//! Metal texture generator — brushed finish or standing-seam roof panels,
//! with optional rust weathering.
//!
//! The algorithm:
//! 1. **Brushed**: anisotropic FBM — high frequency in U (many scratches),
//!    very low frequency in V (scratches run nearly horizontally).
//! 2. **StandingSeam**: sinusoidal ridge profile across V, with micro-detail
//!    FBM overlay.
//! 3. A separate low-frequency FBM drives rust-patch blending: rust areas
//!    receive a warm colour, raised roughness, and reduced metallic value.

use std::f64::consts::TAU;

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Visual style of the metal surface.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MetalStyle {
    /// Fine horizontal scratches (brushed / satin finish).
    Brushed,
    /// Parallel raised ridges running across the tile (standing-seam roof).
    StandingSeam,
}

/// Configures the appearance of a [`MetalGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MetalConfig {
    pub seed: u32,
    /// Surface finish style.
    pub style: MetalStyle,
    /// Base noise scale.
    pub scale: f64,
    /// For `StandingSeam`: number of ridges across the tile.
    pub seam_count: f64,
    /// Ridge sharpness for `StandingSeam` \[0.5 = sinusoidal, 4.0 = sharp\].
    pub seam_sharpness: f64,
    /// Anisotropy factor for `Brushed` — higher = longer horizontal scratches.
    pub brush_stretch: f64,
    /// Micro-roughness amplitude \[0, 1\].
    pub roughness: f64,
    /// Metallic value for clean (rust-free) areas \[0, 1\].
    pub metallic: f32,
    /// Rust-patch coverage \[0 = none, 1 = heavy\].
    pub rust_level: f64,
    /// Base metal colour in linear RGB \[0, 1\].
    pub color_metal: [f32; 3],
    /// Rust colour in linear RGB \[0, 1\].
    pub color_rust: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            seed: 31,
            style: MetalStyle::Brushed,
            scale: 6.0,
            seam_count: 6.0,
            seam_sharpness: 2.5,
            brush_stretch: 8.0,
            roughness: 0.25,
            metallic: 0.85,
            rust_level: 0.15,
            color_metal: [0.42, 0.44, 0.47],
            color_rust: [0.42, 0.24, 0.12],
            normal_strength: 3.0,
        }
    }
}

/// Procedural metal texture generator.
///
/// Produces tileable albedo, normal, and ORM maps.  Upload via
/// [`crate::async_gen::PendingTexture::metal`] / [`crate::generator::map_to_images`].
pub struct MetalGenerator {
    config: MetalConfig,
}

impl MetalGenerator {
    pub fn new(config: MetalConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for MetalGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Scratch / micro-detail noise.
        let fbm_scratch: Fbm<Perlin> = Fbm::new(c.seed).set_octaves(5);

        // Rust patches — separate seed, low frequency for large blotches.
        let fbm_rust: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(41)).set_octaves(4);
        let rust_noise = ToroidalNoise::new(fbm_rust, c.scale * 0.4);
        let rust_grid = sample_grid(&rust_noise, width, height);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;

            // Standing-seam ridge profile (sinusoidal bumps in V).
            // seam_count must be an integer for the pattern to tile; round to nearest.
            let seam_count = c.seam_count.round();
            let seam_h = if c.style == MetalStyle::StandingSeam {
                let phase = (v * seam_count * TAU).sin();
                // Raise to power to sharpen; clamp to [0,1].
                phase.abs().powf(c.seam_sharpness.max(0.1)) * phase.signum() * 0.5 + 0.5
            } else {
                0.0
            };

            for x in 0..w {
                let u = x as f64 / w as f64;

                // Sample scratch noise.
                // Brushed: large radius in U (fast oscillations → many horizontal
                // scratches), small radius in V (slow → scratches run lengthwise).
                // StandingSeam: uniform toroidal sampling for micro-detail.
                let scratch = match c.style {
                    MetalStyle::Brushed => {
                        let nx = (TAU * u).cos() * c.scale * c.brush_stretch;
                        let ny = (TAU * u).sin() * c.scale * c.brush_stretch;
                        let nz = (TAU * v).cos() * c.scale * 0.12;
                        let nw = (TAU * v).sin() * c.scale * 0.12;
                        fbm_scratch.get([nx, ny, nz, nw]) * 0.5 + 0.5
                    }
                    MetalStyle::StandingSeam => {
                        let nx = (TAU * u).cos() * c.scale;
                        let ny = (TAU * u).sin() * c.scale;
                        let nz = (TAU * v).cos() * c.scale;
                        let nw = (TAU * v).sin() * c.scale;
                        fbm_scratch.get([nx, ny, nz, nw]) * 0.5 + 0.5
                    }
                };

                let idx = y * w + x;
                let rust_t = normalize(rust_grid[idx]);
                // Soft threshold → rust coverage.
                let rust_blend = ((rust_t - (1.0 - c.rust_level)).clamp(0.0, c.rust_level)
                    / c.rust_level.max(1e-9))
                .clamp(0.0, 1.0);

                let h_scratch = scratch * c.roughness * 0.3;
                let h_val = match c.style {
                    MetalStyle::Brushed => h_scratch,
                    MetalStyle::StandingSeam => seam_h * 0.7 + h_scratch * 0.3,
                };
                heights[idx] = h_val;

                // Colour: lerp metal → rust.
                let r = lerp(c.color_metal[0], c.color_rust[0], rust_blend as f32);
                let g = lerp(c.color_metal[1], c.color_rust[1], rust_blend as f32);
                let b = lerp(c.color_metal[2], c.color_rust[2], rust_blend as f32);

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // ORM: rust raises roughness and kills metallic.
                let rough = (c.roughness as f32 + rust_blend as f32 * 0.65).clamp(0.0, 1.0);
                let met = (c.metallic - rust_blend as f32 * 0.80).clamp(0.0, 1.0);
                roughness_buf[ai] = 255;
                roughness_buf[ai + 1] = (rough * 255.0).round() as u8;
                roughness_buf[ai + 2] = (met * 255.0).round() as u8;
                roughness_buf[ai + 3] = 255;
            }
        }

        let normal = height_to_normal(
            &heights,
            width,
            height,
            c.normal_strength,
            BoundaryMode::Wrap,
        );

        Ok(TextureMap {
            albedo,
            normal,
            roughness: roughness_buf,
            width,
            height,
        })
    }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}
