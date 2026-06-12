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
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
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
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
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
/// Drives [`TextureGenerator::generate`] using a [`MetalConfig`].  Construct
/// via [`MetalGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::metal`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct MetalGenerator {
    config: MetalConfig,
    fbm_scratch: Fbm<Perlin>,
    rust_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl MetalGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: MetalConfig) -> Self {
        let fbm_scratch: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(5);

        let fbm_rust: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(41)).set_octaves(4);
        let rust_noise = ToroidalNoise::new(fbm_rust, config.scale * 0.4);

        Self {
            config,
            fbm_scratch,
            rust_noise,
        }
    }
}

/// Per-generation sampler: precomputed rust grid + per-pixel anisotropic
/// scratch noise (sampled analytically — the brushed style's stretched torus
/// coordinates have no grid equivalent).
struct MetalCell<'a> {
    config: &'a MetalConfig,
    fbm_scratch: &'a Fbm<Perlin>,
    rust_grid: &'a [f64],
    width: usize,
}

impl SurfaceCell for MetalCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;

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
                self.fbm_scratch.get([nx, ny, nz, nw]) * 0.5 + 0.5
            }
            MetalStyle::StandingSeam => {
                let nx = (TAU * u).cos() * c.scale;
                let ny = (TAU * u).sin() * c.scale;
                let nz = (TAU * v).cos() * c.scale;
                let nw = (TAU * v).sin() * c.scale;
                self.fbm_scratch.get([nx, ny, nz, nw]) * 0.5 + 0.5
            }
        };

        let idx = y as usize * self.width + x as usize;
        let rust_t = normalize(self.rust_grid[idx]);
        // Soft threshold → rust coverage.
        let rust_blend = ((rust_t - (1.0 - c.rust_level)).clamp(0.0, c.rust_level)
            / c.rust_level.max(1e-9))
        .clamp(0.0, 1.0);

        let h_scratch = scratch * c.roughness * 0.3;
        let h_val = match c.style {
            MetalStyle::Brushed => h_scratch,
            MetalStyle::StandingSeam => seam_h * 0.7 + h_scratch * 0.3,
        };

        // Colour: lerp metal → rust.
        let color = [
            lerp(c.color_metal[0], c.color_rust[0], rust_blend as f32),
            lerp(c.color_metal[1], c.color_rust[1], rust_blend as f32),
            lerp(c.color_metal[2], c.color_rust[2], rust_blend as f32),
        ];

        // ORM: rust raises roughness and kills metallic.
        let rough = (c.roughness as f32 + rust_blend as f32 * 0.65).clamp(0.0, 1.0);
        let met = (c.metallic - rust_blend as f32 * 0.80).clamp(0.0, 1.0);

        SurfaceSample {
            height: h_val,
            color,
            roughness: rough,
            metallic: met,
            occlusion: 1.0,
        }
    }
}

impl MetalGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;

        // Rust patches — separate seed, low frequency for large blotches.
        let mut rust_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.rust_noise, width, height, &mut rust_grid);

        let cell = MetalCell {
            config: &self.config,
            fbm_scratch: &self.fbm_scratch,
            rust_grid: &rust_grid,
            width: width as usize,
        };
        let result = generate_surface(
            width,
            height,
            self.config.normal_strength,
            ws.as_deref_mut(),
            &cell,
        );

        if let Some(ws) = ws {
            ws.return_grid(rust_grid);
        }
        result
    }
}

impl TextureGenerator for MetalGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        self.generate_inner(width, height, None)
    }

    fn generate_with_workspace(
        &self,
        width: u32,
        height: u32,
        workspace: &mut Workspace,
    ) -> Result<TextureMap, TextureError> {
        self.generate_inner(width, height, Some(workspace))
    }
}
