//! Corrugated metal texture generator.
//!
//! The algorithm:
//!  1. Precompute two toroidal FBM grids: a micro-detail layer for surface
//!     texture variation, and a separate rust noise layer for weathering.
//!  2. For each pixel the corrugation profile is computed analytically from the
//!     U coordinate using a sine wave.  No trig lookup tables are needed for the
//!     ridge shape because it is a function of U alone and is already O(W) total.
//!  3. Rust accumulates in the valleys of the corrugation (low ridge_h) modulated
//!     by the rust noise and the `rust_level` parameter.
//!  4. Colour interpolates between `color_metal` and `color_rust`.  The micro-detail
//!     FBM adds subtle brightness variation to the metal base to suggest scratches
//!     and manufacturing imperfections.
//!  5. Height is dominated by the ridge profile; micro-detail adds a fine overlay.
//!  6. ORM: metallic surfaces with rust patches that raise roughness and lower
//!     the metallic value proportionally to the rust mask.

use std::f64::consts::TAU;

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Configures the appearance of a [`CorrugatedGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CorrugatedConfig {
    pub seed: u32,
    /// Number of corrugation ridges across the texture U-axis. \[3, 20\]
    pub ridges: f64,
    /// Ridge profile amplitude for the height map. \[0.5, 2.0\]
    pub ridge_depth: f64,
    /// Base surface roughness \[0, 1\].
    pub roughness: f64,
    /// Rust accumulation in valleys \[0, 1\].
    pub rust_level: f64,
    /// Metallic value \[0, 1\].
    pub metallic: f32,
    /// Metal colour in linear RGB.
    pub color_metal: [f32; 3],
    /// Rust colour in linear RGB.
    pub color_rust: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for CorrugatedConfig {
    fn default() -> Self {
        Self {
            seed: 31,
            ridges: 8.0,
            ridge_depth: 1.0,
            roughness: 0.35,
            rust_level: 0.25,
            metallic: 0.85,
            color_metal: [0.72, 0.74, 0.76],
            color_rust: [0.55, 0.30, 0.12],
            normal_strength: 4.0,
        }
    }
}

/// Procedural corrugated-metal texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`CorrugatedConfig`].  Construct
/// via [`CorrugatedGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::corrugated`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct CorrugatedGenerator {
    config: CorrugatedConfig,
    micro_noise: ToroidalNoise<Fbm<Perlin>>,
    rust_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl CorrugatedGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: CorrugatedConfig) -> Self {
        let fbm_micro: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(4);
        let micro_noise = ToroidalNoise::new(fbm_micro, config.ridges * 0.5);
        let fbm_rust: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(100)).set_octaves(3);
        let rust_noise = ToroidalNoise::new(fbm_rust, config.ridges * 0.25);
        Self {
            config,
            micro_noise,
            rust_noise,
        }
    }
}

impl TextureGenerator for CorrugatedGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Micro-detail FBM — higher frequency for surface scratches and
        // manufacturing texture.  Uses `ridges * 0.5` as the toroidal radius
        // so the detail density scales with the number of corrugation ridges.
        let micro_grid = sample_grid(&self.micro_noise, width, height);

        // Rust noise — separate seed, lower frequency for blotchy weathering.
        // A separate V-direction streaking pass uses a portion of this same grid
        // to simulate vertical rust runs from the ridges.
        let rust_grid = sample_grid(&self.rust_noise, width, height);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        // Round ridges to the nearest integer so the pattern tiles exactly.
        let ridges = c.ridges.round();

        for y in 0..h {
            let v = y as f64 / h as f64;

            for x in 0..w {
                let u = x as f64 / w as f64;
                let idx = y * w + x;

                // Corrugation ridge profile: sine wave along U, remapped to [0, 1].
                // Peaks (ridge tops) → 1.0; troughs (valleys) → 0.0.
                let ridge_h = (u * ridges * TAU).sin() * 0.5 + 0.5;

                // Surface micro-detail and rust noise, normalised to [0, 1].
                let surf = normalize(micro_grid[idx]);
                let rust_n = normalize(rust_grid[idx]);

                // V-direction rust streaks: sample rust noise offset slightly in V
                // to create horizontal drips running down from valley centres.
                // The streak factor biases toward the lower half of the V range,
                // simulating gravity-driven rust runs.
                let streak_v = (v + rust_n * 0.15).rem_euclid(1.0);
                let streak_bias = (streak_v * TAU).sin() * 0.5 + 0.5;

                // Rust mask: accumulates in valleys (low ridge_h), scaled by noise
                // and V-direction streaking.  Valleys = (1.0 - ridge_h) raised to
                // a power to concentrate rust at the bottom of the trough.
                let valley_factor = (1.0 - ridge_h).powf(1.5);
                let rust_mask = (valley_factor * rust_n * (0.7 + streak_bias * 0.3) * c.rust_level)
                    .clamp(0.0, 1.0);

                // Height: ridge profile dominates; micro-detail adds fine surface texture.
                let h_val = (ridge_h * c.ridge_depth + surf * 0.05).clamp(0.0, 1.0);
                heights[idx] = h_val;

                // Colour: lerp metal → rust, with a subtle brightness perturbation
                // from the micro-detail layer that suggests scratches and sheen.
                let metal_bright = lerp(0.85, 1.0, surf as f32);
                let rust_mask_f = rust_mask as f32;
                let r = lerp(
                    c.color_metal[0] * metal_bright,
                    c.color_rust[0],
                    rust_mask_f,
                );
                let g = lerp(
                    c.color_metal[1] * metal_bright,
                    c.color_rust[1],
                    rust_mask_f,
                );
                let b = lerp(
                    c.color_metal[2] * metal_bright,
                    c.color_rust[2],
                    rust_mask_f,
                );

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // ORM: rust raises roughness and suppresses metallic.
                let rough = (c.roughness as f32 + rust_mask_f * 0.4).clamp(0.0, 1.0);
                let met = (c.metallic - rust_mask_f * 0.7 * c.metallic).clamp(0.0, 1.0);
                roughness_buf[ai] = 255; // Occlusion = 1.0
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

// --- helpers ----------------------------------------------------------------

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}
