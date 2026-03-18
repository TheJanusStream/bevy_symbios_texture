//! Cast concrete texture generator.
//!
//! The algorithm:
//! 1. Sample a smooth FBM for the main surface relief.
//! 2. Add horizontal formwork-panel lines (cosine grooves in V).
//! 3. Scatter air-pocket pits using a second high-frequency FBM.
//! 4. Blend surface, grooves and pits into the height map and albedo.

use std::f64::consts::TAU;

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Configures the appearance of a [`ConcreteGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConcreteConfig {
    pub seed: u32,
    /// Scale of the main surface FBM.
    pub scale: f64,
    /// FBM octave count.
    pub octaves: usize,
    /// Overall bump amplitude \[0, 1\].
    pub roughness: f64,
    /// Number of horizontal formwork-panel lines per tile \[0 = none\].
    pub formwork_lines: f64,
    /// Groove depth of formwork seams \[0, 1\].
    pub formwork_depth: f64,
    /// Air-pocket / pitting density \[0, 0.5\].
    pub pit_density: f64,
    /// Base concrete colour in linear RGB \[0, 1\].
    pub color_base: [f32; 3],
    /// Pit / shadow colour in linear RGB \[0, 1\].
    pub color_pit: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for ConcreteConfig {
    fn default() -> Self {
        Self {
            seed: 17,
            scale: 5.0,
            octaves: 5,
            roughness: 0.45,
            formwork_lines: 4.0,
            formwork_depth: 0.12,
            pit_density: 0.08,
            color_base: [0.55, 0.54, 0.52],
            color_pit: [0.35, 0.34, 0.33],
            normal_strength: 2.5,
        }
    }
}

/// Procedural cast-concrete texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`ConcreteConfig`].  Construct
/// via [`ConcreteGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::concrete`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct ConcreteGenerator {
    config: ConcreteConfig,
    surf_noise: ToroidalNoise<Fbm<Perlin>>,
    pit_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl ConcreteGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: ConcreteConfig) -> Self {
        let fbm_surf: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(config.octaves);
        let surf_noise = ToroidalNoise::new(fbm_surf, config.scale);

        let fbm_pit: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(77))
            .set_octaves(3)
            .set_frequency(2.0);
        let pit_noise = ToroidalNoise::new(fbm_pit, config.scale * 4.0);

        Self {
            config,
            surf_noise,
            pit_noise,
        }
    }
}

impl TextureGenerator for ConcreteGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Main surface FBM — smooth, low-frequency bumps.
        let surf = sample_grid(&self.surf_noise, width, height);

        // High-frequency pit noise — separate seed.
        let pits = sample_grid(&self.pit_noise, width, height);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;

            // Formwork lines: thin cosine groove repeated `formwork_lines` times in V.
            // Must be an integer count for the pattern to tile; round to nearest.
            let formwork_lines = c.formwork_lines.round();
            let line_groove = if formwork_lines > 0.0 {
                let phase = (v * formwork_lines * TAU).cos();
                // Groove deepest where phase = +1 (peaks), shallow elsewhere.
                ((phase * 0.5 + 0.5) * c.formwork_depth).clamp(0.0, 1.0)
            } else {
                0.0
            };

            for x in 0..w {
                let idx = y * w + x;
                let surf_t = normalize(surf[idx]);
                let pit_t = normalize(pits[idx]);

                // Pits: pixels where pit noise exceeds (1 - density) threshold.
                let threshold = (1.0 - c.pit_density.clamp(0.0, 0.5)).max(0.5);
                let pit_depth = if pit_t > threshold {
                    let d = (pit_t - threshold) / (1.0 - threshold).max(1e-9);
                    d * 0.4
                } else {
                    0.0
                };

                let h_val = (surf_t * c.roughness - line_groove - pit_depth).clamp(0.0, 1.0);
                heights[idx] = h_val;

                // Colour: pits and formwork grooves are darker.
                let shadow = (pit_depth as f32 * 4.0 + line_groove as f32 * 5.0).clamp(0.0, 1.0);
                let r = lerp(c.color_base[0], c.color_pit[0], shadow);
                let g = lerp(c.color_base[1], c.color_pit[1], shadow);
                let b = lerp(c.color_base[2], c.color_pit[2], shadow);

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // ORM: rough, no metallic; pits slightly rougher.
                let rough = (0.80 + shadow * 0.12).clamp(0.0, 1.0);
                roughness_buf[ai] = 255;
                roughness_buf[ai + 1] = (rough * 255.0).round() as u8;
                roughness_buf[ai + 2] = 0;
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
