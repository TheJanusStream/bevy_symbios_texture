//! Asphalt / tarmac texture generator.
//!
//! The algorithm:
//!  1. Three toroidal FBM grids are precomputed at different frequency bands:
//!     - `macro_grid`: low-frequency large-area staining (oil patches, weathering).
//!     - `micro_grid`: mid-frequency micro-texture roughness (aggregate matrix).
//!     - `aggregate_grid`: high-frequency fleck noise for individual stone chips.
//!  2. For each pixel the aggregate grid is threshold-tested: pixels above
//!     `(1.0 - aggregate_density)` are classified as exposed aggregate stones.
//!     These appear as bright, slightly raised flecks in a dark binder matrix.
//!  3. The macro noise drives a stain factor that subtly lightens or darkens the
//!     asphalt base, simulating oil drips, water staining, and UV bleaching.
//!  4. Height is dominated by micro-texture, with aggregate stones sitting
//!     fractionally proud of the surrounding binder.
//!  5. ORM: high roughness throughout (asphalt is never shiny), zero metallic,
//!     with micro-variation reducing roughness slightly at peak elevations.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Configures the appearance of an [`AsphaltGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AsphaltConfig {
    pub seed: u32,
    /// Base noise scale \[2, 12\].
    pub scale: f64,
    /// Aggregate stone density threshold \[0.05, 0.4\].
    pub aggregate_density: f64,
    /// Aggregate fleck scale (high frequency) \[8, 32\].
    pub aggregate_scale: f64,
    /// Overall surface roughness \[0.7, 1.0\].
    pub roughness: f64,
    /// Macro stain / oil variation amplitude \[0, 1\].
    pub stain_level: f64,
    /// Base asphalt colour in linear RGB.
    pub color_base: [f32; 3],
    /// Aggregate stone fleck colour in linear RGB.
    pub color_aggregate: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for AsphaltConfig {
    fn default() -> Self {
        Self {
            seed: 88,
            scale: 4.0,
            aggregate_density: 0.22,
            aggregate_scale: 16.0,
            roughness: 0.90,
            stain_level: 0.25,
            color_base: [0.06, 0.06, 0.07],
            color_aggregate: [0.35, 0.33, 0.30],
            normal_strength: 2.5,
        }
    }
}

/// Procedural asphalt / tarmac texture generator.
///
/// Drives [`TextureGenerator::generate`] using an [`AsphaltConfig`].  Construct
/// via [`AsphaltGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::asphalt`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct AsphaltGenerator {
    config: AsphaltConfig,
    macro_noise: ToroidalNoise<Fbm<Perlin>>,
    micro_noise: ToroidalNoise<Fbm<Perlin>>,
    agg_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl AsphaltGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: AsphaltConfig) -> Self {
        let fbm_macro: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(3);
        let macro_noise = ToroidalNoise::new(fbm_macro, config.scale);
        let fbm_micro: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(50)).set_octaves(4);
        let micro_noise = ToroidalNoise::new(fbm_micro, config.scale * 3.0);
        let fbm_agg: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(200)).set_octaves(2);
        let agg_noise = ToroidalNoise::new(fbm_agg, config.aggregate_scale);
        Self {
            config,
            macro_noise,
            micro_noise,
            agg_noise,
        }
    }
}

impl TextureGenerator for AsphaltGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Macro stain / colour-variation FBM — low frequency for large blotches.
        let macro_grid = sample_grid(&self.macro_noise, width, height);

        // Mid-frequency micro-texture — the fine granular surface of the binder.
        let micro_grid = sample_grid(&self.micro_noise, width, height);

        // High-frequency aggregate fleck noise — sharp, fine-grained.
        let aggregate_grid = sample_grid(&self.agg_noise, width, height);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        // Precompute the aggregate threshold; clamp density to a valid range so
        // the threshold stays in (0, 1) and never produces a degenerate mask.
        let agg_density = c.aggregate_density.clamp(0.01, 0.99);
        let agg_threshold = 1.0 - agg_density;

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;

                // Normalise all three grids to [0, 1].
                let macro_v = normalize(macro_grid[idx]);
                let micro_v = normalize(micro_grid[idx]);
                let agg_v = normalize(aggregate_grid[idx]);

                // Aggregate fleck: pixels above threshold are exposed stone chips.
                let is_agg = agg_v > agg_threshold;

                // Stain factor: macro noise drives subtle lightening / darkening.
                // Centred at 0.5 so the mean effect is neutral.
                let stain = (macro_v - 0.5) * c.stain_level;

                // Height: micro-texture for the binder matrix; aggregate sits proud.
                let agg_bump = if is_agg { 0.3 } else { 0.0 };
                heights[idx] = (micro_v * 0.7 + agg_bump).clamp(0.0, 1.0);

                // Colour: bright stone flecks or stain-modulated asphalt base.
                let (r, g, b) = if is_agg {
                    // Aggregate: slight micro-variation on the stone colour.
                    let brightness = lerp(0.85, 1.0, micro_v as f32);
                    (
                        c.color_aggregate[0] * brightness,
                        c.color_aggregate[1] * brightness,
                        c.color_aggregate[2] * brightness,
                    )
                } else {
                    // Asphalt binder: stain shifts the base colour slightly.
                    let stain_f = stain as f32;
                    (
                        (c.color_base[0] + stain_f).clamp(0.0, 1.0),
                        (c.color_base[1] + stain_f).clamp(0.0, 1.0),
                        (c.color_base[2] + stain_f).clamp(0.0, 1.0),
                    )
                };

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // ORM: asphalt is uniformly rough; micro peaks are fractionally
                // smoother (exposed surface facets of the aggregate matrix).
                let rough = (c.roughness - micro_v * 0.1).clamp(0.0, 1.0) as f32;
                roughness_buf[ai] = 255; // Occlusion = 1.0
                roughness_buf[ai + 1] = (rough * 255.0).round() as u8;
                roughness_buf[ai + 2] = 0; // Metallic = 0.0
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
