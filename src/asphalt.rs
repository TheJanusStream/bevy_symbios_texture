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
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of an [`AsphaltGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AsphaltConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
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

/// Per-generation sampler: three precomputed FBM grids + config.
struct AsphaltCell<'a> {
    config: &'a AsphaltConfig,
    macro_grid: &'a [f64],
    micro_grid: &'a [f64],
    aggregate_grid: &'a [f64],
    /// `1 - clamped aggregate_density`; precomputed so the mask threshold
    /// never goes degenerate.
    agg_threshold: f64,
    width: usize,
}

impl SurfaceCell for AsphaltCell<'_> {
    fn sample(&self, x: u32, y: u32, _u: f64, _v: f64) -> SurfaceSample {
        let c = self.config;
        let idx = y as usize * self.width + x as usize;

        // Normalise all three grids to [0, 1].
        let macro_v = normalize(self.macro_grid[idx]);
        let micro_v = normalize(self.micro_grid[idx]);
        let agg_v = normalize(self.aggregate_grid[idx]);

        // Aggregate fleck: pixels above threshold are exposed stone chips.
        let is_agg = agg_v > self.agg_threshold;

        // Stain factor: macro noise drives subtle lightening / darkening.
        // Centred at 0.5 so the mean effect is neutral.
        let stain = (macro_v - 0.5) * c.stain_level;

        // Height: micro-texture for the binder matrix; aggregate sits proud.
        let agg_bump = if is_agg { 0.3 } else { 0.0 };
        let height = (micro_v * 0.7 + agg_bump).clamp(0.0, 1.0);

        // Colour: bright stone flecks or stain-modulated asphalt base.
        let color = if is_agg {
            // Aggregate: slight micro-variation on the stone colour.
            let brightness = lerp(0.85, 1.0, micro_v as f32);
            [
                c.color_aggregate[0] * brightness,
                c.color_aggregate[1] * brightness,
                c.color_aggregate[2] * brightness,
            ]
        } else {
            // Asphalt binder: stain shifts the base colour slightly.
            let stain_f = stain as f32;
            [
                (c.color_base[0] + stain_f).clamp(0.0, 1.0),
                (c.color_base[1] + stain_f).clamp(0.0, 1.0),
                (c.color_base[2] + stain_f).clamp(0.0, 1.0),
            ]
        };

        // ORM: asphalt is uniformly rough; micro peaks are fractionally
        // smoother (exposed surface facets of the aggregate matrix).
        let rough = (c.roughness - micro_v * 0.1).clamp(0.0, 1.0) as f32;

        SurfaceSample::matte(height, color, rough)
    }
}

impl AsphaltGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Macro stain / colour-variation FBM — low frequency for large blotches.
        let mut macro_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.macro_noise, width, height, &mut macro_grid);

        // Mid-frequency micro-texture — the fine granular surface of the binder.
        let mut micro_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.micro_noise, width, height, &mut micro_grid);

        // High-frequency aggregate fleck noise — sharp, fine-grained.
        let mut aggregate_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.agg_noise, width, height, &mut aggregate_grid);

        let cell = AsphaltCell {
            config: c,
            macro_grid: &macro_grid,
            micro_grid: &micro_grid,
            aggregate_grid: &aggregate_grid,
            agg_threshold: 1.0 - c.aggregate_density.clamp(0.01, 0.99),
            width: width as usize,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        if let Some(ws) = ws {
            ws.return_grid(macro_grid);
            ws.return_grid(micro_grid);
            ws.return_grid(aggregate_grid);
        }
        result
    }
}

impl TextureGenerator for AsphaltGenerator {
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
