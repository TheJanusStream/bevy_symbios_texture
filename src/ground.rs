//! Ground / dirt texture generator.
//!
//! Produces a matted, organic-looking surface by blending two FBM layers at
//! different scales. A low-frequency layer defines broad soil patches; a
//! high-frequency layer adds fine grain and pebble-like micro-detail.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`GroundGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GroundConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Scale of the large soil-patch layer.
    pub macro_scale: f64,
    /// Octaves for the large soil-patch FBM layer.
    pub macro_octaves: usize,
    /// Scale of the fine-grain layer.
    pub micro_scale: f64,
    /// Octaves for the fine-grain FBM layer.
    pub micro_octaves: usize,
    /// Blend weight of the micro layer (0 = only macro, 1 = only micro).
    pub micro_weight: f64,
    /// Dry (light) soil colour in linear RGB \[0, 1\].
    pub color_dry: [f32; 3],
    /// Moist (dark) soil colour in linear RGB \[0, 1\].
    pub color_moist: [f32; 3],
    /// Normal map strength — larger values produce more pronounced surface detail.
    pub normal_strength: f32,
}

impl Default for GroundConfig {
    fn default() -> Self {
        Self {
            seed: 13,
            macro_scale: 2.0,
            macro_octaves: 5,
            micro_scale: 8.0,
            micro_octaves: 4,
            micro_weight: 0.35,
            color_dry: [0.52, 0.40, 0.26],
            color_moist: [0.28, 0.20, 0.12],
            normal_strength: 2.0,
        }
    }
}

/// Procedural ground / dirt texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`GroundConfig`].  Construct
/// via [`GroundGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::ground`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct GroundGenerator {
    config: GroundConfig,
    macro_noise: ToroidalNoise<Fbm<Perlin>>,
    micro_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl GroundGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: GroundConfig) -> Self {
        let fbm_macro: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(config.macro_octaves);
        let macro_noise = ToroidalNoise::new(fbm_macro, config.macro_scale);

        let fbm_micro: Fbm<Perlin> =
            Fbm::new(config.seed.wrapping_add(50)).set_octaves(config.micro_octaves);
        let micro_noise = ToroidalNoise::new(fbm_micro, config.micro_scale);

        Self {
            config,
            macro_noise,
            micro_noise,
        }
    }
}

/// Per-pixel sampler over the two FBM layers.
///
/// Samples the toroidal noise analytically at (u, v) rather than through a
/// precomputed grid: the LUT path computes `TAU * x / w` where the analytic
/// path computes `TAU * (x / w)`, which differs in the last ulp — switching
/// would break byte parity with the pre-driver output.
struct GroundCell<'a> {
    config: &'a GroundConfig,
    macro_noise: &'a ToroidalNoise<Fbm<Perlin>>,
    micro_noise: &'a ToroidalNoise<Fbm<Perlin>>,
}

impl SurfaceCell for GroundCell<'_> {
    fn sample(&self, _x: u32, _y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;

        let macro_val = normalize(self.macro_noise.get(u, v));
        let micro_val = normalize(self.micro_noise.get(u, v));
        let t = macro_val * (1.0 - c.micro_weight) + micro_val * c.micro_weight;

        let tf = t as f32;
        let color = [
            lerp(c.color_moist[0], c.color_dry[0], tf),
            lerp(c.color_moist[1], c.color_dry[1], tf),
            lerp(c.color_moist[2], c.color_dry[2], tf),
        ];

        // Ground is generally rough; slight variation by moisture.
        let rough = 0.80 + (1.0 - tf) * 0.15;

        SurfaceSample::matte(t, color, rough)
    }
}

impl GroundGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let cell = GroundCell {
            config: &self.config,
            macro_noise: &self.macro_noise,
            micro_noise: &self.micro_noise,
        };
        generate_surface(width, height, self.config.normal_strength, ws, &cell)
    }
}

impl TextureGenerator for GroundGenerator {
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
