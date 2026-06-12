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
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`ConcreteGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConcreteConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
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

/// Per-generation sampler: precomputed surface + pit FBM grids and config.
struct ConcreteCell<'a> {
    config: &'a ConcreteConfig,
    surf: &'a [f64],
    pits: &'a [f64],
    width: usize,
}

impl SurfaceCell for ConcreteCell<'_> {
    fn sample(&self, x: u32, y: u32, _u: f64, v: f64) -> SurfaceSample {
        let c = self.config;

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

        let idx = y as usize * self.width + x as usize;
        let surf_t = normalize(self.surf[idx]);
        let pit_t = normalize(self.pits[idx]);

        // Pits: pixels where pit noise exceeds (1 - density) threshold.
        let threshold = (1.0 - c.pit_density.clamp(0.0, 0.5)).max(0.5);
        let pit_depth = if pit_t > threshold {
            let d = (pit_t - threshold) / (1.0 - threshold).max(1e-9);
            d * 0.4
        } else {
            0.0
        };

        let h_val = (surf_t * c.roughness - line_groove - pit_depth).clamp(0.0, 1.0);

        // Colour: pits and formwork grooves are darker.
        let shadow = (pit_depth as f32 * 4.0 + line_groove as f32 * 5.0).clamp(0.0, 1.0);
        let color = [
            lerp(c.color_base[0], c.color_pit[0], shadow),
            lerp(c.color_base[1], c.color_pit[1], shadow),
            lerp(c.color_base[2], c.color_pit[2], shadow),
        ];

        // ORM: rough, no metallic; pits slightly rougher.
        let rough = (0.80 + shadow * 0.12).clamp(0.0, 1.0);

        SurfaceSample::matte(h_val, color, rough)
    }
}

impl ConcreteGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;

        // Main surface FBM — smooth, low-frequency bumps.
        let mut surf = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.surf_noise, width, height, &mut surf);

        // High-frequency pit noise — separate seed.
        let mut pits = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.pit_noise, width, height, &mut pits);

        let cell = ConcreteCell {
            config: &self.config,
            surf: &surf,
            pits: &pits,
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
            ws.return_grid(surf);
            ws.return_grid(pits);
        }
        result
    }
}

impl TextureGenerator for ConcreteGenerator {
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
