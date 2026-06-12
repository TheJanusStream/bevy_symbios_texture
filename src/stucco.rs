//! Stucco / render texture generator.
//!
//! Smooth, high-frequency FBM bumps over a flat matte base — typical of
//! sand-float or pebble-dash exterior render.  The surface is almost flat
//! (low relief) and entirely matte with zero metallic response.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`StuccoGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StuccoConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Spatial frequency — controls bump density (higher = finer texture).
    pub scale: f64,
    /// FBM octave count.
    pub octaves: usize,
    /// Bump amplitude \[0, 1\] — controls surface relief depth.
    pub roughness: f64,
    /// Base stucco colour in linear RGB \[0, 1\].
    pub color_base: [f32; 3],
    /// Shadow / recessed-area colour in linear RGB \[0, 1\].
    pub color_shadow: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for StuccoConfig {
    fn default() -> Self {
        Self {
            seed: 13,
            scale: 8.0,
            octaves: 6,
            roughness: 0.35,
            color_base: [0.92, 0.89, 0.84],
            color_shadow: [0.72, 0.70, 0.66],
            normal_strength: 2.0,
        }
    }
}

/// Procedural stucco / render texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`StuccoConfig`].  Construct
/// via [`StuccoGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::stucco`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct StuccoGenerator {
    config: StuccoConfig,
    noise: ToroidalNoise<Fbm<Perlin>>,
}

impl StuccoGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: StuccoConfig) -> Self {
        let fbm: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(config.octaves);
        let noise = ToroidalNoise::new(fbm, config.scale);
        Self { config, noise }
    }
}

/// Per-generation sampler: precomputed FBM grid + config.
struct StuccoCell<'a> {
    config: &'a StuccoConfig,
    grid: &'a [f64],
    width: usize,
}

impl SurfaceCell for StuccoCell<'_> {
    fn sample(&self, x: u32, y: u32, _u: f64, _v: f64) -> SurfaceSample {
        let c = self.config;
        let raw = self.grid[y as usize * self.width + x as usize];

        // normalize maps [-1,1] → [0,1]; scale by roughness amplitude.
        let t = (normalize(raw) * c.roughness) as f32;

        let color = [
            lerp(c.color_shadow[0], c.color_base[0], t),
            lerp(c.color_shadow[1], c.color_base[1], t),
            lerp(c.color_shadow[2], c.color_base[2], t),
        ];

        // Matte finish: high roughness, zero metallic.
        // Recessed bumps (low t) are slightly rougher (shadow/grit).
        let rough = (0.82 + (1.0 - t) * 0.10).clamp(0.0, 1.0);

        // Scale the height by roughness so the normal map also respects
        // bump amplitude (not just the albedo interpolation above); raw
        // [-1, 1] range, so generate_inner halves the strength.
        SurfaceSample::matte(raw * c.roughness, color, rough)
    }
}

impl StuccoGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;

        let mut grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.noise, width, height, &mut grid);

        let cell = StuccoCell {
            config: &self.config,
            grid: &grid,
            width: width as usize,
        };
        let result = generate_surface(
            width,
            height,
            self.config.normal_strength * 0.5,
            ws.as_deref_mut(),
            &cell,
        );

        if let Some(ws) = ws {
            ws.return_grid(grid);
        }
        result
    }
}

impl TextureGenerator for StuccoGenerator {
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
