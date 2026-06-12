//! Thatch texture generator — dense fibrous roofing material.
//!
//! The algorithm:
//! 1. Build two toroidal FBMs: a high-frequency fibre noise (along U) and a
//!    low-frequency layer-variation noise (along V).  A third low-frequency
//!    warp noise distorts the UV coordinates laterally before sampling, giving
//!    the organic wiggly appearance of real straw bundles.
//! 2. Combine fibre and layer noise into a scalar `fiber_t` value \[0, 1\].
//! 3. Overlay a repeating sawtooth in V with `layer_count` periods; the bottom
//!    of each period is darkened by `layer_shadow` to simulate the shadow cast
//!    by the bundle tip above.
//! 4. Lerp between `color_shadow` and `color_straw` using the combined signal.
//! 5. Height = fiber_t × (1 – shadow gradient) for a convincing normal map.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`ThatchGenerator`].
///
/// Thatch is modelled as densely-packed straw bundles laid in overlapping
/// horizontal layers, like shingles.  The high U-frequency noise creates
/// individual fibre streaks while the V-frequency sawtooth creates the layered
/// overlap shadow pattern.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ThatchConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Fibre density — noise frequency along U (controls how many fibres are
    /// visible across the tile) \[4, 24\].
    pub density: f64,
    /// Anisotropy ratio: the V frequency is `density / anisotropy`, making
    /// fibres appear long and horizontal \[4, 16\].
    pub anisotropy: f64,
    /// Lateral domain-warp strength — how much the fibres wiggle \[0, 0.5\].
    pub warp_strength: f64,
    /// Number of straw-bundle overlap layers visible across the V axis \[4, 16\].
    pub layer_count: f64,
    /// Layer shadow depth — how much darker the bottom of each bundle layer is
    /// \[0, 1\].
    pub layer_shadow: f64,
    /// Base (dry straw) colour in linear RGB \[0, 1\].
    pub color_straw: [f32; 3],
    /// Shadow / rot colour at the bottom of each bundle \[0, 1\].
    pub color_shadow: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for ThatchConfig {
    fn default() -> Self {
        Self {
            seed: 19,
            density: 12.0,
            anisotropy: 8.0,
            warp_strength: 0.15,
            layer_count: 8.0,
            layer_shadow: 0.55,
            color_straw: [0.62, 0.54, 0.28],
            color_shadow: [0.22, 0.17, 0.09],
            normal_strength: 3.5,
        }
    }
}

/// Procedural thatch texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`ThatchConfig`].  Construct
/// via [`ThatchGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::thatch`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct ThatchGenerator {
    config: ThatchConfig,
    warp_noise: ToroidalNoise<Fbm<Perlin>>,
    fibre_noise: ToroidalNoise<Fbm<Perlin>>,
    layer_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl ThatchGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: ThatchConfig) -> Self {
        let warp_freq = (config.density * 0.3).max(0.5);
        let fbm_warp: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(7)).set_octaves(3);
        let warp_noise = ToroidalNoise::new(fbm_warp, warp_freq);

        let fbm_fibre: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(50)).set_octaves(5);
        let fibre_noise = ToroidalNoise::new(fbm_fibre, config.density);

        let layer_freq = (config.density / config.anisotropy.max(1.0)).max(0.5);
        let fbm_layer: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(150)).set_octaves(3);
        let layer_noise = ToroidalNoise::new(fbm_layer, layer_freq);

        Self {
            config,
            warp_noise,
            fibre_noise,
            layer_noise,
        }
    }
}

impl ThatchGenerator {
    /// Core generation logic.  When `ws` is `Some`, borrows grid buffers from
    /// the workspace and returns them when done, avoiding fresh allocations.
    /// Reuses workspace buffers across calls so that generating multiple
    /// size variants does not allocate new backing storage each time.
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Borrow or allocate grid buffers.
        let (mut warp_grid, mut fibre_grid, mut layer_grid) = match ws.as_deref_mut() {
            Some(w) => (w.take_grid(), w.take_grid(), w.take_grid()),
            None => (Vec::new(), Vec::new(), Vec::new()),
        };

        sample_grid_into(&self.warp_noise, width, height, &mut warp_grid);
        sample_grid_into(&self.fibre_noise, width, height, &mut fibre_grid);
        sample_grid_into(&self.layer_noise, width, height, &mut layer_grid);

        let cell = ThatchCell {
            config: c,
            warp_grid: &warp_grid,
            fibre_grid: &fibre_grid,
            layer_grid: &layer_grid,
            layer_count: c.layer_count.round().max(1.0),
            width: width as usize,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        // Return grid buffers to the workspace for reuse.
        if let Some(ws) = ws {
            ws.return_grid(warp_grid);
            ws.return_grid(fibre_grid);
            ws.return_grid(layer_grid);
        }
        result
    }
}

/// Per-generation sampler: warp / fibre / layer grids + config.
struct ThatchCell<'a> {
    config: &'a ThatchConfig,
    warp_grid: &'a [f64],
    fibre_grid: &'a [f64],
    layer_grid: &'a [f64],
    /// `layer_count` rounded and clamped to ≥ 1 so the sawtooth tiles.
    layer_count: f64,
    width: usize,
}

impl SurfaceCell for ThatchCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let w = self.width;
        let idx = y as usize * w + x as usize;

        let layer_v = (v * self.layer_count).fract();
        let shadow_t = (1.0 - layer_v).powf(1.5);

        let warp = self.warp_grid[idx] * c.warp_strength;

        // Lateral domain warp: shift the fibre lookup along the row.
        let warped_x = {
            let ux = (u + warp).rem_euclid(1.0);
            (ux * w as f64) as usize % w
        };
        let warped_idx = y as usize * w + warped_x;
        let fibre_raw = normalize(self.fibre_grid[warped_idx]);
        let layer_raw = normalize(self.layer_grid[idx]);

        let fiber_t = (0.65 * fibre_raw + 0.35 * layer_raw).clamp(0.0, 1.0);

        let h_val =
            (fiber_t * (0.5 + 0.5 * layer_v) - shadow_t * c.layer_shadow * 0.3).clamp(0.0, 1.0);

        let brightness = (fiber_t * (1.0 - shadow_t * c.layer_shadow)).clamp(0.0, 1.0);
        let color = [
            lerp(c.color_shadow[0], c.color_straw[0], brightness as f32),
            lerp(c.color_shadow[1], c.color_straw[1], brightness as f32),
            lerp(c.color_shadow[2], c.color_straw[2], brightness as f32),
        ];

        let rough_val = (0.80 - fiber_t as f32 * 0.15 + shadow_t as f32 * 0.10).clamp(0.65, 0.95);

        SurfaceSample::matte(h_val, color, rough_val)
    }
}

impl TextureGenerator for ThatchGenerator {
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
