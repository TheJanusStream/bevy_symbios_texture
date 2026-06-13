//! Rippled sand texture generator.
//!
//! Wind-rippled dune or beach sand: a directional sine ridge field whose
//! phase is domain-warped by toroidal FBM (so crests meander and branch the
//! way real ripples do), plus high-frequency grain micro-relief and
//! thresholded bright flecks for the sparkle of exposed grains.
//!
//! The ripple count is rounded to an integer so the pattern tiles; both FBM
//! layers are toroidal.

use std::f64::consts::TAU;

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`SandGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SandConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Ripple crests across the tile `[4, 24]` — rounded to an integer so
    /// the pattern tiles exactly.
    pub ripple_count: f64,
    /// Domain-warp strength `[0, 1.5]` — how much the crests meander and
    /// merge.  `0` gives perfectly straight machine-like ridges.
    pub ripple_warp: f64,
    /// Bright-fleck density `[0, 0.5]` — fraction of texels reading as
    /// exposed sparkling grains.
    pub grain_density: f64,
    /// Grain noise frequency `[8, 48]` — controls fleck and micro-relief
    /// size.
    pub grain_scale: f64,
    /// Crest (sunlit) sand colour in linear RGB \[0, 1\].
    pub color_crest: [f32; 3],
    /// Trough (shadowed) sand colour in linear RGB \[0, 1\].
    pub color_trough: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for SandConfig {
    fn default() -> Self {
        Self {
            seed: 91,
            ripple_count: 10.0,
            ripple_warp: 0.6,
            grain_density: 0.12,
            grain_scale: 24.0,
            color_crest: [0.86, 0.74, 0.52],
            color_trough: [0.62, 0.50, 0.34],
            normal_strength: 2.5,
        }
    }
}

/// Procedural rippled-sand texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`SandConfig`].  Construct
/// via [`SandGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::sand`] task for non-blocking generation.
pub struct SandGenerator {
    config: SandConfig,
    warp_noise: ToroidalNoise<Fbm<Perlin>>,
    grain_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl SandGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: SandConfig) -> Self {
        let fbm_warp: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(3);
        let warp_noise = ToroidalNoise::new(fbm_warp, 2.0);
        let fbm_grain: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(70)).set_octaves(2);
        let grain_noise = ToroidalNoise::new(fbm_grain, config.grain_scale.clamp(8.0, 48.0));
        Self {
            config,
            warp_noise,
            grain_noise,
        }
    }

    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let mut warp_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.warp_noise, width, height, &mut warp_grid);

        let mut grain_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.grain_noise, width, height, &mut grain_grid);

        let cell = SandCell {
            config: c,
            warp_grid: &warp_grid,
            grain_grid: &grain_grid,
            ripples: c.ripple_count.round().clamp(1.0, 64.0),
            fleck_threshold: 1.0 - c.grain_density.clamp(0.0, 0.5),
            width: width as usize,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        if let Some(ws) = ws {
            ws.return_grid(warp_grid);
            ws.return_grid(grain_grid);
        }
        result
    }
}

impl TextureGenerator for SandGenerator {
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

/// Per-generation sampler: warp + grain grids and derived ripple constants.
struct SandCell<'a> {
    config: &'a SandConfig,
    warp_grid: &'a [f64],
    grain_grid: &'a [f64],
    /// `ripple_count` rounded so the sine field tiles exactly.
    ripples: f64,
    /// Grain values above this threshold read as bright flecks.
    fleck_threshold: f64,
    width: usize,
}

impl SurfaceCell for SandCell<'_> {
    fn sample(&self, x: u32, y: u32, _u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let idx = y as usize * self.width + x as usize;

        // Ripple field: sine ridges along V, phase-warped by FBM so crests
        // meander and occasionally merge.  The warp term is toroidal, so
        // the phase shift itself tiles.
        let warp = self.warp_grid[idx]; // [-1, 1]
        let phase = v * self.ripples * TAU + warp * c.ripple_warp.clamp(0.0, 1.5) * TAU * 0.5;
        // Slight power sharpens crests relative to the broad troughs.
        let ripple = (phase.sin() * 0.5 + 0.5).powf(1.2);

        // Stretch the FBM distribution: raw fractal output rarely reaches
        // the [0, 1] extremes, which would leave the fleck threshold
        // unreachable at low densities.
        let grain = (normalize(self.grain_grid[idx]) - 0.5) * 2.2 + 0.5;
        let fleck = grain > self.fleck_threshold;
        let micro = (grain - 0.5).clamp(-0.5, 0.5) * 0.08;

        let height =
            (ripple * 0.75 + 0.10 + micro + if fleck { 0.06 } else { 0.0 }).clamp(0.0, 1.0);

        let t = ripple as f32;
        let bright = if fleck { 1.25 } else { 1.0 };
        let color = [
            (lerp(c.color_trough[0], c.color_crest[0], t) * bright).clamp(0.0, 1.0),
            (lerp(c.color_trough[1], c.color_crest[1], t) * bright).clamp(0.0, 1.0),
            (lerp(c.color_trough[2], c.color_crest[2], t) * bright).clamp(0.0, 1.0),
        ];

        // Sand is very rough; sparkling grains read as small specular hits.
        let rough = (0.93 - t * 0.05 - if fleck { 0.25 } else { 0.0 }).clamp(0.0, 1.0);

        SurfaceSample::matte(height, color, rough)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = SandGenerator::new(SandConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
    }

    #[test]
    fn ripples_produce_relief_and_flecks_appear() {
        let map = SandGenerator::new(SandConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        let flat = map.normal.chunks(4).all(|px| px[0] == 128 && px[1] == 128);
        assert!(!flat, "ripples should produce non-flat normals");
        // Some texels should be markedly smoother (flecks) than the base.
        let smooth = map.roughness.chunks(4).any(|px| px[1] < 180);
        assert!(smooth, "flecks should lower roughness locally");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = SandGenerator::new(SandConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = SandGenerator::new(SandConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
