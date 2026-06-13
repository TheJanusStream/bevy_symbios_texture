//! Snow surface texture generator.
//!
//! Wind-drifted snow: soft low-frequency FBM relief with a cool shadow tint
//! in the troughs, plus thresholded sparkle flecks — individual crystals
//! that brighten the albedo and drop the ORM roughness to near zero so they
//! catch specular glints.  Both layers are toroidal and tile seamlessly.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`SnowGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SnowConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Spatial scale of the drift relief `[1, 6]`.
    pub drift_scale: f64,
    /// FBM octave count for the drift layer `[2, 6]`.
    pub drift_octaves: usize,
    /// Sparkle-fleck density `[0, 0.5]` — fraction of texels reading as
    /// glinting crystals.
    pub sparkle_density: f64,
    /// Base roughness of the snow crust `[0.5, 1]`; fresh powder is near 1,
    /// refrozen crust lower.
    pub crust_roughness: f64,
    /// Lit snow colour in linear RGB \[0, 1\].
    pub color_snow: [f32; 3],
    /// Trough / shadow tint in linear RGB \[0, 1\] — typically cool blue.
    pub color_shadow: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for SnowConfig {
    fn default() -> Self {
        Self {
            seed: 73,
            drift_scale: 2.5,
            drift_octaves: 4,
            sparkle_density: 0.08,
            crust_roughness: 0.85,
            color_snow: [0.93, 0.95, 0.99],
            color_shadow: [0.62, 0.70, 0.86],
            normal_strength: 1.8,
        }
    }
}

/// Procedural snow texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`SnowConfig`].  Construct
/// via [`SnowGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::snow`] task for non-blocking generation.
pub struct SnowGenerator {
    config: SnowConfig,
    drift_noise: ToroidalNoise<Fbm<Perlin>>,
    sparkle_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl SnowGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: SnowConfig) -> Self {
        let fbm_drift: Fbm<Perlin> =
            Fbm::new(config.seed).set_octaves(config.drift_octaves.clamp(1, 8));
        let drift_noise = ToroidalNoise::new(fbm_drift, config.drift_scale.clamp(0.5, 8.0));
        let fbm_sparkle: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(120)).set_octaves(2);
        let sparkle_noise = ToroidalNoise::new(fbm_sparkle, 36.0);
        Self {
            config,
            drift_noise,
            sparkle_noise,
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

        let mut drift_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.drift_noise, width, height, &mut drift_grid);

        let mut sparkle_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.sparkle_noise, width, height, &mut sparkle_grid);

        let cell = SnowCell {
            config: c,
            drift_grid: &drift_grid,
            sparkle_grid: &sparkle_grid,
            sparkle_threshold: 1.0 - c.sparkle_density.clamp(0.0, 0.5),
            width: width as usize,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        if let Some(ws) = ws {
            ws.return_grid(drift_grid);
            ws.return_grid(sparkle_grid);
        }
        result
    }
}

impl TextureGenerator for SnowGenerator {
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

/// Per-generation sampler: drift + sparkle grids and the fleck threshold.
struct SnowCell<'a> {
    config: &'a SnowConfig,
    drift_grid: &'a [f64],
    sparkle_grid: &'a [f64],
    /// Stretched sparkle values above this threshold glint.
    sparkle_threshold: f64,
    width: usize,
}

impl SurfaceCell for SnowCell<'_> {
    fn sample(&self, x: u32, y: u32, _u: f64, _v: f64) -> SurfaceSample {
        let c = self.config;
        let idx = y as usize * self.width + x as usize;

        let drift = normalize(self.drift_grid[idx]);

        // Distribution-stretched sparkle field (see sand.rs for rationale).
        let sparkle_n = (normalize(self.sparkle_grid[idx]) - 0.5) * 2.2 + 0.5;
        let sparkle = sparkle_n > self.sparkle_threshold;

        let height = (drift * 0.8 + 0.1 + if sparkle { 0.04 } else { 0.0 }).clamp(0.0, 1.0);

        let t = drift as f32;
        let bright = if sparkle { 1.15 } else { 1.0 };
        let color = [
            (lerp(c.color_shadow[0], c.color_snow[0], t) * bright).clamp(0.0, 1.0),
            (lerp(c.color_shadow[1], c.color_snow[1], t) * bright).clamp(0.0, 1.0),
            (lerp(c.color_shadow[2], c.color_snow[2], t) * bright).clamp(0.0, 1.0),
        ];

        // Crystals glint: near-zero roughness at sparkle texels against the
        // matte crust.
        let rough = if sparkle {
            0.06
        } else {
            (c.crust_roughness.clamp(0.5, 1.0) as f32 - t * 0.05).clamp(0.0, 1.0)
        };

        SurfaceSample::matte(height, color, rough)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = SnowGenerator::new(SnowConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
    }

    #[test]
    fn sparkles_glint_against_matte_crust() {
        let map = SnowGenerator::new(SnowConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        let glints = map.roughness.chunks(4).filter(|px| px[1] < 40).count();
        let matte = map.roughness.chunks(4).filter(|px| px[1] > 150).count();
        assert!(glints > 0, "sparkle texels should be near-smooth");
        assert!(matte > glints, "crust should dominate the surface");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = SnowGenerator::new(SnowConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = SnowGenerator::new(SnowConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
