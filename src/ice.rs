//! Ice surface texture generator.
//!
//! Polished lake ice: a near-mirror pale-blue base crossed by thin internal
//! crack veins (sinusoidal bands over FBM iso-contours — the same vein
//! machinery as marble, inverted and sharpened), with optional frost
//! patches that locally whiten the colour and raise the roughness toward
//! matte.  Cracks are slightly recessed and darkened, faking the depth tint
//! of fractures seen through the surface.

use std::f64::consts::TAU;

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of an [`IceGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IceConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Overall pattern scale `[1, 8]`.
    pub scale: f64,
    /// Crack-vein frequency `[1, 8]` — period of the sinusoidal bands over
    /// the noise contours; higher = more cracks.
    pub crack_density: f64,
    /// Vein sharpness `[2, 12]` — exponent narrowing the cracks; higher =
    /// hairline fractures.
    pub vein_sharpness: f64,
    /// Frost coverage `[0, 1]` — patches that whiten the surface and raise
    /// roughness toward matte.
    pub frost_level: f64,
    /// Clear-ice colour in linear RGB \[0, 1\] (typically pale blue).
    pub color_ice: [f32; 3],
    /// Crack colour in linear RGB \[0, 1\] (deep fracture tint).
    pub color_crack: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for IceConfig {
    fn default() -> Self {
        Self {
            seed: 117,
            scale: 3.0,
            crack_density: 4.0,
            vein_sharpness: 7.0,
            frost_level: 0.25,
            color_ice: [0.72, 0.84, 0.94],
            color_crack: [0.30, 0.44, 0.62],
            normal_strength: 1.5,
        }
    }
}

/// Procedural ice texture generator.
///
/// Drives [`TextureGenerator::generate`] using an [`IceConfig`].  Construct
/// via [`IceGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::ice`] task for non-blocking generation.
pub struct IceGenerator {
    config: IceConfig,
    base_noise: ToroidalNoise<Fbm<Perlin>>,
    frost_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl IceGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: IceConfig) -> Self {
        let fbm_base: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(4);
        let base_noise = ToroidalNoise::new(fbm_base, config.scale.clamp(0.5, 10.0));
        let fbm_frost: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(160)).set_octaves(3);
        let frost_noise = ToroidalNoise::new(fbm_frost, config.scale.clamp(0.5, 10.0) * 1.7);
        Self {
            config,
            base_noise,
            frost_noise,
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

        let mut base_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.base_noise, width, height, &mut base_grid);

        let mut frost_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.frost_noise, width, height, &mut frost_grid);

        let cell = IceCell {
            config: c,
            base_grid: &base_grid,
            frost_grid: &frost_grid,
            width: width as usize,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        if let Some(ws) = ws {
            ws.return_grid(base_grid);
            ws.return_grid(frost_grid);
        }
        result
    }
}

impl TextureGenerator for IceGenerator {
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

/// Per-generation sampler: base + frost grids.
struct IceCell<'a> {
    config: &'a IceConfig,
    base_grid: &'a [f64],
    frost_grid: &'a [f64],
    width: usize,
}

impl SurfaceCell for IceCell<'_> {
    fn sample(&self, x: u32, y: u32, _u: f64, _v: f64) -> SurfaceSample {
        let c = self.config;
        let idx = y as usize * self.width + x as usize;

        // Crack veins: sinusoidal bands over the FBM iso-contours (the
        // contours meander, so the cracks do too).  `vein` is 1 at a crack
        // centre and 0 in clear ice.
        let raw_norm = normalize(self.base_grid[idx]);
        let band = (raw_norm * TAU * c.crack_density.clamp(1.0, 8.0))
            .sin()
            .abs();
        let vein = 1.0 - band.powf(c.vein_sharpness.clamp(2.0, 12.0));

        // Frost patches: soft threshold over the second FBM.
        let frost_level = c.frost_level.clamp(0.0, 1.0);
        let frost_raw = normalize(self.frost_grid[idx]);
        let frost = if frost_level > 0.0 {
            ((frost_raw - (1.0 - frost_level)) / frost_level).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Cracks are recessed; frost lies fractionally proud.
        let height = (0.85 - vein * 0.25 + frost * 0.05).clamp(0.0, 1.0);

        // Colour: clear ice deepens toward the crack tint inside veins, and
        // whitens under frost.
        let vein_f = vein as f32;
        let frost_f = frost as f32;
        let frost_white = [0.94f32, 0.96, 1.0];
        let color = [
            lerp(
                lerp(c.color_ice[0], c.color_crack[0], vein_f),
                frost_white[0],
                frost_f * 0.8,
            ),
            lerp(
                lerp(c.color_ice[1], c.color_crack[1], vein_f),
                frost_white[1],
                frost_f * 0.8,
            ),
            lerp(
                lerp(c.color_ice[2], c.color_crack[2], vein_f),
                frost_white[2],
                frost_f * 0.8,
            ),
        ];

        // Polished base, rougher inside cracks, matte under frost.
        let rough = (0.05 + vein_f * 0.25 + frost_f * 0.55).clamp(0.0, 1.0);

        SurfaceSample::matte(height, color, rough)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = IceGenerator::new(IceConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
    }

    #[test]
    fn surface_is_mostly_polished_with_rough_features() {
        let map = IceGenerator::new(IceConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        let polished = map.roughness.chunks(4).filter(|px| px[1] < 40).count();
        let rough = map.roughness.chunks(4).filter(|px| px[1] > 80).count();
        assert!(polished > 0, "clear ice should be near-mirror");
        assert!(rough > 0, "cracks/frost should raise roughness locally");
        assert!(polished > rough / 4, "polish should be a major fraction");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = IceGenerator::new(IceConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = IceGenerator::new(IceConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
