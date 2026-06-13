//! Lava texture generator — basalt crust with emissive cracks.
//!
//! Cooling lava: dark basalt plates from a toroidal Voronoi decomposition
//! (domed `F1` plates, like cobblestone) separated by molten cracks where
//! the `F2 − F1` boundary field collapses.  The crack field drives the
//! crate's **emissive channel**: the glow colour ramps with crack depth and
//! is written into [`TextureMap::emissive`], which the polling systems
//! assign to `StandardMaterial::emissive_texture`.  Set
//! `MaterialSettings::emission_color` to white and `emission_strength`
//! above zero, or the glow map is multiplied away (see the emissive docs).
//!
//! The albedo carries a faint heat tint near the cracks so the material
//! still reads as hot without bloom.
//!
//! [`TextureMap::emissive`]: crate::generator::TextureMap::emissive

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, cell_hash, normalize, sample_grid_into, toroidal_voronoi},
    surface::{SurfaceCell, SurfaceSample, generate_surface_emissive, lerp},
};

/// Configures the appearance of a [`LavaGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LavaConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Approximate number of crust plates across the tile `[3, 12]`.
    pub plate_scale: f64,
    /// Molten-crack width as a fraction of plate spacing `[0.02, 0.3]`.
    pub crack_width: f64,
    /// Glow falloff exponent `[0.5, 4]` — higher concentrates the glow into
    /// the crack centres.
    pub glow_falloff: f64,
    /// Cooled basalt crust colour in linear RGB \[0, 1\].
    pub color_crust: [f32; 3],
    /// Molten glow colour in linear RGB \[0, 1\] — written to the emissive
    /// map at full crack depth.
    pub color_glow: [f32; 3],
    /// Emissive intensity multiplier `[0, 4]` applied to the glow colour
    /// before it is encoded into the emissive map.
    pub emissive_intensity: f32,
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for LavaConfig {
    fn default() -> Self {
        Self {
            seed: 666,
            plate_scale: 6.0,
            crack_width: 0.14,
            glow_falloff: 1.6,
            color_crust: [0.08, 0.07, 0.07],
            color_glow: [1.0, 0.45, 0.06],
            emissive_intensity: 1.0,
            normal_strength: 4.0,
        }
    }
}

/// Procedural lava texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`LavaConfig`].  Construct
/// via [`LavaGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::lava`] task for non-blocking generation.
pub struct LavaGenerator {
    config: LavaConfig,
    surf_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl LavaGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: LavaConfig) -> Self {
        let fbm_surf: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(50)).set_octaves(4);
        let surf_noise = ToroidalNoise::new(fbm_surf, config.plate_scale * 2.0);
        Self { config, surf_noise }
    }

    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let mut surf_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.surf_noise, width, height, &mut surf_grid);

        let scale = c.plate_scale.round().clamp(2.0, 16.0);
        let cell = LavaCell {
            config: c,
            surf_grid: &surf_grid,
            scale,
            crack_threshold: c.crack_width.clamp(0.02, 0.3) / scale,
            width: width as usize,
        };
        let result =
            generate_surface_emissive(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        if let Some(ws) = ws {
            ws.return_grid(surf_grid);
        }
        result
    }
}

impl TextureGenerator for LavaGenerator {
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

/// Per-generation sampler: crust micro-detail grid + Voronoi constants.
struct LavaCell<'a> {
    config: &'a LavaConfig,
    surf_grid: &'a [f64],
    /// `plate_scale` rounded so the Voronoi lattice tiles.
    scale: f64,
    /// `F2 − F1` below this is molten crack.
    crack_threshold: f64,
    width: usize,
}

impl SurfaceCell for LavaCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let idx = y as usize * self.width + x as usize;
        let micro = normalize(self.surf_grid[idx]);

        let (f1, f2, ci, cj) = toroidal_voronoi(u, v, self.scale, c.seed);
        let gap = f2 - f1;

        // Glow: 1 at the crack centre, fading into the plates.
        let glow = ((1.0 - gap / (self.crack_threshold * 2.0)).clamp(0.0, 1.0))
            .powf(c.glow_falloff.clamp(0.5, 4.0));
        let molten = gap < self.crack_threshold;

        // Plates dome gently; cracks sink.
        let dome = (1.0 - (f1 * self.scale).powf(1.3)).clamp(0.0, 1.0);
        let height = if molten {
            0.12 + micro * 0.04
        } else {
            (0.30 + dome * 0.55 + micro * 0.10).clamp(0.0, 1.0)
        };

        // Albedo: dark basalt with per-plate variance and a faint heat tint
        // toward the cracks; the molten channel itself shows the glow
        // colour dimmed (the emissive map carries the actual light).
        let plate_jitter = ((cell_hash(ci, cj, c.seed.wrapping_add(99)) - 0.5) * 0.10) as f32;
        let heat = (glow * 0.45) as f32;
        let color = [
            (lerp(
                c.color_crust[0] + plate_jitter,
                c.color_glow[0] * 0.55,
                heat,
            ))
            .clamp(0.0, 1.0),
            (lerp(
                c.color_crust[1] + plate_jitter,
                c.color_glow[1] * 0.55,
                heat,
            ))
            .clamp(0.0, 1.0),
            (lerp(
                c.color_crust[2] + plate_jitter,
                c.color_glow[2] * 0.55,
                heat,
            ))
            .clamp(0.0, 1.0),
        ];

        // Molten rock is glassy; crust is rough and dusty.
        let rough = if molten {
            (0.18 - glow as f32 * 0.08).clamp(0.05, 1.0)
        } else {
            (0.85 + (micro as f32 - 0.5) * 0.1).clamp(0.0, 1.0)
        };

        // Emissive: glow colour scaled by intensity, strongest in cracks
        // and flickering slightly with the micro noise.
        let intensity = c.emissive_intensity.clamp(0.0, 4.0);
        let e = (glow as f32) * intensity * (0.85 + micro as f32 * 0.3);
        let emissive = [
            (c.color_glow[0] * e).clamp(0.0, 1.0),
            (c.color_glow[1] * e).clamp(0.0, 1.0),
            (c.color_glow[2] * e).clamp(0.0, 1.0),
        ];

        SurfaceSample {
            height,
            color,
            roughness: rough,
            metallic: 0.0,
            occlusion: 1.0,
            emissive,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_emissive_map() {
        let map = LavaGenerator::new(LavaConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        let emissive = map.emissive.as_ref().expect("lava must emit");
        assert_eq!(emissive.len(), 64 * 64 * 4);
        // Cracks glow, plate interiors stay dark.
        let lit = emissive.chunks(4).filter(|px| px[0] > 120).count();
        let dark = emissive.chunks(4).filter(|px| px[0] < 20).count();
        assert!(lit > 0, "cracks should glow in the emissive map");
        assert!(dark > lit, "plate interiors should dominate and stay dark");
    }

    #[test]
    fn molten_cracks_are_glassy_against_rough_crust() {
        let map = LavaGenerator::new(LavaConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        let glassy = map.roughness.chunks(4).filter(|px| px[1] < 60).count();
        let crusty = map.roughness.chunks(4).filter(|px| px[1] > 180).count();
        assert!(glassy > 0, "molten channels should be glassy");
        assert!(crusty > glassy, "basalt crust should dominate");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = LavaGenerator::new(LavaConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = LavaGenerator::new(LavaConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.emissive, b.emissive);
    }
}
