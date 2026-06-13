//! Woven fabric texture generator.
//!
//! A plain-weave cloth: two perpendicular thread families (vertical warp,
//! horizontal weft) rendered as half-cylinder profiles on an integer thread
//! lattice.  Crossing parity decides which family lies on top, and
//! `weave_contrast` controls how far the under-thread is pressed down —
//! the classic over/under basket texture.  A high-frequency FBM adds fibre
//! fuzz; a low-frequency FBM mottles the colour like slubbed yarn.
//!
//! Threads tile by construction (integer `thread_count` lattice) and both
//! FBM layers are toroidal, so the cloth is seamless.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface},
};

/// Configures the appearance of a [`FabricGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FabricConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Threads per tile edge `[8, 64]` — rounded to an integer so the weave
    /// tiles exactly.
    pub thread_count: f64,
    /// Thread width as a fraction of the lattice cell `[0.3, 0.98]`; lower
    /// values open visible gaps between threads (burlap), higher values
    /// close the weave (tight cotton).
    pub thread_width: f64,
    /// How far the under-thread is pressed down at each crossing `[0, 1]`;
    /// higher values deepen the over/under relief.
    pub weave_contrast: f64,
    /// Fibre fuzz strength `[0, 1]` — high-frequency height and roughness
    /// jitter that softens the cylindrical thread shading.
    pub fuzz: f64,
    /// Warp (vertical) thread colour in linear RGB \[0, 1\].
    pub color_warp: [f32; 3],
    /// Weft (horizontal) thread colour in linear RGB \[0, 1\].  Match the
    /// warp colour for solid cloth, contrast it for shot/two-tone weaves.
    pub color_weft: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self {
            seed: 29,
            thread_count: 24.0,
            thread_width: 0.85,
            weave_contrast: 0.6,
            fuzz: 0.35,
            color_warp: [0.55, 0.36, 0.24],
            color_weft: [0.62, 0.44, 0.30],
            normal_strength: 3.0,
        }
    }
}

/// Procedural woven-fabric texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`FabricConfig`].  Construct
/// via [`FabricGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::fabric`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct FabricGenerator {
    config: FabricConfig,
    fuzz_noise: ToroidalNoise<Fbm<Perlin>>,
    mottle_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl FabricGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: FabricConfig) -> Self {
        let fbm_fuzz: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(50)).set_octaves(4);
        let fuzz_noise = ToroidalNoise::new(fbm_fuzz, config.thread_count.max(1.0) * 1.5);
        let fbm_mottle: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(3);
        let mottle_noise = ToroidalNoise::new(fbm_mottle, 3.0);
        Self {
            config,
            fuzz_noise,
            mottle_noise,
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

        let mut fuzz_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.fuzz_noise, width, height, &mut fuzz_grid);

        let mut mottle_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.mottle_noise, width, height, &mut mottle_grid);

        let cell = FabricCell {
            config: c,
            fuzz_grid: &fuzz_grid,
            mottle_grid: &mottle_grid,
            threads: c.thread_count.round().clamp(2.0, 128.0),
            width: width as usize,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        if let Some(ws) = ws {
            ws.return_grid(fuzz_grid);
            ws.return_grid(mottle_grid);
        }
        result
    }
}

impl TextureGenerator for FabricGenerator {
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

/// Per-generation sampler: fuzz + mottle grids and the rounded lattice size.
struct FabricCell<'a> {
    config: &'a FabricConfig,
    fuzz_grid: &'a [f64],
    mottle_grid: &'a [f64],
    /// `thread_count` rounded so the lattice tiles exactly.
    threads: f64,
    width: usize,
}

impl SurfaceCell for FabricCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let idx = y as usize * self.width + x as usize;

        let su = u * self.threads;
        let sv = v * self.threads;
        let i = su.floor() as i64;
        let j = sv.floor() as i64;
        let fu = su - su.floor();
        let fv = sv - sv.floor();

        // Half-cylinder thread profile across the cell.
        let tw = c.thread_width.clamp(0.3, 0.98);
        let profile = |f: f64| {
            let d = (f - 0.5).abs() * 2.0;
            if d < tw {
                (1.0 - (d / tw) * (d / tw)).sqrt()
            } else {
                0.0
            }
        };
        let warp_p = profile(fu); // vertical threads
        let weft_p = profile(fv); // horizontal threads

        // Plain weave: crossing parity decides the top thread; the
        // under-thread is pressed down by the weave contrast.
        let contrast = c.weave_contrast.clamp(0.0, 1.0);
        let under = 1.0 - 0.4 * contrast;
        let (warp_lift, weft_lift) = if (i + j).rem_euclid(2) == 0 {
            (under, 1.0)
        } else {
            (1.0, under)
        };
        let warp_h = warp_p * warp_lift;
        let weft_h = weft_p * weft_lift;
        let thread_h = warp_h.max(weft_h);

        let fuzz = normalize(self.fuzz_grid[idx]);
        let mottle = normalize(self.mottle_grid[idx]);

        // Gaps between threads sit near 0 (backing shadow); fibre fuzz
        // breaks up the clean cylinder shading.
        let height =
            (thread_h * 0.85 + 0.05 + (fuzz - 0.5) * 0.2 * c.fuzz.clamp(0.0, 1.0)).clamp(0.0, 1.0);

        // Whichever family is higher at this texel shows its colour, shaded
        // by the thread curvature and tinted by the yarn mottle.
        let base = if warp_h >= weft_h {
            c.color_warp
        } else {
            c.color_weft
        };
        let shade = (0.45 + 0.55 * thread_h) as f32;
        let tint = 1.0 + (mottle as f32 - 0.5) * 0.25;
        let color = [
            (base[0] * shade * tint).clamp(0.0, 1.0),
            (base[1] * shade * tint).clamp(0.0, 1.0),
            (base[2] * shade * tint).clamp(0.0, 1.0),
        ];

        // Cloth is matte; crests are fractionally smoother than the gaps.
        let rough = (0.88 - thread_h as f32 * 0.10
            + (fuzz as f32 - 0.5) * 0.12 * c.fuzz.clamp(0.0, 1.0) as f32)
            .clamp(0.0, 1.0);

        SurfaceSample::matte(height, color, rough)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = FabricGenerator::new(FabricConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn weave_relief_varies() {
        // The normal map must not be flat — the weave produces relief.
        let map = FabricGenerator::new(FabricConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        let flat = map.normal.chunks(4).all(|px| px[0] == 128 && px[1] == 128);
        assert!(!flat, "weave should produce non-flat normals");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = FabricGenerator::new(FabricConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = FabricGenerator::new(FabricConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.normal, b.normal);
    }

    #[test]
    fn two_tone_weave_shows_both_colours() {
        let config = FabricConfig {
            color_warp: [1.0, 0.0, 0.0],
            color_weft: [0.0, 0.0, 1.0],
            fuzz: 0.0,
            ..FabricConfig::default()
        };
        let map = FabricGenerator::new(config)
            .generate(128, 128)
            .expect("generate failed");
        let mut saw_red = false;
        let mut saw_blue = false;
        for px in map.albedo.chunks(4) {
            if px[0] > 100 && px[2] < 50 {
                saw_red = true;
            }
            if px[2] > 100 && px[0] < 50 {
                saw_blue = true;
            }
        }
        assert!(saw_red && saw_blue, "both thread families must be visible");
    }
}
