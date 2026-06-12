//! Marble / granite texture generator using domain-warped FBM noise.
//!
//! The algorithm:
//!  1. Precompute toroidal sin/cos lookup tables (one entry per column, one per row).
//!  2. Build three independent FBM sources: two warp layers (warp_u, warp_v) and a
//!     base layer, all evaluated on the same 4-D torus so the result tiles seamlessly.
//!  3. The base FBM is baked into a W×H grid via [`sample_grid_into`].  Warp offsets
//!     are computed inline using the precomputed lookup tables.
//!  4. For each pixel the base grid is sampled at the domain-warped UV coordinates
//!     via bilinear interpolation.  The raw value is normalised to [0, 1] and passed
//!     through a sinusoidal vein function that produces thin dark veins on a light
//!     background.
//!  5. Colour, height, and ORM values are derived from the vein blend factor.
//!
//! Vein function: `((raw_norm · TAU · vein_frequency).sin().abs()).powf(vein_sharpness)`
//! gives 0 at vein centres and 1 on the background rock.  The two endpoints are then
//! mapped to `color_vein` and `color_base` respectively.

use std::f64::consts::TAU;

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, bilinear_sample_torus, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`MarbleGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MarbleConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Overall pattern scale. \[1, 8\]
    pub scale: f64,
    /// FBM octaves for the base layer. \[3, 8\]
    pub octaves: usize,
    /// Domain warp strength — how much the veins meander. \[0, 1.5\]
    pub warp_strength: f64,
    /// Vein frequency: period of sin() applied to warped FBM. \[1, 8\]
    pub vein_frequency: f64,
    /// Vein sharpness: exponent on abs(sin()) making veins narrower. \[0.5, 6\]
    pub vein_sharpness: f64,
    /// Base surface roughness \[0, 0.3\] — keep low for polished marble.
    pub roughness: f64,
    /// Base (light) colour in linear RGB.
    pub color_base: [f32; 3],
    /// Vein colour in linear RGB.
    pub color_vein: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for MarbleConfig {
    fn default() -> Self {
        Self {
            seed: 55,
            scale: 3.0,
            octaves: 5,
            warp_strength: 0.6,
            vein_frequency: 3.0,
            vein_sharpness: 2.0,
            roughness: 0.08,
            color_base: [0.92, 0.90, 0.87],
            color_vein: [0.42, 0.38, 0.34],
            normal_strength: 1.5,
        }
    }
}

/// Procedural marble / granite texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`MarbleConfig`].  Construct
/// via [`MarbleGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::marble`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct MarbleGenerator {
    config: MarbleConfig,
    warp_u_noise: ToroidalNoise<Fbm<Perlin>>,
    warp_v_noise: ToroidalNoise<Fbm<Perlin>>,
    base_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl MarbleGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: MarbleConfig) -> Self {
        let fbm_warp_u: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(config.octaves);
        let fbm_warp_v: Fbm<Perlin> =
            Fbm::new(config.seed.wrapping_add(100)).set_octaves(config.octaves);
        let fbm_base: Fbm<Perlin> =
            Fbm::new(config.seed.wrapping_add(200)).set_octaves(config.octaves);
        let warp_u_noise = ToroidalNoise::new(fbm_warp_u, config.scale);
        let warp_v_noise = ToroidalNoise::new(fbm_warp_v, config.scale);
        let base_noise = ToroidalNoise::new(fbm_base, config.scale);
        Self {
            config,
            warp_u_noise,
            warp_v_noise,
            base_noise,
        }
    }
}

/// Per-generation sampler: torus LUTs, precomputed base grid, warp noises.
struct MarbleCell<'a> {
    config: &'a MarbleConfig,
    warp_u_noise: &'a ToroidalNoise<Fbm<Perlin>>,
    warp_v_noise: &'a ToroidalNoise<Fbm<Perlin>>,
    base_grid: &'a [f64],
    col_cos: &'a [f64],
    col_sin: &'a [f64],
    row_cos: &'a [f64],
    row_sin: &'a [f64],
    w: usize,
    h: usize,
}

impl SurfaceCell for MarbleCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let (x, y) = (x as usize, y as usize);

        // Compute warp offsets using precomputed torus coordinates.
        let nx = self.col_cos[x];
        let ny = self.col_sin[x];
        let nz = self.row_cos[y];
        let nw = self.row_sin[y];
        let du = self.warp_u_noise.get_precomputed(nx, ny, nz, nw) * c.warp_strength;
        let dv = self.warp_v_noise.get_precomputed(nx, ny, nz, nw) * c.warp_strength;

        // Sample the precomputed base grid at the warped UV coordinates.
        // Bilinear interpolation wraps toroidally — no trig per pixel.
        let raw = bilinear_sample_torus(self.base_grid, self.w, self.h, u + du, v + dv);

        // Normalize raw from [-1, 1] to [0, 1] before the vein function.
        let raw_norm = raw * 0.5 + 0.5;

        // Vein pattern: 0 = vein centre, 1 = background rock.
        // High vein_sharpness concentrates the zero region into thin dark lines.
        let vein_t = ((raw_norm * TAU * c.vein_frequency).sin().abs()).powf(c.vein_sharpness);

        // Colour: lerp from vein colour to base rock colour by vein_t.
        let vein_tf = vein_t as f32;
        let color = [
            lerp(c.color_vein[0], c.color_base[0], vein_tf),
            lerp(c.color_vein[1], c.color_base[1], vein_tf),
            lerp(c.color_vein[2], c.color_base[2], vein_tf),
        ];

        // ORM: polished marble is very smooth; veins are only marginally
        // rougher because they are recessed and accumulate micro-debris.
        let rough = (c.roughness * (1.0 - vein_t * 0.3)) as f32;

        // Height: veins are slightly recessed relative to the base rock.
        SurfaceSample::matte(vein_t * 0.8 + normalize(raw) * 0.2, color, rough)
    }
}

impl MarbleGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let w = width as usize;
        let h = height as usize;

        // Precompute toroidal coordinates (W + H entries instead of W × H).
        // All three noise objects share the same `c.scale` frequency so one
        // set of lookup tables covers all of them.
        let freq = c.scale;
        let col_cos: Vec<f64> = (0..w)
            .map(|x| (TAU * x as f64 / w as f64).cos() * freq)
            .collect();
        let col_sin: Vec<f64> = (0..w)
            .map(|x| (TAU * x as f64 / w as f64).sin() * freq)
            .collect();
        let row_cos: Vec<f64> = (0..h)
            .map(|y| (TAU * y as f64 / h as f64).cos() * freq)
            .collect();
        let row_sin: Vec<f64> = (0..h)
            .map(|y| (TAU * y as f64 / h as f64).sin() * freq)
            .collect();

        // Precompute the base noise on a regular grid using the torus LUTs
        // (O(W+H) trig calls).  The warped lookup then becomes a cheap
        // bilinear interpolation rather than per-pixel sin/cos evaluation.
        let mut base_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.base_noise, width, height, &mut base_grid);

        let cell = MarbleCell {
            config: c,
            warp_u_noise: &self.warp_u_noise,
            warp_v_noise: &self.warp_v_noise,
            base_grid: &base_grid,
            col_cos: &col_cos,
            col_sin: &col_sin,
            row_cos: &row_cos,
            row_sin: &row_sin,
            w,
            h,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        if let Some(ws) = ws {
            ws.return_grid(base_grid);
        }
        result
    }
}

impl TextureGenerator for MarbleGenerator {
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
