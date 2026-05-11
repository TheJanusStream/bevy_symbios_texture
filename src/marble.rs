//! Marble / granite texture generator using domain-warped FBM noise.
//!
//! The algorithm:
//!  1. Precompute toroidal sin/cos lookup tables (one entry per column, one per row).
//!  2. Build three independent FBM sources: two warp layers (warp_u, warp_v) and a
//!     base layer, all evaluated on the same 4-D torus so the result tiles seamlessly.
//!  3. The base FBM is baked into a W×H grid via [`sample_grid`].  Warp offsets are
//!     computed inline using the precomputed lookup tables.
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
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::{BoundaryMode, height_to_normal},
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

impl TextureGenerator for MarbleGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

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
        let base_grid = sample_grid(&self.base_noise, width, height);

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let nz = row_cos[y];
            let nw = row_sin[y];
            let v = y as f64 / h as f64;

            for x in 0..w {
                let nx = col_cos[x];
                let ny = col_sin[x];
                let u = x as f64 / w as f64;

                // Compute warp offsets using precomputed torus coordinates.
                let du = self.warp_u_noise.get_precomputed(nx, ny, nz, nw) * c.warp_strength;
                let dv = self.warp_v_noise.get_precomputed(nx, ny, nz, nw) * c.warp_strength;

                // Sample the precomputed base grid at the warped UV coordinates.
                // Bilinear interpolation wraps toroidally — no trig per pixel.
                let raw = bilinear_sample_torus(&base_grid, w, h, u + du, v + dv);

                // Normalize raw from [-1, 1] to [0, 1] before the vein function.
                let raw_norm = raw * 0.5 + 0.5;

                // Vein pattern: 0 = vein centre, 1 = background rock.
                // High vein_sharpness concentrates the zero region into thin dark lines.
                let vein_t =
                    ((raw_norm * TAU * c.vein_frequency).sin().abs()).powf(c.vein_sharpness);

                let idx = y * w + x;

                // Height: veins are slightly recessed relative to the base rock.
                heights[idx] = vein_t * 0.8 + normalize(raw) * 0.2;

                // Colour: lerp from vein colour to base rock colour by vein_t.
                let vein_tf = vein_t as f32;
                let r = lerp(c.color_vein[0], c.color_base[0], vein_tf);
                let g = lerp(c.color_vein[1], c.color_base[1], vein_tf);
                let b = lerp(c.color_vein[2], c.color_base[2], vein_tf);

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // ORM: polished marble is very smooth; veins are only marginally rougher.
                // Roughness = c.roughness * (1 - vein_t * 0.3): veins are fractionally
                // rougher because they are recessed and accumulate micro-debris.
                let rough = (c.roughness * (1.0 - vein_t * 0.3)) as f32;
                roughness_buf[ai] = 255; // Occlusion = 1.0
                roughness_buf[ai + 1] = (rough * 255.0).round() as u8;
                roughness_buf[ai + 2] = 0; // Metallic = 0.0
                roughness_buf[ai + 3] = 255;
            }
        }

        let normal = height_to_normal(
            &heights,
            width,
            height,
            c.normal_strength,
            BoundaryMode::Wrap,
        );

        Ok(TextureMap {
            albedo,
            normal,
            roughness: roughness_buf,
            width,
            height,
        })
    }
}

// --- helpers ----------------------------------------------------------------

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Bilinearly interpolate a value from a toroidal (seamlessly tiling) grid.
///
/// `u` and `v` are in UV space and may fall outside `[0, 1]`; they are wrapped
/// before sampling so the lookup is always valid.  Used to fetch the
/// domain-warped base noise value without additional `sin`/`cos` calls.
#[inline]
fn bilinear_sample_torus(grid: &[f64], w: usize, h: usize, u: f64, v: f64) -> f64 {
    let u = u.rem_euclid(1.0);
    let v = v.rem_euclid(1.0);
    let px = u * w as f64;
    let py = v * h as f64;
    let x0 = px as usize % w;
    let y0 = py as usize % h;
    let x1 = (x0 + 1) % w;
    let y1 = (y0 + 1) % h;
    let fx = px.fract();
    let fy = py.fract();
    let v00 = grid[y0 * w + x0];
    let v10 = grid[y0 * w + x1];
    let v01 = grid[y1 * w + x0];
    let v11 = grid[y1 * w + x1];
    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}
