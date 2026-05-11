//! Plank / siding texture generator using domain-warped anisotropic noise.
//!
//! The algorithm:
//! 1. Divide V into horizontal plank bands.
//! 2. Per band, add a deterministic phase offset to U so grain de-correlates
//!    between planks.
//! 3. Sample anisotropic grain FBM (high U frequency, low V frequency) for the
//!    wood-grain pattern; optionally warp it with a low-frequency FBM.
//! 4. Overlay sparse Worley knots at the scaled knot density.
//! 5. Apply a thin joint gap at plank boundaries.

use std::f64::consts::TAU;

use noise::core::worley::ReturnType;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, Worley};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::ToroidalNoise,
    normal::{BoundaryMode, height_to_normal},
};

/// Configures the appearance of a [`PlankGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PlankConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Number of planks visible vertically.
    pub plank_count: f64,
    /// Grain spatial scale (controls how fine the grain lines are).
    pub grain_scale: f64,
    /// Gap between planks as a fraction of plank height \[0, 0.3\].
    pub joint_width: f64,
    /// Horizontal stagger of end-joints between adjacent planks \[0, 1\].
    pub stagger: f64,
    /// Worley-knot density: fraction of cells that contain a knot \[0, 1\].
    pub knot_density: f64,
    /// Domain-warp strength that bends grain lines \[0, 1\].
    pub grain_warp: f64,
    /// Light wood colour in linear RGB \[0, 1\].
    pub color_wood_light: [f32; 3],
    /// Dark wood colour in linear RGB \[0, 1\].
    pub color_wood_dark: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for PlankConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            plank_count: 5.0,
            grain_scale: 12.0,
            joint_width: 0.06,
            stagger: 0.5,
            knot_density: 0.25,
            grain_warp: 0.35,
            color_wood_light: [0.72, 0.52, 0.30],
            color_wood_dark: [0.42, 0.26, 0.12],
            normal_strength: 2.5,
        }
    }
}

/// Procedural wood-plank / siding texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`PlankConfig`].  Construct
/// via [`PlankGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::plank`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.  Worley knot noise is still
/// constructed in `generate()` because `Worley` is `!Send`.
pub struct PlankGenerator {
    config: PlankConfig,
    fbm_warp: Fbm<Perlin>,
    grain_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl PlankGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: PlankConfig) -> Self {
        let fbm_warp: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(3);
        let grain_fbm: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(100)).set_octaves(5);
        let grain_noise = ToroidalNoise::new(grain_fbm, 1.0);
        Self {
            config,
            fbm_warp,
            grain_noise,
        }
    }
}

impl TextureGenerator for PlankGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // plank_count must be an integer for the grid to tile vertically.
        let plank_count = c.plank_count.round();

        // Worley for knots — constructed here because Worley contains an Rc
        // and is not Send, so it cannot be stored on the struct.
        let worley = Worley::new(c.seed.wrapping_add(200)).set_return_type(ReturnType::Distance);
        let knot_noise = ToroidalNoise::new(worley, plank_count * 1.5);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // Precompute knot Worley grid (isotropic, shared across planks).
        let knot_grid: Vec<f64> = {
            let freq = plank_count * 1.5;
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
            let mut grid = vec![0.0f64; n];
            for y in 0..h {
                for x in 0..w {
                    grid[y * w + x] =
                        knot_noise.get_precomputed(col_cos[x], col_sin[x], row_cos[y], row_sin[y]);
                }
            }
            grid
        };

        // Grain anisotropic frequencies.
        let g_freq_u = c.grain_scale;
        let g_freq_v = c.grain_scale * 0.08; // very low V — long grain lines

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;
            let v_scaled = v * plank_count;
            let y_cell = v_scaled.floor() as i64;
            let v_frac = v_scaled.fract();

            // Per-plank de-correlation phase and stagger.
            let row_phase = cell_hash(y_cell, 0, c.seed);
            let stagger_phase = cell_hash(y_cell, 1, c.seed) * c.stagger;

            // Joint gap at top and bottom of each plank.
            let joint_half = c.joint_width * 0.5;
            let in_joint = v_frac < joint_half || v_frac > 1.0 - joint_half;

            // Precompute row torus coords for grain (V direction, low freq).
            let v_grain = v_frac * 0.1 + row_phase * 0.3; // gently warped per plank
            let g_nz = (TAU * v_grain).cos() * g_freq_v;
            let g_nw = (TAU * v_grain).sin() * g_freq_v;

            for x in 0..w {
                let u = x as f64 / w as f64;

                // Staggered end-joint.
                let u_stagger = (u + stagger_phase).rem_euclid(1.0);
                let stagger_frac = (u_stagger * 3.0).fract(); // ~3 short boards per plank
                let in_end_joint = c.stagger > 0.01
                    && (stagger_frac < c.joint_width * 0.5
                        || stagger_frac > 1.0 - c.joint_width * 0.5);

                let idx = y * w + x;

                if in_joint || in_end_joint {
                    // Joint / shadow line.
                    heights[idx] = 0.0;
                    let ai = idx * 4;
                    let jc = lerp3(c.color_wood_dark, [0.05, 0.03, 0.01], 0.5);
                    albedo[ai] = linear_to_srgb(jc[0]);
                    albedo[ai + 1] = linear_to_srgb(jc[1]);
                    albedo[ai + 2] = linear_to_srgb(jc[2]);
                    albedo[ai + 3] = 255;
                    roughness_buf[ai] = 255;
                    roughness_buf[ai + 1] = (0.92 * 255.0) as u8;
                    roughness_buf[ai + 2] = 0;
                    roughness_buf[ai + 3] = 255;
                    continue;
                }

                // Domain warp: low-freq FBM nudges grain coordinate.
                let warp_u = self.fbm_warp.get([u * 2.0, v * 2.0]) * c.grain_warp * 0.08;

                // Anisotropic grain: per-plank phase shift on U.
                let u_grain = (u + row_phase * 0.7 + warp_u).rem_euclid(1.0);
                let g_nx = (TAU * u_grain).cos() * g_freq_u;
                let g_ny = (TAU * u_grain).sin() * g_freq_u;
                let grain_raw = self.grain_noise.get_precomputed(g_nx, g_ny, g_nz, g_nw);
                let grain_t = normalize(grain_raw); // [0, 1]

                // Knot: Worley cell distance → circular depression.
                let knot_raw = knot_grid[idx];
                // Invert: low distance = near knot centre = depression.
                let knot_t = ((0.5 - knot_raw * 0.5) - (1.0 - c.knot_density))
                    .max(0.0)
                    .min(c.knot_density)
                    / c.knot_density.max(0.01);
                let knot_depression = knot_t.powi(2);

                // Height: grain + knot depression.
                let h_val = (grain_t * (1.0 - knot_depression * 0.6)).clamp(0.0, 1.0);
                heights[idx] = h_val;

                // Colour: lerp light ↔ dark by grain, darken at knots.
                let color_t = (grain_t as f32 - knot_depression as f32 * 0.4).clamp(0.0, 1.0);
                let r = lerp(c.color_wood_dark[0], c.color_wood_light[0], color_t);
                let gr = lerp(c.color_wood_dark[1], c.color_wood_light[1], color_t);
                let b = lerp(c.color_wood_dark[2], c.color_wood_light[2], color_t);

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(gr);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // ORM: knots and dark grain are rougher.
                let rough = 0.50 + (1.0 - color_t) * 0.35;
                roughness_buf[ai] = 255;
                roughness_buf[ai + 1] = (rough * 255.0).round() as u8;
                roughness_buf[ai + 2] = 0;
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

fn cell_hash(cell: i64, salt: u64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (cell as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= salt.wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

#[inline]
fn normalize(v: f64) -> f64 {
    v * 0.5 + 0.5
}
