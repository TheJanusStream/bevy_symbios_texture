//! Bark texture generator using domain-warped FBM noise.
//!
//! The algorithm:
//!  1. Precompute toroidal sin/cos lookup tables (one entry per column, one per row).
//!  2. For each pixel, sample two FBM warp layers inline to produce offsets (du, dv).
//!  3. Sample a third FBM layer at the warped UV coordinates for the final value.
//!  4. Derive colour, roughness and a height field from the result.
//!
//! Computing the warp layers inline (rather than storing full W×H grids) avoids
//! two large intermediate allocations that would otherwise total ~1 GB at 8 K.

use std::f64::consts::TAU;

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::ToroidalNoise,
    normal::height_to_normal,
};

/// Configures the appearance of a [`BarkGenerator`].
#[derive(Clone, Debug)]
pub struct BarkConfig {
    pub seed: u32,
    /// Overall spatial scale of the bark pattern.
    pub scale: f64,
    /// Octaves for the base FBM layer.
    pub octaves: usize,
    /// Horizontal warp strength (small — creates slight lateral texture).
    pub warp_u: f64,
    /// Vertical warp strength (large — creates the fibrous streaks).
    pub warp_v: f64,
    /// Base (light) bark colour in linear RGB [0,1].
    pub color_light: [f32; 3],
    /// Dark groove colour in linear RGB [0,1].
    pub color_dark: [f32; 3],
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for BarkConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            scale: 4.0,
            octaves: 6,
            warp_u: 0.15,
            warp_v: 0.55,
            color_light: [0.45, 0.28, 0.14],
            color_dark: [0.18, 0.10, 0.05],
            normal_strength: 3.0,
        }
    }
}

pub struct BarkGenerator {
    config: BarkConfig,
}

impl BarkGenerator {
    pub fn new(config: BarkConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for BarkGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Three independent FBM sources with offset seeds.
        let fbm_warp_u: Fbm<Perlin> = Fbm::new(c.seed).set_octaves(c.octaves);
        let fbm_warp_v: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(100)).set_octaves(c.octaves);
        let fbm_base: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(200)).set_octaves(c.octaves);

        let warp_u_noise = ToroidalNoise::new(fbm_warp_u, c.scale);
        let warp_v_noise = ToroidalNoise::new(fbm_warp_v, c.scale);
        let base_noise = ToroidalNoise::new(fbm_base, c.scale);

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

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness = vec![0u8; n * 4];

        for y in 0..h {
            let nz = row_cos[y];
            let nw = row_sin[y];
            let v = y as f64 / h as f64;

            for x in 0..w {
                let nx = col_cos[x];
                let ny = col_sin[x];
                let u = x as f64 / w as f64;

                // Compute warp offsets inline — no full-grid storage needed.
                let du = warp_u_noise.get_precomputed(nx, ny, nz, nw) * c.warp_u;
                let dv = warp_v_noise.get_precomputed(nx, ny, nz, nw) * c.warp_v;

                // The warped UV can't use the precomputed tables, so call get().
                let raw = base_noise.get(u + du, v + dv);
                let t = normalize(raw); // [0, 1]

                let idx = y * w + x;
                heights[idx] = t;

                // Colour: lerp between dark and light by height value.
                let r = lerp(c.color_dark[0], c.color_light[0], t as f32);
                let g = lerp(c.color_dark[1], c.color_light[1], t as f32);
                let b = lerp(c.color_dark[2], c.color_light[2], t as f32);

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // Roughness: grooves (dark, low t) are rougher.
                // Packed as ORM: R=Occlusion(1.0), G=Roughness, B=Metallic(0.0).
                let rough = 0.6 + (1.0 - t as f32) * 0.35;
                roughness[ai] = 255; // Occlusion = 1.0 (no shadowing)
                roughness[ai + 1] = (rough * 255.0).round() as u8;
                roughness[ai + 2] = 0; // Metallic = 0.0
                roughness[ai + 3] = 255;
            }
        }

        let normal = height_to_normal(&heights, width, height, c.normal_strength);

        Ok(TextureMap {
            albedo,
            normal,
            roughness,
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

/// Map a raw noise sample from `[-1, 1]` to `[0, 1]`.
#[inline]
fn normalize(v: f64) -> f64 {
    v * 0.5 + 0.5
}
