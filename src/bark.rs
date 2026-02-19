//! Bark texture generator using domain-warped FBM noise.
//!
//! The algorithm:
//!  1. Sample two FBM layers to produce warp offsets (du, dv).
//!  2. Apply a strong vertical warp to create fibrous, vertical streaks.
//!  3. Sample a third FBM layer at the warped coordinates for the final value.
//!  4. Derive colour, roughness and a height field from the result.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureGenerator, TextureMap},
    noise::{ToroidalNoise, normalize, sample_grid},
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
    fn generate(&self, width: u32, height: u32) -> TextureMap {
        let c = &self.config;

        // Three independent FBM sources with offset seeds.
        let fbm_warp_u: Fbm<Perlin> = Fbm::new(c.seed).set_octaves(c.octaves);
        let fbm_warp_v: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(100)).set_octaves(c.octaves);
        let fbm_base: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(200)).set_octaves(c.octaves);

        let warp_u_noise = ToroidalNoise::new(fbm_warp_u, c.scale);
        let warp_v_noise = ToroidalNoise::new(fbm_warp_v, c.scale);
        let base_noise = ToroidalNoise::new(fbm_base, c.scale);

        let warp_u_field = sample_grid(&warp_u_noise, width, height);
        let warp_v_field = sample_grid(&warp_v_noise, width, height);

        let w = width as f64;
        let h = height as f64;
        let n = (width * height) as usize;

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness = vec![0u8; n * 4];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let u = x as f64 / w;
                let v = y as f64 / h;

                let du = warp_u_field[idx] * c.warp_u;
                let dv = warp_v_field[idx] * c.warp_v;

                let raw = base_noise.get(u + du, v + dv);
                let t = normalize(raw); // [0, 1]

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
                // glTF / Bevy StandardMaterial reads roughness from the Green channel.
                let rough = 0.6 + (1.0 - t as f32) * 0.35;
                let ri = idx * 4;
                roughness[ri] = 0;
                roughness[ri + 1] = (rough * 255.0) as u8;
                roughness[ri + 2] = 0;
                roughness[ri + 3] = 255;
            }
        }

        let normal = height_to_normal(&heights, width, height, c.normal_strength);

        TextureMap {
            albedo,
            normal,
            roughness,
            width,
            height,
        }
    }
}

// --- helpers ----------------------------------------------------------------

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Linear → sRGB using the standard piecewise IEC 61966-2-1 transfer function.
#[inline]
fn linear_to_srgb(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let encoded = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0) as u8
}
