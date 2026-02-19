//! Rock texture generator using Ridged Multifractal noise.
//!
//! Ridged multifractal noise produces sharp, ridge-like features that mimic
//! the cracked and faceted appearance of stone surfaces.

use noise::{MultiFractal, Perlin, RidgedMulti};

use crate::{
    generator::{TextureGenerator, TextureMap, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::height_to_normal,
};

/// Configures the appearance of a [`RockGenerator`].
#[derive(Clone, Debug)]
pub struct RockConfig {
    pub seed: u32,
    /// Overall spatial scale.
    pub scale: f64,
    pub octaves: usize,
    /// Attenuation of the ridged multifractal (controls sharpness of ridges).
    pub attenuation: f64,
    /// Base (light) rock colour in linear RGB [0,1].
    pub color_light: [f32; 3],
    /// Shadow (dark) colour in linear RGB [0,1].
    pub color_dark: [f32; 3],
    pub normal_strength: f32,
}

impl Default for RockConfig {
    fn default() -> Self {
        Self {
            seed: 7,
            scale: 3.0,
            octaves: 8,
            attenuation: 2.0,
            color_light: [0.55, 0.52, 0.48],
            color_dark: [0.22, 0.20, 0.18],
            normal_strength: 4.0,
        }
    }
}

pub struct RockGenerator {
    config: RockConfig,
}

impl RockGenerator {
    pub fn new(config: RockConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for RockGenerator {
    fn generate(&self, width: u32, height: u32) -> TextureMap {
        validate_dimensions(width, height);
        let c = &self.config;

        let ridged: RidgedMulti<Perlin> = RidgedMulti::new(c.seed)
            .set_octaves(c.octaves)
            .set_attenuation(c.attenuation);

        let noise = ToroidalNoise::new(ridged, c.scale);
        let heights = sample_grid(&noise, width, height);

        let n = (width as usize) * (height as usize);
        let mut albedo = vec![0u8; n * 4];
        let mut roughness = vec![0u8; n * 4];

        for (i, &height) in heights.iter().enumerate().take(n) {
            let t = normalize(height) as f32;

            let r = lerp(c.color_dark[0], c.color_light[0], t);
            let g = lerp(c.color_dark[1], c.color_light[1], t);
            let b = lerp(c.color_dark[2], c.color_light[2], t);

            let ai = i * 4;
            albedo[ai] = linear_to_srgb(r);
            albedo[ai + 1] = linear_to_srgb(g);
            albedo[ai + 2] = linear_to_srgb(b);
            albedo[ai + 3] = 255;

            // Ridges (high t) are slightly smoother (exposed mineral); cracks rougher.
            // Packed as ORM: R=Occlusion(1.0), G=Roughness, B=Metallic(0.0).
            let rough = 0.75 - t * 0.25;
            roughness[ai] = 255; // Occlusion = 1.0 (no shadowing)
            roughness[ai + 1] = (rough * 255.0) as u8;
            roughness[ai + 2] = 0; // Metallic = 0.0
            roughness[ai + 3] = 255;
        }

        // heights is in [-1, 1]; normalize would scale gradients by 0.5.
        // Halving strength here is equivalent and avoids a full-sized allocation.
        let normal = height_to_normal(&heights, width, height, c.normal_strength * 0.5);

        TextureMap {
            albedo,
            normal,
            roughness,
            width,
            height,
        }
    }
}

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
