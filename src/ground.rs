//! Ground / dirt texture generator.
//!
//! Produces a matted, organic-looking surface by blending two FBM layers at
//! different scales. A low-frequency layer defines broad soil patches; a
//! high-frequency layer adds fine grain and pebble-like micro-detail.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize},
    normal::height_to_normal,
};

/// Configures the appearance of a [`GroundGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GroundConfig {
    pub seed: u32,
    /// Scale of the large soil-patch layer.
    pub macro_scale: f64,
    /// Octaves for the large soil-patch FBM layer.
    pub macro_octaves: usize,
    /// Scale of the fine-grain layer.
    pub micro_scale: f64,
    /// Octaves for the fine-grain FBM layer.
    pub micro_octaves: usize,
    /// Blend weight of the micro layer (0 = only macro, 1 = only micro).
    pub micro_weight: f64,
    /// Dry (light) soil colour in linear RGB \[0, 1\].
    pub color_dry: [f32; 3],
    /// Moist (dark) soil colour in linear RGB \[0, 1\].
    pub color_moist: [f32; 3],
    /// Normal map strength — larger values produce more pronounced surface detail.
    pub normal_strength: f32,
}

impl Default for GroundConfig {
    fn default() -> Self {
        Self {
            seed: 13,
            macro_scale: 2.0,
            macro_octaves: 5,
            micro_scale: 8.0,
            micro_octaves: 4,
            micro_weight: 0.35,
            color_dry: [0.52, 0.40, 0.26],
            color_moist: [0.28, 0.20, 0.12],
            normal_strength: 2.0,
        }
    }
}

/// Procedural ground / dirt texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`GroundConfig`].  Construct
/// via [`GroundGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::ground`] task for non-blocking generation.
pub struct GroundGenerator {
    config: GroundConfig,
}

impl GroundGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: GroundConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for GroundGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let fbm_macro: Fbm<Perlin> = Fbm::new(c.seed).set_octaves(c.macro_octaves);
        let fbm_micro: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(50)).set_octaves(c.micro_octaves);

        let macro_noise = ToroidalNoise::new(fbm_macro, c.macro_scale);
        let micro_noise = ToroidalNoise::new(fbm_micro, c.micro_scale);

        let w = width as f64;
        let h = height as f64;
        let n = (width as usize) * (height as usize);

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness = vec![0u8; n * 4];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let u = x as f64 / w;
                let v = y as f64 / h;

                let macro_val = normalize(macro_noise.get(u, v));
                let micro_val = normalize(micro_noise.get(u, v));

                let t = macro_val * (1.0 - c.micro_weight) + micro_val * c.micro_weight;
                heights[idx] = t;

                let tf = t as f32;
                let r = lerp(c.color_moist[0], c.color_dry[0], tf);
                let g = lerp(c.color_moist[1], c.color_dry[1], tf);
                let b = lerp(c.color_moist[2], c.color_dry[2], tf);

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // Ground is generally rough; slight variation by moisture.
                // Packed as ORM: R=Occlusion(1.0), G=Roughness, B=Metallic(0.0).
                let rough = 0.80 + (1.0 - tf) * 0.15;
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

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}
