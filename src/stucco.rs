//! Stucco / render texture generator.
//!
//! Smooth, high-frequency FBM bumps over a flat matte base — typical of
//! sand-float or pebble-dash exterior render.  The surface is almost flat
//! (low relief) and entirely matte with zero metallic response.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Configures the appearance of a [`StuccoGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StuccoConfig {
    pub seed: u32,
    /// Spatial frequency — controls bump density (higher = finer texture).
    pub scale: f64,
    /// FBM octave count.
    pub octaves: usize,
    /// Bump amplitude \[0, 1\] — controls surface relief depth.
    pub roughness: f64,
    /// Base stucco colour in linear RGB \[0, 1\].
    pub color_base: [f32; 3],
    /// Shadow / recessed-area colour in linear RGB \[0, 1\].
    pub color_shadow: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for StuccoConfig {
    fn default() -> Self {
        Self {
            seed: 13,
            scale: 8.0,
            octaves: 6,
            roughness: 0.35,
            color_base: [0.92, 0.89, 0.84],
            color_shadow: [0.72, 0.70, 0.66],
            normal_strength: 2.0,
        }
    }
}

/// Procedural stucco / render texture generator.
///
/// Produces tileable albedo, normal, and ORM maps.  Upload via
/// [`crate::async_gen::PendingTexture::stucco`] / [`crate::generator::map_to_images`].
pub struct StuccoGenerator {
    config: StuccoConfig,
}

impl StuccoGenerator {
    pub fn new(config: StuccoConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for StuccoGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let fbm: Fbm<Perlin> = Fbm::new(c.seed).set_octaves(c.octaves);
        let noise = ToroidalNoise::new(fbm, c.scale);
        let heights = sample_grid(&noise, width, height);

        let n = (width as usize) * (height as usize);
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for (i, &h) in heights.iter().enumerate() {
            // normalize maps [-1,1] → [0,1]; scale by roughness amplitude.
            let t = (normalize(h) * c.roughness) as f32;

            let r = lerp(c.color_shadow[0], c.color_base[0], t);
            let g = lerp(c.color_shadow[1], c.color_base[1], t);
            let b = lerp(c.color_shadow[2], c.color_base[2], t);

            let ai = i * 4;
            albedo[ai] = linear_to_srgb(r);
            albedo[ai + 1] = linear_to_srgb(g);
            albedo[ai + 2] = linear_to_srgb(b);
            albedo[ai + 3] = 255;

            // Matte finish: high roughness, zero metallic.
            // Recessed bumps (low t) are slightly rougher (shadow/grit).
            let rough = (0.82 + (1.0 - t) * 0.10).clamp(0.0, 1.0);
            roughness_buf[ai] = 255; // Occlusion = 1.0
            roughness_buf[ai + 1] = (rough * 255.0).round() as u8;
            roughness_buf[ai + 2] = 0; // Metallic = 0.0
            roughness_buf[ai + 3] = 255;
        }

        // Scale the height map by roughness so the normal map also respects
        // bump amplitude (not just the albedo interpolation above).
        let heights_scaled: Vec<f64> = heights.iter().map(|&h| h * c.roughness).collect();
        let normal = height_to_normal(
            &heights_scaled,
            width,
            height,
            c.normal_strength * 0.5,
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

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}
