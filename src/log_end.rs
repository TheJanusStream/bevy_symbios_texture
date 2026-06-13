//! Log-end (cut timber) card generator.
//!
//! The sawn end of a log: a round slice silhouette with a slightly
//! irregular outline, concentric growth rings (early/late wood bands over
//! an FBM-warped radial distance, so the rings wobble like real wood),
//! optional radial drying cracks, and a bark rim around the perimeter.
//! Completes the wood set alongside `bark` (trunk surface) and `plank`
//! (sawn boards).
//!
//! Upload with [`map_to_images_card`](crate::generator::map_to_images_card);
//! concentric rings cannot tile, so this is an alpha-masked card.

use std::f64::consts::TAU;

use noise::Perlin;
use rayon::prelude::*;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    normal::{BoundaryMode, dilate_heights, height_to_normal},
    sprite::{fbm2, lerp_color},
};

/// Configures the appearance of a [`LogEndGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LogEndConfig {
    /// PRNG seed for the ring wobble, outline irregularity, and bark
    /// streaks.
    pub seed: u32,
    /// Growth rings from pith to bark `[4, 30]`.
    pub ring_count: f64,
    /// Ring wobble strength `[0, 1]` — FBM warp of the radial distance.
    pub ring_warp: f64,
    /// Early/late wood contrast exponent `[0.5, 4]`; higher sharpens the
    /// dark latewood bands.
    pub ring_contrast: f64,
    /// Radial drying cracks `[0, 12]` — `0` disables them.
    pub crack_count: f64,
    /// Bark rim thickness as a fraction of the slice radius `[0.02, 0.2]`.
    pub bark_width: f64,
    /// Earlywood (light band) colour in linear RGB \[0, 1\].
    pub color_early: [f32; 3],
    /// Latewood (dark band) colour in linear RGB \[0, 1\].
    pub color_late: [f32; 3],
    /// Bark rim colour in linear RGB \[0, 1\].
    pub color_bark: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for LogEndConfig {
    fn default() -> Self {
        Self {
            seed: 7,
            ring_count: 14.0,
            ring_warp: 0.35,
            ring_contrast: 1.8,
            crack_count: 5.0,
            bark_width: 0.07,
            color_early: [0.78, 0.62, 0.42],
            color_late: [0.48, 0.33, 0.18],
            color_bark: [0.30, 0.20, 0.12],
            normal_strength: 2.5,
        }
    }
}

/// Procedural log-end card generator.
///
/// See the [module documentation](self) for the visual model.
pub struct LogEndGenerator {
    config: LogEndConfig,
}

impl LogEndGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: LogEndConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for LogEndGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        let perlin = Perlin::new(c.seed);
        let ring_count = c.ring_count.clamp(4.0, 30.0);
        let warp = c.ring_warp.clamp(0.0, 1.0);
        let contrast = c.ring_contrast.clamp(0.5, 4.0);
        let crack_count = c.crack_count.round().clamp(0.0, 12.0);
        let bark_w = c.bark_width.clamp(0.02, 0.2);

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness = vec![0u8; n * 4];

        heights
            .par_chunks_mut(w)
            .zip(albedo.par_chunks_mut(w * 4))
            .zip(roughness.par_chunks_mut(w * 4))
            .enumerate()
            .for_each(|(y, ((height_row, albedo_row), orm_row))| {
                let v = (y as f64 + 0.5) / h as f64;
                for (x, height_slot) in height_row.iter_mut().enumerate() {
                    let u = (x as f64 + 0.5) / w as f64;

                    let dx = (u - 0.5) * 2.0;
                    let dy = (v - 0.5) * 2.0;
                    let r = (dx * dx + dy * dy).sqrt();
                    let theta = dy.atan2(dx);

                    // Irregular outline: angular noise sampled on a circle
                    // so the silhouette closes seamlessly.
                    let outline = 0.90
                        + 0.05
                            * (fbm2(&perlin, theta.cos() * 1.7 + 11.0, theta.sin() * 1.7, 3) - 0.5)
                            * 2.0;

                    let ai = x * 4;
                    if r > outline {
                        // Outside the slice: transparent, bark RGB to avoid
                        // filtering halos.
                        albedo_row[ai] = linear_to_srgb(c.color_bark[0]);
                        albedo_row[ai + 1] = linear_to_srgb(c.color_bark[1]);
                        albedo_row[ai + 2] = linear_to_srgb(c.color_bark[2]);
                        albedo_row[ai + 3] = 0;
                        orm_row[ai] = 255;
                        orm_row[ai + 1] = 200;
                        orm_row[ai + 2] = 0;
                        orm_row[ai + 3] = 255;
                        continue;
                    }
                    let r_norm = r / outline;

                    // FBM-warped radial distance → wobbling rings.
                    let wobble = (fbm2(&perlin, u * 2.4, v * 2.4, 4) - 0.5) * warp * 0.25;
                    let ring_phase = ((r_norm + wobble) * ring_count * TAU).sin() * 0.5 + 0.5;
                    let ring_t = ring_phase.powf(contrast); // 1 = earlywood

                    // Radial drying cracks: thin wedges widening outward.
                    let crack = if crack_count > 0.0 {
                        let band = (theta * crack_count * 0.5).sin().abs();
                        let line = (1.0 - band.powf(0.06)).clamp(0.0, 1.0);
                        let reach = ((r_norm - 0.15) / 0.55).clamp(0.0, 1.0);
                        line * reach
                    } else {
                        0.0
                    };

                    let in_bark = r_norm > 1.0 - bark_w;

                    let (color, height, rough) = if in_bark {
                        // Bark rim: radial streaks.
                        let streak = fbm2(&perlin, theta.cos() * 4.0, theta.sin() * 4.0 + 31.0, 3);
                        let shade = (0.7 + streak * 0.5) as f32;
                        let color = [
                            (c.color_bark[0] * shade).clamp(0.0, 1.0),
                            (c.color_bark[1] * shade).clamp(0.0, 1.0),
                            (c.color_bark[2] * shade).clamp(0.0, 1.0),
                        ];
                        (color, 0.45 + streak * 0.3, 0.9f32)
                    } else {
                        let base = lerp_color(c.color_late, c.color_early, ring_t as f32);
                        let crack_f = crack as f32;
                        let color = [
                            base[0] * (1.0 - crack_f * 0.7),
                            base[1] * (1.0 - crack_f * 0.7),
                            base[2] * (1.0 - crack_f * 0.7),
                        ];
                        let height = (0.55 + ring_t * 0.2 - crack * 0.45).clamp(0.0, 1.0);
                        let rough =
                            (0.62 + (1.0 - ring_t as f32) * 0.15 + crack_f * 0.2).clamp(0.0, 1.0);
                        (color, height, rough)
                    };

                    *height_slot = height.clamp(0.0, 1.0);
                    albedo_row[ai] = linear_to_srgb(color[0]);
                    albedo_row[ai + 1] = linear_to_srgb(color[1]);
                    albedo_row[ai + 2] = linear_to_srgb(color[2]);
                    albedo_row[ai + 3] = 255;
                    orm_row[ai] = 255;
                    orm_row[ai + 1] = (rough * 255.0).round() as u8;
                    orm_row[ai + 2] = 0;
                    orm_row[ai + 3] = 255;
                }
            });

        dilate_heights(&mut heights, &albedo, w, h);

        let normal = height_to_normal(
            &heights,
            width,
            height,
            c.normal_strength,
            BoundaryMode::Clamp,
        );

        Ok(TextureMap {
            albedo,
            normal,
            roughness,
            emissive: None,
            width,
            height,
            mip_level_count: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = LogEndGenerator::new(LogEndConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
    }

    #[test]
    fn slice_is_opaque_with_transparent_corners() {
        let map = LogEndGenerator::new(LogEndConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        let centre = (64 * 128 + 64) * 4;
        assert_eq!(map.albedo[centre + 3], 255, "pith should be opaque");
        for (cx, cy) in [(2usize, 2usize), (125, 2), (2, 125), (125, 125)] {
            let idx = (cy * 128 + cx) * 4;
            assert_eq!(map.albedo[idx + 3], 0, "corner ({cx},{cy})");
        }
    }

    #[test]
    fn rings_vary_the_albedo() {
        let map = LogEndGenerator::new(LogEndConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        // Walk outward from the pith: the red channel must alternate.
        let mut values = Vec::new();
        for x in 64..120usize {
            let idx = (64 * 128 + x) * 4;
            if map.albedo[idx + 3] == 255 {
                values.push(map.albedo[idx]);
            }
        }
        let min = values.iter().min().copied().unwrap_or(0);
        let max = values.iter().max().copied().unwrap_or(0);
        assert!(max - min > 20, "growth rings should band the albedo");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = LogEndGenerator::new(LogEndConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = LogEndGenerator::new(LogEndConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
