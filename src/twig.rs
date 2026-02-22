//! Twig texture generator — a composite foliage card.
//!
//! A twig arranges multiple leaves along a central stem using 2D affine
//! transforms to map each pixel into the leaf's local UV space, then calls
//! [`LeafSampler::sample`] to composite the result.  The first leaf that
//! covers a pixel wins; the stem SDF is tested first.
//!
//! Like [`LeafGenerator`](crate::leaf::LeafGenerator), the output is an
//! alpha-masked texture.  Upload with
//! [`map_to_images_card`](crate::generator::map_to_images_card).
//!
//! ## Coordinate conventions
//! * Texture UV: `u = 0` left, `u = 1` right, `v = 0` top, `v = 1` bottom.
//! * Stem: vertical line at `u = 0.5`, running the full texture height.
//! * Leaf angle: measured from the downward direction of the stem (`+V`).
//!   `angle = π/2` → leaf points straight out (perpendicular to stem).

use std::f64::consts::FRAC_PI_2;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    leaf::{LeafConfig, LeafSampler},
    normal::height_to_normal,
};

/// Configures the appearance of a [`TwigGenerator`].
#[derive(Clone, Debug)]
pub struct TwigConfig {
    /// Leaf appearance shared by every leaf on the twig.
    pub leaf: LeafConfig,
    /// Stem colour in linear RGB \[0, 1\].
    pub stem_color: [f32; 3],
    /// Half-width of the stem in UV space.
    pub stem_half_width: f64,
    /// Number of opposite leaf pairs along the stem.
    pub leaf_pairs: usize,
    /// Angle of each leaf from the downward direction of the stem (radians).
    /// `π/2` ≈ perpendicular; `π/4` ≈ 45° forward-facing.
    pub leaf_angle: f64,
    /// Scale of each leaf card relative to the texture.  Controls how much
    /// of the texture each leaf occupies along its own V axis.
    pub leaf_scale: f64,
}

impl Default for TwigConfig {
    fn default() -> Self {
        Self {
            leaf: LeafConfig::default(),
            stem_color: [0.25, 0.16, 0.07],
            stem_half_width: 0.015,
            leaf_pairs: 4,
            leaf_angle: FRAC_PI_2 - 0.4, // ≈ 67° from downward → mostly sideways
            leaf_scale: 0.38,
        }
    }
}

/// Describes a single leaf's placement in texture space.
pub struct LeafAttachment {
    /// V position of the stem attachment point.
    pub stem_v: f64,
    /// Signed angle from downward-stem direction.  Positive = right side.
    pub angle: f64,
    /// Uniform scale factor for the leaf (leaf UV mapped over this distance).
    pub scale: f64,
}

/// Procedural twig texture generator.
///
/// Composites multiple leaves and a central stem into a single alpha-masked
/// foliage card.  Upload the result with
/// [`map_to_images_card`](crate::generator::map_to_images_card).
pub struct TwigGenerator {
    config: TwigConfig,
}

impl TwigGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: TwigConfig) -> Self {
        Self { config }
    }

    /// Build the list of leaf attachment descriptors from the configuration.
    fn leaf_attachments(&self) -> Vec<LeafAttachment> {
        let c = &self.config;
        let n = c.leaf_pairs.max(1);
        let mut attachments = Vec::with_capacity(n * 2);

        for i in 0..n {
            // Distribute pairs evenly along V, leaving a 10 % margin at each end.
            let stem_v = 0.1 + (i as f64 / n as f64) * 0.8;

            // Right leaf: positive angle (points right).
            attachments.push(LeafAttachment {
                stem_v,
                angle: c.leaf_angle,
                scale: c.leaf_scale,
            });

            // Left leaf: negative angle (mirror of right).
            attachments.push(LeafAttachment {
                stem_v,
                angle: -c.leaf_angle,
                scale: c.leaf_scale,
            });
        }

        attachments
    }
}

impl TextureGenerator for TwigGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;

        let c = &self.config;
        let sampler = LeafSampler::new(c.leaf.clone());
        let attachments = self.leaf_attachments();

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // Neutral height for transparent pixels → flat normal.
        let mut heights = vec![0.5f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness = vec![0u8; n * 4];

        for y in 0..h {
            let pv = y as f64 / h as f64;
            for x in 0..w {
                let pu = x as f64 / w as f64;
                let idx = y * w + x;
                let ai = idx * 4;

                // --- Stem SDF ---
                let dist_to_stem = (pu - 0.5).abs();
                if dist_to_stem < c.stem_half_width {
                    // Brighter at the center of the stem.
                    let t = 1.0 - (dist_to_stem / c.stem_half_width) as f32;
                    heights[idx] = t as f64 * 0.6;

                    albedo[ai] = linear_to_srgb(lerp(c.stem_color[0] * 0.6, c.stem_color[0], t));
                    albedo[ai + 1] =
                        linear_to_srgb(lerp(c.stem_color[1] * 0.6, c.stem_color[1], t));
                    albedo[ai + 2] =
                        linear_to_srgb(lerp(c.stem_color[2] * 0.6, c.stem_color[2], t));
                    albedo[ai + 3] = 255;

                    roughness[ai] = 255; // occlusion
                    roughness[ai + 1] = (0.75_f32 * 255.0) as u8;
                    roughness[ai + 2] = 0; // metallic
                    roughness[ai + 3] = 255;
                    continue;
                }

                // --- Leaf composite ---
                // Try each attachment in order; first hit wins.
                let mut hit = false;
                for att in &attachments {
                    let (lu, lv) = pixel_to_leaf_uv(pu, pv, att);

                    // Bounds check in leaf UV space before sampling.
                    if !(0.0..=1.0).contains(&lu) || !(0.0..=1.0).contains(&lv) {
                        continue;
                    }

                    if let Some(s) = sampler.sample(lu, lv) {
                        heights[idx] = s.height;

                        albedo[ai] = linear_to_srgb(s.color[0]);
                        albedo[ai + 1] = linear_to_srgb(s.color[1]);
                        albedo[ai + 2] = linear_to_srgb(s.color[2]);
                        albedo[ai + 3] = 255;

                        roughness[ai] = 255; // occlusion
                        roughness[ai + 1] = (s.roughness * 255.0).round() as u8;
                        roughness[ai + 2] = 0; // metallic
                        roughness[ai + 3] = 255;

                        hit = true;
                        break;
                    }
                }

                if !hit {
                    // Fully transparent.
                    albedo[ai + 3] = 0;
                    roughness[ai] = 255;
                    roughness[ai + 1] = 200;
                    roughness[ai + 2] = 0;
                    roughness[ai + 3] = 255;
                }
            }
        }

        let normal = height_to_normal(&heights, width, height, c.leaf.normal_strength);

        Ok(TextureMap {
            albedo,
            normal,
            roughness,
            width,
            height,
        })
    }
}

/// Transform a texture-space pixel `(pu, pv)` into the local UV space of a
/// leaf described by `att`.
///
/// The leaf's coordinate frame:
/// - Local +V points toward the leaf tip, at angle `att.angle` from the
///   downward direction of the stem.
/// - Local +U points 90° clockwise from local +V (the right-hand edge of the
///   leaf as seen from above).
/// - `u_local = 0.5` → midrib; `v_local = 0` → attachment; `v_local = 1` → tip.
fn pixel_to_leaf_uv(pu: f64, pv: f64, att: &LeafAttachment) -> (f64, f64) {
    // Translate pixel relative to the stem attachment point.
    let dx = pu - 0.5;
    let dy = pv - att.stem_v;

    // Rotate into the leaf's local frame.
    // If local Y (tip direction) in texture space = (sin θ, cos θ), then
    // the world-to-local (inverse rotation) maps:
    //   u_raw = dx · cos θ - dy · sin θ   (along leaf X axis)
    //   v_raw = dx · sin θ + dy · cos θ   (along leaf Y / tip axis)
    let cos_a = att.angle.cos();
    let sin_a = att.angle.sin();
    let u_raw = dx * cos_a - dy * sin_a;
    let v_raw = dx * sin_a + dy * cos_a;

    // Scale and map to [0, 1] leaf UV (u=0.5 = midrib, v=0 = attachment).
    let u_local = u_raw / att.scale + 0.5;
    let v_local = v_raw / att.scale;

    (u_local, v_local)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

// --- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_valid_pairs() {
        let twig_gen = TwigGenerator::new(TwigConfig::default());
        let atts = twig_gen.leaf_attachments();
        assert_eq!(atts.len(), TwigConfig::default().leaf_pairs * 2);
        // Left and right angles should be negatives of each other.
        for pair in atts.chunks(2) {
            assert!(
                (pair[0].angle + pair[1].angle).abs() < 1e-12,
                "left/right angles should be symmetric"
            );
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let twig_gen = TwigGenerator::new(TwigConfig::default());
        let map = twig_gen.generate(64, 64).expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn generator_has_transparent_and_opaque_pixels() {
        let twig_gen = TwigGenerator::new(TwigConfig::default());
        let map = twig_gen.generate(128, 128).expect("generate failed");
        let has_transparent = map.albedo.chunks(4).any(|px| px[3] == 0);
        let has_opaque = map.albedo.chunks(4).any(|px| px[3] == 255);
        assert!(
            has_transparent,
            "twig texture should have transparent pixels"
        );
        assert!(has_opaque, "twig texture should have opaque pixels");
    }

    #[test]
    fn stem_center_is_opaque() {
        let twig_gen = TwigGenerator::new(TwigConfig::default());
        let map = twig_gen.generate(128, 128).expect("generate failed");
        // The pixel at (u=0.5, v=0.5) should be on the stem and fully opaque.
        let x = 64usize;
        let y = 64usize;
        let idx = (y * 128 + x) * 4;
        assert_eq!(map.albedo[idx + 3], 255, "stem center should be opaque");
    }

    #[test]
    fn pixel_to_leaf_uv_symmetric() {
        // Symmetric pixels should map to symmetric leaf UVs around the midrib.
        let att_right = LeafAttachment {
            stem_v: 0.5,
            angle: 1.0,
            scale: 0.4,
        };
        let att_left = LeafAttachment {
            stem_v: 0.5,
            angle: -1.0,
            scale: 0.4,
        };

        let (ru, rv) = pixel_to_leaf_uv(0.8, 0.5, &att_right);
        let (lu, lv) = pixel_to_leaf_uv(0.2, 0.5, &att_left);

        // V coordinates should be equal (by symmetry).
        assert!(
            (rv - lv).abs() < 1e-12,
            "v_local should be equal: {rv} vs {lv}"
        );
        // U coordinates should be equidistant from the midrib (0.5).
        assert!(
            ((ru - 0.5).abs() - (lu - 0.5).abs()).abs() < 1e-12,
            "u distance from midrib should be equal: |{ru}-0.5|={} vs |{lu}-0.5|={}",
            (ru - 0.5).abs(),
            (lu - 0.5).abs(),
        );
    }
}
