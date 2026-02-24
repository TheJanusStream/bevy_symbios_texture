//! Twig texture generator — a composite foliage card.
//!
//! A twig composites multiple leaves and an organic stem into a single
//! alpha-masked texture.  The stem tapers from a thick base (`v = 1`) to a
//! pointed terminal tip (`v = 0`) and is gently curved via Perlin noise.
//!
//! # Phyllotaxis modes
//! Controlled by [`TwigConfig::sympodial`]:
//!
//! * **Monopodial (`false`)** — a single continuous axis carries opposite leaf
//!   pairs at each node.  The axis stays relatively straight with a slight
//!   organic curve.  A terminal leaf caps the apex.
//!
//! * **Sympodial (`true`)** — each node produces one dominant lateral and one
//!   suppressed bud.  The axis appears to zigzag because each internode is
//!   really the continuation of a lateral shoot.  Leaves are alternate
//!   (one per node) and positioned at the bend points of the zigzag.
//!
//! # Coordinate conventions
//! * Texture UV: `u = 0` left, `u = 1` right, `v = 0` **tip** (apex),
//!   `v = 1` **base** (attachment to parent branch).
//! * Upload the result with
//!   [`map_to_images_card`](crate::generator::map_to_images_card).

use std::f64::consts::{FRAC_PI_2, PI};

use noise::{NoiseFn, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    leaf::{LeafConfig, LeafSampler},
    normal::height_to_normal,
};

// --- tuning constants -------------------------------------------------------

/// Perlin spatial frequency for the organic stem wiggle.
const STEM_CURVE_FREQ: f64 = 1.8;

/// Seed offset applied to the leaf seed to generate independent stem curvature.
const STEM_PERLIN_SEED_OFFSET: u32 = 77;

/// Relative Y-offset in Perlin space so the stem curve is decorrelated from
/// any future second dimension sampling on the same noise object.
const STEM_PERLIN_Y: f64 = 13.7;

/// Amplitude of the sympodial zigzag relative to `stem_curve`.
const SYMPODIAL_ZZ_SCALE: f64 = 1.5;

/// Power used to taper stem width: 1.0 = linear, < 1.0 = wider for longer.
const STEM_TAPER_POW: f64 = 0.55;

/// Scale of the terminal leaf relative to lateral leaves.
const TERMINAL_SCALE: f64 = 0.72;

// ----------------------------------------------------------------------------

/// Configures the appearance of a [`TwigGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TwigConfig {
    /// Leaf appearance shared by every leaf on the twig.
    pub leaf: LeafConfig,
    /// Stem colour in linear RGB \[0, 1\].
    pub stem_color: [f32; 3],
    /// Half-width of the stem at the base (`v = 1`) in UV space.
    /// The stem tapers to zero at the tip (`v = 0`).
    pub stem_half_width: f64,
    /// Number of lateral leaf nodes.
    /// Monopodial: `leaf_pairs` opposite pairs.  Sympodial: `leaf_pairs`
    /// alternate leaves.
    pub leaf_pairs: usize,
    /// Angle of each lateral leaf measured from the stem axis, in radians.
    ///
    /// * `0` — leaf points along the stem toward the base.
    /// * `π/2` — leaf is perpendicular to the stem (horizontal on a vertical twig).
    /// * `> π/2` — leaf tip angles toward the stem tip (acroscopic / upswept).
    /// * `π` — leaf points straight toward the apex.
    pub leaf_angle: f64,
    /// Scale of each lateral leaf card in UV space.
    pub leaf_scale: f64,
    /// Amplitude of the organic stem curvature in UV space.
    /// `0.0` = perfectly straight; `0.05` is a natural-looking default.
    pub stem_curve: f64,
    /// `false` → monopodial (opposite pairs, continuous axis);
    /// `true` → sympodial (alternate leaves, zigzag axis).
    pub sympodial: bool,
}

impl Default for TwigConfig {
    fn default() -> Self {
        Self {
            leaf: LeafConfig::default(),
            stem_color: [0.25, 0.16, 0.07],
            stem_half_width: 0.015,
            leaf_pairs: 4,
            leaf_angle: FRAC_PI_2 - 0.35, // ≈ 69° — slightly below perpendicular, drooping
            leaf_scale: 0.38,
            stem_curve: 0.05,
            sympodial: false,
        }
    }
}

/// A single leaf card placed on the twig.
pub struct LeafAttachment {
    /// Stem attachment point (U coordinate in texture space).
    pub attach_u: f64,
    /// Stem attachment point (V coordinate in texture space).
    pub attach_v: f64,
    /// Total rotation angle of the leaf axis in texture space, measured from
    /// the downward direction (`+V`).  Positive = clockwise = right side.
    /// Equals `stem_tangent_at(attach_v) + relative_leaf_angle`.
    pub angle: f64,
    /// Uniform scale factor (leaf UV space → texture UV space).
    pub scale: f64,
}

/// Procedural twig texture generator.
///
/// Composites a tapered, curved stem and multiple leaves into an alpha-masked
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

    /// Build the list of leaf attachment descriptors.
    ///
    /// `stem_perlin` must be the same Perlin instance used for the stem
    /// centerline in [`generate`](TextureGenerator::generate).
    pub fn leaf_attachments(&self, stem_perlin: &Perlin) -> Vec<LeafAttachment> {
        let c = &self.config;
        let n = c.leaf_pairs.max(1);

        if c.sympodial {
            self.sympodial_attachments(n, stem_perlin)
        } else {
            self.monopodial_attachments(n, stem_perlin)
        }
    }

    // --- monopodial ----------------------------------------------------------

    /// Opposite leaf pairs + terminal leaf.
    fn monopodial_attachments(&self, n: usize, perlin: &Perlin) -> Vec<LeafAttachment> {
        let c = &self.config;
        // 2 leaves per node + 1 terminal.
        let mut atts = Vec::with_capacity(n * 2 + 1);

        // The terminal leaf card extends toward the tip (v=0) by `leaf_scale *
        // TERMINAL_SCALE` texture units.  Push its attachment point far enough
        // from the edge that the whole card stays inside the texture.
        let term_v = terminal_v(c);

        // Lateral leaves run from just below the terminal to near the base.
        let lat_start = term_v + 0.05;
        let lat_span = 0.88 - lat_start;

        for i in 0..n {
            let attach_v = lat_start + (i as f64 / n as f64) * lat_span;
            let attach_u = stem_center_u(attach_v, c, perlin);
            let tangent = stem_tangent_at(attach_v, c, perlin);

            atts.push(LeafAttachment {
                attach_u,
                attach_v,
                angle: tangent + c.leaf_angle, // right leaf
                scale: c.leaf_scale,
            });
            atts.push(LeafAttachment {
                attach_u,
                attach_v,
                angle: tangent - c.leaf_angle, // left leaf (mirror)
                scale: c.leaf_scale,
            });
        }

        // Terminal leaf: points back along the stem (upward = +PI from downward).
        let term_tangent = stem_tangent_at(term_v, c, perlin);
        atts.push(LeafAttachment {
            attach_u: stem_center_u(term_v, c, perlin),
            attach_v: term_v,
            angle: term_tangent + PI, // pointing toward tip (upward)
            scale: c.leaf_scale * TERMINAL_SCALE,
        });

        atts
    }

    // --- sympodial -----------------------------------------------------------

    /// Alternate leaves at zigzag extrema + terminal leaf.
    ///
    /// In sympodial phyllotaxis the axis appears to zigzag because each
    /// internode is a lateral that became dominant.  Leaves are positioned at
    /// the bend points of the zigzag (sine extrema) and branch off on the
    /// **convex** (outer) side of each bend.
    fn sympodial_attachments(&self, n: usize, perlin: &Perlin) -> Vec<LeafAttachment> {
        let c = &self.config;
        // 1 leaf per node + 1 terminal.
        let mut atts = Vec::with_capacity(n + 1);

        let term_v = terminal_v(c);
        let lat_start = term_v + 0.05;
        let lat_span = 0.88 - lat_start;

        for i in 0..n {
            // Position leaves at the extrema of the sine zigzag.
            // sin(pv * n * PI) has extrema at pv = (2k+1) / (2n).
            let k = i as f64;
            let normalized = (2.0 * k + 1.0) / (2.0 * n as f64);
            let attach_v = lat_start + normalized * lat_span;

            let attach_u = stem_center_u(attach_v, c, perlin);
            let tangent = stem_tangent_at(attach_v, c, perlin);

            // The zigzag is sin(pv * n * PI) * … .  At the k-th extremum the
            // sign is (-1)^k (sin at (2k+1)*PI/2 = cos(k*PI) = (-1)^k).
            // Leaf is on the convex (outer) side: same sign as the zigzag.
            let side = if i % 2 == 0 { 1.0_f64 } else { -1.0 };
            atts.push(LeafAttachment {
                attach_u,
                attach_v,
                angle: tangent + side * c.leaf_angle,
                scale: c.leaf_scale,
            });
        }

        // Terminal leaf.
        let term_tangent = stem_tangent_at(term_v, c, perlin);
        atts.push(LeafAttachment {
            attach_u: stem_center_u(term_v, c, perlin),
            attach_v: term_v,
            angle: term_tangent + PI,
            scale: c.leaf_scale * TERMINAL_SCALE,
        });

        atts
    }
}

impl TextureGenerator for TwigGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;

        let c = &self.config;

        // A separate Perlin instance for the stem so its curve is uncorrelated
        // with the leaf edge-serration noise.
        let stem_perlin = Perlin::new(c.leaf.seed.wrapping_add(STEM_PERLIN_SEED_OFFSET));
        let sampler = LeafSampler::new(c.leaf.clone());
        let attachments = self.leaf_attachments(&stem_perlin);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        let mut heights = vec![0.5f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness = vec![0u8; n * 4];

        for y in 0..h {
            let pv = y as f64 / h as f64;

            // Stem centerline and tapered half-width for this scanline.
            let s_center = stem_center_u(pv, c, &stem_perlin);
            let s_hw = stem_half_width_at(pv, c.stem_half_width);

            for x in 0..w {
                let pu = x as f64 / w as f64;
                let idx = y * w + x;
                let ai = idx * 4;

                // --- Stem SDF ---
                let dist_to_stem = (pu - s_center).abs();
                if s_hw > 1e-9 && dist_to_stem < s_hw {
                    // Bright ridge at the stem centre.
                    let t = 1.0 - (dist_to_stem / s_hw) as f32;
                    heights[idx] = t as f64 * 0.6;

                    albedo[ai] = linear_to_srgb(lerp(c.stem_color[0] * 0.55, c.stem_color[0], t));
                    albedo[ai + 1] =
                        linear_to_srgb(lerp(c.stem_color[1] * 0.55, c.stem_color[1], t));
                    albedo[ai + 2] =
                        linear_to_srgb(lerp(c.stem_color[2] * 0.55, c.stem_color[2], t));
                    albedo[ai + 3] = 255;
                    roughness[ai] = 255;
                    roughness[ai + 1] = (0.78_f32 * 255.0) as u8;
                    roughness[ai + 2] = 0;
                    roughness[ai + 3] = 255;
                    continue;
                }

                // --- Leaf composite ---
                let mut hit = false;
                for att in &attachments {
                    let (lu, lv) = pixel_to_leaf_uv(pu, pv, att);
                    if !(0.0..=1.0).contains(&lu) || !(0.0..=1.0).contains(&lv) {
                        continue;
                    }
                    if let Some(s) = sampler.sample(lu, lv) {
                        heights[idx] = s.height;
                        albedo[ai] = linear_to_srgb(s.color[0]);
                        albedo[ai + 1] = linear_to_srgb(s.color[1]);
                        albedo[ai + 2] = linear_to_srgb(s.color[2]);
                        albedo[ai + 3] = 255;
                        roughness[ai] = 255;
                        roughness[ai + 1] = (s.roughness * 255.0).round() as u8;
                        roughness[ai + 2] = 0;
                        roughness[ai + 3] = 255;
                        hit = true;
                        break;
                    }
                }

                if !hit {
                    // Write edge color into RGB to prevent dark halos from
                    // bilinear filtering at silhouette boundaries.
                    let ec = &c.leaf.color_edge;
                    albedo[ai] = linear_to_srgb(ec[0]);
                    albedo[ai + 1] = linear_to_srgb(ec[1]);
                    albedo[ai + 2] = linear_to_srgb(ec[2]);
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

// --- stem helpers -----------------------------------------------------------

/// V attachment position for the terminal leaf.
///
/// The terminal leaf card extends `leaf_scale * TERMINAL_SCALE` units toward
/// the tip (`v = 0`).  Placing the attachment at this V ensures the whole card
/// remains inside the `[0, 1]` texture with a small margin.
#[inline]
fn terminal_v(config: &TwigConfig) -> f64 {
    config.leaf_scale * TERMINAL_SCALE + 0.03
}

/// U coordinate of the stem centreline at a given V (tip-to-base axis).
///
/// Combines a slow organic Perlin wiggle with an optional sympodial sine
/// zigzag whose amplitude grows from zero at the tip to maximum at the base.
fn stem_center_u(pv: f64, config: &TwigConfig, perlin: &Perlin) -> f64 {
    // Organic curvature: slow single-frequency Perlin wiggle.
    let organic = perlin.get([pv * STEM_CURVE_FREQ, STEM_PERLIN_Y]) * config.stem_curve;

    // Sympodial zigzag: sine wave, amplitude = 0 at tip, grows toward base.
    // Phase is computed over the lateral-leaf span [lat_start, 0.88] so that
    // the sine extrema align with the attach_v positions from
    // sympodial_attachments (which place leaves at normalized = (2k+1)/(2n)
    // within that same span).
    let zigzag = if config.sympodial {
        let lat_start = terminal_v(config) + 0.05;
        let lat_span = 0.88 - lat_start;
        let phase = if lat_span > 0.0 {
            (pv - lat_start) / lat_span * config.leaf_pairs as f64 * PI
        } else {
            0.0
        };
        phase.sin() * config.stem_curve * SYMPODIAL_ZZ_SCALE * pv
    } else {
        0.0
    };

    // Clamp so the stem stays safely inside the texture.
    (0.5 + organic + zigzag).clamp(0.08, 0.92)
}

/// Half-width of the stem at V position `pv` after tapering.
///
/// `pv = 0` (tip) → zero width; `pv = 1` (base) → `half_width`.
fn stem_half_width_at(pv: f64, half_width: f64) -> f64 {
    half_width * pv.powf(STEM_TAPER_POW)
}

/// Angle of the stem tangent at `pv` from the downward direction (`+V`),
/// computed via a numerical derivative.  Central differences are used in the
/// interior; one-sided differences are used at the boundaries so the divisor
/// always matches the actual sample spacing.
fn stem_tangent_at(pv: f64, config: &TwigConfig, perlin: &Perlin) -> f64 {
    let delta = 0.005_f64;
    let (u_lo, u_hi, dv) = if pv < delta {
        // Forward difference at the lower boundary.
        (
            stem_center_u(pv, config, perlin),
            stem_center_u(pv + delta, config, perlin),
            delta,
        )
    } else if pv > 1.0 - delta {
        // Backward difference at the upper boundary.
        (
            stem_center_u(pv - delta, config, perlin),
            stem_center_u(pv, config, perlin),
            delta,
        )
    } else {
        // Central difference in the interior.
        (
            stem_center_u(pv - delta, config, perlin),
            stem_center_u(pv + delta, config, perlin),
            2.0 * delta,
        )
    };
    // du/dv: horizontal displacement per unit of V.
    let du_dv = (u_hi - u_lo) / dv;
    // Angle from the downward (+V) direction.
    du_dv.atan2(1.0)
}

// --- leaf transform ---------------------------------------------------------

/// Map a texture-space pixel `(pu, pv)` into the local UV space of a leaf.
///
/// Uses the leaf attachment point and the baked total angle (stem tangent +
/// relative leaf angle) to invert the 2D rotation.
///
/// Leaf local UV: `u = 0.5` → midrib; `v = 0` → attachment; `v = 1` → tip.
fn pixel_to_leaf_uv(pu: f64, pv: f64, att: &LeafAttachment) -> (f64, f64) {
    let dx = pu - att.attach_u;
    let dy = pv - att.attach_v;

    // World-to-local rotation: local +V = (sin θ, cos θ) in texture space.
    let cos_a = att.angle.cos();
    let sin_a = att.angle.sin();
    let u_raw = dx * cos_a - dy * sin_a;
    let v_raw = dx * sin_a + dy * cos_a;

    (u_raw / att.scale + 0.5, v_raw / att.scale)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

// --- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stem_perlin(config: &TwigConfig) -> Perlin {
        Perlin::new(config.leaf.seed.wrapping_add(STEM_PERLIN_SEED_OFFSET))
    }

    #[test]
    fn monopodial_attachment_count() {
        let config = TwigConfig {
            sympodial: false,
            ..TwigConfig::default()
        };
        let twig_gen = TwigGenerator::new(config.clone());
        let atts = twig_gen.leaf_attachments(&make_stem_perlin(&config));
        // 2 per pair + 1 terminal.
        assert_eq!(atts.len(), config.leaf_pairs * 2 + 1);
    }

    #[test]
    fn sympodial_attachment_count() {
        let config = TwigConfig {
            sympodial: true,
            ..TwigConfig::default()
        };
        let twig_gen = TwigGenerator::new(config.clone());
        let atts = twig_gen.leaf_attachments(&make_stem_perlin(&config));
        // 1 per node + 1 terminal.
        assert_eq!(atts.len(), config.leaf_pairs + 1);
    }

    #[test]
    fn monopodial_leaves_are_opposite() {
        let config = TwigConfig {
            sympodial: false,
            stem_curve: 0.0,
            ..TwigConfig::default()
        };
        let twig_gen = TwigGenerator::new(config.clone());
        let atts = twig_gen.leaf_attachments(&make_stem_perlin(&config));
        // Each pair (excluding the terminal) should have opposite angles.
        let n = config.leaf_pairs;
        for i in 0..n {
            let right = &atts[i * 2];
            let left = &atts[i * 2 + 1];
            assert!(
                (right.angle + left.angle).abs() < 1e-9,
                "monopodial pair {i}: angles should sum to zero (got {} + {})",
                right.angle,
                left.angle,
            );
        }
    }

    #[test]
    fn sympodial_leaves_alternate_sides() {
        let config = TwigConfig {
            sympodial: true,
            stem_curve: 0.0,
            leaf_pairs: 4,
            ..TwigConfig::default()
        };
        let twig_gen = TwigGenerator::new(config.clone());
        let atts = twig_gen.leaf_attachments(&make_stem_perlin(&config));
        // Alternate: right (positive angle), left (negative), right, left, …
        for (i, att) in atts.iter().take(config.leaf_pairs).enumerate() {
            let expected_sign = if i % 2 == 0 { 1.0_f64 } else { -1.0 };
            assert!(
                att.angle * expected_sign > 0.0,
                "sympodial leaf {i}: angle should be on side {expected_sign:+} (got {})",
                att.angle,
            );
        }
    }

    #[test]
    fn stem_tapers_to_zero_at_tip() {
        assert!(stem_half_width_at(0.0, 0.015) < 1e-9);
        assert!(stem_half_width_at(1.0, 0.015) > 0.014);
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
        assert!(
            map.albedo.chunks(4).any(|px| px[3] == 0),
            "twig texture should have transparent pixels"
        );
        assert!(
            map.albedo.chunks(4).any(|px| px[3] == 255),
            "twig texture should have opaque pixels"
        );
    }

    #[test]
    fn stem_center_is_opaque_when_straight() {
        // With stem_curve=0 the centerline stays at u=0.5, making the center
        // pixel at (64, 64) in a 128×128 texture reliably on the stem.
        let config = TwigConfig {
            stem_curve: 0.0,
            sympodial: false,
            ..TwigConfig::default()
        };
        let twig_gen = TwigGenerator::new(config);
        let map = twig_gen.generate(128, 128).expect("generate failed");
        let idx = (64 * 128 + 64) * 4;
        assert_eq!(
            map.albedo[idx + 3],
            255,
            "straight stem center should be opaque"
        );
    }

    #[test]
    fn pixel_to_leaf_uv_symmetric() {
        let att_right = LeafAttachment {
            attach_u: 0.5,
            attach_v: 0.5,
            angle: 1.0,
            scale: 0.4,
        };
        let att_left = LeafAttachment {
            attach_u: 0.5,
            attach_v: 0.5,
            angle: -1.0,
            scale: 0.4,
        };

        let (ru, rv) = pixel_to_leaf_uv(0.8, 0.5, &att_right);
        let (lu, lv) = pixel_to_leaf_uv(0.2, 0.5, &att_left);

        assert!(
            (rv - lv).abs() < 1e-12,
            "v_local should be equal: {rv} vs {lv}"
        );
        assert!(
            ((ru - 0.5).abs() - (lu - 0.5).abs()).abs() < 1e-12,
            "u_local distance from midrib should match: |{ru}-0.5|={} vs |{lu}-0.5|={}",
            (ru - 0.5).abs(),
            (lu - 0.5).abs(),
        );
    }

    #[test]
    fn sympodial_generator_has_transparent_and_opaque() {
        let config = TwigConfig {
            sympodial: true,
            ..TwigConfig::default()
        };
        let twig_gen = TwigGenerator::new(config);
        let map = twig_gen.generate(128, 128).expect("generate failed");
        assert!(map.albedo.chunks(4).any(|px| px[3] == 0));
        assert!(map.albedo.chunks(4).any(|px| px[3] == 255));
    }
}
