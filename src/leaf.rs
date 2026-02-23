//! Leaf texture generator for foliage card rendering.
//!
//! Unlike the tileable generators (Bark, Rock, Ground), this generator
//! produces a discrete leaf silhouette with an alpha channel.  Pixels outside
//! the leaf envelope are fully transparent (`alpha = 0`), defining the
//! silhouette.  Use [`map_to_images_card`](crate::generator::map_to_images_card)
//! when uploading to Bevy so the sampler does not tile.
//!
//! # Architecture
//! The core math lives in [`LeafSampler`], which holds pre-initialised noise
//! objects.  Call [`LeafSampler::sample`] for efficient per-pixel evaluation.
//! The free function [`sample_leaf`] is a convenience wrapper for one-off use.
//! [`LeafGenerator`] iterates the pixel grid using [`LeafSampler`] directly.
//!
//! ## Coordinate convention
//! Leaf UV space: `u = 0.5` is the midrib, `v = 0` is the stem attachment
//! (base), `v = 1` is the leaf tip.

use std::f64::consts::PI;

use noise::core::worley::ReturnType;
use noise::{NoiseFn, Perlin, Worley};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    normal::height_to_normal,
};

// --- tuning constants -------------------------------------------------------

/// Serration noise frequency (UV cycles).  Higher → finer-toothed edge.
const SERRATION_FREQ: f64 = 14.0;

/// Envelope decay rate.  Controls how quickly the leaf narrows from base to
/// tip.  Higher → narrower / more pointed leaf.
const ENVELOPE_DECAY: f64 = 2.0;

/// Maximum half-width of the leaf at its widest point (fraction of UV width).
const MAX_HALF_WIDTH: f64 = 0.44;

/// Frequency of the Worley capillary cells (cells per UV unit).
const WORLEY_FREQ: f64 = 20.0;

/// Frequency of the venule (tertiary vein) Perlin noise (UV cycles).
const VENULE_FREQ: f64 = 28.0;

// ----------------------------------------------------------------------------

/// Configures the appearance of a [`LeafGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LeafConfig {
    pub seed: u32,
    /// Overall colour of the leaf interior in linear RGB \[0, 1\].
    pub color_base: [f32; 3],
    /// Colour at the leaf edges (e.g., autumn tinge, drying) in linear RGB \[0, 1\].
    pub color_edge: [f32; 3],
    /// Tooth depth as a fraction of the local envelope half-width `[0, 1]`.
    /// Serration is scaled by the envelope so teeth stay proportional at the
    /// narrow tip — preventing the splotchy artefacts caused by a fixed offset
    /// exceeding the envelope width.  `0.12` ≈ fine serration; `0.35` ≈ coarse.
    pub serration_strength: f64,
    /// Angle factor for secondary veins.  Controls the ratio of lateral
    /// frequency to longitudinal frequency — higher → more acute vein angle.
    pub vein_angle: f64,
    /// Blend weight of the Worley capillary micro-detail layer \[0, 1\].
    pub micro_detail: f64,
    /// Normal map strength.
    pub normal_strength: f32,
    /// Number of periodic lobe half-cycles along the leaf length.
    /// `0.0` = no lobes (smooth envelope); `4.0` = four bumps per side.
    pub lobe_count: f64,
    /// Lobe modulation depth as a fraction of the base envelope width.
    /// `0.0` = no effect; `1.0` = lobes can fully indent to zero.
    pub lobe_depth: f64,
    /// Controls the shape of each lobe peak.
    /// `1.0` = smooth cosine; `>1.0` = narrower / pointier peaks;
    /// `<1.0` = wide flat peaks with sharp transitions.
    pub lobe_sharpness: f64,
    /// Fraction of the V axis reserved for the petiole (leaf stalk).
    /// `0.0` = no petiole; `0.12` = bottom 12 % of texture is stalk.
    pub petiole_length: f64,
    /// Half-width of the petiole as a fraction of UV width.
    pub petiole_width: f64,
    /// Width of the midrib ridge as a fraction of the base envelope half-width.
    /// `0.05` = crisp narrow ridge; `0.20` = broad pronounced ridge.
    pub midrib_width: f64,
    /// Number of secondary vein pairs branching from the midrib.
    /// Each pair spans one full sine cycle — `6` gives six chevron pairs.
    pub vein_count: f64,
    /// Blend weight of the venule (tertiary vein) network layer \[0, 1\].
    pub venule_strength: f64,
}

impl Default for LeafConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            color_base: [0.12, 0.35, 0.08],
            color_edge: [0.35, 0.28, 0.05],
            serration_strength: 0.12,
            vein_angle: 2.5,
            micro_detail: 0.3,
            normal_strength: 3.0,
            lobe_count: 0.0,
            lobe_depth: 0.35,
            lobe_sharpness: 1.0,
            petiole_length: 0.12,
            petiole_width: 0.022,
            midrib_width: 0.12,
            vein_count: 6.0,
            venule_strength: 0.50,
        }
    }
}

/// Output of [`LeafSampler::sample`] for a single UV coordinate.
#[derive(Clone, Debug)]
pub struct LeafSample {
    /// Height value in `[0, 1]` used to derive the normal map.
    pub height: f64,
    /// Albedo colour in linear RGB \[0, 1\].
    pub color: [f32; 3],
    /// Roughness in `[0, 1]`.
    pub roughness: f32,
}

/// Pre-initialised sampler for efficient per-pixel leaf evaluation.
///
/// Construct once with [`LeafSampler::new`], then call [`sample`](Self::sample)
/// for every pixel.  This avoids re-constructing noise objects on every call.
pub struct LeafSampler {
    config: LeafConfig,
    /// High-frequency Perlin noise used for edge serration.
    perlin: Perlin,
    /// Independent Perlin noise for the venule (tertiary vein) network.
    perlin_venule: Perlin,
    /// Worley (cellular) noise for capillary micro-venation.
    worley: Worley,
}

impl LeafSampler {
    /// Construct a sampler for the given configuration.
    pub fn new(config: LeafConfig) -> Self {
        let perlin = Perlin::new(config.seed);
        let perlin_venule = Perlin::new(config.seed.wrapping_add(2));
        // Use distance-to-feature-point mode: cell boundaries → high values.
        let worley = Worley::new(config.seed.wrapping_add(1))
            .set_return_type(ReturnType::Distance)
            .set_frequency(WORLEY_FREQ);
        Self {
            config,
            perlin,
            perlin_venule,
            worley,
        }
    }

    /// Evaluate the leaf model at normalised coordinates `(u, v)` in `[0, 1]`.
    ///
    /// Returns `None` when the UV lies outside the leaf silhouette.
    pub fn sample(&self, u: f64, v: f64) -> Option<LeafSample> {
        let c = &self.config;

        // --- Petiole region (v < petiole_length) ---
        // The petiole is the narrow stalk connecting the leaf to the stem.
        // It occupies the bottom fraction of the texture in blade UV space.
        if c.petiole_length > 0.0 && v < c.petiole_length {
            let dist = (u - 0.5).abs();
            // Taper slightly at the very bottom, widening toward the blade base.
            let half_width = c.petiole_width * (0.7 + 0.3 * v / c.petiole_length);
            if dist >= half_width {
                return None;
            }
            // Semicircular cross-section profile — raised ridge down the centre.
            let t = dist / half_width;
            let height = (1.0 - t * t).sqrt();
            return Some(LeafSample {
                height,
                color: c.color_base,
                roughness: 0.58,
            });
        }

        // Remap V into blade space [0, 1] so that the petiole region does not
        // compress the leaf blade geometry.
        let v_blade = if c.petiole_length > 0.0 {
            (v - c.petiole_length) / (1.0 - c.petiole_length)
        } else {
            v
        };

        // --- Envelope ---
        // The base envelope is extended at v_blade=0 by the petiole half-width,
        // decaying exponentially so the blade seamlessly continues from the stalk
        // without pinching to zero at the join.
        let envelope_blade = leaf_envelope(v_blade);
        let petiole_base = if c.petiole_length > 0.0 {
            c.petiole_width * (-v_blade * 12.0).exp()
        } else {
            0.0
        };
        let envelope = envelope_blade + petiole_base;
        if envelope <= 0.0 {
            return None;
        }

        // Periodic lobe modulation: a cosine wave along V scales the envelope
        // boundary, producing regular bumps (lobes) or indentations.
        // lobe_sharpness > 1 → narrower / pointier peaks.
        let effective_envelope = lobe_envelope(envelope, v_blade, c);
        if effective_envelope <= 0.0 {
            return None;
        }

        // --- Silhouette test with proportional serration ---
        // Serration is scaled by the local envelope so tooth depth stays
        // proportional everywhere — preventing noise larger than the envelope
        // from punching isolated holes near the narrow leaf tip.
        let raw_dist = (u - 0.5).abs();
        let serration = self.perlin.get([u * SERRATION_FREQ, v_blade * SERRATION_FREQ])
            * c.serration_strength
            * effective_envelope;
        if raw_dist + serration >= effective_envelope {
            return None;
        }

        // From here all height/colour calculations use `raw_dist` (unperturbed)
        // so that the venation field is smooth regardless of edge serration.

        // --- Venation height field ---

        // Blade dome: gentle cross-sectional curvature, highest at the midrib.
        let edge_frac = (raw_dist / effective_envelope).clamp(0.0, 1.0);
        let dome = 1.0 - edge_frac * edge_frac;

        // Midrib: a narrow prominent ridge at u = 0.5.  Uses the un-lobed
        // `envelope` so ridge width stays consistent through lobed leaf shapes.
        let midrib_norm = (raw_dist / (envelope * c.midrib_width.max(0.01))).min(1.0);
        let midrib = (1.0 - midrib_norm).powi(2);

        // Secondary veins: symmetric chevron ridges branching from the midrib.
        // powf(4) narrows the broad sine wave into distinct vein lines.
        let vein_freq = c.vein_count * 2.0;
        let secondary = (v_blade * vein_freq - raw_dist * vein_freq * c.vein_angle)
            .sin()
            .abs()
            .powf(4.0);

        // Venules: fine reticulate network between the secondary veins.
        // Two oblique sine sets, jittered by a low-frequency Perlin field,
        // create an organic diamond mesh.  powf(6) ensures crisp narrow ridges.
        let jitter = self.perlin_venule.get([u * 4.0, v_blade * 4.0]) * 1.8;
        let vn1 = ((u - 0.5) * VENULE_FREQ + v_blade * VENULE_FREQ * 0.38 + jitter)
            .sin()
            .abs()
            .powf(6.0);
        let vn2 = ((u - 0.5) * VENULE_FREQ - v_blade * VENULE_FREQ * 0.38 + jitter)
            .sin()
            .abs()
            .powf(6.0);
        let venule = vn1.max(vn2);

        // Micro (Worley capillary network): bright ridges at Voronoi cell
        // boundaries mimic the spongy mesophyll between the finest veinlets.
        // Worley returns distance * 2 - 1; normalise to [0, 1].
        let micro = (self.worley.get([u, v_blade]) * 0.5 + 0.5).clamp(0.0, 1.0);

        // Combine layers.
        let height = (dome * 0.15
            + midrib * 0.40
            + secondary * 0.25
            + venule * c.venule_strength * 0.15
            + micro * c.micro_detail * 0.05)
            .clamp(0.0, 1.0);

        // --- Colour ---
        // Base: blend from interior colour toward the edge colour.
        // Veins are lightened in albedo — they contain less chlorophyll and
        // reflect more, making them visible even before lighting is applied.
        let edge_t = edge_frac as f32;
        let blade_r = lerp(c.color_base[0], c.color_edge[0], edge_t);
        let blade_g = lerp(c.color_base[1], c.color_edge[1], edge_t);
        let blade_b = lerp(c.color_base[2], c.color_edge[2], edge_t);
        let vein_brightness =
            (midrib as f32 * 0.6 + secondary as f32 * 0.4).clamp(0.0, 1.0) * 0.18;
        let color = [
            (blade_r + vein_brightness).min(1.0),
            (blade_g + vein_brightness * 0.75).min(1.0),
            (blade_b + vein_brightness * 0.25).min(1.0),
        ];

        // --- Roughness ---
        // Vein ridges are slightly smoother than the surrounding mesophyll.
        let roughness = lerp(0.80, 0.52, height as f32);

        Some(LeafSample {
            height,
            color,
            roughness,
        })
    }
}

/// Convenience wrapper: constructs a temporary [`LeafSampler`] for a single call.
///
/// When sampling many pixels prefer constructing a [`LeafSampler`] directly
/// to avoid paying the noise-initialisation cost on every call.
pub fn sample_leaf(u: f64, v: f64, config: &LeafConfig) -> Option<LeafSample> {
    LeafSampler::new(config.clone()).sample(u, v)
}

/// Procedural leaf texture generator.
///
/// Produces an RGBA8 texture where `albedo.alpha` encodes the silhouette
/// (`0` = outside, `255` = inside).  Upload with
/// [`map_to_images_card`](crate::generator::map_to_images_card) so the Bevy
/// sampler does not repeat the texture across the quad.
pub struct LeafGenerator {
    config: LeafConfig,
}

impl LeafGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: LeafConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for LeafGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;

        let sampler = LeafSampler::new(self.config.clone());
        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // Neutral height (0.5) for transparent pixels → flat normal, no artefacts.
        let mut heights = vec![0.5f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;
            for x in 0..w {
                let u = x as f64 / w as f64;
                let idx = y * w + x;
                let ai = idx * 4;

                match sampler.sample(u, v) {
                    None => {
                        // Fully transparent — leave albedo RGB as zero.
                        albedo[ai + 3] = 0;
                        roughness[ai] = 255; // occlusion
                        roughness[ai + 1] = 200; // roughness
                        roughness[ai + 2] = 0; // metallic
                        roughness[ai + 3] = 255;
                    }
                    Some(s) => {
                        heights[idx] = s.height;
                        albedo[ai] = linear_to_srgb(s.color[0]);
                        albedo[ai + 1] = linear_to_srgb(s.color[1]);
                        albedo[ai + 2] = linear_to_srgb(s.color[2]);
                        albedo[ai + 3] = 255;
                        roughness[ai] = 255; // occlusion
                        roughness[ai + 1] = (s.roughness * 255.0).round() as u8;
                        roughness[ai + 2] = 0; // metallic
                        roughness[ai + 3] = 255;
                    }
                }
            }
        }

        let normal = height_to_normal(&heights, width, height, self.config.normal_strength);

        Ok(TextureMap {
            albedo,
            normal,
            roughness,
            width,
            height,
        })
    }
}

// --- private helpers --------------------------------------------------------

/// Apply periodic lobe modulation to the base envelope half-width.
///
/// Multiplies the base envelope by `1 + shaped * lobe_depth` where `shaped`
/// is a sharpness-adjusted cosine of `v * lobe_count * π`.  Returns a value
/// ≥ 0; negative results are clamped to 0 (full indentation).
///
/// When `lobe_count == 0` or `lobe_depth == 0`, returns `base` unchanged.
#[inline]
fn lobe_envelope(base: f64, v: f64, config: &LeafConfig) -> f64 {
    if config.lobe_count <= 0.0 || config.lobe_depth <= 0.0 {
        return base;
    }
    let cos_val = (v * config.lobe_count * PI).cos();
    // sign-preserving power: keeps valleys negative so they indent the envelope.
    let shaped = cos_val.signum() * cos_val.abs().powf(config.lobe_sharpness.max(0.1));
    (base * (1.0 + shaped * config.lobe_depth)).max(0.0)
}

/// Half-width of the leaf envelope at normalised V position.
///
/// Returns a value in `[0, MAX_HALF_WIDTH]`; returns `0.0` outside `(0, 1)`.
/// Shape: `sin(v·π) · exp(-v · ENVELOPE_DECAY) · MAX_HALF_WIDTH`
/// — narrow at the base (v=0, stem attachment), widest around v≈0.35,
/// tapering to a point at the tip (v=1).
fn leaf_envelope(v: f64) -> f64 {
    if v <= 0.0 || v >= 1.0 {
        return 0.0;
    }
    (v * PI).sin() * (-v * ENVELOPE_DECAY).exp() * MAX_HALF_WIDTH
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
    fn envelope_zero_at_boundaries() {
        assert_eq!(leaf_envelope(0.0), 0.0);
        assert_eq!(leaf_envelope(1.0), 0.0);
        assert_eq!(leaf_envelope(-0.1), 0.0);
        assert_eq!(leaf_envelope(1.1), 0.0);
    }

    #[test]
    fn envelope_positive_in_interior() {
        assert!(leaf_envelope(0.3) > 0.0);
        assert!(leaf_envelope(0.5) > 0.0);
        assert!(leaf_envelope(0.9) > 0.0);
    }

    #[test]
    fn midrib_always_inside() {
        let config = LeafConfig {
            serration_strength: 0.0,
            ..LeafConfig::default()
        };
        let sampler = LeafSampler::new(config);
        // u=0.5 (midrib) should be inside for all v in the middle range.
        for vi in 1..=9 {
            let v = vi as f64 / 10.0;
            assert!(
                sampler.sample(0.5, v).is_some(),
                "midrib should be inside at v={v}"
            );
        }
    }

    #[test]
    fn corners_always_outside() {
        let config = LeafConfig::default();
        let sampler = LeafSampler::new(config);
        for &(u, v) in &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            assert!(
                sampler.sample(u, v).is_none(),
                "corner ({u},{v}) should be outside the leaf"
            );
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let leaf_gen = LeafGenerator::new(LeafConfig::default());
        let map = leaf_gen.generate(64, 32).expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 32 * 4);
        assert_eq!(map.normal.len(), 64 * 32 * 4);
        assert_eq!(map.roughness.len(), 64 * 32 * 4);
    }

    #[test]
    fn generator_has_transparent_pixels() {
        let leaf_gen = LeafGenerator::new(LeafConfig::default());
        let map = leaf_gen.generate(64, 64).expect("generate failed");
        // At least some pixels should be transparent (alpha == 0).
        let has_transparent = map.albedo.chunks(4).any(|px| px[3] == 0);
        assert!(
            has_transparent,
            "leaf texture should contain transparent pixels"
        );
        // And at least some should be opaque.
        let has_opaque = map.albedo.chunks(4).any(|px| px[3] == 255);
        assert!(has_opaque, "leaf texture should contain opaque pixels");
    }
}
