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
const SERRATION_FREQ: f64 = 18.0;

/// Envelope decay rate.  Controls how quickly the leaf narrows from base to
/// tip.  Higher → narrower / more pointed leaf.
const ENVELOPE_DECAY: f64 = 2.0;

/// Maximum half-width of the leaf at its widest point (fraction of UV width).
const MAX_HALF_WIDTH: f64 = 0.44;

/// Frequency of the Worley capillary cells (cells per UV unit).
const WORLEY_FREQ: f64 = 20.0;

/// Frequency of secondary vein oscillation along V.
const VEIN_FREQ_V: f64 = 12.0;

// ----------------------------------------------------------------------------

/// Configures the appearance of a [`LeafGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LeafConfig {
    pub seed: u32,
    /// Overall colour of the leaf interior in linear RGB \[0, 1\].
    pub color_base: [f32; 3],
    /// Colour at the leaf edges (e.g., autumn tinge, drying) in linear RGB \[0, 1\].
    pub color_edge: [f32; 3],
    /// UV perturbation magnitude before the envelope check.  Controls how
    /// jagged the silhouette edges appear via Perlin noise.
    pub serration_strength: f64,
    /// Angle factor for secondary veins.  Controls the ratio of U-frequency
    /// to V-frequency of the chevron pattern — higher → more acute vein angle.
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
}

impl Default for LeafConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            color_base: [0.12, 0.35, 0.08],
            color_edge: [0.35, 0.28, 0.05],
            serration_strength: 0.025,
            vein_angle: 2.5,
            micro_detail: 0.3,
            normal_strength: 2.0,
            lobe_count: 0.0,
            lobe_depth: 0.35,
            lobe_sharpness: 1.0,
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
    /// Worley (cellular) noise for capillary micro-venation.
    worley: Worley,
}

impl LeafSampler {
    /// Construct a sampler for the given configuration.
    pub fn new(config: LeafConfig) -> Self {
        let perlin = Perlin::new(config.seed);
        // Use distance-to-feature-point mode: cell boundaries → high values.
        let worley = Worley::new(config.seed.wrapping_add(1))
            .set_return_type(ReturnType::Distance)
            .set_frequency(WORLEY_FREQ);
        Self {
            config,
            perlin,
            worley,
        }
    }

    /// Evaluate the leaf model at normalised coordinates `(u, v)` in `[0, 1]`.
    ///
    /// Returns `None` when the UV lies outside the leaf silhouette.
    pub fn sample(&self, u: f64, v: f64) -> Option<LeafSample> {
        let c = &self.config;

        // --- Envelope check (with lobe modulation + serration perturbation) ---
        let envelope = leaf_envelope(v);
        if envelope <= 0.0 {
            return None;
        }

        // Periodic lobe modulation: a cosine wave along V scales the envelope
        // boundary, producing regular bumps (lobes) or indentations.
        // lobe_sharpness > 1 → narrower / pointier peaks.
        let effective_envelope = lobe_envelope(envelope, v, c);
        if effective_envelope <= 0.0 {
            return None;
        }

        // Perturb the distance-from-midrib with Perlin noise to create organic
        // serrated edges on top of the periodic lobe pattern.
        let serration =
            self.perlin.get([u * SERRATION_FREQ, v * SERRATION_FREQ]) * c.serration_strength;
        let dist_from_midrib = (u - 0.5).abs() + serration;

        if dist_from_midrib >= effective_envelope {
            return None;
        }

        // --- Venation height field ---

        // Primary vein (midrib): falls off from u=0.5 to the base envelope edge
        // so the ridge follows the underlying leaf shape, not the lobed outline.
        let primary = (1.0 - dist_from_midrib / envelope).clamp(0.0, 1.0);

        // Secondary veins: chevron pattern branching symmetrically from the
        // midrib.  `vein_angle` scales the lateral frequency relative to V,
        // controlling how acute the vein branches appear.
        let secondary = (v * VEIN_FREQ_V - (u - 0.5).abs() * VEIN_FREQ_V * c.vein_angle)
            .sin()
            .abs();

        // Tertiary veins (capillary network): Worley distance-to-edge gives
        // bright ridges at Voronoi cell boundaries, mimicking the leaf's
        // spongy mesophyll network.
        // Worley returns `distance * 2.0 - 1.0`; normalise back to [0, 1].
        let micro = (self.worley.get([u, v]) * 0.5 + 0.5).clamp(0.0, 1.0);

        // Combine layers: primary dominates (0.5), secondary adds structure
        // (0.2), micro_detail scales the capillary contribution (up to 0.3).
        let height =
            (primary * 0.5 + secondary * 0.2 + micro * c.micro_detail * 0.3).clamp(0.0, 1.0);

        // --- Colour ---
        // Blend from base colour toward the edge colour as we approach the
        // silhouette boundary, simulating drying at leaf margins.
        let edge_t = (dist_from_midrib / effective_envelope).clamp(0.0, 1.0) as f32;
        let color = [
            lerp(c.color_base[0], c.color_edge[0], edge_t),
            lerp(c.color_base[1], c.color_edge[1], edge_t),
            lerp(c.color_base[2], c.color_edge[2], edge_t),
        ];

        // --- Roughness ---
        // Vein ridges (high height) are slightly smoother; mesophyll cells
        // (lower) are rougher.
        let roughness = lerp(0.80, 0.55, height as f32);

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
