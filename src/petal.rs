//! Petal sprite generator.
//!
//! A single flower petal: an obovate blade with a soft colour gradient
//! from throat to edge and an optional emarginate (notched) tip.  Drifting
//! petal-fall particles, blossom decals, or — at 1×1 — a building block
//! for procedural flowers.  Per-variant cells jitter length, width, skew,
//! and notch so a petal shower reads as organic.
//!
//! # Coordinate conventions
//! Local cell UV: the petal axis runs along V with the attachment
//! (throat) at `v = 1` (bottom) and the tip at `v = 0` (top) — the same
//! base-down convention as the leaf card.
//!
//! Upload with [`map_to_images_card`](crate::generator::map_to_images_card);
//! see [`crate::sprite`] for the shared atlas conventions.

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    sprite::{CellRng, SpriteCell, SpriteSample, generate_atlas, lerp_color},
};

/// Anti-aliasing half-width of the silhouette edge, in cell units.
const EDGE_SOFTNESS: f64 = 0.015;

/// Configures the appearance of a [`PetalGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PetalConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Main blade colour in linear RGB \[0, 1\].
    pub color_base: [f32; 3],
    /// Colour toward the silhouette edge in linear RGB \[0, 1\].
    pub color_edge: [f32; 3],
    /// Colour at the throat (attachment) in linear RGB \[0, 1\] — many
    /// real petals carry a nectar-guide tint here.
    pub color_throat: [f32; 3],
    /// Petal length as a fraction of the cell `[0.4, 1]`.
    pub length: f64,
    /// Maximum blade width as a fraction of the cell `[0.15, 0.95]`.
    pub width: f64,
    /// Position of maximum width along the petal `[0.3, 0.9]` measured
    /// from the throat (`0.65` ≈ obovate, the classic petal silhouette).
    pub peak: f64,
    /// Tip notch radius as a fraction of the cell `[0, 0.25]`.  `0` is an
    /// entire (smooth) tip; larger values cut a heart-shaped notch.
    pub tip_notch: f64,
    /// Lateral shading strength `[0, 1]` — fakes the petal curling toward
    /// the viewer by darkening one rim and lightening the other.
    pub curl: f64,
    /// Maximum per-variant skew of the petal axis `[0, 0.4]`.
    pub asymmetry: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for PetalConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 2,
            variant_cols: 2,
            color_base: [0.98, 0.72, 0.82],
            color_edge: [0.93, 0.5, 0.66],
            color_throat: [0.99, 0.88, 0.55],
            length: 0.92,
            width: 0.6,
            peak: 0.65,
            tip_notch: 0.08,
            curl: 0.4,
            asymmetry: 0.15,
            normal_strength: 1.5,
        }
    }
}

/// One baked petal variant.
struct PetalCell {
    config: PetalConfig,
    length: f64,
    width: f64,
    /// Sine-power exponent positioning the width peak.
    peak_pow: f64,
    notch: f64,
    /// Signed axis skew for this variant.
    skew: f64,
    /// Which rim the curl highlight falls on (±1).
    curl_side: f64,
}

impl PetalCell {
    fn new(config: &PetalConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        let peak = config.peak.clamp(0.3, 0.9);
        Self {
            config: config.clone(),
            length: config.length.clamp(0.4, 1.0) * rng.range(0.88, 1.0),
            width: config.width.clamp(0.15, 0.95) * rng.range(0.85, 1.0),
            // sin(π·t^p) peaks where t^p = 1/2, so p = ln 2⁻¹ / ln(peak).
            peak_pow: (0.5f64).ln() / peak.ln(),
            notch: config.tip_notch.clamp(0.0, 0.25) * rng.range(0.6, 1.2),
            skew: config.asymmetry.clamp(0.0, 0.4) * rng.range(-1.0, 1.0),
            curl_side: if rng.next_f64() < 0.5 { -1.0 } else { 1.0 },
        }
    }
}

impl SpriteCell for PetalCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;

        // Petal-axis parameter: t = 0 at the throat (v = 1), 1 at the tip.
        let margin = (1.0 - self.length) * 0.5;
        let t = ((1.0 - v) - margin) / self.length;

        let out = SpriteSample {
            color: c.color_edge,
            alpha: 0.0,
            height: 0.0,
            roughness: 0.55,
        };
        if !(0.0..=1.0).contains(&t) {
            return out;
        }

        // Obovate envelope with a skewed midline.
        let half_width = self.width * 0.5 * (std::f64::consts::PI * t.powf(self.peak_pow)).sin();
        let centre = 0.5 + self.skew * t * t;
        let lateral = u - centre;

        // Emarginate tip: subtract a notch disc centred just past the tip.
        let tip_y = margin; // v-coordinate of the tip
        let notch_d = ((u - centre).powi(2) + (v - (tip_y - self.notch * 0.4)).powi(2)).sqrt();

        // Signed distance to the silhouette (positive outside).
        let d = (lateral.abs() - half_width).max(self.notch - notch_d);
        let alpha = ((EDGE_SOFTNESS - d) / EDGE_SOFTNESS).clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return out;
        }

        // Lateral position within the blade, –1 .. 1.
        let rim = if half_width > 1e-9 {
            (lateral / half_width).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Colour: throat gradient near the base, edge tint near the rim.
        let throat_blend = (1.0 - t / 0.25).clamp(0.0, 1.0) as f32;
        let edge_blend = (rim.abs().powf(2.0) * 0.9) as f32;
        let mut color = lerp_color(c.color_base, c.color_edge, edge_blend);
        color = lerp_color(color, c.color_throat, throat_blend);

        // Curl shading: one rim catches light, the other falls away.
        let curl = c.curl.clamp(0.0, 1.0);
        let shade = 1.0 + (curl * 0.25 * rim * self.curl_side) as f32;
        color = [
            (color[0] * shade).clamp(0.0, 1.0),
            (color[1] * shade).clamp(0.0, 1.0),
            (color[2] * shade).clamp(0.0, 1.0),
        ];

        // Cross-section dome, tilted by the curl.
        let dome = 1.0 - rim * rim;
        let height = (0.25 + 0.6 * dome + 0.15 * curl * rim * self.curl_side).clamp(0.0, 1.0);

        SpriteSample {
            color,
            alpha,
            height: height * alpha,
            roughness: 0.55,
        }
    }
}

/// Procedural petal sprite generator.
///
/// See the [module documentation](self) for the visual model.
pub struct PetalGenerator {
    config: PetalConfig,
}

impl PetalGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: PetalConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for PetalGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| PetalCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_cell() -> PetalConfig {
        PetalConfig {
            variant_rows: 1,
            variant_cols: 1,
            asymmetry: 0.0,
            ..PetalConfig::default()
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = PetalGenerator::new(PetalConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn blade_is_opaque_with_transparent_margins() {
        let map = PetalGenerator::new(single_cell())
            .generate(128, 128)
            .expect("generate failed");
        let at = |x: usize, y: usize| map.albedo[(y * 128 + x) * 4 + 3];
        // Widest point of the blade (peak = 0.65 from the throat at the
        // bottom → v ≈ 0.39) on the midline.
        assert_eq!(at(64, 50), 255, "blade centre must be opaque");
        assert_eq!(at(4, 64), 0, "left margin must be transparent");
        assert_eq!(at(124, 64), 0, "right margin must be transparent");
    }

    #[test]
    fn tip_notch_cuts_the_apex() {
        let entire = PetalConfig {
            tip_notch: 0.0,
            ..single_cell()
        };
        let notched = PetalConfig {
            tip_notch: 0.25,
            ..single_cell()
        };
        let a = PetalGenerator::new(entire)
            .generate(128, 128)
            .expect("generate failed");
        let b = PetalGenerator::new(notched)
            .generate(128, 128)
            .expect("generate failed");
        // Probe the midline near the tip: the notch removes coverage there.
        let probe = |m: &crate::generator::TextureMap| {
            (8..40)
                .map(|y| m.albedo[(y * 128 + 64) * 4 + 3] as u32)
                .sum::<u32>()
        };
        assert!(
            probe(&b) < probe(&a),
            "notched tip should remove midline coverage near the apex"
        );
    }

    #[test]
    fn variants_differ() {
        let map = PetalGenerator::new(PetalConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        let differs = (0..64usize).any(|i| {
            let a = map.albedo[((40 * 128) + i) * 4 + 3];
            let b = map.albedo[((40 * 128) + i + 64) * 4 + 3];
            a != b
        });
        assert!(differs, "petal atlas cells should not be identical");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = PetalGenerator::new(PetalConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = PetalGenerator::new(PetalConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
