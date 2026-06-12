//! Ring sprite generator.
//!
//! A soft annulus with optional angular waviness: shockwaves, water-drop
//! ripples, magic circles, halos.  Per-variant cells jitter radius and
//! wave phase, so a small atlas gives a ripple train where no two rings
//! are quite alike.
//!
//! Upload with [`map_to_images_card`](crate::generator::map_to_images_card);
//! see [`crate::sprite`] for the shared atlas conventions.

use std::f64::consts::TAU;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    sprite::{CellRng, SpriteCell, SpriteSample, generate_atlas},
};

/// Configures the appearance of a [`RingGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RingConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Ring colour in linear RGB \[0, 1\].
    pub color: [f32; 3],
    /// Centreline radius as a fraction of the cell half-extent
    /// `[0.1, 0.9]`.
    pub radius: f64,
    /// Annulus half-thickness as a fraction of the cell half-extent
    /// `[0.01, 0.5]`.
    pub thickness: f64,
    /// Cross-section falloff exponent `[0.5, 6]`.  Higher → a crisp band
    /// with soft edges; lower → a diffuse glow ring.
    pub falloff: f64,
    /// Angular radius modulation amplitude `[0, 0.3]`.  `0` is a perfect
    /// circle; higher values wobble the ring organically.
    pub waviness: f64,
    /// Number of waviness lobes around the ring `[2, 16]`.
    pub wave_count: usize,
    /// Per-variant radius jitter `[0, 0.4]`.
    pub radius_jitter: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for RingConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 1,
            variant_cols: 1,
            color: [0.85, 0.93, 1.0],
            radius: 0.6,
            thickness: 0.12,
            falloff: 2.0,
            waviness: 0.0,
            wave_count: 6,
            radius_jitter: 0.1,
            normal_strength: 1.0,
        }
    }
}

/// One baked ring variant: jittered radius and wave phase.
struct RingCell {
    config: RingConfig,
    radius: f64,
    phase: f64,
}

impl RingCell {
    fn new(config: &RingConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        let jitter = config.radius_jitter.clamp(0.0, 0.4);
        let base = config.radius.clamp(0.1, 0.9);
        Self {
            config: config.clone(),
            radius: base * (1.0 - rng.next_f64() * jitter),
            phase: rng.range(0.0, TAU),
        }
    }
}

impl SpriteCell for RingCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;
        let dx = (u - 0.5) * 2.0;
        let dy = (v - 0.5) * 2.0;
        let r = (dx * dx + dy * dy).sqrt();
        let theta = dy.atan2(dx);

        let waviness = c.waviness.clamp(0.0, 0.3);
        let wave_count = c.wave_count.clamp(2, 16) as f64;
        let centre = (self.radius * (1.0 + waviness * (wave_count * theta + self.phase).sin()))
            // The wobble must never push the centreline outside the cell.
            .min(0.95);

        let thickness = c.thickness.clamp(0.01, 0.5);
        let falloff = c.falloff.clamp(0.5, 6.0);
        let alpha = (1.0 - (r - centre).abs() / thickness)
            .clamp(0.0, 1.0)
            .powf(falloff);

        SpriteSample {
            color: c.color,
            alpha,
            // Torus cross-section: height crests at the centreline.
            height: alpha,
            roughness: 0.8,
        }
    }
}

/// Procedural ring sprite generator.
///
/// See the [module documentation](self) for the visual model.
pub struct RingGenerator {
    config: RingConfig,
}

impl RingGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: RingConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for RingGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| RingCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_jitter() -> RingConfig {
        RingConfig {
            radius_jitter: 0.0,
            ..RingConfig::default()
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = RingGenerator::new(RingConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn centre_and_corner_are_transparent_but_band_is_not() {
        let map = RingGenerator::new(no_jitter())
            .generate(128, 128)
            .expect("generate failed");
        let at = |x: usize, y: usize| map.albedo[(y * 128 + x) * 4 + 3];
        assert_eq!(at(64, 64), 0, "ring centre must be hollow");
        assert_eq!(at(0, 0), 0, "corner must be transparent");
        // Centreline at radius 0.6 → 64 + 0.6·64 ≈ 102 on the +X axis.
        assert!(at(102, 64) > 200, "ring band should be near-opaque");
    }

    #[test]
    fn waviness_perturbs_the_band() {
        let wavy = RingConfig {
            waviness: 0.25,
            ..no_jitter()
        };
        let a = RingGenerator::new(no_jitter())
            .generate(64, 64)
            .expect("generate failed");
        let b = RingGenerator::new(wavy)
            .generate(64, 64)
            .expect("generate failed");
        assert_ne!(a.albedo, b.albedo);
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = RingGenerator::new(RingConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = RingGenerator::new(RingConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
