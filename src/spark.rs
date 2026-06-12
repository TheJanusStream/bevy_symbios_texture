//! Spark / star sprite generator.
//!
//! An N-pointed streak burst: a bright core with radial arms that fade
//! toward their tips.  Reads as embers, glints, impact sparks, or magic
//! sparkles depending on palette and arm count.  Per-variant cells jitter
//! the rotation phase and individual arm lengths.
//!
//! Upload with [`map_to_images_card`](crate::generator::map_to_images_card);
//! see [`crate::sprite`] for the shared atlas conventions.

use std::f64::consts::TAU;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    sprite::{CellRng, SpriteCell, SpriteSample, generate_atlas, lerp_color},
};

/// Hard cap on the arm count; also the size of the per-cell arm-length
/// table.  Twelve arms at typical sprite resolutions is already past the
/// point where individual streaks resolve.
const MAX_POINTS: usize = 12;

/// Configures the appearance of a [`SparkGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparkConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Number of streak arms `[2, 12]`.
    pub points: usize,
    /// Colour of the central glow in linear RGB \[0, 1\].
    pub color_core: [f32; 3],
    /// Colour toward the arm tips in linear RGB \[0, 1\].
    pub color_tip: [f32; 3],
    /// Radius of the solid central glow as a fraction of the cell
    /// half-extent `[0.02, 0.5]`.
    pub core_radius: f64,
    /// Angular tightness of each arm `[0.5, 10]`.  Higher → needle-thin
    /// streaks; lower → fat lobes that merge into a star polygon.
    pub arm_sharpness: f64,
    /// Radial fade exponent along each arm `[0.5, 6]`.
    pub falloff: f64,
    /// Per-arm length jitter `[0, 0.8]`: each arm is shortened by a random
    /// fraction up to this value, per variant cell.
    pub length_jitter: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for SparkConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 2,
            variant_cols: 2,
            points: 4,
            color_core: [1.0, 0.95, 0.8],
            color_tip: [1.0, 0.45, 0.1],
            core_radius: 0.12,
            arm_sharpness: 3.0,
            falloff: 1.8,
            length_jitter: 0.3,
            normal_strength: 1.0,
        }
    }
}

/// One baked spark variant: rotation phase plus a per-arm length table.
struct SparkCell {
    config: SparkConfig,
    points: usize,
    phase: f64,
    arm_len: [f64; MAX_POINTS],
}

impl SparkCell {
    fn new(config: &SparkConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        let points = config.points.clamp(2, MAX_POINTS);
        let jitter = config.length_jitter.clamp(0.0, 0.8);
        let mut arm_len = [0.0; MAX_POINTS];
        for slot in arm_len.iter_mut().take(points) {
            // 0.96: same clip-safety margin as the soft disc.
            *slot = 0.96 * (1.0 - rng.next_f64() * jitter);
        }
        Self {
            config: config.clone(),
            points,
            phase: rng.range(0.0, TAU),
            arm_len,
        }
    }
}

impl SpriteCell for SparkCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;
        let dx = (u - 0.5) * 2.0;
        let dy = (v - 0.5) * 2.0;
        let r = (dx * dx + dy * dy).sqrt();
        let theta = dy.atan2(dx);

        // Position within the arm fan: integer part selects the nearest
        // arm, fractional distance (0 at an arm centre, 1 halfway between
        // arms) drives the angular profile.
        let fan = (theta - self.phase) / TAU * self.points as f64;
        let nearest = fan.round();
        let angular = (fan - nearest).abs() * 2.0; // [0, 1]
        let arm = (nearest.rem_euclid(self.points as f64)) as usize % self.points;

        let core = c.core_radius.clamp(0.02, 0.5);
        let sharpness = c.arm_sharpness.clamp(0.5, 10.0);
        let falloff = c.falloff.clamp(0.5, 6.0);

        // Envelope radius at this angle: core everywhere, stretching to the
        // arm length at the arm centreline.
        let profile = (1.0 - angular).powf(sharpness);
        let envelope = core + (self.arm_len[arm] - core) * profile;

        // Radial fade inside the envelope plus a soft solid core so the
        // centre never shows the angular structure.
        let arm_alpha = if r < envelope {
            (1.0 - r / envelope).powf(falloff)
        } else {
            0.0
        };
        let core_alpha = (1.0 - r / core).clamp(0.0, 1.0);
        let alpha = arm_alpha.max(core_alpha * core_alpha);

        let core_blend = (alpha * alpha) as f32;
        SpriteSample {
            color: lerp_color(c.color_tip, c.color_core, core_blend),
            alpha,
            height: alpha,
            roughness: 0.9,
        }
    }
}

/// Procedural spark / star sprite generator.
///
/// See the [module documentation](self) for the visual model.
pub struct SparkGenerator {
    config: SparkConfig,
}

impl SparkGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: SparkConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for SparkGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| SparkCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_cell(points: usize) -> SparkConfig {
        SparkConfig {
            variant_rows: 1,
            variant_cols: 1,
            points,
            length_jitter: 0.0,
            ..SparkConfig::default()
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = SparkGenerator::new(SparkConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn centre_is_opaque_with_transparent_surround() {
        let map = SparkGenerator::new(single_cell(4))
            .generate(64, 64)
            .expect("generate failed");
        let centre = (32 * 64 + 32) * 4;
        assert!(map.albedo[centre + 3] > 200, "core should be near-opaque");
        assert!(
            map.albedo.chunks(4).any(|px| px[3] == 0),
            "spark must have fully transparent texels"
        );
    }

    #[test]
    fn arms_extend_past_the_core() {
        // With phase jitter disabled per-arm coverage is hard to probe at a
        // fixed texel, so measure instead: an opaque-ish texel must exist
        // outside the core radius.
        let map = SparkGenerator::new(single_cell(4))
            .generate(128, 128)
            .expect("generate failed");
        let core_px = (0.12 * 64.0) as i32; // core_radius × half-size
        let found = map.albedo.chunks(4).enumerate().any(|(i, px)| {
            let x = (i % 128) as i32 - 64;
            let y = (i / 128) as i32 - 64;
            px[3] > 100 && (x * x + y * y) > (core_px * 3) * (core_px * 3)
        });
        assert!(found, "streaks should reach well beyond the core");
    }

    #[test]
    fn point_count_is_clamped() {
        // 100 points clamps to MAX_POINTS without panicking on the arm table.
        let map = SparkGenerator::new(single_cell(100))
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 32 * 32 * 4);
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = SparkGenerator::new(SparkConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = SparkGenerator::new(SparkConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
