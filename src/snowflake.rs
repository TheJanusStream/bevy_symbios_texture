//! Snowflake sprite generator.
//!
//! A dendritic flake with N-fold symmetry: a small central plate, one main
//! arm per sector, and paired side branches along each arm.  The angle is
//! folded into a single mirrored sector so the maths only ever draws one
//! half-arm — symmetry comes for free.  Per-variant cells jitter arm
//! length, branch stations, branch lengths, and branch angle, which is
//! where the "no two snowflakes alike" character comes from.
//!
//! Upload with [`map_to_images_card`](crate::generator::map_to_images_card);
//! see [`crate::sprite`] for the shared atlas conventions.

use std::f64::consts::TAU;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    sprite::{CellRng, SpriteCell, SpriteSample, generate_atlas},
};

/// Hard cap on side-branch pairs; also sizes the per-cell branch tables.
const MAX_BRANCH_PAIRS: usize = 5;

/// Configures the appearance of a [`SnowflakeGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SnowflakeConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Symmetry order `[3, 8]`.  Real snow is hexagonal (`6`); other
    /// orders read as stylised frost stars.
    pub arms: usize,
    /// Flake colour in linear RGB \[0, 1\].
    pub color: [f32; 3],
    /// Radius of the central plate as a fraction of the cell half-extent
    /// `[0, 0.4]`.
    pub core_radius: f64,
    /// Half-width of the main arm at the centre `[0.01, 0.12]`; arms taper
    /// toward the tip.
    pub arm_width: f64,
    /// Number of side-branch pairs per arm `[0, 5]`.
    pub branch_pairs: usize,
    /// Angle between a side branch and the main arm, radians
    /// `[0.3, 1.4]`.  Real dendrites branch at ~60° (`1.05`).
    pub branch_angle: f64,
    /// Side-branch length relative to the remaining arm length at the
    /// branch station `[0.1, 1]`.
    pub branch_scale: f64,
    /// Anti-aliasing edge width in cell units `[0.005, 0.08]`.
    pub softness: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for SnowflakeConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 2,
            variant_cols: 2,
            arms: 6,
            color: [0.92, 0.96, 1.0],
            core_radius: 0.12,
            arm_width: 0.045,
            branch_pairs: 3,
            branch_angle: 1.05,
            branch_scale: 0.45,
            softness: 0.02,
            normal_strength: 1.5,
        }
    }
}

/// One side branch baked for a variant cell: where it leaves the main arm
/// and how far it reaches.
#[derive(Clone, Copy, Default)]
struct Branch {
    /// Station along the main arm (cell half-extent units).
    station: f64,
    /// Branch length (cell half-extent units).
    length: f64,
}

/// One baked snowflake variant.
struct SnowflakeCell {
    config: SnowflakeConfig,
    arms: usize,
    /// Main-arm length in cell half-extent units.
    arm_len: f64,
    /// Per-cell jittered branch angle.
    branch_angle: f64,
    branch_count: usize,
    branches: [Branch; MAX_BRANCH_PAIRS],
}

impl SnowflakeCell {
    fn new(config: &SnowflakeConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        let arms = config.arms.clamp(3, 8);
        let core = config.core_radius.clamp(0.0, 0.4);
        let arm_len = 0.94 * rng.range(0.85, 1.0);
        let branch_count = config.branch_pairs.min(MAX_BRANCH_PAIRS);
        let branch_scale = config.branch_scale.clamp(0.1, 1.0);

        let mut branches = [Branch::default(); MAX_BRANCH_PAIRS];
        for (k, b) in branches.iter_mut().enumerate().take(branch_count) {
            // Evenly spaced stations between the plate and the tip, each
            // jittered within its slot so variants differ structurally.
            let t = (k as f64 + rng.range(0.55, 1.0)) / (branch_count as f64 + 1.0);
            let station = core + t * (arm_len - core);
            b.station = station;
            b.length = branch_scale * (arm_len - station) * rng.range(0.7, 1.1);
        }

        Self {
            config: config.clone(),
            arms,
            arm_len,
            branch_angle: config.branch_angle.clamp(0.3, 1.4) * rng.range(0.9, 1.1),
            branch_count,
            branches,
        }
    }
}

/// Distance from point `p` to the segment `a → b` (all in 2-D).
fn dist_to_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let ab = (b.0 - a.0, b.1 - a.1);
    let ap = (p.0 - a.0, p.1 - a.1);
    let len_sq = ab.0 * ab.0 + ab.1 * ab.1;
    let t = if len_sq > 1e-12 {
        ((ap.0 * ab.0 + ap.1 * ab.1) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let cx = a.0 + ab.0 * t - p.0;
    let cy = a.1 + ab.1 * t - p.1;
    (cx * cx + cy * cy).sqrt()
}

impl SpriteCell for SnowflakeCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;
        let dx = (u - 0.5) * 2.0;
        let dy = (v - 0.5) * 2.0;
        let r = (dx * dx + dy * dy).sqrt();

        // Fold the angle into one mirrored sector: the arm axis is the
        // sector's x-axis, so every arm and its mirror image are handled by
        // drawing a single half-arm.
        let sector = TAU / self.arms as f64;
        let theta = dy.atan2(dx).rem_euclid(sector);
        let phi = theta.min(sector - theta);
        let p = (r * phi.cos(), r * phi.sin());

        let core = c.core_radius.clamp(0.0, 0.4);
        let width = c.arm_width.clamp(0.01, 0.12);
        let softness = c.softness.clamp(0.005, 0.08);

        // Signed distance to the flake: negative inside.  Start from the
        // central plate (gently scalloped by the fold so it reads as a
        // plate rather than a dot) …
        let plate = r - core * (0.88 + 0.12 * (phi / (sector * 0.5)).cos());
        let mut d = plate;

        // … the tapered main arm …
        let taper = 1.0 - 0.85 * (p.0 / self.arm_len).clamp(0.0, 1.0);
        let d_arm = dist_to_segment(p, (0.0, 0.0), (self.arm_len, 0.0)) - width * taper;
        d = d.min(d_arm);

        // … and the side branches (mirror side covered by the fold).
        let (sin_b, cos_b) = self.branch_angle.sin_cos();
        for b in &self.branches[..self.branch_count] {
            let root = (b.station, 0.0);
            let tip = (b.station + b.length * cos_b, b.length * sin_b);
            let branch_taper = taper * 0.7;
            let d_branch = dist_to_segment(p, root, tip) - width * branch_taper;
            d = d.min(d_branch);
        }

        let alpha = (1.0 - d / softness).clamp(0.0, 1.0);
        // Crystal reads best with a slightly raised spine: height peaks at
        // the centreline of whatever feature we are inside.
        let height = alpha * (0.55 + 0.45 * (1.0 - r).clamp(0.0, 1.0));

        SpriteSample {
            color: c.color,
            alpha,
            height,
            roughness: 0.35,
        }
    }
}

/// Procedural snowflake sprite generator.
///
/// See the [module documentation](self) for the visual model.
pub struct SnowflakeGenerator {
    config: SnowflakeConfig,
}

impl SnowflakeGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: SnowflakeConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for SnowflakeGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| SnowflakeCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    fn single_cell() -> SnowflakeConfig {
        SnowflakeConfig {
            variant_rows: 1,
            variant_cols: 1,
            ..SnowflakeConfig::default()
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = SnowflakeGenerator::new(SnowflakeConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn has_opaque_centre_and_transparent_background() {
        let map = SnowflakeGenerator::new(single_cell())
            .generate(128, 128)
            .expect("generate failed");
        let centre = (64 * 128 + 64) * 4;
        assert_eq!(map.albedo[centre + 3], 255, "plate centre must be opaque");
        // Most of a snowflake cell is air.
        let transparent = map.albedo.chunks(4).filter(|px| px[3] == 0).count();
        assert!(
            transparent > 128 * 128 / 2,
            "snowflake should be mostly transparent (got {transparent} clear texels)"
        );
    }

    #[test]
    fn flake_is_n_fold_symmetric() {
        // With 6 arms, rotating a high-alpha texel by 60° must land on
        // another high-alpha texel.  Probe along the +X arm axis... but the
        // arm axis depends only on the fold, which starts at angle 0, so
        // (r, 0) and (r·cos60°, r·sin60°) are both on arm centrelines.
        let map = SnowflakeGenerator::new(single_cell())
            .generate(256, 256)
            .expect("generate failed");
        let alpha_at = |x: f64, y: f64| -> u8 {
            let px = ((x + 1.0) / 2.0 * 256.0) as usize;
            let py = ((y + 1.0) / 2.0 * 256.0) as usize;
            map.albedo[(py.min(255) * 256 + px.min(255)) * 4 + 3]
        };
        let r = 0.5;
        let a0 = alpha_at(r, 0.0);
        let (s, c) = (PI / 3.0).sin_cos();
        let a60 = alpha_at(r * c, r * s);
        assert!(a0 > 128, "arm centreline should be opaque at r=0.5");
        assert!(
            a0.abs_diff(a60) < 64,
            "6-fold symmetry violated: alpha {a0} vs {a60}"
        );
    }

    #[test]
    fn branch_pairs_zero_is_a_bare_star() {
        let config = SnowflakeConfig {
            branch_pairs: 0,
            ..single_cell()
        };
        let map = SnowflakeGenerator::new(config)
            .generate(64, 64)
            .expect("generate failed");
        assert!(map.albedo.chunks(4).any(|px| px[3] == 255));
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = SnowflakeGenerator::new(SnowflakeConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = SnowflakeGenerator::new(SnowflakeConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
