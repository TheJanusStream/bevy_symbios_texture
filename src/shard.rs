//! Shard sprite generator.
//!
//! An irregular rock-chip / debris-flake silhouette: a star-shaped polygon
//! with jittered vertices, a darkened rim, and noise-grained interior with
//! a random facet tilt.  Impact debris, crumbling masonry, shattered ice,
//! kicked-up gravel.  Per-variant cells re-roll the whole polygon, so an
//! atlas is a handful of distinct chips.
//!
//! Upload with [`map_to_images_card`](crate::generator::map_to_images_card);
//! see [`crate::sprite`] for the shared atlas conventions.

use std::f64::consts::TAU;

use noise::Perlin;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    sprite::{CellRng, SpriteCell, SpriteSample, fbm2, generate_atlas, lerp_color},
};

/// Hard cap on polygon vertex count; also sizes the per-cell vertex tables.
const MAX_SIDES: usize = 9;

/// Anti-aliasing half-width of the silhouette edge, in cell units.
const EDGE_SOFTNESS: f64 = 0.02;

/// Configures the appearance of a [`ShardGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ShardConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Interior colour in linear RGB \[0, 1\].
    pub color_base: [f32; 3],
    /// Rim colour in linear RGB \[0, 1\] — fractured edges read darker
    /// (dirt, shadow) or lighter (fresh stone) than the face.
    pub color_edge: [f32; 3],
    /// Polygon vertex count `[3, 9]`.
    pub sides: usize,
    /// Vertex jitter `[0, 0.9]`: angular and radial randomisation of the
    /// silhouette.  `0` is a regular polygon; high values give jagged
    /// chips.
    pub irregularity: f64,
    /// Width of the darkened rim as a fraction of the shard radius
    /// `[0.02, 0.5]`.
    pub edge_band: f64,
    /// Interior grain strength `[0, 1]` — fractal noise modulating colour
    /// and height.
    pub grain: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 2,
            variant_cols: 2,
            color_base: [0.46, 0.43, 0.4],
            color_edge: [0.24, 0.22, 0.21],
            sides: 5,
            irregularity: 0.45,
            edge_band: 0.18,
            grain: 0.35,
            normal_strength: 2.5,
        }
    }
}

/// One baked shard variant: the polygon's vertex table in polar form plus
/// the facet tilt and noise offset.
struct ShardCell {
    config: ShardConfig,
    sides: usize,
    /// Vertex angles, strictly increasing within one turn.
    angles: [f64; MAX_SIDES],
    /// Vertex radii (cell half-extent units).
    radii: [f64; MAX_SIDES],
    /// Per-cell facet tilt (height gradient across the face).
    tilt: (f64, f64),
    perlin: Perlin,
    noise_offset: (f64, f64),
}

impl ShardCell {
    fn new(config: &ShardConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        let sides = config.sides.clamp(3, MAX_SIDES);
        let irregularity = config.irregularity.clamp(0.0, 0.9);

        let mut angles = [0.0; MAX_SIDES];
        let mut radii = [0.0; MAX_SIDES];
        for i in 0..sides {
            // Jitter each vertex within ±0.45 of its sector so the angular
            // ordering (required by the star-shaped radius interpolation in
            // `sample`) is preserved.
            let jitter = irregularity * rng.range(-0.45, 0.45);
            angles[i] = (i as f64 + jitter) / sides as f64 * TAU;
            radii[i] = 0.9 * (1.0 - irregularity * 0.6 * rng.next_f64());
        }

        let tilt_angle = rng.range(0.0, TAU);
        let tilt_mag = rng.range(0.1, 0.35);
        Self {
            config: config.clone(),
            sides,
            angles,
            radii,
            tilt: (tilt_angle.cos() * tilt_mag, tilt_angle.sin() * tilt_mag),
            perlin: Perlin::new(config.seed),
            noise_offset: (rng.range(0.0, 256.0), rng.range(0.0, 256.0)),
        }
    }

    /// Boundary radius of the star-shaped polygon at angle `theta`,
    /// linearly interpolated between the bracketing vertices.
    fn boundary(&self, theta: f64) -> f64 {
        let n = self.sides;
        // Find the vertex pair whose angular span contains theta.  The
        // table is sorted; wrap-around is handled by offsetting one turn.
        for i in 0..n {
            let a0 = self.angles[i];
            let (a1, r1) = if i + 1 < n {
                (self.angles[i + 1], self.radii[i + 1])
            } else {
                (self.angles[0] + TAU, self.radii[0])
            };
            let th = if theta < a0 { theta + TAU } else { theta };
            if th >= a0 && th <= a1 {
                let t = if a1 > a0 { (th - a0) / (a1 - a0) } else { 0.0 };
                return self.radii[i] + (r1 - self.radii[i]) * t;
            }
        }
        // Unreachable for a well-formed table; fail soft.
        self.radii[0]
    }
}

impl SpriteCell for ShardCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;
        let dx = (u - 0.5) * 2.0;
        let dy = (v - 0.5) * 2.0;
        let r = (dx * dx + dy * dy).sqrt();
        let theta = dy.atan2(dx).rem_euclid(TAU);

        let boundary = self.boundary(theta);
        // Radial signed distance — approximate but adequate at sprite
        // resolutions, and exact along each face normal's radial component.
        let d = r - boundary;
        let alpha = ((EDGE_SOFTNESS - d) / EDGE_SOFTNESS).clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return SpriteSample {
                color: c.color_edge,
                alpha: 0.0,
                height: 0.0,
                roughness: 0.95,
            };
        }

        let grain = c.grain.clamp(0.0, 1.0);
        let n = fbm2(
            &self.perlin,
            u * 5.0 + self.noise_offset.0,
            v * 5.0 + self.noise_offset.1,
            3,
        );

        // Rim shading: distance inside the silhouette, normalised by the
        // configured band width.
        let band = (c.edge_band.clamp(0.02, 0.5) * boundary).max(1e-6);
        let interior = ((-d) / band).clamp(0.0, 1.0).powf(0.7);

        let grain_shade = 1.0 + grain as f32 * (n as f32 - 0.5) * 0.8;
        let mut color = lerp_color(c.color_edge, c.color_base, interior as f32);
        color = [
            (color[0] * grain_shade).clamp(0.0, 1.0),
            (color[1] * grain_shade).clamp(0.0, 1.0),
            (color[2] * grain_shade).clamp(0.0, 1.0),
        ];

        // Facet look: a planar tilt plus grain relief, dropping to zero at
        // the rim so the silhouette bevels.
        let height = (0.55 + self.tilt.0 * dx + self.tilt.1 * dy + grain * (n - 0.5) * 0.3)
            .clamp(0.0, 1.0)
            * (0.4 + 0.6 * interior);

        SpriteSample {
            color,
            alpha,
            height,
            roughness: 0.95,
        }
    }
}

/// Procedural shard sprite generator.
///
/// See the [module documentation](self) for the visual model.
pub struct ShardGenerator {
    config: ShardConfig,
}

impl ShardGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: ShardConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for ShardGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| ShardCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_cell() -> ShardConfig {
        ShardConfig {
            variant_rows: 1,
            variant_cols: 1,
            ..ShardConfig::default()
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = ShardGenerator::new(ShardConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn centre_is_opaque_and_corners_transparent() {
        let map = ShardGenerator::new(single_cell())
            .generate(64, 64)
            .expect("generate failed");
        let centre = (32 * 64 + 32) * 4;
        assert_eq!(map.albedo[centre + 3], 255, "shard centre must be opaque");
        for (x, y) in [(0usize, 0usize), (63, 0), (0, 63), (63, 63)] {
            assert_eq!(
                map.albedo[(y * 64 + x) * 4 + 3],
                0,
                "corner ({x},{y}) must be transparent"
            );
        }
    }

    #[test]
    fn boundary_interpolation_covers_full_turn() {
        // Every angle must produce a positive, bounded radius — exercises
        // the wrap-around branch of the vertex search.
        let cell = ShardCell::new(&single_cell(), 0);
        for i in 0..360 {
            let b = cell.boundary(i as f64 / 360.0 * TAU);
            assert!(b > 0.0 && b <= 0.95, "boundary at {i}° out of range: {b}");
        }
    }

    #[test]
    fn regular_polygon_when_irregularity_zero() {
        let config = ShardConfig {
            irregularity: 0.0,
            ..single_cell()
        };
        let cell = ShardCell::new(&config, 0);
        // All radii equal for a regular polygon.
        for i in 1..cell.sides {
            assert!((cell.radii[i] - cell.radii[0]).abs() < 1e-12);
        }
    }

    #[test]
    fn variants_differ() {
        let map = ShardGenerator::new(ShardConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        let differs = (0..64usize).any(|i| {
            let a = map.albedo[((20 * 128) + i) * 4 + 3];
            let b = map.albedo[((20 * 128) + i + 64) * 4 + 3];
            a != b
        });
        assert!(differs, "shard atlas cells should not be identical");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = ShardGenerator::new(ShardConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = ShardGenerator::new(ShardConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
