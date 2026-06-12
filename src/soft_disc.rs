//! Soft disc / glow sprite generator.
//!
//! A radial-falloff disc with a solid core and a tunable halo — the
//! workhorse particle sprite: fireflies, embers, mist motes, bokeh glints,
//! and generic additive glows.  Per-variant cells jitter scale, elongation,
//! and orientation so an atlas reads as a population of distinct glows
//! rather than copies.
//!
//! Upload with [`map_to_images_card`](crate::generator::map_to_images_card);
//! see [`crate::sprite`] for the shared atlas conventions.

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    sprite::{CellRng, SpriteCell, SpriteSample, generate_atlas, lerp_color},
};

/// Configures the appearance of a [`SoftDiscGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SoftDiscConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Colour at the disc centre in linear RGB \[0, 1\].
    pub color_core: [f32; 3],
    /// Colour of the outer halo in linear RGB \[0, 1\].
    pub color_halo: [f32; 3],
    /// Radius of the fully-opaque core as a fraction of the cell
    /// half-extent `[0, 0.9]`.  `0` makes the falloff start at the centre.
    pub core_radius: f64,
    /// Halo falloff exponent `[0.3, 8]`.  Higher → tighter glow with a
    /// faster fade; lower → wide soft haze.
    pub falloff: f64,
    /// Maximum per-variant elongation `[0, 0.6]`.  Each cell squashes the
    /// disc by a random amount up to this value at a random orientation —
    /// `0` keeps every variant perfectly round.
    pub ellipticity: f64,
    /// Per-variant scale jitter `[0, 0.5]`.  Each cell shrinks the disc by
    /// a random fraction up to this value.
    pub scale_jitter: f64,
    /// Normal map strength.  Soft discs are usually rendered unlit or
    /// additive, so the default is gentle.
    pub normal_strength: f32,
}

impl Default for SoftDiscConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 1,
            variant_cols: 1,
            color_core: [1.0, 0.98, 0.9],
            color_halo: [1.0, 0.72, 0.25],
            core_radius: 0.15,
            falloff: 2.5,
            ellipticity: 0.0,
            scale_jitter: 0.15,
            normal_strength: 1.0,
        }
    }
}

/// One baked soft-disc variant: the config plus this cell's jittered
/// scale, elongation, and orientation.
struct SoftDiscCell {
    config: SoftDiscConfig,
    /// Disc radius after scale jitter, in cell half-extents.
    radius: f64,
    /// Elongation factor applied along the rotated minor axis (`1` = round).
    squash: f64,
    /// Orientation of the elongation axis (radians).
    angle: f64,
}

impl SoftDiscCell {
    fn new(config: &SoftDiscConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        let scale_jitter = config.scale_jitter.clamp(0.0, 0.5);
        let ellipticity = config.ellipticity.clamp(0.0, 0.6);
        Self {
            config: config.clone(),
            // Keep a small safety margin so the halo never clips the cell.
            radius: 0.96 * (1.0 - rng.next_f64() * scale_jitter),
            squash: 1.0 - rng.next_f64() * ellipticity,
            angle: rng.range(0.0, std::f64::consts::PI),
            // rng dropped here; all variant state is baked into the fields.
        }
    }
}

impl SpriteCell for SoftDiscCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;

        // Centre the cell and rotate into the elongation frame.
        let dx = (u - 0.5) * 2.0;
        let dy = (v - 0.5) * 2.0;
        let (sin_a, cos_a) = self.angle.sin_cos();
        let ex = dx * cos_a + dy * sin_a;
        let ey = (-dx * sin_a + dy * cos_a) / self.squash;
        let r = (ex * ex + ey * ey).sqrt() / self.radius;

        let core = c.core_radius.clamp(0.0, 0.9);
        let falloff = c.falloff.clamp(0.3, 8.0);
        let alpha = if r <= core {
            1.0
        } else if r < 1.0 {
            let t = (r - core) / (1.0 - core);
            (1.0 - t).powf(falloff)
        } else {
            0.0
        };

        // Blend toward the core colour faster than the alpha fade so the
        // halo tint stays visible through most of the falloff.
        let core_blend = (alpha * alpha) as f32;
        SpriteSample {
            color: lerp_color(c.color_halo, c.color_core, core_blend),
            alpha,
            height: alpha,
            roughness: 0.9,
        }
    }
}

/// Procedural soft-disc sprite generator.
///
/// See the [module documentation](self) for the visual model.
pub struct SoftDiscGenerator {
    config: SoftDiscConfig,
}

impl SoftDiscGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: SoftDiscConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for SoftDiscGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| SoftDiscCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = SoftDiscGenerator::new(SoftDiscConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn centre_is_opaque_and_corner_transparent() {
        let map = SoftDiscGenerator::new(SoftDiscConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        let centre = (32 * 64 + 32) * 4;
        assert_eq!(map.albedo[centre + 3], 255);
        assert_eq!(map.albedo[3], 0, "corner texel must be transparent");
    }

    #[test]
    fn alpha_fades_gradually() {
        // A soft disc must have fractional alpha between core and rim —
        // that is the difference from a binary foliage-card silhouette.
        let map = SoftDiscGenerator::new(SoftDiscConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert!(
            map.albedo.chunks(4).any(|px| px[3] > 16 && px[3] < 240),
            "soft disc should contain fractional-alpha texels"
        );
    }

    #[test]
    fn atlas_variants_differ() {
        let config = SoftDiscConfig {
            variant_rows: 2,
            variant_cols: 2,
            ellipticity: 0.5,
            scale_jitter: 0.4,
            ..SoftDiscConfig::default()
        };
        let map = SoftDiscGenerator::new(config)
            .generate(128, 128)
            .expect("generate failed");
        // Compare the same texel offset inside two different cells; with
        // strong jitter the alpha must differ for at least one probe.
        let probes = [(20usize, 20usize), (40, 40), (12, 30)];
        let differs = probes.iter().any(|&(px, py)| {
            let a = map.albedo[(py * 128 + px) * 4 + 3];
            let b = map.albedo[(py * 128 + px + 64) * 4 + 3];
            a != b
        });
        assert!(differs, "jittered atlas cells should not be identical");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let config = SoftDiscConfig {
            variant_rows: 2,
            variant_cols: 2,
            ..SoftDiscConfig::default()
        };
        let a = SoftDiscGenerator::new(config.clone())
            .generate(32, 32)
            .expect("generate failed");
        let b = SoftDiscGenerator::new(config)
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
