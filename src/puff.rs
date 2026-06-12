//! Noise puff sprite generator.
//!
//! A billowy blob: domain-warped fractal noise masked by a soft radial
//! falloff.  Reads as dust motes, smoke, fog banks, or sea mist depending
//! on palette, density, and contrast.  Per-variant cells sample different
//! noise offsets so an atlas is a population of distinct puffs.
//!
//! Upload with [`map_to_images_card`](crate::generator::map_to_images_card);
//! see [`crate::sprite`] for the shared atlas conventions.

use noise::Perlin;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    sprite::{CellRng, SpriteCell, SpriteSample, fbm2, generate_atlas, lerp_color},
};

/// Configures the appearance of a [`PuffGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PuffConfig {
    /// PRNG seed for the noise field and per-cell offsets.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Lit colour of the puff in linear RGB \[0, 1\].
    pub color_base: [f32; 3],
    /// Shadowed colour in the noise troughs, linear RGB \[0, 1\].
    pub color_shadow: [f32; 3],
    /// Spatial frequency of the noise across a cell `[1, 8]`.
    pub noise_scale: f64,
    /// Fractal octave count `[1, 8]`.
    pub octaves: usize,
    /// Domain-warp strength `[0, 1.5]` — bends the noise so the silhouette
    /// billows instead of looking like raw Perlin.
    pub warp: f64,
    /// Overall alpha multiplier `[0, 1]`.
    pub density: f64,
    /// Radial mask exponent `[0.5, 6]`.  Higher → tighter, rounder puff;
    /// lower → ragged cloud that reaches the cell edge.
    pub edge_falloff: f64,
    /// Noise remap exponent `[0.5, 4]`.  Higher → wispier interior with
    /// more transparent gaps.
    pub contrast: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for PuffConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 2,
            variant_cols: 2,
            color_base: [0.86, 0.86, 0.9],
            color_shadow: [0.52, 0.52, 0.58],
            noise_scale: 3.0,
            octaves: 4,
            warp: 0.45,
            density: 0.9,
            edge_falloff: 2.0,
            contrast: 1.3,
            normal_strength: 1.0,
        }
    }
}

/// One baked puff variant: the shared noise field plus this cell's offset
/// into it.
struct PuffCell {
    config: PuffConfig,
    perlin: Perlin,
    /// Noise-space offset decorrelating this cell from its siblings.
    offset: (f64, f64),
    /// Per-cell radial scale jitter.
    radius: f64,
}

impl PuffCell {
    fn new(config: &PuffConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        Self {
            config: config.clone(),
            perlin: Perlin::new(config.seed),
            // Far-apart offsets keep cells statistically independent while
            // sampling one noise object.
            offset: (rng.range(0.0, 256.0), rng.range(0.0, 256.0)),
            radius: rng.range(0.85, 1.0),
        }
    }
}

impl SpriteCell for PuffCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;
        let dx = (u - 0.5) * 2.0;
        let dy = (v - 0.5) * 2.0;
        let r = (dx * dx + dy * dy).sqrt() / self.radius;

        let edge_falloff = c.edge_falloff.clamp(0.5, 6.0);
        let mask = (1.0 - r * r).max(0.0).powf(edge_falloff);
        if mask <= 0.0 {
            return SpriteSample {
                color: c.color_shadow,
                alpha: 0.0,
                height: 0.0,
                roughness: 1.0,
            };
        }

        let scale = c.noise_scale.clamp(1.0, 8.0);
        let warp = c.warp.clamp(0.0, 1.5);
        let octaves = c.octaves.clamp(1, 8);

        // Domain warp: displace the sample position by a low-frequency
        // noise read before taking the main fractal sample.
        let nx = u * scale + self.offset.0;
        let ny = v * scale + self.offset.1;
        let wx = fbm2(&self.perlin, nx + 13.7, ny, 2) - 0.5;
        let wy = fbm2(&self.perlin, nx, ny + 71.3, 2) - 0.5;
        let n = fbm2(
            &self.perlin,
            nx + wx * warp * scale,
            ny + wy * warp * scale,
            octaves,
        );

        let contrast = c.contrast.clamp(0.5, 4.0);
        let body = n.powf(contrast);
        let alpha = (mask * body * c.density.clamp(0.0, 1.0)).clamp(0.0, 1.0);

        // Self-shadowing: noise troughs darken, crests lighten.
        let brightness = (0.35 + 0.65 * n) as f32;
        SpriteSample {
            color: lerp_color(c.color_shadow, c.color_base, brightness),
            alpha,
            height: alpha,
            roughness: 1.0,
        }
    }
}

/// Procedural noise-puff sprite generator.
///
/// See the [module documentation](self) for the visual model.
pub struct PuffGenerator {
    config: PuffConfig,
}

impl PuffGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: PuffConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for PuffGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| PuffCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = PuffGenerator::new(PuffConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn puff_is_soft_everywhere() {
        // Puffs are the antithesis of a cutout: corners transparent, body
        // dominated by fractional alpha.
        let map = PuffGenerator::new(PuffConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo[3], 0, "corner must be fully transparent");
        let fractional = map
            .albedo
            .chunks(4)
            .filter(|px| px[3] > 0 && px[3] < 255)
            .count();
        let opaque = map.albedo.chunks(4).filter(|px| px[3] == 255).count();
        assert!(
            fractional > opaque,
            "puff body should be mostly soft alpha ({fractional} soft vs {opaque} solid)"
        );
    }

    #[test]
    fn variants_differ() {
        let map = PuffGenerator::new(PuffConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        // Same probe offset inside two different 64² cells.
        let differs = (0..64usize).any(|i| {
            let a = map.albedo[((32 * 128) + i + 8) * 4 + 3];
            let b = map.albedo[((32 * 128) + i + 8 + 64) * 4 + 3];
            a != b
        });
        assert!(differs, "puff atlas cells should not be identical");
    }

    #[test]
    fn zero_density_is_fully_transparent() {
        let config = PuffConfig {
            density: 0.0,
            ..PuffConfig::default()
        };
        let map = PuffGenerator::new(config)
            .generate(32, 32)
            .expect("generate failed");
        assert!(map.albedo.chunks(4).all(|px| px[3] == 0));
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = PuffGenerator::new(PuffConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = PuffGenerator::new(PuffConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
