//! Flower sprite-atlas generator.
//!
//! A radially composed blossom: `petal_count` petal blades (the
//! [`petal`](crate::petal) cell sampler, re-aimed outward from the flower
//! centre) under a domed centre disc with hash-dotted stamens.  This is the
//! sprite-family counterpart of how [`twig`](crate::twig) composites the
//! leaf sampler — and the "building block for procedural flowers" promised
//! in the petal docs.
//!
//! Per-variant cells jitter the rotation phase and every petal's own shape
//! (each petal draws an independent [`CellRng`] stream), so an atlas gives
//! per-particle blossom variety: petal-fall effects, flower decals, ground
//! scatter.
//!
//! Upload with [`map_to_images_card`](crate::generator::map_to_images_card);
//! see [`crate::sprite`] for the shared atlas conventions.

use std::f64::consts::TAU;

use noise::Perlin;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    petal::{PetalCell, PetalConfig},
    sprite::{CellRng, SpriteCell, SpriteSample, fbm2},
};

/// Configures the appearance of a [`FlowerGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlowerConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Petal appearance shared by every blade (each blade additionally
    /// draws its own per-petal jitter from the petal config's knobs).
    pub petal: PetalConfig,
    /// Number of petal blades `[4, 12]`.
    pub petal_count: usize,
    /// Centre-disc radius as a fraction of the cell half-extent
    /// `[0.05, 0.3]`.
    pub center_radius: f64,
    /// Centre-disc colour in linear RGB \[0, 1\].
    pub center_color: [f32; 3],
    /// Stamen-dot density on the centre disc `[0, 1]`.
    pub dot_density: f64,
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for FlowerConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 2,
            variant_cols: 2,
            petal: PetalConfig::default(),
            petal_count: 6,
            center_radius: 0.14,
            center_color: [0.96, 0.78, 0.25],
            dot_density: 0.5,
            normal_strength: 1.5,
        }
    }
}

/// One baked flower variant: independently-jittered petal blades plus the
/// rotation phase and stamen noise.
struct FlowerCell {
    config: FlowerConfig,
    petals: Vec<PetalCell>,
    perlin: Perlin,
    phase: f64,
}

impl FlowerCell {
    fn new(config: &FlowerConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        let count = config.petal_count.clamp(4, 12);
        // Each blade gets its own decorrelated stream — petals within one
        // blossom differ exactly like petal-atlas cells do.
        let petals = (0..count)
            .map(|i| PetalCell::new(&config.petal, cell * 64 + i))
            .collect();
        Self {
            config: config.clone(),
            petals,
            perlin: Perlin::new(rng.next_u32()),
            phase: rng.range(0.0, TAU),
        }
    }
}

impl SpriteCell for FlowerCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;
        let px = u - 0.5;
        let py = v - 0.5;
        let r = (px * px + py * py).sqrt();

        // Centre disc: a dome with hash-dotted stamens, drawn over the
        // petal throats.
        let cr = c.center_radius.clamp(0.05, 0.3);
        if r < cr {
            let dot_n = fbm2(&self.perlin, u * 26.0, v * 26.0, 2);
            let dotted = (dot_n - 0.5) * 2.2 + 0.5 > 1.0 - c.dot_density.clamp(0.0, 1.0) * 0.5;
            let shade = if dotted { 0.55 } else { 1.0 };
            let dome = 1.0 - (r / cr) * (r / cr);
            return SpriteSample {
                color: [
                    c.center_color[0] * shade,
                    c.center_color[1] * shade,
                    c.center_color[2] * shade,
                ],
                alpha: ((cr - r) / (cr * 0.12)).clamp(0.0, 1.0),
                height: 0.6 + dome * 0.3,
                roughness: 0.75,
            };
        }

        // Petals: rotate the query into each blade's frame (throat at the
        // flower centre, tip pointing outward) and keep the topmost hit.
        let n = self.petals.len();
        let mut best: Option<SpriteSample> = None;
        for (i, petal) in self.petals.iter().enumerate() {
            let angle = self.phase + i as f64 * TAU / n as f64;
            let (sin, cos) = angle.sin_cos();
            // Outward axis of this blade.
            let along = px * cos + py * sin;
            let across = -px * sin + py * cos;
            if along < 0.0 {
                continue; // behind the centre — the opposite blade's side
            }
            // Map into petal cell UV: throat (v = 1) at the centre, tip
            // (v = 0) at the cell edge.
            let pu = 0.5 + across / 0.5;
            let pv = 1.0 - along / 0.5;
            if !(0.0..=1.0).contains(&pu) || !(0.0..=1.0).contains(&pv) {
                continue;
            }
            let s = petal.sample(pu, pv);
            if s.alpha > 0.0 && best.as_ref().is_none_or(|b| s.height > b.height) {
                best = Some(s);
            }
        }

        best.unwrap_or(SpriteSample {
            color: c.petal.color_edge,
            alpha: 0.0,
            height: 0.0,
            roughness: 0.55,
        })
    }
}

/// Procedural flower sprite generator.
///
/// See the [module documentation](self) for the visual model.
pub struct FlowerGenerator {
    config: FlowerConfig,
}

impl FlowerGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: FlowerConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for FlowerGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        crate::sprite::generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| FlowerCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_cell() -> FlowerConfig {
        FlowerConfig {
            variant_rows: 1,
            variant_cols: 1,
            ..FlowerConfig::default()
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = FlowerGenerator::new(FlowerConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
    }

    #[test]
    fn centre_is_opaque_and_corners_transparent() {
        let map = FlowerGenerator::new(single_cell())
            .generate(128, 128)
            .expect("generate failed");
        let centre = (64 * 128 + 64) * 4;
        assert_eq!(map.albedo[centre + 3], 255, "centre disc must be opaque");
        for (cx, cy) in [(2usize, 2usize), (125, 125)] {
            let idx = (cy * 128 + cx) * 4;
            assert_eq!(map.albedo[idx + 3], 0, "corner ({cx},{cy})");
        }
    }

    #[test]
    fn petals_reach_past_the_centre_disc() {
        let map = FlowerGenerator::new(single_cell())
            .generate(128, 128)
            .expect("generate failed");
        let cr_px = (0.14 * 64.0) as i32;
        let petal_px = map.albedo.chunks(4).enumerate().any(|(i, px)| {
            let x = (i % 128) as i32 - 64;
            let y = (i / 128) as i32 - 64;
            px[3] == 255 && (x * x + y * y) > (cr_px * 3) * (cr_px * 3)
        });
        assert!(petal_px, "petal blades should extend beyond the centre");
    }

    #[test]
    fn variants_differ() {
        let map = FlowerGenerator::new(FlowerConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        let differs = (0..64usize).any(|y| {
            (0..64usize).any(|x| {
                let a = ((y * 128) + x) * 4;
                let b = ((y * 128) + x + 64) * 4;
                map.albedo[a..a + 4] != map.albedo[b..b + 4]
            })
        });
        assert!(differs, "atlas cells should bake distinct variants");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = FlowerGenerator::new(FlowerConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        let b = FlowerGenerator::new(FlowerConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
