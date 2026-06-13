//! Chain-link fence card generator.
//!
//! A woven diamond mesh: two wire families running at ±45°, rendered as
//! cylindrical profiles on a diagonal lattice.  Crossing parity lifts one
//! wire over the other (`weave_depth`), and rust accumulates at the
//! crossings where water sits.  The alpha channel is transparent everywhere
//! off the wires, so the card reads as a see-through fence.
//!
//! Upload with [`map_to_images_card`](crate::generator::map_to_images_card);
//! chain-link cards do not tile.

use noise::Perlin;
use rayon::prelude::*;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    normal::{BoundaryMode, dilate_heights, height_to_normal},
    sprite::{fbm2, lerp_color},
};

/// Configures the appearance of a [`ChainLinkGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChainLinkConfig {
    /// PRNG seed for the rust pattern.
    pub seed: u32,
    /// Diamond cells across the card `[4, 16]`.
    pub cell_count: f64,
    /// Wire radius in lattice units `[0.02, 0.2]` — fraction of a diamond
    /// half-diagonal.
    pub wire_radius: f64,
    /// Over/under weave relief at the crossings `[0, 1]`.
    pub weave_depth: f64,
    /// Rust accumulation at the crossings `[0, 1]`.
    pub rust_level: f64,
    /// Galvanised wire colour in linear RGB \[0, 1\].
    pub color_wire: [f32; 3],
    /// Rust colour in linear RGB \[0, 1\].
    pub color_rust: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for ChainLinkConfig {
    fn default() -> Self {
        Self {
            seed: 83,
            cell_count: 8.0,
            wire_radius: 0.07,
            weave_depth: 0.6,
            rust_level: 0.2,
            color_wire: [0.62, 0.64, 0.66],
            color_rust: [0.45, 0.24, 0.10],
            normal_strength: 3.0,
        }
    }
}

/// Procedural chain-link fence card generator.
///
/// See the [module documentation](self) for the visual model.
pub struct ChainLinkGenerator {
    config: ChainLinkConfig,
}

impl ChainLinkGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: ChainLinkConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for ChainLinkGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        let k = c.cell_count.round().clamp(2.0, 32.0);
        let wr = c.wire_radius.clamp(0.02, 0.2);
        let lift = c.weave_depth.clamp(0.0, 1.0) * 0.18;
        let perlin = Perlin::new(c.seed);

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness = vec![0u8; n * 4];

        heights
            .par_chunks_mut(w)
            .zip(albedo.par_chunks_mut(w * 4))
            .zip(roughness.par_chunks_mut(w * 4))
            .enumerate()
            .for_each(|(y, ((height_row, albedo_row), orm_row))| {
                let v = (y as f64 + 0.5) / h as f64;
                for (x, height_slot) in height_row.iter_mut().enumerate() {
                    let u = (x as f64 + 0.5) / w as f64;

                    // Diagonal lattice coordinates: wires run along the
                    // integer lines of each family.
                    let a = (u + v) * k;
                    let b = (u - v) * k;
                    let da = (a - a.round()).abs();
                    let db = (b - b.round()).abs();

                    // Cylindrical wire profiles.
                    let profile = |d: f64| {
                        if d < wr {
                            (1.0 - (d / wr) * (d / wr)).sqrt()
                        } else {
                            0.0
                        }
                    };
                    let pa = profile(da);
                    let pb = profile(db);

                    let ai = x * 4;
                    if pa <= 0.0 && pb <= 0.0 {
                        // Open mesh: transparent, wire colour in RGB to
                        // avoid halos under bilinear filtering.
                        albedo_row[ai] = linear_to_srgb(c.color_wire[0]);
                        albedo_row[ai + 1] = linear_to_srgb(c.color_wire[1]);
                        albedo_row[ai + 2] = linear_to_srgb(c.color_wire[2]);
                        albedo_row[ai + 3] = 0;
                        orm_row[ai] = 255;
                        orm_row[ai + 1] = 150;
                        orm_row[ai + 2] = 0;
                        orm_row[ai + 3] = 255;
                        continue;
                    }

                    // Over/under weave: crossing parity lifts one family.
                    let over_a = ((a.round() + b.round()) as i64).rem_euclid(2) == 0;
                    let ha = pa * 0.45 + 0.35 + if over_a { lift } else { -lift };
                    let hb = pb * 0.45 + 0.35 + if over_a { -lift } else { lift };
                    let (height, prof) = match (pa > 0.0, pb > 0.0) {
                        (true, true) => {
                            if ha >= hb {
                                (ha, pa)
                            } else {
                                (hb, pb)
                            }
                        }
                        (true, false) => (ha, pa),
                        _ => (hb, pb),
                    };
                    *height_slot = height.clamp(0.0, 1.0);

                    // Rust pools where the wires cross.
                    let cross = (pa * pb).clamp(0.0, 1.0);
                    let rust_n = fbm2(&perlin, u * 9.0, v * 9.0, 3);
                    let rust = (cross * rust_n * 1.8 * c.rust_level.clamp(0.0, 1.0)).clamp(0.0, 1.0)
                        as f32;

                    let shade = (0.55 + 0.45 * prof) as f32;
                    let base = lerp_color(c.color_wire, c.color_rust, rust);
                    albedo_row[ai] = linear_to_srgb(base[0] * shade);
                    albedo_row[ai + 1] = linear_to_srgb(base[1] * shade);
                    albedo_row[ai + 2] = linear_to_srgb(base[2] * shade);
                    albedo_row[ai + 3] = 255;

                    let rough = (0.35 + rust * 0.5).clamp(0.0, 1.0);
                    let metallic = (0.85 * (1.0 - rust)).clamp(0.0, 1.0);
                    orm_row[ai] = 255;
                    orm_row[ai + 1] = (rough * 255.0).round() as u8;
                    orm_row[ai + 2] = (metallic * 255.0).round() as u8;
                    orm_row[ai + 3] = 255;
                }
            });

        dilate_heights(&mut heights, &albedo, w, h);

        let normal = height_to_normal(
            &heights,
            width,
            height,
            c.normal_strength,
            BoundaryMode::Clamp,
        );

        Ok(TextureMap {
            albedo,
            normal,
            roughness,
            emissive: None,
            width,
            height,
            mip_level_count: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = ChainLinkGenerator::new(ChainLinkConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
    }

    #[test]
    fn mesh_is_mostly_open_with_opaque_wires() {
        let map = ChainLinkGenerator::new(ChainLinkConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        let open = map.albedo.chunks(4).filter(|px| px[3] == 0).count();
        let wire = map.albedo.chunks(4).filter(|px| px[3] == 255).count();
        assert!(wire > 0, "wires must be opaque");
        assert!(open > wire, "the mesh should be mostly see-through");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = ChainLinkGenerator::new(ChainLinkConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        let b = ChainLinkGenerator::new(ChainLinkConfig::default())
            .generate(32, 32)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
