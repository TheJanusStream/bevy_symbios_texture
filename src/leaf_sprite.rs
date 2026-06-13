//! Leaf sprite-atlas generator.
//!
//! Wraps the [`LeafSampler`] point sampler in the sprite-atlas conventions:
//! every cell bakes a per-cell-seeded leaf variant with bounded jitter on
//! the silhouette shape and colour, so a falling-foliage particle system
//! gets per-particle leaf variety from a single texture bake.  This is the
//! atlas counterpart of the single-leaf [`LeafGenerator`](crate::leaf::LeafGenerator)
//! foliage card.
//!
//! Upload with [`map_to_images_card`](crate::generator::map_to_images_card);
//! see [`crate::sprite`] for the shared atlas conventions.
//!
//! # Implementation note
//!
//! [`LeafSampler`] holds `noise::Worley` (internally reference-counted, so
//! `!Sync`) and cannot be shared across the row-parallel driver in
//! [`generate_atlas`](crate::sprite::generate_atlas).  This generator
//! therefore runs its own atlas loop parallelised per *atlas-row band*:
//! each parallel task constructs the samplers for its band locally and
//! renders the band's pixel rows serially.

use rayon::prelude::*;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    leaf::{LeafConfig, LeafSampler},
    normal::{BoundaryMode, dilate_heights, height_to_normal},
    sprite::{CellRng, clamp_variant_dim},
};

/// Configures the appearance of a [`LeafSpriteGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LeafSpriteConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Base leaf appearance shared by every cell.  Each cell re-seeds the
    /// venation/serration noise and applies the jitter knobs below.
    pub leaf: LeafConfig,
    /// Per-cell silhouette jitter `[0, 1]` — scales bounded perturbations
    /// of serration strength, lobe depth/sharpness, and vein count.
    pub shape_jitter: f64,
    /// Per-cell colour tint jitter `[0, 1]` — shifts the interior colour
    /// with a green-preserving bias so variants read as natural hue drift.
    pub tint_jitter: f32,
}

impl Default for LeafSpriteConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 2,
            variant_cols: 2,
            leaf: LeafConfig::default(),
            shape_jitter: 0.5,
            tint_jitter: 0.25,
        }
    }
}

/// Derive the cell's jittered leaf config — bounded perturbations around the
/// base config, deterministic from `(seed, cell)`.
fn jittered_config(c: &LeafSpriteConfig, cell: usize) -> LeafConfig {
    let mut rng = CellRng::new(c.seed, cell);
    let mut leaf = c.leaf.clone();

    // Fresh noise seed per cell: venation and serration decorrelate even at
    // zero shape jitter.
    leaf.seed = rng.next_u32();

    let shape = c.shape_jitter.clamp(0.0, 1.0);
    leaf.serration_strength =
        (leaf.serration_strength + (rng.next_f64() * 2.0 - 1.0) * 0.05 * shape).clamp(0.0, 0.35);
    leaf.lobe_depth =
        (leaf.lobe_depth + (rng.next_f64() * 2.0 - 1.0) * 0.08 * shape).clamp(0.0, 0.45);
    leaf.lobe_sharpness =
        (leaf.lobe_sharpness + (rng.next_f64() * 2.0 - 1.0) * 0.3 * shape).clamp(0.1, 5.0);
    leaf.vein_count = (leaf.vein_count + (rng.next_f64() * 2.0 - 1.0) * 1.5 * shape)
        .round()
        .clamp(2.0, 14.0);

    // Green-preserving tint: shift all channels but move green least, so
    // variants drift toward autumn/blue-green hues instead of banding in
    // brightness.
    let tint = c.tint_jitter.clamp(0.0, 1.0);
    let t = ((rng.next_f64() * 2.0 - 1.0) as f32) * 0.15 * tint;
    leaf.color_base = [
        (leaf.color_base[0] + t).clamp(0.0, 1.0),
        (leaf.color_base[1] + t * 0.4).clamp(0.0, 1.0),
        (leaf.color_base[2] + t * 0.6).clamp(0.0, 1.0),
    ];

    leaf
}

/// Procedural leaf sprite-atlas generator.
///
/// See the [module documentation](self) for the visual model and the
/// band-parallel implementation note.
pub struct LeafSpriteGenerator {
    config: LeafSpriteConfig,
}

impl LeafSpriteGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: LeafSpriteConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for LeafSpriteGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        let rows = clamp_variant_dim(c.variant_rows);
        let cols = clamp_variant_dim(c.variant_cols);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness = vec![0u8; n * 4];

        // Partition the pixel rows into atlas-row bands (the same
        // `y * rows / h` mapping generate_atlas uses) and split the buffers
        // accordingly so each band renders in its own parallel task.
        let mut bands = Vec::with_capacity(rows);
        {
            let mut h_rest = heights.as_mut_slice();
            let mut a_rest = albedo.as_mut_slice();
            let mut o_rest = roughness.as_mut_slice();
            let mut y0 = 0usize;
            for band in 0..rows {
                let y1 = ((band + 1) * h).div_ceil(rows);
                let len = y1 - y0;
                let (h_band, h_tail) = h_rest.split_at_mut(len * w);
                let (a_band, a_tail) = a_rest.split_at_mut(len * w * 4);
                let (o_band, o_tail) = o_rest.split_at_mut(len * w * 4);
                h_rest = h_tail;
                a_rest = a_tail;
                o_rest = o_tail;
                bands.push((band, y0, h_band, a_band, o_band));
                y0 = y1;
            }
        }

        bands
            .into_par_iter()
            .for_each(|(band, y_start, h_band, a_band, o_band)| {
                // Samplers are built inside the band task: LeafSampler is
                // !Sync (Worley), so it must not cross task boundaries.
                let samplers: Vec<LeafSampler> = (0..cols)
                    .map(|col| LeafSampler::new(jittered_config(c, band * cols + col)))
                    .collect();
                let cell_v0 = band as f64 / rows as f64;
                let edge = c.leaf.color_edge;

                for (local_y, ((h_row, a_row), o_row)) in h_band
                    .chunks_mut(w)
                    .zip(a_band.chunks_mut(w * 4))
                    .zip(o_band.chunks_mut(w * 4))
                    .enumerate()
                {
                    let y = y_start + local_y;
                    for (x, height_slot) in h_row.iter_mut().enumerate() {
                        let col = (x * cols / w).min(cols - 1);
                        let cell_u0 = col as f64 / cols as f64;
                        let u = ((x as f64 + 0.5) / w as f64 - cell_u0) * cols as f64;
                        let v = ((y as f64 + 0.5) / h as f64 - cell_v0) * rows as f64;

                        let ai = x * 4;
                        match samplers[col].sample(u, v) {
                            Some(s) => {
                                *height_slot = s.height;
                                a_row[ai] = linear_to_srgb(s.color[0]);
                                a_row[ai + 1] = linear_to_srgb(s.color[1]);
                                a_row[ai + 2] = linear_to_srgb(s.color[2]);
                                a_row[ai + 3] = 255;
                                o_row[ai] = 255;
                                o_row[ai + 1] = (s.roughness * 255.0).round() as u8;
                                o_row[ai + 2] = 0;
                                o_row[ai + 3] = 255;
                            }
                            None => {
                                // Transparent texel: keep the edge colour in
                                // RGB so bilinear filtering does not pull a
                                // dark halo across the silhouette.
                                a_row[ai] = linear_to_srgb(edge[0]);
                                a_row[ai + 1] = linear_to_srgb(edge[1]);
                                a_row[ai + 2] = linear_to_srgb(edge[2]);
                                a_row[ai + 3] = 0;
                                o_row[ai] = 255;
                                o_row[ai + 1] = 200;
                                o_row[ai + 2] = 0;
                                o_row[ai + 3] = 255;
                            }
                        }
                    }
                }
            });

        // Expand opaque heights into the transparent border so the normal
        // kernel sees no cliff at the silhouette (same as the foliage cards).
        dilate_heights(&mut heights, &albedo, w, h);

        let normal = height_to_normal(
            &heights,
            width,
            height,
            c.leaf.normal_strength,
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
        let map = LeafSpriteGenerator::new(LeafSpriteConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn cells_contain_opaque_leaves_and_transparent_borders() {
        // 2×2 atlas at 128² → each 64² cell has a leaf around its centre and
        // transparent corners.
        let map = LeafSpriteGenerator::new(LeafSpriteConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        for (cx, cy) in [(32usize, 32usize), (96, 32), (32, 96), (96, 96)] {
            let idx = (cy * 128 + cx) * 4;
            assert_eq!(map.albedo[idx + 3], 255, "cell centre ({cx},{cy})");
        }
        assert!(
            map.albedo.chunks(4).any(|px| px[3] == 0),
            "atlas must have fully transparent texels"
        );
    }

    #[test]
    fn variants_differ() {
        let map = LeafSpriteGenerator::new(LeafSpriteConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        // Compare the two top cells texel-by-texel; per-cell seeding must
        // produce different silhouettes/venation.
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
        let a = LeafSpriteGenerator::new(LeafSpriteConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        let b = LeafSpriteGenerator::new(LeafSpriteConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.normal, b.normal);
    }

    #[test]
    fn rejects_invalid_dimensions() {
        assert!(
            LeafSpriteGenerator::new(LeafSpriteConfig::default())
                .generate(0, 64)
                .is_err()
        );
    }
}
