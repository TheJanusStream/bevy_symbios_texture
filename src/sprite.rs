//! Shared scaffolding for sprite-atlas card generators.
//!
//! The sprite family (soft disc, spark, snowflake, puff, ring, petal,
//! shard) produces small alpha-silhouette cards intended primarily for
//! particle billboards.  Unlike the foliage cards (leaf, twig) each sprite
//! generator can bake a `variant_rows × variant_cols` **atlas** in a single
//! image: every cell renders the same configuration with a per-cell derived
//! seed, so a particle system using random atlas frames gets per-particle
//! shape variety from one texture bake.
//!
//! # Architecture
//!
//! Each sprite module defines a config struct plus a *cell sampler* — a
//! struct implementing [`SpriteCell`] that captures the per-cell randomised
//! parameters and answers point queries in local cell UV space.  The shared
//! [`generate_atlas`] driver handles cell layout, buffer assembly, ORM
//! packing, height dilation, and normal-map derivation, so the modules only
//! contain sprite-specific math.
//!
//! Upload results with [`map_to_images_card`](crate::generator::map_to_images_card)
//! (clamp-to-edge samplers); sprites never tile.

use rayon::prelude::*;

use crate::{
    generator::{TextureError, TextureMap, linear_to_srgb, validate_dimensions},
    normal::{BoundaryMode, height_to_normal},
};

/// Maximum atlas dimension (rows or columns) accepted by
/// [`generate_atlas`]; values outside `1..=MAX_VARIANT_DIM` are clamped.
/// Matches the sprite-sheet cap that particle consumers enforce, and keeps
/// the per-cell resolution meaningful (a 16×16 atlas on a 512² bake still
/// leaves 32² texels per cell).
pub const MAX_VARIANT_DIM: usize = 16;

/// One point sample of a sprite cell, in physically-meaningless-but-
/// consistent units: `color` is linear RGB `[0, 1]`, `alpha` is soft
/// coverage `[0, 1]`, `height` feeds the normal-map derivation `[0, 1]`,
/// and `roughness` lands in the ORM green channel `[0, 1]`.
pub struct SpriteSample {
    /// Linear RGB colour of the sprite at this point.  Must be meaningful
    /// even where `alpha == 0` — the driver writes it into transparent
    /// texels to prevent dark halos from bilinear filtering at silhouette
    /// boundaries.
    pub color: [f32; 3],
    /// Soft coverage in `[0, 1]`.  Sprites are allowed (encouraged!) to
    /// return fractional alpha — glows and mist fade out smoothly rather
    /// than cutting like foliage cards.
    pub alpha: f64,
    /// Pseudo-height in `[0, 1]` used to derive the tangent-space normal.
    pub height: f64,
    /// PBR roughness in `[0, 1]`.
    pub roughness: f32,
}

/// A fully-instantiated sprite cell: configuration plus the per-cell
/// randomised parameters, ready to answer point queries.
///
/// Implementations are constructed once per atlas cell by the closure given
/// to [`generate_atlas`] and then sampled for every texel inside that cell.
pub trait SpriteCell {
    /// Sample the sprite at local cell UV (`u`, `v` ∈ `[0, 1]`, `v = 0` at
    /// the top of the cell).
    fn sample(&self, u: f64, v: f64) -> SpriteSample;
}

/// Deterministic per-cell parameter RNG.
///
/// A small [SplitMix64](https://prng.di.unimi.it/splitmix64.c) stream seeded
/// from the config seed and the cell index.  Kept dependency-free and
/// bit-stable so the same config always bakes the same atlas — the contract
/// that [`TextureCache`](crate::cache::TextureCache) fingerprinting and
/// deterministic world records rely on.
pub struct CellRng(u64);

impl CellRng {
    /// Create the parameter stream for `cell` of a sprite seeded `seed`.
    pub fn new(seed: u32, cell: usize) -> Self {
        // Decorrelate cells: distinct odd multiplier per dimension.
        let s = (seed as u64) ^ ((cell as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        Self(s)
    }

    /// Next raw 64-bit value of the stream.
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Next value as `u32` (high bits, better mixed than low).
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // 53 mantissa bits → uniform double in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform `f64` in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

/// Clamp a requested atlas dimension into `1..=MAX_VARIANT_DIM`.
#[inline]
pub fn clamp_variant_dim(n: usize) -> usize {
    n.clamp(1, MAX_VARIANT_DIM)
}

/// Render a `variant_rows × variant_cols` sprite atlas at
/// `width × height` texels.
///
/// `make_cell` is called once per cell (row-major index) to build the cell
/// sampler — typically deriving its randomised parameters from a
/// [`CellRng`].  The driver then:
///
/// 1. samples every texel through [`SpriteCell::sample`] in local cell UV,
/// 2. packs albedo (sRGB-encoded, soft alpha) and ORM (occlusion `255`,
///    roughness from the sample, metallic `0`),
/// 3. dilates heights into fully-transparent texels so the derived normals
///    do not crease at the silhouette, and
/// 4. derives the tangent-space normal map with clamped boundaries.
///
/// Out-of-range atlas dimensions are clamped via [`clamp_variant_dim`]
/// rather than rejected, mirroring how the generators treat other
/// config-value excursions.
///
/// Rows are sampled in parallel — `S` must be `Sync`.  Work runs on the
/// ambient rayon pool (the crate's private pool inside async generation
/// tasks); output is byte-identical to serial evaluation.
pub fn generate_atlas<S, F>(
    width: u32,
    height: u32,
    variant_rows: usize,
    variant_cols: usize,
    normal_strength: f32,
    mut make_cell: F,
) -> Result<TextureMap, TextureError>
where
    S: SpriteCell + Sync,
    F: FnMut(usize) -> S,
{
    validate_dimensions(width, height)?;

    let rows = clamp_variant_dim(variant_rows);
    let cols = clamp_variant_dim(variant_cols);
    let cells: Vec<S> = (0..rows * cols).map(&mut make_cell).collect();

    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    let mut heights = vec![0.0f64; n];
    let mut albedo = vec![0u8; n * 4];
    let mut roughness = vec![0u8; n * 4];

    heights
        .par_chunks_mut(w)
        .zip(albedo.par_chunks_mut(w * 4))
        .zip(roughness.par_chunks_mut(w * 4))
        .enumerate()
        .for_each(|(y, ((height_row, albedo_row), orm_row))| {
            // Row-major cell lookup; the `min` guards the y == h-1 / x == w-1
            // edge where the floating-point cell coordinate can land exactly on
            // the next cell boundary.
            let row = (y * rows / h).min(rows - 1);
            let cell_v0 = row as f64 / rows as f64;
            for (x, height_slot) in height_row.iter_mut().enumerate() {
                let col = (x * cols / w).min(cols - 1);
                let cell = &cells[row * cols + col];

                // Local cell UV of the texel centre.
                let cell_u0 = col as f64 / cols as f64;
                let u = ((x as f64 + 0.5) / w as f64 - cell_u0) * cols as f64;
                let v = ((y as f64 + 0.5) / h as f64 - cell_v0) * rows as f64;

                let s = cell.sample(u, v);
                let ai = x * 4;

                *height_slot = s.height.clamp(0.0, 1.0);
                albedo_row[ai] = linear_to_srgb(s.color[0]);
                albedo_row[ai + 1] = linear_to_srgb(s.color[1]);
                albedo_row[ai + 2] = linear_to_srgb(s.color[2]);
                albedo_row[ai + 3] = (s.alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
                orm_row[ai] = 255;
                orm_row[ai + 1] = (s.roughness.clamp(0.0, 1.0) * 255.0).round() as u8;
                orm_row[ai + 2] = 0;
                orm_row[ai + 3] = 255;
            }
        });

    crate::normal::dilate_heights(&mut heights, &albedo, w, h);

    let normal = height_to_normal(
        &heights,
        width,
        height,
        normal_strength,
        BoundaryMode::Clamp,
    );

    Ok(TextureMap {
        albedo,
        normal,
        roughness,
        width,
        height,
    })
}

/// Normalised fractal Brownian motion over 2-D Perlin noise.
///
/// Sums `octaves` octaves with persistence 0.5 / lacunarity 2.0 and rescales
/// the result into `[0, 1]`.  Shared by the noise-driven sprites (puff,
/// shard grain).
pub fn fbm2(perlin: &noise::Perlin, x: f64, y: f64, octaves: usize) -> f64 {
    use noise::NoiseFn;
    let octaves = octaves.clamp(1, 10);
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        sum += amplitude * perlin.get([x * frequency, y * frequency]);
        norm += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    // Perlin output is roughly [-1, 1]; rescale into [0, 1].
    (sum / norm * 0.5 + 0.5).clamp(0.0, 1.0)
}

/// Linear interpolation between two linear-RGB colours.
#[inline]
pub fn lerp_color(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dot;

    impl SpriteCell for Dot {
        fn sample(&self, u: f64, v: f64) -> SpriteSample {
            let r = ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt() * 2.0;
            SpriteSample {
                color: [1.0, 0.0, 0.0],
                alpha: if r < 0.5 { 1.0 } else { 0.0 },
                height: 1.0 - r,
                roughness: 0.5,
            }
        }
    }

    #[test]
    fn cell_rng_is_deterministic_and_decorrelated() {
        let a1 = CellRng::new(7, 0).next_u64();
        let a2 = CellRng::new(7, 0).next_u64();
        let b = CellRng::new(7, 1).next_u64();
        let c = CellRng::new(8, 0).next_u64();
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
        assert_ne!(a1, c);
    }

    #[test]
    fn cell_rng_range_stays_in_bounds() {
        let mut rng = CellRng::new(42, 3);
        for _ in 0..1000 {
            let v = rng.range(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&v));
        }
    }

    #[test]
    fn atlas_dimensions_are_clamped() {
        assert_eq!(clamp_variant_dim(0), 1);
        assert_eq!(clamp_variant_dim(1), 1);
        assert_eq!(clamp_variant_dim(16), 16);
        assert_eq!(clamp_variant_dim(99), 16);
    }

    #[test]
    fn atlas_renders_each_cell() {
        // 2×2 atlas of centred dots at 64² → each 32² cell has an opaque
        // centre and transparent corners.
        let map = generate_atlas(64, 64, 2, 2, 1.0, |_| Dot).expect("generate");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        for (cx, cy) in [(16usize, 16usize), (48, 16), (16, 48), (48, 48)] {
            let idx = (cy * 64 + cx) * 4;
            assert_eq!(map.albedo[idx + 3], 255, "cell centre ({cx},{cy})");
        }
        // Cell corners (atlas cross-hairs) stay transparent.
        let idx = (32 * 64 + 32) * 4;
        assert_eq!(map.albedo[idx + 3], 0, "cell boundary");
    }

    #[test]
    fn atlas_rejects_zero_dimensions() {
        assert!(generate_atlas(0, 64, 1, 1, 1.0, |_| Dot).is_err());
        assert!(generate_atlas(64, 0, 1, 1, 1.0, |_| Dot).is_err());
    }

    #[test]
    fn fbm2_stays_normalised() {
        let perlin = noise::Perlin::new(5);
        for i in 0..100 {
            let v = fbm2(&perlin, i as f64 * 0.37, i as f64 * 0.61, 4);
            assert!((0.0..=1.0).contains(&v));
        }
    }
}
