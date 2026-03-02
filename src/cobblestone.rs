//! Cobblestone texture generator using Voronoi cell decomposition.
//!
//! The algorithm:
//! 1. Compute a toroidally-wrapped Voronoi diagram (F1 and F2 distances plus
//!    the integer cell ID of the nearest site).
//! 2. Pixels whose `F2 – F1` is below `gap_threshold` lie on a cell boundary
//!    and are rendered as mud/dirt.
//! 3. Stone pixels receive a domed height profile (`1 – (F1·scale)^roundness`)
//!    blended with a toroidal FBM for micro-surface detail.
//! 4. Per-stone colour variance is driven by the integer cell hash of the
//!    nearest Voronoi site.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Configures the appearance of a [`CobblestoneGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CobblestoneConfig {
    pub seed: u32,
    /// Approximate number of stones across the tile \[3, 12\].
    pub scale: f64,
    /// Mud gap threshold as a fraction of stone spacing \[0.02, 0.25\].
    pub gap_width: f64,
    /// Per-stone colour jitter \[0, 1\].  `0.0` = uniform colour.
    pub cell_variance: f64,
    /// Stone roundness — controls how domed the tops of the stones are \[0.5, 2.0\].
    /// Higher values produce flatter stones with steeper sides.
    pub roundness: f64,
    /// Stone colour in linear RGB \[0, 1\].
    pub color_stone: [f32; 3],
    /// Mud / dirt gap colour in linear RGB \[0, 1\].
    pub color_mud: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for CobblestoneConfig {
    fn default() -> Self {
        Self {
            seed: 7,
            scale: 6.0,
            gap_width: 0.12,
            cell_variance: 0.20,
            roundness: 1.2,
            color_stone: [0.46, 0.43, 0.40],
            color_mud: [0.22, 0.18, 0.14],
            normal_strength: 5.0,
        }
    }
}

/// Procedural cobblestone texture generator.
pub struct CobblestoneGenerator {
    config: CobblestoneConfig,
}

impl CobblestoneGenerator {
    pub fn new(config: CobblestoneConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for CobblestoneGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Toroidal FBM for stone-surface micro-detail.
        let fbm_surf: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(50)).set_octaves(4);
        let surf_noise = ToroidalNoise::new(fbm_surf, c.scale * 2.5);
        let surf_grid = sample_grid(&surf_noise, width, height);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // Gap threshold in UV distance units: stones end where the Voronoi
        // boundary is closer than this value.
        let gap_threshold = c.gap_width / c.scale;

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;

            for x in 0..w {
                let u = x as f64 / w as f64;

                let idx = y * w + x;
                let raw_surf = normalize(surf_grid[idx]);

                // Grid-based toroidal Voronoi: returns F1, F2, and the integer
                // cell coordinates of the nearest site.
                let (f1, f2, ci, cj) = voronoi(u, v, c.scale, c.seed);

                let h_val;
                let (r, g, b);

                if f2 - f1 < gap_threshold {
                    // ── Mud / dirt gap ────────────────────────────────────────
                    h_val = raw_surf * 0.04;
                    r = c.color_mud[0];
                    g = c.color_mud[1];
                    b = c.color_mud[2];
                } else {
                    // ── Stone face ────────────────────────────────────────────
                    // Dome profile: peaks at 1.0 directly over the Voronoi
                    // site and falls toward 0.0 at the cell boundary.
                    let dome_base = (1.0 - (f1 * c.scale).powf(c.roundness)).clamp(0.0, 1.0);
                    // Blend FBM micro-detail into the height.
                    h_val = (dome_base * (0.85 + raw_surf * 0.15)).clamp(0.0, 1.0);

                    // Per-stone colour jitter via cell hash.
                    let cv = cell_hash(ci, cj, c.seed.wrapping_add(99));
                    let jitter = (cv - 0.5) * 2.0 * c.cell_variance;
                    r = (c.color_stone[0] + jitter as f32).clamp(0.0, 1.0);
                    g = (c.color_stone[1] + jitter as f32 * 0.85).clamp(0.0, 1.0);
                    b = (c.color_stone[2] + jitter as f32 * 0.65).clamp(0.0, 1.0);
                }

                heights[idx] = h_val;

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // ORM: stone roughness varies with dome height (higher = smoother),
                // mud is nearly fully rough.
                let rough_val = if f2 - f1 < gap_threshold {
                    0.95
                } else {
                    // Smoother at the crown, rougher toward the edges.
                    (0.85 - h_val as f32 * 0.20 + raw_surf as f32 * 0.10).clamp(0.60, 0.90)
                };
                roughness_buf[ai] = 255;
                roughness_buf[ai + 1] = (rough_val * 255.0).round() as u8;
                roughness_buf[ai + 2] = 0;
                roughness_buf[ai + 3] = 255;
            }
        }

        let normal = height_to_normal(
            &heights,
            width,
            height,
            c.normal_strength,
            BoundaryMode::Wrap,
        );

        Ok(TextureMap {
            albedo,
            normal,
            roughness: roughness_buf,
            width,
            height,
        })
    }
}

// --- Voronoi ----------------------------------------------------------------

/// Grid-based toroidal Voronoi in UV space.
///
/// Partitions `[0, 1]²` into `scale × scale` candidate cells and searches a
/// 5×5 neighbourhood around the query point, wrapping toroidally.  Returns
/// `(F1, F2, best_i, best_j)` where F1/F2 are the two nearest site distances
/// in UV units and `(best_i, best_j)` is the integer cell coordinate of the
/// F1 site.
fn voronoi(u: f64, v: f64, scale: f64, seed: u32) -> (f64, f64, i64, i64) {
    let n = scale.round().max(1.0) as i64;
    let su = u * scale;
    let sv = v * scale;
    let gi = su.floor() as i64;
    let gj = sv.floor() as i64;

    let mut f1 = f64::MAX;
    let mut f2 = f64::MAX;
    let mut best_i = gi;
    let mut best_j = gj;

    for di in -2i64..=2 {
        for dj in -2i64..=2 {
            let ni = (gi + di).rem_euclid(n);
            let nj = (gj + dj).rem_euclid(n);

            // Jitter: the site is placed within [0.15, 0.85] of its cell to
            // avoid degenerate near-zero-area cells at the lattice corners.
            let jx = 0.15 + 0.70 * cell_hash(ni, nj, seed);
            let jy = 0.15 + 0.70 * cell_hash(nj, ni, seed.wrapping_add(17));

            // Site position in UV space.
            let cx = (ni as f64 + jx) / scale;
            let cy = (nj as f64 + jy) / scale;

            // Toroidal distance.
            let mut dx = (u - cx).abs();
            let mut dy = (v - cy).abs();
            if dx > 0.5 {
                dx = 1.0 - dx;
            }
            if dy > 0.5 {
                dy = 1.0 - dy;
            }
            let d = (dx * dx + dy * dy).sqrt();

            if d < f1 {
                f2 = f1;
                f1 = d;
                best_i = ni;
                best_j = nj;
            } else if d < f2 {
                f2 = d;
            }
        }
    }

    (f1, f2, best_i, best_j)
}

// --- helpers ----------------------------------------------------------------

/// Deterministic integer hash → \[0, 1\].  Drives Voronoi site jitter and
/// per-stone colour variance.
fn cell_hash(bx: i64, by: i64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (bx as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= (by as u64).wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}
