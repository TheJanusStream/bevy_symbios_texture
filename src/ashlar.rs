//! Ashlar (cut-stone masonry) texture generator.
//!
//! The algorithm:
//! 1. Pre-compute irregular row heights and per-row column widths using integer
//!    hashes so that each row contains a slightly different number of blocks and
//!    each block has a distinct width, simulating hand-cut stonework.
//! 2. For each pixel, locate the enclosing block (row then column) and compute a
//!    rounded-box SDF to separate stone face from mortar joint.
//! 3. Inside the stone, blend a toroidal FBM for surface micro-detail and a
//!    chisel effect near the block edges; apply per-block colour variance.
//! 4. In the mortar joint, render the mortar colour with near-zero height.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Configures the appearance of an [`AshlarGenerator`].
///
/// Ashlar masonry uses irregular but tightly-fitted rectangular stone blocks
/// arranged in horizontal courses.  Each course can have a different number of
/// blocks and each block has a distinct width, giving the characteristic
/// hand-dressed appearance of castle or cathedral walls.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AshlarConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Number of stone courses (rows) across the tile \[2, 8\].
    pub rows: usize,
    /// Base number of blocks per course \[2, 6\].  Each row may vary by ±1.
    pub cols: usize,
    /// Mortar gap as a fraction of average cell size \[0, 0.15\].
    pub mortar_size: f64,
    /// Bevel radius as a fraction of `mortar_size` \[0, 1\].
    pub bevel: f64,
    /// Per-block colour jitter \[0, 1\].  `0.0` = uniform stone colour.
    pub cell_variance: f64,
    /// Chisel-edge depth — strength of the darkening near each block border \[0, 1\].
    pub chisel_depth: f64,
    /// FBM face micro-detail amplitude \[0, 1\].
    pub roughness: f64,
    /// Stone face colour in linear RGB \[0, 1\].
    pub color_stone: [f32; 3],
    /// Mortar joint colour in linear RGB \[0, 1\].
    pub color_mortar: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for AshlarConfig {
    fn default() -> Self {
        Self {
            seed: 13,
            rows: 4,
            cols: 4,
            mortar_size: 0.04,
            bevel: 0.4,
            cell_variance: 0.18,
            chisel_depth: 0.4,
            roughness: 0.45,
            color_stone: [0.52, 0.50, 0.47],
            color_mortar: [0.72, 0.70, 0.65],
            normal_strength: 4.5,
        }
    }
}

/// Procedural ashlar (cut-stone masonry) texture generator.
///
/// Drives [`TextureGenerator::generate`] using an [`AshlarConfig`].  Construct
/// via [`AshlarGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::ashlar`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct AshlarGenerator {
    config: AshlarConfig,
    rough_noise: ToroidalNoise<Fbm<Perlin>>,
    chisel_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl AshlarGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: AshlarConfig) -> Self {
        let fbm_rough: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(50)).set_octaves(5);
        let rough_noise =
            ToroidalNoise::new(fbm_rough, config.cols as f64 * config.rows as f64 * 0.8);
        let fbm_chisel: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(200)).set_octaves(3);
        let chisel_noise =
            ToroidalNoise::new(fbm_chisel, config.cols as f64 * config.rows as f64 * 2.5);
        Self {
            config,
            rough_noise,
            chisel_noise,
        }
    }
}

impl TextureGenerator for AshlarGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Toroidal FBM for stone-face micro-detail and chisel approximation.
        let rough_grid = sample_grid(&self.rough_noise, width, height);

        // Second FBM at higher frequency for fine chisel/crack detail.
        let chisel_grid = sample_grid(&self.chisel_noise, width, height);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // ── Pre-compute row structure ─────────────────────────────────────────
        // Row heights: each varies based on a hash of the row index, then
        // normalised so the full tile height sums to 1.0.
        let rows = c.rows.max(1);
        let row_heights_raw: Vec<f64> = (0..rows)
            .map(|r| 0.6 + 0.8 * cell_hash(r as i64, 99, c.seed.wrapping_add(1)))
            .collect();
        let total_h: f64 = row_heights_raw.iter().sum();
        let row_heights: Vec<f64> = row_heights_raw.iter().map(|h| h / total_h).collect();

        // Cumulative row boundaries: [0.0, h0, h0+h1, …, 1.0].
        let row_cum: Vec<f64> = std::iter::once(0.0)
            .chain(row_heights.iter().scan(0.0, |acc, rh| {
                *acc += rh;
                Some(*acc)
            }))
            .collect();

        // Per-row column count: may vary by ±1 from the base `cols`.
        let cols_base = c.cols.max(1);
        let row_ncols: Vec<usize> = (0..rows)
            .map(|r| {
                let h = cell_hash(r as i64, 77, c.seed.wrapping_add(2));
                if h < 0.33 && cols_base > 2 {
                    cols_base - 1
                } else if h > 0.67 {
                    cols_base + 1
                } else {
                    cols_base
                }
            })
            .collect();

        // Per-row cumulative column widths: [0.0, w0, w0+w1, …, 1.0].
        let col_cums: Vec<Vec<f64>> = (0..rows)
            .map(|r| {
                let ncols = row_ncols[r];
                let widths: Vec<f64> = (0..ncols)
                    .map(|cl| {
                        0.5 + 0.8 * cell_hash(r as i64 * 31 + cl as i64, 13, c.seed.wrapping_add(3))
                    })
                    .collect();
                let total: f64 = widths.iter().sum();
                let normed: Vec<f64> = widths.iter().map(|ww| ww / total).collect();
                std::iter::once(0.0)
                    .chain(normed.iter().scan(0.0, |acc, ww| {
                        *acc += ww;
                        Some(*acc)
                    }))
                    .collect()
            })
            .collect();

        // ── SDF constants ────────────────────────────────────────────────────
        // These are the *relative* half-extents inside a unit cell [0,1]×[0,1].
        // We re-derive them per-pixel using absolute UV distances instead, so
        // the mortar gap is uniform regardless of block aspect ratio.  The
        // config `mortar_size` is expressed as a fraction of the *average* cell
        // size in UV space.
        let avg_cell_size = 1.0 / (rows as f64).max(1.0) / (cols_base as f64).max(1.0);
        // Absolute mortar half-gap in UV units.
        let mortar_gap_uv = c.mortar_size * avg_cell_size * 0.5;
        let bevel_r_uv = (c.bevel * mortar_gap_uv).max(0.0);

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;

            // Find which row this pixel belongs to.
            let row = {
                let idx = row_cum.partition_point(|&b| b <= v).saturating_sub(1);
                idx.min(rows - 1)
            };
            let row_lo = row_cum[row];
            let row_hi = row_cum[row + 1];
            // Local V within this row in [0, 1].
            let v_local = ((v - row_lo) / (row_hi - row_lo)).clamp(0.0, 1.0);
            // Cell-centred V coordinate in [-0.5, 0.5].
            let cy_cell = v_local - 0.5;
            // Half-extent of the stone in V (row height in UV units).
            let row_h_uv = row_hi - row_lo;

            for x in 0..w {
                let u = x as f64 / w as f64;

                // Find which column this pixel belongs to within the current row.
                let cum = &col_cums[row];
                let ncols = row_ncols[row];
                let col = {
                    let idx = cum.partition_point(|&b| b < u).saturating_sub(1);
                    idx.min(ncols - 1)
                };
                let col_lo = cum[col];
                let col_hi = cum[col + 1];
                // Local U within this block in [0, 1].
                let u_local = ((u - col_lo) / (col_hi - col_lo)).clamp(0.0, 1.0);
                // Cell-centred U coordinate in [-0.5, 0.5].
                let cx_cell = u_local - 0.5;
                // Half-extent of the stone in U (column width in UV units).
                let col_w_uv = col_hi - col_lo;

                // ── Rounded-box SDF ───────────────────────────────────────────
                // The SDF is computed in *UV space*, not normalised cell space,
                // so the mortar gap is visually uniform.  We map the centred
                // cell-local coordinates back to UV distances.
                let px = cx_cell * col_w_uv; // UV offset from block centre (U)
                let py = cy_cell * row_h_uv; // UV offset from block centre (V)

                // Inner half-extents (stone face, before bevel).
                let hx = (col_w_uv * 0.5 - mortar_gap_uv - bevel_r_uv).max(0.0);
                let hy = (row_h_uv * 0.5 - mortar_gap_uv - bevel_r_uv).max(0.0);

                let dx = px.abs() - hx;
                let dy = py.abs() - hy;
                let sdf = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() + dx.max(dy).min(0.0)
                    - bevel_r_uv;

                let idx = y * w + x;
                let raw_surf = normalize(rough_grid[idx]);
                let raw_chisel = normalize(chisel_grid[idx]);

                let h_val;
                let (r, gr, b);

                if sdf < 0.0 {
                    // ── Inside the stone block ────────────────────────────────
                    // Bevel ramp: rises from 0 at the border to 1 in the
                    // interior.  Scale by the effective bevel radius, with a
                    // small floor so the ramp has finite width even at bevel=0.
                    let bevel_zone = (bevel_r_uv + mortar_gap_uv * 0.3 + 1e-5).max(1e-5);
                    let edge_t = ((-sdf) / bevel_zone).clamp(0.0, 1.0);

                    // Chisel darkening: strongest near block edges, fades inward.
                    // Use the raw chisel FBM sample to break up the uniformity.
                    let edge_proximity = (1.0 - edge_t).powi(2);
                    let chisel_bump = raw_chisel * c.chisel_depth * edge_proximity;

                    // Face micro-detail from the rough FBM.
                    let face_bump = (raw_surf - 0.5) * c.roughness * 0.35;

                    h_val = (edge_t + face_bump * edge_t - chisel_bump * 0.4).clamp(0.0, 1.0);

                    // Per-block colour jitter via integer cell hash.
                    let block_id = row as i64 * 1000 + col as i64;
                    let cv = cell_hash(block_id, row as i64, c.seed.wrapping_add(77));
                    let jitter = (cv - 0.5) * 2.0 * c.cell_variance;
                    // Chisel darkening also tints the colour.
                    let chisel_darken = (chisel_bump * c.chisel_depth * 0.6) as f32;
                    r = (c.color_stone[0] + jitter as f32 - chisel_darken).clamp(0.0, 1.0);
                    gr = (c.color_stone[1] + jitter as f32 * 0.8 - chisel_darken).clamp(0.0, 1.0);
                    b = (c.color_stone[2] + jitter as f32 * 0.6 - chisel_darken).clamp(0.0, 1.0);
                } else {
                    // ── Mortar joint ──────────────────────────────────────────
                    h_val = raw_surf * c.roughness * 0.03;
                    r = c.color_mortar[0];
                    gr = c.color_mortar[1];
                    b = c.color_mortar[2];
                }

                heights[idx] = h_val;

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(gr);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                let rough_val = if sdf < 0.0 {
                    // Stone face: moderate roughness with FBM variation.
                    0.50 + raw_surf as f32 * 0.30
                } else {
                    // Mortar: high roughness.
                    0.92
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

// --- helpers ----------------------------------------------------------------

/// Deterministic integer hash → \[0, 1\].  Produces per-block colour and
/// geometry jitter with good distribution and no visible lattice patterns.
fn cell_hash(bx: i64, by: i64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (bx as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= (by as u64).wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}
