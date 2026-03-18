//! Pavers / tiles texture generator.
//!
//! The algorithm:
//! 1. Classify each pixel as **stone** or **grout** using a grid SDF.
//!    - `Square`: axis-aligned rectangular cells with a rounded-box SDF.
//!    - `Hexagonal`: flat-top hex grid via axial cube-rounding and IQ's hex SDF.
//! 2. Hash each cell's integer ID → per-paver colour variance.
//! 3. Overlay a toroidal FBM for surface micro-detail and bump height.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Layout of individual paver stones.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PaversLayout {
    /// Rectangular stones arranged in a regular grid.
    Square,
    /// Flat-top hexagonal stones.
    Hexagonal,
}

/// Configures the appearance of a [`PaversGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaversConfig {
    pub seed: u32,
    /// Grid density — roughly the number of pavers across the tile.
    pub scale: f64,
    /// Width-to-height ratio for `Square` stones (ignored for `Hexagonal`).
    pub aspect_ratio: f64,
    /// Grout gap as a fraction of stone size \[0, 0.4\].
    pub grout_width: f64,
    /// Corner bevel radius as a fraction of grout half-width \[0, 1\].
    pub bevel: f64,
    /// Per-paver colour jitter \[0, 1\].
    pub cell_variance: f64,
    /// Surface FBM micro-detail amplitude \[0, 1\].
    pub roughness: f64,
    /// Paving stone colour in linear RGB \[0, 1\].
    pub color_stone: [f32; 3],
    /// Grout / joint colour in linear RGB \[0, 1\].
    pub color_grout: [f32; 3],
    /// Stone layout pattern.
    pub layout: PaversLayout,
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for PaversConfig {
    fn default() -> Self {
        Self {
            seed: 23,
            scale: 5.0,
            aspect_ratio: 1.0,
            grout_width: 0.08,
            bevel: 0.5,
            cell_variance: 0.10,
            roughness: 0.30,
            color_stone: [0.48, 0.44, 0.40],
            color_grout: [0.28, 0.27, 0.26],
            layout: PaversLayout::Square,
            normal_strength: 3.5,
        }
    }
}

/// Procedural pavers / tiles texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`PaversConfig`].  Construct
/// via [`PaversGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::pavers`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct PaversGenerator {
    config: PaversConfig,
    surf_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl PaversGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: PaversConfig) -> Self {
        let fbm: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(50)).set_octaves(4);
        let surf_noise = ToroidalNoise::new(fbm, config.scale * 2.0);

        Self { config, surf_noise }
    }
}

impl TextureGenerator for PaversGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Surface micro-detail FBM.
        let surf_grid = sample_grid(&self.surf_noise, width, height);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        let grout_half = (c.grout_width * 0.5).clamp(0.0, 0.45);
        let bevel_r = (c.bevel * grout_half).max(0.0);
        // Inner half-extents for the stone SDF (before bevel).
        let hx = (0.5 - grout_half - bevel_r).max(0.0);
        let hy = (0.5 - grout_half - bevel_r).max(0.0);

        // Both column and row counts must be integers for the grid to tile.
        let cols = (c.scale * c.aspect_ratio).round();
        let rows = c.scale.round();

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;

            for x in 0..w {
                let u = x as f64 / w as f64;
                let idx = y * w + x;
                let raw_surf = normalize(surf_grid[idx]);

                let (sdf_val, cell_id_u, cell_id_v) = match c.layout {
                    PaversLayout::Square => square_cell(u, v, cols, rows, hx, hy, bevel_r),
                    PaversLayout::Hexagonal => hex_cell(u, v, c.scale),
                };

                let h_val;
                let (r, g, b);

                if sdf_val < 0.0 {
                    // Inside stone.
                    let edge_t = ((-sdf_val) / (bevel_r + 0.005)).clamp(0.0, 1.0);
                    let noise_bump = (raw_surf - 0.5) * c.roughness * 0.4;
                    h_val = (edge_t + noise_bump * edge_t).clamp(0.0, 1.0);

                    let cv = cell_hash(cell_id_u, cell_id_v, c.seed);
                    let jitter = (cv - 0.5) * 2.0 * c.cell_variance;
                    r = (c.color_stone[0] + jitter as f32).clamp(0.0, 1.0);
                    g = (c.color_stone[1] + jitter as f32 * 0.8).clamp(0.0, 1.0);
                    b = (c.color_stone[2] + jitter as f32 * 0.6).clamp(0.0, 1.0);
                } else {
                    // Grout joint.
                    h_val = raw_surf * c.roughness * 0.04;
                    r = c.color_grout[0];
                    g = c.color_grout[1];
                    b = c.color_grout[2];
                }

                heights[idx] = h_val;

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                let rough_val = if sdf_val < 0.0 {
                    0.70 + raw_surf as f32 * 0.20
                } else {
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

// --- Square grid ------------------------------------------------------------

/// Returns `(sdf, cell_id_u, cell_id_v)` for a square-grid paver.
///
/// `sdf < 0` means inside the stone; `sdf >= 0` means in the grout.
fn square_cell(
    u: f64,
    v: f64,
    cols: f64,
    rows: f64,
    hx: f64,
    hy: f64,
    bevel_r: f64,
) -> (f64, i64, i64) {
    let u_scaled = u * cols;
    let v_scaled = v * rows;
    let cell_u = u_scaled.floor() as i64;
    let cell_v = v_scaled.floor() as i64;
    let cx = u_scaled.fract() - 0.5;
    let cy = v_scaled.fract() - 0.5;

    // Rounded-box SDF: negative inside, positive outside.
    let dx = cx.abs() - hx;
    let dy = cy.abs() - hy;
    let sdf = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() + dx.max(dy).min(0.0) - bevel_r;

    (sdf, cell_u, cell_v)
}

// --- Hexagonal grid ---------------------------------------------------------

/// Returns `(sdf, cell_id_q, cell_id_r)` for a flat-top hexagonal paver.
///
/// Uses axial cube-rounding to find the nearest hex center, then IQ's hex SDF.
/// `sdf < 0` inside stone, `sdf >= 0` in grout.
///
/// # Tiling
/// A flat-top hex grid has an intrinsic aspect ratio of `sqrt(3)/1.5 ≈ 1.1547`.
/// There is no integer pair `(cols, rows)` that satisfies both horizontal and
/// vertical tiling on a square exactly.  We fix the horizontal period to
/// `1/scale` (exact for integer `scale`), then round the float row count to
/// the nearest integer and stretch `v` by that correction factor so an integer
/// number of rows fits in `[0, 1]`.  The hexes become very slightly non-regular
/// (< ~8% distortion for scale ≥ 2), which is imperceptible in practice.
fn hex_cell(u: f64, v: f64, scale: f64) -> (f64, i64, i64) {
    const SQRT3: f64 = 1.732_050_807_568_877_3;

    // Vertical tiling requires an integer number of rows; round scale.
    let scale = scale.round().max(1.0);
    // Circumradius so that `scale` hex rows fit exactly across [0, 1] vertically.
    // Row spacing = hex_r * √3, so scale rows require hex_r = 1 / (scale * √3).
    let hex_r = 1.0 / (scale * SQRT3);

    // The natural (float) number of horizontal columns in [0,1].
    // Column spacing = 1.5 * hex_r, so cols = 1 / (1.5 * hex_r) = scale * √3 / 1.5.
    // This is generally not an integer, so stretch u to make it tile.
    let cols_float = scale * SQRT3 / 1.5;
    let cols_int = cols_float.round().max(1.0);
    // Stretch u so that cols_int hex columns span [0, 1] exactly.
    let us = u * (cols_float / cols_int);

    // Convert to fractional axial coordinates (flat-top convention).
    let qf = (2.0 / 3.0) * us / hex_r;
    let rf = (-1.0 / 3.0) * us / hex_r + (SQRT3 / 3.0) * v / hex_r;
    let sf = -qf - rf;

    // Cube-round to nearest hex center.
    let (q, r, _s) = cube_round(qf, rf, sf);

    // Center of this hex in stretched UV space.
    let cx = hex_r * 1.5 * q as f64;
    let cy = hex_r * (SQRT3 / 2.0 * q as f64 + SQRT3 * r as f64);

    // Evaluate the SDF in stretched space (hexes are slightly non-regular).
    let dx = us - cx;
    let dy = v - cy;
    let sdf = hex_sdf(dx, dy, hex_r);

    (sdf, q, r)
}

/// Cube-coordinate rounding (standard hex-grid algorithm).
#[inline]
fn cube_round(qf: f64, rf: f64, sf: f64) -> (i64, i64, i64) {
    let (rq, rr, rs) = (qf.round() as i64, rf.round() as i64, sf.round() as i64);
    let (dq, dr, ds) = (
        (rq as f64 - qf).abs(),
        (rr as f64 - rf).abs(),
        (rs as f64 - sf).abs(),
    );
    if dq > dr && dq > ds {
        (-rr - rs, rr, rs)
    } else if dr > ds {
        (rq, -rq - rs, rs)
    } else {
        (rq, rr, -rq - rr)
    }
}

/// IQ's flat-top hexagon SDF.
///
/// `r` is the circumradius (center → vertex).  Returns negative inside,
/// positive outside.
#[inline]
fn hex_sdf(mut px: f64, mut py: f64, r: f64) -> f64 {
    // k = (-sqrt(3)/2, 0.5, 1/sqrt(3))
    const KX: f64 = -0.866_025_403_784;
    const KY: f64 = 0.5;
    const KZ: f64 = 0.577_350_269_189;

    px = px.abs();
    py = py.abs();
    let d = (KX * px + KY * py).min(0.0);
    px -= 2.0 * d * KX;
    py -= 2.0 * d * KY;
    let qx = px - px.clamp(-KZ * r, KZ * r);
    let qy = py - r;
    (qx * qx + qy * qy).sqrt() * qy.signum()
}

// --- Helpers ----------------------------------------------------------------

/// Deterministic integer cell hash → \[0, 1\].
fn cell_hash(bu: i64, bv: i64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (bu as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= (bv as u64).wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}
