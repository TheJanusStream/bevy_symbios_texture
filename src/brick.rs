//! Brick texture generator using grid-based SDF with per-cell hashing.
//!
//! The algorithm:
//! 1. Scale UV into a brick grid (`u * cols`, `v * rows`).
//! 2. Offset each row by `row_id × row_offset` to create bonding patterns.
//! 3. Compute a rounded-box SDF in cell-local space to separate brick from mortar.
//! 4. Hash each cell's integer ID to derive a per-brick colour variance.
//! 5. Blend toroidal surface-roughness FBM into the height field for micro-detail.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Configures the appearance of a [`BrickGenerator`].
///
/// For perfect vertical tiling the product `scale × row_offset` must be an
/// integer.  The default values (`scale = 4, row_offset = 0.5`) satisfy this
/// constraint (product = 2).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BrickConfig {
    pub seed: u32,
    /// Number of brick rows across the tile (controls coarseness).
    pub scale: f64,
    /// Lateral offset per row as a fraction of brick width.
    /// `0.0` = stack bond, `0.5` = running bond, `0.333` = third bond.
    pub row_offset: f64,
    /// Brick width-to-height ratio (e.g. `2.0` = standard 2:1 brick).
    pub aspect_ratio: f64,
    /// Mortar gap as a fraction of cell height \[0, 0.4\].
    pub mortar_size: f64,
    /// Corner bevel radius as a fraction of `mortar_size` \[0, 1\].
    pub bevel: f64,
    /// Per-brick colour jitter \[0, 1\].  `0.0` = uniform, `1.0` = highly varied.
    pub cell_variance: f64,
    /// Surface pitting / roughness noise intensity \[0, 1\].
    pub roughness: f64,
    /// Brick face colour in linear RGB \[0, 1\].
    pub color_brick: [f32; 3],
    /// Mortar colour in linear RGB \[0, 1\].
    pub color_mortar: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for BrickConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            scale: 4.0,
            row_offset: 0.5,
            aspect_ratio: 2.0,
            mortar_size: 0.10,
            bevel: 0.5,
            cell_variance: 0.12,
            roughness: 0.25,
            color_brick: [0.56, 0.28, 0.18],
            color_mortar: [0.76, 0.73, 0.67],
            normal_strength: 4.0,
        }
    }
}

/// Procedural brick-wall texture generator.
pub struct BrickGenerator {
    config: BrickConfig,
}

impl BrickGenerator {
    pub fn new(config: BrickConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for BrickGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Toroidal surface-roughness FBM for pitting detail.
        let fbm_rough: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(50)).set_octaves(4);
        let rough_noise = ToroidalNoise::new(fbm_rough, c.scale * c.aspect_ratio * 2.0);
        let rough_grid = sample_grid(&rough_noise, width, height);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // Bevel radius in cell-fraction space.
        let bevel_r = (c.bevel * c.mortar_size * 0.5).max(0.0);
        // Inner half-extents for the rounded-box SDF.
        let hx = (0.5 - c.mortar_size - bevel_r).max(0.0);
        let hy = (0.5 - c.mortar_size - bevel_r).max(0.0);

        // Both row count and column count must be integers for the grid to tile.
        let scale = c.scale.round();
        let cols = (scale * c.aspect_ratio).round();

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;
            let v_scaled = v * scale;
            let row_id = v_scaled.floor();
            let v_frac = v_scaled.fract();

            for x in 0..w {
                let u = x as f64 / w as f64;
                let u_shifted = u * cols + row_id * c.row_offset;
                let brick_id_u = u_shifted.floor() as i64;
                let brick_id_v = row_id as i64;
                let u_frac = u_shifted.fract();

                // Cell-centered coordinates in [-0.5, 0.5].
                let cx = u_frac - 0.5;
                let cy = v_frac - 0.5;

                // Rounded-box SDF: negative inside brick, positive in mortar.
                let dx = cx.abs() - hx;
                let dy = cy.abs() - hy;
                let sdf = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() + dx.max(dy).min(0.0)
                    - bevel_r;

                let raw_surf = normalize(rough_grid[y * w + x]);
                let h_val;
                let (r, gr, b);

                if sdf < 0.0 {
                    // Inside brick: bevel ramp + surface roughness.
                    let edge_t = ((-sdf) / (bevel_r + 0.01)).clamp(0.0, 1.0);
                    let noise_bump = (raw_surf - 0.5) * c.roughness * 0.4;
                    h_val = (edge_t + noise_bump * edge_t).clamp(0.0, 1.0);

                    // Per-brick colour variance via integer cell hash.
                    let cv = cell_hash(brick_id_u, brick_id_v, c.seed);
                    let jitter = (cv - 0.5) * 2.0 * c.cell_variance;
                    r = (c.color_brick[0] + jitter as f32).clamp(0.0, 1.0);
                    gr = (c.color_brick[1] + jitter as f32 * 0.7).clamp(0.0, 1.0);
                    b = (c.color_brick[2] + jitter as f32 * 0.5).clamp(0.0, 1.0);
                } else {
                    // Mortar gap: subtle texture.
                    h_val = raw_surf * c.roughness * 0.04;
                    r = c.color_mortar[0];
                    gr = c.color_mortar[1];
                    b = c.color_mortar[2];
                }

                let idx = y * w + x;
                heights[idx] = h_val;

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(gr);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // ORM: roughness higher in mortar, lower on smooth brick.
                let rough_val = if sdf < 0.0 {
                    0.45 + raw_surf as f32 * 0.3
                } else {
                    0.90
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

/// Deterministic integer hash → \[0, 1\].  Produces per-brick colour jitter
/// with good distribution and no visible lattice patterns.
fn cell_hash(bx: i64, by: i64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (bx as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= (by as u64).wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}

#[inline]
fn normalize(v: f64) -> f64 {
    v * 0.5 + 0.5
}
