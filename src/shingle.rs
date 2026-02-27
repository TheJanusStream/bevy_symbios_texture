//! Roof shingle / tile texture generator using overlapping gradient fields.
//!
//! The algorithm:
//! 1. Divide V into rows; stagger alternate rows horizontally.
//! 2. Build a sawtooth height ramp within each row (0 at the exposed lower edge,
//!    1 at the top where the shingle slides under the row above).
//! 3. Apply the shape function: `Square` (0.0) is a flat ramp; `Scalloped` (1.0)
//!    carves a half-circle from the bottom of each shingle.
//! 4. Add moss/algae noise near the lower exposed edge, weighted by `moss_level`.
//! 5. Blend a thin grout line at the row boundary for the sharp shadow step that
//!    makes the overlap visible.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Configures the appearance of a [`ShingleGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ShingleConfig {
    pub seed: u32,
    /// Number of shingle rows across the tile.
    pub scale: f64,
    /// Shape profile blended between square (`0.0`) and scalloped (`1.0`).
    pub shape_profile: f64,
    /// Fraction of each shingle hidden under the row above \[0, 0.8\].
    /// `0.5` = half-lap (standard); `0.0` = no overlap (flat tiles).
    pub overlap: f64,
    /// Horizontal stagger of alternate rows as a fraction of shingle width
    /// \[0, 1\].  `0.5` = running bond.
    pub stagger: f64,
    /// Moss / algae growth intensity on the lower exposed edge \[0, 1\].
    pub moss_level: f64,
    /// Primary tile colour in linear RGB \[0, 1\].
    pub color_tile: [f32; 3],
    /// Shadow / grout colour between rows in linear RGB \[0, 1\].
    pub color_grout: [f32; 3],
    /// Normal-map strength.  Higher values exaggerate the overlap step.
    pub normal_strength: f32,
}

impl Default for ShingleConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            scale: 5.0,
            shape_profile: 0.5,
            overlap: 0.45,
            stagger: 0.5,
            moss_level: 0.18,
            color_tile: [0.40, 0.25, 0.18],
            color_grout: [0.18, 0.14, 0.12],
            normal_strength: 5.0,
        }
    }
}

/// Procedural roof-shingle / tile texture generator.
pub struct ShingleGenerator {
    config: ShingleConfig,
}

impl ShingleGenerator {
    pub fn new(config: ShingleConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for ShingleGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Surface micro-detail FBM (toroidal for seamless tiling).
        let fbm_surf: Fbm<Perlin> = Fbm::new(c.seed).set_octaves(4);
        let surf_noise = ToroidalNoise::new(fbm_surf, c.scale * 3.0);
        let surf_grid = sample_grid(&surf_noise, width, height);

        // Moss noise — low frequency, toroidal.
        let fbm_moss: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(100)).set_octaves(3);
        let moss_noise = ToroidalNoise::new(fbm_moss, c.scale * 1.5);
        let moss_grid = sample_grid(&moss_noise, width, height);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // Exposed (visible) fraction of each shingle from the bottom.
        let exposed = (1.0 - c.overlap).clamp(0.05, 1.0);
        // Thin grout / shadow band at the bottom of each exposed portion.
        let grout_frac = 0.06_f64;

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        // scale must be an integer for the grid to tile; round to nearest.
        let scale = c.scale.round();

        for y in 0..h {
            let v = y as f64 / h as f64;
            let v_scaled = v * scale;
            let row_id = v_scaled.floor() as i64;
            let v_frac = v_scaled.fract(); // 0 = bottom of cell, 1 = top

            for x in 0..w {
                let u = x as f64 / w as f64;

                // Stagger: shift U by row_id × stagger so alternate rows offset.
                let u_stagger = (u + row_id as f64 * c.stagger).rem_euclid(1.0);
                let col_id = (u_stagger * scale).floor() as i64;
                let u_frac = (u_stagger * scale).fract(); // 0..1 within shingle cell

                // Per-shingle colour variance hash.
                let cv = cell_hash(col_id, row_id, c.seed);

                // ── Height ramp ──────────────────────────────────────────────
                // The sawtooth ramp: 0 at the exposed bottom edge, ramps up to 1
                // at the top.  Only the [0, exposed] portion is visible; above
                // that is hidden under the shingle from the row above.
                let ramp = (v_frac / exposed).clamp(0.0, 1.0);

                // ── Shape function ───────────────────────────────────────────
                // `shape_profile=0`: pure ramp (square / flat shingle).
                // `shape_profile=1`: scalloped — carve a half-circle from the
                //   bottom corners so the exposed edge is convex in the middle.
                let dx = (u_frac - 0.5).abs() * 2.0; // 0 at centre, 1 at edge
                let scallop_drop = c.shape_profile
                    * (1.0 - (1.0 - dx * dx).sqrt()) // half-circle profile
                    * (1.0 - ramp).powi(2); // only affects bottom edge
                let ramp_shaped = (ramp - scallop_drop).clamp(0.0, 1.0);

                // ── Grout / shadow line ──────────────────────────────────────
                let in_grout = v_frac < grout_frac && ramp_shaped < 0.15;

                let idx = y * w + x;
                let ai = idx * 4;
                let surf = normalize(surf_grid[idx]);
                let moss_raw = normalize(moss_grid[idx]);

                // Moss grows on the lower exposed portion of each shingle.
                let moss_weight = c.moss_level
                    * moss_raw
                    * (1.0 - ramp_shaped).powi(3)
                    * (1.0 - in_grout as i32 as f64);

                if in_grout {
                    // Grout / shadow line between rows.
                    heights[idx] = 0.0;
                    albedo[ai] = linear_to_srgb(c.color_grout[0]);
                    albedo[ai + 1] = linear_to_srgb(c.color_grout[1]);
                    albedo[ai + 2] = linear_to_srgb(c.color_grout[2]);
                    albedo[ai + 3] = 255;
                    roughness_buf[ai] = 255;
                    roughness_buf[ai + 1] = (0.92 * 255.0) as u8;
                    roughness_buf[ai + 2] = 0;
                    roughness_buf[ai + 3] = 255;
                } else {
                    // Shingle surface with micro-detail and moss.
                    let h_val =
                        (ramp_shaped * (0.9 + surf * 0.1) - moss_weight * 0.05).clamp(0.0, 1.0);
                    heights[idx] = h_val;

                    // Tile colour: jitter per cell, darken with moss.
                    let jitter = (cv - 0.5) * 0.12;
                    let moss_green = [0.15f32, 0.28, 0.10];
                    let base_r = (c.color_tile[0] + jitter as f32).clamp(0.0, 1.0);
                    let base_g = (c.color_tile[1] + jitter as f32 * 0.8).clamp(0.0, 1.0);
                    let base_b = (c.color_tile[2] + jitter as f32 * 0.5).clamp(0.0, 1.0);
                    let r = lerp(base_r, moss_green[0], moss_weight as f32);
                    let g = lerp(base_g, moss_green[1], moss_weight as f32);
                    let b = lerp(base_b, moss_green[2], moss_weight as f32);

                    albedo[ai] = linear_to_srgb(r);
                    albedo[ai + 1] = linear_to_srgb(g);
                    albedo[ai + 2] = linear_to_srgb(b);
                    albedo[ai + 3] = 255;

                    // ORM: lower (exposed) areas and moss are rougher.
                    let rough = 0.55 + (1.0 - ramp_shaped as f32) * 0.3 + moss_weight as f32 * 0.1;
                    roughness_buf[ai] = 255;
                    roughness_buf[ai + 1] = (rough.clamp(0.0, 1.0) * 255.0).round() as u8;
                    roughness_buf[ai + 2] = 0;
                    roughness_buf[ai + 3] = 255;
                }
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

fn cell_hash(col: i64, row: i64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (col as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= (row as u64).wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

#[inline]
fn normalize(v: f64) -> f64 {
    v * 0.5 + 0.5
}
