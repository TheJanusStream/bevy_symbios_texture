//! Stained-glass alpha-card texture generator.
//!
//! The algorithm:
//! 1. Build a toroidal Voronoi diagram using an `n × n` grid of jittered sites
//!    (where `n = round(sqrt(cell_count))`).  For each pixel compute F1 and F2
//!    distances plus the integer cell ID of the nearest site.
//! 2. Pixels whose `F2 – F1` is below `lead_width / n` lie on a lead came
//!    boundary and are rendered as opaque dark metal.
//! 3. Glass pixels receive a vibrant HSV colour derived from the cell hash.
//!    The saturation parameter scales the chroma.  A grime FBM adds subtle
//!    dirt accumulation.
//! 4. Alpha: lead came = 255 (fully opaque); glass = 180 (semi-transparent).
//! 5. Heights: lead = 1.0 (proud of the glass face); glass surface = grime bump.
//!    `dilate_heights` is called before `height_to_normal` so the normal map
//!    has no hard cliff at the alpha silhouette, and `BoundaryMode::Clamp` is
//!    used because this is a card texture that must not tile.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    normal::{BoundaryMode, dilate_heights, height_to_normal},
};

/// Configures the appearance of a [`StainedGlassGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StainedGlassConfig {
    pub seed: u32,
    /// Approximate number of glass cells \[5, 25\].
    pub cell_count: usize,
    /// Lead came (border) width as fraction of cell spacing \[0.02, 0.12\].
    pub lead_width: f64,
    /// Glass colour saturation factor \[0.5, 1.0\].
    pub saturation: f32,
    /// Glass surface roughness (waviness) \[0, 0.15\].
    pub glass_roughness: f64,
    /// Grime/dirt accumulation on glass \[0, 0.5\].
    pub grime_level: f64,
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for StainedGlassConfig {
    fn default() -> Self {
        Self {
            seed: 63,
            cell_count: 12,
            lead_width: 0.05,
            saturation: 0.85,
            glass_roughness: 0.06,
            grime_level: 0.12,
            normal_strength: 2.5,
        }
    }
}

/// Procedural stained-glass texture generator (alpha-card type).
///
/// The result has per-pixel alpha: semi-transparent glass cells separated by
/// fully-opaque lead came lines.  Upload via
/// [`crate::generator::map_to_images_card`] to select the clamp-to-edge
/// sampler automatically.
pub struct StainedGlassGenerator {
    config: StainedGlassConfig,
}

impl StainedGlassGenerator {
    pub fn new(config: StainedGlassConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for StainedGlassGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Grime FBM — no tiling needed for a card texture.
        let grime_fbm: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(11)).set_octaves(5);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // Grid size: n×n gives approximately cell_count cells (n² ≈ cell_count).
        let grid_n = ((c.cell_count as f64).sqrt().round() as i64).max(2);

        // Lead threshold in UV distance units.
        let lead_threshold = c.lead_width / grid_n as f64;

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;

            for x in 0..w {
                let u = x as f64 / w as f64;
                let idx = y * w + x;
                let ai = idx * 4;

                let (f1, f2, ci, cj) = voronoi_f1_f2(u, v, grid_n, c.seed);
                let is_lead = (f2 - f1) < lead_threshold;

                if is_lead {
                    // Lead came: opaque dark metal.
                    heights[idx] = 1.0;

                    let lead_r: f32 = 0.05;
                    let lead_g: f32 = 0.05;
                    let lead_b: f32 = 0.06;
                    albedo[ai] = linear_to_srgb(lead_r);
                    albedo[ai + 1] = linear_to_srgb(lead_g);
                    albedo[ai + 2] = linear_to_srgb(lead_b);
                    albedo[ai + 3] = 255;

                    roughness_buf[ai] = 255;
                    roughness_buf[ai + 1] = (0.40 * 255.0) as u8;
                    roughness_buf[ai + 2] = (0.80 * 255.0) as u8; // metallic lead
                    roughness_buf[ai + 3] = 255;
                } else {
                    // Glass pane: derive vibrant colour from cell hash.
                    let hue = cell_hash(ci, cj, c.seed.wrapping_add(100));
                    let sat_h = cell_hash(cj, ci, c.seed.wrapping_add(200));
                    let saturation = (0.70 + sat_h * 0.30) * c.saturation as f64;
                    let glass_rgb = hsv_to_rgb(hue, saturation.clamp(0.0, 1.0), 0.85);

                    // Grime: FBM dirt on the glass surface.
                    let grime_raw = grime_fbm.get([u * 6.0, v * 6.0]) * 0.5 + 0.5;
                    let grime = (grime_raw * c.grime_level) as f32;

                    heights[idx] = grime_raw * c.grime_level * 0.05;

                    // Darken glass slightly by grime.
                    let r = (glass_rgb[0] - grime * 0.25).clamp(0.0, 1.0);
                    let g = (glass_rgb[1] - grime * 0.20).clamp(0.0, 1.0);
                    let b = (glass_rgb[2] - grime * 0.15).clamp(0.0, 1.0);

                    albedo[ai] = linear_to_srgb(r);
                    albedo[ai + 1] = linear_to_srgb(g);
                    albedo[ai + 2] = linear_to_srgb(b);
                    albedo[ai + 3] = 180; // semi-transparent glass

                    // Glass ORM: low roughness, high metallic (simulates reflections).
                    let glass_rough = (c.glass_roughness + grime_raw * c.grime_level * 0.2)
                        .clamp(0.0, 1.0) as f32;
                    roughness_buf[ai] = 255;
                    roughness_buf[ai + 1] = (glass_rough * 255.0).round() as u8;
                    roughness_buf[ai + 2] = (0.70 * 255.0) as u8; // reflective glass
                    roughness_buf[ai + 3] = 255;
                }
            }
        }

        // Dilate opaque heights one step into transparent neighbours so the
        // normal map avoids a hard cliff at the lead silhouette.
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
            roughness: roughness_buf,
            width,
            height,
        })
    }
}

// --- Voronoi ----------------------------------------------------------------

/// Grid-based toroidal Voronoi returning `(F1, F2, cell_i, cell_j)`.
///
/// The grid has `n × n` cells.  Sites are jittered within `[0.1, 0.9]` of
/// each cell.  Distances wrap toroidally in `[0, 1]²`.
fn voronoi_f1_f2(u: f64, v: f64, n: i64, seed: u32) -> (f64, f64, i64, i64) {
    let su = u * n as f64;
    let sv = v * n as f64;
    let gi = su.floor() as i64;
    let gj = sv.floor() as i64;

    let mut f1 = f64::MAX;
    let mut f2 = f64::MAX;
    let mut bi = gi;
    let mut bj = gj;

    for di in -2i64..=2 {
        for dj in -2i64..=2 {
            let ni = (gi + di).rem_euclid(n);
            let nj = (gj + dj).rem_euclid(n);

            // Jitter within [0.1, 0.9] to avoid degenerate zero-area cells.
            let jx = 0.10 + 0.80 * cell_hash(ni, nj, seed);
            let jy = 0.10 + 0.80 * cell_hash(nj, ni + 1, seed.wrapping_add(31));

            let cx = (ni as f64 + jx) / n as f64;
            let cy = (nj as f64 + jy) / n as f64;

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
                bi = ni;
                bj = nj;
            } else if d < f2 {
                f2 = d;
            }
        }
    }

    (f1, f2, bi, bj)
}

// --- helpers ----------------------------------------------------------------

/// Convert HSV (all in `[0, 1]`) to linear-light RGB `[f32; 3]`.
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [f32; 3] {
    let h6 = (h * 6.0).rem_euclid(6.0);
    let i = h6.floor() as u32;
    let f = h6.fract();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r as f32, g as f32, b as f32]
}

/// Deterministic integer cell hash → `[0, 1]`.
fn cell_hash(bx: i64, by: i64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (bx as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= (by as u64).wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}
