//! Encaustic ceramic tile texture generator.
//!
//! The algorithm:
//! 1. Precompute a toroidal glaze FBM grid for surface waviness.
//! 2. For each pixel, tile the UV space by `scale` to find the integer cell
//!    coordinates `(ci, cj)` and the cell-local offset `(cx, cy) ∈ [-0.5, 0.5]`.
//! 3. Classify the pixel as tile-A, tile-B, or grout according to the chosen
//!    `pattern`:
//!    - **Checkerboard**: alternating squares driven by `(ci + cj) % 2`.
//!    - **Octagon**: octagon with corner-clipped shape in each cell; the small
//!      square that fills the four corners where adjacent octagons meet is
//!      colour-B.
//!    - **Diamond**: rotated 45° grid; a diamond (rotated square) of colour-A
//!      centred in each cell, grout in between.
//! 4. The glaze FBM perturbs the final colour slightly and controls tile-face
//!    height waviness (simulating hand-fired surface irregularity).
//! 5. ORM: tile faces are glossy glazed ceramic (low roughness, zero metallic);
//!    grout is matte (high roughness, zero metallic).
//! 6. `BoundaryMode::Wrap` for seamless tiling.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Geometric pattern used by [`EncausticGenerator`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EncausticPattern {
    /// Simple alternating checkerboard of two colours.
    Checkerboard,
    /// Octagon + small square at cell corners (classic Mediterranean tile).
    Octagon,
    /// Rotated diamond / argyle grid.
    Diamond,
}

/// Configures the appearance of an [`EncausticGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EncausticConfig {
    pub seed: u32,
    /// Tile cells across the texture (both axes) \[2, 10\].
    pub scale: f64,
    /// Geometric pattern.
    pub pattern: EncausticPattern,
    /// Grout line width as fraction of cell \[0.02, 0.15\].
    pub grout_width: f64,
    /// Glaze surface waviness (FBM amplitude) \[0, 0.1\].
    pub glaze_roughness: f64,
    /// Primary shape colour (terra cotta, etc.) in linear RGB \[0, 1\].
    pub color_a: [f32; 3],
    /// Secondary shape colour (blue, white, etc.) in linear RGB \[0, 1\].
    pub color_b: [f32; 3],
    /// Grout colour in linear RGB \[0, 1\].
    pub color_grout: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for EncausticConfig {
    fn default() -> Self {
        Self {
            seed: 47,
            scale: 5.0,
            pattern: EncausticPattern::Octagon,
            grout_width: 0.06,
            glaze_roughness: 0.04,
            color_a: [0.72, 0.38, 0.22],
            color_b: [0.22, 0.35, 0.65],
            color_grout: [0.82, 0.80, 0.75],
            normal_strength: 3.0,
        }
    }
}

/// Procedural encaustic ceramic tile texture generator.
///
/// Drives [`TextureGenerator::generate`] using an [`EncausticConfig`].  Construct
/// via [`EncausticGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::encaustic`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct EncausticGenerator {
    config: EncausticConfig,
    glaze_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl EncausticGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: EncausticConfig) -> Self {
        let glaze_fbm: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(4);
        let glaze_noise = ToroidalNoise::new(glaze_fbm, config.scale * 1.5);
        Self {
            config,
            glaze_noise,
        }
    }
}

impl TextureGenerator for EncausticGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Toroidal glaze FBM: low-frequency surface waviness from hand-firing.
        let glaze_grid = sample_grid(&self.glaze_noise, width, height);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // Round scale to an integer so the grid tiles exactly.
        let scale_f = c.scale.round().max(1.0);
        let grout_half = (c.grout_width * 0.5).clamp(0.0, 0.49);

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;

            for x in 0..w {
                let u = x as f64 / w as f64;
                let idx = y * w + x;
                let ai = idx * 4;

                // Cell coordinates.
                let cell_u = u * scale_f;
                let cell_v = v * scale_f;
                let ci = cell_u.floor() as i64;
                let cj = cell_v.floor() as i64;
                // Local offset centered in [-0.5, 0.5].
                let cx = cell_u.fract() - 0.5;
                let cy = cell_v.fract() - 0.5;

                // Glaze variation in [0, 1].
                let glaze = normalize(glaze_grid[idx]);

                // Classify pixel using the selected pattern.
                let region = classify(&c.pattern, cx, cy, ci, cj, grout_half);

                let (r, g, b, h_val, rough_val) = match region {
                    Region::TileA => {
                        let perturb = (glaze - 0.5) * c.glaze_roughness * 0.6;
                        let r = (c.color_a[0] + perturb as f32).clamp(0.0, 1.0);
                        let g = (c.color_a[1] + perturb as f32).clamp(0.0, 1.0);
                        let b = (c.color_a[2] + perturb as f32).clamp(0.0, 1.0);
                        let hv = 0.85 + glaze * 0.15;
                        let rv = 0.20 + glaze * 0.05;
                        (r, g, b, hv, rv)
                    }
                    Region::TileB => {
                        let perturb = (glaze - 0.5) * c.glaze_roughness * 0.6;
                        let r = (c.color_b[0] + perturb as f32).clamp(0.0, 1.0);
                        let g = (c.color_b[1] + perturb as f32).clamp(0.0, 1.0);
                        let b = (c.color_b[2] + perturb as f32).clamp(0.0, 1.0);
                        let hv = 0.85 + glaze * 0.15;
                        let rv = 0.20 + glaze * 0.05;
                        (r, g, b, hv, rv)
                    }
                    Region::Grout => {
                        let r = c.color_grout[0];
                        let g = c.color_grout[1];
                        let b = c.color_grout[2];
                        (r, g, b, 0.0_f64, 0.85_f64)
                    }
                };

                heights[idx] = h_val;

                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                roughness_buf[ai] = 255;
                roughness_buf[ai + 1] = (rough_val * 255.0).round() as u8;
                roughness_buf[ai + 2] = 0; // ceramic is non-metallic
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

// --- Pattern classification -------------------------------------------------

/// Which region of a cell a pixel belongs to.
enum Region {
    TileA,
    TileB,
    Grout,
}

/// Classify `(cx, cy)` — cell-local position in `[-0.5, 0.5]` — into a
/// [`Region`] for the given pattern.
fn classify(
    pattern: &EncausticPattern,
    cx: f64,
    cy: f64,
    ci: i64,
    cj: i64,
    grout_half: f64,
) -> Region {
    match pattern {
        EncausticPattern::Checkerboard => classify_checkerboard(cx, cy, ci, cj, grout_half),
        EncausticPattern::Octagon => classify_octagon(cx, cy, grout_half),
        EncausticPattern::Diamond => classify_diamond(cx, cy, grout_half),
    }
}

/// Axis-aligned checkerboard: cell `(ci+cj) % 2 == 0` → TileA, else TileB.
/// A grout band of width `grout_half` runs along all four edges.
fn classify_checkerboard(cx: f64, cy: f64, ci: i64, cj: i64, grout_half: f64) -> Region {
    // Grout: near any cell edge.
    if cx.abs() > 0.5 - grout_half || cy.abs() > 0.5 - grout_half {
        return Region::Grout;
    }
    if (ci + cj).rem_euclid(2) == 0 {
        Region::TileA
    } else {
        Region::TileB
    }
}

/// Octagon + small corner square.
///
/// Within each cell the central octagon (colour A) is obtained by cutting the
/// corners of the square at 45°.  The four corner triangles where four
/// adjacent octagons share a vertex become small squares (colour B).
/// Everything else is grout.
fn classify_octagon(cx: f64, cy: f64, grout_half: f64) -> Region {
    // Amount of corner clipping expressed as a fraction of the half-cell.
    // Must be large enough to see the corner square but leave a clear octagon.
    let cut = 0.22_f64; // fraction of 0.5

    // Absolute distances from the cell center.
    let ax = cx.abs();
    let ay = cy.abs();

    // Outer grout border (near the cell edge on either axis).
    let in_grout_border = ax > 0.5 - grout_half || ay > 0.5 - grout_half;

    // Corner region: both |cx| and |cy| exceed (0.5 - cut - grout_half).
    let corner_threshold = 0.5 - cut - grout_half;
    let in_corner = ax > corner_threshold && ay > corner_threshold;

    // Octagon interior: inside the cell edge minus grout, with corners removed.
    // The 45° cut is along |cx| + |cy| < constant.
    let oct_inner = 0.5 - grout_half; // square half-extent before corner cut
    let in_oct_square = ax < oct_inner && ay < oct_inner;
    // Corner cut: remove points where ax + ay > oct_inner + (oct_inner - cut)
    // so the diagonal face sits at the cut distance.
    let diagonal_limit = (0.5 - grout_half) + (0.5 - grout_half - cut);
    let in_oct = in_oct_square && (ax + ay) < diagonal_limit && !in_corner;

    if in_grout_border {
        Region::Grout
    } else if in_corner {
        // Narrow grout gap between corner square and octagon diagonal.
        // The corner square is only the region inside the border grout AND
        // inside the corner threshold — check it is not adjacent to the
        // diagonal face of the octagon (which would be grout).
        // Since in_corner implies ax + ay is large, we always put colour B
        // here; the thin grout diagonal is handled by !in_oct above.
        Region::TileB
    } else if in_oct {
        Region::TileA
    } else {
        // Diagonal grout band between octagon face and corner square.
        Region::Grout
    }
}

/// Rotated 45° diamond grid: a diamond of colour A centred in each cell,
/// grout between diamonds.  A second, smaller diamond (colour B) is not
/// used here — the pattern is a single diamond per cell.
fn classify_diamond(cx: f64, cy: f64, grout_half: f64) -> Region {
    // Rotate 45°: new axes aligned with the diamond diagonals.
    use std::f64::consts::FRAC_1_SQRT_2;
    const INV_SQRT2: f64 = FRAC_1_SQRT_2;
    let rx = (cx + cy) * INV_SQRT2;
    let ry = (cx - cy) * INV_SQRT2;

    // The rotated cell has half-extents 1/sqrt(2) * 0.5 ≈ 0.354 in each axis.
    // Scale so the diamond fills the cell cleanly.
    let half = 0.5 * INV_SQRT2;
    let grout_r = grout_half * INV_SQRT2;

    if rx.abs() < half - grout_r && ry.abs() < half - grout_r {
        Region::TileA
    } else {
        Region::Grout
    }
}
