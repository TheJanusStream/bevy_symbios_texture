//! Iron grille / portcullis alpha-card texture generator.
//!
//! The algorithm:
//! 1. For each pixel compute the signed distance to every vertical bar and
//!    every horizontal bar.  Keep the minimum (nearest) distance for each axis.
//! 2. A pixel is "on a bar" when the combined SDF (min of the two axes) is ≤ 0.
//! 3. When a pixel lies inside both a vertical and a horizontal bar it is a
//!    joint/intersection — rust accumulates more heavily there.
//! 4. A non-toroidal FBM provides the rust distribution noise.
//! 5. When `round_bars` is true the height profile follows a circular cross-
//!    section (`(1 – |d/r|)²`) so the bars look cylindrical in the normal map.
//! 6. Transparent pixels (alpha = 0) get `dilate_heights` applied before
//!    `height_to_normal` with `BoundaryMode::Clamp` to avoid silhouette cliffs.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    normal::{BoundaryMode, dilate_heights, height_to_normal},
};

/// Configures the appearance of an [`IronGrilleGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IronGrilleConfig {
    pub seed: u32,
    /// Number of vertical bars \[2, 10\].
    pub bars_x: usize,
    /// Number of horizontal bars \[2, 10\].
    pub bars_y: usize,
    /// Bar half-width as fraction of card \[0.02, 0.20\].
    pub bar_width: f64,
    /// Round the bar cross-section (`true`) vs rectangular (`false`).
    pub round_bars: bool,
    /// Rust accumulation at joints \[0, 1\].
    pub rust_level: f64,
    /// Iron colour in linear RGB \[0, 1\].
    pub color_iron: [f32; 3],
    /// Rust colour in linear RGB \[0, 1\].
    pub color_rust: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for IronGrilleConfig {
    fn default() -> Self {
        Self {
            seed: 71,
            bars_x: 4,
            bars_y: 6,
            bar_width: 0.04,
            round_bars: true,
            rust_level: 0.30,
            color_iron: [0.14, 0.13, 0.13],
            color_rust: [0.42, 0.22, 0.08],
            normal_strength: 3.5,
        }
    }
}

/// Procedural iron grille / portcullis texture generator (alpha-card type).
///
/// The result has per-pixel alpha: fully transparent in the openings between
/// bars, and fully opaque on the bars themselves.  Upload via
/// [`crate::generator::map_to_images_card`] to select the clamp-to-edge
/// sampler automatically.
pub struct IronGrilleGenerator {
    config: IronGrilleConfig,
}

impl IronGrilleGenerator {
    pub fn new(config: IronGrilleConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for IronGrilleGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Non-toroidal FBM for rust distribution — card texture, no tiling needed.
        let rust_fbm: Fbm<Perlin> = Fbm::new(c.seed).set_octaves(4);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        let bars_x = c.bars_x.max(1);
        let bars_y = c.bars_y.max(1);

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;
            // Center in [-0.5, 0.5].
            let py = v - 0.5;

            for x in 0..w {
                let u = x as f64 / w as f64;
                let px = u - 0.5;

                let idx = y * w + x;
                let ai = idx * 4;

                // Compute minimum SDF to any vertical bar.
                let mut min_v_sdf = f64::MAX;
                for i in 0..bars_x {
                    let x_center = (i as f64 + 0.5) / bars_x as f64 - 0.5;
                    let d = (px - x_center).abs() - c.bar_width;
                    if d < min_v_sdf {
                        min_v_sdf = d;
                    }
                }

                // Compute minimum SDF to any horizontal bar.
                let mut min_h_sdf = f64::MAX;
                for j in 0..bars_y {
                    let y_center = (j as f64 + 0.5) / bars_y as f64 - 0.5;
                    let d = (py - y_center).abs() - c.bar_width;
                    if d < min_h_sdf {
                        min_h_sdf = d;
                    }
                }

                // Combined: a pixel is on the grille when it's inside either axis.
                let bar_sdf = min_v_sdf.min(min_h_sdf);
                let is_bar = bar_sdf <= 0.0;

                if !is_bar {
                    // Fully transparent opening.
                    heights[idx] = 0.0;
                    albedo[ai] = 0;
                    albedo[ai + 1] = 0;
                    albedo[ai + 2] = 0;
                    albedo[ai + 3] = 0;
                    // ORM for transparent pixels uses a neutral sentinel.
                    roughness_buf[ai] = 255;
                    roughness_buf[ai + 1] = 255;
                    roughness_buf[ai + 2] = 0;
                    roughness_buf[ai + 3] = 255;
                    continue;
                }

                // Joint: pixel is inside both a vertical and a horizontal bar.
                let is_joint = min_v_sdf <= 0.0 && min_h_sdf <= 0.0;

                // Rust noise — sample at medium frequency so it blotches naturally.
                let rust_raw = rust_fbm.get([u * 8.0, v * 8.0]) * 0.5 + 0.5; // [0, 1]
                let joint_boost = if is_joint { 1.5_f64 } else { 0.5_f64 };
                let rust_t = (rust_raw * c.rust_level * joint_boost).clamp(0.0, 1.0);

                // Height: round cross-section profile when enabled.
                let h_val = if c.round_bars {
                    // bar_sdf is negative inside; its magnitude / bar_width gives
                    // normalised depth from the bar surface inward.
                    let depth = (bar_sdf.abs() / c.bar_width).clamp(0.0, 1.0);
                    (1.0 - depth).powi(2)
                } else {
                    1.0
                };

                // Joints are slightly proud of the bar surface.
                let joint_bump = if is_joint { 0.15_f64 } else { 0.0_f64 };
                heights[idx] = (h_val + joint_bump).clamp(0.0, 1.0);

                // Colour: lerp iron → rust driven by rust_t.
                let r = lerp(c.color_iron[0], c.color_rust[0], rust_t as f32);
                let g = lerp(c.color_iron[1], c.color_rust[1], rust_t as f32);
                let b = lerp(c.color_iron[2], c.color_rust[2], rust_t as f32);

                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // ORM: iron is metallic; rust degrades both smoothness and metallicness.
                let rough = (0.35 + rust_t * 0.40) as f32;
                let metallic = (0.85 - rust_t * 0.50).max(0.0) as f32;
                roughness_buf[ai] = 255;
                roughness_buf[ai + 1] = (rough * 255.0).round() as u8;
                roughness_buf[ai + 2] = (metallic * 255.0).round() as u8;
                roughness_buf[ai + 3] = 255;
            }
        }

        // Dilate bar heights one step into transparent neighbours so the normal
        // map has no hard silhouette cliff at the bar edges.
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

// --- helpers ----------------------------------------------------------------

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}
