//! Window texture generator using 2-D signed distance functions (SDF).
//!
//! The algorithm:
//! 1. Compute a plain rectangular outer silhouette (full card) and a
//!    rounded-box SDF for the inner glass opening.
//! 2. Classify each pixel as frame, mullion, or glass.
//! 3. Subdivide the glass region into `panes_x × panes_y` panes separated by
//!    mullions using fractional UV within the inner glass area.
//! 4. Add FBM grime noise to the glass surface and roughness map.
//! 5. Produce an alpha-masked card (clamp-to-edge sampler via `map_to_images_card`).

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    normal::{BoundaryMode, dilate_heights, height_to_normal},
};

/// Configures the appearance of a [`WindowGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WindowConfig {
    pub seed: u32,
    /// Frame width as a fraction of the card \[0, 0.4\].
    pub frame_width: f64,
    /// Number of panes in the horizontal direction.
    pub panes_x: usize,
    /// Number of panes in the vertical direction.
    pub panes_y: usize,
    /// Mullion/muntin thickness as a fraction of the glass area \[0, 0.2\].
    pub mullion_thickness: f64,
    /// Inner (glass-opening) corner-rounding radius as a fraction of the card \[0, 0.4\].
    /// The outer silhouette is always a plain rectangle.
    pub corner_radius: f64,
    /// Glass opacity \[0 = clear, 1 = frosted/opaque\].
    pub glass_opacity: f64,
    /// Grime/dirt noise intensity on glass \[0, 1\].
    pub grime_level: f64,
    /// Frame and mullion colour in linear RGB \[0, 1\].
    pub color_frame: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            frame_width: 0.08,
            panes_x: 2,
            panes_y: 3,
            mullion_thickness: 0.025,
            corner_radius: 0.02,
            glass_opacity: 0.30,
            grime_level: 0.15,
            color_frame: [0.85, 0.82, 0.78],
            normal_strength: 3.0,
        }
    }
}

/// Procedural window / glazing texture generator (foliage-card type).
///
/// The result has per-pixel alpha: transparent outside the frame, semi-transparent
/// glass, and opaque frame/mullions.  Upload via
/// [`crate::async_gen::PendingTexture::window`] which selects the
/// clamp-to-edge sampler automatically.
pub struct WindowGenerator {
    config: WindowConfig,
}

impl WindowGenerator {
    pub fn new(config: WindowConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for WindowGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // FBM for glass grime — no tiling needed (card texture).
        let grime_fbm: Fbm<Perlin> = Fbm::new(c.seed).set_octaves(6);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // Inner half-extent = outer half-extent (0.5) minus frame.
        let inner_half = 0.5 - c.frame_width;
        // Inner corner radius: keep at least a small value so the SDF is well-formed.
        // The outer silhouette is a plain rectangle — only the inner opening is rounded.
        let inner_r = c.corner_radius.min(inner_half * 0.9).max(0.005);

        // Glass area extent in UV for pane subdivision.
        let glass_span = inner_half * 2.0; // UV span of glass region (centered)
        let glass_origin = 0.5 - inner_half; // UV origin of glass region

        let mullion_half = (c.mullion_thickness * 0.5).min(0.49);
        let panes_x = c.panes_x.max(1);
        let panes_y = c.panes_y.max(1);

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;
            let py = v - 0.5; // centered in [-0.5, 0.5]

            for x in 0..w {
                let u = x as f64 / w as f64;
                let px = u - 0.5;

                // The outer silhouette (r=0, half-extents 0.5) covers every pixel
                // of the card: px,py ∈ [-0.5,0.5] so outer_sdf ≤ 0 always.
                let inner_sdf =
                    sdf_rounded_box(px, py, inner_half - inner_r, inner_half - inner_r, inner_r);

                let idx = y * w + x;
                let ai = idx * 4;

                if inner_sdf > 0.0 {
                    // Frame band between outer and inner SDF.
                    // Height ramps from 0 at the outer edge inward to 1 deep in the frame.
                    // Distance from the nearest outer edge: min(0.5-|px|, 0.5-|py|).
                    let edge_dist = (0.5 - px.abs()).min(0.5 - py.abs());
                    let edge_t = (edge_dist / (c.frame_width + 0.005)).clamp(0.0, 1.0);
                    heights[idx] = edge_t;

                    albedo[ai] = linear_to_srgb(c.color_frame[0]);
                    albedo[ai + 1] = linear_to_srgb(c.color_frame[1]);
                    albedo[ai + 2] = linear_to_srgb(c.color_frame[2]);
                    albedo[ai + 3] = 255;

                    roughness_buf[ai] = 255;
                    roughness_buf[ai + 1] = (0.75 * 255.0) as u8; // rough wood/paint
                    roughness_buf[ai + 2] = 0;
                    roughness_buf[ai + 3] = 255;
                } else {
                    // Inside glass area.  Check for mullions.
                    // Map pixel to glass-local UV in [0, 1].
                    let gu = ((u - glass_origin) / glass_span).clamp(0.0, 1.0);
                    let gv = ((v - glass_origin) / glass_span).clamp(0.0, 1.0);

                    // Fractional position within each pane.
                    let pu = (gu * panes_x as f64).fract();
                    let pv = (gv * panes_y as f64).fract();

                    // Check if we're on an internal mullion line.
                    // The outer boundary lines coincide with the frame so panes_x=1 never
                    // produces spurious mullions inside the glass.
                    let is_mullion_x = panes_x > 1
                        && (pu < mullion_half
                            || pu > 1.0 - mullion_half
                            || gu < mullion_half / panes_x as f64
                            || gu > 1.0 - mullion_half / panes_x as f64);
                    let is_mullion_y = panes_y > 1
                        && (pv < mullion_half
                            || pv > 1.0 - mullion_half
                            || gv < mullion_half / panes_y as f64
                            || gv > 1.0 - mullion_half / panes_y as f64);

                    // For single-pane axis, still suppress the outer boundary band.
                    let at_x_border = gu < mullion_half || gu > 1.0 - mullion_half;
                    let at_y_border = gv < mullion_half || gv > 1.0 - mullion_half;

                    if is_mullion_x || is_mullion_y || at_x_border || at_y_border {
                        // Mullion — treat like frame.
                        heights[idx] = 1.0;

                        albedo[ai] = linear_to_srgb(c.color_frame[0]);
                        albedo[ai + 1] = linear_to_srgb(c.color_frame[1]);
                        albedo[ai + 2] = linear_to_srgb(c.color_frame[2]);
                        albedo[ai + 3] = 255;

                        roughness_buf[ai] = 255;
                        roughness_buf[ai + 1] = (0.75 * 255.0) as u8;
                        roughness_buf[ai + 2] = 0;
                        roughness_buf[ai + 3] = 255;
                    } else {
                        // Glass pane.
                        let grime_raw = grime_fbm.get([u * 8.0, v * 8.0]) * 0.5 + 0.5;
                        let grime = grime_raw * c.grime_level;

                        heights[idx] = grime * 0.08;

                        // Light blue-grey glass tint, darkened slightly by grime.
                        let gr = (0.82 - grime as f32 * 0.3).clamp(0.0, 1.0);
                        let gg = (0.88 - grime as f32 * 0.2).clamp(0.0, 1.0);
                        let gb = (0.93 - grime as f32 * 0.1).clamp(0.0, 1.0);
                        let alpha = (c.glass_opacity * 255.0).round() as u8;

                        albedo[ai] = linear_to_srgb(gr);
                        albedo[ai + 1] = linear_to_srgb(gg);
                        albedo[ai + 2] = linear_to_srgb(gb);
                        albedo[ai + 3] = alpha;

                        // Glass: low roughness, high metallic (simulates reflection).
                        let glass_rough = (0.05 + grime as f32 * 0.3).clamp(0.0, 1.0);
                        roughness_buf[ai] = 255;
                        roughness_buf[ai + 1] = (glass_rough * 255.0).round() as u8;
                        roughness_buf[ai + 2] = (0.85 * 255.0) as u8; // metallic
                        roughness_buf[ai + 3] = 255;
                    }
                }
            }
        }

        // Fill transparent pixels' heights from opaque neighbours so the normal
        // map doesn't produce hard silhouette cliffs.
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

/// Signed distance to a rounded rectangle centred at the origin.
/// Negative inside, positive outside.
/// `bx`, `by` are inner half-extents (before rounding); `r` is the corner radius.
#[inline]
fn sdf_rounded_box(px: f64, py: f64, bx: f64, by: f64, r: f64) -> f64 {
    let dx = px.abs() - bx;
    let dy = py.abs() - by;
    let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
    let inside = dx.max(dy).min(0.0);
    outside + inside - r
}
