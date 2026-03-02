//! Wainscoting / wood-paneling texture generator.
//!
//! The algorithm:
//! 1. Precompute a toroidal grain FBM grid and a warp FBM grid over the whole
//!    surface.  Both use `sample_grid` so torus coordinates are computed once
//!    per row/column rather than once per pixel.
//! 2. For each pixel, apply domain warp to the grain UV and bilinearly sample
//!    the grain grid on the torus.
//! 3. Determine which panel cell the pixel falls in, then classify it as frame
//!    band, bevel transition, or recessed panel face using a simple margin
//!    test.  This produces the structural height field.
//! 4. The final height field adds a small fraction of grain micro-detail on top
//!    of the structural height so the wood grain shows subtle surface relief.
//! 5. Colour is a lerp between dark and light wood, driven purely by grain.
//! 6. ORM: roughness varies slightly with grain; metallic is always 0.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Configures the appearance of a [`WainscotingGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WainscotingConfig {
    pub seed: u32,
    /// Horizontal panel divisions \[1, 4\].
    pub panels_x: usize,
    /// Vertical panel divisions \[1, 4\].
    pub panels_y: usize,
    /// Rail/stile (frame member) width as fraction of panel cell \[0.05, 0.35\].
    pub frame_width: f64,
    /// Panel inset depth: how recessed the central panel face is \[0, 0.15\].
    pub panel_inset: f64,
    /// Wood grain spatial frequency \[4, 24\].
    pub grain_scale: f64,
    /// Grain domain-warp strength \[0, 0.8\].
    pub grain_warp: f64,
    /// Light wood colour in linear RGB \[0, 1\].
    pub color_wood_light: [f32; 3],
    /// Dark grain colour in linear RGB \[0, 1\].
    pub color_wood_dark: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for WainscotingConfig {
    fn default() -> Self {
        Self {
            seed: 37,
            panels_x: 1,
            panels_y: 2,
            frame_width: 0.20,
            panel_inset: 0.06,
            grain_scale: 10.0,
            grain_warp: 0.30,
            color_wood_light: [0.65, 0.44, 0.20],
            color_wood_dark: [0.28, 0.16, 0.07],
            normal_strength: 4.0,
        }
    }
}

/// Procedural wainscoting / wood-paneling texture generator.
///
/// Produces a tileable albedo, normal, and ORM map.  Upload via
/// [`crate::generator::map_to_images`] for repeat-wrapping samplers.
pub struct WainscotingGenerator {
    config: WainscotingConfig,
}

impl WainscotingGenerator {
    pub fn new(config: WainscotingConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for WainscotingGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Grain FBM: anisotropic-ish, high frequency across the grain direction.
        let grain_fbm: Fbm<Perlin> = Fbm::new(c.seed).set_octaves(5);
        let grain_noise = ToroidalNoise::new(grain_fbm, c.grain_scale);
        let grain_grid = sample_grid(&grain_noise, width, height);

        // Warp FBM: low frequency, used to domain-warp the grain UV.
        let warp_fbm: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(77)).set_octaves(3);
        let warp_noise = ToroidalNoise::new(warp_fbm, c.grain_scale * 0.3);
        let warp_grid = sample_grid(&warp_noise, width, height);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        let panels_x = c.panels_x.max(1);
        let panels_y = c.panels_y.max(1);
        let fw = (c.frame_width * 0.5).clamp(0.0, 0.48);
        let panel_hx = 0.5 - fw;
        let panel_hy = 0.5 - fw;
        let bevel_w = fw * 0.3;

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;

            for x in 0..w {
                let u = x as f64 / w as f64;
                let idx = y * w + x;

                // Domain warp: nudge U coordinate by warp FBM to bend grain lines.
                let warp_u = normalize(warp_grid[idx]) - 0.5; // [-0.5, 0.5]
                let warped_u = (u + warp_u * c.grain_warp * 0.1).rem_euclid(1.0);

                // Bilinearly sample grain grid at warped position.
                let grain_raw = bilinear_sample_torus(&grain_grid, w, h, warped_u, v);
                let grain_t = normalize(grain_raw); // [0, 1]

                // Panel SDF classification.
                // Local position within the cell, centered in [-0.5, 0.5].
                let cell_u = (u * panels_x as f64).fract();
                let cell_v = (v * panels_y as f64).fract();
                let cx = cell_u - 0.5;
                let cy = cell_v - 0.5;

                // Distance from panel interior (positive = inside panel, away from frame).
                let dist_to_frame_u = panel_hx - cx.abs();
                let dist_to_frame_v = panel_hy - cy.abs();
                let dist_inside = dist_to_frame_u.min(dist_to_frame_v);

                let panel_height = if dist_inside < 0.0 {
                    // Frame band — highest surface.
                    1.0_f64
                } else if dist_inside < bevel_w {
                    // Bevel ramp from frame height down to recessed panel face.
                    1.0 - (dist_inside / bevel_w) * c.panel_inset
                } else {
                    // Recessed panel face.
                    1.0 - c.panel_inset
                };

                // Final height: structural panel height + tiny grain micro-detail.
                heights[idx] = (panel_height + grain_t * 0.05).clamp(0.0, 1.0);

                // Colour: dark-to-light lerp driven by grain.
                let r = lerp(c.color_wood_dark[0], c.color_wood_light[0], grain_t as f32);
                let g = lerp(c.color_wood_dark[1], c.color_wood_light[1], grain_t as f32);
                let b = lerp(c.color_wood_dark[2], c.color_wood_light[2], grain_t as f32);

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // ORM: slightly rougher in the dark grain trenches.
                let rough = 0.75 + grain_t * 0.1;
                roughness_buf[ai] = 255;
                roughness_buf[ai + 1] = (rough * 255.0).round() as u8;
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

/// Toroidal bilinear sample of a precomputed grid.
///
/// `u` and `v` are in `[0, 1]` and wrap at the edges.  The four nearest grid
/// pixels are blended with standard bilinear weights.
fn bilinear_sample_torus(grid: &[f64], w: usize, h: usize, u: f64, v: f64) -> f64 {
    let u = u.rem_euclid(1.0);
    let v = v.rem_euclid(1.0);
    let px = u * w as f64;
    let py = v * h as f64;
    let x0 = px as usize % w;
    let y0 = py as usize % h;
    let x1 = (x0 + 1) % w;
    let y1 = (y0 + 1) % h;
    let fx = px.fract();
    let fy = py.fract();
    let v00 = grid[y0 * w + x0];
    let v10 = grid[y0 * w + x1];
    let v01 = grid[y1 * w + x0];
    let v11 = grid[y1 * w + x1];
    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}
