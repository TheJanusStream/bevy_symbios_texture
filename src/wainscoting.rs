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
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, bilinear_sample_torus, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Configures the appearance of a [`WainscotingGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WainscotingConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
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
/// Drives [`TextureGenerator::generate`] using a [`WainscotingConfig`].  Construct
/// via [`WainscotingGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::wainscoting`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct WainscotingGenerator {
    config: WainscotingConfig,
    grain_noise: ToroidalNoise<Fbm<Perlin>>,
    warp_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl WainscotingGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: WainscotingConfig) -> Self {
        let grain_fbm: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(5);
        let grain_noise = ToroidalNoise::new(grain_fbm, config.grain_scale);
        let warp_fbm: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(77)).set_octaves(3);
        let warp_noise = ToroidalNoise::new(warp_fbm, config.grain_scale * 0.3);
        Self {
            config,
            grain_noise,
            warp_noise,
        }
    }
}

/// Per-generation sampler: grain + warp grids and derived panel-layout
/// constants.
struct WainscotingCell<'a> {
    config: &'a WainscotingConfig,
    grain_grid: &'a [f64],
    warp_grid: &'a [f64],
    panels_x: usize,
    panels_y: usize,
    /// Half panel extents (cell-local) and bevel band width derived from
    /// `frame_width`.
    panel_hx: f64,
    panel_hy: f64,
    bevel_w: f64,
    w: usize,
    h: usize,
}

impl SurfaceCell for WainscotingCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let idx = y as usize * self.w + x as usize;

        // Domain warp: nudge U coordinate by warp FBM to bend grain lines.
        let warp_u = normalize(self.warp_grid[idx]) - 0.5; // [-0.5, 0.5]
        let warped_u = (u + warp_u * c.grain_warp * 0.1).rem_euclid(1.0);

        // Bilinearly sample grain grid at warped position.
        let grain_raw = bilinear_sample_torus(self.grain_grid, self.w, self.h, warped_u, v);
        let grain_t = normalize(grain_raw); // [0, 1]

        // Panel SDF classification.
        // Local position within the cell, centered in [-0.5, 0.5].
        let cell_u = (u * self.panels_x as f64).fract();
        let cell_v = (v * self.panels_y as f64).fract();
        let cx = cell_u - 0.5;
        let cy = cell_v - 0.5;

        // Distance from panel interior (positive = inside panel, away from frame).
        let dist_to_frame_u = self.panel_hx - cx.abs();
        let dist_to_frame_v = self.panel_hy - cy.abs();
        let dist_inside = dist_to_frame_u.min(dist_to_frame_v);

        let panel_height = if dist_inside < 0.0 {
            // Frame band — highest surface.
            1.0_f64
        } else if dist_inside < self.bevel_w {
            // Bevel ramp from frame height down to recessed panel face.
            1.0 - (dist_inside / self.bevel_w) * c.panel_inset
        } else {
            // Recessed panel face.
            1.0 - c.panel_inset
        };

        // Colour: dark-to-light lerp driven by grain.
        let color = [
            lerp(c.color_wood_dark[0], c.color_wood_light[0], grain_t as f32),
            lerp(c.color_wood_dark[1], c.color_wood_light[1], grain_t as f32),
            lerp(c.color_wood_dark[2], c.color_wood_light[2], grain_t as f32),
        ];

        // ORM: slightly rougher in the dark grain trenches.
        let rough = (0.75 + grain_t * 0.1) as f32;

        // Final height: structural panel height + tiny grain micro-detail.
        SurfaceSample::matte(
            (panel_height + grain_t * 0.05).clamp(0.0, 1.0),
            color,
            rough,
        )
    }
}

impl WainscotingGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Grain FBM: anisotropic-ish, high frequency across the grain direction.
        let mut grain_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.grain_noise, width, height, &mut grain_grid);

        // Warp FBM: low frequency, used to domain-warp the grain UV.
        let mut warp_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.warp_noise, width, height, &mut warp_grid);

        let fw = (c.frame_width * 0.5).clamp(0.0, 0.48);
        let cell = WainscotingCell {
            config: c,
            grain_grid: &grain_grid,
            warp_grid: &warp_grid,
            panels_x: c.panels_x.max(1),
            panels_y: c.panels_y.max(1),
            panel_hx: 0.5 - fw,
            panel_hy: 0.5 - fw,
            bevel_w: fw * 0.3,
            w: width as usize,
            h: height as usize,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        if let Some(ws) = ws {
            ws.return_grid(grain_grid);
            ws.return_grid(warp_grid);
        }
        result
    }
}

impl TextureGenerator for WainscotingGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        self.generate_inner(width, height, None)
    }

    fn generate_with_workspace(
        &self,
        width: u32,
        height: u32,
        workspace: &mut Workspace,
    ) -> Result<TextureMap, TextureError> {
        self.generate_inner(width, height, Some(workspace))
    }
}
