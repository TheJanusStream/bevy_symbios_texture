//! Cobblestone texture generator using Voronoi cell decomposition.
//!
//! The algorithm:
//! 1. Compute a toroidally-wrapped Voronoi diagram (F1 and F2 distances plus
//!    the integer cell ID of the nearest site).
//! 2. Pixels whose `F2 – F1` is below `gap_threshold` lie on a cell boundary
//!    and are rendered as mud/dirt.
//! 3. Stone pixels receive a domed height profile (`1 – (F1·scale)^roundness`)
//!    blended with a toroidal FBM for micro-surface detail.
//! 4. Per-stone colour variance is driven by the integer cell hash of the
//!    nearest Voronoi site.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, cell_hash, normalize, sample_grid_into, toroidal_voronoi},
    surface::{SurfaceCell, SurfaceSample, generate_surface},
};

/// Configures the appearance of a [`CobblestoneGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CobblestoneConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Approximate number of stones across the tile \[3, 12\].
    pub scale: f64,
    /// Mud gap threshold as a fraction of stone spacing \[0.02, 0.25\].
    pub gap_width: f64,
    /// Per-stone colour jitter \[0, 1\].  `0.0` = uniform colour.
    pub cell_variance: f64,
    /// Stone roundness — controls how domed the tops of the stones are \[0.5, 2.0\].
    /// Higher values produce flatter stones with steeper sides.
    pub roundness: f64,
    /// Stone colour in linear RGB \[0, 1\].
    pub color_stone: [f32; 3],
    /// Mud / dirt gap colour in linear RGB \[0, 1\].
    pub color_mud: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for CobblestoneConfig {
    fn default() -> Self {
        Self {
            seed: 7,
            scale: 6.0,
            gap_width: 0.12,
            cell_variance: 0.20,
            roundness: 1.2,
            color_stone: [0.46, 0.43, 0.40],
            color_mud: [0.22, 0.18, 0.14],
            normal_strength: 5.0,
        }
    }
}

/// Procedural cobblestone texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`CobblestoneConfig`].  Construct
/// via [`CobblestoneGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::cobblestone`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct CobblestoneGenerator {
    config: CobblestoneConfig,
    surf_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl CobblestoneGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: CobblestoneConfig) -> Self {
        let fbm_surf: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(50)).set_octaves(4);
        let surf_noise = ToroidalNoise::new(fbm_surf, config.scale * 2.5);
        Self { config, surf_noise }
    }
}

/// Per-generation sampler: surface grid + Voronoi layout constants.
struct CobblestoneCell<'a> {
    config: &'a CobblestoneConfig,
    surf_grid: &'a [f64],
    /// `scale` rounded and clamped ≥ 1 so the Voronoi lattice tiles.
    scale: f64,
    /// Gap threshold in UV distance units: stones end where the Voronoi
    /// boundary is closer than this value.
    gap_threshold: f64,
    width: usize,
}

impl SurfaceCell for CobblestoneCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;
        let idx = y as usize * self.width + x as usize;
        let raw_surf = normalize(self.surf_grid[idx]);

        // Grid-based toroidal Voronoi: returns F1, F2, and the integer
        // cell coordinates of the nearest site.
        let (f1, f2, ci, cj) = toroidal_voronoi(u, v, self.scale, c.seed);

        let in_gap = f2 - f1 < self.gap_threshold;

        let (h_val, color) = if in_gap {
            // ── Mud / dirt gap ────────────────────────────────────────
            (raw_surf * 0.04, c.color_mud)
        } else {
            // ── Stone face ────────────────────────────────────────────
            // Dome profile: peaks at 1.0 directly over the Voronoi
            // site and falls toward 0.0 at the cell boundary.
            let dome_base = (1.0 - (f1 * c.scale).powf(c.roundness)).clamp(0.0, 1.0);
            // Blend FBM micro-detail into the height.
            let h_val = (dome_base * (0.85 + raw_surf * 0.15)).clamp(0.0, 1.0);

            // Per-stone colour jitter via cell hash.
            let cv = cell_hash(ci, cj, c.seed.wrapping_add(99));
            let jitter = (cv - 0.5) * 2.0 * c.cell_variance;
            let color = [
                (c.color_stone[0] + jitter as f32).clamp(0.0, 1.0),
                (c.color_stone[1] + jitter as f32 * 0.85).clamp(0.0, 1.0),
                (c.color_stone[2] + jitter as f32 * 0.65).clamp(0.0, 1.0),
            ];
            (h_val, color)
        };

        // ORM: stone roughness varies with dome height (higher = smoother),
        // mud is nearly fully rough.
        let rough_val = if in_gap {
            0.95
        } else {
            // Smoother at the crown, rougher toward the edges.
            (0.85 - h_val as f32 * 0.20 + raw_surf as f32 * 0.10).clamp(0.60, 0.90)
        };

        SurfaceSample::matte(h_val, color, rough_val)
    }
}

impl CobblestoneGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // Toroidal FBM for stone-surface micro-detail.
        let mut surf_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.surf_noise, width, height, &mut surf_grid);

        let scale = c.scale.round().max(1.0);
        let cell = CobblestoneCell {
            config: c,
            surf_grid: &surf_grid,
            scale,
            gap_threshold: c.gap_width / scale,
            width: width as usize,
        };
        let result = generate_surface(width, height, c.normal_strength, ws.as_deref_mut(), &cell);

        if let Some(ws) = ws {
            ws.return_grid(surf_grid);
        }
        result
    }
}

impl TextureGenerator for CobblestoneGenerator {
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
