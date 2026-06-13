//! Metal texture generator — brushed finish or standing-seam roof panels,
//! with optional rust weathering.
//!
//! The algorithm:
//! 1. **Brushed**: anisotropic FBM — high frequency in U (many scratches),
//!    very low frequency in V (scratches run nearly horizontally).
//! 2. **StandingSeam**: sinusoidal ridge profile across V, with micro-detail
//!    FBM overlay.
//! 3. A separate low-frequency FBM drives rust-patch blending: rust areas
//!    receive a warm colour, raised roughness, and reduced metallic value.

use std::f64::consts::TAU;

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid_into},
    surface::{SurfaceCell, SurfaceSample, generate_surface, lerp},
};

/// Visual style of the metal surface.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MetalStyle {
    /// Fine horizontal scratches (brushed / satin finish).
    Brushed,
    /// Parallel raised ridges running across the tile (standing-seam roof).
    StandingSeam,
    /// Peened surface of round overlapping dimple depressions (hand-hammered
    /// sheet).  `scale` sets the dimple count across the tile.
    Hammered,
    /// Classic tread plate: a diamond lattice of raised oblong studs with
    /// alternating orientation.  `scale` sets the stud count across the tile.
    DiamondPlate,
}

/// Configures the appearance of a [`MetalGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MetalConfig {
    /// PRNG seed for the deterministic noise pattern; different seeds give
    /// statistically-different textures from otherwise-identical configs.
    pub seed: u32,
    /// Surface finish style.
    pub style: MetalStyle,
    /// Base noise scale.
    pub scale: f64,
    /// For `StandingSeam`: number of ridges across the tile.
    pub seam_count: f64,
    /// Ridge sharpness for `StandingSeam` \[0.5 = sinusoidal, 4.0 = sharp\].
    pub seam_sharpness: f64,
    /// Anisotropy factor for `Brushed` — higher = longer horizontal scratches.
    pub brush_stretch: f64,
    /// Micro-roughness amplitude \[0, 1\].
    pub roughness: f64,
    /// Metallic value for clean (rust-free) areas \[0, 1\].
    pub metallic: f32,
    /// Rust-patch coverage \[0 = none, 1 = heavy\].
    pub rust_level: f64,
    /// Base metal colour in linear RGB \[0, 1\].
    pub color_metal: [f32; 3],
    /// Rust colour in linear RGB \[0, 1\].
    pub color_rust: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            seed: 31,
            style: MetalStyle::Brushed,
            scale: 6.0,
            seam_count: 6.0,
            seam_sharpness: 2.5,
            brush_stretch: 8.0,
            roughness: 0.25,
            metallic: 0.85,
            rust_level: 0.15,
            color_metal: [0.42, 0.44, 0.47],
            color_rust: [0.42, 0.24, 0.12],
            normal_strength: 3.0,
        }
    }
}

/// Procedural metal texture generator.
///
/// Drives [`TextureGenerator::generate`] using a [`MetalConfig`].  Construct
/// via [`MetalGenerator::new`] and call `generate` directly, or spawn a
/// [`crate::async_gen::PendingTexture::metal`] task for non-blocking generation.
///
/// Noise objects are built in the constructor so that calling `generate`
/// multiple times (e.g. producing size variants of the same material)
/// does not repeat the initialisation cost.
pub struct MetalGenerator {
    config: MetalConfig,
    fbm_scratch: Fbm<Perlin>,
    rust_noise: ToroidalNoise<Fbm<Perlin>>,
}

impl MetalGenerator {
    /// Create a new generator with the given configuration.
    ///
    /// Builds the noise objects up front so that repeated
    /// calls to [`generate`](TextureGenerator::generate) skip initialisation.
    pub fn new(config: MetalConfig) -> Self {
        let fbm_scratch: Fbm<Perlin> = Fbm::new(config.seed).set_octaves(5);

        let fbm_rust: Fbm<Perlin> = Fbm::new(config.seed.wrapping_add(41)).set_octaves(4);
        let rust_noise = ToroidalNoise::new(fbm_rust, config.scale * 0.4);

        Self {
            config,
            fbm_scratch,
            rust_noise,
        }
    }
}

/// Per-generation sampler: precomputed rust grid + per-pixel anisotropic
/// scratch noise (sampled analytically — the brushed style's stretched torus
/// coordinates have no grid equivalent).
struct MetalCell<'a> {
    config: &'a MetalConfig,
    fbm_scratch: &'a Fbm<Perlin>,
    rust_grid: &'a [f64],
    width: usize,
}

impl SurfaceCell for MetalCell<'_> {
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample {
        let c = self.config;

        // Standing-seam ridge profile (sinusoidal bumps in V).
        // seam_count must be an integer for the pattern to tile; round to nearest.
        let seam_count = c.seam_count.round();
        let seam_h = if c.style == MetalStyle::StandingSeam {
            let phase = (v * seam_count * TAU).sin();
            // Raise to power to sharpen; clamp to [0,1].
            phase.abs().powf(c.seam_sharpness.max(0.1)) * phase.signum() * 0.5 + 0.5
        } else {
            0.0
        };

        // Sample scratch noise.
        // Brushed: large radius in U (fast oscillations → many horizontal
        // scratches), small radius in V (slow → scratches run lengthwise).
        // Other styles: uniform toroidal sampling for micro-detail.
        let scratch = match c.style {
            MetalStyle::Brushed => {
                let nx = (TAU * u).cos() * c.scale * c.brush_stretch;
                let ny = (TAU * u).sin() * c.scale * c.brush_stretch;
                let nz = (TAU * v).cos() * c.scale * 0.12;
                let nw = (TAU * v).sin() * c.scale * 0.12;
                self.fbm_scratch.get([nx, ny, nz, nw]) * 0.5 + 0.5
            }
            MetalStyle::StandingSeam | MetalStyle::Hammered | MetalStyle::DiamondPlate => {
                let nx = (TAU * u).cos() * c.scale;
                let ny = (TAU * u).sin() * c.scale;
                let nz = (TAU * v).cos() * c.scale;
                let nw = (TAU * v).sin() * c.scale;
                self.fbm_scratch.get([nx, ny, nz, nw]) * 0.5 + 0.5
            }
        };

        let idx = y as usize * self.width + x as usize;
        let rust_t = normalize(self.rust_grid[idx]);
        // Soft threshold → rust coverage.
        let rust_blend = ((rust_t - (1.0 - c.rust_level)).clamp(0.0, c.rust_level)
            / c.rust_level.max(1e-9))
        .clamp(0.0, 1.0);

        let h_scratch = scratch * c.roughness * 0.3;
        let h_val = match c.style {
            MetalStyle::Brushed => h_scratch,
            MetalStyle::StandingSeam => seam_h * 0.7 + h_scratch * 0.3,
            MetalStyle::Hammered => {
                // Plateau between dimples at 1, dimple centres at 0.
                let dimple = dimple_height(u, v, c.scale, c.seed);
                (dimple * 0.8 + h_scratch * 0.2).clamp(0.0, 1.0)
            }
            MetalStyle::DiamondPlate => {
                (0.35 + diamond_stud(u, v, c.scale) * 0.5 + h_scratch * 0.3).clamp(0.0, 1.0)
            }
        };

        // Colour: lerp metal → rust.
        let color = [
            lerp(c.color_metal[0], c.color_rust[0], rust_blend as f32),
            lerp(c.color_metal[1], c.color_rust[1], rust_blend as f32),
            lerp(c.color_metal[2], c.color_rust[2], rust_blend as f32),
        ];

        // ORM: rust raises roughness and kills metallic.
        let rough = (c.roughness as f32 + rust_blend as f32 * 0.65).clamp(0.0, 1.0);
        let met = (c.metallic - rust_blend as f32 * 0.80).clamp(0.0, 1.0);

        SurfaceSample {
            height: h_val,
            color,
            roughness: rough,
            metallic: met,
            occlusion: 1.0,
            emissive: [0.0, 0.0, 0.0],
        }
    }
}

impl MetalGenerator {
    fn generate_inner(
        &self,
        width: u32,
        height: u32,
        mut ws: Option<&mut Workspace>,
    ) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;

        // Rust patches — separate seed, low frequency for large blotches.
        let mut rust_grid = ws.as_deref_mut().map_or_else(Vec::new, |w| w.take_grid());
        sample_grid_into(&self.rust_noise, width, height, &mut rust_grid);

        let cell = MetalCell {
            config: &self.config,
            fbm_scratch: &self.fbm_scratch,
            rust_grid: &rust_grid,
            width: width as usize,
        };
        let result = generate_surface(
            width,
            height,
            self.config.normal_strength,
            ws.as_deref_mut(),
            &cell,
        );

        if let Some(ws) = ws {
            ws.return_grid(rust_grid);
        }
        result
    }
}

impl TextureGenerator for MetalGenerator {
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

// --- style helpers ------------------------------------------------------------

/// Hammered dimple field: `1` on the plateau between dimples, falling to `0`
/// at each dimple centre.  A jittered toroidal point grid (one dimple per
/// cell, 3×3 neighbourhood search) keeps the pattern seamless.
fn dimple_height(u: f64, v: f64, count: f64, seed: u32) -> f64 {
    let n = count.round().max(1.0);
    let gi = (u * n).floor() as i64;
    let gj = (v * n).floor() as i64;

    let mut best = f64::MAX;
    for di in -1i64..=1 {
        for dj in -1i64..=1 {
            let ni = (gi + di).rem_euclid(n as i64);
            let nj = (gj + dj).rem_euclid(n as i64);
            // Keep centres away from cell borders so neighbouring dimples
            // overlap without leaving flat seams.
            let jx = 0.2 + 0.6 * cell_hash(ni, nj, seed);
            let jy = 0.2 + 0.6 * cell_hash(nj, ni, seed.wrapping_add(13));
            let cx = (ni as f64 + jx) / n;
            let cy = (nj as f64 + jy) / n;

            // Toroidal distance in cell units.
            let mut dx = (u - cx).abs();
            if dx > 0.5 {
                dx = 1.0 - dx;
            }
            let mut dy = (v - cy).abs();
            if dy > 0.5 {
                dy = 1.0 - dy;
            }
            best = best.min((dx * dx + dy * dy).sqrt() * n);
        }
    }

    // Spherical cap profile: depth 1 at the centre, flat beyond ~0.75 cells.
    let t = (best / 0.75).min(1.0);
    1.0 - (1.0 - t * t).max(0.0)
}

/// Diamond-plate stud field: `1` on a raised stud, `0` on the base plate.
///
/// The UV plane is sheared into diagonal lattice coordinates; each lattice
/// cell carries one rounded-bar stud whose orientation alternates with cell
/// parity.  Integer `scale` keeps both diagonal families tiling.
fn diamond_stud(u: f64, v: f64, scale: f64) -> f64 {
    let k = scale.round().max(1.0);
    let s = (u + v) * k;
    let t = (u - v) * k;
    let cell_s = s.floor();
    let cell_t = t.floor();
    let fs = s - cell_s - 0.5; // [-0.5, 0.5] within the lattice cell
    let ft = t - cell_t - 0.5;

    // Alternate stud orientation per cell.
    let (along, across) = if ((cell_s + cell_t) as i64).rem_euclid(2) == 0 {
        (fs, ft)
    } else {
        (ft, fs)
    };

    // Rounded-bar SDF: half-length 0.28, half-width via the 0.16 falloff.
    let dx = (along.abs() - 0.28).max(0.0);
    let dy = across.abs();
    let d = (dx * dx + dy * dy).sqrt();
    let stud = (1.0 - d / 0.16).clamp(0.0, 1.0);
    stud * stud * (3.0 - 2.0 * stud) // smoothstep shoulder
}

/// Deterministic integer hash → \[0, 1\] for the hammered dimple jitter.
fn cell_hash(bx: i64, by: i64, seed: u32) -> f64 {
    let mut h = seed as u64;
    h ^= (bx as u64).wrapping_mul(6_364_136_223_846_793_005);
    h ^= (by as u64).wrapping_mul(1_442_695_040_888_963_407);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    (h as f64) * (1.0 / u64::MAX as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn styled(style: MetalStyle) -> MetalConfig {
        MetalConfig {
            style,
            ..MetalConfig::default()
        }
    }

    #[test]
    fn all_styles_generate() {
        for style in [
            MetalStyle::Brushed,
            MetalStyle::StandingSeam,
            MetalStyle::Hammered,
            MetalStyle::DiamondPlate,
        ] {
            let map = MetalGenerator::new(styled(style))
                .generate(32, 32)
                .expect("generate failed");
            assert_eq!(map.albedo.len(), 32 * 32 * 4);
        }
    }

    #[test]
    fn new_styles_differ_from_brushed() {
        let brushed = MetalGenerator::new(styled(MetalStyle::Brushed))
            .generate(64, 64)
            .expect("generate failed");
        for style in [MetalStyle::Hammered, MetalStyle::DiamondPlate] {
            let other = MetalGenerator::new(styled(style.clone()))
                .generate(64, 64)
                .expect("generate failed");
            assert_ne!(
                brushed.normal, other.normal,
                "{style:?} must shape the surface differently from Brushed"
            );
        }
    }

    #[test]
    fn dimple_and_stud_fields_tile() {
        for x in [0.0, 0.25, 0.75] {
            assert!((dimple_height(0.0, x, 6.0, 7) - dimple_height(1.0, x, 6.0, 7)).abs() < 1e-12);
            assert!((dimple_height(x, 0.0, 6.0, 7) - dimple_height(x, 1.0, 6.0, 7)).abs() < 1e-12);
            assert!((diamond_stud(0.0, x, 6.0) - diamond_stud(1.0, x, 6.0)).abs() < 1e-12);
            assert!((diamond_stud(x, 0.0, 6.0) - diamond_stud(x, 1.0, 6.0)).abs() < 1e-12);
        }
    }
}
