//! Shared scaffolding for tileable surface generators.
//!
//! Mirrors the [`sprite`](crate::sprite) architecture for the surface
//! family: each generator module defines a *cell sampler* — a struct
//! implementing [`SurfaceCell`] that captures the per-generation state
//! (config, precomputed noise grids) and answers point queries — and the
//! shared [`generate_surface`] driver owns buffer allocation, sRGB albedo
//! packing, ORM packing, and the toroidal normal-map derivation.
//!
//! # Height-field convention
//!
//! [`SurfaceSample::height`] feeds [`height_to_normal`] **unmodified**: the
//! driver neither normalises nor clamps it, and `normal_strength` is passed
//! straight through.  Generators therefore control the gradient scale
//! themselves — e.g. a generator emitting raw `[-1, 1]` noise passes
//! `config.normal_strength * 0.5` to compensate for the doubled range,
//! exactly as the hand-rolled implementations did.
//!
//! # UV convention
//!
//! Cells are sampled at `u = x / width`, `v = y / height` (texel corner),
//! matching the toroidal grid samplers in [`crate::noise`], so cells that
//! index a precomputed [`sample_grid_into`](crate::noise::sample_grid_into)
//! buffer and cells that evaluate analytically agree on coordinates.

use rayon::prelude::*;

use crate::{
    generator::{TextureError, TextureMap, Workspace, linear_to_srgb, validate_dimensions},
    normal::{BoundaryMode, height_to_normal},
};

/// One point sample of a tileable surface.
///
/// `color` is linear RGB `[0, 1]` (the driver sRGB-encodes it);
/// `roughness` / `metallic` / `occlusion` land in the ORM green / blue /
/// red channels respectively; `height` feeds the normal-map derivation
/// (see the module docs for the scale convention).
pub struct SurfaceSample {
    /// Height value handed to [`height_to_normal`] unmodified.
    pub height: f64,
    /// Linear RGB albedo in `[0, 1]`.
    pub color: [f32; 3],
    /// PBR roughness `[0, 1]` (ORM green channel).
    pub roughness: f32,
    /// PBR metallic `[0, 1]` (ORM blue channel).
    pub metallic: f32,
    /// Ambient occlusion `[0, 1]` (ORM red channel).
    pub occlusion: f32,
}

impl SurfaceSample {
    /// A dielectric, un-occluded sample — the common case for natural
    /// materials (`metallic = 0`, `occlusion = 1`).
    #[inline]
    pub fn matte(height: f64, color: [f32; 3], roughness: f32) -> Self {
        Self {
            height,
            color,
            roughness,
            metallic: 0.0,
            occlusion: 1.0,
        }
    }
}

/// A fully-instantiated surface sampler: configuration plus any precomputed
/// per-generation state (noise grids, lookup tables), ready to answer point
/// queries.
///
/// Implementations are constructed once per `generate()` call and sampled
/// for every texel by [`generate_surface`].
pub trait SurfaceCell {
    /// Sample the surface at texel `(x, y)` / UV `(u, v)`.
    ///
    /// Both coordinate forms are provided so grid-backed cells can index
    /// `y * width + x` directly (keeping the trigonometry-free fast path of
    /// precomputed toroidal grids) while analytic cells use UV.
    fn sample(&self, x: u32, y: u32, u: f64, v: f64) -> SurfaceSample;
}

/// Render a tileable `width × height` surface through `cell`.
///
/// The driver:
///
/// 1. samples every texel via [`SurfaceCell::sample`],
/// 2. packs albedo (sRGB-encoded, opaque) and ORM
///    (occlusion / roughness / metallic from the sample),
/// 3. derives the tangent-space normal map from the height field with
///    toroidal ([`BoundaryMode::Wrap`]) neighbours, so normals tile
///    seamlessly alongside the colour data.
///
/// Rows are sampled in parallel — `cell` must be `Sync`.  Work runs on the
/// ambient rayon pool: async generation tasks already execute on the
/// crate's private pool, so nested row-parallelism work-steals across that
/// pool's threads and [`AsyncTextureConfig::pool_threads`] remains the
/// effective CPU cap; direct synchronous calls parallelise on the caller's
/// pool (usually the global one).  Output is byte-identical to serial
/// evaluation — every sample is a pure function of its coordinates.
///
/// `workspace` (optional) pools the height-field buffer across calls; pass
/// the same [`Workspace`] from
/// [`generate_with_workspace`](crate::generator::TextureGenerator::generate_with_workspace)
/// to avoid re-allocating large grids at high resolutions.
///
/// [`AsyncTextureConfig::pool_threads`]: crate::AsyncTextureConfig
pub fn generate_surface<C: SurfaceCell + Sync>(
    width: u32,
    height: u32,
    normal_strength: f32,
    mut workspace: Option<&mut Workspace>,
    cell: &C,
) -> Result<TextureMap, TextureError> {
    validate_dimensions(width, height)?;

    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    let mut heights = workspace
        .as_deref_mut()
        .map_or_else(Vec::new, |ws| ws.take_grid());
    heights.clear();
    heights.resize(n, 0.0);

    let mut albedo = vec![0u8; n * 4];
    let mut roughness = vec![0u8; n * 4];

    heights
        .par_chunks_mut(w)
        .zip(albedo.par_chunks_mut(w * 4))
        .zip(roughness.par_chunks_mut(w * 4))
        .enumerate()
        .for_each(|(y, ((height_row, albedo_row), orm_row))| {
            let v = y as f64 / h as f64;
            for (x, height_slot) in height_row.iter_mut().enumerate() {
                let u = x as f64 / w as f64;
                let s = cell.sample(x as u32, y as u32, u, v);

                *height_slot = s.height;

                let ai = x * 4;
                albedo_row[ai] = linear_to_srgb(s.color[0]);
                albedo_row[ai + 1] = linear_to_srgb(s.color[1]);
                albedo_row[ai + 2] = linear_to_srgb(s.color[2]);
                albedo_row[ai + 3] = 255;

                orm_row[ai] = (s.occlusion.clamp(0.0, 1.0) * 255.0).round() as u8;
                orm_row[ai + 1] = (s.roughness.clamp(0.0, 1.0) * 255.0).round() as u8;
                orm_row[ai + 2] = (s.metallic.clamp(0.0, 1.0) * 255.0).round() as u8;
                orm_row[ai + 3] = 255;
            }
        });

    let normal = height_to_normal(&heights, width, height, normal_strength, BoundaryMode::Wrap);

    if let Some(ws) = workspace {
        ws.return_grid(heights);
    }

    Ok(TextureMap {
        albedo,
        normal,
        roughness,
        width,
        height,
    })
}

/// Linear interpolation between two `f32` values with `t` clamped to
/// `[0, 1]` — the shared home for the helper every surface module used to
/// define locally.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constant-output cell for driver-level assertions.
    struct Flat;

    impl SurfaceCell for Flat {
        fn sample(&self, _x: u32, _y: u32, _u: f64, _v: f64) -> SurfaceSample {
            SurfaceSample {
                height: 0.5,
                color: [1.0, 0.0, 0.0],
                roughness: 0.5,
                metallic: 1.0,
                occlusion: 0.0,
            }
        }
    }

    #[test]
    fn driver_packs_orm_channels() {
        let map = generate_surface(4, 4, 1.0, None, &Flat).expect("generate");
        // O=R, R=G, M=B channel order.
        assert_eq!(map.roughness[0], 0, "occlusion 0.0 → 0");
        assert_eq!(map.roughness[1], 128, "roughness 0.5 → 128");
        assert_eq!(map.roughness[2], 255, "metallic 1.0 → 255");
        assert_eq!(map.roughness[3], 255);
        // Albedo is opaque and sRGB-encoded.
        assert_eq!(map.albedo[0], 255);
        assert_eq!(map.albedo[3], 255);
        // Flat height field → neutral normal (128, 128, 255).
        assert_eq!(&map.normal[0..4], &[128, 128, 255, 255]);
    }

    #[test]
    fn driver_rejects_invalid_dimensions() {
        assert!(generate_surface(0, 4, 1.0, None, &Flat).is_err());
        assert!(generate_surface(4, 0, 1.0, None, &Flat).is_err());
    }

    #[test]
    fn driver_reuses_workspace_buffers() {
        let mut ws = Workspace::new();
        let a = generate_surface(8, 8, 1.0, Some(&mut ws), &Flat).expect("first");
        let b = generate_surface(8, 8, 1.0, Some(&mut ws), &Flat).expect("second");
        assert_eq!(a.albedo, b.albedo);
        assert_eq!(a.normal, b.normal);
    }

    #[test]
    fn matte_sets_dielectric_defaults() {
        let s = SurfaceSample::matte(0.3, [0.1, 0.2, 0.3], 0.7);
        assert_eq!(s.metallic, 0.0);
        assert_eq!(s.occlusion, 1.0);
    }

    #[test]
    fn lerp_clamps_t() {
        assert_eq!(lerp(0.0, 1.0, -1.0), 0.0);
        assert_eq!(lerp(0.0, 1.0, 2.0), 1.0);
        assert_eq!(lerp(0.0, 1.0, 0.5), 0.5);
    }
}
