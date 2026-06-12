//! Core trait and data types shared by all texture generators.

use std::sync::OnceLock;

use rayon::prelude::*;

use bevy::{
    asset::{Assets, RenderAssetUsages},
    image::{Image, ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    prelude::Handle,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

/// Error returned when texture dimensions are invalid.
#[derive(Debug)]
pub enum TextureError {
    /// Either `width` or `height` was zero, which is not a valid wgpu texture size.
    ZeroDimension {
        /// Requested width (texels) — at least one of `width`/`height` is zero.
        width: u32,
        /// Requested height (texels) — at least one of `width`/`height` is zero.
        height: u32,
    },
    /// One or both dimensions exceeded [`MAX_DIMENSION`].
    DimensionTooLarge {
        /// Requested width (texels).
        width: u32,
        /// Requested height (texels).
        height: u32,
        /// The hard cap; equal to [`MAX_DIMENSION`] at the time the error was raised.
        max: u32,
    },
}

impl std::fmt::Display for TextureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextureError::ZeroDimension { width, height } => write!(
                f,
                "texture dimensions must be non-zero (got {width}×{height})"
            ),
            TextureError::DimensionTooLarge { width, height, max } => write!(
                f,
                "texture dimensions {width}×{height} exceed MAX_DIMENSION={max}"
            ),
        }
    }
}

impl std::error::Error for TextureError {}

/// Raw pixel buffers produced by a [`TextureGenerator`].
pub struct TextureMap {
    /// RGBA8 sRGB-encoded colour (albedo) pixels, row-major.
    ///
    /// The base level occupies the first `width × height × 4` bytes.  When
    /// [`mip_level_count`](TextureMap::mip_level_count) is greater than 1,
    /// the remaining mip levels follow contiguously (see
    /// [`with_mips`](TextureMap::with_mips)).
    pub albedo: Vec<u8>,
    /// RGBA8 linear tangent-space normal map pixels, row-major.  Same
    /// base-plus-mips layout as `albedo`.
    pub normal: Vec<u8>,
    /// RGBA8 ORM (Occlusion/Roughness/Metallic) pixels, row-major.  Same
    /// base-plus-mips layout as `albedo`.
    pub roughness: Vec<u8>,
    /// Optional RGBA8 sRGB-encoded emissive (glow) pixels, row-major.  Same
    /// base-plus-mips layout as `albedo`.  `None` for the non-glowing
    /// generators (currently all of them); when present the polling systems
    /// assign it to `StandardMaterial::emissive_texture`, where it is
    /// multiplied by the material's `emissive` colour factor — set
    /// [`MaterialSettings::emission_color`] /
    /// [`emission_strength`](crate::MaterialSettings::emission_strength)
    /// to white / 1.0 for unmodified map output.
    ///
    /// [`MaterialSettings::emission_color`]: crate::MaterialSettings::emission_color
    pub emissive: Option<Vec<u8>>,
    /// Texture width in texels.
    pub width: u32,
    /// Texture height in texels.
    pub height: u32,
    /// Number of mip levels contained in the pixel buffers, including the
    /// base level.  `1` means base level only (what `generate()` produces);
    /// larger values mean [`with_mips`](TextureMap::with_mips) has appended
    /// the full chain and upload becomes a pure move.
    pub mip_level_count: u32,
}

impl TextureMap {
    /// Compute the full mipmap pyramid for all three maps, appending the
    /// levels to the pixel buffers (type-correct averaging per map — see
    /// the module docs of [`map_to_images`]).
    ///
    /// The async generation tasks call this **on the worker thread** right
    /// after `generate()`, so the main-thread upload in the polling systems
    /// is a pure buffer move instead of a multi-megapixel box-filter pass
    /// (a guaranteed frame hitch at 4096²).  Synchronous callers may invoke
    /// it themselves; [`map_to_images`] computes the chain on demand when
    /// it is absent.
    ///
    /// No-op if the chain is already present.
    pub fn with_mips(mut self) -> Self {
        if self.mip_level_count > 1 {
            return self;
        }
        let (albedo, count) =
            generate_mipmaps(self.albedo, self.width, self.height, MipmapMode::Srgb);
        let (normal, _) =
            generate_mipmaps(self.normal, self.width, self.height, MipmapMode::Normal);
        let (roughness, _) =
            generate_mipmaps(self.roughness, self.width, self.height, MipmapMode::Linear);
        self.albedo = albedo;
        self.normal = normal;
        self.roughness = roughness;
        if let Some(emissive) = self.emissive.take() {
            let (emissive, _) =
                generate_mipmaps(emissive, self.width, self.height, MipmapMode::Srgb);
            self.emissive = Some(emissive);
        }
        self.mip_level_count = count;
        self
    }

    /// The byte length of the base level (`width × height × 4`) — the
    /// prefix of each pixel buffer that excludes any appended mip levels.
    #[inline]
    pub fn base_len(&self) -> usize {
        self.width as usize * self.height as usize * 4
    }
}

/// Handles returned after uploading a [`TextureMap`] into Bevy's asset system.
///
/// Cloning is cheap — `Handle<Image>` is a reference-counted asset id.
#[derive(Clone)]
pub struct GeneratedHandles {
    /// Handle to the albedo (colour) image.
    pub albedo: Handle<Image>,
    /// Handle to the tangent-space normal map image.
    pub normal: Handle<Image>,
    /// Handle to the ORM (Occlusion/Roughness/Metallic) image.
    pub roughness: Handle<Image>,
    /// Handle to the emissive (glow) image, when the generator produced one.
    pub emissive: Option<Handle<Image>>,
}

/// Reusable scratch buffers for texture generation.
///
/// At high resolutions each `Vec<f64>` noise grid is large (128 MB at
/// 4096×4096).  Generators that produce multiple grids can spike memory by
/// hundreds of megabytes per task.  A `Workspace` lets callers pre-allocate
/// these buffers once and pass them into [`TextureGenerator::generate_with_workspace`]
/// so the same heap memory is reused across generations instead of being
/// allocated and freed on every call.
///
/// # Example
///
/// ```rust,ignore
/// use bevy_symbios_texture::generator::{Workspace, TextureGenerator};
/// use bevy_symbios_texture::thatch::{ThatchConfig, ThatchGenerator};
///
/// let gen = ThatchGenerator::new(ThatchConfig::default());
/// let mut ws = Workspace::new();
///
/// // First call allocates; subsequent calls reuse the same buffers.
/// let map1 = gen.generate_with_workspace(2048, 2048, &mut ws).unwrap();
/// let map2 = gen.generate_with_workspace(2048, 2048, &mut ws).unwrap();
/// ```
pub struct Workspace {
    /// Pool of reusable `f64` grid buffers (noise samples, height maps, etc.).
    ///
    /// Generators call [`Workspace::take_grid`] to borrow a buffer and
    /// [`Workspace::return_grid`] to put it back when done.
    grids: Vec<Vec<f64>>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

impl Workspace {
    /// Create an empty workspace.  Buffers are allocated on first use.
    pub fn new() -> Self {
        Self { grids: Vec::new() }
    }

    /// Take a grid buffer from the pool, or create a new empty one.
    ///
    /// The returned `Vec` may have leftover capacity from a previous call —
    /// callers should `clear()` or use [`sample_grid_into`] which handles
    /// resizing.
    ///
    /// [`sample_grid_into`]: crate::noise::sample_grid_into
    pub fn take_grid(&mut self) -> Vec<f64> {
        self.grids.pop().unwrap_or_default()
    }

    /// Return a grid buffer to the pool for reuse by the next generation.
    pub fn return_grid(&mut self, buf: Vec<f64>) {
        self.grids.push(buf);
    }
}

/// Trait for procedural texture configuration structs.
///
/// Each struct that drives a specific texture type (bark, rock, ground, …)
/// should provide an implementation that turns its configuration into a
/// fully-populated [`TextureMap`].
pub trait TextureGenerator {
    /// Generate albedo, normal, and roughness pixel buffers at the given size.
    ///
    /// Returns [`TextureError`] if `width` or `height` is zero or exceeds
    /// [`MAX_DIMENSION`].
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError>;

    /// Generate using pre-allocated scratch buffers from `workspace`.
    ///
    /// The default implementation ignores the workspace and delegates to
    /// [`generate`](TextureGenerator::generate).  Generators that allocate
    /// large intermediate grids (e.g. [`ThatchGenerator`], [`BarkGenerator`])
    /// override this to pull buffers from the workspace, avoiding repeated
    /// 128 MB+ allocations at high resolutions.
    ///
    /// [`ThatchGenerator`]: crate::thatch::ThatchGenerator
    /// [`BarkGenerator`]: crate::bark::BarkGenerator
    fn generate_with_workspace(
        &self,
        width: u32,
        height: u32,
        _workspace: &mut Workspace,
    ) -> Result<TextureMap, TextureError> {
        self.generate(width, height)
    }
}

/// Maximum allowed texture dimension (per side).
///
/// Capped at 4096 to bound peak memory usage.  At 8192 the bark generator
/// alone requires ~1.75 GB per task; with four concurrent tasks that exceeds
/// 7 GB and OOMs mid-range machines.  At 4096 the peak is ~450 MB per task.
pub const MAX_DIMENSION: u32 = 4096;

/// Dimension guard for texture generators.
///
/// Call at the top of every [`TextureGenerator::generate`] implementation.
/// Returns an error for zero-sized textures (invalid wgpu resources) or
/// dimensions that exceed [`MAX_DIMENSION`].
#[inline]
pub fn validate_dimensions(width: u32, height: u32) -> Result<(), TextureError> {
    if width == 0 || height == 0 {
        return Err(TextureError::ZeroDimension { width, height });
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(TextureError::DimensionTooLarge {
            width,
            height,
            max: MAX_DIMENSION,
        });
    }
    Ok(())
}

/// Upload a [`TextureMap`] into [`Assets<Image>`] with repeat-wrapping samplers.
///
/// Takes `map` by value to move the pixel buffers directly into the `Image`
/// assets, avoiding an extra copy of up to 3 × W × H × 4 bytes.
pub fn map_to_images(map: TextureMap, images: &mut Assets<Image>) -> GeneratedHandles {
    GeneratedHandles {
        albedo: images.add(make_image(
            map.albedo,
            map.width,
            map.height,
            map.mip_level_count,
            TextureFormat::Rgba8UnormSrgb,
            ImageAddressMode::Repeat,
            MipmapMode::Srgb,
        )),
        normal: images.add(make_image(
            map.normal,
            map.width,
            map.height,
            map.mip_level_count,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::Repeat,
            MipmapMode::Normal,
        )),
        roughness: images.add(make_image(
            map.roughness,
            map.width,
            map.height,
            map.mip_level_count,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::Repeat,
            MipmapMode::Linear,
        )),
        emissive: map.emissive.map(|data| {
            images.add(make_image(
                data,
                map.width,
                map.height,
                map.mip_level_count,
                TextureFormat::Rgba8UnormSrgb,
                ImageAddressMode::Repeat,
                MipmapMode::Srgb,
            ))
        }),
    }
}

/// Upload a [`TextureMap`] into [`Assets<Image>`] with clamp-to-edge samplers.
///
/// Use this for alpha-masked cards (leaf, twig, window, stained glass, iron
/// grille) and sprite atlases, where the texture must not tile and the alpha
/// silhouette must not bleed across edges.  For tileable surfaces use
/// [`map_to_images`] instead.
pub fn map_to_images_card(map: TextureMap, images: &mut Assets<Image>) -> GeneratedHandles {
    GeneratedHandles {
        albedo: images.add(make_image(
            map.albedo,
            map.width,
            map.height,
            map.mip_level_count,
            TextureFormat::Rgba8UnormSrgb,
            ImageAddressMode::ClampToEdge,
            MipmapMode::Srgb,
        )),
        normal: images.add(make_image(
            map.normal,
            map.width,
            map.height,
            map.mip_level_count,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::ClampToEdge,
            MipmapMode::Normal,
        )),
        roughness: images.add(make_image(
            map.roughness,
            map.width,
            map.height,
            map.mip_level_count,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::ClampToEdge,
            MipmapMode::Linear,
        )),
        emissive: map.emissive.map(|data| {
            images.add(make_image(
                data,
                map.width,
                map.height,
                map.mip_level_count,
                TextureFormat::Rgba8UnormSrgb,
                ImageAddressMode::ClampToEdge,
                MipmapMode::Srgb,
            ))
        }),
    }
}

/// Controls how mipmap averages are computed for different texture types.
#[derive(Clone, Copy)]
enum MipmapMode {
    /// Albedo: decode from sRGB, average in linear light, re-encode to sRGB.
    /// Averaging in non-linear space makes mipmaps artificially dark.
    Srgb,
    /// Normal map: decode XYZ to [-1, 1], average, renormalize, re-encode.
    /// Averaging without renormalization shrinks or zeroes the normal length.
    Normal,
    /// ORM / linear maps: average directly in u8 space (already linear).
    Linear,
}

/// Decode an sRGB u8 value to linear-light f32.
fn srgb_to_linear(v: u8) -> f32 {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|i| {
            let c = i as f32 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        })
    })[v as usize]
}

/// Average a 2×2 block of RGBA8 pixels according to `mode`.
fn average_block(pixels: &[[u8; 4]], mode: MipmapMode) -> [u8; 4] {
    let n = pixels.len() as f32;
    match mode {
        MipmapMode::Linear => {
            let mut rgba = [0u32; 4];
            for p in pixels {
                for i in 0..4 {
                    rgba[i] += p[i] as u32;
                }
            }
            let count = pixels.len() as u32;
            [
                (rgba[0] / count) as u8,
                (rgba[1] / count) as u8,
                (rgba[2] / count) as u8,
                (rgba[3] / count) as u8,
            ]
        }
        MipmapMode::Srgb => {
            // Linearise, average in linear light, re-encode as sRGB.
            // Alpha is always linear — average directly.
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            let mut a = 0u32;
            for p in pixels {
                r += srgb_to_linear(p[0]);
                g += srgb_to_linear(p[1]);
                b += srgb_to_linear(p[2]);
                a += p[3] as u32;
            }
            [
                linear_to_srgb(r / n),
                linear_to_srgb(g / n),
                linear_to_srgb(b / n),
                (a / pixels.len() as u32) as u8,
            ]
        }
        MipmapMode::Normal => {
            // Decode XYZ from [0,255] → [-1,1], average, renormalize, re-encode.
            // Without renormalization, averaging +X and -X gives a zero vector
            // which produces black pixels and NaN propagation in PBR shaders.
            let mut nx = 0.0f32;
            let mut ny = 0.0f32;
            let mut nz = 0.0f32;
            for p in pixels {
                nx += p[0] as f32 / 127.5 - 1.0;
                ny += p[1] as f32 / 127.5 - 1.0;
                nz += p[2] as f32 / 127.5 - 1.0;
            }
            nx /= n;
            ny /= n;
            nz /= n;
            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
            nx /= len;
            ny /= len;
            nz /= len;
            let enc = |v: f32| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            [enc(nx), enc(ny), enc(nz), 255]
        }
    }
}

/// Recursively downsamples a base RGBA8 image to generate all mipmap levels.
///
/// Appends each successive level (half width, half height) directly onto
/// `data` using a 2×2 box filter.  `mode` controls how the box filter
/// averages pixels — see [`MipmapMode`].  Non-power-of-two dimensions are
/// handled by clamping the source 2×2 block to the actual image boundary.
///
/// Returns the expanded buffer and the total number of mip levels
/// (including level 0).
fn generate_mipmaps(
    mut data: Vec<u8>,
    base_width: u32,
    base_height: u32,
    mode: MipmapMode,
) -> (Vec<u8>, u32) {
    let mut mip_level_count = 1u32;
    let mut current_width = base_width as usize;
    let mut current_height = base_height as usize;
    let mut prev_offset = 0usize;

    while current_width > 1 || current_height > 1 {
        let next_width = current_width.max(2) / 2;
        let next_height = current_height.max(2) / 2;
        let next_offset = data.len();

        data.resize(next_offset + next_width * next_height * 4, 0);

        // The freshly-appended level is disjoint from its source level, so
        // split the buffer and fill the new level's rows in parallel on the
        // ambient rayon pool.  Byte-identical to serial filling.
        let (prev_all, next_level) = data.split_at_mut(next_offset);
        let prev_level = &prev_all[prev_offset..];

        next_level
            .par_chunks_mut(next_width * 4)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..next_width {
                    let sx = x * 2;
                    let sy = y * 2;

                    let mut pixels = [[0u8; 4]; 4];
                    let mut count = 0usize;

                    for dy in 0..2usize {
                        if sy + dy >= current_height {
                            continue;
                        }
                        for dx in 0..2usize {
                            if sx + dx >= current_width {
                                continue;
                            }
                            let src_idx = ((sy + dy) * current_width + (sx + dx)) * 4;
                            pixels[count] = [
                                prev_level[src_idx],
                                prev_level[src_idx + 1],
                                prev_level[src_idx + 2],
                                prev_level[src_idx + 3],
                            ];
                            count += 1;
                        }
                    }

                    let avg = average_block(&pixels[..count], mode);
                    let dst = x * 4;
                    row[dst..dst + 4].copy_from_slice(&avg);
                }
            });

        prev_offset = next_offset;
        current_width = next_width;
        current_height = next_height;
        mip_level_count += 1;
    }

    (data, mip_level_count)
}

fn make_image(
    data: Vec<u8>,
    width: u32,
    height: u32,
    mip_level_count: u32,
    format: TextureFormat,
    address_mode: ImageAddressMode,
    mipmap_mode: MipmapMode,
) -> Image {
    // Accept a chain precomputed on the worker ([`TextureMap::with_mips`]) or
    // compute one here for base-only buffers (synchronous callers, FileStore
    // loads).  Either way the buffer handed to the Image carries every level.
    let (mip_data, mip_level_count) = if mip_level_count > 1 {
        (data, mip_level_count)
    } else {
        generate_mipmaps(data, width, height, mipmap_mode)
    };

    let mut image = Image::new_uninit(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        format,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.mip_level_count = mip_level_count;
    image.data = Some(mip_data);
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        // wgpu requires all filter modes to be Linear when anisotropy_clamp > 1.
        mag_filter: bevy::image::ImageFilterMode::Linear,
        min_filter: bevy::image::ImageFilterMode::Linear,
        mipmap_filter: bevy::image::ImageFilterMode::Linear,
        anisotropy_clamp: 16,
        ..Default::default()
    });
    image
}

/// Convert a linear-light `f32` in `[0, 1]` to an sRGB-encoded `u8`.
///
/// Uses a 4096-entry lookup table (built once via [`OnceLock`]) to avoid
/// calling `f32::powf` millions of times per texture.  The input is quantised
/// to the nearest 1/4095 step before the lookup; the step is ~0.000244,
/// which keeps the maximum output error well below one count in u8.
///
/// A 256-entry table would be insufficient: the sRGB curve is steep near
/// zero and the first non-zero bin (linear ≈ 1/255) maps to sRGB ≈ 13,
/// making output values 1–12 unreachable.  4096 bins avoid that gap.
#[inline]
pub(crate) fn linear_to_srgb(linear: f32) -> u8 {
    const N: usize = 4096;
    static LUT: OnceLock<[u8; N]> = OnceLock::new();
    let lut = LUT.get_or_init(|| {
        std::array::from_fn(|i| {
            let c = i as f32 / (N - 1) as f32;
            let encoded = if c <= 0.003_130_8 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            };
            (encoded * 255.0).round() as u8
        })
    });
    lut[(linear.clamp(0.0, 1.0) * (N - 1) as f32).round() as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rock::{RockConfig, RockGenerator};

    #[test]
    fn with_mips_appends_full_chain_and_is_idempotent() {
        let map = RockGenerator::new(RockConfig::default())
            .generate(8, 8)
            .expect("8x8 generation")
            .with_mips();
        // 8 → 4 → 2 → 1: four levels.
        assert_eq!(map.mip_level_count, 4);
        let expected = (64 + 16 + 4 + 1) * 4;
        assert_eq!(map.albedo.len(), expected);
        assert_eq!(map.normal.len(), expected);
        assert_eq!(map.roughness.len(), expected);

        let again = map.with_mips();
        assert_eq!(again.mip_level_count, 4);
        assert_eq!(again.albedo.len(), expected, "with_mips must be a no-op");
    }

    #[test]
    fn emissive_maps_chain_and_upload() {
        let mut map = RockGenerator::new(RockConfig::default())
            .generate(8, 8)
            .expect("8x8 generation");
        map.emissive = Some(vec![128u8; map.base_len()]);

        let map = map.with_mips();
        let expected = (64 + 16 + 4 + 1) * 4;
        assert_eq!(
            map.emissive.as_ref().expect("emissive kept").len(),
            expected,
            "with_mips must chain the emissive map too"
        );

        let mut images = Assets::<Image>::default();
        let handles = map_to_images(map, &mut images);
        let handle = handles.emissive.as_ref().expect("emissive handle");
        let img = images.get(handle).expect("emissive image");
        assert_eq!(img.texture_descriptor.mip_level_count, 4);
        assert_eq!(img.texture_descriptor.format, TextureFormat::Rgba8UnormSrgb);
    }

    /// The worker-precomputed chain and the upload-time fallback must
    /// produce byte-identical images.
    #[test]
    fn precomputed_and_on_demand_uploads_are_identical() {
        let generator = RockGenerator::new(RockConfig::default());
        let mut images = Assets::<Image>::default();

        let on_demand = map_to_images(generator.generate(16, 16).expect("gen"), &mut images);
        let precomputed = map_to_images(
            generator.generate(16, 16).expect("gen").with_mips(),
            &mut images,
        );

        for (a, b) in [
            (&on_demand.albedo, &precomputed.albedo),
            (&on_demand.normal, &precomputed.normal),
            (&on_demand.roughness, &precomputed.roughness),
        ] {
            let ia = images.get(a).expect("on-demand image");
            let ib = images.get(b).expect("precomputed image");
            assert_eq!(
                ia.texture_descriptor.mip_level_count,
                ib.texture_descriptor.mip_level_count
            );
            assert_eq!(ia.texture_descriptor.size, ib.texture_descriptor.size);
            assert_eq!(ia.data, ib.data, "upload paths must be byte-identical");
        }
    }
}
