//! Core trait and data types shared by all texture generators.

use std::sync::OnceLock;

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
    ZeroDimension { width: u32, height: u32 },
    /// One or both dimensions exceeded [`MAX_DIMENSION`].
    DimensionTooLarge { width: u32, height: u32, max: u32 },
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
    pub albedo: Vec<u8>,
    /// RGBA8 linear tangent-space normal map pixels, row-major.
    pub normal: Vec<u8>,
    /// RGBA8 ORM (Occlusion/Roughness/Metallic) pixels, row-major.
    pub roughness: Vec<u8>,
    /// Texture width in texels.
    pub width: u32,
    /// Texture height in texels.
    pub height: u32,
}

/// Handles returned after uploading a [`TextureMap`] into Bevy's asset system.
pub struct GeneratedHandles {
    /// Handle to the albedo (colour) image.
    pub albedo: Handle<Image>,
    /// Handle to the tangent-space normal map image.
    pub normal: Handle<Image>,
    /// Handle to the ORM (Occlusion/Roughness/Metallic) image.
    pub roughness: Handle<Image>,
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
            TextureFormat::Rgba8UnormSrgb,
            ImageAddressMode::Repeat,
            MipmapMode::Srgb,
        )),
        normal: images.add(make_image(
            map.normal,
            map.width,
            map.height,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::Repeat,
            MipmapMode::Normal,
        )),
        roughness: images.add(make_image(
            map.roughness,
            map.width,
            map.height,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::Repeat,
            MipmapMode::Linear,
        )),
    }
}

/// Upload a [`TextureMap`] into [`Assets<Image>`] with clamp-to-edge samplers.
///
/// Use this for foliage cards (leaf, twig) where the texture must not tile
/// and the alpha silhouette must not bleed across edges.  For tileable
/// surfaces use [`map_to_images`] instead.
pub fn map_to_images_card(map: TextureMap, images: &mut Assets<Image>) -> GeneratedHandles {
    GeneratedHandles {
        albedo: images.add(make_image(
            map.albedo,
            map.width,
            map.height,
            TextureFormat::Rgba8UnormSrgb,
            ImageAddressMode::ClampToEdge,
            MipmapMode::Srgb,
        )),
        normal: images.add(make_image(
            map.normal,
            map.width,
            map.height,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::ClampToEdge,
            MipmapMode::Normal,
        )),
        roughness: images.add(make_image(
            map.roughness,
            map.width,
            map.height,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::ClampToEdge,
            MipmapMode::Linear,
        )),
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

        for y in 0..next_height {
            for x in 0..next_width {
                let dst_idx = next_offset + (y * next_width + x) * 4;
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
                        let src_idx = prev_offset + ((sy + dy) * current_width + (sx + dx)) * 4;
                        pixels[count] = [
                            data[src_idx],
                            data[src_idx + 1],
                            data[src_idx + 2],
                            data[src_idx + 3],
                        ];
                        count += 1;
                    }
                }

                let avg = average_block(&pixels[..count], mode);
                data[dst_idx] = avg[0];
                data[dst_idx + 1] = avg[1];
                data[dst_idx + 2] = avg[2];
                data[dst_idx + 3] = avg[3];
            }
        }

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
    format: TextureFormat,
    address_mode: ImageAddressMode,
    mipmap_mode: MipmapMode,
) -> Image {
    // Pass base-level data directly — its length equals width * height * 4, which
    // is exactly what Image::new expects.  No dummy zeroed buffer needed.
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        format,
        RenderAssetUsages::default(),
    );
    let base_data = image.data.take().unwrap();
    let (mip_data, mip_level_count) = generate_mipmaps(base_data, width, height, mipmap_mode);
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
