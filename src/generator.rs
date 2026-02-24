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
/// Prevents allocation of unbounded memory and keeps sizes within GPU limits
/// that are commonly supported across all major platforms.
pub const MAX_DIMENSION: u32 = 8192;

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
        )),
        normal: images.add(make_image(
            map.normal,
            map.width,
            map.height,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::Repeat,
        )),
        roughness: images.add(make_image(
            map.roughness,
            map.width,
            map.height,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::Repeat,
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
        )),
        normal: images.add(make_image(
            map.normal,
            map.width,
            map.height,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::ClampToEdge,
        )),
        roughness: images.add(make_image(
            map.roughness,
            map.width,
            map.height,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::ClampToEdge,
        )),
    }
}

/// Recursively downsamples a base RGBA8 image to generate all mipmap levels.
///
/// Appends each successive level (half width, half height) directly onto
/// `data` using a 2×2 box filter.  Non-power-of-two dimensions are handled
/// by clamping the source 2×2 block to the actual image boundary.
///
/// Returns the expanded buffer and the total number of mip levels
/// (including level 0).
fn generate_mipmaps_rgba(mut data: Vec<u8>, base_width: u32, base_height: u32) -> (Vec<u8>, u32) {
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

                let mut r = 0u32;
                let mut g = 0u32;
                let mut b = 0u32;
                let mut a = 0u32;
                let mut samples = 0u32;

                for dy in 0..2usize {
                    if sy + dy >= current_height {
                        continue;
                    }
                    for dx in 0..2usize {
                        if sx + dx >= current_width {
                            continue;
                        }
                        let src_idx = prev_offset + ((sy + dy) * current_width + (sx + dx)) * 4;
                        r += data[src_idx] as u32;
                        g += data[src_idx + 1] as u32;
                        b += data[src_idx + 2] as u32;
                        a += data[src_idx + 3] as u32;
                        samples += 1;
                    }
                }

                data[dst_idx] = (r / samples) as u8;
                data[dst_idx + 1] = (g / samples) as u8;
                data[dst_idx + 2] = (b / samples) as u8;
                data[dst_idx + 3] = (a / samples) as u8;
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
) -> Image {
    let base_size = (width as usize) * (height as usize) * 4;
    let (mip_data, mip_level_count) = generate_mipmaps_rgba(data, width, height);

    // Image::new validates data.len() == width * height * bytes_per_pixel,
    // so we construct with the level-0 slice, then overwrite with the full chain.
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        mip_data[..base_size].to_vec(),
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
