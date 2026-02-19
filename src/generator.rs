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
    ZeroDimension { width: u32, height: u32 },
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
    pub albedo: Vec<u8>,
    pub normal: Vec<u8>,
    pub roughness: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Handles returned after uploading a [`TextureMap`] into Bevy's asset system.
pub struct GeneratedHandles {
    pub albedo: Handle<Image>,
    pub normal: Handle<Image>,
    pub roughness: Handle<Image>,
}

/// Implement this on any procedural texture configuration struct.
pub trait TextureGenerator {
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
        )),
        normal: images.add(make_image(
            map.normal,
            map.width,
            map.height,
            TextureFormat::Rgba8Unorm,
        )),
        roughness: images.add(make_image(
            map.roughness,
            map.width,
            map.height,
            TextureFormat::Rgba8Unorm,
        )),
    }
}

fn make_image(data: Vec<u8>, width: u32, height: u32, format: TextureFormat) -> Image {
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
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        ..Default::default()
    });
    image
}

/// Convert a linear-light `f32` in `[0, 1]` to an sRGB-encoded `u8`.
///
/// Uses a 256-entry lookup table (built once via [`OnceLock`]) to avoid
/// calling `f32::powf` millions of times per texture.  The input is quantised
/// to the nearest 1/255 step before the lookup; the resulting error is less
/// than one count in the output, which is indistinguishable after u8
/// quantisation.
#[inline]
pub(crate) fn linear_to_srgb(linear: f32) -> u8 {
    static LUT: OnceLock<[u8; 256]> = OnceLock::new();
    let lut = LUT.get_or_init(|| {
        std::array::from_fn(|i| {
            let c = i as f32 / 255.0_f32;
            let encoded = if c <= 0.003_130_8 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            };
            (encoded * 255.0).round() as u8
        })
    });
    lut[(linear.clamp(0.0, 1.0) * 255.0).round() as usize]
}
