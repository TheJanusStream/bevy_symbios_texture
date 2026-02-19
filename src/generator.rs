//! Core trait and data types shared by all texture generators.

use bevy::{
    asset::{Assets, RenderAssetUsages},
    image::{Image, ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    prelude::Handle,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

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
    fn generate(&self, width: u32, height: u32) -> TextureMap;
}

/// Maximum allowed texture dimension (per side).
///
/// Prevents allocation of unbounded memory and keeps sizes within GPU limits
/// that are commonly supported across all major platforms.
pub const MAX_DIMENSION: u32 = 8192;

/// Panic-guard for texture dimensions.
///
/// Call at the top of every [`TextureGenerator::generate`] implementation.
/// Catches both degenerate zero-sized textures (which produce invalid
/// `wgpu` resources) and absurdly large ones (which would OOM or overflow).
#[inline]
pub fn validate_dimensions(width: u32, height: u32) {
    assert!(
        width > 0 && height > 0,
        "texture dimensions must be non-zero (got {width}×{height})"
    );
    assert!(
        width <= MAX_DIMENSION && height <= MAX_DIMENSION,
        "texture dimensions exceed MAX_DIMENSION={MAX_DIMENSION} (got {width}×{height})"
    );
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
