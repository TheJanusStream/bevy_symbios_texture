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

/// Upload a [`TextureMap`] into [`Assets<Image>`] with repeat-wrapping samplers.
pub fn map_to_images(map: &TextureMap, images: &mut Assets<Image>) -> GeneratedHandles {
    GeneratedHandles {
        albedo: images.add(make_image(
            map.albedo.clone(),
            map.width,
            map.height,
            TextureFormat::Rgba8UnormSrgb,
        )),
        normal: images.add(make_image(
            map.normal.clone(),
            map.width,
            map.height,
            TextureFormat::Rgba8Unorm,
        )),
        roughness: images.add(make_image(
            map.roughness.clone(),
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
