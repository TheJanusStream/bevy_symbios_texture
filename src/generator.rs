//! Bevy asset adapters for the core texture pipeline.
//!
//! The pure trait and data types ([`TextureGenerator`], [`TextureMap`],
//! [`TextureError`], [`Workspace`], [`validate_dimensions`], [`MAX_DIMENSION`])
//! live in the Bevy-free [`symbios_texture::generator`] core module and are
//! re-exported here so the public path `bevy_symbios_texture::generator::*`
//! is unchanged.  This module adds the Bevy-coupled upload helpers that turn
//! a [`TextureMap`] into [`Handle<Image>`] assets.

pub use symbios_texture::generator::{
    MAX_DIMENSION, TextureError, TextureGenerator, TextureMap, Workspace, validate_dimensions,
};

use bevy::{
    asset::{Assets, RenderAssetUsages},
    image::{Image, ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    prelude::Handle,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

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

/// Controls how the upload-time mipmap fallback averages pixels for different
/// texture types.  Mirrors the core's per-channel averaging modes.
#[derive(Clone, Copy)]
enum MipmapMode {
    /// Albedo: decode from sRGB, average in linear light, re-encode to sRGB.
    Srgb,
    /// Normal map: decode XYZ to [-1, 1], average, renormalize, re-encode.
    Normal,
    /// ORM / linear maps: average directly in u8 space (already linear).
    Linear,
}

/// Upload a [`TextureMap`] into [`Assets<Image>`] with repeat-wrapping samplers.
///
/// Takes `map` by value to move the pixel buffers directly into the `Image`
/// assets, avoiding an extra copy of up to 3 × W × H × 4 bytes.
///
/// Uploads with [`RenderAssetUsages::RENDER_WORLD`] only, so the CPU pixel
/// buffer is released once the texture reaches the GPU. That is correct for
/// textures bound straight to a material and never sampled back on the CPU, and
/// a meaningful saving on wasm where linear memory never returns to the OS.
/// Callers that read the pixels back after upload (e.g. concatenating layers
/// into a texture array) must use [`map_to_images_with_usages`] with
/// [`RenderAssetUsages::MAIN_WORLD`] to keep `Image::data` resident.
pub fn map_to_images(map: TextureMap, images: &mut Assets<Image>) -> GeneratedHandles {
    map_to_images_with_usages(map, RenderAssetUsages::RENDER_WORLD, images)
}

/// [`map_to_images`] with an explicit [`RenderAssetUsages`] (repeat-wrapping
/// samplers). Pass [`RenderAssetUsages::MAIN_WORLD`] when the resulting
/// `Image::data` must stay CPU-resident after upload; pass
/// [`RenderAssetUsages::RENDER_WORLD`] (what [`map_to_images`] uses) to free it.
pub fn map_to_images_with_usages(
    map: TextureMap,
    usages: RenderAssetUsages,
    images: &mut Assets<Image>,
) -> GeneratedHandles {
    GeneratedHandles {
        albedo: images.add(make_image(
            map.albedo,
            map.width,
            map.height,
            map.mip_level_count,
            TextureFormat::Rgba8UnormSrgb,
            ImageAddressMode::Repeat,
            MipmapMode::Srgb,
            usages,
        )),
        normal: images.add(make_image(
            map.normal,
            map.width,
            map.height,
            map.mip_level_count,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::Repeat,
            MipmapMode::Normal,
            usages,
        )),
        roughness: images.add(make_image(
            map.roughness,
            map.width,
            map.height,
            map.mip_level_count,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::Repeat,
            MipmapMode::Linear,
            usages,
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
                usages,
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
///
/// Like [`map_to_images`], uploads with [`RenderAssetUsages::RENDER_WORLD`]
/// only (CPU buffer freed after GPU upload); use
/// [`map_to_images_card_with_usages`] when the pixels must stay CPU-resident.
pub fn map_to_images_card(map: TextureMap, images: &mut Assets<Image>) -> GeneratedHandles {
    map_to_images_card_with_usages(map, RenderAssetUsages::RENDER_WORLD, images)
}

/// [`map_to_images_card`] with an explicit [`RenderAssetUsages`] (clamp-to-edge
/// samplers). See [`map_to_images_with_usages`] for when to choose
/// `MAIN_WORLD` over the default `RENDER_WORLD`.
pub fn map_to_images_card_with_usages(
    map: TextureMap,
    usages: RenderAssetUsages,
    images: &mut Assets<Image>,
) -> GeneratedHandles {
    GeneratedHandles {
        albedo: images.add(make_image(
            map.albedo,
            map.width,
            map.height,
            map.mip_level_count,
            TextureFormat::Rgba8UnormSrgb,
            ImageAddressMode::ClampToEdge,
            MipmapMode::Srgb,
            usages,
        )),
        normal: images.add(make_image(
            map.normal,
            map.width,
            map.height,
            map.mip_level_count,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::ClampToEdge,
            MipmapMode::Normal,
            usages,
        )),
        roughness: images.add(make_image(
            map.roughness,
            map.width,
            map.height,
            map.mip_level_count,
            TextureFormat::Rgba8Unorm,
            ImageAddressMode::ClampToEdge,
            MipmapMode::Linear,
            usages,
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
                usages,
            ))
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn make_image(
    data: Vec<u8>,
    width: u32,
    height: u32,
    mip_level_count: u32,
    format: TextureFormat,
    address_mode: ImageAddressMode,
    mipmap_mode: MipmapMode,
    usages: RenderAssetUsages,
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
        usages,
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

// --- upload-time mipmap fallback --------------------------------------------
//
// Mirrors `symbios_texture::generator`'s box-filter chain so a base-only
// `TextureMap` (synchronous callers, FileStore loads) gets a full mip chain at
// upload time.  Worker-precomputed chains skip this path entirely.

use std::sync::OnceLock;

use rayon::prelude::*;

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

#[inline]
fn linear_to_srgb(linear: f32) -> u8 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use symbios_texture::rock::{RockConfig, RockGenerator};

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
