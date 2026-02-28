//! `bevy_symbios_texture` — procedural texture generation for Bevy.
//!
//! # Generators
//!
//! **Tileable surface textures** (bark, rock, ground, brick, plank, concrete,
//! metal, shingle, pavers, stucco): wrap seamlessly via toroidal 4-D noise
//! mapping.  Upload with [`map_to_images`] to get repeat-wrapping samplers.
//!
//! **Alpha-masked cards** (leaf, twig, window): produce silhouettes with
//! per-pixel alpha that must not tile.  Upload with [`map_to_images_card`] to
//! get clamp-to-edge samplers.
//!
//! # Architecture
//! Every generator implements [`TextureGenerator`], which produces a
//! [`TextureMap`] (raw pixel buffers for albedo, normal, roughness/ORM).
//!
//! Seamless tiling for surface textures is guaranteed by the [`ToroidalNoise`]
//! wrapper, which maps 2-D UV coordinates to a 4-D torus so noise wraps at
//! every edge with no seam.
//!
//! Both upload functions generate a full mipmap chain with type-correct
//! averaging: sRGB-linear averaging for albedo, renormalized averaging for
//! normal maps, and direct linear averaging for ORM maps.
//!
//! # Genetics
//! All config types implement `symbios_genetics::Genotype` (see [`genetics`]),
//! making them compatible with evolutionary search algorithms such as
//! `SimpleGA`, `Nsga2`, and `MapElites` from the `symbios-genetics` crate.

pub mod async_gen;
pub mod bark;
pub mod brick;
pub mod concrete;
pub mod generator;
pub mod genetics;
pub mod ground;
pub mod leaf;
pub mod metal;
pub mod noise;
pub mod normal;
pub mod pavers;
pub mod plank;
pub mod rock;
pub mod shingle;
pub mod stucco;
pub mod twig;
pub mod window;

#[cfg(feature = "egui")]
pub mod ui;

pub use generator::{
    GeneratedHandles, TextureError, TextureGenerator, TextureMap, map_to_images, map_to_images_card,
};
pub use leaf::{LeafConfig, LeafGenerator, LeafSample, LeafSampler, sample_leaf};
pub use noise::ToroidalNoise;
pub use twig::{TwigConfig, TwigGenerator};

use bevy::prelude::*;

/// Bevy plugin — registers the async-generation polling system.
pub struct SymbiosTexturePlugin;

impl Plugin for SymbiosTexturePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, async_gen::poll_texture_tasks);
    }
}
