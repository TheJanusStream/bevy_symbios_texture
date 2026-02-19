//! `bevy_symbios_texture` — procedural, tileable texture generation for Bevy.
//!
//! # Architecture
//! Every generator implements [`TextureGenerator`], which produces a
//! [`TextureMap`] (raw pixel buffers for albedo, normal, roughness).
//! Call [`map_to_images`] to upload those buffers into [`bevy::asset::Assets<Image>`].
//!
//! Tileability is guaranteed by the [`ToroidalNoise`] wrapper, which maps 2-D
//! UV coordinates to a 4-D torus so noise wraps at every edge with no seam.

pub mod async_gen;
pub mod bark;
pub mod generator;
pub mod ground;
pub mod noise;
pub mod normal;
pub mod rock;

pub use generator::{GeneratedHandles, TextureError, TextureGenerator, TextureMap, map_to_images};
pub use noise::ToroidalNoise;

use bevy::prelude::*;

/// Bevy plugin — registers the async-generation polling system.
pub struct SymbiosTexturePlugin;

impl Plugin for SymbiosTexturePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, async_gen::poll_texture_tasks);
    }
}
