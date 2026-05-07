//! `bevy_symbios_texture` — procedural texture generation for Bevy.
//!
//! # Generators
//!
//! **Tileable surface textures** (bark, rock, ground, brick, plank, concrete,
//! metal, shingle, pavers, stucco, ashlar, cobblestone, thatch, marble,
//! corrugated, asphalt, wainscoting, encaustic): wrap seamlessly via toroidal
//! 4-D noise mapping.  Upload with [`map_to_images`] to get repeat-wrapping
//! samplers.
//!
//! **Alpha-masked cards** (leaf, twig, window, stained_glass, iron_grille):
//! produce silhouettes with per-pixel alpha that must not tile.  Upload with
//! [`map_to_images_card`] to get clamp-to-edge samplers.
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

pub mod ashlar;
pub mod asphalt;
pub mod async_gen;
pub mod bark;
pub mod brick;
pub mod cache;
pub mod cobblestone;
pub mod concrete;
pub mod corrugated;
pub mod curve;
pub mod encaustic;
pub mod generator;
pub mod genetics;
pub mod ground;
pub mod iron_grille;
pub mod leaf;
pub mod marble;
pub mod material;
pub mod metal;
pub mod noise;
pub mod normal;
pub mod pavers;
pub mod plank;
pub mod rock;
pub mod shingle;
pub mod stained_glass;
pub mod stucco;
pub mod thatch;
pub mod twig;
pub mod wainscoting;
pub mod window;

#[cfg(feature = "egui")]
pub mod ui;

pub use async_gen::{AsyncTextureConfig, DEFAULT_POOL_THREADS};
pub use cache::{
    DEFAULT_MEMORY_CACHE_ENTRIES, FileStore, MemoryStore, TextureCache, TextureCacheKey,
    TextureCacheStore,
};
pub use curve::{
    AnimatedProceduralMaterial, EaseInOut, Linear, ParameterCurve, ScriptedFn, Stepped,
    TextureCurve,
};
pub use generator::{
    GeneratedHandles, TextureError, TextureGenerator, TextureMap, Workspace, map_to_images,
    map_to_images_card,
};
pub use leaf::{LeafConfig, LeafGenerator, LeafSample, LeafSampler, sample_leaf};
pub use material::{
    MaterialSettings, PatchMaterialTextures, RenderProperties, TextureConfig,
    build_procedural_material_async,
};
pub use noise::ToroidalNoise;
pub use twig::{TwigConfig, TwigGenerator};

use bevy::prelude::*;

/// Bevy plugin — registers the async-generation polling system and applies
/// the [`AsyncTextureConfig`] to the private texture-generation thread pool.
///
/// Construct via [`Default`] for the standard [`DEFAULT_POOL_THREADS`] cap,
/// or set [`SymbiosTexturePlugin::config`] explicitly for custom pool sizing:
///
/// ```rust,ignore
/// app.add_plugins(SymbiosTexturePlugin {
///     config: AsyncTextureConfig { pool_threads: 0 }, // auto = cores / 2
/// });
/// ```
///
/// The pool configuration is applied at plugin-build time and frozen on the
/// first generation request — re-adding the plugin with a different config
/// after a task has been dispatched has no effect.
#[derive(Default, Clone)]
pub struct SymbiosTexturePlugin {
    /// Configuration for the private texture-generation thread pool.
    pub config: AsyncTextureConfig,
}

impl Plugin for SymbiosTexturePlugin {
    fn build(&self, app: &mut App) {
        // Best-effort apply: silently ignored if a config has already been
        // installed (e.g. the plugin is added more than once, or another path
        // ran first).  The first call wins; documented above.
        let _ = async_gen::set_pool_config(self.config.clone());
        app.insert_resource(self.config.clone());
        app.add_systems(
            Update,
            (
                async_gen::poll_texture_tasks,
                material::patch_procedural_material_textures,
                curve::tick_animated_procedural_materials,
            ),
        );
    }
}
