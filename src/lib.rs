//! `bevy_symbios_texture` — procedural texture generation for Bevy.
//!
//! # Crate split
//!
//! The pure, Bevy-free generation core lives in the [`symbios_texture`] crate
//! (every generator module, its `Config`/`Generator` types, the
//! [`TextureGenerator`] trait, the [`TextureMap`] pipeline, [`ToroidalNoise`],
//! the [`sprite`]/[`surface`] kits, the per-config genetics, and the
//! registry).  This wrapper re-exports that core wholesale and adds the
//! Bevy-coupled layer: the [`SymbiosTexturePlugin`], the async generation
//! pool, the `bevy` `Image` upload adapters, the
//! [`TextureCache`] resource, the procedural-material builder, and the egui
//! UI.  The public API is unchanged across the split.
//!
//! # Generators
//!
//! **Tileable surface textures** (bark, rock, ground, brick, plank, concrete,
//! metal, shingle, pavers, stucco, ashlar, cobblestone, thatch, marble,
//! corrugated, asphalt, wainscoting, encaustic, fabric, sand, snow, ice,
//! lava): wrap seamlessly via toroidal 4-D noise mapping.  Upload with
//! [`map_to_images`] to get repeat-wrapping samplers.  `lava` additionally
//! produces an emissive (glow) map.
//!
//! **Alpha-masked cards** (leaf, twig, window, stained_glass, iron_grille,
//! chain_link, log_end): produce silhouettes with per-pixel alpha that must
//! not tile.  Upload with [`map_to_images_card`] to get clamp-to-edge
//! samplers.
//!
//! **Sprite atlases** (soft_disc, spark, snowflake, puff, ring, petal,
//! shard, leaf_sprite, flame, flower): alpha-silhouette cards aimed at
//! particle billboards.  Each can bake a `variant_rows × variant_cols`
//! atlas where every cell renders a per-cell-seeded variant of the same
//! config — one bake gives a particle system per-particle shape variety via
//! random atlas frames.  Shared conventions live in [`sprite`]; upload with
//! [`map_to_images_card`].
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
//! # Plugin & async generation
//! [`SymbiosTexturePlugin`] registers the polling systems and applies an
//! [`AsyncTextureConfig`] to a private rayon thread pool dedicated to
//! texture generation.  Spawn [`async_gen::PendingTexture`] components to
//! offload work; the result lands on the entity as
//! [`async_gen::TextureReady`].
//!
//! # Procedural materials
//! [`build_procedural_material_async`] is a one-shot helper that returns a
//! `Handle<StandardMaterial>` immediately and patches the generated textures
//! in once the background task completes.  Pair with an optional
//! [`TextureCache`] resource to avoid regenerating identical configs.
//!
//! # Animated parameters
//! [`AnimatedProceduralMaterial`] drives time-varying texture parameters by
//! re-evaluating a closure each frame, regenerating only when the
//! fingerprint of the resulting [`TextureConfig`] changes (with a
//! configurable wall-clock cooldown).
//!
//! # Genetics
//! All config types implement `symbios_genetics::Genotype` (see [`genetics`]),
//! making them compatible with evolutionary search algorithms such as
//! `SimpleGA`, `Nsga2`, and `MapElites` from the `symbios-genetics` crate.
//!
//! [`TextureCache`]: cache::TextureCache
//! [`TextureConfig`]: material::TextureConfig

// Re-export the entire Bevy-free core so the public module paths
// (`bevy_symbios_texture::ashlar`, `::sprite`, `::genetics`, …) and every
// pure `Config`/`Generator`/helper type are preserved exactly.  The wrapper's
// own `generator` module below shadows the core's `generator` in this glob
// (it re-exports the core items and adds the Bevy upload adapters), so
// `bevy_symbios_texture::generator` is the merged view.
pub use symbios_texture::*;

// Bevy-coupled modules kept in the wrapper.
pub mod async_gen;
pub mod cache;
pub mod curve;
pub mod generator;
pub mod material;

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
    map_to_images_card, map_to_images_card_with_usages, map_to_images_with_usages,
};
pub use material::{
    MaterialSettings, PatchMaterialTextures, RenderProperties, TextureConfig,
    build_procedural_material_async,
};
pub use symbios_texture::leaf::{LeafConfig, LeafGenerator, LeafSample, LeafSampler, sample_leaf};
pub use symbios_texture::noise::ToroidalNoise;
pub use symbios_texture::sprite::{CellRng, SpriteCell, SpriteSample, generate_atlas};
pub use symbios_texture::surface::{SurfaceCell, SurfaceSample, generate_surface};
pub use symbios_texture::twig::{TwigConfig, TwigGenerator};

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
