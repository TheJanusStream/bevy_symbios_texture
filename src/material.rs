//! Procedural-material settings + one-shot async builder.
//!
//! [`MaterialSettings`] is a generator-agnostic PBR description: base colour,
//! emission, roughness, metallic, plus an embedded [`TextureConfig`] enum
//! that selects which generator (if any) drives the texture slots.
//!
//! [`build_procedural_material_async`] is the central entry point: from a
//! `&MaterialSettings` it returns a [`Handle<StandardMaterial>`] immediately
//! and dispatches the texture-generation work in the background.  When the
//! generator finishes, [`patch_procedural_material_textures`] (registered
//! automatically by [`SymbiosTexturePlugin`](crate::SymbiosTexturePlugin))
//! writes the resulting albedo, normal, and ORM images directly into the
//! same material — callers never touch raw [`PendingTexture`] entities.
//!
//! Optional caching is provided by inserting a [`TextureCache`] resource
//! before the helper runs.  Cache hits return the previous handles without
//! re-running the generator.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use bevy::asset::Assets;
use bevy::ecs::component::Component;
use bevy::ecs::entity::Entity;
use bevy::ecs::system::{Commands, Query, ResMut};
use bevy::image::Image;
use bevy::pbr::StandardMaterial;
use bevy::prelude::{AlphaMode, Color, Handle};
use bevy::render::render_resource::Face;

use crate::ashlar::AshlarConfig;
use crate::asphalt::AsphaltConfig;
use crate::async_gen::PendingTexture;
use crate::bark::BarkConfig;
use crate::brick::BrickConfig;
use crate::cache::{TextureCache, TextureCacheKey};
use crate::cobblestone::CobblestoneConfig;
use crate::concrete::ConcreteConfig;
use crate::corrugated::CorrugatedConfig;
use crate::encaustic::EncausticConfig;
use crate::generator::{map_to_images, map_to_images_card};
use crate::ground::GroundConfig;
use crate::iron_grille::IronGrilleConfig;
use crate::leaf::LeafConfig;
use crate::marble::MarbleConfig;
use crate::metal::MetalConfig;
use crate::pavers::PaversConfig;
use crate::plank::PlankConfig;
use crate::rock::RockConfig;
use crate::shingle::ShingleConfig;
use crate::stained_glass::StainedGlassConfig;
use crate::stucco::StuccoConfig;
use crate::thatch::ThatchConfig;
use crate::twig::TwigConfig;
use crate::wainscoting::WainscotingConfig;
use crate::window::WindowConfig;

/// PBR rendering hints derived from a [`TextureConfig`] variant.
///
/// Card-style textures (foliage, glass, grilles) need clamp-to-edge sampling
/// and alpha masking; tiling surfaces (bark, brick, plank, …) need opaque
/// rendering with back-face culling and repeat sampling.
#[derive(Copy, Clone, Debug)]
pub struct RenderProperties {
    /// `StandardMaterial::alpha_mode` to apply — `Opaque` for surfaces,
    /// `Mask(0.5)` for alpha-masked cards.
    pub alpha_mode: AlphaMode,
    /// `StandardMaterial::double_sided` flag — `true` for cards so a flat
    /// quad is visible from both sides.
    pub double_sided: bool,
    /// `StandardMaterial::cull_mode` — `Some(Face::Back)` for surfaces,
    /// `None` for cards (no culling, so both sides render).
    pub cull_mode: Option<Face>,
    /// `true` when generated images should be uploaded with
    /// [`map_to_images_card`] (clamp-to-edge); `false` for tiling surfaces.
    pub is_card: bool,
}

/// Default `StandardMaterial` flags for an opaque tiling surface.
fn surface_render_properties() -> RenderProperties {
    RenderProperties {
        alpha_mode: AlphaMode::Opaque,
        double_sided: false,
        cull_mode: Some(Face::Back),
        is_card: false,
    }
}

/// Default `StandardMaterial` flags for an alpha-masked, double-sided card.
fn card_render_properties() -> RenderProperties {
    RenderProperties {
        alpha_mode: AlphaMode::Mask(0.5),
        double_sided: true,
        cull_mode: None,
        is_card: true,
    }
}

/// Generates the [`TextureConfig`] enum and its supporting impls for every
/// generator in the crate.  Each row is `(variant, module, ConfigTy, kind)`.
macro_rules! define_texture_config {
    ($(($variant:ident, $module:ident, $config_ty:ty, $kind:ident)),* $(,)?) => {
        /// Tagged union of every generator config supported by the crate, plus
        /// a [`None`](TextureConfig::None) variant for materials that do not
        /// drive a procedural texture.
        ///
        /// Serialisation is `#[serde(tag = "$type")]` for forward-compat: a
        /// future variant deserialised by an older binary lands in
        /// [`TextureConfig::None`] via the catch-all default.
        #[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
        #[serde(tag = "$type")]
        pub enum TextureConfig {
            /// No procedural texture — leaves the `StandardMaterial`'s
            /// texture slots untouched.
            #[default]
            None,
            $(
                #[doc = concat!("Procedural ", stringify!($variant), " generator config.")]
                $variant($config_ty)
            ),*,
        }

        impl TextureConfig {
            /// Human-readable variant name suitable for UI combo boxes / logs.
            pub fn label(&self) -> &'static str {
                match self {
                    Self::None => "None",
                    $(Self::$variant(_) => stringify!($variant)),*,
                }
            }

            /// PBR rendering properties (alpha mode, culling, card flag) that
            /// match the generator's expected sampling and shading.
            pub fn render_properties(&self) -> RenderProperties {
                match self {
                    Self::None => surface_render_properties(),
                    $(Self::$variant(_) => kind_to_render_properties(TextureKind::$kind)),*,
                }
            }

            /// Submit a generation task for this config at `width × height`.
            ///
            /// Returns `None` for [`TextureConfig::None`].  The returned
            /// [`PendingTexture`] resolves on a background rayon thread (or
            /// inline if pool construction failed; see [`AsyncTextureConfig`]).
            ///
            /// [`AsyncTextureConfig`]: crate::AsyncTextureConfig
            pub fn spawn(&self, width: u32, height: u32) -> Option<PendingTexture> {
                match self {
                    Self::None => None,
                    $(Self::$variant(c) =>
                        Some(PendingTexture::$module(c.clone(), width, height))),*,
                }
            }

            /// Stable per-config fingerprint suitable for cache keys.
            ///
            /// Hashes the `Debug` representation of the variant; the output
            /// is stable for a given config value within one process and
            /// across re-runs of the same binary, which is the contract that
            /// [`TextureCache`](crate::cache::TextureCache) relies on.  It is
            /// **not** stable across compiler versions or struct-field order
            /// changes — bump the manifest version when those change.
            pub fn fingerprint(&self) -> u64 {
                let mut h = DefaultHasher::new();
                // Tag the kind separately so two distinct configs that
                // happen to share a Debug body stay distinguishable.
                self.label().hash(&mut h);
                match self {
                    Self::None => {}
                    $(Self::$variant(c) => format!("{c:?}").hash(&mut h)),*,
                }
                h.finish()
            }
        }
    };
}

/// Internal classification used by [`define_texture_config!`] to pick render
/// properties.  Cards need clamp-to-edge + alpha; surfaces need opaque +
/// back-face culling + repeat tiling.
enum TextureKind {
    Surface,
    Card,
}

fn kind_to_render_properties(kind: TextureKind) -> RenderProperties {
    match kind {
        TextureKind::Surface => surface_render_properties(),
        TextureKind::Card => card_render_properties(),
    }
}

define_texture_config!(
    (Leaf, leaf, LeafConfig, Card),
    (Twig, twig, TwigConfig, Card),
    (Bark, bark, BarkConfig, Surface),
    (Window, window, WindowConfig, Card),
    (StainedGlass, stained_glass, StainedGlassConfig, Card),
    (IronGrille, iron_grille, IronGrilleConfig, Card),
    (Ground, ground, GroundConfig, Surface),
    (Rock, rock, RockConfig, Surface),
    (Brick, brick, BrickConfig, Surface),
    (Plank, plank, PlankConfig, Surface),
    (Shingle, shingle, ShingleConfig, Surface),
    (Stucco, stucco, StuccoConfig, Surface),
    (Concrete, concrete, ConcreteConfig, Surface),
    (Metal, metal, MetalConfig, Surface),
    (Pavers, pavers, PaversConfig, Surface),
    (Ashlar, ashlar, AshlarConfig, Surface),
    (Cobblestone, cobblestone, CobblestoneConfig, Surface),
    (Thatch, thatch, ThatchConfig, Surface),
    (Marble, marble, MarbleConfig, Surface),
    (Corrugated, corrugated, CorrugatedConfig, Surface),
    (Asphalt, asphalt, AsphaltConfig, Surface),
    (Wainscoting, wainscoting, WainscotingConfig, Surface),
    (Encaustic, encaustic, EncausticConfig, Surface),
);

/// PBR material settings driven by a [`TextureConfig`].
///
/// All numeric fields are plain `f32`/`[f32; 3]` — applications that need
/// DAG-CBOR / fixed-point serialisation (e.g. blockchain payloads) should
/// keep their own mirror type and convert at the boundary.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MaterialSettings {
    /// Base colour (linear RGB).
    pub base_color: [f32; 3],
    /// Emission colour (linear RGB) before strength scaling.
    pub emission_color: [f32; 3],
    /// Multiplier applied to `emission_color` when computing the
    /// `StandardMaterial::emissive` linear value.
    pub emission_strength: f32,
    /// Perceptual roughness in `[0, 1]`.
    pub roughness: f32,
    /// Metallic factor in `[0, 1]`.
    pub metallic: f32,
    /// UV repeat scale.  `1.0` means one tile across the mesh; `2.0` means
    /// the texture repeats twice along U and V.
    pub uv_scale: f32,
    /// Procedural texture configuration; [`TextureConfig::None`] leaves the
    /// material's texture slots untouched.
    #[serde(default)]
    pub texture: TextureConfig,
}

impl Default for MaterialSettings {
    fn default() -> Self {
        Self {
            base_color: [0.6, 0.4, 0.2],
            emission_color: [0.0, 0.0, 0.0],
            emission_strength: 0.0,
            roughness: 0.5,
            metallic: 0.0,
            uv_scale: 1.0,
            texture: TextureConfig::None,
        }
    }
}

/// Marker for an in-flight procedural-texture task whose result should be
/// patched directly onto a [`StandardMaterial`].
///
/// Spawned alongside a [`PendingTexture`] by
/// [`build_procedural_material_async`].  Consumed by
/// [`patch_procedural_material_textures`].  Removing or despawning the entity
/// before completion cancels the task (via `PendingTexture`'s drop flag) and
/// leaves the material untouched.
#[derive(Component)]
pub struct PatchMaterialTextures {
    /// Material whose `base_color_texture` / `normal_map_texture` /
    /// `metallic_roughness_texture` slots receive the generated images.
    pub target: Handle<StandardMaterial>,
    /// Cache key the result should be stored under, when a [`TextureCache`]
    /// is present.  `None` disables caching for this task.
    pub cache_key: Option<TextureCacheKey>,
}

/// One-shot helper: build a [`StandardMaterial`] from `settings`, dispatch
/// any required texture generation in the background, and return the handle
/// immediately.
///
/// Texture slots are populated by [`patch_procedural_material_textures`]
/// once the generator finishes — callers can use the returned handle
/// straight away (the material renders with `base_color` / `roughness` /
/// `metallic` until the textures arrive a few frames later).
///
/// If a [`TextureCache`] resource is provided, the cache is consulted
/// before dispatching.  On cache hit the texture handles are written
/// synchronously and no background task is spawned.
///
/// Returns `Handle<StandardMaterial>`.
pub fn build_procedural_material_async(
    commands: &mut Commands,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    cache: Option<&mut TextureCache>,
    settings: &MaterialSettings,
    width: u32,
    height: u32,
) -> Handle<StandardMaterial> {
    let props = settings.texture.render_properties();
    let emissive =
        Color::srgb_from_array(settings.emission_color).to_linear() * settings.emission_strength;

    let mut material = StandardMaterial {
        base_color: Color::srgb_from_array(settings.base_color),
        perceptual_roughness: settings.roughness,
        metallic: settings.metallic,
        emissive,
        alpha_mode: props.alpha_mode,
        double_sided: props.double_sided,
        cull_mode: props.cull_mode,
        ..Default::default()
    };

    let cache_key = if matches!(settings.texture, TextureConfig::None) {
        None
    } else {
        Some(TextureCacheKey {
            kind: settings.texture.label(),
            fingerprint: settings.texture.fingerprint(),
            width,
            height,
        })
    };

    // Cache hit: write handles into the material before we hand it to Bevy.
    if let (Some(key), Some(cache_ref)) = (cache_key.as_ref(), cache.as_deref())
        && let Some(handles) = cache_ref.get_handles(key)
    {
        material.base_color_texture = Some(handles.albedo.clone());
        material.normal_map_texture = Some(handles.normal.clone());
        material.metallic_roughness_texture = Some(handles.roughness.clone());
        return materials.add(material);
    }

    let handle = materials.add(material);

    // Cache miss (or no cache): dispatch generation if a generator is selected.
    if let Some(pending) = settings.texture.spawn(width, height) {
        commands.spawn((
            pending,
            PatchMaterialTextures {
                target: handle.clone(),
                cache_key,
            },
        ));
    }

    // Touch `images` to keep its &mut binding live across the cache lookup
    // — the polling system below is the only consumer that mutates it.
    let _ = images;

    handle
}

/// Bevy system — drains finished [`PendingTexture`]s tagged with
/// [`PatchMaterialTextures`] and writes the generated images straight into
/// the target material's albedo / normal / ORM slots.
///
/// Registered automatically by
/// [`SymbiosTexturePlugin`](crate::SymbiosTexturePlugin) alongside the
/// generic [`poll_texture_tasks`](crate::async_gen::poll_texture_tasks).
/// The two systems are mutually exclusive: an entity with
/// `PatchMaterialTextures` is consumed here and never reaches the generic
/// poller (which only handles bare `PendingTexture` entities).
pub fn patch_procedural_material_textures(
    mut commands: Commands,
    tasks: Query<(Entity, &PendingTexture, &PatchMaterialTextures)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut cache: Option<ResMut<TextureCache>>,
) {
    use std::sync::mpsc::TryRecvError;

    for (entity, pending, patch) in &tasks {
        let poll = pending
            .rx
            .lock()
            .expect("texture thread poisoned")
            .try_recv();

        match poll {
            Ok(Ok(map)) => {
                let is_card = pending.is_card();
                let handles = if is_card {
                    map_to_images_card(map, &mut images)
                } else {
                    map_to_images(map, &mut images)
                };

                if let Some(cache_ref) = cache.as_deref_mut()
                    && let Some(key) = patch.cache_key.clone()
                {
                    cache_ref.insert(key, Arc::new(handles.clone()));
                }

                if let Some(mat) = materials.get_mut(&patch.target) {
                    mat.base_color_texture = Some(handles.albedo);
                    mat.normal_map_texture = Some(handles.normal);
                    mat.metallic_roughness_texture = Some(handles.roughness);
                }
                commands.entity(entity).despawn();
            }
            Ok(Err(e)) => {
                bevy::log::error!("Procedural material texture generation failed: {e}");
                commands.entity(entity).despawn();
            }
            Err(TryRecvError::Disconnected) => {
                bevy::log::error!("Procedural material texture thread panicked");
                commands.entity(entity).despawn();
            }
            Err(TryRecvError::Empty) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant has a distinct, non-empty label.
    #[test]
    fn labels_are_unique_and_non_empty() {
        let configs = [
            TextureConfig::None,
            TextureConfig::Bark(BarkConfig::default()),
            TextureConfig::Leaf(LeafConfig::default()),
            TextureConfig::Brick(BrickConfig::default()),
            TextureConfig::StainedGlass(StainedGlassConfig::default()),
            TextureConfig::IronGrille(IronGrilleConfig::default()),
        ];
        let mut seen = std::collections::HashSet::new();
        for cfg in &configs {
            let label = cfg.label();
            assert!(!label.is_empty());
            assert!(seen.insert(label), "duplicate label: {label}");
        }
    }

    /// Two equal configs hash to the same fingerprint; differing seeds produce
    /// different fingerprints.
    #[test]
    fn fingerprint_is_stable_and_distinguishes_seeds() {
        let a = TextureConfig::Bark(BarkConfig::default());
        let b = TextureConfig::Bark(BarkConfig::default());
        assert_eq!(a.fingerprint(), b.fingerprint());

        let mut differ = BarkConfig::default();
        differ.seed = differ.seed.wrapping_add(1);
        let c = TextureConfig::Bark(differ);
        assert_ne!(a.fingerprint(), c.fingerprint());
    }

    /// Card-style variants emit Mask + double-sided + no culling; surface
    /// variants emit Opaque + back-face culling.
    #[test]
    fn render_properties_match_kind() {
        let card = TextureConfig::Leaf(LeafConfig::default()).render_properties();
        assert!(card.is_card);
        assert!(card.double_sided);
        assert!(card.cull_mode.is_none());

        let surface = TextureConfig::Bark(BarkConfig::default()).render_properties();
        assert!(!surface.is_card);
        assert!(!surface.double_sided);
        assert!(matches!(surface.cull_mode, Some(Face::Back)));
    }

    /// `TextureConfig::None` produces no PendingTexture; every other variant does.
    #[test]
    fn spawn_returns_none_for_none_variant() {
        assert!(TextureConfig::None.spawn(8, 8).is_none());
        assert!(
            TextureConfig::Bark(BarkConfig::default())
                .spawn(8, 8)
                .is_some()
        );
    }
}
