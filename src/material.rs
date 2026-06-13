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

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use bevy::asset::Assets;
use bevy::ecs::component::Component;
use bevy::ecs::entity::Entity;
use bevy::ecs::system::{Commands, Query, ResMut};
use bevy::image::Image;
use bevy::math::{Affine2, Vec2};
use bevy::pbr::StandardMaterial;
use bevy::prelude::{AlphaMode, Color, Handle, LinearRgba};
use bevy::render::render_resource::Face;

use crate::async_gen::PendingTexture;
use crate::cache::{TextureCache, TextureCacheKey};
use crate::generator::{map_to_images, map_to_images_card};

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
/// generator in the crate.  Receives the registry rows
/// `(Variant, module, ConfigTy, GeneratorTy, Kind)` from
/// [`for_each_generator!`](crate::registry::for_each_generator); the
/// generator type is unused here (the async constructors consume it).
macro_rules! define_texture_config {
    ($(($variant:ident, $module:ident, $config_ty:ty, $generator_ty:ty, $kind:ident)),* $(,)?) => {
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

            /// One default-config instance of every generator variant, in
            /// registry order.
            ///
            /// Drives generator dropdowns and benchmark suites without a
            /// hand-maintained list — new registry rows appear here
            /// automatically.  [`TextureConfig::None`] is not included.
            pub fn all_defaults() -> Vec<TextureConfig> {
                vec![$(Self::$variant(<$config_ty>::default())),*]
            }

            /// The generator's module name (e.g. `"stained_glass"`) — the
            /// snake_case counterpart of [`label`](TextureConfig::label),
            /// useful for stable benchmark and file identifiers.
            pub fn module_name(&self) -> &'static str {
                match self {
                    Self::None => "none",
                    $(Self::$variant(_) => stringify!($module)),*,
                }
            }

            /// Run the generator synchronously on the calling thread.
            ///
            /// Returns `None` for [`TextureConfig::None`].  The generator is
            /// constructed per call — microseconds of setup against the
            /// milliseconds of pixel work; hold a concrete generator
            /// (e.g. [`BarkGenerator`](crate::bark::BarkGenerator)) when
            /// producing many size variants of one config.
            pub fn generate_sync(
                &self,
                width: u32,
                height: u32,
            ) -> Option<Result<crate::generator::TextureMap, crate::generator::TextureError>> {
                use crate::generator::TextureGenerator as _;
                match self {
                    Self::None => None,
                    $(Self::$variant(c) =>
                        Some(<$generator_ty>::new(c.clone()).generate(width, height))),*,
                }
            }

            /// Stable per-config fingerprint suitable for cache keys.
            ///
            /// Structurally hashes the config through its serde
            /// representation (every primitive as little-endian bits, floats
            /// via `to_bits`) with a fixed FNV-1a hasher — no allocation,
            /// and the output is stable across runs, Rust versions, and
            /// platforms.  The fingerprint rolls only when field values
            /// change, fields are added/removed, or the serde representation
            /// changes; for generator-*internal* changes bump
            /// [`TextureCache::manifest_version`](crate::cache::TextureCache::manifest_version)
            /// instead.
            pub fn fingerprint(&self) -> u64 {
                let mut h = crate::fingerprint::Fnv1a::new();
                // Tag the kind separately so two distinct configs that
                // happen to share a field layout stay distinguishable.
                self.label().hash(&mut h);
                match self {
                    Self::None => {}
                    $(Self::$variant(c) => crate::fingerprint::hash_value(c, &mut h)),*,
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

crate::registry::for_each_generator!(define_texture_config);

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
    ///
    /// Bevy multiplies any generator-produced emissive *texture* by this
    /// factor.  When a generator emits a glow map (e.g.
    /// [`Lava`](crate::lava::LavaGenerator)) and you leave both this and
    /// `emission_color` at their defaults, the factor is auto-defaulted to
    /// white so the map shows at its encoded values — set a non-default
    /// `emission_color` / `emission_strength` here only to tint or brighten
    /// (e.g. above 1.0 for HDR bloom).
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

/// Assign a generator-produced emissive map to `material`, defaulting the
/// emissive *factor* so the glow is actually visible.
///
/// Bevy multiplies `emissive_texture` by [`StandardMaterial::emissive`], so
/// the black factor that a non-emissive material carries by default would
/// hide the map entirely.  This helper treats a *white* factor as the
/// sentinel for "auto-enabled for a generated glow map":
///
/// * a map arrived while the factor is still the black default → set it to
///   white so the map shows at its encoded values;
/// * no map arrived but the factor is the auto-white → reset it to black so
///   an animated regeneration that drops the glow does not leave the whole
///   surface emitting white.
///
/// A caller-supplied non-black, non-white factor (e.g. a tinted or
/// brightened glow set via [`MaterialSettings::emission_color`] /
/// [`emission_strength`](MaterialSettings::emission_strength)) is left
/// untouched in both directions.
fn apply_emissive_map(material: &mut StandardMaterial, emissive: Option<Handle<Image>>) {
    // Compare RGB only: the emissive factor's alpha is not used for emission,
    // and `emission_color × emission_strength` yields `{0,0,0,0}` (alpha 0) at
    // the defaults — distinct from `LinearRgba::BLACK` (alpha 1).  White is
    // the sentinel meaning "auto-enabled for a generated glow map".
    let e = material.emissive;
    let factor_is_unset = e.red == 0.0 && e.green == 0.0 && e.blue == 0.0;
    let factor_is_auto_white = e.red == 1.0 && e.green == 1.0 && e.blue == 1.0;
    match &emissive {
        Some(_) if factor_is_unset => material.emissive = LinearRgba::WHITE,
        None if factor_is_auto_white => material.emissive = LinearRgba::BLACK,
        _ => {}
    }
    material.emissive_texture = emissive;
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
/// synchronously and no background task is spawned.  Disk-backed stores
/// ([`FileStore`](crate::cache::FileStore)) upload their persisted pixels
/// into `images` during that lookup, so hits survive process restarts.
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
        uv_transform: Affine2::from_scale(Vec2::splat(settings.uv_scale)),
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
    // Full lookup — disk-backed stores read their blob and upload it into
    // `images` here, so a FileStore hit short-circuits generation exactly
    // like a memory hit.
    if let (Some(key), Some(cache_ref)) = (cache_key.as_ref(), cache.as_deref())
        && let Some(handles) = cache_ref.get(key, images)
    {
        material.base_color_texture = Some(handles.albedo.clone());
        material.normal_map_texture = Some(handles.normal.clone());
        material.metallic_roughness_texture = Some(handles.roughness.clone());
        apply_emissive_map(&mut material, handles.emissive.clone());
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

                // Persist raw pixels for disk-backed stores while the map is
                // still available — the upload below consumes it.
                if let Some(cache_ref) = cache.as_deref()
                    && let Some(key) = patch.cache_key.as_ref()
                {
                    cache_ref.persist_pixels(key, &map, is_card);
                }

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
                    // Defaults the emissive factor to white when a glow map
                    // is present (and undoes it when one is not), so the map
                    // is visible without the caller configuring emission.
                    apply_emissive_map(mat, handles.emissive);
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

    use crate::bark::BarkConfig;
    use crate::brick::BrickConfig;
    use crate::iron_grille::IronGrilleConfig;
    use crate::leaf::LeafConfig;
    use crate::stained_glass::StainedGlassConfig;

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

    /// Cross-version stability canary: the fingerprint of a fixed config is
    /// pinned.  If this fails, the hashing scheme or the config's field set
    /// changed — both invalidate `FileStore` caches, so note it in the
    /// CHANGELOG and re-pin.
    #[test]
    fn fingerprint_is_pinned_for_known_config() {
        let fp = TextureConfig::Bark(BarkConfig::default()).fingerprint();
        println!("bark default fingerprint: {fp:#018x}");
        assert_eq!(fp, GOLDEN_BARK_FINGERPRINT);
    }

    const GOLDEN_BARK_FINGERPRINT: u64 = 0xf63c_f22d_3946_c257;

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

    use bevy::ecs::system::SystemState;
    use bevy::ecs::world::World;

    /// Minimal world with the asset containers the builder needs.
    fn asset_world() -> World {
        let mut world = World::new();
        world.insert_resource(Assets::<StandardMaterial>::default());
        world.insert_resource(Assets::<Image>::default());
        world
    }

    /// The system params `build_procedural_material_async` consumes.
    type BuilderParams = (
        Commands<'static, 'static>,
        ResMut<'static, Assets<StandardMaterial>>,
        ResMut<'static, Assets<Image>>,
    );

    /// `uv_scale` lands on `StandardMaterial::uv_transform`; the default of
    /// `1.0` produces the identity transform.
    #[test]
    fn uv_scale_is_applied_to_uv_transform() {
        let mut world = asset_world();
        let mut state: SystemState<BuilderParams> = SystemState::new(&mut world);
        let (mut commands, mut materials, mut images) = state.get_mut(&mut world);

        let scaled = MaterialSettings {
            uv_scale: 2.0,
            ..MaterialSettings::default()
        };
        let handle = build_procedural_material_async(
            &mut commands,
            &mut materials,
            &mut images,
            None,
            &scaled,
            8,
            8,
        );
        let mat = materials.get(&handle).expect("material registered");
        assert_eq!(mat.uv_transform, Affine2::from_scale(Vec2::splat(2.0)));

        let default = MaterialSettings::default();
        let handle = build_procedural_material_async(
            &mut commands,
            &mut materials,
            &mut images,
            None,
            &default,
            8,
            8,
        );
        let mat = materials.get(&handle).expect("material registered");
        assert_eq!(mat.uv_transform, Affine2::IDENTITY);

        state.apply(&mut world);
    }

    /// End-to-end FileStore regression test through the real plugin systems:
    /// pass 1 generates and persists to disk; pass 2 (fresh world, same
    /// directory) must hit the blob synchronously and dispatch no task.
    #[test]
    fn file_cache_round_trips_through_material_flow() {
        let dir = std::env::temp_dir().join(format!("bst-matflow-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let settings = MaterialSettings {
            texture: TextureConfig::Bark(BarkConfig::default()),
            ..MaterialSettings::default()
        };

        // Pass 1: cold cache — generate through the patch system.
        {
            let mut world = asset_world();
            world.insert_resource(TextureCache::file(dir.clone(), 0).expect("create cache dir"));

            let mut state: SystemState<BuilderParams> = SystemState::new(&mut world);
            let (mut commands, mut materials, mut images) = state.get_mut(&mut world);
            let handle = build_procedural_material_async(
                &mut commands,
                &mut materials,
                &mut images,
                None,
                &settings,
                8,
                8,
            );
            state.apply(&mut world);

            let mut schedule = bevy::ecs::schedule::Schedule::default();
            schedule.add_systems(patch_procedural_material_textures);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                schedule.run(&mut world);
                let patched = world
                    .resource::<Assets<StandardMaterial>>()
                    .get(&handle)
                    .is_some_and(|m| m.base_color_texture.is_some());
                if patched {
                    break;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "texture generation timed out"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }

        let blobs = std::fs::read_dir(&dir).expect("cache dir readable").count();
        assert_eq!(blobs, 1, "FileStore must have persisted exactly one blob");

        // Pass 2: fresh world, same directory — synchronous disk hit.
        {
            let mut world = asset_world();
            let mut cache = TextureCache::file(dir.clone(), 0).expect("reopen cache dir");

            let mut state: SystemState<BuilderParams> = SystemState::new(&mut world);
            let (mut commands, mut materials, mut images) = state.get_mut(&mut world);
            let handle = build_procedural_material_async(
                &mut commands,
                &mut materials,
                &mut images,
                Some(&mut cache),
                &settings,
                8,
                8,
            );
            let mat = materials.get(&handle).expect("material registered");
            assert!(
                mat.base_color_texture.is_some(),
                "disk hit must populate the albedo slot synchronously"
            );
            assert!(mat.normal_map_texture.is_some());
            assert!(mat.metallic_roughness_texture.is_some());
            state.apply(&mut world);

            let mut pending = world.query::<&PendingTexture>();
            assert_eq!(
                pending.iter(&world).count(),
                0,
                "a cache hit must not dispatch a generation task"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn dummy_image_handle() -> Handle<Image> {
        Handle::<Image>::default()
    }

    #[test]
    fn apply_emissive_map_auto_enables_white_for_a_glow_map() {
        let mut mat = StandardMaterial::default();
        assert_eq!(mat.emissive, LinearRgba::BLACK, "precondition");
        apply_emissive_map(&mut mat, Some(dummy_image_handle()));
        assert_eq!(
            mat.emissive,
            LinearRgba::WHITE,
            "a glow map with the black default factor must auto-enable white"
        );
        assert!(mat.emissive_texture.is_some());
    }

    #[test]
    fn apply_emissive_map_respects_a_caller_supplied_factor() {
        let tint = LinearRgba::new(0.2, 0.4, 0.6, 1.0);
        let mut mat = StandardMaterial {
            emissive: tint,
            ..Default::default()
        };
        apply_emissive_map(&mut mat, Some(dummy_image_handle()));
        assert_eq!(
            mat.emissive, tint,
            "a non-default factor must be left alone"
        );
    }

    #[test]
    fn apply_emissive_map_undoes_auto_white_when_the_map_drops() {
        // Simulate an animated regeneration: first a glow map (auto-white),
        // then a frame with no emissive — the factor must reset so the whole
        // surface does not emit white.
        let mut mat = StandardMaterial::default();
        apply_emissive_map(&mut mat, Some(dummy_image_handle()));
        assert_eq!(mat.emissive, LinearRgba::WHITE);
        apply_emissive_map(&mut mat, None);
        assert_eq!(mat.emissive, LinearRgba::BLACK, "auto-white must be undone");
        assert!(mat.emissive_texture.is_none());
    }

    /// End-to-end: a lava material built with default `MaterialSettings`
    /// (no emission configured) must end up with both an emissive texture
    /// and a non-black emissive factor after the patch system runs, so the
    /// glow is actually visible.
    #[test]
    fn lava_glow_is_visible_with_default_material_settings() {
        use crate::lava::LavaConfig;

        let settings = MaterialSettings {
            texture: TextureConfig::Lava(LavaConfig::default()),
            ..MaterialSettings::default()
        };
        assert_eq!(settings.emission_strength, 0.0, "precondition: no emission");

        let mut world = asset_world();
        let mut state: SystemState<BuilderParams> = SystemState::new(&mut world);
        let (mut commands, mut materials, mut images) = state.get_mut(&mut world);
        let handle = build_procedural_material_async(
            &mut commands,
            &mut materials,
            &mut images,
            None,
            &settings,
            16,
            16,
        );
        // Before generation finishes the factor's RGB is the unset (zero)
        // default and no emissive map is present.
        let before = materials.get(&handle).unwrap();
        assert_eq!(
            (
                before.emissive.red,
                before.emissive.green,
                before.emissive.blue
            ),
            (0.0, 0.0, 0.0)
        );
        assert!(before.emissive_texture.is_none());
        state.apply(&mut world);

        let mut schedule = bevy::ecs::schedule::Schedule::default();
        schedule.add_systems(patch_procedural_material_textures);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            schedule.run(&mut world);
            let done = world
                .resource::<Assets<StandardMaterial>>()
                .get(&handle)
                .is_some_and(|m| m.emissive_texture.is_some());
            if done {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "lava generation timed out"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let mat = world
            .resource::<Assets<StandardMaterial>>()
            .get(&handle)
            .unwrap();
        assert!(
            mat.emissive_texture.is_some(),
            "lava must set an emissive map"
        );
        assert_eq!(
            mat.emissive,
            LinearRgba::WHITE,
            "emissive factor must auto-default to white so the glow shows"
        );
    }
}
