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
//! writes the resulting albedo, normal, ORM, and (when produced) emissive
//! images directly into the same material — callers never touch raw
//! [`PendingTexture`] entities.
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
use crate::generator::{GeneratedHandles, TextureMap, map_to_images, map_to_images_card};

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
/// [`for_each_generator!`](symbios_texture::for_each_generator); the
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
                let mut h = symbios_texture::fingerprint::Fnv1a::new();
                // Tag the kind separately so two distinct configs that
                // happen to share a field layout stay distinguishable.
                self.label().hash(&mut h);
                match self {
                    Self::None => {}
                    $(Self::$variant(c) => symbios_texture::fingerprint::hash_value(c, &mut h)),*,
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

symbios_texture::for_each_generator!(define_texture_config);

/// Generates the [`symbios_genetics::Genotype`] dispatch for the
/// [`TextureConfig`] enum from the registry rows: mutation delegates to the
/// wrapped config; crossover recombines like variants field-wise and falls
/// back to a uniform parent pick for mismatched variants (their fields cannot
/// be recombined).
///
/// This lives in the wrapper crate (not the Bevy-free `symbios-texture` core)
/// because `TextureConfig` is defined here — its inherent `spawn`/
/// `render_properties` methods are Bevy-coupled — so the orphan rule requires
/// the `impl Genotype for TextureConfig` to be in this crate too.  The
/// per-config `impl Genotype for FooConfig` blocks stay in the core
/// (`symbios_texture::genetics`).
macro_rules! impl_texture_config_genotype {
    ($(($variant:ident, $module:ident, $config_ty:ty, $generator_ty:ty, $kind:ident)),* $(,)?) => {
        impl symbios_genetics::Genotype for TextureConfig {
            fn mutate<R: rand::Rng>(&mut self, rng: &mut R, rate: f32) {
                match self {
                    TextureConfig::None => {}
                    $(TextureConfig::$variant(c) => c.mutate(rng, rate)),*,
                }
            }

            fn crossover<R: rand::Rng>(&self, other: &Self, rng: &mut R) -> Self {
                match (self, other) {
                    $((TextureConfig::$variant(a), TextureConfig::$variant(b)) =>
                        TextureConfig::$variant(a.crossover(b, rng)),)*
                    (a, b) => {
                        if rng.random::<bool>() {
                            a.clone()
                        } else {
                            b.clone()
                        }
                    }
                }
            }
        }
    };
}

symbios_texture::for_each_generator!(impl_texture_config_genotype);

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

impl MaterialSettings {
    /// Build the [`StandardMaterial`] these settings describe, with the four
    /// texture slots left empty.
    ///
    /// This is the pure half of [`build_procedural_material_async`] — the same
    /// field-for-field construction, without the `&mut Commands` that builder
    /// needs in order to dispatch generation.  A consumer whose textures are
    /// baked somewhere this crate's rayon pool cannot reach (a Web Worker, a
    /// job queue, another process) forks the *dispatch* by calling this and
    /// driving its own baking, and still gets the *appearance* from the one
    /// body the built-in path uses.
    ///
    /// Slots are filled in afterwards, either by
    /// [`patch_procedural_material_textures`] or, for such a consumer, by
    /// [`apply_generated_handles`].
    ///
    /// `uv_transform` is set to a uniform scale of [`uv_scale`]; a caller with
    /// its own UV offset/rotation convention overwrites the field after the
    /// call rather than reimplementing the rest.
    ///
    /// [`uv_scale`]: MaterialSettings::uv_scale
    pub fn standard_material(&self) -> StandardMaterial {
        let props = self.texture.render_properties();
        let emissive =
            Color::srgb_from_array(self.emission_color).to_linear() * self.emission_strength;

        StandardMaterial {
            base_color: Color::srgb_from_array(self.base_color),
            perceptual_roughness: self.roughness,
            metallic: self.metallic,
            emissive,
            alpha_mode: props.alpha_mode,
            double_sided: props.double_sided,
            cull_mode: props.cull_mode,
            uv_transform: Affine2::from_scale(Vec2::splat(self.uv_scale)),
            ..Default::default()
        }
    }

    /// The [`TextureCacheKey`] a bake of these settings at `width` x `height`
    /// should be stored under, or `None` when [`TextureConfig::None`] means
    /// there is nothing to bake.
    ///
    /// The key fingerprints the *config*, not the material, so two materials
    /// differing only in base colour share one baked texture set.
    pub fn cache_key(&self, width: u32, height: u32) -> Option<TextureCacheKey> {
        if matches!(self.texture, TextureConfig::None) {
            return None;
        }
        Some(TextureCacheKey {
            kind: self.texture.label(),
            fingerprint: self.texture.fingerprint(),
            width,
            height,
        })
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
///
/// **Public because a consumer that applies generated textures ITSELF needs
/// it** (#23). A caller that bakes off the main thread — a Web Worker, say —
/// receives finished pixels and writes the slots without ever going through
/// this crate's patch system, and the emissive slot cannot simply be assigned:
/// the factor has to move with it or the glow does not show. Private, this
/// left such a caller no option but to copy the body and keep the copy in
/// step by hand, which is a silent-drift hazard rather than a coupling — a
/// change here would apply to one of that consumer's two rendering paths and
/// not the other, with no compile error and no failing test.
pub fn apply_emissive_map(material: &mut StandardMaterial, emissive: Option<Handle<Image>>) {
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

/// Write a finished [`GeneratedHandles`] set into `material`'s texture slots.
///
/// The three always-present slots are assigned directly; the emissive slot
/// goes through [`apply_emissive_map`], which also moves the emissive *factor*
/// so a generated glow map is actually visible.
///
/// Public for the same reason [`apply_emissive_map`] is: a consumer that bakes
/// off this crate's own pool receives finished pixels and writes the slots
/// itself, and "assign three handles, then call `apply_emissive_map`" is
/// exactly the kind of four-line convention that drifts silently once it is
/// copied rather than called.
pub fn apply_generated_handles(material: &mut StandardMaterial, handles: &GeneratedHandles) {
    material.base_color_texture = Some(handles.albedo.clone());
    material.normal_map_texture = Some(handles.normal.clone());
    material.metallic_roughness_texture = Some(handles.roughness.clone());
    apply_emissive_map(material, handles.emissive.clone());
}

/// Persist, upload and cache a finished [`TextureMap`], returning the handles.
///
/// The callable core of [`patch_procedural_material_textures`]: raw pixels are
/// handed to a disk-backed store *before* the upload (which consumes `map`),
/// the map is uploaded with the sampler mode `is_card` selects, and the
/// resulting handles are written into the cache under `key`.
///
/// The system applies the returned handles to one material; a consumer with
/// several materials awaiting the same bake, or with no `PendingTexture`
/// entity at all because it baked elsewhere, calls this and then
/// [`apply_generated_handles`] per target.  Split out so that ordering — and
/// the fact that `persist_pixels` must precede the upload — lives in one
/// place.
///
/// Pass `key: None` to skip caching entirely.
pub fn store_generated_texture_map(
    map: TextureMap,
    is_card: bool,
    key: Option<&TextureCacheKey>,
    cache: Option<&mut TextureCache>,
    images: &mut Assets<Image>,
) -> GeneratedHandles {
    // Persist raw pixels for disk-backed stores while the map is still
    // available — the upload below consumes it.
    if let (Some(cache_ref), Some(key)) = (cache.as_deref(), key) {
        cache_ref.persist_pixels(key, &map, is_card);
    }

    let handles = if is_card {
        map_to_images_card(map, images)
    } else {
        map_to_images(map, images)
    };

    if let (Some(cache_ref), Some(key)) = (cache, key) {
        cache_ref.insert(key.clone(), Arc::new(handles.clone()));
    }

    handles
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
    let mut material = settings.standard_material();
    let cache_key = settings.cache_key(width, height);

    // Cache hit: write handles into the material before we hand it to Bevy.
    // Full lookup — disk-backed stores read their blob and upload it into
    // `images` here, so a FileStore hit short-circuits generation exactly
    // like a memory hit.
    if let (Some(key), Some(cache_ref)) = (cache_key.as_ref(), cache.as_deref())
        && let Some(handles) = cache_ref.get(key, images)
    {
        apply_generated_handles(&mut material, &handles);
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
                let handles = store_generated_texture_map(
                    map,
                    pending.is_card(),
                    patch.cache_key.as_ref(),
                    cache.as_deref_mut(),
                    &mut images,
                );

                if let Some(mut mat) = materials.get_mut(&patch.target) {
                    // Also defaults the emissive factor to white when a glow
                    // map is present (and undoes it when one is not), so the
                    // map is visible without the caller configuring emission.
                    apply_generated_handles(&mut mat, &handles);
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

    /// The `impl Genotype for TextureConfig` dispatch (kept in this wrapper
    /// crate alongside the enum) delegates mutation to the wrapped config and
    /// recombines like variants field-wise.  Smoke-test both paths.
    #[test]
    fn texture_config_genotype_dispatch() {
        use rand::SeedableRng;
        use symbios_genetics::Genotype;

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);

        // None mutates to itself; carried-config mutation does not panic.
        let mut none = TextureConfig::None;
        none.mutate(&mut rng, 1.0);
        assert!(matches!(none, TextureConfig::None));

        let mut bark = TextureConfig::Bark(BarkConfig::default());
        bark.mutate(&mut rng, 1.0);
        assert!(matches!(bark, TextureConfig::Bark(_)));

        // Like-variant crossover keeps the variant; mismatched variants pick a
        // whole parent.
        let a = TextureConfig::Bark(BarkConfig::default());
        let b = TextureConfig::Bark(BarkConfig {
            seed: 99,
            ..BarkConfig::default()
        });
        assert!(matches!(a.crossover(&b, &mut rng), TextureConfig::Bark(_)));

        let leaf = TextureConfig::Leaf(LeafConfig::default());
        let mixed = a.crossover(&leaf, &mut rng);
        assert!(matches!(
            mixed,
            TextureConfig::Bark(_) | TextureConfig::Leaf(_)
        ));
    }

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
        let (mut commands, mut materials, mut images) = state
            .get_mut(&mut world)
            .expect("builder params are resolvable on the asset world");

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
            let (mut commands, mut materials, mut images) = state
                .get_mut(&mut world)
                .expect("builder params are resolvable on the asset world");
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
            let (mut commands, mut materials, mut images) = state
                .get_mut(&mut world)
                .expect("builder params are resolvable on the asset world");
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

    // -----------------------------------------------------------------
    // The pure halves of the async builder (#102): a consumer that has to
    // fork *dispatch* — a wasm build baking in a Web Worker, say — must be
    // able to reach the *appearance* without reimplementing it.  These
    // tests pin the contract that consumer relies on.
    // -----------------------------------------------------------------

    /// `standard_material` carries every field the async builder used to
    /// build inline, and leaves the texture slots for the caller to fill.
    #[test]
    fn standard_material_carries_every_appearance_field() {
        let settings = MaterialSettings {
            base_color: [0.2, 0.6, 0.9],
            emission_color: [0.9, 0.3, 0.1],
            emission_strength: 2.5,
            roughness: 0.35,
            metallic: 0.8,
            uv_scale: 3.0,
            texture: TextureConfig::None,
        };
        let mat = settings.standard_material();

        assert_eq!(mat.base_color, Color::srgb_from_array([0.2, 0.6, 0.9]));
        assert_eq!(mat.perceptual_roughness, 0.35);
        assert_eq!(mat.metallic, 0.8);
        assert_eq!(
            mat.emissive,
            Color::srgb_from_array([0.9, 0.3, 0.1]).to_linear() * 2.5,
            "emissive is the emission colour scaled by strength"
        );
        assert_eq!(mat.uv_transform, Affine2::from_scale(Vec2::splat(3.0)));
        assert!(
            mat.base_color_texture.is_none()
                && mat.normal_map_texture.is_none()
                && mat.metallic_roughness_texture.is_none()
                && mat.emissive_texture.is_none(),
            "slots are filled later, by the patch system or the caller"
        );
    }

    /// The render properties a config carries (alpha mode, culling) reach the
    /// material through `standard_material`, so a card built this way is a
    /// card.
    #[test]
    fn standard_material_takes_render_properties_from_the_config() {
        use crate::bark::BarkConfig;
        use crate::leaf::LeafConfig;

        let surface = MaterialSettings {
            texture: TextureConfig::Bark(BarkConfig::default()),
            ..MaterialSettings::default()
        }
        .standard_material();
        let card = MaterialSettings {
            texture: TextureConfig::Leaf(LeafConfig::default()),
            ..MaterialSettings::default()
        }
        .standard_material();

        let surface_props = TextureConfig::Bark(BarkConfig::default()).render_properties();
        let card_props = TextureConfig::Leaf(LeafConfig::default()).render_properties();

        assert_eq!(surface.alpha_mode, surface_props.alpha_mode);
        assert_eq!(surface.double_sided, surface_props.double_sided);
        assert_eq!(surface.cull_mode, surface_props.cull_mode);
        assert_eq!(card.alpha_mode, card_props.alpha_mode);
        assert_eq!(card.double_sided, card_props.double_sided);
        assert_eq!(card.cull_mode, card_props.cull_mode);
        assert_ne!(
            surface.alpha_mode, card.alpha_mode,
            "precondition: the two shapes actually differ"
        );
    }

    /// The builder and the pure half must not drift: what
    /// `build_procedural_material_async` adds to `Assets` is exactly what
    /// `standard_material` returns, and its key is `cache_key`'s.
    #[test]
    fn async_builder_and_pure_half_agree() {
        use crate::bark::BarkConfig;
        use bevy::ecs::system::SystemState;
        use bevy::ecs::world::World;

        let settings = MaterialSettings {
            base_color: [0.1, 0.2, 0.3],
            emission_color: [0.4, 0.5, 0.6],
            emission_strength: 1.5,
            roughness: 0.25,
            metallic: 0.75,
            uv_scale: 4.0,
            texture: TextureConfig::Bark(BarkConfig::default()),
        };

        let mut world = World::new();
        world.init_resource::<Assets<StandardMaterial>>();
        world.init_resource::<Assets<Image>>();
        /// The parameter set `build_procedural_material_async` needs, named
        /// so the `SystemState` below stays inside `clippy::type_complexity`.
        type BuilderParams<'w, 's> = (
            Commands<'w, 's>,
            ResMut<'w, Assets<StandardMaterial>>,
            ResMut<'w, Assets<Image>>,
        );

        let mut state: SystemState<BuilderParams> = SystemState::new(&mut world);
        let (mut commands, mut materials, mut images) = state
            .get_mut(&mut world)
            .expect("asset world resolves the builder params");
        let handle = build_procedural_material_async(
            &mut commands,
            &mut materials,
            &mut images,
            None,
            &settings,
            64,
            32,
        );
        state.apply(&mut world);

        let materials = world.resource::<Assets<StandardMaterial>>();
        let built = materials.get(&handle).expect("added synchronously");
        let pure = settings.standard_material();

        assert_eq!(built.base_color, pure.base_color);
        assert_eq!(built.perceptual_roughness, pure.perceptual_roughness);
        assert_eq!(built.metallic, pure.metallic);
        assert_eq!(built.emissive, pure.emissive);
        assert_eq!(built.alpha_mode, pure.alpha_mode);
        assert_eq!(built.double_sided, pure.double_sided);
        assert_eq!(built.cull_mode, pure.cull_mode);
        assert_eq!(built.uv_transform, pure.uv_transform);

        // …and the dispatched task carries the key `cache_key` names.
        let mut patches = world.query::<&PatchMaterialTextures>();
        let patch = patches.iter(&world).next().expect("bark dispatches a bake");
        assert_eq!(patch.cache_key, settings.cache_key(64, 32));
    }

    /// `cache_key` fingerprints the config at the requested dimensions, and
    /// declines to key a material that has nothing to bake.
    #[test]
    fn cache_key_fingerprints_the_config_and_skips_none() {
        use crate::bark::BarkConfig;

        let bark = MaterialSettings {
            texture: TextureConfig::Bark(BarkConfig::default()),
            ..MaterialSettings::default()
        };
        let key = bark
            .cache_key(256, 128)
            .expect("a bakeable config is keyed");
        assert_eq!(key.kind, bark.texture.label());
        assert_eq!(key.fingerprint, bark.texture.fingerprint());
        assert_eq!((key.width, key.height), (256, 128));

        assert!(
            MaterialSettings::default().cache_key(256, 128).is_none(),
            "TextureConfig::None has nothing to bake, so nothing to key"
        );

        // Base colour is not part of the identity: two materials differing
        // only in appearance share one baked texture set.
        let tinted = MaterialSettings {
            base_color: [1.0, 0.0, 0.0],
            ..bark.clone()
        };
        assert_eq!(tinted.cache_key(256, 128), bark.cache_key(256, 128));
    }

    /// `apply_generated_handles` writes all four slots, and the emissive
    /// factor moves with the glow map (the half a copy of the three plain
    /// assignments would silently omit).
    #[test]
    fn apply_generated_handles_writes_every_slot_and_moves_the_factor() {
        let handles = GeneratedHandles {
            albedo: dummy_image_handle(),
            normal: dummy_image_handle(),
            roughness: dummy_image_handle(),
            emissive: Some(dummy_image_handle()),
        };
        let mut mat = StandardMaterial::default();
        assert_eq!(mat.emissive, LinearRgba::BLACK, "precondition");

        apply_generated_handles(&mut mat, &handles);

        assert!(mat.base_color_texture.is_some());
        assert!(mat.normal_map_texture.is_some());
        assert!(mat.metallic_roughness_texture.is_some());
        assert!(mat.emissive_texture.is_some());
        assert_eq!(
            mat.emissive,
            LinearRgba::WHITE,
            "the glow map is invisible unless the factor moves with it"
        );

        // No glow map: the three plain slots still land, and the auto-white
        // is undone rather than left emitting.
        let dark = GeneratedHandles {
            emissive: None,
            ..handles
        };
        apply_generated_handles(&mut mat, &dark);
        assert!(mat.emissive_texture.is_none());
        assert_eq!(mat.emissive, LinearRgba::BLACK);
    }

    fn flat_map(width: u32, height: u32, emissive: bool) -> TextureMap {
        let px = (width * height * 4) as usize;
        TextureMap {
            albedo: vec![200; px],
            normal: vec![128; px],
            roughness: vec![64; px],
            emissive: emissive.then(|| vec![255; px]),
            width,
            height,
            mip_level_count: 1,
        }
    }

    /// `store_generated_texture_map` uploads the map, and — when a key and a
    /// cache are supplied — leaves the result retrievable, so a consumer
    /// baking outside this crate shares the cache with the built-in path.
    #[test]
    fn store_generated_texture_map_uploads_and_caches() {
        let mut images = Assets::<Image>::default();
        let mut cache = TextureCache::memory(8);
        let key = TextureCacheKey {
            kind: "test",
            fingerprint: 42,
            width: 4,
            height: 4,
        };

        let handles = store_generated_texture_map(
            flat_map(4, 4, true),
            false,
            Some(&key),
            Some(&mut cache),
            &mut images,
        );
        assert!(images.get(&handles.albedo).is_some());
        assert!(images.get(&handles.normal).is_some());
        assert!(images.get(&handles.roughness).is_some());
        assert!(
            handles.emissive.is_some(),
            "the glow map survives the upload"
        );

        let hit = cache.get(&key, &mut images).expect("the bake is cached");
        assert_eq!(hit.albedo, handles.albedo);
        assert_eq!(hit.emissive, handles.emissive);
    }

    /// A `None` key opts out of caching entirely without changing the upload.
    #[test]
    fn store_generated_texture_map_without_a_key_skips_the_cache() {
        let mut images = Assets::<Image>::default();
        let mut cache = TextureCache::memory(8);

        let would_have_been = TextureCacheKey {
            kind: "test",
            fingerprint: 7,
            width: 4,
            height: 4,
        };

        let handles = store_generated_texture_map(
            flat_map(4, 4, false),
            true,
            None,
            Some(&mut cache),
            &mut images,
        );

        assert!(images.get(&handles.albedo).is_some());
        assert!(handles.emissive.is_none());
        assert!(
            cache.get(&would_have_been, &mut images).is_none(),
            "an unkeyed bake must not enter the cache under any key"
        );
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
        let (mut commands, mut materials, mut images) = state
            .get_mut(&mut world)
            .expect("builder params are resolvable on the asset world");
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
