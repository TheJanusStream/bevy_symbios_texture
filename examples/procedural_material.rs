//! Demonstrates [`build_procedural_material_async`] vs the manual
//! `PendingTexture` + `Handle<StandardMaterial>` patching dance.
//!
//! Run with `cargo run --release --example procedural_material`.
//!
//! The left cube is built via the one-call helper:
//! ```ignore
//! let material = build_procedural_material_async(
//!     &mut commands, &mut materials, &mut images, cache,
//!     &settings, 512, 512,
//! );
//! ```
//!
//! The right cube reproduces the equivalent flow by hand: spawn a
//! `PendingTexture::brick`, pre-allocate a `StandardMaterial` handle, then
//! query the entity in a custom system to write the textures back into the
//! material when generation completes.  The helper collapses ~40 lines of
//! per-call-site plumbing.

use bevy::prelude::*;
use bevy_symbios_texture::async_gen::{PendingTexture, TextureReady};
use bevy_symbios_texture::brick::BrickConfig;
use bevy_symbios_texture::{
    DEFAULT_MEMORY_CACHE_ENTRIES, MaterialSettings, SymbiosTexturePlugin, TextureCache,
    TextureConfig, build_procedural_material_async,
};

#[derive(Component)]
struct ManualBrickTask(Handle<StandardMaterial>);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SymbiosTexturePlugin::default())
        // Enables hit-detection for the helper.  Try toggling this off and
        // observe both cubes regenerate from scratch on every restart.
        .insert_resource(TextureCache::memory(DEFAULT_MEMORY_CACHE_ENTRIES))
        .add_systems(Startup, setup)
        .add_systems(Update, drain_manual_task)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut cache: ResMut<TextureCache>,
) {
    let cube = meshes.add(Cuboid::default());

    // ── Left cube: one-call helper ────────────────────────────────────────
    let settings = MaterialSettings {
        base_color: [0.65, 0.30, 0.20],
        roughness: 0.85,
        metallic: 0.0,
        texture: TextureConfig::Brick(BrickConfig::default()),
        ..MaterialSettings::default()
    };
    let helper_material = build_procedural_material_async(
        &mut commands,
        &mut materials,
        &mut images,
        Some(&mut cache),
        &settings,
        512,
        512,
    );
    commands.spawn((
        Mesh3d(cube.clone()),
        MeshMaterial3d(helper_material),
        Transform::from_xyz(-1.5, 0.0, 0.0),
    ));

    // ── Right cube: manual flow for comparison ────────────────────────────
    let manual_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.65, 0.30, 0.20),
        perceptual_roughness: 0.85,
        ..default()
    });
    commands.spawn((
        Mesh3d(cube),
        MeshMaterial3d(manual_material.clone()),
        Transform::from_xyz(1.5, 0.0, 0.0),
    ));
    commands.spawn((
        PendingTexture::brick(BrickConfig::default(), 512, 512),
        ManualBrickTask(manual_material),
    ));

    // Camera + light.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_xyz(2.0, 4.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Manual-flow polling: when the bare `PendingTexture` resolves it surfaces a
/// `TextureReady` component; we copy the handles into the pre-allocated
/// material here.  The helper does this for you.
fn drain_manual_task(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    ready: Query<(Entity, &TextureReady, &ManualBrickTask)>,
) {
    for (entity, ready, manual) in &ready {
        if let Some(mat) = materials.get_mut(&manual.0) {
            mat.base_color_texture = Some(ready.0.albedo.clone());
            mat.normal_map_texture = Some(ready.0.normal.clone());
            mat.metallic_roughness_texture = Some(ready.0.roughness.clone());
        }
        commands.entity(entity).despawn();
    }
}
