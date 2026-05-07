//! Animates rust coverage on a metal panel from 0% to 100% over 10 seconds.
//!
//! Run with `cargo run --release --example animated_rust`.
//!
//! Demonstrates:
//! * [`AnimatedProceduralMaterial`] — drives a per-frame curve evaluation.
//! * [`Linear`] / [`EaseInOut`] — built-in [`ParameterCurve`] impls.
//! * Throttling via `min_regen_interval` — without it the rayon pool would
//!   thrash regenerating the entire metal texture every frame.
//!
//! The metal panel is built once via
//! [`build_procedural_material_async`](bevy_symbios_texture::build_procedural_material_async)
//! at `t = 0`, then re-textured roughly four times a second as the
//! [`Linear`] rust-curve advances.  At `t >= 10 s` the curve plateaus at
//! `rust_level = 1.0`, the fingerprint stops changing, and the system stops
//! regenerating.

use bevy::prelude::*;
use bevy_symbios_texture::metal::{MetalConfig, MetalStyle};
use bevy_symbios_texture::{
    AnimatedProceduralMaterial, DEFAULT_MEMORY_CACHE_ENTRIES, Linear, MaterialSettings,
    ParameterCurve, SymbiosTexturePlugin, TextureCache, TextureConfig,
    build_procedural_material_async,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SymbiosTexturePlugin::default())
        .insert_resource(TextureCache::memory(DEFAULT_MEMORY_CACHE_ENTRIES))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut cache: ResMut<TextureCache>,
) {
    // Base metal config — same for every frame except `rust_level`.
    let base = MetalConfig {
        style: MetalStyle::Brushed,
        ..MetalConfig::default()
    };

    // Initial material: pristine metal at t = 0.
    let initial_settings = MaterialSettings {
        base_color: [0.85, 0.85, 0.86],
        roughness: 0.55,
        metallic: 0.95,
        texture: TextureConfig::Metal(MetalConfig {
            rust_level: 0.0,
            ..base.clone()
        }),
        ..MaterialSettings::default()
    };
    let material = build_procedural_material_async(
        &mut commands,
        &mut materials,
        &mut images,
        Some(&mut cache),
        &initial_settings,
        512,
        512,
    );

    // Curve: 0 → 1 over 10 seconds.  After 10 s the value clamps at 1.0;
    // the fingerprint stops changing and regeneration halts.
    let rust = Linear {
        from: 0.0_f64,
        to: 1.0,
        duration: 10.0,
    };

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 2.0, 0.1))),
        MeshMaterial3d(material.clone()),
        Transform::from_xyz(0.0, 0.5, 0.0),
        AnimatedProceduralMaterial::new(material, 512, 512, move |t| {
            TextureConfig::Metal(MetalConfig {
                rust_level: rust.eval(t),
                ..base.clone()
            })
        })
        // Quarter-second cadence: smooth enough to read, cheap enough to
        // saturate one rayon thread instead of all four.
        .with_min_regen_interval(0.25),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.5, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            ..default()
        },
        Transform::from_xyz(2.0, 4.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
