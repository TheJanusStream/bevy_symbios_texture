//! `texture_viewer` — displays all five generators side-by-side in a window.
//!
//! Run with:
//!   cargo run --example texture_viewer

use bevy::prelude::*;
use bevy_symbios_texture::{
    SymbiosTexturePlugin,
    async_gen::{PendingTexture, TextureReady},
    bark::BarkConfig,
    ground::GroundConfig,
    leaf::LeafConfig,
    rock::RockConfig,
    twig::TwigConfig,
};

const TEX_SIZE: u32 = 512;
const N_PANELS: usize = 5;
const SPACING: f32 = TEX_SIZE as f32 + 20.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_symbios_texture — viewer".into(),
                resolution: ((SPACING * N_PANELS as f32 + 40.0) as u32, TEX_SIZE + 80).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(SymbiosTexturePlugin)
        .add_systems(Startup, spawn_tasks)
        .add_systems(Update, show_ready_textures)
        .run();
}

/// Marker so we know which slot to place a finished texture in.
#[derive(Component)]
struct TextureSlot(usize);

fn spawn_tasks(mut commands: Commands) {
    commands.spawn(Camera2d);

    commands.spawn((
        PendingTexture::bark(BarkConfig::default(), TEX_SIZE, TEX_SIZE),
        TextureSlot(0),
    ));
    commands.spawn((
        PendingTexture::rock(RockConfig::default(), TEX_SIZE, TEX_SIZE),
        TextureSlot(1),
    ));
    commands.spawn((
        PendingTexture::ground(GroundConfig::default(), TEX_SIZE, TEX_SIZE),
        TextureSlot(2),
    ));
    commands.spawn((
        PendingTexture::leaf(LeafConfig::default(), TEX_SIZE, TEX_SIZE),
        TextureSlot(3),
    ));
    commands.spawn((
        PendingTexture::twig(TwigConfig::default(), TEX_SIZE, TEX_SIZE),
        TextureSlot(4),
    ));
}

fn show_ready_textures(
    mut commands: Commands,
    ready: Query<(Entity, &TextureReady, &TextureSlot)>,
) {
    for (entity, ready_tex, slot) in &ready {
        // Despawn the task entity so this query won't match it again next frame.
        commands.entity(entity).despawn();

        // Centre each panel evenly across the window.
        let x = (slot.0 as f32 - (N_PANELS as f32 - 1.0) * 0.5) * SPACING;

        commands.spawn((
            Sprite {
                image: ready_tex.0.albedo.clone(),
                custom_size: Some(Vec2::splat(TEX_SIZE as f32)),
                ..default()
            },
            Transform::from_translation(Vec3::new(x, 0.0, 0.0)),
        ));

        let label = match slot.0 {
            0 => "Bark",
            1 => "Rock",
            2 => "Ground",
            3 => "Leaf",
            _ => "Twig",
        };
        commands.spawn((
            Text2d::new(label),
            Transform::from_translation(Vec3::new(x, -(TEX_SIZE as f32 * 0.5 + 18.0), 0.0)),
        ));
    }
}
