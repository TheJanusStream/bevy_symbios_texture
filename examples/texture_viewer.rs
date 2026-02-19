//! `texture_viewer` — displays all three generators side-by-side in a window.
//!
//! Run with:
//!   cargo run --example texture_viewer

use bevy::prelude::*;
use bevy_symbios_texture::{
    SymbiosTexturePlugin,
    async_gen::{PendingTexture, TextureReady},
    bark::BarkConfig,
    ground::GroundConfig,
    rock::RockConfig,
};

const TEX_SIZE: u32 = 512;
const SPACING: f32 = TEX_SIZE as f32 + 20.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_symbios_texture — viewer".into(),
                resolution: ((SPACING * 3.0 + 40.0) as u32, (TEX_SIZE + 80)).into(),
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
}

fn show_ready_textures(
    mut commands: Commands,
    ready: Query<(Entity, &TextureReady, &TextureSlot)>,
    mut shown: Local<Vec<Entity>>,
) {
    for (entity, ready_tex, slot) in &ready {
        if shown.contains(&entity) {
            continue;
        }
        shown.push(entity);

        let x = (slot.0 as f32 - 1.0) * SPACING;

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
            _ => "Ground",
        };
        commands.spawn((
            Text2d::new(label),
            Transform::from_translation(Vec3::new(x, -(TEX_SIZE as f32 * 0.5 + 18.0), 0.0)),
        ));
    }
}
