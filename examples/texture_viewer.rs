//! `texture_viewer` — displays all five generators side-by-side in a window.
//!
//! **Click any texture panel to mutate it** and regenerate with the new
//! parameters.  Each click applies a random perturbation (rate = 0.3) to
//! every parameter of the corresponding generator config.
//!
//! Run with:
//!   cargo run --example texture_viewer

use bevy::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use symbios_genetics::Genotype;

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

/// Y-centre of the albedo row (upper) and normal-map row (lower).
const ALBEDO_Y: f32 = TEX_SIZE as f32 * 0.5 + 20.0;
const NORMAL_Y: f32 = -(TEX_SIZE as f32 * 0.5 + 20.0);

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "bevy_symbios_texture — click to mutate".into(),
                    resolution: (
                        (SPACING * N_PANELS as f32 + 40.0) as u32,
                        TEX_SIZE * 2 + 120,
                    )
                        .into(),
                    ..default()
                }),
                ..default()
            }),
        )
        .add_plugins(SymbiosTexturePlugin)
        .add_systems(Startup, spawn_tasks)
        .add_systems(Update, (show_ready_textures, handle_click))
        .run();
}

// --- panel config -----------------------------------------------------------

/// Texture configuration stored on both task entities and live panel sprites.
///
/// Carried along so that `handle_click` can mutate and re-queue any panel
/// without needing a separate registry.
#[derive(Component, Clone)]
enum PanelConfig {
    Bark(BarkConfig),
    Rock(RockConfig),
    Ground(GroundConfig),
    Leaf(LeafConfig),
    Twig(TwigConfig),
}

impl PanelConfig {
    fn mutate_in_place<R: rand::Rng>(&mut self, rng: &mut R, rate: f32) {
        match self {
            PanelConfig::Bark(c) => c.mutate(rng, rate),
            PanelConfig::Rock(c) => c.mutate(rng, rate),
            PanelConfig::Ground(c) => c.mutate(rng, rate),
            PanelConfig::Leaf(c) => c.mutate(rng, rate),
            PanelConfig::Twig(c) => c.mutate(rng, rate),
        }
    }

    fn spawn_pending(&self, width: u32, height: u32) -> PendingTexture {
        match self {
            PanelConfig::Bark(c) => PendingTexture::bark(c.clone(), width, height),
            PanelConfig::Rock(c) => PendingTexture::rock(c.clone(), width, height),
            PanelConfig::Ground(c) => PendingTexture::ground(c.clone(), width, height),
            PanelConfig::Leaf(c) => PendingTexture::leaf(c.clone(), width, height),
            PanelConfig::Twig(c) => PendingTexture::twig(c.clone(), width, height),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            PanelConfig::Bark(_) => "Bark",
            PanelConfig::Rock(_) => "Rock",
            PanelConfig::Ground(_) => "Ground",
            PanelConfig::Leaf(_) => "Leaf",
            PanelConfig::Twig(_) => "Twig",
        }
    }
}

// --- components & helpers ---------------------------------------------------

/// Which display slot (0–4) this entity belongs to.
#[derive(Component, Clone, Copy)]
struct TextureSlot(usize);

/// Marker for the normal-map sprite in a panel slot (not clickable).
#[derive(Component)]
struct NormalPanel;

fn slot_x(slot: usize) -> f32 {
    (slot as f32 - (N_PANELS as f32 - 1.0) * 0.5) * SPACING
}

// --- systems ----------------------------------------------------------------

fn spawn_tasks(mut commands: Commands) {
    commands.spawn(Camera2d);

    let defaults: [PanelConfig; N_PANELS] = [
        PanelConfig::Bark(BarkConfig::default()),
        PanelConfig::Rock(RockConfig::default()),
        PanelConfig::Ground(GroundConfig::default()),
        PanelConfig::Leaf(LeafConfig::default()),
        PanelConfig::Twig(TwigConfig::default()),
    ];

    for (slot, config) in defaults.into_iter().enumerate() {
        let pending = config.spawn_pending(TEX_SIZE, TEX_SIZE);
        commands.spawn((pending, config, TextureSlot(slot)));
    }
}

/// Promotes completed task entities into visible sprite panels.
///
/// The `PanelConfig` that travelled on the task entity is cloned onto the
/// sprite so `handle_click` can read and mutate it later.  The label is
/// spawned as a child so `despawn_recursive` removes both together.
fn show_ready_textures(
    mut commands: Commands,
    ready: Query<(Entity, &TextureReady, &TextureSlot, &PanelConfig)>,
) {
    for (entity, ready_tex, &slot, config) in &ready {
        commands.entity(entity).despawn();

        let x = slot_x(slot.0);
        let label = config.label();

        // Albedo panel (top row) — this is the clickable panel.
        commands
            .spawn((
                Sprite {
                    image: ready_tex.0.albedo.clone(),
                    custom_size: Some(Vec2::splat(TEX_SIZE as f32)),
                    ..default()
                },
                Transform::from_translation(Vec3::new(x, ALBEDO_Y, 0.0)),
                slot,
                config.clone(),
            ))
            .with_children(|parent| {
                parent.spawn((
                    Text2d::new(label),
                    Transform::from_translation(Vec3::new(
                        0.0,
                        -(TEX_SIZE as f32 * 0.5 + 18.0),
                        0.0,
                    )),
                ));
            });

        // Normal-map panel (bottom row) — display only.
        commands
            .spawn((
                Sprite {
                    image: ready_tex.0.normal.clone(),
                    custom_size: Some(Vec2::splat(TEX_SIZE as f32)),
                    ..default()
                },
                Transform::from_translation(Vec3::new(x, NORMAL_Y, 0.0)),
                NormalPanel,
                slot,
            ))
            .with_children(|parent| {
                parent.spawn((
                    Text2d::new(format!("{label} (normal)")),
                    Transform::from_translation(Vec3::new(
                        0.0,
                        -(TEX_SIZE as f32 * 0.5 + 18.0),
                        0.0,
                    )),
                ));
            });
    }
}

/// Detects left-clicks and mutates whichever panel the cursor is over.
///
/// On click the panel entity (and its label child) are despawned; a new async
/// task is immediately queued with the mutated config.
fn handle_click(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    panels: Query<(Entity, &Transform, &PanelConfig, &TextureSlot), With<Sprite>>,
    normal_panels: Query<(Entity, &TextureSlot), With<NormalPanel>>,
    mut commands: Commands,
    mut rng: Local<Option<StdRng>>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let rng = rng.get_or_insert_with(|| StdRng::seed_from_u64(0xdead_beef_cafe));

    let Ok(window) = windows.single() else { return };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };
    let Some(world_pos) = camera.viewport_to_world_2d(cam_transform, cursor_pos).ok() else {
        return;
    };

    let half = TEX_SIZE as f32 * 0.5;
    for (entity, transform, config, &slot) in &panels {
        let center = transform.translation.truncate();
        let delta = (world_pos - center).abs();
        if delta.x <= half && delta.y <= half {
            let mut new_config = config.clone();
            new_config.mutate_in_place(rng, 0.3);
            let pending = new_config.spawn_pending(TEX_SIZE, TEX_SIZE);
            // despawn() is recursive by default in Bevy 0.15+, removing the label child too.
            commands.entity(entity).despawn();
            for (normal_entity, &normal_slot) in &normal_panels {
                if normal_slot.0 == slot.0 {
                    commands.entity(normal_entity).despawn();
                    break;
                }
            }
            commands.spawn((pending, new_config, slot));
            break;
        }
    }
}
