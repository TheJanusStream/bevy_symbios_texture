//! `texture_viewer` — displays one material at a time with Prev/Next navigation.
//!
//! Shows the **albedo** (top) and **normal map** (bottom) for the active material.
//! Use the **< Prev** and **Next >** buttons to cycle through all nine materials.
//! **Click the albedo panel** to mutate the current material with new random parameters.
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
    brick::BrickConfig,
    ground::GroundConfig,
    leaf::LeafConfig,
    plank::PlankConfig,
    rock::RockConfig,
    shingle::ShingleConfig,
    twig::TwigConfig,
    window::WindowConfig,
};

const TEX_SIZE: u32 = 512;

/// World-space Y centres for the albedo (top) and normal-map (bottom) sprites.
const ALBEDO_Y: f32 = TEX_SIZE as f32 * 0.5 + 20.0;
const NORMAL_Y: f32 = -(TEX_SIZE as f32 * 0.5 + 20.0);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_symbios_texture — material viewer".into(),
                resolution: (TEX_SIZE + 40, TEX_SIZE * 2 + 140).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(SymbiosTexturePlugin)
        .init_resource::<MaterialStore>()
        .init_resource::<CurrentSlot>()
        .add_systems(Startup, (setup_scene, spawn_tasks))
        .add_systems(
            Update,
            (
                collect_ready_textures,
                handle_nav_buttons,
                handle_albedo_click,
                update_display,
            )
                .chain(),
        )
        .run();
}

// ---------------------------------------------------------------------------
// Material configs
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum PanelConfig {
    Bark(BarkConfig),
    Rock(RockConfig),
    Ground(GroundConfig),
    Leaf(LeafConfig),
    Twig(TwigConfig),
    Brick(BrickConfig),
    Window(WindowConfig),
    Plank(PlankConfig),
    Shingle(ShingleConfig),
}

impl PanelConfig {
    fn mutate_in_place<R: rand::Rng>(&mut self, rng: &mut R, rate: f32) {
        match self {
            PanelConfig::Bark(c) => c.mutate(rng, rate),
            PanelConfig::Rock(c) => c.mutate(rng, rate),
            PanelConfig::Ground(c) => c.mutate(rng, rate),
            PanelConfig::Leaf(c) => c.mutate(rng, rate),
            PanelConfig::Twig(c) => c.mutate(rng, rate),
            PanelConfig::Brick(c) => c.mutate(rng, rate),
            PanelConfig::Window(c) => c.mutate(rng, rate),
            PanelConfig::Plank(c) => c.mutate(rng, rate),
            PanelConfig::Shingle(c) => c.mutate(rng, rate),
        }
    }

    fn spawn_pending(&self, width: u32, height: u32) -> PendingTexture {
        match self {
            PanelConfig::Bark(c) => PendingTexture::bark(c.clone(), width, height),
            PanelConfig::Rock(c) => PendingTexture::rock(c.clone(), width, height),
            PanelConfig::Ground(c) => PendingTexture::ground(c.clone(), width, height),
            PanelConfig::Leaf(c) => PendingTexture::leaf(c.clone(), width, height),
            PanelConfig::Twig(c) => PendingTexture::twig(c.clone(), width, height),
            PanelConfig::Brick(c) => PendingTexture::brick(c.clone(), width, height),
            PanelConfig::Window(c) => PendingTexture::window(c.clone(), width, height),
            PanelConfig::Plank(c) => PendingTexture::plank(c.clone(), width, height),
            PanelConfig::Shingle(c) => PendingTexture::shingle(c.clone(), width, height),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            PanelConfig::Bark(_) => "Bark",
            PanelConfig::Rock(_) => "Rock",
            PanelConfig::Ground(_) => "Ground",
            PanelConfig::Leaf(_) => "Leaf",
            PanelConfig::Twig(_) => "Twig",
            PanelConfig::Brick(_) => "Brick",
            PanelConfig::Window(_) => "Window",
            PanelConfig::Plank(_) => "Plank",
            PanelConfig::Shingle(_) => "Shingle",
        }
    }
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// All material configs and their cached texture handles.
#[derive(Resource, Default)]
struct MaterialStore {
    configs: Vec<PanelConfig>,
    /// `(albedo, normal)` handles; `None` while still generating.
    textures: Vec<Option<(Handle<Image>, Handle<Image>)>>,
    /// Monotonic counter per slot — stale task results are discarded.
    generations: Vec<u32>,
}

/// Index of the material currently displayed on screen.
#[derive(Resource, Default)]
struct CurrentSlot(usize);

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Sprite showing the albedo of the active material (clickable).
#[derive(Component)]
struct AlbedoDisplay;

/// Sprite showing the normal map of the active material.
#[derive(Component)]
struct NormalDisplay;

/// UI text label with the active material's name.
#[derive(Component)]
struct MaterialLabel;

/// Carried on async-task entities to route results back to the right slot.
#[derive(Component, Clone, Copy)]
struct TaskSlot {
    slot: usize,
    generation: u32,
}

/// Navigation button direction.
#[derive(Component)]
enum NavButton {
    Prev,
    Next,
}

// ---------------------------------------------------------------------------
// Startup systems
// ---------------------------------------------------------------------------

fn spawn_tasks(mut commands: Commands, mut store: ResMut<MaterialStore>) {
    let configs = vec![
        PanelConfig::Bark(BarkConfig::default()),
        PanelConfig::Rock(RockConfig::default()),
        PanelConfig::Ground(GroundConfig::default()),
        PanelConfig::Leaf(LeafConfig::default()),
        PanelConfig::Twig(TwigConfig::default()),
        PanelConfig::Brick(BrickConfig::default()),
        PanelConfig::Window(WindowConfig::default()),
        PanelConfig::Plank(PlankConfig::default()),
        PanelConfig::Shingle(ShingleConfig::default()),
    ];

    store.textures = vec![None; configs.len()];
    store.generations = vec![0; configs.len()];

    for (i, config) in configs.iter().enumerate() {
        let pending = config.spawn_pending(TEX_SIZE, TEX_SIZE);
        commands.spawn((
            pending,
            TaskSlot {
                slot: i,
                generation: 0,
            },
        ));
    }

    store.configs = configs;
}

fn setup_scene(mut commands: Commands) {
    commands.spawn(Camera2d);

    // Albedo sprite — grey placeholder until texture is ready.
    commands.spawn((
        Sprite {
            color: Color::srgb(0.25, 0.25, 0.25),
            custom_size: Some(Vec2::splat(TEX_SIZE as f32)),
            ..default()
        },
        Transform::from_translation(Vec3::new(0.0, ALBEDO_Y, 0.0)),
        AlbedoDisplay,
    ));

    // Normal-map sprite — grey placeholder.
    commands.spawn((
        Sprite {
            color: Color::srgb(0.25, 0.25, 0.25),
            custom_size: Some(Vec2::splat(TEX_SIZE as f32)),
            ..default()
        },
        Transform::from_translation(Vec3::new(0.0, NORMAL_Y, 0.0)),
        NormalDisplay,
    ));

    // UI: bottom bar — < Prev | material name | Next >
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::FlexEnd,
            ..default()
        })
        .with_children(|root| {
            root.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(20.0),
                padding: UiRect::all(Val::Px(10.0)),
                ..default()
            })
            .with_children(|row| {
                // Prev button
                row.spawn((
                    Button,
                    Node {
                        width: Val::Px(110.0),
                        height: Val::Px(40.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderColor::all(Color::srgb(0.5, 0.5, 0.5)),
                    BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
                    NavButton::Prev,
                ))
                .with_children(|btn| {
                    btn.spawn((
                        Text::new("< Prev"),
                        TextFont {
                            font_size: 18.0,
                            ..default()
                        },
                        TextColor(Color::WHITE),
                    ));
                });

                row.spawn((
                    Node {
                        min_width: Val::Px(240.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    Text::new("Loading\u{2026}"),
                    TextFont {
                        font_size: 20.0,
                        ..default()
                    },
                    TextColor(Color::WHITE),
                    MaterialLabel,
                ));

                // Next button
                row.spawn((
                    Button,
                    Node {
                        width: Val::Px(110.0),
                        height: Val::Px(40.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderColor::all(Color::srgb(0.5, 0.5, 0.5)),
                    BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
                    NavButton::Next,
                ))
                .with_children(|btn| {
                    btn.spawn((
                        Text::new("Next >"),
                        TextFont {
                            font_size: 18.0,
                            ..default()
                        },
                        TextColor(Color::WHITE),
                    ));
                });
            });
        });
}

// ---------------------------------------------------------------------------
// Update systems
// ---------------------------------------------------------------------------

/// Moves completed task results into `MaterialStore`.
fn collect_ready_textures(
    mut commands: Commands,
    ready: Query<(Entity, &TextureReady, &TaskSlot)>,
    mut store: ResMut<MaterialStore>,
) {
    for (entity, tex, task_slot) in &ready {
        commands.entity(entity).despawn();
        if store.generations[task_slot.slot] == task_slot.generation {
            store.textures[task_slot.slot] = Some((tex.0.albedo.clone(), tex.0.normal.clone()));
        }
    }
}

/// Advances or retreats `CurrentSlot` when a nav button is pressed.
fn handle_nav_buttons(
    nav_q: Query<(&Interaction, &NavButton), Changed<Interaction>>,
    mut current: ResMut<CurrentSlot>,
    store: Res<MaterialStore>,
) {
    let n = store.configs.len();
    if n == 0 {
        return;
    }
    for (interaction, nav) in &nav_q {
        if *interaction == Interaction::Pressed {
            match nav {
                NavButton::Prev => current.0 = (current.0 + n - 1) % n,
                NavButton::Next => current.0 = (current.0 + 1) % n,
            }
        }
    }
}

/// Mutates the current material when the user left-clicks the albedo panel.
fn handle_albedo_click(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    albedo_q: Query<&Transform, With<AlbedoDisplay>>,
    mut store: ResMut<MaterialStore>,
    current: Res<CurrentSlot>,
    mut commands: Commands,
    mut rng: Local<Option<StdRng>>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let rng = rng.get_or_insert_with(|| {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        StdRng::seed_from_u64(seed)
    });

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

    let Ok(transform) = albedo_q.single() else {
        return;
    };
    let center = transform.translation.truncate();
    let half = TEX_SIZE as f32 * 0.5;
    if (world_pos - center).abs().cmple(Vec2::splat(half)).all() {
        let slot = current.0;
        store.configs[slot].mutate_in_place(rng, 0.3);
        store.textures[slot] = None;
        let next_gen = store.generations[slot].wrapping_add(1);
        store.generations[slot] = next_gen;
        let pending = store.configs[slot].spawn_pending(TEX_SIZE, TEX_SIZE);
        commands.spawn((
            pending,
            TaskSlot {
                slot,
                generation: next_gen,
            },
        ));
    }
}

/// Keeps the displayed sprites and label in sync with `CurrentSlot`.
fn update_display(
    store: Res<MaterialStore>,
    current: Res<CurrentSlot>,
    mut albedo_q: Query<&mut Sprite, (With<AlbedoDisplay>, Without<NormalDisplay>)>,
    mut normal_q: Query<&mut Sprite, (With<NormalDisplay>, Without<AlbedoDisplay>)>,
    mut label_q: Query<&mut Text, With<MaterialLabel>>,
) {
    if store.configs.is_empty() {
        return;
    }

    let slot = current.0;
    let name = store.configs[slot].label();

    if let Ok(mut text) = label_q.single_mut() {
        let new_label = if store.textures[slot].is_none() {
            format!("{name} (loading\u{2026})")
        } else {
            format!("{name}  \u{00b7}  click albedo to mutate")
        };
        if text.0 != new_label {
            text.0 = new_label;
        }
    }

    match &store.textures[slot] {
        Some((albedo, normal)) => {
            if let Ok(mut sprite) = albedo_q.single_mut() {
                if sprite.image != *albedo {
                    sprite.image = albedo.clone();
                    sprite.color = Color::WHITE;
                }
            }
            if let Ok(mut sprite) = normal_q.single_mut() {
                if sprite.image != *normal {
                    sprite.image = normal.clone();
                    sprite.color = Color::WHITE;
                }
            }
        }
        None => {
            // Show grey while loading.
            if let Ok(mut sprite) = albedo_q.single_mut() {
                sprite.image = Handle::default();
                sprite.color = Color::srgb(0.25, 0.25, 0.25);
            }
            if let Ok(mut sprite) = normal_q.single_mut() {
                sprite.image = Handle::default();
                sprite.color = Color::srgb(0.25, 0.25, 0.25);
            }
        }
    }
}
