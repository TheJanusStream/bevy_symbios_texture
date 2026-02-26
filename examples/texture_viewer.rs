//! `texture_viewer` — interactive material viewer with egui editor.
//!
//! Displays the **albedo** (top) and **normal map** (bottom) for the active
//! material. Use the egui side panel to cycle materials with **< Prev** /
//! **Next >**, trigger a random **Mutate**, and edit every parameter live.
//!
//! Run with:
//!   cargo run --example texture_viewer --features egui

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
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
    ui::{
        bark_config_editor, brick_config_editor, ground_config_editor, leaf_config_editor,
        plank_config_editor, rock_config_editor, shingle_config_editor, twig_config_editor,
        window_config_editor,
    },
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
                resolution: (TEX_SIZE + 40, TEX_SIZE * 2 + 40).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((SymbiosTexturePlugin, EguiPlugin::default()))
        .init_resource::<MaterialStore>()
        .init_resource::<CurrentSlot>()
        .init_resource::<ViewerRng>()
        .add_systems(Startup, (setup_scene, spawn_tasks))
        .add_systems(EguiPrimaryContextPass, render_ui)
        .add_systems(Update, (collect_ready_textures, update_display).chain())
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

/// Seeded RNG used for config mutation.
#[derive(Resource)]
struct ViewerRng(StdRng);

impl Default for ViewerRng {
    fn default() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self(StdRng::seed_from_u64(seed))
    }
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Sprite showing the albedo of the active material.
#[derive(Component)]
struct AlbedoDisplay;

/// Sprite showing the normal map of the active material.
#[derive(Component)]
struct NormalDisplay;

/// Carried on async-task entities to route results back to the right slot.
#[derive(Component, Clone, Copy)]
struct TaskSlot {
    slot: usize,
    generation: u32,
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

/// Egui panel: navigation, loading indicator, mutate button, config editor.
fn render_ui(
    mut contexts: EguiContexts,
    mut store: ResMut<MaterialStore>,
    mut current: ResMut<CurrentSlot>,
    mut rng: ResMut<ViewerRng>,
    mut commands: Commands,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let n = store.configs.len();
    if n == 0 {
        return;
    }

    let slot = current.0;
    let title = store.configs[slot].label();
    let loading = store.textures[slot].is_none();

    let mut nav_delta: i32 = 0;
    let mut mutate = false;
    let mut regen = false;

    egui::Window::new(title)
        .anchor(egui::Align2::RIGHT_TOP, egui::Vec2::new(-10.0, 10.0))
        .default_width(280.0)
        .resizable(true)
        .show(ctx, |ui| {
            // Navigation row
            ui.horizontal(|ui| {
                if ui.button("< Prev").clicked() {
                    nav_delta = -1;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Next >").clicked() {
                        nav_delta = 1;
                    }
                });
            });

            if loading {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Generating...");
                });
            }

            if ui.button("Mutate").clicked() {
                mutate = true;
            }

            ui.separator();

            let id = egui::Id::new("viewer_config");
            let (_, r) = match &mut store.configs[slot] {
                PanelConfig::Bark(c) => bark_config_editor(ui, c, id),
                PanelConfig::Rock(c) => rock_config_editor(ui, c, id),
                PanelConfig::Ground(c) => ground_config_editor(ui, c, id),
                PanelConfig::Leaf(c) => leaf_config_editor(ui, c, id),
                PanelConfig::Twig(c) => twig_config_editor(ui, c, id),
                PanelConfig::Brick(c) => brick_config_editor(ui, c, id),
                PanelConfig::Window(c) => window_config_editor(ui, c, id),
                PanelConfig::Plank(c) => plank_config_editor(ui, c, id),
                PanelConfig::Shingle(c) => shingle_config_editor(ui, c, id),
            };
            regen = r;
        });

    // Apply navigation (takes effect next frame).
    if nav_delta != 0 {
        current.0 = ((slot as i32 + nav_delta).rem_euclid(n as i32)) as usize;
    }

    // Mutation randomises the config then triggers regen.
    if mutate {
        store.configs[slot].mutate_in_place(&mut rng.0, 0.3);
        regen = true;
    }

    // Spawn a new generation task whenever regen is requested.
    if regen {
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

/// Keeps the displayed sprites in sync with `CurrentSlot`.
fn update_display(
    store: Res<MaterialStore>,
    current: Res<CurrentSlot>,
    mut albedo_q: Query<&mut Sprite, (With<AlbedoDisplay>, Without<NormalDisplay>)>,
    mut normal_q: Query<&mut Sprite, (With<NormalDisplay>, Without<AlbedoDisplay>)>,
) {
    if store.configs.is_empty() {
        return;
    }

    let slot = current.0;

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
