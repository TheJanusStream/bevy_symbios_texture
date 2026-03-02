//! `texture_viewer` — interactive material viewer with egui editor.
//!
//! Layout: **albedo** (left) | **normal map** (centre) | **spinning 3-D cube**
//! (right) — a live PBR preview with the generated material applied.
//!
//! Use the egui panel to cycle materials with **< Prev** / **Next >**, trigger
//! a random **Mutate**, and edit every parameter live.
//!
//! Run with:
//!   cargo run --example texture_viewer --features egui

use bevy::{camera::Viewport, prelude::*};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use rand::{SeedableRng, rngs::StdRng};
use symbios_genetics::Genotype;

use bevy_symbios_texture::{
    SymbiosTexturePlugin,
    ashlar::AshlarConfig,
    asphalt::AsphaltConfig,
    async_gen::{PendingTexture, TextureReady},
    bark::BarkConfig,
    brick::BrickConfig,
    cobblestone::CobblestoneConfig,
    concrete::ConcreteConfig,
    corrugated::CorrugatedConfig,
    encaustic::EncausticConfig,
    ground::GroundConfig,
    iron_grille::IronGrilleConfig,
    leaf::LeafConfig,
    marble::MarbleConfig,
    metal::MetalConfig,
    pavers::PaversConfig,
    plank::PlankConfig,
    rock::RockConfig,
    shingle::ShingleConfig,
    stained_glass::StainedGlassConfig,
    stucco::StuccoConfig,
    thatch::ThatchConfig,
    twig::TwigConfig,
    ui::{
        ashlar_config_editor, asphalt_config_editor, bark_config_editor, brick_config_editor,
        cobblestone_config_editor, concrete_config_editor, corrugated_config_editor,
        encaustic_config_editor, ground_config_editor, iron_grille_config_editor,
        leaf_config_editor, marble_config_editor, metal_config_editor, pavers_config_editor,
        plank_config_editor, rock_config_editor, shingle_config_editor,
        stained_glass_config_editor, stucco_config_editor, thatch_config_editor,
        twig_config_editor, wainscoting_config_editor, window_config_editor,
    },
    wainscoting::WainscotingConfig,
    window::WindowConfig,
};

const TEX_SIZE: u32 = 512;
const GAP: u32 = 20;

// Window dimensions: three 512-wide columns separated and padded by 20-px gaps.
const WINDOW_W: u32 = 4 * GAP + 3 * TEX_SIZE; // 1616
const WINDOW_H: u32 = 2 * GAP + TEX_SIZE; // 552

// Camera2d sprite X positions (world-space; Camera2d origin = window centre).
// Albedo panel centre = pixel (GAP + TEX_SIZE/2) = 276; window centre = 808.
const ALBEDO_X: f32 = -((GAP + TEX_SIZE) as f32); // -532
const NORMAL_X: f32 = 0.0; // centre of window

// Camera3d viewport for the rightmost column (physical pixels).
const CUBE_VP_X: u32 = 3 * GAP + 2 * TEX_SIZE; // 1084
const CUBE_VP_W: u32 = TEX_SIZE + GAP; // 532

// Size of the preview cube (world units) and spinning rates.
const CUBE_SIZE: f32 = 240.0;
const SPIN_X: f32 = 0.20; // rad/s around X
const SPIN_Y: f32 = 0.45; // rad/s around Y

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_symbios_texture — material viewer".into(),
                resolution: (WINDOW_W, WINDOW_H).into(),
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
        .add_systems(Update, spin_cube)
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
    Stucco(StuccoConfig),
    Concrete(ConcreteConfig),
    Metal(MetalConfig),
    Pavers(PaversConfig),
    Ashlar(AshlarConfig),
    Cobblestone(CobblestoneConfig),
    Thatch(ThatchConfig),
    Marble(MarbleConfig),
    Corrugated(CorrugatedConfig),
    Asphalt(AsphaltConfig),
    Wainscoting(WainscotingConfig),
    StainedGlass(StainedGlassConfig),
    IronGrille(IronGrilleConfig),
    Encaustic(EncausticConfig),
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
            PanelConfig::Stucco(c) => c.mutate(rng, rate),
            PanelConfig::Concrete(c) => c.mutate(rng, rate),
            PanelConfig::Metal(c) => c.mutate(rng, rate),
            PanelConfig::Pavers(c) => c.mutate(rng, rate),
            PanelConfig::Ashlar(c) => c.mutate(rng, rate),
            PanelConfig::Cobblestone(c) => c.mutate(rng, rate),
            PanelConfig::Thatch(c) => c.mutate(rng, rate),
            PanelConfig::Marble(c) => c.mutate(rng, rate),
            PanelConfig::Corrugated(c) => c.mutate(rng, rate),
            PanelConfig::Asphalt(c) => c.mutate(rng, rate),
            PanelConfig::Wainscoting(c) => c.mutate(rng, rate),
            PanelConfig::StainedGlass(c) => c.mutate(rng, rate),
            PanelConfig::IronGrille(c) => c.mutate(rng, rate),
            PanelConfig::Encaustic(c) => c.mutate(rng, rate),
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
            PanelConfig::Stucco(c) => PendingTexture::stucco(c.clone(), width, height),
            PanelConfig::Concrete(c) => PendingTexture::concrete(c.clone(), width, height),
            PanelConfig::Metal(c) => PendingTexture::metal(c.clone(), width, height),
            PanelConfig::Pavers(c) => PendingTexture::pavers(c.clone(), width, height),
            PanelConfig::Ashlar(c) => PendingTexture::ashlar(c.clone(), width, height),
            PanelConfig::Cobblestone(c) => PendingTexture::cobblestone(c.clone(), width, height),
            PanelConfig::Thatch(c) => PendingTexture::thatch(c.clone(), width, height),
            PanelConfig::Marble(c) => PendingTexture::marble(c.clone(), width, height),
            PanelConfig::Corrugated(c) => PendingTexture::corrugated(c.clone(), width, height),
            PanelConfig::Asphalt(c) => PendingTexture::asphalt(c.clone(), width, height),
            PanelConfig::Wainscoting(c) => PendingTexture::wainscoting(c.clone(), width, height),
            PanelConfig::StainedGlass(c) => PendingTexture::stained_glass(c.clone(), width, height),
            PanelConfig::IronGrille(c) => PendingTexture::iron_grille(c.clone(), width, height),
            PanelConfig::Encaustic(c) => PendingTexture::encaustic(c.clone(), width, height),
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
            PanelConfig::Stucco(_) => "Stucco",
            PanelConfig::Concrete(_) => "Concrete",
            PanelConfig::Metal(_) => "Metal",
            PanelConfig::Pavers(_) => "Pavers",
            PanelConfig::Ashlar(_) => "Ashlar",
            PanelConfig::Cobblestone(_) => "Cobblestone",
            PanelConfig::Thatch(_) => "Thatch",
            PanelConfig::Marble(_) => "Marble",
            PanelConfig::Corrugated(_) => "Corrugated Metal",
            PanelConfig::Asphalt(_) => "Asphalt",
            PanelConfig::Wainscoting(_) => "Wainscoting",
            PanelConfig::StainedGlass(_) => "Stained Glass",
            PanelConfig::IronGrille(_) => "Iron Grille",
            PanelConfig::Encaustic(_) => "Encaustic Tile",
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

/// Sprite showing the albedo of the active material (left column).
#[derive(Component)]
struct AlbedoDisplay;

/// Sprite showing the normal map of the active material (centre column).
#[derive(Component)]
struct NormalDisplay;

/// Spinning 3-D cube in the right column — previews the full PBR material.
#[derive(Component)]
struct CubeDisplay;

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
        PanelConfig::Stucco(StuccoConfig::default()),
        PanelConfig::Concrete(ConcreteConfig::default()),
        PanelConfig::Metal(MetalConfig::default()),
        PanelConfig::Pavers(PaversConfig::default()),
        PanelConfig::Ashlar(AshlarConfig::default()),
        PanelConfig::Cobblestone(CobblestoneConfig::default()),
        PanelConfig::Thatch(ThatchConfig::default()),
        PanelConfig::Marble(MarbleConfig::default()),
        PanelConfig::Corrugated(CorrugatedConfig::default()),
        PanelConfig::Asphalt(AsphaltConfig::default()),
        PanelConfig::Wainscoting(WainscotingConfig::default()),
        PanelConfig::StainedGlass(StainedGlassConfig::default()),
        PanelConfig::IronGrille(IronGrilleConfig::default()),
        PanelConfig::Encaustic(EncausticConfig::default()),
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

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera2d — renders albedo and normal-map sprites to the full window.
    commands.spawn((
        Camera2d,
        Camera {
            order: 0,
            ..default()
        },
    ));

    // Camera3d — renders the spinning PBR cube into the rightmost column only.
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 1,
            viewport: Some(Viewport {
                physical_position: UVec2::new(CUBE_VP_X, 0),
                physical_size: UVec2::new(CUBE_VP_W, WINDOW_H),
                ..default()
            }),
            ..default()
        },
        // Slightly elevated camera angle for a clearer 3-D impression.
        Transform::from_xyz(0.0, 160.0, 560.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Directional light so the cube shows shading and normal-map detail.
    commands.spawn((
        DirectionalLight {
            illuminance: 3500.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(4.0, 6.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Albedo sprite — left column, grey placeholder until texture is ready.
    commands.spawn((
        Sprite {
            color: Color::srgb(0.25, 0.25, 0.25),
            custom_size: Some(Vec2::splat(TEX_SIZE as f32)),
            ..default()
        },
        Transform::from_translation(Vec3::new(ALBEDO_X, 0.0, 0.0)),
        AlbedoDisplay,
    ));

    // Normal-map sprite — centre column, grey placeholder.
    commands.spawn((
        Sprite {
            color: Color::srgb(0.25, 0.25, 0.25),
            custom_size: Some(Vec2::splat(TEX_SIZE as f32)),
            ..default()
        },
        Transform::from_translation(Vec3::new(NORMAL_X, 0.0, 0.0)),
        NormalDisplay,
    ));

    // 3-D preview cube — material is updated once textures arrive.
    // Tangents must be generated explicitly; Cuboid doesn't include them by
    // default and they are required for normal-map shading to work.
    let mut cube_mesh = Cuboid::new(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE).mesh().build();
    let _ = cube_mesh.generate_tangents();
    commands.spawn((
        Mesh3d(meshes.add(cube_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.0, 0.0),
            metallic: 0.0,
            perceptual_roughness: 1.0,
            ..default()
        })),
        CubeDisplay,
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

/// Slowly rotates the preview cube each frame.
fn spin_cube(time: Res<Time>, mut query: Query<&mut Transform, With<CubeDisplay>>) {
    let t = time.elapsed_secs();
    for mut transform in &mut query {
        transform.rotation = Quat::from_euler(EulerRot::XYZ, t * SPIN_X, t * SPIN_Y, 0.0);
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
    let loading = store.textures[slot].is_none();

    let mut new_slot: Option<usize> = None;
    let mut mutate = false;
    let mut regen = false;

    egui::Window::new("Material Viewer")
        .default_width(280.0)
        .resizable(true)
        .show(ctx, |ui| {
            // Material selector dropdown.
            egui::ComboBox::from_id_salt("material_select")
                .selected_text(store.configs[slot].label())
                .width(ui.available_width() - 8.0)
                .show_ui(ui, |ui| {
                    for i in 0..n {
                        let label = store.configs[i].label();
                        if ui.selectable_label(i == slot, label).clicked() {
                            new_slot = Some(i);
                        }
                    }
                });

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
                PanelConfig::Stucco(c) => stucco_config_editor(ui, c, id),
                PanelConfig::Concrete(c) => concrete_config_editor(ui, c, id),
                PanelConfig::Metal(c) => metal_config_editor(ui, c, id),
                PanelConfig::Pavers(c) => pavers_config_editor(ui, c, id),
                PanelConfig::Ashlar(c) => ashlar_config_editor(ui, c, id),
                PanelConfig::Cobblestone(c) => cobblestone_config_editor(ui, c, id),
                PanelConfig::Thatch(c) => thatch_config_editor(ui, c, id),
                PanelConfig::Marble(c) => marble_config_editor(ui, c, id),
                PanelConfig::Corrugated(c) => corrugated_config_editor(ui, c, id),
                PanelConfig::Asphalt(c) => asphalt_config_editor(ui, c, id),
                PanelConfig::Wainscoting(c) => wainscoting_config_editor(ui, c, id),
                PanelConfig::StainedGlass(c) => stained_glass_config_editor(ui, c, id),
                PanelConfig::IronGrille(c) => iron_grille_config_editor(ui, c, id),
                PanelConfig::Encaustic(c) => encaustic_config_editor(ui, c, id),
            };

            if loading {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Generating...");
                });
            }

            regen = r;
        });

    // Apply dropdown selection.
    if let Some(s) = new_slot {
        current.0 = s;
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

/// Keeps the displayed sprites and PBR cube in sync with `CurrentSlot`.
fn update_display(
    store: Res<MaterialStore>,
    current: Res<CurrentSlot>,
    mut albedo_q: Query<&mut Sprite, (With<AlbedoDisplay>, Without<NormalDisplay>)>,
    mut normal_q: Query<&mut Sprite, (With<NormalDisplay>, Without<AlbedoDisplay>)>,
    cube_q: Query<&MeshMaterial3d<StandardMaterial>, With<CubeDisplay>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
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
            // Update cube material only when the albedo handle actually changed.
            if let Ok(cube_mat) = cube_q.single() {
                let needs_update = std_materials
                    .get(&cube_mat.0)
                    .map(|m| m.base_color_texture.as_ref() != Some(albedo))
                    .unwrap_or(false);
                if needs_update {
                    if let Some(mat) = std_materials.get_mut(&cube_mat.0) {
                        mat.base_color_texture = Some(albedo.clone());
                        mat.normal_map_texture = Some(normal.clone());
                        mat.base_color = Color::WHITE;
                    }
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
            if let Ok(cube_mat) = cube_q.single() {
                let needs_reset = std_materials
                    .get(&cube_mat.0)
                    .map(|m| m.base_color_texture.is_some())
                    .unwrap_or(false);
                if needs_reset {
                    if let Some(mat) = std_materials.get_mut(&cube_mat.0) {
                        mat.base_color_texture = None;
                        mat.normal_map_texture = None;
                        mat.base_color = Color::srgb(0.4, 0.4, 0.4);
                    }
                }
            }
        }
    }
}
