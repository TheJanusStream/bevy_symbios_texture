//! `texture_viewer` — interactive material viewer with egui editor.
//!
//! Layout: **albedo** (left) | **normal map** (centre) | **3-D preview**
//! (right) — a live PBR preview with the generated material applied.
//! Tileable surface textures get a spinning cube; alpha-masked cards and
//! sprite atlases get a gently swaying alpha-blended quad in front of a
//! checkerboard backdrop, so per-pixel alpha is actually visible.
//!
//! Use the egui panel to select a material from the dropdown, trigger a
//! random **Mutate**, and edit every parameter live.
//!
//! Run with:
//!   cargo run --release --example texture_viewer --features egui
//!
//! (`--release` matters: unoptimized texture generation takes tens of
//! seconds per 512² map.)

use bevy::{
    asset::RenderAssetUsages,
    camera::Viewport,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use rand::{SeedableRng, rngs::StdRng};
use symbios_genetics::Genotype;

use bevy_symbios_texture::{
    SymbiosTexturePlugin, TextureConfig, async_gen::TextureReady, ui::texture_config_editor,
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

// Card preview: a flat quad swaying around Y. A full spin would put the quad
// edge-on half the time, so it oscillates instead — enough to show the
// normal-map response under the directional light.
const CARD_SIZE: f32 = 300.0;
const SWAY_RATE: f32 = 0.7; // sway phase speed (rad/s)
const SWAY_AMPLITUDE: f32 = 0.6; // max yaw deflection (rad, ~34°)

// Checkerboard backdrop behind the card quad — makes alpha edges judgeable.
// Far enough back that the swaying quad's corners never reach it
// (half-width × sin(amplitude) ≈ 85 world units of excursion), and sized so
// it still projects to a visible frame around the quad despite being deeper.
const BACKDROP_SIZE: f32 = 480.0;
const BACKDROP_Z: f32 = -130.0;
const CHECKER_TEX: u32 = 512; // backdrop texture side (texels)
const CHECKER_CELL: u32 = 64; // checker cell side (texels)

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
        .add_plugins((SymbiosTexturePlugin::default(), EguiPlugin::default()))
        .init_resource::<MaterialStore>()
        .init_resource::<CurrentSlot>()
        .init_resource::<ViewerRng>()
        .add_systems(Startup, (setup_scene, spawn_tasks))
        .add_systems(EguiPrimaryContextPass, render_ui)
        .add_systems(Update, (collect_ready_textures, update_display).chain())
        .add_systems(Update, (spin_cube, sway_card))
        .run();
}

// ---------------------------------------------------------------------------
// Material configs
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// All material configs and their cached texture handles.
#[derive(Resource, Default)]
struct MaterialStore {
    configs: Vec<TextureConfig>,
    /// `(albedo, normal)` handles; `None` while still generating.
    textures: Vec<Option<(Handle<Image>, Handle<Image>)>>,
    /// Monotonic counter per slot — stale task results are discarded.
    generations: Vec<u32>,
    /// `true` for alpha-masked cards / sprite atlases (card-quad preview);
    /// `false` for tileable surfaces (cube preview). Captured from
    /// `PendingTexture::is_card()` so the library stays the single source
    /// of truth for the classification.
    is_card: Vec<bool>,
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

/// Spinning 3-D cube in the right column — previews tileable surface materials.
#[derive(Component)]
struct CubeDisplay;

/// Root of the card preview (quad + checkerboard backdrop) — shown instead of
/// the cube for alpha-masked cards and sprite atlases. Visibility is toggled
/// here so both children follow.
#[derive(Component)]
struct CardPreviewRoot;

/// Swaying alpha-blended quad showing the card material itself.
#[derive(Component)]
struct CardDisplay;

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
    // The registry order drives the dropdown — new generators appear
    // automatically (TextureConfig::all_defaults derives from the registry).
    let configs = TextureConfig::all_defaults();

    store.textures = vec![None; configs.len()];
    store.generations = vec![0; configs.len()];
    store.is_card = Vec::with_capacity(configs.len());

    for (i, config) in configs.iter().enumerate() {
        let pending = config
            .spawn(TEX_SIZE, TEX_SIZE)
            .expect("all_defaults never yields TextureConfig::None");
        store.is_card.push(pending.is_card());
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

/// Bakes the grey checkerboard texture shown behind the card-preview quad.
fn checkerboard_image(images: &mut Assets<Image>) -> Handle<Image> {
    let mut data = Vec::with_capacity((CHECKER_TEX * CHECKER_TEX * 4) as usize);
    for y in 0..CHECKER_TEX {
        for x in 0..CHECKER_TEX {
            let light = ((x / CHECKER_CELL) + (y / CHECKER_CELL)).is_multiple_of(2);
            let v = if light { 140 } else { 90 };
            data.extend_from_slice(&[v, v, v, 255]);
        }
    }
    images.add(Image::new(
        Extent3d {
            width: CHECKER_TEX,
            height: CHECKER_TEX,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    ))
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
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
            base_color: Color::srgb(0.4, 0.4, 0.4),
            metallic: 0.0,
            perceptual_roughness: 1.0,
            ..default()
        })),
        CubeDisplay,
    ));

    // Card preview — hidden until a card-style material is selected.
    // Alpha-blended so the per-pixel alpha that defines card generators is
    // actually visible; double-sided so the sway never shows a culled face.
    let mut card_mesh = Rectangle::new(CARD_SIZE, CARD_SIZE).mesh().build();
    let _ = card_mesh.generate_tangents();
    commands
        .spawn((Transform::default(), Visibility::Hidden, CardPreviewRoot))
        .with_children(|parent| {
            // Unlit checkerboard backdrop, behind the swaying quad.
            parent.spawn((
                Mesh3d(meshes.add(Rectangle::new(BACKDROP_SIZE, BACKDROP_SIZE))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color_texture: Some(checkerboard_image(&mut images)),
                    unlit: true,
                    ..default()
                })),
                Transform::from_xyz(0.0, 0.0, BACKDROP_Z),
            ));
            // The card quad itself.
            parent.spawn((
                Mesh3d(meshes.add(card_mesh)),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.4, 0.4, 0.4),
                    metallic: 0.0,
                    perceptual_roughness: 1.0,
                    alpha_mode: AlphaMode::Blend,
                    double_sided: true,
                    cull_mode: None,
                    ..default()
                })),
                CardDisplay,
            ));
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

/// Slowly rotates the preview cube each frame.
fn spin_cube(time: Res<Time>, mut query: Query<&mut Transform, With<CubeDisplay>>) {
    let t = time.elapsed_secs();
    for mut transform in &mut query {
        transform.rotation = Quat::from_euler(EulerRot::XYZ, t * SPIN_X, t * SPIN_Y, 0.0);
    }
}

/// Oscillates the card quad around Y so lighting and normal-map response are
/// visible without ever turning the quad edge-on.
fn sway_card(time: Res<Time>, mut query: Query<&mut Transform, With<CardDisplay>>) {
    let yaw = (time.elapsed_secs() * SWAY_RATE).sin() * SWAY_AMPLITUDE;
    for mut transform in &mut query {
        transform.rotation = Quat::from_rotation_y(yaw);
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
            let (_, r) = texture_config_editor(ui, &mut store.configs[slot], id);

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
        store.configs[slot].mutate(&mut rng.0, 0.3);
        regen = true;
    }

    // Spawn a new generation task whenever regen is requested.
    if regen {
        store.textures[slot] = None;
        let next_gen = store.generations[slot].wrapping_add(1);
        store.generations[slot] = next_gen;
        let pending = store.configs[slot]
            .spawn(TEX_SIZE, TEX_SIZE)
            .expect("viewer slots never hold TextureConfig::None");
        commands.spawn((
            pending,
            TaskSlot {
                slot,
                generation: next_gen,
            },
        ));
    }
}

/// Applies the generated maps to a preview material, guarded so the asset is
/// only mutated when the albedo handle actually changed.
fn apply_preview_textures(
    materials: &mut Assets<StandardMaterial>,
    handle: &Handle<StandardMaterial>,
    albedo: &Handle<Image>,
    normal: &Handle<Image>,
) {
    let needs_update = materials
        .get(handle)
        .map(|m| m.base_color_texture.as_ref() != Some(albedo))
        .unwrap_or(false);
    if needs_update && let Some(mat) = materials.get_mut(handle) {
        mat.base_color_texture = Some(albedo.clone());
        mat.normal_map_texture = Some(normal.clone());
        mat.base_color = Color::WHITE;
    }
}

/// Resets a preview material to its grey placeholder while regenerating.
fn reset_preview_textures(
    materials: &mut Assets<StandardMaterial>,
    handle: &Handle<StandardMaterial>,
) {
    let needs_reset = materials
        .get(handle)
        .map(|m| m.base_color_texture.is_some())
        .unwrap_or(false);
    if needs_reset && let Some(mat) = materials.get_mut(handle) {
        mat.base_color_texture = None;
        mat.normal_map_texture = None;
        mat.base_color = Color::srgb(0.4, 0.4, 0.4);
    }
}

/// Query filter matching either 3-D preview mesh (cube or card quad).
type AnyPreviewMesh = Or<(With<CubeDisplay>, With<CardDisplay>)>;

/// Query filter matching the two visibility-toggled preview roots.
type AnyPreviewRoot = Or<(With<CubeDisplay>, With<CardPreviewRoot>)>;

/// Keeps the displayed sprites and the 3-D preview in sync with `CurrentSlot`.
fn update_display(
    store: Res<MaterialStore>,
    current: Res<CurrentSlot>,
    mut albedo_q: Query<&mut Sprite, (With<AlbedoDisplay>, Without<NormalDisplay>)>,
    mut normal_q: Query<&mut Sprite, (With<NormalDisplay>, Without<AlbedoDisplay>)>,
    preview_q: Query<&MeshMaterial3d<StandardMaterial>, AnyPreviewMesh>,
    mut vis_q: Query<(&mut Visibility, Has<CardPreviewRoot>), AnyPreviewRoot>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
) {
    if store.configs.is_empty() {
        return;
    }

    let slot = current.0;

    // Cube for tileable surfaces; swaying quad + checkerboard for cards.
    let is_card = store.is_card[slot];
    for (mut vis, is_card_preview) in &mut vis_q {
        vis.set_if_neq(if is_card_preview == is_card {
            Visibility::Visible
        } else {
            Visibility::Hidden
        });
    }

    match &store.textures[slot] {
        Some((albedo, normal)) => {
            if let Ok(mut sprite) = albedo_q.single_mut()
                && sprite.image != *albedo
            {
                sprite.image = albedo.clone();
                sprite.color = Color::WHITE;
            }
            if let Ok(mut sprite) = normal_q.single_mut()
                && sprite.image != *normal
            {
                sprite.image = normal.clone();
                sprite.color = Color::WHITE;
            }
            // Both preview materials stay in sync; the handle guard makes
            // touching the hidden one free.
            for mat in &preview_q {
                apply_preview_textures(&mut std_materials, &mat.0, albedo, normal);
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
            for mat in &preview_q {
                reset_preview_textures(&mut std_materials, &mat.0);
            }
        }
    }
}
