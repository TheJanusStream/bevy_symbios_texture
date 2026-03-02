//! Egui UI helpers for editing texture generator configs.
//!
//! Provides reusable widgets for every texture config type so any application
//! with `bevy_egui` can embed texture parameter controls without duplicating
//! editor code.
//!
//! # Signature convention
//!
//! Every config editor has the form:
//! ```text
//! pub fn xxx_config_editor(ui: &mut egui::Ui, cfg: &mut XxxConfig, id: egui::Id) -> (bool, bool)
//! ```
//!
//! - `id` — used as the collapsing-header id salt; build with `egui::Id::new(x)`
//!   or derive a child ID via `parent.with("label")`.
//! - Returns `(writeback, regen)`:
//!   - `writeback` — any widget changed (including mid-drag). Write the config
//!     back to your resource to prevent slider snap-back.
//!   - `regen` — a value was *committed*: drag ended or a non-drag widget changed.
//!     Only regenerate the texture when this is `true`.
//!
//! Enabled via the `egui` Cargo feature.

use bevy_egui::egui;

use crate::ashlar::AshlarConfig;
use crate::asphalt::AsphaltConfig;
use crate::bark::BarkConfig;
use crate::brick::BrickConfig;
use crate::cobblestone::CobblestoneConfig;
use crate::concrete::ConcreteConfig;
use crate::corrugated::CorrugatedConfig;
use crate::encaustic::{EncausticConfig, EncausticPattern};
use crate::ground::GroundConfig;
use crate::iron_grille::IronGrilleConfig;
use crate::leaf::LeafConfig;
use crate::marble::MarbleConfig;
use crate::metal::{MetalConfig, MetalStyle};
use crate::pavers::{PaversConfig, PaversLayout};
use crate::plank::PlankConfig;
use crate::rock::RockConfig;
use crate::shingle::ShingleConfig;
use crate::stained_glass::StainedGlassConfig;
use crate::stucco::StuccoConfig;
use crate::thatch::ThatchConfig;
use crate::twig::TwigConfig;
use crate::wainscoting::WainscotingConfig;
use crate::window::WindowConfig;

// ---------------------------------------------------------------------------
// Foliage card editors
// ---------------------------------------------------------------------------

/// Renders all [`LeafConfig`] parameters inside a collapsing header.
pub fn leaf_config_editor(ui: &mut egui::Ui, cfg: &mut LeafConfig, id: egui::Id) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Leaf Config")
        .id_salt(id)
        .show(ui, |ui| {
            color_instant(ui, "Base Color", &mut cfg.color_base, &mut wb, &mut regen);
            color_instant(ui, "Edge Color", &mut cfg.color_edge, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.serration_strength, 0.0..=0.5).text("Serration"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.vein_angle, 1.0..=5.0).text("Vein Angle"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.vein_count, 2.0..=12.0).text("Vein Count"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.lobe_count, 0.0..=6.0).text("Lobe Count"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.lobe_depth, 0.0..=1.0).text("Lobe Depth"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.micro_detail, 0.0..=1.0).text("Micro Detail"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=8.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.petiole_length, 0.0..=0.3).text("Petiole"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`TwigConfig`] parameters inside a collapsing header,
/// including an embedded [`leaf_config_editor`] for the twig's leaf appearance.
pub fn twig_config_editor(ui: &mut egui::Ui, cfg: &mut TwigConfig, id: egui::Id) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Twig Config")
        .id_salt(id)
        .show(ui, |ui| {
            color_instant(ui, "Stem Color", &mut cfg.stem_color, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.stem_half_width, 0.005..=0.05).text("Stem Width"),
                &mut wb,
                &mut regen,
            );
            usize_instant(
                ui,
                &mut cfg.leaf_pairs,
                1..=8,
                "Leaf Pairs",
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.leaf_angle, 0.0..=std::f64::consts::PI)
                    .text("Leaf Angle"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.leaf_scale, 0.1..=0.6).text("Leaf Scale"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.stem_curve, 0.0..=0.15).text("Stem Curve"),
                &mut wb,
                &mut regen,
            );
            bool_instant(ui, &mut cfg.sympodial, "Sympodial", &mut wb, &mut regen);
            let (lwb, lregen) = leaf_config_editor(ui, &mut cfg.leaf, id.with("twig_leaf"));
            wb |= lwb;
            regen |= lregen;
        });
    (wb, regen)
}

/// Renders all [`BarkConfig`] parameters inside a collapsing header.
pub fn bark_config_editor(ui: &mut egui::Ui, cfg: &mut BarkConfig, id: egui::Id) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Bark Config")
        .id_salt(id)
        .show(ui, |ui| {
            color_instant(ui, "Light Color", &mut cfg.color_light, &mut wb, &mut regen);
            color_instant(ui, "Dark Color", &mut cfg.color_dark, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.scale, 1.0..=12.0).text("Scale"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.warp_u, 0.0..=0.5).text("Warp H"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.warp_v, 0.0..=1.5).text("Warp V"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=8.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
            usize_instant(ui, &mut cfg.octaves, 1..=8, "Octaves", &mut wb, &mut regen);
            ui.separator();
            ui.label("Rhytidome Plates:");
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.furrow_multiplier, 0.0..=1.0).text("Furrow Blend"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.furrow_scale_u, 0.5..=6.0).text("Plate Width"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.furrow_scale_v, 0.05..=1.0).text("Plate Length"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.furrow_shape, 0.1..=2.0).text("Plate Shape"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`WindowConfig`] parameters inside a collapsing header.
///
/// Window is a foliage-card type with alpha masking; upload results with
/// `map_to_images_card`.
pub fn window_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut WindowConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Window Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.frame_width, 0.0..=0.4).text("Frame Width"),
                &mut wb,
                &mut regen,
            );
            usize_instant(ui, &mut cfg.panes_x, 1..=6, "Panes X", &mut wb, &mut regen);
            usize_instant(ui, &mut cfg.panes_y, 1..=6, "Panes Y", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.mullion_thickness, 0.0..=0.2).text("Mullion Thickness"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.corner_radius, 0.0..=0.4).text("Corner Radius"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.glass_opacity, 0.0..=1.0).text("Glass Opacity"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.grime_level, 0.0..=1.0).text("Grime"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Frame Color", &mut cfg.color_frame, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=8.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

// ---------------------------------------------------------------------------
// Surface (tileable) texture editors
// ---------------------------------------------------------------------------

/// Renders all [`GroundConfig`] parameters inside a collapsing header.
pub fn ground_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut GroundConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Ground Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.macro_scale, 0.5..=8.0).text("Macro Scale"),
                &mut wb,
                &mut regen,
            );
            usize_instant(
                ui,
                &mut cfg.macro_octaves,
                1..=8,
                "Macro Octaves",
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.micro_scale, 2.0..=20.0).text("Micro Scale"),
                &mut wb,
                &mut regen,
            );
            usize_instant(
                ui,
                &mut cfg.micro_octaves,
                1..=6,
                "Micro Octaves",
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.micro_weight, 0.0..=1.0).text("Micro Weight"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Color Dry", &mut cfg.color_dry, &mut wb, &mut regen);
            color_instant(ui, "Color Moist", &mut cfg.color_moist, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=8.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`RockConfig`] parameters inside a collapsing header.
pub fn rock_config_editor(ui: &mut egui::Ui, cfg: &mut RockConfig, id: egui::Id) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Rock Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.scale, 0.5..=12.0).text("Scale"),
                &mut wb,
                &mut regen,
            );
            usize_instant(ui, &mut cfg.octaves, 1..=12, "Octaves", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.attenuation, 0.5..=6.0).text("Attenuation"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Color Gaps", &mut cfg.color_light, &mut wb, &mut regen);
            color_instant(ui, "Color Stone", &mut cfg.color_dark, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=8.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`BrickConfig`] parameters inside a collapsing header.
pub fn brick_config_editor(ui: &mut egui::Ui, cfg: &mut BrickConfig, id: egui::Id) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Brick Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.scale, 1.0..=16.0)
                    .step_by(1.0)
                    .text("Scale (Rows)"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.row_offset, 0.0..=1.0).text("Row Offset"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.aspect_ratio, 1.0..=4.0).text("Aspect Ratio"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.mortar_size, 0.0..=0.4).text("Mortar Size"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.bevel, 0.0..=1.0).text("Bevel"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.cell_variance, 0.0..=1.0).text("Color Variance"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.roughness, 0.0..=1.0).text("Surface Roughness"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Brick Color", &mut cfg.color_brick, &mut wb, &mut regen);
            color_instant(
                ui,
                "Mortar Color",
                &mut cfg.color_mortar,
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=8.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`PlankConfig`] parameters inside a collapsing header.
pub fn plank_config_editor(ui: &mut egui::Ui, cfg: &mut PlankConfig, id: egui::Id) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Plank Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.plank_count, 1.0..=16.0)
                    .step_by(1.0)
                    .text("Plank Count"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.grain_scale, 2.0..=32.0).text("Grain Scale"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.joint_width, 0.0..=0.3).text("Joint Width"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.stagger, 0.0..=1.0).text("Stagger"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.knot_density, 0.0..=1.0).text("Knot Density"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.grain_warp, 0.0..=1.0).text("Grain Warp"),
                &mut wb,
                &mut regen,
            );
            color_instant(
                ui,
                "Wood Light",
                &mut cfg.color_wood_light,
                &mut wb,
                &mut regen,
            );
            color_instant(
                ui,
                "Wood Dark",
                &mut cfg.color_wood_dark,
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=8.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`ShingleConfig`] parameters inside a collapsing header.
pub fn shingle_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut ShingleConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Shingle Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.scale, 2.0..=16.0)
                    .step_by(1.0)
                    .text("Scale (Rows)"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.shape_profile, 0.0..=1.0).text("Shape (Square→Scallop)"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.overlap, 0.0..=0.8).text("Overlap"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.stagger, 0.0..=1.0).text("Stagger"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.moss_level, 0.0..=1.0).text("Moss"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Tile Color", &mut cfg.color_tile, &mut wb, &mut regen);
            color_instant(ui, "Grout Color", &mut cfg.color_grout, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=8.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`StuccoConfig`] parameters inside a collapsing header.
pub fn stucco_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut StuccoConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Stucco Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.scale, 1.0..=20.0).text("Scale"),
                &mut wb,
                &mut regen,
            );
            usize_instant(ui, &mut cfg.octaves, 1..=10, "Octaves", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.roughness, 0.0..=1.0).text("Roughness"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Base Color", &mut cfg.color_base, &mut wb, &mut regen);
            color_instant(
                ui,
                "Shadow Color",
                &mut cfg.color_shadow,
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=6.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`ConcreteConfig`] parameters inside a collapsing header.
pub fn concrete_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut ConcreteConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Concrete Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.scale, 1.0..=16.0).text("Scale"),
                &mut wb,
                &mut regen,
            );
            usize_instant(ui, &mut cfg.octaves, 1..=10, "Octaves", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.roughness, 0.0..=1.0).text("Roughness"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.formwork_lines, 0.0..=12.0)
                    .step_by(1.0)
                    .text("Formwork Lines"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.formwork_depth, 0.0..=0.5).text("Formwork Depth"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.pit_density, 0.0..=0.45).text("Pit Density"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Base Color", &mut cfg.color_base, &mut wb, &mut regen);
            color_instant(ui, "Pit Color", &mut cfg.color_pit, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=6.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`MetalConfig`] parameters inside a collapsing header.
pub fn metal_config_editor(ui: &mut egui::Ui, cfg: &mut MetalConfig, id: egui::Id) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Metal Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            // Style selector
            ui.horizontal(|ui| {
                ui.label("Style:");
                let brushed = cfg.style == MetalStyle::Brushed;
                if ui.selectable_label(brushed, "Brushed").clicked() && !brushed {
                    cfg.style = MetalStyle::Brushed;
                    wb = true;
                    regen = true;
                }
                let seam = cfg.style == MetalStyle::StandingSeam;
                if ui.selectable_label(seam, "Standing Seam").clicked() && !seam {
                    cfg.style = MetalStyle::StandingSeam;
                    wb = true;
                    regen = true;
                }
            });
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.scale, 1.0..=16.0).text("Scale"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.seam_count, 1.0..=16.0)
                    .step_by(1.0)
                    .text("Seam Count"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.seam_sharpness, 0.5..=6.0).text("Seam Sharpness"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.brush_stretch, 1.0..=20.0).text("Brush Stretch"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.roughness, 0.0..=1.0).text("Roughness"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.metallic, 0.0..=1.0).text("Metallic"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.rust_level, 0.0..=1.0).text("Rust"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Metal Color", &mut cfg.color_metal, &mut wb, &mut regen);
            color_instant(ui, "Rust Color", &mut cfg.color_rust, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=6.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`PaversConfig`] parameters inside a collapsing header.
pub fn pavers_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut PaversConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Pavers Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            // Layout selector
            ui.horizontal(|ui| {
                ui.label("Layout:");
                let sq = cfg.layout == PaversLayout::Square;
                if ui.selectable_label(sq, "Square").clicked() && !sq {
                    cfg.layout = PaversLayout::Square;
                    wb = true;
                    regen = true;
                }
                let hx = cfg.layout == PaversLayout::Hexagonal;
                if ui.selectable_label(hx, "Hexagonal").clicked() && !hx {
                    cfg.layout = PaversLayout::Hexagonal;
                    wb = true;
                    regen = true;
                }
            });
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.scale, 1.0..=16.0)
                    .step_by(1.0)
                    .text("Scale"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.aspect_ratio, 0.5..=3.0).text("Aspect Ratio"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.grout_width, 0.0..=0.35).text("Grout Width"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.bevel, 0.0..=1.0).text("Bevel"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.cell_variance, 0.0..=0.8).text("Color Variance"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.roughness, 0.0..=1.0).text("Surface Roughness"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Stone Color", &mut cfg.color_stone, &mut wb, &mut regen);
            color_instant(ui, "Grout Color", &mut cfg.color_grout, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=8.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`AshlarConfig`] parameters inside a collapsing header.
pub fn ashlar_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut AshlarConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Ashlar Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            usize_instant(ui, &mut cfg.rows, 2..=8, "Rows", &mut wb, &mut regen);
            usize_instant(ui, &mut cfg.cols, 2..=6, "Cols", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.mortar_size, 0.005..=0.15).text("Mortar Size"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.bevel, 0.0..=1.0).text("Bevel"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.cell_variance, 0.0..=1.0).text("Color Variance"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.chisel_depth, 0.0..=1.0).text("Chisel Depth"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.roughness, 0.0..=1.0).text("Roughness"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Stone Color", &mut cfg.color_stone, &mut wb, &mut regen);
            color_instant(ui, "Mortar Color", &mut cfg.color_mortar, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=8.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`CobblestoneConfig`] parameters inside a collapsing header.
pub fn cobblestone_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut CobblestoneConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Cobblestone Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.scale, 2.0..=14.0).text("Scale"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.gap_width, 0.01..=0.3).text("Gap Width"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.cell_variance, 0.0..=1.0).text("Color Variance"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.roundness, 0.3..=2.5).text("Roundness"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Stone Color", &mut cfg.color_stone, &mut wb, &mut regen);
            color_instant(ui, "Mud Color", &mut cfg.color_mud, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=8.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`ThatchConfig`] parameters inside a collapsing header.
pub fn thatch_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut ThatchConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Thatch Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.density, 3.0..=24.0).text("Fibre Density"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.anisotropy, 2.0..=20.0).text("Anisotropy"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.warp_strength, 0.0..=0.6).text("Warp"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.layer_count, 2.0..=20.0).text("Layer Count"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.layer_shadow, 0.0..=1.0).text("Layer Shadow"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Straw Color", &mut cfg.color_straw, &mut wb, &mut regen);
            color_instant(ui, "Shadow Color", &mut cfg.color_shadow, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=6.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`MarbleConfig`] parameters inside a collapsing header.
pub fn marble_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut MarbleConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Marble Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.scale, 0.5..=10.0).text("Scale"),
                &mut wb,
                &mut regen,
            );
            usize_instant(ui, &mut cfg.octaves, 2..=10, "Octaves", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.warp_strength, 0.0..=2.0).text("Warp Strength"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.vein_frequency, 0.5..=10.0).text("Vein Frequency"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.vein_sharpness, 0.3..=8.0).text("Vein Sharpness"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.roughness, 0.0..=0.4).text("Roughness"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Base Color", &mut cfg.color_base, &mut wb, &mut regen);
            color_instant(ui, "Vein Color", &mut cfg.color_vein, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=4.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`CorrugatedConfig`] parameters inside a collapsing header.
pub fn corrugated_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut CorrugatedConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Corrugated Metal Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.ridges, 2.0..=20.0)
                    .step_by(1.0)
                    .text("Ridges"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.ridge_depth, 0.3..=2.5).text("Ridge Depth"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.roughness, 0.0..=1.0).text("Roughness"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.rust_level, 0.0..=1.0).text("Rust"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.metallic, 0.0..=1.0).text("Metallic"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Metal Color", &mut cfg.color_metal, &mut wb, &mut regen);
            color_instant(ui, "Rust Color", &mut cfg.color_rust, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=6.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`AsphaltConfig`] parameters inside a collapsing header.
pub fn asphalt_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut AsphaltConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Asphalt Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.scale, 1.0..=14.0).text("Scale"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.aggregate_density, 0.02..=0.5)
                    .text("Aggregate Density"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.aggregate_scale, 4.0..=40.0).text("Aggregate Scale"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.roughness, 0.5..=1.0).text("Roughness"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.stain_level, 0.0..=1.0).text("Stain Level"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Base Color", &mut cfg.color_base, &mut wb, &mut regen);
            color_instant(
                ui,
                "Aggregate Color",
                &mut cfg.color_aggregate,
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=4.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`WainscotingConfig`] parameters inside a collapsing header.
pub fn wainscoting_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut WainscotingConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Wainscoting Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            usize_instant(ui, &mut cfg.panels_x, 1..=4, "Panels X", &mut wb, &mut regen);
            usize_instant(ui, &mut cfg.panels_y, 1..=4, "Panels Y", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.frame_width, 0.05..=0.4).text("Frame Width"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.panel_inset, 0.0..=0.2).text("Panel Inset"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.grain_scale, 4.0..=28.0).text("Grain Scale"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.grain_warp, 0.0..=1.0).text("Grain Warp"),
                &mut wb,
                &mut regen,
            );
            color_instant(
                ui,
                "Wood Light",
                &mut cfg.color_wood_light,
                &mut wb,
                &mut regen,
            );
            color_instant(
                ui,
                "Wood Dark",
                &mut cfg.color_wood_dark,
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=8.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`StainedGlassConfig`] parameters inside a collapsing header.
pub fn stained_glass_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut StainedGlassConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Stained Glass Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            usize_instant(ui, &mut cfg.cell_count, 3..=30, "Cell Count", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.lead_width, 0.01..=0.15).text("Lead Width"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.saturation, 0.3..=1.0).text("Saturation"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.glass_roughness, 0.0..=0.2).text("Glass Roughness"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.grime_level, 0.0..=0.6).text("Grime"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=4.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`IronGrilleConfig`] parameters inside a collapsing header.
pub fn iron_grille_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut IronGrilleConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Iron Grille Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            usize_instant(ui, &mut cfg.bars_x, 1..=12, "Bars X", &mut wb, &mut regen);
            usize_instant(ui, &mut cfg.bars_y, 1..=12, "Bars Y", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.bar_width, 0.01..=0.25).text("Bar Width"),
                &mut wb,
                &mut regen,
            );
            bool_instant(ui, &mut cfg.round_bars, "Round Bars", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.rust_level, 0.0..=1.0).text("Rust"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Iron Color", &mut cfg.color_iron, &mut wb, &mut regen);
            color_instant(ui, "Rust Color", &mut cfg.color_rust, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=6.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

/// Renders all [`EncausticConfig`] parameters inside a collapsing header.
pub fn encaustic_config_editor(
    ui: &mut egui::Ui,
    cfg: &mut EncausticConfig,
    id: egui::Id,
) -> (bool, bool) {
    let mut wb = false;
    let mut regen = false;
    egui::CollapsingHeader::new("Encaustic Tile Config")
        .id_salt(id)
        .show(ui, |ui| {
            u32_instant(ui, &mut cfg.seed, "Seed", &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.scale, 1.0..=12.0)
                    .step_by(1.0)
                    .text("Scale"),
                &mut wb,
                &mut regen,
            );
            // Pattern selector
            ui.horizontal(|ui| {
                ui.label("Pattern:");
                let cb = cfg.pattern == EncausticPattern::Checkerboard;
                if ui.selectable_label(cb, "Checker").clicked() && !cb {
                    cfg.pattern = EncausticPattern::Checkerboard;
                    wb = true;
                    regen = true;
                }
                let oct = cfg.pattern == EncausticPattern::Octagon;
                if ui.selectable_label(oct, "Octagon").clicked() && !oct {
                    cfg.pattern = EncausticPattern::Octagon;
                    wb = true;
                    regen = true;
                }
                let dia = cfg.pattern == EncausticPattern::Diamond;
                if ui.selectable_label(dia, "Diamond").clicked() && !dia {
                    cfg.pattern = EncausticPattern::Diamond;
                    wb = true;
                    regen = true;
                }
            });
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.grout_width, 0.01..=0.2).text("Grout Width"),
                &mut wb,
                &mut regen,
            );
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.glaze_roughness, 0.0..=0.15).text("Glaze Roughness"),
                &mut wb,
                &mut regen,
            );
            color_instant(ui, "Color A", &mut cfg.color_a, &mut wb, &mut regen);
            color_instant(ui, "Color B", &mut cfg.color_b, &mut wb, &mut regen);
            color_instant(ui, "Grout Color", &mut cfg.color_grout, &mut wb, &mut regen);
            slider_debounced(
                ui,
                egui::Slider::new(&mut cfg.normal_strength, 0.0..=6.0).text("Normal Strength"),
                &mut wb,
                &mut regen,
            );
        });
    (wb, regen)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Adds a slider with drag-aware debouncing.
///
/// - `writeback` accumulates on any `changed()` (including mid-drag) so the
///   caller can write the value back and prevent visual snap-back.
/// - `regen` accumulates only on `drag_stopped()` or a non-drag change, avoiding
///   unnecessary texture regeneration during continuous slider drags.
pub fn slider_debounced(
    ui: &mut egui::Ui,
    slider: impl egui::Widget,
    writeback: &mut bool,
    regen: &mut bool,
) {
    let r = ui.add(slider);
    *writeback |= r.changed();
    *regen |= r.drag_stopped() || (r.changed() && !r.dragged());
}

/// Horizontal labeled slider for `f32` values. Returns `true` on any change.
pub fn f32_slider(
    ui: &mut egui::Ui,
    val: &mut f32,
    label: &str,
    range: std::ops::RangeInclusive<f32>,
) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::Slider::new(val, range)).changed()
    })
    .inner
}

/// Horizontal labeled slider for `f64` values. Returns `true` on any change.
pub fn f64_slider(
    ui: &mut egui::Ui,
    val: &mut f64,
    label: &str,
    range: std::ops::RangeInclusive<f64>,
) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::Slider::new(val, range)).changed()
    })
    .inner
}

/// Horizontal labeled slider for `usize` values. Returns `true` on any change.
pub fn usize_slider(
    ui: &mut egui::Ui,
    val: &mut usize,
    label: &str,
    range: std::ops::RangeInclusive<usize>,
) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::Slider::new(val, range)).changed()
    })
    .inner
}

/// Horizontal labeled drag value for `u32`. Returns `true` on any change.
pub fn u32_drag(ui: &mut egui::Ui, val: &mut u32, label: &str) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(val).speed(1.0)).changed()
    })
    .inner
}

// ---------------------------------------------------------------------------
// Private inline helpers used only within this module
// ---------------------------------------------------------------------------

/// Color picker that immediately sets both writeback and regen flags.
fn color_instant(
    ui: &mut egui::Ui,
    label: &str,
    color: &mut [f32; 3],
    wb: &mut bool,
    regen: &mut bool,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        let r = ui.color_edit_button_rgb(color);
        *wb |= r.changed();
        *regen |= r.changed();
    });
}

/// Checkbox that immediately sets both writeback and regen flags.
fn bool_instant(ui: &mut egui::Ui, val: &mut bool, label: &str, wb: &mut bool, regen: &mut bool) {
    let r = ui.checkbox(val, label);
    *wb |= r.changed();
    *regen |= r.changed();
}

/// Integer usize slider that immediately sets both flags (integer steps are cheap to regen).
fn usize_instant(
    ui: &mut egui::Ui,
    val: &mut usize,
    range: std::ops::RangeInclusive<usize>,
    label: &str,
    wb: &mut bool,
    regen: &mut bool,
) {
    let r = ui.add(egui::Slider::new(val, range).text(label));
    *wb |= r.changed();
    *regen |= r.changed();
}

/// Integer u32 drag that immediately sets both flags.
fn u32_instant(ui: &mut egui::Ui, val: &mut u32, label: &str, wb: &mut bool, regen: &mut bool) {
    ui.horizontal(|ui| {
        ui.label(label);
        let r = ui.add(egui::DragValue::new(val).speed(1.0));
        *wb |= r.changed();
        *regen |= r.changed();
    });
}
