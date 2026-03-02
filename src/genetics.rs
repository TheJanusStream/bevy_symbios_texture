//! [`Genotype`] implementations for all texture configuration types.
//!
//! Each config struct implements [`symbios_genetics::Genotype`], making it
//! compatible with `SimpleGA`, `Nsga2`, and `MapElites` from that crate.
//!
//! # Mutation
//! Each numeric field is perturbed independently with probability `rate`.
//! Floating-point fields receive a uniform perturbation scaled to the field's
//! natural range.  Integer fields step by ±1.  Boolean fields are flipped.
//! Seed fields are replaced entirely.
//!
//! # Crossover
//! Uniform field crossover: each field is drawn independently from one of the
//! two parents at random (50/50).  Color channels are crossed over per-channel
//! for finer-grained colour mixing.

use std::f64::consts::PI;

use rand::Rng;
use symbios_genetics::Genotype;

use crate::{
    ashlar::AshlarConfig,
    asphalt::AsphaltConfig,
    bark::BarkConfig,
    brick::BrickConfig,
    cobblestone::CobblestoneConfig,
    concrete::ConcreteConfig,
    corrugated::CorrugatedConfig,
    encaustic::{EncausticConfig, EncausticPattern},
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
    wainscoting::WainscotingConfig,
    window::WindowConfig,
};

// --- shared helpers ---------------------------------------------------------

/// Perturb a `f64` by a uniform step in `(-half_range, +half_range)` with
/// probability `rate`, clamped to `[min, max]`.
#[inline]
fn mutate_f64<R: Rng>(
    val: f64,
    rng: &mut R,
    rate: f32,
    half_range: f64,
    min: f64,
    max: f64,
) -> f64 {
    if rng.random::<f32>() < rate {
        (val + (rng.random::<f64>() - 0.5) * 2.0 * half_range).clamp(min, max)
    } else {
        val
    }
}

/// Perturb a `f32` by a uniform step in `(-half_range, +half_range)` with
/// probability `rate`, clamped to `[min, max]`.
#[inline]
fn mutate_f32<R: Rng>(
    val: f32,
    rng: &mut R,
    rate: f32,
    half_range: f32,
    min: f32,
    max: f32,
) -> f32 {
    if rng.random::<f32>() < rate {
        (val + (rng.random::<f32>() - 0.5) * 2.0 * half_range).clamp(min, max)
    } else {
        val
    }
}

/// Perturb a `usize` by ±1 with probability `rate`, clamped to `[min, max]`.
#[inline]
fn mutate_usize<R: Rng>(val: usize, rng: &mut R, rate: f32, min: usize, max: usize) -> usize {
    if rng.random::<f32>() < rate {
        if rng.random::<bool>() {
            val.saturating_add(1).min(max)
        } else {
            val.saturating_sub(1).max(min)
        }
    } else {
        val
    }
}

/// Replace a `u32` seed entirely with probability `rate`.
#[inline]
fn mutate_seed<R: Rng>(val: u32, rng: &mut R, rate: f32) -> u32 {
    if rng.random::<f32>() < rate {
        rng.random::<u32>()
    } else {
        val
    }
}

/// Mutate each channel of an RGB `[f32; 3]` colour independently.
#[inline]
fn mutate_color3<R: Rng>(color: [f32; 3], rng: &mut R, rate: f32, half_range: f32) -> [f32; 3] {
    [
        mutate_f32(color[0], rng, rate, half_range, 0.0, 1.0),
        mutate_f32(color[1], rng, rate, half_range, 0.0, 1.0),
        mutate_f32(color[2], rng, rate, half_range, 0.0, 1.0),
    ]
}

/// Crossover two RGB colours channel-by-channel.
#[inline]
fn crossover_color3<R: Rng>(a: [f32; 3], b: [f32; 3], rng: &mut R) -> [f32; 3] {
    [
        if rng.random::<bool>() { a[0] } else { b[0] },
        if rng.random::<bool>() { a[1] } else { b[1] },
        if rng.random::<bool>() { a[2] } else { b[2] },
    ]
}

// --- BarkConfig -------------------------------------------------------------

impl Genotype for BarkConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.scale = mutate_f64(self.scale, rng, rate, 1.0, 0.5, 16.0);
        self.octaves = mutate_usize(self.octaves, rng, rate, 1, 12);
        self.warp_u = mutate_f64(self.warp_u, rng, rate, 0.1, 0.0, 1.0);
        self.warp_v = mutate_f64(self.warp_v, rng, rate, 0.2, 0.0, 2.0);
        self.color_light = mutate_color3(self.color_light, rng, rate, 0.07);
        self.color_dark = mutate_color3(self.color_dark, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 8.0);
        self.furrow_multiplier = mutate_f64(self.furrow_multiplier, rng, rate, 0.2, 0.0, 1.0);
        self.furrow_scale_u = mutate_f64(self.furrow_scale_u, rng, rate, 0.5, 0.5, 6.0);
        self.furrow_scale_v = mutate_f64(self.furrow_scale_v, rng, rate, 0.1, 0.05, 1.0);
        self.furrow_shape = mutate_f64(self.furrow_shape, rng, rate, 0.15, 0.1, 2.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() {
                self.seed
            } else {
                other.seed
            },
            scale: if rng.random::<bool>() {
                self.scale
            } else {
                other.scale
            },
            octaves: if rng.random::<bool>() {
                self.octaves
            } else {
                other.octaves
            },
            warp_u: if rng.random::<bool>() {
                self.warp_u
            } else {
                other.warp_u
            },
            warp_v: if rng.random::<bool>() {
                self.warp_v
            } else {
                other.warp_v
            },
            color_light: crossover_color3(self.color_light, other.color_light, rng),
            color_dark: crossover_color3(self.color_dark, other.color_dark, rng),
            normal_strength: if rng.random::<bool>() {
                self.normal_strength
            } else {
                other.normal_strength
            },
            furrow_multiplier: if rng.random::<bool>() {
                self.furrow_multiplier
            } else {
                other.furrow_multiplier
            },
            furrow_scale_u: if rng.random::<bool>() {
                self.furrow_scale_u
            } else {
                other.furrow_scale_u
            },
            furrow_scale_v: if rng.random::<bool>() {
                self.furrow_scale_v
            } else {
                other.furrow_scale_v
            },
            furrow_shape: if rng.random::<bool>() {
                self.furrow_shape
            } else {
                other.furrow_shape
            },
        }
    }
}

// --- RockConfig -------------------------------------------------------------

impl Genotype for RockConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.scale = mutate_f64(self.scale, rng, rate, 0.75, 0.5, 12.0);
        self.octaves = mutate_usize(self.octaves, rng, rate, 1, 14);
        self.attenuation = mutate_f64(self.attenuation, rng, rate, 0.25, 1.0, 4.0);
        self.color_light = mutate_color3(self.color_light, rng, rate, 0.07);
        self.color_dark = mutate_color3(self.color_dark, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 8.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() {
                self.seed
            } else {
                other.seed
            },
            scale: if rng.random::<bool>() {
                self.scale
            } else {
                other.scale
            },
            octaves: if rng.random::<bool>() {
                self.octaves
            } else {
                other.octaves
            },
            attenuation: if rng.random::<bool>() {
                self.attenuation
            } else {
                other.attenuation
            },
            color_light: crossover_color3(self.color_light, other.color_light, rng),
            color_dark: crossover_color3(self.color_dark, other.color_dark, rng),
            normal_strength: if rng.random::<bool>() {
                self.normal_strength
            } else {
                other.normal_strength
            },
        }
    }
}

// --- GroundConfig -----------------------------------------------------------

impl Genotype for GroundConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.macro_scale = mutate_f64(self.macro_scale, rng, rate, 0.5, 0.5, 8.0);
        self.macro_octaves = mutate_usize(self.macro_octaves, rng, rate, 1, 10);
        self.micro_scale = mutate_f64(self.micro_scale, rng, rate, 1.0, 1.0, 20.0);
        self.micro_octaves = mutate_usize(self.micro_octaves, rng, rate, 1, 10);
        self.micro_weight = mutate_f64(self.micro_weight, rng, rate, 0.1, 0.0, 1.0);
        self.color_dry = mutate_color3(self.color_dry, rng, rate, 0.07);
        self.color_moist = mutate_color3(self.color_moist, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 8.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() {
                self.seed
            } else {
                other.seed
            },
            macro_scale: if rng.random::<bool>() {
                self.macro_scale
            } else {
                other.macro_scale
            },
            macro_octaves: if rng.random::<bool>() {
                self.macro_octaves
            } else {
                other.macro_octaves
            },
            micro_scale: if rng.random::<bool>() {
                self.micro_scale
            } else {
                other.micro_scale
            },
            micro_octaves: if rng.random::<bool>() {
                self.micro_octaves
            } else {
                other.micro_octaves
            },
            micro_weight: if rng.random::<bool>() {
                self.micro_weight
            } else {
                other.micro_weight
            },
            color_dry: crossover_color3(self.color_dry, other.color_dry, rng),
            color_moist: crossover_color3(self.color_moist, other.color_moist, rng),
            normal_strength: if rng.random::<bool>() {
                self.normal_strength
            } else {
                other.normal_strength
            },
        }
    }
}

// --- LeafConfig -------------------------------------------------------------

impl Genotype for LeafConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.color_base = mutate_color3(self.color_base, rng, rate, 0.07);
        self.color_edge = mutate_color3(self.color_edge, rng, rate, 0.07);
        self.serration_strength = mutate_f64(self.serration_strength, rng, rate, 0.01, 0.0, 0.15);
        self.vein_angle = mutate_f64(self.vein_angle, rng, rate, 0.3, 0.5, 6.0);
        self.micro_detail = mutate_f64(self.micro_detail, rng, rate, 0.1, 0.0, 1.0);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.3, 0.5, 6.0);
        self.lobe_count = mutate_f64(self.lobe_count, rng, rate, 1.0, 0.0, 10.0);
        self.lobe_depth = mutate_f64(self.lobe_depth, rng, rate, 0.15, 0.0, 1.0);
        self.lobe_sharpness = mutate_f64(self.lobe_sharpness, rng, rate, 0.4, 0.1, 5.0);
        self.petiole_length = mutate_f64(self.petiole_length, rng, rate, 0.02, 0.0, 0.25);
        self.petiole_width = mutate_f64(self.petiole_width, rng, rate, 0.003, 0.008, 0.05);
        self.midrib_width = mutate_f64(self.midrib_width, rng, rate, 0.02, 0.03, 0.35);
        self.vein_count = mutate_f64(self.vein_count, rng, rate, 1.0, 2.0, 14.0);
        self.venule_strength = mutate_f64(self.venule_strength, rng, rate, 0.1, 0.0, 1.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() {
                self.seed
            } else {
                other.seed
            },
            color_base: crossover_color3(self.color_base, other.color_base, rng),
            color_edge: crossover_color3(self.color_edge, other.color_edge, rng),
            serration_strength: if rng.random::<bool>() {
                self.serration_strength
            } else {
                other.serration_strength
            },
            vein_angle: if rng.random::<bool>() {
                self.vein_angle
            } else {
                other.vein_angle
            },
            micro_detail: if rng.random::<bool>() {
                self.micro_detail
            } else {
                other.micro_detail
            },
            normal_strength: if rng.random::<bool>() {
                self.normal_strength
            } else {
                other.normal_strength
            },
            lobe_count: if rng.random::<bool>() {
                self.lobe_count
            } else {
                other.lobe_count
            },
            lobe_depth: if rng.random::<bool>() {
                self.lobe_depth
            } else {
                other.lobe_depth
            },
            lobe_sharpness: if rng.random::<bool>() {
                self.lobe_sharpness
            } else {
                other.lobe_sharpness
            },
            petiole_length: if rng.random::<bool>() {
                self.petiole_length
            } else {
                other.petiole_length
            },
            petiole_width: if rng.random::<bool>() {
                self.petiole_width
            } else {
                other.petiole_width
            },
            midrib_width: if rng.random::<bool>() {
                self.midrib_width
            } else {
                other.midrib_width
            },
            vein_count: if rng.random::<bool>() {
                self.vein_count
            } else {
                other.vein_count
            },
            venule_strength: if rng.random::<bool>() {
                self.venule_strength
            } else {
                other.venule_strength
            },
        }
    }
}

// --- TwigConfig -------------------------------------------------------------

impl Genotype for TwigConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.leaf.mutate(rng, rate);
        self.stem_color = mutate_color3(self.stem_color, rng, rate, 0.07);
        self.stem_half_width = mutate_f64(self.stem_half_width, rng, rate, 0.005, 0.005, 0.05);
        self.leaf_pairs = mutate_usize(self.leaf_pairs, rng, rate, 1, 8);
        self.leaf_angle = mutate_f64(self.leaf_angle, rng, rate, 0.15, 0.1, PI);
        self.leaf_scale = mutate_f64(self.leaf_scale, rng, rate, 0.05, 0.15, 0.6);
        self.stem_curve = mutate_f64(self.stem_curve, rng, rate, 0.02, 0.0, 0.2);
        if rng.random::<f32>() < rate {
            self.sympodial = !self.sympodial;
        }
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            leaf: self.leaf.crossover(&other.leaf, rng),
            stem_color: crossover_color3(self.stem_color, other.stem_color, rng),
            stem_half_width: if rng.random::<bool>() {
                self.stem_half_width
            } else {
                other.stem_half_width
            },
            leaf_pairs: if rng.random::<bool>() {
                self.leaf_pairs
            } else {
                other.leaf_pairs
            },
            leaf_angle: if rng.random::<bool>() {
                self.leaf_angle
            } else {
                other.leaf_angle
            },
            leaf_scale: if rng.random::<bool>() {
                self.leaf_scale
            } else {
                other.leaf_scale
            },
            stem_curve: if rng.random::<bool>() {
                self.stem_curve
            } else {
                other.stem_curve
            },
            sympodial: if rng.random::<bool>() {
                self.sympodial
            } else {
                other.sympodial
            },
        }
    }
}

// --- BrickConfig ------------------------------------------------------------

impl Genotype for BrickConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.scale = mutate_f64(self.scale, rng, rate, 1.0, 1.0, 12.0).round();
        self.row_offset = mutate_f64(self.row_offset, rng, rate, 0.1, 0.0, 1.0);
        self.aspect_ratio = mutate_f64(self.aspect_ratio, rng, rate, 0.3, 1.0, 4.0);
        self.mortar_size = mutate_f64(self.mortar_size, rng, rate, 0.03, 0.01, 0.35);
        self.bevel = mutate_f64(self.bevel, rng, rate, 0.2, 0.0, 1.0);
        self.cell_variance = mutate_f64(self.cell_variance, rng, rate, 0.1, 0.0, 0.8);
        self.roughness = mutate_f64(self.roughness, rng, rate, 0.1, 0.0, 1.0);
        self.color_brick = mutate_color3(self.color_brick, rng, rate, 0.07);
        self.color_mortar = mutate_color3(self.color_mortar, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 8.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() {
                self.seed
            } else {
                other.seed
            },
            scale: if rng.random::<bool>() {
                self.scale
            } else {
                other.scale
            },
            row_offset: if rng.random::<bool>() {
                self.row_offset
            } else {
                other.row_offset
            },
            aspect_ratio: if rng.random::<bool>() {
                self.aspect_ratio
            } else {
                other.aspect_ratio
            },
            mortar_size: if rng.random::<bool>() {
                self.mortar_size
            } else {
                other.mortar_size
            },
            bevel: if rng.random::<bool>() {
                self.bevel
            } else {
                other.bevel
            },
            cell_variance: if rng.random::<bool>() {
                self.cell_variance
            } else {
                other.cell_variance
            },
            roughness: if rng.random::<bool>() {
                self.roughness
            } else {
                other.roughness
            },
            color_brick: crossover_color3(self.color_brick, other.color_brick, rng),
            color_mortar: crossover_color3(self.color_mortar, other.color_mortar, rng),
            normal_strength: if rng.random::<bool>() {
                self.normal_strength
            } else {
                other.normal_strength
            },
        }
    }
}

// --- WindowConfig -----------------------------------------------------------

impl Genotype for WindowConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.frame_width = mutate_f64(self.frame_width, rng, rate, 0.02, 0.02, 0.4);
        self.panes_x = mutate_usize(self.panes_x, rng, rate, 1, 6);
        self.panes_y = mutate_usize(self.panes_y, rng, rate, 1, 8);
        self.mullion_thickness = mutate_f64(self.mullion_thickness, rng, rate, 0.005, 0.005, 0.15);
        self.corner_radius = mutate_f64(self.corner_radius, rng, rate, 0.01, 0.0, 0.35);
        self.glass_opacity = mutate_f64(self.glass_opacity, rng, rate, 0.1, 0.0, 1.0);
        self.grime_level = mutate_f64(self.grime_level, rng, rate, 0.1, 0.0, 1.0);
        self.color_frame = mutate_color3(self.color_frame, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 6.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() {
                self.seed
            } else {
                other.seed
            },
            frame_width: if rng.random::<bool>() {
                self.frame_width
            } else {
                other.frame_width
            },
            panes_x: if rng.random::<bool>() {
                self.panes_x
            } else {
                other.panes_x
            },
            panes_y: if rng.random::<bool>() {
                self.panes_y
            } else {
                other.panes_y
            },
            mullion_thickness: if rng.random::<bool>() {
                self.mullion_thickness
            } else {
                other.mullion_thickness
            },
            corner_radius: if rng.random::<bool>() {
                self.corner_radius
            } else {
                other.corner_radius
            },
            glass_opacity: if rng.random::<bool>() {
                self.glass_opacity
            } else {
                other.glass_opacity
            },
            grime_level: if rng.random::<bool>() {
                self.grime_level
            } else {
                other.grime_level
            },
            color_frame: crossover_color3(self.color_frame, other.color_frame, rng),
            normal_strength: if rng.random::<bool>() {
                self.normal_strength
            } else {
                other.normal_strength
            },
        }
    }
}

// --- PlankConfig ------------------------------------------------------------

impl Genotype for PlankConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.plank_count = mutate_f64(self.plank_count, rng, rate, 1.0, 2.0, 12.0).round();
        self.grain_scale = mutate_f64(self.grain_scale, rng, rate, 2.0, 4.0, 24.0);
        self.joint_width = mutate_f64(self.joint_width, rng, rate, 0.02, 0.01, 0.25);
        self.stagger = mutate_f64(self.stagger, rng, rate, 0.15, 0.0, 1.0);
        self.knot_density = mutate_f64(self.knot_density, rng, rate, 0.1, 0.0, 1.0);
        self.grain_warp = mutate_f64(self.grain_warp, rng, rate, 0.1, 0.0, 1.0);
        self.color_wood_light = mutate_color3(self.color_wood_light, rng, rate, 0.07);
        self.color_wood_dark = mutate_color3(self.color_wood_dark, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 6.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() {
                self.seed
            } else {
                other.seed
            },
            plank_count: if rng.random::<bool>() {
                self.plank_count
            } else {
                other.plank_count
            },
            grain_scale: if rng.random::<bool>() {
                self.grain_scale
            } else {
                other.grain_scale
            },
            joint_width: if rng.random::<bool>() {
                self.joint_width
            } else {
                other.joint_width
            },
            stagger: if rng.random::<bool>() {
                self.stagger
            } else {
                other.stagger
            },
            knot_density: if rng.random::<bool>() {
                self.knot_density
            } else {
                other.knot_density
            },
            grain_warp: if rng.random::<bool>() {
                self.grain_warp
            } else {
                other.grain_warp
            },
            color_wood_light: crossover_color3(self.color_wood_light, other.color_wood_light, rng),
            color_wood_dark: crossover_color3(self.color_wood_dark, other.color_wood_dark, rng),
            normal_strength: if rng.random::<bool>() {
                self.normal_strength
            } else {
                other.normal_strength
            },
        }
    }
}

// --- ShingleConfig ----------------------------------------------------------

impl Genotype for ShingleConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.scale = mutate_f64(self.scale, rng, rate, 1.0, 2.0, 12.0).round();
        self.shape_profile = mutate_f64(self.shape_profile, rng, rate, 0.2, 0.0, 1.0);
        self.overlap = mutate_f64(self.overlap, rng, rate, 0.1, 0.0, 0.8);
        self.stagger = mutate_f64(self.stagger, rng, rate, 0.15, 0.0, 1.0);
        self.moss_level = mutate_f64(self.moss_level, rng, rate, 0.1, 0.0, 1.0);
        self.color_tile = mutate_color3(self.color_tile, rng, rate, 0.07);
        self.color_grout = mutate_color3(self.color_grout, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 8.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() {
                self.seed
            } else {
                other.seed
            },
            scale: if rng.random::<bool>() {
                self.scale
            } else {
                other.scale
            },
            shape_profile: if rng.random::<bool>() {
                self.shape_profile
            } else {
                other.shape_profile
            },
            overlap: if rng.random::<bool>() {
                self.overlap
            } else {
                other.overlap
            },
            stagger: if rng.random::<bool>() {
                self.stagger
            } else {
                other.stagger
            },
            moss_level: if rng.random::<bool>() {
                self.moss_level
            } else {
                other.moss_level
            },
            color_tile: crossover_color3(self.color_tile, other.color_tile, rng),
            color_grout: crossover_color3(self.color_grout, other.color_grout, rng),
            normal_strength: if rng.random::<bool>() {
                self.normal_strength
            } else {
                other.normal_strength
            },
        }
    }
}

// --- StuccoConfig -----------------------------------------------------------

impl Genotype for StuccoConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.scale = mutate_f64(self.scale, rng, rate, 1.5, 1.0, 20.0);
        self.octaves = mutate_usize(self.octaves, rng, rate, 1, 10);
        self.roughness = mutate_f64(self.roughness, rng, rate, 0.1, 0.0, 1.0);
        self.color_base = mutate_color3(self.color_base, rng, rate, 0.07);
        self.color_shadow = mutate_color3(self.color_shadow, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 6.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() {
                self.seed
            } else {
                other.seed
            },
            scale: if rng.random::<bool>() {
                self.scale
            } else {
                other.scale
            },
            octaves: if rng.random::<bool>() {
                self.octaves
            } else {
                other.octaves
            },
            roughness: if rng.random::<bool>() {
                self.roughness
            } else {
                other.roughness
            },
            color_base: crossover_color3(self.color_base, other.color_base, rng),
            color_shadow: crossover_color3(self.color_shadow, other.color_shadow, rng),
            normal_strength: if rng.random::<bool>() {
                self.normal_strength
            } else {
                other.normal_strength
            },
        }
    }
}

// --- ConcreteConfig ---------------------------------------------------------

impl Genotype for ConcreteConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.scale = mutate_f64(self.scale, rng, rate, 1.0, 1.0, 16.0);
        self.octaves = mutate_usize(self.octaves, rng, rate, 1, 10);
        self.roughness = mutate_f64(self.roughness, rng, rate, 0.1, 0.0, 1.0);
        self.formwork_lines = mutate_f64(self.formwork_lines, rng, rate, 1.0, 0.0, 12.0).round();
        self.formwork_depth = mutate_f64(self.formwork_depth, rng, rate, 0.05, 0.0, 0.5);
        self.pit_density = mutate_f64(self.pit_density, rng, rate, 0.04, 0.0, 0.45);
        self.color_base = mutate_color3(self.color_base, rng, rate, 0.07);
        self.color_pit = mutate_color3(self.color_pit, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 6.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() {
                self.seed
            } else {
                other.seed
            },
            scale: if rng.random::<bool>() {
                self.scale
            } else {
                other.scale
            },
            octaves: if rng.random::<bool>() {
                self.octaves
            } else {
                other.octaves
            },
            roughness: if rng.random::<bool>() {
                self.roughness
            } else {
                other.roughness
            },
            formwork_lines: if rng.random::<bool>() {
                self.formwork_lines
            } else {
                other.formwork_lines
            },
            formwork_depth: if rng.random::<bool>() {
                self.formwork_depth
            } else {
                other.formwork_depth
            },
            pit_density: if rng.random::<bool>() {
                self.pit_density
            } else {
                other.pit_density
            },
            color_base: crossover_color3(self.color_base, other.color_base, rng),
            color_pit: crossover_color3(self.color_pit, other.color_pit, rng),
            normal_strength: if rng.random::<bool>() {
                self.normal_strength
            } else {
                other.normal_strength
            },
        }
    }
}

// --- MetalConfig ------------------------------------------------------------

impl Genotype for MetalConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        // style: flip randomly with probability rate.
        if rng.random::<f32>() < rate {
            self.style = match self.style {
                crate::metal::MetalStyle::Brushed => crate::metal::MetalStyle::StandingSeam,
                crate::metal::MetalStyle::StandingSeam => crate::metal::MetalStyle::Brushed,
            };
        }
        self.scale = mutate_f64(self.scale, rng, rate, 1.0, 1.0, 16.0);
        self.seam_count = mutate_f64(self.seam_count, rng, rate, 1.0, 1.0, 16.0).round();
        self.seam_sharpness = mutate_f64(self.seam_sharpness, rng, rate, 0.5, 0.5, 6.0);
        self.brush_stretch = mutate_f64(self.brush_stretch, rng, rate, 1.5, 1.0, 20.0);
        self.roughness = mutate_f64(self.roughness, rng, rate, 0.1, 0.0, 1.0);
        self.metallic = mutate_f32(self.metallic, rng, rate, 0.1, 0.0, 1.0);
        self.rust_level = mutate_f64(self.rust_level, rng, rate, 0.1, 0.0, 1.0);
        self.color_metal = mutate_color3(self.color_metal, rng, rate, 0.07);
        self.color_rust = mutate_color3(self.color_rust, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 6.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() {
                self.seed
            } else {
                other.seed
            },
            style: if rng.random::<bool>() {
                self.style.clone()
            } else {
                other.style.clone()
            },
            scale: if rng.random::<bool>() {
                self.scale
            } else {
                other.scale
            },
            seam_count: if rng.random::<bool>() {
                self.seam_count
            } else {
                other.seam_count
            },
            seam_sharpness: if rng.random::<bool>() {
                self.seam_sharpness
            } else {
                other.seam_sharpness
            },
            brush_stretch: if rng.random::<bool>() {
                self.brush_stretch
            } else {
                other.brush_stretch
            },
            roughness: if rng.random::<bool>() {
                self.roughness
            } else {
                other.roughness
            },
            metallic: if rng.random::<bool>() {
                self.metallic
            } else {
                other.metallic
            },
            rust_level: if rng.random::<bool>() {
                self.rust_level
            } else {
                other.rust_level
            },
            color_metal: crossover_color3(self.color_metal, other.color_metal, rng),
            color_rust: crossover_color3(self.color_rust, other.color_rust, rng),
            normal_strength: if rng.random::<bool>() {
                self.normal_strength
            } else {
                other.normal_strength
            },
        }
    }
}

// --- PaversConfig -----------------------------------------------------------

impl Genotype for PaversConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        if rng.random::<f32>() < rate {
            self.layout = match self.layout {
                crate::pavers::PaversLayout::Square => crate::pavers::PaversLayout::Hexagonal,
                crate::pavers::PaversLayout::Hexagonal => crate::pavers::PaversLayout::Square,
            };
        }
        self.scale = mutate_f64(self.scale, rng, rate, 1.0, 1.0, 16.0).round();
        self.aspect_ratio = mutate_f64(self.aspect_ratio, rng, rate, 0.2, 0.5, 3.0);
        self.grout_width = mutate_f64(self.grout_width, rng, rate, 0.02, 0.01, 0.35);
        self.bevel = mutate_f64(self.bevel, rng, rate, 0.15, 0.0, 1.0);
        self.cell_variance = mutate_f64(self.cell_variance, rng, rate, 0.08, 0.0, 0.8);
        self.roughness = mutate_f64(self.roughness, rng, rate, 0.1, 0.0, 1.0);
        self.color_stone = mutate_color3(self.color_stone, rng, rate, 0.07);
        self.color_grout = mutate_color3(self.color_grout, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 8.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() {
                self.seed
            } else {
                other.seed
            },
            layout: if rng.random::<bool>() {
                self.layout.clone()
            } else {
                other.layout.clone()
            },
            scale: if rng.random::<bool>() {
                self.scale
            } else {
                other.scale
            },
            aspect_ratio: if rng.random::<bool>() {
                self.aspect_ratio
            } else {
                other.aspect_ratio
            },
            grout_width: if rng.random::<bool>() {
                self.grout_width
            } else {
                other.grout_width
            },
            bevel: if rng.random::<bool>() {
                self.bevel
            } else {
                other.bevel
            },
            cell_variance: if rng.random::<bool>() {
                self.cell_variance
            } else {
                other.cell_variance
            },
            roughness: if rng.random::<bool>() {
                self.roughness
            } else {
                other.roughness
            },
            color_stone: crossover_color3(self.color_stone, other.color_stone, rng),
            color_grout: crossover_color3(self.color_grout, other.color_grout, rng),
            normal_strength: if rng.random::<bool>() {
                self.normal_strength
            } else {
                other.normal_strength
            },
        }
    }
}

// --- AshlarConfig -----------------------------------------------------------

impl Genotype for AshlarConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.rows = mutate_usize(self.rows, rng, rate, 2, 8);
        self.cols = mutate_usize(self.cols, rng, rate, 2, 6);
        self.mortar_size = mutate_f64(self.mortar_size, rng, rate, 0.02, 0.005, 0.15);
        self.bevel = mutate_f64(self.bevel, rng, rate, 0.2, 0.0, 1.0);
        self.cell_variance = mutate_f64(self.cell_variance, rng, rate, 0.1, 0.0, 0.8);
        self.chisel_depth = mutate_f64(self.chisel_depth, rng, rate, 0.1, 0.0, 1.0);
        self.roughness = mutate_f64(self.roughness, rng, rate, 0.1, 0.0, 1.0);
        self.color_stone = mutate_color3(self.color_stone, rng, rate, 0.07);
        self.color_mortar = mutate_color3(self.color_mortar, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 8.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() { self.seed } else { other.seed },
            rows: if rng.random::<bool>() { self.rows } else { other.rows },
            cols: if rng.random::<bool>() { self.cols } else { other.cols },
            mortar_size: if rng.random::<bool>() { self.mortar_size } else { other.mortar_size },
            bevel: if rng.random::<bool>() { self.bevel } else { other.bevel },
            cell_variance: if rng.random::<bool>() { self.cell_variance } else { other.cell_variance },
            chisel_depth: if rng.random::<bool>() { self.chisel_depth } else { other.chisel_depth },
            roughness: if rng.random::<bool>() { self.roughness } else { other.roughness },
            color_stone: crossover_color3(self.color_stone, other.color_stone, rng),
            color_mortar: crossover_color3(self.color_mortar, other.color_mortar, rng),
            normal_strength: if rng.random::<bool>() { self.normal_strength } else { other.normal_strength },
        }
    }
}

// --- CobblestoneConfig -------------------------------------------------------

impl Genotype for CobblestoneConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.scale = mutate_f64(self.scale, rng, rate, 1.0, 2.0, 14.0);
        self.gap_width = mutate_f64(self.gap_width, rng, rate, 0.03, 0.01, 0.3);
        self.cell_variance = mutate_f64(self.cell_variance, rng, rate, 0.1, 0.0, 0.8);
        self.roundness = mutate_f64(self.roundness, rng, rate, 0.15, 0.3, 2.5);
        self.color_stone = mutate_color3(self.color_stone, rng, rate, 0.07);
        self.color_mud = mutate_color3(self.color_mud, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 8.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() { self.seed } else { other.seed },
            scale: if rng.random::<bool>() { self.scale } else { other.scale },
            gap_width: if rng.random::<bool>() { self.gap_width } else { other.gap_width },
            cell_variance: if rng.random::<bool>() { self.cell_variance } else { other.cell_variance },
            roundness: if rng.random::<bool>() { self.roundness } else { other.roundness },
            color_stone: crossover_color3(self.color_stone, other.color_stone, rng),
            color_mud: crossover_color3(self.color_mud, other.color_mud, rng),
            normal_strength: if rng.random::<bool>() { self.normal_strength } else { other.normal_strength },
        }
    }
}

// --- ThatchConfig ------------------------------------------------------------

impl Genotype for ThatchConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.density = mutate_f64(self.density, rng, rate, 2.0, 3.0, 24.0);
        self.anisotropy = mutate_f64(self.anisotropy, rng, rate, 1.0, 2.0, 20.0);
        self.warp_strength = mutate_f64(self.warp_strength, rng, rate, 0.05, 0.0, 0.6);
        self.layer_count = mutate_f64(self.layer_count, rng, rate, 1.0, 2.0, 20.0);
        self.layer_shadow = mutate_f64(self.layer_shadow, rng, rate, 0.1, 0.0, 1.0);
        self.color_straw = mutate_color3(self.color_straw, rng, rate, 0.07);
        self.color_shadow = mutate_color3(self.color_shadow, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 6.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() { self.seed } else { other.seed },
            density: if rng.random::<bool>() { self.density } else { other.density },
            anisotropy: if rng.random::<bool>() { self.anisotropy } else { other.anisotropy },
            warp_strength: if rng.random::<bool>() { self.warp_strength } else { other.warp_strength },
            layer_count: if rng.random::<bool>() { self.layer_count } else { other.layer_count },
            layer_shadow: if rng.random::<bool>() { self.layer_shadow } else { other.layer_shadow },
            color_straw: crossover_color3(self.color_straw, other.color_straw, rng),
            color_shadow: crossover_color3(self.color_shadow, other.color_shadow, rng),
            normal_strength: if rng.random::<bool>() { self.normal_strength } else { other.normal_strength },
        }
    }
}

// --- MarbleConfig ------------------------------------------------------------

impl Genotype for MarbleConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.scale = mutate_f64(self.scale, rng, rate, 1.0, 0.5, 10.0);
        self.octaves = mutate_usize(self.octaves, rng, rate, 2, 10);
        self.warp_strength = mutate_f64(self.warp_strength, rng, rate, 0.15, 0.0, 2.0);
        self.vein_frequency = mutate_f64(self.vein_frequency, rng, rate, 0.5, 0.5, 10.0);
        self.vein_sharpness = mutate_f64(self.vein_sharpness, rng, rate, 0.5, 0.3, 8.0);
        self.roughness = mutate_f64(self.roughness, rng, rate, 0.05, 0.0, 0.4);
        self.color_base = mutate_color3(self.color_base, rng, rate, 0.07);
        self.color_vein = mutate_color3(self.color_vein, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.3, 0.0, 4.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() { self.seed } else { other.seed },
            scale: if rng.random::<bool>() { self.scale } else { other.scale },
            octaves: if rng.random::<bool>() { self.octaves } else { other.octaves },
            warp_strength: if rng.random::<bool>() { self.warp_strength } else { other.warp_strength },
            vein_frequency: if rng.random::<bool>() { self.vein_frequency } else { other.vein_frequency },
            vein_sharpness: if rng.random::<bool>() { self.vein_sharpness } else { other.vein_sharpness },
            roughness: if rng.random::<bool>() { self.roughness } else { other.roughness },
            color_base: crossover_color3(self.color_base, other.color_base, rng),
            color_vein: crossover_color3(self.color_vein, other.color_vein, rng),
            normal_strength: if rng.random::<bool>() { self.normal_strength } else { other.normal_strength },
        }
    }
}

// --- CorrugatedConfig --------------------------------------------------------

impl Genotype for CorrugatedConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.ridges = mutate_f64(self.ridges, rng, rate, 2.0, 2.0, 20.0).round();
        self.ridge_depth = mutate_f64(self.ridge_depth, rng, rate, 0.2, 0.3, 2.5);
        self.roughness = mutate_f64(self.roughness, rng, rate, 0.1, 0.0, 1.0);
        self.rust_level = mutate_f64(self.rust_level, rng, rate, 0.1, 0.0, 1.0);
        self.metallic = mutate_f32(self.metallic, rng, rate, 0.1, 0.0, 1.0);
        self.color_metal = mutate_color3(self.color_metal, rng, rate, 0.07);
        self.color_rust = mutate_color3(self.color_rust, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 6.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() { self.seed } else { other.seed },
            ridges: if rng.random::<bool>() { self.ridges } else { other.ridges },
            ridge_depth: if rng.random::<bool>() { self.ridge_depth } else { other.ridge_depth },
            roughness: if rng.random::<bool>() { self.roughness } else { other.roughness },
            rust_level: if rng.random::<bool>() { self.rust_level } else { other.rust_level },
            metallic: if rng.random::<bool>() { self.metallic } else { other.metallic },
            color_metal: crossover_color3(self.color_metal, other.color_metal, rng),
            color_rust: crossover_color3(self.color_rust, other.color_rust, rng),
            normal_strength: if rng.random::<bool>() { self.normal_strength } else { other.normal_strength },
        }
    }
}

// --- AsphaltConfig -----------------------------------------------------------

impl Genotype for AsphaltConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.scale = mutate_f64(self.scale, rng, rate, 1.0, 1.0, 14.0);
        self.aggregate_density = mutate_f64(self.aggregate_density, rng, rate, 0.05, 0.02, 0.5);
        self.aggregate_scale = mutate_f64(self.aggregate_scale, rng, rate, 3.0, 4.0, 40.0);
        self.roughness = mutate_f64(self.roughness, rng, rate, 0.05, 0.5, 1.0);
        self.stain_level = mutate_f64(self.stain_level, rng, rate, 0.1, 0.0, 1.0);
        self.color_base = mutate_color3(self.color_base, rng, rate, 0.05);
        self.color_aggregate = mutate_color3(self.color_aggregate, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.3, 0.0, 4.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() { self.seed } else { other.seed },
            scale: if rng.random::<bool>() { self.scale } else { other.scale },
            aggregate_density: if rng.random::<bool>() { self.aggregate_density } else { other.aggregate_density },
            aggregate_scale: if rng.random::<bool>() { self.aggregate_scale } else { other.aggregate_scale },
            roughness: if rng.random::<bool>() { self.roughness } else { other.roughness },
            stain_level: if rng.random::<bool>() { self.stain_level } else { other.stain_level },
            color_base: crossover_color3(self.color_base, other.color_base, rng),
            color_aggregate: crossover_color3(self.color_aggregate, other.color_aggregate, rng),
            normal_strength: if rng.random::<bool>() { self.normal_strength } else { other.normal_strength },
        }
    }
}

// --- WainscotingConfig -------------------------------------------------------

impl Genotype for WainscotingConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.panels_x = mutate_usize(self.panels_x, rng, rate, 1, 4);
        self.panels_y = mutate_usize(self.panels_y, rng, rate, 1, 4);
        self.frame_width = mutate_f64(self.frame_width, rng, rate, 0.05, 0.05, 0.4);
        self.panel_inset = mutate_f64(self.panel_inset, rng, rate, 0.02, 0.0, 0.2);
        self.grain_scale = mutate_f64(self.grain_scale, rng, rate, 2.0, 4.0, 28.0);
        self.grain_warp = mutate_f64(self.grain_warp, rng, rate, 0.1, 0.0, 1.0);
        self.color_wood_light = mutate_color3(self.color_wood_light, rng, rate, 0.07);
        self.color_wood_dark = mutate_color3(self.color_wood_dark, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 8.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() { self.seed } else { other.seed },
            panels_x: if rng.random::<bool>() { self.panels_x } else { other.panels_x },
            panels_y: if rng.random::<bool>() { self.panels_y } else { other.panels_y },
            frame_width: if rng.random::<bool>() { self.frame_width } else { other.frame_width },
            panel_inset: if rng.random::<bool>() { self.panel_inset } else { other.panel_inset },
            grain_scale: if rng.random::<bool>() { self.grain_scale } else { other.grain_scale },
            grain_warp: if rng.random::<bool>() { self.grain_warp } else { other.grain_warp },
            color_wood_light: crossover_color3(self.color_wood_light, other.color_wood_light, rng),
            color_wood_dark: crossover_color3(self.color_wood_dark, other.color_wood_dark, rng),
            normal_strength: if rng.random::<bool>() { self.normal_strength } else { other.normal_strength },
        }
    }
}

// --- StainedGlassConfig ------------------------------------------------------

impl Genotype for StainedGlassConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.cell_count = mutate_usize(self.cell_count, rng, rate, 3, 30);
        self.lead_width = mutate_f64(self.lead_width, rng, rate, 0.01, 0.01, 0.15);
        self.saturation = mutate_f32(self.saturation, rng, rate, 0.1, 0.3, 1.0);
        self.glass_roughness = mutate_f64(self.glass_roughness, rng, rate, 0.02, 0.0, 0.2);
        self.grime_level = mutate_f64(self.grime_level, rng, rate, 0.05, 0.0, 0.6);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.3, 0.0, 4.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() { self.seed } else { other.seed },
            cell_count: if rng.random::<bool>() { self.cell_count } else { other.cell_count },
            lead_width: if rng.random::<bool>() { self.lead_width } else { other.lead_width },
            saturation: if rng.random::<bool>() { self.saturation } else { other.saturation },
            glass_roughness: if rng.random::<bool>() { self.glass_roughness } else { other.glass_roughness },
            grime_level: if rng.random::<bool>() { self.grime_level } else { other.grime_level },
            normal_strength: if rng.random::<bool>() { self.normal_strength } else { other.normal_strength },
        }
    }
}

// --- IronGrilleConfig --------------------------------------------------------

impl Genotype for IronGrilleConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.bars_x = mutate_usize(self.bars_x, rng, rate, 1, 12);
        self.bars_y = mutate_usize(self.bars_y, rng, rate, 1, 12);
        self.bar_width = mutate_f64(self.bar_width, rng, rate, 0.02, 0.01, 0.25);
        if rng.random::<f32>() < rate {
            self.round_bars = !self.round_bars;
        }
        self.rust_level = mutate_f64(self.rust_level, rng, rate, 0.1, 0.0, 1.0);
        self.color_iron = mutate_color3(self.color_iron, rng, rate, 0.07);
        self.color_rust = mutate_color3(self.color_rust, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 6.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() { self.seed } else { other.seed },
            bars_x: if rng.random::<bool>() { self.bars_x } else { other.bars_x },
            bars_y: if rng.random::<bool>() { self.bars_y } else { other.bars_y },
            bar_width: if rng.random::<bool>() { self.bar_width } else { other.bar_width },
            round_bars: if rng.random::<bool>() { self.round_bars } else { other.round_bars },
            rust_level: if rng.random::<bool>() { self.rust_level } else { other.rust_level },
            color_iron: crossover_color3(self.color_iron, other.color_iron, rng),
            color_rust: crossover_color3(self.color_rust, other.color_rust, rng),
            normal_strength: if rng.random::<bool>() { self.normal_strength } else { other.normal_strength },
        }
    }
}

// --- EncausticConfig ---------------------------------------------------------

impl Genotype for EncausticConfig {
    fn mutate<R: Rng>(&mut self, rng: &mut R, rate: f32) {
        self.seed = mutate_seed(self.seed, rng, rate);
        self.scale = mutate_f64(self.scale, rng, rate, 1.0, 1.0, 12.0).round();
        if rng.random::<f32>() < rate {
            self.pattern = match self.pattern {
                EncausticPattern::Checkerboard => EncausticPattern::Octagon,
                EncausticPattern::Octagon => EncausticPattern::Diamond,
                EncausticPattern::Diamond => EncausticPattern::Checkerboard,
            };
        }
        self.grout_width = mutate_f64(self.grout_width, rng, rate, 0.02, 0.01, 0.2);
        self.glaze_roughness = mutate_f64(self.glaze_roughness, rng, rate, 0.02, 0.0, 0.15);
        self.color_a = mutate_color3(self.color_a, rng, rate, 0.07);
        self.color_b = mutate_color3(self.color_b, rng, rate, 0.07);
        self.color_grout = mutate_color3(self.color_grout, rng, rate, 0.07);
        self.normal_strength = mutate_f32(self.normal_strength, rng, rate, 0.5, 0.5, 6.0);
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        Self {
            seed: if rng.random::<bool>() { self.seed } else { other.seed },
            scale: if rng.random::<bool>() { self.scale } else { other.scale },
            pattern: if rng.random::<bool>() { self.pattern.clone() } else { other.pattern.clone() },
            grout_width: if rng.random::<bool>() { self.grout_width } else { other.grout_width },
            glaze_roughness: if rng.random::<bool>() { self.glaze_roughness } else { other.glaze_roughness },
            color_a: crossover_color3(self.color_a, other.color_a, rng),
            color_b: crossover_color3(self.color_b, other.color_b, rng),
            color_grout: crossover_color3(self.color_grout, other.color_grout, rng),
            normal_strength: if rng.random::<bool>() { self.normal_strength } else { other.normal_strength },
        }
    }
}

// --- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn bark_mutate_is_deterministic() {
        let base = BarkConfig::default();
        let mut a = base.clone();
        let mut b = base.clone();
        a.mutate(&mut seeded_rng(), 1.0);
        b.mutate(&mut seeded_rng(), 1.0);
        // Same seed → same result.
        assert_eq!(a.seed, b.seed);
        assert_eq!(a.octaves, b.octaves);
    }

    #[test]
    fn bark_mutate_rate_zero_is_identity() {
        let base = BarkConfig::default();
        let mut c = base.clone();
        c.mutate(&mut seeded_rng(), 0.0);
        assert_eq!(c.seed, base.seed);
        assert_eq!(c.octaves, base.octaves);
        assert!((c.scale - base.scale).abs() < f64::EPSILON);
    }

    #[test]
    fn bark_crossover_fields_from_parents() {
        let a = BarkConfig::default();
        let b = BarkConfig {
            seed: 99,
            octaves: 3,
            scale: 8.0,
            ..BarkConfig::default()
        };
        let child = a.crossover(&b, &mut seeded_rng());
        // Every field must come from one of the two parents.
        assert!(child.seed == a.seed || child.seed == b.seed);
        assert!(child.octaves == a.octaves || child.octaves == b.octaves);
        assert!(child.scale == a.scale || child.scale == b.scale);
    }

    #[test]
    fn rock_mutate_rate_zero_is_identity() {
        let base = RockConfig::default();
        let mut c = base.clone();
        c.mutate(&mut seeded_rng(), 0.0);
        assert_eq!(c.seed, base.seed);
        assert!((c.attenuation - base.attenuation).abs() < f64::EPSILON);
    }

    #[test]
    fn ground_mutate_clamps_micro_weight() {
        let mut c = GroundConfig {
            micro_weight: 0.99,
            ..GroundConfig::default()
        };
        // Mutate with rate=1.0 many times — micro_weight must stay in [0, 1].
        let mut rng = seeded_rng();
        for _ in 0..50 {
            c.mutate(&mut rng, 1.0);
            assert!((0.0..=1.0).contains(&c.micro_weight));
        }
    }

    #[test]
    fn leaf_crossover_valid() {
        let a = LeafConfig::default();
        let b = LeafConfig {
            seed: 77,
            vein_angle: 4.0,
            ..LeafConfig::default()
        };
        let child = a.crossover(&b, &mut seeded_rng());
        assert!(child.seed == a.seed || child.seed == b.seed);
        assert!(child.vein_angle == a.vein_angle || child.vein_angle == b.vein_angle);
    }

    #[test]
    fn twig_mutate_delegates_to_leaf() {
        let base = TwigConfig::default();
        let mut c = base.clone();
        // Rate = 1.0 always mutates — leaf seed must change from default (rarely stays same).
        // We just check it doesn't panic and leaf_pairs stays in range.
        c.mutate(&mut seeded_rng(), 1.0);
        assert!((1..=8).contains(&c.leaf_pairs));
        assert!(c.stem_half_width >= 0.005 && c.stem_half_width <= 0.05);
    }

    #[test]
    fn twig_crossover_valid() {
        let a = TwigConfig {
            sympodial: false,
            leaf_pairs: 2,
            ..TwigConfig::default()
        };
        let b = TwigConfig {
            sympodial: true,
            leaf_pairs: 6,
            ..TwigConfig::default()
        };
        let child = a.crossover(&b, &mut seeded_rng());
        assert!(child.leaf_pairs == a.leaf_pairs || child.leaf_pairs == b.leaf_pairs);
        assert!(child.sympodial == a.sympodial || child.sympodial == b.sympodial);
    }
}
