//! Byte-parity guard for surface generators ported to the
//! `surface::generate_surface` driver.
//!
//! The golden hashes below were captured from the pre-port (hand-rolled
//! pixel-loop) implementations at 64×64.  The port moves packing into the
//! shared driver without changing any math, so output must stay
//! byte-identical.  If a hash changes on purpose (intentional visual
//! change), re-capture with `cargo test --test surface_parity -- --nocapture`
//! after temporarily printing the new values.

use bevy_symbios_texture::ashlar::{AshlarConfig, AshlarGenerator};
use bevy_symbios_texture::asphalt::{AsphaltConfig, AsphaltGenerator};
use bevy_symbios_texture::bark::{BarkConfig, BarkGenerator};
use bevy_symbios_texture::brick::{BrickConfig, BrickGenerator};
use bevy_symbios_texture::cobblestone::{CobblestoneConfig, CobblestoneGenerator};
use bevy_symbios_texture::concrete::{ConcreteConfig, ConcreteGenerator};
use bevy_symbios_texture::corrugated::{CorrugatedConfig, CorrugatedGenerator};
use bevy_symbios_texture::encaustic::{EncausticConfig, EncausticGenerator, EncausticPattern};
use bevy_symbios_texture::generator::{TextureGenerator, TextureMap};
use bevy_symbios_texture::ground::{GroundConfig, GroundGenerator};
use bevy_symbios_texture::marble::{MarbleConfig, MarbleGenerator};
use bevy_symbios_texture::metal::{MetalConfig, MetalGenerator, MetalStyle};
use bevy_symbios_texture::pavers::{PaversConfig, PaversGenerator, PaversLayout};
use bevy_symbios_texture::plank::{PlankConfig, PlankGenerator};
use bevy_symbios_texture::rock::{RockConfig, RockGenerator};
use bevy_symbios_texture::shingle::{ShingleConfig, ShingleGenerator};
use bevy_symbios_texture::stucco::{StuccoConfig, StuccoGenerator};
use bevy_symbios_texture::thatch::{ThatchConfig, ThatchGenerator};
use bevy_symbios_texture::wainscoting::{WainscotingConfig, WainscotingGenerator};

/// FNV-1a over all three pixel buffers — dependency-free and stable across
/// platforms and Rust versions (unlike `DefaultHasher`).
fn fnv1a(map: &TextureMap) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for buf in [&map.albedo, &map.normal, &map.roughness] {
        for &b in buf.iter() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

fn hash_of(generator: &dyn TextureGenerator) -> u64 {
    fnv1a(&generator.generate(64, 64).expect("64x64 generation"))
}

#[test]
fn rock_output_is_byte_stable() {
    let default_hash = hash_of(&RockGenerator::new(RockConfig::default()));
    let varied_hash = hash_of(&RockGenerator::new(RockConfig {
        seed: 99,
        scale: 5.0,
        attenuation: 1.5,
        ..RockConfig::default()
    }));
    println!("rock: default={default_hash:#018x} varied={varied_hash:#018x}");
    assert_eq!(default_hash, GOLDEN_ROCK_DEFAULT);
    assert_eq!(varied_hash, GOLDEN_ROCK_VARIED);
}

#[test]
fn stucco_output_is_byte_stable() {
    let default_hash = hash_of(&StuccoGenerator::new(StuccoConfig::default()));
    let varied_hash = hash_of(&StuccoGenerator::new(StuccoConfig {
        seed: 5,
        scale: 4.0,
        roughness: 0.7,
        ..StuccoConfig::default()
    }));
    println!("stucco: default={default_hash:#018x} varied={varied_hash:#018x}");
    assert_eq!(default_hash, GOLDEN_STUCCO_DEFAULT);
    assert_eq!(varied_hash, GOLDEN_STUCCO_VARIED);
}

#[test]
fn concrete_output_is_byte_stable() {
    let default_hash = hash_of(&ConcreteGenerator::new(ConcreteConfig::default()));
    let varied_hash = hash_of(&ConcreteGenerator::new(ConcreteConfig {
        seed: 3,
        formwork_lines: 0.0,
        pit_density: 0.3,
        ..ConcreteConfig::default()
    }));
    println!("concrete: default={default_hash:#018x} varied={varied_hash:#018x}");
    assert_eq!(default_hash, GOLDEN_CONCRETE_DEFAULT);
    assert_eq!(varied_hash, GOLDEN_CONCRETE_VARIED);
}

/// One (name, default-hash, varied-hash) parity case per ported generator.
/// Each `case!` row builds the default config and a varied config touching
/// seed plus shape parameters (including enum branches where present).
macro_rules! parity_case {
    ($name:ident, $gen:ident, $default:expr, $varied:expr, $gd:expr, $gv:expr) => {
        #[test]
        fn $name() {
            let default_hash = hash_of(&$gen::new($default));
            let varied_hash = hash_of(&$gen::new($varied));
            println!(
                "{}: default={default_hash:#018x} varied={varied_hash:#018x}",
                stringify!($name)
            );
            assert_eq!(default_hash, $gd, "default config drifted");
            assert_eq!(varied_hash, $gv, "varied config drifted");
        }
    };
}

parity_case!(
    ground_output_is_byte_stable,
    GroundGenerator,
    GroundConfig::default(),
    GroundConfig {
        seed: 99,
        macro_scale: 4.0,
        micro_weight: 0.6,
        ..GroundConfig::default()
    },
    GOLDEN_GROUND_DEFAULT,
    GOLDEN_GROUND_VARIED
);

parity_case!(
    marble_output_is_byte_stable,
    MarbleGenerator,
    MarbleConfig::default(),
    MarbleConfig {
        seed: 9,
        warp_strength: 1.0,
        vein_frequency: 5.0,
        ..MarbleConfig::default()
    },
    GOLDEN_MARBLE_DEFAULT,
    GOLDEN_MARBLE_VARIED
);

parity_case!(
    asphalt_output_is_byte_stable,
    AsphaltGenerator,
    AsphaltConfig::default(),
    AsphaltConfig {
        seed: 4,
        aggregate_density: 0.35,
        stain_level: 0.6,
        ..AsphaltConfig::default()
    },
    GOLDEN_ASPHALT_DEFAULT,
    GOLDEN_ASPHALT_VARIED
);

parity_case!(
    metal_output_is_byte_stable,
    MetalGenerator,
    MetalConfig::default(),
    MetalConfig {
        seed: 8,
        style: MetalStyle::StandingSeam,
        rust_level: 0.5,
        ..MetalConfig::default()
    },
    GOLDEN_METAL_DEFAULT,
    GOLDEN_METAL_VARIED
);

parity_case!(
    corrugated_output_is_byte_stable,
    CorrugatedGenerator,
    CorrugatedConfig::default(),
    CorrugatedConfig {
        seed: 2,
        ridges: 12.0,
        rust_level: 0.6,
        ..CorrugatedConfig::default()
    },
    GOLDEN_CORRUGATED_DEFAULT,
    GOLDEN_CORRUGATED_VARIED
);

parity_case!(
    thatch_output_is_byte_stable,
    ThatchGenerator,
    ThatchConfig::default(),
    ThatchConfig {
        seed: 3,
        density: 18.0,
        layer_count: 12.0,
        ..ThatchConfig::default()
    },
    GOLDEN_THATCH_DEFAULT,
    GOLDEN_THATCH_VARIED
);

parity_case!(
    shingle_output_is_byte_stable,
    ShingleGenerator,
    ShingleConfig::default(),
    ShingleConfig {
        seed: 5,
        shape_profile: 1.0,
        moss_level: 0.5,
        ..ShingleConfig::default()
    },
    GOLDEN_SHINGLE_DEFAULT,
    GOLDEN_SHINGLE_VARIED
);

parity_case!(
    wainscoting_output_is_byte_stable,
    WainscotingGenerator,
    WainscotingConfig::default(),
    WainscotingConfig {
        seed: 6,
        panels_x: 2,
        panels_y: 1,
        ..WainscotingConfig::default()
    },
    GOLDEN_WAINSCOTING_DEFAULT,
    GOLDEN_WAINSCOTING_VARIED
);

parity_case!(
    encaustic_output_is_byte_stable,
    EncausticGenerator,
    EncausticConfig::default(),
    EncausticConfig {
        seed: 7,
        pattern: EncausticPattern::Diamond,
        scale: 3.0,
        ..EncausticConfig::default()
    },
    GOLDEN_ENCAUSTIC_DEFAULT,
    GOLDEN_ENCAUSTIC_VARIED
);

parity_case!(
    pavers_output_is_byte_stable,
    PaversGenerator,
    PaversConfig::default(),
    PaversConfig {
        seed: 11,
        layout: PaversLayout::Hexagonal,
        grout_width: 0.15,
        ..PaversConfig::default()
    },
    GOLDEN_PAVERS_DEFAULT,
    GOLDEN_PAVERS_VARIED
);

parity_case!(
    ashlar_output_is_byte_stable,
    AshlarGenerator,
    AshlarConfig::default(),
    AshlarConfig {
        seed: 12,
        rows: 6,
        cols: 3,
        ..AshlarConfig::default()
    },
    GOLDEN_ASHLAR_DEFAULT,
    GOLDEN_ASHLAR_VARIED
);

parity_case!(
    cobblestone_output_is_byte_stable,
    CobblestoneGenerator,
    CobblestoneConfig::default(),
    CobblestoneConfig {
        seed: 14,
        scale: 9.0,
        roundness: 0.7,
        ..CobblestoneConfig::default()
    },
    GOLDEN_COBBLESTONE_DEFAULT,
    GOLDEN_COBBLESTONE_VARIED
);

parity_case!(
    brick_output_is_byte_stable,
    BrickGenerator,
    BrickConfig::default(),
    BrickConfig {
        seed: 15,
        row_offset: 0.333,
        bevel: 0.2,
        ..BrickConfig::default()
    },
    GOLDEN_BRICK_DEFAULT,
    GOLDEN_BRICK_VARIED
);

parity_case!(
    bark_output_is_byte_stable,
    BarkGenerator,
    BarkConfig::default(),
    BarkConfig {
        seed: 1,
        furrow_multiplier: 0.5,
        ..BarkConfig::default()
    },
    GOLDEN_BARK_DEFAULT,
    GOLDEN_BARK_VARIED
);

parity_case!(
    plank_output_is_byte_stable,
    PlankGenerator,
    PlankConfig::default(),
    PlankConfig {
        seed: 2,
        plank_count: 8.0,
        knot_density: 0.5,
        ..PlankConfig::default()
    },
    GOLDEN_PLANK_DEFAULT,
    GOLDEN_PLANK_VARIED
);

// Captured from the pre-port implementations (this commit, 64×64).
const GOLDEN_ROCK_DEFAULT: u64 = 0x5305_c95c_840b_981f;
const GOLDEN_ROCK_VARIED: u64 = 0x7df4_7bb0_3f32_dad0;
const GOLDEN_STUCCO_DEFAULT: u64 = 0xbf2a_3e3f_927e_ccfd;
const GOLDEN_STUCCO_VARIED: u64 = 0xfe3d_6385_d1b8_bcd6;
const GOLDEN_CONCRETE_DEFAULT: u64 = 0x0bc3_50b4_b305_85d9;
const GOLDEN_CONCRETE_VARIED: u64 = 0x3a6c_e43c_ec5f_72a3;
const GOLDEN_GROUND_DEFAULT: u64 = 0x277f_ed4b_5ff6_dfed;
const GOLDEN_GROUND_VARIED: u64 = 0xe344_864b_bae8_b317;
// Re-captured for the 0.6.0 warp_octaves change — see the bark note below.
const GOLDEN_MARBLE_DEFAULT: u64 = 0x9586_ebd0_46a9_d68a;
const GOLDEN_MARBLE_VARIED: u64 = 0xc401_f50f_b8e0_05ec;
const GOLDEN_ASPHALT_DEFAULT: u64 = 0xc7b3_3ae6_5e1d_bb6e;
const GOLDEN_ASPHALT_VARIED: u64 = 0x8fba_66ad_f373_942f;
const GOLDEN_METAL_DEFAULT: u64 = 0xae99_2134_2632_3ede;
const GOLDEN_METAL_VARIED: u64 = 0xd7b1_5559_aaa5_e5c5;
const GOLDEN_CORRUGATED_DEFAULT: u64 = 0xeb4b_7dd3_71b8_1b62;
const GOLDEN_CORRUGATED_VARIED: u64 = 0x1b86_4605_53c1_6d2e;
const GOLDEN_THATCH_DEFAULT: u64 = 0x4a44_44c6_e8e1_9029;
const GOLDEN_THATCH_VARIED: u64 = 0xa17f_cdd2_97e5_8d8c;
const GOLDEN_SHINGLE_DEFAULT: u64 = 0xb08d_44d7_0fc6_0f7b;
const GOLDEN_SHINGLE_VARIED: u64 = 0x7579_1303_36c3_7c6e;
const GOLDEN_WAINSCOTING_DEFAULT: u64 = 0xdb94_4dce_df5e_88d8;
const GOLDEN_WAINSCOTING_VARIED: u64 = 0xce1c_1f7b_58a0_e510;
const GOLDEN_ENCAUSTIC_DEFAULT: u64 = 0x0d09_564f_4dfd_0155;
const GOLDEN_ENCAUSTIC_VARIED: u64 = 0x03ae_8793_a709_1791;
const GOLDEN_PAVERS_DEFAULT: u64 = 0xa63b_6090_7b5e_f446;
const GOLDEN_PAVERS_VARIED: u64 = 0x9a03_b216_1fee_cfd4;
const GOLDEN_ASHLAR_DEFAULT: u64 = 0x05fa_5166_f4bd_cf6c;
const GOLDEN_ASHLAR_VARIED: u64 = 0xb70c_f3b9_bf55_c6bf;
const GOLDEN_COBBLESTONE_DEFAULT: u64 = 0x6666_008f_1d37_58e5;
const GOLDEN_COBBLESTONE_VARIED: u64 = 0xede8_e451_cbc8_2c5c;
const GOLDEN_BRICK_DEFAULT: u64 = 0x1975_181d_137c_9798;
const GOLDEN_BRICK_VARIED: u64 = 0x04dd_2e4f_0977_76fd;
// Bark and marble were re-captured after the intentional visual change in
// 0.6.0: warp layers now run `warp_octaves` (default 3) instead of the full
// base `octaves` count (accepted drift, issue #78).
const GOLDEN_BARK_DEFAULT: u64 = 0x8433_25c5_18fe_7eb3;
const GOLDEN_BARK_VARIED: u64 = 0x8e3c_0a26_cc91_3674;
const GOLDEN_PLANK_DEFAULT: u64 = 0x07fe_04ba_940c_ae21;
const GOLDEN_PLANK_VARIED: u64 = 0x94e1_59e5_9544_8098;
