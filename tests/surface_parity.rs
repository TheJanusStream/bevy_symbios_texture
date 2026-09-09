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
use bevy_symbios_texture::chain_link::{ChainLinkConfig, ChainLinkGenerator};
use bevy_symbios_texture::cobblestone::{CobblestoneConfig, CobblestoneGenerator};
use bevy_symbios_texture::concrete::{ConcreteConfig, ConcreteGenerator};
use bevy_symbios_texture::corrugated::{CorrugatedConfig, CorrugatedGenerator};
use bevy_symbios_texture::encaustic::{EncausticConfig, EncausticGenerator, EncausticPattern};
use bevy_symbios_texture::generator::{TextureGenerator, TextureMap};
use bevy_symbios_texture::ground::{GroundConfig, GroundGenerator};
use bevy_symbios_texture::iron_grille::{IronGrilleConfig, IronGrilleGenerator};
use bevy_symbios_texture::marble::{MarbleConfig, MarbleGenerator};
use bevy_symbios_texture::metal::{MetalConfig, MetalGenerator, MetalStyle};
use bevy_symbios_texture::pavers::{PaversConfig, PaversGenerator, PaversLayout};
use bevy_symbios_texture::plank::{PlankConfig, PlankGenerator};
use bevy_symbios_texture::rock::{RockConfig, RockGenerator};
use bevy_symbios_texture::shingle::{ShingleConfig, ShingleGenerator};
use bevy_symbios_texture::stained_glass::{StainedGlassConfig, StainedGlassGenerator};
use bevy_symbios_texture::stucco::{StuccoConfig, StuccoGenerator};
use bevy_symbios_texture::thatch::{ThatchConfig, ThatchGenerator};
use bevy_symbios_texture::wainscoting::{WainscotingConfig, WainscotingGenerator};
use bevy_symbios_texture::window::{WindowConfig, WindowGenerator};

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

// The four alpha-card generators `symbios-texture` 0.7.0 moved onto the shared
// surface driver (its #19).  Their baselines were captured on 0.6.0, before the
// port, and the port left every one of the eight unmoved — so these rows pin the
// port, not a re-capture of it.  `hash_of` covers albedo, normal and roughness;
// the alpha channel a card also writes is not in the hash.
parity_case!(
    chain_link_output_is_byte_stable,
    ChainLinkGenerator,
    ChainLinkConfig::default(),
    ChainLinkConfig {
        seed: 5,
        cell_count: 12.0,
        wire_radius: 0.1,
        rust_level: 0.6,
        ..ChainLinkConfig::default()
    },
    GOLDEN_CHAIN_LINK_DEFAULT,
    GOLDEN_CHAIN_LINK_VARIED
);

parity_case!(
    iron_grille_output_is_byte_stable,
    IronGrilleGenerator,
    IronGrilleConfig::default(),
    IronGrilleConfig {
        seed: 9,
        bars_x: 6,
        bars_y: 3,
        round_bars: false,
        rust_level: 0.7,
        ..IronGrilleConfig::default()
    },
    GOLDEN_IRON_GRILLE_DEFAULT,
    GOLDEN_IRON_GRILLE_VARIED
);

parity_case!(
    stained_glass_output_is_byte_stable,
    StainedGlassGenerator,
    StainedGlassConfig::default(),
    StainedGlassConfig {
        seed: 4,
        cell_count: 20,
        lead_width: 0.08,
        grime_level: 0.3,
        ..StainedGlassConfig::default()
    },
    GOLDEN_STAINED_GLASS_DEFAULT,
    GOLDEN_STAINED_GLASS_VARIED
);

parity_case!(
    window_output_is_byte_stable,
    WindowGenerator,
    WindowConfig::default(),
    WindowConfig {
        seed: 7,
        panes_x: 3,
        panes_y: 2,
        glass_opacity: 0.55,
        grime_level: 0.4,
        ..WindowConfig::default()
    },
    GOLDEN_WINDOW_DEFAULT,
    GOLDEN_WINDOW_VARIED
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
// Re-blessed for `symbios-texture` 0.7.0.  The varied case is the only
// `Hexagonal` row here, and 0.6.0 fixed two things about that layout: the U
// seam did not tile (its #13 — a flat-top lattice needs an even column count)
// and `hex_sdf`'s `r` is the apothem, not the circumradius, so `hex_cell` drew
// a hexagon that strictly contained its own Voronoi cell and no pixel was ever
// grout (its #17).  `PaversLayout::Square` is untouched, hence the default
// holding.
const GOLDEN_PAVERS_VARIED: u64 = 0x0101_c664_d376_0696;
const GOLDEN_ASHLAR_DEFAULT: u64 = 0x05fa_5166_f4bd_cf6c;
const GOLDEN_ASHLAR_VARIED: u64 = 0xb70c_f3b9_bf55_c6bf;
const GOLDEN_COBBLESTONE_DEFAULT: u64 = 0x6666_008f_1d37_58e5;
const GOLDEN_COBBLESTONE_VARIED: u64 = 0xede8_e451_cbc8_2c5c;
// Re-blessed for `symbios-texture` 0.4.3, which wrapped the brick column
// index so a brick straddling the U seam is one colour rather than two. Its
// changelog called the move out and said downstream goldens needed
// re-capturing; this pair was missed, so these two cases had been failing
// since 0.4.3 published — on the untouched tree, at 0.4.4, before any of the
// 0.11 work. Both configs stagger their courses (`BrickConfig::default` has a
// non-zero `row_offset`, and the varied case sets 0.333), which is exactly the
// case the fix moves.
const GOLDEN_BRICK_DEFAULT: u64 = 0x92ec_9e8c_cc1f_1a4e;
// Re-blessed again for `symbios-texture` 0.7.0: `BrickGenerator::new` now snaps
// `row_offset` to a whole fraction of the scale exactly as the genotype fixup
// already did (its #14), so the varied case's 0.333 at scale 4 becomes 0.25.
// The default's `row_offset` is already snapped, which is why it holds.
const GOLDEN_BRICK_VARIED: u64 = 0x5dd4_30c4_7b79_4145;
// Bark and marble were re-captured after the intentional visual change in
// 0.6.0: warp layers now run `warp_octaves` (default 3) instead of the full
// base `octaves` count (accepted drift, issue #78).
const GOLDEN_BARK_DEFAULT: u64 = 0x8433_25c5_18fe_7eb3;
const GOLDEN_BARK_VARIED: u64 = 0x8e3c_0a26_cc91_3674;
// Re-blessed for `symbios-texture` 0.7.0.  0.6.0 moved plank onto the
// `SurfaceCell` driver byte-for-byte by preserving the old loop's truncated
// joint byte (`(0.92 * 255.0) as u8` = 234, where `surface::pack_texel`
// rounds); 0.7.0 harmonised it to the rounded 235 (its #18), which moves both
// rows and nothing else.
const GOLDEN_PLANK_DEFAULT: u64 = 0x317f_2908_1aba_2cc1;
const GOLDEN_PLANK_VARIED: u64 = 0x8b18_3859_5ed1_ff38;

// Captured on `symbios-texture` 0.6.0 for the four alpha-card generators, and
// unmoved by the 0.7.0 port that put them on the shared surface driver.
const GOLDEN_CHAIN_LINK_DEFAULT: u64 = 0xcf38_9fce_4b5f_f9ad;
const GOLDEN_CHAIN_LINK_VARIED: u64 = 0x6450_202a_9380_34c4;
const GOLDEN_IRON_GRILLE_DEFAULT: u64 = 0x13ae_3556_6bcd_8e76;
const GOLDEN_IRON_GRILLE_VARIED: u64 = 0x108f_6a8e_b810_dd81;
const GOLDEN_STAINED_GLASS_DEFAULT: u64 = 0xe82d_df95_1c46_e6fa;
const GOLDEN_STAINED_GLASS_VARIED: u64 = 0xdf93_dfd1_7b18_9623;
const GOLDEN_WINDOW_DEFAULT: u64 = 0x0cdb_aaa8_9459_b670;
const GOLDEN_WINDOW_VARIED: u64 = 0xbb27_9539_7751_c6ba;
