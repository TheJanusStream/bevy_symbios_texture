# bevy_symbios_texture

Procedural, tileable texture generation for [Bevy](https://bevyengine.org/).

Generates albedo, normal, and roughness (ORM) maps entirely on the CPU — no
asset files required. All surface textures are seamlessly tileable by
construction via toroidal 4-D noise mapping. Alpha-masked card textures (leaf,
twig, window, stained glass, iron grille) produce per-pixel transparency and
do not tile.

## Bevy compatibility

| bevy_symbios_texture | Bevy |
|----------------------|------|
| 0.3                  | 0.18 |

## Installation

```toml
[dependencies]
bevy_symbios_texture = "0.3"

# Optional: egui editor panel (required for the texture_viewer example)
bevy_symbios_texture = { version = "0.3", features = ["egui"] }
```

## Quick start

### Synchronous (blocking)

Suitable for startup systems or contexts where a small generation time is acceptable.

```rust
use bevy::prelude::*;
use bevy_symbios_texture::{
    bark::{BarkConfig, BarkGenerator},
    generator::{TextureGenerator, map_to_images},
};

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let map = BarkGenerator::new(BarkConfig::default())
        .generate(512, 512)
        .expect("valid dimensions");

    let handles = map_to_images(map, &mut images);

    commands.spawn(Sprite {
        image: handles.albedo,
        ..default()
    });
}
```

### Asynchronous (non-blocking, recommended)

Offloads pixel math to a private, bounded rayon thread pool (capped at 4
concurrent tasks) so the main thread is never stalled. On WASM, falls back to
Bevy's `AsyncComputeTaskPool`.

```rust
use bevy::prelude::*;
use bevy_symbios_texture::{
    SymbiosTexturePlugin,
    async_gen::{PendingTexture, TextureReady},
    bark::BarkConfig,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SymbiosTexturePlugin)   // registers the polling system
        .add_systems(Startup, spawn_task)
        .add_systems(Update, on_ready)
        .run();
}

fn spawn_task(mut commands: Commands) {
    commands.spawn(PendingTexture::bark(BarkConfig::default(), 1024, 1024));
}

fn on_ready(
    mut commands: Commands,
    ready: Query<(Entity, &TextureReady)>,
) {
    for (entity, tex) in &ready {
        commands.entity(entity).despawn();
        // tex.0.albedo / tex.0.normal / tex.0.roughness are Handle<Image>
    }
}
```

Dropping a `PendingTexture` entity before generation completes sets a
cancellation flag; tasks that have not yet started exit without doing any work.

## Generators

### Surface textures (tileable)

All tileable generators produce three seamlessly-repeating maps:

| Map        | Format           | Contents                                  |
|------------|------------------|-------------------------------------------|
| `albedo`   | `Rgba8UnormSrgb` | Base colour                               |
| `normal`   | `Rgba8Unorm`     | Tangent-space normal (R=X, G=Y, B=Z)      |
| `roughness`| `Rgba8Unorm`     | ORM: R=Occlusion, G=Roughness, B=Metallic |

Upload with [`map_to_images`] to get repeat-wrapping samplers.

#### Bark

Domain-warped FBM noise with an anisotropic Worley plate layer for rhytidome
furrows, producing fibrous, streaked bark grain.

```rust
use bevy_symbios_texture::bark::BarkConfig;

let config = BarkConfig {
    seed: 42,
    scale: 4.0,             // spatial frequency of the pattern
    octaves: 6,             // FBM detail levels
    warp_u: 0.15,           // lateral warp strength
    warp_v: 0.55,           // vertical (fibre) warp strength
    color_light: [0.45, 0.28, 0.14],  // ridge colour, linear RGB
    color_dark:  [0.18, 0.10, 0.05],  // groove colour, linear RGB
    normal_strength: 3.0,
    furrow_multiplier: 0.55, // blend weight of the Worley plate layer [0, 1]
    furrow_scale_u: 2.0,     // horizontal cell frequency (higher = narrower plates)
    furrow_scale_v: 0.25,    // vertical cell frequency (lower = longer plates)
    furrow_shape: 0.4,       // plate height power (<1 widens plateau, sharpens cracks)
};
```

#### Rock

Ridged multifractal noise for cracked, faceted stone.

```rust
use bevy_symbios_texture::rock::RockConfig;

let config = RockConfig {
    seed: 7,
    scale: 3.0,
    octaves: 8,
    attenuation: 2.0,    // ridge sharpness (higher = sharper)
    color_light: [0.37, 0.42, 0.36],
    color_dark:  [0.22, 0.20, 0.18],
    normal_strength: 4.0,
};
```

#### Ground

Blended dual-scale FBM for organic soil / dirt surfaces.

```rust
use bevy_symbios_texture::ground::GroundConfig;

let config = GroundConfig {
    seed: 13,
    macro_scale: 2.0,    // large soil-patch scale
    macro_octaves: 5,
    micro_scale: 8.0,    // fine grain scale
    micro_octaves: 4,
    micro_weight: 0.35,  // 0.0 = all macro, 1.0 = all micro
    color_dry:   [0.52, 0.40, 0.26],
    color_moist: [0.28, 0.20, 0.12],
    normal_strength: 2.0,
};
```

#### Brick

Grid-based SDF with per-cell colour hashing and configurable mortar/bonding pattern.

```rust
use bevy_symbios_texture::brick::BrickConfig;

let config = BrickConfig {
    seed: 42,
    scale: 4.0,          // number of brick rows across the tile
    row_offset: 0.5,     // 0.0 = stack bond, 0.5 = running bond, 0.333 = third bond
    aspect_ratio: 2.0,   // brick width-to-height ratio
    mortar_size: 0.05,   // mortar gap as a fraction of cell height [0, 0.4]
    bevel: 0.5,          // corner bevel radius as a fraction of mortar_size [0, 1]
    cell_variance: 0.15, // per-brick colour jitter [0, 1]
    roughness: 0.5,      // surface pitting noise intensity [0, 1]
    color_brick:  [0.56, 0.28, 0.18],
    color_mortar: [0.76, 0.73, 0.67],
    normal_strength: 4.0,
};
```

#### Plank

Anisotropic grain FBM with domain warp, Worley knots, and horizontal joint gaps.
Each plank row has an independent de-correlated grain phase.

```rust
use bevy_symbios_texture::plank::PlankConfig;

let config = PlankConfig {
    seed: 42,
    plank_count: 5.0,     // number of planks visible vertically
    grain_scale: 12.0,    // controls how fine the grain lines are
    joint_width: 0.06,    // gap between planks as a fraction of plank height [0, 0.3]
    stagger: 0.5,         // horizontal stagger of end-joints [0, 1]
    knot_density: 0.25,   // fraction of cells that contain a Worley knot [0, 1]
    grain_warp: 0.35,     // domain-warp strength that bends grain lines [0, 1]
    color_wood_light: [0.72, 0.52, 0.30],
    color_wood_dark:  [0.42, 0.26, 0.12],
    normal_strength: 2.5,
};
```

#### Concrete

Smooth FBM surface relief with optional horizontal formwork-panel seams and
scattered air-pocket pits.

```rust
use bevy_symbios_texture::concrete::ConcreteConfig;

let config = ConcreteConfig {
    seed: 17,
    scale: 5.0,
    octaves: 5,
    roughness: 0.45,          // overall bump amplitude [0, 1]
    formwork_lines: 4.0,      // number of horizontal panel seams [0 = none]
    formwork_depth: 0.12,     // groove depth of seams [0, 1]
    pit_density: 0.08,        // air-pocket density [0, 0.5]
    color_base: [0.55, 0.54, 0.52],
    color_pit:  [0.35, 0.34, 0.33],
    normal_strength: 2.5,
};
```

#### Metal

Brushed-metal (anisotropic FBM scratches) or standing-seam roof panels, with
optional rust-patch weathering.

```rust
use bevy_symbios_texture::metal::{MetalConfig, MetalStyle};

let config = MetalConfig {
    seed: 31,
    style: MetalStyle::Brushed, // or MetalStyle::StandingSeam
    scale: 6.0,
    seam_count: 6.0,      // StandingSeam: number of ridges across the tile
    seam_sharpness: 2.5,  // StandingSeam: 0.5 = sinusoidal, 4.0 = sharp
    brush_stretch: 8.0,   // Brushed: anisotropy (higher = longer horizontal scratches)
    roughness: 0.25,      // micro-roughness amplitude [0, 1]
    metallic: 0.85,       // metallic value for clean areas [0, 1]
    rust_level: 0.15,     // rust-patch coverage [0 = none, 1 = heavy]
    color_metal: [0.42, 0.44, 0.47],
    color_rust:  [0.42, 0.24, 0.12],
    normal_strength: 3.0,
};
```

#### Shingle

Overlapping roof shingles or tiles with configurable profile shape, moss growth,
and staggered bonding.

```rust
use bevy_symbios_texture::shingle::ShingleConfig;

let config = ShingleConfig {
    seed: 42,
    scale: 5.0,           // number of shingle rows across the tile
    shape_profile: 0.5,   // 0.0 = square/flat, 1.0 = scalloped (half-circle cut)
    overlap: 0.45,        // fraction of each shingle hidden under the row above [0, 0.8]
    stagger: 0.5,         // horizontal stagger of alternate rows [0, 1]
    moss_level: 0.18,     // moss/algae growth on the lower exposed edge [0, 1]
    color_tile:  [0.40, 0.25, 0.18],
    color_grout: [0.18, 0.14, 0.12],
    normal_strength: 5.0,
};
```

#### Pavers

Square or flat-top hexagonal paving stones with grout joints, per-stone colour
variance, and a rounded-box SDF bevel.

```rust
use bevy_symbios_texture::pavers::{PaversConfig, PaversLayout};

let config = PaversConfig {
    seed: 23,
    scale: 5.0,           // roughly the number of pavers across the tile
    aspect_ratio: 1.0,    // width-to-height ratio for Square layout (ignored for Hexagonal)
    grout_width: 0.08,    // grout gap as a fraction of stone size [0, 0.4]
    bevel: 0.5,           // corner bevel radius as a fraction of grout half-width [0, 1]
    cell_variance: 0.10,  // per-paver colour jitter [0, 1]
    roughness: 0.30,      // surface FBM micro-detail amplitude [0, 1]
    color_stone: [0.48, 0.44, 0.40],
    color_grout: [0.28, 0.27, 0.26],
    layout: PaversLayout::Square, // or PaversLayout::Hexagonal
    normal_strength: 3.5,
};
```

#### Stucco

High-frequency FBM bumps over a flat matte base — typical of sand-float or
pebble-dash exterior render.  Entirely matte with zero metallic response.

```rust
use bevy_symbios_texture::stucco::StuccoConfig;

let config = StuccoConfig {
    seed: 13,
    scale: 8.0,       // bump density (higher = finer texture)
    octaves: 6,
    roughness: 0.35,  // bump amplitude / surface relief depth [0, 1]
    color_base:   [0.92, 0.89, 0.84],
    color_shadow: [0.72, 0.70, 0.66],
    normal_strength: 2.0,
};
```

#### Ashlar

Irregular cut-stone masonry with per-block colour variance, chisel-edge
darkening, and configurable mortar joints.

```rust
use bevy_symbios_texture::ashlar::AshlarConfig;

let config = AshlarConfig {
    seed: 13,
    rows: 4,              // number of stone courses (rows) [2, 8]
    cols: 4,              // base blocks per course [2, 6]; each row may vary ±1
    mortar_size: 0.04,    // mortar gap as a fraction of average cell size [0, 0.15]
    bevel: 0.4,           // corner bevel as fraction of mortar_size [0, 1]
    cell_variance: 0.18,  // per-block colour jitter [0, 1]
    chisel_depth: 0.4,    // darkening near each block border [0, 1]
    roughness: 0.45,      // FBM face micro-detail amplitude [0, 1]
    color_stone: [0.52, 0.50, 0.47],
    color_mortar: [0.72, 0.70, 0.65],
    normal_strength: 4.5,
};
```

#### Cobblestone

Voronoi cell decomposition producing domed, irregularly shaped stones separated
by mud/dirt gaps.

```rust
use bevy_symbios_texture::cobblestone::CobblestoneConfig;

let config = CobblestoneConfig {
    seed: 7,
    scale: 6.0,           // approximate number of stones across the tile [3, 12]
    gap_width: 0.12,      // mud gap threshold as fraction of stone spacing [0.02, 0.25]
    cell_variance: 0.20,  // per-stone colour jitter [0, 1]
    roundness: 1.2,       // dome profile power [0.5, 2.0]; higher = flatter tops
    color_stone: [0.46, 0.43, 0.40],
    color_mud: [0.22, 0.18, 0.14],
    normal_strength: 5.0,
};
```

#### Marble

Domain-warped FBM noise passed through a sinusoidal vein function for polished
marble or granite with thin dark veins on a light background.

```rust
use bevy_symbios_texture::marble::MarbleConfig;

let config = MarbleConfig {
    seed: 55,
    scale: 3.0,            // overall pattern scale [1, 8]
    octaves: 5,            // FBM octaves for the base layer [3, 8]
    warp_strength: 0.6,    // how much the veins meander [0, 1.5]
    vein_frequency: 3.0,   // period of sin() on warped FBM [1, 8]
    vein_sharpness: 2.0,   // exponent narrowing the veins [0.5, 6]
    roughness: 0.08,       // surface roughness [0, 0.3]; low for polished marble
    color_base: [0.92, 0.90, 0.87],
    color_vein: [0.42, 0.38, 0.34],
    normal_strength: 1.5,
};
```

#### Thatch

Dense fibrous roofing material with anisotropic straw fibres, lateral
domain-warp wiggle, and layered bundle overlap shadows.

```rust
use bevy_symbios_texture::thatch::ThatchConfig;

let config = ThatchConfig {
    seed: 19,
    density: 12.0,         // fibre frequency along U [4, 24]
    anisotropy: 8.0,       // V frequency = density / anisotropy [4, 16]
    warp_strength: 0.15,   // lateral domain-warp wiggle [0, 0.5]
    layer_count: 8.0,      // straw-bundle overlap layers along V [4, 16]
    layer_shadow: 0.55,    // shadow depth at bundle bottom [0, 1]
    color_straw: [0.62, 0.54, 0.28],
    color_shadow: [0.22, 0.17, 0.09],
    normal_strength: 3.5,
};
```

#### Corrugated

Corrugated metal sheets with sine-wave ridges and valley-concentrated rust
weathering.

```rust
use bevy_symbios_texture::corrugated::CorrugatedConfig;

let config = CorrugatedConfig {
    seed: 31,
    ridges: 8.0,           // number of corrugation ridges across U [3, 20]
    ridge_depth: 1.0,      // ridge profile amplitude [0.5, 2.0]
    roughness: 0.35,       // base surface roughness [0, 1]
    rust_level: 0.25,      // rust accumulation in valleys [0, 1]
    metallic: 0.85,        // metallic value [0, 1]
    color_metal: [0.72, 0.74, 0.76],
    color_rust: [0.55, 0.30, 0.12],
    normal_strength: 4.0,
};
```

#### Asphalt

Three-band toroidal FBM (macro staining, micro roughness, aggregate flecks) for
tarmac / asphalt with exposed stone chips.

```rust
use bevy_symbios_texture::asphalt::AsphaltConfig;

let config = AsphaltConfig {
    seed: 88,
    scale: 4.0,              // base noise scale [2, 12]
    aggregate_density: 0.22, // exposed stone chip density [0.05, 0.4]
    aggregate_scale: 16.0,   // fleck noise frequency [8, 32]
    roughness: 0.90,         // overall surface roughness [0.7, 1.0]
    stain_level: 0.25,       // macro stain / oil variation [0, 1]
    color_base: [0.06, 0.06, 0.07],
    color_aggregate: [0.35, 0.33, 0.30],
    normal_strength: 2.5,
};
```

#### Wainscoting

Wood-panel wainscoting with recessed panel faces, rail/stile framing, and
anisotropic grain FBM with domain warp.

```rust
use bevy_symbios_texture::wainscoting::WainscotingConfig;

let config = WainscotingConfig {
    seed: 37,
    panels_x: 1,           // horizontal panel divisions [1, 4]
    panels_y: 2,           // vertical panel divisions [1, 4]
    frame_width: 0.20,     // rail/stile width as fraction of cell [0.05, 0.35]
    panel_inset: 0.06,     // panel recess depth [0, 0.15]
    grain_scale: 10.0,     // wood grain spatial frequency [4, 24]
    grain_warp: 0.30,      // grain domain-warp strength [0, 0.8]
    color_wood_light: [0.65, 0.44, 0.20],
    color_wood_dark: [0.28, 0.16, 0.07],
    normal_strength: 4.0,
};
```

#### Encaustic

Decorative ceramic tiles with glazed surfaces in configurable geometric patterns
(checkerboard, octagon, diamond).

```rust
use bevy_symbios_texture::encaustic::{EncausticConfig, EncausticPattern};

let config = EncausticConfig {
    seed: 47,
    scale: 5.0,            // tile cells across the texture [2, 10]
    pattern: EncausticPattern::Octagon, // or Checkerboard, Diamond
    grout_width: 0.06,     // grout line width as fraction of cell [0.02, 0.15]
    glaze_roughness: 0.04, // glaze surface waviness [0, 0.1]
    color_a: [0.72, 0.38, 0.22],  // primary tile colour
    color_b: [0.22, 0.35, 0.65],  // secondary tile colour
    color_grout: [0.82, 0.80, 0.75],
    normal_strength: 3.0,
};
```

### Alpha-masked cards

Card generators produce an RGBA8 texture where `albedo.alpha` encodes the
silhouette (`0` = fully transparent, `255` = fully opaque).  Upload with
[`map_to_images_card`] so the sampler does not tile and the alpha silhouette
does not bleed at edges.

#### Leaf

A discrete leaf silhouette with procedural venation: midrib, secondary veins,
a Perlin venule (tertiary vein) network, Worley capillaries, and optional
lobed margins.

```rust
use bevy_symbios_texture::leaf::LeafConfig;

let config = LeafConfig {
    seed: 0,
    color_base: [0.12, 0.19, 0.11],  // interior colour, linear RGB
    color_edge: [0.35, 0.28, 0.05],  // edge / autumn tinge
    serration_strength: 0.12,         // tooth depth [0, ~0.35]
    vein_angle: 2.5,                  // secondary vein acuteness
    micro_detail: 0.3,                // Worley capillary blend weight
    normal_strength: 1.0,
    lobe_count: 4.0,                  // 0 = smooth; >0 = lobed margins
    lobe_depth: 0.23,
    lobe_sharpness: 1.0,
    petiole_length: 0.12,             // fraction of V reserved for the stalk
    petiole_width: 0.022,
    midrib_width: 0.12,
    vein_count: 6.0,
    venule_strength: 0.50,
};
```

`LeafSampler` can also be used directly for per-pixel evaluation without
going through the full generator (e.g., inside a twig compositor):

```rust
use bevy_symbios_texture::leaf::{LeafConfig, LeafSampler};

let sampler = LeafSampler::new(LeafConfig::default());
if let Some(sample) = sampler.sample(0.5, 0.4) {
    // sample.height, sample.color, sample.roughness
}
```

#### Twig

A composite foliage card: a tapered, organically curved stem carrying multiple
leaf cards.  Supports two phyllotaxis modes:

* **Monopodial** (`sympodial: false`) — opposite leaf pairs on a straight axis
  with a terminal leaf at the apex.
* **Sympodial** (`sympodial: true`) — alternate leaves on a zigzag axis,
  with a terminal leaf at the apex.

```rust
use std::f64::consts::FRAC_PI_2;
use bevy_symbios_texture::twig::TwigConfig;
use bevy_symbios_texture::leaf::LeafConfig;

let config = TwigConfig {
    leaf: LeafConfig::default(),
    stem_color: [0.18, 0.08, 0.06],
    stem_half_width: 0.021,
    leaf_pairs: 4,
    leaf_angle: FRAC_PI_2 - 0.35,  // ~70° from stem axis
    leaf_scale: 0.38,
    stem_curve: 0.015,
    sympodial: true,
};
```

#### Window

An SDF-based window card with configurable frame, mullions/muntins, and
per-pane glass.  The alpha channel is transparent outside the frame and
semi-transparent over glass panes.

```rust
use bevy_symbios_texture::window::WindowConfig;

let config = WindowConfig {
    seed: 42,
    frame_width: 0.08,       // frame width as a fraction of the card [0, 0.4]
    panes_x: 2,              // number of panes horizontally
    panes_y: 3,              // number of panes vertically
    mullion_thickness: 0.025, // mullion/muntin thickness as a fraction of the glass area
    corner_radius: 0.02,     // inner glass-opening corner rounding [0, 0.4]
    glass_opacity: 0.30,     // glass alpha [0 = clear, 1 = frosted/opaque]
    grime_level: 0.15,       // grime/dirt noise on glass [0, 1]
    color_frame: [0.85, 0.82, 0.78],
    normal_strength: 3.0,
};
```

#### Stained Glass

Voronoi-based stained-glass panel with lead came borders and semi-transparent
coloured glass panes.  Glass alpha is 180 (semi-transparent); lead is 255
(fully opaque).

```rust
use bevy_symbios_texture::stained_glass::StainedGlassConfig;

let config = StainedGlassConfig {
    seed: 63,
    cell_count: 12,        // approximate number of glass cells [5, 25]
    lead_width: 0.05,      // lead came width as fraction of cell spacing [0.02, 0.12]
    saturation: 0.85,      // glass colour saturation [0.5, 1.0]
    glass_roughness: 0.06, // glass surface waviness [0, 0.15]
    grime_level: 0.12,     // grime/dirt accumulation on glass [0, 0.5]
    normal_strength: 2.5,
};
```

#### Iron Grille

Rectangular or round-bar iron grille / portcullis with configurable bar count
and joint-concentrated rust weathering.

```rust
use bevy_symbios_texture::iron_grille::IronGrilleConfig;

let config = IronGrilleConfig {
    seed: 71,
    bars_x: 4,             // vertical bars [2, 10]
    bars_y: 6,             // horizontal bars [2, 10]
    bar_width: 0.04,       // bar half-width as fraction of card [0.02, 0.20]
    round_bars: true,      // true = cylindrical cross-section, false = rectangular
    rust_level: 0.30,      // rust at joints [0, 1]
    color_iron: [0.14, 0.13, 0.13],
    color_rust: [0.42, 0.22, 0.08],
    normal_strength: 3.5,
};
```

## Evolutionary parameter search (genetics)

All config types implement `symbios_genetics::Genotype`, making them
compatible with the evolutionary algorithms in the `symbios-genetics` crate
(`SimpleGA`, `Nsga2`, `MapElites`).

Each field is independently perturbed during mutation and drawn uniformly from
one of two parents during crossover:

```rust
use symbios_genetics::Genotype;
use bevy_symbios_texture::bark::BarkConfig;
use rand::SeedableRng;

let mut config = BarkConfig::default();
let mut rng = rand::rngs::StdRng::seed_from_u64(42);
config.mutate(&mut rng, 0.3);  // perturb each field with 30 % probability

let parent_b = BarkConfig { seed: 99, ..BarkConfig::default() };
let child = config.crossover(&parent_b, &mut rng);
```

The `texture_viewer` example uses this to mutate any displayed material when
you click **Mutate**.

## Architecture

```
TextureGenerator (trait)
    │
    │  Tileable surface textures
    ├── BarkGenerator       ─── ToroidalNoise (domain-warped FBM + Worley plates)
    ├── RockGenerator       ─── ToroidalNoise (RidgedMulti)
    ├── GroundGenerator     ─── ToroidalNoise × 2 (dual-scale FBM)
    ├── BrickGenerator      ─── ToroidalNoise FBM + rounded-box SDF grid
    ├── PlankGenerator      ─── ToroidalNoise FBM + Worley knots (anisotropic)
    ├── ConcreteGenerator   ─── ToroidalNoise FBM + cosine formwork + pit FBM
    ├── MetalGenerator      ─── ToroidalNoise FBM (brushed/standing-seam) + rust FBM
    ├── ShingleGenerator    ─── ToroidalNoise FBM + sawtooth overlap ramp
    ├── PaversGenerator     ─── ToroidalNoise FBM + square/hex SDF grid
    ├── StuccoGenerator     ─── ToroidalNoise FBM (high-frequency, matte)
    ├── AshlarGenerator     ─── ToroidalNoise FBM + irregular SDF grid + chisel edge
    ├── CobblestoneGenerator─── toroidal Voronoi (domed F1, mud gap at F2−F1)
    ├── MarbleGenerator     ─── ToroidalNoise FBM (domain-warped sinusoidal veins)
    ├── ThatchGenerator     ─── ToroidalNoise FBM (anisotropic fibre + sawtooth layers)
    ├── CorrugatedGenerator ─── sine-wave ridge profile + rust FBM
    ├── AsphaltGenerator    ─── ToroidalNoise FBM × 3 (macro/micro/aggregate)
    ├── WainscotingGenerator─── ToroidalNoise grain FBM + panel margin SDF
    ├── EncausticGenerator  ─── ToroidalNoise glaze FBM + geometric pattern SDF
    │
    │  Alpha-masked cards
    ├── LeafGenerator       ─── LeafSampler (silhouette + venation)
    ├── TwigGenerator       ─── LeafSampler × N (composite stem + leaves)
    ├── WindowGenerator     ─── rounded-box SDF frame/mullions + FBM grime
    ├── StainedGlassGenerator── toroidal Voronoi + lead came SDF + grime FBM
    └── IronGrilleGenerator ─── bar SDF grid + joint rust FBM
                                │
                        height_to_normal() → normal map
                        linear_to_srgb()   → albedo encoding
                                │
                         TextureMap { albedo, normal, roughness }
                                │
                map_to_images()      → GeneratedHandles (repeat sampler)
                map_to_images_card() → GeneratedHandles (clamp sampler)
                                │
                        full mipmap chain (type-correct averaging)
```

**Seamless tiling** is provided by [`ToroidalNoise`], which maps 2-D UV
coordinates onto a 4-D torus so that noise wraps perfectly at every edge:

```
nx = cos(2π·u) · frequency
ny = sin(2π·u) · frequency
nz = cos(2π·v) · frequency
nw = sin(2π·v) · frequency
```

Because `cos(0) = cos(2π)` and `sin(0) = sin(2π)`, `u=0` and `u=1` always
resolve to the same 4-D point, guaranteeing zero-seam tiling.

**Normal maps** are derived from the height field via central-difference
gradients.  For the tileable surface textures the neighbours wrap toroidally,
so the normals are also seamless.  For card textures (Leaf, Twig, Window, Stained Glass, Iron Grille) the
boundary uses clamp-to-edge so normals do not bleed across the transparent
silhouette border.

**Colour encoding** uses a 4096-entry sRGB lookup table (built once via
`OnceLock`) to avoid repeated `f32::powf` calls during rasterisation.
A 256-entry table would be insufficient because the sRGB curve is steep
near zero; 4096 bins keep the maximum quantisation error well below one
count in u8.

**Mipmap generation** is performed by `map_to_images` / `map_to_images_card`
using a 2×2 box filter with type-correct averaging: sRGB values are decoded to
linear light before averaging and re-encoded afterward (avoiding dark mipmaps),
normal-map XYZ vectors are averaged and renormalized (avoiding zero-length
normals in PBR shaders), and ORM values are averaged directly in linear space.
16× anisotropic filtering is enabled on all samplers.

## Running the viewer example

```sh
cargo run --example texture_viewer --features egui
```

Displays an interactive material viewer with three columns: **albedo** (left),
**normal map** (centre), and a **spinning 3-D PBR cube** (right) with the
generated material applied.  An egui panel on the left lets you cycle through
all 23 generators with **< Prev** / **Next >**, trigger a random **Mutate**
(rate = 0.3), and edit every parameter live.

## License

MIT — see [LICENSE](LICENSE).
