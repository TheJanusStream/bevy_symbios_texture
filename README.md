# bevy_symbios_texture

Procedural, tileable texture generation for [Bevy](https://bevyengine.org/).

Generates albedo, normal, and roughness (ORM) maps entirely on the CPU — no
asset files required. All surface textures are seamlessly tileable by
construction via toroidal 4-D noise mapping. Alpha-masked card textures (leaf,
twig, window) produce per-pixel transparency and do not tile.

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
    mortar_size: 0.10,   // mortar gap as a fraction of cell height [0, 0.4]
    bevel: 0.5,          // corner bevel radius as a fraction of mortar_size [0, 1]
    cell_variance: 0.12, // per-brick colour jitter [0, 1]
    roughness: 0.25,     // surface pitting noise intensity [0, 1]
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
    color_base: [0.12, 0.35, 0.08],  // interior colour, linear RGB
    color_edge: [0.35, 0.28, 0.05],  // edge / autumn tinge
    serration_strength: 0.12,         // tooth depth [0, ~0.35]
    vein_angle: 2.5,                  // secondary vein acuteness
    micro_detail: 0.3,                // Worley capillary blend weight
    normal_strength: 3.0,
    lobe_count: 0.0,                  // 0 = smooth; >0 = lobed margins
    lobe_depth: 0.35,
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
    stem_color: [0.25, 0.16, 0.07],
    stem_half_width: 0.015,
    leaf_pairs: 4,
    leaf_angle: FRAC_PI_2 - 0.35,  // ~70° from stem axis
    leaf_scale: 0.38,
    stem_curve: 0.05,
    sympodial: false,
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
    ├── BarkGenerator    ─── ToroidalNoise (domain-warped FBM + Worley plates)
    ├── RockGenerator    ─── ToroidalNoise (RidgedMulti)
    ├── GroundGenerator  ─── ToroidalNoise × 2 (dual-scale FBM)
    ├── BrickGenerator   ─── ToroidalNoise FBM + rounded-box SDF grid
    ├── PlankGenerator   ─── ToroidalNoise FBM + Worley knots (anisotropic)
    ├── ConcreteGenerator─── ToroidalNoise FBM + cosine formwork + pit FBM
    ├── MetalGenerator   ─── ToroidalNoise FBM (brushed/standing-seam) + rust FBM
    ├── ShingleGenerator ─── ToroidalNoise FBM + sawtooth overlap ramp
    ├── PaversGenerator  ─── ToroidalNoise FBM + square/hex SDF grid
    ├── StuccoGenerator  ─── ToroidalNoise FBM (high-frequency, matte)
    ├── LeafGenerator    ─── LeafSampler (silhouette + venation)
    ├── TwigGenerator    ─── LeafSampler × N (composite stem + leaves)
    └── WindowGenerator  ─── rounded-box SDF frame/mullions + FBM grime
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
so the normals are also seamless.  For card textures (Leaf, Twig, Window) the
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
all 13 generators with **< Prev** / **Next >**, trigger a random **Mutate**
(rate = 0.3), and edit every parameter live.

## License

MIT — see [LICENSE](LICENSE).
