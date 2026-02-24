# bevy_symbios_texture

Procedural, tileable texture generation for [Bevy](https://bevyengine.org/).

Generates albedo, normal, and roughness (ORM) maps entirely on the CPU — no
asset files required. All surface textures are seamlessly tileable by
construction via toroidal 4-D noise mapping. Foliage cards (leaf, twig) produce
alpha-masked silhouettes and do not tile.

## Bevy compatibility

| bevy_symbios_texture | Bevy |
|----------------------|------|
| 0.2                  | 0.18 |

## Installation

```toml
[dependencies]
bevy_symbios_texture = "0.2"
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

Offloads pixel math to Bevy's `AsyncComputeTaskPool` so the main thread is
never stalled.

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

## Generators

### Surface textures (tileable)

All three tileable generators produce three seamlessly-repeating maps:

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

### Foliage cards (alpha-masked)

Foliage generators produce an RGBA8 texture where `albedo.alpha` encodes the
silhouette (`0` = outside, `255` = inside).  Upload with [`map_to_images_card`]
so the sampler does not tile and the alpha channel does not bleed at edges.

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
    leaf_angle: FRAC_PI_2 - 0.35,  // ~69° from stem axis
    leaf_scale: 0.38,
    stem_curve: 0.05,
    sympodial: false,
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

The `texture_viewer` example uses this to mutate any displayed panel when you
click it.

## Architecture

```
TextureGenerator (trait)
    │
    ├── BarkGenerator   ─── ToroidalNoise (domain-warped FBM + Worley plates)
    ├── RockGenerator   ─── ToroidalNoise (RidgedMulti)
    ├── GroundGenerator ─── ToroidalNoise × 2 (dual-scale FBM)
    ├── LeafGenerator   ─── LeafSampler (silhouette + venation)
    └── TwigGenerator   ─── LeafSampler × N (composite stem + leaves)
                                │
                        height_to_normal() → normal map
                        linear_to_srgb()   → albedo encoding
                                │
                         TextureMap { albedo, normal, roughness }
                                │
                map_to_images()      → GeneratedHandles (repeat sampler)
                map_to_images_card() → GeneratedHandles (clamp sampler)
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
gradients with wrap-around neighbours, so the normals are also seamless.

**Colour encoding** uses a 4096-entry sRGB lookup table (built once via
`OnceLock`) to avoid repeated `f32::powf` calls during rasterisation.
A 256-entry table would be insufficient because the sRGB curve is steep
near zero; 4096 bins keep the maximum quantisation error well below one
count in u8.

## Running the viewer example

```sh
cargo run --example texture_viewer
```

Displays all five generators (Bark, Rock, Ground, Leaf, Twig) in two rows:
albedo maps on top, normal maps below.  **Left-click any albedo panel** to
apply a random mutation (rate = 0.3) and regenerate that texture.

## License

MIT — see [LICENSE](LICENSE).
