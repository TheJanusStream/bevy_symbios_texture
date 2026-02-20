# bevy_symbios_texture

Procedural, tileable texture generation for [Bevy](https://bevyengine.org/).

Generates albedo, normal, and roughness (ORM) maps entirely on the CPU — no
asset files required. All textures are seamlessly tileable by construction
via toroidal 4-D noise mapping.

## Bevy compatibility

| bevy_symbios_texture | Bevy |
|----------------------|------|
| 0.1                  | 0.18 |

## Installation

```toml
[dependencies]
bevy_symbios_texture = "0.1"
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

All generators produce three maps:

| Map       | Format          | Contents                              |
|-----------|-----------------|---------------------------------------|
| `albedo`  | `Rgba8UnormSrgb`| Base colour                           |
| `normal`  | `Rgba8Unorm`    | Tangent-space normal (R=X, G=Y, B=Z)  |
| `roughness`| `Rgba8Unorm`   | ORM: R=Occlusion, G=Roughness, B=Metallic |

### Bark

Domain-warped FBM noise producing fibrous, streaked bark grain.

```rust
use bevy_symbios_texture::bark::BarkConfig;

let config = BarkConfig {
    seed: 42,
    scale: 4.0,          // spatial frequency of the pattern
    octaves: 6,          // FBM detail levels
    warp_u: 0.15,        // lateral warp strength
    warp_v: 0.55,        // vertical (fibre) warp strength
    color_light: [0.45, 0.28, 0.14],  // ridge colour, linear RGB
    color_dark:  [0.18, 0.10, 0.05],  // groove colour, linear RGB
    normal_strength: 3.0,
};
```

### Rock

Ridged multifractal noise for cracked, faceted stone.

```rust
use bevy_symbios_texture::rock::RockConfig;

let config = RockConfig {
    seed: 7,
    scale: 3.0,
    octaves: 8,
    attenuation: 2.0,    // ridge sharpness (higher = sharper)
    color_light: [0.55, 0.52, 0.48],
    color_dark:  [0.22, 0.20, 0.18],
    normal_strength: 4.0,
};
```

### Ground

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

## Architecture

```
TextureGenerator (trait)
    │
    ├── BarkGenerator   ─── ToroidalNoise (domain-warped FBM)
    ├── RockGenerator   ─── ToroidalNoise (RidgedMulti)
    └── GroundGenerator ─── ToroidalNoise × 2 (dual-scale FBM)
                                │
                        height_to_normal() → normal map
                        linear_to_srgb()   → albedo encoding
                                │
                         TextureMap { albedo, normal, roughness }
                                │
                        map_to_images() → GeneratedHandles { albedo, normal, roughness }
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

**Colour encoding** uses a 256-entry sRGB lookup table (built once via
`OnceLock`) to avoid repeated `f32::powf` calls during rasterisation.

## Running the viewer example

```sh
cargo run --example texture_viewer
```

Displays bark, rock, and ground albedo maps side-by-side in a window.

## License

MIT — see [LICENSE](LICENSE).
