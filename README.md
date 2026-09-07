# bevy_symbios_texture

Procedural, tileable texture generation for [Bevy](https://bevyengine.org/).

Generates albedo, normal, roughness (ORM), and optional emissive maps
entirely on the CPU — no asset files required. Generation is multi-core
(rows are produced in parallel) and seamlessly tileable for all surface
textures via toroidal 4-D noise mapping. Alpha-masked card textures (leaf,
twig, window, stained glass, iron grille, chain-link, log-end) produce
per-pixel transparency and do not tile. Atlas-capable card generators —
particle sprites (soft disc, spark, snowflake, puff, ring, petal, shard,
flame, flower) and foliage billboards (leaf sprite, grass tuft, frond,
reed, needle, broadleaf) — bake alpha-silhouette sheets where every atlas
cell is a per-cell-seeded variant of the same config.

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

Offloads pixel math to a private, bounded rayon thread pool (default 4
concurrent tasks; configurable via `AsyncTextureConfig::pool_threads`) so the
main thread is never stalled. On WASM, falls back to Bevy's
`AsyncComputeTaskPool`.

Within each texture, rows are generated in parallel: async tasks work-steal
across the private pool (so `pool_threads` remains the CPU cap), while direct
synchronous `generate()` calls parallelise on the caller's rayon pool —
usually the global one, using every core.  Output is byte-identical to
serial generation.

If `rayon::ThreadPoolBuilder::build()` fails at first init (out-of-memory, OS
thread limit, sandboxed environments) the library logs a warning and falls
back to running each generator inline on the calling thread.  Texture
generation still works — slower and blocking the spawning thread — instead of
panicking.

```rust
use bevy::prelude::*;
use bevy_symbios_texture::{
    AsyncTextureConfig, SymbiosTexturePlugin,
    async_gen::{PendingTexture, TextureReady},
    bark::BarkConfig,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // Default pool: 4 concurrent tasks.
        .add_plugins(SymbiosTexturePlugin::default())
        // Or explicitly: 0 = auto (available_parallelism() / 2),
        // any positive value = exact thread count.
        // .add_plugins(SymbiosTexturePlugin {
        //     config: AsyncTextureConfig { pool_threads: 0 },
        // })
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

### One-shot procedural materials

`build_procedural_material_async` collapses the StandardMaterial-allocate +
PendingTexture-spawn + post-completion-patch dance into a single call.  Define
a `MaterialSettings` (PBR fields + a `TextureConfig` enum that selects the
generator), call the helper, and use the returned handle immediately:

```rust
use bevy_symbios_texture::{
    MaterialSettings, TextureConfig, build_procedural_material_async,
    brick::BrickConfig,
};

fn spawn_brick_wall(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let settings = MaterialSettings {
        base_color: [0.65, 0.30, 0.20],
        roughness: 0.85,
        texture: TextureConfig::Brick(BrickConfig::default()),
        ..MaterialSettings::default()
    };
    let material = build_procedural_material_async(
        &mut commands, &mut materials, &mut images, /*cache=*/ None,
        &settings, 512, 512,
    );
    commands.spawn((Mesh3d(meshes.add(Cuboid::default())), MeshMaterial3d(material)));
}
```

#### Baking somewhere else

The helper fuses *construction* with *dispatch* — it needs `&mut Commands`
because it spawns the generation task. A consumer whose textures are baked
somewhere this crate's rayon pool cannot reach (a Web Worker, a job queue,
another process) has to fork the dispatch, and the four pieces it needs are
public so it does not also have to fork the appearance:

| Item | Does |
|------|------|
| `MaterialSettings::standard_material()` | Builds the `StandardMaterial` these settings describe, texture slots empty |
| `MaterialSettings::cache_key(w, h)` | The `TextureCacheKey` a bake of them belongs under; `None` for `TextureConfig::None` |
| `store_generated_texture_map(map, is_card, key, cache, images)` | Persists raw pixels, uploads the map, writes the cache; returns the handles |
| `apply_generated_handles(&mut material, &handles)` | Writes all four slots, moving the emissive *factor* with the glow map |

`build_procedural_material_async` and `patch_procedural_material_textures` are
written in terms of these, so a consumer driving them by hand is running the
same bodies rather than a copy that drifts.

`uv_transform` is set to a uniform scale of `uv_scale`; a caller with its own UV
offset/rotation convention overwrites that one field after the call.

### Texture cache

To avoid regenerating the same `(generator, config, size)` tuple across
spawns, insert a `TextureCache` resource:

```rust
use bevy_symbios_texture::{DEFAULT_MEMORY_CACHE_ENTRIES, TextureCache};

app.insert_resource(TextureCache::memory(DEFAULT_MEMORY_CACHE_ENTRIES));
// or, for cross-process persistence:
// app.insert_resource(TextureCache::file("./.texture-cache", manifest_version)?);
```

Cache hits return previously-uploaded `Handle<Image>` clones synchronously
and skip the rayon dispatch entirely.  Cache keys derive from a fingerprint
of the config struct, so any field change automatically invalidates the
prior entry.  `manifest_version` is mixed into every `FileStore` on-disk
key, so bumping it rotates the persisted cache without deleting the
directory — use it when generator internals change without a config-field
change.

The library ships two built-in stores — `MemoryStore` (bounded, FIFO
eviction, default) and `FileStore` (binary blobs on disk) — and exposes the
`TextureCacheStore` trait for custom backends.  `FileStore` persists the raw
pixel blobs as generation completes and re-uploads them (regenerating
mipmaps) on the first hit after a restart, so warm caches survive across
processes.

For diagnostics, `TextureCache::entry_count()` reports how many entries the
backing store currently holds in memory — `Some(len)` for `MemoryStore`,
`None` for `FileStore`, whose entries live on disk.  Because the cache
retains `Handle<Image>` clones, its entry count is often the missing term
when attributing image-asset growth in a running application.

### Animated parameter curves

Time-varying weathering, age, and seasonal change are first-class via the
`AnimatedProceduralMaterial` component plus a small set of
`ParameterCurve` impls (`Linear`, `EaseInOut`, `Stepped`, `ScriptedFn`).
Attach the component to a material entity and the
`tick_animated_procedural_materials` system regenerates the texture
whenever the curve's output changes:

```rust
use bevy_symbios_texture::{
    AnimatedProceduralMaterial, Linear, ParameterCurve, TextureConfig,
    metal::MetalConfig,
};

let rust = Linear { from: 0.0_f64, to: 1.0, duration: 10.0 };
let base = MetalConfig::default();
let animator = AnimatedProceduralMaterial::new(material, 512, 512, move |t| {
    TextureConfig::Metal(MetalConfig {
        rust_level: rust.eval(t),
        ..base.clone()
    })
})
.with_min_regen_interval(0.25); // throttle regeneration to 4 Hz
```

Two thresholds gate regeneration: a wall-clock cooldown
(`min_regen_interval`, default 0.25 s) and fingerprint equality.  Stepped
or plateaued curves cost essentially nothing once the value stops changing.

For sub-second smoothness across the steady-state pixels, drive a
fragment-shader uniform on the material — generator output is RGBA8 and is
the wrong knob for sub-frame interpolation.

## Compute-shader fast path

A wgpu compute-shader port of the hottest generators (FBM-based bark,
brick, marble) is on the roadmap but **not implemented**.  The remaining
work — porting toroidal noise to WGSL with bit-equivalent CPU/GPU output,
dispatch + readback plumbing, a feature flag that swaps in the GPU path
when available, and a benchmark suite — is multi-week and was deliberately
deferred so this release could ship the asynchronous + cached + animated
path on a known-good CPU baseline.

If you need realtime texture editing at 60 FPS today, the alternatives are:

* Rely on the row-parallel CPU path: on a modern many-core desktop the
  heaviest generator (bark) renders a 512² map in ~20 ms, so interactive
  editing at moderate resolutions is already feasible without the GPU port.
* Bake the texture once via the regular CPU path and animate a material
  uniform (rust mask weight, colour blend, etc.) in the fragment shader.
* Use `AnimatedProceduralMaterial` with a coarse `min_regen_interval` and
  accept the staircase update cadence.

## Generators

### Surface textures (tileable)

All tileable generators produce three seamlessly-repeating maps:

| Map        | Format           | Contents                                  |
|------------|------------------|-------------------------------------------|
| `albedo`   | `Rgba8UnormSrgb` | Base colour                               |
| `normal`   | `Rgba8Unorm`     | Tangent-space normal (R=X, G=Y, B=Z)      |
| `roughness`| `Rgba8Unorm`     | ORM: R=Occlusion, G=Roughness, B=Metallic |
| `emissive` | `Rgba8UnormSrgb` | Optional emissive / glow map              |

Upload with `map_to_images` to get repeat-wrapping samplers.  When a
generator produces an emissive map the polling systems assign it to
`StandardMaterial::emissive_texture`; Bevy multiplies it by the material's
emissive colour factor, which the material flow auto-defaults to white when
`MaterialSettings::emission_color` / `emission_strength` are left unset, so
the glow shows out of the box.  Set them only to tint or brighten beyond the
map's encoded values.

#### Bark

Domain-warped FBM noise with an anisotropic Worley plate layer for rhytidome
furrows, producing fibrous, streaked bark grain.

```rust
use bevy_symbios_texture::bark::BarkConfig;

let config = BarkConfig {
    seed: 42,
    scale: 2.0,             // spatial frequency of the pattern
    octaves: 6,             // FBM detail levels (base layer)
    warp_octaves: 3,        // FBM detail for the warp layers (3 is plenty)
    warp_u: 0.15,           // lateral warp strength
    warp_v: 0.55,           // vertical (fibre) warp strength
    color_light: [0.45, 0.28, 0.14],  // ridge colour, linear RGB
    color_dark:  [0.09, 0.05, 0.03],  // groove colour, linear RGB
    normal_strength: 3.0,
    furrow_multiplier: 0.78, // blend weight of the Worley plate layer [0, 1]
    furrow_scale_u: 2.0,     // horizontal cell frequency (higher = narrower plates)
    furrow_scale_v: 0.48,    // vertical cell frequency (lower = longer plates)
    furrow_shape: 2.0,       // plate height power (<1 widens plateau, sharpens cracks)
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

Brushed metal (anisotropic FBM scratches), standing-seam roof panels,
hand-hammered dimples, or diamond tread plate — all with optional rust-patch
weathering.  For `Hammered` and `DiamondPlate`, `scale` sets the dimple /
stud count across the tile.

```rust
use bevy_symbios_texture::metal::{MetalConfig, MetalStyle};

let config = MetalConfig {
    seed: 31,
    style: MetalStyle::Brushed, // or StandingSeam, Hammered, DiamondPlate
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
    warp_octaves: 3,       // FBM octaves for the warp layers [1, 6]
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

#### Sand

Wind-rippled sand: a directional sine ridge field phase-warped by FBM so
crests meander and merge, plus grain micro-relief and thresholded bright
flecks (exposed sparkling grains read as local smooth spots in the ORM).

```rust
use bevy_symbios_texture::sand::SandConfig;

let config = SandConfig {
    seed: 91,
    ripple_count: 10.0,    // crests across the tile [4, 24]
    ripple_warp: 0.6,      // crest meander strength [0, 1.5]
    grain_density: 0.12,   // bright-fleck density [0, 0.5]
    grain_scale: 24.0,     // grain noise frequency [8, 48]
    color_crest: [0.86, 0.74, 0.52],
    color_trough: [0.62, 0.50, 0.34],
    normal_strength: 2.5,
};
```

#### Snow

Wind-drifted snow: soft FBM relief with a cool shadow tint in the troughs
and thresholded sparkle flecks — crystals that brighten the albedo and drop
ORM roughness to near zero for specular glints.

```rust
use bevy_symbios_texture::snow::SnowConfig;

let config = SnowConfig {
    seed: 73,
    drift_scale: 2.5,      // drift relief scale [1, 6]
    drift_octaves: 4,      // FBM octaves [2, 6]
    sparkle_density: 0.08, // glinting-crystal density [0, 0.5]
    crust_roughness: 0.85, // base crust roughness [0.5, 1]
    color_snow: [0.93, 0.95, 0.99],
    color_shadow: [0.62, 0.70, 0.86],  // cool trough tint
    normal_strength: 1.8,
};
```

#### Ice

Polished lake ice: a near-mirror pale-blue base crossed by thin recessed
crack veins (sinusoidal bands over FBM contours), with frost patches that
whiten the colour and raise roughness toward matte.

```rust
use bevy_symbios_texture::ice::IceConfig;

let config = IceConfig {
    seed: 117,
    scale: 3.0,            // pattern scale [1, 8]
    crack_density: 4.0,    // vein frequency [1, 8]
    vein_sharpness: 7.0,   // crack narrowing exponent [2, 12]
    frost_level: 0.25,     // matte frost coverage [0, 1]
    color_ice: [0.72, 0.84, 0.94],
    color_crack: [0.30, 0.44, 0.62],
    normal_strength: 1.5,
};
```

#### Moss

A dense velvety moss carpet: broad FBM cushion hummocks overlaid with a
fine filament-tip stipple, plus scattered patches bleached toward a dry
straw tone.  Shaded crevices read deep green while the cushion crowns catch
a bright yellow-green.

```rust
use bevy_symbios_texture::moss::MossConfig;

let config = MossConfig {
    seed: 21,
    cushion_scale: 5.0,     // hummock scale (lower = broader mounds)
    cushion_octaves: 4,
    filament_scale: 34.0,   // filament-tip stipple frequency (higher = finer)
    filament_octaves: 3,
    filament_weight: 0.45,  // stipple blend weight [0, 1]
    color_deep: [0.03, 0.09, 0.03],  // shaded crevice colour
    color_tip: [0.26, 0.44, 0.10],   // cushion-crown colour
    color_dry: [0.38, 0.34, 0.14],   // bleached straw tone
    dry_patches: 0.25,      // share of the carpet bleached dry [0, 1]
    dry_scale: 2.5,         // dry-patch scale (lower = larger patches)
    cushion_depth: 0.6,     // mound vs stipple share of the height field [0, 1]
    normal_strength: 2.4,
};
```

#### Lichen

Crustose lichen colonies over bare rock: a thresholded FBM patch field with
pale growing margins, a granular interior, and two species tints (sage
grey-green and rusty orange) selected by a slower field, so a rock face
shows several colonies rather than one flat wash.  Uncolonised texels keep
the rock substrate colour.

```rust
use bevy_symbios_texture::lichen::LichenConfig;

let config = LichenConfig {
    seed: 7,
    patch_scale: 3.0,       // colony field scale (lower = broader colonies)
    patch_octaves: 2,       // more octaves = more raggedly-lobed outlines
    coverage: 0.45,         // colonised fraction [0, 1]; 0 = bare rock
    rim_width: 0.06,        // pale growing-margin width [0, 0.4]; 0 = none
    species_scale: 1.8,     // species mix (lower = larger single-species areas)
    color_rock: [0.13, 0.13, 0.12],      // bare substrate
    color_lichen_a: [0.14, 0.17, 0.10],  // sage grey-green species
    color_lichen_b: [0.26, 0.13, 0.04],  // rusty orange species
    color_rim: [0.38, 0.40, 0.32],
    grain_scale: 40.0,      // interior grain frequency (higher = finer)
    grain_strength: 0.18,   // interior grain strength [0, 1]
    relief: 0.5,            // how proud the crust stands of the rock [0, 1]
    normal_strength: 1.8,
};
```

#### Cactus Skin

A tileable succulent hide: vertical accordion ribs, a periodic lattice of
felted areoles seated on the rib crests, and pale spines radiating from
each areole.  Ribs and areoles are integer-periodic, so the tile wraps
seamlessly around an L-system cactus stem.

```rust
use bevy_symbios_texture::cactus::CactusSkinConfig;

let config = CactusSkinConfig {
    seed: 0,
    rib_count: 8,           // vertical ribs around the tile [3, 40]
    areole_rows: 9,         // areole rows up the tile [2, 40]
    rib_depth: 0.85,        // accordion-pleat relief [0, 1]
    rib_sharpness: 0.85,    // crest sharpness [0.3, 3]
    color_skin: [0.22, 0.42, 0.27],    // waxy ridge colour
    color_valley: [0.07, 0.17, 0.11],  // shaded pleat colour
    color_areole: [0.55, 0.50, 0.40],  // felt cushions
    color_spine: [0.86, 0.82, 0.66],
    areole_size: 0.022,     // felt radius in UV units [0.005, 0.08]
    spine_reach: 3.2,       // spine length as multiple of areole_size [1, 6]
    spine_count: 8,         // spines per areole [0, 24]
    waxiness: 0.55,         // gloss (higher = lower roughness) [0, 1]
    normal_strength: 1.4,
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

#### Fabric

Plain-weave cloth: perpendicular warp/weft threads as half-cylinder
profiles, over/under crossing relief, fibre fuzz, and yarn-mottle tinting.
Match the two colours for solid cloth or contrast them for two-tone weaves.

```rust
use bevy_symbios_texture::fabric::FabricConfig;

let config = FabricConfig {
    seed: 29,
    thread_count: 24.0,    // threads per tile edge [8, 64]
    thread_width: 0.85,    // thread width as cell fraction [0.3, 0.98]
    weave_contrast: 0.6,   // over/under relief depth [0, 1]
    fuzz: 0.35,            // fibre fuzz strength [0, 1]
    color_warp: [0.55, 0.36, 0.24],  // vertical threads
    color_weft: [0.62, 0.44, 0.30],  // horizontal threads
    normal_strength: 3.0,
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

#### Cracked earth

Dried mud plates with curled rims, separated by cracks of *constant* width —
the cheaper `F2 − F1` mask widens with the cell, giving large plates canyons
and small ones hairlines.

```rust
use bevy_symbios_texture::cracked_earth::CrackedEarthConfig;

let config = CrackedEarthConfig {
    seed: 11,
    scale: 7.0,            // plates across the tile [2, 20]
    jitter: 0.85,          // plate irregularity [0, 1]
    crack_width: 0.010,    // crack width in UV units (fraction of the tile)
    crack_depth: 0.55,     // how far cracks cut into the height field
    curl: 0.22,            // how far plate rims lift as they dry
    curl_reach: 0.035,     // how far back from a crack the curl reaches (UV)
    plate_variance: 0.10,  // per-plate tint spread [0, 1]
    grain_scale: 26.0,
    grain_strength: 0.12,
    color_plate: [0.44, 0.33, 0.22],
    color_crack: [0.13, 0.09, 0.06],
    normal_strength: 3.0,
};
```

#### Gravel

Packed graded aggregate over dust.  Each stone is sized against the distance
to its neighbour rather than a fixed radius, so it fills its own cell; the
metric decides whether stones read as water-rounded shingle or crushed rock.

```rust
use bevy_symbios_texture::gravel::GravelConfig;
use bevy_symbios_texture::noise::CellMetric;

let config = GravelConfig {
    seed: 23,
    scale: 20.0,           // stones across the tile (≈20 roadbase, ≈8 ballast)
    metric: CellMetric::Euclidean, // or Manhattan / Chebyshev for angular stone
    jitter: 0.9,
    roundness: 1.6,        // dome profile exponent
    size_variance: 0.45,   // grading spread [0, 1]
    cell_variance: 0.13,   // per-stone tint spread [0, 1]
    fines_level: 0.55,     // dust filling the gaps [0, 1]
    grain_scale: 60.0,
    color_stone: [0.40, 0.38, 0.35],
    color_dark: [0.17, 0.16, 0.15],
    color_fines: [0.26, 0.24, 0.21],
    normal_strength: 2.5,
};
```

#### Forest floor

Fallen leaf litter over humus.  Leaves are stamped as discrete objects and
layered with a per-leaf depth so they interleave — noise-based "litter" has no
edges you can follow around a single leaf.

```rust
use bevy_symbios_texture::forest_floor::ForestFloorConfig;

let config = ForestFloorConfig {
    seed: 31,
    litter_scale: 7.0,     // leaves across the coarsest layer
    layers: 3,             // stacked litter layers [1, 4]
    coverage: 0.85,        // fraction of cells carrying a leaf [0, 1]
    leaf_length: 1.15,     // multiple of the lattice cell
    leaf_width: 0.5,       // multiple of the leaf length
    leaf_thickness: 0.35,
    midrib: 0.22,          // darkening of the leaf spine [0, 1]
    humus_scale: 14.0,
    color_humus: [0.09, 0.07, 0.05],
    color_leaf: [0.46, 0.31, 0.12],
    color_leaf_old: [0.22, 0.16, 0.09],
    normal_strength: 2.2,
};
```

#### Enamel

A smooth fired glaze, optionally crazed.  Its character is the *absence* of
directional structure, which is what a brushed-metal finish cannot fake.

```rust
use bevy_symbios_texture::enamel::EnamelConfig;

let config = EnamelConfig {
    seed: 17,
    color: [0.62, 0.20, 0.16],       // fired glaze
    color_body: [0.80, 0.78, 0.74],  // unglazed body, seen through the craze
    gloss_roughness: 0.18,           // above ~0.4 reads as matt paint
    metallic: 0.0,
    crackle: 0.0,                    // 0 leaves the coat perfectly clear
    crackle_scale: 26.0,
    crackle_width: 0.0025,           // UV units, so the web stays hairline
    orange_peel: 0.11,
    orange_peel_scale: 34.0,
    ..Default::default()             // `weathering`, `normal_strength`
};
```

#### Obsidian

Near-black polished glass carrying warped flow banding.  Keep `band_warp` well
under half a turn: past that the bend exceeds half a band period and the bands
fold back through one another instead of flowing.

```rust
use bevy_symbios_texture::obsidian::ObsidianConfig;

let config = ObsidianConfig {
    seed: 29,
    color: [0.035, 0.032, 0.045],
    color_sheen: [0.16, 0.15, 0.22],
    band_cycles_u: 5.0,    // whole cycles; the field only tiles at integers
    band_cycles_v: 2.0,
    band_warp: 0.26,       // turns of phase — keep well under 0.5
    band_warp_scale: 1.6,
    band_sharpness: 0.35,
    band_contrast: 0.8,
    gloss_roughness: 0.12,
    metallic: 0.6,
    relief: 0.05,          // polished glass is nearly flat
    ..Default::default()
};
```

#### Chitin

Carapace plating.  Built on a soft minimum so plates swell into one another
rather than meeting at the crease a hard minimum gives; the suture is then
drawn back in deliberately.

```rust
use bevy_symbios_texture::chitin::ChitinConfig;

let config = ChitinConfig {
    seed: 37,
    scale: 6.0,            // plates across the tile
    jitter: 0.75,
    softness: 24.0,        // low merges plates, high approaches cut stone
    plate_fill: 0.9,       // how far a plate swells toward its neighbour
    plate_relief: 0.55,
    seam_width: 0.006,     // UV units
    seam_depth: 0.75,
    iridescence: 0.22,     // per-plate sheen, applied as a multiplier
    color: [0.20, 0.34, 0.20],
    color_deep: [0.05, 0.09, 0.07],
    gloss_roughness: 0.28,
    metallic: 0.45,
    pit_scale: 40.0,
    ..Default::default()
};
```

#### Solar panel

Photovoltaic wafers behind glass.  The wiring is laid in cell-local space so
it runs continuously across cells while the silicon does not — that continuity
is what reads as a panel rather than a tiled floor.

```rust
use bevy_symbios_texture::solar_panel::SolarPanelConfig;

let config = SolarPanelConfig {
    seed: 41,
    cells_x: 4.0,
    cells_y: 4.0,
    cell_gap: 0.06,        // fraction of a cell
    corner_cut: 0.14,      // wafers are cut from a round ingot
    busbars: 3.0,
    busbar_width: 0.014,   // coverage is width × count — keep both small
    fingers: 18.0,
    finger_width: 0.003,
    color_cell: [0.020, 0.030, 0.075],
    color_backing: [0.72, 0.72, 0.70],
    color_wire: [0.62, 0.63, 0.65],
    cell_variance: 0.18,   // multiplicative, so near-black silicon stays dark
    crystal_mottle: 0.30,
    crystal_scale: 22.0,
    glass_roughness: 0.10,
    ..Default::default()
};
```

#### Parquet

Short boards laid in a repeating figure.  Herringbone is not a grid of blocks:
a cell's board direction falls out of `(i − j) mod 2·aspect`, and because that
key shifts along both axes the runs interlock into the zig-zag.

```rust
use bevy_symbios_texture::parquet::{ParquetConfig, ParquetLayout};

let config = ParquetConfig {
    seed: 43,
    layout: ParquetLayout::Herringbone, // or Basket, Brick
    scale: 8.0,            // board slots across the tile
    aspect: 4.0,           // board length as a multiple of its width
    joint_width: 0.05,
    joint_depth: 0.5,
    grain_lines: 7.0,
    grain_contrast: 0.35,
    grain_warp: 0.22,
    board_variance: 0.13,
    color_wood: [0.36, 0.20, 0.09],
    color_grain: [0.19, 0.10, 0.04],
    color_joint: [0.07, 0.04, 0.02],
    gloss_roughness: 0.32,
    ..Default::default()
};
```

#### Truchet

Hashed quarter-arcs that meet at every tile edge whichever way the neighbour
fell, so a grid of coin flips reads as one routed network.  The emissive
channel is only collected when `emissive_intensity` is above zero.

```rust
use bevy_symbios_texture::truchet::TruchetConfig;

let config = TruchetConfig {
    seed: 47,
    scale: 6.0,            // tiles across the panel
    trace_width: 0.09,     // fraction of a tile
    trace_relief: 0.6,
    density: 0.85,         // below 1 the network breaks into runs and stubs
    color_panel: [0.035, 0.055, 0.050],
    color_trace: [0.16, 0.42, 0.34],
    color_glow: [0.10, 0.85, 0.60],
    emissive_intensity: 1.0, // 0 skips the emissive buffer entirely
    panel_roughness: 0.72,
    trace_roughness: 0.30,
    trace_metallic: 0.65,
    mottle_scale: 18.0,
    ..Default::default()
};
```

#### Weathering (shared)

Every generator depicting a **built or dressed** surface carries an optional
`weathering` block that ages it after generation: ashlar, asphalt, brick,
chitin, cobblestone, concrete, corrugated, enamel, encaustic, fabric, marble,
metal, obsidian, parquet, pavers, rock, shingle, solar panel, stucco, thatch,
truchet and wainscoting.

Natural surfaces are deliberately excluded — sand, snow, moss, bark and their
kin already read as weathered, and a second ageing pass over them fights the
generator rather than helping it. `plank` is the one gap: it still hand-rolls
its pixel loop instead of using the surface driver, so it cannot be handed a
config yet.  Every layer
defaults to an amount of zero, so an untouched block leaves the material
exactly as the generator drew it and costs nothing to bake.

Layers are applied in the order material actually ages: edge wear rubs raised
arrises back to the substrate, corrosion creeps out of crevices (adding its own
crust relief), grime settles into recesses, and runoff streaks draw down from
ledges.

```rust
use bevy_symbios_texture::rock::RockConfig;
use bevy_symbios_texture::weathering::{Streaks, WeatheringConfig};

let config = RockConfig {
    weathering: WeatheringConfig {
        seed: 4,
        streaks: Streaks {
            amount: 0.9,
            density: 0.5,  // fraction of candidate ledges that actually run
            length: 0.35,  // fraction of the tile height, so it survives a
                           // change of bake resolution
            ..Default::default()
        },
        ..Default::default()
    },
    ..Default::default()
};
```

### Alpha-masked cards

Card generators produce an RGBA8 texture where `albedo.alpha` encodes the
silhouette (`0` = fully transparent, `255` = fully opaque).  Upload with
`map_to_images_card` so the sampler does not tile and the alpha silhouette
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

#### Chain-Link

A woven diamond wire mesh: two cylindrical wire families at ±45° with
over/under crossing relief and rust pooling at the joints.  Transparent
between the wires.

```rust
use bevy_symbios_texture::chain_link::ChainLinkConfig;

let config = ChainLinkConfig {
    seed: 83,
    cell_count: 8.0,       // diamond cells across the card [4, 16]
    wire_radius: 0.07,     // wire radius in lattice units [0.02, 0.2]
    weave_depth: 0.6,      // over/under relief [0, 1]
    rust_level: 0.2,       // crossing rust [0, 1]
    color_wire: [0.62, 0.64, 0.66],
    color_rust: [0.45, 0.24, 0.10],
    normal_strength: 3.0,
};
```

#### Log End

The sawn end of a log: irregular round silhouette, FBM-wobbled concentric
growth rings, optional radial drying cracks, and a streaked bark rim.
Completes the wood set alongside `bark` and `plank`.

```rust
use bevy_symbios_texture::log_end::LogEndConfig;

let config = LogEndConfig {
    seed: 7,
    ring_count: 14.0,      // growth rings [4, 30]
    ring_warp: 0.35,       // ring wobble [0, 1]
    ring_contrast: 1.8,    // latewood band sharpness [0.5, 4]
    crack_count: 5.0,      // radial drying cracks [0, 12]; 0 = none
    bark_width: 0.07,      // bark rim thickness [0.02, 0.2]
    color_early: [0.78, 0.62, 0.42],
    color_late: [0.48, 0.33, 0.18],
    color_bark: [0.30, 0.20, 0.12],
    normal_strength: 2.5,
};
```

#### Lava

Cooling lava: dark basalt plates from a toroidal Voronoi decomposition,
separated by molten cracks that drive the **emissive map** — the glow colour
ramps with crack depth and is written to `StandardMaterial::emissive_texture`.
The material flow auto-enables a white emissive factor when a glow map is
present, so lava glows out of the box; set `emission_color` /
`emission_strength` only to tint or brighten it.

```rust
use bevy_symbios_texture::lava::LavaConfig;

let config = LavaConfig {
    seed: 666,
    plate_scale: 6.0,        // crust plates across the tile [3, 12]
    crack_width: 0.14,       // molten-crack width [0.02, 0.3]
    glow_falloff: 1.6,       // glow concentration exponent [0.5, 4]
    color_crust: [0.08, 0.07, 0.07],
    color_glow: [1.0, 0.45, 0.06],
    emissive_intensity: 1.0, // glow multiplier [0, 4]
    normal_strength: 4.0,
};
```

### Sprite atlases

The atlas family produces alpha-silhouette cards for billboards.  Unlike
the single-image cards above, each generator here can bake a
`variant_rows × variant_cols` **atlas** in a single image: every cell renders
the same config with a per-cell derived seed, so a particle system using
random atlas frames gets per-particle shape variety from one texture bake.
Atlas dimensions are clamped to `1..=16` per axis; `1 × 1` bakes a single
card.

The family spans two idioms.  The **particle sprites** (soft disc, spark,
snowflake, puff, ring, petal, shard, flame) use soft fractional alpha —
glows and mist fade out smoothly.  The **foliage billboards** (leaf sprite,
grass tuft, frond, reed, needle, broadleaf) cut hard silhouettes like the
foliage cards and default to `1 × 1` — a single vegetation card — with the
atlas as an opt-in for per-instance variety.

Shared scaffolding (the `SpriteCell` trait, the `generate_atlas` driver, and
the deterministic `CellRng` parameter stream) lives in the `sprite` module.
Upload with `map_to_images_card`; sprites never tile.

```rust
use bevy_symbios_texture::{
    generator::{TextureGenerator, map_to_images_card},
    spark::{SparkConfig, SparkGenerator},
};

let map = SparkGenerator::new(SparkConfig {
    variant_rows: 4,       // 4 × 4 atlas = 16 spark variants in one bake
    variant_cols: 4,
    ..SparkConfig::default()
})
.generate(512, 512)
.expect("valid dimensions");

let handles = map_to_images_card(map, &mut images);
```

#### Soft Disc

Radial-falloff disc with a solid core and tunable halo — the workhorse
particle sprite: fireflies, embers, mist motes, bokeh glints, additive glows.

```rust
use bevy_symbios_texture::soft_disc::SoftDiscConfig;

let config = SoftDiscConfig {
    seed: 0,
    variant_rows: 1,       // atlas rows [1, 16]
    variant_cols: 1,       // atlas columns [1, 16]
    color_core: [1.0, 0.98, 0.9],   // centre colour, linear RGB
    color_halo: [1.0, 0.72, 0.25],  // outer halo colour
    core_radius: 0.15,     // fully-opaque core radius [0, 0.9]
    falloff: 2.5,          // halo falloff exponent [0.3, 8]; higher = tighter glow
    ellipticity: 0.0,      // max per-variant elongation [0, 0.6]; 0 = always round
    scale_jitter: 0.15,    // per-variant shrink fraction [0, 0.5]
    normal_strength: 1.0,
};
```

#### Spark

N-pointed streak burst: a bright core with radial arms fading toward their
tips.  Embers, glints, impact sparks, magic sparkles.

```rust
use bevy_symbios_texture::spark::SparkConfig;

let config = SparkConfig {
    seed: 0,
    variant_rows: 2,
    variant_cols: 2,
    points: 4,             // number of streak arms [2, 12]
    color_core: [1.0, 0.95, 0.8],
    color_tip: [1.0, 0.45, 0.1],
    core_radius: 0.12,     // solid central glow radius [0.02, 0.5]
    arm_sharpness: 3.0,    // angular tightness [0.5, 10]; higher = needle-thin
    falloff: 1.8,          // radial fade exponent along each arm [0.5, 6]
    length_jitter: 0.3,    // per-arm length jitter [0, 0.8]
    normal_strength: 1.0,
};
```

#### Snowflake

Dendritic flake with N-fold symmetry: a central plate, one main arm per
sector, and paired side branches.  Per-variant jitter is where the "no two
snowflakes alike" character comes from.

```rust
use bevy_symbios_texture::snowflake::SnowflakeConfig;

let config = SnowflakeConfig {
    seed: 0,
    variant_rows: 2,
    variant_cols: 2,
    arms: 6,               // symmetry order [3, 8]; real snow is hexagonal
    color: [0.92, 0.96, 1.0],
    core_radius: 0.12,     // central plate radius [0, 0.4]
    arm_width: 0.045,      // main-arm half-width at the centre [0.01, 0.12]
    branch_pairs: 3,       // side-branch pairs per arm [0, 5]
    branch_angle: 1.05,    // branch angle, radians [0.3, 1.4]; ~60° is realistic
    branch_scale: 0.45,    // branch length vs remaining arm length [0.1, 1]
    softness: 0.02,        // anti-aliasing edge width [0.005, 0.08]
    normal_strength: 1.5,
};
```

#### Puff

Billowy blob of domain-warped fractal noise masked by a soft radial falloff.
Dust motes, smoke, fog banks, sea mist.

```rust
use bevy_symbios_texture::puff::PuffConfig;

let config = PuffConfig {
    seed: 0,
    variant_rows: 2,
    variant_cols: 2,
    color_base: [0.86, 0.86, 0.9],    // lit colour
    color_shadow: [0.52, 0.52, 0.58], // noise-trough colour
    noise_scale: 3.0,      // noise frequency across a cell [1, 8]
    octaves: 4,            // fractal octave count [1, 8]
    warp: 0.45,            // domain-warp strength [0, 1.5]; billows the silhouette
    density: 0.9,          // overall alpha multiplier [0, 1]
    edge_falloff: 2.0,     // radial mask exponent [0.5, 6]; higher = rounder puff
    contrast: 1.3,         // noise remap exponent [0.5, 4]; higher = wispier
    normal_strength: 1.0,
};
```

#### Ring

Soft annulus with optional angular waviness: shockwaves, water-drop ripples,
magic circles, halos.

```rust
use bevy_symbios_texture::ring::RingConfig;

let config = RingConfig {
    seed: 0,
    variant_rows: 1,
    variant_cols: 1,
    color: [0.85, 0.93, 1.0],
    radius: 0.6,           // centreline radius [0.1, 0.9]
    thickness: 0.12,       // annulus half-thickness [0.01, 0.5]
    falloff: 2.0,          // cross-section falloff exponent [0.5, 6]
    waviness: 0.0,         // angular radius modulation [0, 0.3]; 0 = perfect circle
    wave_count: 6,         // waviness lobes around the ring [2, 16]
    radius_jitter: 0.1,    // per-variant radius jitter [0, 0.4]
    normal_strength: 1.0,
};
```

#### Petal

A single flower petal: an obovate blade with a soft throat-to-edge gradient
and an optional notched tip.  Petal-fall particles, blossom decals, or — at
`1 × 1` — a building block for procedural flowers.

```rust
use bevy_symbios_texture::petal::PetalConfig;

let config = PetalConfig {
    seed: 0,
    variant_rows: 2,
    variant_cols: 2,
    color_base: [0.98, 0.72, 0.82],   // main blade colour
    color_edge: [0.93, 0.5, 0.66],    // silhouette-edge colour
    color_throat: [0.99, 0.88, 0.55], // attachment-point (nectar guide) tint
    length: 0.92,          // petal length as fraction of the cell [0.4, 1]
    width: 0.6,            // max blade width [0.15, 0.95]
    peak: 0.65,            // position of max width from the throat [0.3, 0.9]
    tip_notch: 0.08,       // tip notch radius [0, 0.25]; 0 = smooth tip
    curl: 0.4,             // lateral shading strength [0, 1]; fakes curl
    asymmetry: 0.15,       // max per-variant axis skew [0, 0.4]
    normal_strength: 1.5,
};
```

#### Shard

Irregular rock-chip / debris-flake silhouette: a jittered polygon with a
darkened rim and noise-grained interior.  Impact debris, crumbling masonry,
shattered ice, kicked-up gravel.

```rust
use bevy_symbios_texture::shard::ShardConfig;

let config = ShardConfig {
    seed: 0,
    variant_rows: 2,
    variant_cols: 2,
    color_base: [0.46, 0.43, 0.4],  // interior colour
    color_edge: [0.24, 0.22, 0.21], // fractured-rim colour
    sides: 5,              // polygon vertex count [3, 9]
    irregularity: 0.45,    // vertex jitter [0, 0.9]; 0 = regular polygon
    edge_band: 0.18,       // darkened-rim width as fraction of radius [0.02, 0.5]
    grain: 0.35,           // interior fractal-grain strength [0, 1]
    normal_strength: 2.5,
};
```

#### Flame

A single tongue of fire: a teardrop envelope displaced by fractal
turbulence that grows toward the tip, with a core→mid→tip colour ramp.
Per-variant cells jitter lean, elongation, and turbulence phase, so random
atlas frames read as flicker.  Pairs well with additive blending.

```rust
use bevy_symbios_texture::flame::FlameConfig;

let config = FlameConfig {
    seed: 0,
    variant_rows: 2,       // atlas rows [1, 16]
    variant_cols: 2,       // atlas columns [1, 16]
    elongation: 1.6,       // vertical stretch [1, 3]; taller = lazier wisp
    turbulence: 0.55,      // tip displacement strength [0, 1.5]
    lean_jitter: 0.25,     // max per-variant sideways lean [0, 0.5]
    falloff: 1.6,          // envelope fade exponent [0.5, 4]
    color_core: [1.0, 0.97, 0.78],  // hot base
    color_mid: [1.0, 0.55, 0.10],
    color_tip: [0.85, 0.16, 0.02],  // cool fringe
    normal_strength: 1.0,
};
```

#### Flower

A radially composed blossom: petal blades (the petal sampler re-aimed
outward) under a domed, stamen-dotted centre disc — the sprite counterpart
of how `twig` composites leaves.  Every petal in every variant draws its
own jitter stream.

```rust
use bevy_symbios_texture::flower::FlowerConfig;
use bevy_symbios_texture::petal::PetalConfig;

let config = FlowerConfig {
    seed: 0,
    variant_rows: 2,       // atlas rows [1, 16]
    variant_cols: 2,       // atlas columns [1, 16]
    petal: PetalConfig::default(), // shared blade appearance
    petal_count: 6,        // blades [4, 12]
    center_radius: 0.14,   // centre disc radius [0.05, 0.3]
    center_color: [0.96, 0.78, 0.25],
    dot_density: 0.5,      // stamen dots [0, 1]
    normal_strength: 1.5,
};
```

#### Leaf Sprite

The atlas counterpart of the single-leaf foliage card: every cell bakes a
per-cell-seeded leaf variant with bounded jitter on serration, lobes, vein
count, and a green-preserving colour tint.  Falling-foliage particle
systems get per-particle leaf variety from one bake.

```rust
use bevy_symbios_texture::leaf::LeafConfig;
use bevy_symbios_texture::leaf_sprite::LeafSpriteConfig;

let config = LeafSpriteConfig {
    seed: 0,
    variant_rows: 2,       // atlas rows [1, 16]
    variant_cols: 2,       // atlas columns [1, 16]
    leaf: LeafConfig::default(), // base leaf appearance for every cell
    shape_jitter: 0.5,     // silhouette jitter [0, 1]
    tint_jitter: 0.25,     // colour tint jitter [0, 1]
};
```

#### Grass Tuft

A clump of curved, tip-tapered grass blades fanning from a common root line
at the bottom edge — the workhorse ground-cover billboard.  Blades jitter
height, lean, curvature, width, and dryness per variant.

```rust
use bevy_symbios_texture::grass::GrassTuftConfig;

let config = GrassTuftConfig {
    seed: 0,
    variant_rows: 1,       // atlas rows [1, 16]; 1 × 1 bakes a single tuft
    variant_cols: 1,
    blade_count: 9,        // blades per tuft [1, 24]
    color_base: [0.11, 0.17, 0.06],  // shaded root colour
    color_tip: [0.36, 0.46, 0.14],   // bright tip colour
    color_dry: [0.46, 0.39, 0.15],   // dry/dead blade tone
    blade_width: 0.05,     // root half-width as cell fraction [0.01, 0.12]
    blade_taper: 1.3,      // tip taper exponent [0.5, 4]
    height_min: 0.55,      // shortest blade height [0.2, 1]
    height_max: 0.96,      // tallest blade height [0.2, 1]
    fan_spread: 0.34,      // lateral tip splay [0, 0.5]
    curve: 0.14,           // outward blade arc toward the tip [0, 0.5]
    base_spread: 0.16,     // horizontal root spread [0, 0.4]
    dry_fraction: 0.22,    // share of dry blades [0, 1]
    normal_strength: 1.2,
};
```

#### Reed

A shoreline reed / cattail clump: tall, near-vertical strap leaves rising
from a common waterline base, with an optional share of stalks topped by
the cattail's brown catkin spike.  Far taller and straighter than the grass
tuft.

```rust
use bevy_symbios_texture::reed::ReedConfig;

let config = ReedConfig {
    seed: 0,
    variant_rows: 1,       // atlas rows [1, 16]
    variant_cols: 1,
    blade_count: 6,        // leaves per clump [1, 12]
    color_base: [0.10, 0.16, 0.06],
    color_tip: [0.38, 0.44, 0.16],
    color_catkin: [0.24, 0.13, 0.05],  // seed-head colour
    blade_width: 0.022,    // base half-width as cell fraction [0.008, 0.08]
    height_min: 0.62,      // shortest leaf height [0.3, 1]
    height_max: 0.98,      // tallest leaf height [0.3, 1]
    lean: 0.09,            // lateral tip lean [0, 0.3]; reeds stand straight
    tip_fraction: 0.28,    // leaf-length fraction that tapers [0.05, 0.8]
    catkin_share: 0.4,     // stalks bearing a catkin [0, 1]
    catkin_length: 0.2,    // catkin length as cell fraction [0, 0.4]
    catkin_width: 0.022,   // catkin half-width [0.005, 0.06]
    normal_strength: 1.2,
};
```

#### Needle

A conifer needle-cluster shoot — the conifer counterpart of the broadleaf
twig card.  Paired needles splay outward and forward from a woody axis,
shortening toward the tip.  A wide `needle_angle` with long needles reads
as pine, a narrow angle with short needles as spruce, a high `pair_count`
with minimal taper as fir.

```rust
use bevy_symbios_texture::needle::NeedleConfig;

let config = NeedleConfig {
    seed: 0,
    variant_rows: 1,       // atlas rows [1, 16]
    variant_cols: 1,
    pair_count: 11,        // needle pairs along the shoot [1, 24]
    color_base: [0.05, 0.13, 0.07],
    color_tip: [0.16, 0.31, 0.14],
    color_shoot: [0.21, 0.13, 0.07],  // woody axis colour
    needle_angle: 42.0,    // splay from the shoot axis, degrees [5, 85]
    needle_length: 0.3,    // needle length as cell fraction [0.05, 0.6]
    needle_width: 0.009,   // needle half-width [0.002, 0.03]
    length_taper: 0.55,    // shortening toward the shoot tip [0, 1]
    shoot_length: 0.9,     // shoot length as cell fraction [0.2, 1]
    shoot_width: 0.009,    // shoot half-width [0.002, 0.04]
    normal_strength: 1.2,
};
```

#### Frond

A single leaflet (pinna) of a pinnate frond: a narrow lanceolate strap with
a strong central midrib and shallow pinnate veins.  One depth knob spans an
entire (palm-leaflet) margin to the lobed pinnule of a fern — the drop-in
leaflet card for an L-system palm or fern whose rachis geometry is drawn by
the grammar.

```rust
use bevy_symbios_texture::frond::FrondConfig;

let config = FrondConfig {
    seed: 0,
    variant_rows: 1,       // atlas rows [1, 16]
    variant_cols: 1,
    color_base: [0.11, 0.30, 0.09],
    color_edge: [0.22, 0.42, 0.13],  // margin / tip colour
    width: 0.13,           // max half-width as cell fraction [0.04, 0.30]
    tip_taper: 1.4,        // tip acuteness [0.4, 3]
    midrib_width: 0.16,    // midrib ridge width as local half-width fraction
    vein_count: 9.0,       // pinnate secondary vein pairs
    lobe_count: 0.0,       // margin lobes per side; 0 = entire margin
    lobe_depth: 0.0,       // lobe cut depth [0, 0.6]; 0 = smooth palm leaflet
    normal_strength: 1.3,
};
```

#### Broadleaf

A palmate broadleaf built in polar coordinates about its petiole
attachment: a radius function with `lobe_count` peaks carves the classic
maple / sycamore / ivy silhouettes, with main veins radiating to each lobe
tip and an optional cordate (heart) base notch.  `lobe_count: 1.0` with a
shallow depth yields a plain ovate blade, so one generator covers both the
palmate and simple-broadleaf families — a different leaf *form* from the
pinnate, midrib-based leaf card.

```rust
use bevy_symbios_texture::broadleaf::BroadleafConfig;

let config = BroadleafConfig {
    seed: 0,
    variant_rows: 1,       // atlas rows [1, 16]
    variant_cols: 1,
    color_base: [0.13, 0.26, 0.08],
    color_edge: [0.28, 0.38, 0.12],  // margin colour
    lobe_count: 5.0,       // palmate lobes [1, 9]; 5 = maple, 3 = ivy
    lobe_depth: 0.34,      // sinus depth between lobes [0, 0.8]
    fan_angle: 78.0,       // fan half-angle, degrees [30, 110]
    radius: 0.92,          // blade radius as cell fraction [0.3, 1]
    base_notch: 0.18,      // cordate basal notch depth [0, 0.5]; 0 = wedge
    vein_width: 0.05,      // radiating main-vein width [0.01, 0.2]
    petiole_length: 0.1,   // V-axis fraction reserved for the stalk [0, 0.3]
    normal_strength: 1.4,
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

The `Genotype` implementations and the egui editor widgets are generated by
declarative macros (`impl_genotype!` / `impl_config_editor!`) rather than
hand-written per-config boilerplate.  Each macro invocation declares the
config struct, field kinds (seed, f64, colour, enum, etc.), and optional
post-hooks for tiling-invariant fixups, keeping the per-config call site
small while covering all 48 config types.

`TextureConfig` itself also implements `Genotype` (mutation delegates to the
wrapped config; crossover recombines like variants field-wise) and exposes
registry-derived helpers — `all_defaults()` for dropdowns and benches,
`module_name()` for stable identifiers, `generate_sync()` for synchronous
dispatch, and (behind the `egui` feature) `ui::texture_config_editor` for
variant-generic parameter editing.  The `texture_viewer` example and the
criterion bench suite are built entirely on these, so they extend
automatically when a generator is added to the registry.

## Architecture

The crate is a two-layer stack: the Bevy-free core —
[`symbios-texture`](https://crates.io/crates/symbios-texture) — owns
everything in the diagram below and is re-exported wholesale, while this
wrapper adds the Bevy layer on top: `SymbiosTexturePlugin`, the async
generation pool, the `Image` upload adapters (`map_to_images` /
`map_to_images_card`), the `TextureCache` resource, the
procedural-material builder, and the egui editors.

```text
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
    ├── FabricGenerator     ─── perpendicular thread lattice + over/under weave
    ├── SandGenerator       ─── warped sine ripples + grain flecks
    ├── SnowGenerator       ─── FBM drift relief + sparkle flecks
    ├── IceGenerator        ─── sinusoidal crack veins + frost patches
    ├── LavaGenerator       ─── toroidal Voronoi plates + emissive crack glow
    ├── MossGenerator       ─── ToroidalNoise FBM × 3 (cushion/filament/dry)
    ├── LichenGenerator     ─── ToroidalNoise FBM (thresholded thallus patches)
    ├── CactusSkinGenerator ─── periodic rib/areole lattice + sinusoid mottle
    │
    │  Alpha-masked cards
    ├── LeafGenerator       ─── LeafSampler (silhouette + venation)
    ├── TwigGenerator       ─── LeafSampler × N (composite stem + leaves)
    ├── WindowGenerator     ─── rounded-box SDF frame/mullions + FBM grime
    ├── StainedGlassGenerator── toroidal Voronoi + lead came SDF + grime FBM
    ├── IronGrilleGenerator ─── bar SDF grid + joint rust FBM
    ├── ChainLinkGenerator  ─── diagonal wire lattice + over/under weave
    ├── LogEndGenerator     ─── warped concentric rings + bark rim
    │
    │  Sprite atlases (alpha-masked cards, via sprite::generate_atlas)
    ├── SoftDiscGenerator   ─── radial-falloff disc (core + halo)
    ├── SparkGenerator      ─── N-armed streak burst
    ├── SnowflakeGenerator  ─── dendritic N-fold flake
    ├── PuffGenerator       ─── domain-warped FBM blob + radial mask
    ├── RingGenerator       ─── soft annulus + angular waviness
    ├── PetalGenerator      ─── obovate blade + throat/edge gradient
    ├── ShardGenerator      ─── jittered polygon chip + grain FBM
    ├── LeafSpriteGenerator ─── LeafSampler atlas (per-cell leaf variants)
    ├── FlameGenerator      ─── teardrop envelope + FBM turbulence
    ├── FlowerGenerator     ─── PetalCell × N (radial composite blossom)
    ├── GrassTuftGenerator  ─── fanned, curved blade ribbons
    ├── FrondGenerator      ─── lanceolate pinna + pinnate veins
    ├── ReedGenerator       ─── strap leaves + catkin spikes
    ├── NeedleGenerator     ─── paired-needle conifer shoot
    └── BroadleafGenerator  ─── polar palmate silhouette + radiating veins
                                │
                        height_to_normal() → normal map
                        linear_to_srgb()   → albedo encoding
                                │
                 TextureMap { albedo, normal, roughness, emissive? }
                                │
                map_to_images()      → GeneratedHandles (repeat sampler)
                map_to_images_card() → GeneratedHandles (clamp sampler)
                                │
                        full mipmap chain (type-correct averaging)
```

**Noise-in-constructor** — the surface generators and the SDF-based cards
(window, stained glass, iron grille) build their noise objects
(`Fbm<Perlin>`, `RidgedMulti<Perlin>`, `ToroidalNoise<…>`) once in `new()`
and store them as struct fields.  Calling `generate()` multiple times (e.g.
to produce size variants of the same material) skips the initialisation cost.
`Worley` is the exception: it contains an `Rc` and is therefore `!Send` /
`!Sync`, so the generators that use it (`BarkGenerator`, `PlankGenerator`)
construct it locally — and, since generation is row-parallel, once per row
inside the parallel loop (the construction cost is microseconds against the
per-row pixel work).  The foliage cards (`LeafGenerator`, `TwigGenerator` —
leaf sampling also uses Worley) and the atlas generators hold only their
config and build their samplers per `generate()` call.  The cell-decomposition
surfaces (`CobblestoneGenerator`, `LavaGenerator`) instead use a dependency-free,
`Sync` hash-based toroidal Voronoi (`noise::toroidal_voronoi`).

**Workspace buffer pooling** — generators that allocate large intermediate
grids (e.g. `BarkGenerator`, `ThatchGenerator`) accept an optional
`Workspace` via `generate_with_workspace()`.  The workspace maintains a
pool of `Vec<f64>` buffers that are borrowed and returned across calls,
eliminating repeated 128 MB+ allocations at 4096×4096 resolution.

**Seamless tiling** is provided by `ToroidalNoise`, which maps 2-D UV
coordinates onto a 4-D torus so that noise wraps perfectly at every edge:

```text
nx = cos(2π·u) · frequency
ny = sin(2π·u) · frequency
nz = cos(2π·v) · frequency
nw = sin(2π·v) · frequency
```

Because `cos(0) = cos(2π)` and `sin(0) = sin(2π)`, `u=0` and `u=1` always
resolve to the same 4-D point, guaranteeing zero-seam tiling.

**Normal maps** are derived from the height field via central-difference
gradients.  For the tileable surface textures the neighbours wrap toroidally,
so the normals are also seamless.  For card textures (the foliage and
architectural cards and the whole atlas family) the boundary uses clamp-to-edge
so normals do not bleed across the transparent silhouette border.  Sprite
atlases additionally dilate heights into fully-transparent texels before
derivation so the normals do not crease at silhouette edges.

**Colour encoding** uses a 4096-entry sRGB lookup table (built once via
`OnceLock`) to avoid repeated `f32::powf` calls during rasterisation.
A 256-entry table would be insufficient because the sRGB curve is steep
near zero; 4096 bins keep the maximum quantisation error well below one
count in u8.

**Mipmap generation** uses a 2×2 box filter with type-correct averaging: sRGB
values are decoded to linear light before averaging and re-encoded afterward
(avoiding dark mipmaps), normal-map XYZ vectors are averaged and renormalized
(avoiding zero-length normals in PBR shaders), and ORM values are averaged
directly in linear space.  Async generation tasks precompute the full chain
on the worker thread (`TextureMap::with_mips`), so the main-thread upload in
the polling systems is a pure buffer move; `map_to_images` /
`map_to_images_card` compute the chain on demand for maps without one
(synchronous callers, cache loads).  16× anisotropic filtering is enabled on
all samplers.

## Examples

### texture_viewer

```sh
cargo run --release --example texture_viewer --features egui
```

Displays an interactive material viewer with three columns: **albedo** (left),
**normal map** (centre), and a **3-D PBR preview** (right) with the generated
material applied.  Tileable surface textures are shown on a spinning cube;
alpha-masked cards and sprite atlases get a gently swaying alpha-blended quad
in front of a checkerboard backdrop instead, so per-pixel alpha is visible.
An egui panel on the left lets you select any of the 48 generators from a
dropdown, trigger a random **Mutate** (rate = 0.3), and edit every parameter
live.

### procedural_material

```sh
cargo run --release --example procedural_material
```

Side-by-side comparison of `build_procedural_material_async` (left cube)
against the manual `PendingTexture` + material-patching flow it replaces
(right cube).

### animated_rust

```sh
cargo run --release --example animated_rust
```

Animates rust coverage on a metal panel from 0 % to 100 % over ten seconds
via `AnimatedProceduralMaterial` driving a `Linear` curve, throttled to
roughly four regenerations per second.

## License

MIT — see [LICENSE](LICENSE).
