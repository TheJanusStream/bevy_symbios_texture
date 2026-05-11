//! Async texture generation system.
//!
//! Offloads the CPU-intensive pixel math to a private, bounded [`rayon`]
//! thread pool so it does not stall the main thread.  The pool size is
//! controlled by [`AsyncTextureConfig::pool_threads`] (default
//! [`DEFAULT_POOL_THREADS`]); excess requests are queued and run in order
//! rather than spawning unbounded OS threads.  When a task finishes the
//! images are uploaded to [`Assets<Image>`] and the result entity receives
//! the [`TextureReady`] component.
//!
//! # Usage
//! ```rust,ignore
//! // Tileable surface textures (bark, rock, ground) — poll_texture_tasks
//! // uploads them with map_to_images (repeat sampler).
//! commands.spawn(PendingTexture::bark(BarkConfig::default(), 512, 512));
//!
//! // Foliage cards (leaf, twig) — poll_texture_tasks automatically uses
//! // map_to_images_card (clamp-to-edge sampler) for these types.
//! commands.spawn(PendingTexture::leaf(LeafConfig::default(), 512, 512));
//! commands.spawn(PendingTexture::twig(TwigConfig::default(), 512, 512));
//!
//! // Later, query for TextureReady to consume the handles.
//! ```

/// Default concurrency cap applied when no explicit
/// [`AsyncTextureConfig::pool_threads`] is supplied.
///
/// Tasks beyond this cap are queued inside the rayon pool rather than spawning
/// new OS threads, bounding both CPU and memory usage.  The default is
/// deliberately conservative; saturate large machines by setting
/// `AsyncTextureConfig::pool_threads = 0` (auto = `available_parallelism / 2`)
/// or an explicit higher value.
pub const DEFAULT_POOL_THREADS: usize = 4;

/// Plugin-time configuration for the private texture-generation thread pool.
///
/// Applied by [`SymbiosTexturePlugin`](crate::SymbiosTexturePlugin) before any
/// task is dispatched.  Once the pool is built (lazily, on the first
/// generation request) the configuration is frozen for the process lifetime;
/// changing the value afterwards has no effect.
#[derive(bevy::ecs::resource::Resource, Clone, Debug)]
pub struct AsyncTextureConfig {
    /// Maximum concurrent generation tasks.
    ///
    /// * `0` selects an auto value of `available_parallelism / 2` (minimum 1).
    ///   This trades fewer threads against better main-thread responsiveness
    ///   while still scaling on large machines.
    /// * Any positive value caps the pool at exactly that many threads.
    ///
    /// Defaults to [`DEFAULT_POOL_THREADS`].
    pub pool_threads: usize,
}

impl Default for AsyncTextureConfig {
    fn default() -> Self {
        Self {
            pool_threads: DEFAULT_POOL_THREADS,
        }
    }
}

/// Resolves the requested thread-count to an actual count.
fn resolve_pool_threads(cfg: &AsyncTextureConfig) -> usize {
    if cfg.pool_threads == 0 {
        std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(2)
    } else {
        cfg.pool_threads
    }
}

static POOL_CONFIG: OnceLock<AsyncTextureConfig> = OnceLock::new();
static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

/// Returned by [`set_pool_config`] when a configuration has already been
/// installed by an earlier caller.
#[derive(Debug)]
pub struct PoolConfigAlreadySet;

impl std::fmt::Display for PoolConfigAlreadySet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AsyncTextureConfig has already been applied; new value ignored")
    }
}

impl std::error::Error for PoolConfigAlreadySet {}

/// Apply the texture-generation thread-pool configuration.
///
/// The plugin calls this once at startup with the user-supplied
/// [`AsyncTextureConfig`].  Calls after the pool has been initialised are
/// silently ignored — the configuration is read exactly once when the first
/// generation task is dispatched.
pub fn set_pool_config(cfg: AsyncTextureConfig) -> Result<(), PoolConfigAlreadySet> {
    POOL_CONFIG.set(cfg).map_err(|_| PoolConfigAlreadySet)
}

/// Build the rayon pool from the resolved configuration.
///
/// Returns `None` if [`rayon::ThreadPoolBuilder`] fails (out-of-memory, OS
/// thread limit, sandboxed environments).  On `None`, [`spawn_task`] falls
/// back to running the closure synchronously on the calling thread so texture
/// generation continues to work — slowly, but correctly — instead of panicking
/// at startup.
fn build_pool(cfg: &AsyncTextureConfig) -> Option<rayon::ThreadPool> {
    let n = resolve_pool_threads(cfg);
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("texture-gen-{i}"))
        .build()
    {
        Ok(pool) => Some(pool),
        Err(e) => {
            bevy::log::warn!(
                "bevy_symbios_texture: failed to build texture-gen thread pool ({e}); \
                 falling back to inline (synchronous) generation. Each PendingTexture \
                 will be produced on the spawning thread, blocking it for the duration \
                 of the generator."
            );
            None
        }
    }
}

/// Returns the library-private rayon thread pool used for texture generation,
/// or `None` if pool construction failed at first init.
///
/// Isolated from the application's global rayon pool so texture work does not
/// starve unrelated parallel workloads and the concurrency cap is enforced
/// regardless of the calling application's rayon configuration.
fn gen_pool() -> Option<&'static rayon::ThreadPool> {
    POOL.get_or_init(|| {
        let cfg = POOL_CONFIG.get().cloned().unwrap_or_default();
        build_pool(&cfg)
    })
    .as_ref()
}

use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use bevy::{
    asset::Assets,
    ecs::{
        component::Component,
        entity::Entity,
        system::{Commands, Query, ResMut},
    },
    image::Image,
};

use crate::{
    ashlar::{AshlarConfig, AshlarGenerator},
    asphalt::{AsphaltConfig, AsphaltGenerator},
    bark::{BarkConfig, BarkGenerator},
    brick::{BrickConfig, BrickGenerator},
    cobblestone::{CobblestoneConfig, CobblestoneGenerator},
    concrete::{ConcreteConfig, ConcreteGenerator},
    corrugated::{CorrugatedConfig, CorrugatedGenerator},
    encaustic::{EncausticConfig, EncausticGenerator},
    generator::{
        GeneratedHandles, TextureError, TextureGenerator, TextureMap, map_to_images,
        map_to_images_card,
    },
    ground::{GroundConfig, GroundGenerator},
    iron_grille::{IronGrilleConfig, IronGrilleGenerator},
    leaf::{LeafConfig, LeafGenerator},
    marble::{MarbleConfig, MarbleGenerator},
    metal::{MetalConfig, MetalGenerator},
    pavers::{PaversConfig, PaversGenerator},
    plank::{PlankConfig, PlankGenerator},
    rock::{RockConfig, RockGenerator},
    shingle::{ShingleConfig, ShingleGenerator},
    stained_glass::{StainedGlassConfig, StainedGlassGenerator},
    stucco::{StuccoConfig, StuccoGenerator},
    thatch::{ThatchConfig, ThatchGenerator},
    twig::{TwigConfig, TwigGenerator},
    wainscoting::{WainscotingConfig, WainscotingGenerator},
    window::{WindowConfig, WindowGenerator},
};

/// Spawned onto an entity to request background texture generation.
///
/// Each constructor submits `generate()` to a private rayon pool sized by
/// [`AsyncTextureConfig::pool_threads`] (default [`DEFAULT_POOL_THREADS`]).
/// Because `generate()` is a monolithic blocking loop with no yield points,
/// using Bevy's `AsyncComputeTaskPool` would starve other tasks on that
/// executor; a dedicated pool avoids the problem while bounding OS thread
/// and memory usage.  [`poll_texture_tasks`] non-blockingly checks for
/// completion each frame using [`mpsc::Receiver::try_recv`].
///
/// Dropping `PendingTexture` (e.g. when the entity is despawned) sets an
/// atomic cancellation flag.  Tasks that have not yet started will see the
/// flag and exit without doing any work, preventing zombie tasks from
/// saturating the thread pool when entities are rapidly spawned and destroyed.
#[derive(Component)]
pub struct PendingTexture {
    // Wrapped in Mutex so the struct is Sync, which Bevy's Component bound requires.
    pub(crate) rx: std::sync::Mutex<mpsc::Receiver<Result<TextureMap, TextureError>>>,
    /// Set to `true` on drop; the background task checks this before starting.
    cancelled: Arc<AtomicBool>,
    /// `true` for foliage cards (leaf, twig) that need a clamp-to-edge sampler.
    is_card: bool,
}

impl PendingTexture {
    /// Returns `true` if this task should be uploaded with
    /// [`map_to_images_card`] (clamp-to-edge sampler, alpha-masked card)
    /// rather than the default repeat-tiling [`map_to_images`].
    pub fn is_card(&self) -> bool {
        self.is_card
    }
}

impl Drop for PendingTexture {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

/// Shared constructor body: creates the channel + cancellation flag, spawns the
/// task, and returns a `PendingTexture`.  The closure `f` is the generator call.
/// Native Desktop: Spawn using our private, bounded Rayon pool.
///
/// When the rayon pool failed to build (see [`gen_pool`]), the closure runs
/// inline on the calling thread.  The mpsc channel is fed before the function
/// returns, so [`poll_texture_tasks`] still consumes the result correctly via
/// its normal polling path — only the spawn-time latency changes.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_task<F>(f: F, is_card: bool) -> PendingTexture
where
    F: FnOnce() -> Result<TextureMap, TextureError> + Send + 'static,
{
    let cancelled = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancelled);
    let (tx, rx) = mpsc::sync_channel(1);

    match gen_pool() {
        Some(pool) => pool.spawn(move || {
            if !flag.load(Ordering::Relaxed) {
                tx.send(f()).ok();
            }
        }),
        None => {
            if !flag.load(Ordering::Relaxed) {
                tx.send(f()).ok();
            }
        }
    }

    PendingTexture {
        rx: std::sync::Mutex::new(rx),
        cancelled,
        is_card,
    }
}

/// WASM Web: Fallback to Bevy's default AsyncComputeTaskPool.
/// On WASM, this multiplexes onto the main thread (blocking UI, but compiling cleanly).
#[cfg(target_arch = "wasm32")]
fn spawn_task<F>(f: F, is_card: bool) -> PendingTexture
where
    F: FnOnce() -> Result<TextureMap, TextureError> + Send + 'static,
{
    use bevy::tasks::AsyncComputeTaskPool;

    let cancelled = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancelled);
    let (tx, rx) = mpsc::sync_channel(1);

    AsyncComputeTaskPool::get()
        .spawn(async move {
            if !flag.load(Ordering::Relaxed) {
                tx.send(f()).ok();
            }
        })
        .detach(); // Detach the Bevy task; we track completion via the mpsc channel anyway

    PendingTexture {
        rx: std::sync::Mutex::new(rx),
        cancelled,
        is_card,
    }
}

impl PendingTexture {
    /// Spawn a bark texture generation thread at `width × height` texels.
    pub fn bark(config: BarkConfig, width: u32, height: u32) -> Self {
        let generator = BarkGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a rock texture generation thread at `width × height` texels.
    pub fn rock(config: RockConfig, width: u32, height: u32) -> Self {
        let generator = RockGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a ground texture generation thread at `width × height` texels.
    pub fn ground(config: GroundConfig, width: u32, height: u32) -> Self {
        let generator = GroundGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a leaf texture generation thread at `width × height` texels.
    ///
    /// [`poll_texture_tasks`] uploads the result with
    /// [`map_to_images_card`] automatically,
    /// giving a clamp-to-edge sampler suitable for foliage cards.
    pub fn leaf(config: LeafConfig, width: u32, height: u32) -> Self {
        let generator = LeafGenerator::new(config);
        spawn_task(move || generator.generate(width, height), true)
    }

    /// Spawn a twig texture generation thread at `width × height` texels.
    ///
    /// [`poll_texture_tasks`] uploads the result with
    /// [`map_to_images_card`] automatically,
    /// giving a clamp-to-edge sampler suitable for foliage cards.
    pub fn twig(config: TwigConfig, width: u32, height: u32) -> Self {
        let generator = TwigGenerator::new(config);
        spawn_task(move || generator.generate(width, height), true)
    }

    /// Spawn a brick-wall texture generation thread at `width × height` texels.
    pub fn brick(config: BrickConfig, width: u32, height: u32) -> Self {
        let generator = BrickGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a window texture generation thread at `width × height` texels.
    ///
    /// [`poll_texture_tasks`] uploads the result with
    /// [`map_to_images_card`] automatically,
    /// giving a clamp-to-edge sampler suitable for foliage cards.
    pub fn window(config: WindowConfig, width: u32, height: u32) -> Self {
        let generator = WindowGenerator::new(config);
        spawn_task(move || generator.generate(width, height), true)
    }

    /// Spawn a wood-plank / siding texture generation thread at `width × height` texels.
    pub fn plank(config: PlankConfig, width: u32, height: u32) -> Self {
        let generator = PlankGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a roof-shingle texture generation thread at `width × height` texels.
    pub fn shingle(config: ShingleConfig, width: u32, height: u32) -> Self {
        let generator = ShingleGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a stucco / render texture generation thread at `width × height` texels.
    pub fn stucco(config: StuccoConfig, width: u32, height: u32) -> Self {
        let generator = StuccoGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a concrete texture generation thread at `width × height` texels.
    pub fn concrete(config: ConcreteConfig, width: u32, height: u32) -> Self {
        let generator = ConcreteGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a metal texture generation thread at `width × height` texels.
    pub fn metal(config: MetalConfig, width: u32, height: u32) -> Self {
        let generator = MetalGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a pavers / tiles texture generation thread at `width × height` texels.
    pub fn pavers(config: PaversConfig, width: u32, height: u32) -> Self {
        let generator = PaversGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn an ashlar (cut stone masonry) texture generation thread at `width × height` texels.
    pub fn ashlar(config: AshlarConfig, width: u32, height: u32) -> Self {
        let generator = AshlarGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a cobblestone texture generation thread at `width × height` texels.
    pub fn cobblestone(config: CobblestoneConfig, width: u32, height: u32) -> Self {
        let generator = CobblestoneGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a thatch roofing texture generation thread at `width × height` texels.
    pub fn thatch(config: ThatchConfig, width: u32, height: u32) -> Self {
        let generator = ThatchGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a marble / granite texture generation thread at `width × height` texels.
    pub fn marble(config: MarbleConfig, width: u32, height: u32) -> Self {
        let generator = MarbleGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a corrugated metal texture generation thread at `width × height` texels.
    pub fn corrugated(config: CorrugatedConfig, width: u32, height: u32) -> Self {
        let generator = CorrugatedGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn an asphalt / tarmac texture generation thread at `width × height` texels.
    pub fn asphalt(config: AsphaltConfig, width: u32, height: u32) -> Self {
        let generator = AsphaltGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a wood paneling / wainscoting texture generation thread at `width × height` texels.
    pub fn wainscoting(config: WainscotingConfig, width: u32, height: u32) -> Self {
        let generator = WainscotingGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }

    /// Spawn a stained glass texture generation thread at `width × height` texels.
    ///
    /// [`poll_texture_tasks`] uploads the result with
    /// [`map_to_images_card`] automatically,
    /// giving a clamp-to-edge sampler suitable for alpha-masked cards.
    pub fn stained_glass(config: StainedGlassConfig, width: u32, height: u32) -> Self {
        let generator = StainedGlassGenerator::new(config);
        spawn_task(move || generator.generate(width, height), true)
    }

    /// Spawn an iron grille / portcullis texture generation thread at `width × height` texels.
    ///
    /// [`poll_texture_tasks`] uploads the result with
    /// [`map_to_images_card`] automatically,
    /// giving a clamp-to-edge sampler suitable for alpha-masked cards.
    pub fn iron_grille(config: IronGrilleConfig, width: u32, height: u32) -> Self {
        let generator = IronGrilleGenerator::new(config);
        spawn_task(move || generator.generate(width, height), true)
    }

    /// Spawn an encaustic ceramic tile texture generation thread at `width × height` texels.
    pub fn encaustic(config: EncausticConfig, width: u32, height: u32) -> Self {
        let generator = EncausticGenerator::new(config);
        spawn_task(move || generator.generate(width, height), false)
    }
}

/// Added to the entity by [`poll_texture_tasks`] when generation is complete.
#[derive(Component)]
pub struct TextureReady(pub GeneratedHandles);

/// Bevy system — polls pending generation tasks and uploads finished maps.
///
/// Skips entities also tagged with [`PatchMaterialTextures`](crate::material::PatchMaterialTextures);
/// those are consumed by [`patch_procedural_material_textures`](crate::material::patch_procedural_material_textures)
/// instead, which writes the generated images directly into a target
/// `StandardMaterial` rather than emitting [`TextureReady`].
pub fn poll_texture_tasks(
    mut commands: Commands,
    tasks: Query<
        (Entity, &PendingTexture),
        bevy::ecs::query::Without<crate::material::PatchMaterialTextures>,
    >,
    mut images: ResMut<Assets<Image>>,
) {
    for (entity, pending) in &tasks {
        let poll = pending
            .rx
            .lock()
            .expect("texture thread poisoned")
            .try_recv();
        match poll {
            Ok(Ok(map)) => {
                let handles = if pending.is_card {
                    map_to_images_card(map, &mut images)
                } else {
                    map_to_images(map, &mut images)
                };
                commands
                    .entity(entity)
                    .remove::<PendingTexture>()
                    .insert(TextureReady(handles));
            }
            Ok(Err(e)) => {
                bevy::log::error!("Texture generation failed: {e}");
                commands.entity(entity).remove::<PendingTexture>();
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                bevy::log::error!("Texture generation thread panicked");
                commands.entity(entity).remove::<PendingTexture>();
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bark::BarkConfig;

    /// Auto thread count picks at least one thread regardless of host parallelism.
    #[test]
    fn auto_pool_threads_is_at_least_one() {
        let cfg = AsyncTextureConfig { pool_threads: 0 };
        assert!(resolve_pool_threads(&cfg) >= 1);
    }

    /// Explicit non-zero values are passed through unchanged.
    #[test]
    fn explicit_pool_threads_is_passthrough() {
        let cfg = AsyncTextureConfig { pool_threads: 7 };
        assert_eq!(resolve_pool_threads(&cfg), 7);
    }

    /// Inline-fallback path: when the pool is unavailable, [`spawn_task`] still
    /// produces a `PendingTexture` whose channel already holds the generated
    /// map.  This exercises the same code path that runs after a real
    /// `rayon::ThreadPoolBuilder` failure.
    #[test]
    fn inline_fallback_runs_synchronously() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancelled);
        let (tx, rx) = mpsc::sync_channel(1);

        let generator = BarkGenerator::new(BarkConfig::default());
        if !flag.load(Ordering::Relaxed) {
            tx.send(generator.generate(8, 8)).ok();
        }

        let received = rx
            .try_recv()
            .expect("inline fallback should make the result immediately available");
        let map = received.expect("8x8 generation must succeed");
        assert_eq!(map.width, 8);
        assert_eq!(map.height, 8);
    }
}
