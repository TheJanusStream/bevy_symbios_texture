//! Async texture generation system.
//!
//! Offloads the CPU-intensive pixel math to a private, bounded [`rayon`]
//! thread pool so it does not stall the main thread.  The pool is limited to
//! [`MAX_GENERATION_THREADS`] concurrent tasks; excess requests are queued and
//! run in order rather than spawning unbounded OS threads.  When a task
//! finishes the images are uploaded to [`Assets<Image>`] and the result entity
//! receives the [`TextureReady`] component.
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

/// Maximum number of texture generation tasks that run concurrently.
///
/// Additional tasks are queued inside the rayon pool rather than spawning new
/// OS threads, bounding both CPU and memory usage.
const MAX_GENERATION_THREADS: usize = 4;

/// Returns the library-private rayon thread pool used for texture generation.
///
/// Isolated from the application's global rayon pool so texture work does not
/// starve unrelated parallel workloads and the concurrency cap is enforced
/// regardless of the calling application's rayon configuration.
fn gen_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(MAX_GENERATION_THREADS)
            .thread_name(|i| format!("texture-gen-{i}"))
            .build()
            .expect("failed to build texture generation thread pool")
    })
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
/// Each constructor submits `generate()` to the private [`gen_pool`] rayon
/// pool (capped at [`MAX_GENERATION_THREADS`] concurrent tasks).  Because
/// `generate()` is a monolithic blocking loop with no yield points, using
/// Bevy's `AsyncComputeTaskPool` would starve other tasks on that executor;
/// a dedicated pool avoids the problem while bounding OS thread and memory
/// usage.  [`poll_texture_tasks`] non-blockingly checks for completion each
/// frame using [`mpsc::Receiver::try_recv`].
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

impl Drop for PendingTexture {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

/// Shared constructor body: creates the channel + cancellation flag, spawns the
/// task, and returns a `PendingTexture`.  The closure `f` is the generator call.
/// Native Desktop: Spawn using our private, bounded Rayon pool.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_task<F>(f: F, is_card: bool) -> PendingTexture
where
    F: FnOnce() -> Result<TextureMap, TextureError> + Send + 'static,
{
    let cancelled = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancelled);
    let (tx, rx) = mpsc::sync_channel(1);

    gen_pool().spawn(move || {
        if !flag.load(Ordering::Relaxed) {
            tx.send(f()).ok();
        }
    });

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
    /// [`map_to_images_card`](crate::generator::map_to_images_card) automatically,
    /// giving a clamp-to-edge sampler suitable for foliage cards.
    pub fn leaf(config: LeafConfig, width: u32, height: u32) -> Self {
        let generator = LeafGenerator::new(config);
        spawn_task(move || generator.generate(width, height), true)
    }

    /// Spawn a twig texture generation thread at `width × height` texels.
    ///
    /// [`poll_texture_tasks`] uploads the result with
    /// [`map_to_images_card`](crate::generator::map_to_images_card) automatically,
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
    /// [`map_to_images_card`](crate::generator::map_to_images_card) automatically,
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
    /// [`map_to_images_card`](crate::generator::map_to_images_card) automatically,
    /// giving a clamp-to-edge sampler suitable for alpha-masked cards.
    pub fn stained_glass(config: StainedGlassConfig, width: u32, height: u32) -> Self {
        let generator = StainedGlassGenerator::new(config);
        spawn_task(move || generator.generate(width, height), true)
    }

    /// Spawn an iron grille / portcullis texture generation thread at `width × height` texels.
    ///
    /// [`poll_texture_tasks`] uploads the result with
    /// [`map_to_images_card`](crate::generator::map_to_images_card) automatically,
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
pub fn poll_texture_tasks(
    mut commands: Commands,
    tasks: Query<(Entity, &PendingTexture)>,
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
