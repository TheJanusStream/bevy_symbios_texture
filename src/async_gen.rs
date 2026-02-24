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

use std::sync::{OnceLock, mpsc};

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
    bark::{BarkConfig, BarkGenerator},
    generator::{
        GeneratedHandles, TextureError, TextureGenerator, TextureMap, map_to_images,
        map_to_images_card,
    },
    ground::{GroundConfig, GroundGenerator},
    leaf::{LeafConfig, LeafGenerator},
    rock::{RockConfig, RockGenerator},
    twig::{TwigConfig, TwigGenerator},
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
#[derive(Component)]
pub struct PendingTexture {
    // Wrapped in Mutex so the struct is Sync, which Bevy's Component bound requires.
    pub(crate) rx: std::sync::Mutex<mpsc::Receiver<Result<TextureMap, TextureError>>>,
    /// `true` for foliage cards (leaf, twig) that need a clamp-to-edge sampler.
    is_card: bool,
}

impl PendingTexture {
    /// Spawn a bark texture generation thread at `width × height` texels.
    pub fn bark(config: BarkConfig, width: u32, height: u32) -> Self {
        let generator = BarkGenerator::new(config);
        let (tx, rx) = mpsc::sync_channel(1);
        gen_pool().spawn(move || {
            tx.send(generator.generate(width, height)).ok();
        });
        Self {
            rx: std::sync::Mutex::new(rx),
            is_card: false,
        }
    }

    /// Spawn a rock texture generation thread at `width × height` texels.
    pub fn rock(config: RockConfig, width: u32, height: u32) -> Self {
        let generator = RockGenerator::new(config);
        let (tx, rx) = mpsc::sync_channel(1);
        gen_pool().spawn(move || {
            tx.send(generator.generate(width, height)).ok();
        });
        Self {
            rx: std::sync::Mutex::new(rx),
            is_card: false,
        }
    }

    /// Spawn a ground texture generation thread at `width × height` texels.
    pub fn ground(config: GroundConfig, width: u32, height: u32) -> Self {
        let generator = GroundGenerator::new(config);
        let (tx, rx) = mpsc::sync_channel(1);
        gen_pool().spawn(move || {
            tx.send(generator.generate(width, height)).ok();
        });
        Self {
            rx: std::sync::Mutex::new(rx),
            is_card: false,
        }
    }

    /// Spawn a leaf texture generation thread at `width × height` texels.
    ///
    /// [`poll_texture_tasks`] uploads the result with
    /// [`map_to_images_card`](crate::generator::map_to_images_card) automatically,
    /// giving a clamp-to-edge sampler suitable for foliage cards.
    pub fn leaf(config: LeafConfig, width: u32, height: u32) -> Self {
        let generator = LeafGenerator::new(config);
        let (tx, rx) = mpsc::sync_channel(1);
        gen_pool().spawn(move || {
            tx.send(generator.generate(width, height)).ok();
        });
        Self {
            rx: std::sync::Mutex::new(rx),
            is_card: true,
        }
    }

    /// Spawn a twig texture generation thread at `width × height` texels.
    ///
    /// [`poll_texture_tasks`] uploads the result with
    /// [`map_to_images_card`](crate::generator::map_to_images_card) automatically,
    /// giving a clamp-to-edge sampler suitable for foliage cards.
    pub fn twig(config: TwigConfig, width: u32, height: u32) -> Self {
        let generator = TwigGenerator::new(config);
        let (tx, rx) = mpsc::sync_channel(1);
        gen_pool().spawn(move || {
            tx.send(generator.generate(width, height)).ok();
        });
        Self {
            rx: std::sync::Mutex::new(rx),
            is_card: true,
        }
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
