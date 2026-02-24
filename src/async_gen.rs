//! Async texture generation system.
//!
//! Offloads the CPU-intensive pixel math to Bevy's `AsyncComputeTaskPool` so
//! it does not stall the main thread.  When the task finishes the images are
//! uploaded to [`Assets<Image>`] and the result entity receives the
//! [`TextureReady`] component.
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

use bevy::{
    asset::Assets,
    ecs::{
        component::Component,
        entity::Entity,
        system::{Commands, Query, ResMut},
    },
    image::Image,
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
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

/// Spawned onto an entity to request async texture generation.
#[derive(Component)]
pub struct PendingTexture {
    pub(crate) task: Task<Result<TextureMap, TextureError>>,
    /// `true` for foliage cards (leaf, twig) that need a clamp-to-edge sampler.
    is_card: bool,
}

impl PendingTexture {
    /// Spawn an async bark texture generation task at `width × height` texels.
    pub fn bark(config: BarkConfig, width: u32, height: u32) -> Self {
        let generator = BarkGenerator::new(config);
        let task =
            AsyncComputeTaskPool::get().spawn(async move { generator.generate(width, height) });
        Self {
            task,
            is_card: false,
        }
    }

    /// Spawn an async rock texture generation task at `width × height` texels.
    pub fn rock(config: RockConfig, width: u32, height: u32) -> Self {
        let generator = RockGenerator::new(config);
        let task =
            AsyncComputeTaskPool::get().spawn(async move { generator.generate(width, height) });
        Self {
            task,
            is_card: false,
        }
    }

    /// Spawn an async ground texture generation task at `width × height` texels.
    pub fn ground(config: GroundConfig, width: u32, height: u32) -> Self {
        let generator = GroundGenerator::new(config);
        let task =
            AsyncComputeTaskPool::get().spawn(async move { generator.generate(width, height) });
        Self {
            task,
            is_card: false,
        }
    }

    /// Spawn an async leaf texture generation task at `width × height` texels.
    ///
    /// [`poll_texture_tasks`] uploads the result with
    /// [`map_to_images_card`](crate::generator::map_to_images_card) automatically,
    /// giving a clamp-to-edge sampler suitable for foliage cards.
    pub fn leaf(config: LeafConfig, width: u32, height: u32) -> Self {
        let generator = LeafGenerator::new(config);
        let task =
            AsyncComputeTaskPool::get().spawn(async move { generator.generate(width, height) });
        Self {
            task,
            is_card: true,
        }
    }

    /// Spawn an async twig texture generation task at `width × height` texels.
    ///
    /// [`poll_texture_tasks`] uploads the result with
    /// [`map_to_images_card`](crate::generator::map_to_images_card) automatically,
    /// giving a clamp-to-edge sampler suitable for foliage cards.
    pub fn twig(config: TwigConfig, width: u32, height: u32) -> Self {
        let generator = TwigGenerator::new(config);
        let task =
            AsyncComputeTaskPool::get().spawn(async move { generator.generate(width, height) });
        Self {
            task,
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
    mut tasks: Query<(Entity, &mut PendingTexture)>,
    mut images: ResMut<Assets<Image>>,
) {
    for (entity, mut pending) in &mut tasks {
        if let Some(result) = block_on(future::poll_once(&mut pending.task)) {
            let is_card = pending.is_card;
            match result {
                Ok(map) => {
                    let handles = if is_card {
                        map_to_images_card(map, &mut images)
                    } else {
                        map_to_images(map, &mut images)
                    };
                    commands
                        .entity(entity)
                        .remove::<PendingTexture>()
                        .insert(TextureReady(handles));
                }
                Err(e) => {
                    bevy::log::error!("Texture generation failed: {e}");
                    commands.entity(entity).remove::<PendingTexture>();
                }
            }
        }
    }
}
