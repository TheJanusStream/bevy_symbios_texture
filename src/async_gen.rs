//! Async texture generation system.
//!
//! Offloads the CPU-intensive pixel math to Bevy's `AsyncComputeTaskPool` so
//! it does not stall the main thread.  When the task finishes the images are
//! uploaded to [`Assets<Image>`] and the result entity receives the
//! [`TextureReady`] component.
//!
//! # Usage
//! ```rust,ignore
//! commands.spawn(PendingTexture::bark(BarkConfig::default(), 512, 512));
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
    generator::{GeneratedHandles, TextureError, TextureGenerator, TextureMap, map_to_images},
    ground::{GroundConfig, GroundGenerator},
    rock::{RockConfig, RockGenerator},
};

/// Spawned onto an entity to request async texture generation.
#[derive(Component)]
pub struct PendingTexture(pub(crate) Task<Result<TextureMap, TextureError>>);

impl PendingTexture {
    pub fn bark(config: BarkConfig, width: u32, height: u32) -> Self {
        let generator = BarkGenerator::new(config);
        let task =
            AsyncComputeTaskPool::get().spawn(async move { generator.generate(width, height) });
        Self(task)
    }

    pub fn rock(config: RockConfig, width: u32, height: u32) -> Self {
        let generator = RockGenerator::new(config);
        let task =
            AsyncComputeTaskPool::get().spawn(async move { generator.generate(width, height) });
        Self(task)
    }

    pub fn ground(config: GroundConfig, width: u32, height: u32) -> Self {
        let generator = GroundGenerator::new(config);
        let task =
            AsyncComputeTaskPool::get().spawn(async move { generator.generate(width, height) });
        Self(task)
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
        if let Some(result) = block_on(future::poll_once(&mut pending.0)) {
            match result {
                Ok(map) => {
                    let handles = map_to_images(map, &mut images);
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
