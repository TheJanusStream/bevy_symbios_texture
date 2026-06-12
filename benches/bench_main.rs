//! Criterion benchmarks for every registered generator.
//!
//! The generator list derives from the registry via
//! [`TextureConfig::all_defaults`], so new generators are benched
//! automatically and the suite can never drift from the roster again.
//! Names stay `<module>_512` (e.g. `bark_512`), matching the historical
//! per-generator benchmark IDs.
//!
//! [`TextureConfig::generate_sync`] constructs the generator inside the
//! measured closure; construction is microseconds of noise-object setup
//! against milliseconds of pixel work, so the numbers remain comparable
//! with the old construct-once benchmarks.

use std::hint::black_box;
use std::sync::Arc;

use bevy_symbios_texture::brick::{BrickConfig, BrickGenerator};
use bevy_symbios_texture::cache::DEFAULT_MEMORY_CACHE_ENTRIES;
use bevy_symbios_texture::generator::{GeneratedHandles, TextureGenerator};
use bevy_symbios_texture::{MemoryStore, TextureCacheKey, TextureCacheStore, TextureConfig};
use criterion::{Criterion, criterion_group, criterion_main};

/// One `<module>_512` benchmark per registry row.
fn bench_all_generators(c: &mut Criterion) {
    for config in TextureConfig::all_defaults() {
        let name = format!("{}_512", config.module_name());
        c.bench_function(&name, |b| {
            b.iter(|| config.generate_sync(black_box(512), black_box(512)))
        });
    }
}

/// Cache value proposition: a cold 512² generation against a hot
/// memory-store lookup for the same key.
fn bench_cache_cold_vs_hot(c: &mut Criterion) {
    let cfg = TextureConfig::Brick(BrickConfig::default());
    let key = TextureCacheKey {
        kind: cfg.label(),
        fingerprint: cfg.fingerprint(),
        width: 512,
        height: 512,
    };
    let mut store = MemoryStore::new(DEFAULT_MEMORY_CACHE_ENTRIES);
    store.put(
        key.clone(),
        Arc::new(GeneratedHandles {
            albedo: Default::default(),
            normal: Default::default(),
            roughness: Default::default(),
            emissive: None,
        }),
        false,
        None,
    );

    let mut group = c.benchmark_group("cache_brick_512");

    let generator = BrickGenerator::new(BrickConfig::default());
    group.bench_function("cold_generate", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
    group.bench_function("hot_lookup", |b| {
        b.iter(|| store.peek_memory_only(black_box(&key)))
    });
    group.finish();
}

criterion_group!(benches, bench_all_generators, bench_cache_cold_vs_hot);
criterion_main!(benches);
