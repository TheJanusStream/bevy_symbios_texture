use std::hint::black_box;

use bevy_symbios_texture::bark::{BarkConfig, BarkGenerator};
use bevy_symbios_texture::generator::TextureGenerator;
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_bark(c: &mut Criterion) {
    let generator = BarkGenerator::new(BarkConfig::default());
    c.bench_function("bark_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

criterion_group!(benches, bench_bark);
criterion_main!(benches);
