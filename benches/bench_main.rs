use std::hint::black_box;

use bevy_symbios_texture::bark::{BarkConfig, BarkGenerator};
use bevy_symbios_texture::generator::TextureGenerator;
use bevy_symbios_texture::ground::{GroundConfig, GroundGenerator};
use bevy_symbios_texture::leaf::{LeafConfig, LeafGenerator};
use bevy_symbios_texture::rock::{RockConfig, RockGenerator};
use bevy_symbios_texture::twig::{TwigConfig, TwigGenerator};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_bark(c: &mut Criterion) {
    let generator = BarkGenerator::new(BarkConfig::default());
    c.bench_function("bark_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_rock(c: &mut Criterion) {
    let generator = RockGenerator::new(RockConfig::default());
    c.bench_function("rock_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_ground(c: &mut Criterion) {
    let generator = GroundGenerator::new(GroundConfig::default());
    c.bench_function("ground_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_leaf(c: &mut Criterion) {
    let generator = LeafGenerator::new(LeafConfig::default());
    c.bench_function("leaf_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_twig(c: &mut Criterion) {
    let generator = TwigGenerator::new(TwigConfig::default());
    c.bench_function("twig_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

criterion_group!(
    benches,
    bench_bark,
    bench_rock,
    bench_ground,
    bench_leaf,
    bench_twig
);
criterion_main!(benches);
