use std::hint::black_box;

use bevy_symbios_texture::ashlar::{AshlarConfig, AshlarGenerator};
use bevy_symbios_texture::asphalt::{AsphaltConfig, AsphaltGenerator};
use bevy_symbios_texture::bark::{BarkConfig, BarkGenerator};
use bevy_symbios_texture::brick::{BrickConfig, BrickGenerator};
use bevy_symbios_texture::cobblestone::{CobblestoneConfig, CobblestoneGenerator};
use bevy_symbios_texture::concrete::{ConcreteConfig, ConcreteGenerator};
use bevy_symbios_texture::corrugated::{CorrugatedConfig, CorrugatedGenerator};
use bevy_symbios_texture::encaustic::{EncausticConfig, EncausticGenerator};
use bevy_symbios_texture::generator::TextureGenerator;
use bevy_symbios_texture::ground::{GroundConfig, GroundGenerator};
use bevy_symbios_texture::iron_grille::{IronGrilleConfig, IronGrilleGenerator};
use bevy_symbios_texture::leaf::{LeafConfig, LeafGenerator};
use bevy_symbios_texture::marble::{MarbleConfig, MarbleGenerator};
use bevy_symbios_texture::metal::{MetalConfig, MetalGenerator};
use bevy_symbios_texture::pavers::{PaversConfig, PaversGenerator};
use bevy_symbios_texture::plank::{PlankConfig, PlankGenerator};
use bevy_symbios_texture::rock::{RockConfig, RockGenerator};
use bevy_symbios_texture::shingle::{ShingleConfig, ShingleGenerator};
use bevy_symbios_texture::stained_glass::{StainedGlassConfig, StainedGlassGenerator};
use bevy_symbios_texture::stucco::{StuccoConfig, StuccoGenerator};
use bevy_symbios_texture::thatch::{ThatchConfig, ThatchGenerator};
use bevy_symbios_texture::twig::{TwigConfig, TwigGenerator};
use bevy_symbios_texture::wainscoting::{WainscotingConfig, WainscotingGenerator};
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

fn bench_brick(c: &mut Criterion) {
    let generator = BrickGenerator::new(BrickConfig::default());
    c.bench_function("brick_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_plank(c: &mut Criterion) {
    let generator = PlankGenerator::new(PlankConfig::default());
    c.bench_function("plank_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_shingle(c: &mut Criterion) {
    let generator = ShingleGenerator::new(ShingleConfig::default());
    c.bench_function("shingle_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_stucco(c: &mut Criterion) {
    let generator = StuccoGenerator::new(StuccoConfig::default());
    c.bench_function("stucco_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_concrete(c: &mut Criterion) {
    let generator = ConcreteGenerator::new(ConcreteConfig::default());
    c.bench_function("concrete_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_metal(c: &mut Criterion) {
    let generator = MetalGenerator::new(MetalConfig::default());
    c.bench_function("metal_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_pavers(c: &mut Criterion) {
    let generator = PaversGenerator::new(PaversConfig::default());
    c.bench_function("pavers_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_ashlar(c: &mut Criterion) {
    let generator = AshlarGenerator::new(AshlarConfig::default());
    c.bench_function("ashlar_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_cobblestone(c: &mut Criterion) {
    let generator = CobblestoneGenerator::new(CobblestoneConfig::default());
    c.bench_function("cobblestone_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_thatch(c: &mut Criterion) {
    let generator = ThatchGenerator::new(ThatchConfig::default());
    c.bench_function("thatch_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_marble(c: &mut Criterion) {
    let generator = MarbleGenerator::new(MarbleConfig::default());
    c.bench_function("marble_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_corrugated(c: &mut Criterion) {
    let generator = CorrugatedGenerator::new(CorrugatedConfig::default());
    c.bench_function("corrugated_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_asphalt(c: &mut Criterion) {
    let generator = AsphaltGenerator::new(AsphaltConfig::default());
    c.bench_function("asphalt_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_wainscoting(c: &mut Criterion) {
    let generator = WainscotingGenerator::new(WainscotingConfig::default());
    c.bench_function("wainscoting_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_stained_glass(c: &mut Criterion) {
    let generator = StainedGlassGenerator::new(StainedGlassConfig::default());
    c.bench_function("stained_glass_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_iron_grille(c: &mut Criterion) {
    let generator = IronGrilleGenerator::new(IronGrilleConfig::default());
    c.bench_function("iron_grille_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

fn bench_encaustic(c: &mut Criterion) {
    let generator = EncausticGenerator::new(EncausticConfig::default());
    c.bench_function("encaustic_512", |b| {
        b.iter(|| generator.generate(black_box(512), black_box(512)))
    });
}

criterion_group!(
    benches,
    bench_bark,
    bench_rock,
    bench_ground,
    bench_leaf,
    bench_twig,
    bench_brick,
    bench_plank,
    bench_shingle,
    bench_stucco,
    bench_concrete,
    bench_metal,
    bench_pavers,
    bench_ashlar,
    bench_cobblestone,
    bench_thatch,
    bench_marble,
    bench_corrugated,
    bench_asphalt,
    bench_wainscoting,
    bench_stained_glass,
    bench_iron_grille,
    bench_encaustic,
);
criterion_main!(benches);
