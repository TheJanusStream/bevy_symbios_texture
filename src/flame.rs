//! Flame wisp sprite generator.
//!
//! A single tongue of fire: a teardrop envelope — rounded at the base,
//! tapering to a point at the tip — displaced by domain-warped fractal
//! turbulence that grows toward the tip, with a vertical colour ramp from
//! the hot core through the mid-flame to the cool tip.  Per-variant cells
//! jitter the lean angle, elongation, and turbulence phase, so an atlas
//! reads as a flickering fire when particles pick random frames.
//!
//! Pairs well with additive blending; the soft fractional alpha fades the
//! wisp out instead of cutting like the foliage cards.
//!
//! Upload with [`map_to_images_card`](crate::generator::map_to_images_card);
//! see [`crate::sprite`] for the shared atlas conventions.

use noise::Perlin;

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap},
    sprite::{CellRng, SpriteCell, SpriteSample, fbm2, lerp_color},
};

/// Cell V coordinate of the flame base (bottom of the tongue).
const BASE_V: f64 = 0.90;
/// Cell V coordinate of the flame tip.
const TIP_V: f64 = 0.06;
/// Thickness of the rounded cap below the base, in flame-axis units.
const CAP: f64 = 0.12;

/// Configures the appearance of a [`FlameGenerator`].
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlameConfig {
    /// PRNG seed for the per-cell variant jitter.
    pub seed: u32,
    /// Atlas rows; each cell bakes an independent variant (clamped to
    /// `1..=16`).
    pub variant_rows: usize,
    /// Atlas columns; see `variant_rows`.
    pub variant_cols: usize,
    /// Vertical stretch of the tongue `[1, 3]` — higher values narrow the
    /// flame for the same height, reading as a taller, lazier wisp.
    pub elongation: f64,
    /// Turbulent displacement strength `[0, 1.5]`.  Applied increasingly
    /// toward the tip, so the base stays anchored while the tip wisps.
    pub turbulence: f64,
    /// Maximum per-variant sideways lean of the tip `[0, 0.5]` (cell-width
    /// fraction; each cell draws uniformly in `±lean_jitter`).
    pub lean_jitter: f64,
    /// Radial fade exponent of the envelope `[0.5, 4]`; higher = a tighter,
    /// harder tongue.
    pub falloff: f64,
    /// Hot core colour (base of the flame) in linear RGB \[0, 1\].
    pub color_core: [f32; 3],
    /// Mid-flame colour in linear RGB \[0, 1\].
    pub color_mid: [f32; 3],
    /// Tip / fringe colour in linear RGB \[0, 1\].
    pub color_tip: [f32; 3],
    /// Normal map strength.
    pub normal_strength: f32,
}

impl Default for FlameConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            variant_rows: 2,
            variant_cols: 2,
            elongation: 1.6,
            turbulence: 0.55,
            lean_jitter: 0.25,
            falloff: 1.6,
            color_core: [1.0, 0.97, 0.78],
            color_mid: [1.0, 0.55, 0.10],
            color_tip: [0.85, 0.16, 0.02],
            normal_strength: 1.0,
        }
    }
}

/// One baked flame variant: turbulence noise + per-cell lean/stretch/phase.
struct FlameCell {
    config: FlameConfig,
    perlin: Perlin,
    /// Noise-space offset decorrelating this cell's turbulence.
    phase: (f64, f64),
    /// Signed tip lean for this variant (cell-width fraction).
    lean: f64,
    /// Per-variant elongation (config value ±15 %).
    stretch: f64,
}

impl FlameCell {
    fn new(config: &FlameConfig, cell: usize) -> Self {
        let mut rng = CellRng::new(config.seed, cell);
        let lean = rng.range(-1.0, 1.0) * config.lean_jitter.clamp(0.0, 0.5);
        let stretch = config.elongation.clamp(1.0, 3.0) * rng.range(0.85, 1.15);
        let phase = (rng.range(0.0, 64.0), rng.range(0.0, 64.0));
        Self {
            config: config.clone(),
            perlin: Perlin::new(rng.next_u32()),
            phase,
            lean,
            stretch,
        }
    }
}

impl SpriteCell for FlameCell {
    fn sample(&self, u: f64, v: f64) -> SpriteSample {
        let c = &self.config;

        // Flame-axis coordinate: 0 at the base, 1 at the tip; the rounded
        // cap extends slightly below 0.
        let s = (BASE_V - v) / (BASE_V - TIP_V);
        if !(-CAP..=1.0).contains(&s) {
            return SpriteSample {
                color: c.color_tip,
                alpha: 0.0,
                height: 0.0,
                roughness: 0.9,
            };
        }
        let s_up = s.max(0.0);

        // Tip drifts sideways quadratically — the base stays anchored.
        let centre = 0.5 + self.lean * s_up * s_up;

        // Teardrop width: widest at the base, pinching to a point at the
        // tip; elongation narrows the whole tongue.
        let width = (0.30 / self.stretch) * (1.0 - s_up).powf(0.65);

        // Turbulent sideways displacement, growing toward the tip.  The
        // noise field is sampled in cell UV so the wisps are continuous
        // along the tongue.
        let n = fbm2(
            &self.perlin,
            u * 3.0 + self.phase.0,
            v * 3.0 * self.stretch + self.phase.1,
            4,
        );
        let wobble = (n - 0.5) * 2.0 * c.turbulence.clamp(0.0, 1.5) * (0.15 + 0.85 * s_up);

        // Radial distance in flame-local units: horizontal from the
        // (displaced) centreline, vertical only inside the base cap.
        let dx = (u - centre) / width.max(1e-4) + wobble;
        let dy = (s.min(0.0)) / CAP;
        let d = (dx * dx + dy * dy).sqrt();

        let envelope = (1.0 - d).clamp(0.0, 1.0);
        let alpha = (envelope.powf(c.falloff.clamp(0.5, 4.0)) * (0.75 + 0.5 * n)).clamp(0.0, 1.0);

        // Colour ramp: hot at the anchored base core, cooling along the
        // axis and toward the fringe.
        let ramp = (s_up + (1.0 - envelope) * 0.35).clamp(0.0, 1.0);
        let color = if ramp < 0.45 {
            lerp_color(c.color_core, c.color_mid, (ramp / 0.45) as f32)
        } else {
            lerp_color(c.color_mid, c.color_tip, ((ramp - 0.45) / 0.55) as f32)
        };

        SpriteSample {
            color,
            alpha,
            height: alpha,
            roughness: 0.9,
        }
    }
}

/// Procedural flame wisp sprite generator.
///
/// See the [module documentation](self) for the visual model.
pub struct FlameGenerator {
    config: FlameConfig,
}

impl FlameGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: FlameConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for FlameGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        let c = &self.config;
        crate::sprite::generate_atlas(
            width,
            height,
            c.variant_rows,
            c.variant_cols,
            c.normal_strength,
            |cell| FlameCell::new(c, cell),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_cell() -> FlameConfig {
        FlameConfig {
            variant_rows: 1,
            variant_cols: 1,
            lean_jitter: 0.0,
            ..FlameConfig::default()
        }
    }

    #[test]
    fn generator_produces_correct_buffer_sizes() {
        let map = FlameGenerator::new(FlameConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(map.albedo.len(), 64 * 64 * 4);
        assert_eq!(map.normal.len(), 64 * 64 * 4);
        assert_eq!(map.roughness.len(), 64 * 64 * 4);
    }

    #[test]
    fn base_is_bright_with_transparent_top_corners() {
        let map = FlameGenerator::new(single_cell())
            .generate(128, 128)
            .expect("generate failed");
        // Near the anchored base centre (v ≈ 0.82) the tongue is solid.
        let base = ((0.82 * 128.0) as usize * 128 + 64) * 4;
        assert!(map.albedo[base + 3] > 120, "flame base should be visible");
        // Top corners stay clear.
        for x in [2usize, 125] {
            let idx = (2 * 128 + x) * 4;
            assert_eq!(map.albedo[idx + 3], 0, "top corner ({x},2)");
        }
    }

    #[test]
    fn variants_differ() {
        let map = FlameGenerator::new(FlameConfig::default())
            .generate(128, 128)
            .expect("generate failed");
        let differs = (0..64usize).any(|y| {
            (0..64usize).any(|x| {
                let a = ((y * 128) + x) * 4;
                let b = ((y * 128) + x + 64) * 4;
                map.albedo[a..a + 4] != map.albedo[b..b + 4]
            })
        });
        assert!(differs, "atlas cells should bake distinct variants");
    }

    #[test]
    fn deterministic_for_same_seed() {
        let a = FlameGenerator::new(FlameConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        let b = FlameGenerator::new(FlameConfig::default())
            .generate(64, 64)
            .expect("generate failed");
        assert_eq!(a.albedo, b.albedo);
    }
}
