//! Thatch texture generator — dense fibrous roofing material.
//!
//! The algorithm:
//! 1. Build two toroidal FBMs: a high-frequency fibre noise (along U) and a
//!    low-frequency layer-variation noise (along V).  A third low-frequency
//!    warp noise distorts the UV coordinates laterally before sampling, giving
//!    the organic wiggly appearance of real straw bundles.
//! 2. Combine fibre and layer noise into a scalar `fiber_t` value \[0, 1\].
//! 3. Overlay a repeating sawtooth in V with `layer_count` periods; the bottom
//!    of each period is darkened by `layer_shadow` to simulate the shadow cast
//!    by the bundle tip above.
//! 4. Lerp between `color_shadow` and `color_straw` using the combined signal.
//! 5. Height = fiber_t × (1 – shadow gradient) for a convincing normal map.

use noise::{Fbm, MultiFractal, Perlin};

use crate::{
    generator::{TextureError, TextureGenerator, TextureMap, linear_to_srgb, validate_dimensions},
    noise::{ToroidalNoise, normalize, sample_grid},
    normal::{BoundaryMode, height_to_normal},
};

/// Configures the appearance of a [`ThatchGenerator`].
///
/// Thatch is modelled as densely-packed straw bundles laid in overlapping
/// horizontal layers, like shingles.  The high U-frequency noise creates
/// individual fibre streaks while the V-frequency sawtooth creates the layered
/// overlap shadow pattern.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ThatchConfig {
    pub seed: u32,
    /// Fibre density — noise frequency along U (controls how many fibres are
    /// visible across the tile) \[4, 24\].
    pub density: f64,
    /// Anisotropy ratio: the V frequency is `density / anisotropy`, making
    /// fibres appear long and horizontal \[4, 16\].
    pub anisotropy: f64,
    /// Lateral domain-warp strength — how much the fibres wiggle \[0, 0.5\].
    pub warp_strength: f64,
    /// Number of straw-bundle overlap layers visible across the V axis \[4, 16\].
    pub layer_count: f64,
    /// Layer shadow depth — how much darker the bottom of each bundle layer is
    /// \[0, 1\].
    pub layer_shadow: f64,
    /// Base (dry straw) colour in linear RGB \[0, 1\].
    pub color_straw: [f32; 3],
    /// Shadow / rot colour at the bottom of each bundle \[0, 1\].
    pub color_shadow: [f32; 3],
    /// Normal-map strength.
    pub normal_strength: f32,
}

impl Default for ThatchConfig {
    fn default() -> Self {
        Self {
            seed: 19,
            density: 12.0,
            anisotropy: 8.0,
            warp_strength: 0.15,
            layer_count: 8.0,
            layer_shadow: 0.55,
            color_straw: [0.62, 0.54, 0.28],
            color_shadow: [0.22, 0.17, 0.09],
            normal_strength: 3.5,
        }
    }
}

/// Procedural thatch texture generator.
pub struct ThatchGenerator {
    config: ThatchConfig,
}

impl ThatchGenerator {
    pub fn new(config: ThatchConfig) -> Self {
        Self { config }
    }
}

impl TextureGenerator for ThatchGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<TextureMap, TextureError> {
        validate_dimensions(width, height)?;
        let c = &self.config;

        // ── Noise layers ─────────────────────────────────────────────────────
        //
        // Three toroidal FBMs:
        //   warp  – low-frequency lateral domain warp (bends the fibres).
        //   fibre – high U-frequency for individual straw streaks.
        //   layer – low V-frequency for broad bundle-layer variation.
        //
        // We achieve anisotropy by using two separate toroidal noise objects
        // whose frequencies are matched to the desired U and V wavelengths.
        // `fibre_noise` uses `density` for a high horizontal frequency.
        // `layer_noise` uses `density / anisotropy` for a low vertical one.

        let warp_freq = (c.density * 0.3).max(0.5);
        let fbm_warp: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(7)).set_octaves(3);
        let warp_noise = ToroidalNoise::new(fbm_warp, warp_freq);
        let warp_grid = sample_grid(&warp_noise, width, height);

        let fbm_fibre: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(50)).set_octaves(5);
        let fibre_noise = ToroidalNoise::new(fbm_fibre, c.density);
        let fibre_grid = sample_grid(&fibre_noise, width, height);

        let layer_freq = (c.density / c.anisotropy.max(1.0)).max(0.5);
        let fbm_layer: Fbm<Perlin> = Fbm::new(c.seed.wrapping_add(150)).set_octaves(3);
        let layer_noise = ToroidalNoise::new(fbm_layer, layer_freq);
        let layer_grid = sample_grid(&layer_noise, width, height);

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        let mut heights = vec![0.0f64; n];
        let mut albedo = vec![0u8; n * 4];
        let mut roughness_buf = vec![0u8; n * 4];

        for y in 0..h {
            let v = y as f64 / h as f64;

            // Sawtooth layer pattern in V: `layer_v` goes from 0 (bottom of
            // the exposed bundle tip) to 1 (top, hidden under next layer).
            // `layer_count` must be an integer for the pattern to tile; round
            // to the nearest integer.
            let layer_count = c.layer_count.round().max(1.0);
            let layer_v = (v * layer_count).fract(); // [0, 1) per layer

            // Shadow gradient: dark at the bottom (layer_v ≈ 0, the exposed
            // tip hanging down), bright at the top.
            let shadow_t = (1.0 - layer_v).powf(1.5);

            for x in 0..w {
                let u = x as f64 / w as f64;

                let idx = y * w + x;

                // Domain warp: offset U before sampling the fibre noise.
                // `warp_grid` is in [-1, 1]; scale by warp_strength to get
                // a small UV displacement that bends the fibres laterally.
                let warp = warp_grid[idx] * c.warp_strength;

                // Fibre noise: use the warped U index.  Since we only have a
                // precomputed grid at regular (u, v) positions, approximate the
                // warped sample by remapping to the nearest warped pixel column.
                // A true warp would require per-pixel noise evaluation; instead
                // we index the precomputed grid with a clamped warped offset so
                // the texture tiles correctly.
                let warped_x = {
                    let ux = (u + warp).rem_euclid(1.0);
                    (ux * w as f64) as usize % w
                };
                let warped_idx = y * w + warped_x;
                let fibre_raw = normalize(fibre_grid[warped_idx]);

                // Layer variation: broad V-direction modulation.
                let layer_raw = normalize(layer_grid[idx]);

                // Combined fibre signal: weight fibre detail strongly, layer
                // variation adds subtle brightness bands along V.
                let fiber_t = (0.65 * fibre_raw + 0.35 * layer_raw).clamp(0.0, 1.0);

                // Height combines the fiber intensity and the sawtooth ramp
                // so that each bundle layer rises from bottom to top.
                let h_val = (fiber_t * (0.5 + 0.5 * layer_v) - shadow_t * c.layer_shadow * 0.3)
                    .clamp(0.0, 1.0);
                heights[idx] = h_val;

                // Colour: lerp from shadow colour (dark, bottom of layer) to
                // straw colour (bright, top of layer / fibre highlight).
                let brightness = (fiber_t * (1.0 - shadow_t * c.layer_shadow)).clamp(0.0, 1.0);
                let r = lerp(c.color_shadow[0], c.color_straw[0], brightness as f32);
                let g = lerp(c.color_shadow[1], c.color_straw[1], brightness as f32);
                let b = lerp(c.color_shadow[2], c.color_straw[2], brightness as f32);

                let ai = idx * 4;
                albedo[ai] = linear_to_srgb(r);
                albedo[ai + 1] = linear_to_srgb(g);
                albedo[ai + 2] = linear_to_srgb(b);
                albedo[ai + 3] = 255;

                // ORM: thatch is rough throughout; slightly less rough on the
                // bright fibre highlights, more rough in shadow areas.
                let rough_val =
                    (0.80 - fiber_t as f32 * 0.15 + shadow_t as f32 * 0.10).clamp(0.65, 0.95);
                roughness_buf[ai] = 255;
                roughness_buf[ai + 1] = (rough_val * 255.0).round() as u8;
                roughness_buf[ai + 2] = 0;
                roughness_buf[ai + 3] = 255;
            }
        }

        let normal = height_to_normal(
            &heights,
            width,
            height,
            c.normal_strength,
            BoundaryMode::Wrap,
        );

        Ok(TextureMap {
            albedo,
            normal,
            roughness: roughness_buf,
            width,
            height,
        })
    }
}

// --- helpers ----------------------------------------------------------------

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}
