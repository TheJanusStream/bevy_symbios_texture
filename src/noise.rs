//! Toroidal 4D noise mapping for seamless texture tiling.
//!
//! Maps 2D UV coordinates (in \[0, 1\]) to a 4D point on a torus so that noise
//! sampled there wraps perfectly at all four edges with no seam.
//!
//! The mapping is:
//!   nx = cos(2π·u) · frequency
//!   ny = sin(2π·u) · frequency
//!   nz = cos(2π·v) · frequency
//!   nw = sin(2π·v) · frequency
//!
//! `frequency` is the torus radius in noise-space. Larger values push the
//! sample point further from the origin, crossing more noise-lattice cells and
//! producing higher-frequency / more-detailed patterns.
//!
//! Seam-freedom is guaranteed because cos(0)=cos(2π) and sin(0)=sin(2π), so
//! u=0 and u=1 always resolve to the identical 4D coordinate.

use noise::NoiseFn;
use std::f64::consts::TAU;

/// Wraps any 4-dimensional noise function and samples it on a torus, producing
/// output that tiles seamlessly when `u` and `v` are each in `[0, 1]`.
pub struct ToroidalNoise<N> {
    noise: N,
    /// Torus radius in noise-space.  Larger → more detail per texture tile.
    pub frequency: f64,
}

impl<N: NoiseFn<f64, 4>> ToroidalNoise<N> {
    pub fn new(noise: N, frequency: f64) -> Self {
        Self { noise, frequency }
    }

    /// Sample the noise at normalised UV coordinates in [0, 1].
    ///
    /// Both `u` and `v` wrap continuously; there is no seam.
    pub fn get(&self, u: f64, v: f64) -> f64 {
        // Radius = frequency: as u/v sweep [0,1] the 4D point traces a circle
        // of this radius through noise space, giving arc-length = 2π·frequency.
        // With Perlin lattice cells of size 1, a radius of ~1 gives ~6 cells of
        // variation; radius 4 gives ~25.
        let nx = (TAU * u).cos() * self.frequency;
        let ny = (TAU * u).sin() * self.frequency;
        let nz = (TAU * v).cos() * self.frequency;
        let nw = (TAU * v).sin() * self.frequency;
        self.noise.get([nx, ny, nz, nw])
    }

    /// Sample at an offset UV — useful when building domain-warp chains.
    pub fn get_offset(&self, u: f64, v: f64, du: f64, dv: f64) -> f64 {
        self.get(u + du, v + dv)
    }

    /// Sample the noise at pre-projected 4D torus coordinates.
    ///
    /// Use this with lookup tables (see [`sample_grid`]) to avoid recomputing
    /// `sin`/`cos` for every sample in a regular grid.
    #[inline]
    pub fn get_precomputed(&self, nx: f64, ny: f64, nz: f64, nw: f64) -> f64 {
        self.noise.get([nx, ny, nz, nw])
    }
}

/// Convenience: iterate over a `width × height` grid and collect samples.
///
/// Returns a `Vec<f64>` of length `width * height`, values in `[-1, 1]`.
///
/// Torus coordinates are precomputed into lookup tables of size `W` and `H`,
/// reducing trigonometric calls from `O(W × H)` to `O(W + H)`.
pub fn sample_grid<N: NoiseFn<f64, 4>>(
    noise: &ToroidalNoise<N>,
    width: u32,
    height: u32,
) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let freq = noise.frequency;

    // One sin/cos pair per column and per row instead of per pixel.
    let col_cos: Vec<f64> = (0..w)
        .map(|x| (TAU * x as f64 / w as f64).cos() * freq)
        .collect();
    let col_sin: Vec<f64> = (0..w)
        .map(|x| (TAU * x as f64 / w as f64).sin() * freq)
        .collect();
    let row_cos: Vec<f64> = (0..h)
        .map(|y| (TAU * y as f64 / h as f64).cos() * freq)
        .collect();
    let row_sin: Vec<f64> = (0..h)
        .map(|y| (TAU * y as f64 / h as f64).sin() * freq)
        .collect();

    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let nz = row_cos[y];
        let nw = row_sin[y];
        for x in 0..w {
            out.push(noise.get_precomputed(col_cos[x], col_sin[x], nz, nw));
        }
    }
    out
}

/// Map a raw noise sample from `[-1, 1]` to `[0, 1]`.
#[inline]
pub fn normalize(v: f64) -> f64 {
    v * 0.5 + 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use noise::Perlin;

    /// Verify that the sampler actually varies across the texture.
    /// With the (broken) inverted formula the stddev was < 0.001; correct
    /// formula gives > 0.1 for frequency=4.
    #[test]
    fn samples_vary_with_frequency() {
        let noise = ToroidalNoise::new(Perlin::new(1), 4.0);
        let samples = sample_grid(&noise, 64, 64);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let stddev = variance.sqrt();
        assert!(
            stddev > 0.1,
            "noise has almost no variation (stddev={stddev:.4}); torus radius is likely wrong"
        );
    }

    /// Verify left/right and top/bottom edges match (seamless tiling).
    #[test]
    fn tiles_seamlessly() {
        let noise = ToroidalNoise::new(Perlin::new(42), 3.0);
        // u=0 and u=1 should give the same value for any v
        for v in [0.0, 0.25, 0.5, 0.75] {
            let at_0 = noise.get(0.0, v);
            let at_1 = noise.get(1.0, v);
            assert!(
                (at_0 - at_1).abs() < 1e-10,
                "horizontal seam at v={v}: {at_0} != {at_1}"
            );
        }
        // v=0 and v=1 should give the same value for any u
        for u in [0.0, 0.25, 0.5, 0.75] {
            let at_0 = noise.get(u, 0.0);
            let at_1 = noise.get(u, 1.0);
            assert!(
                (at_0 - at_1).abs() < 1e-10,
                "vertical seam at u={u}: {at_0} != {at_1}"
            );
        }
    }
}
