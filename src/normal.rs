//! Convert a greyscale heightmap into a tangent-space normal map.
//!
//! Uses central differences (Sobel-like) to estimate the surface gradient at
//! each texel, then encodes the result as RGBA8 with:
//!   R = X  (tangent)
//!   G = Y  (bitangent)
//!   B = Z  (surface normal, always points outward)
//!   A = 255
//!
//! The encoding follows Bevy's convention: values are remapped from [-1,1]
//! to \[0, 255\] via `((n + 1.0) * 0.5 * 255.0) as u8`.

/// Convert a slice of normalised height values `[0, 1]` into a tangent-space
/// normal map encoded as RGBA8.
///
/// `strength` scales the gradient — larger values produce more pronounced normals.
pub fn height_to_normal(heights: &[f64], width: u32, height: u32, strength: f32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let s = strength as f64;

    let mut out = vec![0u8; w * h * 4];

    for y in 0..h {
        for x in 0..w {
            // Wrap-around neighbours for seamless tiling.
            let xm = (x + w - 1) % w;
            let xp = (x + 1) % w;
            let ym = (y + h - 1) % h;
            let yp = (y + 1) % h;

            let left = heights[y * w + xm]; // x - 1
            let right = heights[y * w + xp]; // x + 1
            let above = heights[ym * w + x]; // y - 1
            let below = heights[yp * w + x]; // y + 1

            // Central difference gradient.
            let dx = (right - left) * s;
            let dy = (below - above) * s;

            // Normal = normalize(-dx, -dy, 1).
            let len = (dx * dx + dy * dy + 1.0).sqrt();
            let nx = -dx / len;
            let ny = -dy / len;
            let nz = 1.0 / len;

            let idx = (y * w + x) * 4;
            out[idx] = encode_normal(nx);
            out[idx + 1] = encode_normal(ny);
            out[idx + 2] = encode_normal(nz);
            out[idx + 3] = 255;
        }
    }

    out
}

#[inline]
fn encode_normal(n: f64) -> u8 {
    ((n * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8
}
