//! Convert a greyscale heightmap into a tangent-space normal map.
//!
//! Uses central differences (Sobel-like) to estimate the surface gradient at
//! each texel, then encodes the result as RGBA8 with:
//!   R = X  (tangent)
//!   G = Y  (bitangent, points up / toward –V in Bevy's OpenGL convention)
//!   B = Z  (surface normal, always points outward)
//!   A = 255
//!
//! The encoding follows Bevy's convention: values are remapped from [-1,1]
//! to \[0, 255\] via `((n + 1.0) * 0.5 * 255.0) as u8`.

/// How to handle pixel neighbours at the texture boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoundaryMode {
    /// Wrap indices toroidally — correct for tileable textures (bark, rock, ground).
    Wrap,
    /// Clamp indices to the edge — correct for foliage cards (leaf, twig) that
    /// must not bleed normals across the transparent border.
    Clamp,
}

/// Convert a slice of normalised height values `[0, 1]` into a tangent-space
/// normal map encoded as RGBA8.
///
/// `strength` scales the gradient — larger values produce more pronounced
/// normals.  The gradient is divided by the pixel spacing in UV space
/// (`2 / width` and `2 / height` for central differences), so the output is
/// resolution-independent: the same `strength` value produces identical
/// surface steepness at any resolution.
///
/// `boundary` controls how neighbours are fetched at the texture edges.  Use
/// [`BoundaryMode::Wrap`] for tileable textures and [`BoundaryMode::Clamp`]
/// for foliage cards.
pub fn height_to_normal(
    heights: &[f64],
    width: u32,
    height: u32,
    strength: f32,
    boundary: BoundaryMode,
) -> Vec<u8> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let w = width as usize;
    let h = height as usize;
    let s = strength as f64;

    let mut out = vec![0u8; w * h * 4];

    for y in 0..h {
        for x in 0..w {
            let (xm, xp, ym, yp) = match boundary {
                BoundaryMode::Wrap => ((x + w - 1) % w, (x + 1) % w, (y + h - 1) % h, (y + 1) % h),
                BoundaryMode::Clamp => (
                    x.saturating_sub(1),
                    (x + 1).min(w - 1),
                    y.saturating_sub(1),
                    (y + 1).min(h - 1),
                ),
            };

            let left = heights[y * w + xm];
            let right = heights[y * w + xp];
            let above = heights[ym * w + x];
            let below = heights[yp * w + x];

            // True derivative: divide height difference by pixel spacing in UV
            // space (2/width for central differences).  Equivalent to
            // multiplying by width/2.  This makes gradient magnitude
            // resolution-independent.
            let dx = (right - left) * s * w as f64 * 0.5;
            let dy = (below - above) * s * h as f64 * 0.5;

            // Normal = normalize(-dx, dy, 1) in Bevy / OpenGL tangent space.
            //
            // X: negate because a rightward slope tilts the normal leftward.
            // Y: no negation — Bevy's bitangent points toward –V (up in image),
            //    so a positive dH/dV (below > above) yields a positive Y
            //    component (normal tilts toward the top of the texture).
            let len = (dx * dx + dy * dy + 1.0).sqrt();
            let nx = -dx / len;
            let ny = dy / len;
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
