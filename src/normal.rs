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

            // True derivative: divide height difference by the actual sample
            // spacing in UV space.  Wrap mode always uses a central difference
            // (spacing = 2/width), so x_dist is always 2.  Clamp mode uses a
            // one-sided difference at the edges (spacing = 1/width, x_dist 1)
            // and a central difference in the interior (x_dist 2).
            // NOTE: under Wrap, xm may be larger than xp (e.g. xm=w-1, xp=1),
            // so we must not compute `xp - xm` directly — that would underflow.
            // NOTE: under Clamp with w == 1, xm == xp == 0, so the difference
            // would be 0.  Guard with .max(1) — the numerator (right - left) is
            // also 0 in that case (same pixel both sides), so dx == 0 (flat normal).
            let (x_dist, y_dist) = match boundary {
                BoundaryMode::Wrap => (2.0f64, 2.0f64),
                BoundaryMode::Clamp => ((xp - xm).max(1) as f64, (yp - ym).max(1) as f64),
            };
            let dx = (right - left) * s * w as f64 / x_dist;
            let dy = (below - above) * s * h as f64 / y_dist;

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

/// Dilate opaque-pixel heights one step into the transparent border.
///
/// For each transparent pixel (`albedo[idx*4 + 3] == 0`) that has at least
/// one opaque 4-connected neighbour, replace its height with the average of
/// those neighbours.  Pixels that remain fully surrounded by transparency
/// keep their original value (0.5 neutral flat).
///
/// Call this **before** [`height_to_normal`] on foliage cards so the central-
/// difference kernel does not see an artificial height cliff at the silhouette
/// edge.  One dilation pass is sufficient because the kernel only samples one
/// pixel away.
pub(crate) fn dilate_heights(heights: &mut [f64], albedo: &[u8], w: usize, h: usize) {
    let mut dilated = heights.to_vec();
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if albedo[idx * 4 + 3] != 0 {
                continue; // opaque — leave unchanged
            }
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for (dy, dx) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let ny = y as i32 + dy;
                let nx = x as i32 + dx;
                if ny < 0 || ny >= h as i32 || nx < 0 || nx >= w as i32 {
                    continue;
                }
                let nidx = ny as usize * w + nx as usize;
                if albedo[nidx * 4 + 3] != 0 {
                    sum += heights[nidx];
                    count += 1;
                }
            }
            if count > 0 {
                dilated[idx] = sum / count as f64;
            }
        }
    }
    heights.copy_from_slice(&dilated);
}

#[inline]
fn encode_normal(n: f64) -> u8 {
    ((n * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8
}
