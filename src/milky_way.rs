//! The cinematic look's Milky Way (look.rs), baked once at startup from the
//! star panorama the renderer already loads.
//!
//! The panorama is 8-bit BC1 with its stars baked in: brightened in the
//! shader it showed 4x4 blocks, 565 colour noise and blurred duplicates of
//! the catalogue stars. Here a ~0.09 degree mip is decoded, its point stars
//! (~1 texel there) are removed from the luminance by a 3 x 3 morphological
//! opening (min then max), and the result is blurred. The map's colour is not
//! used: BC1's 565 endpoints quantise green finer than red and blue, which
//! turned dark regions green and magenta. The integrated starlight is tinted
//! instead: bluish in the faint arms, warm (G/K giants) in the bright band.
//! The brightness is normalised from the data (area-weighted percentiles):
//! the sky's median glow is the black point, so only the band shows, and
//! its 99.5th percentile lands at `BAND_DISPLAY` through a gentle contrast
//! curve. Without the black point and at 0.32 the broad galactic glow
//! turned the whole background into grey fog. The output is display-
//! referred sRGB, equirectangular like the panorama (same UVs).

use crate::star_panorama::StarPanoramaMip;

/// Linear display value of the bright band (99.5th percentile).
const BAND_DISPLAY: f32 = 0.05;
/// Percentile of the sky's glow taken as black.
const BLACK_PERCENTILE: f32 = 0.5;
/// Contrast of the band above the black point.
const CONTRAST: f32 = 1.6;
/// Tints of the faint arms and the bright band (unit luminance).
const FAINT_TINT: [f32; 3] = [0.90, 0.99, 1.16];
const BRIGHT_TINT: [f32; 3] = [1.10, 0.98, 0.84];
const MAX_WIDTH: u32 = 4096;

/// BGRA8 sRGB texels, width, height.
pub struct MilkyWay {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

fn srgb_to_linear(value: u8) -> f32 {
    let c = f32::from(value) / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(value: f32) -> u8 {
    let c = value.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// Decode one BC1 level to linear RGB.
fn decode_bc1(bytes: &[u8], width: u32, height: u32) -> Option<Vec<[f32; 3]>> {
    let (bw, bh) = (width.div_ceil(4) as usize, height.div_ceil(4) as usize);
    if bytes.len() < bw * bh * 8 {
        return None;
    }
    let lut: Vec<f32> = (0..=255).map(srgb_to_linear).collect();
    let (width, height) = (width as usize, height as usize);
    let mut out = vec![[0.0; 3]; width * height];
    let rgb565 = |c: u16| {
        let r = ((c >> 11) & 31) as u32 * 255 / 31;
        let g = ((c >> 5) & 63) as u32 * 255 / 63;
        let b = (c & 31) as u32 * 255 / 31;
        [r as f32, g as f32, b as f32]
    };
    for by in 0..bh {
        for bx in 0..bw {
            let block = &bytes[(by * bw + bx) * 8..][..8];
            let c0 = u16::from_le_bytes([block[0], block[1]]);
            let c1 = u16::from_le_bytes([block[2], block[3]]);
            let (p0, p1) = (rgb565(c0), rgb565(c1));
            let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
            let palette: [[f32; 3]; 4] = if c0 > c1 {
                [p0, p1, [0, 1, 2].map(|i| mix(p0[i], p1[i], 1.0 / 3.0)), [0, 1, 2].map(|i| mix(p0[i], p1[i], 2.0 / 3.0))]
            } else {
                [p0, p1, [0, 1, 2].map(|i| mix(p0[i], p1[i], 0.5)), [0.0; 3]]
            };
            let indices = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
            for py in 0..4 {
                for px in 0..4 {
                    let (x, y) = (bx * 4 + px, by * 4 + py);
                    if x >= width || y >= height {
                        continue;
                    }
                    let index = (indices >> (2 * (py * 4 + px))) & 3;
                    let colour = palette[index as usize];
                    out[y * width + x] = colour.map(|c| lut[(c + 0.5) as usize]);
                }
            }
        }
    }
    Some(out)
}

/// Decode a BGRA8 sRGB level, box-averaged down by `factor`.
fn decode_bgra(bytes: &[u8], width: u32, height: u32, factor: u32) -> Option<Vec<[f32; 3]>> {
    let (w, h) = (width as usize, height as usize);
    if bytes.len() < w * h * 4 || factor == 0 {
        return None;
    }
    let lut: Vec<f32> = (0..=255).map(srgb_to_linear).collect();
    let f = factor as usize;
    let (ow, oh) = (w / f, h / f);
    let mut out = vec![[0.0; 3]; ow * oh];
    for y in 0..oh {
        for x in 0..ow {
            let mut sum = [0.0; 3];
            for dy in 0..f {
                for dx in 0..f {
                    let t = &bytes[((y * f + dy) * w + x * f + dx) * 4..][..4];
                    sum[0] += lut[t[2] as usize];
                    sum[1] += lut[t[1] as usize];
                    sum[2] += lut[t[0] as usize];
                }
            }
            out[y * ow + x] = sum.map(|s| s / (f * f) as f32);
        }
    }
    Some(out)
}

/// Separable running min (or max) over a (2r+1)^2 window; wraps in x.
fn morph(input: &[f32], width: usize, height: usize, radius: usize, max: bool) -> Vec<f32> {
    let pick = |a: f32, b: f32| if max { a.max(b) } else { a.min(b) };
    let mut rows = vec![0.0; input.len()];
    for y in 0..height {
        let row = &input[y * width..][..width];
        for x in 0..width {
            let mut v = row[x];
            for d in 1..=radius {
                v = pick(v, pick(row[(x + d) % width], row[(x + width - d) % width]));
            }
            rows[y * width + x] = v;
        }
    }
    let mut out = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut v = rows[y * width + x];
            for d in 1..=radius {
                let up = rows[y.saturating_sub(d) * width + x];
                let down = rows[(y + d).min(height - 1) * width + x];
                v = pick(v, pick(up, down));
            }
            out[y * width + x] = v;
        }
    }
    out
}

/// Separable [1 4 6 4 1] / 16 blur; wraps in x.
fn blur(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let mut rows = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            rows[y * width + x] = (0..5).map(|i| K[i] * input[y * width + (x + width + i - 2) % width]).sum();
        }
    }
    let mut out = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            out[y * width + x] = (0..5)
                .map(|i| K[i] * rows[(y + i).saturating_sub(2).min(height - 1) * width + x])
                .sum();
        }
    }
    out
}

fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Bake from a BC1 mip chain (`bc1 = true`) or a single BGRA8 level.
pub fn bake(bytes: &[u8], mips: &[StarPanoramaMip], full: (u32, u32), bc1: bool) -> Option<MilkyWay> {
    let level = |target: u32| -> Option<(Vec<[f32; 3]>, u32, u32)> {
        if bc1 {
            let mip = mips.iter().find(|mip| mip.width <= target)?;
            let data = bytes.get(usize::try_from(mip.byte_offset).ok()?..)?;
            Some((decode_bc1(data, mip.width, mip.height)?, mip.width, mip.height))
        } else {
            let factor = (full.0 / target).max(1);
            Some((decode_bgra(bytes, full.0, full.1, factor)?, full.0 / factor, full.1 / factor))
        }
    };
    let (source, width, height) = level(MAX_WIDTH)?;
    let (w, h) = (width as usize, height as usize);

    let lum: Vec<f32> = source.iter().map(|&c| luminance(c)).collect();
    drop(source);
    let opened = morph(&morph(&lum, w, h, 1, false), w, h, 1, true);
    let glow = blur(&blur(&blur(&opened, w, h), w, h), w, h);

    // Area-weighted percentiles of the glow set the black point and scale.
    let mut weighted: Vec<(f32, f32)> = glow.iter().enumerate()
        .map(|(i, &v)| (v, ((0.5 - ((i / w) as f32 + 0.5) / h as f32) * std::f32::consts::PI).cos()))
        .collect();
    weighted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f32 = weighted.iter().map(|(_, wt)| wt).sum();
    let percentile = |q: f32| {
        let mut accumulated = 0.0;
        for (value, wt) in &weighted {
            accumulated += wt;
            if accumulated >= q * total {
                return *value;
            }
        }
        weighted.last().map_or(0.0, |v| v.0)
    };
    let black = percentile(BLACK_PERCENTILE);
    let band = percentile(0.995);
    drop(weighted);
    let range = (band - black).max(1.0e-6);

    let mut out = Vec::with_capacity(w * h * 4);
    for &glow in &glow {
        let n = ((glow - black) / range).max(0.0);
        let value = BAND_DISPLAY * n.powf(CONTRAST);
        let t = n.min(1.0);
        let t = t * t * (3.0 - 2.0 * t);
        let colour = [0, 1, 2].map(|i| value * (FAINT_TINT[i] + (BRIGHT_TINT[i] - FAINT_TINT[i]) * t));
        out.extend_from_slice(&[linear_to_srgb(colour[2]), linear_to_srgb(colour[1]), linear_to_srgb(colour[0]), 255]);
    }
    Some(MilkyWay { bytes: out, width, height })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_removes_point_stars_and_keeps_the_band() {
        let (w, h) = (32usize, 16usize);
        let mut field = vec![0.1_f32; w * h];
        for x in 0..w {
            for y in 6..10 {
                field[y * w + x] = 0.5; // a 4-texel-wide band
            }
        }
        field[2 * w + 5] = 5.0; // a point star off the band
        let opened = morph(&morph(&field, w, h, 1, false), w, h, 1, true);
        assert!((opened[2 * w + 5] - 0.1).abs() < 1.0e-6);
        assert!((opened[8 * w + 12] - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn bc1_decodes_endpoints() {
        // One block, both endpoints white (0xFFFF), all indices 0.
        let block = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0];
        let texels = decode_bc1(&block, 4, 4).unwrap();
        assert!(texels.iter().all(|t| t.iter().all(|&c| (c - 1.0).abs() < 1.0e-6)));
    }
}
