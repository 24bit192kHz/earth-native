//! Point-source stars from the Hipparcos Main Catalogue (ESA 1997): every
//! star with V <= 8.0, i.e. the whole naked-eye sky and a magnitude beyond.
//!
//! Each star is drawn as its own GPU primitive (one quad per star,
//! `shaders/stars_points.vert`), never through an image, so it stays a sharp
//! point at any resolution. The vertex shader moves it to the current date:
//! proper motion from the catalogue epoch, then precession, nutation and
//! Earth rotation (one matrix from Astronomy Engine), then annual aberration.
//!
//! The catalogue is uploaded as a small BGRA8 texture, `STARS_PER_ROW` stars
//! per row and four texels per star (shader channel order r, g, b, a):
//!   texel 0: r,g,b = RA bits 23..0            a = Dec bits 23..16
//!   texel 1: r,g   = Dec bits 15..0           b = magnitude code  a = B-V code
//!   texel 2: r,g   = pmRA*cos(Dec) code (16)  b,a = pmDec code (16)
//!   texel 3: unused
//! RA is a 24-bit fraction of a turn and Dec a 24-bit fraction of 180 degrees
//! from the south pole (both far below a pixel); proper motions are in
//! quarter mas/yr around 32768.

pub const STARS_PER_ROW: usize = 256;
pub const TEXELS_PER_STAR: usize = 4;
pub const WIDTH: usize = STARS_PER_ROW * TEXELS_PER_STAR;
pub const MAG_MIN: f64 = -2.0;
pub const MAG_MAX: f64 = 8.0;
pub const BV_MIN: f64 = -0.5;
pub const BV_MAX: f64 = 2.5;
pub const PM_STEP_MAS_PER_YEAR: f64 = 0.25;
/// Hipparcos catalogue epoch J1991.25 (TT) as a Unix time.
pub const EPOCH_UNIX_SECONDS: f64 = 946_728_000.0 - 8.75 * 365.25 * 86_400.0;
pub const SECONDS_PER_JULIAN_YEAR: f64 = 365.25 * 86_400.0;

const CATALOG: &str = include_str!("../assets/stars/hipparcos-v8.csv");

#[derive(Clone, Copy, Debug)]
struct Star {
    ra: f64,
    dec: f64,
    vmag: f64,
    bv: f64,
    pm_ra: f64,
    pm_dec: f64,
}

fn parse() -> Vec<Star> {
    CATALOG
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .filter_map(|line| {
            let values: Vec<f64> = line.split(',').map(|v| v.trim().parse().ok()).collect::<Option<_>>()?;
            let [ra, dec, vmag, bv, pm_ra, pm_dec] = values[..] else { return None };
            Some(Star { ra, dec, vmag, bv, pm_ra, pm_dec })
        })
        .collect()
}

fn code(value: f64, min: f64, max: f64, levels: f64) -> u32 {
    (((value - min) / (max - min)).clamp(0.0, 1.0) * levels).round() as u32
}

fn pm_code(mas_per_year: f64) -> u32 {
    ((mas_per_year / PM_STEP_MAS_PER_YEAR).round() + 32768.0).clamp(0.0, 65535.0) as u32
}

/// BGRA8 memory order for shader-visible (r, g, b, a).
fn texel(r: u32, g: u32, b: u32, a: u32) -> [u8; 4] {
    [b as u8, g as u8, r as u8, a as u8]
}

/// The packed catalogue: (bytes, width, height, star count).
pub fn bake() -> (Vec<u8>, u32, u32, u32) {
    let mut stars = parse();
    // Faint first, so where two PSFs overlap the brighter one blends last.
    stars.sort_by(|a, b| b.vmag.total_cmp(&a.vmag));
    let height = stars.len().div_ceil(STARS_PER_ROW).max(1);
    let mut bytes = vec![0u8; WIDTH * height * 4];
    for (index, star) in stars.iter().enumerate() {
        let ra = (star.ra.rem_euclid(360.0) / 360.0 * 16_777_216.0).round().min(16_777_215.0) as u32;
        let dec = code(star.dec, -90.0, 90.0, 16_777_215.0);
        let mag = 1 + code(star.vmag, MAG_MIN, MAG_MAX, 254.0);
        let bv = code(star.bv, BV_MIN, BV_MAX, 255.0);
        let (pm_ra, pm_dec) = (pm_code(star.pm_ra), pm_code(star.pm_dec));
        let texels = [
            texel(ra >> 16, ra >> 8, ra, dec >> 16),
            texel(dec >> 8, dec, mag, bv),
            texel(pm_ra >> 8, pm_ra, pm_dec >> 8, pm_dec),
            [0; 4],
        ];
        let base = index * TEXELS_PER_STAR * 4;
        for (k, value) in texels.iter().enumerate() {
            bytes[base + k * 4..base + k * 4 + 4].copy_from_slice(value);
        }
    }
    (bytes, WIDTH as u32, height as u32, stars.len() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(bytes: &[u8], index: usize, texel_index: usize) -> [u32; 4] {
        let base = (index * TEXELS_PER_STAR + texel_index) * 4;
        let t = &bytes[base..base + 4];
        [t[2], t[1], t[0], t[3]].map(u32::from)
    }

    #[test]
    fn catalogue_parses_every_star() {
        assert!(parse().len() > 41_000);
    }

    #[test]
    fn sirius_round_trips_last_and_brightest() {
        let (bytes, _, _, count) = bake();
        let index = count as usize - 1;
        let [r0, g0, b0, a0] = channel(&bytes, index, 0);
        let [r1, g1, b1, _] = channel(&bytes, index, 1);
        let [r2, g2, b2, a2] = channel(&bytes, index, 2);
        let ra = f64::from((r0 << 16) | (g0 << 8) | b0) / 16_777_216.0 * 360.0;
        let dec = f64::from((a0 << 16) | (r1 << 8) | g1) / 16_777_215.0 * 180.0 - 90.0;
        let mag = MAG_MIN + f64::from(b1 - 1) / 254.0 * (MAG_MAX - MAG_MIN);
        let pm_ra = (f64::from((r2 << 8) | g2) - 32768.0) * PM_STEP_MAS_PER_YEAR;
        let pm_dec = (f64::from((b2 << 8) | a2) - 32768.0) * PM_STEP_MAS_PER_YEAR;
        assert!((ra - 101.288_54).abs() < 1.0e-4 && (dec + 16.713_14).abs() < 1.0e-4, "{ra} {dec}");
        assert!((mag + 1.44).abs() < 0.03, "{mag}");
        // Hipparcos: -546.0, -1223.1 mas/yr.
        assert!((pm_ra + 546.0).abs() < 0.2 && (pm_dec + 1223.1).abs() < 0.2, "{pm_ra} {pm_dec}");
    }
}
