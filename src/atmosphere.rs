//! Earth-calibrated atmosphere dimensions shared by CPU validation and shaders.
#![allow(dead_code)] // The integrator is compiled by build.rs; runtime embeds its output.

pub const EARTH_RADIUS_KM: f32 = 6_378.137;
pub const ATMOSPHERE_TOP_KM: f32 = 100.0;
pub const ATMOSPHERE_SCALE_HEIGHT_KM: f32 = 8.5;
pub const CLOUD_ALTITUDE_KM: f32 = 5.5;
pub const LUT_WIDTH: usize = 512;
pub const MAX_COLUMN_KM: f64 = 400.0;

pub const fn normalized_top_radius(earth_radius: f32) -> f32 {
    earth_radius * (1.0 + ATMOSPHERE_TOP_KM / EARTH_RADIUS_KM)
}

/// Exponential molecular density column from the surface to 100 km.
/// Build-time integration, in km; mu is the cosine of the outward zenith angle.
pub fn column_km(mu: f64) -> f64 {
    let radius = EARTH_RADIUS_KM as f64;
    let top = radius + ATMOSPHERE_TOP_KM as f64;
    let length = (radius * radius * mu * mu + top * top - radius * radius).sqrt() - radius * mu;
    let step = length / 2048.0;
    (0..2048).map(|i| {
        let distance = (i as f64 + 0.5) * step;
        let altitude = (radius * radius + distance * distance + 2.0 * radius * distance * mu).sqrt() - radius;
        (-altitude / ATMOSPHERE_SCALE_HEIGHT_KM as f64).exp() * step
    }).sum()
}

/// Two UNORM channels encode one 16-bit column; hardware interpolation remains
/// linear after decoding. Quadratic mu spacing concentrates samples at the limb.
pub fn bake_column_lut() -> Vec<u8> {
    (0..LUT_WIDTH).flat_map(|i| {
        let u = i as f64 / (LUT_WIDTH - 1) as f64;
        let encoded = (column_km(u * u) / MAX_COLUMN_KM * 65535.0).round() as u16;
        [0, encoded as u8, (encoded >> 8) as u8, 255]
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn top_of_atmosphere_is_100_km_above_surface() {
        let surface = 0.78;
        let top = normalized_top_radius(surface);
        assert!(((top - surface) / surface * EARTH_RADIUS_KM - ATMOSPHERE_TOP_KM).abs() < 0.001);
    }
    #[test]
    fn scale_height_is_physical_and_below_shell() {
        assert!(ATMOSPHERE_SCALE_HEIGHT_KM > 7.0 && ATMOSPHERE_SCALE_HEIGHT_KM < 10.0);
        assert!(ATMOSPHERE_SCALE_HEIGHT_KM < ATMOSPHERE_TOP_KM);
    }
    #[test]
    fn baked_columns_match_integration_and_vertical_analytic_solution() {
        assert!((column_km(1.0) - 8.5 * (1.0 - (-100.0_f64 / 8.5).exp())).abs() < 0.0001);
        assert!(column_km(0.0) > 280.0 && column_km(0.0) < 300.0);
        let lut = bake_column_lut();
        for i in 0..1000 {
            let mu = i as f64 / 999.0;
            let coordinate = mu.sqrt() * (LUT_WIDTH - 1) as f64;
            let index = (coordinate.floor() as usize).min(LUT_WIDTH - 2);
            let decode = |j: usize| u16::from_be_bytes([lut[j * 4 + 2], lut[j * 4 + 1]]) as f64 * MAX_COLUMN_KM / 65535.0;
            let interpolated = decode(index) + (decode(index + 1) - decode(index)) * (coordinate - index as f64);
            assert!((interpolated / column_km(mu) - 1.0).abs() < 0.0005, "mu={mu}");
        }
    }
}
