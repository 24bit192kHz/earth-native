//! Physically based atmosphere: the lookup tables the Earth shader's view-ray
//! march reads (Bruneton & Neyret 2008 parameterisation, Hillaire 2020
//! multiple-scattering approximation).
//!
//! Everything is in kilometres and relative to a unit solar irradiance at the
//! top of the atmosphere; the shader scales by its own sunlight units. RGB are
//! sRGB bands, not single wavelengths: each coefficient is the spectrum
//! averaged over that channel's colour-matching function (CIE 1931 via the
//! Wyman-Sloan-Shirley fit, times the sRGB matrix) under the 5778 K Sun, which
//! is exact for thin paths. Single lines (650/550/450 nm) missed that the red
//! channel spans the Chappuis ozone peak at 603 nm: they absorbed green over
//! red and tinted long grazing paths magenta, where orbital photographs show
//! the limb and the far haze cyan.
//!
//! Constituents:
//! - air: Rayleigh scattering (Bodhaine et al. 1999 dispersion, 13.56e-3 /km
//!   at 550 nm), scale height 8 km;
//! - ozone: Chappuis-band absorption (Serdyuchenko/Gorshelev 2013 cross
//!   sections at 223 K), a 25 km-peaked layer (tent, 15 km half width) of
//!   ~300 DU column;
//! - aerosol: optical depth 0.18 at 550 nm, Angstrom exponent 0.5, scale
//!   height 3.0 km, single-scattering albedo 0.94, Cornette-Shanks g = 0.68:
//!   a dust/marine mix calibrated so haze brightens toward the horizon as in
//!   ISS Earth-observation footage (the MODIS global mean is 0.12-0.15; the
//!   subtropical dust belts under the orbit are 0.3-0.6).

use std::f64::consts::PI;

pub const GROUND_KM: f64 = 6_378.137;
pub const TOP_KM: f64 = GROUND_KM + 100.0;

pub const RAYLEIGH_SCATTERING: [f64; 3] = [7.117e-3, 13.745e-3, 33.233e-3];
pub const RAYLEIGH_SCALE_KM: f64 = 8.0;
pub const OZONE_ABSORPTION: [f64; 3] = [2.785e-3, 1.779e-3, 0.0];
pub const OZONE_PEAK_KM: f64 = 25.0;
pub const OZONE_HALF_WIDTH_KM: f64 = 15.0;
pub const AEROSOL_OPTICAL_DEPTH_550: f64 = 0.18;
pub const AEROSOL_ANGSTROM: f64 = 0.5;
pub const AEROSOL_SCALE_KM: f64 = 3.0;
pub const AEROSOL_ALBEDO: f64 = 0.94;
pub const AEROSOL_G: f64 = 0.68;
/// Mean reflectance of what lies under the air (ocean ~0.06, land ~0.2,
/// cloud excluded), only for light the ground returns into the sky.
pub const GROUND_ALBEDO: f64 = 0.12;

pub const TRANSMITTANCE_WIDTH: usize = 256;
pub const TRANSMITTANCE_HEIGHT: usize = 64;
pub const MULTISCATTER_SIZE: usize = 32;
pub const IRRADIANCE_WIDTH: usize = 64;
pub const IRRADIANCE_HEIGHT: usize = 16;
/// The irradiance table spans the lowest 16 km (surface, cloud tops).
pub const IRRADIANCE_TOP_KM: f64 = 16.0;

/// Wavelengths at which a power law (the aerosol Angstrom law) takes its
/// sRGB band average, for exponents 0-1.5 to within 0.3 %.
pub const EFFECTIVE_WAVELENGTHS_NM: [f64; 3] = [640.0, 545.0, 440.0];

fn aerosol_scattering() -> [f64; 3] {
    EFFECTIVE_WAVELENGTHS_NM.map(|nm| {
        AEROSOL_OPTICAL_DEPTH_550 / AEROSOL_SCALE_KM * (550.0 / nm).powf(AEROSOL_ANGSTROM) * AEROSOL_ALBEDO
    })
}

fn aerosol_extinction() -> [f64; 3] {
    aerosol_scattering().map(|s| s / AEROSOL_ALBEDO)
}

/// Sea-level-normalised scattering and extinction at altitude `h` km.
#[derive(Clone, Copy)]
struct Medium {
    rayleigh: [f64; 3],
    mie: [f64; 3],
    extinction: [f64; 3],
}

fn medium(h: f64) -> Medium {
    let h = h.max(0.0);
    let dr = (-h / RAYLEIGH_SCALE_KM).exp();
    let dm = (-h / AEROSOL_SCALE_KM).exp();
    let dozone = (1.0 - (h - OZONE_PEAK_KM).abs() / OZONE_HALF_WIDTH_KM).max(0.0);
    let (ms, me) = (aerosol_scattering(), aerosol_extinction());
    let rayleigh = RAYLEIGH_SCATTERING.map(|b| b * dr);
    let mie = ms.map(|b| b * dm);
    let extinction = [0, 1, 2].map(|c| rayleigh[c] + me[c] * dm + OZONE_ABSORPTION[c] * dozone);
    Medium { rayleigh, mie, extinction }
}

pub fn rayleigh_phase(cosine: f64) -> f64 {
    3.0 / (16.0 * PI) * (1.0 + cosine * cosine)
}

/// Cornette-Shanks, normalised over the sphere.
pub fn aerosol_phase(cosine: f64) -> f64 {
    let g = AEROSOL_G;
    let k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    k * (1.0 + cosine * cosine) / (1.0 + g * g - 2.0 * g * cosine).powf(1.5)
}

/// Distance from radius `r` along zenith cosine `mu` to the top of the air.
fn distance_to_top(r: f64, mu: f64) -> f64 {
    (-r * mu + (r * r * (mu * mu - 1.0) + TOP_KM * TOP_KM).max(0.0).sqrt()).max(0.0)
}

/// Distance to the ground, or None when the ray misses it.
fn distance_to_ground(r: f64, mu: f64) -> Option<f64> {
    let discriminant = r * r * (mu * mu - 1.0) + GROUND_KM * GROUND_KM;
    (mu < 0.0 && discriminant >= 0.0).then(|| (-r * mu - discriminant.sqrt()).max(0.0))
}

fn exp3(v: [f64; 3]) -> [f64; 3] {
    v.map(|x| (-x).exp())
}

/// Exact (fine midpoint) transmittance from (r, mu) to the top of the air.
pub fn integrate_transmittance(r: f64, mu: f64) -> [f64; 3] {
    const STEPS: usize = 256;
    let length = distance_to_top(r, mu);
    let dt = length / STEPS as f64;
    let mut depth = [0.0; 3];
    for i in 0..STEPS {
        let t = (i as f64 + 0.5) * dt;
        let ri = (r * r + t * t + 2.0 * r * mu * t).sqrt();
        let m = medium(ri - GROUND_KM);
        for c in 0..3 {
            depth[c] += m.extinction[c] * dt;
        }
    }
    exp3(depth)
}

fn unit_to_texel(x: f64, size: usize) -> f64 {
    0.5 / size as f64 + x * (1.0 - 1.0 / size as f64)
}

fn texel_to_unit(u: f64, size: usize) -> f64 {
    (u - 0.5 / size as f64) / (1.0 - 1.0 / size as f64)
}

/// Bruneton's (r, mu) -> uv, for rays that do not hit the ground.
pub fn transmittance_uv(r: f64, mu: f64) -> (f64, f64) {
    let h = (TOP_KM * TOP_KM - GROUND_KM * GROUND_KM).sqrt();
    let rho = (r * r - GROUND_KM * GROUND_KM).max(0.0).sqrt();
    let d = distance_to_top(r, mu);
    let d_min = TOP_KM - r;
    let d_max = rho + h;
    let x_mu = ((d - d_min) / (d_max - d_min)).clamp(0.0, 1.0);
    let x_r = (rho / h).clamp(0.0, 1.0);
    (unit_to_texel(x_mu, TRANSMITTANCE_WIDTH), unit_to_texel(x_r, TRANSMITTANCE_HEIGHT))
}

fn transmittance_r_mu(u: f64, v: f64) -> (f64, f64) {
    let x_mu = texel_to_unit(u, TRANSMITTANCE_WIDTH);
    let x_r = texel_to_unit(v, TRANSMITTANCE_HEIGHT);
    let h = (TOP_KM * TOP_KM - GROUND_KM * GROUND_KM).sqrt();
    let rho = h * x_r;
    let r = (rho * rho + GROUND_KM * GROUND_KM).sqrt();
    let d_min = TOP_KM - r;
    let d_max = rho + h;
    let d = d_min + x_mu * (d_max - d_min);
    let mu = if d == 0.0 { 1.0 } else { ((h * h - rho * rho - d * d) / (2.0 * r * d)).clamp(-1.0, 1.0) };
    (r, mu)
}

/// Transmittance of the air along a straight ray from `origin` (km, Earth
/// centred) in unit direction `dir`; zero when the ray meets the ground.
pub fn ray_transmittance(origin: [f64; 3], dir: [f64; 3]) -> [f64; 3] {
    let b = origin[0] * dir[0] + origin[1] * dir[1] + origin[2] * dir[2];
    let c = origin.iter().map(|v| v * v).sum::<f64>();
    let ground = b * b - (c - GROUND_KM * GROUND_KM);
    if ground > 0.0 && -b - ground.sqrt() > 0.0 {
        return [0.0; 3];
    }
    let top = b * b - (c - TOP_KM * TOP_KM);
    if top <= 0.0 {
        return [1.0; 3];
    }
    let t0 = (-b - top.sqrt()).max(0.0);
    let t1 = -b + top.sqrt();
    if t1 <= t0 {
        return [1.0; 3];
    }
    // Sample densely around the closest approach, where the density peaks.
    const STEPS: usize = 160;
    let closest = (-b).clamp(t0, t1);
    let mut depth = [0.0; 3];
    let mut previous = t0;
    for i in 1..=STEPS {
        let u = i as f64 / STEPS as f64;
        // Piecewise-quadratic map concentrating samples near `closest`.
        let t = if u <= 0.5 {
            let v = u / 0.5;
            closest - (closest - t0) * (1.0 - v) * (1.0 - v)
        } else {
            let v = (u - 0.5) / 0.5;
            closest + (t1 - closest) * v * v
        };
        let mid = 0.5 * (previous + t);
        let dt = t - previous;
        previous = t;
        let p = [origin[0] + dir[0] * mid, origin[1] + dir[1] * mid, origin[2] + dir[2] * mid];
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let m = medium(r - GROUND_KM);
        for ch in 0..3 {
            depth[ch] += m.extinction[ch] * dt;
        }
    }
    exp3(depth)
}

pub struct Tables {
    pub transmittance: Vec<[f32; 3]>,
    pub multiscatter: Vec<[f32; 3]>,
    pub irradiance: Vec<[f32; 3]>,
}

struct TransmittanceTable<'a>(&'a [[f32; 3]]);

impl TransmittanceTable<'_> {
    fn sample(&self, r: f64, mu: f64) -> [f64; 3] {
        let (u, v) = transmittance_uv(r, mu);
        bilinear(self.0, TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT, u, v)
    }

    /// Sunlight reaching radius r with sun zenith cosine mu_s, zero when the
    /// Earth is in the way (no penumbra on the CPU side).
    fn sun(&self, r: f64, mu_s: f64) -> [f64; 3] {
        let horizon = -(1.0 - (GROUND_KM / r).powi(2)).max(0.0).sqrt();
        if mu_s < horizon { [0.0; 3] } else { self.sample(r, mu_s) }
    }
}

fn bilinear(table: &[[f32; 3]], width: usize, height: usize, u: f64, v: f64) -> [f64; 3] {
    let x = (u * width as f64 - 0.5).clamp(0.0, (width - 1) as f64);
    let y = (v * height as f64 - 0.5).clamp(0.0, (height - 1) as f64);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let at = |xx: usize, yy: usize| table[yy * width + xx];
    [0, 1, 2].map(|c| {
        let top = at(x0, y0)[c] as f64 * (1.0 - fx) + at(x1, y0)[c] as f64 * fx;
        let bottom = at(x0, y1)[c] as f64 * (1.0 - fx) + at(x1, y1)[c] as f64 * fx;
        top * (1.0 - fy) + bottom * fy
    })
}

fn multiscatter_texel(transmittance: &TransmittanceTable, r: f64, mu_s: f64) -> [f64; 3] {
    // Hillaire 2020, section 5.5: second-order isotropic radiance L2 and the
    // transfer factor f_ms from 64 directions; Psi = L2 / (1 - f_ms).
    const SQRT_SAMPLES: usize = 8;
    const STEPS: usize = 32;
    let sun = [(1.0 - mu_s * mu_s).max(0.0).sqrt(), 0.0, mu_s];
    let isotropic = 1.0 / (4.0 * PI);
    let solid_angle = 4.0 * PI / (SQRT_SAMPLES * SQRT_SAMPLES) as f64;
    let mut l2 = [0.0; 3];
    let mut fms = [0.0; 3];
    for i in 0..SQRT_SAMPLES {
        for j in 0..SQRT_SAMPLES {
            let theta = 2.0 * PI * (i as f64 + 0.5) / SQRT_SAMPLES as f64;
            let cos_phi = 1.0 - 2.0 * (j as f64 + 0.5) / SQRT_SAMPLES as f64;
            let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
            let dir = [theta.cos() * sin_phi, theta.sin() * sin_phi, cos_phi];
            let mu = dir[2];
            let ground = distance_to_ground(r, mu);
            let length = ground.unwrap_or_else(|| distance_to_top(r, mu));
            let dt = length / STEPS as f64;
            let mut throughput = [1.0; 3];
            let mut l = [0.0; 3];
            let mut f = [0.0; 3];
            for s in 0..STEPS {
                let t = (s as f64 + 0.5) * dt;
                let p = [dir[0] * t, dir[1] * t, r + dir[2] * t];
                let rp = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                let m = medium(rp - GROUND_KM);
                let mu_sp = (p[0] * sun[0] + p[2] * sun[2]) / rp;
                let sun_t = transmittance.sun(rp, mu_sp);
                for c in 0..3 {
                    let scattering = m.rayleigh[c] + m.mie[c];
                    let step_t = (-m.extinction[c] * dt).exp();
                    let integral = (1.0 - step_t) / m.extinction[c].max(1.0e-12);
                    l[c] += throughput[c] * sun_t[c] * scattering * isotropic * integral;
                    f[c] += throughput[c] * scattering * integral;
                    throughput[c] *= step_t;
                }
            }
            if let Some(tg) = ground {
                let p = [dir[0] * tg, dir[1] * tg, r + dir[2] * tg];
                let rp = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                let mu_sg = (p[0] * sun[0] + p[2] * sun[2]) / rp;
                let sun_t = transmittance.sun(rp, mu_sg);
                for c in 0..3 {
                    l[c] += throughput[c] * sun_t[c] * mu_sg.max(0.0) * GROUND_ALBEDO / PI;
                }
            }
            for c in 0..3 {
                l2[c] += l[c] * solid_angle;
                fms[c] += f[c] * solid_angle;
            }
        }
    }
    [0, 1, 2].map(|c| {
        let l2 = l2[c] * isotropic;
        let f = (fms[c] * isotropic).min(0.99);
        l2 / (1.0 - f)
    })
}

fn multiscatter_uv_to_r_mu(u: f64, v: f64) -> (f64, f64) {
    let mu_s = texel_to_unit(u, MULTISCATTER_SIZE) * 2.0 - 1.0;
    let r = GROUND_KM + texel_to_unit(v, MULTISCATTER_SIZE) * (TOP_KM - GROUND_KM);
    (r.max(GROUND_KM + 0.01), mu_s)
}

fn sample_multiscatter(table: &[[f32; 3]], r: f64, mu_s: f64) -> [f64; 3] {
    let u = unit_to_texel(0.5 + 0.5 * mu_s, MULTISCATTER_SIZE);
    let v = unit_to_texel((r - GROUND_KM) / (TOP_KM - GROUND_KM), MULTISCATTER_SIZE);
    bilinear(table, MULTISCATTER_SIZE, MULTISCATTER_SIZE, u, v)
}

/// Downward sky irradiance on a horizontal surface at radius r (clear sky,
/// relative to unit top-of-atmosphere solar irradiance): single scattering
/// with the full phase functions plus the multiple-scattering term.
fn irradiance_texel(transmittance: &TransmittanceTable, multiscatter: &[[f32; 3]], r: f64, mu_s: f64) -> [f64; 3] {
    const AZIMUTHS: usize = 16;
    const ZENITHS: usize = 8;
    const STEPS: usize = 24;
    let sun = [(1.0 - mu_s * mu_s).max(0.0).sqrt(), 0.0, mu_s];
    let mut e = [0.0; 3];
    for a in 0..AZIMUTHS {
        let phi = 2.0 * PI * (a as f64 + 0.5) / AZIMUTHS as f64;
        for z in 0..ZENITHS {
            // Cosine-weighted zenith rings: mu uniform in mu^2.
            let mu = ((z as f64 + 0.5) / ZENITHS as f64).sqrt();
            let sin = (1.0 - mu * mu).sqrt();
            let dir = [phi.cos() * sin, phi.sin() * sin, mu];
            let cosine = dir[0] * sun[0] + dir[2] * sun[2];
            let (pr, pm) = (rayleigh_phase(cosine), aerosol_phase(cosine));
            let length = distance_to_top(r, mu);
            let dt = length / STEPS as f64;
            let mut throughput = [1.0; 3];
            let mut l = [0.0; 3];
            for s in 0..STEPS {
                let t = (s as f64 + 0.5) * dt;
                let p = [dir[0] * t, dir[1] * t, r + dir[2] * t];
                let rp = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                let m = medium(rp - GROUND_KM);
                let mu_sp = (p[0] * sun[0] + p[2] * sun[2]) / rp;
                let sun_t = transmittance.sun(rp, mu_sp);
                let psi = sample_multiscatter(multiscatter, rp, mu_sp);
                for c in 0..3 {
                    let source = sun_t[c] * (m.rayleigh[c] * pr + m.mie[c] * pm)
                        + psi[c] * (m.rayleigh[c] + m.mie[c]);
                    let step_t = (-m.extinction[c] * dt).exp();
                    l[c] += throughput[c] * source * (1.0 - step_t) / m.extinction[c].max(1.0e-12);
                    throughput[c] *= step_t;
                }
            }
            // Cosine-weighted rings: each sample carries pi / N of E.
            for c in 0..3 {
                e[c] += l[c] * PI / (AZIMUTHS * ZENITHS) as f64;
            }
        }
    }
    e
}

#[cfg(test)]
pub fn irradiance_uv(r: f64, mu_s: f64) -> (f64, f64) {
    (
        unit_to_texel(0.5 + 0.5 * mu_s, IRRADIANCE_WIDTH),
        unit_to_texel(((r - GROUND_KM) / IRRADIANCE_TOP_KM).clamp(0.0, 1.0), IRRADIANCE_HEIGHT),
    )
}

/// `f(0..count)` on all cores in contiguous chunks (texels are
/// independent); the same values as a serial map, in order.
fn parallel_map<T: Send + Default + Clone, F: Fn(usize) -> T + Sync>(count: usize, f: F) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(count.max(1));
    let chunk = count.div_ceil(threads);
    let mut out = vec![T::default(); count];
    std::thread::scope(|scope| {
        for (index, slice) in out.chunks_mut(chunk).enumerate() {
            let f = &f;
            scope.spawn(move || {
                for (offset, value) in slice.iter_mut().enumerate() {
                    *value = f(index * chunk + offset);
                }
            });
        }
    });
    out
}

/// The three tables, each spread over all cores (367 ms serially at
/// startup, the largest share of the time to the first frame).
pub fn bake() -> Tables {
    let transmittance: Vec<[f32; 3]> = parallel_map(TRANSMITTANCE_WIDTH * TRANSMITTANCE_HEIGHT, |index| {
        let (x, y) = (index % TRANSMITTANCE_WIDTH, index / TRANSMITTANCE_WIDTH);
        let (r, mu) = transmittance_r_mu(
            (x as f64 + 0.5) / TRANSMITTANCE_WIDTH as f64,
            (y as f64 + 0.5) / TRANSMITTANCE_HEIGHT as f64,
        );
        integrate_transmittance(r, mu).map(|v| v as f32)
    });
    let table = TransmittanceTable(&transmittance);
    let multiscatter: Vec<[f32; 3]> = parallel_map(MULTISCATTER_SIZE * MULTISCATTER_SIZE, |index| {
        let (x, y) = (index % MULTISCATTER_SIZE, index / MULTISCATTER_SIZE);
        let (r, mu_s) = multiscatter_uv_to_r_mu(
            (x as f64 + 0.5) / MULTISCATTER_SIZE as f64,
            (y as f64 + 0.5) / MULTISCATTER_SIZE as f64,
        );
        multiscatter_texel(&table, r, mu_s).map(|v| v as f32)
    });
    let irradiance: Vec<[f32; 3]> = parallel_map(IRRADIANCE_WIDTH * IRRADIANCE_HEIGHT, |index| {
        let (x, y) = (index % IRRADIANCE_WIDTH, index / IRRADIANCE_WIDTH);
        let mu_s = texel_to_unit((x as f64 + 0.5) / IRRADIANCE_WIDTH as f64, IRRADIANCE_WIDTH) * 2.0 - 1.0;
        let h = texel_to_unit((y as f64 + 0.5) / IRRADIANCE_HEIGHT as f64, IRRADIANCE_HEIGHT) * IRRADIANCE_TOP_KM;
        irradiance_texel(&table, &multiscatter, GROUND_KM + h.max(0.0), mu_s).map(|v| v as f32)
    });
    Tables { transmittance, multiscatter, irradiance }
}

/// IEEE half from f32 (round to nearest even; finite inputs only here).
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7bff; // clamp to the largest finite half
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let m = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let half = m >> shift;
        let remainder = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let round = (remainder > halfway || (remainder == halfway && (half & 1) == 1)) as u32;
        return sign | (half + round) as u16;
    }
    let half = ((half_exponent as u32) << 10) | (mantissa >> 13);
    let remainder = mantissa & 0x1fff;
    let round = (remainder > 0x1000 || (remainder == 0x1000 && (half & 1) == 1)) as u32;
    sign | (half + round) as u16
}

pub fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half as u32) & 0x8000) << 16;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let mantissa = (half & 0x3ff) as u32;
    let bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

/// RGBA16F bytes (alpha 1) for an RGB table.
pub fn rgba16f_bytes(table: &[[f32; 3]]) -> Vec<u8> {
    table
        .iter()
        .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 1.0])
        .flat_map(|v| f32_to_f16(v).to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_optical_depths_match_the_published_constituents() {
        let t = integrate_transmittance(GROUND_KM, 1.0);
        let tau = t.map(|v| -v.ln());
        // Rayleigh (0.057/0.110/0.266) + aerosol + ozone (~0.04/0.03/0).
        let aerosol = aerosol_extinction().map(|b| b * AEROSOL_SCALE_KM);
        for c in 0..3 {
            let rayleigh = RAYLEIGH_SCATTERING[c] * RAYLEIGH_SCALE_KM;
            let ozone = OZONE_ABSORPTION[c] * OZONE_HALF_WIDTH_KM;
            let expected = rayleigh + aerosol[c] + ozone;
            assert!((tau[c] / expected - 1.0).abs() < 0.01, "channel {c}: {} vs {expected}", tau[c]);
        }
        // The green band's effective wavelength is 545 nm.
        assert!((aerosol[1] / AEROSOL_OPTICAL_DEPTH_550 - 1.0).abs() < 0.005);
    }

    #[test]
    fn ray_transmittance_matches_the_table_integrator() {
        // From just above the air straight down to the ground is blocked;
        // a ray leaving the surface upward equals the zenith column.
        assert_eq!(ray_transmittance([0.0, 0.0, TOP_KM + 300.0], [0.0, 0.0, -1.0]), [0.0; 3]);
        let up = ray_transmittance([0.0, 0.0, GROUND_KM + 0.001], [0.0, 0.0, 1.0]);
        let reference = integrate_transmittance(GROUND_KM + 0.001, 1.0);
        for c in 0..3 {
            assert!((up[c] / reference[c] - 1.0).abs() < 2e-3, "{up:?} vs {reference:?}");
        }
        // Grazing sunlight through the troposphere reddens (the orange
        // sunset layer), but 20 km up the Chappuis band, which the red
        // channel spans, takes red and green and leaves blue: the blue band
        // above it in orbital photographs (Hulburt 1953).
        let low = ray_transmittance([-3000.0, 0.0, GROUND_KM + 5.0], [1.0, 0.0, 0.0]);
        assert!(low[0] > low[1] && low[1] > low[2], "{low:?}");
        let limb = ray_transmittance([-3000.0, 0.0, GROUND_KM + 20.0], [1.0, 0.0, 0.0]);
        assert!(limb[2] > limb[1] && limb[1] > limb[0] && limb[2] < 0.5, "{limb:?}");
    }

    #[test]
    fn transmittance_mapping_round_trips() {
        for &(r, mu) in &[(GROUND_KM, 1.0), (GROUND_KM + 5.0, 0.3), (GROUND_KM + 60.0, -0.1), (TOP_KM - 0.5, 0.9)] {
            let (u, v) = transmittance_uv(r, mu);
            let (r2, mu2) = transmittance_r_mu(u, v);
            assert!((r2 - r).abs() < 1e-6, "r {r} -> {r2}");
            assert!((mu2 - mu).abs() < 1e-6, "mu {mu} -> {mu2}");
        }
    }

    #[test]
    fn the_earth_shader_uses_these_constants() {
        let shader = include_str!("../shaders/earth_textured.frag");
        let vec3 = |v: [f64; 3]| format!("vec3({:.6e}, {:.6e}, {:.6e})", v[0], v[1], v[2]);
        assert!(shader.contains(&format!("RAYLEIGH_SCATTERING = {};", vec3(RAYLEIGH_SCATTERING))), "Rayleigh");
        assert!(shader.contains(&format!("OZONE_ABSORPTION = {};", vec3(OZONE_ABSORPTION))), "ozone");
        assert!(shader.contains(&format!("AEROSOL_EXTINCTION = {};", vec3(aerosol_extinction()))), "aerosol {}", vec3(aerosol_extinction()));
        assert!(shader.contains(&format!("AEROSOL_ALBEDO = {AEROSOL_ALBEDO:?};")), "albedo");
        assert!(shader.contains(&format!("AEROSOL_SCALE = {AEROSOL_SCALE_KM:?};")), "scale");
        assert!(shader.contains(&format!("AEROSOL_G = {AEROSOL_G:?};")), "g");
    }

    #[test]
    fn half_floats_round_trip() {
        for &v in &[0.0f32, 1.0, 0.5, 3.1e-5, 1.0e-7, 65000.0, 0.018_75, 0.333] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= v.abs() * 1.0e-3 + 6.0e-8, "{v} -> {back}");
        }
    }

    #[test]
    fn tables_are_physical() {
        let tables = bake();
        assert_eq!(tables.transmittance.len(), TRANSMITTANCE_WIDTH * TRANSMITTANCE_HEIGHT);
        // Zenith sun at sea level: blue is attenuated most.
        let (u, v) = transmittance_uv(GROUND_KM, 1.0);
        let t = bilinear(&tables.transmittance, TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT, u, v);
        assert!(t[0] > t[1] && t[1] > t[2] && t[2] > 0.5, "{t:?}");
        // Overhead-sun clear sky irradiance on the ground is ~10-20% of the
        // direct beam and blue-dominated.
        let (u, v) = irradiance_uv(GROUND_KM, 1.0);
        let e = bilinear(&tables.irradiance, IRRADIANCE_WIDTH, IRRADIANCE_HEIGHT, u, v);
        assert!(e[2] > e[0] && e[1] > 0.05 && e[1] < 0.3, "sky irradiance {e:?}");
        assert!(tables.multiscatter.iter().all(|m| m.iter().all(|v| v.is_finite() && *v >= 0.0)));
    }
}
