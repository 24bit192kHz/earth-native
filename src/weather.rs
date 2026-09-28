use std::{env, fs, path::PathBuf};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Deserialize)]
pub struct WeatherState {
    pub source: String,
    pub kind: String,
    pub valid_unix_utc: i64,
    #[serde(default)]
    pub aurora_unix_utc: i64,
    pub texture: PathBuf,
    pub sha256: String,
    pub width: u32,
    pub height: u32,
    /// Observed cloud cover (NOAA GMGSI visible/infrared + GFS), 4096x2048
    /// BGRA: R cover; G sea-ice concentration (else cloud-top coldness);
    /// with aerosol, B the 440-645 nm Angstrom exponent and A the 550 nm
    /// aerosol optical depth.
    #[serde(default)]
    pub clouds_texture: Option<PathBuf>,
    #[serde(default)]
    pub clouds_sha256: Option<String>,
    #[serde(default)]
    pub clouds_unix_utc: i64,
    /// NOAA GEFS-Aerosols analysis time packed into the cloud texture (0: none).
    #[serde(default)]
    pub aerosol_unix_utc: i64,
    /// OSI SAF sea-ice concentration day packed into G of the cloud texture.
    #[serde(default)]
    pub sea_ice_unix_utc: i64,
}

impl WeatherState {
    pub fn load() -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let Some(path) = env::var_os("EARTH_NATIVE_WEATHER_MANIFEST") else { return Ok(None); };
        // The weather feed may start after the renderer; until it has written
        // its first manifest there is simply no weather yet.
        let data = match fs::read(path) {
            Ok(data) => data,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        if data.len() > 16384 { return Err("weather manifest exceeds 16 KiB".into()); }
        let state: Self = serde_json::from_slice(&data)?;
        state.validate()?;
        let file = fs::File::open(&state.texture)?;
        if file.metadata()?.len() != u64::from(state.width) * u64::from(state.height) * 4 {
            return Err("weather payload size mismatch".into());
        }
        let mut digest = Sha256::new();
        std::io::copy(&mut std::io::BufReader::new(file), &mut digest)?;
        if format!("{:x}", digest.finalize()) != state.sha256 {
            return Err("weather payload checksum mismatch".into());
        }
        let preview = crate::day_color::load_preview_texture("EARTH_NATIVE_WEATHER_MANIFEST",
            state.texture.clone().into_os_string(), crate::day_color::PreviewColorSpace::Linear)?;
        if preview.extent() != (state.width, state.height) { return Err("weather sidecar geometry mismatch".into()); }
        let mut state = state;
        // Live clouds are optional: a bad or missing file keeps the static map.
        if let Err(error) = state.verify_clouds() {
            eprintln!("earth-native: live clouds ignored: {error}");
            state.clouds_texture = None;
        }
        Ok(Some(state))
    }

    fn verify_clouds(&self) -> Result<(), Box<dyn std::error::Error>> {
        let (Some(path), Some(sha256)) = (&self.clouds_texture, &self.clouds_sha256) else { return Ok(()); };
        let file = fs::File::open(path)?;
        if file.metadata()?.len() != 4096 * 2048 * 4 {
            return Err("cloud payload size mismatch".into());
        }
        let mut digest = Sha256::new();
        std::io::copy(&mut std::io::BufReader::new(file), &mut digest)?;
        if format!("{:x}", digest.finalize()) != *sha256 {
            return Err("cloud payload checksum mismatch".into());
        }
        Ok(())
    }

    /// Live clouds are shown while no more than six hours old.
    pub fn live_clouds(&self, now: i64) -> Option<&PathBuf> {
        self.clouds_texture.as_ref().filter(|_| (now - self.clouds_unix_utc).abs() < 6 * 3600)
    }

    /// The aerosol analysis is used while no more than a day old (it
    /// changes over days; GEFS-Aerosols runs every six hours).
    pub fn live_aerosol(&self, now: i64) -> bool {
        self.aerosol_unix_utc > 0 && (now - self.aerosol_unix_utc).abs() < 24 * 3600
    }

    /// Daily sea ice is used while no more than four days old.
    pub fn live_sea_ice(&self, now: i64) -> bool {
        self.sea_ice_unix_utc > 0 && (now - self.sea_ice_unix_utc).abs() < 4 * 86_400
    }

    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.source != "NOAA-GFS" || self.kind != "model-analysis"
            || (self.width, self.height) != (1440, 720) || self.valid_unix_utc <= 0 || self.aurora_unix_utc < 0
            || self.sha256.len() != 64 || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("unsupported weather manifest".into());
        }
        Ok(())
    }

    pub fn age_label(&self, now: i64) -> &'static str {
        match now - self.valid_unix_utc {
            age if age < -3600 => "future",
            age if age > 18 * 3600 => "stale",
            _ => "current-model",
        }
    }
}

/// Where the aurora is brightest in the dark right now: the 1-degree cell
/// of highest NOAA OVATION probability (alpha of the 1440x720 BGRA field
/// texture) whose centre has the Sun at least 10 degrees below the horizon.
/// `sun` is the Earth-fixed (ECEF) unit vector to the Sun. Returns latitude,
/// longitude and probability (0-1).
pub fn strongest_dark_aurora(fields: &[u8], width: usize, height: usize, sun: [f32; 3]) -> Option<(f32, f32, f32)> {
    if width % 4 != 0 || height % 4 != 0 || fields.len() != width * height * 4 {
        return None;
    }
    let dark = -(10.0_f32.to_radians().sin());
    let mut best: Option<(f32, f32, f32)> = None;
    for cell_row in 0..height / 4 {
        let latitude = 90.0 - (cell_row as f32 * 4.0 + 2.0) * 180.0 / height as f32;
        if !(45.0..=85.0).contains(&latitude.abs()) {
            continue;
        }
        for cell_column in 0..width / 4 {
            let longitude = -180.0 + (cell_column as f32 * 4.0 + 2.0) * 360.0 / width as f32;
            let (lat, lon) = (latitude.to_radians(), longitude.to_radians());
            let up = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
            if up[0] * sun[0] + up[1] * sun[1] + up[2] * sun[2] > dark {
                continue;
            }
            let mut sum = 0u32;
            for row in cell_row * 4..cell_row * 4 + 4 {
                for column in cell_column * 4..cell_column * 4 + 4 {
                    sum += u32::from(fields[(row * width + column) * 4 + 3]);
                }
            }
            let probability = sum as f32 / (16.0 * 255.0);
            if best.map_or(true, |(_, _, p)| probability > p) {
                best = Some((latitude, longitude, probability));
            }
        }
    }
    best
}

/// Point `distance` (radians of arc) from (lat, lon) along initial
/// `bearing` (radians from north), and the bearing back from there.
fn great_circle(lat: f64, lon: f64, bearing: f64, distance: f64) -> (f64, f64, f64) {
    let lat2 = (lat.sin() * distance.cos() + lat.cos() * distance.sin() * bearing.cos()).asin();
    let lon2 = lon + (bearing.sin() * distance.sin() * lat.cos()).atan2(distance.cos() - lat.sin() * lat2.sin());
    let d_lon = lon - lon2;
    let back = (d_lon.sin() * lat.cos()).atan2(lat2.cos() * lat.sin() - lat2.sin() * lat.cos() * d_lon.cos());
    (lat2, lon2, back)
}

fn sun_sine(lat: f64, lon: f64, sun: [f32; 3]) -> f64 {
    lat.cos() * lon.cos() * f64::from(sun[0]) + lat.cos() * lon.sin() * f64::from(sun[1]) + lat.sin() * f64::from(sun[2])
}

/// Where to hover (420 km up, `arc_degrees` of arc from the aurora) and
/// which way to look so the aurora stands against a dark sky: of twelve
/// vantage points around it, one whose horizon (the limb ~21-26 degrees
/// ahead, where sunlit air would glow and set the exposure) is in night. (A sunlit camera is fine: the Sun is then behind it.) Ties favour
/// the equatorward side, as from the ISS.
/// Returns camera latitude, longitude and heading in degrees.
pub fn aurora_vantage(latitude: f32, longitude: f32, arc_degrees: f32, sun: [f32; 3]) -> (f32, f32, f32) {
    let (lat, lon) = (f64::from(latitude).to_radians(), f64::from(longitude).to_radians());
    let distance = f64::from(arc_degrees).to_radians();
    let mut best = (f64::INFINITY, 0.0, 0.0, 0.0);
    for step in 0..12 {
        let bearing = f64::from(step) * 30f64.to_radians();
        let (camera_lat, camera_lon, heading) = great_circle(lat, lon, bearing, distance);
        let mut worst = f64::NEG_INFINITY;
        for offset in [-35.0f64, -15.0, 0.0, 15.0, 35.0] {
            for reach in [21.0f64, 26.0] {
                let (p_lat, p_lon, _) = great_circle(camera_lat, camera_lon, heading + offset.to_radians(), reach.to_radians());
                worst = worst.max(sun_sine(p_lat, p_lon, sun));
            }
        }
        // Air whose Sun is ~12 degrees down is dark to ~150 km: any such
        // horizon is fully dark, and the ISS-like equatorward side wins, where
        // the arcs on the oval's poleward flank stand edge-on above the limb.
        let equatorward = 0.002 * (camera_lat.abs() - lat.abs()).to_degrees();
        let score = worst.max(-0.2) + equatorward;
        if score < best.0 {
            best = (score, camera_lat, camera_lon, heading);
        }
    }
    let heading = best.3.to_degrees().rem_euclid(360.0);
    (best.1.to_degrees() as f32, ((best.2.to_degrees() + 540.0).rem_euclid(360.0) - 180.0) as f32, heading as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aurora_vantage_faces_the_aurora_away_from_sunlit_air() {
        // Northern summer, local midnight under the oval at 65 N: the pole
        // is sunlit, so looking poleward would put glowing air on the
        // horizon. The chosen view must not face north, and its horizon
        // must be in night.
        let sun = { let lat = 20f32.to_radians(); [lat.cos(), 0.0, lat.sin()] };
        let (lat, lon, heading) = aurora_vantage(65.0, 180.0, 11.0, sun);
        let north = heading.to_radians().cos();
        assert!(north < 0.5, "heading {heading} from {lat} {lon}");
        let (h_lat, h_lon, _) = great_circle(f64::from(lat).to_radians(), f64::from(lon).to_radians(),
            f64::from(heading).to_radians(), 21f64.to_radians());
        assert!(sun_sine(h_lat, h_lon, sun) < -0.1, "horizon lit");
        // Night all round: the ISS-like equatorward side, looking north.
        let sun = [0.0, 0.0, -1.0];
        let (lat, _lon, heading) = aurora_vantage(65.0, 180.0, 11.0, sun);
        assert!((lat - 54.0).abs() < 0.5, "camera {lat}");
        assert!(heading < 1.0 || heading > 359.0, "heading {heading}");
    }

    #[test]
    fn strongest_aurora_is_taken_only_in_the_dark() {
        let (width, height) = (1440, 720);
        let mut fields = vec![0u8; width * height * 4];
        let mut paint = |lat: f32, lon: f32, value: u8| {
            let row = ((90.0 - lat) / 180.0 * height as f32) as usize;
            let column = ((lon + 180.0) / 360.0 * width as f32) as usize;
            for r in row - 4..row + 4 {
                for c in column - 4..column + 4 {
                    fields[(r * width + c) * 4 + 3] = value;
                }
            }
        };
        // A strong oval over sunlit Alaska and a weaker one over dark Norway.
        paint(65.0, -150.0, 250);
        paint(68.0, 18.0, 120);
        let sun = { let (lat, lon) = (0.0_f32, -150.0_f32.to_radians()); [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()] };
        let (lat, lon, probability) = strongest_dark_aurora(&fields, width, height, sun).unwrap();
        assert!((lat - 68.0).abs() < 1.5 && (lon - 18.0).abs() < 1.5, "{lat} {lon}");
        assert!((probability - 120.0 / 255.0).abs() < 0.02);
    }
    #[test]
    fn weather_age_does_not_disguise_stale_or_future_data() {
        let state = WeatherState { source: "NOAA-GFS".into(), kind: "model-analysis".into(),
            valid_unix_utc: 100000, aurora_unix_utc: 0, texture: PathBuf::new(), sha256: "a".repeat(64), width: 1440, height: 720,
            clouds_texture: None, clouds_sha256: None, clouds_unix_utc: 0, aerosol_unix_utc: 0, sea_ice_unix_utc: 0 };
        assert!(state.validate().is_ok());
        assert_eq!(state.age_label(100000 + 6 * 3600), "current-model");
        assert_eq!(state.age_label(100000 + 19 * 3600), "stale");
        assert_eq!(state.age_label(90000), "future");
        let mut invalid = state;
        invalid.width = 16000;
        assert!(invalid.validate().is_err());
    }
}
