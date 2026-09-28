//! One-second ISS propagation with asynchronous six-hour TLE refreshes.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::mpsc::{self, Receiver},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::{camera::Vec3, sgp4::IssOrbit};

const ISS_TLE_URL: &str = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE";
const TLE_REFRESH_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);
const TLE_STALE_AGE_SECONDS: i64 = 48 * 60 * 60;
const FALLBACK_TLE: &str = include_str!("../assets/iss.tle");

pub struct IssTracker {
    orbit: Option<IssOrbit>,
    cache_path: PathBuf,
    refresh_receiver: Option<Receiver<Result<PathBuf, String>>>,
    last_refresh_attempt: Instant,
    sampled_second: Option<i64>,
    previous_direction: Vec3,
    next_direction: Vec3,
    last_error: Option<String>,
    tle_epoch_unix_seconds: Option<i64>,
}

impl IssTracker {
    pub fn new() -> Self {
        let cache_path = tle_cache_path();
        let now = Instant::now();
        // CelesTrak blocks clients that poll too often, so the throttle must
        // survive restarts: it counts from the last attempt (successful or
        // not), recorded by the cache file and a stamp file, never from the
        // element epoch (which is hours old even right after a download).
        let since_last_attempt = [cache_path.clone(), attempt_stamp_path(&cache_path)]
            .iter()
            .filter_map(|path| fs::metadata(path).ok()?.modified().ok())
            .filter_map(|modified| SystemTime::now().duration_since(modified).ok())
            .min();

        let mut tracker = Self {
            orbit: None,
            cache_path,
            refresh_receiver: None,
            // Match the Unreal launcher: do not replace a valid shared cache
            // merely because the native renderer has started.
            last_refresh_attempt: match since_last_attempt {
                Some(elapsed) if elapsed < TLE_REFRESH_INTERVAL => now.checked_sub(elapsed).unwrap_or(now),
                _ => now.checked_sub(TLE_REFRESH_INTERVAL).unwrap_or(now),
            },
            sampled_second: None,
            previous_direction: Vec3::X,
            next_direction: Vec3::X,
            last_error: None,
            tle_epoch_unix_seconds: None,
        };
        tracker.load_best_available_tle();
        tracker.request_refresh_if_due();
        tracker
    }

    pub fn update_unix_utc(&mut self, seconds: i64, microseconds: i32) -> Option<Vec3> {
        self.poll_refresh();
        self.request_refresh_if_due();
        if !(0..1_000_000).contains(&microseconds) {
            self.last_error = Some("Unix microseconds must be in [0,1000000)".to_owned());
            return None;
        }
        if self.sampled_second != Some(seconds) {
            self.sample_directions(seconds);
        }
        self.sampled_second.map(|_| {
            let fraction = microseconds as f32 / 1_000_000.0;
            slerp(self.previous_direction, self.next_direction, fraction)
        })
    }

    /// ISS position (scene units, the renderer's longitude-mirrored Earth
    /// frame) and unit velocity at this instant, for the onboard camera. The
    /// radius is the scene sphere plus the true height above the ellipsoid,
    /// so horizon dip and ground scale match what the crew sees.
    pub fn onboard_state(&mut self, seconds: i64, microseconds: i32) -> Option<IssState> {
        self.poll_refresh();
        self.request_refresh_if_due();
        let orbit = self.orbit.as_mut()?;
        let sample = |orbit: &mut IssOrbit, offset_us: i64| -> Option<([f64; 3], f64)> {
            let total = seconds as i128 * 1_000_000 + microseconds as i128 + offset_us as i128;
            let (s, us) = (total.div_euclid(1_000_000) as i64, total.rem_euclid(1_000_000) as i32);
            let position = orbit.propagate_unix_utc(s, us).ok()?;
            let latitude = geocentric_latitude(position.latitude_radians, position.altitude_kilometres);
            let radius_km = crate::sky::GROUND_KM + position.altitude_kilometres;
            let direction = scene_direction(latitude, position.longitude_radians);
            Some((direction.map(|v| v * radius_km), position.altitude_kilometres))
        };
        let (before, _) = sample(orbit, -500_000)?;
        let (now, altitude_km) = sample(orbit, 0)?;
        let (after, _) = sample(orbit, 500_000)?;
        let scale = f64::from(crate::camera::SCENE_EARTH_RADIUS) / crate::sky::GROUND_KM;
        let velocity = [0, 1, 2].map(|c| after[c] - before[c]);
        let speed = velocity.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0e-9);
        Some(IssState {
            position: Vec3::new((now[0] * scale) as f32, (now[1] * scale) as f32, (now[2] * scale) as f32),
            velocity: Vec3::new((velocity[0] / speed) as f32, (velocity[1] / speed) as f32, (velocity[2] / speed) as f32),
            altitude_km,
            ground_speed_km_s: speed,
        })
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn tle_epoch_unix_seconds(&self) -> Option<i64> {
        self.tle_epoch_unix_seconds
    }

    pub fn tle_age_seconds(&self, unix_seconds: i64) -> Option<i64> {
        self.tle_epoch_unix_seconds
            .map(|epoch| unix_seconds.saturating_sub(epoch))
    }

    pub fn tle_is_stale(&self, unix_seconds: i64) -> bool {
        self.tle_age_seconds(unix_seconds)
            .map_or(true, |age| age.abs() > TLE_STALE_AGE_SECONDS)
    }

    fn load_best_available_tle(&mut self) {
        let cache_path = self.cache_path.clone();
        let cached = fs::read_to_string(&cache_path).ok().and_then(|content| {
            let epoch = parse_tle(&content)?.epoch_unix_seconds;
            Some((content, epoch))
        });
        let fallback =
            parse_tle(FALLBACK_TLE).map(|tle| (FALLBACK_TLE.to_owned(), tle.epoch_unix_seconds));
        let selected = match (cached, fallback) {
            (Some(cached), Some(fallback)) if cached.1 >= fallback.1 => Some(cached),
            (Some(_), Some(fallback)) => Some(fallback),
            (Some(cached), None) => Some(cached),
            (None, Some(fallback)) => Some(fallback),
            (None, None) => None,
        };
        match selected {
            Some((content, _)) => {
                self.load_tle_content(&content);
            }
            None => {
                self.last_error =
                    Some("no valid cached or embedded ISS TLE is available".to_owned());
            }
        }
    }

    fn load_tle_path(&mut self, path: &Path) -> bool {
        let content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(error) => {
                self.last_error = Some(format!("could not read {}: {error}", path.display()));
                return false;
            }
        };
        self.load_tle_content(&content)
    }

    fn load_tle_content(&mut self, content: &str) -> bool {
        let Some(tle) = parse_tle(content) else {
            self.last_error = Some("ISS TLE content is invalid".to_owned());
            return false;
        };
        let mut orbit = match IssOrbit::new() {
            Ok(orbit) => orbit,
            Err(error) => {
                self.last_error = Some(error);
                return false;
            }
        };
        match orbit.load_tle(tle.name, tle.line_one, tle.line_two) {
            Ok(()) => {
                self.orbit = Some(orbit);
                self.sampled_second = None;
                self.tle_epoch_unix_seconds = Some(tle.epoch_unix_seconds);
                self.last_error = None;
                true
            }
            Err(error) => {
                self.last_error = Some(format!("could not load ISS TLE: {error}"));
                false
            }
        }
    }

    fn sample_directions(&mut self, seconds: i64) {
        let Some(orbit) = self.orbit.as_mut() else {
            return;
        };
        let current = orbit.propagate_unix_utc(seconds, 0);
        let next = orbit.propagate_unix_utc(seconds.saturating_add(1), 0);
        match (current, next) {
            (Ok(current), Ok(next)) => {
                self.previous_direction =
                    asset_direction(geocentric_latitude(current.latitude_radians, current.altitude_kilometres), current.longitude_radians);
                self.next_direction =
                    asset_direction(geocentric_latitude(next.latitude_radians, next.altitude_kilometres), next.longitude_radians);
                self.sampled_second = Some(seconds);
                self.last_error = None;
            }
            (Err(error), _) | (_, Err(error)) => self.last_error = Some(error),
        }
    }

    fn request_refresh_if_due(&mut self) {
        if self.refresh_receiver.is_some()
            || self.last_refresh_attempt.elapsed() < TLE_REFRESH_INTERVAL
        {
            return;
        }
        self.last_refresh_attempt = Instant::now();
        let cache_path = self.cache_path.clone();
        if let Some(parent) = cache_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(attempt_stamp_path(&cache_path), b"");
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let _ = sender.send(refresh_tle_cache(&cache_path));
        });
        self.refresh_receiver = Some(receiver);
    }

    fn poll_refresh(&mut self) {
        let Some(receiver) = self.refresh_receiver.as_ref() else {
            return;
        };
        let result = match receiver.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("TLE refresh worker disconnected".to_owned())
            }
        };
        self.refresh_receiver = None;
        match result {
            Ok(path) => {
                self.load_tle_path(&path);
            }
            Err(error) => self.last_error = Some(error),
        }
    }
}

fn tle_cache_path() -> PathBuf {
    let base = env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    // Atomically refreshed from Celestrak at most once per cache lifetime.
    base.join("earth-native").join("iss.tle")
}

#[cfg(test)]
fn best_tle_epoch(cache_path: &Path) -> Option<i64> {
    let cached = fs::read_to_string(cache_path)
        .ok()
        .and_then(|content| parse_tle(&content).map(|tle| tle.epoch_unix_seconds));
    let fallback = parse_tle(FALLBACK_TLE).map(|tle| tle.epoch_unix_seconds);
    cached.into_iter().chain(fallback).max()
}

fn attempt_stamp_path(cache_path: &Path) -> PathBuf {
    cache_path.with_extension("last-attempt")
}

fn refresh_tle_cache(cache_path: &Path) -> Result<PathBuf, String> {
    let parent = cache_path.parent().ok_or("TLE cache path has no parent")?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let temporary = cache_path.with_extension(format!("tmp.{}", std::process::id()));
    let status = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "5",
            "--user-agent",
            concat!("earth-native/", env!("CARGO_PKG_VERSION"), " (+https://github.com/24bit192kHz/earth-native)"),
            ISS_TLE_URL,
            "--output",
        ])
        .arg(&temporary)
        .status()
        .map_err(|error| format!("could not start curl: {error}"))?;
    if !status.success() {
        let _ = fs::remove_file(&temporary);
        return Err("Celestrak TLE request failed; keeping cached fallback".to_owned());
    }
    let content = fs::read_to_string(&temporary).map_err(|error| error.to_string())?;
    if !is_valid_iss_tle(&content) {
        let _ = fs::remove_file(&temporary);
        return Err("Celestrak response did not contain a valid ISS TLE".to_owned());
    }
    fs::rename(&temporary, cache_path).map_err(|error| error.to_string())?;
    Ok(cache_path.to_owned())
}

fn is_valid_iss_tle(content: &str) -> bool {
    parse_tle(content).is_some()
}

struct ParsedTle<'a> {
    name: &'a str,
    line_one: &'a str,
    line_two: &'a str,
    epoch_unix_seconds: i64,
}

fn parse_tle(content: &str) -> Option<ParsedTle<'_>> {
    let mut name = "ISS (ZARYA)";
    let mut line_one = None;
    let mut line_two = None;
    for line in content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        if line.starts_with("1 25544") && line.len() == 69 {
            line_one = Some(line);
        } else if line.starts_with("2 25544") && line.len() == 69 && line_one.is_some() {
            line_two = Some(line);
            break;
        } else if line_one.is_none() {
            name = line;
        }
    }
    let line_one = line_one?;
    let line_two = line_two?;
    if !tle_checksum_valid(line_one) || !tle_checksum_valid(line_two) {
        return None;
    }
    let epoch = line_one.get(18..32)?;
    let short_year = epoch.get(0..2)?.parse::<i32>().ok()?;
    let day_of_year = epoch.get(2..)?.parse::<f64>().ok()?;
    if !(1.0..367.0).contains(&day_of_year) {
        return None;
    }
    let year = if short_year >= 57 {
        1900 + short_year
    } else {
        2000 + short_year
    };
    let epoch_unix_seconds = days_from_civil(year, 1, 1)
        .checked_mul(86_400)?
        .checked_add(((day_of_year - 1.0) * 86_400.0).round() as i64)?;
    Some(ParsedTle {
        name,
        line_one,
        line_two,
        epoch_unix_seconds,
    })
}

fn tle_checksum_valid(line: &str) -> bool {
    let bytes = line.as_bytes();
    if bytes.len() != 69 || !bytes[68].is_ascii_digit() {
        return false;
    }
    let checksum = bytes[..68].iter().fold(0_u32, |sum, byte| {
        sum + if byte.is_ascii_digit() {
            u32::from(byte - b'0')
        } else if *byte == b'-' {
            1
        } else {
            0
        }
    }) % 10;
    checksum == u32::from(bytes[68] - b'0')
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = i64::from(year) - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month = i64::from(month);
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

pub(crate) fn unix_utc_timestamp(now: SystemTime) -> Option<(i64, i32)> {
    let total_nanoseconds = match now.duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            i128::from(duration.as_secs()) * 1_000_000_000 + i128::from(duration.subsec_nanos())
        }
        Err(error) => {
            let duration = error.duration();
            -(i128::from(duration.as_secs()) * 1_000_000_000 + i128::from(duration.subsec_nanos()))
        }
    };
    let total_microseconds = total_nanoseconds.div_euclid(1_000);
    let seconds = total_microseconds.div_euclid(1_000_000);
    let microseconds = total_microseconds.rem_euclid(1_000_000);
    Some((i64::try_from(seconds).ok()?, microseconds as i32))
}

// SGP4 reports WGS72 geodetic latitude and height. Convert the actual satellite
// position, including altitude, before using its geocentric radial direction.
fn geocentric_latitude(latitude: f64, altitude_km: f64) -> f64 {
    let flattening = 1.0 / 298.26;
    let eccentricity_sq = flattening * (2.0 - flattening);
    let n = 6378.135 / (1.0 - eccentricity_sq * latitude.sin().powi(2)).sqrt();
    ((n * (1.0 - eccentricity_sq) + altitude_km) * latitude.sin())
        .atan2((n + altitude_km) * latitude.cos())
}

/// Unit vector of a geocentric latitude/longitude in the renderer's Earth
/// frame: ECEF with Y negated (the equirectangular maps grow U westward).
pub fn scene_direction(latitude: f64, longitude: f64) -> [f64; 3] {
    [latitude.cos() * longitude.cos(), -latitude.cos() * longitude.sin(), latitude.sin()]
}

#[derive(Clone, Copy, Debug)]
pub struct IssState {
    pub position: Vec3,
    pub velocity: Vec3,
    pub altitude_km: f64,
    /// Inertial speed relative to the rotating Earth, km/s.
    pub ground_speed_km_s: f64,
}

fn asset_direction(latitude: f64, longitude: f64) -> Vec3 {
    let cosine_latitude = latitude.cos();
    let asset_longitude = longitude + std::f64::consts::PI;
    // OrbitCamera's calibrated 180-degree yaw negates the horizontal axes
    // before deriving its position. Mirror latitude here so its final camera
    // position matches Unreal's ISS-orbit convention instead of reflecting
    // the view across the equator.
    Vec3::new(
        (cosine_latitude * asset_longitude.cos()) as f32,
        (cosine_latitude * asset_longitude.sin()) as f32,
        -latitude.sin() as f32,
    )
    .normalized()
}

fn slerp(from: Vec3, to: Vec3, fraction: f32) -> Vec3 {
    let from = from.normalized();
    let to = to.normalized();
    let dot = from.dot(to).clamp(-1.0, 1.0);
    if dot > 0.9995 {
        return (from * (1.0 - fraction) + to * fraction).normalized();
    }
    let angle = dot.acos();
    let sine = angle.sin();
    ((from * ((1.0 - fraction) * angle).sin() + to * (fraction * angle).sin()) / sine).normalized()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onboard_state_is_at_iss_altitude_and_orbital_speed() {
        let mut tracker = IssTracker::new();
        let epoch = tracker.tle_epoch_unix_seconds().expect("embedded TLE");
        let state = tracker.onboard_state(epoch + 600, 250_000).expect("state");
        let radius_km = state.position.length() as f64 / f64::from(crate::camera::SCENE_EARTH_RADIUS) * crate::sky::GROUND_KM;
        assert!((radius_km - crate::sky::GROUND_KM - state.altitude_km).abs() < 0.5);
        assert!((380.0..460.0).contains(&state.altitude_km), "{}", state.altitude_km);
        // ~7.66 km/s inertial, minus up to ~0.45 km/s of Earth rotation.
        assert!((7.0..7.8).contains(&state.ground_speed_km_s), "{}", state.ground_speed_km_s);
        assert!(state.velocity.dot(state.position.normalized()).abs() < 0.01);
    }

    #[test]
    fn fallback_tle_is_valid() {
        assert!(is_valid_iss_tle(FALLBACK_TLE));
    }

    #[test]
    fn tle_cache_matches_the_unreal_wallpaper_cache() {
        assert!(tle_cache_path().ends_with("earth-native/iss.tle"));
    }

    #[test]
    fn cache_selection_uses_tle_epoch_instead_of_file_mtime() {
        let path = std::env::temp_dir().join(format!(
            "earth-native-tle-epoch-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos()
        ));
        fs::write(&path, FALLBACK_TLE).expect("write temporary TLE cache");
        let expected = parse_tle(FALLBACK_TLE).unwrap().epoch_unix_seconds;
        assert_eq!(best_tle_epoch(&path), Some(expected));
        fs::remove_file(path).expect("remove temporary TLE cache");
    }

    #[test]
    fn tle_epoch_and_checksum_are_validated() {
        let parsed = parse_tle(FALLBACK_TLE).expect("embedded ISS TLE");
        assert!(parsed.epoch_unix_seconds > 1_767_225_600); // 2026-01-01 UTC
        let mut invalid = FALLBACK_TLE.to_owned();
        let index = invalid.find("9995").unwrap() + 3;
        invalid.replace_range(index..=index, "0");
        assert!(parse_tle(&invalid).is_none());
    }

    #[test]
    fn interpolation_stays_on_the_unit_sphere() {
        let direction = slerp(Vec3::X, Vec3::Y, 0.5);
        assert!((direction.length() - 1.0).abs() < 0.0001);
        assert!(direction.x > 0.7 && direction.y > 0.7);
    }

    #[test]
    fn system_time_conversion_uses_shared_floor_based_unix_components() {
        assert_eq!(unix_utc_timestamp(UNIX_EPOCH), Some((0, 0)));
        assert_eq!(
            unix_utc_timestamp(UNIX_EPOCH + Duration::new(12, 345_678_999)),
            Some((12, 345_678))
        );
        assert_eq!(
            unix_utc_timestamp(UNIX_EPOCH - Duration::from_micros(1)),
            Some((-1, 999_999))
        );
        assert_eq!(
            unix_utc_timestamp(UNIX_EPOCH - Duration::from_nanos(1)),
            Some((-1, 999_999))
        );
    }

    #[test]
    fn asset_latitude_compensates_the_calibrated_camera_yaw() {
        let mut camera = crate::camera::OrbitCamera::new(1.0);
        assert!(camera.set_base_orbit_direction(asset_direction(std::f64::consts::FRAC_PI_2, 0.0,)));
        // A positive ISS latitude must put the final orbit camera above the
        // equator, matching Unreal's camera-location convention.
        assert!(camera.pose().position.z > 0.999);
    }

    #[test]
    fn iss_pitch_uses_the_unreal_camera_location_sign() {
        let mut camera = crate::camera::OrbitCamera::new(1.0);
        assert!(camera.set_base_orbit_direction(asset_direction(0.0, 0.0)));
        assert!(camera.set_orbit_angles(180.0, 30.0));

        // Unreal rotates the ISS radial vector around +Z cross the vector.
        // At longitude zero this moves a positive pitch toward negative Z.
        let position = camera.pose().position;
        assert!(
            (position.x + 5.5 * 30.0_f32.to_radians().cos()).abs() < 0.001,
            "unexpected ISS camera position: {position:?}"
        );
        assert!(position.z < -0.49);
    }

    #[test]
    fn iss_pose_matches_unreal_for_nontrivial_orbit_angles() {
        fn rotate(vector: Vec3, axis: Vec3, radians: f32) -> Vec3 {
            let axis = axis.normalized();
            let (sine, cosine) = radians.sin_cos();
            vector * cosine + axis.cross(vector) * sine + axis * (axis.dot(vector) * (1.0 - cosine))
        }

        let latitude = 0.37_f64;
        let longitude = -0.91_f64;
        let manual_yaw = 17.0_f32;
        let manual_pitch = 23.0_f32;
        let cosine_latitude = latitude.cos() as f32;
        let unreal_iss = Vec3::new(
            cosine_latitude * (longitude + std::f64::consts::PI).cos() as f32,
            cosine_latitude * (longitude + std::f64::consts::PI).sin() as f32,
            latitude.sin() as f32,
        );
        let yawed = rotate(unreal_iss, Vec3::Z, manual_yaw.to_radians());
        let expected_radial = rotate(
            yawed,
            Vec3::Z.cross(yawed).normalized(),
            manual_pitch.to_radians(),
        );

        let mut camera = crate::camera::OrbitCamera::new(1.0);
        assert!(camera.set_base_orbit_direction(asset_direction(latitude, longitude)));
        assert!(camera.set_orbit_angles(manual_yaw + 180.0, manual_pitch));
        let pose = camera.pose();

        let expected_position = expected_radial * 5.5;
        assert!((pose.position.x - expected_position.x).abs() < 0.001);
        assert!((pose.position.y - expected_position.y).abs() < 0.001);
        assert!((pose.position.z - expected_position.z).abs() < 0.001);
        assert!((pose.forward.x + expected_radial.x).abs() < 0.001);
        assert!((pose.forward.y + expected_radial.y).abs() < 0.001);
        assert!((pose.forward.z + expected_radial.z).abs() < 0.001);
    }

    #[test]
    fn sgp4_geodetic_conversion_includes_satellite_altitude() {
        let phi = std::f64::consts::FRAC_PI_4;
        let surface = geocentric_latitude(phi, 0.0);
        let iss = geocentric_latitude(phi, 420.0);
        assert!((surface.to_degrees() - 44.80758).abs() < 0.0001);
        assert!(surface < iss && iss < phi);
        assert_eq!(geocentric_latitude(0.0, 420.0), 0.0);
    }
}
