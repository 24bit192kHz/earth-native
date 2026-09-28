//! Locates the installed texture pack and exports the per-map variables the
//! loaders read, so `earth-native start` runs without a launcher script.
//!
//! Search order: `$EARTH_NATIVE_DATA_DIR`, `$XDG_DATA_HOME/earth-native`,
//! `~/.local/share/earth-native`, `<exe>/../share/earth-native`,
//! `/usr/local/share/earth-native`, `/usr/share/earth-native`. A data dir
//! holds `active.json` (naming its texture set, default `textures`) and an
//! optional `weather/current.json` written by the weather pipeline. Variables
//! that are already set are never overridden.

use std::{
    env,
    path::{Path, PathBuf},
};

/// (variable, file stem, extensions in preference order)
const MAPS: &[(&str, &str, &[&str])] = &[
    ("EARTH_NATIVE_DAY_COLOR_EAST", "day-east", &["bc3", "bgra"]),
    ("EARTH_NATIVE_DAY_COLOR_WEST", "day-west", &["bc3", "bgra"]),
    ("EARTH_NATIVE_NIGHT_EMISSION", "night", &["bc4", "bgra"]),
    ("EARTH_NATIVE_SURFACE_NORMAL_EAST", "normal-east", &["bgra"]),
    ("EARTH_NATIVE_SURFACE_NORMAL_WEST", "normal-west", &["bgra"]),
    ("EARTH_NATIVE_HEIGHT", "height", &["bc4", "bgra"]),
    ("EARTH_NATIVE_RELIEF", "relief", &["bc5"]),
    ("EARTH_NATIVE_MOON", "moon", &["bc1", "bgra"]),
    ("EARTH_NATIVE_MERCURY", "mercury", &["bc1", "bgra"]),
    ("EARTH_NATIVE_VENUS", "venus", &["bc1", "bgra"]),
    ("EARTH_NATIVE_MARS", "mars", &["bc1", "bgra"]),
    ("EARTH_NATIVE_JUPITER", "jupiter", &["bc1", "bgra"]),
    ("EARTH_NATIVE_SATURN", "saturn", &["bc1", "bgra"]),
    ("EARTH_NATIVE_SATURN_RING", "saturn-ring", &["bgra"]),
    ("EARTH_NATIVE_URANUS", "uranus", &["bc1", "bgra"]),
    ("EARTH_NATIVE_NEPTUNE", "neptune", &["bc1", "bgra"]),
    ("EARTH_NATIVE_CLOUDS_A", "clouds", &["bc4"]),
    ("EARTH_NATIVE_CLOUDS_BA", "cloud-empty", &["bgra"]),
    ("EARTH_NATIVE_DESERT_CLOUD_MASK", "desert", &["bgra"]),
    ("EARTH_NATIVE_CLOUD_HEIGHT", "cloud-empty", &["bgra"]),
    ("EARTH_NATIVE_TILING_NOISE", "tiling-noise", &["bgra"]),
    ("EARTH_NATIVE_STAR_PANORAMA", "stars", &["bc1", "bgra"]),
];

fn candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(dir) = env::var_os("EARTH_NATIVE_DATA_DIR") {
        paths.push(PathBuf::from(dir));
    }
    if let Some(dir) = env::var_os("XDG_DATA_HOME") {
        paths.push(PathBuf::from(dir).join("earth-native"));
    }
    if let Some(home) = env::var_os("HOME") {
        paths.push(PathBuf::from(home).join(".local/share/earth-native"));
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(prefix) = exe.parent().and_then(Path::parent) {
            paths.push(prefix.join("share/earth-native"));
        }
    }
    paths.push(PathBuf::from("/usr/local/share/earth-native"));
    paths.push(PathBuf::from("/usr/share/earth-native"));
    paths
}

fn texture_dir(data_dir: &Path) -> PathBuf {
    let name = std::fs::read(data_dir.join("active.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|value| value.get("textures")?.as_str().map(str::to_owned))
        .filter(|name| !name.contains('/') && !name.starts_with('.'))
        .unwrap_or_else(|| "textures".to_owned());
    data_dir.join(name)
}

fn with_sidecar(path: PathBuf) -> Option<PathBuf> {
    let mut sidecar = path.clone().into_os_string();
    sidecar.push(".json");
    (path.is_file() && Path::new(&sidecar).is_file()).then_some(path)
}

/// Export the texture pack's variables; returns the texture directory used.
pub fn configure_environment() -> Option<PathBuf> {
    if env::var_os("EARTH_NATIVE_NASA_DATA").is_some() {
        return None; // A launcher already configured everything explicitly.
    }
    let (data_dir, textures) = candidates().into_iter().find_map(|dir| {
        let textures = texture_dir(&dir);
        with_sidecar(textures.join("day-east.bc3"))
            .or_else(|| with_sidecar(textures.join("day-east.bgra")))
            .map(|_| (dir, textures))
    })?;
    // SAFETY (single-threaded): runs at startup before any thread is spawned.
    env::set_var("EARTH_NATIVE_NASA_DATA", &textures);
    let clouds_ok = with_sidecar(textures.join("clouds.bc4")).is_some();
    if clouds_ok && env::var_os("EARTH_NATIVE_CLOUDS_A").is_none() {
        env::set_var("EARTH_NATIVE_NASA_CLOUDS", "1");
    }
    for (variable, stem, extensions) in MAPS {
        if env::var_os(variable).is_some() {
            continue;
        }
        if let Some(path) = extensions
            .iter()
            .find_map(|extension| with_sidecar(textures.join(format!("{stem}.{extension}"))))
        {
            env::set_var(variable, path);
        }
    }
    // The 500 m Blue Marble virtual texture: the month's own composite
    // (snow, vegetation and sea ice follow the season) if baked, else the
    // nearest baked month, else any.
    if env::var_os("EARTH_NATIVE_EARTHVT").is_none() {
        let mut names: Vec<String> = months_nearest_first(current_month())
            .map(|month| format!("earth-day-{month:02}.earthvt"))
            .collect();
        names.push("earth-day.earthvt".to_owned());
        if let Some(path) = names.iter().flat_map(|name| [textures.join(name), data_dir.join("vt").join(name)])
            .find(|path| path.is_file())
        {
            env::set_var("EARTH_NATIVE_EARTHVT", path);
        }
    }
    // Point at the weather feed's manifest even before it exists, so a feed
    // started later is picked up by `earth-native weather reload`.
    if env::var_os("EARTH_NATIVE_WEATHER_MANIFEST").is_none() {
        env::set_var("EARTH_NATIVE_WEATHER_MANIFEST", data_dir.join("weather/current.json"));
    }
    Some(textures)
}

/// UTC calendar month (1-12) of the system clock.
fn current_month() -> u32 {
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs() / 86_400) as i64;
    // Civil-from-days (Howard Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    (if mp < 10 { mp + 3 } else { mp - 9 }) as u32
}

/// All twelve months ordered by seasonal distance from `month`: the month
/// itself, the next, the previous, two ahead, two back, and so on.
fn months_nearest_first(month: u32) -> impl Iterator<Item = u32> {
    (0..12i32).map(move |step| {
        let offset = if step % 2 == 1 { (step + 1) / 2 } else { -(step / 2) };
        (month as i32 - 1 + offset).rem_euclid(12) as u32 + 1
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn month_is_in_range() {
        assert!((1..=12).contains(&super::current_month()));
    }

    #[test]
    fn months_are_tried_nearest_first() {
        let order: Vec<u32> = super::months_nearest_first(1).collect();
        assert_eq!(order, [1, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7]);
        let mut sorted = super::months_nearest_first(9).collect::<Vec<_>>();
        sorted.sort();
        assert_eq!(sorted, (1..=12).collect::<Vec<_>>());
    }
}
