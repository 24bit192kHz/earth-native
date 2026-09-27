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
        Ok(Some(state))
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weather_age_does_not_disguise_stale_or_future_data() {
        let state = WeatherState { source: "NOAA-GFS".into(), kind: "model-analysis".into(),
            valid_unix_utc: 100000, aurora_unix_utc: 0, texture: PathBuf::new(), sha256: "a".repeat(64), width: 1440, height: 720 };
        assert!(state.validate().is_ok());
        assert_eq!(state.age_label(100000 + 6 * 3600), "current-model");
        assert_eq!(state.age_label(100000 + 19 * 3600), "stale");
        assert_eq!(state.age_label(90000), "future");
        let mut invalid = state;
        invalid.width = 16000;
        assert!(invalid.validate().is_err());
    }
}
