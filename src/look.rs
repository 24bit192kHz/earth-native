//! Presentation look of the Earth view.
//!
//! `realistic` is one physical camera: a single exposure for the whole frame,
//! so the Sun in view (or the daylit Earth) hides the stars, and the Sun
//! carries its full lens glare. `cinematic` shows the scene as the adapted
//! eye would want it on a desktop: the Milky Way and the catalogue stars are
//! always visible at a fixed brightness (in colour), even next to the daylit
//! Earth, and the Sun is a soft glow with faint rays.
//!
//! The startup look is `$EARTH_NATIVE_LOOK`, else the last one chosen with
//! `earth-native look` (saved in `$XDG_STATE_HOME/earth-native/look`), else
//! realistic.

use std::{env, fs, io, path::PathBuf};

pub const LOOK_ENV: &str = "EARTH_NATIVE_LOOK";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Look {
    #[default]
    Realistic,
    Cinematic,
}

impl Look {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "realistic" => Some(Self::Realistic),
            "cinematic" => Some(Self::Cinematic),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Realistic => "realistic",
            Self::Cinematic => "cinematic",
        }
    }

    fn state_path() -> Option<PathBuf> {
        env::var_os("XDG_STATE_HOME")
            .filter(|dir| !dir.is_empty())
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state")))
            .map(|dir| dir.join("earth-native").join("look"))
    }

    pub fn startup() -> Self {
        if let Some(look) = env::var(LOOK_ENV).ok().as_deref().and_then(Self::parse) {
            return look;
        }
        Self::state_path()
            .and_then(|path| fs::read_to_string(path).ok())
            .as_deref()
            .and_then(Self::parse)
            .unwrap_or_default()
    }

    /// Remember the look for the next start.
    pub fn save(self) -> io::Result<()> {
        let path = Self::state_path().ok_or_else(|| io::Error::other("no XDG_STATE_HOME or HOME"))?;
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, format!("{}\n", self.name()))
    }
}

#[cfg(test)]
mod tests {
    use super::Look;

    #[test]
    fn names_round_trip() {
        for look in [Look::Realistic, Look::Cinematic] {
            assert_eq!(Look::parse(look.name()), Some(look));
        }
        assert_eq!(Look::parse(" Cinematic\n"), Some(Look::Cinematic));
        assert_eq!(Look::parse("pretty"), None);
    }
}
