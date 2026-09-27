//! Switchable rendered body.
//!
//! The renderer shows Earth (the full production pipeline), the Moon, or any
//! other planet as a textured, physically lit globe. The body is chosen at
//! startup from `EARTH_NATIVE_BODY` and may be switched live over IPC
//! (`earth-native body NAME`) or with Ctrl+Left/Right. Switching is a
//! per-frame concern: the draw path and the fragment shaders branch on the
//! current body, so Earth stays fully intact while another body renders.

use std::env;

/// The body currently being rendered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Body {
    /// Full Earth pipeline (procedural or textured).
    #[default]
    Earth,
    Jupiter,
    Mercury,
    Mars,
    /// Gas giant with its main rings.
    Saturn,
    Venus,
    Uranus,
    Neptune,
    /// Earth's Moon as a selectable body (the sky Moon is always drawn).
    Moon,
}

impl Body {
    /// Every body, indexed by `id()` (the shader selector).
    pub const ALL: [Self; 9] = [
        Self::Earth,
        Self::Jupiter,
        Self::Mercury,
        Self::Mars,
        Self::Saturn,
        Self::Venus,
        Self::Uranus,
        Self::Neptune,
        Self::Moon,
    ];
    pub const COUNT: usize = Self::ALL.len();

    /// Interactive cycling order: outward from the Sun, the Moon after Earth.
    pub const TOUR: [Self; 9] = [
        Self::Mercury,
        Self::Venus,
        Self::Earth,
        Self::Moon,
        Self::Mars,
        Self::Jupiter,
        Self::Saturn,
        Self::Uranus,
        Self::Neptune,
    ];

    /// NASA/JPL planetary physical parameters, equatorial radius in km.
    pub const fn equatorial_radius_km(self) -> f64 {
        match self {
            Self::Earth => crate::astronomy::EARTH_EQUATORIAL_RADIUS_KM,
            Self::Jupiter => 71_492.0,
            Self::Mercury => 2_439.7,
            Self::Mars => 3_396.19,
            Self::Saturn => 60_268.0,
            Self::Venus => 6_051.8,
            Self::Uranus => 25_559.0,
            Self::Neptune => 24_764.0,
            Self::Moon => 1_737.4,
        }
    }

    /// Parse a body name (case-insensitive).
    pub fn parse(value: &str) -> Option<Self> {
        let value = value.to_ascii_lowercase();
        Self::ALL.into_iter().find(|body| body.name() == value)
    }

    /// Canonical CLI/log name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Earth => "earth",
            Self::Jupiter => "jupiter",
            Self::Mercury => "mercury",
            Self::Mars => "mars",
            Self::Saturn => "saturn",
            Self::Venus => "venus",
            Self::Uranus => "uranus",
            Self::Neptune => "neptune",
            Self::Moon => "moon",
        }
    }

    /// Fragment-shader selector: an integer body id carried in the
    /// otherwise-unused `.w` of the shared `sun_direction` push constant so
    /// the layout stays identical across every pipeline. Equal to the index
    /// in `ALL`: 0 Earth, 1 Jupiter, 2 Mercury, 3 Mars, 4 Saturn, 5 Venus,
    /// 6 Uranus, 7 Neptune, 8 Moon.
    pub fn id(self) -> f32 {
        Self::ALL.iter().position(|body| *body == self).unwrap_or(0) as f32
    }
}

/// Environment variable selecting the startup body.
pub const BODY_ENV: &str = "EARTH_NATIVE_BODY";

/// Resolve the startup body from `EARTH_NATIVE_BODY`, defaulting to Earth.
pub fn from_environment() -> Body {
    env::var(BODY_ENV)
        .ok()
        .and_then(|value| Body::parse(&value))
        .unwrap_or(Body::Earth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_is_case_insensitive_and_closed() {
        assert_eq!(Body::parse("earth"), Some(Body::Earth));
        assert_eq!(Body::parse("EARTH"), Some(Body::Earth));
        for body in Body::ALL {
            assert_eq!(Body::parse(body.name()), Some(body));
        }
        assert_eq!(Body::parse("pluto"), None);
        assert_eq!(Body::parse(""), None);
    }

    #[test]
    fn ids_are_distinct_sequential_and_tour_covers_all() {
        for (index, body) in Body::ALL.iter().enumerate() {
            assert_eq!(body.id(), index as f32);
            assert!(Body::TOUR.contains(body));
        }
    }
}
