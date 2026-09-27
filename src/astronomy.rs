//! Offline astronomical state derived from Astronomy Engine.

use std::{error::Error, f64::consts::PI, fmt};

use astronomy_engine_bindings as ffi;

// IAU 2012 exact AU, WGS84 equatorial Earth radius, IAU nominal solar
// radius and NASA/NSSDC mean lunar radius (all in kilometres).
pub const ASTRONOMICAL_UNIT_KM: f64 = 149_597_870.7;
pub const EARTH_EQUATORIAL_RADIUS_KM: f64 = 6_378.137;
pub const SUN_RADIUS_KM: f64 = 695_700.0;
pub const MOON_RADIUS_KM: f64 = 1_737.4;

/// NASA SORCE/TSIS nominal total solar irradiance at 1 AU, inverse-square falloff.
pub fn solar_irradiance_w_m2(distance_km: f64) -> f64 {
    1361.0 * (ASTRONOMICAL_UNIT_KM / distance_km).powi(2)
}

#[test]
fn solar_flux_uses_inverse_square_si_units() {
    assert_eq!(solar_irradiance_w_m2(ASTRONOMICAL_UNIT_KM), 1361.0);
    assert_eq!(solar_irradiance_w_m2(2.0 * ASTRONOMICAL_UNIT_KM), 340.25);
}

const SECONDS_PER_DAY: i64 = 86_400;
const UNIX_SECONDS_AT_J2000: i64 = 946_728_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CelestialState {
    pub unix_seconds: i64,
    pub sun_direction: [f32; 3],
    pub sun_distance_earth_radii: f32,
    pub moon_direction: [f32; 3],
    pub moon_distance_earth_radii: f32,
    pub greenwich_apparent_sidereal_angle_radians: f32,
    pub moon_illuminated_fraction: f32,
    pub moon_phase_angle_radians: f32,
    /// Row-major matrix: `body_direction = moon_world_to_body * asset_direction`.
    /// Rows are the lunar +X prime-meridian, +Y east, and +Z north axes,
    /// expressed in Earth asset coordinates. The IAU periodic terms supplied by
    /// Astronomy Engine keep the frame body-fixed and include physical libration.
    pub moon_world_to_body: [[f32; 3]; 3],
    /// Sun direction as seen from each switchable body (index = body id,
    /// 0 = Earth) in the same Earth-fixed world frame as `sun_direction`.
    /// From another planet the Sun sits at a different place against the
    /// fixed stars (planetary parallax), so switching bodies moves the Sun
    /// glow and the lighting to the currently correct position.
    pub body_sun_directions: [[f32; 3]; crate::body::Body::COUNT],
    /// Sun distance from each body in Earth radii (index = body id), setting
    /// the Sun's apparent size per planet.
    pub body_sun_distances_earth_radii: [f32; crate::body::Body::COUNT],
    /// ECEF-to-body matrices from IAU poles and prime meridians. Earth uses
    /// ECEF directly; other planets rotate at their own sidereal rates.
    pub body_world_to_fixed: [[[f32; 3]; 3]; crate::body::Body::COUNT],
    pub sun_angular_radius_radians: f32,
    pub moon_angular_radius_radians: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AstronomyError {
    FfiStatus {
        operation: &'static str,
        status: i32,
    },
    NonFinite(&'static str),
    InvalidValue(&'static str),
}

impl fmt::Display for AstronomyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FfiStatus { operation, status } => {
                write!(
                    f,
                    "{operation} failed with Astronomy Engine status {status}"
                )
            }
            Self::NonFinite(value) => write!(f, "Astronomy Engine returned non-finite {value}"),
            Self::InvalidValue(value) => write!(f, "Astronomy Engine returned invalid {value}"),
        }
    }
}

impl Error for AstronomyError {}

pub fn celestial_state(unix_seconds: i64) -> Result<CelestialState, AstronomyError> {
    let mut time = time_from_unix_seconds(unix_seconds)?;

    let rotation = unsafe { ffi::Astronomy_Rotation_EQJ_EQD(&mut time) };
    check_status("Astronomy_Rotation_EQJ_EQD", rotation.status)?;
    if !rotation.rot.iter().flatten().all(|value| value.is_finite()) {
        return Err(AstronomyError::NonFinite("EQJ-to-EQD rotation"));
    }

    let sidereal_hours = unsafe { ffi::Astronomy_SiderealTime(&mut time) };
    check_finite("Greenwich apparent sidereal time", sidereal_hours)?;
    let sidereal_radians = (sidereal_hours * PI / 12.0).rem_euclid(2.0 * PI);

    let sun = geocentric_asset_vector(
        ffi::astro_body_t_BODY_SUN,
        "Sun",
        time,
        rotation,
        sidereal_radians,
    )?;
    let moon = geocentric_asset_vector(
        ffi::astro_body_t_BODY_MOON,
        "Moon",
        time,
        rotation,
        sidereal_radians,
    )?;

    let illumination = unsafe { ffi::Astronomy_Illumination(ffi::astro_body_t_BODY_MOON, time) };
    check_status("Astronomy_Illumination(Moon)", illumination.status)?;
    check_finite("Moon illuminated fraction", illumination.phase_fraction)?;
    check_finite("Moon phase angle", illumination.phase_angle)?;
    if !(0.0..=1.0).contains(&illumination.phase_fraction)
        || !(0.0..=180.0).contains(&illumination.phase_angle)
    {
        return Err(AstronomyError::InvalidValue("Moon illumination"));
    }

    let axis = unsafe { ffi::Astronomy_RotationAxis(ffi::astro_body_t_BODY_MOON, &mut time) };
    check_status("Astronomy_RotationAxis(Moon)", axis.status)?;
    check_status("Astronomy_RotationAxis(Moon).north", axis.north.status)?;
    for (name, value) in [
        ("Moon pole right ascension", axis.ra),
        ("Moon pole declination", axis.dec),
        ("Moon prime-meridian spin", axis.spin),
        ("Moon north x", axis.north.x),
        ("Moon north y", axis.north.y),
        ("Moon north z", axis.north.z),
    ] {
        check_finite(name, value)?;
    }
    let moon_world_to_body = lunar_world_to_body(axis, time, rotation, sidereal_radians)?;

    // Sun direction as seen from each switchable body. From another planet the
    // Sun sits at a different place against the fixed stars (planetary
    // parallax) and at a different distance, so switching bodies moves the
    // Sun glow and lighting to the currently correct position and size.
    const COUNT: usize = crate::body::Body::COUNT;
    let mut body_sun_directions = [[0.0_f32; 3]; COUNT];
    let mut body_sun_distances = [0.0_f32; COUNT];
    let mut body_world_to_fixed = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; COUNT];
    body_sun_directions[0] = to_f32_vector(sun.direction, "body sun direction")?;
    body_sun_distances[0] = to_f32(
        sun.distance_au * ASTRONOMICAL_UNIT_KM / EARTH_EQUATORIAL_RADIUS_KM,
        "body sun distance",
    )?;
    for (slot, body) in [
        (1, ffi::astro_body_t_BODY_JUPITER),
        (2, ffi::astro_body_t_BODY_MERCURY),
        (3, ffi::astro_body_t_BODY_MARS),
        (4, ffi::astro_body_t_BODY_SATURN),
        (5, ffi::astro_body_t_BODY_VENUS),
        (6, ffi::astro_body_t_BODY_URANUS),
        (7, ffi::astro_body_t_BODY_NEPTUNE),
        (8, ffi::astro_body_t_BODY_MOON),
    ] {
        let vector = body_sun_vector(body, time, rotation, sidereal_radians)?;
        let body_axis = unsafe { ffi::Astronomy_RotationAxis(body, &mut time) };
        check_status("planet rotation axis", body_axis.status)?;
        body_world_to_fixed[slot] = lunar_world_to_body(body_axis, time, rotation, sidereal_radians)?;
        body_sun_directions[slot] = to_f32_vector(vector.direction, "body sun direction")?;
        body_sun_distances[slot] = to_f32(
            vector.distance_au * ASTRONOMICAL_UNIT_KM / EARTH_EQUATORIAL_RADIUS_KM,
            "body sun distance",
        )?;
    }

    let sun_radius = angular_radius(SUN_RADIUS_KM, sun.distance_au, "Sun angular radius")?;
    let moon_radius = angular_radius(MOON_RADIUS_KM, moon.distance_au, "Moon angular radius")?;

    Ok(CelestialState {
        unix_seconds,
        sun_direction: to_f32_vector(sun.direction, "Sun direction")?,
        sun_distance_earth_radii: to_f32(
            sun.distance_au * ASTRONOMICAL_UNIT_KM / EARTH_EQUATORIAL_RADIUS_KM,
            "Sun distance",
        )?,
        moon_direction: to_f32_vector(moon.direction, "Moon direction")?,
        moon_distance_earth_radii: to_f32(
            moon.distance_au * ASTRONOMICAL_UNIT_KM / EARTH_EQUATORIAL_RADIUS_KM,
            "Moon distance",
        )?,
        greenwich_apparent_sidereal_angle_radians: to_f32(
            sidereal_radians,
            "Greenwich apparent sidereal angle",
        )?,
        moon_illuminated_fraction: to_f32(
            illumination.phase_fraction,
            "Moon illuminated fraction",
        )?,
        moon_phase_angle_radians: to_f32(
            illumination.phase_angle.to_radians(),
            "Moon phase angle",
        )?,
        moon_world_to_body,
        body_sun_directions,
        body_sun_distances_earth_radii: body_sun_distances,
        body_world_to_fixed,
        sun_angular_radius_radians: to_f32(sun_radius, "Sun angular radius")?,
        moon_angular_radius_radians: to_f32(moon_radius, "Moon angular radius")?,
    })
}

#[derive(Clone, Copy)]
struct AssetVector {
    direction: [f64; 3],
    distance_au: f64,
}

fn time_from_unix_seconds(unix_seconds: i64) -> Result<ffi::astro_time_t, AstronomyError> {
    let relative = i128::from(unix_seconds) - i128::from(UNIX_SECONDS_AT_J2000);
    let days = relative.div_euclid(i128::from(SECONDS_PER_DAY));
    let seconds = relative.rem_euclid(i128::from(SECONDS_PER_DAY));
    let ut_days = days as f64 + seconds as f64 / SECONDS_PER_DAY as f64;
    check_finite("Unix-to-J2000 day conversion", ut_days)?;

    let time = unsafe { ffi::Astronomy_TimeFromDays(ut_days) };
    if !time.ut.is_finite() || !time.tt.is_finite() {
        return Err(AstronomyError::NonFinite("astronomical time"));
    }
    Ok(time)
}

fn geocentric_asset_vector(
    body: ffi::astro_body_t,
    name: &'static str,
    time: ffi::astro_time_t,
    rotation: ffi::astro_rotation_t,
    sidereal_radians: f64,
) -> Result<AssetVector, AstronomyError> {
    let eqj = unsafe { ffi::Astronomy_GeoVector(body, time, ffi::astro_aberration_t_ABERRATION) };
    check_vector(name, &eqj)?;
    let distance_au = vector_length([eqj.x, eqj.y, eqj.z]);
    if !distance_au.is_finite() || distance_au <= 0.0 {
        return Err(AstronomyError::InvalidValue("geocentric distance"));
    }

    let eqd = unsafe { ffi::Astronomy_RotateVector(rotation, eqj) };
    check_vector("EQJ-to-EQD vector", &eqd)?;
    let ecef = eqd_to_ecef([eqd.x, eqd.y, eqd.z], sidereal_radians);
    let ecef_length = vector_length(ecef);
    if !ecef_length.is_finite() || ecef_length <= 0.0 {
        return Err(AstronomyError::InvalidValue("Earth-fixed direction"));
    }

    Ok(AssetVector {
        direction: ecef.map(|component| component / ecef_length),
        distance_au,
    })
}

/// Sun direction as seen from `body`, in the Earth-fixed world frame. The
/// planet-to-Sun vector is the negated heliocentric position of the body;
/// rotating it into the equator of date and then into ECEF puts it in the
/// same frame as `sun_direction`, so the Sun glow and the planet lighting
/// move to the body's currently correct sky position.
fn body_sun_vector(
    body: ffi::astro_body_t,
    time: ffi::astro_time_t,
    rotation: ffi::astro_rotation_t,
    sidereal_radians: f64,
) -> Result<AssetVector, AstronomyError> {
    let helio = unsafe { ffi::Astronomy_HelioVector(body, time) };
    check_vector("heliocentric body position", &helio)?;
    let distance_au = vector_length([helio.x, helio.y, helio.z]);
    if !distance_au.is_finite() || distance_au <= 0.0 {
        return Err(AstronomyError::InvalidValue("heliocentric distance"));
    }
    let to_sun = ffi::astro_vector_t {
        status: ffi::astro_status_t_ASTRO_SUCCESS,
        x: -helio.x,
        y: -helio.y,
        z: -helio.z,
        t: time,
    };
    let eqd = unsafe { ffi::Astronomy_RotateVector(rotation, to_sun) };
    check_vector("EQJ-to-EQD sun vector", &eqd)?;
    let ecef = eqd_to_ecef([eqd.x, eqd.y, eqd.z], sidereal_radians);
    let ecef_length = vector_length(ecef);
    if !ecef_length.is_finite() || ecef_length <= 0.0 {
        return Err(AstronomyError::InvalidValue("body sun direction"));
    }
    Ok(AssetVector {
        direction: ecef.map(|component| component / ecef_length),
        distance_au,
    })
}

fn lunar_world_to_body(
    axis: ffi::astro_axis_t,
    time: ffi::astro_time_t,
    rotation: ffi::astro_rotation_t,
    sidereal_radians: f64,
) -> Result<[[f32; 3]; 3], AstronomyError> {
    let ra = (axis.ra * 15.0).to_radians();
    let dec = axis.dec.to_radians();
    let spin = axis.spin.to_radians();
    let (sin_ra, cos_ra) = ra.sin_cos();
    let (sin_dec, cos_dec) = dec.sin_cos();
    let (sin_spin, cos_spin) = spin.sin_cos();

    let eqj_axes = [
        [
            -sin_ra * cos_spin - cos_ra * sin_dec * sin_spin,
            cos_ra * cos_spin - sin_ra * sin_dec * sin_spin,
            cos_dec * sin_spin,
        ],
        [
            sin_ra * sin_spin - cos_ra * sin_dec * cos_spin,
            -cos_ra * sin_spin - sin_ra * sin_dec * cos_spin,
            cos_dec * cos_spin,
        ],
        [cos_ra * cos_dec, sin_ra * cos_dec, sin_dec],
    ];

    let mut matrix = [[0.0_f32; 3]; 3];
    for (row, axis_eqj) in eqj_axes.into_iter().enumerate() {
        let vector = ffi::astro_vector_t {
            status: ffi::astro_status_t_ASTRO_SUCCESS,
            x: axis_eqj[0],
            y: axis_eqj[1],
            z: axis_eqj[2],
            t: time,
        };
        let eqd = unsafe { ffi::Astronomy_RotateVector(rotation, vector) };
        check_vector("lunar body axis rotation", &eqd)?;
        matrix[row] = to_f32_vector(
            eqd_to_ecef([eqd.x, eqd.y, eqd.z], sidereal_radians),
            "lunar body axis",
        )?;
    }
    Ok(matrix)
}

// Rotate an equator-of-date (EQD) vector into the Earth-fixed (ECEF) frame
// using Greenwich apparent sidereal time. This is the same frame the day /
// night / Moon textures live in (the baker pins longitude -180 at U=0, so
// scene +X is the prime meridian), so the Sun and Moon *direction* vectors
// must come out un-flipped here. A previous version negated X and Y to match
// the star-panorama's own 180-degree convention; that flipped the live
// sub-solar point by half a turn and lit the night hemisphere (e.g. Saudi
// Arabia at 02:00 local rendered as day). The panorama de-rotation in
// stars_textured.frag carries that 180-degree term itself and is untouched.
fn eqd_to_ecef(eqd: [f64; 3], sidereal_radians: f64) -> [f64; 3] {
    let (sin_theta, cos_theta) = sidereal_radians.sin_cos();
    let ecef_x = cos_theta * eqd[0] + sin_theta * eqd[1];
    let ecef_y = -sin_theta * eqd[0] + cos_theta * eqd[1];
    [ecef_x, ecef_y, eqd[2]]
}

fn angular_radius(
    body_radius_km: f64,
    distance_au: f64,
    name: &'static str,
) -> Result<f64, AstronomyError> {
    let ratio = body_radius_km / (distance_au * ASTRONOMICAL_UNIT_KM);
    if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
        return Err(AstronomyError::InvalidValue(name));
    }
    Ok(ratio.asin())
}

fn check_status(
    operation: &'static str,
    status: ffi::astro_status_t,
) -> Result<(), AstronomyError> {
    if status == ffi::astro_status_t_ASTRO_SUCCESS {
        Ok(())
    } else {
        Err(AstronomyError::FfiStatus {
            operation,
            status: status as i32,
        })
    }
}

fn check_vector(name: &'static str, vector: &ffi::astro_vector_t) -> Result<(), AstronomyError> {
    check_status(name, vector.status)?;
    if [vector.x, vector.y, vector.z]
        .iter()
        .all(|value| value.is_finite())
    {
        Ok(())
    } else {
        Err(AstronomyError::NonFinite(name))
    }
}

fn check_finite(name: &'static str, value: f64) -> Result<(), AstronomyError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AstronomyError::NonFinite(name))
    }
}

fn to_f32(value: f64, name: &'static str) -> Result<f32, AstronomyError> {
    let value = value as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AstronomyError::NonFinite(name))
    }
}

fn to_f32_vector(vector: [f64; 3], name: &'static str) -> Result<[f32; 3], AstronomyError> {
    Ok([
        to_f32(vector[0], name)?,
        to_f32(vector[1], name)?,
        to_f32(vector[2], name)?,
    ])
}

fn vector_length(vector: [f64; 3]) -> f64 {
    vector
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a.into_iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn unix_epoch_conversion_matches_j2000() {
        let j2000 = time_from_unix_seconds(UNIX_SECONDS_AT_J2000).unwrap();
        assert_eq!(j2000.ut, 0.0);

        let unix_epoch = time_from_unix_seconds(0).unwrap();
        assert!((unix_epoch.ut + 10_957.5).abs() < 1.0e-12);

        let before_j2000 = time_from_unix_seconds(UNIX_SECONDS_AT_J2000 - 1).unwrap();
        assert!((before_j2000.ut + 1.0 / 86_400.0).abs() < 1.0e-12);
    }

    #[test]
    fn vectors_are_normalized_and_physical_scales_are_sensible() {
        let state = celestial_state(1_712_600_400).unwrap(); // 2024-04-08 18:20 UTC
        for direction in [state.sun_direction, state.moon_direction] {
            assert!(direction.iter().all(|value| value.is_finite()));
            assert!((dot(direction, direction) - 1.0).abs() < 2.0e-6);
        }
        assert!((22_000.0..24_000.0).contains(&state.sun_distance_earth_radii));
        assert!((55.0..64.0).contains(&state.moon_distance_earth_radii));

        let sun_diameter_degrees = (2.0 * state.sun_angular_radius_radians).to_degrees();
        let moon_diameter_degrees = (2.0 * state.moon_angular_radius_radians).to_degrees();
        assert!((0.51..0.55).contains(&sun_diameter_degrees));
        assert!((0.48..0.58).contains(&moon_diameter_degrees));
    }

    #[test]
    fn body_sun_directions_are_unit_vectors_with_orbital_distance_ordering() {
        let state = celestial_state(1_712_600_400).unwrap(); // 2024-04-08 18:20 UTC
                                                             // Slot 0 is Earth and must equal the geocentric Sun exactly.
        assert!((dot(state.body_sun_directions[0], state.sun_direction) - 1.0).abs() < 1.0e-6);
        for direction in state.body_sun_directions {
            assert!(direction.iter().all(|value| value.is_finite()));
            assert!((dot(direction, direction) - 1.0).abs() < 2.0e-6);
        }
        // Orbits never cross, so the Sun distance ordering is invariant:
        // Mercury < Earth < Mars < Jupiter < Saturn.
        let d = state.body_sun_distances_earth_radii;
        assert!(d[2] < d[0]);
        assert!(d[0] < d[3]);
        assert!(d[3] < d[1]);
        assert!(d[1] < d[4]);
        // Mercury < Venus < Earth and Saturn < Uranus < Neptune; the Moon
        // shares Earth's heliocentric distance to within ~0.3 %.
        assert!(d[2] < d[5] && d[5] < d[0]);
        assert!(d[4] < d[6] && d[6] < d[7]);
        assert!((d[8] / d[0] - 1.0).abs() < 0.004);
        // Uranus 18.3-20.1 AU, Neptune 29.8-30.4 AU.
        assert!((429_000.0..472_000.0).contains(&d[6]));
        assert!((699_000.0..713_000.0).contains(&d[7]));
        // Physical bounds in Earth radii (AU * 23455): Mercury 0.31-0.47 AU,
        // Saturn 9.0-10.1 AU.
        assert!((7_000.0..11_500.0).contains(&d[2]));
        assert!((210_000.0..240_000.0).contains(&d[4]));
    }

    #[test]
    fn planetary_frames_are_orthonormal_and_follow_their_own_solar_days() {
        let first = celestial_state(1_710_936_000).unwrap();
        let later = celestial_state(1_710_939_600).unwrap();
        for matrix in first.body_world_to_fixed {
            for i in 0..3 {
                assert!((dot(matrix[i], matrix[i]) - 1.0).abs() < 1.0e-5);
                assert!(dot(matrix[i], matrix[(i + 1) % 3]).abs() < 1.0e-5);
            }
        }
        // Hourly change in the subsolar longitude on each body. (The raw
        // angle is scaled by cos(subsolar latitude): Uranus' 98-degree tilt
        // currently puts the Sun near 64 degrees north there.)
        let motion = |body: usize| {
            let a = first.body_world_to_fixed[body].map(|row| dot(row, first.body_sun_directions[body]));
            let b = later.body_world_to_fixed[body].map(|row| dot(row, later.body_sun_directions[body]));
            let delta = (b[1].atan2(b[0]) - a[1].atan2(a[0])).to_degrees().rem_euclid(360.0);
            delta.min(360.0 - delta)
        };
        assert!((30.0..42.0).contains(&motion(1)), "Jupiter: {}", motion(1));
        // Solar days: Venus ~116.75 d, Uranus ~17.24 h, Neptune ~16.11 h,
        // Moon ~29.53 d.
        assert!(motion(5) < 0.2, "Venus: {}", motion(5));
        assert!((18.0..24.0).contains(&motion(6)), "Uranus: {}", motion(6));
        assert!((19.0..25.0).contains(&motion(7)), "Neptune: {}", motion(7));
        assert!(motion(8) < 1.0, "Moon: {}", motion(8));
        assert!(motion(2) < 0.2, "Mercury: {}", motion(2));
        assert!((10.0..20.0).contains(&motion(3)), "Mars: {}", motion(3));
        assert!((25.0..40.0).contains(&motion(4)), "Saturn: {}", motion(4));
    }
    #[test]
    fn known_2024_lunar_phases_have_expected_illumination() {
        let full = celestial_state(1_711_348_400).unwrap(); // 2024-03-25 07:00 UTC
        let quarter = celestial_state(1_713_186_000).unwrap(); // 2024-04-15 13:00 UTC
        let new = celestial_state(1_712_600_400).unwrap(); // 2024-04-08 18:20 UTC
        assert!(full.moon_illuminated_fraction > 0.97);
        assert!((0.40..0.60).contains(&quarter.moon_illuminated_fraction));
        assert!(new.moon_illuminated_fraction < 0.03);
        assert!(full.moon_phase_angle_radians < 0.35);
        assert!(new.moon_phase_angle_radians > 2.8);
    }

    // Regression for the half-turn Sun/Moon frame bug: at this instant Riyadh
    // (24.71 N, 46.68 E) is at 01:54 local time and must be on the night side.
    // The Sun direction is in the same Earth-fixed frame as the day texture
    // (scene +X = prime meridian), so its dot product with the city's ECEF
    // unit vector is the sine of the solar elevation and must be negative.
    // Before the eqd_to_ecef fix the X/Y negation made this dot positive and
    // lit the night hemisphere.
    #[test]
    fn sun_direction_is_earth_fixed_night_over_riyadh_at_local_0154() {
        let state = celestial_state(1_785_279_368).unwrap(); // 2026-07-27 22:56 UTC
        let lat = 24.71_f64.to_radians();
        let lon = 46.68_f64.to_radians();
        let riyadh = [
            (lat.cos() * lon.cos()) as f32,
            (lat.cos() * lon.sin()) as f32,
            lat.sin() as f32,
        ];
        let solar_elevation_sine = dot(state.sun_direction, riyadh);
        assert!(
            solar_elevation_sine < -0.3,
            "Riyadh must be at night, got solar-elevation sine {solar_elevation_sine}"
        );
        // Sub-solar longitude must be ~197.6 E, not the flipped ~17.6 E.
        let subsolar_longitude = state.sun_direction[1]
            .atan2(state.sun_direction[0])
            .to_degrees();
        let subsolar_longitude = subsolar_longitude.rem_euclid(360.0);
        assert!(
            (190.0..205.0).contains(&subsolar_longitude),
            "sub-solar longitude {subsolar_longitude} not near 197.6 E"
        );
    }

    #[test]
    fn sidereal_angle_advances_by_about_six_hours() {
        let first = celestial_state(1_704_067_200).unwrap(); // 2024-01-01 00:00 UTC
        let second = celestial_state(1_704_088_800).unwrap(); // six hours later
        let delta = (second.greenwich_apparent_sidereal_angle_radians
            - first.greenwich_apparent_sidereal_angle_radians)
            .rem_euclid(2.0 * std::f32::consts::PI);
        assert!((1.55..1.60).contains(&delta));
    }

    #[test]
    fn lunar_basis_is_orthonormal_and_continuous_over_a_day() {
        let first = celestial_state(1_704_067_200).unwrap();
        let next = celestial_state(1_704_153_600).unwrap();
        for matrix in [first.moon_world_to_body, next.moon_world_to_body] {
            for row in matrix {
                assert!((dot(row, row) - 1.0).abs() < 2.0e-6);
            }
            assert!(dot(matrix[0], matrix[1]).abs() < 2.0e-6);
            assert!(dot(matrix[0], matrix[2]).abs() < 2.0e-6);
            assert!(dot(matrix[1], matrix[2]).abs() < 2.0e-6);
        }
        for row in 0..3 {
            assert!(dot(first.moon_world_to_body[row], next.moon_world_to_body[row]) > 0.95);
        }

        // The visible near side points back toward Earth and must stay close
        // to the Moon's body-fixed prime meridian, with only libration offsets.
        let near_side = first.moon_direction.map(|value| -value);
        let body_near_side = first.moon_world_to_body.map(|row| dot(row, near_side));
        assert!(
            body_near_side[0] > 0.97,
            "unexpected lunar near-side basis: {body_near_side:?}"
        );
        assert!(body_near_side[1].abs() < 0.2);
        assert!(body_near_side[2].abs() < 0.2);
    }
}
