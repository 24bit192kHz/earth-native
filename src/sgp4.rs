//! Exception-safe Rust ownership around the narrow libsgp4 C ABI.

use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_double, c_int},
    ptr::NonNull,
};

#[repr(C)]
struct RawSgp4 {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn earth_sgp4_create() -> *mut RawSgp4;
    fn earth_sgp4_destroy(handle: *mut RawSgp4);
    fn earth_sgp4_load_tle(
        handle: *mut RawSgp4,
        name: *const c_char,
        line_one: *const c_char,
        line_two: *const c_char,
    ) -> c_int;
    fn earth_sgp4_propagate_unix_utc(
        handle: *mut RawSgp4,
        unix_seconds: i64,
        microseconds: i32,
        latitude_radians: *mut c_double,
        longitude_radians: *mut c_double,
        altitude_kilometres: *mut c_double,
    ) -> c_int;
    fn earth_sgp4_last_error(handle: *const RawSgp4) -> *const c_char;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodeticPosition {
    pub latitude_radians: f64,
    pub longitude_radians: f64,
    pub altitude_kilometres: f64,
}

pub struct IssOrbit {
    raw: NonNull<RawSgp4>,
}

impl IssOrbit {
    pub fn new() -> Result<Self, String> {
        let raw = NonNull::new(unsafe { earth_sgp4_create() })
            .ok_or_else(|| "libsgp4 could not allocate an orbit handle".to_owned())?;
        Ok(Self { raw })
    }

    pub fn load_tle(&mut self, name: &str, line_one: &str, line_two: &str) -> Result<(), String> {
        let name = CString::new(name).map_err(|_| "TLE name contains an interior NUL byte")?;
        let line_one =
            CString::new(line_one).map_err(|_| "TLE line one contains an interior NUL byte")?;
        let line_two =
            CString::new(line_two).map_err(|_| "TLE line two contains an interior NUL byte")?;
        self.check(unsafe {
            earth_sgp4_load_tle(
                self.raw.as_ptr(),
                name.as_ptr(),
                line_one.as_ptr(),
                line_two.as_ptr(),
            )
        })
    }

    pub fn propagate_unix_utc(
        &mut self,
        unix_seconds: i64,
        microseconds: i32,
    ) -> Result<GeodeticPosition, String> {
        let mut latitude = 0.0;
        let mut longitude = 0.0;
        let mut altitude = 0.0;
        self.check(unsafe {
            earth_sgp4_propagate_unix_utc(
                self.raw.as_ptr(),
                unix_seconds,
                microseconds,
                &mut latitude,
                &mut longitude,
                &mut altitude,
            )
        })?;
        Ok(GeodeticPosition {
            latitude_radians: latitude,
            longitude_radians: longitude,
            altitude_kilometres: altitude,
        })
    }

    fn check(&self, status: c_int) -> Result<(), String> {
        if status == 0 {
            return Ok(());
        }
        let message = unsafe { earth_sgp4_last_error(self.raw.as_ptr()) };
        let message = if message.is_null() {
            "unknown libsgp4 error".to_owned()
        } else {
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        };
        Err(format!("libsgp4 status {status}: {message}"))
    }
}

impl Drop for IssOrbit {
    fn drop(&mut self) {
        unsafe { earth_sgp4_destroy(self.raw.as_ptr()) };
    }
}
