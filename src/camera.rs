#![allow(dead_code)] // Public camera/projection API is consumed by later renderer stages.

//! Shared orbit-camera and continuous-desktop projection math.
//!
//! The renderer has one camera but a layer-shell surface for every output.
//! `DesktopLayout` describes the compositor's logical desktop, while a
//! `ProjectionSlice` maps a pixel from one output back into that common
//! desktop.  Keeping the mapping here prevents a rotated or scaled monitor
//! from receiving an independent camera projection.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Unreal's `UCameraComponent::FieldOfView` value in the reference project.
pub const CAMERA_FOV_X_DEGREES: f32 = 50.0;
/// The camera component's default aspect ratio in Unreal.
pub const CAMERA_REFERENCE_ASPECT_RATIO: f32 = 16.0 / 9.0;
/// Shared rendered Earth radius. All camera and celestial distances use this
/// same scene-unit scale.
pub const SCENE_EARTH_RADIUS: f32 = 0.78;
pub const INITIAL_DISTANCE_SCALE: f32 = 5.5;
pub const MIN_DISTANCE_SCALE: f32 = 1.08;
pub const MAX_DISTANCE_SCALE: f32 = 12.0;
pub const ZOOM_FACTOR_PER_STEP: f32 = 0.84;
pub const ZOOM_SMOOTHING_RATE: f32 = 7.0;
pub const MOUSE_ORBIT_DEGREES_PER_UNIT: f32 = 0.06;
pub const KEYBOARD_ORBIT_DEGREES_PER_SECOND: f32 = 70.0;
pub const MIN_PITCH_DEGREES: f32 = -89.0;
pub const MAX_PITCH_DEGREES: f32 = 89.0;
pub const INITIAL_YAW_DEGREES: f32 = 180.0;
/// `UE_SMALL_NUMBER`, used by Unreal's `FInterpTo` early-out.
pub const INTERP_SNAP_DISTANCE_SQUARED: f32 = 1.0e-8;

/// A compact dependency-free two-component float vector.
///
/// It is used for normalized UV, NDC, and logical desktop coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

fn rotate_around_axis(vector: Vec3, axis: Vec3, radians: f32) -> Vec3 {
    let axis = axis.normalized();
    if axis == Vec3::ZERO {
        return vector;
    }
    let (sine, cosine) = radians.sin_cos();
    vector * cosine + axis.cross(vector) * sine + axis * (axis.dot(vector) * (1.0 - cosine))
}

impl Vec2 {
    pub const ZERO: Self = Self::new(0.0, 0.0);

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    pub fn min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y))
    }

    pub fn max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y))
    }
}

impl Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

/// A compact dependency-free three-component float vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn normalized(self) -> Self {
        let length = self.length();
        if length.is_finite() && length > f32::EPSILON {
            self / length
        } else {
            Self::ZERO
        }
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// Camera axes in Unreal's X-forward, Y-right, Z-up world convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

/// A world-space ray suitable for analytical sphere intersection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

/// Shared orbital state.  Distances are expressed in the same units as
/// `earth_radius`, so the renderer may use a normalized sphere radius of one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitCamera {
    center: Vec3,
    earth_radius: f32,
    base_orbit_direction: Vec3,
    distance: f32,
    target_distance: f32,
    yaw_degrees: f32,
    pitch_degrees: f32,
    keyboard_yaw_axis: f32,
    keyboard_pitch_axis: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl OrbitCamera {
    /// Sets the direction used before local yaw and pitch are applied. This
    /// is normally +X, or the one-second-sampled ISS direction when orbital
    /// tracking is enabled. Invalid values preserve the last stable pose.
    pub fn set_base_orbit_direction(&mut self, direction: Vec3) -> bool {
        let direction = direction.normalized();
        if direction == Vec3::ZERO || direction == self.base_orbit_direction {
            return false;
        }
        self.base_orbit_direction = direction;
        true
    }

    /// Creates an orbit camera centered at the world origin.
    ///
    /// A non-positive or non-finite radius is replaced with one.  This keeps a
    /// malformed discovery result from ever producing invalid shader uniforms.
    pub fn new(earth_radius: f32) -> Self {
        Self::with_center(Vec3::ZERO, earth_radius)
    }

    pub fn with_center(center: Vec3, earth_radius: f32) -> Self {
        let center = if center.is_finite() {
            center
        } else {
            Vec3::ZERO
        };
        let earth_radius = sanitize_radius(earth_radius);
        let distance = earth_radius * INITIAL_DISTANCE_SCALE;

        Self {
            center,
            earth_radius,
            base_orbit_direction: Vec3::X,
            distance,
            target_distance: distance,
            yaw_degrees: INITIAL_YAW_DEGREES,
            pitch_degrees: 0.0,
            keyboard_yaw_axis: 0.0,
            keyboard_pitch_axis: 0.0,
        }
    }

    pub fn center(&self) -> Vec3 {
        self.center
    }

    pub fn earth_radius(&self) -> f32 {
        self.earth_radius
    }

    pub fn distance(&self) -> f32 {
        self.distance
    }

    pub fn target_distance(&self) -> f32 {
        self.target_distance
    }

    pub fn distance_scale(&self) -> f32 {
        self.distance / self.earth_radius
    }

    pub fn target_distance_scale(&self) -> f32 {
        self.target_distance / self.earth_radius
    }

    pub fn yaw_degrees(&self) -> f32 {
        self.yaw_degrees
    }

    pub fn pitch_degrees(&self) -> f32 {
        self.pitch_degrees
    }

    /// Changes the orbit center without changing the camera's distance or
    /// angular state.
    pub fn set_center(&mut self, center: Vec3) -> bool {
        if !center.is_finite() || center == self.center {
            return false;
        }

        self.center = center;
        true
    }

    /// Re-scales current and target distances so their scale relative to the
    /// Earth remains stable when the discovered Earth bounds change.
    pub fn set_earth_radius(&mut self, earth_radius: f32) -> bool {
        let earth_radius = sanitize_radius(earth_radius);
        if earth_radius == self.earth_radius {
            return false;
        }

        let current_scale = self
            .distance_scale()
            .clamp(MIN_DISTANCE_SCALE, MAX_DISTANCE_SCALE);
        let target_scale = self
            .target_distance_scale()
            .clamp(MIN_DISTANCE_SCALE, MAX_DISTANCE_SCALE);
        self.earth_radius = earth_radius;
        self.distance = current_scale * earth_radius;
        self.target_distance = target_scale * earth_radius;
        true
    }

    /// Sets both current and target distance immediately.  This is intended
    /// for restoring a saved camera state or matching a reference capture.
    pub fn set_distance_scale_immediate(&mut self, scale: f32) -> bool {
        let scale = clamp_distance_scale(scale);
        let distance = self.earth_radius * scale;
        if self.distance == distance && self.target_distance == distance {
            return false;
        }

        self.distance = distance;
        self.target_distance = distance;
        true
    }

    /// Sets the smoothed zoom destination without changing the current zoom.
    pub fn set_target_distance_scale(&mut self, scale: f32) -> bool {
        let target_distance = self.earth_radius * clamp_distance_scale(scale);
        if self.target_distance == target_distance {
            return false;
        }

        self.target_distance = target_distance;
        true
    }

    /// Applies the exact main-project zoom rule: `target *= 0.84^steps`.
    pub fn zoom_steps(&mut self, steps: f32) -> bool {
        if !steps.is_finite() || steps == 0.0 {
            return false;
        }

        let zoom_factor = ZOOM_FACTOR_PER_STEP.powf(steps);
        let min_distance = self.earth_radius * MIN_DISTANCE_SCALE;
        let max_distance = self.earth_radius * MAX_DISTANCE_SCALE;
        let candidate = self.target_distance * zoom_factor;
        let target_distance = if candidate.is_finite() {
            candidate.clamp(min_distance, max_distance)
        } else if steps.is_sign_positive() {
            min_distance
        } else {
            max_distance
        };

        if target_distance == self.target_distance {
            return false;
        }

        self.target_distance = target_distance;
        true
    }

    /// Applies a mouse drag.  Call this only while the layer-shell input region
    /// is enabled for interactive control.
    pub fn orbit_mouse(&mut self, delta_x: f32, delta_y: f32) -> bool {
        let previous_yaw = self.yaw_degrees;
        let previous_pitch = self.pitch_degrees;

        if delta_x.is_finite() {
            let yaw = self.yaw_degrees + delta_x * MOUSE_ORBIT_DEGREES_PER_UNIT;
            if yaw.is_finite() {
                self.yaw_degrees = yaw;
            }
        }
        if delta_y.is_finite() {
            let pitch = self.pitch_degrees + delta_y * MOUSE_ORBIT_DEGREES_PER_UNIT;
            if pitch.is_finite() {
                self.pitch_degrees = self.clamp_pitch(pitch);
            }
        }

        self.yaw_degrees != previous_yaw || self.pitch_degrees != previous_pitch
    }

    /// Pitch is applied on top of the base orbit direction, which ISS
    /// tracking tilts by up to its 51.6-degree inclination. Clamping pitch
    /// alone let the total pass a pole, flipping the camera's up vector and
    /// inverting the view; bound the total elevation instead.
    fn clamp_pitch(&self, pitch: f32) -> f32 {
        let base_elevation = self.base_orbit_direction.z.clamp(-1.0, 1.0).asin().to_degrees();
        let low = MIN_PITCH_DEGREES.max(MIN_PITCH_DEGREES - base_elevation);
        let high = MAX_PITCH_DEGREES.min(MAX_PITCH_DEGREES - base_elevation);
        pitch.clamp(low, high.max(low))
    }

    /// Mirrors the two persistent keyboard axis values held by the Unreal
    /// player controller.  Values are normally -1, 0, or 1.
    pub fn set_keyboard_axes(&mut self, yaw_axis: f32, pitch_axis: f32) {
        self.keyboard_yaw_axis = sanitize_axis(yaw_axis);
        self.keyboard_pitch_axis = sanitize_axis(pitch_axis);
    }

    /// Sets an orbital pose directly, preserving Unreal's unclamped yaw and
    /// clamped pitch semantics.
    pub fn set_orbit_angles(&mut self, yaw_degrees: f32, pitch_degrees: f32) -> bool {
        let previous_yaw = self.yaw_degrees;
        let previous_pitch = self.pitch_degrees;

        if yaw_degrees.is_finite() {
            self.yaw_degrees = yaw_degrees;
        }
        if pitch_degrees.is_finite() {
            self.pitch_degrees = self.clamp_pitch(pitch_degrees);
        }

        self.yaw_degrees != previous_yaw || self.pitch_degrees != previous_pitch
    }

    /// Advances keyboard orbit and zoom smoothing by monotonic elapsed time.
    ///
    /// Zoom intentionally uses Unreal's `FInterpTo` rule rather than an
    /// exponential ease: `current += (target - current) * clamp(dt * 7, 0, 1)`.
    pub fn tick(&mut self, delta_seconds: f32) -> bool {
        if !delta_seconds.is_finite() || delta_seconds <= 0.0 {
            return false;
        }

        let previous_yaw = self.yaw_degrees;
        let previous_pitch = self.pitch_degrees;
        let previous_distance = self.distance;

        let yaw = self.yaw_degrees
            + self.keyboard_yaw_axis * KEYBOARD_ORBIT_DEGREES_PER_SECOND * delta_seconds;
        if yaw.is_finite() {
            self.yaw_degrees = yaw;
        }

        let pitch = self.pitch_degrees
            + self.keyboard_pitch_axis * KEYBOARD_ORBIT_DEGREES_PER_SECOND * delta_seconds;
        if pitch.is_finite() {
            self.pitch_degrees = self.clamp_pitch(pitch);
        }

        let distance_delta = self.target_distance - self.distance;
        if distance_delta * distance_delta < INTERP_SNAP_DISTANCE_SQUARED {
            self.distance = self.target_distance;
        } else {
            let interpolation = (delta_seconds * ZOOM_SMOOTHING_RATE).clamp(0.0, 1.0);
            self.distance += distance_delta * interpolation;
        }

        self.yaw_degrees != previous_yaw
            || self.pitch_degrees != previous_pitch
            || self.distance != previous_distance
    }

    /// Produces the world-space camera axes.  At the default yaw of 180
    /// degrees this places the camera on +X looking toward the origin, exactly
    /// as `OrbitCenter - FRotator(Pitch, Yaw, 0).Vector() * Distance` does.
    pub fn pose(&self) -> CameraPose {
        let yaw = self.yaw_degrees.to_radians();
        // The ISS camera basis is mirrored on Z before the calibrated 180
        // degree yaw. Negate pitch here so positive interactive pitch follows
        // Unreal's `FQuat(PitchAxis, ManualPitch)` camera-location convention.
        // Re-clamp against the current base: ISS tracking moves it after the
        // stored pitch was accepted.
        let pitch = -self.clamp_pitch(self.pitch_degrees).to_radians();
        let yawed_direction = rotate_around_axis(self.base_orbit_direction, Vec3::Z, yaw);
        let mut pitch_axis = Vec3::Z.cross(yawed_direction).normalized();
        if pitch_axis == Vec3::ZERO {
            pitch_axis = Vec3::Y;
        }
        let forward = rotate_around_axis(yawed_direction, pitch_axis, pitch).normalized();

        // The pitch limit keeps this non-zero.  Keep a finite fallback so a
        // corrupted restored state still leaves the renderer with a valid basis.
        let mut right = Vec3::Z.cross(forward).normalized();
        if right == Vec3::ZERO {
            right = Vec3::Y;
        }
        let up = forward.cross(right).normalized();

        CameraPose {
            position: self.center - forward * self.distance,
            forward,
            right,
            up,
        }
    }
}

fn sanitize_radius(radius: f32) -> f32 {
    if radius.is_finite() && radius > 0.0 {
        radius
    } else {
        1.0
    }
}

fn clamp_distance_scale(scale: f32) -> f32 {
    if scale.is_finite() {
        scale.clamp(MIN_DISTANCE_SCALE, MAX_DISTANCE_SCALE)
    } else {
        INITIAL_DISTANCE_SCALE
    }
}

fn sanitize_axis(axis: f32) -> f32 {
    if axis.is_finite() {
        axis
    } else {
        0.0
    }
}

/// Physical mode size before the output transform and fractional scale are
/// applied.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelSize {
    pub width: u32,
    pub height: u32,
}

impl PixelSize {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// The eight `wl_output.transform` values.
///
/// `map_buffer_uv` maps UV coordinates in an untransformed buffer into the
/// visual, top-left-origin coordinate system of the output.  The 90-degree
/// names follow the Wayland protocol: rotation is counter-clockwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputTransform {
    #[default]
    Normal,
    Rotate90,
    Rotate180,
    Rotate270,
    Flipped,
    Flipped90,
    Flipped180,
    Flipped270,
}

impl OutputTransform {
    pub const fn from_wl_output(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Normal),
            1 => Some(Self::Rotate90),
            2 => Some(Self::Rotate180),
            3 => Some(Self::Rotate270),
            4 => Some(Self::Flipped),
            5 => Some(Self::Flipped90),
            6 => Some(Self::Flipped180),
            7 => Some(Self::Flipped270),
            _ => None,
        }
    }

    pub const fn as_wl_output(self) -> u32 {
        match self {
            Self::Normal => 0,
            Self::Rotate90 => 1,
            Self::Rotate180 => 2,
            Self::Rotate270 => 3,
            Self::Flipped => 4,
            Self::Flipped90 => 5,
            Self::Flipped180 => 6,
            Self::Flipped270 => 7,
        }
    }

    /// Returns whether transformed logical width and height are swapped.
    pub const fn swaps_axes(self) -> bool {
        matches!(
            self,
            Self::Rotate90 | Self::Rotate270 | Self::Flipped90 | Self::Flipped270
        )
    }

    /// Maps UVs in an untransformed buffer to visual output UVs.
    pub fn map_buffer_uv(self, uv: Vec2) -> Vec2 {
        match self {
            Self::Normal => uv,
            Self::Rotate90 => Vec2::new(uv.y, 1.0 - uv.x),
            Self::Rotate180 => Vec2::new(1.0 - uv.x, 1.0 - uv.y),
            Self::Rotate270 => Vec2::new(1.0 - uv.y, uv.x),
            Self::Flipped => Vec2::new(1.0 - uv.x, uv.y),
            Self::Flipped90 => Vec2::new(uv.y, uv.x),
            Self::Flipped180 => Vec2::new(uv.x, 1.0 - uv.y),
            Self::Flipped270 => Vec2::new(1.0 - uv.y, 1.0 - uv.x),
        }
    }

    pub const fn inverse(self) -> Self {
        match self {
            Self::Normal => Self::Normal,
            Self::Rotate90 => Self::Rotate270,
            Self::Rotate180 => Self::Rotate180,
            Self::Rotate270 => Self::Rotate90,
            Self::Flipped => Self::Flipped,
            Self::Flipped90 => Self::Flipped90,
            Self::Flipped180 => Self::Flipped180,
            Self::Flipped270 => Self::Flipped270,
        }
    }

    /// Maps visual output UVs back into the untransformed buffer.
    pub fn unmap_output_uv(self, uv: Vec2) -> Vec2 {
        self.inverse().map_buffer_uv(uv)
    }
}

/// An axis-aligned rectangle in Hyprland/Wayland logical desktop coordinates.
/// Its origin is the top-left corner and its Y axis grows downward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogicalRect {
    pub origin: Vec2,
    pub size: Vec2,
}

impl LogicalRect {
    pub fn new(origin: Vec2, size: Vec2) -> Result<Self, ProjectionError> {
        let rect = Self { origin, size };
        if rect.is_valid() {
            Ok(rect)
        } else {
            Err(ProjectionError::InvalidLogicalRect)
        }
    }

    pub fn is_valid(self) -> bool {
        self.origin.is_finite() && self.size.is_finite() && self.size.x > 0.0 && self.size.y > 0.0
    }

    pub fn right(self) -> f32 {
        self.origin.x + self.size.x
    }

    pub fn bottom(self) -> f32 {
        self.origin.y + self.size.y
    }

    pub fn center(self) -> Vec2 {
        self.origin + self.size * 0.5
    }

    pub fn aspect_ratio(self) -> f32 {
        self.size.x / self.size.y
    }

    pub fn union(self, other: Self) -> Self {
        let min = self.origin.min(other.origin);
        let max =
            Vec2::new(self.right(), self.bottom()).max(Vec2::new(other.right(), other.bottom()));
        Self {
            origin: min,
            size: max - min,
        }
    }

    pub fn global_uv(self, point: Vec2) -> Vec2 {
        Vec2::new(
            (point.x - self.origin.x) / self.size.x,
            (point.y - self.origin.y) / self.size.y,
        )
    }
}

/// Errors returned instead of allowing a bad hotplug/configure event to put
/// NaN or an empty rectangle into a camera uniform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionError {
    EmptyOutputSet,
    InvalidLogicalRect,
    InvalidOutputScale,
    EmptyPixelSize,
}

/// Output information reported by Wayland/Hyprland.  `position` is already in
/// global logical coordinates; `pixel_size` is the current native output mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputGeometry {
    pub position: Vec2,
    pub pixel_size: PixelSize,
    pub scale: f32,
    pub transform: OutputTransform,
}

impl OutputGeometry {
    pub const fn new(
        position: Vec2,
        pixel_size: PixelSize,
        scale: f32,
        transform: OutputTransform,
    ) -> Self {
        Self {
            position,
            pixel_size,
            scale,
            transform,
        }
    }

    pub fn logical_size(self) -> Result<Vec2, ProjectionError> {
        if self.pixel_size.is_empty() {
            return Err(ProjectionError::EmptyPixelSize);
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(ProjectionError::InvalidOutputScale);
        }

        let (width, height) = if self.transform.swaps_axes() {
            (self.pixel_size.height, self.pixel_size.width)
        } else {
            (self.pixel_size.width, self.pixel_size.height)
        };
        let size = Vec2::new(width as f32 / self.scale, height as f32 / self.scale);
        if size.is_finite() && size.x > 0.0 && size.y > 0.0 {
            Ok(size)
        } else {
            Err(ProjectionError::InvalidLogicalRect)
        }
    }

    pub fn logical_rect(self) -> Result<LogicalRect, ProjectionError> {
        LogicalRect::new(self.position, self.logical_size()?)
    }
}

/// The bounding logical rectangle containing all active outputs.  Gaps remain
/// part of the canvas, matching the current wide Unreal wallpaper behavior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DesktopLayout {
    pub bounds: LogicalRect,
}

impl DesktopLayout {
    pub fn new(bounds: LogicalRect) -> Result<Self, ProjectionError> {
        if bounds.is_valid() {
            Ok(Self { bounds })
        } else {
            Err(ProjectionError::InvalidLogicalRect)
        }
    }

    pub fn from_outputs(outputs: &[OutputGeometry]) -> Result<Self, ProjectionError> {
        let mut bounds: Option<LogicalRect> = None;
        for output in outputs {
            let output_rect = output.logical_rect()?;
            bounds = Some(match bounds {
                Some(current) => current.union(output_rect),
                None => output_rect,
            });
        }

        Self::new(bounds.ok_or(ProjectionError::EmptyOutputSet)?)
    }

    pub fn projection_slice(
        self,
        output: OutputGeometry,
    ) -> Result<ProjectionSlice, ProjectionError> {
        ProjectionSlice::new(self.bounds, output)
    }
}

/// The global NDC limits occupied by an output.  This is useful when a
/// renderer derives a scissor rectangle from the global camera projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NdcRect {
    pub min: Vec2,
    pub max: Vec2,
}

/// Maps one layer-shell output into the global logical desktop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionSlice {
    pub desktop: LogicalRect,
    pub output: LogicalRect,
    pub pixel_size: PixelSize,
    pub output_transform: OutputTransform,
}

impl ProjectionSlice {
    pub fn new(desktop: LogicalRect, output: OutputGeometry) -> Result<Self, ProjectionError> {
        if !desktop.is_valid() {
            return Err(ProjectionError::InvalidLogicalRect);
        }

        Ok(Self {
            desktop,
            output: output.logical_rect()?,
            pixel_size: output.pixel_size,
            output_transform: output.transform,
        })
    }

    /// Maps visual output UVs into global desktop UVs.  Use this path when the
    /// swapchain/pre-transform already exposes pixels in visual output space.
    pub fn global_uv_from_output_uv(self, output_uv: Vec2) -> Vec2 {
        let point = self.output.origin
            + Vec2::new(
                output_uv.x * self.output.size.x,
                output_uv.y * self.output.size.y,
            );
        self.desktop.global_uv(point)
    }

    /// Maps an untransformed native-buffer UV into global desktop UVs.
    pub fn global_uv_from_buffer_uv(self, buffer_uv: Vec2) -> Vec2 {
        self.global_uv_from_output_uv(self.output_transform.map_buffer_uv(buffer_uv))
    }

    /// Maps visual output UVs into the conventional Vulkan/graphics NDC
    /// convention: X right, Y up.
    pub fn global_ndc_from_output_uv(self, output_uv: Vec2) -> Vec2 {
        desktop_uv_to_ndc(self.global_uv_from_output_uv(output_uv))
    }

    /// Maps untransformed native-buffer UVs into global NDC.
    pub fn global_ndc_from_buffer_uv(self, buffer_uv: Vec2) -> Vec2 {
        desktop_uv_to_ndc(self.global_uv_from_buffer_uv(buffer_uv))
    }

    pub fn ndc_bounds(self) -> NdcRect {
        let top_left = self.global_ndc_from_output_uv(Vec2::new(0.0, 0.0));
        let bottom_right = self.global_ndc_from_output_uv(Vec2::new(1.0, 1.0));
        NdcRect {
            min: top_left.min(bottom_right),
            max: top_left.max(bottom_right),
        }
    }
}

/// GPU-ready tangent values for a perspective ray: `forward + right * x + up
/// * y`, followed by normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionParameters {
    pub tan_half_fov_x: f32,
    pub tan_half_fov_y: f32,
}

/// Perspective projection for the one continuous global desktop canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalProjection {
    pub desktop: DesktopLayout,
    parameters: ProjectionParameters,
}

impl GlobalProjection {
    pub fn new(desktop: DesktopLayout) -> Self {
        // The Unreal project uses AspectRatio_MaintainYFOV.  The camera's
        // 50-degree horizontal FOV was authored at a 16:9 camera aspect, so
        // derive that fixed vertical field of view before adapting to desktop.
        let reference_half_x = (CAMERA_FOV_X_DEGREES.to_radians()) * 0.5;
        let tan_half_fov_y = reference_half_x.tan() / CAMERA_REFERENCE_ASPECT_RATIO;
        let tan_half_fov_x = tan_half_fov_y * desktop.bounds.aspect_ratio();

        Self {
            desktop,
            parameters: ProjectionParameters {
                tan_half_fov_x,
                tan_half_fov_y,
            },
        }
    }

    pub fn parameters(self) -> ProjectionParameters {
        self.parameters
    }

    pub fn vertical_fov_radians(self) -> f32 {
        2.0 * self.parameters.tan_half_fov_y.atan()
    }

    pub fn horizontal_fov_radians(self) -> f32 {
        2.0 * self.parameters.tan_half_fov_x.atan()
    }

    pub fn vertical_fov_degrees(self) -> f32 {
        self.vertical_fov_radians().to_degrees()
    }

    pub fn horizontal_fov_degrees(self) -> f32 {
        self.horizontal_fov_radians().to_degrees()
    }

    pub fn projection_slice(
        self,
        output: OutputGeometry,
    ) -> Result<ProjectionSlice, ProjectionError> {
        self.desktop.projection_slice(output)
    }

    pub fn ray_from_ndc(self, pose: CameraPose, ndc: Vec2) -> Ray {
        let direction = (pose.forward
            + pose.right * (ndc.x * self.parameters.tan_half_fov_x)
            + pose.up * (ndc.y * self.parameters.tan_half_fov_y))
            .normalized();
        Ray {
            origin: pose.position,
            direction,
        }
    }

    pub fn ray_from_output_uv(
        self,
        pose: CameraPose,
        slice: ProjectionSlice,
        output_uv: Vec2,
    ) -> Ray {
        self.ray_from_ndc(pose, slice.global_ndc_from_output_uv(output_uv))
    }

    pub fn ray_from_buffer_uv(
        self,
        pose: CameraPose,
        slice: ProjectionSlice,
        buffer_uv: Vec2,
    ) -> Ray {
        self.ray_from_ndc(pose, slice.global_ndc_from_buffer_uv(buffer_uv))
    }
}

fn desktop_uv_to_ndc(uv: Vec2) -> Vec2 {
    Vec2::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.001;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_vec2_close(actual: Vec2, expected: Vec2) {
        assert_close(actual.x, expected.x);
        assert_close(actual.y, expected.y);
    }

    fn assert_vec3_close(actual: Vec3, expected: Vec3) {
        assert_close(actual.x, expected.x);
        assert_close(actual.y, expected.y);
        assert_close(actual.z, expected.z);
    }

    fn reference_outputs() -> [OutputGeometry; 2] {
        [
            // DP-1 from the current Hyprland layout: native 2560x1080 but
            // rotated 90 degrees, therefore 1080x2560 logical pixels.
            OutputGeometry::new(
                Vec2::new(0.0, 0.0),
                PixelSize::new(2560, 1080),
                1.0,
                OutputTransform::Rotate90,
            ),
            OutputGeometry::new(
                Vec2::new(1080.0, 757.0),
                PixelSize::new(3440, 1440),
                1.0,
                OutputTransform::Normal,
            ),
        ]
    }

    #[test]
    fn tilted_base_cannot_carry_the_camera_over_a_pole() {
        for base_z in [-0.78_f32, -0.3, 0.0, 0.3, 0.78] {
            let mut camera = OrbitCamera::new(1.0);
            let horizontal = (1.0 - base_z * base_z).sqrt();
            camera.set_base_orbit_direction(Vec3::new(horizontal, 0.0, base_z));
            for pitch in [-400.0_f32, -89.0, 89.0, 400.0] {
                camera.set_orbit_angles(180.0, pitch);
                for _ in 0..50 {
                    camera.orbit_mouse(0.0, pitch.signum() * 50.0);
                }
                let pose = camera.pose();
                // Crossing a pole reverses the horizontal heading, which turns
                // the picture upside down and makes drags run backwards.
                let yawed = rotate_around_axis(Vec3::new(horizontal, 0.0, base_z), Vec3::Z, 180.0_f32.to_radians());
                let heading = pose.forward.x * yawed.x + pose.forward.y * yawed.y;
                assert!(heading > 0.0, "inverted: base_z={base_z} pitch={pitch} forward={:?}", pose.forward);
            }
        }
    }

    #[test]
    fn camera_defaults_match_the_unreal_contract() {
        let camera = OrbitCamera::new(1000.0);

        assert_close(camera.distance(), 5500.0);
        assert_close(camera.target_distance(), 5500.0);
        assert_close(camera.distance_scale(), 5.5);
        assert_close(camera.yaw_degrees(), 180.0);
        assert_close(camera.pitch_degrees(), 0.0);

        let pose = camera.pose();
        assert_vec3_close(pose.position, Vec3::new(5500.0, 0.0, 0.0));
        assert_vec3_close(pose.forward, Vec3::new(-1.0, 0.0, 0.0));
        assert_vec3_close(pose.right, Vec3::new(0.0, -1.0, 0.0));
        assert_vec3_close(pose.up, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn zoom_and_smoothing_match_the_reference_rules() {
        let mut camera = OrbitCamera::new(1.0);
        assert!(camera.zoom_steps(1.0));
        assert_close(camera.target_distance_scale(), 4.62);
        assert_close(camera.distance_scale(), 5.5);

        assert!(camera.tick(0.1));
        assert_close(camera.distance_scale(), 4.884);

        camera.tick(1.0);
        assert_close(camera.distance_scale(), 4.62);

        camera.zoom_steps(10_000.0);
        assert_close(camera.target_distance_scale(), MIN_DISTANCE_SCALE);
        camera.zoom_steps(-10_000.0);
        assert_close(camera.target_distance_scale(), MAX_DISTANCE_SCALE);
    }

    #[test]
    fn mouse_and_keyboard_orbit_match_the_reference_rates_and_limits() {
        let mut camera = OrbitCamera::new(1.0);
        assert!(camera.orbit_mouse(10.0, 20.0));
        assert_close(camera.yaw_degrees(), 180.6);
        assert_close(camera.pitch_degrees(), 1.2);

        camera.set_keyboard_axes(1.0, 1.0);
        assert!(camera.tick(2.0));
        assert_close(camera.yaw_degrees(), 320.6);
        assert_close(camera.pitch_degrees(), MAX_PITCH_DEGREES);

        camera.set_keyboard_axes(0.0, -1.0);
        camera.tick(3.0);
        assert_close(camera.pitch_degrees(), MIN_PITCH_DEGREES);
    }

    #[test]
    fn transformed_output_logical_size_accounts_for_rotation_and_scale() {
        let output = OutputGeometry::new(
            Vec2::new(-20.0, 10.0),
            PixelSize::new(3840, 2160),
            1.5,
            OutputTransform::Flipped90,
        );

        assert_vec2_close(output.logical_size().unwrap(), Vec2::new(1440.0, 2560.0));
        let rect = output.logical_rect().unwrap();
        assert_vec2_close(rect.origin, Vec2::new(-20.0, 10.0));
        assert_vec2_close(rect.size, Vec2::new(1440.0, 2560.0));
    }

    #[test]
    fn output_transform_round_trips_every_protocol_variant() {
        let source = Vec2::new(0.23, 0.71);
        let transforms = [
            OutputTransform::Normal,
            OutputTransform::Rotate90,
            OutputTransform::Rotate180,
            OutputTransform::Rotate270,
            OutputTransform::Flipped,
            OutputTransform::Flipped90,
            OutputTransform::Flipped180,
            OutputTransform::Flipped270,
        ];

        for transform in transforms {
            let visual = transform.map_buffer_uv(source);
            assert_vec2_close(transform.unmap_output_uv(visual), source);
            assert_eq!(
                OutputTransform::from_wl_output(transform.as_wl_output()),
                Some(transform)
            );
        }

        assert_vec2_close(
            OutputTransform::Rotate90.map_buffer_uv(Vec2::new(0.0, 0.0)),
            Vec2::new(0.0, 1.0),
        );
    }

    #[test]
    fn current_two_monitor_layout_matches_the_wide_unreal_canvas() {
        let outputs = reference_outputs();
        let desktop = DesktopLayout::from_outputs(&outputs).unwrap();

        assert_vec2_close(desktop.bounds.origin, Vec2::new(0.0, 0.0));
        assert_vec2_close(desktop.bounds.size, Vec2::new(4520.0, 2560.0));

        let dp1 = desktop.projection_slice(outputs[0]).unwrap();
        let dp2 = desktop.projection_slice(outputs[1]).unwrap();
        assert_vec2_close(
            dp1.global_ndc_from_output_uv(Vec2::new(1.0, 757.0 / 2560.0)),
            dp2.global_ndc_from_output_uv(Vec2::new(0.0, 0.0)),
        );

        // DP-1's untransformed buffer has a 90 degree Wayland rotation.
        // This source pixel maps to the exact same global desktop point.
        assert_vec2_close(
            dp1.global_ndc_from_buffer_uv(Vec2::new(1.0 - 757.0 / 2560.0, 1.0)),
            dp2.global_ndc_from_output_uv(Vec2::new(0.0, 0.0)),
        );
    }

    #[test]
    fn projection_is_continuous_across_monitor_boundaries() {
        let outputs = reference_outputs();
        let desktop = DesktopLayout::from_outputs(&outputs).unwrap();
        let projection = GlobalProjection::new(desktop);
        let dp1 = projection.projection_slice(outputs[0]).unwrap();
        let dp2 = projection.projection_slice(outputs[1]).unwrap();
        let pose = OrbitCamera::new(1.0).pose();

        let dp1_ray = projection.ray_from_output_uv(pose, dp1, Vec2::new(1.0, 757.0 / 2560.0));
        let dp2_ray = projection.ray_from_output_uv(pose, dp2, Vec2::new(0.0, 0.0));
        assert_vec3_close(dp1_ray.direction, dp2_ray.direction);
    }

    #[test]
    fn fov_preserves_the_unreal_maintain_y_fov_policy() {
        let desktop =
            DesktopLayout::new(LogicalRect::new(Vec2::ZERO, Vec2::new(1920.0, 1080.0)).unwrap())
                .unwrap();
        let projection = GlobalProjection::new(desktop);

        assert_close(projection.horizontal_fov_degrees(), CAMERA_FOV_X_DEGREES);
        assert!(projection.vertical_fov_degrees() < CAMERA_FOV_X_DEGREES);
    }

    #[test]
    fn invalid_hotplug_geometry_is_rejected() {
        let invalid_scale = OutputGeometry::new(
            Vec2::ZERO,
            PixelSize::new(1920, 1080),
            0.0,
            OutputTransform::Normal,
        );
        assert_eq!(
            invalid_scale.logical_size(),
            Err(ProjectionError::InvalidOutputScale)
        );
        assert_eq!(
            DesktopLayout::from_outputs(&[]),
            Err(ProjectionError::EmptyOutputSet)
        );
    }
}
