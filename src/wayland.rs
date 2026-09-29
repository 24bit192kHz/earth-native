use std::{
    collections::BTreeMap,
    env,
    time::{Duration, Instant, SystemTime},
};

use ash::vk;
use wayland_client::{
    delegate_noop,
    protocol::{
        wl_compositor::WlCompositor,
        wl_keyboard::{self, WlKeyboard},
        wl_output::{self, WlOutput},
        wl_pointer::{self, WlPointer},
        wl_region::WlRegion,
        wl_registry::{self, WlRegistry},
        wl_seat::{self, WlSeat},
        wl_surface::WlSurface,
    },
    Connection, Dispatch, Proxy, QueueHandle, WEnum,
};
use wayland_protocols::xdg::xdg_output::zv1::client::{
    zxdg_output_manager_v1::{self, ZxdgOutputManagerV1},
    zxdg_output_v1::{self, ZxdgOutputV1},
};
use wayland_protocols_wlr::layer_shell::v1::client::{
    zwlr_layer_shell_v1::{self, ZwlrLayerShellV1},
    zwlr_layer_surface_v1::{self, ZwlrLayerSurfaceV1},
};

use crate::{
    astronomy::{
        celestial_state, CelestialState, ASTRONOMICAL_UNIT_KM, EARTH_EQUATORIAL_RADIUS_KM,
        MOON_RADIUS_KM, SUN_RADIUS_KM,
    },
    camera::{
        horizon_dip_degrees, pov_pose, CameraPose, DesktopLayout, GlobalProjection,
        LogicalRect as CameraRect, OrbitCamera, OutputGeometry, OutputTransform, PixelSize,
        PovLook, ProjectionParameters, Vec2, Vec3, POV_MAX_FOV_DEGREES, POV_MIN_FOV_DEGREES,
        SCENE_EARTH_RADIUS,
    },
    debug_scenes::{self, DebugScene},
    ipc,
    orbit::{unix_utc_timestamp, IssTracker},
    vulkan::{FrameUniforms, LogicalRect, Renderer, RendererResult},
};

pub mod x11;

const IDLE_FRAME_RATE: u32 = 15;
// Riding the ISS the ground moves continuously (~16-39 px/s on a 3440 px
// monitor); 20 fps steps it by ~1-2 px, a third less GPU work than 30.
const ONBOARD_FRAME_RATE: u32 = 20;
// The globe's own motion is gated below; this cadence only animates
// lightning flashes and aurora drift (was 15 fps).
const WEATHER_ANIMATION_FRAME_RATE: u32 = 8;
/// Control-mode look-around from the window, at the default 78 degree lens
/// (scaled with the field of view so zoomed-in views turn finer).
const POV_DRAG_DEGREES_PER_PIXEL: f32 = 0.05;
const POV_KEY_DEGREES_PER_SECOND: f32 = 30.0;
/// Control-mode WASD flight over the Earth: ground speed in altitudes per
/// second (~420 km/s from the ISS), Shift for FLY_BOOST times that. It moves
/// only while a key is held.
const FLY_ALTITUDES_PER_SECOND: f64 = 1.0;
const FLY_BOOST: f32 = 4.0;
/// WASD on the globe orbits this many times faster than the arrow keys.
const FLY_ORBIT_AXIS: f32 = 2.5;
const INTERACTIVE_FRAME_RATE: u32 = 30;
const DEBUG_FRAME_RATE: u32 = 10;
const INTERACTIVE_GRACE: Duration = Duration::from_secs(2);
// Motion-adaptive idle (knob 5): skip presents while the accumulated
// projected drift since the last present stays sub-pixel. Structural changes
// keep using `dirty`; only continuous motion (camera easing, ISS tracking,
// the 1 Hz celestial refresh) is gated by this crossing.
const IDLE_MOTION_THRESHOLD_PX: f32 = 0.5;
const IDLE_MIN_INTERVAL: Duration = Duration::from_millis(66);
const IDLE_MAX_INTERVAL: Duration = Duration::from_secs(1);
const LIVE_INITIAL_DISTANCE_SCALE: f32 = 7.3;
const LIVE_FOCUS_Y_OFFSET_FRACTION: f32 = 0.01;

fn frame_interval(frame_rate: u32) -> Duration {
    Duration::from_nanos(1_000_000_000 / u64::from(frame_rate))
}

/// Rolling count of actually-rendered frames over the last whole second.
/// `status` already reports the *target* frame rate; a compositor/GPU stall
/// (slow fence waits, starved swapchain acquires) only shows up as the
/// measured cadence falling behind that target.
#[derive(Debug)]
struct FrameCadence {
    frames: u32,
    measured_per_second: u32,
    window_started: Instant,
}

impl FrameCadence {
    fn new(started: Instant) -> Self {
        Self {
            frames: 0,
            measured_per_second: 0,
            window_started: started,
        }
    }

    fn record_render(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.window_started);
        if elapsed >= Duration::from_secs(1) {
            self.measured_per_second = (self.frames as f64 / elapsed.as_secs_f64()).round() as u32;
            self.frames = 0;
            self.window_started = now;
        }
        self.frames += 1;
    }
}

const MEAN_SUN_DISTANCE_EARTH_RADII: f64 = ASTRONOMICAL_UNIT_KM / EARTH_EQUATORIAL_RADIUS_KM;
const MEAN_MOON_DISTANCE_EARTH_RADII: f64 = 384_400.0 / EARTH_EQUATORIAL_RADIUS_KM;
const IDENTITY_MATRIX_3: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[derive(Clone, Copy, Debug)]
pub struct OutputToken {
    global_name: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceToken {
    output_id: u32,
}

#[derive(Default)]
struct OutputState {
    output: Option<WlOutput>,
    xdg_output: Option<ZxdgOutputV1>,
    name: Option<String>,
    mode: PixelSize,
    scale: i32,
    transform: OutputTransform,
    logical_position: Option<Vec2>,
    logical_size: Option<Vec2>,
}

impl OutputState {
    fn scale_factor(&self) -> i32 {
        self.scale.max(1)
    }

    fn fallback_geometry(&self) -> Option<OutputGeometry> {
        if self.mode.is_empty() {
            return None;
        }
        Some(OutputGeometry::new(
            self.logical_position.unwrap_or(Vec2::ZERO),
            self.mode,
            self.scale_factor() as f32,
            self.transform,
        ))
    }

    fn logical_rect(&self) -> Option<CameraRect> {
        if let (Some(position), Some(size)) = (self.logical_position, self.logical_size) {
            return CameraRect::new(position, size).ok();
        }
        self.fallback_geometry()?.logical_rect().ok()
    }
}

struct LayerState {
    surface: WlSurface,
    layer_surface: ZwlrLayerSurfaceV1,
    configured_width: u32,
    configured_height: u32,
}

#[derive(Default)]
struct PointerState {
    output_id: Option<u32>,
    last_position: Option<(f64, f64)>,
    dragging: bool,
}

/// The live view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ViewMode {
    /// A camera aboard the ISS (SGP4 position and flight direction), the
    /// default: the Earth as the Earth Observation photographs show it.
    Onboard,
    /// The whole globe from far away, centred under the ISS.
    Globe,
    /// A fixed onboard-style viewpoint (tests and aimed captures).
    FixedPov { latitude: f32, longitude: f32, altitude_km: f32 },
}

/// Cached [`GlobalProjection::parameters`] output plus the resolved focus point.
/// Both derive from the desktop bounds, the focus-output name, and the debug
/// flag, so they are recomputed only when those inputs change instead of once
/// per frame. Tan/aspect math lives in [`GlobalProjection::new`]; this only
/// memoizes its result.
#[derive(Clone, Debug, PartialEq)]
struct CachedProjection {
    bounds: CameraRect,
    params: ProjectionParameters,
    focus: Vec2,
    /// Logical size of the monitor the view is centred on.
    focus_size: Vec2,
    focus_output_name: String,
    debug_mode: bool,
}

pub struct NativeApp {
    // Rust drops fields in declaration order. Vulkan must release every
    // VkSurfaceKHR while its wl_surface and wl_display proxies are still live.
    renderer: Renderer,
    // Xorg frontend (None under Wayland). Declared after the renderer so the
    // Vulkan surfaces go before their X windows and connection.
    x11: Option<x11::X11Parts>,
    // Wayland frontend (None under Xorg).
    display: Option<wayland_client::protocol::wl_display::WlDisplay>,
    queue_handle: Option<QueueHandle<Self>>,
    compositor: Option<WlCompositor>,
    layer_shell: Option<ZwlrLayerShellV1>,
    xdg_output_manager: Option<ZxdgOutputManagerV1>,
    outputs: BTreeMap<u32, OutputState>,
    layers: BTreeMap<u32, LayerState>,
    camera: OrbitCamera,
    fixed_camera: bool,
    /// What the live view shows; `camera` over IPC switches it.
    view_mode: ViewMode,
    pov_look: PovLook,
    /// Pitch the "auto" framing last chose, so looking around starts there.
    pov_auto_pitch: f32,
    /// Last onboard state, for status.
    iss_state: Option<crate::orbit::IssState>,
    iss: IssTracker,
    desktop: Option<DesktopLayout>,
    pointer: PointerState,
    controlled_output: Option<u32>,
    keyboard_yaw: f32,
    keyboard_pitch: f32,
    /// Held WASD keys (W, A, S, D) and Shift, for control-mode flight.
    fly_keys: [bool; 4],
    shift_held: bool,
    ctrl_held: bool,
    interactive_until: Instant,
    last_tick: Instant,
    last_render_ms: f32,
    frame_cadence: FrameCadence,
    fixed_unix_seconds: Option<i64>,
    active_unix_seconds: i64,
    last_astronomy_second: i64,
    last_astronomy_error: Option<String>,
    celestial_state: CelestialState,
    dirty: bool,
    // Output events only mark the desktop layout stale; the recompute and the
    // renderer viewport updates are applied once at the top of render_frame,
    // so a hotplug burst no longer repeats the layout/projection work per event.
    layout_dirty: bool,
    // Last presented scene vectors plus the motion gate (knob 5). `prev_*`
    // is the previous timer sample for the drift-rate estimate; `presented_*`
    // is the last actually presented frame the skip threshold measures
    // against. `idle_interval` arms the event-loop timer to the predicted
    // threshold crossing so wakeups also drop with presents.
    presented_forward: Option<[f32; 3]>,
    presented_distance: f32,
    presented_sun: [f32; 3],
    presented_moon: [f32; 3],
    prev_forward: Option<[f32; 3]>,
    prev_distance: f32,
    prev_sun: [f32; 3],
    prev_moon: [f32; 3],
    prev_sample_at: Option<Instant>,
    motion_rate_px_per_s: f32,
    renderer_clean: bool,
    idle_interval: Duration,
    stopping: bool,
    // Memoized GlobalProjection parameters + focus point, keyed on the desktop
    // bounds, focus_output_name, and debug_mode. Recomputed only when one of
    // those inputs changes; render_frame reads the cache when clean.
    cached_projection: Option<CachedProjection>,
    celestial_sun: Option<[f32; 3]>,
    celestial_moon: Option<[f32; 3]>,
    celestial_sun_distance_earth_radii: Option<f32>,
    celestial_moon_distance_earth_radii: Option<f32>,
    focus_output_name: String,
    debug_mode: bool,
    debug_scene_index: usize,
    debug_scene_started: Instant,
    debug_scene: DebugScene,
}

impl NativeApp {
    pub fn new(
        display: wayland_client::protocol::wl_display::WlDisplay,
        queue_handle: QueueHandle<Self>,
        renderer: Renderer,
        debug_mode: bool,
    ) -> RendererResult<Self> {
        Self::with_frontend(Some(display), Some(queue_handle), renderer, debug_mode)
    }

    /// An Xorg-hosted app; `x11::run` attaches the windows afterwards.
    pub fn new_x11(renderer: Renderer, debug_mode: bool) -> RendererResult<Self> {
        Self::with_frontend(None, None, renderer, debug_mode)
    }

    fn with_frontend(
        display: Option<wayland_client::protocol::wl_display::WlDisplay>,
        queue_handle: Option<QueueHandle<Self>>,
        renderer: Renderer,
        debug_mode: bool,
    ) -> RendererResult<Self> {
        let now = Instant::now();
        let (unix_seconds, _) = unix_utc_timestamp(SystemTime::now())
            .ok_or("system time is outside the supported Unix timestamp range")?;
        let initial_celestial_state = celestial_state(unix_seconds)?;
        let mut app = Self {
            renderer,
            x11: None,
            display,
            queue_handle,
            compositor: None,
            layer_shell: None,
            xdg_output_manager: None,
            outputs: BTreeMap::new(),
            layers: BTreeMap::new(),
            camera: OrbitCamera::new(SCENE_EARTH_RADIUS),
            fixed_camera: false,
            view_mode: ViewMode::Onboard,
            pov_look: PovLook::default(),
            pov_auto_pitch: 0.0,
            iss_state: None,
            iss: IssTracker::new(),
            desktop: None,
            pointer: PointerState::default(),
            presented_forward: None,
            presented_distance: 0.0,
            presented_sun: [0.0; 3],
            presented_moon: [0.0; 3],
            prev_forward: None,
            prev_distance: 0.0,
            prev_sun: [0.0; 3],
            prev_moon: [0.0; 3],
            prev_sample_at: None,
            motion_rate_px_per_s: 0.0,
            renderer_clean: true,
            idle_interval: IDLE_MIN_INTERVAL,
            controlled_output: None,
            keyboard_yaw: 0.0,
            keyboard_pitch: 0.0,
            fly_keys: [false; 4],
            shift_held: false,
            ctrl_held: false,
            interactive_until: now,
            last_tick: now,
            last_render_ms: 0.0,
            frame_cadence: FrameCadence::new(now),
            fixed_unix_seconds: None,
            active_unix_seconds: unix_seconds,
            last_astronomy_second: unix_seconds,
            last_astronomy_error: None,
            celestial_state: initial_celestial_state,
            dirty: true,
            layout_dirty: false,
            cached_projection: None,
            stopping: false,
            celestial_sun: None,
            celestial_moon: None,
            celestial_sun_distance_earth_radii: None,
            celestial_moon_distance_earth_radii: None,
            // Empty selects the largest monitor (see `focus_point`).
            focus_output_name: env::var("EARTH_NATIVE_FOCUS_OUTPUT").unwrap_or_default(),
            debug_mode,
            debug_scene_index: 0,
            debug_scene_started: now,
            debug_scene: debug_scenes::scene(0),
        };
        app.camera
            .set_distance_scale_immediate(LIVE_INITIAL_DISTANCE_SCALE);
        if debug_mode {
            app.apply_debug_scene();
        }
        Ok(app)
    }

    pub fn request_stop(&mut self) {
        self.stopping = true;
    }

    pub fn stopping(&self) -> bool {
        self.stopping
    }

    /// Must run before calloop drops its Wayland connection.
    pub fn shutdown_renderer(&mut self) {
        self.renderer.destroy_outputs();
    }

    pub fn next_frame_interval(&self) -> Duration {
        if self.debug_mode {
            frame_interval(DEBUG_FRAME_RATE)
        } else if self.is_interactive(Instant::now()) {
            frame_interval(INTERACTIVE_FRAME_RATE)
        } else if self.riding_iss() {
            frame_interval(ONBOARD_FRAME_RATE)
        } else if self.fixed_unix_seconds.is_none() && self.renderer.has_weather_animation() {
            frame_interval(WEATHER_ANIMATION_FRAME_RATE)
        } else {
            self.idle_interval
        }
    }

    fn current_frame_rate(&self) -> u32 {
        if self.debug_mode {
            DEBUG_FRAME_RATE
        } else if self.is_interactive(Instant::now()) {
            INTERACTIVE_FRAME_RATE
        } else if self.riding_iss() {
            ONBOARD_FRAME_RATE
        } else if self.fixed_unix_seconds.is_none() && self.renderer.has_weather_animation() {
            WEATHER_ANIMATION_FRAME_RATE
        } else {
            IDLE_FRAME_RATE
        }
    }

    fn riding_iss(&self) -> bool {
        !self.debug_mode && !self.fixed_camera && self.view_mode == ViewMode::Onboard
            && self.renderer.body() == crate::body::Body::Earth
    }

    fn pov_active(&self) -> bool {
        !self.debug_mode && !self.fixed_camera && self.renderer.body() == crate::body::Body::Earth
            && matches!(self.view_mode, ViewMode::Onboard | ViewMode::FixedPov { .. })
    }

    /// The onboard pose for this instant, or None to use the orbit camera.
    fn onboard_pose(&mut self, seconds: i64, microseconds: i32, projection: &CachedProjection) -> Option<(CameraPose, f32)> {
        if !self.pov_active() {
            return None;
        }
        let (position, reference, altitude_km) = match self.view_mode {
            ViewMode::Onboard => {
                let state = self.iss.onboard_state(seconds, microseconds)?;
                self.iss_state = Some(state);
                (state.position, state.velocity, state.altitude_km)
            }
            ViewMode::FixedPov { latitude, longitude, altitude_km } => {
                let direction = crate::orbit::scene_direction(f64::from(latitude).to_radians(), f64::from(longitude).to_radians());
                let radius = f64::from(SCENE_EARTH_RADIUS) * (1.0 + f64::from(altitude_km) / EARTH_EQUATORIAL_RADIUS_KM);
                let position = Vec3::new((direction[0] * radius) as f32, (direction[1] * radius) as f32, (direction[2] * radius) as f32);
                // North in the mirrored frame.
                let north = Vec3::new(
                    (-f64::from(latitude).to_radians().sin() * f64::from(longitude).to_radians().cos()) as f32,
                    (f64::from(latitude).to_radians().sin() * f64::from(longitude).to_radians().sin()) as f32,
                    f64::from(latitude).to_radians().cos() as f32,
                );
                (position, north, f64::from(altitude_km))
            }
            ViewMode::Globe => return None,
        };
        let zoom = self.pov_zoom(projection);
        let pitch = match self.pov_look.pitch_degrees {
            Some(pitch) => pitch,
            None => {
                // Horizon ~20 % below the top of the focus monitor.
                let tan_half_y = projection.params.tan_half_fov_y * zoom * projection.focus_size.y / projection.bounds.size.y;
                self.pov_auto_pitch = horizon_dip_degrees(altitude_km) + (0.6 * tan_half_y).atan().to_degrees();
                self.pov_auto_pitch
            }
        };
        Some((pov_pose(position, reference, self.pov_look.heading_degrees, pitch), zoom))
    }

    /// Factor on the projection tangents that gives the onboard lens its
    /// horizontal field of view across the focus monitor.
    fn pov_zoom(&self, projection: &CachedProjection) -> f32 {
        let focus_fraction = (projection.focus_size.x / projection.bounds.size.x).max(1.0e-3);
        let wanted = (self.pov_look.fov_degrees.clamp(POV_MIN_FOV_DEGREES, POV_MAX_FOV_DEGREES).to_radians() * 0.5).tan();
        wanted / (projection.params.tan_half_fov_x * focus_fraction).max(1.0e-6)
    }

    fn focus_rect(&self, bounds: CameraRect) -> CameraRect {
        let named = self
            .outputs
            .values()
            .find(|output| output.name.as_deref() == Some(self.focus_output_name.as_str()))
            .and_then(OutputState::logical_rect);
        let largest = || {
            self.outputs
                .values()
                .filter_map(OutputState::logical_rect)
                .max_by(|a, b| (a.size.x * a.size.y).total_cmp(&(b.size.x * b.size.y)))
        };
        named.or_else(largest).unwrap_or(bounds)
    }

    fn focus_point(&self, bounds: CameraRect) -> Vec2 {
        // The named focus output, else the largest monitor, else the desktop.
        let named = self
            .outputs
            .values()
            .find(|output| output.name.as_deref() == Some(self.focus_output_name.as_str()))
            .and_then(OutputState::logical_rect);
        let largest = || {
            self.outputs
                .values()
                .filter_map(OutputState::logical_rect)
                .max_by(|a, b| (a.size.x * a.size.y).total_cmp(&(b.size.x * b.size.y)))
        };
        let focus_rect = named.or_else(largest).unwrap_or(bounds);
        let mut focus = focus_rect.center();
        if !self.debug_mode {
            focus.y += focus_rect.size.y * LIVE_FOCUS_Y_OFFSET_FRACTION;
        }
        focus
    }

    /// Recompute the cached projection parameters and focus point from the
    /// current desktop bounds, focus-output selection, and debug flag. Runs
    /// only when one of those inputs changes; the per-frame path reads the
    /// cache. The tan/aspect math is unchanged: it still flows through
    /// [`GlobalProjection::new`].
    fn refresh_projection_cache(&mut self) {
        let Some(desktop) = self.desktop else {
            self.cached_projection = None;
            return;
        };
        let bounds = desktop.bounds;
        let params = GlobalProjection::new(desktop).parameters();
        let focus = self.focus_point(bounds);
        let focus_size = self.focus_rect(bounds).size;
        self.cached_projection = Some(CachedProjection {
            bounds,
            params,
            focus,
            focus_size,
            focus_output_name: self.focus_output_name.clone(),
            debug_mode: self.debug_mode,
        });
    }

    pub fn render_frame(&mut self) -> RendererResult<()> {
        let now = Instant::now();
        if self.layout_dirty {
            self.update_desktop_layout();
            self.layout_dirty = false;
        }
        let wall_timestamp = unix_utc_timestamp(SystemTime::now())
            .ok_or("system time is outside the supported Unix timestamp range")?;
        let (active_seconds, active_microseconds) = if self.debug_mode {
            (1_710_936_000, 0) // 2024-03-20 12:00 UTC, repeatable weather animation.
        } else {
            active_timestamp(self.fixed_unix_seconds, wall_timestamp)
        };
        self.active_unix_seconds = active_seconds;
        let elapsed = now.saturating_duration_since(self.last_tick);
        self.last_tick = now;
        if self.debug_mode {
            self.advance_debug_scene(now);
        }
        // Camera easing mutates the pose every tick while settling, but at
        // idle that drift is usually sub-pixel: the motion gate below decides
        // whether it needs a present. Structural `dirty` is untouched.
        self.camera.tick(elapsed.as_secs_f32());
        self.fly(elapsed.as_secs_f32());
        if self.pov_active() && (self.keyboard_yaw != 0.0 || self.keyboard_pitch != 0.0) {
            let degrees = POV_KEY_DEGREES_PER_SECOND * self.pov_scale() * elapsed.as_secs_f32().min(0.1);
            self.turn_pov(self.keyboard_yaw * degrees, -self.keyboard_pitch * degrees);
        }

        if !self.debug_mode && !self.fixed_camera && self.view_mode == ViewMode::Globe
            && self.renderer.body() == crate::body::Body::Earth {
            if let Some(direction) = self
                .iss
                .update_unix_utc(active_seconds, active_microseconds)
            {
                // ISS tracking pans the camera base continuously; like easing
                // above, the accumulated drift below gates its presents.
                self.camera.set_base_orbit_direction(direction);
            }
        }
        self.update_celestial_state(active_seconds);

        // Motion-adaptive idle: continuous drift (easing, ISS, celestial) no
        // longer sets `dirty`, so present once its accumulated projection
        // crosses the threshold. Below it, skip the present and re-arm the
        // timer for the predicted crossing. Debug, input, structural dirt,
        // the first frame, and a renderer that asked for another frame all
        // present exactly as before.
        if self.debug_mode {
            if !self.dirty && !self.is_interactive(now) {
                return Ok(());
            }
        } else if !self.dirty
            && !self.riding_iss()
            && !(self.fixed_unix_seconds.is_none() && self.renderer.has_weather_animation())
            && !self.is_interactive(now)
            && self.renderer_clean
            && self.presented_forward.is_some()
        {
            let motion_px = self.idle_motion_px(now);
            if motion_px < IDLE_MOTION_THRESHOLD_PX {
                let remaining = IDLE_MOTION_THRESHOLD_PX - motion_px;
                let predicted_secs = remaining / self.motion_rate_px_per_s.max(1.0e-3);
                self.idle_interval = Duration::from_secs_f32(predicted_secs)
                    .clamp(IDLE_MIN_INTERVAL, IDLE_MAX_INTERVAL);
                return Ok(());
            }
            self.idle_interval = IDLE_MIN_INTERVAL;
        }
        let Some(desktop) = self.desktop else {
            return Ok(());
        };
        let celestial = self.celestial_state;
        // The debug scene pins the Sun for Earth regression views; every other
        // body gets its own currently correct Sun against the fixed stars.
        let body = self.renderer.body();
        let debug_sun_active = self.debug_mode && body == crate::body::Body::Earth;
        let sun_direction = if debug_sun_active {
            let direction = self.debug_scene.sun_direction;
            [direction.x, direction.y, direction.z]
        } else if self.celestial_sun.is_none() {
            celestial.body_sun_directions[body.id() as usize]
        } else {
            self.celestial_sun.unwrap_or(celestial.sun_direction)
        };
        let moon_direction = if self.debug_mode {
            let direction = self.debug_scene.moon_direction;
            [direction.x, direction.y, direction.z]
        } else {
            self.celestial_moon.unwrap_or(celestial.moon_direction)
        };
        let sun_distance_earth_radii = if debug_sun_active {
            MEAN_SUN_DISTANCE_EARTH_RADII as f32
        } else if self.celestial_sun_distance_earth_radii.is_none() {
            celestial.body_sun_distances_earth_radii[body.id() as usize]
        } else {
            self.celestial_sun_distance_earth_radii
                .unwrap_or(celestial.sun_distance_earth_radii)
        };
        let moon_distance_earth_radii = if self.debug_mode {
            MEAN_MOON_DISTANCE_EARTH_RADII as f32
        } else {
            self.celestial_moon_distance_earth_radii
                .unwrap_or(celestial.moon_distance_earth_radii)
        };
        let (
            moon_world_to_body,
            greenwich_sidereal_radians,
            moon_illuminated_fraction,
            moon_phase_angle_radians,
        ) = if self.debug_mode && body == crate::body::Body::Earth {
            let (fraction, phase) = debug_moon_phase(sun_direction, moon_direction);
            (IDENTITY_MATRIX_3, 0.0, fraction, phase)
        } else {
            (
                celestial.moon_world_to_body,
                celestial.greenwich_apparent_sidereal_angle_radians,
                celestial.moon_illuminated_fraction,
                celestial.moon_phase_angle_radians,
            )
        };
        let bounds = desktop.bounds;
        // The outputs-map scan inside focus_point and the GlobalProjection tan
        // math run only when the cache key changes (bounds, focus-output name,
        // or debug flag). A clean frame only compares the key, so hotplug and
        // focus/debug changes each pay exactly one recompute.
        let cache_current = self.cached_projection.as_ref().is_some_and(|cached| {
            cached.bounds == bounds
                && cached.focus_output_name == self.focus_output_name
                && cached.debug_mode == self.debug_mode
        });
        if !cache_current {
            self.refresh_projection_cache();
        }
        let Some(cached) = self.cached_projection.clone() else {
            return Ok(());
        };
        let focus = cached.focus;
        let (pose, projection) = match self.onboard_pose(active_seconds, active_microseconds, &cached) {
            Some((pose, zoom)) => (pose, ProjectionParameters {
                tan_half_fov_x: cached.params.tan_half_fov_x * zoom,
                tan_half_fov_y: cached.params.tan_half_fov_y * zoom,
            }),
            None => (self.camera.pose(), cached.params),
        };
        let camera_distance = pose.position.length();
        let uniforms = FrameUniforms {
            unix_seconds: active_seconds,
            nasa_materials: false,
            nasa_clouds: false,
            weather_valid_unix_utc: 0,
            aurora_valid_unix_utc: 0,
            live_clouds: false,
            live_aerosol: false,
            live_sea_ice: false,
            camera_position: [pose.position.x, pose.position.y, pose.position.z],
            camera_distance,
            forward: [pose.forward.x, pose.forward.y, pose.forward.z],
            right: [pose.right.x, pose.right.y, pose.right.z],
            up: [pose.up.x, pose.up.y, pose.up.z],
            canvas: LogicalRect {
                x: bounds.origin.x,
                y: bounds.origin.y,
                width: bounds.size.x,
                height: bounds.size.y,
            },
            sun_direction,
            moon_direction,
            sun_distance_earth_radii,
            moon_distance_earth_radii,
            sun_radius_earth_radii: (SUN_RADIUS_KM / EARTH_EQUATORIAL_RADIUS_KM) as f32,
            moon_radius_earth_radii: (MOON_RADIUS_KM / EARTH_EQUATORIAL_RADIUS_KM) as f32,
            moon_world_to_body: if body == crate::body::Body::Earth {
                moon_world_to_body
            } else {
                celestial.body_world_to_fixed[body.id() as usize]
            },
            greenwich_sidereal_radians,
            moon_illuminated_fraction,
            moon_phase_angle_radians,
            time_of_day_seconds: ((active_seconds % 86_400) as f32)
                + (active_microseconds as f32) * 1.0e-6,
            tan_half_fov_x: projection.tan_half_fov_x,
            tan_half_fov_y: projection.tan_half_fov_y,
            focus_x: focus.x,
            focus_y: focus.y,
            star_eqj_to_world: celestial.eqj_to_world,
            star_aberration: celestial.earth_velocity_over_c,
        };
        let render_started = Instant::now();
        // EARTH_NATIVE_CONTINUOUS=1 disables the motion gate, so a pinned
        // scene keeps rendering: a steady GPU benchmark for A/B work.
        let still_dirty = self.renderer.render(uniforms)? || continuous_rendering();
        self.dirty = still_dirty;
        self.renderer_clean = !still_dirty;
        // The frame on screen now matches the current vectors: rebase the
        // motion gate here (and only here) so skipped drift never compounds.
        let pose = self.camera.pose();
        self.presented_forward = Some([pose.forward.x, pose.forward.y, pose.forward.z]);
        self.presented_distance = self.camera.distance();
        let (sun_now, moon_now) = self.effective_celestial();
        self.presented_sun = sun_now;
        self.presented_moon = moon_now;
        self.prev_forward = self.presented_forward;
        self.prev_distance = self.presented_distance;
        self.prev_sun = sun_now;
        self.prev_moon = moon_now;
        self.prev_sample_at = Some(render_started);
        self.last_render_ms = render_started.elapsed().as_secs_f32() * 1000.0;
        self.frame_cadence.record_render(render_started);
        Ok(())
    }

    fn update_celestial_state(&mut self, unix_seconds: i64) {
        if unix_seconds == self.last_astronomy_second {
            return;
        }
        self.last_astronomy_second = unix_seconds;
        match celestial_state(unix_seconds) {
            Ok(state) => {
                self.celestial_state = state;
                self.last_astronomy_error = None;
                // No `dirty` here: the 1 Hz refresh drifts the terminator by
                // micro-pixels per tick; the motion gate presents once the
                // accumulated drift crosses the threshold instead.
            }
            Err(error) => {
                // Sanitize once at set time (<=1 Hz, error path only) so
                // status reads below stay allocation-flat.
                self.last_astronomy_error = Some(format!(
                    "error:{}",
                    error.to_string().split_whitespace().collect::<Vec<_>>().join("_")
                ))
            }
        }
    }

    /// Sun/Moon vectors currently on screen, mirroring the selection in
    /// `render_frame` (debug pins, socket overrides, live ephemeris).
    fn effective_celestial(&self) -> ([f32; 3], [f32; 3]) {
        let body = self.renderer.body();
        let celestial = &self.celestial_state;
        let sun = if self.debug_mode && body == crate::body::Body::Earth {
            let direction = self.debug_scene.sun_direction;
            [direction.x, direction.y, direction.z]
        } else if self.celestial_sun.is_none() {
            celestial.body_sun_directions[body.id() as usize]
        } else {
            self.celestial_sun.unwrap_or(celestial.sun_direction)
        };
        let moon = if self.debug_mode {
            let direction = self.debug_scene.moon_direction;
            [direction.x, direction.y, direction.z]
        } else {
            self.celestial_moon.unwrap_or(celestial.moon_direction)
        };
        (sun, moon)
    }

    /// Densest output's pixels per scene radian, from the cached shared
    /// projection. Every output shows a slice of the same canvas, so the
    /// canvas height times the largest scale factor covers all outputs.
    fn pixels_per_radian(&self) -> Option<f32> {
        let cached = self.cached_projection.as_ref()?;
        let tan = cached.params.tan_half_fov_y;
        if !tan.is_finite() || tan <= 0.0 {
            return None;
        }
        let canvas_h = cached.bounds.size.y;
        if !canvas_h.is_finite() || canvas_h <= 0.0 {
            return None;
        }
        let max_scale = self
            .outputs
            .values()
            .map(|output| output.scale_factor())
            .max()
            .unwrap_or(1)
            .max(1) as f32;
        Some(canvas_h * max_scale / (2.0 * tan))
    }

    /// Projected drift since the last present, in pixels, while updating the
    /// drift-rate estimate from consecutive timer samples. Camera rotation
    /// covers both slew and orbital parallax to first order; the zoom term
    /// scales the disk radius by the fractional distance change.
    fn idle_motion_px(&mut self, now: Instant) -> f32 {
        // Chord length between (near-)unit vectors: identical to the angle
        // below ~1e-3 rad, where f32 acos(dot) loses all precision to
        // catastrophic cancellation. Per-tick drift here is ~1e-4 rad, so
        // acos would quantize every sample and randomize the rate estimate.
        fn angle(a: [f32; 3], b: [f32; 3]) -> f32 {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        }
        let pose = self.camera.pose();
        let forward = [pose.forward.x, pose.forward.y, pose.forward.z];
        let distance = self.camera.distance();
        let (sun, moon) = self.effective_celestial();
        let Some(ppr) = self.pixels_per_radian() else {
            // No projection yet: report matured motion so this frame presents
            // and rebases the gate.
            return IDLE_MOTION_THRESHOLD_PX;
        };
        let disk_px = ppr * (SCENE_EARTH_RADIUS / distance.max(1.0e-5)).asin();
        let presented = self.presented_forward.unwrap_or(forward);
        let motion = angle(forward, presented) * ppr
            + (distance - self.presented_distance).abs() / distance.max(1.0e-5) * disk_px
            + angle(sun, self.presented_sun) * ppr
            + angle(moon, self.presented_moon) * ppr;
        if let (Some(prev), Some(sampled_at)) = (self.prev_forward, self.prev_sample_at) {
            let dt = now.saturating_duration_since(sampled_at).as_secs_f32();
            if dt > 1.0e-3 {
                let step = angle(forward, prev) * ppr
                    + (distance - self.prev_distance).abs() / distance.max(1.0e-5) * disk_px
                    + angle(sun, self.prev_sun) * ppr
                    + angle(moon, self.prev_moon) * ppr;
                let instant = step / dt;
                if instant.is_finite() {
                    self.motion_rate_px_per_s +=
                        (instant - self.motion_rate_px_per_s) * 0.25;
                }
            }
        }
        self.prev_forward = Some(forward);
        self.prev_distance = distance;
        self.prev_sun = sun;
        self.prev_moon = moon;
        self.prev_sample_at = Some(now);
        motion
    }

    fn apply_debug_scene(&mut self) {
        self.camera.set_base_orbit_direction(Vec3::X);
        self.camera
            .set_orbit_angles(self.debug_scene.camera_yaw, self.debug_scene.camera_pitch);
        self.camera
            .set_distance_scale_immediate(self.debug_scene.camera_distance);
        self.renderer.set_debug_scene(self.debug_scene.name);
        self.dirty = true;
    }

    fn advance_debug_scene(&mut self, now: Instant) {
        if self.debug_scene_index + 1 >= debug_scenes::SCENE_COUNT
            || now.duration_since(self.debug_scene_started) < debug_scenes::SCENE_DURATION
        {
            return;
        }
        self.debug_scene_index += 1;
        self.debug_scene_started = now;
        self.debug_scene = debug_scenes::scene(self.debug_scene_index);
        self.apply_debug_scene();
    }

    pub fn handle_socket_request(&mut self, request: ipc::Request) -> String {
        match request {
            ipc::Request::Status => self.status_line(),
            ipc::Request::Camera { yaw, pitch, distance } => {
                let values = [yaw, pitch, distance].map(|value| value.parse::<f32>().unwrap_or(f32::NAN));
                if !values.iter().all(|value| value.is_finite())
                    || !(-89.0..=89.0).contains(&values[1])
                    || !(crate::camera::MIN_DISTANCE_SCALE..=crate::camera::MAX_DISTANCE_SCALE).contains(&values[2]) {
                    return "error camera requires finite degrees, pitch [-89,89], and valid zoom distance".into();
                }
                if self.debug_mode { return "error camera setter requires normal mode".into(); }
                self.fixed_camera = true;
                self.renderer.snap_exposure();
                self.camera.set_base_orbit_direction(Vec3::X);
                self.camera.set_orbit_angles(values[0], values[1]);
                self.camera.set_distance_scale_immediate(values[2]);
                self.dirty = true;
                "ok camera=fixed".into()
            }
            ipc::Request::CameraLive => {
                self.fixed_camera = false;
                self.view_mode = ViewMode::Onboard;
                self.pov_look = PovLook::default();
                self.renderer.snap_exposure();
                self.dirty = true;
                "ok camera=live view=iss".into()
            }
            ipc::Request::CameraGlobe => {
                self.fixed_camera = false;
                self.view_mode = ViewMode::Globe;
                self.renderer.snap_exposure();
                self.dirty = true;
                "ok camera=live view=globe".into()
            }
            ipc::Request::CameraNext => {
                let view = self.cycle_view();
                format!("ok camera=live view={view}")
            }
            ipc::Request::CameraZoom { steps } => {
                if self.control_zoom(steps as f32) {
                    self.mark_interactive();
                }
                format!("ok fov_degrees={:.1}", self.pov_look.fov_degrees)
            }
            ipc::Request::CameraReset => {
                self.reset_look();
                "ok camera=reset".into()
            }
            ipc::Request::CameraAurora => self.aim_at_aurora(),
            ipc::Request::CameraIss { heading, pitch, fov } => match parse_look(&heading, &pitch, &fov) {
                Ok(look) => {
                    if self.debug_mode { return "error camera setter requires normal mode".into(); }
                    self.fixed_camera = false;
                    self.view_mode = ViewMode::Onboard;
                    self.pov_look = look;
                    self.renderer.snap_exposure();
                    self.dirty = true;
                    "ok camera=live view=iss".into()
                }
                Err(error) => format!("error {error}"),
            },
            ipc::Request::CameraPov { latitude, longitude, altitude, heading, pitch, fov } => {
                let place = [latitude, longitude, altitude].map(|value| value.parse::<f32>().unwrap_or(f32::NAN));
                if !place.iter().all(|value| value.is_finite()) || !(-90.0..=90.0).contains(&place[0])
                    || !(150.0..=40000.0).contains(&place[2]) {
                    return "error pov needs latitude [-90,90], longitude, altitude 150-40000 km".into();
                }
                match parse_look(&heading, &pitch, &fov) {
                    Ok(look) => {
                        if self.debug_mode { return "error camera setter requires normal mode".into(); }
                        self.fixed_camera = false;
                        self.view_mode = ViewMode::FixedPov { latitude: place[0], longitude: place[1], altitude_km: place[2] };
                        self.pov_look = look;
                        self.renderer.snap_exposure();
                        self.dirty = true;
                        "ok camera=pov".into()
                    }
                    Err(error) => format!("error {error}"),
                }
            }
            ipc::Request::CaptureFrame => match self.renderer.capture_frame() {
                Ok(mut ticket) => {
                    self.dirty = true;
                    ticket["state"] = self.status_line().into();
                    ticket.to_string()
                }
                Err(error) => format!("error {error}"),
            },
            ipc::Request::WeatherReload => match self.renderer.reload_weather() {
                Ok(()) => { self.dirty = true; "ok weather queued".to_owned() }
                Err(error) => format!("error {error}"),
            },
            ipc::Request::Stop => {
                self.stopping = true;
                "ok stopping".to_owned()
            }
            ipc::Request::Control { monitor } => match self.toggle_control(&monitor) {
                Ok(enabled) => {
                    if enabled {
                        format!("ok control {monitor}")
                    } else {
                        "ok control released".to_owned()
                    }
                }
                Err(error) => format!("error {error}"),
            },
            ipc::Request::Body { body } => match crate::body::Body::parse(&body) {
                Some(body) => {
                    self.renderer.set_body(body);
                    self.dirty = true;
                    format!("ok body {}", body.name())
                }
                None => "error body must be earth, jupiter, mercury, mars, or saturn".to_owned(),
            },
            ipc::Request::CelestialShow => self.celestial_status(),
            ipc::Request::CelestialFreeze => match unix_utc_timestamp(SystemTime::now()) {
                Some((seconds, _)) => {
                    self.set_fixed_time(seconds);
                    self.celestial_status()
                }
                None => {
                    "error system time is outside the supported Unix timestamp range".to_owned()
                }
            },
            ipc::Request::CelestialSun { yaw, pitch } => match celestial_direction(&yaw, &pitch) {
                Ok(direction) => {
                    self.celestial_sun = Some(direction);
                    self.celestial_sun_distance_earth_radii =
                        Some(MEAN_SUN_DISTANCE_EARTH_RADII as f32);
                    self.dirty = true;
                    self.celestial_status()
                }
                Err(error) => format!("error {error}"),
            },
            ipc::Request::CelestialMoon { yaw, pitch } => match celestial_direction(&yaw, &pitch) {
                Ok(direction) => {
                    self.celestial_moon = Some(direction);
                    self.celestial_moon_distance_earth_radii =
                        Some(MEAN_MOON_DISTANCE_EARTH_RADII as f32);
                    self.dirty = true;
                    self.celestial_status()
                }
                Err(error) => format!("error {error}"),
            },
            ipc::Request::CelestialLive => {
                self.celestial_sun = None;
                self.celestial_moon = None;
                self.celestial_sun_distance_earth_radii = None;
                self.celestial_moon_distance_earth_radii = None;
                self.restore_live_time();
                self.celestial_status()
            }
            ipc::Request::TimeShow => self.time_status(),
            ipc::Request::TimeUnix { seconds } => {
                self.set_fixed_time(seconds);
                self.time_status()
            }
            ipc::Request::TimeLive => {
                self.restore_live_time();
                self.time_status()
            }
        }
    }

    fn set_fixed_time(&mut self, seconds: i64) {
        self.renderer.snap_exposure();
        self.fixed_unix_seconds = Some(seconds);
        self.active_unix_seconds = seconds;
        self.update_celestial_state(seconds);
        self.dirty = true;
    }

    fn restore_live_time(&mut self) {
        self.fixed_unix_seconds = None;
        if let Some((seconds, _)) = unix_utc_timestamp(SystemTime::now()) {
            self.active_unix_seconds = seconds;
            self.update_celestial_state(seconds);
        }
        self.dirty = true;
    }

    fn displayed_celestial_fields(
        &self,
    ) -> (
        [f32; 3],
        [f32; 3],
        f32,
        f32,
        f32,
        f32,
        &'static str,
        &'static str,
    ) {
        if self.debug_mode {
            let sun = direction_array(self.debug_scene.sun_direction);
            let moon = direction_array(self.debug_scene.moon_direction);
            let (illumination, phase) = debug_moon_phase(sun, moon);
            (
                sun,
                moon,
                MEAN_SUN_DISTANCE_EARTH_RADII as f32,
                MEAN_MOON_DISTANCE_EARTH_RADII as f32,
                illumination,
                phase,
                "debug",
                "debug",
            )
        } else {
            (
                self.celestial_sun
                    .unwrap_or(self.celestial_state.sun_direction),
                self.celestial_moon
                    .unwrap_or(self.celestial_state.moon_direction),
                self.celestial_sun_distance_earth_radii
                    .unwrap_or(self.celestial_state.sun_distance_earth_radii),
                self.celestial_moon_distance_earth_radii
                    .unwrap_or(self.celestial_state.moon_distance_earth_radii),
                self.celestial_state.moon_illuminated_fraction,
                self.celestial_state.moon_phase_angle_radians,
                if self.celestial_sun.is_some() {
                    "manual"
                } else {
                    "live"
                },
                if self.celestial_moon.is_some() {
                    "manual"
                } else {
                    "live"
                },
            )
        }
    }

    fn celestial_detail_fields(&self) -> String {
        let (mut sun, moon, mut sun_distance, moon_distance, illumination, phase, mut sun_state, mut moon_state) =
            self.displayed_celestial_fields();
        let body = self.renderer.body();
        if body != crate::body::Body::Earth {
            sun = self.celestial_sun.unwrap_or(self.celestial_state.body_sun_directions[body.id() as usize]);
            sun_distance = self.celestial_sun_distance_earth_radii
                .unwrap_or(self.celestial_state.body_sun_distances_earth_radii[body.id() as usize]);
            sun_state = if self.celestial_sun.is_some() { "manual" } else { "live" };
            moon_state = "hidden";
        }
        let sun_apparent_diameter_arcminutes =
            2.0 * angular_radius_degrees(
                SUN_RADIUS_KM / EARTH_EQUATORIAL_RADIUS_KM,
                sun_distance as f64,
            ) * 60.0;
        let moon_apparent_diameter_arcminutes =
            2.0 * angular_radius_degrees(
                MOON_RADIUS_KM / EARTH_EQUATORIAL_RADIUS_KM,
                moon_distance as f64,
            ) * 60.0;
        format!(
            "time_mode={} active_unix_utc={} astronomy={} sun={} sun_yaw={:.3} sun_pitch={:.3} sun_distance_earth_radii={sun_distance:.3} sun_apparent_diameter_arcmin={sun_apparent_diameter_arcminutes:.3} moon={} moon_yaw={:.3} moon_pitch={:.3} moon_distance_earth_radii={moon_distance:.3} moon_apparent_diameter_arcmin={moon_apparent_diameter_arcminutes:.3} moon_illumination={illumination:.6} moon_phase_radians={phase:.6}",
            if self.fixed_unix_seconds.is_some() { "fixed" } else { "live" },
            self.active_unix_seconds,
            self.astronomy_status(),
            sun_state,
            direction_yaw(sun),
            direction_pitch(sun),
            moon_state,
            direction_yaw(moon),
            direction_pitch(moon),
        )
    }

    fn astronomy_status(&self) -> &str {
        self.last_astronomy_error.as_deref().unwrap_or("ready")
    }

    fn time_status(&self) -> String {
        format!("ok time {}", self.celestial_detail_fields())
    }

    fn celestial_status(&self) -> String {
        format!(
            "ok celestial {} manual_sun={} manual_moon={} debug_scene={}",
            self.celestial_detail_fields(),
            self.celestial_sun.is_some(),
            self.celestial_moon.is_some(),
            if self.debug_mode {
                self.debug_scene.name
            } else {
                "none"
            },
        )
    }

    fn status_line(&self) -> String {
        let control = self
            .controlled_output
            .and_then(|id| self.outputs.get(&id))
            .and_then(|output| output.name.as_deref())
            .unwrap_or("none");
        let iss = if self.iss.last_error().is_some() {
            "degraded"
        } else if self.iss.tle_is_stale(self.active_unix_seconds) {
            "stale"
        } else {
            "ready"
        };
        let iss_tle_epoch = self
            .iss
            .tle_epoch_unix_seconds()
            .map_or_else(|| "none".to_owned(), |value| value.to_string());
        let iss_tle_age_hours = self
            .iss
            .tle_age_seconds(self.active_unix_seconds)
            .map_or(f64::NAN, |seconds| seconds as f64 / 3600.0);
        let view = if self.fixed_camera {
            "fixed".to_owned()
        } else {
            match self.view_mode {
                ViewMode::Onboard => match self.iss_state {
                    Some(state) => format!("iss iss_altitude_km={:.1} iss_ground_speed_km_s={:.2} heading={:.1} fov={:.0}",
                        state.altitude_km, state.ground_speed_km_s, self.pov_look.heading_degrees, self.pov_look.fov_degrees),
                    None => "iss".to_owned(),
                },
                ViewMode::Globe => "globe".to_owned(),
                ViewMode::FixedPov { latitude, longitude, altitude_km } =>
                    format!("pov pov_lat={latitude:.3} pov_lon={longitude:.3} pov_altitude_km={altitude_km:.1}"),
            }
        };
        format!(
            "ok running outputs={} configured={} control={} view={} exposure_ev={:.2} iss={} iss_tle_epoch={} iss_tle_age_hours={:.1} fps={} measured_fps={} frame_ms={:.2} {} {}",
            self.outputs.len(),
            self.x11.as_ref().map_or(self.layers.len(), x11::X11Parts::window_count),
            control,
            view,
            self.renderer.exposure_ev(),
            iss,
            iss_tle_epoch,
            iss_tle_age_hours,
            self.current_frame_rate(),
            self.frame_cadence.measured_per_second,
            self.last_render_ms,
            self.renderer.status_fields(),
            self.celestial_detail_fields(),
        )
    }

    fn is_interactive(&self, now: Instant) -> bool {
        self.pointer.dragging
            || self.keyboard_yaw != 0.0
            || self.keyboard_pitch != 0.0
            || self.fly_keys.iter().any(|&held| held)
            || now < self.interactive_until
    }

    fn mark_interactive(&mut self) {
        self.interactive_until = Instant::now() + INTERACTIVE_GRACE;
        self.idle_interval = IDLE_MIN_INTERVAL;
        self.dirty = true;
    }

    fn toggle_control(&mut self, monitor: &str) -> Result<bool, &'static str> {
        let output_id = self
            .outputs
            .iter()
            .find_map(|(id, output)| (output.name.as_deref() == Some(monitor)).then_some(*id))
            .ok_or("selected monitor is not currently available")?;
        if self.controlled_output == Some(output_id) {
            self.release_control();
            return Ok(false);
        }

        self.release_control();
        self.apply_input_region(output_id, true);
        self.controlled_output = Some(output_id);
        self.mark_interactive();
        Ok(true)
    }

    fn release_control(&mut self) {
        if let Some(output_id) = self.controlled_output.take() {
            self.apply_input_region(output_id, false);
        }
        self.pointer = PointerState::default();
        self.keyboard_yaw = 0.0;
        self.keyboard_pitch = 0.0;
        self.camera.set_keyboard_axes(0.0, 0.0);
    }

    fn apply_input_region(&mut self, output_id: u32, enabled: bool) {
        if let Some(x11) = self.x11.as_ref() {
            x11.set_input(output_id, enabled);
            return;
        }
        let (Some(compositor), Some(queue_handle)) = (self.compositor.as_ref(), self.queue_handle.as_ref()) else {
            return;
        };
        let Some(layer) = self.layers.get(&output_id) else {
            return;
        };
        let region = compositor.create_region(queue_handle, ());
        if enabled {
            let width = layer.configured_width.min(i32::MAX as u32) as i32;
            let height = layer.configured_height.min(i32::MAX as u32) as i32;
            if width > 0 && height > 0 {
                region.add(0, 0, width, height);
            }
            layer
                .layer_surface
                .set_keyboard_interactivity(zwlr_layer_surface_v1::KeyboardInteractivity::OnDemand);
        } else {
            layer
                .layer_surface
                .set_keyboard_interactivity(zwlr_layer_surface_v1::KeyboardInteractivity::None);
        }
        layer.surface.set_input_region(Some(&region));
        region.destroy();
        layer.surface.commit();
    }

    fn update_desktop_layout(&mut self) {
        let output_rectangles: Vec<_> = self
            .outputs
            .values()
            .filter_map(OutputState::logical_rect)
            .collect();
        let Some(first) = output_rectangles.first().copied() else {
            self.desktop = None;
            self.cached_projection = None;
            return;
        };
        let bounds = output_rectangles
            .iter()
            .copied()
            .skip(1)
            .fold(first, CameraRect::union);
        self.desktop = DesktopLayout::new(bounds).ok();
        for (output_id, output) in &self.outputs {
            if let Some(rect) = output.logical_rect() {
                self.renderer.set_viewport(
                    *output_id,
                    LogicalRect {
                        x: rect.origin.x,
                        y: rect.origin.y,
                        width: rect.size.x,
                        height: rect.size.y,
                    },
                );
            }
        }
        self.dirty = true;
        // Bounds changed here, so the memoized projection/focus is stale:
        // refresh it now while the layout is dirty so clean frames only read.
        self.refresh_projection_cache();
    }

    fn try_create_layer_surface(&mut self, output_id: u32, qh: &QueueHandle<Self>) {
        if self.layers.contains_key(&output_id) {
            return;
        }
        let (Some(compositor), Some(layer_shell), Some(output)) = (
            self.compositor.as_ref(),
            self.layer_shell.as_ref(),
            self.outputs
                .get(&output_id)
                .and_then(|state| state.output.as_ref()),
        ) else {
            return;
        };
        let surface = compositor.create_surface(qh, ());
        let layer_surface = layer_shell.get_layer_surface(
            &surface,
            Some(output),
            zwlr_layer_shell_v1::Layer::Background,
            "earth-native".to_owned(),
            qh,
            SurfaceToken { output_id },
        );
        layer_surface.set_size(0, 0);
        layer_surface.set_anchor(
            zwlr_layer_surface_v1::Anchor::Top
                | zwlr_layer_surface_v1::Anchor::Bottom
                | zwlr_layer_surface_v1::Anchor::Left
                | zwlr_layer_surface_v1::Anchor::Right,
        );
        layer_surface.set_exclusive_zone(-1);
        layer_surface
            .set_keyboard_interactivity(zwlr_layer_surface_v1::KeyboardInteractivity::None);
        let empty_region = compositor.create_region(qh, ());
        surface.set_input_region(Some(&empty_region));
        empty_region.destroy();
        surface.commit();
        self.layers.insert(
            output_id,
            LayerState {
                surface,
                layer_surface,
                configured_width: 0,
                configured_height: 0,
            },
        );
    }

    fn attach_xdg_output(&mut self, output_id: u32, qh: &QueueHandle<Self>) {
        let Some(manager) = self.xdg_output_manager.as_ref() else {
            return;
        };
        let Some(output) = self.outputs.get_mut(&output_id) else {
            return;
        };
        if output.xdg_output.is_some() {
            return;
        }
        let Some(wl_output) = output.output.as_ref() else {
            return;
        };
        output.xdg_output = Some(manager.get_xdg_output(
            wl_output,
            qh,
            OutputToken {
                global_name: output_id,
            },
        ));
    }

    fn create_pending_surfaces(&mut self, qh: &QueueHandle<Self>) {
        let output_ids: Vec<_> = self.outputs.keys().copied().collect();
        for output_id in output_ids {
            self.try_create_layer_surface(output_id, qh);
        }
    }

    fn attach_pending_xdg_outputs(&mut self, qh: &QueueHandle<Self>) {
        let output_ids: Vec<_> = self.outputs.keys().copied().collect();
        for output_id in output_ids {
            self.attach_xdg_output(output_id, qh);
        }
    }

    fn configure_layer_surface(&mut self, output_id: u32, serial: u32, width: u32, height: u32) {
        let Some(layer) = self.layers.get_mut(&output_id) else {
            return;
        };
        layer.layer_surface.ack_configure(serial);
        layer.configured_width = width;
        layer.configured_height = height;
        let scale = self
            .outputs
            .get(&output_id)
            .map(OutputState::scale_factor)
            .unwrap_or(1);
        layer.surface.set_buffer_scale(scale);
        // No wl_surface.set_buffer_transform: the layer-shell configure size is
        // already in the output's logical (post-rotation) space and the
        // compositor rotates the composited frame itself at scanout. Declaring
        // a buffer transform on top double-rotates the surface, so a portrait
        // output clips the buffer to a square intersection and the per-frame
        // mismatch work stalls the compositor (slow swapchain acquires).
        let viewport = self
            .outputs
            .get(&output_id)
            .and_then(OutputState::logical_rect)
            .map(|rect| LogicalRect {
                x: rect.origin.x,
                y: rect.origin.y,
                width: rect.size.x,
                height: rect.size.y,
            })
            .unwrap_or(LogicalRect {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
            });
        let extent = vk::Extent2D {
            width: width.saturating_mul(scale as u32),
            height: height.saturating_mul(scale as u32),
        };
        let output_name = self
            .outputs
            .get(&output_id)
            .and_then(|output| output.name.clone())
            .unwrap_or_else(|| format!("output-{output_id}"));
        let Some(display) = self.display.as_ref() else {
            return;
        };
        let source = crate::vulkan::SurfaceSource::Wayland {
            display: display.id().as_ptr().cast::<vk::wl_display>(),
            surface: layer.surface.id().as_ptr().cast::<vk::wl_surface>(),
        };
        if let Err(error) = unsafe {
            self.renderer.configure_output(output_id, source, extent, viewport, &output_name)
        } {
            eprintln!("earth-native: Vulkan output {output_id} configuration failed: {error}");
        }
        self.layout_dirty = true;
        if self.controlled_output == Some(output_id) {
            self.apply_input_region(output_id, true);
        }
    }

    fn pointer_output_for_surface(&self, surface: &WlSurface) -> Option<u32> {
        self.layers.iter().find_map(|(output_id, layer)| {
            (layer.surface.id() == surface.id()).then_some(*output_id)
        })
    }

    fn update_keyboard_axes(&mut self) {
        // On the globe WASD orbits too, faster than the arrows (Shift more).
        let (forward, right) = self.fly_axes();
        let (orbit_yaw, orbit_pitch) = if self.view_mode == ViewMode::Globe && !self.pov_active_view() {
            let speed = FLY_ORBIT_AXIS * if self.shift_held { FLY_BOOST } else { 1.0 };
            (right * speed, forward * speed)
        } else {
            (0.0, 0.0)
        };
        self.camera
            .set_keyboard_axes(self.keyboard_yaw + orbit_yaw, self.keyboard_pitch + orbit_pitch);
    }

    fn pov_active_view(&self) -> bool {
        matches!(self.view_mode, ViewMode::Onboard | ViewMode::FixedPov { .. })
    }

    /// (forward, right) from the held WASD keys, each -1, 0 or 1.
    fn fly_axes(&self) -> (f32, f32) {
        let [w, a, s, d] = self.fly_keys.map(f32::from);
        (w - s, d - a)
    }

    /// WASD flight from the window: while a key is held the viewpoint moves
    /// over the Earth along a great circle, W where the camera looks, A/D
    /// sideways, and the view direction is carried along (parallel
    /// transport) so it does not swing. Leaving the ISS, the flight starts
    /// at its current place and heading; C (or `camera next`) returns.
    fn fly(&mut self, seconds: f32) {
        let (forward, right) = self.fly_axes();
        if (forward == 0.0 && right == 0.0) || !self.pov_active() {
            return;
        }
        if self.view_mode == ViewMode::Onboard && !self.leave_iss_here() {
            return;
        }
        let ViewMode::FixedPov { latitude, longitude, altitude_km } = self.view_mode else { return };
        let boost = if self.shift_held { f64::from(FLY_BOOST) } else { 1.0 };
        let speed_km_s = f64::from(altitude_km).max(100.0) * FLY_ALTITUDES_PER_SECOND * boost;
        let angle = speed_km_s * f64::from(seconds.min(0.1)) / (EARTH_EQUATORIAL_RADIUS_KM + f64::from(altitude_km));
        let (new_latitude, new_longitude, new_heading) =
            fly_step(latitude, longitude, self.pov_look.heading_degrees, forward, right, angle as f32);
        self.view_mode = ViewMode::FixedPov { latitude: new_latitude, longitude: new_longitude, altitude_km };
        self.pov_look.heading_degrees = new_heading;
        self.dirty = true;
    }

    /// Replace the ISS ride by a fixed viewpoint at its current place,
    /// altitude and absolute heading (the view does not jump).
    fn leave_iss_here(&mut self) -> bool {
        let Some(state) = self.iss_state else { return false };
        let up = state.position.normalized();
        let latitude = up.z.clamp(-1.0, 1.0).asin().to_degrees().clamp(-89.9, 89.9);
        let longitude = (-up.y).atan2(up.x).to_degrees();
        let (_, north) = surface_frame(latitude, longitude);
        let east = up.cross(north).normalized();
        let level = (state.velocity - up * state.velocity.dot(up)).normalized();
        let heading = self.pov_look.heading_degrees.to_radians();
        let facing = level * heading.cos() + up.cross(level).normalized() * heading.sin();
        self.view_mode = ViewMode::FixedPov { latitude, longitude, altitude_km: state.altitude_km as f32 };
        self.pov_look.heading_degrees = facing.dot(east).atan2(facing.dot(north)).to_degrees().rem_euclid(360.0);
        true
    }
}

fn orbit_delta_from_surface_motion(delta_x: f64, delta_y: f64) -> (f32, f32) {
    // wl_pointer positions use a top-left origin, while Unreal's MouseY axis
    // increases when the pointer moves up. Keep the camera API in Unreal's
    // convention and translate at the Wayland boundary.
    (delta_x as f32, -(delta_y as f32))
}

impl Dispatch<WlRegistry, ()> for NativeApp {
    fn event(
        state: &mut Self,
        registry: &WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_registry::Event::Global {
                name,
                interface,
                version,
            } => match interface.as_str() {
                "wl_compositor" => {
                    state.compositor = Some(registry.bind(name, version.min(4), qh, ()));
                    state.create_pending_surfaces(qh);
                }
                "zwlr_layer_shell_v1" => {
                    state.layer_shell = Some(registry.bind(name, version.min(5), qh, ()));
                    state.create_pending_surfaces(qh);
                }
                "zxdg_output_manager_v1" => {
                    state.xdg_output_manager = Some(registry.bind(name, version.min(3), qh, ()));
                    state.attach_pending_xdg_outputs(qh);
                }
                "wl_output" => {
                    let output = registry.bind::<WlOutput, _, _>(
                        name,
                        version.min(4),
                        qh,
                        OutputToken { global_name: name },
                    );
                    state.outputs.insert(
                        name,
                        OutputState {
                            output: Some(output),
                            scale: 1,
                            ..OutputState::default()
                        },
                    );
                    state.attach_xdg_output(name, qh);
                    state.try_create_layer_surface(name, qh);
                }
                "wl_seat" => {
                    registry.bind::<WlSeat, _, _>(name, version.min(5), qh, ());
                }
                _ => {}
            },
            wl_registry::Event::GlobalRemove { name } => {
                state.renderer.destroy_output(name);
                if let Some(layer) = state.layers.remove(&name) {
                    layer.layer_surface.destroy();
                    layer.surface.destroy();
                }
                state.outputs.remove(&name);
                if state.controlled_output == Some(name) {
                    state.release_control();
                }
                state.layout_dirty = true;
            }
            _ => {}
        }
    }
}

impl Dispatch<WlOutput, OutputToken> for NativeApp {
    fn event(
        state: &mut Self,
        _: &WlOutput,
        event: wl_output::Event,
        token: &OutputToken,
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        let Some(output) = state.outputs.get_mut(&token.global_name) else {
            return;
        };
        match event {
            wl_output::Event::Geometry { transform, .. } => {
                if let WEnum::Value(transform) = transform {
                    output.transform = OutputTransform::from_wl_output(transform as u32)
                        .unwrap_or(OutputTransform::Normal);
                }
            }
            wl_output::Event::Mode {
                flags,
                width,
                height,
                ..
            } => {
                if matches!(flags, WEnum::Value(value) if value.contains(wl_output::Mode::Current))
                {
                    output.mode = PixelSize::new(width.max(0) as u32, height.max(0) as u32);
                }
            }
            wl_output::Event::Scale { factor } => output.scale = factor.max(1),
            wl_output::Event::Name { name } => output.name = Some(name),
            wl_output::Event::Done => {
                state.layout_dirty = true;
                state.try_create_layer_surface(token.global_name, qh);
            }
            _ => {}
        }
    }
}

impl Dispatch<ZxdgOutputManagerV1, ()> for NativeApp {
    fn event(
        _: &mut Self,
        _: &ZxdgOutputManagerV1,
        _: zxdg_output_manager_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZxdgOutputV1, OutputToken> for NativeApp {
    fn event(
        state: &mut Self,
        _: &ZxdgOutputV1,
        event: zxdg_output_v1::Event,
        token: &OutputToken,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        let Some(output) = state.outputs.get_mut(&token.global_name) else {
            return;
        };
        match event {
            zxdg_output_v1::Event::LogicalPosition { x, y } => {
                output.logical_position = Some(Vec2::new(x as f32, y as f32));
            }
            zxdg_output_v1::Event::LogicalSize { width, height } => {
                if width > 0 && height > 0 {
                    output.logical_size = Some(Vec2::new(width as f32, height as f32));
                }
            }
            zxdg_output_v1::Event::Name { name } => output.name = Some(name),
            zxdg_output_v1::Event::Done => state.layout_dirty = true,
            _ => {}
        }
    }
}

impl Dispatch<ZwlrLayerShellV1, ()> for NativeApp {
    fn event(
        _: &mut Self,
        _: &ZwlrLayerShellV1,
        _: zwlr_layer_shell_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwlrLayerSurfaceV1, SurfaceToken> for NativeApp {
    fn event(
        state: &mut Self,
        _: &ZwlrLayerSurfaceV1,
        event: zwlr_layer_surface_v1::Event,
        token: &SurfaceToken,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_layer_surface_v1::Event::Configure {
                serial,
                width,
                height,
            } => state.configure_layer_surface(token.output_id, serial, width, height),
            zwlr_layer_surface_v1::Event::Closed => {
                state.renderer.destroy_output(token.output_id);
                if let Some(layer) = state.layers.remove(&token.output_id) {
                    layer.layer_surface.destroy();
                    layer.surface.destroy();
                }
                state.dirty = true;
            }
            _ => {}
        }
    }
}

impl Dispatch<WlSeat, ()> for NativeApp {
    fn event(
        _: &mut Self,
        seat: &WlSeat,
        event: wl_seat::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(capabilities),
        } = event
        {
            if capabilities.contains(wl_seat::Capability::Pointer) {
                seat.get_pointer(qh, ());
            }
            if capabilities.contains(wl_seat::Capability::Keyboard) {
                seat.get_keyboard(qh, ());
            }
        }
    }
}

impl Dispatch<WlPointer, ()> for NativeApp {
    fn event(
        state: &mut Self,
        _: &WlPointer,
        event: wl_pointer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_pointer::Event::Enter {
                surface,
                surface_x,
                surface_y,
                ..
            } => {
                state.pointer.output_id = state.pointer_output_for_surface(&surface);
                state.pointer.last_position = Some((surface_x, surface_y));
            }
            wl_pointer::Event::Leave { .. } => state.pointer = PointerState::default(),
            wl_pointer::Event::Motion {
                surface_x,
                surface_y,
                ..
            } => {
                let previous = state.pointer.last_position.replace((surface_x, surface_y));
                if state.pointer.dragging && state.pointer.output_id == state.controlled_output {
                    if let Some((previous_x, previous_y)) = previous {
                        let (orbit_x, orbit_y) = orbit_delta_from_surface_motion(
                            surface_x - previous_x,
                            surface_y - previous_y,
                        );
                        if state.control_drag(orbit_x, orbit_y) {
                            state.mark_interactive();
                        }
                    }
                }
            }
            wl_pointer::Event::Button {
                button,
                state: button_state,
                ..
            } if button == 272 => {
                if let WEnum::Value(button_state) = button_state {
                    state.pointer.dragging = button_state == wl_pointer::ButtonState::Pressed
                        && state.pointer.output_id == state.controlled_output;
                    if state.pointer.dragging {
                        state.mark_interactive();
                    }
                }
            }
            wl_pointer::Event::AxisDiscrete { axis, discrete } => {
                if state.pointer.output_id == state.controlled_output
                    && matches!(axis, WEnum::Value(wl_pointer::Axis::VerticalScroll))
                    && state.control_zoom(-(discrete as f32))
                {
                    state.mark_interactive();
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<WlKeyboard, ()> for NativeApp {
    fn event(
        state: &mut Self,
        _: &WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_keyboard::Event::Modifiers { mods_depressed, .. } = event {
            // Wayland modifier bit 2 is Control.
            state.ctrl_held = mods_depressed & (1 << 2) != 0;
            return;
        }
        let wl_keyboard::Event::Key {
            key,
            state: key_state,
            ..
        } = event
        else {
            return;
        };
        let WEnum::Value(key_state) = key_state else {
            return;
        };
        state.handle_evdev_key(key, key_state == wl_keyboard::KeyState::Pressed, state.ctrl_held);
    }
}

impl NativeApp {
    /// Switch between the ISS window and the globe (from any fixed or
    /// planetary view, back to the ISS window over Earth).
    fn cycle_view(&mut self) -> &'static str {
        let to_globe = !self.fixed_camera && self.view_mode == ViewMode::Onboard
            && self.renderer.body() == crate::body::Body::Earth;
        if self.renderer.body() != crate::body::Body::Earth {
            self.renderer.set_body(crate::body::Body::Earth);
        }
        self.fixed_camera = false;
        self.view_mode = if to_globe { ViewMode::Globe } else { ViewMode::Onboard };
        self.renderer.snap_exposure();
        self.mark_interactive();
        if to_globe { "globe" } else { "iss" }
    }

    /// Hover at ISS altitude ~6 degrees (670 km) from the strongest aurora
    /// in the dark and face it, from the side whose horizon is darkest: its
    /// 110-250 km curtains then stand just above the horizon against black
    /// sky, as in ISS window photographs.
    fn aim_at_aurora(&mut self) -> String {
        if self.debug_mode {
            return "error camera setter requires normal mode".into();
        }
        let Some(weather) = self.renderer.weather_state() else {
            return "error no NOAA aurora data yet (is earth-native-weather running?)".into();
        };
        let (path, width, height) = (weather.texture.clone(), weather.width as usize, weather.height as usize);
        let fields = match std::fs::read(&path) {
            Ok(fields) => fields,
            Err(error) => return format!("error reading {}: {error}", path.display()),
        };
        let Some((latitude, longitude, probability)) =
            crate::weather::strongest_dark_aurora(&fields, width, height, self.celestial_state.sun_direction)
        else {
            return "error no dark auroral zone right now".into();
        };
        let (camera_latitude, camera_longitude, heading) =
            crate::weather::aurora_vantage(latitude, longitude, 6.0, self.celestial_state.sun_direction);
        self.fixed_camera = false;
        if self.renderer.body() != crate::body::Body::Earth {
            self.renderer.set_body(crate::body::Body::Earth);
        }
        self.view_mode = ViewMode::FixedPov { latitude: camera_latitude, longitude: camera_longitude, altitude_km: 420.0 };
        self.pov_look = PovLook { heading_degrees: heading, pitch_degrees: Some(15.0), fov_degrees: PovLook::default().fov_degrees };
        self.renderer.snap_exposure();
        self.mark_interactive();
        format!("ok camera=aurora aurora_lat={latitude:.1} aurora_lon={longitude:.1} probability={:.0}% camera_lat={camera_latitude:.1} camera_lon={camera_longitude:.1} heading={heading:.0}{}",
            probability * 100.0,
            if probability < 0.05 { " (quiet oval: faint)" } else { "" })
    }

    /// Look ahead again with the default lens (the view mode stays).
    fn reset_look(&mut self) {
        let heading = match self.view_mode {
            ViewMode::FixedPov { .. } => self.pov_look.heading_degrees,
            _ => PovLook::default().heading_degrees,
        };
        self.pov_look = PovLook { heading_degrees: heading, ..PovLook::default() };
        self.renderer.snap_exposure();
        self.mark_interactive();
    }

    /// Field-of-view scale for look-around rates (1 at the default lens).
    fn pov_scale(&self) -> f32 {
        self.pov_look.fov_degrees / PovLook::default().fov_degrees
    }

    /// Turn the window view: heading to the right, pitch further down.
    fn turn_pov(&mut self, heading_degrees: f32, pitch_down_degrees: f32) -> bool {
        if heading_degrees == 0.0 && pitch_down_degrees == 0.0 {
            return false;
        }
        let pitch = self.pov_look.pitch_degrees.unwrap_or(self.pov_auto_pitch);
        self.pov_look.heading_degrees = (self.pov_look.heading_degrees + heading_degrees).rem_euclid(360.0);
        self.pov_look.pitch_degrees = Some((pitch + pitch_down_degrees).clamp(-89.0, 89.0));
        self.dirty = true;
        true
    }

    /// Control-mode drag (camera convention: x right, y up). From the ISS or
    /// a fixed viewpoint it grabs the view and turns the camera; otherwise it
    /// orbits the globe.
    fn control_drag(&mut self, delta_x: f32, delta_y: f32) -> bool {
        if !self.pov_active() {
            return self.camera.orbit_mouse(delta_x, delta_y);
        }
        let degrees = POV_DRAG_DEGREES_PER_PIXEL * self.pov_scale();
        self.turn_pov(-delta_x * degrees, delta_y * degrees)
    }

    /// Control-mode zoom, positive steps in: the window lens's field of view
    /// (8-120 degrees) from the ISS or a fixed viewpoint, else the orbit.
    fn control_zoom(&mut self, steps: f32) -> bool {
        if !self.pov_active() {
            return self.camera.zoom_steps(steps);
        }
        let fov = (self.pov_look.fov_degrees * 0.84_f32.powf(steps)).clamp(POV_MIN_FOV_DEGREES, POV_MAX_FOV_DEGREES);
        if fov == self.pov_look.fov_degrees {
            return false;
        }
        self.pov_look.fov_degrees = fov;
        self.dirty = true;
        true
    }

    /// Keyboard control shared by the Wayland and Xorg frontends (Linux
    /// evdev key codes): arrows orbit (or look around from the ISS), W/A/S/D
    /// fly over the Earth while held (orbit faster on the globe), Shift
    /// boosts them, Q/E zoom, C switches ISS window/globe, R resets the
    /// look, Esc releases control, and Ctrl+Left/Right tours the bodies.
    fn handle_evdev_key(&mut self, key: u32, pressed: bool, ctrl_held: bool) {
        if self.controlled_output.is_none() {
            return;
        }
        if pressed && ctrl_held && (key == 105 || key == 106) {
            let bodies = crate::body::Body::TOUR;
            let current = self.renderer.body();
            let current_index = bodies.iter().position(|body| *body == current).unwrap_or(0);
            let next_index = if key == 106 {
                (current_index + 1) % bodies.len()
            } else {
                (current_index + bodies.len() - 1) % bodies.len()
            };
            self.renderer.set_body(bodies[next_index]);
            self.dirty = true;
            self.mark_interactive();
            return;
        }
        match key {
            1 if pressed => self.release_control(),
            // C: ISS window <-> globe; R: look ahead with the default lens.
            46 if pressed => {
                self.cycle_view();
                return;
            }
            19 if pressed => {
                self.reset_look();
                return;
            }
            // Shift boosts WASD; W A S D fly (or orbit the globe).
            42 | 54 => self.shift_held = pressed,
            17 => self.fly_keys[0] = pressed,
            30 => self.fly_keys[1] = pressed,
            31 => self.fly_keys[2] = pressed,
            32 => self.fly_keys[3] = pressed,
            105 => self.keyboard_yaw = if pressed { -1.0 } else { 0.0 },
            106 => self.keyboard_yaw = if pressed { 1.0 } else { 0.0 },
            103 => self.keyboard_pitch = if pressed { 1.0 } else { 0.0 },
            108 => self.keyboard_pitch = if pressed { -1.0 } else { 0.0 },
            16 if pressed => {
                if self.control_zoom(-1.0) {
                    self.mark_interactive();
                }
            }
            18 if pressed => {
                if self.control_zoom(1.0) {
                    self.mark_interactive();
                }
            }
            _ => return,
        }
        self.update_keyboard_axes();
        if pressed {
            self.mark_interactive();
        }
    }
}

delegate_noop!(NativeApp: ignore WlCompositor);
delegate_noop!(NativeApp: ignore WlSurface);
delegate_noop!(NativeApp: ignore WlRegion);

fn parse_look(heading: &str, pitch: &str, fov: &str) -> Result<PovLook, &'static str> {
    let heading = heading.parse::<f32>().map_err(|_| "heading must be degrees")?;
    let pitch = if pitch == "auto" {
        None
    } else {
        Some(pitch.parse::<f32>().ok().filter(|p| (-89.0..=89.0).contains(p)).ok_or("pitch must be auto or degrees in [-89,89]")?)
    };
    let fov = fov.parse::<f32>().ok()
        .filter(|f| (POV_MIN_FOV_DEGREES..=POV_MAX_FOV_DEGREES).contains(f))
        .ok_or("field of view must be 8-120 degrees")?;
    if !heading.is_finite() {
        return Err("heading must be finite");
    }
    Ok(PovLook { heading_degrees: heading, pitch_degrees: pitch, fov_degrees: fov })
}

fn celestial_direction(yaw: &str, pitch: &str) -> Result<[f32; 3], &'static str> {
    let yaw = yaw
        .parse::<f32>()
        .map_err(|_| "celestial yaw must be finite degrees")?;
    let pitch = pitch
        .parse::<f32>()
        .map_err(|_| "celestial pitch must be finite degrees")?;
    if !yaw.is_finite() || !pitch.is_finite() {
        return Err("celestial angles must be finite");
    }
    if !(-90.0..=90.0).contains(&pitch) {
        return Err("celestial pitch must be in [-90,90] degrees");
    }
    let yaw = yaw.to_radians();
    let pitch = pitch.to_radians();
    let direction = [
        pitch.cos() * yaw.cos(),
        pitch.cos() * yaw.sin(),
        pitch.sin(),
    ];
    direction
        .iter()
        .all(|value| value.is_finite())
        .then_some(direction)
        .ok_or("celestial direction is not finite")
}

fn direction_array(direction: Vec3) -> [f32; 3] {
    [direction.x, direction.y, direction.z]
}

fn direction_yaw(direction: [f32; 3]) -> f32 {
    direction[1]
        .atan2(direction[0])
        .to_degrees()
        .rem_euclid(360.0)
}

fn direction_pitch(direction: [f32; 3]) -> f32 {
    direction[2].clamp(-1.0, 1.0).asin().to_degrees()
}

fn active_timestamp(fixed_unix_seconds: Option<i64>, wall: (i64, i32)) -> (i64, i32) {
    fixed_unix_seconds.map_or(wall, |seconds| (seconds, 0))
}

fn debug_moon_phase(sun: [f32; 3], moon: [f32; 3]) -> (f32, f32) {
    let separation = sun
        .into_iter()
        .zip(moon)
        .map(|(left, right)| left * right)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
        .acos();
    let phase = std::f32::consts::PI - separation;
    ((1.0 + phase.cos()) * 0.5, phase)
}

fn angular_radius_degrees(radius_earth_radii: f64, distance_earth_radii: f64) -> f64 {
    (radius_earth_radii / distance_earth_radii.max(radius_earth_radii))
        .clamp(0.0, 0.999_999_999)
        .asin()
        .to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasd_flight_follows_great_circles() {
        let near = |a: f32, b: f32| (a - b).abs() < 0.05;
        // East along the equator: longitude grows, latitude and heading stay.
        let (lat, lon, heading) = fly_step(0.0, 10.0, 90.0, 1.0, 0.0, 5.0_f32.to_radians());
        assert!(near(lat, 0.0) && near(lon, 15.0) && near(heading, 90.0), "{lat} {lon} {heading}");
        // North: latitude grows by the arc.
        let (lat, lon, heading) = fly_step(10.0, 20.0, 0.0, 1.0, 0.0, 30.0_f32.to_radians());
        assert!(near(lat, 40.0) && near(lon, 20.0) && near(heading, 0.0), "{lat} {lon} {heading}");
        // Over the pole: down the far side, now facing south.
        let (lat, lon, heading) = fly_step(80.0, 0.0, 0.0, 1.0, 0.0, 20.0_f32.to_radians());
        assert!(near(lat, 80.0) && near(lon.abs(), 180.0) && near(heading, 180.0), "{lat} {lon} {heading}");
        // S goes back, D strafes right (east when facing north).
        let (lat, ..) = fly_step(10.0, 0.0, 0.0, -1.0, 0.0, 5.0_f32.to_radians());
        assert!(near(lat, 5.0), "{lat}");
        let (lat, lon, _) = fly_step(0.0, 0.0, 0.0, 0.0, 1.0, 5.0_f32.to_radians());
        assert!(near(lat, 0.0) && near(lon, 5.0), "{lat} {lon}");
    }

    #[test]
    fn frame_intervals_match_the_native_idle_interactive_policy() {
        assert_eq!(
            frame_interval(IDLE_FRAME_RATE),
            Duration::from_nanos(66_666_666)
        );
        assert_eq!(
            frame_interval(INTERACTIVE_FRAME_RATE),
            Duration::from_nanos(33_333_333)
        );
        assert_eq!(frame_interval(DEBUG_FRAME_RATE), Duration::from_millis(100));
    }

    #[test]
    fn frame_cadence_reports_rendered_frames_per_second() {
        let start = Instant::now();
        let mut cadence = FrameCadence::new(start);
        assert_eq!(cadence.measured_per_second, 0);
        // Ten frames inside the first second publish when the window closes.
        for i in 1..=10 {
            cadence.record_render(start + Duration::from_millis(i * 90));
        }
        cadence.record_render(start + Duration::from_millis(1000));
        assert_eq!(cadence.measured_per_second, 10);
    }

    #[test]
    fn frame_cadence_detects_a_stalled_renderer() {
        let start = Instant::now();
        let mut cadence = FrameCadence::new(start);
        for i in 1..=10 {
            cadence.record_render(start + Duration::from_millis(i * 90));
        }
        cadence.record_render(start + Duration::from_millis(1000));
        // A stall follows: nothing renders for a full second, then one frame.
        cadence.record_render(start + Duration::from_millis(2000));
        assert_eq!(cadence.measured_per_second, 1);
    }

    #[test]
    fn celestial_angles_use_source_facing_world_directions() {
        let direction = celestial_direction("90", "30").unwrap();
        assert!(direction[0].abs() < 0.0001);
        assert!((direction[1] - 0.8660254).abs() < 0.0001);
        assert!((direction[2] - 0.5).abs() < 0.0001);
        assert!((direction_yaw(direction) - 90.0).abs() < 0.001);
        assert!((direction_pitch(direction) - 30.0).abs() < 0.001);
        assert!(celestial_direction("0", "91").is_err());
        assert!(celestial_direction("nan", "0").is_err());
    }

    #[test]
    fn fixed_time_uses_zero_microseconds_and_live_time_uses_the_wall_sample() {
        assert_eq!(active_timestamp(None, (123, 456_789)), (123, 456_789));
        assert_eq!(active_timestamp(Some(-42), (123, 456_789)), (-42, 0));
    }

    #[test]
    fn debug_phase_is_derived_from_sun_moon_separation() {
        let (new_fraction, new_phase) = debug_moon_phase([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(new_fraction.abs() < 1.0e-6);
        assert!((new_phase - std::f32::consts::PI).abs() < 1.0e-6);

        let (full_fraction, full_phase) = debug_moon_phase([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!((full_fraction - 1.0).abs() < 1.0e-6);
        assert!(full_phase.abs() < 1.0e-6);
    }

    #[test]
    fn physical_body_sizes_match_half_degree_sky_discs() {
        let sun_diameter_arcminutes =
            2.0 * angular_radius_degrees(
                SUN_RADIUS_KM / EARTH_EQUATORIAL_RADIUS_KM,
                MEAN_SUN_DISTANCE_EARTH_RADII,
            ) * 60.0;
        let moon_diameter_arcminutes =
            2.0 * angular_radius_degrees(
                MOON_RADIUS_KM / EARTH_EQUATORIAL_RADIUS_KM,
                MEAN_MOON_DISTANCE_EARTH_RADII,
            ) * 60.0;
        assert!((31.0..=33.0).contains(&sun_diameter_arcminutes));
        assert!((30.0..=33.0).contains(&moon_diameter_arcminutes));
    }

    #[test]
    fn wayland_pointer_y_is_translated_to_unreal_mouse_y() {
        assert_eq!(orbit_delta_from_surface_motion(12.5, -8.0), (12.5, 8.0));
        assert_eq!(orbit_delta_from_surface_motion(-3.0, 6.0), (-3.0, -6.0));
    }
}

fn continuous_rendering() -> bool {
    static CONTINUOUS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CONTINUOUS.get_or_init(|| std::env::var_os("EARTH_NATIVE_CONTINUOUS").is_some_and(|value| value == "1"))
}

/// One flight step: from (latitude, longitude) facing `heading` (degrees
/// from north), move `angle` radians of arc along the great circle toward
/// forward/right (-1..1); returns the new place and the parallel-transported
/// heading.
fn fly_step(latitude: f32, longitude: f32, heading: f32, forward: f32, right: f32, angle: f32) -> (f32, f32, f32) {
    let (up, north) = surface_frame(latitude, longitude);
    let east = up.cross(north).normalized();
    let heading = heading.to_radians();
    let facing = (north * heading.cos() + east * heading.sin()).normalized();
    let side = up.cross(facing).normalized();
    let travel = (facing * forward + side * right).normalized();
    let (sin, cos) = (angle.sin(), angle.cos());
    let new_up = (up * cos + travel * sin).normalized();
    // Parallel transport: the component along the travel turns with the
    // great circle, the rest is unchanged.
    let along = facing.dot(travel);
    let new_facing = (facing - travel * along + (travel * cos - up * sin) * along).normalized();
    let new_latitude = new_up.z.clamp(-1.0, 1.0).asin().to_degrees().clamp(-89.9, 89.9);
    let new_longitude = (-new_up.y).atan2(new_up.x).to_degrees();
    let (_, new_north) = surface_frame(new_latitude, new_longitude);
    let new_east = new_up.cross(new_north).normalized();
    let new_heading = new_facing.dot(new_east).atan2(new_facing.dot(new_north)).to_degrees().rem_euclid(360.0);
    (new_latitude, new_longitude, new_heading)
}

/// Local vertical and north at a latitude/longitude, in the scene's
/// longitude-mirrored frame (as `onboard_pose` builds a fixed viewpoint).
fn surface_frame(latitude: f32, longitude: f32) -> (Vec3, Vec3) {
    let (lat, lon) = (f64::from(latitude).to_radians(), f64::from(longitude).to_radians());
    let direction = crate::orbit::scene_direction(lat, lon);
    let up = Vec3::new(direction[0] as f32, direction[1] as f32, direction[2] as f32);
    let north = Vec3::new((-lat.sin() * lon.cos()) as f32, (lat.sin() * lon.sin()) as f32, lat.cos() as f32);
    (up, north)
}
