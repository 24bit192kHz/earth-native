use std::{
    collections::HashMap,
    env,
    ffi::CString,
    io::Cursor,
    mem::size_of,
    ptr,
    time::{Duration, Instant},
};

use ash::{util::read_spv, vk, Device, Entry, Instance};

use crate::{
    asset_bundle::{RenderQuality, ValidatedBundle},
    camera::SCENE_EARTH_RADIUS,
    day_color::{
        load_optional_cloud_previews, load_optional_day_color_hemispheres,
        load_optional_desert_cloud_mask, load_optional_height_preview, load_optional_night_emission,
        load_optional_moon_preview, load_optional_planet_previews, load_optional_saturn_ring_preview,
        load_optional_surface_normal_hemispheres, load_optional_tiling_noise, CloudPreviews,
        DayColorHemispheres, DesertCloudMaskPreview, HeightPreview, MoonPreview,
        BlockFormat, NightEmissionPreview, PlanetPreview, PlanetPreviews, PreviewTexture, RingPreview, SurfaceNormalHemispheres, TilingNoisePreview,
    },
    debug_capture::{self, PixelOrder},
    earthvt::{
        LayerDescriptor, PixelFormat, TextureChannel, TileKey, TileRequest, PADDED_TILE_SIZE,
        LAYER_FLAG_CLAMP_Y, LAYER_FLAG_SRGB, LAYER_FLAG_WRAP_X,
    },
    vt_feedback::{Feedback, OutputDescriptor},
    vt_streamer::{StreamKind, UploadJob, VirtualTextureStreamer},
    star_panorama::{
        load_optional_star_panorama, StarPanorama, StarPanoramaFormat, StarPanoramaMip,
    },
};

mod post;

use post::{HdrTarget, PostFrame, PostPipeline, HDR_FORMAT};

pub type RendererResult<T> = Result<T, Box<dyn std::error::Error>>;

// The reference actor's bounds are wider than its visible Earth shell. These
// normalized radii are calibrated against the 5.5x main capture on the active
// 4520x2560 desktop: the surface projects to about 1398 pixels and the
// surface remains calibrated while atmosphere height uses physical units.
const REFERENCE_SURFACE_RADIUS: f32 = SCENE_EARTH_RADIUS;
// Physical 100 km atmosphere in the same units as the visible Earth.
const REFERENCE_ATMOSPHERE_RADIUS: f32 = crate::atmosphere::normalized_top_radius(SCENE_EARTH_RADIUS);
const REFERENCE_GLOW_RADIUS: f32 = REFERENCE_ATMOSPHERE_RADIUS;
// NASA A-ring outer edge, conservatively bounded for any axial orientation.
const REFERENCE_RING_RADIUS: f32 = SCENE_EARTH_RADIUS * (136_775.0 / 60_268.0) + 0.002;
// Unreal's raw export preserves its native BGRA8 byte order. This sRGB format
// interprets those bytes directly and returns linear RGB from texture().
const STAR_PANORAMA_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const STAR_PANORAMA_BC1_FORMAT: vk::Format = vk::Format::BC1_RGB_SRGB_BLOCK;
const LINEAR_PREVIEW_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
// Offline block-compressed NASA previews (see NativeRenderer/bcn.py).
const PREVIEW_BC3_SRGB_FORMAT: vk::Format = vk::Format::BC3_SRGB_BLOCK;
const PREVIEW_BC3_LINEAR_FORMAT: vk::Format = vk::Format::BC3_UNORM_BLOCK;
const PREVIEW_BC4_FORMAT: vk::Format = vk::Format::BC4_UNORM_BLOCK;
const PREVIEW_BC5_FORMAT: vk::Format = vk::Format::BC5_UNORM_BLOCK;
/// Atmosphere lookup tables (sky.rs). Linear filtering, sampling and transfer
/// of this format are mandatory in Vulkan, so no feature query is needed.
const SKY_LUT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

fn is_block_compressed(format: vk::Format) -> bool {
    matches!(
        format,
        STAR_PANORAMA_BC1_FORMAT | PREVIEW_BC3_SRGB_FORMAT | PREVIEW_BC3_LINEAR_FORMAT | PREVIEW_BC4_FORMAT
    )
}
const VT_ATLAS_FORMAT: vk::Format = vk::Format::BC7_SRGB_BLOCK;
const VT_PAGE_TABLE_FORMAT: vk::Format = vk::Format::R32_UINT;
const VT_ENV: &str = "EARTH_NATIVE_EARTHVT";
const VT_BUDGET_ENV: &str = "EARTH_NATIVE_EARTHVT_BUDGET_MB";
/// Atlas budgets. A view needs ~70-100 day tiles and ~20-60 static tiles
/// (about one texel per pixel over 3440x1440 + 2560x1080); the budgets hold
/// several times that as cache, and batched streaming refills the rest in a
/// few frames. 256 MB (capped at 2048 slots, 143 MB) and 96 MB were reserved.
const VT_DEFAULT_BUDGET_MB: u32 = 48;
const STATIC_VT_DEFAULT_BUDGET_MB: u64 = 40;
const STATIC_VT_ENV: &str = "EARTH_NATIVE_STATIC_VT";
const STATIC_VT_BUDGET_ENV: &str = "EARTH_NATIVE_STATIC_VT_BUDGET_MB";
const VT_ATLAS_SIZE: u32 = PADDED_TILE_SIZE;
const VT_DEFAULT_SLOT_BUDGET: u32 = 64;
const DEBUG_CAPTURE_INTERVAL: Duration = Duration::from_secs(1);
// GPU frame timing: three timestamp queries per output (frame top, after the
// star draw, after the Earth draw), addressed by output id in a shared pool.
const TIMESTAMP_QUERIES_PER_OUTPUT: u32 = 4;
const TIMESTAMP_OUTPUT_SLOTS: u32 = 32;
// GPU timing is a diagnostics side channel for the status line, sampled once
// every N presented frames per output instead of every frame.
const TIMESTAMP_SAMPLE_EVERY: u32 = 30;
const _: () = assert!(REFERENCE_SURFACE_RADIUS > 0.0);
const _: () = assert!(REFERENCE_SURFACE_RADIUS < REFERENCE_ATMOSPHERE_RADIUS);

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct LogicalRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl LogicalRect {
    pub fn is_valid(self) -> bool {
        self.width > 0.0 && self.height > 0.0
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct FrameUniforms {
    pub unix_seconds: i64,
    pub nasa_materials: bool,
    /// NASA Blue Marble cloud map in place of the authored procedural layers.
    pub nasa_clouds: bool,
    pub weather_valid_unix_utc: i64,
    pub aurora_valid_unix_utc: i64,
    /// Observed (GMGSI) clouds are bound at 12 and current.
    pub live_clouds: bool,
    /// Binding 12 also carries the live GEFS-Aerosols optical depth (B, A).
    pub live_aerosol: bool,
    /// Binding 12 also carries OSI SAF sea-ice concentration (G).
    pub live_sea_ice: bool,
    pub camera_position: [f32; 3],
    pub camera_distance: f32,
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub canvas: LogicalRect,
    pub sun_direction: [f32; 3],
    pub moon_direction: [f32; 3],
    pub sun_distance_earth_radii: f32,
    pub moon_distance_earth_radii: f32,
    pub sun_radius_earth_radii: f32,
    pub moon_radius_earth_radii: f32,
    pub moon_world_to_body: [[f32; 3]; 3],
    pub greenwich_sidereal_radians: f32,
    pub moon_illuminated_fraction: f32,
    pub moon_phase_angle_radians: f32,
    // Seconds since midnight UTC (mod 86400), driving aurora curtain drift and
    // lightning flash timing. f32 precision at 86400 is ~8 ms, plenty.
    pub time_of_day_seconds: f32,
    pub tan_half_fov_x: f32,
    pub tan_half_fov_y: f32,
    pub focus_x: f32,
    pub focus_y: f32,
    /// Row-major J2000 -> world rotation (precession, nutation, sidereal).
    pub star_eqj_to_world: [[f32; 3]; 3],
    /// Earth's barycentric velocity / c in J2000, for annual aberration.
    pub star_aberration: [f32; 3],
}

impl Default for FrameUniforms {
    fn default() -> Self {
        Self {
            unix_seconds: 946728000,
            nasa_materials: false,
            nasa_clouds: false,
            weather_valid_unix_utc: 0,
            aurora_valid_unix_utc: 0,
            live_clouds: false,
            live_aerosol: false,
            live_sea_ice: false,
            camera_position: [-5.5, 0.0, 0.0],
            camera_distance: 5.5,
            forward: [1.0, 0.0, 0.0],
            right: [0.0, 1.0, 0.0],
            up: [0.0, 0.0, 1.0],
            canvas: LogicalRect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
            sun_direction: [1.0, 0.0, 0.0],
            moon_direction: [-1.0, 0.0, 0.0],
            sun_distance_earth_radii: 23_455.0,
            moon_distance_earth_radii: 60.27,
            sun_radius_earth_radii: 109.076,
            moon_radius_earth_radii: 0.272507,
            moon_world_to_body: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            greenwich_sidereal_radians: 0.0,
            moon_illuminated_fraction: 0.5,
            moon_phase_angle_radians: std::f32::consts::FRAC_PI_2,
            time_of_day_seconds: 0.0,
            tan_half_fov_x: 0.262_3,
            tan_half_fov_y: 0.262_3,
            focus_x: 0.5,
            focus_y: 0.5,
            star_eqj_to_world: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            star_aberration: [0.0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ShaderFrame {
    camera_position_distance: [f32; 4],
    camera_forward: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    viewport_rect: [f32; 4],
    canvas_rect: [f32; 4],
    sun_direction: [f32; 4],
    projection_tangents: [f32; 4],
    celestial_distances: [f32; 4],
    moon_body_x: [f32; 4],
    moon_body_y: [f32; 4],
    moon_body_z: [f32; 4],
    celestial_state: [f32; 4],
    // Precomputed per-frame celestial geometry, in scene space. The star pass
    // previously recomputed these camera-relative directions, apparent radii,
    // and the sidereal sin/cos pair per fragment; the padding slots of
    // moon_body_x/y/z carry (sidereal sine, sidereal cosine, sun-gate
    // threshold = cos(sun apparent radius * 40)). Moon-disc smoothstep edges
    // use the otherwise unused celestial_state.x and material_state.w lanes.
    celestial_sun_view: [f32; 4],
    celestial_moon_view: [f32; 4],
    material_state: [f32; 4],
}

/// Push constants of the per-star quads (`stars_points.vert`). The camera
/// basis is expressed in J2000 so the vertex shader only needs dot products.
#[repr(C)]
#[derive(Clone, Copy)]
struct StarFrame {
    view_right: [f32; 4],
    view_up: [f32; 4],
    view_forward: [f32; 4],
    projection_tangents: [f32; 4],
    canvas_rect: [f32; 4],
    viewport_rect: [f32; 4],
    params: [f32; 4],
}

impl StarFrame {
    fn from_uniforms(uniforms: FrameUniforms, viewport: LogicalRect, physical_width: u32) -> Self {
        // world = M * eqj, so a world-space basis vector b is M^T b in J2000.
        let m = uniforms.star_eqj_to_world;
        let to_eqj = |b: [f32; 3]| -> [f32; 3] {
            [0, 1, 2].map(|column| (0..3).map(|row| m[row][column] * b[row]).sum())
        };
        let (right, up, forward) = (to_eqj(uniforms.right), to_eqj(uniforms.up), to_eqj(uniforms.forward));
        let v = uniforms.star_aberration;
        let years = (uniforms.unix_seconds as f64 - crate::star_catalog::EPOCH_UNIX_SECONDS)
            / crate::star_catalog::SECONDS_PER_JULIAN_YEAR;
        Self {
            view_right: [right[0], right[1], right[2], v[0]],
            view_up: [up[0], up[1], up[2], v[1]],
            view_forward: [forward[0], forward[1], forward[2], v[2]],
            projection_tangents: [uniforms.tan_half_fov_x, uniforms.tan_half_fov_y, uniforms.focus_x, uniforms.focus_y],
            canvas_rect: [uniforms.canvas.x, uniforms.canvas.y, uniforms.canvas.width, uniforms.canvas.height],
            viewport_rect: [viewport.x, viewport.y, viewport.width, viewport.height],
            params: [years as f32, physical_width as f32 / viewport.width.max(1.0), 1.0, 0.0],
        }
    }
}

impl ShaderFrame {
    fn from_uniforms(uniforms: FrameUniforms, viewport: LogicalRect, body_selector: f32) -> Self {
        // Camera-relative celestial geometry, mirroring the star shader's old
        // per-fragment derivation: scene position = earth-fixed direction *
        // distance * SCENE_EARTH_RADIUS, then direction/apparent radius as
        // seen from the camera. Computed once per frame here instead of once
        // per fragment on both outputs.
        fn normalize3(vector: [f32; 3]) -> [f32; 3] {
            let length = vector
                .iter()
                .map(|component| component * component)
                .sum::<f32>()
                .sqrt()
                .max(1.0e-5);
            [vector[0] / length, vector[1] / length, vector[2] / length]
        }
        fn camera_view(
            earth_direction: [f32; 3],
            distance_earth_radii: f32,
            radius_earth_radii: f32,
            camera_position: [f32; 3],
            scene_units_per_earth_radius: f32,
        ) -> ([f32; 3], f32, f32) {
            let from_earth = normalize3(earth_direction);
            let distance = distance_earth_radii.max(1.0);
            let position = [
                from_earth[0] * distance * scene_units_per_earth_radius,
                from_earth[1] * distance * scene_units_per_earth_radius,
                from_earth[2] * distance * scene_units_per_earth_radius,
            ];
            let camera_to_body = [
                position[0] - camera_position[0],
                position[1] - camera_position[1],
                position[2] - camera_position[2],
            ];
            let camera_distance = camera_to_body
                .iter()
                .map(|component| component * component)
                .sum::<f32>()
                .sqrt()
                .max(1.0e-5);
            let direction = [
                camera_to_body[0] / camera_distance,
                camera_to_body[1] / camera_distance,
                camera_to_body[2] / camera_distance,
            ];
            let angular_radius = (radius_earth_radii * scene_units_per_earth_radius / camera_distance)
                .clamp(0.0, 0.999999)
                .asin();
            (direction, angular_radius, camera_distance)
        }
        // SM_SphereLow UV0 increases U westward, so the Earth day map is a
        // longitude mirror of ECEF. Lighting, the sky disc, and the Moon all
        // use this Y-mirrored vector so the terminator and the Sun stay on
        // the same side of the globe. Other bodies keep the un-mirrored
        // Earth-fixed direction from `body_sun_vector`.
        let (sun_direction, moon_direction) = if body_selector == 0.0 {
            (
                mirror_earth_texture_longitude(uniforms.sun_direction),
                mirror_earth_texture_longitude(uniforms.moon_direction),
            )
        } else {
            (uniforms.sun_direction, uniforms.moon_direction)
        };
        let body = crate::body::Body::ALL[body_selector as usize];
        let scene_units_per_earth_radius = SCENE_EARTH_RADIUS
            * (crate::astronomy::EARTH_EQUATORIAL_RADIUS_KM / body.equatorial_radius_km()) as f32;
        let mut world_to_body = uniforms.moon_world_to_body;
        if body == crate::body::Body::Earth {
            // The Moon's surface normal shares the mirrored scene Y axis.
            for row in &mut world_to_body { row[1] = -row[1]; }
        }
        let (sun_direction_from_camera, sun_angular_radius, _sun_camera_distance) = camera_view(
            sun_direction,
            uniforms.sun_distance_earth_radii,
            uniforms.sun_radius_earth_radii,
            uniforms.camera_position,
            scene_units_per_earth_radius,
        );
        let (moon_direction_from_camera, moon_angular_radius, _moon_camera_distance) =
            camera_view(
                moon_direction,
                uniforms.moon_distance_earth_radii,
                uniforms.moon_radius_earth_radii,
                uniforms.camera_position,
                scene_units_per_earth_radius,
            );
        // Panorama (J2000) orientation: world = M * eqj with M ~ Rz(-angle),
        // so this is sidereal time plus precession in right ascension (the
        // remaining <= 0.15 degree of precession in declination is invisible
        // in the diffuse Milky Way; catalogue stars use the full matrix).
        let m = uniforms.star_eqj_to_world;
        let panorama_angle = m[0][1].atan2(m[0][0]);
        let sidereal_sine = panorama_angle.sin();
        let sidereal_cosine = panorama_angle.cos();
        // Widest corona layer decays over 35 apparent radii; gate the whole
        // stack a little past that. Coherent across the frame, so the branch
        // costs nothing when the Sun is out of view.
        let sun_glow_threshold = (sun_angular_radius * 40.0).cos();
        let moon_disc_edge = moon_angular_radius * 0.05;
        let moon_disc_lo = (moon_angular_radius + moon_disc_edge).cos();
        let moon_disc_hi = (moon_angular_radius - moon_disc_edge).max(0.0).cos();
        Self {
            camera_position_distance: [
                uniforms.camera_position[0],
                uniforms.camera_position[1],
                uniforms.camera_position[2],
                uniforms.camera_distance,
            ],
            // x: 0 authored, 1 NASA surface, 2 NASA surface + NASA cloud map;
            // +4 when observed live clouds are bound (binding 12), +8 when
            // that texture also carries aerosol optical depth, +16 sea ice.
            material_state: [(u32::from(uniforms.nasa_materials) + u32::from(uniforms.nasa_clouds)
                + 4 * u32::from(uniforms.live_clouds)
                + 8 * u32::from(uniforms.live_clouds && uniforms.live_aerosol)
                + 16 * u32::from(uniforms.live_clouds && uniforms.live_sea_ice)) as f32,
                u32::from(uniforms.weather_valid_unix_utc > 0) as f32,
                u32::from(uniforms.aurora_valid_unix_utc > 0) as f32, moon_disc_hi],
            camera_forward: [
                uniforms.forward[0],
                uniforms.forward[1],
                uniforms.forward[2],
                moon_direction[0],
            ],
            camera_right: [
                uniforms.right[0],
                uniforms.right[1],
                uniforms.right[2],
                moon_direction[1],
            ],
            camera_up: [
                uniforms.up[0],
                uniforms.up[1],
                uniforms.up[2],
                moon_direction[2],
            ],
            viewport_rect: [viewport.x, viewport.y, viewport.width, viewport.height],
            canvas_rect: [
                uniforms.canvas.x,
                uniforms.canvas.y,
                uniforms.canvas.width,
                uniforms.canvas.height,
            ],
            sun_direction: [
                sun_direction[0],
                sun_direction[1],
                sun_direction[2],
                body_selector,
            ],
            projection_tangents: [
                uniforms.tan_half_fov_x,
                uniforms.tan_half_fov_y,
                uniforms.focus_x,
                uniforms.focus_y,
            ],
            celestial_distances: [
                uniforms.sun_distance_earth_radii,
                uniforms.moon_distance_earth_radii,
                uniforms.sun_radius_earth_radii,
                uniforms.moon_radius_earth_radii,
            ],
            moon_body_x: [
                world_to_body[0][0],
                world_to_body[0][1],
                world_to_body[0][2],
                sidereal_sine,
            ],
            moon_body_y: [
                world_to_body[1][0],
                world_to_body[1][1],
                world_to_body[1][2],
                sidereal_cosine,
            ],
            moon_body_z: [
                world_to_body[2][0],
                world_to_body[2][1],
                world_to_body[2][2],
                sun_glow_threshold,
            ],
            celestial_state: [
                moon_disc_lo,
                uniforms.moon_illuminated_fraction,
                uniforms.moon_phase_angle_radians,
                uniforms.time_of_day_seconds,
            ],
            celestial_sun_view: [
                sun_direction_from_camera[0],
                sun_direction_from_camera[1],
                sun_direction_from_camera[2],
                sun_angular_radius,
            ],
            celestial_moon_view: [
                moon_direction_from_camera[0],
                moon_direction_from_camera[1],
                moon_direction_from_camera[2],
                moon_angular_radius,
            ],
        }
    }
}

/// The native window a Vulkan output presents to.
#[derive(Clone, Copy, Debug)]
pub enum SurfaceSource {
    /// A wlr-layer-shell background surface.
    Wayland { display: *mut vk::wl_display, surface: *mut vk::wl_surface },
    /// An Xorg desktop-type window on an XCB connection.
    Xcb { connection: *mut vk::xcb_connection_t, window: vk::xcb_window_t },
}

fn mirror_earth_texture_longitude(direction: [f32; 3]) -> [f32; 3] {
    [direction[0], -direction[1], direction[2]]
}

fn projected_sphere_half_ndc(
    camera_distance: f32,
    radius: f32,
    tan_half_fov_x: f32,
    tan_half_fov_y: f32,
) -> Option<(f32, f32)> {
    if !camera_distance.is_finite()
        || !radius.is_finite()
        || !tan_half_fov_x.is_finite()
        || !tan_half_fov_y.is_finite()
        || camera_distance <= radius
        || radius <= 0.0
        || tan_half_fov_x <= 0.0
        || tan_half_fov_y <= 0.0
    {
        return None;
    }
    let tangent = radius / (camera_distance * camera_distance - radius * radius).sqrt();
    let half_ndc_x = tangent / tan_half_fov_x;
    let half_ndc_y = tangent / tan_half_fov_y;
    (half_ndc_x.is_finite() && half_ndc_y.is_finite()).then_some((half_ndc_x, half_ndc_y))
}

pub struct Renderer {
    // Keeps libvulkan loaded until every instance-owned object is destroyed.
    _entry: Entry,
    instance: Instance,
    surface_loader: ash::khr::surface::Instance,
    wayland_surface_loader: Option<ash::khr::wayland_surface::Instance>,
    xcb_surface_loader: Option<ash::khr::xcb_surface::Instance>,
    device: Option<DeviceState>,
    outputs: HashMap<u32, OutputTarget>,
    // A CPU-side descriptor kept only until its one-time, device-local upload.
    star_panorama: Option<StarPanorama>,
    // Stage-three fixed-resolution source previews. Production uses `.earthvt`
    // atlases, but these make the analytical material path testable now.
    day_color: Option<DayColorHemispheres>,
    surface_normals: Option<SurfaceNormalHemispheres>,
    night_emission: Option<NightEmissionPreview>,
    cloud_previews: Option<CloudPreviews>,
    tiling_noise: Option<TilingNoisePreview>,
    desert_cloud_mask: Option<DesertCloudMaskPreview>,
    height_preview: Option<HeightPreview>,
    moon_preview: Option<MoonPreview>,
    planet_previews: PlanetPreviews,
    saturn_ring_preview: Option<RingPreview>,
    vt_config: Option<(LayerDescriptor, u32)>,
    vt_streamer: Option<VirtualTextureStreamer>,
    /// Static VT (night lights, NASA cloud map, relief): config, streamer and
    /// the upload batch in flight.
    static_vt_config: Option<(LayerDescriptor, u32)>,
    static_streamer: Option<VirtualTextureStreamer>,
    static_pending_jobs: Vec<UploadJob>,
    /// Tiles the last feedback pass wanted (working sets, for `status`).
    vt_wanted_tiles: (usize, usize),
    vt_feedback: Feedback,
    vt_frame: u64,
    /// Tiles of the upload batch in flight.
    vt_pending_jobs: Vec<UploadJob>,
    /// Frame and view of the last CPU feedback pass (see `VT_FEEDBACK_INTERVAL`).
    vt_feedback_frame: u64,
    vt_feedback_view: Option<[f32; 7]>,
    debug_screenshots: bool,
    debug_scene_name: String,
    weather: Option<crate::weather::WeatherState>,
    pending_weather: Option<crate::weather::WeatherState>,
    nasa_materials: bool,
    nasa_clouds: bool,
    render_quality: RenderQuality,
    channels_label: String,
    textures_label: &'static str,
    star_format: &'static str,
    current_body: crate::body::Body,
    exposure: ExposureController,
    /// The GMGSI cloud texture occupies binding 12.
    live_clouds_bound: bool,
}

/// Camera auto-exposure, like an ISS photographer's: sunlit scenes, even
/// dark ocean, are shot at one "sunny 16" exposure (pre-exposure 1); night
/// series are long high-ISO exposures that render moonlit cloud nearly as
/// bright as daylight while city lights saturate (Earth Observation night
/// frames are ~15-19 stops above day frames).
struct ExposureController {
    ev: f32,
    target_ev: f32,
    last_update: Option<Instant>,
    snap_frames: u32,
    frame: u32,
}

impl ExposureController {
    const DAY_LUMINANCE_LOG2: f32 = -1.32; // 85th percentile of a sunlit scene ~0.4
    const ADAPTATION: f32 = 1.0;
    const MIN_EV: f32 = -1.5;
    const MAX_EV: f32 = 19.0;
    const TIME_CONSTANT_S: f32 = 0.9;

    fn new() -> Self {
        Self { ev: 0.0, target_ev: 0.0, last_update: None, snap_frames: 4, frame: 0 }
    }

    /// Daylight scenes, even dark ocean, are shot at the same "sunny 16"
    /// exposure (no opening up within this many stops of the day key).
    const DAY_LATITUDE_STOPS: f32 = 1.5;

    fn target_for(log2_luminance: f32) -> f32 {
        let delta = log2_luminance - Self::DAY_LUMINANCE_LOG2;
        let ev = if delta < 0.0 {
            (-Self::ADAPTATION * delta - Self::DAY_LATITUDE_STOPS).max(0.0)
        } else {
            -delta
        };
        // Night series keep moonlit land dark (~3 % grey) and let cloud and
        // lights carry the frame: ~1.5 stops below a full "normal" exposure.
        let night = ((ev - 4.0) / 8.0).clamp(0.0, 1.0);
        (ev - 1.6 * night * night * (3.0 - 2.0 * night)).clamp(Self::MIN_EV, Self::MAX_EV)
    }

    /// Log-space contrast of the photographic grade: the processed night
    /// frames are much harder than the day ones.
    fn contrast(&self) -> f32 {
        let night = ((self.ev - 4.0) / 8.0).clamp(0.0, 1.0);
        1.22 + 0.33 * night * night * (3.0 - 2.0 * night)
    }

    /// Light adaptation (toward a lower exposure) is fast, as for the eye
    /// and a camera's highlight protection; dark adaptation keeps the slow
    /// constant. Symmetric easing left the daylit Earth white for seconds
    /// after a look at the night sky (EV 17 -> 0 at ~1 stop per second).
    const LIGHT_TIME_CONSTANT_S: f32 = 0.12;
    const MAX_OVEREXPOSURE_STOPS: f32 = 1.5;

    fn update(&mut self, reading: Option<post::MeterReading>, sun_cap: Option<f32>, now: Instant) {
        if let Some(reading) = reading {
            self.target_ev = Self::target_for(reading.log2_luminance);
        }
        // A photographer never shoots a night exposure with the Sun in the
        // frame: its glare (added after metering) would white it out.
        if let Some(cap) = sun_cap {
            self.target_ev = self.target_ev.min(cap.max(Self::MIN_EV));
        }
        let dt = self.last_update.map_or(0.0, |last| now.saturating_duration_since(last).as_secs_f32());
        self.last_update = Some(now);
        if self.snap_frames > 0 {
            // Readings lag a frame or two behind the cut; keep snapping (and
            // drawing, see `converging`) until they describe the new view.
            self.snap_frames -= 1;
            self.ev = self.target_ev;
        } else if self.target_ev < self.ev {
            self.ev = self.ev.min(self.target_ev + Self::MAX_OVEREXPOSURE_STOPS);
            self.ev += (self.target_ev - self.ev) * (1.0 - (-dt / Self::LIGHT_TIME_CONSTANT_S).exp());
        } else {
            self.ev += (self.target_ev - self.ev) * (1.0 - (-dt / Self::TIME_CONSTANT_S).exp());
        }
        self.frame = self.frame.wrapping_add(1);
    }

    fn converging(&self) -> bool {
        self.snap_frames > 0 || (self.target_ev - self.ev).abs() > 0.03
    }
}

/// Fraction of a luminous disc's light reaching the camera per channel:
/// nine rays across the disc, each blocked by the ground or dimmed and
/// reddened by the air (the Sun rising through the limb).
fn disc_visibility(camera_km: [f64; 3], direction: [f64; 3], angular_radius: f64) -> [f64; 3] {
    let d = {
        let length = direction.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0e-9);
        direction.map(|v| v / length)
    };
    let reference = if d[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
    let cross = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let normalize = |v: [f64; 3]| {
        let length = v.iter().map(|c| c * c).sum::<f64>().sqrt().max(1.0e-12);
        v.map(|c| c / length)
    };
    let u = normalize(cross(reference, d));
    let v = cross(d, u);
    let mut sum = [0.0; 3];
    let mut count = 0.0;
    let mut add = |offset_u: f64, offset_v: f64| {
        let ray = normalize([0, 1, 2].map(|c| d[c] + u[c] * offset_u + v[c] * offset_v));
        let t = crate::sky::ray_transmittance(camera_km, ray);
        for c in 0..3 {
            sum[c] += t[c];
        }
        count += 1.0;
    };
    add(0.0, 0.0);
    for k in 0..8 {
        let a = k as f64 * std::f64::consts::FRAC_PI_4;
        add(0.72 * angular_radius * a.cos(), 0.72 * angular_radius * a.sin());
    }
    sum.map(|value| value / count)
}

impl Renderer {
    /// Switch which body the procedural material renders. Earth keeps the full
    /// pipeline; Jupiter forces the procedural gas-giant path (there is no
    /// Jupiter virtual-texture bundle).
    pub fn set_body(&mut self, body: crate::body::Body) {
        self.current_body = body;
    }

    pub fn body(&self) -> crate::body::Body {
        self.current_body
    }

    /// The loaded weather/aurora manifest, if any.
    pub fn weather_state(&self) -> Option<&crate::weather::WeatherState> {
        self.weather.as_ref()
    }

    pub fn has_weather_animation(&self) -> bool {
        self.weather.is_some() && self.current_body == crate::body::Body::Earth
    }
}

struct DeviceState {
    physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_family: u32,
    device: Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    // Shared GPU timing pool; each OutputTarget owns TIMESTAMP_QUERIES_PER_OUTPUT
    // slots at (output_id % TIMESTAMP_OUTPUT_SLOTS) * TIMESTAMP_QUERIES_PER_OUTPUT.
    timestamp_pool: vk::QueryPool,
    timestamp_period_ns: f32,
    swapchain_loader: ash::khr::swapchain::Device,
    pipeline: Option<Pipeline>,
    star_texture: Option<PinnedStarTexture>,
    // This set-0 material stays resident even when previews are absent. The
    // VT day-colour path is set 1 and can therefore be enabled independently.
    day_color_textures: Option<PinnedDayColorTextures>,
    virtual_texture: Option<VirtualTexture>,
    static_texture: Option<VirtualTexture>,
    /// Atmosphere tables bound at 17..=19 of the Earth set.
    sky_luts: Vec<PinnedBgraTexture>,
}

struct Pipeline {
    color_format: vk::Format,
    layout: vk::PipelineLayout,
    star_descriptor_set_layout: vk::DescriptorSetLayout,
    star_descriptor_pool: vk::DescriptorPool,
    vt_descriptor_set_layout: vk::DescriptorSetLayout,
    vt_descriptor_pool: vk::DescriptorPool,
    stars_procedural: vk::Pipeline,
    stars_textured: vk::Pipeline,
    earth_procedural: vk::Pipeline,
    earth_textured: vk::Pipeline,
    /// One additive quad per catalogue star, with its own push-constant block.
    star_points_layout: vk::PipelineLayout,
    star_points: vk::Pipeline,
    /// Camera stage: HDR scene target -> swapchain.
    post: PostPipeline,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct VtParams {
    base_dimensions_mip_count_enabled: [u32; 4],
    page_table_dimensions_slots_tile_size: [u32; 4],
}

impl VtParams {
    fn disabled() -> Self {
        Self {
            base_dimensions_mip_count_enabled: [1, 1, 1, 0],
            page_table_dimensions_slots_tile_size: [1, 1, 1, VT_ATLAS_SIZE],
        }
    }
}

/// A native-resolution equirectangular panorama kept resident for the entire
/// renderer lifetime. The source uses explicit BGRA8 sRGB sidecar metadata.
struct PinnedStarTexture {
    texture: PinnedBgraTexture,
    /// Hipparcos catalogue (binding 16), drawn as `star_count` quads.
    catalog: PinnedBgraTexture,
    star_count: u32,
    moon: PinnedBgraTexture,
    /// False while a non-Earth body is selected: binding 10 carries a 1x1
    /// fallback because the Moon disc only renders for Earth.
    moon_resident: bool,
    planets: PinnedPlanetTextures,
    descriptor_set: vk::DescriptorSet,
}

/// The authored albedo texture for the currently selected planet plus
/// Saturn's ring alpha, resident in VRAM only while that body is selected.
/// Every other body's binding carries a 1x1 fallback so the shared descriptor
/// set always holds a valid sampler; switching bodies swaps the resident
/// texture for the new one and frees the old. Each real equirectangular map
/// is tens to hundreds of MB, so pinning all five at once needlessly exhausts
/// VRAM on a wallpaper.
struct PinnedPlanetTextures {
    /// The body whose albedo is currently resident.
    resident: crate::body::Body,
    /// Resident albedo for `resident` (1x1 fallback when the body has no
    /// preview or is Earth).
    albedo: PinnedBgraTexture,
    /// Resident Saturn ring alpha (1x1 fallback unless Saturn is selected).
    ring: PinnedBgraTexture,
    /// 1x1 fallback bound to every non-resident planet binding.
    fallback: PinnedBgraTexture,
    resources: StarTextureResources,
}

/// Two fixed-resolution source hemispheres, used only until the virtual
/// texture streamer owns the day-colour channel.
struct PinnedDayColorTextures {
    resources: StarTextureResources,
    east: PinnedBgraTexture,
    west: PinnedBgraTexture,
    normal_east: PinnedBgraTexture,
    normal_west: PinnedBgraTexture,
    // A 1x1 black texture occupies this slot when the optional reference map
    // is absent, so the textured material always has a complete descriptor.
    night: PinnedBgraTexture,
    cloud_a: PinnedBgraTexture,
    cloud_ba: PinnedBgraTexture,
    tiling_noise: PinnedBgraTexture,
    desert_cloud_mask: PinnedBgraTexture,
    height: PinnedBgraTexture,
    atmosphere: PinnedBgraTexture,
    weather_fields: Option<PinnedBgraTexture>,
    cloud_height: Option<PinnedBgraTexture>,
    /// False while a non-Earth body is selected: bindings 1-9/11/12 carry
    /// 1x1 fallbacks; the procedural path never binds this set off-Earth.
    earth_resident: bool,
    descriptor_set: vk::DescriptorSet,
}

fn uses_textured_celestial_pass(has_panorama: bool, has_moon: bool) -> bool {
    has_panorama || has_moon
}

struct PinnedBgraTexture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    sampler: vk::Sampler,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VtSlotState {
    Free,
    Pending(TileKey),
    Resident(TileKey),
    Evicting(TileKey),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VtSlotAllocation {
    pub slot: u32,
    pub evicted: Option<TileKey>,
}

#[derive(Debug)]
pub struct VtSlotAllocator {
    slots: Vec<VtSlotState>,
    last_used: Vec<u64>,
    clock: u64,
}

impl VtSlotAllocator {
    pub fn new(slot_budget: u32) -> RendererResult<Self> {
        if slot_budget == 0 {
            return Err("virtual-texture slot budget must be non-zero".into());
        }
        Ok(Self {
            slots: vec![VtSlotState::Free; slot_budget as usize],
            last_used: vec![0; slot_budget as usize],
            clock: 0,
        })
    }

    pub fn states(&self) -> &[VtSlotState] { &self.slots }

    /// Whether `key` already owns a slot (uploaded, or upload in flight).
    pub fn holds(&self, key: TileKey) -> bool {
        self.slots.iter().any(|state| matches!(state, VtSlotState::Resident(k) | VtSlotState::Pending(k) if *k == key))
    }

    /// Refresh recency for resident tiles the current view still samples, so
    /// eviction takes tiles that left the view rather than the oldest upload
    /// (which evicted the always-visible coarse mips and thrashed).
    pub fn touch_visible(&mut self, visible: &std::collections::HashSet<TileKey>) {
        if visible.is_empty() {
            return;
        }
        self.clock = self.clock.saturating_add(1);
        for (slot, state) in self.slots.iter().enumerate() {
            if let VtSlotState::Resident(key) = state {
                if visible.contains(key) {
                    self.last_used[slot] = self.clock;
                }
            }
        }
    }

    pub fn request(&mut self, key: TileKey) -> Option<VtSlotAllocation> {
        if let Some(slot) = self
            .slots
            .iter()
            .position(|state| *state == VtSlotState::Resident(key))
        {
            self.clock = self.clock.saturating_add(1);
            self.last_used[slot] = self.clock;
            return None;
        }
        if self.slots.iter().any(|state| *state == VtSlotState::Pending(key)) {
            return None;
        }
        let slot = self.slots.iter().position(|state| *state == VtSlotState::Free)
            .or_else(|| self.slots.iter().enumerate()
                .filter(|(_, state)| matches!(state, VtSlotState::Resident(_)))
                .min_by_key(|(index, _)| (self.last_used[*index], *index))
                .map(|(index, _)| index))?;
        let evicted = match self.slots[slot] {
            VtSlotState::Resident(old) => {
                self.slots[slot] = VtSlotState::Evicting(old);
                Some(old)
            }
            _ => None,
        };
        self.slots[slot] = VtSlotState::Pending(key);
        self.clock = self.clock.saturating_add(1);
        self.last_used[slot] = self.clock;
        Some(VtSlotAllocation { slot: slot as u32, evicted })
    }

    pub fn mark_resident(&mut self, slot: u32, key: TileKey) -> bool {
        let Some(state) = self.slots.get_mut(slot as usize) else { return false };
        if *state != VtSlotState::Pending(key) && *state != VtSlotState::Evicting(key) { return false; }
        *state = VtSlotState::Resident(key);
        true
    }

    pub fn evict(&mut self, slot: u32) -> Option<TileKey> {
        let state = self.slots.get_mut(slot as usize)?;
        let old = match *state {
            VtSlotState::Resident(key) | VtSlotState::Evicting(key) => key,
            _ => return None,
        };
        *state = VtSlotState::Free;
        Some(old)
    }

    fn cancel_pending(&mut self, slot: u32, key: TileKey, evicted: Option<TileKey>) {
        if let Some(state) = self.slots.get_mut(slot as usize) {
            *state = evicted.map(VtSlotState::Resident).unwrap_or(VtSlotState::Free);
            if matches!(*state, VtSlotState::Resident(_)) {
                self.last_used[slot as usize] = self.clock;
            }
        }
        let _ = key;
    }
}

/// Which virtual texture an instance is: the monthly 500 m day colour, or
/// the static layers (night lights, NASA cloud map, relief normals) whose
/// three finest levels stream and whose tails stay resident.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VtKind {
    Day,
    Static,
}

/// Levels of the static layers that stream (32K, 16K, 8K); coarser ones are
/// the resident tails.
const STATIC_VT_MIPS: u16 = 3;

impl VtKind {
    /// Atlas formats in payload order (one per layer).
    fn atlas_formats(self) -> &'static [vk::Format] {
        match self {
            Self::Day => &[VT_ATLAS_FORMAT],
            Self::Static => &[vk::Format::BC4_UNORM_BLOCK, vk::Format::BC4_UNORM_BLOCK, vk::Format::BC5_UNORM_BLOCK],
        }
    }

    fn pixel_formats(self) -> &'static [PixelFormat] {
        match self {
            Self::Day => &[PixelFormat::Bc7],
            Self::Static => &[PixelFormat::Bc4, PixelFormat::Bc4, PixelFormat::Bc5],
        }
    }

    /// Bytes of one tile across all layers.
    fn tile_bytes(self) -> u64 {
        self.pixel_formats().iter().map(|format| format.encoded_tile_bytes()).sum()
    }

    /// Set-1 bindings: page table, parameters, atlases.
    fn bindings(self) -> (u32, u32, &'static [u32]) {
        match self {
            Self::Day => (1, 2, &[0]),
            Self::Static => (3, 4, &[5, 6, 7]),
        }
    }

    fn accepts(self, layer: LayerDescriptor) -> bool {
        match self {
            Self::Day => layer.channel == TextureChannel::DayColor && layer.format == PixelFormat::Bc7
                && layer.flags & (LAYER_FLAG_SRGB | LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y)
                    == (LAYER_FLAG_SRGB | LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y),
            Self::Static => layer.channel == TextureChannel::NightEmission && layer.format == PixelFormat::Bc4
                && layer.flags & (LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y) == (LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y),
        }
    }

    /// Levels the page table addresses.
    fn streamed_mips(self, layer: LayerDescriptor) -> u16 {
        match self {
            Self::Day => layer.mip_count,
            Self::Static => layer.mip_count.min(STATIC_VT_MIPS),
        }
    }
}

struct VirtualTexture {
    kind: VtKind,
    /// One atlas per layer; a slot index addresses the same tile in each.
    atlases: Vec<(ImageAllocation, vk::ImageView)>,
    atlas_sampler: vk::Sampler,
    page_table: ImageAllocation,
    page_table_view: vk::ImageView,
    page_table_sampler: vk::Sampler,
    params_buffer: vk::Buffer,
    params_memory: vk::DeviceMemory,
    descriptor_set: vk::DescriptorSet,
    params: VtParams,
    layer: Option<LayerDescriptor>,
    allocator: VtSlotAllocator,
    staging: Option<VtUploadStaging>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Staging bytes per tile: all layers' payloads, then the page-table
    /// words (eviction marker, new slot), 16-byte aligned.
    tile_stride: u64,
}

struct VtUploadStaging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    capacity: vk::DeviceSize,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    /// Tiles of the batch in flight (empty when the staging is free).
    pending: Vec<VtPendingUpload>,
}

/// Tiles uploaded per batch (one command buffer, one fence). One tile per
/// frame capped streaming at 30 tiles/s (15 on the globe), so a new view
/// visibly sharpened chunk by chunk for seconds; 48 x 70 KB is 3.3 MB.
const VT_UPLOAD_BATCH: usize = 48;
/// Frames between CPU feedback passes when the view moves smoothly.
const VT_FEEDBACK_INTERVAL: u64 = 6;

#[derive(Clone, Copy, Debug)]
struct VtPendingUpload {
    key: TileKey,
    slot: u32,
    evicted: Option<TileKey>,
}

impl VirtualTexture {
    /// Create the atlases, page table and parameters of `kind` (1x1 and
    /// disabled without `config`) and bind them in set 1: in `descriptor_set`
    /// when given (the static VT shares the day VT's set), else a new one.
    #[allow(clippy::too_many_arguments)]
    fn create_disabled(
        device: &Device,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        config: Option<(LayerDescriptor, u32)>,
        kind: VtKind,
        descriptor_set: Option<vk::DescriptorSet>,
    ) -> RendererResult<Self> {
        let required_transfer = vk::FormatFeatureFlags::TRANSFER_DST | vk::FormatFeatureFlags::SAMPLED_IMAGE;
        let supports = |format| unsafe { instance.get_physical_device_format_properties(physical_device, format) }
            .optimal_tiling_features.contains(required_transfer);
        if !kind.atlas_formats().iter().all(|&format| supports(format)) || !supports(VT_PAGE_TABLE_FORMAT) {
            return Err("GPU does not support the required virtual-texture atlas/page-table formats".into());
        }
        let (base_width, base_height, mip_count, requested_slots, layer) = match config {
            Some((layer, budget)) if kind.accepts(layer)
                && layer.base_width > 0 && layer.base_height > 0 && layer.mip_count > 0
                && budget > 0 => (layer.base_width, layer.base_height, kind.streamed_mips(layer), budget, Some(layer)),
            _ => (1, 1, 1, 1, None),
        };
        let max_array_layers = unsafe { instance.get_physical_device_properties(physical_device).limits.max_image_array_layers };
        let slot_budget = requested_slots.min(max_array_layers).max(1);
        let page_width = layer.map(|layer| layer.tile_grid(0).map(|grid| grid.0).unwrap_or(1)).unwrap_or(1);
        let page_height = layer.map(|layer| layer.tile_grid(0).map(|grid| grid.1).unwrap_or(1)).unwrap_or(1);
        let tile_stride = (kind.tile_bytes() + 8).div_ceil(16) * 16;
        // Handles start null (destroying a null handle is a no-op), so any
        // failure below can release whatever was created so far.
        let mut vt = Self {
            kind,
            atlases: Vec::new(),
            atlas_sampler: vk::Sampler::null(),
            page_table: ImageAllocation { image: vk::Image::null(), memory: vk::DeviceMemory::null() },
            page_table_view: vk::ImageView::null(),
            page_table_sampler: vk::Sampler::null(),
            params_buffer: vk::Buffer::null(),
            params_memory: vk::DeviceMemory::null(),
            descriptor_set: descriptor_set.unwrap_or_default(),
            params: VtParams {
                base_dimensions_mip_count_enabled: [base_width, base_height, u32::from(mip_count), u32::from(layer.is_some())],
                page_table_dimensions_slots_tile_size: [page_width, page_height, slot_budget, VT_ATLAS_SIZE],
            },
            layer,
            allocator: VtSlotAllocator::new(slot_budget)?,
            staging: None,
            memory_properties,
            tile_stride,
        };
        let built = (|| -> RendererResult<()> {
            unsafe {
                for &format in kind.atlas_formats() {
                    let atlas = ImageAllocation::create_array(device, memory_properties, format, VT_ATLAS_SIZE, VT_ATLAS_SIZE, slot_budget, vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)?;
                    let view = create_image_view(device, atlas.image, format, vk::ImageViewType::TYPE_2D_ARRAY, slot_budget);
                    let view = match view {
                        Ok(view) => view,
                        Err(error) => { atlas.destroy(device); return Err(error); }
                    };
                    vt.atlases.push((atlas, view));
                }
                vt.page_table = ImageAllocation::create_array(device, memory_properties, VT_PAGE_TABLE_FORMAT, page_width, page_height, u32::from(mip_count), vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)?;
                vt.page_table_view = create_image_view(device, vt.page_table.image, VT_PAGE_TABLE_FORMAT, vk::ImageViewType::TYPE_2D_ARRAY, u32::from(mip_count))?;
                vt.atlas_sampler = device.create_sampler(&vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR).min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(f32::from(mip_count)), None)?;
                vt.page_table_sampler = device.create_sampler(&vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST).min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE), None)?;
                let params_size = size_of::<VtParams>() as u64;
                vt.params_buffer = device.create_buffer(&vk::BufferCreateInfo::default().size(params_size).usage(vk::BufferUsageFlags::UNIFORM_BUFFER).sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
                let requirements = device.get_buffer_memory_requirements(vt.params_buffer);
                let memory_type = find_memory_type(memory_properties, requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
                vt.params_memory = device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(memory_type), None)?;
                device.bind_buffer_memory(vt.params_buffer, vt.params_memory, 0)?;
                let mapped = device.map_memory(vt.params_memory, 0, params_size, vk::MemoryMapFlags::empty())?;
                ptr::copy_nonoverlapping((&vt.params as *const VtParams).cast::<u8>(), mapped.cast::<u8>(), size_of::<VtParams>());
                device.unmap_memory(vt.params_memory);
                if descriptor_set.is_none() {
                    let layouts = [descriptor_set_layout];
                    vt.descriptor_set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts))?.remove(0);
                }
                let (page_binding, params_binding, atlas_bindings) = kind.bindings();
                let atlas_infos: Vec<_> = vt.atlases.iter().map(|(_, view)| [vk::DescriptorImageInfo::default().sampler(vt.atlas_sampler).image_view(*view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]).collect();
                let page_info = [vk::DescriptorImageInfo::default().sampler(vt.page_table_sampler).image_view(vt.page_table_view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let buffer_info = [vk::DescriptorBufferInfo::default().buffer(vt.params_buffer).offset(0).range(params_size)];
                let mut writes = vec![
                    vk::WriteDescriptorSet::default().dst_set(vt.descriptor_set).dst_binding(page_binding).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&page_info),
                    vk::WriteDescriptorSet::default().dst_set(vt.descriptor_set).dst_binding(params_binding).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&buffer_info),
                ];
                for (binding, info) in atlas_bindings.iter().zip(&atlas_infos) {
                    writes.push(vk::WriteDescriptorSet::default().dst_set(vt.descriptor_set).dst_binding(*binding).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(info));
                }
                device.update_descriptor_sets(&writes, &[]);
                let atlas_images: Vec<_> = vt.atlases.iter().map(|(atlas, _)| atlas.image).collect();
                initialize_vt_images(device, queue, command_pool, &atlas_images, slot_budget, vt.page_table.image, u32::from(mip_count))?;
                vt.staging = Some(create_vt_staging(device, memory_properties, command_pool, tile_stride * VT_UPLOAD_BATCH as u64)?);
            }
            Ok(())
        })();
        if let Err(error) = built {
            unsafe { vt.destroy(device, command_pool) };
            return Err(error);
        }
        Ok(vt)
    }

    /// Submit up to `VT_UPLOAD_BATCH` already-encoded tiles (each payload is
    /// every layer's 264x264 tile, concatenated) in one command buffer.
    /// Returns the indices of `tiles` that were admitted (a key that already
    /// owns a slot is skipped). Another batch is rejected until `poll_upload`
    /// observes the fence, bounding staging and residency.
    unsafe fn submit_tiles(&mut self, device: &Device, queue: vk::Queue, tiles: &[(TileKey, &[u8])]) -> RendererResult<Vec<usize>> {
        let Some(layer) = self.layer else { return Err("virtual texture is disabled".into()); };
        let tile_bytes = self.kind.tile_bytes();
        if tiles.len() > VT_UPLOAD_BATCH {
            return Err("VT upload batch is larger than its staging".into());
        }
        if tiles.iter().any(|(key, payload)| !layer.contains_key(*key) || key.mip >= self.kind.streamed_mips(layer) || payload.len() as u64 != tile_bytes) {
            return Err("VT tile key or payload size is invalid".into());
        }
        let Some(mut staging) = self.staging.take() else {
            return Err("VT staging must be initialized by the renderer integration hook".into());
        };
        if !staging.pending.is_empty() {
            self.staging = Some(staging);
            return Err("VT upload staging is busy".into());
        }
        let mut admitted = Vec::new();
        let mut uploads = Vec::new();
        for (index, (key, payload)) in tiles.iter().enumerate() {
            let Some(allocation) = self.allocator.request(*key) else { continue };
            let offset = self.tile_stride * uploads.len() as u64;
            ptr::copy_nonoverlapping(payload.as_ptr(), staging.mapped.add(offset as usize), payload.len());
            ptr::copy_nonoverlapping(u32::MAX.to_ne_bytes().as_ptr(), staging.mapped.add((offset + tile_bytes) as usize), 4);
            ptr::copy_nonoverlapping(allocation.slot.to_ne_bytes().as_ptr(), staging.mapped.add((offset + tile_bytes + 4) as usize), 4);
            uploads.push((VtPendingUpload { key: *key, slot: allocation.slot, evicted: allocation.evicted }, offset));
            admitted.push(index);
        }
        if uploads.is_empty() {
            self.staging = Some(staging);
            return Ok(admitted);
        }
        let layer_bytes: Vec<u64> = self.kind.pixel_formats().iter().map(|format| format.encoded_tile_bytes()).collect();
        let result = (|| -> RendererResult<()> {
            device.reset_fences(&[staging.fence])?;
            device.reset_command_buffer(staging.command_buffer, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(staging.command_buffer, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let color = |base_layer: u32, layer_count: u32| vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).base_array_layer(base_layer).layer_count(layer_count);
            let page_layers = u32::from(self.kind.streamed_mips(layer));
            let barrier = |image: vk::Image, range: vk::ImageSubresourceRange, to_transfer: bool| {
                let (old, new, src, dst) = if to_transfer {
                    (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::SHADER_READ, vk::AccessFlags::TRANSFER_WRITE)
                } else {
                    (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ)
                };
                vk::ImageMemoryBarrier::default().old_layout(old).new_layout(new).src_access_mask(src).dst_access_mask(dst).image(image).subresource_range(range)
            };
            let barriers = |to_transfer: bool| -> Vec<vk::ImageMemoryBarrier> {
                let mut list: Vec<_> = uploads.iter().flat_map(|(upload, _)| self.atlases.iter()
                    .map(move |(atlas, _)| barrier(atlas.image, color(upload.slot, 1), to_transfer))).collect();
                list.push(barrier(self.page_table.image, color(0, page_layers), to_transfer));
                list
            };
            device.cmd_pipeline_barrier(staging.command_buffer, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &barriers(true));
            let page_copy = |offset: u64, mip: u16, x: u32, y: u32| vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(0).base_array_layer(u32::from(mip)).layer_count(1))
                .image_offset(vk::Offset3D { x: x as i32, y: y as i32, z: 0 })
                .image_extent(vk::Extent3D { width: 1, height: 1, depth: 1 });
            for (upload, offset) in &uploads {
                // Invalidate the evicted tile's page before this tile claims
                // its slot; a later tile in the batch never takes a pending slot.
                if let Some(evicted) = upload.evicted {
                    let (old_x, old_y) = key_page(layer, evicted);
                    device.cmd_copy_buffer_to_image(staging.command_buffer, staging.buffer, self.page_table.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[page_copy(offset + tile_bytes, evicted.mip, old_x, old_y)]);
                }
                let mut layer_offset = *offset;
                for ((atlas, _), bytes) in self.atlases.iter().zip(&layer_bytes) {
                    let copy = vk::BufferImageCopy::default().buffer_offset(layer_offset)
                        .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(0).base_array_layer(upload.slot).layer_count(1))
                        .image_extent(vk::Extent3D { width: VT_ATLAS_SIZE, height: VT_ATLAS_SIZE, depth: 1 });
                    device.cmd_copy_buffer_to_image(staging.command_buffer, staging.buffer, atlas.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy]);
                    layer_offset += bytes;
                }
                let (page_x, page_y) = key_page(layer, upload.key);
                device.cmd_copy_buffer_to_image(staging.command_buffer, staging.buffer, self.page_table.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[page_copy(offset + tile_bytes + 4, upload.key.mip, page_x, page_y)]);
            }
            device.cmd_pipeline_barrier(staging.command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &barriers(false));
            device.end_command_buffer(staging.command_buffer)?;
            let command_buffers = [staging.command_buffer];
            device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&command_buffers)], staging.fence)?;
            Ok(())
        })();
        if let Err(error) = result {
            for (upload, _) in &uploads {
                self.allocator.cancel_pending(upload.slot, upload.key, upload.evicted);
            }
            self.staging = Some(staging);
            return Err(error);
        }
        staging.pending = uploads.into_iter().map(|(upload, _)| upload).collect();
        self.staging = Some(staging);
        Ok(admitted)
    }

    /// Poll the batch fence. Returns the committed uploads when ready.
    unsafe fn poll_upload(&mut self, device: &Device) -> RendererResult<Option<Vec<VtPendingUpload>>> {
        let Some(staging) = self.staging.as_mut() else { return Ok(None); };
        if staging.pending.is_empty() {
            return Ok(None);
        }
        match device.get_fence_status(staging.fence) {
            Ok(true) => {
                let pending = std::mem::take(&mut staging.pending);
                for upload in &pending {
                    if !self.allocator.mark_resident(upload.slot, upload.key) {
                        return Err("VT upload fence completed for an unexpected allocator state".into());
                    }
                }
                Ok(Some(pending))
            }
            Ok(false) | Err(vk::Result::NOT_READY) => Ok(None),
            Err(error) => {
                for upload in std::mem::take(&mut staging.pending) {
                    self.allocator.cancel_pending(upload.slot, upload.key, upload.evicted);
                }
                Err(error.into())
            }
        }
    }

    unsafe fn destroy(&mut self, device: &Device, command_pool: vk::CommandPool) {
        if let Some(staging) = self.staging.take() { device.unmap_memory(staging.memory); device.destroy_fence(staging.fence, None); device.free_command_buffers(command_pool, &[staging.command_buffer]); device.destroy_buffer(staging.buffer, None); device.free_memory(staging.memory, None); }
        device.destroy_sampler(self.atlas_sampler, None);
        for (atlas, view) in self.atlases.drain(..) {
            device.destroy_image_view(view, None);
            atlas.destroy(device);
        }
        device.destroy_sampler(self.page_table_sampler, None); device.destroy_image_view(self.page_table_view, None); device.destroy_image(self.page_table.image, None); device.free_memory(self.page_table.memory, None);
        device.destroy_buffer(self.params_buffer, None); device.free_memory(self.params_memory, None);
    }
}

/// Commit a finished upload batch, then submit the next one (up to
/// `VT_UPLOAD_BATCH` tiles the I/O worker has read).
fn pump_vt_uploads(
    streamer: &mut VirtualTextureStreamer,
    texture: &mut VirtualTexture,
    pending: &mut Vec<UploadJob>,
    device: &Device,
    queue: vk::Queue,
    frame: u64,
) -> RendererResult<()> {
    if let Some(completed) = unsafe { texture.poll_upload(device)? } {
        let jobs = std::mem::take(pending);
        if jobs.len() != completed.len() || jobs.iter().zip(&completed).any(|(job, upload)| job.key != upload.key) {
            return Err("VT upload batch completed for the wrong tiles".into());
        }
        let now = Instant::now();
        for (job, upload) in jobs.iter().zip(&completed) {
            if let Some(evicted) = upload.evicted {
                streamer.residency_mut().remove(evicted);
            }
            streamer.mark_uploaded(job, frame, now)?;
        }
    }
    if pending.is_empty() {
        // A tile can be requested again while its first upload is still
        // on the GPU; drop such duplicates instead of failing the frame.
        let mut batch: Vec<UploadJob> = Vec::new();
        while batch.len() < VT_UPLOAD_BATCH {
            let Some(job) = streamer.poll_one()? else { break };
            if !texture.allocator.holds(job.key) && !batch.iter().any(|queued| queued.key == job.key) {
                batch.push(job);
            }
        }
        if !batch.is_empty() {
            let tiles: Vec<(TileKey, &[u8])> = batch.iter().map(|job| (job.key, job.payload.as_slice())).collect();
            let admitted = unsafe { texture.submit_tiles(device, queue, &tiles)? };
            drop(tiles);
            let mut admitted = admitted.into_iter().peekable();
            *pending = batch.into_iter().enumerate()
                .filter_map(|(index, job)| admitted.next_if_eq(&index).is_some().then_some(job))
                .collect();
        }
    }
    Ok(())
}

fn key_page(layer: LayerDescriptor, key: TileKey) -> (u32, u32) {
    let (tiles_x, tiles_y) = layer.tile_grid(key.mip).unwrap_or((1, 1));
    (key.x.min(tiles_x.saturating_sub(1)), key.y.min(tiles_y.saturating_sub(1)))
}

unsafe fn create_vt_staging(
    device: &Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    command_pool: vk::CommandPool,
    capacity: vk::DeviceSize,
) -> RendererResult<VtUploadStaging> {
    let buffer = device.create_buffer(&vk::BufferCreateInfo::default().size(capacity).usage(vk::BufferUsageFlags::TRANSFER_SRC).sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
    let requirements = device.get_buffer_memory_requirements(buffer);
    let memory_type = match find_memory_type(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
        Ok(index) => index,
        Err(error) => { device.destroy_buffer(buffer, None); return Err(error); }
    };
    let memory = match device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(memory_type), None) {
        Ok(memory) => memory,
        Err(error) => { device.destroy_buffer(buffer, None); return Err(error.into()); }
    };
    if let Err(error) = device.bind_buffer_memory(buffer, memory, 0) {
        device.free_memory(memory, None); device.destroy_buffer(buffer, None); return Err(error.into());
    }
    let mapped = match device.map_memory(memory, 0, capacity, vk::MemoryMapFlags::empty()) {
        Ok(mapped) => mapped.cast::<u8>(),
        Err(error) => { device.free_memory(memory, None); device.destroy_buffer(buffer, None); return Err(error.into()); }
    };
    let command_buffer = match device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)) {
        Ok(mut buffers) => buffers.remove(0),
        Err(error) => { device.unmap_memory(memory); device.free_memory(memory, None); device.destroy_buffer(buffer, None); return Err(error.into()); }
    };
    let fence = match device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None) {
        Ok(fence) => fence,
        Err(error) => { device.unmap_memory(memory); device.free_command_buffers(command_pool, &[command_buffer]); device.free_memory(memory, None); device.destroy_buffer(buffer, None); return Err(error.into()); }
    };
    Ok(VtUploadStaging { buffer, memory, mapped, capacity, command_buffer, fence, pending: Vec::new() })
}

fn create_image_view(device: &Device, image: vk::Image, format: vk::Format, view_type: vk::ImageViewType, layers: u32) -> RendererResult<vk::ImageView> {
    let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(layers);
    Ok(unsafe { device.create_image_view(&vk::ImageViewCreateInfo::default().image(image).view_type(view_type).format(format).subresource_range(range), None)? })
}

unsafe fn initialize_vt_images(
    device: &Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    atlases: &[vk::Image],
    atlas_layers: u32,
    page_table: vk::Image,
    page_layers: u32,
) -> RendererResult<()> {
    let command = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
    let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
    let result = (|| -> RendererResult<()> {
        device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        let atlas_range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(atlas_layers);
        let page_range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(page_layers);
        let transition = |image: vk::Image, range: vk::ImageSubresourceRange, old, new, src, dst| vk::ImageMemoryBarrier::default()
            .old_layout(old).new_layout(new).src_access_mask(src).dst_access_mask(dst).image(image).subresource_range(range);
        let mut to_transfer: Vec<_> = atlases.iter().map(|&atlas| transition(atlas, atlas_range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE)).collect();
        to_transfer.push(transition(page_table, page_range, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE));
        device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &to_transfer);
        let page_clear = vk::ClearColorValue { uint32: [u32::MAX, 0, 0, 0] };
        device.cmd_clear_color_image(command, page_table, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &page_clear, &[page_range]);
        let mut to_sampled: Vec<_> = atlases.iter().map(|&atlas| transition(atlas, atlas_range, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ)).collect();
        to_sampled.push(transition(page_table, page_range, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ));
        device.cmd_pipeline_barrier(command, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &to_sampled);
        device.end_command_buffer(command)?;
        let commands = [command];
        device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&commands)], fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
        Ok(())
    })();
    device.destroy_fence(fence, None);
    device.free_command_buffers(command_pool, &[command]);
    result
}

struct StagingBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

struct ImageAllocation {
    image: vk::Image,
    memory: vk::DeviceMemory,
}

#[derive(Clone, Copy)]
struct StarTextureResources {
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    max_image_dimension: u32,
    srgb_format_features: vk::FormatFeatureFlags,
    bc1_format_features: vk::FormatFeatureFlags,
    bc3_format_features: vk::FormatFeatureFlags,
    bc4_format_features: vk::FormatFeatureFlags,
    linear_format_features: vk::FormatFeatureFlags,
    /// 0 when sampler anisotropy is unavailable.
    max_anisotropy: f32,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

// A fence or acquire wait that takes longer than this is anomalous: a healthy
// idle frame finishes in ~0.2 ms of CPU time and its fence clears long before
// the next frame's wait begins. Everything between this threshold and the 3 s
// bounded cap used to be a silent zone - the compositor or driver could starve
// the sync object for multi-second bursts (observed on NVIDIA + Hyprland
// explicit sync: no pending GPU work, fence signals late, restart fixes it)
// and leave no trace. Slow waits are now logged, and a run of them forces a
// swapchain rebuild so the session self-heals the way a restart would.
const SLOW_WAIT_THRESHOLD_MS: f32 = 250.0;
const SELF_HEAL_CONSECUTIVE_SLOW_WAITS: u32 = 3;

#[derive(Debug, Default)]
struct SlowWaitTracker {
    consecutive_slow: u32,
    // Set when the tracker trips; render() recreates the swapchain once and
    // clears it.
    self_heal_requested: bool,
}

impl SlowWaitTracker {
    /// Returns true when this individual wait exceeded the slow threshold.
    fn record(&mut self, wait_ms: f32) -> bool {
        let slow = wait_ms >= SLOW_WAIT_THRESHOLD_MS;
        if slow {
            self.consecutive_slow += 1;
            if self.consecutive_slow >= SELF_HEAL_CONSECUTIVE_SLOW_WAITS {
                self.self_heal_requested = true;
            }
        } else {
            self.consecutive_slow = 0;
        }
        slow
    }

    fn take_self_heal(&mut self) -> bool {
        std::mem::take(&mut self.self_heal_requested)
    }
}

struct OutputTarget {
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: vk::Semaphore,
    // Presentation may still be waiting on a binary semaphore after the
    // graphics fence signals. Associate each signal semaphore with one image
    // and only reuse it after that image has been acquired again.
    render_finished: Vec<vk::Semaphore>,
    in_flight: vk::Fence,
    needs_recreate: bool,
    viewport: LogicalRect,
    debug_capture: Option<DebugCapture>,
    output_name: String,
    capture_supported: bool,
    // Slot in DeviceState::timestamp_pool owned by this output.
    query_base: u32,
    // Raw ticks from the last completed frame: [frame top, after star draw,
    // after Earth draw]. Multiply by DeviceState::timestamp_period_ns.
    /// Frame top, after stars, after the Earth, after the camera stage.
    last_gpu_ticks: [u64; 4],
    // Sample every N submissions, read after that submission's fence signals.
    query_clock: u32,
    query_pending: bool,
    slow_waits: SlowWaitTracker,
    /// Scene-referred radiance target of the camera stage (created on first
    /// render, once the pipeline exists; rebuilt with the extent).
    hdr: Option<HdrTarget>,
    /// Exposure metering of the last completed frame.
    last_meter: Option<post::MeterReading>,
}

/// Per-frame camera state shared by every output.
#[derive(Clone, Copy)]
struct CameraSettings {
    preexposure: f32,
    post: PostFrame,
}

struct DebugCapture {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    capacity: vk::DeviceSize,
    monitor_name: String,
    output_name: String,
    pixel_order: PixelOrder,
    pending: bool,
    last_written: Option<Instant>,
    periodic: bool,
    resume_scene: Option<String>,
}

impl DebugCapture {
    fn create(
        device: &Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        extent: vk::Extent2D,
        format: vk::Format,
        monitor_name: &str,
        scene_name: &str,
    ) -> RendererResult<Self> {
        let pixel_order = match format {
            vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM => PixelOrder::Bgra,
            vk::Format::R8G8B8A8_SRGB | vk::Format::R8G8B8A8_UNORM => PixelOrder::Rgba,
            _ => return Err("debug screenshot readback requires an 8-bit RGBA swapchain".into()),
        };
        let capacity = u64::from(extent.width)
            .checked_mul(u64::from(extent.height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or("debug screenshot buffer size overflows")?;
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(capacity)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?
        };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_type = match find_readback_memory_type(
            memory_properties,
            requirements.memory_type_bits,
        ) {
            Ok(memory_type) => memory_type,
            Err(error) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(error);
            }
        };
        let memory = match unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(error.into());
            }
        };
        if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            return Err(error.into());
        }
        Ok(Self {
            buffer,
            memory,
            capacity,
            monitor_name: monitor_name.to_owned(),
            output_name: format!("scene-{scene_name}-{monitor_name}"),
            pixel_order,
            pending: true,
            last_written: None,
            periodic: true,
            resume_scene: None,
        })
    }

    fn is_due(&self) -> bool {
        self.pending || (self.periodic && self.last_written
            .map(|last_written| last_written.elapsed() >= DEBUG_CAPTURE_INTERVAL)
            .unwrap_or(true))
    }

    fn set_scene(&mut self, scene_name: &str) {
        self.output_name = format!("scene-{scene_name}-{}", self.monitor_name);
        self.pending = true;
        self.last_written = None;
    }

    unsafe fn write_jpeg(
        &mut self,
        device: &Device,
        extent: vk::Extent2D,
        uniforms: FrameUniforms,
        body_selector: f32,
        viewport: LogicalRect,
    ) -> RendererResult<bool> {
        let mapped = device.map_memory(
            self.memory,
            0,
            self.capacity,
            vk::MemoryMapFlags::empty(),
        )?;
        let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), self.capacity as usize);
        let pixels = bytes.to_vec();
        device.unmap_memory(self.memory);
        if !debug_capture::submit_frame(&self.output_name, extent.width, extent.height,
            pixels, self.pixel_order, serde_json::json!({"uniforms": uniforms,
                "body_id": body_selector, "viewport": viewport,
                "body_radius_km": crate::body::Body::ALL[body_selector as usize].equatorial_radius_km(),
                "camera_altitude_km": (uniforms.camera_distance as f64 / SCENE_EARTH_RADIUS as f64 - 1.0)
                    * crate::body::Body::ALL[body_selector as usize].equatorial_radius_km(),
                "pixel_order": format!("{:?}", self.pixel_order),
                "scene_surface_radius": SCENE_EARTH_RADIUS,
                "cloud_altitude_km": crate::atmosphere::CLOUD_ALTITUDE_KM,
                "solar_irradiance_w_m2": crate::astronomy::solar_irradiance_w_m2(
                    uniforms.sun_distance_earth_radii as f64 * crate::astronomy::EARTH_EQUATORIAL_RADIUS_KM),
                "earth_radius_km": crate::astronomy::EARTH_EQUATORIAL_RADIUS_KM}))? {
            return Ok(false);
        }
        self.pending = false;
        self.last_written = Some(Instant::now());
        Ok(true)
    }

    unsafe fn destroy(&self, device: &Device) {
        device.destroy_buffer(self.buffer, None);
        device.free_memory(self.memory, None);
    }
}

impl Renderer {
    /// Read back the next rendered frame on every output without entering debug mode.
    /// Normal operation allocates no capture buffer; successful one-shot buffers
    /// are released only after their submission fence has signaled.
    pub fn capture_frame(&mut self) -> RendererResult<serde_json::Value> {
        let device = self.device.as_ref().ok_or("no Vulkan device")?;
        if self.outputs.is_empty() { return Err("no configured outputs".into()); }
        if self.outputs.values().any(|target| !target.capture_supported
            || target.debug_capture.as_ref().is_some_and(|capture| !capture.periodic)) {
            return Err("capture unavailable or already pending".into());
        }
        // Never replace a debug buffer that a timed-out submission still owns.
        for target in self.outputs.values() {
            if target.debug_capture.is_some() && !unsafe { device.device.get_fence_status(target.in_flight)? } {
                return Err("previous debug readback is in flight; retry capture".into());
            }
        }
        let id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
        let mut captures = Vec::new();
        for (&output_id, target) in &self.outputs {
            match DebugCapture::create(&device.device, device.memory_properties,
                target.extent, target.format, &target.output_name, &format!("capture-{id}")) {
                Ok(mut capture) => {
                    capture.periodic = false;
                    capture.resume_scene = self.debug_screenshots.then(|| self.debug_scene_name.clone());
                    captures.push((output_id, capture));
                }
                Err(error) => {
                    for (_, capture) in captures { unsafe { capture.destroy(&device.device); } }
                    return Err(error);
                }
            }
        }
        let mut images = Vec::new();
        for (output_id, capture) in captures {
            images.push(debug_capture::output_path(&capture.output_name));
            let target = self.outputs.get_mut(&output_id).unwrap();
            if let Some(old) = target.debug_capture.replace(capture) {
                // A debug readback can still be in flight following a timeout.
                unsafe { old.destroy(&device.device); }
            }
        }
        Ok(serde_json::json!({"status": "queued", "id": id.to_string(), "images": images}))
    }

    pub fn new(
        debug_screenshots: bool,
        render_quality: RenderQuality,
        bundle: Option<ValidatedBundle>,
    ) -> RendererResult<Self> {
        let vt_streamer = match env::var_os(VT_ENV) {
            Some(path) => {
                let budget_mb = env::var(VT_BUDGET_ENV)
                    .ok()
                    .and_then(|value| value.parse::<u32>().ok())
                    .unwrap_or(VT_DEFAULT_BUDGET_MB);
                let budget_bytes = u64::from(budget_mb)
                    .checked_mul(1024 * 1024)
                    .ok_or("virtual-texture budget is too large")?;
                Some(VirtualTextureStreamer::spawn(path, budget_bytes)?)
            }
            None => None,
        };
        let vt_config = vt_streamer.as_ref().and_then(|streamer| {
            streamer.layer_for_channel(TextureChannel::DayColor).map(|layer| {
                let budget_mb = env::var(VT_BUDGET_ENV)
                    .ok()
                    .and_then(|value| value.parse::<u32>().ok())
                    .unwrap_or(VT_DEFAULT_BUDGET_MB);
                let budget_bytes = u64::from(budget_mb) * 1024 * 1024;
                let tile_bytes = PixelFormat::Bc7.encoded_tile_bytes();
                let slots = (budget_bytes / tile_bytes).clamp(1, u64::from(u32::MAX)) as u32;
                (layer, slots)
            })
        });
        // The static layers stream their three finest levels; their tails
        // (data_dir.rs points the night, cloud and relief maps at them) stay
        // resident and stand in until a tile arrives.
        let static_budget_mb = env::var(STATIC_VT_BUDGET_ENV).ok().and_then(|value| value.parse::<u64>().ok()).unwrap_or(STATIC_VT_DEFAULT_BUDGET_MB);
        let static_streamer = env::var_os(STATIC_VT_ENV).and_then(|path| {
            VirtualTextureStreamer::spawn_kind(path, static_budget_mb * 1024 * 1024, StreamKind::Static)
                .map_err(|error| eprintln!("earth-native: static virtual texture disabled: {error}"))
                .ok()
        });
        let static_vt_config = static_streamer.as_ref()
            .and_then(|streamer| streamer.layer_for_channel(TextureChannel::NightEmission))
            .map(|layer| (layer, (static_budget_mb * 1024 * 1024 / VtKind::Static.tile_bytes()).clamp(1, u64::from(u32::MAX)) as u32));
        let star_panorama = load_optional_star_panorama()?;
        let star_format = match star_panorama.as_ref().map(StarPanorama::format) {
            Some(StarPanoramaFormat::Bgra8Srgb) => "bgra8",
            Some(StarPanoramaFormat::Bc1Srgb) => "bc1",
            None => "procedural",
        };
        // The fixed-resolution day map stays loaded with a virtual texture:
        // it fills pages that are not resident yet, and the other Earth
        // layers (lights, clouds, relief) load only alongside it.
        let day_color = load_optional_day_color_hemispheres()?;
        let surface_normals = load_optional_surface_normal_hemispheres(day_color.is_some())?;
        let night_emission = load_optional_night_emission(day_color.is_some())?;
        let cloud_previews = load_optional_cloud_previews(day_color.is_some())?;
        let tiling_noise = load_optional_tiling_noise(cloud_previews.is_some())?;
        let desert_cloud_mask = load_optional_desert_cloud_mask(cloud_previews.is_some())?;
        let height_preview = load_optional_height_preview(cloud_previews.is_some())?;
        let moon_preview = load_optional_moon_preview()?;
        let planet_previews = load_optional_planet_previews()?;
        let saturn_ring_preview = load_optional_saturn_ring_preview()?;
        let entry = unsafe { Entry::load()? };
        let application_name = CString::new("earth-native")?;
        let engine_name = CString::new("earth-native")?;
        let application_info = vk::ApplicationInfo::default()
            .application_name(&application_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);
        // Enable whichever window-system surface extensions the loader offers:
        // Wayland (layer-shell) and XCB (Xorg desktop windows).
        let available = unsafe { entry.enumerate_instance_extension_properties(None)? };
        let has_extension = |name: &std::ffi::CStr| {
            available.iter().any(|property| property.extension_name_as_c_str() == Ok(name))
        };
        let has_wayland = has_extension(vk::KHR_WAYLAND_SURFACE_NAME);
        let has_xcb = has_extension(vk::KHR_XCB_SURFACE_NAME);
        if !has_wayland && !has_xcb {
            return Err("Vulkan loader offers neither VK_KHR_wayland_surface nor VK_KHR_xcb_surface".into());
        }
        let mut extensions = vec![vk::KHR_SURFACE_NAME.as_ptr()];
        if has_wayland {
            extensions.push(vk::KHR_WAYLAND_SURFACE_NAME.as_ptr());
        }
        if has_xcb {
            extensions.push(vk::KHR_XCB_SURFACE_NAME.as_ptr());
        }
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_extension_names(&extensions);
        let instance = unsafe { entry.create_instance(&create_info, None)? };
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let wayland_surface_loader =
            has_wayland.then(|| ash::khr::wayland_surface::Instance::new(&entry, &instance));
        let xcb_surface_loader =
            has_xcb.then(|| ash::khr::xcb_surface::Instance::new(&entry, &instance));

        // Process-lifetime status labels, computed once: the bundle moves into
        // the struct below, and the env is fixed at startup.
        let channels_label = match bundle.as_ref().map(|bundle| bundle.channels.as_slice()) {
            None | Some([]) => "preview".to_owned(),
            Some(channels) => channels.iter().map(|channel| format!("{channel:?}").to_lowercase()).collect::<Vec<_>>().join(","),
        };
        let textures_label: &'static str = if std::env::var_os("EARTH_NATIVE_NASA_DATA").is_some() { "NASA" } else { "reference" };
        Ok(Self {
            _entry: entry,
            instance,
            surface_loader,
            wayland_surface_loader,
            xcb_surface_loader,
            device: None,
            outputs: HashMap::new(),
            vt_config,
            vt_streamer,
            static_vt_config,
            static_streamer,
            static_pending_jobs: Vec::new(),
            vt_wanted_tiles: (0, 0),
            vt_feedback: Feedback::default(),
            vt_frame: 0,
            vt_pending_jobs: Vec::new(),
            vt_feedback_frame: 0,
            vt_feedback_view: None,
            star_panorama,
            day_color,
            surface_normals,
            night_emission,
            cloud_previews,
            tiling_noise,
            desert_cloud_mask,
            height_preview,
            moon_preview,
            planet_previews,
            saturn_ring_preview,
            debug_scene_name: "day-earth".to_owned(),
            weather: None,
            pending_weather: if std::env::var_os("EARTH_NATIVE_NASA_DATA").is_some() {
                crate::weather::WeatherState::load().unwrap_or_else(|error| {
                    eprintln!("weather unavailable; rendering without live fields: {error}"); None
                })
            } else { None },
            nasa_materials: std::env::var_os("EARTH_NATIVE_NASA_DATA").is_some(),
            nasa_clouds: std::env::var_os("EARTH_NATIVE_NASA_DATA").is_some()
                && std::env::var_os("EARTH_NATIVE_NASA_CLOUDS").is_some(),
            debug_screenshots,
            render_quality,
            channels_label,
            textures_label,
            star_format,
            current_body: crate::body::from_environment(),
            exposure: ExposureController::new(),
            live_clouds_bound: false,
        })
    }

    pub fn status_fields(&self) -> String {
        let channels = self.channels_label.as_str();
        let vt_bytes = self.vt_streamer.as_ref().map(|streamer| streamer.residency().resident_bytes()).unwrap_or(0);
        // GPU frame timing from the timestamp queries, summed over outputs
        // (each renders its own frame): stars, Earth, camera stage (mip
        // pyramid, glare, tone curve).
        let (gpu_total_ms, gpu_star_ms, gpu_earth_ms, gpu_post_ms) = match self.device.as_ref() {
            Some(device) => {
                let mut ticks = [0u64; 3];
                for target in self.outputs.values() {
                    let [top, after_stars, after_earth, after_post] = target.last_gpu_ticks;
                    if after_post > after_earth && after_earth >= after_stars && after_stars >= top {
                        ticks[0] += after_stars - top;
                        ticks[1] += after_earth - after_stars;
                        ticks[2] += after_post - after_earth;
                    }
                }
                let period = f64::from(device.timestamp_period_ns) / 1_000_000.0;
                let [star, earth, post] = ticks.map(|value| value as f64 * period);
                (star + earth + post, star, earth, post)
            }
            None => (0.0, 0.0, 0.0, 0.0),
        };
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let weather = self.weather.as_ref().map(|state| format!("{} weather_utc={} weather_age_hours={:.1} aurora=NOAA-OVATION aurora_utc={} lightning=simulated",
            state.age_label(now), state.valid_unix_utc, (now - state.valid_unix_utc) as f64 / 3600.0, state.aurora_unix_utc)).unwrap_or_else(|| "unavailable".to_owned());
        let textures = self.textures_label;
        let live = self.weather.as_ref().filter(|_| self.live_clouds_bound);
        let clouds = match live {
            Some(state) if state.live_clouds(now).is_some() => "NOAA-GMGSI",
            _ if self.nasa_clouds => "NASA",
            _ => "authored",
        };
        let aerosol = match live {
            Some(state) if state.live_clouds(now).is_some() && state.live_aerosol(now) => "NOAA-GEFS",
            _ => "climatology",
        };
        let sea_ice = match live {
            Some(state) if state.live_clouds(now).is_some() && state.live_sea_ice(now) => "OSI-SAF",
            _ => "none",
        };
        format!("vt_tiles={} static_tiles={} body={} quality={} hdr={} stars={} textures={textures} clouds={clouds} aerosol={aerosol} sea_ice={sea_ice} star_map=reference weather={weather} vt_vram_mb={:.1} channels={channels} gpu_ms={gpu_total_ms:.2}(star={gpu_star_ms:.2},earth={gpu_earth_ms:.2},post={gpu_post_ms:.2})", self.vt_wanted_tiles.0, self.vt_wanted_tiles.1, self.current_body.name(), self.render_quality.name(), "swapchain-sdr", self.star_format, vt_bytes as f64 / (1024.0 * 1024.0))
    }

    pub unsafe fn configure_output(
        &mut self,
        output_id: u32,
        source: SurfaceSource,
        extent: vk::Extent2D,
        viewport: LogicalRect,
        output_name: &str,
    ) -> RendererResult<()> {
        if extent.width == 0 || extent.height == 0 || !viewport.is_valid() {
            return Ok(());
        }

        if self.outputs.contains_key(&output_id) {
            self.recreate_output(output_id, extent, viewport)?;
            return Ok(());
        }

        let surface = match source {
            SurfaceSource::Wayland { display, surface } => {
                let loader = self
                    .wayland_surface_loader
                    .as_ref()
                    .ok_or("Vulkan loader lacks VK_KHR_wayland_surface")?;
                let create_info = vk::WaylandSurfaceCreateInfoKHR::default()
                    .display(display)
                    .surface(surface);
                loader.create_wayland_surface(&create_info, None)?
            }
            SurfaceSource::Xcb { connection, window } => {
                let loader = self
                    .xcb_surface_loader
                    .as_ref()
                    .ok_or("Vulkan loader lacks VK_KHR_xcb_surface")?;
                let create_info = vk::XcbSurfaceCreateInfoKHR::default()
                    .connection(connection)
                    .window(window);
                loader.create_xcb_surface(&create_info, None)?
            }
        };
        self.ensure_device(surface)?;
        let device = self
            .device
            .as_ref()
            .expect("device initialized for a Vulkan surface");
        let debug_output_name = self.debug_screenshots.then_some(output_name);
        let mut target = OutputTarget::create(
            &self.surface_loader,
            device,
            surface,
            extent,
            viewport,
            debug_output_name,
            &self.debug_scene_name,
        )?;
        target.output_name = output_name.to_owned();
        target.query_base =
            (output_id % TIMESTAMP_OUTPUT_SLOTS) * TIMESTAMP_QUERIES_PER_OUTPUT;
        self.ensure_pipeline(target.format)?;
        self.outputs.insert(output_id, target);
        Ok(())
    }

    /// Queue one encoded BC7 tile for the configured day-colour VT.
    pub fn submit_day_color_tile(&mut self, key: TileKey, payload: &[u8]) -> RendererResult<bool> {
        let device = self.device.as_mut().ok_or("Vulkan device is not initialized")?;
        let virtual_texture = device.virtual_texture.as_mut().ok_or("VT resources are not initialized")?;
        unsafe { Ok(!virtual_texture.submit_tiles(&device.device, device.queue, &[(key, payload)])?.is_empty()) }
    }

    /// Poll the asynchronous VT upload fence and commit the ready tiles.
    pub fn poll_day_color_upload(&mut self) -> RendererResult<Vec<TileKey>> {
        let device = self.device.as_mut().ok_or("Vulkan device is not initialized")?;
        let virtual_texture = device.virtual_texture.as_mut().ok_or("VT resources are not initialized")?;
        unsafe {
            Ok(virtual_texture
                .poll_upload(&device.device)?
                .map(|uploads| uploads.iter().map(|upload| upload.key).collect())
                .unwrap_or_default())
        }
    }

    /// Configure the day-colour VT layer before the first output is created.
    /// The descriptor and GPU resources are still created when no layer is
    /// supplied, keeping the Earth textured path available with fallbacks.
    pub fn initialize_day_color_vt(&mut self, layer: LayerDescriptor, slot_budget: u32) -> RendererResult<()> {
        if layer.channel != TextureChannel::DayColor || layer.format != PixelFormat::Bc7 {
            return Err("day-colour VT requires a BC7 layer".into());
        }
        if slot_budget == 0 {
            return Err("day-colour VT slot budget must be non-zero".into());
        }
        if self.device.is_some() {
            return Err("day-colour VT must be configured before Vulkan device creation".into());
        }
        self.vt_config = Some((layer, slot_budget));
        Ok(())
    }

    pub fn set_viewport(&mut self, output_id: u32, viewport: LogicalRect) {
        if let Some(target) = self.outputs.get_mut(&output_id) {
            target.viewport = viewport;
        }
    }

    pub fn destroy_output(&mut self, output_id: u32) {
        let Some(device) = self.device.as_ref() else {
            return;
        };
        let Some(target) = self.outputs.remove(&output_id) else {
            return;
        };
        unsafe {
            let _ = device.device.device_wait_idle();
            let surface = target.surface;
            target.destroy(
                &device.device,
                device.command_pool,
                &device.swapchain_loader,
            );
            self.surface_loader.destroy_surface(surface, None);
        }
    }

    /// Release every WSI surface before the Wayland connection goes away.
    ///
    /// `calloop` owns that connection, so application shutdown must call this
    /// while its event loop is still in scope. Doing it from `Drop` is too late.
    pub fn destroy_outputs(&mut self) {
        let output_ids: Vec<_> = self.outputs.keys().copied().collect();
        for output_id in output_ids {
            self.destroy_output(output_id);
        }
    }

    pub fn reload_weather(&mut self) -> RendererResult<()> {
        if !self.nasa_materials {
            return Err("weather requires the NASA data material".into());
        }
        let state = crate::weather::WeatherState::load()?.ok_or("weather manifest is not configured")?;
        if self.weather.as_ref().map_or(true, |old| old.sha256 != state.sha256) {
            self.pending_weather = Some(state);
        } else {
            self.weather = Some(state);
        }
        Ok(())
    }

    fn apply_weather(&mut self) -> RendererResult<()> {
        let Some(state) = self.pending_weather.as_ref() else { return Ok(()); };
        let Some(device) = self.device.as_mut() else { return Ok(()); };
        // Descriptor updates cannot race any output's submitted command buffer.
        for target in self.outputs.values() {
            if !unsafe { device.device.get_fence_status(target.in_flight)? } { return Ok(()); }
        }
        let Some(textures) = device.day_color_textures.as_mut() else { return Ok(()); };
        let source = crate::day_color::load_preview_texture("EARTH_NATIVE_WEATHER_MANIFEST",
            state.texture.clone().into_os_string(), crate::day_color::PreviewColorSpace::Linear)?;
        let mapped = source.map_payload()?;
        let replacement = PinnedBgraTexture::create_from_preview(&device.device, &textures.resources, LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, &source, mapped.bytes(), "NOAA weather and aurora")?;
        let info = [vk::DescriptorImageInfo::default().sampler(replacement.sampler)
            .image_view(replacement.view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = [vk::WriteDescriptorSet::default().dst_set(textures.descriptor_set).dst_binding(11)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&info)];
        unsafe {
            device.device.update_descriptor_sets(&write, &[]);
            if let Some(previous) = textures.weather_fields.take() {
                previous.destroy(&device.device);
            }
        }
        textures.weather_fields = Some(replacement);
        // Observed clouds (binding 12, in the slot of the authored cloud
        // height map, unused by the NASA material).
        let clouds = state.clouds_texture.clone();
        self.live_clouds_bound = false;
        if let Some(path) = clouds {
            let upload = crate::day_color::load_preview_texture("EARTH_NATIVE_WEATHER_MANIFEST", path.into_os_string(),
                    crate::day_color::PreviewColorSpace::Linear)
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() })
                .and_then(|source| {
                    let mapped = source.map_payload()?;
                    PinnedBgraTexture::create_from_preview(&device.device, &textures.resources, LINEAR_PREVIEW_FORMAT,
                        vk::SamplerAddressMode::CLAMP_TO_EDGE, &source, mapped.bytes(), "GMGSI live clouds")
                });
            match upload {
                Ok(texture) => {
                    let info = [vk::DescriptorImageInfo::default().sampler(texture.sampler)
                        .image_view(texture.view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                    let write = [vk::WriteDescriptorSet::default().dst_set(textures.descriptor_set).dst_binding(12)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&info)];
                    unsafe {
                        device.device.update_descriptor_sets(&write, &[]);
                        if let Some(previous) = textures.cloud_height.take() {
                            previous.destroy(&device.device);
                        }
                    }
                    textures.cloud_height = Some(texture);
                    self.live_clouds_bound = true;
                }
                Err(error) => eprintln!("earth-native: live clouds unavailable: {error}"),
            }
        }
        self.weather = self.pending_weather.take();
        Ok(())
    }

    /// EXP-001 kill switch: residency eviction runs only with
    /// `EARTH_NATIVE_BODY_RESIDENCY=1` (or `true`). A routine rebuild without
    /// the variable restores exact baseline residency behavior even though
    /// the eviction code stays compiled in.
    fn body_residency_enabled() -> bool {
        Self::body_residency_flag(std::env::var_os("EARTH_NATIVE_BODY_RESIDENCY"))
    }

    fn body_residency_flag(value: Option<std::ffi::OsString>) -> bool {
        value
            .and_then(|value| value.into_string().ok())
            .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    }

    /// Evict the Earth preview stack and Moon albedo while a non-Earth body
    /// is selected; restore them before Earth presents again. Runs ahead of
    /// `apply_weather` so a restored stack re-uploads live weather in the
    /// same frame. Every transition no-ops when nothing changed, and the
    /// shared descriptor sets stay valid throughout (1x1 fallbacks).
    fn sync_body_residency(&mut self) -> RendererResult<()> {
        if self.current_body == crate::body::Body::Earth {
            let Some(device_state) = self.device.as_mut() else { return Ok(()); };
            let Some(textures) = device_state.day_color_textures.as_mut() else { return Ok(()); };
            if !textures.earth_resident {
                let Some(hemispheres) = self.day_color.as_ref() else { return Ok(()); };
                textures.restore_earth_stack(
                    &device_state.device,
                    hemispheres,
                    self.surface_normals.as_ref(),
                    self.night_emission.as_ref(),
                    self.cloud_previews.as_ref(),
                    self.desert_cloud_mask.as_ref(),
                    self.height_preview.as_ref(),
                )?;
                // The evicted stack dropped the live weather upload; re-queue
                // it so `apply_weather` restores binding 11 this same frame.
                if self.weather.is_some() && self.pending_weather.is_none() {
                    self.pending_weather = self.weather.clone();
                }
            }
        } else if self.day_color.is_some() {
            let Some(device_state) = self.device.as_mut() else { return Ok(()); };
            if let Some(textures) = device_state.day_color_textures.as_mut() {
                textures.evict_earth_stack(&device_state.device)?;
            }
        }
        let Some(device_state) = self.device.as_mut() else { return Ok(()); };
        if let Some(star_texture) = device_state.star_texture.as_mut() {
            star_texture.sync_moon(
                &device_state.device,
                self.current_body,
                self.moon_preview.as_ref(),
            )?;
        }
        Ok(())
    }

    pub fn render(&mut self, mut uniforms: FrameUniforms) -> RendererResult<bool> {
        if Self::body_residency_enabled() {
            self.sync_body_residency()?;
        }
        if let Err(error) = self.apply_weather() {
            eprintln!("weather upload failed; retaining previous fields: {error}");
            self.pending_weather = None;
        }
        uniforms.nasa_materials = self.nasa_materials;
        uniforms.nasa_clouds = self.nasa_clouds;
        if let Some(weather) = &self.weather {
            uniforms.weather_valid_unix_utc = weather.valid_unix_utc;
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
            if (now - weather.aurora_unix_utc).abs() < 6 * 3600 {
                uniforms.aurora_valid_unix_utc = weather.aurora_unix_utc;
            }
            // Off-Earth eviction drops the texture; the flag follows it.
            let resident = self.device.as_ref()
                .and_then(|device| device.day_color_textures.as_ref())
                .is_some_and(|textures| textures.earth_resident);
            uniforms.live_clouds = self.live_clouds_bound && resident && weather.live_clouds(now).is_some();
            uniforms.live_aerosol = uniforms.live_clouds && weather.live_aerosol(now);
            uniforms.live_sea_ice = uniforms.live_clouds && weather.live_sea_ice(now);
        }
        self.process_virtual_texture(uniforms)?;
        // Fast path: outputs almost never need recreation, so skip the
        // per-frame heap allocation unless one actually does.
        if self.outputs.values().any(|target| target.needs_recreate) {
            let recreations: Vec<_> = self
                .outputs
                .iter()
                .filter_map(|(output_id, target)| {
                    target
                        .needs_recreate
                        .then_some((*output_id, target.extent, target.viewport))
                })
                .collect();
            for (output_id, extent, viewport) in recreations {
                self.recreate_output(output_id, extent, viewport)?;
            }
        }
        // Swap the resident planet texture when the body changed: only the
        // selected body's albedo (and Saturn's ring) stays in VRAM.
        if let Some(device_state) = self.device.as_mut() {
            if let Some(star_texture) = device_state.star_texture.as_mut() {
                star_texture.planets.swap(
                    &device_state.device,
                    star_texture.descriptor_set,
                    self.current_body,
                    &self.planet_previews,
                    self.saturn_ring_preview.as_ref(),
                )?;
            }
        }
        let Some(device) = self.device.as_ref() else {
            return Ok(false);
        };
        let Some(pipeline) = device.pipeline.as_ref() else {
            return Ok(false);
        };
        let mut needs_redraw = self.pending_weather.is_some()
            || !self.vt_pending_jobs.is_empty()
            || self.vt_streamer.as_ref().is_some_and(VirtualTextureStreamer::busy)
            || !self.static_pending_jobs.is_empty()
            || self.static_streamer.as_ref().is_some_and(VirtualTextureStreamer::busy);
        // Integer body id carried in `sun_direction.w` (0=Earth, 1=Jupiter,
        // 2=Mercury, 3=Mars, 4=Saturn) so the shared push-constant layout
        // stays identical across every pipeline.
        let body_selector = self.current_body.id();
        // Meter from the largest output that has a reading.
        let reading = self
            .outputs
            .values()
            .filter(|target| target.last_meter.is_some())
            .max_by_key(|target| target.extent.width as u64 * target.extent.height as u64)
            .and_then(|target| target.last_meter);
        let earth = self.current_body == crate::body::Body::Earth;
        let sun_cap = if earth { sun_exposure_cap(uniforms, body_selector) } else { None };
        self.exposure.update(if earth { reading } else { None }, sun_cap, Instant::now());
        needs_redraw |= earth && self.exposure.converging();
        let camera = camera_settings(uniforms, body_selector, &self.exposure, earth);
        for target in self.outputs.values_mut() {
            if target.format != pipeline.color_format {
                continue;
            }
            target.render(device, pipeline, uniforms, body_selector, &camera)?;
            needs_redraw |= target.needs_recreate
                || target
                    .debug_capture
                    .as_ref()
                    .is_some_and(|capture| capture.pending);
        }
        Ok(needs_redraw)
    }

    /// Jump cuts (camera or time set over IPC) re-meter instead of easing,
    /// so a capture right after the cut is correctly exposed.
    pub fn snap_exposure(&mut self) {
        self.exposure.snap_frames = 8;
    }

    pub fn exposure_ev(&self) -> f32 {
        self.exposure.ev
    }


    pub fn set_debug_scene(&mut self, scene_name: &str) {
        if !self.debug_screenshots || self.debug_scene_name == scene_name {
            return;
        }
        self.debug_scene_name = scene_name.to_owned();
        for target in self.outputs.values_mut() {
            if let Some(capture) = target.debug_capture.as_mut() {
                if capture.periodic {
                    capture.set_scene(scene_name);
                } else if capture.resume_scene.is_some() {
                    capture.resume_scene = Some(scene_name.to_owned());
                }
            }
        }
    }

    fn process_virtual_texture(&mut self, uniforms: FrameUniforms) -> RendererResult<()> {
        let Self { device, vt_streamer, vt_pending_jobs, static_streamer, static_pending_jobs, vt_frame, .. } = self;
        let Some(device) = device.as_mut() else {
            return Ok(());
        };
        let frame = *vt_frame;
        let pairs = [
            (vt_streamer.as_mut(), device.virtual_texture.as_mut(), vt_pending_jobs),
            (static_streamer.as_mut(), device.static_texture.as_mut(), static_pending_jobs),
        ];
        let mut day_layer = None;
        let mut static_layer = None;
        for (streamer, texture, pending) in pairs {
            let (Some(streamer), Some(texture)) = (streamer, texture) else { continue };
            if texture.layer.is_none() {
                continue;
            }
            streamer.dispatch_feedback(frame)?;
            pump_vt_uploads(streamer, texture, pending, &device.device, device.queue, frame)?;
            match texture.kind {
                VtKind::Day => day_layer = texture.layer,
                VtKind::Static => static_layer = texture.layer,
            }
        }
        if day_layer.is_none() && static_layer.is_none() {
            return Ok(());
        }

        // The visible tile set changes slowly (from the ISS the camera moves
        // ~0.06 % of its altitude per frame), so the CPU feedback, ~100k ray
        // casts and the largest share of the renderer's CPU, runs every
        // `VT_FEEDBACK_INTERVAL` frames, and at once when the view jumps.
        let view = [uniforms.camera_position[0], uniforms.camera_position[1], uniforms.camera_position[2],
            uniforms.forward[0], uniforms.forward[1], uniforms.forward[2], uniforms.tan_half_fov_x];
        let jumped = self.vt_feedback_view.is_none_or(|last| {
            let distance = |v: &[f32]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let moved = distance(&[view[0] - last[0], view[1] - last[1], view[2] - last[2]]);
            let altitude = (distance(&view[..3]) - crate::camera::SCENE_EARTH_RADIUS).max(1.0e-3);
            let turned = view[3] * last[3] + view[4] * last[4] + view[5] * last[5];
            moved > 0.01 * altitude || turned < 0.99996 || (view[6] / last[6] - 1.0).abs() > 0.01
        });
        if !jumped && self.vt_frame.wrapping_sub(self.vt_feedback_frame) < VT_FEEDBACK_INTERVAL {
            self.vt_frame = self.vt_frame.wrapping_add(1);
            return Ok(());
        }
        self.vt_feedback_frame = self.vt_frame;
        self.vt_feedback_view = Some(view);
        let outputs = self
            .outputs
            .iter()
            .map(|(output_id, target)| OutputDescriptor {
                output_id: *output_id,
                viewport: target.viewport,
                physical_extent: [target.extent.width, target.extent.height],
            })
            .collect::<Vec<_>>();
        // One pass over the rays serves both textures. The static layer gets
        // a distinct feedback id (its file's night layer shares id 1 with
        // the day layer) and is mapped back below.
        const STATIC_FEEDBACK_ID: u16 = 0x8001;
        let mut layers = Vec::with_capacity(2);
        layers.extend(day_layer);
        layers.extend(static_layer.map(|layer| LayerDescriptor { id: STATIC_FEEDBACK_ID, ..layer }));
        let requests = self
            .vt_feedback
            .collect_requests(uniforms, &outputs, &layers, self.vt_frame);
        let (static_requests, day_requests): (Vec<_>, Vec<_>) =
            requests.into_iter().partition(|request| request.key.layer == STATIC_FEEDBACK_ID);
        let Some(device) = self.device.as_mut() else { return Ok(()) };
        if let (Some(streamer), Some(texture)) = (self.vt_streamer.as_mut(), device.virtual_texture.as_mut()) {
            if std::env::var_os("EARTH_NATIVE_VT_DEBUG").is_some() && (self.vt_frame % 60 == 0 || self.vt_frame < 5) {
                let mut mips = std::collections::BTreeMap::<u16, usize>::new();
                for request in &day_requests { *mips.entry(request.key.mip).or_default() += 1; }
                eprintln!("vt-debug frame={} requests={} by_mip={:?} resident_bytes={} pending_jobs={}",
                    self.vt_frame, day_requests.len(), mips, streamer.residency().resident_bytes(), self.vt_pending_jobs.len());
            }
            if day_layer.is_some() {
                self.vt_wanted_tiles.0 = day_requests.len();
                let visible: std::collections::HashSet<TileKey> = day_requests.iter().map(|request| request.key).collect();
                texture.allocator.touch_visible(&visible);
                streamer.submit_feedback(self.vt_frame, day_requests, false);
            }
        }
        if let (Some(streamer), Some(texture), Some(layer)) = (self.static_streamer.as_mut(), device.static_texture.as_mut(), static_layer) {
            // Only the streamed levels, each with the next coarser one for the
            // shader's blend between the two nearest levels.
            let mut wanted = std::collections::BTreeMap::<TileKey, TileRequest>::new();
            for request in static_requests.into_iter().filter(|request| request.key.mip < STATIC_VT_MIPS) {
                let key = TileKey::new(layer.id, request.key.mip, request.key.x, request.key.y);
                let parent = (key.mip + 1 < STATIC_VT_MIPS).then(|| TileKey::new(layer.id, key.mip + 1, key.x / 2, key.y / 2));
                for key in std::iter::once(key).chain(parent) {
                    let entry = wanted.entry(key).or_insert(TileRequest::visible(key, 0, request.requested_frame));
                    entry.priority = entry.priority.saturating_add(request.priority);
                }
            }
            let mut static_requests: Vec<_> = wanted.into_values().collect();
            static_requests.sort_by(|left, right| right.priority.cmp(&left.priority).then_with(|| left.key.cmp(&right.key)));
            self.vt_wanted_tiles.1 = static_requests.len();
            let visible: std::collections::HashSet<TileKey> = static_requests.iter().map(|request| request.key).collect();
            texture.allocator.touch_visible(&visible);
            streamer.submit_feedback(self.vt_frame, static_requests, false);
        }
        self.vt_frame = self.vt_frame.wrapping_add(1);
        Ok(())
    }

    fn ensure_device(&mut self, surface: vk::SurfaceKHR) -> RendererResult<()> {
        if let Some(device) = self.device.as_ref() {
            let supported = unsafe {
                self.surface_loader.get_physical_device_surface_support(
                    device.physical_device,
                    device.queue_family,
                    surface,
                )?
            };
            if !supported {
                return Err("the selected GPU queue cannot present to this Wayland output".into());
            }
            return Ok(());
        }

        let physical_devices = unsafe { self.instance.enumerate_physical_devices()? };
        let selected = physical_devices.into_iter().find_map(|physical_device| {
            let queue_families = unsafe {
                self.instance
                    .get_physical_device_queue_family_properties(physical_device)
            };
            queue_families
                .iter()
                .enumerate()
                .find_map(|(index, family)| {
                    if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        return None;
                    }
                    let present_supported = unsafe {
                        self.surface_loader
                            .get_physical_device_surface_support(
                                physical_device,
                                index as u32,
                                surface,
                            )
                            .ok()
                    };
                    present_supported
                        .filter(|supported| *supported)
                        .map(|_| (physical_device, index as u32))
                })
        });
        let Some((physical_device, queue_family)) = selected else {
            return Err("no Vulkan graphics queue can present to the Wayland surface".into());
        };

        let queue_priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities)];
        let extensions = [vk::KHR_SWAPCHAIN_NAME.as_ptr()];
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        // BC star/NASA previews require this core feature; every desktop GPU has it.
        let supported = unsafe { self.instance.get_physical_device_features(physical_device) };
        let base_features = vk::PhysicalDeviceFeatures::default()
            .texture_compression_bc(supported.texture_compression_bc == vk::TRUE)
            .sampler_anisotropy(supported.sampler_anisotropy == vk::TRUE)
            .dual_src_blend(true);
        if supported.dual_src_blend != vk::TRUE {
            return Err("the GPU lacks dual-source blending (needed for atmospheric transmittance)".into());
        }
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&extensions)
            .enabled_features(&base_features)
            .push_next(&mut features13);
        let device = unsafe {
            self.instance
                .create_device(physical_device, &create_info, None)?
        };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let memory_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(physical_device)
        };
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };
        let timestamp_pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(TIMESTAMP_OUTPUT_SLOTS * TIMESTAMP_QUERIES_PER_OUTPUT);
        let timestamp_pool = unsafe { device.create_query_pool(&timestamp_pool_info, None)? };
        let timestamp_period_ns = unsafe {
            self.instance
                .get_physical_device_properties(physical_device)
                .limits
                .timestamp_period
        };
        let swapchain_loader = ash::khr::swapchain::Device::new(&self.instance, &device);
        self.device = Some(DeviceState {
            physical_device,
            memory_properties,
            queue_family,
            device,
            queue,
            command_pool,
            timestamp_pool,
            timestamp_period_ns,
            swapchain_loader,
            pipeline: None,
            star_texture: None,
            day_color_textures: None,
            virtual_texture: None,
            static_texture: None,
            sky_luts: Vec::new(),
        });
        Ok(())
    }

    fn ensure_pipeline(&mut self, color_format: vk::Format) -> RendererResult<()> {
        let (
            memory_properties,
            max_star_image_dimension,
            max_push_constants_size,
            srgb_format_features,
            bc1_format_features,
            bc3_format_features,
            bc4_format_features,
            linear_format_features,
            max_anisotropy,
        ) = {
            let device = self
                .device
                .as_ref()
                .expect("device initialized before pipeline");
            let physical_device = device.physical_device;
            let memory_properties = unsafe {
                self.instance
                    .get_physical_device_memory_properties(physical_device)
            };
            let properties = unsafe {
                self.instance
                    .get_physical_device_properties(physical_device)
            };
            let srgb_format_properties = unsafe {
                self.instance
                    .get_physical_device_format_properties(physical_device, STAR_PANORAMA_FORMAT)
            };
            let linear_format_properties = unsafe {
                self.instance
                    .get_physical_device_format_properties(physical_device, LINEAR_PREVIEW_FORMAT)
            };
            let format_features = |format| unsafe {
                self.instance
                    .get_physical_device_format_properties(physical_device, format)
                    .optimal_tiling_features
            };
            let bc1_format_properties = unsafe {
                self.instance.get_physical_device_format_properties(
                    physical_device,
                    STAR_PANORAMA_BC1_FORMAT,
                )
            };
            (
                memory_properties,
                properties.limits.max_image_dimension2_d,
                properties.limits.max_push_constants_size,
                srgb_format_properties.optimal_tiling_features,
                bc1_format_properties.optimal_tiling_features,
                format_features(PREVIEW_BC3_SRGB_FORMAT),
                format_features(PREVIEW_BC4_FORMAT),
                linear_format_properties.optimal_tiling_features,
                {
                    let features = unsafe { self.instance.get_physical_device_features(physical_device) };
                    if features.sampler_anisotropy == vk::TRUE {
                        properties.limits.max_sampler_anisotropy.min(8.0)
                    } else {
                        0.0
                    }
                },
            )
        };
        let device = self
            .device
            .as_mut()
            .expect("device initialized before pipeline");
        if max_push_constants_size < size_of::<ShaderFrame>() as u32 {
            return Err(format!(
                "GPU supports {max_push_constants_size} push-constant bytes; celestial frame requires {}",
                size_of::<ShaderFrame>()
            )
            .into());
        }
        if let Some(pipeline) = device.pipeline.as_ref() {
            if pipeline.color_format == color_format {
                return Ok(());
            }
            return Err("Wayland outputs selected incompatible swapchain color formats".into());
        }
        // Borrow, never take: EXP-001 re-uploads these after an off-Earth
        // eviction, and the descriptors hold only paths and dimensions.
        let star_panorama = self.star_panorama.as_ref();
        let day_color = self.day_color.as_ref();
        let surface_normals = self.surface_normals.as_ref();
        let night_emission = self.night_emission.as_ref();
        let cloud_previews = self.cloud_previews.as_ref();
        let tiling_noise = self.tiling_noise.as_ref();
        let desert_cloud_mask = self.desert_cloud_mask.as_ref();
        let height_preview = self.height_preview.as_ref();
        let moon_preview = self.moon_preview.as_ref();
        let planet_previews = &self.planet_previews;
        let saturn_ring_preview = self.saturn_ring_preview.as_ref();
        let pipeline = Pipeline::create(&device.device, color_format)?;
        let star_resources = StarTextureResources {
            memory_properties,
            max_image_dimension: max_star_image_dimension,
            srgb_format_features,
            bc1_format_features,
            bc3_format_features,
            bc4_format_features,
            linear_format_features,
            max_anisotropy,
            queue: device.queue,
            command_pool: device.command_pool,
            descriptor_pool: pipeline.star_descriptor_pool,
            descriptor_set_layout: pipeline.star_descriptor_set_layout,
        };
        let star_texture = if uses_textured_celestial_pass(
            star_panorama.is_some(),
            moon_preview.is_some(),
        ) || planet_previews.any()
        {
            match PinnedStarTexture::create(
                &device.device,
                &star_resources,
                star_panorama,
                moon_preview,
                self.current_body,
                planet_previews,
                saturn_ring_preview,
            ) {
                Ok(texture) => Some(texture),
                Err(error) => {
                    unsafe { pipeline.destroy(&device.device) };
                    return Err(error);
                }
            }
        } else {
            None
        };
        let day_color_textures = match day_color {
            Some(hemispheres) => {
                match PinnedDayColorTextures::create(
                    &device.device,
                    &star_resources,
                    hemispheres,
                    surface_normals,
                    night_emission,
                    cloud_previews,
                    tiling_noise,
                    desert_cloud_mask,
                    height_preview,
                ) {
                    Ok(textures) => Some(textures),
                    Err(error) => {
                        unsafe {
                            if let Some(texture) = star_texture.as_ref() {
                                texture.destroy(&device.device);
                            }
                            pipeline.destroy(&device.device);
                        }
                        return Err(error);
                    }
                }
            }
            None => Some(PinnedDayColorTextures::create_fallback(&device.device, &star_resources)?),
        };
        let vt = VirtualTexture::create_disabled(
            &device.device,
            &self.instance,
            device.physical_device,
            device.queue,
            device.command_pool,
            pipeline.vt_descriptor_pool,
            pipeline.vt_descriptor_set_layout,
            memory_properties,
            self.vt_config,
            VtKind::Day,
            None,
        )?;
        // Always created (1x1 and disabled without the file): the shader
        // statically uses bindings 3-7 of the day VT's set.
        let mut vt = vt;
        let static_texture = match VirtualTexture::create_disabled(
            &device.device,
            &self.instance,
            device.physical_device,
            device.queue,
            device.command_pool,
            pipeline.vt_descriptor_pool,
            pipeline.vt_descriptor_set_layout,
            memory_properties,
            self.static_vt_config,
            VtKind::Static,
            Some(vt.descriptor_set),
        ) {
            Ok(texture) => texture,
            Err(error) => {
                unsafe { vt.destroy(&device.device, device.command_pool) };
                return Err(error);
            }
        };
        let sky_luts = match upload_sky_luts(&device.device, &star_resources) {
            Ok(luts) => luts,
            Err(error) => {
                unsafe {
                    if let Some(textures) = day_color_textures.as_ref() {
                        textures.destroy(&device.device);
                    }
                    if let Some(texture) = star_texture.as_ref() {
                        texture.destroy(&device.device);
                    }
                    pipeline.destroy(&device.device);
                }
                return Err(error);
            }
        };
        if let Some(textures) = day_color_textures.as_ref() {
            let infos: Vec<_> = sky_luts.iter().map(|texture| vk::DescriptorImageInfo::default()
                .sampler(texture.sampler)
                .image_view(texture.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)).collect();
            let writes: Vec<_> = infos.iter().enumerate().map(|(index, info)| vk::WriteDescriptorSet::default()
                .dst_set(textures.descriptor_set)
                .dst_binding(17 + index as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(info))).collect();
            unsafe { device.device.update_descriptor_sets(&writes, &[]) };
        }
        device.sky_luts = sky_luts;
        device.star_texture = star_texture;
        device.day_color_textures = day_color_textures;
        device.virtual_texture = Some(vt);
        device.static_texture = Some(static_texture);
        device.pipeline = Some(pipeline);
        Ok(())
    }

    fn recreate_output(
        &mut self,
        output_id: u32,
        extent: vk::Extent2D,
        viewport: LogicalRect,
    ) -> RendererResult<()> {
        let Some(device) = self.device.as_ref() else {
            return Ok(());
        };
        let Some(target) = self.outputs.get_mut(&output_id) else {
            return Ok(());
        };
        if target.extent == extent && !target.needs_recreate {
            target.viewport = viewport;
            return Ok(());
        }
        unsafe { device.device.device_wait_idle()? };
        let surface = target.surface;
        let old_swapchain = target.swapchain;
        let debug_output_name = target
            .debug_capture
            .as_ref()
            .map(|capture| capture.monitor_name.clone());
        let debug_scene_name = self.debug_scene_name.clone();
        let mut replacement = OutputTarget::create_with_old(
            &self.surface_loader,
            device,
            surface,
            extent,
            viewport,
            debug_output_name.as_deref(),
            &debug_scene_name,
            old_swapchain,
        )?;
        replacement.output_name = target.output_name.clone();
        if let (Some(old), Some(new)) = (&target.debug_capture, &mut replacement.debug_capture) {
            new.periodic = old.periodic;
            new.output_name = old.output_name.clone();
            new.resume_scene = old.resume_scene.clone();
        }
        if replacement.format != target.format {
            return Err("swapchain recreation changed to an incompatible color format".into());
        }
        let old_target = std::mem::replace(target, replacement);
        target.query_base =
            (output_id % TIMESTAMP_OUTPUT_SLOTS) * TIMESTAMP_QUERIES_PER_OUTPUT;
        unsafe {
            old_target.destroy_swapchain_resources(
                &device.device,
                device.command_pool,
                &device.swapchain_loader,
            )
        };
        Ok(())
    }
}

/// Veiling-glare point-spread function of the lens model (post.frag's
/// `source_glare` lobes), per steradian at `angle` radians off the source.
fn lens_glare_psf(angle: f64) -> f64 {
    let near = angle / 0.010;
    let far = angle / 0.15;
    0.004 * 0.6 / (std::f64::consts::PI * 0.010 * 0.010) * (1.0 + near * near).powf(-1.6)
        + 0.0015 / (std::f64::consts::PI * 0.15 * 0.15) * (1.0 + far * far).powf(-2.0)
}

/// Exposure ceiling (EV) set by the Sun. The scene meter reads the HDR
/// target before post.frag adds the lens glare, so the Sun's veiling glare is
/// metered here over a 33 x 15 grid across the whole frame, as a camera's
/// matrix meter would see it: the exposure keeps the frame's mean glare
/// under 8 % and all but 8 % of the frame under half white. A Sun just off
/// the edge then flares that side instead of whiting out the view, and a
/// night exposure cannot open up while the (sunlit) ISS camera faces the
/// Sun. A Sun in frame caps it at log2(1 / visible fraction): "sunny 16"
/// (EV 0) for an open Sun, more for one sinking through the limb.
fn sun_exposure_cap(uniforms: FrameUniforms, body_selector: f32) -> Option<f32> {
    let frame = ShaderFrame::from_uniforms(uniforms, LogicalRect::default(), body_selector);
    let view = frame.celestial_sun_view;
    let sun = [view[0], view[1], view[2]].map(f64::from);
    let dot = |a: [f64; 3], b: [f32; 3]| a[0] * f64::from(b[0]) + a[1] * f64::from(b[1]) + a[2] * f64::from(b[2]);
    let (x, y, z) = (dot(sun, uniforms.right), dot(sun, uniforms.up), dot(sun, uniforms.forward));
    // Same acceptance as post.frag: no glare from > ~75 degrees off axis.
    let t = ((z - 0.26) / (0.64 - 0.26)).clamp(0.0, 1.0);
    let acceptance = t * t * (3.0 - 2.0 * t);
    if acceptance <= 0.0 {
        return None;
    }
    let km_per_unit = crate::sky::GROUND_KM / f64::from(SCENE_EARTH_RADIUS);
    let camera_km = uniforms.camera_position.map(|v| f64::from(v) * km_per_unit);
    let visible = disc_visibility(camera_km, sun, f64::from(view[3]));
    let luminance = 0.2126 * visible[0] + 0.7152 * visible[1] + 0.0722 * visible[2];
    if luminance <= 1.0e-7 {
        return None;
    }
    let (tan_x, tan_y) = (f64::from(uniforms.tan_half_fov_x), f64::from(uniforms.tan_half_fov_y));
    let in_frame = z > 0.0 && (x / z / tan_x).abs() < 1.0 && (y / z / tan_y).abs() < 1.0;
    let mut glare = Vec::with_capacity(33 * 15);
    for row in 0..15 {
        for column in 0..33 {
            let (u, v) = (f64::from(column) / 16.0 - 1.0, f64::from(row) / 7.0 - 1.0);
            let direction = [0, 1, 2].map(|c| f64::from(uniforms.forward[c])
                + f64::from(uniforms.right[c]) * u * tan_x + f64::from(uniforms.up[c]) * v * tan_y);
            let length = direction.iter().map(|d| d * d).sum::<f64>().sqrt();
            let cosine = (direction.iter().zip(sun).map(|(d, s)| d * s).sum::<f64>() / length).clamp(-1.0, 1.0);
            glare.push(std::f64::consts::PI * luminance * acceptance * lens_glare_psf(cosine.acos()));
        }
    }
    glare.sort_by(f64::total_cmp);
    let mean = glare.iter().sum::<f64>() / glare.len() as f64;
    let high = glare[glare.len() * 92 / 100];
    let glare_cap = (0.08 / mean).log2().min((0.5 / high).log2());
    let cap = if in_frame { glare_cap.min((1.0 / luminance).log2()) } else { glare_cap };
    Some(cap as f32)
}

fn camera_settings(uniforms: FrameUniforms, body_selector: f32, exposure: &ExposureController, earth: bool) -> CameraSettings {
    if !earth {
        // Other bodies keep their per-body presentation (earth.frag's own
        // exposure and the legacy filmic curve).
        return CameraSettings {
            preexposure: 1.0,
            post: PostFrame { tone: [1.0, 1.0, 0.0, 0.0], ..PostFrame::default() },
        };
    }
    let preexposure = exposure.ev.exp2();
    let frame = ShaderFrame::from_uniforms(uniforms, LogicalRect::default(), body_selector);
    let to_camera = |d: [f32; 4]| {
        let dot = |b: [f32; 3]| d[0] * b[0] + d[1] * b[1] + d[2] * b[2];
        [dot(uniforms.right), dot(uniforms.up), dot(uniforms.forward), d[3]]
    };
    let km_per_unit = crate::sky::GROUND_KM / f64::from(SCENE_EARTH_RADIUS);
    let camera_km = uniforms.camera_position.map(|v| f64::from(v) * km_per_unit);
    let source = |view: [f32; 4], irradiance: f64| {
        let direction = [0, 1, 2].map(|c| f64::from(view[c]));
        let visible = disc_visibility(camera_km, direction, f64::from(view[3]));
        let scale = f64::from(preexposure) * std::f64::consts::PI * irradiance;
        [visible[0] * scale, visible[1] * scale, visible[2] * scale, 1.0].map(|v| v as f32)
    };
    // Lunar irradiance relative to the Sun: Allen's phase law, mean full
    // Moon 2.5e-6 at 60.27 Earth radii (the shader's moonlight_scale).
    let phase_degrees = f64::from(uniforms.moon_phase_angle_radians).to_degrees();
    let phase_law = 10f64.powf(-0.4 * (0.026 * phase_degrees + 4.0e-9 * phase_degrees.powi(4)));
    let distance_ratio = 60.27 / f64::from(uniforms.moon_distance_earth_radii).max(1.0);
    let moon_irradiance = 2.5e-6 * phase_law * distance_ratio * distance_ratio;
    let solar_au = f64::from(uniforms.sun_distance_earth_radii) * crate::sky::GROUND_KM / 149_597_870.7;
    let sun_irradiance = 1.0 / (solar_au * solar_au).max(1.0e-6);
    let noise = 0.003 + 0.022 * ((exposure.ev - 7.0) / 8.0).clamp(0.0, 1.0);
    CameraSettings {
        preexposure,
        post: PostFrame {
            tone: [exposure.contrast(), 0.0, (exposure.frame % 4096) as f32, noise],
            sun: to_camera(frame.celestial_sun_view),
            sun_light: source(frame.celestial_sun_view, sun_irradiance),
            moon: to_camera(frame.celestial_moon_view),
            // The Moon disc is compressed toward display white like the
            // eye's local adaptation (stars_textured.frag); so is its glare.
            moon_light: source(frame.celestial_moon_view,
                moon_irradiance * (0.35 / (0.12 * f64::from(preexposure) * sun_irradiance)).min(1.0)),
            ..PostFrame::default()
        },
    }
}

/// Bakes the atmosphere tables (sky.rs) and uploads them as half floats.
fn upload_sky_luts(device: &Device, resources: &StarTextureResources) -> RendererResult<Vec<PinnedBgraTexture>> {
    let started = Instant::now();
    let tables = crate::sky::bake();
    let specs = [
        (&tables.transmittance, crate::sky::TRANSMITTANCE_WIDTH, crate::sky::TRANSMITTANCE_HEIGHT, "sky transmittance LUT"),
        (&tables.multiscatter, crate::sky::MULTISCATTER_SIZE, crate::sky::MULTISCATTER_SIZE, "sky multiple-scattering LUT"),
        (&tables.irradiance, crate::sky::IRRADIANCE_WIDTH, crate::sky::IRRADIANCE_HEIGHT, "sky irradiance LUT"),
    ];
    let mut textures = Vec::with_capacity(specs.len());
    for (table, width, height, label) in specs {
        let bytes = crate::sky::rgba16f_bytes(table);
        match PinnedBgraTexture::create(device, resources, SKY_LUT_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE,
            width as u32, height as u32, &bytes, label)
        {
            Ok(texture) => textures.push(texture),
            Err(error) => {
                unsafe { for texture in &textures { texture.destroy(device); } }
                return Err(error);
            }
        }
    }
    eprintln!("earth-native: atmosphere tables baked in {:?}", started.elapsed());
    // Relief normals (binding 20): GEBCO slopes as BC5, or flat.
    let relief = match std::env::var_os("EARTH_NATIVE_RELIEF") {
        Some(path) => crate::day_color::load_preview_texture("EARTH_NATIVE_RELIEF", path, crate::day_color::PreviewColorSpace::Linear)
            .map_err(|error| -> Box<dyn std::error::Error> { error.into() })
            .and_then(|texture| {
                let mapped = texture.map_payload()?;
                PinnedBgraTexture::create_from_preview(device, resources, LINEAR_PREVIEW_FORMAT,
                    vk::SamplerAddressMode::CLAMP_TO_EDGE, &texture, mapped.bytes(), "relief normals")
            }),
        None => Err("unset".into()),
    };
    let relief = match relief {
        Ok(texture) => {
            eprintln!("earth-native: relief normals {}", std::env::var("EARTH_NATIVE_RELIEF").unwrap_or_default());
            texture
        }
        Err(error) => {
            if std::env::var_os("EARTH_NATIVE_RELIEF").is_some() {
                eprintln!("earth-native: relief normals unavailable ({error}); using a flat surface");
            }
            match PinnedBgraTexture::create(device, resources, LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, 1, 1, &[128, 128, 128, 255], "flat relief") {
                Ok(texture) => texture,
                Err(error) => {
                    unsafe { for texture in &textures { texture.destroy(device); } }
                    return Err(error);
                }
            }
        }
    };
    textures.push(relief);
    Ok(textures)
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            if let Some(device) = self.device.as_mut() {
                let _ = device.device.device_wait_idle();
                for (_, target) in self.outputs.drain() {
                    let surface = target.surface;
                    target.destroy(
                        &device.device,
                        device.command_pool,
                        &device.swapchain_loader,
                    );
                    self.surface_loader.destroy_surface(surface, None);
                }
                if let Some(textures) = device.day_color_textures.as_ref() {
                    textures.destroy(&device.device);
                }
                for texture in device.sky_luts.drain(..) {
                    texture.destroy(&device.device);
                }
                if let Some(mut virtual_texture) = device.virtual_texture.take() {
                    virtual_texture.destroy(&device.device, device.command_pool);
                }
                if let Some(mut static_texture) = device.static_texture.take() {
                    static_texture.destroy(&device.device, device.command_pool);
                }
                if let Some(texture) = device.star_texture.as_ref() {
                    texture.destroy(&device.device);
                }
                if let Some(pipeline) = device.pipeline.as_ref() {
                    pipeline.destroy(&device.device);
                }
                device
                    .device
                    .destroy_command_pool(device.command_pool, None);
                device
                    .device
                    .destroy_query_pool(device.timestamp_pool, None);
                device.device.destroy_device(None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

impl Pipeline {
    fn create(device: &Device, color_format: vk::Format) -> RendererResult<Self> {
        let vertex_code = read_spv(&mut Cursor::new(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/earth.vert.spv"
        ))))?;
        let stars_code = read_spv(&mut Cursor::new(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/stars.frag.spv"
        ))))?;
        let stars_textured_code = read_spv(&mut Cursor::new(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/stars_textured.frag.spv"
        ))))?;
        let earth_code = read_spv(&mut Cursor::new(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/earth.frag.spv"
        ))))?;
        let earth_textured_code = read_spv(&mut Cursor::new(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/earth_textured.frag.spv"
        ))))?;
        let vertex_module_info = vk::ShaderModuleCreateInfo::default().code(&vertex_code);
        let stars_procedural_module_info = vk::ShaderModuleCreateInfo::default().code(&stars_code);
        let stars_textured_module_info =
            vk::ShaderModuleCreateInfo::default().code(&stars_textured_code);
        let earth_module_info = vk::ShaderModuleCreateInfo::default().code(&earth_code);
        let earth_textured_module_info =
            vk::ShaderModuleCreateInfo::default().code(&earth_textured_code);
        let vertex_module = unsafe { device.create_shader_module(&vertex_module_info, None)? };
        let stars_procedural_module =
            match unsafe { device.create_shader_module(&stars_procedural_module_info, None) } {
                Ok(module) => module,
                Err(error) => {
                    unsafe { device.destroy_shader_module(vertex_module, None) };
                    return Err(error.into());
                }
            };
        let stars_textured_module =
            match unsafe { device.create_shader_module(&stars_textured_module_info, None) } {
                Ok(module) => module,
                Err(error) => {
                    unsafe {
                        device.destroy_shader_module(vertex_module, None);
                        device.destroy_shader_module(stars_procedural_module, None);
                    }
                    return Err(error.into());
                }
            };
        let earth_module = match unsafe { device.create_shader_module(&earth_module_info, None) } {
            Ok(module) => module,
            Err(error) => {
                unsafe {
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_shader_module(stars_procedural_module, None);
                    device.destroy_shader_module(stars_textured_module, None);
                }
                return Err(error.into());
            }
        };
        let earth_textured_module =
            match unsafe { device.create_shader_module(&earth_textured_module_info, None) } {
                Ok(module) => module,
                Err(error) => {
                    unsafe {
                        device.destroy_shader_module(vertex_module, None);
                        device.destroy_shader_module(stars_procedural_module, None);
                        device.destroy_shader_module(stars_textured_module, None);
                        device.destroy_shader_module(earth_module, None);
                    }
                    return Err(error.into());
                }
            };
        let descriptor_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(8)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(9)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(10)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(11)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(12)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(13)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(14)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(15)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),            // Hipparcos star catalogue (star set only, stars_points.vert).
            vk::DescriptorSetLayoutBinding::default()
                .binding(16)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            // Atmosphere tables (Earth set only): transmittance, multiple
            // scattering, sky irradiance.
            vk::DescriptorSetLayoutBinding::default()
                .binding(17)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(18)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(19)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // GEBCO relief normals (Earth set only).
            vk::DescriptorSetLayoutBinding::default()
                .binding(20)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let descriptor_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
        let star_descriptor_set_layout =
            match unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) } {
                Ok(layout) => layout,
                Err(error) => {
                    unsafe {
                        device.destroy_shader_module(vertex_module, None);
                        device.destroy_shader_module(stars_procedural_module, None);
                        device.destroy_shader_module(stars_textured_module, None);
                        device.destroy_shader_module(earth_module, None);
                        device.destroy_shader_module(earth_textured_module, None);
                    }
                    return Err(error.into());
                }
            };
        let descriptor_pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(42)];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&descriptor_pool_sizes);
        let star_descriptor_pool =
            match unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) } {
                Ok(pool) => pool,
                Err(error) => {
                    unsafe {
                        device.destroy_shader_module(vertex_module, None);
                        device.destroy_shader_module(stars_procedural_module, None);
                        device.destroy_shader_module(stars_textured_module, None);
                        device.destroy_shader_module(earth_module, None);
                        device.destroy_shader_module(earth_textured_module, None);
                        device.destroy_descriptor_set_layout(star_descriptor_set_layout, None);
                    }
                    return Err(error.into());
                }
            };
        let vt_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            // Static VT (VtKind::Static): page table, parameters, night,
            // cloud and relief atlases.
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let vt_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&vt_bindings);
        let vt_descriptor_set_layout = match unsafe {
            device.create_descriptor_set_layout(&vt_layout_info, None)
        } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe {
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_shader_module(stars_procedural_module, None);
                    device.destroy_shader_module(stars_textured_module, None);
                    device.destroy_shader_module(earth_module, None);
                    device.destroy_shader_module(earth_textured_module, None);
                    device.destroy_descriptor_pool(star_descriptor_pool, None);
                    device.destroy_descriptor_set_layout(star_descriptor_set_layout, None);
                }
                return Err(error.into());
            }
        };
        let vt_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(6),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(2),
        ];
        let vt_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&vt_pool_sizes);
        let vt_descriptor_pool = match unsafe { device.create_descriptor_pool(&vt_pool_info, None) } {
            Ok(pool) => pool,
            Err(error) => {
                unsafe {
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_shader_module(stars_procedural_module, None);
                    device.destroy_shader_module(stars_textured_module, None);
                    device.destroy_shader_module(earth_module, None);
                    device.destroy_shader_module(earth_textured_module, None);
                    device.destroy_descriptor_set_layout(vt_descriptor_set_layout, None);
                    device.destroy_descriptor_pool(star_descriptor_pool, None);
                    device.destroy_descriptor_set_layout(star_descriptor_set_layout, None);
                }
                return Err(error.into());
            }
        };
        let push_constants = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<ShaderFrame>() as u32)];
        let set_layouts = [star_descriptor_set_layout, vt_descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constants);
        let layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe {
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_shader_module(stars_procedural_module, None);
                    device.destroy_shader_module(stars_textured_module, None);
                    device.destroy_shader_module(earth_module, None);
                    device.destroy_shader_module(earth_textured_module, None);
                    device.destroy_descriptor_pool(vt_descriptor_pool, None);
                    device.destroy_descriptor_set_layout(vt_descriptor_set_layout, None);
                    device.destroy_descriptor_pool(star_descriptor_pool, None);
                    device.destroy_descriptor_set_layout(star_descriptor_set_layout, None);
                }
                return Err(error.into());
            }
        };
        let stars_procedural = create_graphics_pipeline(
            device,
            vertex_module,
            stars_procedural_module,
            layout,
            HDR_FORMAT,
            Blend::Opaque,
        );
        let stars_procedural = match stars_procedural {
            Ok(pipeline) => pipeline,
            Err(error) => {
                unsafe {
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_shader_module(stars_procedural_module, None);
                    device.destroy_shader_module(stars_textured_module, None);
                    device.destroy_shader_module(earth_module, None);
                    device.destroy_shader_module(earth_textured_module, None);
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_pool(star_descriptor_pool, None);
                    device.destroy_descriptor_set_layout(star_descriptor_set_layout, None);
                }
                return Err(error);
            }
        };
        let stars_textured = create_graphics_pipeline(
            device,
            vertex_module,
            stars_textured_module,
            layout,
            HDR_FORMAT,
            Blend::Opaque,
        );
        let stars_textured = match stars_textured {
            Ok(pipeline) => pipeline,
            Err(error) => {
                unsafe {
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_shader_module(stars_procedural_module, None);
                    device.destroy_shader_module(stars_textured_module, None);
                    device.destroy_shader_module(earth_module, None);
                    device.destroy_shader_module(earth_textured_module, None);
                    device.destroy_pipeline(stars_procedural, None);
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_pool(star_descriptor_pool, None);
                    device.destroy_descriptor_set_layout(star_descriptor_set_layout, None);
                }
                return Err(error);
            }
        };
        let earth_procedural = create_graphics_pipeline(
            device,
            vertex_module,
            earth_module,
            layout,
            HDR_FORMAT,
            Blend::Transmittance,
        );
        let earth_procedural = match earth_procedural {
            Ok(pipeline) => pipeline,
            Err(error) => {
                unsafe {
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_shader_module(stars_procedural_module, None);
                    device.destroy_shader_module(stars_textured_module, None);
                    device.destroy_shader_module(earth_module, None);
                    device.destroy_shader_module(earth_textured_module, None);
                    device.destroy_pipeline(stars_procedural, None);
                    device.destroy_pipeline(stars_textured, None);
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_pool(star_descriptor_pool, None);
                    device.destroy_descriptor_set_layout(star_descriptor_set_layout, None);
                }
                return Err(error);
            }
        };
        let earth_textured = create_graphics_pipeline(
            device,
            vertex_module,
            earth_textured_module,
            layout,
            HDR_FORMAT,
            Blend::Transmittance,
        );
        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(stars_procedural_module, None);
            device.destroy_shader_module(stars_textured_module, None);
            device.destroy_shader_module(earth_module, None);
            device.destroy_shader_module(earth_textured_module, None);
        }
        let earth_textured = match earth_textured {
            Ok(pipeline) => pipeline,
            Err(error) => {
                unsafe {
                    device.destroy_pipeline(stars_procedural, None);
                    device.destroy_pipeline(stars_textured, None);
                    device.destroy_pipeline(earth_procedural, None);
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_pool(star_descriptor_pool, None);
                    device.destroy_descriptor_set_layout(star_descriptor_set_layout, None);
                }
                return Err(error);
            }
        };
        let (star_points_layout, star_points) =
            match create_star_points_pipeline(device, star_descriptor_set_layout, HDR_FORMAT) {
                Ok(created) => created,
                Err(error) => {
                    unsafe {
                        device.destroy_pipeline(stars_procedural, None);
                        device.destroy_pipeline(stars_textured, None);
                        device.destroy_pipeline(earth_procedural, None);
                        device.destroy_pipeline(earth_textured, None);
                        device.destroy_pipeline_layout(layout, None);
                        device.destroy_descriptor_pool(star_descriptor_pool, None);
                        device.destroy_descriptor_set_layout(star_descriptor_set_layout, None);
                    }
                    return Err(error);
                }
            };
        let post = match PostPipeline::create(device, color_format) {
            Ok(post) => post,
            Err(error) => {
                unsafe {
                    device.destroy_pipeline(stars_procedural, None);
                    device.destroy_pipeline(stars_textured, None);
                    device.destroy_pipeline(earth_procedural, None);
                    device.destroy_pipeline(earth_textured, None);
                    device.destroy_pipeline(star_points, None);
                    device.destroy_pipeline_layout(star_points_layout, None);
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_pool(star_descriptor_pool, None);
                    device.destroy_descriptor_set_layout(star_descriptor_set_layout, None);
                }
                return Err(error);
            }
        };
        Ok(Self {
            post,
            star_points_layout,
            star_points,
            color_format,
            layout,
            star_descriptor_set_layout,
            star_descriptor_pool,
            vt_descriptor_set_layout,
            vt_descriptor_pool,
            stars_procedural,
            stars_textured,
            earth_procedural,
            earth_textured,
        })
    }

    unsafe fn destroy(&self, device: &Device) {
        self.post.destroy(device);
        device.destroy_pipeline(self.stars_procedural, None);
        device.destroy_pipeline(self.stars_textured, None);
        device.destroy_pipeline(self.earth_procedural, None);
        device.destroy_pipeline(self.earth_textured, None);
        device.destroy_pipeline(self.star_points, None);
        device.destroy_pipeline_layout(self.star_points_layout, None);
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_descriptor_pool(self.vt_descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.vt_descriptor_set_layout, None);
        device.destroy_descriptor_pool(self.star_descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.star_descriptor_set_layout, None);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Blend {
    Opaque,
    Additive,
    /// Dual-source: premultiplied radiance plus per-channel transmittance of
    /// what lies behind (the air reddens the Sun and stars it covers).
    Transmittance,
}

/// Layout (star texture set + `StarFrame` push constants) and additive
/// pipeline for the per-star quads.
fn create_star_points_pipeline(
    device: &Device,
    star_descriptor_set_layout: vk::DescriptorSetLayout,
    color_format: vk::Format,
) -> RendererResult<(vk::PipelineLayout, vk::Pipeline)> {
    let vertex_code = read_spv(&mut Cursor::new(include_bytes!(concat!(env!("OUT_DIR"), "/stars_points.vert.spv"))))?;
    let fragment_code = read_spv(&mut Cursor::new(include_bytes!(concat!(env!("OUT_DIR"), "/stars_points.frag.spv"))))?;
    let set_layouts = [star_descriptor_set_layout];
    let push_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(size_of::<StarFrame>() as u32)];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }?;
    let modules = unsafe {
        let vertex = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vertex_code), None);
        let fragment = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&fragment_code), None);
        match (vertex, fragment) {
            (Ok(vertex), Ok(fragment)) => Ok((vertex, fragment)),
            (vertex, fragment) => {
                for module in [vertex.ok(), fragment.ok()].into_iter().flatten() {
                    device.destroy_shader_module(module, None);
                }
                device.destroy_pipeline_layout(layout, None);
                Err(vertex.err().or(fragment.err()).unwrap_or(vk::Result::ERROR_UNKNOWN))
            }
        }
    }?;
    let pipeline = create_graphics_pipeline(device, modules.0, modules.1, layout, color_format, Blend::Additive);
    unsafe {
        device.destroy_shader_module(modules.0, None);
        device.destroy_shader_module(modules.1, None);
    }
    match pipeline {
        Ok(pipeline) => Ok((layout, pipeline)),
        Err(error) => {
            unsafe { device.destroy_pipeline_layout(layout, None) };
            Err(error)
        }
    }
}

fn create_graphics_pipeline(
    device: &Device,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
    blend: Blend,
) -> RendererResult<vk::Pipeline> {
    let entry_name = CString::new("main")?;
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(&entry_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(&entry_name),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    // Additive keeps the destination alpha: (0 * src) + (1 * dst).
    let (src_color, dst_color, src_alpha, dst_alpha) = match blend {
        Blend::Additive => (vk::BlendFactor::ONE, vk::BlendFactor::ONE, vk::BlendFactor::ZERO, vk::BlendFactor::ONE),
        _ => (vk::BlendFactor::ONE, vk::BlendFactor::SRC1_COLOR, vk::BlendFactor::ONE, vk::BlendFactor::ONE_MINUS_SRC_ALPHA),
    };
    let color_attachment = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(blend != Blend::Opaque)
        .src_color_blend_factor(src_color)
        .dst_color_blend_factor(dst_color)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(src_alpha)
        .dst_alpha_blend_factor(dst_alpha)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_attachment);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = [color_format];
    let mut rendering_info =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
    let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
        .push_next(&mut rendering_info)
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .render_pass(vk::RenderPass::null())];
    match unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
    } {
        Ok(mut pipelines) => Ok(pipelines.remove(0)),
        Err((pipelines, error)) => {
            unsafe {
                for pipeline in pipelines {
                    device.destroy_pipeline(pipeline, None);
                }
            }
            Err(error.into())
        }
    }
}

impl PinnedBgraTexture {
    fn create(
        device: &Device,
        resources: &StarTextureResources,
        format: vk::Format,
        address_mode_v: vk::SamplerAddressMode,
        width: u32,
        height: u32,
        bytes: &[u8],
        label: &str,
    ) -> RendererResult<Self> {
        let mips = [StarPanoramaMip {
            byte_offset: 0,
            width,
            height,
        }];
        Self::create_mipped(
            device,
            resources,
            format,
            address_mode_v,
            width,
            height,
            &mips,
            bytes,
            label,
        )
    }

    /// Uploads a preview texture: raw BGRA8 as `bgra_format`, or its offline
    /// block-compressed mips, whose colour space follows `bgra_format`.
    fn create_from_preview(
        device: &Device,
        resources: &StarTextureResources,
        bgra_format: vk::Format,
        address_mode_v: vk::SamplerAddressMode,
        texture: &PreviewTexture,
        bytes: &[u8],
        label: &str,
    ) -> RendererResult<Self> {
        let (width, height) = texture.extent();
        let srgb = bgra_format == STAR_PANORAMA_FORMAT;
        let format = match texture.block_format() {
            None => {
                return Self::create(device, resources, bgra_format, address_mode_v, width, height, bytes, label)
            }
            Some(BlockFormat::Bc1) if srgb => STAR_PANORAMA_BC1_FORMAT,
            Some(BlockFormat::Bc1) => return Err(format!("{label}: BC1 previews must be sRGB colour").into()),
            Some(BlockFormat::Bc3) if srgb => PREVIEW_BC3_SRGB_FORMAT,
            Some(BlockFormat::Bc3) => PREVIEW_BC3_LINEAR_FORMAT,
            // Single-channel data; sRGB-encoded sources decode in the shader.
            Some(BlockFormat::Bc4) => PREVIEW_BC4_FORMAT,
            Some(BlockFormat::Bc5) => PREVIEW_BC5_FORMAT,
        };
        Self::create_mipped(device, resources, format, address_mode_v, width, height, texture.mips(), bytes, label)
    }

    fn create_mipped(
        device: &Device,
        resources: &StarTextureResources,
        format: vk::Format,
        address_mode_v: vk::SamplerAddressMode,
        width: u32,
        height: u32,
        mips: &[StarPanoramaMip],
        bytes: &[u8],
        label: &str,
    ) -> RendererResult<Self> {
        if mips.is_empty() {
            return Err(format!("{label} has no mip levels").into());
        }
        let format_features = match format {
            STAR_PANORAMA_FORMAT => resources.srgb_format_features,
            STAR_PANORAMA_BC1_FORMAT => resources.bc1_format_features,
            PREVIEW_BC3_SRGB_FORMAT | PREVIEW_BC3_LINEAR_FORMAT => resources.bc3_format_features,
            PREVIEW_BC4_FORMAT | PREVIEW_BC5_FORMAT => resources.bc4_format_features,
            LINEAR_PREVIEW_FORMAT => resources.linear_format_features,
            SKY_LUT_FORMAT => vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
                | vk::FormatFeatureFlags::TRANSFER_DST,
            _ => return Err("unsupported pinned texture format".into()),
        };
        validate_pinned_bgra_image_support(
            label,
            width,
            height,
            resources.max_image_dimension,
            format_features,
        )?;
        // Canonical colour/mask previews get a one-time GPU mip bake. Raw
        // 5x5 normal pages cannot be filtered across their unrelated neighbours.
        let generate_mips = mips.len() == 1 && width > 1 && height > 1
            && !is_block_compressed(format)
            && !label.contains("surface-normal")
            && format != SKY_LUT_FORMAT
            && format_features.contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST);
        let mip_levels = if generate_mips { u32::BITS - width.max(height).leading_zeros() } else { mips.len() as u32 };
        let staging = unsafe { StagingBuffer::create(device, resources.memory_properties, bytes)? };
        let allocation = match unsafe {
            ImageAllocation::create_mipped(
                device,
                resources.memory_properties,
                format,
                width,
                height,
                mip_levels,
                generate_mips,
            )
        } {
            Ok(allocation) => allocation,
            Err(error) => {
                unsafe { staging.destroy(device) };
                return Err(error);
            }
        };
        let upload = unsafe {
            upload_staging_mip_image(
                device,
                resources.queue,
                resources.command_pool,
                staging.buffer,
                allocation.image,
                mips,
                mip_levels,
            )
        };
        unsafe { staging.destroy(device) };
        if let Err(error) = upload {
            unsafe { allocation.destroy(device) };
            return Err(error);
        }
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1);
        let view_info = vk::ImageViewCreateInfo::default()
            .image(allocation.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresource_range);
        let view = match unsafe { device.create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(error) => {
                unsafe { allocation.destroy(device) };
                return Err(error.into());
            }
        };
        // With a single resident level and max_lod=0 the sampled level is
        // always 0, so inter-level blend mode is a no-op: NEAREST skips the
        // trilinear setup. A 1x1 has one texel, so NEAREST mag/min is exact.
        let single_mip = mip_levels == 1;
        let single_texel = single_mip && width == 1 && height == 1;
        // Equirect maps wrap in U. A 1-texel-high strip (Saturn ring, atmosphere
        // LUT) is radial/tabular, not periodic: inherit V so LINEAR cannot blend
        // the inner edge with the outer edge.
        let address_mode_u = if height == 1 || format == SKY_LUT_FORMAT {
            address_mode_v
        } else {
            vk::SamplerAddressMode::REPEAT
        };
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(if single_texel { vk::Filter::NEAREST } else { vk::Filter::LINEAR })
            .min_filter(if single_texel { vk::Filter::NEAREST } else { vk::Filter::LINEAR })
            .mipmap_mode(if single_mip { vk::SamplerMipmapMode::NEAREST } else { vk::SamplerMipmapMode::LINEAR })
            .address_mode_u(address_mode_u)
            .address_mode_v(address_mode_v)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            // Grazing views (the limb, the poles of equirect maps) need
            // anisotropic footprints; isotropic mips smeared them radially.
            .anisotropy_enable(!single_mip && resources.max_anisotropy >= 1.0)
            .max_anisotropy(resources.max_anisotropy.max(1.0))
            .min_lod(0.0)
            .max_lod(mip_levels.saturating_sub(1) as f32);
        let sampler = match unsafe { device.create_sampler(&sampler_info, None) } {
            Ok(sampler) => sampler,
            Err(error) => {
                unsafe {
                    device.destroy_image_view(view, None);
                    allocation.destroy(device);
                }
                return Err(error.into());
            }
        };
        Ok(Self {
            image: allocation.image,
            memory: allocation.memory,
            view,
            sampler,
        })
    }

    unsafe fn destroy(&self, device: &Device) {
        device.destroy_sampler(self.sampler, None);
        device.destroy_image_view(self.view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}

fn allocate_texture_descriptor_set(
    device: &Device,
    resources: &StarTextureResources,
) -> RendererResult<vk::DescriptorSet> {
    let descriptor_set_layouts = [resources.descriptor_set_layout];
    let descriptor_allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(resources.descriptor_pool)
        .set_layouts(&descriptor_set_layouts);
    match unsafe { device.allocate_descriptor_sets(&descriptor_allocate_info) } {
        Ok(mut sets) if sets.len() == 1 => Ok(sets.remove(0)),
        Ok(_) => Err("Vulkan allocated an unexpected texture descriptor-set count".into()),
        Err(error) => Err(error.into()),
    }
}

impl PinnedPlanetTextures {
    fn create_fallback(device: &Device, resources: &StarTextureResources) -> RendererResult<PinnedBgraTexture> {
        PinnedBgraTexture::create(
            device,
            resources,
            STAR_PANORAMA_FORMAT,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            1,
            1,
            &[0, 0, 0, 255],
            "planet fallback",
        )
    }

    fn create_for_body(
        device: &Device,
        resources: &StarTextureResources,
        body: crate::body::Body,
        planet_previews: &PlanetPreviews,
        saturn_ring_preview: Option<&RingPreview>,
    ) -> RendererResult<Self> {
        let fallback = Self::create_fallback(device, resources)?;
        let albedo = match PinnedStarTexture::create_preview_or_fallback(
            device,
            resources,
            planet_previews.get(body),
            [0, 0, 0, 255],
            "planet albedo",
        ) {
            Ok(texture) => texture,
            Err(error) => {
                unsafe { fallback.destroy(device) };
                return Err(error);
            }
        };
        let ring = if body == crate::body::Body::Saturn {
            match saturn_ring_preview {
                Some(preview) => {
                    let source = preview.texture();
                    let mapped = source.map_payload()?;
                    PinnedBgraTexture::create_from_preview(device, resources, STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, source, mapped.bytes(), "saturn ring")
                }
                None => Self::create_fallback(device, resources),
            }
        } else {
            Self::create_fallback(device, resources)
        };
        let ring = match ring {
            Ok(texture) => texture,
            Err(error) => {
                unsafe {
                    albedo.destroy(device);
                    fallback.destroy(device);
                }
                return Err(error);
            }
        };
        Ok(Self {
            resident: body,
            albedo,
            ring,
            fallback,
            resources: *resources,
        })
    }

    /// Descriptor infos for bindings 11..=15: binding 11 carries whichever
    /// body's albedo is resident (the shader reads it for every non-Earth
    /// body), 15 Saturn's ring; 12..=14 are legacy slots kept as fallbacks so
    /// the descriptor layout is unchanged.
    fn descriptor_infos(&self) -> [vk::DescriptorImageInfo; 5] {
        let info = |texture: &PinnedBgraTexture| {
            vk::DescriptorImageInfo::default()
                .sampler(texture.sampler)
                .image_view(texture.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        };
        let resident_id = self.resident.id() as usize;
        [
            info(if resident_id != 0 { &self.albedo } else { &self.fallback }),
            info(&self.fallback),
            info(&self.fallback),
            info(&self.fallback),
            info(if resident_id == 4 { &self.ring } else { &self.fallback }),
        ]
    }

    /// Make `body`'s albedo (and Saturn's ring) resident, freeing the previous
    /// one. The descriptor set is shared by every output, so the GPU must be
    /// idle before the old texture is destroyed; the one-off stall only
    /// happens on a body switch.
    fn swap(
        &mut self,
        device: &Device,
        descriptor_set: vk::DescriptorSet,
        body: crate::body::Body,
        planet_previews: &PlanetPreviews,
        saturn_ring_preview: Option<&RingPreview>,
    ) -> RendererResult<()> {
        if body == self.resident {
            return Ok(());
        }
        let albedo = PinnedStarTexture::create_preview_or_fallback(
            device,
            &self.resources,
            planet_previews.get(body),
            [0, 0, 0, 255],
            "planet albedo",
        )?;
        let ring = if body == crate::body::Body::Saturn {
            match saturn_ring_preview {
                Some(preview) => {
                    let source = preview.texture();
                    let mapped = source.map_payload()?;
                    PinnedBgraTexture::create_from_preview(device, &self.resources, STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, source, mapped.bytes(), "saturn ring")?
                }
                None => Self::create_fallback(device, &self.resources)?,
            }
        } else {
            Self::create_fallback(device, &self.resources)?
        };
        unsafe { device.queue_wait_idle(self.resources.queue)? };
        let infos = {
            let old_albedo = std::mem::replace(&mut self.albedo, albedo);
            let old_ring = std::mem::replace(&mut self.ring, ring);
            self.resident = body;
            unsafe {
                old_albedo.destroy(device);
                old_ring.destroy(device);
            }
            self.descriptor_infos()
        };
        let mut writes = Vec::with_capacity(infos.len());
        for (binding, info) in infos.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(11 + binding as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(info)),
            );
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok(())
    }

    unsafe fn destroy(&self, device: &Device) {
        self.albedo.destroy(device);
        self.ring.destroy(device);
        self.fallback.destroy(device);
    }
}

impl PinnedStarTexture {
    /// Upload a planet albedo or ring texture, falling back to a 1x1 texture
    /// when the preview is absent so the shared descriptor set always carries
    /// a valid sampler for every switchable body.
    fn create_preview_or_fallback(
        device: &Device,
        resources: &StarTextureResources,
        preview: Option<&PlanetPreview>,
        fallback: [u8; 4],
        label: &str,
    ) -> RendererResult<PinnedBgraTexture> {
        match preview {
            Some(preview) => {
                let source = preview.texture();
                let mapped = source.map_payload()?;
                PinnedBgraTexture::create_from_preview(device, resources, STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, source, mapped.bytes(), label)
            }
            None => PinnedBgraTexture::create(
                device,
                resources,
                STAR_PANORAMA_FORMAT,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                1,
                1,
                &fallback,
                &format!("{label} fallback"),
            ),
        }
    }
    /// Keep the Moon albedo resident only while Earth is selected. The Moon
    /// disc branch is gated on `body == 0` in the star shader, so every other
    /// body carries the same 1x1 fallback a missing preview would. Mirrors
    /// `PinnedPlanetTextures::swap`: the replacement uploads first, the queue
    /// drains, and only then is the old image destroyed and binding 10
    /// rewritten, so the shared descriptor set never dangles.
    fn sync_moon(
        &mut self,
        device: &Device,
        body: crate::body::Body,
        moon_preview: Option<&MoonPreview>,
    ) -> RendererResult<()> {
        let want_moon = body == crate::body::Body::Earth;
        if want_moon == self.moon_resident {
            return Ok(());
        }
        let started = std::time::Instant::now();
        // `create_preview_or_fallback` only handles planet albedos; the Moon
        // match below mirrors the `create` arms exactly.
        let replacement = match moon_preview.filter(|_| want_moon) {
            Some(preview) => {
                let moon_source = preview.texture();
                let mapped = moon_source.map_payload()?;
                PinnedBgraTexture::create_from_preview(device, &self.planets.resources, STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, moon_source, mapped.bytes(), "moon albedo")?
            }
            None => PinnedBgraTexture::create(
                device,
                &self.planets.resources,
                STAR_PANORAMA_FORMAT,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                1,
                1,
                &[0, 0, 0, 255],
                "moon albedo fallback",
            )?,
        };
        unsafe { device.queue_wait_idle(self.planets.resources.queue)? };
        let old = std::mem::replace(&mut self.moon, replacement);
        unsafe { old.destroy(device) };
        self.moon_resident = want_moon;
        let info = [vk::DescriptorImageInfo::default()
            .sampler(self.moon.sampler)
            .image_view(self.moon.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(10)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info)];
        unsafe { device.update_descriptor_sets(&write, &[]) };
        eprintln!(
            "earth-native: moon albedo {} in {:?}",
            if want_moon { "restored" } else { "evicted" },
            started.elapsed()
        );
        Ok(())
    }

    fn create(
        device: &Device,
        resources: &StarTextureResources,
        panorama: Option<&StarPanorama>,
        moon_preview: Option<&MoonPreview>,
        initial_body: crate::body::Body,
        planet_previews: &PlanetPreviews,
        saturn_ring_preview: Option<&RingPreview>,
    ) -> RendererResult<Self> {
        let panorama_texture = match panorama {
            Some(panorama) => {
                let (width, height) = panorama.extent();
                let mapped = panorama.map_payload()?;
                match panorama.format() {
                    StarPanoramaFormat::Bgra8Srgb => PinnedBgraTexture::create(
                        device,
                        resources,
                        STAR_PANORAMA_FORMAT,
                        vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        width,
                        height,
                        mapped.bytes(),
                        "star panorama",
                    )?,
                    StarPanoramaFormat::Bc1Srgb => PinnedBgraTexture::create_mipped(
                        device,
                        resources,
                        STAR_PANORAMA_BC1_FORMAT,
                        vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        width,
                        height,
                        panorama.mips(),
                        mapped.bytes(),
                        "BC1 star panorama",
                    )?,
                }
            }
            None => PinnedBgraTexture::create(
                device,
                resources,
                STAR_PANORAMA_FORMAT,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                1,
                1,
                &[0, 0, 0, 255],
                "star panorama fallback",
            )?,
        };
        // The Moon disc only renders for Earth, so any other startup body
        // keeps a 1x1 fallback here exactly like a missing Moon preview.
        let moon_preview = if Renderer::body_residency_enabled() {
            moon_preview.filter(|_| initial_body == crate::body::Body::Earth)
        } else {
            moon_preview
        };
        let moon = match moon_preview {
            Some(preview) => {
                let moon_source = preview.texture();
                let mapped = moon_source.map_payload()?;
                PinnedBgraTexture::create_from_preview(device, resources, STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, moon_source, mapped.bytes(), "moon albedo")?
            }
            None => PinnedBgraTexture::create(
                device,
                resources,
                STAR_PANORAMA_FORMAT,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                1,
                1,
                &[0, 0, 0, 255],
                "moon albedo fallback",
            )?,
        };
        let planets = PinnedPlanetTextures::create_for_body(
            device,
            resources,
            initial_body,
            planet_previews,
            saturn_ring_preview,
        )?;
        let (catalog_bytes, catalog_width, catalog_height, star_count) = crate::star_catalog::bake();
        let catalog = match PinnedBgraTexture::create(
            device,
            resources,
            LINEAR_PREVIEW_FORMAT,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            catalog_width,
            catalog_height,
            &catalog_bytes,
            "bright star catalogue",
        ) {
            Ok(texture) => texture,
            Err(error) => {
                unsafe {
                    panorama_texture.destroy(device);
                    moon.destroy(device);
                    planets.destroy(device);
                }
                return Err(error);
            }
        };
        let descriptor_set = match allocate_texture_descriptor_set(device, resources) {
            Ok(descriptor_set) => descriptor_set,
            Err(error) => {
                unsafe {
                    panorama_texture.destroy(device);
                    moon.destroy(device);
                    planets.destroy(device);
                    catalog.destroy(device);
                }
                return Err(error);
            }
        };
        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(panorama_texture.sampler)
            .image_view(panorama_texture.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let moon_info = [vk::DescriptorImageInfo::default()
            .sampler(moon.sampler)
            .image_view(moon.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let catalog_info = [vk::DescriptorImageInfo::default()
            .sampler(catalog.sampler)
            .image_view(catalog.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

        let planet_infos = planets.descriptor_infos();
        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(16)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&catalog_info),

            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(10)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&moon_info),
        ];
        for (binding, info) in planet_infos.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(11 + binding as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(info)),
            );
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok(Self {
            texture: panorama_texture,
            catalog,
            star_count,
            moon,
            planets,
            descriptor_set,
            moon_resident: !Renderer::body_residency_enabled() || initial_body == crate::body::Body::Earth,
        })
    }

    unsafe fn destroy(&self, device: &Device) {
        self.texture.destroy(device);
        self.moon.destroy(device);
        self.planets.destroy(device);
        self.catalog.destroy(device);
    }
}

impl PinnedDayColorTextures {
    fn bind_environment_fields(&self, device: &Device) {
        // Valid fallback descriptors are required even when weather is disabled.
        // Weather uploads own binding 11 and must never replace authored clouds.
        let weather = self.weather_fields.as_ref().unwrap_or(&self.cloud_a);
        let height = self.cloud_height.as_ref().unwrap_or(&self.height);
        let infos = [weather, height].map(|texture| vk::DescriptorImageInfo::default()
            .sampler(texture.sampler).image_view(texture.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
        let writes: Vec<_> = infos.iter().enumerate().map(|(index, info)|
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_set)
                .dst_binding(11 + index as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(info))).collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    fn load_cloud_height(&mut self, device: &Device) -> RendererResult<()> {
        let Some(path) = std::env::var_os("EARTH_NATIVE_CLOUD_HEIGHT") else { return Ok(()); };
        let source = crate::day_color::load_preview_texture("EARTH_NATIVE_CLOUD_HEIGHT",
            path, crate::day_color::PreviewColorSpace::Linear)?;
        let mapped = source.map_payload()?;
        self.cloud_height = Some(PinnedBgraTexture::create_from_preview(device, &self.resources, LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, &source, mapped.bytes(), "authored cloud terrain height")?);
        Ok(())
    }

    fn atmosphere_lut(device: &Device, resources: &StarTextureResources) -> RendererResult<PinnedBgraTexture> {
        PinnedBgraTexture::create(device, resources, LINEAR_PREVIEW_FORMAT,
            vk::SamplerAddressMode::CLAMP_TO_EDGE, crate::atmosphere::LUT_WIDTH as u32, 1,
            include_bytes!(concat!(env!("OUT_DIR"), "/atmosphere-column.bgra")), "atmosphere column LUT")
    }
    /// Rewrite one core binding (0..=9) after a swap. Binding 10 (atmosphere
    /// LUT) is never evicted; bindings 11/12 go through
    /// `bind_environment_fields`.
    fn write_core_binding(&self, device: &Device, binding: u32, texture: &PinnedBgraTexture) {
        let info = [vk::DescriptorImageInfo::default()
            .sampler(texture.sampler)
            .image_view(texture.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info)];
        unsafe { device.update_descriptor_sets(&write, &[]) };
    }

    /// Rewrite bindings 0..=9 from the current fields.
    fn write_core_bindings(&self, device: &Device) {
        let fields: [&PinnedBgraTexture; 10] = [
            &self.tiling_noise,
            &self.east,
            &self.west,
            &self.night,
            &self.cloud_a,
            &self.cloud_ba,
            &self.desert_cloud_mask,
            &self.height,
            &self.normal_east,
            &self.normal_west,
        ];
        for (binding, texture) in fields.into_iter().enumerate() {
            self.write_core_binding(device, binding as u32, texture);
        }
    }

    /// Replace the big Earth previews with 1x1 fallbacks once the last GPU
    /// use completes. Only the procedural path runs off-Earth and it never
    /// binds this descriptor set, yet every binding stays valid throughout.
    /// The tiling-noise preview (64 KiB) and atmosphere LUT (2 KiB) stay.
    fn evict_earth_stack(&mut self, device: &Device) -> RendererResult<()> {
        if !self.earth_resident {
            return Ok(());
        }
        let started = std::time::Instant::now();
        // Upload every fallback before touching the resident stack, mirroring
        // `create_fallback`, so a failure leaves the current stack intact.
        let specs = [
            (STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, [0, 0, 0, 255], "evicted day east"),
            (STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, [0, 0, 0, 255], "evicted day west"),
            (STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::REPEAT, [0, 0, 0, 255], "evicted night"),
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, [0, 0, 0, 255], "evicted clouds A"),
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, [0, 0, 0, 255], "evicted clouds BA"),
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, [0, 0, 0, 255], "evicted desert mask"),
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, [0, 0, 0, 255], "evicted height"),
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, [255, 128, 128, 255], "evicted normal east"),
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, [255, 128, 128, 255], "evicted normal west"),
        ];
        let mut fallbacks = Vec::with_capacity(specs.len());
        for (format, address, bytes, label) in specs {
            match PinnedBgraTexture::create(device, &self.resources, format, address, 1, 1, &bytes, label) {
                Ok(texture) => fallbacks.push(texture),
                Err(error) => {
                    unsafe { for texture in fallbacks { texture.destroy(device); } }
                    return Err(error);
                }
            }
        }
        unsafe { device.queue_wait_idle(self.resources.queue)? };
        let mut fallbacks = fallbacks.into_iter();
        let olds = [
            std::mem::replace(&mut self.east, fallbacks.next().unwrap()),
            std::mem::replace(&mut self.west, fallbacks.next().unwrap()),
            std::mem::replace(&mut self.night, fallbacks.next().unwrap()),
            std::mem::replace(&mut self.cloud_a, fallbacks.next().unwrap()),
            std::mem::replace(&mut self.cloud_ba, fallbacks.next().unwrap()),
            std::mem::replace(&mut self.desert_cloud_mask, fallbacks.next().unwrap()),
            std::mem::replace(&mut self.height, fallbacks.next().unwrap()),
            std::mem::replace(&mut self.normal_east, fallbacks.next().unwrap()),
            std::mem::replace(&mut self.normal_west, fallbacks.next().unwrap()),
        ];
        if let Some(previous) = self.weather_fields.take() {
            unsafe { previous.destroy(device) };
        }
        if let Some(previous) = self.cloud_height.take() {
            unsafe { previous.destroy(device) };
        }
        unsafe { for old in olds { old.destroy(device) } }
        self.earth_resident = false;
        self.write_core_bindings(device);
        self.bind_environment_fields(device);
        eprintln!("earth-native: earth preview stack evicted in {:?}", started.elapsed());
        Ok(())
    }

    fn upload_day_east(
        device: &Device,
        resources: &StarTextureResources,
        hemispheres: &DayColorHemispheres,
    ) -> RendererResult<PinnedBgraTexture> {
        let mapped = hemispheres.east().map_canonical_payload(true)?;
        PinnedBgraTexture::create_from_preview(device, resources, STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, hemispheres.east(), &mapped, "day-colour east hemisphere")
    }

    fn upload_day_west(
        device: &Device,
        resources: &StarTextureResources,
        hemispheres: &DayColorHemispheres,
    ) -> RendererResult<PinnedBgraTexture> {
        let mapped = hemispheres.west().map_canonical_payload(false)?;
        PinnedBgraTexture::create_from_preview(device, resources, STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, hemispheres.west(), &mapped, "day-colour west hemisphere")
    }

    fn upload_or_fallback_night(
        device: &Device,
        resources: &StarTextureResources,
        night: Option<&NightEmissionPreview>,
    ) -> RendererResult<PinnedBgraTexture> {
        match night {
            Some(preview) => {
                let texture = preview.texture();
                let mapped = texture.map_payload()?;
                PinnedBgraTexture::create_from_preview(device, resources, STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::REPEAT, texture, mapped.bytes(), "night-emission preview")
            }
            None => PinnedBgraTexture::create(
                device,
                resources,
                STAR_PANORAMA_FORMAT,
                vk::SamplerAddressMode::REPEAT,
                1,
                1,
                &[0, 0, 0, 255],
                "night-emission fallback",
            ),
        }
    }

    fn upload_or_fallback_cloud(
        device: &Device,
        resources: &StarTextureResources,
        cloud: Option<&PreviewTexture>,
        label: &str,
        fallback_label: &str,
    ) -> RendererResult<PinnedBgraTexture> {
        match cloud {
            Some(texture) => {
                let mapped = texture.map_payload()?;
                PinnedBgraTexture::create_from_preview(device, resources, LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, texture, mapped.bytes(), label)
            }
            None => PinnedBgraTexture::create(
                device,
                resources,
                LINEAR_PREVIEW_FORMAT,
                vk::SamplerAddressMode::REPEAT,
                1,
                1,
                &[0, 0, 0, 255],
                fallback_label,
            ),
        }
    }

    fn upload_or_fallback_normal(
        device: &Device,
        resources: &StarTextureResources,
        normal: Option<&PreviewTexture>,
        label: &str,
        fallback_label: &str,
    ) -> RendererResult<PinnedBgraTexture> {
        match normal {
            Some(texture) => {
                let mapped = texture.map_payload()?;
                PinnedBgraTexture::create_from_preview(device, resources, LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, texture, mapped.bytes(), label)
            }
            None => PinnedBgraTexture::create(
                device,
                resources,
                LINEAR_PREVIEW_FORMAT,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                1,
                1,
                &[255, 128, 128, 255],
                fallback_label,
            ),
        }
    }

    /// Re-upload the full Earth stack before Earth presents again. Each field
    /// swaps in place and rewrites its own binding immediately, so an upload
    /// failure leaves a valid mixed stack, keeps `earth_resident` false, and
    /// the next frame retries. Labels, formats, and fallback bytes match
    /// `create` exactly.
    fn restore_earth_stack(
        &mut self,
        device: &Device,
        hemispheres: &DayColorHemispheres,
        surface_normals: Option<&SurfaceNormalHemispheres>,
        night_emission: Option<&NightEmissionPreview>,
        cloud_previews: Option<&CloudPreviews>,
        desert_cloud_mask: Option<&DesertCloudMaskPreview>,
        height_preview: Option<&HeightPreview>,
    ) -> RendererResult<()> {
        if self.earth_resident {
            return Ok(());
        }
        let started = std::time::Instant::now();
        unsafe { device.queue_wait_idle(self.resources.queue)? };
        let replacement = Self::upload_day_east(device, &self.resources, hemispheres)?;
        let old = std::mem::replace(&mut self.east, replacement);
        unsafe { old.destroy(device) };
        self.write_core_binding(device, 1, &self.east);
        let replacement = Self::upload_day_west(device, &self.resources, hemispheres)?;
        let old = std::mem::replace(&mut self.west, replacement);
        unsafe { old.destroy(device) };
        self.write_core_binding(device, 2, &self.west);
        let replacement = Self::upload_or_fallback_night(device, &self.resources, night_emission)?;
        let old = std::mem::replace(&mut self.night, replacement);
        unsafe { old.destroy(device) };
        self.write_core_binding(device, 3, &self.night);
        let replacement = Self::upload_or_fallback_cloud(
            device,
            &self.resources,
            cloud_previews.map(|previews| previews.a()),
            "cloud A preview",
            "cloud A fallback",
        )?;
        let old = std::mem::replace(&mut self.cloud_a, replacement);
        unsafe { old.destroy(device) };
        self.write_core_binding(device, 4, &self.cloud_a);
        let replacement = Self::upload_or_fallback_cloud(
            device,
            &self.resources,
            cloud_previews.map(|previews| previews.ba()),
            "cloud BA preview",
            "cloud BA fallback",
        )?;
        let old = std::mem::replace(&mut self.cloud_ba, replacement);
        unsafe { old.destroy(device) };
        self.write_core_binding(device, 5, &self.cloud_ba);
        let replacement = Self::upload_or_fallback_cloud(
            device,
            &self.resources,
            desert_cloud_mask.map(|preview| preview.texture()),
            "desert-cloud permission preview",
            "desert-cloud permission fallback",
        )?;
        let old = std::mem::replace(&mut self.desert_cloud_mask, replacement);
        unsafe { old.destroy(device) };
        self.write_core_binding(device, 6, &self.desert_cloud_mask);
        let replacement = Self::upload_or_fallback_cloud(
            device,
            &self.resources,
            height_preview.map(|preview| preview.texture()),
            "terrain-height preview",
            "terrain-height fallback",
        )?;
        let old = std::mem::replace(&mut self.height, replacement);
        unsafe { old.destroy(device) };
        self.write_core_binding(device, 7, &self.height);
        let replacement = Self::upload_or_fallback_normal(
            device,
            &self.resources,
            surface_normals.map(|normals| normals.east()),
            "surface-normal east hemisphere",
            "surface-normal east fallback",
        )?;
        let old = std::mem::replace(&mut self.normal_east, replacement);
        unsafe { old.destroy(device) };
        self.write_core_binding(device, 8, &self.normal_east);
        let replacement = Self::upload_or_fallback_normal(
            device,
            &self.resources,
            surface_normals.map(|normals| normals.west()),
            "surface-normal west hemisphere",
            "surface-normal west fallback",
        )?;
        let old = std::mem::replace(&mut self.normal_west, replacement);
        unsafe { old.destroy(device) };
        self.write_core_binding(device, 9, &self.normal_west);
        if let Err(error) = self.load_cloud_height(device) {
            self.bind_environment_fields(device);
            return Err(error);
        }
        self.earth_resident = true;
        self.bind_environment_fields(device);
        eprintln!("earth-native: earth preview stack restored in {:?}", started.elapsed());
        Ok(())
    }

    fn create_fallback(
        device: &Device,
        resources: &StarTextureResources,
    ) -> RendererResult<Self> {
        let specs = [
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, [255, 255, 255, 255], "fallback tiling noise"),
            (STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, [0, 0, 0, 255], "fallback day east"),
            (STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, [0, 0, 0, 255], "fallback day west"),
            (STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::REPEAT, [0, 0, 0, 255], "fallback night"),
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, [0, 0, 0, 255], "fallback clouds A"),
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, [0, 0, 0, 255], "fallback clouds BA"),
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, [0, 0, 0, 255], "fallback desert mask"),
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, [0, 0, 0, 255], "fallback height"),
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, [128, 128, 255, 255], "fallback normal east"),
            (LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, [128, 128, 255, 255], "fallback normal west"),
        ];
        let mut textures = Vec::with_capacity(specs.len());
        for (format, address, bytes, label) in specs {
            match PinnedBgraTexture::create(device, resources, format, address, 1, 1, &bytes, label) {
                Ok(texture) => textures.push(texture),
                Err(error) => {
                    unsafe { for texture in textures { texture.destroy(device); } }
                    return Err(error);
                }
            }
        }
        match Self::atmosphere_lut(device, resources) {
            Ok(texture) => textures.push(texture),
            Err(error) => {
                unsafe { for texture in textures { texture.destroy(device); } }
                return Err(error);
            }
        }
        let descriptor_set = match allocate_texture_descriptor_set(device, resources) {
            Ok(set) => set,
            Err(error) => {
                unsafe { for texture in textures { texture.destroy(device); } }
                return Err(error);
            }
        };
        let infos: Vec<_> = textures.iter().map(|texture| vk::DescriptorImageInfo::default()
            .sampler(texture.sampler)
            .image_view(texture.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)).collect();
        let writes: Vec<_> = infos.iter().enumerate().map(|(binding, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(info))
        }).collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        let mut textures = textures.into_iter();
        let result = Self {
            tiling_noise: textures.next().unwrap(),
            resources: *resources,
            east: textures.next().unwrap(),
            west: textures.next().unwrap(),
            night: textures.next().unwrap(),
            cloud_a: textures.next().unwrap(),
            cloud_ba: textures.next().unwrap(),
            desert_cloud_mask: textures.next().unwrap(),
            height: textures.next().unwrap(),
            normal_east: textures.next().unwrap(),
            normal_west: textures.next().unwrap(),
            atmosphere: textures.next().unwrap(),
            weather_fields: None,
            cloud_height: None,
            descriptor_set,
            earth_resident: true,
        };
        result.bind_environment_fields(device);
        Ok(result)
    }

    fn create(
        device: &Device,
        resources: &StarTextureResources,
        hemispheres: &DayColorHemispheres,
        surface_normals: Option<&SurfaceNormalHemispheres>,
        night_emission: Option<&NightEmissionPreview>,
        cloud_previews: Option<&CloudPreviews>,
        tiling_noise: Option<&TilingNoisePreview>,
        desert_cloud_mask: Option<&DesertCloudMaskPreview>,
        height_preview: Option<&HeightPreview>,
    ) -> RendererResult<Self> {
        let east = {
            let mapped = hemispheres.east().map_canonical_payload(true)?;
            PinnedBgraTexture::create_from_preview(device, resources, STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, hemispheres.east(), &mapped, "day-colour east hemisphere")?
        };
        let west = {
            let mapped = hemispheres.west().map_canonical_payload(false)?;
            match PinnedBgraTexture::create_from_preview(device, resources, STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, hemispheres.west(), &mapped, "day-colour west hemisphere") {
                Ok(texture) => texture,
                Err(error) => {
                    unsafe { east.destroy(device) };
                    return Err(error);
                }
            }
        };
        let night = match night_emission {
            Some(preview) => {
                let texture = preview.texture();
                let mapped = match texture.map_payload() {
                    Ok(mapped) => mapped,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                        }
                        return Err(error.into());
                    }
                };
                match PinnedBgraTexture::create_from_preview(device, resources, STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::REPEAT, texture, mapped.bytes(), "night-emission preview") {
                    Ok(texture) => texture,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                        }
                        return Err(error);
                    }
                }
            }
            None => match PinnedBgraTexture::create(
                device,
                resources,
                STAR_PANORAMA_FORMAT,
                vk::SamplerAddressMode::REPEAT,
                1,
                1,
                &[0, 0, 0, 255],
                "night-emission fallback",
            ) {
                Ok(texture) => texture,
                Err(error) => {
                    unsafe {
                        east.destroy(device);
                        west.destroy(device);
                    }
                    return Err(error);
                }
            },
        };
        let cloud_a = match cloud_previews {
            Some(previews) => {
                let texture = previews.a();
                let mapped = match texture.map_payload() {
                    Ok(mapped) => mapped,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                        }
                        return Err(error.into());
                    }
                };
                match PinnedBgraTexture::create_from_preview(device, resources, LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, texture, mapped.bytes(), "cloud A preview") {
                    Ok(texture) => texture,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                        }
                        return Err(error);
                    }
                }
            }
            None => match PinnedBgraTexture::create(
                device,
                resources,
                LINEAR_PREVIEW_FORMAT,
                vk::SamplerAddressMode::REPEAT,
                1,
                1,
                &[0, 0, 0, 255],
                "cloud A fallback",
            ) {
                Ok(texture) => texture,
                Err(error) => {
                    unsafe {
                        east.destroy(device);
                        west.destroy(device);
                        night.destroy(device);
                    }
                    return Err(error);
                }
            },
        };
        let cloud_ba = match cloud_previews {
            Some(previews) => {
                let texture = previews.ba();
                let mapped = match texture.map_payload() {
                    Ok(mapped) => mapped,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                            cloud_a.destroy(device);
                        }
                        return Err(error.into());
                    }
                };
                match PinnedBgraTexture::create_from_preview(device, resources, LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, texture, mapped.bytes(), "cloud BA preview") {
                    Ok(texture) => texture,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                            cloud_a.destroy(device);
                        }
                        return Err(error);
                    }
                }
            }
            None => match PinnedBgraTexture::create(
                device,
                resources,
                LINEAR_PREVIEW_FORMAT,
                vk::SamplerAddressMode::REPEAT,
                1,
                1,
                &[0, 0, 0, 255],
                "cloud BA fallback",
            ) {
                Ok(texture) => texture,
                Err(error) => {
                    unsafe {
                        east.destroy(device);
                        west.destroy(device);
                        night.destroy(device);
                        cloud_a.destroy(device);
                    }
                    return Err(error);
                }
            },
        };
        let tiling_noise = match tiling_noise {
            Some(preview) => {
                let texture = preview.texture();
                let mapped = match texture.map_payload() {
                    Ok(mapped) => mapped,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                            cloud_a.destroy(device);
                            cloud_ba.destroy(device);
                        }
                        return Err(error.into());
                    }
                };
                match PinnedBgraTexture::create_from_preview(device, resources, STAR_PANORAMA_FORMAT, vk::SamplerAddressMode::REPEAT, texture, mapped.bytes(), "cloud tiling-noise preview") {
                    Ok(texture) => texture,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                            cloud_a.destroy(device);
                            cloud_ba.destroy(device);
                        }
                        return Err(error);
                    }
                }
            }
            None => match PinnedBgraTexture::create(
                device,
                resources,
                STAR_PANORAMA_FORMAT,
                vk::SamplerAddressMode::REPEAT,
                1,
                1,
                &[255, 255, 255, 255],
                "cloud tiling-noise fallback",
            ) {
                Ok(texture) => texture,
                Err(error) => {
                    unsafe {
                        east.destroy(device);
                        west.destroy(device);
                        night.destroy(device);
                        cloud_a.destroy(device);
                        cloud_ba.destroy(device);
                    }
                    return Err(error);
                }
            },
        };
        let desert_cloud_mask = match desert_cloud_mask {
            Some(preview) => {
                let texture = preview.texture();
                let mapped = match texture.map_payload() {
                    Ok(mapped) => mapped,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                            cloud_a.destroy(device);
                            cloud_ba.destroy(device);
                            tiling_noise.destroy(device);
                        }
                        return Err(error.into());
                    }
                };
                match PinnedBgraTexture::create_from_preview(device, resources, LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, texture, mapped.bytes(), "desert-cloud permission preview") {
                    Ok(texture) => texture,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                            cloud_a.destroy(device);
                            cloud_ba.destroy(device);
                            tiling_noise.destroy(device);
                        }
                        return Err(error);
                    }
                }
            }
            // A zero desert-distribution source means no authored terrain
            // suppression, preserving the legacy diagnostic when omitted.
            None => match PinnedBgraTexture::create(
                device,
                resources,
                LINEAR_PREVIEW_FORMAT,
                vk::SamplerAddressMode::REPEAT,
                1,
                1,
                &[0, 0, 0, 255],
                "desert-cloud permission fallback",
            ) {
                Ok(texture) => texture,
                Err(error) => {
                    unsafe {
                        east.destroy(device);
                        west.destroy(device);
                        night.destroy(device);
                        cloud_a.destroy(device);
                        cloud_ba.destroy(device);
                        tiling_noise.destroy(device);
                    }
                    return Err(error);
                }
            },
        };
        let height = match height_preview {
            Some(preview) => {
                let texture = preview.texture();
                let mapped = match texture.map_payload() {
                    Ok(mapped) => mapped,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                            cloud_a.destroy(device);
                            cloud_ba.destroy(device);
                            tiling_noise.destroy(device);
                            desert_cloud_mask.destroy(device);
                        }
                        return Err(error.into());
                    }
                };
                match PinnedBgraTexture::create_from_preview(device, resources, LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::REPEAT, texture, mapped.bytes(), "terrain-height preview") {
                    Ok(texture) => texture,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                            cloud_a.destroy(device);
                            cloud_ba.destroy(device);
                            tiling_noise.destroy(device);
                            desert_cloud_mask.destroy(device);
                        }
                        return Err(error);
                    }
                }
            }
            None => match PinnedBgraTexture::create(
                device,
                resources,
                LINEAR_PREVIEW_FORMAT,
                vk::SamplerAddressMode::REPEAT,
                1,
                1,
                &[0, 0, 0, 255],
                "terrain-height fallback",
            ) {
                Ok(texture) => texture,
                Err(error) => {
                    unsafe {
                        east.destroy(device);
                        west.destroy(device);
                        night.destroy(device);
                        cloud_a.destroy(device);
                        cloud_ba.destroy(device);
                        tiling_noise.destroy(device);
                        desert_cloud_mask.destroy(device);
                    }
                    return Err(error);
                }
            },
        };
        let normal_east = match surface_normals {
            Some(normals) => {
                let texture = normals.east();
                let mapped = match texture.map_payload() {
                    Ok(mapped) => mapped,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                            cloud_a.destroy(device);
                            cloud_ba.destroy(device);
                            tiling_noise.destroy(device);
                            desert_cloud_mask.destroy(device);
                            height.destroy(device);
                        }
                        return Err(error.into());
                    }
                };
                match PinnedBgraTexture::create_from_preview(device, resources, LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, texture, mapped.bytes(), "surface-normal east hemisphere") {
                    Ok(texture) => texture,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                            cloud_a.destroy(device);
                            cloud_ba.destroy(device);
                            tiling_noise.destroy(device);
                            desert_cloud_mask.destroy(device);
                            height.destroy(device);
                        }
                        return Err(error);
                    }
                }
            }
            None => match PinnedBgraTexture::create(
                device,
                resources,
                LINEAR_PREVIEW_FORMAT,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                1,
                1,
                &[255, 128, 128, 255],
                "surface-normal east fallback",
            ) {
                Ok(texture) => texture,
                Err(error) => {
                    unsafe {
                        east.destroy(device);
                        west.destroy(device);
                        night.destroy(device);
                        cloud_a.destroy(device);
                        cloud_ba.destroy(device);
                        tiling_noise.destroy(device);
                        desert_cloud_mask.destroy(device);
                        height.destroy(device);
                    }
                    return Err(error);
                }
            },
        };
        let normal_west = match surface_normals {
            Some(normals) => {
                let texture = normals.west();
                let mapped = match texture.map_payload() {
                    Ok(mapped) => mapped,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                            cloud_a.destroy(device);
                            cloud_ba.destroy(device);
                            tiling_noise.destroy(device);
                            desert_cloud_mask.destroy(device);
                            height.destroy(device);
                            normal_east.destroy(device);
                        }
                        return Err(error.into());
                    }
                };
                match PinnedBgraTexture::create_from_preview(device, resources, LINEAR_PREVIEW_FORMAT, vk::SamplerAddressMode::CLAMP_TO_EDGE, texture, mapped.bytes(), "surface-normal west hemisphere") {
                    Ok(texture) => texture,
                    Err(error) => {
                        unsafe {
                            east.destroy(device);
                            west.destroy(device);
                            night.destroy(device);
                            cloud_a.destroy(device);
                            cloud_ba.destroy(device);
                            tiling_noise.destroy(device);
                            desert_cloud_mask.destroy(device);
                            height.destroy(device);
                            normal_east.destroy(device);
                        }
                        return Err(error);
                    }
                }
            }
            None => match PinnedBgraTexture::create(
                device,
                resources,
                LINEAR_PREVIEW_FORMAT,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                1,
                1,
                &[255, 128, 128, 255],
                "surface-normal west fallback",
            ) {
                Ok(texture) => texture,
                Err(error) => {
                    unsafe {
                        east.destroy(device);
                        west.destroy(device);
                        night.destroy(device);
                        cloud_a.destroy(device);
                        cloud_ba.destroy(device);
                        tiling_noise.destroy(device);
                        desert_cloud_mask.destroy(device);
                        height.destroy(device);
                        normal_east.destroy(device);
                    }
                    return Err(error);
                }
            },
        };
        let descriptor_set = match allocate_texture_descriptor_set(device, resources) {
            Ok(descriptor_set) => descriptor_set,
            Err(error) => {
                unsafe {
                    east.destroy(device);
                    west.destroy(device);
                    night.destroy(device);
                    cloud_a.destroy(device);
                    cloud_ba.destroy(device);
                    tiling_noise.destroy(device);
                    desert_cloud_mask.destroy(device);
                    height.destroy(device);
                    normal_east.destroy(device);
                    normal_west.destroy(device);
                }
                return Err(error);
            }
        };
        let east_info = [vk::DescriptorImageInfo::default()
            .sampler(east.sampler)
            .image_view(east.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let west_info = [vk::DescriptorImageInfo::default()
            .sampler(west.sampler)
            .image_view(west.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let night_info = [vk::DescriptorImageInfo::default()
            .sampler(night.sampler)
            .image_view(night.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let cloud_a_info = [vk::DescriptorImageInfo::default()
            .sampler(cloud_a.sampler)
            .image_view(cloud_a.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let cloud_ba_info = [vk::DescriptorImageInfo::default()
            .sampler(cloud_ba.sampler)
            .image_view(cloud_ba.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let tiling_noise_info = [vk::DescriptorImageInfo::default()
            .sampler(tiling_noise.sampler)
            .image_view(tiling_noise.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let desert_cloud_mask_info = [vk::DescriptorImageInfo::default()
            .sampler(desert_cloud_mask.sampler)
            .image_view(desert_cloud_mask.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let height_info = [vk::DescriptorImageInfo::default()
            .sampler(height.sampler)
            .image_view(height.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let normal_east_info = [vk::DescriptorImageInfo::default()
            .sampler(normal_east.sampler)
            .image_view(normal_east.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let normal_west_info = [vk::DescriptorImageInfo::default()
            .sampler(normal_west.sampler)
            .image_view(normal_west.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&tiling_noise_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&east_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&west_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&night_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&cloud_a_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&cloud_ba_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&desert_cloud_mask_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(7)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&height_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(8)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&normal_east_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(9)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&normal_west_info),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        let mut result = Self {
            east,
            resources: *resources,
            west,
            normal_east,
            normal_west,
            night,
            cloud_a,
            cloud_ba,
            tiling_noise,
            desert_cloud_mask,
            height,
            atmosphere: Self::atmosphere_lut(device, resources)?,
            weather_fields: None,
            cloud_height: None,
            descriptor_set,
            earth_resident: true,
        };
        if let Err(error) = result.load_cloud_height(device) {
            unsafe { result.destroy(device); }
            return Err(error);
        }
        result.bind_environment_fields(device);
        let info = [vk::DescriptorImageInfo::default()
            .sampler(result.atmosphere.sampler).image_view(result.atmosphere.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = [vk::WriteDescriptorSet::default().dst_set(descriptor_set).dst_binding(10)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&info)];
        unsafe { device.update_descriptor_sets(&write, &[]) };
        Ok(result)
    }

    unsafe fn destroy(&self, device: &Device) {
        self.east.destroy(device);
        self.west.destroy(device);
        self.normal_east.destroy(device);
        self.normal_west.destroy(device);
        self.night.destroy(device);
        self.cloud_a.destroy(device);
        self.cloud_ba.destroy(device);
        self.tiling_noise.destroy(device);
        self.desert_cloud_mask.destroy(device);
        self.height.destroy(device);
        self.atmosphere.destroy(device);
        if let Some(texture) = &self.weather_fields { texture.destroy(device); }
        if let Some(texture) = &self.cloud_height { texture.destroy(device); }
    }
}

fn validate_pinned_bgra_image_support(
    label: &str,
    width: u32,
    height: u32,
    max_image_dimension: u32,
    format_features: vk::FormatFeatureFlags,
) -> RendererResult<()> {
    if width > max_image_dimension || height > max_image_dimension {
        return Err(format!(
            "{label} {width}x{height} exceeds this GPU's max 2D image dimension {max_image_dimension}"
        )
        .into());
    }
    let required = vk::FormatFeatureFlags::TRANSFER_DST
        | vk::FormatFeatureFlags::SAMPLED_IMAGE
        | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
    if !format_features.contains(required) {
        return Err(
            "GPU cannot linearly sample and upload B8G8R8A8_SRGB raw textures".into(),
        );
    }
    Ok(())
}

impl StagingBuffer {
    unsafe fn create(
        device: &Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        bytes: &[u8],
    ) -> RendererResult<Self> {
        let size = u64::try_from(bytes.len()).map_err(|_| "star panorama exceeds Vulkan size")?;
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = device.create_buffer(&create_info, None)?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        // Prefer coherent host memory so the per-texture flush path below is
        // skipped; fall back to any host-visible type. Same bytes either way.
        let memory_type_index = match find_memory_type(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .or_else(|_| {
            find_memory_type(
                memory_properties,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            )
        }) {
            Ok(index) => index,
            Err(error) => {
                device.destroy_buffer(buffer, None);
                return Err(error);
            }
        };
        let host_coherent = memory_properties.memory_types[memory_type_index as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = match device.allocate_memory(&allocate_info, None) {
            Ok(memory) => memory,
            Err(error) => {
                device.destroy_buffer(buffer, None);
                return Err(error.into());
            }
        };
        if let Err(error) = device.bind_buffer_memory(buffer, memory, 0) {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
            return Err(error.into());
        }
        let mapped = match device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) {
            Ok(mapped) => mapped,
            Err(error) => {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
                return Err(error.into());
            }
        };
        ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
        if !host_coherent {
            let ranges = [vk::MappedMemoryRange::default()
                .memory(memory)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            if let Err(error) = device.flush_mapped_memory_ranges(&ranges) {
                device.unmap_memory(memory);
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
                return Err(error.into());
            }
        }
        device.unmap_memory(memory);
        Ok(Self { buffer, memory })
    }

    unsafe fn destroy(self, device: &Device) {
        device.destroy_buffer(self.buffer, None);
        device.free_memory(self.memory, None);
    }
}

impl ImageAllocation {
    unsafe fn create_array(
        device: &Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        format: vk::Format,
        width: u32,
        height: u32,
        array_layers: u32,
        usage: vk::ImageUsageFlags,
    ) -> RendererResult<Self> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(array_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = device.create_image(&image_info, None)?;
        let requirements = device.get_image_memory_requirements(image);
        let memory_type_index = match find_memory_type(memory_properties, requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            Ok(index) => index,
            Err(error) => { device.destroy_image(image, None); return Err(error); }
        };
        let memory = match device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(memory_type_index), None) {
            Ok(memory) => memory,
            Err(error) => { device.destroy_image(image, None); return Err(error.into()); }
        };
        if let Err(error) = device.bind_image_memory(image, memory, 0) {
            device.free_memory(memory, None); device.destroy_image(image, None); return Err(error.into());
        }
        Ok(Self { image, memory })
    }

    unsafe fn create_mipped(
        device: &Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        format: vk::Format,
        width: u32,
        height: u32,
        mip_levels: u32,
        needs_blit: bool,
    ) -> RendererResult<Self> {
        // TRANSFER_SRC opts the image out of lossless compression on several
        // drivers; request it only when the mip-bake blit loop will use it.
        let usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED
            | if needs_blit { vk::ImageUsageFlags::TRANSFER_SRC } else { vk::ImageUsageFlags::empty() };
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = device.create_image(&image_info, None)?;
        let requirements = device.get_image_memory_requirements(image);
        let memory_type_index = match find_memory_type(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) {
            Ok(index) => index,
            Err(error) => {
                device.destroy_image(image, None);
                return Err(error);
            }
        };
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = match device.allocate_memory(&allocate_info, None) {
            Ok(memory) => memory,
            Err(error) => {
                device.destroy_image(image, None);
                return Err(error.into());
            }
        };
        if let Err(error) = device.bind_image_memory(image, memory, 0) {
            device.free_memory(memory, None);
            device.destroy_image(image, None);
            return Err(error.into());
        }
        Ok(Self { image, memory })
    }

    unsafe fn destroy(self, device: &Device) {
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}

unsafe fn upload_staging_mip_image(
    device: &Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    staging_buffer: vk::Buffer,
    image: vk::Image,
    mips: &[StarPanoramaMip],
    mip_levels: u32,
) -> RendererResult<()> {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];
    let upload = (|| -> RendererResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(command_buffer, &begin_info)?;
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1);
        let to_transfer = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(range);
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_transfer],
        );
        let copy = mips
            .iter()
            .enumerate()
            .map(|(mip_level, mip)| {
                vk::BufferImageCopy::default()
                    .buffer_offset(mip.byte_offset)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(mip_level as u32)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: mip.width,
                        height: mip.height,
                        depth: 1,
                    })
            })
            .collect::<Vec<_>>();
        device.cmd_copy_buffer_to_image(
            command_buffer,
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &copy,
        );
        for level in mips.len() as u32..mip_levels {
            let previous = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(level - 1)
                .level_count(1).base_array_layer(0).layer_count(1);
            let to_source = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL).new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(image).subresource_range(previous);
            device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_source]);
            let layers = |mip| vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(mip).layer_count(1);
            let end = |mip: u32| vk::Offset3D { x: (mips[0].width >> mip).max(1) as i32,
                y: (mips[0].height >> mip).max(1) as i32, z: 1 };
            let blit = vk::ImageBlit::default().src_subresource(layers(level - 1))
                .src_offsets([vk::Offset3D::default(), end(level - 1)])
                .dst_subresource(layers(level)).dst_offsets([vk::Offset3D::default(), end(level)]);
            device.cmd_blit_image(command_buffer, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], vk::Filter::LINEAR);
            // Return the source to the common layout before the final whole-image barrier.
            let restore = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_READ).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL).new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image).subresource_range(previous);
            device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[restore]);
        }
        let to_sampled = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(image)
            .subresource_range(range);
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_sampled],
        );
        device.end_command_buffer(command_buffer)?;
        let command_buffers = [command_buffer];
        let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        device.queue_submit(queue, &submit_info, vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
        Ok(())
    })();
    if upload.is_err() {
        let _ = device.queue_wait_idle(queue);
    }
    device.free_command_buffers(command_pool, &[command_buffer]);
    upload
}

/// Memory for buffers the CPU reads back (the exposure meter, frame
/// captures). HOST_VISIBLE | HOST_COHERENT alone picked the first such type,
/// which with resizable BAR is device-local VRAM mapped over PCIe: every CPU
/// read was an uncached bus round trip, and the per-frame meter conversion
/// alone took ~20 % of the renderer's CPU. Prefer cached system memory.
pub(super) fn find_readback_memory_type(
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    compatible_types: u32,
) -> RendererResult<u32> {
    let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED;
    let pick = |required: vk::MemoryPropertyFlags, avoid_device_local: bool| (0..memory_properties.memory_type_count).find(|&index| {
        let properties = memory_properties.memory_types[index as usize].property_flags;
        compatible_types & (1 << index) != 0 && properties.contains(required)
            && !(avoid_device_local && properties.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
    });
    pick(wanted, true)
        .or_else(|| pick(wanted, false))
        .or_else(|| pick(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, true))
        .map_or_else(|| find_memory_type(memory_properties, compatible_types,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT), Ok)
}

fn find_memory_type(
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    compatible_types: u32,
    required: vk::MemoryPropertyFlags,
) -> RendererResult<u32> {
    for index in 0..memory_properties.memory_type_count {
        let supported = compatible_types & (1_u32 << index) != 0;
        let properties = memory_properties.memory_types[index as usize].property_flags;
        if supported && properties.contains(required) {
            return Ok(index);
        }
    }
    Err("no Vulkan memory type supports the required texture allocation properties".into())
}

impl OutputTarget {
    fn create(
        surface_loader: &ash::khr::surface::Instance,
        device: &DeviceState,
        surface: vk::SurfaceKHR,
        extent: vk::Extent2D,
        viewport: LogicalRect,
        debug_output_name: Option<&str>,
        debug_scene_name: &str,
    ) -> RendererResult<Self> {
        Self::create_with_old(
            surface_loader,
            device,
            surface,
            extent,
            viewport,
            debug_output_name,
            debug_scene_name,
            vk::SwapchainKHR::null(),
        )
    }

    fn create_with_old(
        surface_loader: &ash::khr::surface::Instance,
        device: &DeviceState,
        surface: vk::SurfaceKHR,
        requested_extent: vk::Extent2D,
        viewport: LogicalRect,
        debug_output_name: Option<&str>,
        debug_scene_name: &str,
        old_swapchain: vk::SwapchainKHR,
    ) -> RendererResult<Self> {
        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(device.physical_device, surface)?
        };
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(device.physical_device, surface)?
        };
        let surface_format = formats
            .iter()
            .copied()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first().copied())
            .ok_or("the Wayland surface has no Vulkan color format")?;
        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(device.physical_device, surface)?
        };
        let present_mode = if present_modes.contains(&vk::PresentModeKHR::FIFO) {
            vk::PresentModeKHR::FIFO
        } else {
            *present_modes
                .first()
                .ok_or("the Wayland surface has no presentation mode")?
        };
        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: requested_extent.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: requested_extent.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        };
        if debug_output_name.is_some()
            && !capabilities
                .supported_usage_flags
                .contains(vk::ImageUsageFlags::TRANSFER_SRC)
        {
            return Err("the Wayland swapchain does not support debug screenshot readback".into());
        }
        let capture_supported = capabilities.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_SRC);
        let image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | if capture_supported { vk::ImageUsageFlags::TRANSFER_SRC } else { vk::ImageUsageFlags::empty() };
        // FIFO already provides compositor pacing; asking for an extra image
        // duplicates a full-resolution BGRA surface per output and increases
        // VRAM without improving image quality. Use the surface minimum while
        // respecting the implementation's advertised bounds.
        let min_image_count = capabilities.min_image_count.max(1).min(
            if capabilities.max_image_count > 0 {
                capabilities.max_image_count
            } else {
                u32::MAX
            },
        );
        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|alpha| capabilities.supported_composite_alpha.contains(*alpha))
        .ok_or("the Wayland surface has no compatible alpha mode")?;
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(image_usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);
        let swapchain = unsafe {
            device
                .swapchain_loader
                .create_swapchain(&swapchain_info, None)?
        };
        let images = unsafe { device.swapchain_loader.get_swapchain_images(swapchain)? };
        let mut views = Vec::with_capacity(images.len());
        for &image in &images {
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .subresource_range(subresource_range);
            views.push(unsafe { device.device.create_image_view(&view_info, None)? });
        }
        let command_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(images.len() as u32);
        let command_buffers = unsafe { device.device.allocate_command_buffers(&command_info)? };
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let image_available = unsafe { device.device.create_semaphore(&semaphore_info, None)? };
        let mut render_finished = Vec::with_capacity(images.len());
        for _ in &images {
            render_finished.push(unsafe { device.device.create_semaphore(&semaphore_info, None)? });
        }
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight = unsafe { device.device.create_fence(&fence_info, None)? };
        let debug_capture = debug_output_name
            .map(|output_name| {
                DebugCapture::create(
                    &device.device,
                    device.memory_properties,
                    extent,
                    surface_format.format,
                    output_name,
                    debug_scene_name,
                )
            })
            .transpose()?;
        Ok(Self {
            surface,
            swapchain,
            format: surface_format.format,
            extent,
            images,
            views,
            command_buffers,
            image_available,
            render_finished,
            in_flight,
            needs_recreate: false,
            viewport,
            debug_capture,
            output_name: debug_output_name.unwrap_or("output").to_owned(),
            capture_supported,
            query_base: 0,
            last_gpu_ticks: [0; 4],
            // Presented-frame counter; queries are written+read only when
            // `query_clock % TIMESTAMP_SAMPLE_EVERY == 0`, otherwise the last
            // sample is reused by status_fields.
            query_clock: 0,
            query_pending: false,
            slow_waits: SlowWaitTracker::default(),
            hdr: None,
            last_meter: None,
        })
    }

    fn render(
        &mut self,
        device: &DeviceState,
        pipeline: &Pipeline,
        uniforms: FrameUniforms,
        body_selector: f32,
        camera: &CameraSettings,
    ) -> RendererResult<()> {
        unsafe {
            let capture_buffer = self.debug_capture.as_mut().and_then(|capture| {
                capture.is_due().then_some(capture.buffer)
            });
            // Fast path: the previous frame finished long before this tick,
            // so poll without blocking first. Healthy frames skip the 3 s
            // kernel wait and both clock reads; only a genuinely unsignaled
            // fence pays for Instant + the bounded blocking wait below.
            match device.device.wait_for_fences(&[self.in_flight], true, 0) {
                Ok(()) => {
                    self.slow_waits.record(0.0);
                }
                Err(vk::Result::TIMEOUT) => {
                    let fence_wait_started = Instant::now();
                    match device
                        .device
                        .wait_for_fences(&[self.in_flight], true, 3_000_000_000)
                    {
                        Ok(()) => {
                            let fence_wait_ms =
                                fence_wait_started.elapsed().as_secs_f32() * 1000.0;
                            if self.slow_waits.record(fence_wait_ms) {
                                eprintln!(
                                    "earth-native: slow fence wait {:.0} ms ({}x{}); compositor/GPU starved the fence",
                                    fence_wait_ms, self.extent.width, self.extent.height
                                );
                            }
                        }
                        Err(vk::Result::TIMEOUT) => {
                            self.slow_waits.record(3000.0);
                            eprintln!(
                                "earth-native: GPU fence timeout ({}x{}, ticks {:?}); skipping frame",
                                self.extent.width, self.extent.height, self.last_gpu_ticks
                            );
                            return Ok(());
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
                Err(error) => return Err(error.into()),
            }
            // The fence guarantees this output's previous submission (the one
            // that wrote these queries) has completed, so the prior frame's
            // GPU timing can be read back before this frame reuses the slots.
            // Sampled 1/N frames; in between the last sample stays put.
            let sample_gpu = self.query_clock % TIMESTAMP_SAMPLE_EVERY == 0;
            let mut ticks = [0u64; TIMESTAMP_QUERIES_PER_OUTPUT as usize];
            if self.query_pending
                && device
                .device
                .get_query_pool_results(
                    device.timestamp_pool,
                    self.query_base,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64,
                )
                .is_ok()
            {
                self.last_gpu_ticks = ticks;
                self.query_pending = false;
            }
            // The same fence covers the last frame's metering copy.
            if let Some(hdr) = self.hdr.as_mut() {
                if hdr.pending_preexposure.is_some() {
                    hdr.meter_preexposure = hdr.pending_preexposure;
                    self.last_meter = hdr.read_meter();
                }
            }
            if self.hdr.as_ref().map(|hdr| hdr.extent) != Some(self.extent) {
                if let Some(old) = self.hdr.take() {
                    old.destroy(&device.device);
                }
                self.hdr = Some(HdrTarget::create(&device.device, device.memory_properties, &pipeline.post, self.extent)?);
                self.last_meter = None;
            }
            let hdr = self.hdr.as_ref().expect("HDR target created above");
            // Non-blocking acquire: timeout 0 polls the compositor for an
            // image without stalling the render thread or churning the
            // swapchain. When the compositor is starving (not releasing
            // images - a known NVIDIA + Hyprland explicit-sync pattern) the
            // frame is skipped and retried on the next timer tick.
            //
            // Recreating the swapchain on mere starvation was a leak: each
            // rebuild allocated a fresh image set that the compositor keeps
            // holding, so VRAM accumulated to ~9 GiB over a long session.
            // Only a genuine surface change (OUT_OF_DATE) rebuilds now.
            let acquired = device.swapchain_loader.acquire_next_image(
                self.swapchain,
                0,
                self.image_available,
                vk::Fence::null(),
            );
            let (image_index, suboptimal) = match acquired {
                Ok(image) => image,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.needs_recreate = true;
                    return Ok(());
                }
                Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => {
                    // Compositor is holding all images; skip this frame. The
                    // next timer tick acquires again.
                    return Ok(());
                }
                Err(error) => return Err(error.into()),
            };
            self.needs_recreate |= suboptimal;
            if self.slow_waits.take_self_heal() {
                // A run of slow waits means the sync state behind this
                // swapchain is confused (driver/compositor starvation). A
                // rebuild recreates fences and sync objects from scratch -
                // the same recovery a service restart gives, without losing
                // the session. The fence we just waited on is still
                // signalled, so the normal frame path below stays correct.
                eprintln!(
                    "earth-native: self-healing swapchain for {}x{} after repeated slow GPU waits",
                    self.extent.width, self.extent.height
                );
                self.needs_recreate = true;
                return Ok(());
            }
            device.device.reset_fences(&[self.in_flight])?;
            let command_buffer = self.command_buffers[image_index as usize];
            device
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            // Re-recorded fresh on every submit; ONE_TIME_SUBMIT lets the driver
            // skip retaining internal state for this buffer across submits.
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .device
                .begin_command_buffer(command_buffer, &begin_info)?;
            if sample_gpu {
                device.device.cmd_reset_query_pool(
                    command_buffer,
                    device.timestamp_pool,
                    self.query_base,
                    TIMESTAMP_QUERIES_PER_OUTPUT,
                );
                device.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    device.timestamp_pool,
                    self.query_base,
                );
            }
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            hdr.record_begin(&device.device, command_buffer);
            // The opaque star pass overwrites every pixel of the scene target,
            // so the prior contents are never read; DONT_CARE skips a
            // full-resolution clear on every output every frame. (If the star
            // pass ever gains discarded pixels, restore CLEAR or add depth.)
            let color_attachment = [vk::RenderingAttachmentInfo::default()
                .image_view(hdr.attachment_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachment);
            device
                .device
                .cmd_begin_rendering(command_buffer, &rendering_info);
            let viewport = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.extent.width as f32,
                height: self.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissor = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            }];
            let mut shader_frame = ShaderFrame::from_uniforms(uniforms, self.viewport, body_selector);
            // The otherwise unused .w lane carries the camera pre-exposure.
            shader_frame.camera_position_distance[3] = camera.preexposure;
            let bytes = std::slice::from_raw_parts(
                (&shader_frame as *const ShaderFrame).cast::<u8>(),
                size_of::<ShaderFrame>(),
            );
            device.device.cmd_push_constants(
                command_buffer,
                pipeline.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytes,
            );
            device.device.cmd_set_viewport(command_buffer, 0, &viewport);
            device.device.cmd_set_scissor(command_buffer, 0, &scissor);
            let star_pipeline = if let Some(texture) = device.star_texture.as_ref() {
                device.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.layout,
                    0,
                    &[texture.descriptor_set],
                    &[],
                );
                pipeline.stars_textured
            } else {
                pipeline.stars_procedural
            };
            device.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                star_pipeline,
            );
            device.device.cmd_draw(command_buffer, 3, 1, 0, 0);
            // Catalogue stars: one additive quad each, on top of the sky and
            // under the planet (whose opaque disc covers them).
            if let Some(texture) = device.star_texture.as_ref().filter(|texture| texture.star_count > 0) {
                let mut star_frame = StarFrame::from_uniforms(uniforms, self.viewport, self.extent.width);
                // Radiance per unit of catalogue irradiance: pre-exposure times
                // the sunlight unit (pi) over one physical pixel's solid angle.
                let pixel_angle = 2.0 * uniforms.tan_half_fov_y
                    / (uniforms.canvas.height * star_frame.params[1]).max(1.0);
                if camera.post.tone[1] > 0.5 {
                    star_frame.params[2] = 1.0;
                    star_frame.params[3] = 1.0;
                } else {
                    star_frame.params[2] = camera.preexposure * std::f32::consts::PI / (pixel_angle * pixel_angle);
                }
                let star_bytes = std::slice::from_raw_parts(
                    (&star_frame as *const StarFrame).cast::<u8>(),
                    size_of::<StarFrame>(),
                );
                device.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.star_points);
                device.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.star_points_layout,
                    0,
                    &[texture.descriptor_set],
                    &[],
                );
                device.device.cmd_push_constants(
                    command_buffer,
                    pipeline.star_points_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    star_bytes,
                );
                device.device.cmd_draw(command_buffer, texture.star_count * 6, 1, 0, 0);
                // Different push-constant ranges make the layouts
                // incompatible: restore set 0 and the push constants the
                // body pass expects under the shared layout.
                device.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.layout,
                    0,
                    &[texture.descriptor_set],
                    &[],
                );
                device.device.cmd_push_constants(
                    command_buffer,
                    pipeline.layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytes,
                );
            }
            if sample_gpu {
                device.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    device.timestamp_pool,
                    self.query_base + 1,
                );
            }
            if let Some(earth_scissor) = self.earth_scissor(uniforms, body_selector) {
                device
                    .device
                    .cmd_set_scissor(command_buffer, 0, &[earth_scissor]);
                let earth_pipeline = if body_selector >= 1.0 {
                    // Jupiter has no virtual-texture bundle; always render the
                    // procedural gas-giant material.
                    pipeline.earth_procedural
                } else if let (Some(textures), Some(virtual_texture)) = (
                    device.day_color_textures.as_ref(),
                    device.virtual_texture.as_ref(),
                ) {
                    device.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        0,
                        &[textures.descriptor_set, virtual_texture.descriptor_set],
                        &[],
                    );
                    pipeline.earth_textured
                } else {
                    pipeline.earth_procedural
                };
                device.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    earth_pipeline,
                );
                device.device.cmd_draw(command_buffer, 3, 1, 0, 0);
            }
            device.device.cmd_end_rendering(command_buffer);
            if sample_gpu {
                device.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    device.timestamp_pool,
                    self.query_base + 2,
                );
            }
            hdr.record_resolve(&device.device, command_buffer);
            hdr.record_bloom(&device.device, command_buffer, &pipeline.post);
            let to_color = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(self.images[image_index as usize])
                .subresource_range(range);
            device.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_color],
            );
            let present_attachment = [vk::RenderingAttachmentInfo::default()
                .image_view(self.views[image_index as usize])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)];
            let present_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                })
                .layer_count(1)
                .color_attachments(&present_attachment);
            device.device.cmd_begin_rendering(command_buffer, &present_info);
            device.device.cmd_set_viewport(command_buffer, 0, &viewport);
            device.device.cmd_set_scissor(command_buffer, 0, &scissor);
            let mut post_frame = camera.post;
            post_frame.projection = [uniforms.tan_half_fov_x, uniforms.tan_half_fov_y, uniforms.focus_x, uniforms.focus_y];
            post_frame.canvas = [uniforms.canvas.x, uniforms.canvas.y, uniforms.canvas.width, uniforms.canvas.height];
            post_frame.viewport = [self.viewport.x, self.viewport.y, self.viewport.width, self.viewport.height];
            device.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.post.pipeline);
            device.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.post.layout,
                0,
                &[hdr.descriptor_set],
                &[],
            );
            device.device.cmd_push_constants(
                command_buffer,
                pipeline.post.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                std::slice::from_raw_parts((&post_frame as *const PostFrame).cast::<u8>(), size_of::<PostFrame>()),
            );
            device.device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.device.cmd_end_rendering(command_buffer);
            if sample_gpu {
                device.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    device.timestamp_pool,
                    self.query_base + 3,
                );
            }
            if let Some(capture_buffer) = capture_buffer {
                let to_transfer = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(self.images[image_index as usize])
                    .subresource_range(range);
                device.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_transfer],
                );
                let copy = [vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: self.extent.width,
                        height: self.extent.height,
                        depth: 1,
                    })];
                device.device.cmd_copy_image_to_buffer(
                    command_buffer,
                    self.images[image_index as usize],
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    capture_buffer,
                    &copy,
                );
                let host_read = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .buffer(capture_buffer)
                    .size(vk::WHOLE_SIZE);
                device.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[host_read],
                    &[],
                );
                let to_present = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(self.images[image_index as usize])
                    .subresource_range(range);
                device.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_present],
                );
            } else {
                let to_present = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(self.images[image_index as usize])
                    .subresource_range(range);
                device.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_present],
                );
            }
            device.device.end_command_buffer(command_buffer)?;
            let wait_semaphores = [self.image_available];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [self.render_finished[image_index as usize]];
            let command_buffers = [command_buffer];
            let submit = [vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)];
            device
                .device
                .queue_submit(device.queue, &submit, self.in_flight)?;
            if let Some(hdr) = self.hdr.as_mut() {
                hdr.pending_preexposure = Some(camera.preexposure);
            }
            self.query_clock = self.query_clock.wrapping_add(1);
            self.query_pending |= sample_gpu;
            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            match device
                .swapchain_loader
                .queue_present(device.queue, &present_info)
            {
                Ok(_) => {}
                Err(vk::Result::SUBOPTIMAL_KHR) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.needs_recreate = true;
                }
                Err(error) => return Err(error.into()),
            }
            if capture_buffer.is_some() {
                match device
                    .device
                    .wait_for_fences(&[self.in_flight], true, 3_000_000_000)
                {
                    Ok(()) => {
                        if let Some(capture) = self.debug_capture.as_mut() {
                            capture.write_jpeg(&device.device, self.extent, uniforms, body_selector, self.viewport)?;
                        }
                        if self.debug_capture.as_ref().is_some_and(|capture| !capture.periodic && !capture.pending) {
                            if let Some(mut capture) = self.debug_capture.take() {
                                if let Some(scene) = capture.resume_scene.take() {
                                    capture.periodic = true;
                                    capture.set_scene(&scene);
                                    self.debug_capture = Some(capture);
                                } else {
                                    capture.destroy(&device.device);
                                }
                            }
                        }
                    }
                    Err(vk::Result::TIMEOUT) => {
                        eprintln!("earth-native: GPU fence timeout before debug readback; skipping screenshot");
                    }
                    Err(error) => return Err(error.into()),
                }
            }
        }
        Ok(())
    }

    fn earth_scissor(&self, uniforms: FrameUniforms, body_selector: f32) -> Option<vk::Rect2D> {
        // Saturn's ring extends beyond two equatorial radii.
        // the scissor is normally sized for. Use a ring-sized radius so the
        // ring isn't cut off at a square box around the planet.
        let scissor_radius = if (body_selector - 4.0).abs() < 0.01 {
            REFERENCE_RING_RADIUS
        } else if body_selector < 0.5 && uniforms.aurora_valid_unix_utc > 0 {
            SCENE_EARTH_RADIUS * (1.0 + 320.0 / 6378.137)
        } else {
            REFERENCE_GLOW_RADIUS
        };
        let Some((half_ndc_x, half_ndc_y)) = projected_sphere_half_ndc(
            uniforms.camera_distance,
            scissor_radius,
            uniforms.tan_half_fov_x,
            uniforms.tan_half_fov_y,
        ) else {
            return Some(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            });
        };
        let center_x = uniforms.focus_x;
        let center_y = uniforms.focus_y;
        let half_width = uniforms.canvas.width * half_ndc_x * 0.5;
        let half_height = uniforms.canvas.height * half_ndc_y * 0.5;
        let left = center_x - half_width;
        let right = center_x + half_width;
        let top = center_y - half_height;
        let bottom = center_y + half_height;
        let local_left = ((left - self.viewport.x) / self.viewport.width * self.extent.width as f32)
            .floor()
            .clamp(0.0, self.extent.width as f32) as u32;
        let local_right = ((right - self.viewport.x) / self.viewport.width
            * self.extent.width as f32)
            .ceil()
            .clamp(0.0, self.extent.width as f32) as u32;
        let local_top = ((top - self.viewport.y) / self.viewport.height * self.extent.height as f32)
            .floor()
            .clamp(0.0, self.extent.height as f32) as u32;
        let local_bottom = ((bottom - self.viewport.y) / self.viewport.height
            * self.extent.height as f32)
            .ceil()
            .clamp(0.0, self.extent.height as f32) as u32;
        (local_left < local_right && local_top < local_bottom).then_some(vk::Rect2D {
            offset: vk::Offset2D {
                x: local_left as i32,
                y: local_top as i32,
            },
            extent: vk::Extent2D {
                width: local_right - local_left,
                height: local_bottom - local_top,
            },
        })
    }

    unsafe fn destroy_swapchain_resources(
        &self,
        device: &Device,
        command_pool: vk::CommandPool,
        swapchain_loader: &ash::khr::swapchain::Device,
    ) {
        if let Some(capture) = self.debug_capture.as_ref() {
            capture.destroy(device);
        }
        if let Some(hdr) = self.hdr.as_ref() {
            hdr.destroy(device);
        }
        device.destroy_fence(self.in_flight, None);
        device.destroy_semaphore(self.image_available, None);
        for semaphore in &self.render_finished {
            device.destroy_semaphore(*semaphore, None);
        }
        device.free_command_buffers(command_pool, &self.command_buffers);
        for view in &self.views {
            device.destroy_image_view(*view, None);
        }
        swapchain_loader.destroy_swapchain(self.swapchain, None);
    }

    unsafe fn destroy(
        self,
        device: &Device,
        command_pool: vk::CommandPool,
        swapchain_loader: &ash::khr::swapchain::Device,
    ) {
        self.destroy_swapchain_resources(device, command_pool, swapchain_loader);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposure_adapts_to_light_fast_and_to_dark_slowly() {
        let start = Instant::now();
        let mut exposure = ExposureController::new();
        exposure.snap_frames = 0;
        exposure.ev = 17.0;
        // Panning from the night sky back to the daylit Earth (no cut):
        // at most 1.5 stops over on the first frame, within 0.1 stop by 0.4 s.
        let day = post::MeterReading { log2_luminance: ExposureController::DAY_LUMINANCE_LOG2, coverage: 1.0 };
        exposure.update(Some(day), None, start);
        exposure.update(Some(day), None, start + Duration::from_millis(33));
        assert!(exposure.ev <= 1.5, "first frames {:.2} EV over", exposure.ev);
        exposure.update(Some(day), None, start + Duration::from_millis(400));
        assert!(exposure.ev < 0.1, "{:.2}", exposure.ev);
        // Back to the night: still the slow ~0.9 s easing.
        let night = post::MeterReading { log2_luminance: -21.4, coverage: 1.0 };
        exposure.update(Some(night), None, start + Duration::from_millis(433));
        exposure.update(Some(night), None, start + Duration::from_millis(533));
        assert!(exposure.ev < 3.0, "{:.2}", exposure.ev);
    }

    #[test]
    fn moon_basis_undoes_the_scene_longitude_mirror() {
        let uniforms = FrameUniforms::default();
        let earth = ShaderFrame::from_uniforms(uniforms, LogicalRect::default(), 0.0);
        assert_eq!(&earth.moon_body_x[..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&earth.moon_body_y[..3], &[0.0, -1.0, 0.0]);
        let planet = ShaderFrame::from_uniforms(uniforms, LogicalRect::default(), 1.0);
        assert_eq!(&planet.moon_body_y[..3], &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn solar_parallax_uses_selected_body_kilometres() {
        let uniforms = FrameUniforms { camera_position: [0.0, 4.0, 0.0], ..FrameUniforms::default() };
        for body in crate::body::Body::ALL {
            let frame = ShaderFrame::from_uniforms(uniforms, LogicalRect::default(), body.id());
            let camera_km = 4.0 / SCENE_EARTH_RADIUS as f64 * body.equatorial_radius_km();
            let sun_km = uniforms.sun_distance_earth_radii as f64 * crate::astronomy::EARTH_EQUATORIAL_RADIUS_KM;
            let expected = -camera_km / sun_km.hypot(camera_km);
            assert!((frame.celestial_sun_view[1] as f64 - expected).abs() < 1.0e-7);
        }
    }

    #[test]
    fn calibrated_5_5x_projection_matches_the_main_capture_scale() {
        let canvas_width = 4520.0_f32;
        let canvas_height = 2560.0_f32;
        let tan_half_fov_y = (25.0_f32.to_radians()).tan() / (16.0 / 9.0);
        let tan_half_fov_x = tan_half_fov_y * (canvas_width / canvas_height);

        let (surface_x, surface_y) = projected_sphere_half_ndc(
            5.5,
            REFERENCE_SURFACE_RADIUS,
            tan_half_fov_x,
            tan_half_fov_y,
        )
        .expect("the reference surface remains outside the camera");
        let (_, atmosphere_y) = projected_sphere_half_ndc(
            5.5,
            REFERENCE_ATMOSPHERE_RADIUS,
            tan_half_fov_x,
            tan_half_fov_y,
        )
        .expect("the reference atmosphere remains outside the camera");

        assert!((canvas_width * surface_x - 1398.26).abs() < 1.0);
        assert!((canvas_height * surface_y - 1398.26).abs() < 1.0);
        assert!((atmosphere_y / surface_y - 1.016).abs() < 0.001);
    }

    #[test]
    fn slot_allocator_is_deterministic_and_keeps_pending_slots_reserved() {
        let first = TileKey::new(1, 0, 0, 0);
        let second = TileKey::new(1, 0, 1, 0);
        let third = TileKey::new(1, 0, 2, 0);
        let mut allocator = VtSlotAllocator::new(2).unwrap();
        let a = allocator.request(first).unwrap();
        assert_eq!(a.slot, 0);
        assert!(allocator.request(first).is_none());
        assert!(allocator.mark_resident(a.slot, first));
        let b = allocator.request(second).unwrap();
        assert_eq!(b.slot, 1);
        assert!(allocator.mark_resident(b.slot, second));
        let c = allocator.request(third).unwrap();
        assert_eq!(c.slot, 0);
        assert_eq!(c.evicted, Some(first));
        assert!(matches!(allocator.states()[0], VtSlotState::Pending(key) if key == third));
        let retry = allocator.request(first).unwrap();
        assert_eq!(retry.slot, 1);
        assert_eq!(retry.evicted, Some(second));
        assert!(matches!(allocator.states()[0], VtSlotState::Pending(key) if key == third));
        assert!(allocator.mark_resident(c.slot, third));
    }

    #[test]
    fn slot_allocator_evicts_tiles_that_left_the_view() {
        let coarse = TileKey::new(1, 5, 0, 0);
        let fine = TileKey::new(1, 0, 7, 3);
        let next = TileKey::new(1, 0, 8, 3);
        let mut allocator = VtSlotAllocator::new(2).unwrap();
        let a = allocator.request(coarse).unwrap();
        assert!(allocator.holds(coarse));
        assert!(allocator.mark_resident(a.slot, coarse));
        let b = allocator.request(fine).unwrap();
        assert!(allocator.mark_resident(b.slot, fine));
        // The coarse tile was uploaded first but is still on screen.
        allocator.touch_visible(&[coarse].into_iter().collect());
        let c = allocator.request(next).unwrap();
        assert_eq!(c.evicted, Some(fine));
        assert!(allocator.holds(coarse));
        assert!(allocator.holds(next));
        assert!(!allocator.holds(fine));
    }

    #[test]
    fn body_residency_experiment_defaults_off() {
        use std::ffi::OsString;
        assert!(!Renderer::body_residency_flag(None));
        assert!(!Renderer::body_residency_flag(Some(OsString::from("0"))));
        assert!(!Renderer::body_residency_flag(Some(OsString::from(""))));
        assert!(Renderer::body_residency_flag(Some(OsString::from("1"))));
        assert!(Renderer::body_residency_flag(Some(OsString::from("true"))));
        assert!(Renderer::body_residency_flag(Some(OsString::from("TRUE"))));
    }

    #[test]
    fn moon_albedo_is_earth_only_for_residency_eviction() {
        // EXP-001 evicts the Moon albedo off-Earth; the star shader must keep
        // gating its only sample on `body == 0`.
        let stars = include_str!("../shaders/stars_textured.frag");
        assert!(stars.contains("if (body == 0 && moon_alpha > 0.0)"));
        assert!(stars.contains("textureGrad(moon_albedo, moon_uv, moon_dx, moon_dy)"));
    }

    #[test]
    fn star_panorama_sampling_folds_antimeridian_gradients() {
        // fract() wraps the panorama value but not its screen derivative;
        // implicit LOD would collapse to the coarsest mip in a sky strip.
        let stars = include_str!("../shaders/stars_textured.frag");
        assert!(stars.contains("textureGrad(star_panorama, panorama_uv + erosion * vec2(1.0, 1.0), panorama_dx, panorama_dy)"));
        assert!(!stars.contains("texture(star_panorama,"));
    }

    #[test]
    fn planet_sampling_folds_antimeridian_gradients() {
        // atan's branch cut spikes implicit derivatives at the wraparound;
        // every equirect body sampler must use wrap-folded explicit gradients.
        let earth = include_str!("../shaders/earth.frag");
        for sampler in ["planet_texture"] {
            assert!(
                earth.contains(&format!("textureGrad({sampler}, uv, planet_dx, planet_dy)")),
                "{sampler} must sample with folded gradients"
            );
            assert!(
                !earth.contains(&format!("texture({sampler},")),
                "{sampler} must not use implicit LOD across the cut"
            );
        }
        assert!(earth.contains("textureGrad(saturn_ring_texture, ring_uv, ring_dx, ring_dy)"));
        assert!(!earth.contains("texture(saturn_ring_texture,"));
    }

    #[test]
    fn seam_gradients_are_computed_in_uniform_flow() {
        // dFdx/dFdy inside a divergent branch are undefined at silhouette
        // quads; textureGrad with precomputed gradients is well-defined.
        // These pins guard the evaluation order, not the rendered result.
        let earth = include_str!("../shaders/earth.frag");
        let grad = earth.find("vec2 planet_dx = dFdx(planet_uv)").unwrap();
        let branch = earth.find("if (body_distance > 0.0)").unwrap();
        assert!(grad < branch, "planet gradients must precede the hit branch");
        let stars = include_str!("../shaders/stars_textured.frag");
        let grad = stars.find("vec2 moon_dx = dFdx(moon_uv)").unwrap();
        let branch = stars.find("if (body == 0 && moon_alpha > 0.0)").unwrap();
        assert!(grad < branch, "moon gradients must precede the disc branch");
        let grad = stars.find("vec2 panorama_dx = dFdx(panorama_uv)").unwrap();
        let branch = stars.find("if (closest < 0.0 && hidden > 0.0)").unwrap();
        assert!(grad < branch, "panorama gradients must precede the early-out");
        let moon_grad = stars.find("vec2 moon_dx = dFdx(moon_uv)").unwrap();
        assert!(moon_grad < branch, "moon gradients must precede the planet occlusion early-out");
        let ring_grad = earth.find("vec2 ring_dx = dFdx(ring_uv)").unwrap();
        let ring_gate = earth.find("if (abs(ray.z) < 1.0e-7) return vec4(0.0);").unwrap();
        assert!(ring_grad < ring_gate, "ring gradients must precede the silhouette gates");
        let earth_width = earth.find("float horizon_width = max(fwidth(sunlight), 1.0e-4);").unwrap();
        let earth_discard = earth.find("if (atmosphere_distance < 0.0)").unwrap();
        assert!(earth_width < earth_discard, "earth fwidth must precede the atmosphere discard");
        let textured = include_str!("../shaders/earth_textured.frag");
        let map_dx = textured.find("vec2 map_dx = dFdx(map_uv)").unwrap();
        let tex_discard = textured.find("if (!in_air && (frame.material_state.z < 0.5 ||").unwrap();
        assert!(map_dx < tex_discard, "earth_textured map gradients must precede discard");
    }

    #[test]
    fn clouds_come_from_imagery_and_stars_are_independent_of_weather() {
        let earth = include_str!("../shaders/earth_textured.frag");
        let cloud_start = earth.find("float sample_cloud_density(").unwrap();
        let cloud_end = earth[cloud_start..].find("float city_signal_at(").unwrap() + cloud_start;
        // Model weather fields drive lightning only; clouds come from observed
        // imagery (NOAA GMGSI) when the feed provides it, else the NASA map.
        assert!(!earth[cloud_start..cloud_end].contains("frame.material_state"));
        assert!(!earth[cloud_start..cloud_end].contains("weather_fields"));
        assert!(earth.contains(
            "return live_clouds() ? live_cloud_opacity(mesh_uv0, cloud_normal) : nasa_cloud_opacity(mesh_uv0);"));
        let noise_start = earth.find("float cloud_noise(").unwrap();
        let noise_end = earth[noise_start..].find("float cloud_detail(").unwrap() + noise_start;
        // Cloud detail is projected from the sphere, never sheared lat/lon.
        assert!(earth[noise_start..noise_end].contains("vec3 n"));
        assert!(!earth[noise_start..noise_end].contains("cos_lat"));
        assert!(earth.contains("binding = 4) uniform sampler2D clouds_a"));
        assert!(earth.contains("binding = 11) uniform sampler2D weather_fields"));
        assert!(!earth.contains("texture(clouds_a, cell_center_uv)"));
        let stars = include_str!("../shaders/stars_textured.frag");
        assert!(stars.contains("fract(0.5 - longitude * inv_two_pi + star_dome_yaw_turns)"));
    }

    #[test]
    fn cloud_mips_use_continuous_gradients_before_tiling() {
        let shader = include_str!("../shaders/earth_textured.frag");
        // The NASA cloud map goes through sample_static (static VT or tail).
        let start = shader.find("vec4 sample_static(").unwrap();
        let end = shader[start..].find("// Terrain normal").unwrap() + start;
        let sampling = &shader[start..end];
        assert!(sampling.contains("dx.x -= round(dx.x)"));
        assert!(sampling.contains("dy.x -= round(dy.x)"));
        assert!(sampling.contains("textureGrad(tail, uv, dx, dy)"));
        assert!(!sampling.contains("texture("), "cloud maps must not derive mips from wrapped UVs");
        let start = shader.find("vec4 sample_nasa_clouds(").unwrap();
        let clouds = &shader[start..start + 200];
        assert!(clouds.contains("dFdx(map_uv), dFdy(map_uv)"));
        // Adjacent pixels straddling longitude zero must keep their small
        // footprint, including the half-scale mist and highly tiled noise.
        for tiling in [0.5_f32, 2.0, 4.0, 6.0, 14.0, 33.0] {
            for raw_delta in [0.0002_f32, -0.9998, 0.9998] {
                let gradient = (raw_delta - raw_delta.round()) * tiling;
                assert!((gradient.abs() - 0.0002 * tiling).abs() < 2.0e-6);
            }
        }
    }

    #[test]
    fn longitude_half_maps_do_not_filter_the_opposite_edge() {
        let shader = include_str!("../shaders/earth_textured.frag");
        assert!(shader.contains("preview_uv = half_map_uv(preview_uv, vec2(textureSize(day_color_west, 0)), dx, dy)"));
        assert!(shader.contains("ceil(max(log2(max(footprint, 1.0)), 0.0))"));
        assert!(shader.contains("clamp(uv.x, inset, 1.0 - inset)"));
        // The inset protects the coarser of both trilinear levels, not just
        // the base image. A one-texel mip must sample its center exactly.
        for width in [1024.0_f32, 4096.0] {
            for footprint in [0.01_f32, 1.0, 1.1, 3.9, 16.0, 1024.0, 8192.0] {
                let mip = footprint.max(1.0).log2().ceil();
                let inset = (0.5 * mip.exp2() / width).min(0.5);
                let coarse_width = (width / mip.exp2()).max(1.0);
                for u in [0.0_f32, 0.000001, 0.5, 0.999999, 1.0] {
                    let sampled = u.clamp(inset, 1.0 - inset);
                    assert!(sampled * coarse_width >= 0.5);
                    assert!((1.0 - sampled) * coarse_width >= 0.5);
                }
            }
        }
    }

    #[test]
    fn textured_earth_marches_a_physical_atmosphere() {
        let shader = include_str!("../shaders/earth_textured.frag");
        // Precomputed Bruneton/Hillaire tables from sky.rs, not an analytic glow.
        assert!(shader.contains("binding = 17) uniform sampler2D sky_transmittance"));
        assert!(shader.contains("binding = 18) uniform sampler2D sky_multiscatter"));
        assert!(shader.contains("binding = 19) uniform sampler2D sky_irradiance"));
        assert!(shader.contains("void march("));
        assert!(shader.contains("multiscatter("));
        assert!(shader.contains("sky_irradiance_at("));
        assert!(shader.contains("rayleigh_phase"));
        assert!(shader.contains("aerosol_phase"));
        assert!(shader.contains("OZONE"));
        // Scene unit: white Lambertian under the zenith Sun at 1 AU is 1, so
        // in-scattered radiance (per steradian) carries the factor pi.
        assert!(shader.contains("PI * sun_irradiance"));
        assert!(shader.contains("solar_visibility"));
        // Per-channel transmittance leaves through the second blend source.
        assert!(shader.contains("layout(location = 0, index = 1) out vec4 out_transmittance"));
        // Night side: lightning, city underglow, OVATION aurora, airglow.
        assert!(shader.contains("storm_random(strike_cell, uint(phase))"));
        assert!(!shader.contains("phase * 5.31"));
        assert!(shader.contains("city_signal_cloud"));
        assert!(shader.contains("cloud_shadow"));
        assert!(shader.contains("cox_munk_glint"));
        assert!(!shader.contains("wave_slope"));
        assert!(shader.contains("aurora_emission"));
        assert!(shader.contains("OVATION"));
        assert!(shader.contains("aurora_top_radius"));
        assert!(shader.contains("557.7 nm"));
        assert!(shader.contains("textureLod(weather_fields, uv, 3.0).a"));
        assert!(shader.contains("night_airglow"));
        assert!(shader.contains("moonlight_scale()"));
        assert!(!shader.contains("aurora_oval"));

        // Cloud over ground, then the air in front of both; the surface is
        // opaque (transmittance zero) so stars cannot show through.
        let cloud_composite = shader.find("under = mix(under, cloud_colour, cloud_opacity);").unwrap();
        let air_composite = shader.find("radiance = above_radiance + above_transmittance * under;").unwrap();
        let opaque = shader[air_composite..].find("transmittance = vec3(0.0);").unwrap() + air_composite;
        assert!(cloud_composite < air_composite);
        assert!(air_composite < opaque);
    }

    #[test]
    fn textured_sun_uses_layered_reference_corona() {
        let shader = include_str!("../shaders/stars_textured.frag");
        // Camera-relative Sun/Moon geometry arrives precomputed per frame.
        assert!(shader.contains("frame.celestial_sun_view.xyz"));
        assert!(shader.contains("frame.celestial_sun_view.w"));
        assert!(shader.contains("frame.celestial_moon_view.xyz"));
        assert!(shader.contains("frame.celestial_moon_view.w"));
        // The corona stack stays gated behind the precomputed 40-radius
        // threshold instead of running on every fragment.
        assert!(shader.contains("frame.moon_body_z.w"));
        // Eclipse geometry still derives scene-space positions from the
        // Earth-fixed directions inside the moon-disc branch.
        assert!(shader.contains("frame.celestial_distances.x"));
        assert!(shader.contains("frame.celestial_distances.y"));
        assert!(shader.contains("sun_position"));
        assert!(shader.contains("moon_position"));
        assert!(shader.contains("sun_from_moon"));
        assert!(shader.contains("limb_darkening"));
        assert!(shader.contains("sun_aureole"));
        assert!(shader.contains("sun_shoulder"));
        assert!(shader.contains("sun_core * vec3(1.55, 1.45, 1.28)"));
        assert!(shader.contains("sun_inner_corona * vec3(0.55, 0.30, 0.08)"));
         // Wide halo kept tight and faint neutral-warm so a Sun in-frame does
         // not paint a muddy vignette across the night sky; deep sky returns
         // to black (the old 35/14-radius gold stack read as brown fog).
         assert!(shader.contains("sun_aureole * vec3(0.0032, 0.0028, 0.0024)"));
         assert!(shader.contains("sun_outer_corona * vec3(0.022, 0.016, 0.010)"));
        assert!(shader.contains("asset_to_equator_of_date"));
        // Sidereal de-rotation uses the precomputed sine/cosine pair.
        assert!(shader.contains("frame.moon_body_x.w"));
        assert!(shader.contains("frame.moon_body_y.w"));
        assert!(shader.contains("frame.moon_body_x.xyz"));
        assert!(shader.contains("eclipse_visibility"));
        assert!(shader.contains("earthshine"));
        // Lunar opposition surge and directional bluish earthshine.
        assert!(shader.contains("phase_surge"));
        assert!(shader.contains("earth_facing"));
        assert!(!shader.contains("pow(max(sun_cosine, 0.0), 18000.0)"));
    }

    #[test]
    fn physical_celestial_frame_fits_the_target_gpu_push_constant_budget() {
        assert_eq!(size_of::<ShaderFrame>(), 16 * 16);
        assert_eq!(size_of::<ShaderFrame>() % 16, 0);
        assert!(size_of::<ShaderFrame>() <= 256);
    }

    #[test]
    fn earth_sky_sun_and_lighting_share_the_texture_longitude_mirror() {
        let mut uniforms = FrameUniforms::default();
        uniforms.sun_direction = [0.6, 0.8, 0.0];
        uniforms.moon_direction = [0.0, 1.0, 0.0];
        let viewport = LogicalRect {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        };
        let earth = ShaderFrame::from_uniforms(uniforms, viewport, 0.0);
        let jupiter = ShaderFrame::from_uniforms(uniforms, viewport, 1.0);
        assert!((earth.sun_direction[0] - 0.6).abs() < 1.0e-6);
        assert!((earth.sun_direction[1] + 0.8).abs() < 1.0e-6);
        assert!((earth.sun_direction[3] - 0.0).abs() < 1.0e-6);
        assert!((jupiter.sun_direction[1] - 0.8).abs() < 1.0e-6);
        assert!((jupiter.sun_direction[3] - 1.0).abs() < 1.0e-6);
        // Distant Sun: camera-relative view tracks the mirrored ECEF vector.
        assert!(earth.celestial_sun_view[1] < -0.5);
        assert!(jupiter.celestial_sun_view[1] > 0.5);
        // New Moon stays next to the Sun: both get the same longitude mirror.
        assert!(earth.camera_right[3] < -0.5);
        assert!(jupiter.camera_right[3] > 0.5);
        let shader = include_str!("../shaders/earth_textured.frag");
        assert!(
            !shader.contains("vec3(sun_raw.x, -sun_raw.y, sun_raw.z)"),
            "Earth lighting must not Y-mirror again; ShaderFrame already did"
        );
    }

    #[test]
    fn pinned_bgra_texture_requires_a_supported_native_image() {
        let supported = vk::FormatFeatureFlags::TRANSFER_DST
            | vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
        assert!(validate_pinned_bgra_image_support("test", 16_384, 8_192, 16_384, supported).is_ok());
        assert!(validate_pinned_bgra_image_support("test", 16_384, 8_192, 8_192, supported).is_err());
        assert!(validate_pinned_bgra_image_support(
            "test",
            16_384,
            8_192,
            16_384,
            vk::FormatFeatureFlags::SAMPLED_IMAGE
        )
        .is_err());
    }

    #[test]
    fn slow_wait_tracker_flags_only_waits_over_the_threshold() {
        let mut tracker = SlowWaitTracker::default();
        assert!(!tracker.record(0.1));
        assert!(!tracker.record(SLOW_WAIT_THRESHOLD_MS - 1.0));
        assert!(tracker.record(SLOW_WAIT_THRESHOLD_MS));
        assert!(tracker.record(3000.0));
    }

    #[test]
    fn slow_wait_tracker_self_heals_after_the_consecutive_run() {
        let mut tracker = SlowWaitTracker::default();
        for _ in 0..SELF_HEAL_CONSECUTIVE_SLOW_WAITS - 1 {
            tracker.record(400.0);
        }
        assert!(!tracker.take_self_heal());
        tracker.record(400.0);
        assert!(tracker.take_self_heal());
        // The trip is consumed exactly once.
        assert!(!tracker.take_self_heal());
    }

    #[test]
    fn slow_wait_tracker_resets_its_run_on_a_fast_wait() {
        let mut tracker = SlowWaitTracker::default();
        tracker.record(400.0);
        tracker.record(400.0);
        tracker.record(0.2);
        tracker.record(400.0);
        assert!(!tracker.take_self_heal());
    }

}
