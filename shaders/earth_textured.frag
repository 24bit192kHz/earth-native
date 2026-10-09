#version 460

// Earth: surface, clouds, atmosphere, airglow and aurora, in scene-linear
// radiance. Unit: a white Lambertian surface under a zenith Sun at 1 AU is
// 1.0 (so solar irradiance is pi); every output is multiplied by the camera
// pre-exposure carried in camera_position_distance.w. The camera stage
// (post.frag) applies lens glare, the tone curve and quantisation.
//
// Output uses dual-source blending: location 0 index 0 is premultiplied
// radiance, index 1 the per-channel transmittance of what lies behind (the
// air reddens the Sun and stars seen through the limb).

layout(location = 0) noperspective in vec2 in_uv;
layout(location = 0, index = 0) out vec4 out_color;
layout(location = 0, index = 1) out vec4 out_transmittance;

// Bindings one through three are optional fixed-resolution source previews.
// Production EarthVT atlases replace this stage-three bridge.
layout(set = 0, binding = 0) uniform sampler2D tiling_noise;
layout(set = 0, binding = 1) uniform sampler2D day_color_east;
layout(set = 0, binding = 2) uniform sampler2D day_color_west;
layout(set = 0, binding = 3) uniform sampler2D night_emission;
layout(set = 0, binding = 4) uniform sampler2D clouds_a;
layout(set = 0, binding = 5) uniform sampler2D clouds_ba;
layout(set = 0, binding = 6) uniform sampler2D desert_cloud_mask;
layout(set = 0, binding = 7) uniform sampler2D terrain_height;
layout(set = 0, binding = 8) uniform sampler2D surface_normal_east;
layout(set = 0, binding = 9) uniform sampler2D surface_normal_west;
layout(set = 0, binding = 10) uniform sampler2D atmosphere_column;
layout(set = 0, binding = 11) uniform sampler2D weather_fields;
layout(set = 0, binding = 12) uniform sampler2D cloud_terrain_height;
// Atmosphere tables baked by src/sky.rs (keep the constants below in sync).
layout(set = 0, binding = 17) uniform sampler2D sky_transmittance;
layout(set = 0, binding = 18) uniform sampler2D sky_multiscatter;
layout(set = 0, binding = 19) uniform sampler2D sky_irradiance;
// GEBCO 2026 relief as unit-normal east/north components (0.5 + 0.5 n),
// flat over water; a 1x1 flat texture when not installed.
layout(set = 0, binding = 20) uniform sampler2D relief_normals;

layout(set = 1, binding = 0) uniform sampler2DArray vt_day_atlas;
layout(set = 1, binding = 1) uniform usampler2DArray vt_page_table;
layout(set = 1, binding = 2, std140) uniform VtParams {
    uvec4 base_dimensions_mip_count_enabled;
    uvec4 page_table_dimensions_slots_tile_size;
} vt;
// Static virtual texture (earth-static.earthvt): the 500 m night lights,
// the 1 km NASA cloud map and the GEBCO relief normals stream their three
// finest levels through one page table; the rest (and anything not yet
// resident) comes from the resident tails bound as night_emission,
// clouds_a and relief_normals.
layout(set = 1, binding = 3) uniform usampler2DArray static_page_table;
layout(set = 1, binding = 4, std140) uniform StaticVtParams {
    uvec4 base_dimensions_mip_count_enabled;
    uvec4 page_table_dimensions_slots_tile_size;
} svt;
layout(set = 1, binding = 5) uniform sampler2DArray static_night_atlas;
layout(set = 1, binding = 6) uniform sampler2DArray static_clouds_atlas;
layout(set = 1, binding = 7) uniform sampler2DArray static_relief_atlas;

layout(push_constant) uniform FrameData {
    vec4 camera_position_distance;  // xyz camera, w pre-exposure
    vec4 camera_forward;
    vec4 camera_right;
    vec4 camera_up;
    vec4 viewport_rect;
    vec4 canvas_rect;
    vec4 sun_direction;
    vec4 projection_tangents;
    vec4 celestial_distances;
    vec4 moon_body_x;
    vec4 moon_body_y;
    vec4 moon_body_z;
    vec4 celestial_state;
    // Precomputed camera-relative celestial geometry (see ShaderFrame). The
    // Earth material path does not read these; they keep the push-constant
    // block identical across every pipeline sharing this layout.
    vec4 celestial_sun_view;
    vec4 celestial_moon_view;
    vec4 material_state;
} frame;

const float surface_radius = 0.78;
const float KM_PER_UNIT = 6378.137 / 0.78;
// Global mean cloud-top height (ISCCP ~500-600 hPa). At 12 km every cloud's
// shadow landed twice as far out, reading as a doubled cloud field.
const float cloud_radius = surface_radius * (1.0 + 5.5 / 6378.137);
const float atmosphere_radius = surface_radius * (1.0 + 100.0 / 6378.137);
const float inv_two_pi = 0.15915494309189535;
const float inv_pi = 0.3183098861837907;
const float PI = 3.14159265358979;
// Blue Marble NG is a contrast-enhanced mosaic; measured against DSCOVR/EPIC
// true colour its land is ~1.4x too saturated. Pull the albedo toward its
// own luminance by that calibrated amount.
const float surface_saturation = 1.0;

// --- Atmosphere (src/sky.rs) ---------------------------------------------
const float R_GROUND = 6378.137;
const float R_TOP = 6478.137;
const vec3 RAYLEIGH_SCATTERING = vec3(7.117000e-3, 1.374500e-2, 3.323300e-2);
const float RAYLEIGH_SCALE = 8.0;
const vec3 OZONE_ABSORPTION = vec3(2.785000e-3, 1.779000e-3, 0.000000e0);
// Aerosol optical depth 0.18 at 550 nm, Angstrom 0.5, scale height 3.0 km:
// boundary-layer haze plus the free troposphere's. With all of it under a
// 1.8 km scale height, the limb's lower half above ~5 km was pure Rayleigh
// blue (R/G 0.1-0.3 at 50-80 % of its peak, footage 0.2-0.8): ISS footage
// shows it pale, a white haze layer under the blue.
const vec3 AEROSOL_EXTINCTION = vec3(5.562149e-2, 6.027460e-2, 6.708204e-2);
const float AEROSOL_ALBEDO = 0.94;
const float AEROSOL_SCALE = 3.0;
const float AEROSOL_G = 0.68;
const float SUN_ANGULAR_RADIUS = 0.004654;
const float MOON_ANGULAR_RADIUS = 0.004516;

float preexposure() {
    return frame.camera_position_distance.w;
}

// Pre-exposure of the night series, EV 16 (post.rs holds the same number for
// the meter, and a test keeps the two equal).
const float NIGHT_SERIES_PREEXPOSURE = 65536.0;

// The night side has an exposure of its own, as the adapted eye (or a
// composited film) shows it: whatever the camera's exposure, city lights,
// moonlit cloud, lightning, airglow and aurora are drawn as a night series
// (EV 16) records them. They stay visible beside the daylit Earth and
// under a sunrise. At a night exposure the factor is one.
float night_gain() {
    return max(1.0, NIGHT_SERIES_PREEXPOSURE / preexposure());
}

// The same gain eased in through twilight (in stops, so it has no edge):
// none where the Sun is up, all of it once the Sun is 14 degrees down.
float night_gain(float mu_sun) {
    return pow(night_gain(), 1.0 - smoothstep(-0.25, -0.03, mu_sun));
}

// The stars' daylight gate (stars_textured.frag), for the limb airglow: it
// is recorded only at a night exposure, once the Sun is more than 5 degrees
// below the camera's nadir horizon (full from 20 degrees down). Over a
// twilit limb seen from the sunlit ISS the footage shows none.
float night_sky_gate(vec3 camera, vec3 sun) {
    float t = clamp((dot(normalize(camera), sun) + 0.0872) / -0.2548, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Moonlight and starlight as the dark-adapted eye and night footage show
// them: slightly blue (the Purkinje shift).
const vec3 NIGHT_TINT = vec3(0.80, 0.92, 1.12);

float sphere_intersection(vec3 origin, vec3 direction, float radius, out float discriminant) {
    float b = dot(origin, direction);
    float c = dot(origin, origin) - radius * radius;
    discriminant = b * b - c;
    if (discriminant < 0.0) {
        return -1.0;
    }
    float root = sqrt(discriminant);
    float near_distance = -b - root;
    return near_distance > 0.0 ? near_distance : -b + root;
}

float sphere_intersection(vec3 origin, vec3 direction, float radius) {
    float discriminant;
    return sphere_intersection(origin, direction, radius, discriminant);
}

// Near and far roots (either may be negative); false when missed.
bool sphere_span(vec3 origin, vec3 direction, float radius, out float t0, out float t1) {
    float b = dot(origin, direction);
    float c = dot(origin, origin) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant < 0.0) {
        t0 = -1.0;
        t1 = -1.0;
        return false;
    }
    float root = sqrt(discriminant);
    t0 = -b - root;
    t1 = -b + root;
    return true;
}

vec2 transmittance_uv(float r, float mu) {
    float h = sqrt(R_TOP * R_TOP - R_GROUND * R_GROUND);
    float rho = sqrt(max(r * r - R_GROUND * R_GROUND, 0.0));
    float d = max(-r * mu + sqrt(max(r * r * (mu * mu - 1.0) + R_TOP * R_TOP, 0.0)), 0.0);
    float d_min = R_TOP - r;
    float d_max = rho + h;
    float x_mu = clamp((d - d_min) / max(d_max - d_min, 1.0e-3), 0.0, 1.0);
    float x_r = clamp(rho / h, 0.0, 1.0);
    return vec2(0.5 / 256.0 + x_mu * (255.0 / 256.0), 0.5 / 64.0 + x_r * (63.0 / 64.0));
}

// Transmittance from radius r toward a light at zenith cosine mu, with the
// Earth's shadow softened over the light's angular radius (penumbra).
vec3 light_transmittance(float r, float mu, float angular_radius) {
    float sin_horizon = R_GROUND / max(r, R_GROUND);
    float cos_horizon = -sqrt(max(1.0 - sin_horizon * sin_horizon, 0.0));
    float visible = smoothstep(-angular_radius, angular_radius, mu - cos_horizon);
    if (visible <= 0.0) return vec3(0.0);
    return textureLod(sky_transmittance, transmittance_uv(r, max(mu, cos_horizon + 2.0e-4)), 0.0).rgb * visible;
}

vec3 multiscatter(float r, float mu) {
    vec2 uv = vec2(0.5 / 32.0 + (0.5 + 0.5 * mu) * (31.0 / 32.0),
        0.5 / 32.0 + clamp((r - R_GROUND) / (R_TOP - R_GROUND), 0.0, 1.0) * (31.0 / 32.0));
    return textureLod(sky_multiscatter, uv, 0.0).rgb;
}

// Clear-sky diffuse irradiance on a horizontal surface, relative to unit
// top-of-atmosphere solar irradiance.
vec3 sky_irradiance_at(float r, float mu) {
    vec2 uv = vec2(0.5 / 64.0 + (0.5 + 0.5 * mu) * (63.0 / 64.0),
        0.5 / 16.0 + clamp((r - R_GROUND) / 16.0, 0.0, 1.0) * (15.0 / 16.0));
    return textureLod(sky_irradiance, uv, 0.0).rgb;
}

float rayleigh_phase(float c) {
    return 0.0596831 * (1.0 + c * c);
}

float aerosol_phase(float c) {
    const float g = AEROSOL_G;
    const float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + c * c) / pow(1.0 + g * g - 2.0 * g * c, 1.5);
}

struct Light {
    vec3 direction;
    vec3 irradiance;   // relative to the Sun at 1 AU (Sun: 1/au^2)
    float angular_radius;
    float rayleigh;    // phase values for this view ray
    float aerosol;
};

vec2 sphere_uv(vec3 normal);

// Live NOAA GEFS-Aerosols analysis packed with the observed clouds.
bool live_aerosol() {
    return (uint(frame.material_state.x) & 8u) != 0u;
}

// Live EUMETSAT OSI SAF sea-ice concentration in G of binding 12.
bool live_sea_ice() {
    return (uint(frame.material_state.x) & 16u) != 0u;
}

// Texture review (EARTH_NATIVE_NO_CLOUDS): no cloud layer, no cloud shadows.
bool clouds_off() {
    return (uint(frame.material_state.x) & 32u) != 0u;
}

// Aerosol extinction at the ground (km^-1 at 650/550/450 nm) below `unit`:
// the analysed 550 nm optical depth over the aerosol scale height, spread
// spectrally by the analysed 440-645 nm Angstrom exponent (dust ~0.2,
// smoke/pollution ~1.5). Else the climatological constant the tables use.
vec3 aerosol_extinction_at(vec3 unit) {
    if (!live_aerosol()) return AEROSOL_EXTINCTION;
    vec4 packed = textureLod(cloud_terrain_height, sphere_uv(unit), 3.0);
    float optical_depth = 4.0 * packed.a * packed.a;
    float angstrom = 3.0 * packed.b - 0.5;
    return optical_depth / AEROSOL_SCALE * pow(vec3(640.0, 545.0, 440.0) / 550.0, vec3(-angstrom));
}

// Single + multiple scattering of Sun (and Moon) light and the extinction
// along [t0, t1] km of the ray origin + t * direction (both in km). Samples
// crowd toward `dense` (clamped into the segment), where the air is densest:
// the ground end of a downward ray or the tangent point of a limb ray. The
// aerosol load is taken at that point, where the ray meets the haze layer.
void march(vec3 origin, vec3 direction, float t0, float t1, float dense, int steps,
    Light sun, Light moon, bool with_moon, inout vec3 radiance, inout vec3 transmittance) {
    if (t1 <= t0) return;
    dense = clamp(dense, t0, t1);
    vec3 aerosol_extinction = aerosol_extinction_at(normalize(origin + direction * dense));
    float split = (dense - t0) / (t1 - t0);
    float previous = t0;
    for (int i = 1; i <= steps; ++i) {
        float u = float(i) / float(steps);
        float t;
        if (u <= split) {
            float v = 1.0 - u / max(split, 1.0e-6);
            t = dense - (dense - t0) * v * v;
        } else {
            float v = (u - split) / max(1.0 - split, 1.0e-6);
            t = dense + (t1 - dense) * v * v;
        }
        float dt = t - previous;
        vec3 p = origin + direction * (previous + 0.5 * dt);
        previous = t;
        float r = length(p);
        float h = r - R_GROUND;
        float rayleigh_density = exp(-max(h, 0.0) / RAYLEIGH_SCALE);
        float aerosol_density = exp(-max(h, 0.0) / AEROSOL_SCALE);
        float ozone_density = max(1.0 - abs(h - 25.0) / 15.0, 0.0);
        vec3 rayleigh = RAYLEIGH_SCATTERING * rayleigh_density;
        vec3 aerosol = aerosol_extinction * AEROSOL_ALBEDO * aerosol_density;
        vec3 extinction = rayleigh + aerosol_extinction * aerosol_density + OZONE_ABSORPTION * ozone_density;
        vec3 n = p / r;
        float mu_sun = dot(n, sun.direction);
        vec3 source = sun.irradiance * (light_transmittance(r, mu_sun, sun.angular_radius)
                * (rayleigh * sun.rayleigh + aerosol * sun.aerosol)
            + multiscatter(r, mu_sun) * (rayleigh + aerosol));
        if (with_moon) {
            float mu_moon = dot(n, moon.direction);
            source += moon.irradiance * (light_transmittance(r, mu_moon, moon.angular_radius)
                    * (rayleigh * moon.rayleigh + aerosol * moon.aerosol)
                + multiscatter(r, mu_moon) * (rayleigh + aerosol));
        }
        vec3 step_transmittance = exp(-extinction * dt);
        radiance += transmittance * source * (1.0 - step_transmittance) / max(extinction, vec3(1.0e-9));
        transmittance *= step_transmittance;
    }
}

// Moonlight: Allen's phase law, 10^(-0.4(0.026a + 4e-9 a^4)) with a in
// degrees, scaled by distance; full Moon at its mean distance gives ~2.5e-6
// of sunlight. The camera's exposure, not a gain here, makes it visible.
vec3 moon_direction() {
    return normalize(vec3(frame.camera_forward.w, frame.camera_right.w, frame.camera_up.w));
}

float moonlight_scale() {
    float a = degrees(frame.celestial_state.z);
    float phase_law = pow(10.0, -0.4 * (0.026 * a + 4.0e-9 * a * a * a * a));
    float distance_ratio = 60.27 / max(frame.celestial_distances.y, 1.0);
    return 2.5e-6 * phase_law * distance_ratio * distance_ratio;
}

// Starlight, zodiacal light and airglow illuminate the moonless night ground
// at ~2e-3 lux, ~2e-8 of sunlight.
const float NIGHT_SKY_IRRADIANCE = 2.0e-8;

// Fraction of the solar disc visible from scene point p, with the Moon at its
// ephemeris position and both apparent radii exact (area overlap of two
// discs, no limb darkening). Only near new Moon can this be below one; the
// early-out is per-frame uniform, so other frames pay one dot product.
float solar_visibility(vec3 p) {
    vec3 sun_dir = normalize(frame.sun_direction.xyz);
    vec3 moon_dir = moon_direction();
    if (dot(sun_dir, moon_dir) < 0.9994) return 1.0; // > ~2 degrees apart
    vec3 to_sun = sun_dir * frame.celestial_distances.x * surface_radius - p;
    vec3 to_moon = moon_dir * frame.celestial_distances.y * surface_radius - p;
    float sun_distance = length(to_sun);
    float moon_distance = length(to_moon);
    float a = frame.celestial_distances.z * surface_radius / sun_distance;
    float b = frame.celestial_distances.w * surface_radius / moon_distance;
    // Chord form keeps the sub-degree separation precise in float32.
    float d = 2.0 * asin(0.5 * length(to_sun / sun_distance - to_moon / moon_distance));
    if (d >= a + b) return 1.0;
    float covered;
    if (d <= abs(a - b)) {
        covered = min(a, b) * min(a, b);
    } else {
        float alpha = acos(clamp((d * d + a * a - b * b) / (2.0 * d * a), -1.0, 1.0));
        float beta = acos(clamp((d * d + b * b - a * a) / (2.0 * d * b), -1.0, 1.0));
        float kite = sqrt(max((-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b), 0.0));
        covered = (a * a * alpha + b * b * beta - 0.5 * kite) / 3.14159265;
    }
    return clamp(1.0 - covered / (a * a), 0.0, 1.0);
}

float cox_munk_glint(vec3 normal, vec3 light, vec3 ray) {
    const float slope_variance = 0.003 + 0.00512 * 7.0;
    vec3 h = normalize(light - ray);
    float nh = max(dot(normal, h), 1.0e-3);
    float nh2 = nh * nh;
    float facets = exp(-(1.0 - nh2) / (nh2 * slope_variance))
        / (3.14159265 * slope_variance * nh2 * nh2);
    // Schlick water Fresnel: ((1.333 - 1) / (1.333 + 1))^2.
    float fresnel = 0.02037 + 0.97963 * pow(1.0 - max(dot(h, light), 0.0), 5.0);
    float nv = max(dot(normal, -ray), 0.05);
    // The glint fades as the Sun sets on the water (waves shadow each
    // other); a step here drew the terminator as a straight cut.
    return 3.14159265 * fresnel * facets / (4.0 * nv) * smoothstep(0.0, 0.08, dot(normal, light));
}

vec2 sphere_uv(vec3 normal) {
    // Equirectangular map coordinates shared by every Earth layer: U grows
    // westward from the antimeridian mirror convention, V southward.
    float longitude = atan(normal.y, normal.x);
    return vec2(
        fract(0.5 - longitude * inv_two_pi),
        0.5 - asin(clamp(normal.z, -1.0, 1.0)) * inv_pi);
}

vec2 half_map_uv(vec2 uv, vec2 size, vec2 dx, vec2 dy) {
    // Keep both trilinear levels inside this longitude half. REPEAT would
    // blend its opposite edge instead of the adjacent half in another map/channel.
    float footprint = max(length(dx * size), length(dy * size));
    float mip = ceil(max(log2(max(footprint, 1.0)), 0.0));
    float inset = min(0.5 * exp2(mip) / size.x, 0.5);
    return vec2(clamp(uv.x, inset, 1.0 - inset), uv.y);
}

vec4 sample_day_preview(vec2 mesh_uv0, vec2 dx, vec2 dy) {
    dx.x -= round(dx.x);
    dy.x -= round(dy.x);
    dx.x *= 2.0;
    dy.x *= 2.0;
    vec2 preview_uv = mesh_uv0.x > 0.5
        ? vec2(mesh_uv0.x * 2.0 - 1.0, mesh_uv0.y)
        : vec2(mesh_uv0.x * 2.0, mesh_uv0.y);
    preview_uv = half_map_uv(preview_uv, vec2(textureSize(day_color_west, 0)), dx, dy);
    // MF_MakeMask returns one precisely when its U gradient exceeds 0.5.
    if (mesh_uv0.x > 0.5) {
        return textureGrad(day_color_east, preview_uv, dx, dy);
    }
    return textureGrad(day_color_west, preview_uv, dx, dy);
}

uint vt_page(uint x, uint y, uint mip) {
    uvec2 dims = max(vt.page_table_dimensions_slots_tile_size.xy >> mip, uvec2(1));
    x = x % dims.x;
    y = min(y, dims.y - 1u);
    return texelFetch(vt_page_table, ivec3(ivec2(x, y), int(mip)), 0).r;
}

// Must equal `vt_feedback::VT_MIP_BIAS`: shared mip bias, 0.0 = exact match.
const float VT_MIP_BIAS = 0.0;
vec4 sample_day_vt(vec2 uv, vec2 dx, vec2 dy, out bool hit) {
    hit = false;
    if (vt.base_dimensions_mip_count_enabled.w == 0u) {
        return vec4(0.0);
    }
    uint mip_count = vt.base_dimensions_mip_count_enabled.z;
    float base_width = float(vt.base_dimensions_mip_count_enabled.x);
    float base_height = float(vt.base_dimensions_mip_count_enabled.y);
    dx.x -= round(dx.x);
    dy.x -= round(dy.x);
    vec2 dimensions = vec2(base_width, base_height);
    float texels_per_pixel = max(max(length(dx * dimensions), length(dy * dimensions)), 1.0e-5);
    int requested_mip = clamp(int(floor(log2(texels_per_pixel) + VT_MIP_BIAS)), 0, int(mip_count) - 1);
    for (int mip = requested_mip; mip < int(mip_count); ++mip) {
        uvec2 dims = max(vt.page_table_dimensions_slots_tile_size.xy >> uint(mip), uvec2(1));
        vec2 wrapped_uv = vec2(fract(uv.x), clamp(uv.y, 0.0, 1.0));
        uvec2 page = min(uvec2(wrapped_uv * vec2(dims)), dims - 1u);
        uint slot = vt_page(page.x, page.y, uint(mip));
        if (slot != 0xffffffffu && slot < vt.page_table_dimensions_slots_tile_size.z) {
            vec2 mip_uv = wrapped_uv * vec2(dims);
            vec2 local = fract(mip_uv);
            vec2 atlas_uv = (local * 256.0 + 4.0) / float(vt.page_table_dimensions_slots_tile_size.w);
            hit = true;
            return texture(vt_day_atlas, vec3(atlas_uv, float(slot)));
        }
    }
    return vec4(0.0);
}

vec4 sample_day_material(vec2 mesh_uv0, vec2 dx, vec2 dy) {
    // EarthVT uses the canonical equirectangular coordinates directly. The
    // 5x5 block remap belongs only to the diagnostic fallback preview.
    bool hit;
    vec4 vt_colour = sample_day_vt(mesh_uv0, dx, dy, hit);
    return hit ? vt_colour : sample_day_preview(mesh_uv0, dx, dy);
}

vec3 sample_surface_normal(vec2 mesh_uv0, vec2 dx, vec2 dy) {
    dx.x -= round(dx.x);
    dy.x -= round(dy.x);
    // Two longitude halves (east/west maps); fold U into the half.
    vec2 uv = vec2(fract(mesh_uv0.x * 2.0), mesh_uv0.y);
    vec2 n_dx = vec2(dx.x * 2.0, dx.y);
    vec2 n_dy = vec2(dy.x * 2.0, dy.y);
    float inset = 0.5 / float(textureSize(surface_normal_west, 0).x);
    uv.x = clamp(uv.x, inset, 1.0 - inset);
    return mesh_uv0.x > 0.5
        ? textureGrad(surface_normal_east, uv, n_dx, n_dy).rgb
        : textureGrad(surface_normal_west, uv, n_dx, n_dy).rgb;
}

float sample_packed_height(vec2 map_uv, vec2 dx, vec2 dy) {
    dx.x -= round(dx.x);
    dy.x -= round(dy.x);
    return textureGrad(terrain_height, map_uv, dx, dy).r;
}

vec3 perturb_surface_normal(vec3 geometric_normal, vec2 map_uv, vec2 map_dx, vec2 map_dy, vec3 sun, float water_amount) {
    vec2 encoded_xy = sample_surface_normal(map_uv, map_dx, map_dy).xy * 2.0 - 1.0;
    float encoded_z = sqrt(max(1.0 - dot(encoded_xy, encoded_xy), 0.0));
    vec3 raw_normal = vec3(encoded_xy, encoded_z);
     // Relief normals are flattened over low terrain and under a high Sun
     // (where shading would over-emphasise the derived slopes), then applied
     // at half strength. Ocean roughness is statistical (Cox-Munk glint).
    float first_flatten = clamp(3.0 * sample_packed_height(map_uv, map_dx, map_dy), 0.88, 0.95)
        - clamp(dot(geometric_normal, sun), 0.0, 1.0) / 5.0;
    vec3 height_flattened = normalize(mix(
        raw_normal,
        vec3(0.0, 0.0, 1.0),
        clamp(first_flatten, 0.0, 1.0)));
    vec3 tangent_normal = normalize(mix(
        height_flattened,
        vec3(0.0, 0.0, 1.0),
        0.5));
    float equatorial_length = length(geometric_normal.xy);
    if (equatorial_length < 1.0e-4) {
        return geometric_normal;
    }
    // U increases westward and V increases southward in the map coordinates.
    vec3 tangent_u = vec3(geometric_normal.y, -geometric_normal.x, 0.0)
        / equatorial_length;
    vec3 tangent_v = normalize(cross(geometric_normal, tangent_u));
    return normalize(
        tangent_u * tangent_normal.x
        + tangent_v * tangent_normal.y
        + geometric_normal * tangent_normal.z);
}

bool static_vt() {
    return svt.base_dimensions_mip_count_enabled.w != 0u;
}

// log2 of full-resolution texels per pixel (the static layers' mip).
float static_lod(vec2 dx, vec2 dy) {
    vec2 dimensions = vec2(svt.base_dimensions_mip_count_enabled.xy);
    return log2(max(max(length(dx * dimensions), length(dy * dimensions)), 1.0e-5));
}

// One level of a static layer: the resident page of `mip`, or the nearest
// coarser resident one, or the tail.
vec4 static_level(sampler2DArray atlas, sampler2D tail, vec2 uv, int mip) {
    int mips = int(svt.base_dimensions_mip_count_enabled.z);
    vec2 wrapped_uv = vec2(fract(uv.x), clamp(uv.y, 0.0, 1.0));
    for (int level = mip; level < mips; ++level) {
        uvec2 dims = max(svt.page_table_dimensions_slots_tile_size.xy >> uint(level), uvec2(1));
        uvec2 page = min(uvec2(wrapped_uv * vec2(dims)), dims - 1u);
        uint slot = texelFetch(static_page_table, ivec3(ivec2(page), level), 0).r;
        if (slot != 0xffffffffu && slot < svt.page_table_dimensions_slots_tile_size.z) {
            vec2 local = fract(wrapped_uv * vec2(dims));
            vec2 atlas_uv = (local * 256.0 + 4.0) / float(svt.page_table_dimensions_slots_tile_size.w);
            return textureLod(atlas, vec3(atlas_uv, float(slot)), 0.0);
        }
    }
    return textureLod(tail, uv, 0.0);
}

// A static layer with trilinear blending between its two nearest levels;
// coarser than the streamed levels it is the tail with the hardware's
// anisotropic filtering. Without the static VT the tail is the full map.
vec4 sample_static(sampler2DArray atlas, sampler2D tail, vec2 uv, vec2 dx, vec2 dy) {
    dx.x -= round(dx.x);
    dy.x -= round(dy.x);
    if (!static_vt()) return textureGrad(tail, uv, dx, dy);
    float lod = static_lod(dx, dy);
    float mips = float(svt.base_dimensions_mip_count_enabled.z);
    if (lod >= mips) return textureGrad(tail, uv, dx, dy);
    float level = max(lod, 0.0);
    int first = int(floor(level));
    float blend = level - float(first);
    vec4 fine = static_level(atlas, tail, uv, first);
    if (blend < 1.0 / 256.0) return fine;
    vec4 coarse = first + 1 < int(mips) ? static_level(atlas, tail, uv, first + 1) : textureLod(tail, uv, 0.0);
    return mix(fine, coarse, blend);
}

// Terrain normal from the GEBCO slopes, in the renderer's frame (east =
// n x z is geographic east in the longitude-mirrored scene).
vec3 relief_normal(vec3 n, vec2 map_uv, vec2 dx, vec2 dy) {
    dx.x -= round(dx.x);
    dy.x -= round(dy.x);
    vec2 stored = sample_static(static_relief_atlas, relief_normals, map_uv, dx, dy).rg * 2.0 - 1.0;
    vec3 east = cross(n, vec3(0.0, 0.0, 1.0));
    float east_length = length(east);
    if (east_length < 1.0e-4) return n;
    east /= east_length;
    vec3 north = cross(east, n);
    // Gradients from 1.2 km mean heights are gentle, and mips average them
    // further, but a pixel of real terrain still holds lit and shaded
    // slopes: exaggerate (as shaded-relief maps do), more for coarser mips.
    float lod = max(static_vt() ? static_lod(dx, dy) : textureQueryLod(relief_normals, map_uv).y, 0.0);
    float exaggeration = 2.5 * pow(1.3, lod);
    vec2 gradient = stored / max(sqrt(max(1.0 - dot(stored, stored), 0.0)), 0.2) * exaggeration;
    return normalize(east * gradient.x + north * gradient.y + n);
}

// The NASA cloud map (static VT or its tail). Derivatives are taken before
// any wrap, with the sphere's longitude discontinuity repaired.
vec4 sample_nasa_clouds(vec2 map_uv) {
    return sample_static(static_clouds_atlas, clouds_a, map_uv, dFdx(map_uv), dFdy(map_uv));
}

// NASA Blue Marble cloud map: BC4 display-encoded
// cloud brightness. Reflectance saturates with optical depth, so anything
// displayed moderately bright is an optically thick deck and must be opaque;
// a literal brightness inversion left bright cumulus ~40% transparent, which
// showed each cloud's own offset shadow through it as a dark "double".
// Faint values stay translucent haze and thin cirrus.
float nasa_cloud_opacity(vec2 map_uv) {
    float brightness = sample_nasa_clouds(map_uv).r;
    return smoothstep(0.12, 0.72, brightness);
}

// Thin cloud is optically thin in albedo too: scale the lit cloud top from
// grey (thin) to white (thick deck) so tops keep texture instead of flat white.
float nasa_cloud_albedo(vec2 map_uv) {
    float brightness = sample_nasa_clouds(map_uv).r;
    return mix(0.62, 0.9, smoothstep(0.25, 0.8, brightness));
}

// Observed clouds (NOAA GMGSI visible by day, 10.7 um infrared by night,
// GFS low cloud and polar fill), ~10 km cover in R of binding 12.
bool live_clouds() {
    return (uint(frame.material_state.x) & 4u) != 0u;
}

// Sub-grid structure the 10 km observation cannot carry: the tileable
// 1/f^2 fractal at two scales (tiles of ~220 km and ~28 km, texels of
// ~0.9 km and ~110 m), projected triplanar from the sphere normal so it is
// isotropic on the ground with no lat/lon shear (which smeared it into
// streaks away from the prime meridian). The variance-preserving blend
// keeps its contrast where two projections overlap.
float cloud_noise(vec3 n, float tile_km, vec2 offset, float bias) {
    vec3 p = n * (R_GROUND / tile_km);
    vec3 w = pow(abs(n), vec3(8.0));
    w /= w.x + w.y + w.z;
    float mean = textureLod(tiling_noise, vec2(0.5), 16.0).r;
    vec3 s = vec3(texture(tiling_noise, p.yz + offset, bias).r,
                  texture(tiling_noise, p.zx + offset, bias).r,
                  texture(tiling_noise, p.xy + offset, bias).r);
    return mean + (dot(w, s) - mean) * inversesqrt(dot(w, w));
}

// The fine octave is read two mips down: cloud elements are rounded at
// the ~0.5 km scale, and the fractal's last octaves only frayed their edges
// into specks. Broken cloud is many small cells: trade cumulus in ISS
// footage is ~27 clouds per 10^4 pixels at ~1 km per pixel (median 2 km,
// the largest 8 % of the cloud area), where the coarse tile carried the
// field and cut it into a few large blobs (5.6-7 per 10^4 pixels, the
// largest 67-90 %). With little cover (`fine` 1) the fine tile carries it;
// the weights keep the field's mean and variance, so the cover quantiles
// stay valid.
float cloud_detail(vec3 n, float fine) {
    float coarse_weight = mix(0.68, 0.32, fine);
    float fine_weight = 1.0 - coarse_weight;
    float mean = textureLod(tiling_noise, vec2(0.5), 16.0).r;
    float coarse = cloud_noise(n, 222.0, vec2(0.0), 0.0) - mean;
    float detail = cloud_noise(n, 28.0, vec2(0.37, 0.71), 2.0) - mean;
    return mean + (coarse_weight * coarse + fine_weight * detail)
        * sqrt(0.5648 / (coarse_weight * coarse_weight + fine_weight * fine_weight));
}

// Real cloud morphology at 1 km (cloud streets, open and closed cells,
// fronts) from the NASA Blue Marble cloud composite, mixed into the detail
// so observed cover is sculpted like clouds rather than noise. Where that
// historical map was clear the fractal alone decides.
// Broken cloud (`fine` 1) leans on the fine fractal: the composite's large
// static decks clumped it into a few big blobs.
float cloud_morphology(vec2 map_uv, vec3 n, float fine) {
    float composite = sample_nasa_clouds(map_uv).r;
    return mix(cloud_detail(n, fine), composite, 0.45 * (1.0 - 0.6 * fine));
}

// Cubic B-spline upsampling of the ~10 km cover in four bilinear taps.
// Bilinear alone has kinks on the texel grid, which a sharp cloud edge
// traces as straight, rectangular outlines.
float live_cloud_cover(vec2 map_uv, vec2 dx, vec2 dy) {
    vec2 size = vec2(textureSize(cloud_terrain_height, 0));
    vec2 p = map_uv * size - 0.5;
    vec2 i = floor(p);
    vec2 f = p - i;
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;
    vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    vec2 w3 = f3 / 6.0;
    vec2 g0 = w0 + w1;
    vec2 g1 = w2 + w3;
    vec2 uv0 = (i + 0.5 - 1.0 + w1 / g0) / size;
    vec2 uv1 = (i + 0.5 + 1.0 + w3 / g1) / size;
    return g0.y * (g0.x * textureGrad(cloud_terrain_height, vec2(uv0.x, uv0.y), dx, dy).r
                 + g1.x * textureGrad(cloud_terrain_height, vec2(uv1.x, uv0.y), dx, dy).r)
         + g1.y * (g0.x * textureGrad(cloud_terrain_height, vec2(uv0.x, uv1.y), dx, dy).r
                 + g1.x * textureGrad(cloud_terrain_height, vec2(uv1.x, uv1.y), dx, dy).r);
}

// Quantile of the morphology field (0.55 fractal + 0.45 NASA composite)
// relative to its mean, measured over the textures (cos-latitude weighted,
// sd 0.117, skewed bright): q = 0, 0.1, 0.5, 0.9, 1 (the ends just past
// the 1st/99th percentiles).
float morphology_quantile_offset(float q) {
    if (q < 0.1) return mix(-0.22, -0.135, q / 0.1);
    if (q < 0.5) return mix(-0.135, -0.023, (q - 0.1) / 0.4);
    if (q < 0.9) return mix(-0.023, 0.172, (q - 0.5) / 0.4);
    return mix(0.172, 0.33, (q - 0.9) / 0.1);
}

// The observed cover (a real ~10 km cloud image) is the cloud fraction; the
// morphology decides where inside that area the cloud is. The field is cut
// at its (1 - cover) quantile, so the cloudy area keeps the observed
// fraction while edges, holes and cells come from the fractal and the NASA
// composite. Cumulus and deck edges are sharp at ISS scale (~100 m): the cut
// is a narrow band that widens to one pixel's footprint when that is larger
// (antialiased, and area-averaging to the same cover from afar), with a
// thin translucent fringe. A fixed 0.08-0.92 ramp spread every edge over
// 5-10 km and left wide 30-60 % translucent areas that read as grey smears
// over sunglint.
// x: opacity, y: cloud-top albedo. How far the field rises above the cut
// stands in for optical depth, and reflectance grows with it (two-stream
// R ~ (1-g)tau / (2 + (1-g)tau)): thin fringes stay grey, cores go white,
// so a deck keeps its lumpy texture instead of a flat cut-out fill.
// z: relief of the cloud top (0 at the edge, 1 over a core), for shading.
//
// `slant` is tan(view zenith angle) at the shell. Clouds have sides: a field
// of fraction f and height/width ratio a hides 1 - (1 - f)^(1 + a tan) of
// what is behind it (random overlap), so broken cloud closes up toward the
// horizon, as in every oblique photograph from orbit. Zero for the Sun's
// path (shadows are cast by the cover itself).
const float CLOUD_ASPECT = 0.25;
vec3 live_cloud(vec2 map_uv, vec3 n, float slant) {
    vec2 dx = dFdx(map_uv);
    vec2 dy = dFdy(map_uv);
    dx.x -= round(dx.x);
    dy.x -= round(dy.x);
    float cover = clamp(live_cloud_cover(map_uv, dx, dy), 0.0, 1.0);
    cover = 1.0 - pow(1.0 - cover, 1.0 + CLOUD_ASPECT * slant);
    float morphology = cloud_morphology(map_uv, n, 1.0 - smoothstep(0.15, 0.6, cover));
    float mean = 0.55 * textureLod(tiling_noise, vec2(0.5), 16.0).r + 0.45 * 0.244;
    float threshold = mean + morphology_quantile_offset(1.0 - cover);
    // A pixel or two wide: the minimum of +-0.027 in field units spread
    // edges over several pixels where the field is flat (as soft as 0.04
    // of the contrast per pixel against the footage's 0.21-0.29); +-0.018
    // keeps them crisp without the hard cut-out look of a one-pixel step.
    float band = max(0.009, 0.5 * fwidth(morphology));
    float opacity = smoothstep(threshold - band - 0.009, threshold + band + 0.009, morphology);
    // Thin fringes grey, cores white, reached sooner: decks read as bright
    // lumpy cloud (footage interior L 0.58) rather than a grey mottle (0.48).
    float depth = smoothstep(0.0, 0.14, morphology - threshold);
    float rise = max(morphology - threshold, 0.0);
    vec3 near = vec3(opacity, mix(0.62, 0.93, depth), rise / (rise + 0.12));
    // From afar the mips average the morphology toward its mean, so the
    // quantile cut above collapses into an on/off switch at 50 % cover:
    // flat white cut-outs tracing the 10 km grid on the globe. Once a pixel
    // spans several km, show the observed cover as the fraction it is, with
    // the (low-passed) morphology left as texture, and thin cover greyer.
    float km_x = length(vec2(dx.x * 40075.0 * sqrt(max(1.0 - n.z * n.z, 0.0)), dx.y * 20037.5));
    float km_y = length(vec2(dy.x * 40075.0 * sqrt(max(1.0 - n.z * n.z, 0.0)), dy.y * 20037.5));
    float far = smoothstep(1.5, 8.0, max(km_x, km_y));
    float wide_opacity = clamp(cover + (morphology - mean) * 1.2 * (1.0 - abs(2.0 * cover - 1.0)), 0.0, 1.0);
    vec3 wide = vec3(wide_opacity, mix(0.55, 0.9, smoothstep(0.3, 0.95, cover)), wide_opacity);
    return mix(near, wide, far);
}

float live_cloud_opacity(vec2 map_uv, vec3 n) {
    return live_cloud(map_uv, n, 0.0).x;
}

float sample_cloud_density(vec2 mesh_uv0, vec3 cloud_normal) {
    if (clouds_off()) return 0.0;
    return live_clouds() ? live_cloud_opacity(mesh_uv0, cloud_normal) : nasa_cloud_opacity(mesh_uv0);
}


float hash21(vec2 value) {
    value = fract(value * vec2(123.34, 456.21));
    value += dot(value, value + 45.32);
    return fract(value.x * value.y);
}

float storm_random(vec2 cell, uint phase) {
    // PCG RXS-M-XS hash preserves entropy at every UTC second. Multiplying
    // a large float time by the old hash constants erased fractional bits.
    uint state = (uint(cell.x) * 1973u ^ uint(cell.y) * 9277u ^ phase * 26699u)
        * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) * (1.0 / 4294967296.0);
}

// Four independent numbers for one (cell, phase): the same hash chained.
vec4 storm_random4(vec2 cell, uint phase) {
    uint word = uint(cell.x) * 1973u ^ uint(cell.y) * 9277u ^ phase * 26699u;
    vec4 value;
    for (int i = 0; i < 4; ++i) {
        uint state = (word + uint(i) * 2654435761u) * 747796405u + 2891336453u;
        word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        value[i] = float((word >> 22u) ^ word) * (1.0 / 4294967296.0);
    }
    return value;
}

// Black Marble 2016 is an 8-bit display product (sRGB-coded BC4). Mapped to
// radiance so the brightest city cores reach ~25x a full-Moon-lit desert
// (VIIRS DNB: ~500 vs ~20 nW cm^-2 sr^-1), in sunlight units. The product's
// stretch is gentler than sRGB at the faint end: decoded with 2.2, the
// lights of villages and roads fell into the camera's toe and northern
// India showed a third of the lit ground of 2024-26 ISS footage. Cores
// (encoded 1) are unchanged.
const float CITY_RADIANCE = 1.5e-5;
float nasa_lights(float encoded) {
    return pow(encoded, 1.6);
}

// The glows (halo over a city, light diffused by cloud) sample blurred mips,
// where the gentler decode would lift wide faint areas 4x into a grey fog:
// they keep the 2.2 decode they were calibrated with.
float nasa_glow(float encoded) {
    return pow(encoded, 2.2);
}

// City-light radiance at a map position (linear, from the sRGB-coded BC4).
float city_signal_at(vec2 map_uv, vec2 offset) {
    vec2 uv = map_uv + offset;
    return nasa_glow(sample_static(static_night_atlas, night_emission, uv, dFdx(uv), dFdy(uv)).r);
}

float city_glow_at(vec2 map_uv, float lod) {
    return nasa_glow(textureLod(night_emission, map_uv, lod).r);
}

// Share of city light that is high-pressure sodium. Street lighting has
// largely moved to white LEDs (the US, Europe, East Asia, and India's
// national LED programme), and ISS night footage of the 2020s reads
// white with a warm cast nearly everywhere. Black Marble cannot tell lamp
// types apart, so one global share keeps that warm cast.
const float SODIUM_SHARE = 0.15;
// The same lamps seen diffused through air or cloud: scattering treats every
// colour alike, so a lit cloud deck keeps the lamps' mixed colour, the
// cream-white of 2024-26 ISS footage (R/G ~1.1, B/G ~0.8), not a sodium
// orange.
const vec3 DIFFUSE_LAMP = vec3(1.0, 0.90, 0.76);

// Aurora shell: emission lives between 90 and 400 km.
const float aurora_top_radius = surface_radius * (1.0 + 400.0 / 6378.137);
// Night glows reach 400 km (the 630 nm airglow layer): rays that pass over
// the air but under this still shade.
const float night_glow_top_radius = surface_radius * (1.0 + 400.0 / 6378.137);

// Noise with an explicit LOD: derivatives are undefined inside the march.
float aurora_noise(vec2 uv, vec2 tiling, float footprint_uv) {
    float texels = footprint_uv * max(tiling.x, tiling.y) * float(textureSize(tiling_noise, 0).x);
    return textureLod(tiling_noise, uv * tiling, max(log2(max(texels, 1.0e-6)), 0.0)).r;
}

// Smooth value noise on the sphere map, periodic in longitude; two octaves.
// The tiling detail texture is too grainy to steer whole arcs.
float aurora_value_noise(vec2 uv, vec2 cells) {
    float sum = 0.0;
    float amplitude = 0.65;
    for (int octave = 0; octave < 2; ++octave) {
        vec2 x = uv * cells;
        vec2 i = floor(x);
        vec2 f = x - i;
        f = f * f * (3.0 - 2.0 * f);
        vec2 j = vec2(mod(i.x + 1.0, cells.x), i.y + 1.0);
        i.x = mod(i.x, cells.x);
        float a = hash21(i + 0.5);
        float b = hash21(vec2(j.x, i.y) + 0.5);
        float c = hash21(vec2(i.x, j.y) + 0.5);
        float d = hash21(j + 0.5);
        sum += amplitude * mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        cells *= 2.0;
        amplitude = 0.35;
    }
    return sum;
}

// Aurora. Volume emission rate in kilorayleigh per km of path (a
// column of 1 kR per km summed along the ray is 1 kR); 1 kR at 557.7 nm is
// 4.7e-9 in sunlight units (see the airglow note). Bright discrete arcs are
// 10-100 kR overhead and several times that edge-on at the limb.
const float KILORAYLEIGH = 4.7e-9;
// Display gain: quiet ovals (OVATION 10-20 %) are ~1-5 kR, under the
// night key; this brings them to the brightness of the curtains in
// orbital time-lapses and in the Incredible Earth reference renders.
const float AURORA_GAIN = 150.0;

// Volumetric march through the 90-400 km auroral shell. NOAA OVATION gives
// where the oval is and how active it is; the structure follows DMSP/VIIRS
// night imagery and ISS photography:
//  - a diffuse, patchy glow filling the oval (brightest equatorward);
//  - discrete arcs along its poleward flank that follow the oval's contours,
//    fold and break up, multiplying with activity;
//  - field-aligned rays, constant along the (near-vertical) magnetic field:
//    vertical striations when a curtain is seen edge-on at the limb;
//  - altitude profiles: O(1S) 557.7 nm green with a sharp lower border near
//    100 km and a ~30 km scale height above; O(1D) 630 nm red from ~180 km,
//    peaking near 235 km and fading slowly above (a dim, tall maroon haze
//    over the curtains in ISS photos, crimson only in the ray tops); N2+
//    391/428 nm and N2 1PG pink-violet along the lower edge of bright arcs,
//    and violet ray tops where the aurora stands in sunlight.
vec3 aurora_emission(vec3 camera, vec3 ray, vec3 sun, float surface_hit) {
    if (frame.material_state.z < 0.5) return vec3(0.0);
    float b = dot(camera, ray);
    float c = dot(camera, camera) - aurora_top_radius * aurora_top_radius;
    float disc = b * b - c;
    if (disc <= 0.0) return vec3(0.0);
    float root = sqrt(disc);
    float t0 = max(-b - root, 0.0);
    float t1 = -b + root;
    // Only the ground occludes. Clipping at the 90 km sphere would drop the
    // far half of grazing rays and draw a seam along the limb.
    if (surface_hit > 0.0) t1 = min(t1, surface_hit);
    if (t1 <= t0) return vec3(0.0);
    // The oval never reaches below ~45 degrees geomagnetic; skip rays whose
    // whole segment stays at lower latitude.
    vec3 p0 = normalize(camera + ray * t0);
    vec3 p1 = normalize(camera + ray * t1);
    vec3 pm = normalize(camera + ray * (0.5 * (t0 + t1)));
    if (max(max(abs(p0.z), abs(p1.z)), abs(pm.z)) < 0.55) return vec3(0.0);

    // ~25 km steps: a downward ray crosses the 90-320 km layer in a few
    // hundred km and needs ~10 samples; grazing limb rays keep 40.
    int steps = int(clamp(ceil((t1 - t0) * KM_PER_UNIT / 25.0), 10.0, 40.0));
    float dt = (t1 - t0) / float(steps);
    // White-noise stratification that changes every frame: step banding
    // becomes fine grain, like the camera's own noise, never a fixed pattern.
    float time = frame.celestial_state.w;
    float jitter = hash21(gl_FragCoord.xy * 0.713 + vec2(fract(time * 7.31) * 97.0, fract(time * 3.17) * 61.0));
    float pixel_angle = 2.0 * frame.projection_tangents.y / frame.canvas_rect.w;
    float step_km = dt * KM_PER_UNIT;
    vec3 emission = vec3(0.0);
    for (int i = 0; i < steps; ++i) {
        float ts = t0 + (float(i) + jitter) * dt;
        vec3 p = camera + ray * ts;
        float r = length(p);
        vec3 n = p / r;
        float h = (r - surface_radius) * KM_PER_UNIT;
        if (h < 90.0) continue;
        // Hidden only over sunlit ground (Sun more than ~1-5 degrees up
        // below it): ISS footage shows the curtains bright up to the dawn
        // terminator, beside the blue sunlit limb, where fading them through
        // nautical twilight (-9 to -1 degrees) left no aurora near any dawn
        // or dusk. (A fade by the ray's background cost 8 more registers.)
        float visible = 1.0 - smoothstep(-0.02, 0.08, dot(n, sun));
        if (visible <= 0.0) continue;
        vec2 uv = sphere_uv(n);
        float probability = textureLod(weather_fields, uv, 3.0).a;
        if (probability < 0.01) continue;
        // Oval side and contour spacing from the probability ~1.1 degrees
        // (120 km) poleward and equatorward.
        vec2 pole_step = vec2(0.0, n.z > 0.0 ? -0.006 : 0.006);
        float p_pole = textureLod(weather_fields, uv + pole_step, 3.0).a;
        float p_equator = textureLod(weather_fields, uv - pole_step, 3.0).a;
        float slope = (p_pole - p_equator) / 240.0;
        float poleward = smoothstep(0.0, -0.3, (p_pole - p_equator) / (probability + 0.02));
        float activity = smoothstep(0.0, 0.12, max(probability, p_equator));
        // What one sample stands for horizontally: the pixel footprint or
        // the march step along the ground track, whichever is larger.
        vec3 horizontal = ray - n * dot(ray, n);
        float footprint_km = max(ts * pixel_angle * KM_PER_UNIT, step_km * length(horizontal));
        float footprint_uv = footprint_km / (6.2831853 * R_GROUND * max(length(n.xy), 0.2));

        // Discrete arcs: contours of the OVATION field at fixed levels, so
        // quiet ovals carry one or two and storms several; folds displace
        // them by up to ~100 km and drift over minutes.
        float fold = aurora_noise(uv + vec2(time * 3.0e-4, time * 4.0e-5), vec2(3.0, 1.5), footprint_uv)
            + 0.35 * aurora_noise(uv - vec2(time * 9.0e-4, 0.0), vec2(11.0, 4.0), footprint_uv);
        float fold_km = (fold - 0.675) * 320.0;
        float spacing = max(abs(slope), 4.0e-5);
        // A bright arc is a bundle of 1-10 km sheets; seen from orbit it
        // reads as a soft band ~30 km wide. Under a footprint it widens with
        // its energy conserved rather than aliasing.
        const float arc_km = 14.0;
        float arc_sigma = sqrt(arc_km * arc_km + footprint_km * footprint_km);
        float arc_gain = arc_km / arc_sigma;
        float arcs = 0.0;
        // Contours up to 60 %: quiet ovals carry the first two or three, a
        // storm all six. An arc's brightness follows its level (the local
        // precipitating energy flux), ~20 kR at 7.5 % to ~130 kR at 60 %:
        // IBC II to III-IV, as in storm-time ISS footage.
        const float levels[6] = float[6](0.045, 0.075, 0.115, 0.2, 0.35, 0.6);
        const float weights[6] = float[6](0.6, 1.0, 1.2, 2.2, 3.5, 6.0);
        for (int k = 0; k < 6; ++k) {
            if (max(probability, p_equator) < 0.5 * levels[k]) break;
            vec2 offset = vec2(0.31, 0.17) * float(k + 1);
            float own = aurora_value_noise(vec2(uv.x - time * (1.2e-5 + 0.6e-5 * float(k % 3)), uv.y) + offset,
                vec2(18.0, 30.0));
            float along = (probability - levels[k]) / spacing + fold_km + (own - 0.5) * 180.0;
            float segment = smoothstep(0.35, 0.70, own);
            arcs += weights[k] * segment * exp(-0.5 * along * along / (arc_sigma * arc_sigma));
        }
        arcs *= arc_gain * poleward * activity;
        // Field-aligned rays: horizontal noise only (constant along the
        // field), 15-40 km cells drifting eastward along the arcs; filtered
        // by the footprint so distant or coarsely sampled rays average out.
        float ray_noise = aurora_noise(vec2(uv.x + time * 2.5e-4, uv.y), vec2(160.0, 90.0), footprint_uv);
        float rays = mix(1.0, 0.15 + 2.2 * ray_noise * ray_noise, 0.9 * (1.0 - smoothstep(80.0, 250.0, footprint_km)));
        float pulse = 0.88 + 0.12 * sin(time * 1.1 + ray_noise * 9.0);
        // Diffuse aurora: large patches over the whole oval, patchy and
        // contrasted as ISS footage shows it from above (p95/p5 ~4).
        float patches = aurora_noise(uv + vec2(time * 1.0e-4, 0.0), vec2(26.0, 13.0), footprint_uv);
        // ~1 kR at 25 % probability, rising with the energy flux in storms.
        float diffuse = smoothstep(0.05, 0.25, probability) * max(1.0, probability / 0.25)
            * (1.0 - 0.5 * poleward) * (0.15 + 2.0 * patches * patches * patches);

        float green = smoothstep(94.0, 104.0, h) * exp(-max(h - 108.0, 0.0) / 30.0);
        float red_x = (h - 235.0) / 65.0;
        float red = (h < 235.0 ? exp(-red_x * red_x) : exp(-(h - 235.0) / 110.0)) * smoothstep(150.0, 190.0, h);
        // Outside the Earth's shadow N2+ ions resonantly scatter sunlight.
        float shadow_cos = -sqrt(max(1.0 - surface_radius * surface_radius / (r * r), 0.0));
        float sunlit = smoothstep(shadow_cos - 0.02, shadow_cos + 0.03, dot(n, sun));
        float fringe_x = (h - 97.0) / 4.0;
        float fringe = exp(-fringe_x * fringe_x);
        // kR per km: diffuse ~1 kR overhead over ~50 km of column, arcs
        // ~20 kR over ~40 km.
        float discrete = arcs * rays * pulse;
        // Arcs carry the display along the limb; seen from above, the
        // diffuse glow lights the ground under the oval green (footage:
        // 0.2-0.4 of the limb band, 6-14x what 0.02 gave). The 630 nm red
        // is ~1/15 of the green, desaturated toward maroon: footage shows a
        // dim tall haze (red/green 0.02-0.08) with crimson ray tops, where
        // 1/5 drew a saturated ribbon.
        vec3 local = vec3(0.15, 1.0, 0.25) * green * (0.08 * diffuse + 0.9 * discrete)
            + vec3(1.0, 0.15, 0.13) * red * (0.002 * diffuse + 0.07 * discrete)
            + vec3(0.9, 0.25, 0.8) * fringe * 0.25 * discrete * smoothstep(0.3, 1.0, discrete)
            + vec3(0.45, 0.30, 1.0) * red * 0.06 * discrete * sunlit;
        emission += local * (visible * step_km);
    }
    return emission * KILORAYLEIGH;
}

// Night airglow layers, integrated analytically along the ray: a layer of
// Gaussian vertical profile (peak h0, width w) and zenith intensity I seen
// along a chord with tangent altitude ht has column
//   I * sqrt(2 R / w) / sqrt(pi) * g((ht - h0) / w),
// g(x) = int exp(-(x + y^2)^2) dy, ~sqrt(pi / (c - x)) below the layer and
// exp(-x^2) sqrt(pi / (c + 2x)) above (c = 0.956 matches g(0) = 1.813). The
// tangent (limb) path is ~80x the zenith column: the thin band ISS night
// photographs show on the horizon.
//   O(1S) 557.7 nm at 97 km, ~400 R near solar maximum: green, which a
//     camera sensor records as a teal green (ISS footage: B/G 0.4-0.5)
//   Na D 589 nm at 91 km, ~100 R: yellow-orange
//   OH Meinel (visible red tail) at 87 km, ~130 R: the thin amber line
//     under the green one in moonless footage
//   O(1D) 630 nm from the F region, peak ~250 km and ~75 km thick: ~60 R at
//     middle latitudes, ~300 R over the equatorial anomaly near solar maximum. ISS footage
//     shows it as a red-orange band at 235-250 km above a darker gap.
// 1 R of 557.7 nm is 2.8e-10 W m^-2 sr^-1, 1/59 of a white Lambertian
// surface under the Sun in the green channel per 1e-10: sunlight units.
float airglow_column(float tangent_km, float h0, float w, float slant) {
    float x = (tangent_km - h0) / w;
    const float c = 0.956;
    float g = x < 0.0 ? sqrt(PI / (c - x)) : exp(-x * x) * sqrt(PI / (c + 2.0 * x));
    float chord = sqrt(2.0 * R_GROUND / w) / sqrt(PI) * g;
    // A ray that meets the ground crosses the layer once: its column is the
    // slant factor 1/cos(zenith), which the tangent chord overestimates.
    return slant > 0.0 ? min(0.5 * chord, slant) : chord;
}

// Display gain, as AURORA_GAIN: ISS night footage is shot at high gain,
// where the limb airglow reads as a distinct green-yellow band; at the
// physical scale under the night key it was all but invisible.
const float AIRGLOW_GAIN = 15.0;
// Like the stars and the Milky Way, the airglow is shown as a night series
// at a fixed exposure (EV 17) records it, whatever the camera's exposure:
// following the camera it was too bright in moonless views (EV 18) and lost
// under moonlight (EV 16), where the footage still shows the line and the
// red 630 nm band.
const float AIRGLOW_DISPLAY_PREEXPOSURE = 131072.0;

// Geomagnetic north pole (IGRF dipole, 80.8 N 72.6 W) in the scene's
// mirrored frame (east longitude = -atan(y, x), see sphere_uv).
const vec3 GEOMAGNETIC_POLE = vec3(0.0478, 0.1526, 0.9871);

// `far_half` dims the second crossing of a limb ray (which misses the
// ground) by the air below its tangent point: the march's transmittance.
vec3 night_airglow(vec3 camera, vec3 ray, vec3 sun, float surface_hit, float far_half) {
    vec3 tangent = camera - ray * dot(camera, ray);
    float slant = 0.0;
    if (surface_hit > 0.0) {
        if (dot(tangent - camera, ray) > surface_hit) tangent = camera + ray * surface_hit;
        float t_layer = sphere_intersection(camera, ray, surface_radius * (1.0 + 94.0 / 6378.137));
        vec3 layer_normal = normalize(camera + ray * max(t_layer, 0.0));
        slant = 1.0 / max(dot(-ray, layer_normal), 0.02);
    }
    float tangent_km = (length(tangent) - surface_radius) * KM_PER_UNIT;
    if (tangent_km > 400.0) return vec3(0.0);
    vec3 up = normalize(tangent);
    float dark = 1.0 - smoothstep(-0.25, -0.05, dot(up, sun));
    if (dark <= 0.0) return vec3(0.0);
    // Beyond a pixel the layers are unresolved: widen them, keeping the
    // height-integrated brightness.
    float footprint_km = length(tangent - camera) * KM_PER_UNIT
        * 2.0 * frame.projection_tangents.y / frame.canvas_rect.w;
    float w_green = sqrt(25.0 + footprint_km * footprint_km);
    float w_na = sqrt(16.0 + footprint_km * footprint_km);
    float w_red = sqrt(2025.0 + footprint_km * footprint_km);
    // 630 nm: brightest within ~18 degrees of the magnetic equator.
    float magnetic_latitude = asin(clamp(dot(up, GEOMAGNETIC_POLE), -1.0, 1.0));
    float red_rayleigh = 60.0 + 240.0 * exp(-magnetic_latitude * magnetic_latitude / 0.0987);
    vec3 glow = vec3(0.20, 1.0, 0.42) * 400.0 * airglow_column(tangent_km, 97.0, w_green, slant) * sqrt(5.0 / w_green)
        + vec3(1.0, 0.62, 0.02) * 100.0 * airglow_column(tangent_km, 91.0, w_na, slant) * sqrt(4.0 / w_na)
        + vec3(1.0, 0.30, 0.04) * 130.0 * airglow_column(tangent_km, 87.0, w_na, slant) * sqrt(4.0 / w_na)
        + vec3(1.0, 0.10, 0.03) * red_rayleigh * airglow_column(tangent_km, 250.0, w_red, slant) * sqrt(45.0 / w_red);
    if (slant <= 0.0) glow *= far_half;
    return glow * 4.7e-12 * AIRGLOW_GAIN * dark;
}

// --- Lightning -----------------------------------------------------------
// Flashes belong to the 1 degree cells (~110 km, wider than any glow) of the
// weather fields where NOAA GFS reports convective energy, rain and cloud
// water (the cell's storm blob). Every cell keeps its own clock, so no two
// flash on a common beat, and draws a new flash in each slot of FLASH_SLOT
// seconds: position, size, length and stroke pattern all differ, nothing
// repeats. A flash is the intracloud kind that fills the top of a storm: a
// soft elongated glow, 5-17 km across (the Gaussian sigma), that lasts
// 0.1-0.4 s, several return strokes (each decaying in ~20-50 ms) and a
// dimmer continuing glow behind them. Big flashes are bright, wide and long;
// most are small. The cells around the pixel are all asked, so a glow is
// never cut at a cell edge.
const float FLASH_SLOT = 0.8;
// Flashes per second of a cell with blob 1. The pipeline puts storms on the
// observed cold cloud tops (data_pipeline.py steer_storms), ~2.7x the
// model's storm area; 0.17 keeps the global rate near the observed
// 44-46 flashes per second.
const float FLASH_RATE = 0.17;
// Frames are drawn at 8-20 per second: averaging each stroke over the last
// 90 ms, as a camera does, shows every flash in the frame it falls in.
const float FLASH_WINDOW = 0.09;
// Radiance scale of a flash in sunlight units: its centre reaches a few 1e-5,
// the brightest ~1e-4 of sunlit cloud.
const float FLASH_RADIANCE = 5.0e-5;
// Brightness of the 1st-4th return stroke of a flash, before its own variation.
const vec4 STROKE_WEIGHT = vec4(1.0, 0.62, 0.48, 0.38);

// The night series is EV 16 (night_gain brings a flash there from a shorter
// exposure). A camera opened further, EV 19 from the ISS at night, would burn
// every flash out to a flat white blob: a flash keeps the display brightness
// it has at EV ~17.5 instead, so its structure stays while its core still
// clips. Held at EV 16 (3 stops back) flashes peaked at 0.04-0.13 beside
// city lights at 1.0; in ISS footage every flash core clips like a city.
float flash_gain() {
    return min(1.0, 2.8 * NIGHT_SERIES_PREEXPOSURE / preexposure());
}

// Mean over the window ending at `time` of a pulse that starts at `onset` and
// decays with time constant `tau`.
float flash_pulse(float time, float onset, float tau) {
    if (time <= onset) return 0.0;
    float from = max(time - FLASH_WINDOW, onset);
    return tau * (exp(-(from - onset) / tau) - exp(-(time - onset) / tau)) / FLASH_WINDOW;
}

// How thick and lumpy the cloud top is under a flash, 0 (a gap) to 1 (a
// tower): the glow follows it, so a flash shows the cauliflower top it lights
// instead of a smooth disc. The fractal is read three times coarser than a
// pixel: towers a few km across, not pixel noise.
float flash_lumps(vec3 normal, float pixel_km) {
    vec3 w = pow(abs(normal), vec3(8.0));
    w /= w.x + w.y + w.z;
    float mean = textureLod(tiling_noise, vec2(0.5), 16.0).r;
    float lumps = 0.0;
    for (int octave = 0; octave < 2; ++octave) {
        float tile_km = octave == 0 ? 160.0 : 45.0;
        float lod = log2(max(3.0 * pixel_km * 256.0 / tile_km, 1.0));
        vec3 p = normal * (R_GROUND / tile_km) + float(octave) * 0.37;
        vec3 s = vec3(textureLod(tiling_noise, p.yz, lod).r,
                      textureLod(tiling_noise, p.zx, lod).r,
                      textureLod(tiling_noise, p.xy, lod).r);
        float n = mean + (dot(w, s) - mean) * inversesqrt(dot(w, w));
        lumps += (octave == 0 ? 0.65 : 0.35) * (n - mean);
    }
    return clamp(0.5 + 3.5 * lumps, 0.0, 1.0);
}

// Light the storms add to the cloud tops at map position `uv` (`normal` is the
// same point as a direction); `pixel_km` is the ground size of a pixel.
vec3 lightning_light(vec2 uv, vec3 normal, float pixel_km, float time) {
    if (frame.material_state.y < 0.5) return vec3(0.0);
    ivec2 size = textureSize(weather_fields, 0) / 4;
    vec2 grid = vec2(size);
    ivec2 home = ivec2(floor(uv * grid));
    vec3 light = vec3(0.0);
    float lumps = -1.0;
    // The compiler must not be able to count these as nine. Unrolled, it
    // fetched all nine cells up front and the whole Earth shader needed 116
    // registers instead of 64: every pixel ran slower, not only the storms.
    int cells = 9 + int(min(preexposure(), 0.0));
    for (int cell = 0; cell < cells; ++cell) {
        int row = home.y + cell / 3 - 1;
        if (row < 0 || row >= size.y) continue;
        ivec2 texel = ivec2((home.x + cell % 3 - 1 + size.x) % size.x, row);
        vec3 weather = textureLod(weather_fields, (vec2(texel) + 0.5) / grid, 2.0).rgb;
        float storm_blob = smoothstep(0.025, 0.4, weather.g) * smoothstep(0.05, 0.5, weather.b)
            * smoothstep(0.3, 0.8, weather.r);
        if (storm_blob <= 0.0) continue;
        // A cell is 111 km tall and 111 cos(latitude) km wide.
        float cos_lat = cos((0.5 - (float(row) + 0.5) / grid.y) * PI);
        float cell_km = 111.0 * max(cos_lat, 0.25);
        vec2 strike_cell = vec2(texel);
        // The cell's own clock, and an activity that waxes and wanes over
        // ~9 s (mean 1): storms flash in bursts, not at an even rate.
        vec4 own = storm_random4(strike_cell, 0u);
        float slot_time = time / FLASH_SLOT + own.x;
        float phase = floor(slot_time);
        float local = (slot_time - phase) * FLASH_SLOT;
        // Read at the start of the slot: were it read now, the rate would
        // move while a flash is on and start or stop it halfway.
        float wave = 0.5 - 0.5 * cos(6.2831853 * fract((phase - own.x) * (FLASH_SLOT / 9.0) + own.y));
        float rate = FLASH_RATE * storm_blob * cos_lat * (0.25 + 2.0 * wave * wave);
        float roll = storm_random(strike_cell, uint(phase));
        if (roll >= rate * FLASH_SLOT) continue;

        vec4 draw = storm_random4(strike_cell, uint(phase) ^ 0x68e31da4u);
        vec4 jitter = storm_random4(strike_cell, uint(phase) ^ 0xb5297a4du);
        vec4 place = storm_random4(strike_cell, uint(phase) ^ 0x1b56c4e9u);
        float flash_size = pow(draw.x, 2.2);
        float duration = 0.10 + 0.30 * pow(flash_size, 0.7);
        // The flash, tail included, ends inside its slot.
        float start = draw.y * max(FLASH_SLOT - duration - 0.25, 0.0);
        float finish = start + duration + 0.24;
        if (local < start || local > finish) continue;

        // Fades out towards the edge of the cells asked, so no glow can
        // be cut there however bright it is.
        vec2 from_cell = uv * grid - (strike_cell + 0.5);
        from_cell.x -= round(from_cell.x / grid.x) * grid.x;
        float inside = 1.0 - smoothstep(1.0, 1.5, max(abs(from_cell.x), abs(from_cell.y)));
        if (inside <= 0.0) continue;

        // A storm re-flashes around its core: every cell has a hot spot,
        // each flash lands within a quarter of a cell of it.
        vec2 center = (strike_cell + clamp(0.2 + 0.6 * own.zw + 0.5 * (place.xy - 0.5), 0.05, 0.95)) / grid;
        vec2 offset = uv - center;
        offset.x -= round(offset.x);
        vec2 km = vec2(offset.x * 2.0 * PI * cos((0.5 - center.y) * PI), offset.y * PI) * R_GROUND;
        float sigma = 4.5 + 13.0 * pow(flash_size, 0.8);
        // The odd flash lights a whole cloud shield. None is wider than
        // the cells asked about it can carry (a tail of 4 sigma).
        if (jitter.w > 0.96) sigma *= 1.6;
        sigma = min(sigma, 0.22 * cell_km);
        // The glow spreads over the first ~100 ms, as the discharge branches.
        sigma *= 0.55 + 0.45 * smoothstep(0.0, 0.12, local - start);
        // Under a pixel the glow cannot be narrower than the pixel: widen
        // it and give up a share of its peak.
        float wide = sqrt(sigma * sigma + 0.3 * pixel_km * pixel_km);
        float stretch = 1.0 + 1.2 * jitter.x;
        vec2 axis = normalize(place.zw - 0.5 + 1.0e-3);
        float along = dot(km, axis);
        float across = dot(km, vec2(-axis.y, axis.x));
        float r2 = (along * along / stretch + across * across * stretch) / (wide * wide);
        // Out to ~10 widths for the halo below; the core and glow end by 6.
        if (r2 > 100.0) continue;
        if (lumps < 0.0) lumps = flash_lumps(normal, pixel_km);
        float reach = 0.6 + 0.9 * lumps;       // thick lobes of the cloud reach further
        float strength = 0.55 + 0.9 * lumps;   // and glow brighter
        // The tail is faded out over its last 0.2 s: it is still a few
        // percent of the peak where the flash ends, and would vanish at once.
        float fade = 1.0 - smoothstep(finish - 0.2, finish, local);
        float amplitude = (0.45 + 0.75 * flash_size) * sigma / wide * inside * fade;

        // The continuing glow sits on the flash; every return stroke lights
        // the cloud a little further along it, so the lit part changes
        // from one stroke to the next instead of one disc fading.
        float tau = 0.018 + 0.030 * draw.z;
        int strokes = 2 + int(draw.w * 3.0);
        float glow = 0.2 * flash_pulse(local, start, 0.04 + 0.5 * duration);
        float temporal = glow;
        for (int k = 0; k < 4; ++k) {
            if (k >= strokes) break;
            float onset = start + duration * (float(k) + (k == 0 ? 0.0 : 0.8 * jitter[k])) / float(strokes);
            float pulse = STROKE_WEIGHT[k] * (0.6 + 0.8 * fract(jitter[k] * 17.31 + draw[k]))
                * flash_pulse(local, onset, tau);
            temporal += pulse;
            if (pulse < 1.0e-3) continue;
            float shift = k == 0 ? 0.0 : (fract(jitter[k] * 7.7 + draw.x) - 0.5) * 2.4 * wide;
            float a = along - shift;
            float stroke_r2 = (a * a / stretch + across * across * stretch) / (wide * wide * reach);
            if (stroke_r2 > 18.0) continue;
            // Blue-violet in the deck (footage halos R/G 0.5-0.8, B/G
            // 1.3-2.4) around a small white core that clips.
            light += (vec3(0.55, 0.50, 1.0) * exp(-0.5 * stroke_r2) + vec3(2.2, 2.2, 2.4) * exp(-4.0 * stroke_r2))
                * (pulse * amplitude * strength);
        }
        float glow_r2 = r2 / (1.4 * reach);
        if (glow_r2 < 18.0) {
            light += vec3(0.55, 0.50, 1.0) * (exp(-0.5 * glow_r2) * glow * amplitude * strength);
        }
        // The cloud deck around the flash, lit through its thick parts out
        // to 5-15 core widths at ~10 % of the core, as in the footage. Kept
        // inside the cells asked, and faded to nothing by the cut above (it
        // is still ~15 % of its peak at 6 widths: a hard rim there).
        float halo_sigma = min(3.5 * wide, 0.30 * cell_km);
        float halo_r2 = (along * along / sqrt(stretch) + across * across * sqrt(stretch)) / (halo_sigma * halo_sigma);
        if (halo_r2 < 18.0) {
            light += vec3(0.45, 0.45, 1.0) * (exp(-0.5 * halo_r2) * (1.0 - smoothstep(45.0, 100.0, r2))
                * lumps * lumps * 0.12 * amplitude * temporal);
        }
    }
    return light;
}

void main() {
    vec2 surface_uv = in_uv;
    vec2 global_xy = frame.viewport_rect.xy + surface_uv * frame.viewport_rect.zw;
    vec2 canonical_ndc = vec2(
        2.0 * (global_xy.x - frame.projection_tangents.z) / frame.canvas_rect.z,
        2.0 * (frame.projection_tangents.w - global_xy.y) / frame.canvas_rect.w
    );
    vec3 ray = normalize(
        frame.camera_forward.xyz
        + frame.camera_right.xyz * canonical_ndc.x * frame.projection_tangents.x
        + frame.camera_up.xyz * canonical_ndc.y * frame.projection_tangents.y);

    vec3 camera = frame.camera_position_distance.xyz;
    // Earth uniforms already Y-mirror the Sun/Moon to match the westward map
    // U, so lighting and the sky disc share one vector. Do not mirror again.
    vec3 sun = normalize(frame.sun_direction.xyz);
    vec3 moon = moon_direction();
    float earth_distance = sphere_intersection(camera, ray, surface_radius);
    // map_uv + gradients in uniform control flow: the discard and the
    // earth_distance branch both diverge at silhouettes.
    vec3 earth_point = camera + ray * earth_distance;
    vec3 earth_normal = normalize(earth_point);
    vec2 map_uv = sphere_uv(earth_normal);
    vec2 map_dx = dFdx(map_uv);
    vec2 map_dy = dFdy(map_uv);
    float atmosphere_t0, atmosphere_t1;
    bool in_air = sphere_span(camera, ray, atmosphere_radius, atmosphere_t0, atmosphere_t1) && atmosphere_t1 > 0.0;
    if (!in_air && sphere_intersection(camera, ray, night_glow_top_radius) < 0.0) {
        discard;
    }

    float solar_au = frame.celestial_distances.x * (6378.137 / 149597870.7);
    float sun_irradiance = 1.0 / (solar_au * solar_au);
    float moon_irradiance = moonlight_scale();
    float exposure = preexposure();
    // Moonlight matters only once the camera has opened up for the night.
    bool with_moon = moon_irradiance * exposure > 2.0e-4;
    float view_sun = dot(ray, sun);
    float view_moon = dot(ray, moon);
    // Scattered radiance per unit irradiance is per steradian; the scene
    // unit (white Lambertian = 1) is irradiance / pi, hence the factor pi.
    Light sun_light = Light(sun, vec3(PI * sun_irradiance), SUN_ANGULAR_RADIUS,
        rayleigh_phase(view_sun), aerosol_phase(view_sun));
    Light moon_light = Light(moon, vec3(PI * moon_irradiance), MOON_ANGULAR_RADIUS,
        rayleigh_phase(view_moon), aerosol_phase(view_moon));

    vec3 origin_km = camera * KM_PER_UNIT;
    float t_entry = max(atmosphere_t0, 0.0) * KM_PER_UNIT;
    float t_exit = atmosphere_t1 * KM_PER_UNIT;
    vec3 radiance = vec3(0.0);
    vec3 transmittance = vec3(1.0);
    float coverage = 0.0;
    // For the meter (post.rs read_meter), which sees this texel scaled by
    // the camera's exposure: true where it is (sunlit ground or air), false
    // where night_gain holds it at the night series' EV 16 instead. The meter
    // must undo that gain, or the night side looks darker to it at every
    // stop the camera opens up and the exposure chases itself. Only the sign
    // of the alpha carries it: its size stays the coverage.
    bool metered_lit = true;

    if (earth_distance > 0.0) {
        float t_ground = earth_distance * KM_PER_UNIT;
        vec3 point = earth_point;
        vec3 normal = earth_normal;
        float r_ground = R_GROUND;
        vec4 day_material = sample_day_material(map_uv, map_dx, map_dy);
        float water = day_material.a;
        vec3 shading_normal = relief_normal(normal, map_uv, map_dx, map_dy);
        float mu_sun = dot(normal, sun);
        float mu_moon = dot(normal, moon);
        // Sunlit or twilit ground: until the Sun is ~9 degrees down, the
        // sunlit air above the ground (over ~80 km) still outshines the night
        // terms at a twilight exposure. Cut at -0.10, that air read 2^(16-EV)
        // too dark to the meter, which then jumped to the night exposure at
        // dusk while the twilit side of the frame blew out.
        metered_lit = mu_sun > -0.16;

        // Cloud shadows: march the sun-leg from the surface into the cloud
        // shell and attenuate direct sun by the density found there.
        float cloud_shadow = 0.0;
        if (mu_sun > -0.02) {
            float sun_leg = sphere_intersection(point + normal * 0.0005, sun, cloud_radius);
            if (sun_leg > 0.0) {
                vec3 shadow_normal = normalize(point + normal * 0.0005 + sun * sun_leg);
                cloud_shadow = clamp(sample_cloud_density(sphere_uv(shadow_normal), shadow_normal), 0.0, 1.0);
            }
        }
        // Thick cloud blocks ~90 % of the direct beam; thin cloud forward-
        // scatters most of it. At 70 % the shadows of cumulus stayed ~1.6
        // stops dark, where ISS footage shows them nearly black on the sea
        // (sea in shadow / cloud top ~1/20).
        float sun_shadow = 1.0 - 0.9 * pow(cloud_shadow, 1.6);

        vec3 albedo = mix(vec3(dot(day_material.rgb, vec3(0.2126, 0.7152, 0.0722))), day_material.rgb, surface_saturation);
        // Open ocean: Blue Marble paints water a flat, too-dark fill. Use
        // Case-1 water-leaving reflectance averaged over the sRGB colour
        // bands (single wavelengths, Rrs 443/555/670 x pi, made it 1.7x too
        // dark and navy against the footage's deep water) and keep the map
        // where it is brighter (shallow banks, turbid coasts).
        const vec3 deep_water = vec3(0.0006, 0.0075, 0.033);
        albedo = mix(albedo, max(day_material.rgb, deep_water), water);
        // Blue Marble oceans are open all year; lay today's pack ice over
        // them. Snow-covered first-year ice is a little greyer and bluer
        // than the ice sheet, and it has no specular glint.
        if (live_sea_ice() && water > 0.0) {
            float ice = water * textureGrad(cloud_terrain_height, map_uv, map_dx, map_dy).g;
            albedo = mix(albedo, vec3(0.70, 0.75, 0.80), ice);
            water -= ice;
        }

        float sun_visible = solar_visibility(point);
        vec3 sun_beam = light_transmittance(r_ground, mu_sun, SUN_ANGULAR_RADIUS) * sun_irradiance * sun_visible;
        vec3 sky = sky_irradiance_at(r_ground, mu_sun) * sun_irradiance * sun_visible;
        // The tables assume the climatological aerosol; correct the beam for
        // the local load (dust or smoke dims and reddens it) and hand what
        // the extra aerosol scatters, mostly forward, to the diffuse sky.
        vec3 extra_depth = (aerosol_extinction_at(normal) - AEROSOL_EXTINCTION) * AEROSOL_SCALE;
        vec3 beam_change = exp(-extra_depth / max(mu_sun, 0.05));
        vec3 horizontal_beam = sun_beam * max(mu_sun, 0.0);
        sky = max(sky + horizontal_beam * (1.0 - beam_change) * AEROSOL_ALBEDO * 0.8, vec3(0.0));
        sun_beam *= beam_change;
        float lambert = max(dot(shading_normal, sun), 0.0);
        vec3 ground = albedo * (sun_beam * lambert * sun_shadow + sky * (1.0 - 0.6 * cloud_shadow));
        // Ocean surface: Cox-Munk sunglint plus the Fresnel reflection of
        // the sky (~2 % near nadir, rising toward grazing views, where the
        // mirrored ray sees the bright sky near the horizon rather than the
        // sky's average).
        float view_cos = max(dot(normal, -ray), 0.0);
        float fresnel = 0.02037 + 0.97963 * pow(1.0 - view_cos, 5.0);
        // Calm streaks and slicks (a few km to a few hundred) glint
        // brighter, rough water dimmer: the texture inside ISS sunglint,
        // where one 7 m/s everywhere drew a smooth even glow. The clouds'
        // 1/f^2 fractal stretched 4:1 along the mostly zonal winds (tiles
        // ~500 by 125 km, read ~4 km fine); isotropic it read as haze.
        // (Varying the Cox-Munk slope variance itself cost 8 more registers.)
        float calm = 1.0;
        if (water > 0.0) {
            calm = clamp(1.0 + 1.5 * (textureLod(tiling_noise, map_uv * vec2(80.0, 160.0), 2.0).r
                - textureLod(tiling_noise, vec2(0.5), 16.0).r), 0.55, 1.45);
        }
        // Light scattered through a cloud makes no glint: the cloud's shadow
        // takes it all, a dark hole in the glint as in the footage. The
        // camera's saturation tinted a white glint peach; footage shows it
        // silver under a high Sun.
        vec3 glint = cox_munk_glint(normal, sun, ray) * sun_beam * ((1.0 - cloud_shadow) * calm);
        glint = mix(vec3(dot(glint, vec3(0.2126, 0.7152, 0.0722))), glint, 0.75);
        ground += water * (glint + fresnel * sky * (1.0 + 1.5 * (1.0 - view_cos)));

        // Night: moonlight, starlight/airglow and city lights.
        vec3 moon_beam = light_transmittance(r_ground, mu_moon, MOON_ANGULAR_RADIUS) * moon_irradiance;
        float night = 1.0 - smoothstep(-0.05, 0.02, mu_sun);
        vec3 night_light = NIGHT_TINT * night_gain(mu_sun);
        ground += albedo * (moon_beam * max(dot(shading_normal, moon), 0.0) + NIGHT_SKY_IRRADIANCE) * night_light;
        ground += water * cox_munk_glint(normal, moon, ray) * moon_beam * night_light;
        // Lights are hidden by daylight at any exposure; skip the fetches.
        if (night > 0.0) {
            // Black Marble lights are grayscale radiance; colour them by lamp
            // type and intensity: sodium suburbs read orange, LED suburbs
            // cool white, and dense cores warm white in both.
            float lights = nasa_lights(sample_static(static_night_atlas, night_emission, map_uv, map_dx, map_dy).r);
            float bright = smoothstep(0.02, 0.25, lights);
            vec3 sodium_lamp = mix(vec3(1.0, 0.42, 0.08), vec3(1.0, 0.64, 0.24), bright);
            vec3 led_lamp = mix(vec3(0.80, 0.90, 1.0), vec3(1.0, 0.97, 0.90), bright);
            vec3 city = mix(led_lamp, sodium_lamp, SODIUM_SHARE) * lights;
            // Light scattered by the air over a city: a faint wide halo.
            float texture_width = float(textureSize(night_emission, 0).x);
            float narrow = city_glow_at(map_uv, max(log2(texture_width * 0.0044), 0.0));
            float wide = city_glow_at(map_uv, max(log2(texture_width * 0.0120), 0.0));
            city += DIFFUSE_LAMP * (narrow * 0.06 + wide * 0.04);
            ground += city * CITY_RADIANCE * night * night_gain(mu_sun);
        }

        // Air between the cloud tops and the ground, then the ground itself.
        float t_cloud = sphere_intersection(camera, ray, cloud_radius) * KM_PER_UNIT;
        bool through_cloud = t_cloud > 0.0 && t_cloud < t_ground && !clouds_off();
        float t_split = through_cloud ? t_cloud : t_ground;
        vec3 above_radiance = vec3(0.0);
        vec3 above_transmittance = vec3(1.0);
        // Short downward paths need fewer samples than grazing views;
        // preserve the 16-step integration near the horizon and the
        // separate 32-step limb march below.
        int air_steps = view_cos > 0.35 ? 12 : 16;
        march(origin_km, ray, t_entry, t_split, t_split, air_steps, sun_light, moon_light, with_moon,
            above_radiance, above_transmittance);
        vec3 below_radiance = vec3(0.0);
        vec3 below_transmittance = vec3(1.0);
        if (through_cloud) {
            march(origin_km, ray, t_cloud, t_ground, t_ground, 6, sun_light, moon_light, with_moon,
                below_radiance, below_transmittance);
        }
        vec3 under = below_radiance + below_transmittance * ground;

        if (through_cloud) {
            vec3 cloud_normal = normalize(camera + ray * (t_cloud / KM_PER_UNIT));
            vec2 cloud_uv = sphere_uv(cloud_normal);
            float view_mu = max(dot(cloud_normal, -ray), 0.05);
            float slant = min(sqrt(1.0 - view_mu * view_mu) / view_mu, 12.0);
            vec3 cloud_sample;
            if (live_clouds()) {
                cloud_sample = live_cloud(cloud_uv, cloud_normal, slant);
            } else {
                float nasa = 1.0 - pow(1.0 - nasa_cloud_opacity(cloud_uv), 1.0 + CLOUD_ASPECT * slant);
                cloud_sample = vec3(nasa, nasa_cloud_albedo(cloud_uv), nasa);
            }
            float cloud_opacity = cloud_sample.x;
            float r_cloud = R_GROUND + 5.5;
            // Cloud tops are lumpy: shade them as a height field whose
            // relief follows the cloud's depth (~1.5 km from edge to core).
            // Its slope comes from screen-space derivatives (no fetches),
            // solved into the tangent plane of the shell.
            vec3 shell_km = cloud_normal * r_cloud;
            vec3 px = dFdx(shell_km);
            vec3 py = dFdy(shell_km);
            vec2 relief_d = 1.5 * vec2(dFdx(cloud_sample.z), dFdy(cloud_sample.z));
            float gxx = dot(px, px);
            float gxy = dot(px, py);
            float gyy = dot(py, py);
            float det = gxx * gyy - gxy * gxy;
            vec3 slope = det > 1.0e-12
                ? ((gyy * relief_d.x - gxy * relief_d.y) * px + (gxx * relief_d.y - gxy * relief_d.x) * py) / det
                : vec3(0.0);
            slope *= min(1.0, 1.2 / max(length(slope), 1.0e-6));
            vec3 top_normal = normalize(cloud_normal - slope);
            float cloud_mu = dot(cloud_normal, sun);
            // Light diffuses inside a cloud, so the shading wraps: a face
            // turned from the Sun is dimmed, not black.
            float top_mu = dot(top_normal, sun);
            float relief_light = clamp(1.0 + 0.75 * (top_mu - cloud_mu) / max(cloud_mu, 0.12), 0.35, 1.6);
            float moon_relief = clamp(1.0 + 0.75 * (dot(top_normal, moon) - dot(cloud_normal, moon))
                / max(dot(cloud_normal, moon), 0.12), 0.35, 1.6);
            float cloud_moon_mu = dot(cloud_normal, moon);
            float cloud_albedo = cloud_sample.y;
            // Thick cloud tops scatter strongly back toward the Sun and stay
            // bright at grazing light (not Lambertian): a small forward/back
            // term keeps sunset tops lit, as ISS photos show.
            float cloud_lambert = 0.12 * smoothstep(-0.02, 0.05, cloud_mu) + 0.88 * max(cloud_mu, 0.0);
            vec3 cloud_sun = light_transmittance(r_cloud, cloud_mu, SUN_ANGULAR_RADIUS) * sun_irradiance
                * solar_visibility(cloud_normal * cloud_radius);
            vec3 cloud_sky = sky_irradiance_at(r_cloud, cloud_mu) * sun_irradiance;
            vec3 cloud_moon = light_transmittance(r_cloud, cloud_moon_mu, MOON_ANGULAR_RADIUS) * moon_irradiance;
            float cloud_night = 1.0 - smoothstep(-0.12, 0.0, cloud_mu);
            vec3 cloud_colour = cloud_albedo * (cloud_sun * cloud_lambert * relief_light + 0.6 * cloud_sky
                + (cloud_moon * max(cloud_moon_mu, 0.0) * moon_relief + NIGHT_SKY_IRRADIANCE)
                    * NIGHT_TINT * night_gain(cloud_mu));
            if (cloud_night > 0.0) {
                // Thunderstorm lightning as seen from orbit: the flash lights the
                // cloud from inside, so it takes the cloud's own shape,
                // brightest through its thick cores.
                float scatter = smoothstep(0.20, 0.60, cloud_opacity);
                float thick = 0.3 + 0.9 * cloud_sample.z;
                if (scatter > 0.0) {
                    cloud_colour += lightning_light(cloud_uv, cloud_normal, max(length(px), length(py)), frame.celestial_state.w)
                        * (cloud_night * scatter * thick * FLASH_RADIANCE * night_gain(cloud_mu) * flash_gain());
                }
                // City lights glow upward into low night cloud.
                // The cloud diffuses them: a wide glow in the lamps' colour.
                float city_signal_cloud = city_signal_at(cloud_uv, vec2(0.0));
                float glow_width = float(textureSize(night_emission, 0).x);
                float city_diffuse = city_glow_at(cloud_uv, max(log2(glow_width * 0.0012), 0.0))
                    + city_glow_at(cloud_uv, max(log2(glow_width * 0.0044), 0.0));
                cloud_colour += DIFFUSE_LAMP * (0.35 * pow(city_signal_cloud, 1.4) + 0.35 * city_diffuse)
                    * cloud_night * CITY_RADIANCE * night_gain(cloud_mu);
            }
            under = mix(under, cloud_colour, cloud_opacity);
        }
        radiance = above_radiance + above_transmittance * under;
        transmittance = vec3(0.0);
        coverage = 1.0;
    } else if (in_air) {
        // Limb: the densest air is at the ray's closest approach.
        float t_closest = -dot(origin_km, ray);
        march(origin_km, ray, t_entry, t_exit, t_closest, 32, sun_light, moon_light, with_moon,
            radiance, transmittance);
        coverage = 1.0 - dot(transmittance, vec3(0.2126, 0.7152, 0.0722));
        // Sunlit air: the ray's closest approach lies outside the Earth's
        // shadow, eased over the lowest 60 km, where the higher air along
        // the ray still catches the Sun. Otherwise only the airglow and the
        // aurora light it, at the night gain. (A deeper cut made the twilit
        // limb of a night view read as a daylit crescent, and the meter
        // exposed for it: aurora and lights went dim.)
        vec3 tangent = origin_km + ray * clamp(t_closest, t_entry, t_exit);
        float r_tangent = length(tangent);
        float mu_tangent = dot(tangent, sun) / r_tangent;
        float shadow_axis = r_tangent * sqrt(max(1.0 - mu_tangent * mu_tangent, 0.0));
        metered_lit = mu_tangent >= 0.0 || shadow_axis > R_GROUND - 30.0;
    }
    float far_half = 0.5 + 0.5 * dot(transmittance, vec3(0.2126, 0.7152, 0.0722));
    radiance += AURORA_GAIN * aurora_emission(camera, ray, sun, earth_distance) * night_gain()
        + night_airglow(camera, ray, sun, earth_distance, far_half)
        * (night_sky_gate(camera, sun) * AIRGLOW_DISPLAY_PREEXPOSURE / exposure);
    // The coverage's sign is the meter's flag (see metered_lit); the blend
    // passes it through (stars leave alpha 0) and nothing else reads it.
    out_color = vec4(min(radiance * exposure, vec3(30000.0)), metered_lit ? coverage : -coverage);
    out_transmittance = vec4(transmittance, 1.0);
}
