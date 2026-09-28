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
// Aerosol optical depth 0.18 at 550 nm, Angstrom 0.5, scale height 1.8 km.
const vec3 AEROSOL_EXTINCTION = vec3(9.270248e-2, 1.004577e-1, 1.118034e-1);
const float AEROSOL_ALBEDO = 0.94;
const float AEROSOL_SCALE = 1.8;
const float AEROSOL_G = 0.68;
const float SUN_ANGULAR_RADIUS = 0.004654;
const float MOON_ANGULAR_RADIUS = 0.004516;

float preexposure() {
    return frame.camera_position_distance.w;
}

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
    return 3.14159265 * fresnel * facets / (4.0 * nv) * step(0.0, dot(normal, light));
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
float cloud_noise(vec3 n, float tile_km, vec2 offset) {
    vec3 p = n * (R_GROUND / tile_km);
    vec3 w = pow(abs(n), vec3(8.0));
    w /= w.x + w.y + w.z;
    float mean = textureLod(tiling_noise, vec2(0.5), 16.0).r;
    vec3 s = vec3(texture(tiling_noise, p.yz + offset).r,
                  texture(tiling_noise, p.zx + offset).r,
                  texture(tiling_noise, p.xy + offset).r);
    return mean + (dot(w, s) - mean) * inversesqrt(dot(w, w));
}

float cloud_detail(vec3 n) {
    return 0.68 * cloud_noise(n, 222.0, vec2(0.0))
        + 0.32 * cloud_noise(n, 28.0, vec2(0.37, 0.71));
}

// Real cloud morphology at 1 km (cloud streets, open and closed cells,
// fronts) from the NASA Blue Marble cloud composite, mixed into the detail
// so observed cover is sculpted like clouds rather than noise. Where that
// historical map was clear the fractal alone decides.
float cloud_morphology(vec2 map_uv, vec3 n) {
    float composite = sample_nasa_clouds(map_uv).r;
    return mix(cloud_detail(n), composite, 0.45);
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
vec2 live_cloud(vec2 map_uv, vec3 n) {
    vec2 dx = dFdx(map_uv);
    vec2 dy = dFdy(map_uv);
    dx.x -= round(dx.x);
    dy.x -= round(dy.x);
    float cover = clamp(live_cloud_cover(map_uv, dx, dy), 0.0, 1.0);
    float morphology = cloud_morphology(map_uv, n);
    float mean = 0.55 * textureLod(tiling_noise, vec2(0.5), 16.0).r + 0.45 * 0.244;
    float threshold = mean + morphology_quantile_offset(1.0 - cover);
    float band = max(0.012, 0.5 * fwidth(morphology));
    float opacity = smoothstep(threshold - band - 0.015, threshold + band + 0.015, morphology);
    float depth = smoothstep(0.0, 0.22, morphology - threshold);
    vec2 near = vec2(opacity, mix(0.5, 0.92, depth));
    // From afar the mips average the morphology toward its mean, so the
    // quantile cut above collapses into an on/off switch at 50 % cover:
    // flat white cut-outs tracing the 10 km grid on the globe. Once a pixel
    // spans several km, show the observed cover as the fraction it is, with
    // the (low-passed) morphology left as texture, and thin cover greyer.
    float km_x = length(vec2(dx.x * 40075.0 * sqrt(max(1.0 - n.z * n.z, 0.0)), dx.y * 20037.5));
    float km_y = length(vec2(dy.x * 40075.0 * sqrt(max(1.0 - n.z * n.z, 0.0)), dy.y * 20037.5));
    float far = smoothstep(1.5, 8.0, max(km_x, km_y));
    vec2 wide = vec2(clamp(cover + (morphology - mean) * 1.2 * (1.0 - abs(2.0 * cover - 1.0)), 0.0, 1.0),
        mix(0.55, 0.9, smoothstep(0.3, 0.95, cover)));
    return mix(near, wide, far);
}

float live_cloud_opacity(vec2 map_uv, vec3 n) {
    return live_cloud(map_uv, n).x;
}

float sample_cloud_density(vec2 mesh_uv0, vec3 cloud_normal) {
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

// Black Marble 2016 is an 8-bit display product (sRGB-coded BC4). Mapped to
// radiance so the brightest city cores reach ~25x a full-Moon-lit desert
// (VIIRS DNB: ~500 vs ~20 nW cm^-2 sr^-1), in sunlight units.
const float CITY_RADIANCE = 1.5e-5;
float nasa_lights(float encoded) {
    return pow(encoded, 2.2);
}

// City-light radiance at a map position (linear, from the sRGB-coded BC4).
float city_signal_at(vec2 map_uv, vec2 offset) {
    vec2 uv = map_uv + offset;
    return nasa_lights(sample_static(static_night_atlas, night_emission, uv, dFdx(uv), dFdy(uv)).r);
}

float city_glow_at(vec2 map_uv, float lod) {
    return nasa_lights(textureLod(night_emission, map_uv, lod).r);
}

// Aurora shell: emission lives between 90 and 320 km.
const float aurora_top_radius = surface_radius * (1.0 + 320.0 / 6378.137);

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

// Volumetric march through the 90-320 km auroral shell. NOAA OVATION gives
// where the oval is and how active it is; the structure follows DMSP/VIIRS
// night imagery and ISS photography:
//  - a diffuse, patchy glow filling the oval (brightest equatorward);
//  - discrete arcs along its poleward flank that follow the oval's contours,
//    fold and break up, multiplying with activity;
//  - field-aligned rays, constant along the (near-vertical) magnetic field:
//    vertical striations when a curtain is seen edge-on at the limb;
//  - altitude profiles: O(1S) 557.7 nm green with a sharp lower border near
//    100 km and a ~30 km scale height above; O(1D) 630 nm red spread over
//    180-320 km (the crimson top in ISS photos); N2+ 391/428 nm and N2 1PG
//    pink-violet along the lower edge of bright arcs.
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
        // Invisible against sunlit air: fade through nautical twilight.
        float visible = 1.0 - smoothstep(-0.16, -0.02, dot(n, sun));
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
        float fold_km = (fold - 0.675) * 150.0;
        float spacing = max(abs(slope), 4.0e-5);
        // A bright arc is a bundle of 1-10 km sheets; seen from orbit it
        // reads as a soft band ~30 km wide. Under a footprint it widens with
        // its energy conserved rather than aliasing.
        const float arc_km = 30.0;
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
        float rays = mix(1.0, 0.35 + 1.3 * ray_noise * ray_noise, 0.8 * (1.0 - smoothstep(20.0, 60.0, footprint_km)));
        float pulse = 0.88 + 0.12 * sin(time * 1.1 + ray_noise * 9.0);
        // Diffuse aurora: smooth large patches over the whole oval.
        float patches = aurora_noise(uv + vec2(time * 1.0e-4, 0.0), vec2(26.0, 13.0), footprint_uv);
        // ~1 kR at 25 % probability, rising with the energy flux in storms.
        float diffuse = smoothstep(0.02, 0.25, probability) * max(1.0, probability / 0.25)
            * (1.0 - 0.5 * poleward) * (0.4 + 1.2 * patches * patches);

        float green = smoothstep(94.0, 104.0, h) * exp(-max(h - 108.0, 0.0) / 30.0);
        float red_x = (h - 235.0) / 65.0;
        float red = exp(-red_x * red_x) * smoothstep(150.0, 190.0, h);
        float fringe_x = (h - 97.0) / 4.0;
        float fringe = exp(-fringe_x * fringe_x);
        // kR per km: diffuse ~1 kR overhead over ~50 km of column, arcs
        // ~20 kR over ~40 km.
        float discrete = arcs * rays * pulse;
        vec3 local = vec3(0.30, 1.0, 0.30) * green * (0.03 * diffuse + 0.55 * discrete)
            + vec3(1.0, 0.07, 0.12) * red * (0.012 * diffuse + 0.05 * discrete)
            + vec3(0.9, 0.25, 0.8) * fringe * 0.25 * discrete * smoothstep(0.3, 1.0, discrete);
        emission += local * visible * step_km;
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
//   O(1S) 557.7 nm at 97 km, ~400 R near solar maximum: green
//   Na D 589 nm at 91 km, ~100 R: yellow-orange
//   OH Meinel (visible red tail) at 87 km
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

vec3 night_airglow(vec3 camera, vec3 ray, vec3 sun, float surface_hit) {
    vec3 tangent = camera - ray * dot(camera, ray);
    float slant = 0.0;
    if (surface_hit > 0.0) {
        if (dot(tangent - camera, ray) > surface_hit) tangent = camera + ray * surface_hit;
        float t_layer = sphere_intersection(camera, ray, surface_radius * (1.0 + 94.0 / 6378.137));
        vec3 layer_normal = normalize(camera + ray * max(t_layer, 0.0));
        slant = 1.0 / max(dot(-ray, layer_normal), 0.02);
    }
    float tangent_km = (length(tangent) - surface_radius) * KM_PER_UNIT;
    if (tangent_km > 160.0) return vec3(0.0);
    float dark = 1.0 - smoothstep(-0.25, -0.05, dot(normalize(tangent), sun));
    if (dark <= 0.0) return vec3(0.0);
    // Beyond a pixel the layers are unresolved: widen them, keeping the
    // height-integrated brightness.
    float footprint_km = length(tangent - camera) * KM_PER_UNIT
        * 2.0 * frame.projection_tangents.y / frame.canvas_rect.w;
    float w_green = sqrt(25.0 + footprint_km * footprint_km);
    float w_na = sqrt(16.0 + footprint_km * footprint_km);
    vec3 glow = vec3(0.35, 1.0, 0.05) * 400.0 * airglow_column(tangent_km, 97.0, w_green, slant) * sqrt(5.0 / w_green)
        + vec3(1.0, 0.62, 0.02) * 100.0 * airglow_column(tangent_km, 91.0, w_na, slant) * sqrt(4.0 / w_na)
        + vec3(1.0, 0.18, 0.03) * 40.0 * airglow_column(tangent_km, 87.0, w_na, slant) * sqrt(4.0 / w_na);
    return glow * 4.7e-12 * dark;
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
    if (!in_air && (frame.material_state.z < 0.5 ||
        sphere_intersection(camera, ray, aurora_top_radius) < 0.0)) {
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
        // Thick decks block ~70 % of the direct beam (the rest arrives as
        // cloud-scattered diffuse light); thin cloud forward-scatters most.
        float sun_shadow = 1.0 - 0.7 * pow(cloud_shadow, 1.6);

        vec3 albedo = mix(vec3(dot(day_material.rgb, vec3(0.2126, 0.7152, 0.0722))), day_material.rgb, surface_saturation);
        // Open ocean: Blue Marble paints water a flat, too-dark fill. Use
        // Case-1 water-leaving reflectance (Rrs 443/555/670 ~ 8e-3, 1.5e-3,
        // 2e-4 sr^-1, times pi) and keep the map where it is brighter
        // (shallow banks, turbid coasts).
        const vec3 deep_water = vec3(0.0006, 0.0047, 0.025);
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
        vec3 ground = albedo * (sun_beam * lambert * sun_shadow + sky * (1.0 - 0.45 * cloud_shadow));
        // Ocean surface: Cox-Munk sunglint plus the Fresnel reflection of
        // the sky (~2 % near nadir, rising toward grazing views).
        float view_cos = max(dot(normal, -ray), 0.0);
        float fresnel = 0.02037 + 0.97963 * pow(1.0 - view_cos, 5.0);
        ground += water * (cox_munk_glint(normal, sun, ray) * sun_beam * sun_shadow + fresnel * sky);

        // Night: moonlight, starlight/airglow and city lights.
        vec3 moon_beam = light_transmittance(r_ground, mu_moon, MOON_ANGULAR_RADIUS) * moon_irradiance;
        ground += albedo * (moon_beam * max(dot(shading_normal, moon), 0.0) + NIGHT_SKY_IRRADIANCE);
        ground += water * cox_munk_glint(normal, moon, ray) * moon_beam;
        // Lights are hidden by daylight at any exposure; skip the fetches.
        float night = 1.0 - smoothstep(-0.05, 0.02, mu_sun);
        if (night > 0.0) {
            // Black Marble lights are grayscale radiance; colour them from
            // intensity: dim suburbs read sodium orange, dense cores warm white.
            float lights = nasa_lights(sample_static(static_night_atlas, night_emission, map_uv, map_dx, map_dy).r);
            vec3 city = mix(vec3(1.0, 0.48, 0.14), vec3(1.0, 0.72, 0.38), smoothstep(0.02, 0.25, lights)) * lights;
            // Light scattered by the air over a city: a faint wide halo.
            float texture_width = float(textureSize(night_emission, 0).x);
            float narrow = city_glow_at(map_uv, max(log2(texture_width * 0.0044), 0.0));
            float wide = city_glow_at(map_uv, max(log2(texture_width * 0.0120), 0.0));
            city += vec3(1.0, 0.55, 0.25) * (narrow * 0.06 + wide * 0.04);
            ground += city * CITY_RADIANCE * night;
        }

        // Air between the cloud tops and the ground, then the ground itself.
        float t_cloud = sphere_intersection(camera, ray, cloud_radius) * KM_PER_UNIT;
        bool through_cloud = t_cloud > 0.0 && t_cloud < t_ground;
        float t_split = through_cloud ? t_cloud : t_ground;
        vec3 above_radiance = vec3(0.0);
        vec3 above_transmittance = vec3(1.0);
        march(origin_km, ray, t_entry, t_split, t_split, 24, sun_light, moon_light, with_moon,
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
            vec2 cloud_sample = live_clouds() ? live_cloud(cloud_uv, cloud_normal)
                : vec2(nasa_cloud_opacity(cloud_uv), nasa_cloud_albedo(cloud_uv));
            float cloud_opacity = cloud_sample.x;
            float r_cloud = R_GROUND + 5.5;
            float cloud_mu = dot(cloud_normal, sun);
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
            vec3 cloud_colour = cloud_albedo * (cloud_sun * cloud_lambert + 0.6 * cloud_sky
                + cloud_moon * max(cloud_moon_mu, 0.0) + NIGHT_SKY_IRRADIANCE);
            float cloud_night = 1.0 - smoothstep(-0.12, 0.0, cloud_mu);
            if (cloud_night > 0.0) {
                // Thunderstorm lightning as seen from orbit: brief flashes that
                // bloom under a cloud top, flicker (several return strokes), and
                // cluster inside mesoscale convective systems (NOAA GFS CAPE,
                // precipitation and cloud water).
                const vec2 grid_size = vec2(720.0, 360.0);
                vec2 strike_grid = cloud_uv * grid_size;
                vec2 strike_cell = floor(strike_grid);
                vec2 suv = fract(strike_grid);
                vec2 cell_center_uv = (strike_cell + 0.5) / grid_size;
                vec3 weather = texture(weather_fields, cell_center_uv).rgb;
                float storm_blob = smoothstep(0.025, 0.4, weather.g) * smoothstep(0.05, 0.5, weather.b)
                    * smoothstep(0.3, 0.8, weather.r) * frame.material_state.y;
                float t = frame.celestial_state.w;
                float life = 1.4;
                float phase = floor(t / life);
                float roll = storm_random(strike_cell, uint(phase));
                float fx = fract(t / life);
                float p1 = hash21(strike_cell + 9.1) * 0.5;
                float p2 = p1 + 0.12 + hash21(strike_cell + 4.4) * 0.18;
                float stroke_one = (fx - p1) * 26.0;
                float stroke_two = (fx - p2) * 22.0;
                float flicker = exp(-(stroke_one * stroke_one))
                    + 0.7 * exp(-(stroke_two * stroke_two))
                    + 0.3 * step(p1, fx) * exp(-max(fx - p1, 0.0) * 18.0);
                const float frequency = 0.035;
                float strike = step(1.0 - frequency * storm_blob, roll) * flicker * cloud_night;
                vec2 bloom_center = vec2(hash21(strike_cell + 11.3), hash21(strike_cell + 47.9));
                float d = length((suv - bloom_center) * vec2(1.6, 1.0));
                float bloom_arg = d * 7.0;
                float core_arg = d * 19.0;
                float bloom = exp(-(bloom_arg * bloom_arg));
                float core = exp(-(core_arg * core_arg));
                float scatter = smoothstep(0.20, 0.60, cloud_opacity);
                // A lightning-lit cloud top is ~1e-4 of sunlit cloud.
                cloud_colour += (vec3(0.55, 0.68, 1.0) * bloom * 3.0 + vec3(0.92, 0.96, 1.0) * core * 4.0)
                    * strike * scatter * 4.0e-5;
                // City lights glow upward into low night cloud.
                float city_signal_cloud = city_signal_at(cloud_uv, vec2(0.0));
                cloud_colour += vec3(1.0, 0.52, 0.22) * pow(city_signal_cloud, 1.4) * cloud_night
                    * 0.35 * CITY_RADIANCE;
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
    }
    radiance += aurora_emission(camera, ray, sun, earth_distance) + night_airglow(camera, ray, sun, earth_distance);
    out_color = vec4(min(radiance * exposure, vec3(30000.0)), coverage);
    out_transmittance = vec4(transmittance, 1.0);
}
