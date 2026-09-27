#version 460

layout(location = 0) noperspective in vec2 in_uv;
layout(location = 0) out vec4 out_color;

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

layout(set = 1, binding = 0) uniform sampler2DArray vt_day_atlas;
layout(set = 1, binding = 1) uniform usampler2DArray vt_page_table;
layout(set = 1, binding = 2, std140) uniform VtParams {
    uvec4 base_dimensions_mip_count_enabled;
    uvec4 page_table_dimensions_slots_tile_size;
} vt;

layout(push_constant) uniform FrameData {
    vec4 camera_position_distance;
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
// Global mean cloud-top height (ISCCP ~500-600 hPa). At 12 km every cloud's
// shadow landed twice as far out, reading as a doubled cloud field.
const float cloud_radius = surface_radius * (1.0 + 5.5 / 6378.137);
const float atmosphere_radius = surface_radius * (1.0 + 100.0 / 6378.137);
const float civil_twilight_sine = -0.10452846326765347;
const float inv_two_pi = 0.15915494309189535;
const float inv_pi = 0.3183098861837907;
const float atmosphere_shell_thickness = atmosphere_radius - surface_radius;
// SDR presentation. The exposure keeps mid-tones (0.3 linear) where the old
// linear 0.75 scale put them.
const float exposure = 0.82;
const float surface_saturation = 0.72;
// Sea-level Rayleigh optical depth per km of the 8.5 km scale-height column at
// the RGB channel centres (680/550/440 nm), Bodhaine et al. (1999): tau =
// 0.0410 / 0.0971 / 0.2426. The former constants were ~17% high, over-blueing the sea.
const vec3 rayleigh_per_km = vec3(0.004819, 0.011419, 0.028542);
// Global-mean aerosol optical depth at 550 nm (MODIS/AERONET ~0.1-0.15).
const float vertical_aerosol_optical_depth = 0.05;

vec3 filmic(vec3 x) {
    // Toe: a slight deepening of the darkest values; stronger toes crushed
    // clear ocean far below DSCOVR/EPIC's measured grey-blue. Shoulder: an exponential roll-off above 0.72 so cloud tops,
    // glint and the limb compress instead of hard-clipping in the 8-bit target.
    x = max(x, vec3(0.0));
    x = x * x / (x + 0.011);
    const float knee = 0.72;
    vec3 shoulder = knee + (1.0 - knee) * (1.0 - exp(-(x - knee) / (1.0 - knee)));
    return mix(x, shoulder, step(knee, x));
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

float density_column_km(float mu) {
    vec2 encoded = textureLod(atmosphere_column,
        vec2((sqrt(clamp(mu, 0.0, 1.0)) * 511.0 + 0.5) / 512.0, 0.5), 0.0).rg;
    return dot(encoded, vec2(65280.0, 255.0)) * (400.0 / 65535.0);
}

float front_atmosphere_path(
    vec3 origin,
    vec3 direction,
    float atmosphere_entry,
    float surface_hit) {
    // Every permitted camera is above the 100 km shell. Its surface-to-space
    // column therefore depends only on zenith angle, baked once at build time.
    vec3 normal = normalize(origin + direction * surface_hit);
    return density_column_km(dot(normal, -direction)) / 8.5;
}

// Mean sunlight transmittance over the air a view path actually scatters
// from, for single scattering with both extinctions growing along the path:
// [(1 - e^-(tv+ts)) / (tv+ts)] / [(1 - e^-tv) / tv]. When Sun and view share
// the path (full-disk rim, ts ~ tv) it stays near (1 + e^-tv)/2 and the rim
// keeps its blue haze; for a short view through a grazing sun path (the
// terminator from above) it removes the blue and leaves twilight orange.
vec3 sun_path_factor(vec3 view_depth, vec3 sun_depth) {
    vec3 total = view_depth + sun_depth;
    vec3 both = (vec3(1.0) - exp(-total)) / max(total, vec3(1.0e-4));
    vec3 view = (vec3(1.0) - exp(-view_depth)) / max(view_depth, vec3(1.0e-4));
    return both / max(view, vec3(1.0e-4));
}

vec3 composite_front_atmosphere(
    vec3 surface_colour,
    vec3 origin,
    vec3 direction,
    vec3 sun,
    float atmosphere_entry,
    float surface_hit,
    float surface_sun_height,
    float sun_visible) {
    vec3 atmosphere_point = origin + direction * atmosphere_entry;
    vec3 atmosphere_normal = normalize(atmosphere_point);
    float sun_height = dot(atmosphere_normal, sun);
    float daylight = smoothstep(civil_twilight_sine, 0.0, sun_height);
    // The shell must never outlive the surface night: near the terminator
    // the entry point can stay lit while the ground below is dark, which
    // painted a blue fringe onto the night side. Unify on the tighter gate.
    float surface_gate = smoothstep(civil_twilight_sine, 0.0, surface_sun_height);
    daylight = min(daylight, surface_gate);
    // The front shell is a volume path, not a Fresnel edge mask: an edge term
    // would make the atmosphere vanish at the disk center, while real air
    // scatters over the whole visible disk. The shell-only branch below
    // handles the outer limb.
    const float vertical_rayleigh_optical_depth = 0.097065;
    float path_length = front_atmosphere_path(
        origin,
        direction,
        atmosphere_entry,
        surface_hit);
    float optical_depth = vertical_rayleigh_optical_depth * path_length;
    // Relative Rayleigh scattering at representative red, green, and blue
    // wavelengths, normalized at 550 nm (Bodhaine optical depths) folded offline so
    // no per-fragment pow() evaluates a pure constant.
    const vec3 rayleigh = vec3(0.421976, 1.0, 2.499406);
    float aerosol_transmittance = exp(-vertical_aerosol_optical_depth * path_length);
    vec3 transmittance = exp(-optical_depth * rayleigh) * aerosol_transmittance;
    // Night early-out: daylight is exactly 0.0 below the civil-twilight gate,
    // so both additive scattering terms below contribute exactly 0.0. Surface
    // attenuation (transmittance) is still required and stays on this path.
    if (daylight <= 0.0) {
        return surface_colour * transmittance;
    }
    // Rayleigh phase relative to Lambertian reflected sunlight (pi / 4pi).
    float scattering_cosine = dot(direction, sun);
    float rayleigh_phase = 0.1875 * (1.0 + scattering_cosine * scattering_cosine);
    // Physical (1.0) relative to the Lambertian surface: the former 1.6 SDR
    // lift washed the whole day disk pale blue. The outer limb branch keeps
    // its own exposure, so the thin limb stays visible at globe scale.
    const float atmosphere_exposure = 1.0;
    // Sunlight reaching the scattering air has itself crossed the atmosphere:
    // near the terminator that path is up to ~38 air masses, which removes the
    // blue and leaves the orange twilight band seen from orbit. Without it
    // the sky glowed full-strength blue past the terminator. The Moon's
    // shadow (solar eclipse) dims it too.
    float scatter_sun_height = max(0.5 * (sun_height + surface_sun_height), 0.0);
    vec3 sun_depth = rayleigh_per_km * density_column_km(scatter_sun_height);
    vec3 sun_path = sun_path_factor(optical_depth * rayleigh, sun_depth) * sun_visible;
    vec3 atmosphere_colour = surface_colour * transmittance
        + atmosphere_exposure * rayleigh_phase * (vec3(1.0) - transmittance) * daylight * sun_path;

    // Aerosols: global-mean optical depth ~0.12 at 550 nm (MODIS/AERONET
    // climatology), single-scattering albedo 0.95, spectrally near-grey.
    // Phase is 85 % Henyey-Greenstein g = 0.72 (forward glare when the Sun
    // is behind Earth) + 15 % isotropic, in the same pi-scaled units as the
    // Rayleigh term. This haze is what makes the real ocean grey-blue and
    // softens land colour in DSCOVR/EPIC images.
    const float aerosol_g = 0.72;
    float view_sun_cosine = clamp(dot(direction, sun), -1.0, 1.0);
    float hg_denominator = max(1.0 + aerosol_g * aerosol_g - 2.0 * aerosol_g * view_sun_cosine, 1.0e-4);
    float aerosol_phase = 0.85 * (1.0 - aerosol_g * aerosol_g)
            * inversesqrt(hg_denominator * hg_denominator * hg_denominator) * 0.25
        + 0.15 * 0.25;
    float mie_scatter = 0.95 * (1.0 - aerosol_transmittance) * min(aerosol_phase, 6.0);
    return atmosphere_colour + mie_scatter * daylight * sun_path;
}

// Moonlight. Lunar illuminance follows Allen's phase law, 10^(-0.4(0.026a +
// 4e-9 a^4)) with a in degrees, scaled by distance; full Moon at its mean
// distance gives ~2.5e-6 of sunlight. That is invisible at daylight exposure,
// so a night-adaptation gain presents it the way a dark-adapted eye or an
// ISS night photograph does: under a full Moon, land and cloud tops are
// faintly readable but clearly darker than day. Around new Moon the night
// side correctly goes dark apart from lights.
const float night_adaptation = 20000.0;

vec3 moon_direction() {
    return normalize(vec3(frame.camera_forward.w, frame.camera_right.w, frame.camera_up.w));
}

float moonlight_scale() {
    float a = degrees(frame.celestial_state.z);
    float phase_law = pow(10.0, -0.4 * (0.026 * a + 4.0e-9 * a * a * a * a));
    float distance_ratio = 60.27 / max(frame.celestial_distances.y, 1.0);
    return 2.5e-6 * phase_law * distance_ratio * distance_ratio * night_adaptation;
}

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
     // Ocean waves are far below pixel scale from orbit; their slopes are
     // handled statistically by the Cox-Munk glint lobe, not by visible
     // sine trains (which drew ~27 km diagonal stripes across the sea).
     vec3 wavy_normal = tangent_normal;
    float equatorial_length = length(geometric_normal.xy);
    if (equatorial_length < 1.0e-4) {
        return geometric_normal;
    }
    // U increases westward and V increases southward in the map coordinates.
    vec3 tangent_u = vec3(geometric_normal.y, -geometric_normal.x, 0.0)
        / equatorial_length;
    vec3 tangent_v = normalize(cross(geometric_normal, tangent_u));
     return normalize(
        tangent_u * wavy_normal.x
        + tangent_v * wavy_normal.y
        + geometric_normal * wavy_normal.z);
}

vec4 sample_cloud_map(sampler2D source, vec2 map_uv, vec2 tiling) {
    // Choose mips before tiling/wrapping. fract() derivatives across a tile
    // edge select coarse mips and draw a dark line through otherwise fine clouds.
    // Repair the sphere's longitude discontinuity before scaling derivatives;
    // the REPEAT sampler handles tile boundaries without a coordinate jump.
    vec2 dx = dFdx(map_uv);
    vec2 dy = dFdy(map_uv);
    dx.x -= round(dx.x);
    dy.x -= round(dy.x);
    return textureGrad(source, map_uv * tiling, dx * tiling, dy * tiling);
}

float cloud_noise(vec2 map_uv, vec2 tiling) {
    return sample_cloud_map(tiling_noise, map_uv, tiling).r;
}

// NASA Blue Marble cloud map: BC4 display-encoded
// cloud brightness. Reflectance saturates with optical depth, so anything
// displayed moderately bright is an optically thick deck and must be opaque;
// a literal brightness inversion left bright cumulus ~40% transparent, which
// showed each cloud's own offset shadow through it as a dark "double".
// Faint values stay translucent haze and thin cirrus.
float nasa_cloud_opacity(vec2 map_uv) {
    float brightness = sample_cloud_map(clouds_a, map_uv, vec2(1.0)).r;
    return smoothstep(0.12, 0.72, brightness);
}

// Thin cloud is optically thin in albedo too: scale the lit cloud top from
// grey (thin) to white (thick deck) so tops keep texture instead of flat white.
float nasa_cloud_albedo(vec2 map_uv) {
    float brightness = sample_cloud_map(clouds_a, map_uv, vec2(1.0)).r;
    return mix(0.72, 1.0, smoothstep(0.25, 0.8, brightness));
}

float sample_cloud_density(vec2 mesh_uv0, vec3 cloud_normal) {
    return nasa_cloud_opacity(mesh_uv0);
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
// NASA night lights are BC4 grayscale Black Marble, stored sRGB-encoded (no
// sRGB BC4 format exists), so decode here.
float nasa_lights(float encoded) {
    return pow(encoded, 2.2);
}

// City-light radiance at a map position (linear, from the sRGB-coded BC4).
float city_signal_at(vec2 map_uv, vec2 offset) {
    return nasa_lights(texture(night_emission, map_uv + offset).r);
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

vec3 aurora_emission(vec3 camera, vec3 ray, vec3 sun, float surface_hit) {
    if (frame.material_state.z < 0.5) return vec3(0.0);
    // Volumetric march through the auroral shell. OVATION supplies where the
    // oval is; discrete arcs are bands along its probability contours, and
    // field-aligned rays are constant in altitude, so from the side (the limb)
    // they become vertical curtains and from above they read as thin arcs.
    // Emission profiles: O(1S) 557.7 nm green with a sharp ~100 km lower
    // border peaking near 110 km; O(1D) 630 nm red around 200-300 km; N2+
    // violet on the lower edge of bright arcs.
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

    const int steps = 12;
    float dt = (t1 - t0) / float(steps);
    // Stable per-pixel dither (interleaved gradient noise) hides step banding.
    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float t = frame.celestial_state.w;
    float pixel_angle = 2.0 * frame.projection_tangents.y / frame.canvas_rect.w;
    float km_per_unit = 6378.137 / surface_radius;
    vec3 emission = vec3(0.0);
    for (int i = 0; i < steps; ++i) {
        float ts = t0 + (float(i) + jitter) * dt;
        vec3 p = camera + ray * ts;
        float r = length(p);
        vec3 n = p / r;
        float h = (r - surface_radius) * km_per_unit;
        // Aurora vanishes against a sunlit sky; a wide ramp keeps the
        // terminator crossing soft when a curtain is seen edge-on at the limb.
        float visible = 1.0 - smoothstep(-0.25, 0.15, dot(n, sun));
        if (visible <= 0.0) continue;
        vec2 uv = sphere_uv(n);
        // OVATION is a coarse grid; a ~2-degree mip keeps contours smooth.
        float probability = textureLod(weather_fields, uv, 3.0).a;
        if (probability < 0.015) continue;
        float footprint = ts * pixel_angle / (6.2831853 * r * max(length(n.xy), 0.2));
        // Latitude-direction footprint, across which arcs are spaced.
        float footprint_v = ts * pixel_angle / (3.14159265 * r);
        float oval = smoothstep(0.02, 0.16, probability);
        // Folds evolve over tens of seconds; arcs drift slowly equatorward.
        float fold = aurora_noise(uv + vec2(t * 3.0e-4, t * 4.0e-5), vec2(3.0, 1.5), footprint)
            + 0.35 * aurora_noise(uv - vec2(t * 9.0e-4, 0.0), vec2(11.0, 4.0), footprint);
        float phase = probability * 18.0 + fold * 0.9 - t * 0.012;
        float arc = pow(0.5 + 0.5 * sin(phase * 6.2831853), 16.0);
        // Arcs sit ~0.015 in V apart (OVATION gradient x 18 bands) and
        // each is ~1/6 of that wide; below ~2 pixels fade to the mean,
        // C(32,16)/2^32, instead of aliasing.
        float arc_pixels = 0.015 / (6.0 * max(footprint_v, 1.0e-7));
        arc = mix(0.14, arc, smoothstep(1.5, 3.0, arc_pixels));
        // Field-aligned rays: independent of altitude, flickering over
        // seconds and surging eastward along the arcs.
        // Tiling is ~isotropic on the ground at auroral latitudes (~60 km
        // cells), so from above rays read as beads, from the side as curtains.
        float ray_noise = aurora_noise(vec2(uv.x + t * 2.5e-4, uv.y), vec2(220.0, 200.0), footprint);
        float rays = 0.35 + 1.4 * ray_noise * ray_noise;
        float pulse = 0.8 + 0.2 * sin(t * 1.3 + ray_noise * 12.0);
        float green = smoothstep(92.0, 108.0, h) * exp(-max(h - 110.0, 0.0) / 38.0);
        float red_h = (h - 245.0) / 55.0;
        float red = exp(-red_h * red_h);
        float violet_h = (h - 100.0) / 5.0;
        float violet = exp(-violet_h * violet_h);
        float structure = 0.12 + arc * rays * pulse;
        vec3 local = vec3(0.20, 1.0, 0.42) * green * structure
            + vec3(0.9, 0.08, 0.12) * red * (0.10 + 0.25 * arc) * 0.15
            + vec3(0.45, 0.25, 1.0) * violet * arc * rays * 0.25;
        emission += local * oval * visible * (dt * km_per_unit);
    }
    return emission * 0.007;
}

// Night airglow: a thin O(1S) layer near 95 km. Invisible looking straight
// down, but a tangent line of sight crosses ~40x more of it, which is the
// faint green band ISS photos show along the night limb.
vec3 night_airglow(vec3 camera, vec3 ray, vec3 sun, float surface_hit) {
    vec3 tangent = camera - ray * dot(camera, ray);
    if (surface_hit > 0.0 && dot(tangent - camera, ray) > surface_hit) {
        tangent = camera + ray * surface_hit;
    }
    float km_per_unit = 6378.137 / surface_radius;
    float h = (length(tangent) - surface_radius) * km_per_unit;
    // Integrate the 7 km layer over the pixel footprint: from far away the
    // band is sub-pixel, and evaluating it at the pixel centre drew a razor-
    // sharp 1-pixel line around the planet. Widening conserves its energy.
    float footprint_km = length(tangent - camera) * km_per_unit
        * 2.0 * frame.projection_tangents.y / frame.canvas_rect.w;
    float sigma = sqrt(49.0 + footprint_km * footprint_km);
    float band_h = (h - 95.0) / sigma;
    float band = exp(-band_h * band_h) * (7.0 / sigma);
    float dark = 1.0 - smoothstep(-0.25, -0.05, dot(normalize(tangent), sun));
    return vec3(0.35, 0.75, 0.30) * band * dark * 0.02;
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
    float atmosphere_discriminant;
    float atmosphere_distance = sphere_intersection(camera, ray, atmosphere_radius, atmosphere_discriminant);
    float earth_distance = sphere_intersection(camera, ray, surface_radius);
    // map_uv + gradients and atmosphere fwidth in uniform control flow: the
    // discard and earth_distance branch both diverge at silhouettes.
    vec3 earth_point = camera + ray * earth_distance;
    vec3 earth_normal = normalize(earth_point);
    vec2 map_uv = sphere_uv(earth_normal);
    vec2 map_dx = dFdx(map_uv);
    vec2 map_dy = dFdy(map_uv);
    float atmosphere_coverage_width = max(fwidth(atmosphere_discriminant), 1.0e-5);
    if (atmosphere_distance < 0.0 && (frame.material_state.z < 0.5 ||
        sphere_intersection(camera, ray, aurora_top_radius) < 0.0)) {
        discard;
    }
    vec3 colour = vec3(0.0);
    float alpha = 0.0;
    if (earth_distance > 0.0) {
        vec3 point = earth_point;
        vec3 normal = earth_normal;
        vec4 day_material = sample_day_material(map_uv, map_dx, map_dy);
        vec3 shading_normal = perturb_surface_normal(normal, map_uv, map_dx, map_dy, sun, day_material.a);
        float geometric_sunlight = dot(normal, sun);
        // The Sun is a 0.53-degree disk; its hard penumbra is only used for
        // the specular water glint. Diffuse albedo uses a wider soft gate so
        // the terminator reads as a graded sunset, not a 0.26-degree line.
        float solar_penumbra = frame.celestial_distances.z / frame.celestial_distances.x;
        float direct_day = smoothstep(-solar_penumbra, solar_penumbra, geometric_sunlight);
        // Surface albedo is lit only while the Sun is at or above the horizon.
        // The half-degree soft toe absorbs grazing-terrain roughness without
        // leaking day texture onto the night hemisphere (the old violet /
        // blue-hour terms painted albedo below the horizon, showing the dark
        // side as if it were day).
        float surface_day = smoothstep(-0.010, 0.085, geometric_sunlight);
        float night_visibility = 1.0 - smoothstep(-0.060, 0.010, geometric_sunlight);
        float sunlight = dot(shading_normal, sun);
        // Grazing blend: micro-facet perturbation would kill the diffuse
        // before geometric sunset, hiding the warm bands under noise. Ease
        // toward the flat-normal response as the sun drops so the terminator
        // stays smooth and tintable; detail rules the high sun.
        float flat_sun = max(geometric_sunlight, 0.0);
        float graze = 1.0 - smoothstep(0.0, 0.25, geometric_sunlight);
        float daylight = max(mix(sunlight, flat_sun, graze), 0.0);
        // Cloud shadows: march the sun-leg from the surface into the cloud
        // shell and attenuate direct sun by the density found there, so
        // convective decks ground themselves instead of floating shadowless.
        // Day-gated: no sun, no shadows, no second density evaluation.
        float cloud_shadow = 0.0;
        if (surface_day > 0.02) {
             float sun_leg = sphere_intersection(point + normal * 0.0005, sun, cloud_radius);
             if (sun_leg > 0.0) {
                 vec3 shadow_hit = point + normal * 0.0005 + sun * sun_leg;
                 vec3 shadow_normal = normalize(shadow_hit);
                 cloud_shadow = clamp(
                     sample_cloud_density(sphere_uv(shadow_normal), shadow_normal),
                     0.0, 1.0);
             }
        }
        // Thin cloud forward-scatters most sunlight through, so the shadow
        // grows faster than linearly with opacity; thick decks block ~60 %
        // (the rest arrives as diffuse skylight and cloud-scattered light).
        float sun_shadow = 1.0 - 0.6 * pow(cloud_shadow, 1.6) * clamp(surface_day, 0.0, 1.0);
        // Blue Marble NG is a contrast-enhanced mosaic; measured against
        // DSCOVR/EPIC true colour its land is ~1.4x too saturated. Pull the
        // albedo toward its own luminance by that calibrated amount.
        vec3 day = mix(vec3(dot(day_material.rgb, vec3(0.2126, 0.7152, 0.0722))), day_material.rgb, surface_saturation);
        // Beer-Lambert attenuation on the sunlight path gives a spectral
        // sunset without adding energy to unlit terrain.
        vec3 light_tint = exp(-rayleigh_per_km
            * density_column_km(max(geometric_sunlight, 0.0)));
        float sun_visible = solar_visibility(point);
        light_tint *= sun_visible;
        float solar_distance_au = frame.celestial_distances.x * (6378.137 / 149597870.7);
        light_tint /= solar_distance_au * solar_distance_au;
        float diffuse = daylight;

        // Black Marble lights are grayscale radiance; colour them from
        // intensity: dim suburbs read sodium orange, dense cores warm white.
        float lights = nasa_lights(texture(night_emission, map_uv).r);
        vec3 night = mix(vec3(1.0, 0.50, 0.16), vec3(1.0, 0.80, 0.52), smoothstep(0.02, 0.25, lights))
            * lights * 2.6;
        // Offline-filtered mip levels provide two bloom scales in two fetches.
        vec3 city_halo = vec3(0.0);
        if (night_visibility > 0.02) {
             float texture_width = float(textureSize(night_emission, 0).x);
             float narrow = city_glow_at(map_uv, max(log2(texture_width * 0.0044), 0.0));
             float wide = city_glow_at(map_uv, max(log2(texture_width * 0.0120), 0.0));
            city_halo = vec3(1.0, 0.55, 0.25) * (narrow * 0.60 + wide * 0.45);
            city_halo *= 0.25;
        }
        // Cox-Munk sunglint: wave facets have a Gaussian slope distribution
        // (variance 0.003 + 0.00512 * wind m/s, ~7 m/s here), giving the
        // smooth, broad glint patch seen from orbit. Expressed in the same
        // albedo-times-cosine units as the diffuse term (pi * BRDF * cos_i).
        float water_glint = day_material.a * direct_day * cox_munk_glint(normal, sun, ray);
        colour = day * diffuse * light_tint * surface_day * sun_shadow
             + night * night_visibility + city_halo * night_visibility;
        // Moonlit ground and moon glint on water, only where the Sun is down.
        vec3 moon = moon_direction();
        float moonlight = moonlight_scale() * night_visibility;
        float moon_glint = cox_munk_glint(normal, moon, ray) * day_material.a;
        colour += (day * max(dot(normal, moon), 0.0) + moon_glint)
            * moonlight * exp(-rayleigh_per_km * density_column_km(max(dot(normal, moon), 0.0)));
        // Specular reflection carries the Sun's own spectrum; light_tint
        // already reddens it near the terminator. A blue tint turned lakes
        // at the specular point cyan.
        colour += water_glint * light_tint * sun_shadow;

        // Starlight-scale residual, relative to the daylight exposure.
        const float night_floor = 0.000002;
        float planet_day = smoothstep(-0.23, 0.23, geometric_sunlight);
        colour += day * night_floor * (1.0 - planet_day);

        float cloud_distance = sphere_intersection(camera, ray, cloud_radius);
        if (cloud_distance > 0.0 && cloud_distance < earth_distance) {
            vec3 cloud_normal = normalize(camera + ray * cloud_distance);
            vec2 cloud_uv = sphere_uv(cloud_normal);
            float cloud_density = sample_cloud_density(cloud_uv, cloud_normal);
            float cloud_opacity = cloud_density;
            float cloud_sunlight = dot(cloud_normal, sun);
            float cloud_wrap = max(cloud_sunlight, 0.0);
            float cloud_daylight = smoothstep(civil_twilight_sine, 0.0, cloud_sunlight);
            float cloud_golden = (1.0 - smoothstep(0.010, 0.22, cloud_sunlight))
                * smoothstep(-0.010, 0.020, cloud_sunlight);
            float cloud_ember = (1.0 - smoothstep(0.000, 0.060, cloud_sunlight))
                * smoothstep(-0.012, 0.000, cloud_sunlight);
            vec3 cloud_tint = vec3(0.86, 0.93, 1.0);
            cloud_tint = mix(cloud_tint, vec3(1.0, 0.62, 0.32), cloud_golden * 0.9);
            cloud_tint = mix(cloud_tint, vec3(1.0, 0.42, 0.26), cloud_ember * 0.95);
            cloud_tint *= nasa_cloud_albedo(cloud_uv);
            vec3 cloud_colour = cloud_tint * (0.08 + 0.92 * cloud_wrap) * sun_visible;
            float cloud_night = 1.0 - cloud_daylight;
            // Jupiter's night floor is the surface albedo. Earth's clouds are a
            // separate shell, so the same 7% lift turns night into a grey haze.
            // Keep night cloud as a thin, dark veil so continents stay readable.
            float cloud_planet_day = smoothstep(-0.23, 0.23, cloud_sunlight);
            // Night clouds are moonlit (dark without a Moon) and still hide
            // much of the ground; lights glow through thin cloud below.
            float cloud_moonlight = moonlight_scale() * max(dot(cloud_normal, moon_direction()), 0.0);
            cloud_colour = mix(vec3(0.9, 0.93, 1.0) * cloud_moonlight, cloud_colour, cloud_planet_day);
            cloud_opacity *= mix(0.7, 1.0, cloud_planet_day);
            // Thunderstorm lightning as seen from orbit: brief flashes that
            // bloom under a cloud top, flicker (several return strokes), and
            // cluster inside mesoscale convective systems, with quiet sky
            // between. They are localized glows, not the whole cloud
            // silhouette and not a uniform random sprinkle per cell.
            const vec2 grid_size = vec2(720.0, 360.0);
            vec2 strike_grid = cloud_uv * grid_size;
            vec2 strike_cell = floor(strike_grid);
            vec2 suv = fract(strike_grid);
            vec2 cell_center_uv = (strike_cell + 0.5) / grid_size;
            // Storms come from the NOAA GFS fields: convective energy (CAPE),
            // precipitation and cloud water, gated to deep night cloud, so
            // flashes cluster in the day's real convective systems.
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
            // Two sharp strokes plus a short exponential afterglow.
            float stroke_one = (fx - p1) * 26.0;
            float stroke_two = (fx - p2) * 22.0;
            float flicker = exp(-(stroke_one * stroke_one))
                + 0.7 * exp(-(stroke_two * stroke_two))
                + 0.3 * step(p1, fx) * exp(-max(fx - p1, 0.0) * 18.0);
            // Instability controls event frequency, not the energy of every
            // individual stroke. Small forecast probabilities must not turn
            // all otherwise valid lightning into invisible dim glows.
            const float frequency = 0.035;
            float strike = step(1.0 - frequency * storm_blob, roll) * flicker * cloud_night;
            // A small bright core with a soft halo at the strike point, seen
            // through the cloud top; local thickness modulates brightness so
            // clear gaps stay dark, but the shape is the bloom, not the cloud.
            vec2 bloom_center = vec2(
                hash21(strike_cell + 11.3),
                hash21(strike_cell + 47.9));
            float d = length((suv - bloom_center) * vec2(1.6, 1.0));
            float bloom_arg = d * 7.0;
            float core_arg = d * 19.0;
            float bloom = exp(-(bloom_arg * bloom_arg));
            float core = exp(-(core_arg * core_arg));
            float scatter = smoothstep(0.20, 0.60, cloud_density);
            cloud_colour += vec3(0.55, 0.68, 1.0) * strike * bloom * scatter * 3.0
                + vec3(0.92, 0.96, 1.0) * strike * core * scatter * 4.0;
            // City lights glow upward into low night cloud.
            float city_signal_cloud = city_signal_at(cloud_uv, vec2(0.0));
            cloud_colour += vec3(1.0, 0.52, 0.22)
                * pow(city_signal_cloud, 1.4) * cloud_night * 0.10;
            colour = mix(colour, cloud_colour, cloud_opacity);
        }
        colour = composite_front_atmosphere(
            colour,
            camera,
            ray,
            sun,
            atmosphere_distance,
            earth_distance,
            geometric_sunlight,
            sun_visible);
        alpha = 1.0;
    } else {
        vec3 tangent_point = camera - ray * dot(camera, ray);
        vec3 atmosphere_normal = normalize(tangent_point);
        float altitude_km = max((length(tangent_point) / surface_radius - 1.0) * 6378.137, 0.0);
        // Smooth the analytical sphere silhouette over the current pixel
        // footprint. Without this coverage term the discard boundary becomes
        // visibly stair-stepped on the narrow outputs.
        float silhouette = smoothstep(
            0.0,
            atmosphere_coverage_width,
            atmosphere_discriminant);
        float sun_height = dot(atmosphere_normal, sun);
        float daylight = smoothstep(civil_twilight_sine, 0.0, sun_height);
        float column_km = sqrt(6.2831853 * 6378.137 * 8.5) * exp(-altitude_km / 8.5);
        vec3 extinction = vec3(1.0) - exp(-rayleigh_per_km * column_km);
        alpha = max(max(extinction.r, extinction.g), extinction.b) * silhouette;
        // The Vulkan pipeline uses straight-alpha blending. Keep atmosphere
        // RGB unassociated so coverage is applied exactly once by the blend.
        // Night limb early-out: colour carries a daylight factor, so it is
        // exactly zero here; alpha (coverage) is unaffected and still flows
        // to the blend below.
        vec3 limb_colour = vec3(0.0);
        if (daylight > 0.0) {
            float phase = 0.1875 * (1.0 + dot(ray, sun) * dot(ray, sun));
            vec3 sun_path = sun_path_factor(rayleigh_per_km * column_km,
                rayleigh_per_km * density_column_km(max(sun_height, 0.0)));
            limb_colour = extinction * phase * daylight * sun_path * 1.6 / max(alpha, 1.0e-5);
        }
        colour = limb_colour;
    }
    vec3 aurora = aurora_emission(camera, ray, sun, earth_distance) + night_airglow(camera, ray, sun, earth_distance);
    vec3 premultiplied = colour * alpha + aurora;
    alpha = max(alpha, clamp(max(aurora.r, max(aurora.g, aurora.b)), 0.0, 1.0));
    out_color = vec4(filmic(premultiplied * exposure / max(alpha, 1.0e-5)), alpha);
}
