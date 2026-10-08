#version 460

// One quad (two triangles, six vertices) per Hipparcos star. Positions are
// computed for the current UTC every frame: proper motion from the catalogue
// epoch, then J2000 -> view (precession, nutation, Earth rotation and camera,
// folded into one basis on the CPU), then annual aberration.
// Catalogue layout: see src/star_catalog.rs.

layout(set = 0, binding = 16) uniform sampler2D star_catalog;

layout(push_constant) uniform StarFrame {
    // Camera basis in J2000 coordinates; .w lanes carry Earth's velocity / c.
    vec4 view_right;
    vec4 view_up;
    vec4 view_forward;
    vec4 projection_tangents; // tan half-FOV x, y; focus x, y (canvas units)
    vec4 canvas_rect;
    vec4 viewport_rect;
    // x: Julian years since the catalogue epoch; y: physical pixels per canvas
    // unit; z: sky daylight gate (0 in daylight, 1 at night, see vulkan.rs).
    vec4 params;
    // Camera position in J2000 (scene units, Earth radius 0.78) for the limb
    // extinction of each star's line of sight.
    vec4 observer;
} frame;

layout(location = 0) noperspective out vec2 out_offset_px;
layout(location = 1) flat out vec4 out_colour_energy;
layout(location = 2) flat out float out_sigma_px;

const int stars_per_row = 256;
const float mas_to_radians = 4.84813681e-9;

const vec2 CORNERS[6] = vec2[](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));

void cull() {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    out_offset_px = vec2(0.0);
    out_colour_energy = vec4(0.0);
    out_sigma_px = 1.0;
}

// Sea-level Rayleigh optical depths for R, G, B (lambda^-4 from 0.097 at 550 nm).
// A line of sight that grazes the air at tangent altitude h crosses the one-sided
// grazing column sqrt(pi R / 2H) exp(-h / H) (Chapman), so a star at the limb is
// dimmed and reddened; clear sky from about 100 km up.
vec3 limb_transmission(vec3 observer, vec3 direction) {
    if (dot(observer, direction) >= 0.0) return vec3(1.0);
    float impact = length(cross(observer, direction));
    if (impact <= 0.78) return vec3(1.0);
    float altitude_km = (impact - 0.78) / 0.78 * 6371.0;
    float column = 34.3 * exp(-altitude_km / 8.5);
    return exp(-vec3(0.050, 0.097, 0.216) * column);
}

void main() {
    int star = gl_VertexIndex / 6;
    vec2 corner = CORNERS[gl_VertexIndex % 6];
    // Daylight: the camera shoots the sky only at a night exposure (gate 0).
    if (frame.params.z <= 0.0) { cull(); return; }
    ivec2 base = ivec2((star % stars_per_row) * 4, star / stars_per_row);
    vec4 t0 = floor(texelFetch(star_catalog, base, 0) * 255.0 + 0.5);
    vec4 t1 = floor(texelFetch(star_catalog, base + ivec2(1, 0), 0) * 255.0 + 0.5);
    vec4 t2 = floor(texelFetch(star_catalog, base + ivec2(2, 0), 0) * 255.0 + 0.5);
    if (t1.b < 0.5) { cull(); return; }

    float ra = (t0.r * 65536.0 + t0.g * 256.0 + t0.b) / 16777216.0 * 6.28318531;
    float dec = (t0.a * 65536.0 + t1.r * 256.0 + t1.g) / 16777215.0 * 3.14159265 - 1.57079633;
    float vmag = -2.0 + (t1.b - 1.0) / 254.0 * 10.0;
    float bv = -0.5 + t1.a / 255.0 * 3.0;
    vec2 pm = (vec2(t2.r * 256.0 + t2.g, t2.b * 256.0 + t2.a) - 32768.0) * 0.25;

    float cd = cos(dec), sd = sin(dec), cr = cos(ra), sr = sin(ra);
    vec3 direction = vec3(cd * cr, cd * sr, sd);
    // Proper motion along the local east and north unit vectors.
    vec3 east = vec3(-sr, cr, 0.0);
    vec3 north = vec3(-sd * cr, -sd * sr, cd);
    direction += (east * pm.x + north * pm.y) * (mas_to_radians * frame.params.x);
    // Annual aberration (first order, <= 20.5 arcsec).
    direction = normalize(direction + vec3(frame.view_right.w, frame.view_up.w, frame.view_forward.w));

    vec3 view = vec3(dot(direction, frame.view_right.xyz),
        dot(direction, frame.view_up.xyz),
        dot(direction, frame.view_forward.xyz));
    if (view.z <= 1.0e-4) { cull(); return; }
    vec2 ndc = view.xy / view.z / frame.projection_tangents.xy;
    if (any(greaterThan(abs(ndc), vec2(4.0)))) { cull(); return; }

    // Brightness steps 1.8x per magnitude on a fixed sky scale; the camera
    // exposure does not enter it, the daylight gate does (see params.z).
    float energy = frame.params.z * 0.35 * exp(0.587787 * (6.5 - vmag));
    vec3 transmission = limb_transmission(frame.observer.xyz, direction);
    float sigma = 0.70 * pow(max(energy, 1.0), 0.2);
    float radius_px = max(4.0 * sigma, 1.5);

    vec2 global_xy = vec2(
        frame.projection_tangents.z + ndc.x * frame.canvas_rect.z * 0.5,
        frame.projection_tangents.w - ndc.y * frame.canvas_rect.w * 0.5);
    global_xy += corner * radius_px / frame.params.y;
    vec2 local = (global_xy - frame.viewport_rect.xy) / frame.viewport_rect.zw;
    gl_Position = vec4(local * 2.0 - 1.0, 0.0, 1.0);
    out_offset_px = corner * radius_px;

    // B-V to colour: blackbody chromaticities (Mitchell Charity's table, D65
    // white) sampled at the Ballesteros temperature, pulled 15% toward white.
    vec3 c = bv < 0.0 ? mix(vec3(0.62, 0.71, 1.00), vec3(0.79, 0.85, 1.00), (bv + 0.35) / 0.35)
        : bv < 0.4 ? mix(vec3(0.79, 0.85, 1.00), vec3(0.97, 0.96, 1.00), bv / 0.4)
        : bv < 0.8 ? mix(vec3(0.97, 0.96, 1.00), vec3(1.00, 0.92, 0.80), (bv - 0.4) / 0.4)
        : bv < 1.4 ? mix(vec3(1.00, 0.92, 0.80), vec3(1.00, 0.78, 0.55), (bv - 0.8) / 0.6)
        : mix(vec3(1.00, 0.78, 0.55), vec3(1.00, 0.66, 0.40), clamp((bv - 1.4) / 0.6, 0.0, 1.0));
    // Unit luminance, so the V magnitude sets the brightness.
    c /= dot(c, vec3(0.2126, 0.7152, 0.0722));
    out_colour_energy = vec4(mix(vec3(1.0), c, 0.85) * transmission, energy);
    out_sigma_px = sigma;
}
