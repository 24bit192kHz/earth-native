#version 460

layout(location = 0) noperspective in vec2 in_uv;
// Dual-source output: premultiplied radiance and the transmittance of what
// lies behind; the camera stage (post.frag) applies the tone curve.
layout(location = 0, index = 0) out vec4 out_color;
layout(location = 0, index = 1) out vec4 out_transmittance;

// These values are supplied independently for every output. The shared camera
// uses the complete logical desktop, so adjacent outputs sample the same frustum.
// The planet albedos (equirectangular 2:1) plus Saturn's ring alpha
// (radial strip), bound in the same descriptor set as the star panorama. Only
// the selected body's sampler is read, so Earth never samples them.
// Binding 11 holds the resident body's albedo (only one body is ever shown);
// 12..=14 are legacy fallback slots kept for an unchanged descriptor layout.
layout(set = 0, binding = 11) uniform sampler2D planet_texture;
layout(set = 0, binding = 15) uniform sampler2D saturn_ring_texture;

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
    // procedural path does not read these; they keep the push-constant block
    // identical across every pipeline sharing this layout.
    vec4 celestial_sun_view;
    vec4 celestial_moon_view;
    vec4 material_state;
} frame;

// Calibrated to the main 5.5x capture. The actor bounds that define the
// camera scale include a wider shell than the visible Earth surface.
const float surface_radius = 0.78;
// Global mean cloud-top height (ISCCP ~500-600 hPa). At 12 km every cloud's
// shadow landed twice as far out, reading as a doubled cloud field.
const float cloud_radius = surface_radius * (1.0 + 5.5 / 6378.137);
const float atmosphere_radius = surface_radius * (1.0 + 100.0 / 6378.137);
const float civil_twilight_sine = -0.10452846326765347;

float sphere_intersection(vec3 origin, vec3 direction, float radius) {
    float b = dot(origin, direction);
    float c = dot(origin, origin) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant < 0.0) {
        return -1.0;
    }
    float root = sqrt(discriminant);
    float near_distance = -b - root;
    return near_distance > 0.0 ? near_distance : -b + root;
}


float hash21(vec2 value) {
    value = fract(value * vec2(123.34, 456.21));
    value += dot(value, value + 45.32);
    return fract(value.x * value.y);
}

// The planet albedo sampled through the same lighting treatment as
// the Earth day face: a day/night terminator from the shared sun direction,
// subtle limb darkening, and the star field already composited behind the
// planet. Sampling is equirectangular (2:1) from the sphere normal, with
// latitude from world +Z (camera up) so the pole points to screen top.
vec3 world_to_body(vec3 vector) {
    return vec3(dot(frame.moon_body_x.xyz, vector),
        dot(frame.moon_body_y.xyz, vector), dot(frame.moon_body_z.xyz, vector));
}

vec3 body_to_world(vec3 vector) {
    return frame.moon_body_x.xyz * vector.x
        + frame.moon_body_y.xyz * vector.y + frame.moon_body_z.xyz * vector.z;
}

// Polar / equatorial radius (NASA/JPL planetary physical parameters).
// 1 Jupiter, 2 Mercury, 3 Mars, 4 Saturn, 5 Venus, 6 Uranus, 7 Neptune, 8 Moon.
float polar_ratio(int body) {
    if (body == 1) return 66854.0 / 71492.0;
    if (body == 3) return 3376.2 / 3396.19;
    if (body == 4) return 54364.0 / 60268.0;
    if (body == 6) return 24973.0 / 25559.0;
    if (body == 7) return 24341.0 / 24764.0;
    return 1.0;
}

// Same SDR presentation as the Earth pass (earth_textured.frag).
const float exposure = 0.82;
vec3 filmic(vec3 x) {
    x = max(x, vec3(0.0));
    x = x * x / (x + 0.011);
    const float knee = 0.72;
    vec3 shoulder = knee + (1.0 - knee) * (1.0 - exp(-(x - knee) / (1.0 - knee)));
    return mix(x, shoulder, step(knee, x));
}

// Disk-resolved photometry, in albedo-times-cosine units like Lambert.
// Airless regoliths (Moon, Mercury) barely limb-darken: Lunar-Lambert with
// McEwen's (1996) phase function L(alpha). Bodies with atmospheres follow
// Minnaert's law mu0^k mu^(k-1): Mars k~0.75 (dust haze), Venus' cloud deck
// and the giant planets k~0.8-0.9, which gives their visible limb darkening.
float photometry(int body, float mu0, float mu, float phase_degrees) {
    mu0 = max(mu0, 0.0);
    mu = max(mu, 0.05);
    if (body == 2 || body == 8) {
        float a = phase_degrees;
        float l = clamp(1.0 - 0.019 * a + 2.42e-4 * a * a - 1.46e-6 * a * a * a, 0.0, 1.0);
        return l * 2.0 * mu0 / (mu0 + mu) + (1.0 - l) * mu0;
    }
    float k = body == 3 ? 0.75 : (body == 5 ? 0.80 : (body == 6 || body == 7 ? 0.85 : 0.90));
    return pow(mu0, k) * pow(mu, k - 1.0);
}

float planet_intersection(vec3 camera, vec3 ray, float ratio) {
    vec3 origin = world_to_body(camera) / vec3(1.0, 1.0, ratio);
    vec3 direction = world_to_body(ray) / vec3(1.0, 1.0, ratio);
    float a = dot(direction, direction);
    float b = dot(origin, direction);
    float c = dot(origin, origin) - surface_radius * surface_radius;
    float discriminant = b * b - a * c;
    if (discriminant < 0.0) return -1.0;
    float near_hit = (-b - sqrt(discriminant)) / a;
    return near_hit > 0.0 ? near_hit : (-b + sqrt(discriminant)) / a;
}

vec3 planet_textured(vec3 normal, vec2 uv, vec2 planet_dx, vec2 planet_dy, vec3 sun, vec3 ray, int body) {
    // uv + wrap-folded gradients arrive precomputed in uniform control flow
    // at the call site: dFdx/dFdy inside the disc branch below would be
    // undefined at silhouette quads (non-uniform participation). textureGrad
    // needs no derivatives itself, so fetching here stays well-defined.
    vec3 albedo = textureGrad(planet_texture, uv, planet_dx, planet_dy).rgb;

    // Solar response with a finite angular solar disk at the terminator.
    float sunlight = dot(normal, sun);
    float terminator_width = frame.celestial_sun_view.w;
    float day = smoothstep(-terminator_width, terminator_width, sunlight);
    float phase_degrees = degrees(acos(clamp(dot(sun, -ray), -1.0, 1.0)));
    float lit = photometry(body, sunlight, dot(normal, -ray), phase_degrees) * day;
    // Exposure adapts to the selected body's solar flux (as a camera would),
    // preserving inspectable albedo; it is not common radiometry across bodies.
    return albedo * lit;
}

// The Saturn ring alpha/colour, sampled as a flat annulus in the planet's
// equatorial plane (z = 0 in world space, pole at +Z). Returns RGBA; the
// alpha is zero when the ray misses the ring or the ring is behind the planet.
vec4 saturn_ring_colour(vec3 ray, vec3 camera, float planet_distance) {
    ray = world_to_body(ray);
    camera = world_to_body(camera);
    const float ring_inner = surface_radius * (74658.0 / 60268.0);
    const float ring_outer = surface_radius * (136775.0 / 60268.0);
    // Saturn's main rings are only tens of metres thick. At orbital display
    // scales, a plane is both more accurate and cheaper than the former slab.
    // Sampling coordinate + gradients in uniform control flow: the range gates
    // below diverge at the ring silhouette, where implicit LOD (and fresh
    // derivatives) would be undefined. u is radial (no wraparound), so no fold.
    float hit = abs(ray.z) < 1.0e-7 ? -1.0 : -camera.z / ray.z;
    vec3 ring_point = camera + ray * hit;
    float ring_radius = length(ring_point.xy);
    vec2 ring_uv = vec2((ring_radius - ring_inner) / (ring_outer - ring_inner), 0.5);
    vec2 ring_dx = dFdx(ring_uv);
    vec2 ring_dy = dFdy(ring_uv);
    if (abs(ray.z) < 1.0e-7) return vec4(0.0);
    if (hit <= 0.0 || (planet_distance > 0.0 && hit > planet_distance)) return vec4(0.0);
    vec3 point = ring_point;
    float radius = ring_radius;
    if (radius < ring_inner || radius > ring_outer) return vec4(0.0);
    vec4 ring = textureGrad(saturn_ring_texture, ring_uv, ring_dx, ring_dy);
    vec3 sun = normalize(world_to_body(frame.sun_direction.xyz));
    vec3 scaled_point = point / vec3(1.0, 1.0, 54364.0 / 60268.0);
    vec3 scaled_sun = normalize(sun / vec3(1.0, 1.0, 54364.0 / 60268.0));
    float projection = dot(scaled_point, scaled_sun);
    float miss = length(scaled_point - scaled_sun * projection) - surface_radius;
    float penumbra = max(length(point) * frame.celestial_sun_view.w, 1.0e-5);
    float visibility = projection < 0.0 ? smoothstep(-penumbra, penumbra, miss) : 1.0;
    float lighting = abs(sun.z) * visibility + 0.000002;
    float opacity = 1.0 - pow(max(1.0 - ring.a, 0.0), 1.0 / max(abs(ray.z), 0.02));
    float edge = smoothstep(ring_inner, ring_inner + 0.001, radius)
        * (1.0 - smoothstep(ring_outer - 0.001, ring_outer, radius));
    return vec4(ring.rgb * lighting, opacity * edge);
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
    vec3 sun = normalize(frame.sun_direction.xyz);
    // The atmosphere discard must NOT gate the planet/ring path: Saturn's ring
    // extends well beyond the atmosphere sphere, so a ray to the ring would be
    // discarded here before the ring is ever drawn. It is enforced only on the
    // Earth material below.
    float atmosphere_distance = sphere_intersection(camera, ray, atmosphere_radius);
    float earth_distance = sphere_intersection(camera, ray, surface_radius);

    // Switchable body: `sun_direction.w` carries the integer body id
    // (0 Earth, 1 Jupiter, 2 Mercury, 3 Mars, 4 Saturn, 5 Venus, 6 Uranus,
    // 7 Neptune, 8 Moon).
    int body = int(frame.sun_direction.w + 0.5);
    if (body >= 1) {
        float ratio = polar_ratio(body);
        float body_distance = planet_intersection(camera, ray, ratio);
        vec3 colour = vec3(0.0);
        float alpha = 0.0;
        // Equirect UV + wrap-folded gradients in uniform control flow (this
        // `body >= 1` branch is push-constant-uniform). body_distance is
        // garbage off-disc, but derivatives only need uniform participation,
        // not valid values; the hit branch below diverges at the limb.
        vec3 raw_normal = normalize(world_to_body(camera + ray * body_distance));
        float raw_lon = atan(raw_normal.y, raw_normal.x);
        vec2 planet_uv = vec2(0.5 + raw_lon / 6.28318530718,
            0.5 - asin(clamp(raw_normal.z, -1.0, 1.0)) / 3.14159265359);
        vec2 planet_dx = dFdx(planet_uv);
        vec2 planet_dy = dFdy(planet_uv);
        planet_dx.x -= floor(planet_dx.x + 0.5);
        planet_dy.x -= floor(planet_dy.x + 0.5);
        if (body_distance > 0.0) {
            vec3 body_point = world_to_body(camera + ray * body_distance);
            vec3 body_normal = normalize(body_to_world(body_point / vec3(1.0, 1.0, ratio * ratio)));
            colour = planet_textured(body_normal, planet_uv, planet_dx, planet_dy, sun, ray, body);
            alpha = 1.0;
        }
        // Saturn's ring composites over the planet (and shows alone where the
        // disk is absent), with correct front/back occlusion against the globe.
        if (body == 4) {
            vec4 ring = saturn_ring_colour(ray, camera, body_distance);
            if (ring.a > 0.0) {
                if (alpha > 0.0) {
                    colour = mix(colour, ring.rgb, ring.a);
                } else {
                    colour = ring.rgb;
                    alpha = ring.a;
                }
            }
        }
        if (alpha <= 0.0) {
            discard;
        }
        out_color = vec4(colour * exposure * alpha, alpha);
        out_transmittance = vec4(vec3(1.0 - alpha), 1.0);
        return;
    }

    // sunlight + fwidth in uniform control flow: the atmosphere discard and
    // the earth_distance branch both diverge at silhouettes, where fresh
    // derivatives would be undefined. Off-disc values are unused except as
    // derivative donors.
    vec3 earth_point = camera + ray * earth_distance;
    vec3 earth_normal = normalize(earth_point);
    float sunlight = dot(earth_normal, sun);
    float horizon_width = max(fwidth(sunlight), 1.0e-4);
    // Earth-only: a ray that misses the atmosphere is outside the Earth disc.
    if (atmosphere_distance < 0.0) {
        discard;
    }
    vec3 colour = vec3(0.0);
    float alpha = 0.0;
    if (earth_distance > 0.0) {
        vec3 point = earth_point;
        vec3 normal = earth_normal;
        float daylight = max(sunlight, 0.0);
        float direct_day = smoothstep(-horizon_width, horizon_width, sunlight);
        float night_visibility = 1.0 - smoothstep(civil_twilight_sine, 0.0, sunlight);
        vec3 day = mix(vec3(0.035, 0.12, 0.20), vec3(0.20, 0.34, 0.15), normal.z * 0.5 + 0.5);
        vec3 night = vec3(0.003, 0.007, 0.015);
        night *= 0.70;
        float city_pattern = step(0.997, hash21(floor(normal.xy * 400.0)));
        night += city_pattern * vec3(1.0, 0.48, 0.12) * night_visibility;
        vec3 half_vector = normalize(sun - ray);
        float fresnel = pow(1.0 - max(dot(normal, -ray), 0.0), 5.0);
        float water_glint = pow(max(dot(normal, half_vector), 0.0), 220.0) * fresnel;
        colour = day * (0.18 + 0.82 * daylight) * direct_day + night * night_visibility;
        colour += water_glint * direct_day * vec3(0.55, 0.78, 1.0);
        // Same 7% albedo floor as the other bodies.
        float planet_day = smoothstep(-0.23, 0.23, sunlight);
        colour += day * 0.07 * (1.0 - planet_day);

        // A cheap analytical cloud shell is retained until density and normal
        // tiles are available. It stays spatially stable at idle frame rates.
        float cloud_distance = sphere_intersection(camera, ray, cloud_radius);
        if (cloud_distance > 0.0 && cloud_distance < earth_distance) {
            vec3 cloud_normal = normalize(camera + ray * cloud_distance);
            float cloud_noise = hash21(floor(cloud_normal.xy * 180.0));
            float cloud_alpha = smoothstep(0.82, 0.98, cloud_noise) * 0.45;
            float cloud_daylight = smoothstep(civil_twilight_sine, 0.0, dot(cloud_normal, sun));
            vec3 cloud_colour = vec3(0.72, 0.78, 0.84)
                * (0.25 + 0.75 * max(dot(cloud_normal, sun), 0.0))
                * mix(0.05, 1.0, cloud_daylight);
            colour = mix(colour, cloud_colour, cloud_alpha * mix(0.16, 1.0, cloud_daylight));
        }
        alpha = 1.0;
    } else {
        vec3 atmosphere_point = camera + ray * atmosphere_distance;
        vec3 atmosphere_normal = normalize(atmosphere_point);
        float edge = clamp(1.0 - dot(atmosphere_normal, -ray), 0.0, 1.0);
        float daylight = smoothstep(civil_twilight_sine, 0.0, dot(atmosphere_normal, sun));
        alpha = pow(edge, 2.5) * (0.02 + 0.40 * daylight);
        colour = vec3(0.08, 0.28, 0.72) * alpha;
    }

    out_color = vec4(colour * 0.75 * alpha * frame.camera_position_distance.w, alpha);
    out_transmittance = vec4(vec3(1.0 - alpha), 1.0);
}
