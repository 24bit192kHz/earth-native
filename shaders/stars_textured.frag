#version 460

layout(location = 0) noperspective in vec2 in_uv;
layout(location = 0) out vec4 out_color;

// The raw BGRA8 payload is copied directly into a B8G8R8A8_SRGB image, so this
// sampler returns linear scene values. The procedural shader remains separate
// when the optional production panorama is absent.
layout(set = 0, binding = 0) uniform sampler2D star_panorama;
layout(set = 0, binding = 10) uniform sampler2D moon_albedo;

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
    // Per-frame precomputed celestial geometry (see ShaderFrame): camera-space
    // body directions plus apparent radii. moon_body_x.w / moon_body_y.w carry
    // the sidereal sine/cosine pair and moon_body_z.w the sun-gate threshold.
    vec4 celestial_sun_view;
    vec4 celestial_moon_view;
    vec4 material_state;
} frame;

const float inv_two_pi = 0.15915494309189535;
const float inv_pi = 0.3183098861837907;
const float scene_earth_radius = 0.78;

vec2 spherical_uv(vec3 direction) {
    return vec2(
        fract(0.5 - atan(direction.y, direction.x) * inv_two_pi),
        clamp(0.5 - asin(clamp(direction.z, -1.0, 1.0)) * inv_pi, 0.0, 1.0));
}

vec3 asset_to_equator_of_date(vec3 direction) {
    // Asset longitude zero is -X. Undo that asset rotation, then undo the
    // Earth-fixed sidereal rotation so the panorama remains inertial. The
    // sidereal sine/cosine pair is precomputed once per frame on the CPU.
    vec3 ecef = frame.sun_direction.w == 0.0
        ? vec3(-direction.x, direction.y, direction.z)
        : vec3(-direction.x, -direction.y, direction.z);
    float sine = frame.moon_body_x.w;
    float cosine = frame.moon_body_y.w;
    return vec3(
        cosine * ecef.x - sine * ecef.y,
        sine * ecef.x + cosine * ecef.y,
        ecef.z);
}

vec3 stable_perpendicular(vec3 direction) {
    vec3 reference = abs(direction.z) < 0.95 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    return normalize(cross(reference, direction));
}

float celestial_disc(float cosine_angle, float angular_radius, float edge) {
    return smoothstep(
        cos(angular_radius + edge),
        cos(max(angular_radius - edge, 0.0)),
        cosine_angle);
}

void main() {
    vec2 output_uv = in_uv;
    vec2 global_xy = frame.viewport_rect.xy + output_uv * frame.viewport_rect.zw;
    vec2 canonical_ndc = vec2(
        2.0 * (global_xy.x - frame.projection_tangents.z) / frame.canvas_rect.z,
        2.0 * (frame.projection_tangents.w - global_xy.y) / frame.canvas_rect.w
    );
    vec3 ray = normalize(
        frame.camera_forward.xyz
        + frame.camera_right.xyz * canonical_ndc.x * frame.projection_tangents.x
        + frame.camera_up.xyz * canonical_ndc.y * frame.projection_tangents.y);
    // Return an opaque value so DONT_CARE remains valid. The planet will
    // overwrite this interior; the small inset preserves every silhouette pixel.
    vec3 camera = frame.camera_position_distance.xyz;
    float closest = dot(camera, ray);
    // An inscribed sphere is conservative for the oblate gas giants.
    float opaque_radius = frame.sun_direction.w == 1.0 ? 0.78 * (66854.0 / 71492.0)
        : frame.sun_direction.w == 3.0 ? 0.78 * (3376.2 / 3396.19)
        : frame.sun_direction.w == 4.0 ? 0.78 * (54364.0 / 60268.0)
        : frame.sun_direction.w == 6.0 ? 0.78 * (24973.0 / 25559.0)
        : frame.sun_direction.w == 7.0 ? 0.78 * (24341.0 / 24764.0) : 0.78;
    opaque_radius -= 0.001;
    float hidden = closest * closest - dot(camera, camera) + opaque_radius * opaque_radius;
    // Panorama UV: U=0.5-atan(Y,X)/(2*pi), V=0.5-asin(Z)/pi.
    // The NASA SVS map is in J2000 equatorial coordinates with RA 0h at the
    // centre and RA increasing leftward, so no extra yaw is applied.
    // REPEAT on U keeps the antimeridian value-continuous (but not its screen
    // derivative — folded explicit gradients below keep mip selection continuous);
    // CLAMP_TO_EDGE preserves the polar rows exactly as the raw top-left-origin export stored them.
    const float star_dome_yaw_turns = 0.0;
    vec3 inertial_ray = asset_to_equator_of_date(ray);
    float longitude = atan(inertial_ray.y, inertial_ray.x);
    float latitude = asin(clamp(inertial_ray.z, -1.0, 1.0));
    vec2 panorama_uv = vec2(
        fract(0.5 - longitude * inv_two_pi + star_dome_yaw_turns),
        clamp(0.5 - latitude * inv_pi, 0.0, 1.0));
    vec2 panorama_dx = dFdx(panorama_uv);
    vec2 panorama_dy = dFdy(panorama_uv);
    panorama_dx.x -= floor(panorama_dx.x + 0.5);
    panorama_dy.x -= floor(panorama_dy.x + 0.5);
    // panorama_uv + gradients are computed above this occluded-pixel branch on
    // purpose: it diverges at the planet silhouette, where fresh derivatives
    // would be undefined. textureGrad itself stays valid inside the branch.
    // Same precompute for the Moon: camera-relative direction and apparent
    // radius. Scene-space positions are only needed by the eclipse math and
    // are derived inside the tiny moon-disc branch below. The Moon only
    // belongs to Earth; every other body hides it (`sun_direction.w` carries
    // the body id: 0 = Earth).
    int body = int(frame.sun_direction.w + 0.5);
    vec3 moon_direction = frame.celestial_moon_view.xyz;
    float moon_reference_radius = frame.celestial_moon_view.w;
    float moon_cosine = dot(ray, moon_direction);
    float moon_alpha = smoothstep(frame.celestial_state.x, frame.material_state.w, moon_cosine);
    // Disc geometry + wrap-folded gradients in uniform control flow. This sits
    // above both the planet-occlusion early-out and the fetch branch below:
    // both diverge (silhouette quads, varying moon_alpha), where fresh
    // derivatives would be undefined. textureGrad itself stays valid.
    float sin_theta = sqrt(max(1.0 - moon_cosine * moon_cosine, 0.0));
    float radial = clamp(sin_theta / sin(moon_reference_radius), 0.0, 1.0);
    vec3 tangent = ray - moon_direction * moon_cosine;
    float tangent_length = length(tangent);
    vec3 tangent_direction = tangent_length > 1.0e-6
        ? tangent / tangent_length
        : stable_perpendicular(moon_direction);
    vec3 surface_normal = normalize(
        -moon_direction * sqrt(max(1.0 - radial * radial, 0.0))
        + tangent_direction * radial);
    vec3 body_normal = normalize(vec3(
        dot(frame.moon_body_x.xyz, surface_normal),
        dot(frame.moon_body_y.xyz, surface_normal),
        dot(frame.moon_body_z.xyz, surface_normal)));
    vec2 moon_uv = spherical_uv(body_normal);
    // Same antimeridian treatment as the planet branch: fract() wraps the
    // value but not its screen derivative across the cut.
    vec2 moon_dx = dFdx(moon_uv);
    vec2 moon_dy = dFdy(moon_uv);
    moon_dx.x -= floor(moon_dx.x + 0.5);
    moon_dy.x -= floor(moon_dy.x + 0.5);
    if (closest < 0.0 && hidden > 0.0) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec3 source_colour = textureGrad(star_panorama, panorama_uv, panorama_dx, panorama_dy).rgb;
    // The source panorama is sampled into linear scene values. Match the
    // reference's display-space compression so its low-level scan texture
    // remains black while the catalogue stars stay visible at native scale.
    vec3 colour = pow(max(source_colour, vec3(0.0)), vec3(1.7)) * 0.75;

    // Camera-relative Sun direction and apparent radius arrive precomputed
    // once per frame (the old per-fragment normalize/length/asin chain).
    vec3 sun = frame.celestial_sun_view.xyz;
    float sun_angular_radius = frame.celestial_sun_view.w;
    float sun_cosine = dot(ray, sun);
    float sun_angle = acos(clamp(sun_cosine, -1.0, 1.0));
    // The two widest layers stay unconditional (they shape the glow when the
    // Sun sits just off-frame), but they are kept tight and faint: wide gold
    // washes read as brown fog on black space, and the gated inner layers
    // already carry the near-disc energy.
    float sun_outer_corona = exp(-max(sun_angle - sun_angular_radius, 0.0) / (sun_angular_radius * 8.0));
    float sun_aureole = exp(-max(sun_angle, 0.0) / (sun_angular_radius * 12.0));
    // Wide halo kept faint and neutral-warm so the Sun in-frame does not paint
    // a muddy vignette across the night sky; deep sky returns to black.
    colour += sun_aureole * vec3(0.0032, 0.0028, 0.0024);
    colour += sun_outer_corona * vec3(0.022, 0.016, 0.010);
    // The remaining layers decay to under 1e-3 of full by 40 apparent radii
    // (shoulder e^-10.2, inner e^-6.6, core 0), so past the precomputed gate
    // threshold they contribute less than a quantization step. The test is
    // frame-uniform except at the screen-edge crossing, so the branch is
    // coherent and free when the Sun is out of view.
    if (sun_cosine > frame.moon_body_z.w) {
        // The photosphere uses the physical apparent radius. The surrounding
        // corona remains an intentionally display-scale bloom, since the real
        // corona is not visible at ordinary wallpaper exposure.
        float disc_edge = max(sun_angular_radius - sun_angle, 0.0) / max(sun_angular_radius, 1.0e-4);
        float limb_darkening = mix(0.55, 1.0, pow(disc_edge, 0.65));
        float sun_core = celestial_disc(sun_cosine, sun_angular_radius, sun_angular_radius * 0.05) * limb_darkening;
        float sun_shoulder = exp(-max(sun_angle - sun_angular_radius * 0.35, 0.0) / (sun_angular_radius * 3.9));
        float sun_inner_corona = exp(-max(sun_angle - sun_angular_radius * 0.7, 0.0) / (sun_angular_radius * 6.0));
        colour += sun_inner_corona * vec3(0.55, 0.30, 0.08);
        colour += sun_shoulder * vec3(1.05, 0.78, 0.28);
        colour += sun_core * vec3(1.55, 1.45, 1.28);
    }

    if (body == 0 && moon_alpha > 0.0) {
        vec3 albedo = textureGrad(moon_albedo, moon_uv, moon_dx, moon_dy).rgb;
        // Scene-space body positions for the eclipse geometry, reconstructed
        // from the Earth-fixed directions only inside this small disc branch.
        vec3 sun_from_earth = normalize(frame.sun_direction.xyz);
        vec3 sun_position = sun_from_earth
            * (max(frame.celestial_distances.x, 1.0) * scene_earth_radius);
        vec3 moon_from_earth = normalize(vec3(
            frame.camera_forward.w,
            frame.camera_right.w,
            frame.camera_up.w));
        vec3 moon_position = moon_from_earth
            * (max(frame.celestial_distances.y, 1.0) * scene_earth_radius);
        vec3 sun_from_moon = normalize(sun_position - moon_position);
        float lighting = max(dot(surface_normal, sun_from_moon), 0.0);
        // Opposition surge: lunar regolith backscatters, so the full Moon is
        // ~40% brighter near zero phase than a Lambert surface predicts
        // (Hapke-style coherent backscatter; phase angle in celestial_state.z).
        float phase_surge = 1.0 + 0.4 * exp(-abs(frame.celestial_state.z) / 0.12);
        vec3 earth_from_moon = normalize(-moon_position);
        float shadow_separation = acos(clamp(dot(earth_from_moon, sun_from_moon), -1.0, 1.0));
        float earth_angular_radius = asin(clamp(
            scene_earth_radius / max(length(moon_position), scene_earth_radius),
            0.0,
            0.999999));
        float sun_from_moon_radius = asin(clamp(
            frame.celestial_distances.z * scene_earth_radius
                / max(length(sun_position - moon_position), 1.0e-5),
            0.0,
            0.999999));
        float eclipse_visibility = smoothstep(
            max(earth_angular_radius - sun_from_moon_radius, 0.0),
            earth_angular_radius + sun_from_moon_radius,
            shadow_separation);
        // Earthshine: sunlight reflected by the whole sunlit Earth falls on
        // the Moon's Earth-facing hemisphere, bluish and phase-dependent.
        float earth_facing = max(dot(surface_normal, earth_from_moon), 0.0);
        vec3 earthshine = vec3(0.55, 0.62, 0.85)
            * (0.028 * earth_facing * (1.0 - clamp(frame.celestial_state.y, 0.0, 1.0)));
        vec3 moon_colour = albedo
            * (earthshine + lighting * phase_surge * eclipse_visibility);
        colour = mix(colour, moon_colour, moon_alpha);
    }

    out_color = vec4(colour, 1.0);
}
