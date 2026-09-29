#version 460

// Camera stage: lens glare, exposure, sensor response and quantisation. The
// scene passes wrote scene-linear, pre-exposed radiance (white Lambertian
// under a zenith Sun = 1 x pre-exposure) with a full box-filtered mip chain.
//
// The model follows a wide-angle lens on a full-frame sensor behind the ISS
// Cupola window, as in the Earth Observation photo archive:
//  - glare: the lens point-spread function has a long power-law tail
//    (scatter from glass, coatings and the window); near field comes from the
//    mip pyramid, the far field from the bright sources analytically so it
//    stays continuous across monitors;
//  - natural vignetting (cos^4, partly corrected as a camera profile does);
//  - a filmic response, mild saturation and dithered 8-bit output with
//    exposure-dependent sensor noise.

layout(location = 0) noperspective in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D scene;
layout(set = 0, binding = 1) uniform sampler2D bloom_map;

layout(push_constant) uniform PostFrame {
    vec4 tone;        // x contrast, y mode (0 camera, 1 legacy), z seed, w noise
    vec4 projection;  // tan half-fov x/y, optical centre x/y (canvas units)
    vec4 canvas;
    vec4 viewport;
    vec4 sun;         // camera-space direction, w angular radius
    vec4 sun_light;   // visible irradiance (pre-exposed), w in front
    vec4 moon;
    vec4 moon_light;
} post;

const float PI = 3.14159265;

vec3 legacy_filmic(vec3 x) {
    x = max(x, vec3(0.0));
    x = x * x / (x + 0.011);
    const float knee = 0.72;
    vec3 shoulder = knee + (1.0 - knee) * (1.0 - exp(-(x - knee) / (1.0 - knee)));
    return mix(x, shoulder, step(knee, x));
}

// Film-like response: a short toe, linear mid-tones and a long shoulder that
// reaches white only asymptotically (clouds keep texture, lights bloom to
// white instead of clipping to a flat colour).
vec3 camera_curve(vec3 x) {
    // Clamped and written as x * (x / (x + c)) so very bright sources cannot
    // overflow to inf/NaN (which printed coloured specks on the Moon).
    x = clamp(x, vec3(0.0), vec3(1.0e4));
    x = x * (x / (x + 0.004));
    const float knee = 0.55;
    vec3 shoulder = knee + (1.0 - knee) * (1.0 - exp(-(x - knee) / (1.0 - knee)));
    return mix(x, shoulder, step(knee, x));
}

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

// Far-field glare and diffraction of one compact source whose visible
// irradiance (pre-exposed, sunlight units x sr) is `light.rgb`.
vec3 source_glare(vec3 ray, vec2 pixel_ndc, vec4 source, vec4 light, float starburst, float px_per_rad) {
    if (light.w < 0.5) return vec3(0.0);
    vec3 energy = light.rgb;
    if (max(energy.r, max(energy.g, energy.b)) <= 0.0) return vec3(0.0);
    float angle = 2.0 * asin(clamp(0.5 * length(ray - source.xyz), 0.0, 1.0));
    // The lens barrel (and the window frame around it) takes in no light
    // from sources more than ~75 degrees off the optical axis.
    float acceptance = smoothstep(0.26, 0.64, source.z);
    if (acceptance <= 0.0) return vec3(0.0);
    energy *= acceptance;
    // Veiling glare: two normalised power-law lobes (each integrates to its
    // fraction of the source energy). The narrow one is the lens's own
    // scatter around the core, the wide one the window and barrel. Keep in
    // sync with lens_glare_psf in vulkan.rs (the exposure cap uses it).
    float x_near = angle / 0.010;
    float x_far = angle / 0.15;
    float psf = 0.004 * 0.6 / (PI * 0.010 * 0.010) * pow(1.0 + x_near * x_near, -1.6)
        + 0.0015 * 1.0 / (PI * 0.15 * 0.15) * pow(1.0 + x_far * x_far, -2.0);
    vec3 glare = energy * psf;

    // Diffraction starburst: 2 x 9 blades = 18 rays. Along a ray the edge
    // diffraction falls as 1/r^2; ~2 % of the energy goes into the rays.
    if (starburst > 0.0 && source.z > 0.0) {
        vec2 source_ndc = source.xy / source.z / post.projection.xy;
        vec2 delta = (pixel_ndc - source_ndc) * post.projection.xy * px_per_rad;
        float r_px = length(delta);
        float phi = atan(delta.y, delta.x);
        const float rays = 18.0;
        const float rotation = 0.21;
        float nearest = abs(fract((phi - rotation) * rays / (2.0 * PI) + 0.5) - 0.5) * (2.0 * PI / rays);
        float across_px = r_px * sin(nearest);
        float width_px = 1.1 + 0.004 * r_px;
        float spike = exp(-0.5 * across_px * across_px / (width_px * width_px)) / (2.5066 * width_px);
        float core_px = max(source.w * px_per_rad, 2.0);
        // Rays alternate in length (odd/even blade edges), as real ones do.
        float ray_index = floor((phi - rotation) * rays / (2.0 * PI) + 0.5);
        float alternate = mod(ray_index, 2.0) < 0.5 ? 1.0 : 0.55;
        // Chromatic: the diffraction angle scales with wavelength, so ray
        // ends turn warm while the base stays white.
        vec3 tint = mix(vec3(1.0), vec3(1.2, 1.0, 0.78), smoothstep(60.0, 700.0, r_px));
        // Real rays fade faster than the ideal edge: finite aperture edges
        // and the partial coherence of the solar disc.
        float along = 0.006 / max(r_px * r_px, core_px * core_px) * exp(-r_px / 900.0);
        glare += energy * px_per_rad * px_per_rad * tint * spike * along * alternate * starburst;
    }
    return glare;
}

void main() {
    vec2 uv = in_uv;
    vec3 colour = min(textureLod(scene, uv, 0.0).rgb, vec3(65000.0));

    if (post.tone.y > 0.5) {
        // Dithered like the camera path: the smooth limb darkening of a
        // planet's disc banded in 8 bits.
        vec3 legacy = clamp(legacy_filmic(colour), 0.0, 1.0);
        vec3 encoded = mix(12.92 * legacy, 1.055 * pow(legacy, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, legacy));
        vec3 seed = vec3(gl_FragCoord.xy, 0.0);
        encoded = clamp(encoded + (hash13(seed + 3.7) + hash13(seed + 9.1) - 1.0) / 255.0, 0.0, 1.0);
        out_color = vec4(mix(encoded / 12.92, pow((encoded + 0.055) / 1.055, vec3(2.4)), step(0.04045, encoded)), 1.0);
        return;
    }

    // Local adaptation, as for the Moon's disc: above 0.5 the luminance is
    // compressed toward 1.2 with the hue kept. At a night exposure the
    // twilight arc is 2^7 over white: uncompressed it was a featureless
    // white band; now it keeps its orange, white and blue layers beside
    // the city lights, which stay golden instead of clipping. The meter
    // reads the scene before this.
    float scene_luminance = dot(colour, vec3(0.2126, 0.7152, 0.0722));
    if (scene_luminance > 0.5) {
        float over = scene_luminance - 0.5;
        colour *= (0.5 + over / (1.0 + over / 0.7)) / scene_luminance;
    }

    // Near-field glare: the pyramid mixture, precomputed at half resolution
    // (bloom.frag).
    const float bloom = 0.012;
    colour = mix(colour, textureLod(bloom_map, uv, 0.0).rgb, bloom);

    vec2 global_xy = post.viewport.xy + uv * post.viewport.zw;
    vec2 ndc = vec2(
        2.0 * (global_xy.x - post.projection.z) / post.canvas.z,
        2.0 * (post.projection.w - global_xy.y) / post.canvas.w);
    vec3 ray = normalize(vec3(ndc * post.projection.xy, 1.0));
    float px_per_rad = post.canvas.w * 0.5 / post.projection.y;

    // The Sun is a soft glowing disc: no diffraction rays, no ghosts.
    colour += source_glare(ray, ndc, post.sun, post.sun_light, 0.0, px_per_rad);
    colour += source_glare(ray, ndc, post.moon, post.moon_light, 0.0, px_per_rad);

    // Natural vignetting, 60 % corrected (a lens profile's residual).
    float cos_axis = ray.z;
    colour *= mix(1.0, cos_axis * cos_axis * cos_axis * cos_axis, 0.4);

    // Camera colour rendering. The sensor's green channel reaches well into
    // the blue (and its blue into the green), and the raw conversion only
    // partly undoes it: Rayleigh blue comes out azure, open water teal.
    // Rows sum to one, so greys stay grey. Calibrated on the footage: open
    // ocean blue/green 2.3 -> 1.7 (footage 1.4-2.0), red/green 0.51 -> 0.40
    // (0.26-0.36); land is left within 0.05.
    colour = vec3(colour.r, mix(colour.g, colour.b, 0.2), mix(colour.b, colour.g, 0.1));

    // Photographic grade, as in the processed Earth Observation frames:
    // contrast in log space around mid-grey and a saturation boost (a raw
    // file developed with a "vivid" picture style), then the film curve.
    float scene_luma = max(dot(colour, vec3(0.2126, 0.7152, 0.0722)), 1.0e-6);
    float contrast = post.tone.x;
    colour *= pow(min(scene_luma, 1.0e3) / 0.18, contrast - 1.0);
    scene_luma = max(dot(colour, vec3(0.2126, 0.7152, 0.0722)), 1.0e-6);
    // Past clipping a sensor's channels saturate together: highlights run
    // to white. Without this an overexposed twilight arc kept its hue and
    // printed hard yellow and cyan lines where one channel was absent.
    float saturation = 1.3 * (1.0 - smoothstep(1.0, 6.0, scene_luma));
    colour = max(mix(vec3(scene_luma), colour, saturation), vec3(0.0));
    colour = camera_curve(colour);
    float luma = dot(colour, vec3(0.2126, 0.7152, 0.0722));

    // Sensor noise grows with gain (read + shot noise), then triangular
    // dither to the 8-bit swapchain in the sRGB-encoded domain.
    vec3 srgb = mix(12.92 * colour, 1.055 * pow(colour, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, colour));
    vec3 seed = vec3(gl_FragCoord.xy, post.tone.z);
    float n1 = hash13(seed);
    float n2 = hash13(seed + 17.13);
    float grain = (n1 + n2 - 1.0) * post.tone.w * sqrt(max(luma, 0.002));
    srgb += grain;
    srgb += (hash13(seed + 3.7) + hash13(seed + 9.1) - 1.0) / 255.0;
    srgb = clamp(srgb, 0.0, 1.0);
    colour = mix(srgb / 12.92, pow((srgb + 0.055) / 1.055, vec3(2.4)), step(0.04045, srgb));
    out_color = vec4(colour, 1.0);
}
