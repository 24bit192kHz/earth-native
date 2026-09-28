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
//  - diffraction by a nine-blade iris: an 18-ray starburst on the Sun;
//  - internal reflections: faint coloured ghosts mirrored through the centre;
//  - natural vignetting (cos^4, partly corrected as a camera profile does);
//  - a filmic response, mild saturation and dithered 8-bit output with
//    exposure-dependent sensor noise.

layout(location = 0) noperspective in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D scene;

layout(push_constant) uniform PostFrame {
    vec4 tone;        // x contrast, y mode (0 camera, 1 legacy), z seed, w noise
    vec4 projection;  // tan half-fov x/y, optical centre x/y (canvas units)
    vec4 canvas;      // x Sun starburst strength, y ghost strength, zw canvas size
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

vec4 cubic_weights(float v) {
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0 / 6.0);
}

// Cubic B-spline filtered read of one pyramid level (4 bilinear taps), which
// turns the blocky box pyramid into smooth Gaussian-like blurs.
vec3 smooth_level(float level, vec2 uv) {
    vec2 size = vec2(textureSize(scene, int(level)));
    vec2 texel = uv * size - 0.5;
    vec2 f = fract(texel);
    texel -= f;
    vec4 xw = cubic_weights(f.x);
    vec4 yw = cubic_weights(f.y);
    vec4 c = texel.xxyy + vec2(-0.5, 1.5).xyxy;
    vec4 s = vec4(xw.xz + xw.yw, yw.xz + yw.yw);
    vec4 offset = (c + vec4(xw.yw, yw.yw) / s) / size.xxyy;
    vec3 a = textureLod(scene, offset.xz, level).rgb;
    vec3 b = textureLod(scene, offset.yz, level).rgb;
    vec3 d = textureLod(scene, offset.xw, level).rgb;
    vec3 e = textureLod(scene, offset.yw, level).rgb;
    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);
    return mix(mix(e, d, sx), mix(b, a, sx), sy);
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

// Internal reflections: discs mirrored through the optical centre.
vec3 ghosts(vec2 pixel_ndc, vec4 source, vec4 light, float px_per_rad) {
    if (light.w < 0.5 || source.z <= 0.05) return vec3(0.0);
    vec2 s = source.xy / source.z / post.projection.xy;
    if (any(greaterThan(abs(s), vec2(1.6)))) return vec3(0.0);
    vec3 energy = light.rgb;
    const int count = 5;
    const float along[count] = float[](-0.35, -0.72, -1.15, 0.42, -1.55);
    const float radius[count] = float[](0.030, 0.085, 0.050, 0.022, 0.16);
    const vec3 colour[count] = vec3[](vec3(0.45, 1.0, 0.55), vec3(0.75, 0.45, 1.0),
        vec3(1.0, 0.75, 0.35), vec3(0.5, 0.8, 1.0), vec3(0.6, 1.0, 0.8));
    vec3 sum = vec3(0.0);
    float aspect = post.projection.x / post.projection.y;
    for (int i = 0; i < count; ++i) {
        vec2 centre = s * along[i];
        vec2 d = (pixel_ndc - centre) * vec2(aspect, 1.0);
        float r = length(d) / radius[i];
        float disc = smoothstep(1.0, 0.82, r) * (0.55 + 0.45 * r * r);
        sum += colour[i] * disc / (radius[i] * radius[i]);
    }
    return energy * sum * 2.5e-6;
}

void main() {
    vec2 uv = in_uv;
    vec3 colour = min(textureLod(scene, uv, 0.0).rgb, vec3(65000.0));

    if (post.tone.y > 0.5) {
        out_color = vec4(legacy_filmic(colour), 1.0);
        return;
    }

    // Near-field glare: a power-law mixture of blurred pyramid levels, each
    // softly compressed above BLOOM_KNEE (local adaptation). A night exposure
    // puts a thin sunlit limb ~2^10 over white; uncompressed, its bloom was
    // a white fog, which is why the meter used to hold a daylight exposure
    // for it and the night side and stars went black.
    const float BLOOM_KNEE = 16.0;
    vec3 blur = vec3(0.0);
    float weight_sum = 0.0;
    float weight = 1.0;
    int top = min(textureQueryLevels(scene) - 1, 9);
    for (int level = 1; level <= top; ++level) {
        vec3 level_colour = smooth_level(float(level), uv);
        level_colour /= 1.0 + max(level_colour.r, max(level_colour.g, level_colour.b)) / BLOOM_KNEE;
        blur += level_colour * weight;
        weight_sum += weight;
        weight *= 0.78;
    }
    const float bloom = 0.012;
    colour = mix(colour, min(blur / max(weight_sum, 1.0e-6), vec3(65000.0)), bloom);

    vec2 global_xy = post.viewport.xy + uv * post.viewport.zw;
    vec2 ndc = vec2(
        2.0 * (global_xy.x - post.projection.z) / post.canvas.z,
        2.0 * (post.projection.w - global_xy.y) / post.canvas.w);
    vec3 ray = normalize(vec3(ndc * post.projection.xy, 1.0));
    float px_per_rad = post.canvas.w * 0.5 / post.projection.y;

    colour += source_glare(ray, ndc, post.sun, post.sun_light, post.canvas.x, px_per_rad);
    colour += source_glare(ray, ndc, post.moon, post.moon_light, 0.15, px_per_rad);
    colour += ghosts(ndc, post.sun, post.sun_light, px_per_rad) * post.canvas.y;

    // Natural vignetting, 60 % corrected (a lens profile's residual).
    float cos_axis = ray.z;
    colour *= mix(1.0, cos_axis * cos_axis * cos_axis * cos_axis, 0.4);

    // Photographic grade, as in the processed Earth Observation frames:
    // contrast in log space around mid-grey and a saturation boost (a raw
    // file developed with a "vivid" picture style), then the film curve.
    float scene_luma = max(dot(colour, vec3(0.2126, 0.7152, 0.0722)), 1.0e-6);
    float contrast = post.tone.x;
    colour *= pow(min(scene_luma, 1.0e3) / 0.18, contrast - 1.0);
    scene_luma = max(dot(colour, vec3(0.2126, 0.7152, 0.0722)), 1.0e-6);
    colour = max(mix(vec3(scene_luma), colour, 1.3), vec3(0.0));
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
