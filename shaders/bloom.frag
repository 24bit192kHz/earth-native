#version 460

// Near-field glare of the camera stage (post.frag), at half resolution: the
// power-law mixture of blurred pyramid levels is a smooth, low-frequency
// field, so one pass here at a quarter of the pixels, read back with one
// bilinear tap, replaces 36 texture fetches per full-resolution pixel (that
// loop was a third of the frame's GPU time).

layout(location = 0) noperspective in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D scene;

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

// Push constants (the camera stage's block): tone.x selects the pass.
layout(push_constant) uniform BloomPass {
    vec4 tone; // x: 0 fine pass (levels 1-2 + coarse), 1 coarse pass (levels 3+)
} pass_info;

// Levels 1-2 (2-4 px blurs) at half resolution; levels 3+ (8 px and wider)
// at one eighth, in their own pass, then added here with one bilinear tap:
// per level the same compression and weight 0.78^(level - 1), normalised
// by the sum over all levels, as when every level was sampled at every
// half-resolution pixel (36 fetches, most of the camera stage's cost).
const int FINE_LEVELS = 2;
layout(set = 0, binding = 1) uniform sampler2D coarse_bloom;

void main() {
    vec2 uv = in_uv;
    // Near-field glare: a power-law mixture of blurred pyramid levels, each
    // softly compressed above BLOOM_KNEE (local adaptation). A night exposure
    // puts a thin sunlit limb ~2^10 over white; uncompressed, its bloom was
    // a white fog, which is why the meter used to hold a daylight exposure
    // for it and the night side and stars went black.
    const float BLOOM_KNEE = 16.0;
    bool coarse_pass = pass_info.tone.x > 0.5;
    int top = min(textureQueryLevels(scene) - 1, 9);
    int first = coarse_pass ? FINE_LEVELS + 1 : 1;
    int last = coarse_pass ? top : min(FINE_LEVELS, top);
    vec3 blur = vec3(0.0);
    float weight_sum = 0.0;
    float weight = 1.0;
    for (int level = 1; level <= top; ++level) {
        if (level >= first && level <= last) {
            vec3 level_colour = smooth_level(float(level), uv);
            level_colour /= 1.0 + max(level_colour.r, max(level_colour.g, level_colour.b)) / BLOOM_KNEE;
            blur += level_colour * weight;
        }
        weight_sum += weight;
        weight *= 0.78;
    }
    if (coarse_pass) {
        // Unnormalised weighted sum of the coarse levels.
        out_color = vec4(min(blur, vec3(65000.0)), 1.0);
        return;
    }
    blur += textureLod(coarse_bloom, uv, 0.0).rgb;
    out_color = vec4(min(blur / max(weight_sum, 1.0e-6), vec3(65000.0)), 1.0);
}
