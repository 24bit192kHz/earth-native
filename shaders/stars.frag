#version 460

layout(location = 0) noperspective in vec2 in_uv;
layout(location = 0) out vec4 out_color;

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

float hash21(vec2 value) {
    value = fract(value * vec2(123.34, 456.21));
    value += dot(value, value + 45.32);
    return fract(value.x * value.y);
}

vec3 star_field(vec2 canvas_uv) {
    // The optional pinned panorama has its own pipeline. This remains the
    // zero-resource fallback while no production asset is available.
    vec2 grid = canvas_uv * vec2(4096.0, 2048.0);
    vec2 cell = floor(grid);
    float seed = hash21(cell);
    // Empty cells are 99.65% of the field: return before the pow / sqrt /
    // second-hash star math so the cold path stays out of the hot stream.
    // Bit-exact — the old value on this path was tint * point * 0.0.
    if (seed < 0.9965) {
        return vec3(0.0);
    }
    vec2 local = fract(grid) - 0.5;
    float radius = mix(0.025, 0.18, pow(seed, 18.0));
    float point = smoothstep(radius, 0.0, length(local));
    vec3 tint = mix(vec3(0.55, 0.68, 1.0), vec3(1.0, 0.82, 0.58), hash21(cell + 17.0));
    // The gate is taken here (seed >= 0.9965), so step() was 1.0; x * 1.0
    // is exact in IEEE-754, making this identical to tint * point * step.
    return tint * point;
}

void main() {
    vec2 output_uv = in_uv;
    vec2 global_xy = frame.viewport_rect.xy + output_uv * frame.viewport_rect.zw;
    vec2 canvas_uv = (global_xy - frame.canvas_rect.xy) / frame.canvas_rect.zw;
    vec3 background = vec3(0.003, 0.006, 0.014) + star_field(canvas_uv);
    out_color = vec4(background, 1.0);
}
