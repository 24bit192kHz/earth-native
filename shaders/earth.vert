#version 460

// Exactly what the fragment stages consume: output UV in [0,1] in-viewport
// (vertices map exactly to (0,0),(2,0),(0,2)). `noperspective` is exact here
// because gl_Position.w is identically 1.0, so perspective correction is a
// no-op divide the hardware can skip.
layout(location = 0) noperspective out vec2 out_uv;

const vec2 POSITIONS[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2(3.0, -1.0),
    vec2(-1.0, 3.0)
);

void main() {
    vec2 position = POSITIONS[gl_VertexIndex];
    out_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
