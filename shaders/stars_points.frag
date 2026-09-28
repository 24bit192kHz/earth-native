#version 460

// Point-spread function of one catalogue star, integrated over the pixel
// (a Gaussian's pixel integral is a product of erf differences), so faint
// stars neither alias nor twinkle as the view moves by sub-pixel steps.

layout(location = 0) noperspective in vec2 in_offset_px;
layout(location = 1) flat in vec4 in_colour_energy;
layout(location = 2) flat in float in_sigma_px;
layout(location = 0) out vec4 out_color;

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7.
float erf_approx(float x) {
    float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
        * t * exp(-x * x);
    return sign(x) * y;
}

float pixel_integral(float centre, float sigma) {
    float k = 0.70710678 / sigma;
    return 0.5 * (erf_approx((centre + 0.5) * k) - erf_approx((centre - 0.5) * k));
}

void main() {
    float sigma = in_sigma_px;
    float psf = pixel_integral(in_offset_px.x, sigma) * pixel_integral(in_offset_px.y, sigma);
    vec3 light = in_colour_energy.rgb * in_colour_energy.a * psf;
    out_color = vec4(min(light, vec3(30000.0)), 0.0);
}
