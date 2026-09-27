//! CPU virtual-texture feedback matching the Earth textured fragment projection.
//!
//! Feedback is intentionally independent of Vulkan and disk I/O. A renderer can
//! collect it after building its output viewports, then hand the requests to the
//! virtual-texture streamer on the following frame.

use std::collections::BTreeMap;

use crate::{
    camera::SCENE_EARTH_RADIUS,
    earthvt::{LayerDescriptor, TileKey, TileRequest},
    vulkan::{FrameUniforms, LogicalRect},
};

const SURFACE_RADIUS: f32 = SCENE_EARTH_RADIUS;
const PI: f32 = std::f32::consts::PI;
const TWO_PI: f32 = 2.0 * PI;

/// One display view consumed by [`Feedback::collect`].
#[derive(Clone, Copy, Debug)]
pub struct OutputDescriptor {
    pub output_id: u32,
    pub viewport: LogicalRect,
    pub physical_extent: [u32; 2],
}

/// Must equal `VT_MIP_BIAS` in `shaders/earth_textured.frag`: the single
/// shared bias added to log2(texels-per-pixel) on both the CPU estimator
/// below and the GPU `sample_day_vt`. Keep at 0.0 (no over/under-fetch).
pub const VT_MIP_BIAS: f32 = 0.0;

/// Limits for bounded CPU feedback work.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeedbackConfig {
    /// Maximum number of ray samples generated for any one output.
    pub max_rays_per_output: usize,
    /// Maximum number of deduplicated requests returned for a frame.
    pub max_requests: usize,
    /// Bias applied after the projected texel footprint is converted to a mip.
    pub mip_bias: f32,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            max_rays_per_output: 16_384,
            max_requests: 16_384,
            mip_bias: VT_MIP_BIAS,
        }
    }
}

/// CPU feedback collector. It has no mutable state and can be reused per frame.
#[derive(Clone, Copy, Debug)]
pub struct Feedback {
    pub config: FeedbackConfig,
}

impl Default for Feedback {
    fn default() -> Self {
        Self {
            config: FeedbackConfig::default(),
        }
    }
}

/// The output descriptors and prioritized visible tile requests for one frame.
#[derive(Clone, Debug, Default)]
pub struct FeedbackFrame {
    pub outputs: Vec<OutputDescriptor>,
    pub requests: Vec<TileRequest>,
}

impl Feedback {
    /// Request-only variant of [`Feedback::collect`] for the per-frame path:
    /// `process_virtual_texture` consumes only `.requests`, so callers can
    /// skip cloning the output descriptors into a second Vec.
    pub fn collect_requests(
        self,
        uniforms: FrameUniforms,
        outputs: &[OutputDescriptor],
        layers: &[LayerDescriptor],
        frame: u64,
    ) -> Vec<TileRequest> {
        let mut priorities = BTreeMap::<TileKey, u32>::new();

        for &output in outputs {
            let (grid_x, grid_y) = grid_dimensions(output, self.config.max_rays_per_output);
            if grid_x == 0 || grid_y == 0 || !output.viewport.is_valid() {
                continue;
            }
            let pixel_dx = output.viewport.width / output.physical_extent[0] as f32;
            let pixel_dy = output.viewport.height / output.physical_extent[1] as f32;
            for sample_y in 0..grid_y {
                for sample_x in 0..grid_x {
                    let x = (sample_x as f32 + 0.5) / grid_x as f32;
                    let y = (sample_y as f32 + 0.5) / grid_y as f32;
                    let Some(uv) = surface_uv(uniforms, output.viewport, x, y) else {
                        continue;
                    };
                    let uv_dx = surface_uv_at_global(
                        uniforms,
                        output.viewport,
                        output.viewport.x + x * output.viewport.width + pixel_dx,
                        output.viewport.y + y * output.viewport.height,
                    );
                    let uv_dy = surface_uv_at_global(
                        uniforms,
                        output.viewport,
                        output.viewport.x + x * output.viewport.width,
                        output.viewport.y + y * output.viewport.height + pixel_dy,
                    );
                    for &layer in layers {
                        let mip = estimate_mip(layer, uv, uv_dx, uv_dy, self.config.mip_bias);
                        let Some((mip_tiles_x, mip_tiles_y)) = layer.tile_grid(mip) else {
                            continue;
                        };
                        let key = TileKey::new(
                            layer.id,
                            mip,
                            tile_coordinate(uv[0], mip_tiles_x),
                            tile_coordinate_clamped(uv[1], mip_tiles_y),
                        );
                        let priority = priorities.entry(key).or_default();
                        *priority = priority.saturating_add(1);
                    }
                }
            }
        }

        let mut requests = priorities
            .into_iter()
            .map(|(key, priority)| TileRequest::visible(key, priority, frame))
            .collect::<Vec<_>>();
        requests.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.key.cmp(&right.key))
        });
        requests.truncate(self.config.max_requests);

        requests
    }

    /// Include output descriptors for diagnostics that need the full feedback frame.
    #[cfg(test)]
    pub fn collect(
        self,
        uniforms: FrameUniforms,
        outputs: &[OutputDescriptor],
        layers: &[LayerDescriptor],
        frame: u64,
    ) -> FeedbackFrame {
        FeedbackFrame {
            outputs: outputs.to_vec(),
            requests: self.collect_requests(uniforms, outputs, layers, frame),
        }
    }
}

fn grid_dimensions(output: OutputDescriptor, max_rays: usize) -> (usize, usize) {
    if max_rays == 0 || output.physical_extent[0] == 0 || output.physical_extent[1] == 0 {
        return (0, 0);
    }
    let width = output.physical_extent[0] as usize;
    let height = output.physical_extent[1] as usize;
    let aspect = width as f64 / height as f64;
    let mut grid_x = ((max_rays as f64 * aspect).sqrt().ceil() as usize)
        .max(1)
        .min(width);
    let mut grid_y = ((max_rays as f64 / aspect).sqrt().ceil() as usize)
        .max(1)
        .min(height);
    while grid_x.saturating_mul(grid_y) > max_rays {
        if grid_x >= grid_y && grid_x > 1 {
            grid_x -= 1;
        } else if grid_y > 1 {
            grid_y -= 1;
        } else {
            break;
        }
    }
    (grid_x, grid_y)
}

fn surface_uv(uniforms: FrameUniforms, viewport: LogicalRect, x: f32, y: f32) -> Option<[f32; 2]> {
    surface_uv_at_global(
        uniforms,
        viewport,
        viewport.x + x * viewport.width,
        viewport.y + y * viewport.height,
    )
}

fn surface_uv_at_global(
    uniforms: FrameUniforms,
    _viewport: LogicalRect,
    x: f32,
    y: f32,
) -> Option<[f32; 2]> {
    let canonical_x = 2.0 * (x - uniforms.focus_x) / uniforms.canvas.width;
    let canonical_y = 2.0 * (uniforms.focus_y - y) / uniforms.canvas.height;
    let ray = normalize(add(
        uniforms.forward,
        add(
            scale(uniforms.right, canonical_x * uniforms.tan_half_fov_x),
            scale(uniforms.up, canonical_y * uniforms.tan_half_fov_y),
        ),
    ))?;
    let distance = sphere_intersection(uniforms.camera_position, ray, SURFACE_RADIUS);
    if distance <= 0.0 {
        return None;
    }
    let point = add(uniforms.camera_position, scale(ray, distance));
    let normal = normalize(point)?;
    let longitude = normal[1].atan2(normal[0]);
    Some([
        fract(0.5 - longitude / TWO_PI),
        0.5 - normal[2].clamp(-1.0, 1.0).asin() / PI,
    ])
}

fn sphere_intersection(origin: [f32; 3], direction: [f32; 3], radius: f32) -> f32 {
    let b = dot(origin, direction);
    let c = dot(origin, origin) - radius * radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return -1.0;
    }
    let root = discriminant.sqrt();
    let near = -b - root;
    if near > 0.0 {
        near
    } else {
        -b + root
    }
}

fn estimate_mip(
    layer: LayerDescriptor,
    center: [f32; 2],
    right: Option<[f32; 2]>,
    down: Option<[f32; 2]>,
    bias: f32,
) -> u16 {
    let Some((base_width, base_height)) = layer.mip_dimensions(0) else {
        return 0;
    };
    let footprint = [right, down]
        .into_iter()
        .flatten()
        .map(|uv| {
            let du = wrapped_distance(uv[0], center[0]) * base_width as f32;
            let dv = (uv[1] - center[1]).abs() * base_height as f32;
            // Mirror the shader's length(dx*dimensions): L2, not L-infinity,
            // so diagonal gradients estimate the same mip the GPU requests.
            du.hypot(dv)
        })
        .fold(1.0_f32, f32::max);
    let mip = (footprint.log2() + bias).floor().max(0.0) as u16;
    mip.min(layer.mip_count.saturating_sub(1))
}

fn tile_coordinate(uv: f32, tiles: u32) -> u32 {
    if tiles == 0 {
        return 0;
    }
    ((uv.clamp(0.0, 1.0 - f32::EPSILON) * tiles as f32).floor() as u32) % tiles
}

fn tile_coordinate_clamped(uv: f32, tiles: u32) -> u32 {
    if tiles == 0 {
        return 0;
    }
    (uv.clamp(0.0, 1.0 - f32::EPSILON) * tiles as f32).floor() as u32
}

fn wrapped_distance(left: f32, right: f32) -> f32 {
    let distance = (left - right).abs();
    distance.min(1.0 - distance)
}

fn fract(value: f32) -> f32 {
    value - value.floor()
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn add(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn scale(value: [f32; 3], scalar: f32) -> [f32; 3] {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

fn normalize(value: [f32; 3]) -> Option<[f32; 3]> {
    let length = dot(value, value).sqrt();
    (length.is_finite() && length > f32::EPSILON).then(|| scale(value, length.recip()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::earthvt::{PixelFormat, TextureChannel};

    fn layer(id: u16, width: u32, height: u32, mip_count: u16) -> LayerDescriptor {
        LayerDescriptor {
            id,
            channel: TextureChannel::DayColor,
            format: PixelFormat::Bc7,
            mip_count,
            base_width: width,
            base_height: height,
            first_index: 0,
            index_count: 0,
            flags: 0,
        }
    }

    fn output(id: u32, width: u32, height: u32) -> OutputDescriptor {
        OutputDescriptor {
            output_id: id,
            viewport: LogicalRect {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
            physical_extent: [width, height],
        }
    }

    #[test]
    fn seam_uses_short_wrapped_distance_and_wraps_tile_x() {
        assert!((wrapped_distance(0.998, 0.002) - 0.004).abs() < 0.0001);
        assert_eq!(tile_coordinate(1.0, 8), 7);
        assert_eq!(tile_coordinate(0.0, 8), 0);
    }

    #[test]
    fn mip_estimate_is_bounded_by_layer_pyramid() {
        let layer = layer(1, 65_536, 32_768, 8);
        assert_eq!(
            estimate_mip(layer, [0.5, 0.5], Some([0.501, 0.5]), None, 0.0),
            6
        );
        assert_eq!(
            estimate_mip(layer, [0.5, 0.5], Some([0.501, 0.5]), None, 20.0),
            7
        );
    }
    #[test]
    fn diagonal_gradient_matches_l2_mip() {
        let layer = layer(1, 65_536, 32_768, 8);
        // Exactly 1.5 texels per axis: L-infinity requests mip 0, L2 mip 1.
        let uv = [0.5 + 1.5 / 65_536.0, 0.5 + 1.5 / 32_768.0];
        assert_eq!(estimate_mip(layer, [0.5, 0.5], Some([uv[0], 0.5]), None, 0.0), 0);
        assert_eq!(estimate_mip(layer, [0.5, 0.5], Some(uv), None, 0.0), 1);
    }

    #[test]
    fn requests_are_deduplicated_and_priority_is_visible_coverage() {
        let mut config = FeedbackConfig::default();
        config.max_rays_per_output = 64;
        let frame = Feedback { config }.collect(
            FrameUniforms::default(),
            &[output(1, 64, 64)],
            &[layer(3, 256, 256, 1)],
            9,
        );
        assert!(!frame.requests.is_empty());
        assert!(frame
            .requests
            .windows(2)
            .all(|pair| pair[0].key != pair[1].key));
        assert!(frame
            .requests
            .iter()
            .all(|request| request.kind == crate::earthvt::RequestKind::Visible));
        assert!(frame.requests.windows(2).all(|pair| {
            pair[0].priority > pair[1].priority
                || (pair[0].priority == pair[1].priority && pair[0].key <= pair[1].key)
        }));
    }

    #[test]
    fn identical_multi_output_views_share_one_request_set() {
        let config = FeedbackConfig {
            max_rays_per_output: 64,
            ..FeedbackConfig::default()
        };
        let one = (Feedback { config }).collect(
            FrameUniforms::default(),
            &[output(1, 64, 64)],
            &[layer(3, 256, 256, 1)],
            4,
        );
        let two = (Feedback { config }).collect(
            FrameUniforms::default(),
            &[output(1, 64, 64), output(2, 64, 64)],
            &[layer(3, 256, 256, 1)],
            4,
        );
        assert_eq!(two.outputs.len(), 2);
        assert_eq!(one.requests.len(), two.requests.len());
        assert_eq!(one.requests[0].key, two.requests[0].key);
    }
}
