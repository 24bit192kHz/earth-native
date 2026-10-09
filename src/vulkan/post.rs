//! Camera stage. Every scene pass writes scene-linear, pre-exposed radiance
//! into a half-float target; this stage builds that target's mip pyramid (the
//! lens-glare and metering source) and draws the final image through a
//! lens/sensor model into the swapchain.

use std::{io::Cursor, mem::size_of};

use ash::{util::read_spv, vk, Device};

use super::{find_memory_type, find_readback_memory_type, RendererResult};

pub(super) const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Exposure metering reads the first pyramid level at most this wide.
const METER_MAX_WIDTH: u32 = 64;
const MAX_TARGETS: u32 = 32;

/// Push constants of `post.frag`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PostFrame {
    /// x: log-space contrast of the grade, y: grade strength (1 Earth, 0 other bodies
    /// planet presentation), z: frame seed, w: sensor noise at mid-grey.
    pub tone: [f32; 4],
    /// tan half-FOV x/y, optical centre x/y in canvas units.
    pub projection: [f32; 4],
    pub canvas: [f32; 4],
    pub viewport: [f32; 4],
    /// Camera-space direction (right, up, forward) and angular radius.
    pub sun: [f32; 4],
    /// Unoccluded-and-transmitted solar irradiance reaching the lens,
    /// pre-exposed, RGB; w = 1 when the Sun is in front of the camera.
    pub sun_light: [f32; 4],
    pub moon: [f32; 4],
    pub moon_light: [f32; 4],
}

pub(super) struct PostPipeline {
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    /// Half-resolution near-field glare (bloom.frag) into the HDR format.
    pub bloom_pipeline: vk::Pipeline,
    sampler: vk::Sampler,
}

impl PostPipeline {
    pub fn create(device: &Device, swapchain_format: vk::Format) -> RendererResult<Self> {
        let vertex_code = read_spv(&mut Cursor::new(include_bytes!(concat!(env!("OUT_DIR"), "/earth.vert.spv"))))?;
        let fragment_code = read_spv(&mut Cursor::new(include_bytes!(concat!(env!("OUT_DIR"), "/post.frag.spv"))))?;
        let bloom_code = read_spv(&mut Cursor::new(include_bytes!(concat!(env!("OUT_DIR"), "/bloom.frag.spv"))))?;
        // 0: the HDR pyramid, 1: the half-resolution bloom.
        let bindings = [0, 1].map(|binding| vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT));
        unsafe {
            let set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            // Three sets per target (camera, fine and coarse bloom passes),
            // two images each.
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_TARGETS * 6)];
            let pool = match device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(MAX_TARGETS * 3)
                    .pool_sizes(&sizes),
                None,
            ) {
                Ok(pool) => pool,
                Err(error) => {
                    device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(error.into());
                }
            };
            let set_layouts = [set_layout];
            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .size(size_of::<PostFrame>() as u32)];
            let layout = match device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&push),
                None,
            ) {
                Ok(layout) => layout,
                Err(error) => {
                    device.destroy_descriptor_pool(pool, None);
                    device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(error.into());
                }
            };
            let sampler = match device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            ) {
                Ok(sampler) => sampler,
                Err(error) => {
                    device.destroy_pipeline_layout(layout, None);
                    device.destroy_descriptor_pool(pool, None);
                    device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(error.into());
                }
            };
            let destroy_all = |device: &Device| {
                device.destroy_sampler(sampler, None);
                device.destroy_pipeline_layout(layout, None);
                device.destroy_descriptor_pool(pool, None);
                device.destroy_descriptor_set_layout(set_layout, None);
            };
            let vertex = match device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vertex_code), None) {
                Ok(module) => module,
                Err(error) => {
                    destroy_all(device);
                    return Err(error.into());
                }
            };
            let fragment = match device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&fragment_code), None) {
                Ok(module) => module,
                Err(error) => {
                    device.destroy_shader_module(vertex, None);
                    destroy_all(device);
                    return Err(error.into());
                }
            };
            let bloom_fragment = match device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&bloom_code), None) {
                Ok(module) => module,
                Err(error) => {
                    device.destroy_shader_module(fragment, None);
                    device.destroy_shader_module(vertex, None);
                    destroy_all(device);
                    return Err(error.into());
                }
            };
            let pipeline = super::create_graphics_pipeline(
                device, vertex, fragment, layout, swapchain_format, super::Blend::Opaque);
            let bloom_pipeline = super::create_graphics_pipeline(
                device, vertex, bloom_fragment, layout, HDR_FORMAT, super::Blend::Opaque);
            device.destroy_shader_module(vertex, None);
            device.destroy_shader_module(fragment, None);
            device.destroy_shader_module(bloom_fragment, None);
            match (pipeline, bloom_pipeline) {
                (Ok(pipeline), Ok(bloom_pipeline)) => Ok(Self { set_layout, pool, layout, pipeline, bloom_pipeline, sampler }),
                (pipeline, bloom_pipeline) => {
                    if let Ok(pipeline) = pipeline { device.destroy_pipeline(pipeline, None); }
                    if let Ok(pipeline) = bloom_pipeline { device.destroy_pipeline(pipeline, None); }
                    destroy_all(device);
                    Err("could not create the camera-stage pipelines".into())
                }
            }
        }
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline(self.bloom_pipeline, None);
        device.destroy_sampler(self.sampler, None);
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_descriptor_pool(self.pool, None);
        device.destroy_descriptor_set_layout(self.set_layout, None);
    }
}

/// Scene-referred luminance statistics from one completed frame.
#[derive(Clone, Copy, Debug)]
pub(super) struct MeterReading {
    /// log2 of the metering key: the 85th-percentile Earth luminance (exposure
    /// removed), in the renderer's units (white Lambertian under a zenith Sun
    /// = 1).
    pub log2_luminance: f32,
    /// Fraction of the frame the Earth and its air cover.
    pub coverage: f32,
    /// How far the key is sunlit or twilit Earth at its own 85th percentile
    /// (1), not a bright band, a crescent or the night side (0): the camera
    /// then opens up for only part of a deficit, as a low Sun and dusk
    /// darken the ground in the footage at one exposure. It fades over the
    /// first stop the highlight or crescent rule raises the key: a flag
    /// here closed the exposure by ~4 stops in one frame at dusk, when the
    /// last of a crescent's share left the frame.
    pub sunlit: f32,
    /// log2 key for a thin sunlit band far over the key (the sunrise and
    /// sunset arc), which the controller uses while the Sun is near the
    /// limb in front of the camera.
    pub band_log2: Option<f32>,
}

/// One meter texel of the Earth: its scene luminance, its weight (coverage,
/// centre-weighted) and whether it is sunlit (scaled by the camera's
/// exposure) or drawn at the night side's own exposure.
struct MeterSample {
    luminance: f32,
    weight: f32,
    lit: bool,
}

/// Pre-exposure of the night series the night side is drawn at (EV 16,
/// `NIGHT_SERIES_PREEXPOSURE` in earth_textured.frag; a test in vulkan.rs
/// keeps the two equal).
pub(super) const NIGHT_SERIES_PREEXPOSURE: f32 = 65536.0;

/// Scene luminance above which a meter texel holds sunlit Earth: a tenth
/// of a percent of white under a zenith Sun, far over moonlit cloud
/// (~1e-6) and city lights, and still under a crescent at a low Sun.
const DAYLIGHT_FLOOR: f32 = 1.0e-3;

pub(super) struct HdrTarget {
    pool: vk::DescriptorPool,
    image: vk::Image,
    memory: vk::DeviceMemory,
    pub attachment_view: vk::ImageView,
    sampled_view: vk::ImageView,
    pub descriptor_set: vk::DescriptorSet,
    pub extent: vk::Extent2D,
    mip_levels: u32,
    /// Bloom (bloom.frag): levels 1-2 at half resolution plus levels 3+
    /// from a one-eighth-resolution pass; each with the set its pass reads.
    bloom: BloomImage,
    coarse: BloomImage,
    meter_level: u32,
    meter_extent: vk::Extent2D,
    meter_buffer: vk::Buffer,
    meter_memory: vk::DeviceMemory,
    meter_ptr: *const u16,
    /// Pre-exposure of the frame whose pyramid the meter buffer holds.
    pub meter_preexposure: Option<f32>,
    /// Pre-exposure of the frame being recorded (becomes the meter's on
    /// completion).
    pub pending_preexposure: Option<f32>,
}

fn full_mip_count(extent: vk::Extent2D) -> u32 {
    u32::BITS - extent.width.max(extent.height).max(1).leading_zeros()
}

impl HdrTarget {
    pub fn create(
        device: &Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        post: &PostPipeline,
        extent: vk::Extent2D,
    ) -> RendererResult<Self> {
        let mip_levels = full_mip_count(extent);
        let mut meter_level = 0;
        while (extent.width >> meter_level) > METER_MAX_WIDTH && meter_level + 1 < mip_levels {
            meter_level += 1;
        }
        let meter_extent = vk::Extent2D {
            width: (extent.width >> meter_level).max(1),
            height: (extent.height >> meter_level).max(1),
        };
        unsafe {
            let image = device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(HDR_FORMAT)
                    .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
                    .mip_levels(mip_levels)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let requirements = device.get_image_memory_requirements(image);
            let memory = match find_memory_type(memory_properties, requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .and_then(|index| Ok(device.allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(index), None)?))
            {
                Ok(memory) => memory,
                Err(error) => {
                    device.destroy_image(image, None);
                    return Err(error);
                }
            };
            let cleanup_image = |device: &Device| {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            };
            if let Err(error) = device.bind_image_memory(image, memory, 0) {
                cleanup_image(device);
                return Err(error.into());
            }
            let view = |levels: u32| {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(HDR_FORMAT)
                        .subresource_range(vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(levels)
                            .layer_count(1)),
                    None,
                )
            };
            let attachment_view = match view(1) {
                Ok(view) => view,
                Err(error) => {
                    cleanup_image(device);
                    return Err(error.into());
                }
            };
            let sampled_view = match view(mip_levels) {
                Ok(view) => view,
                Err(error) => {
                    device.destroy_image_view(attachment_view, None);
                    cleanup_image(device);
                    return Err(error.into());
                }
            };
            let cleanup_views = |device: &Device| {
                device.destroy_image_view(sampled_view, None);
                device.destroy_image_view(attachment_view, None);
                cleanup_image(device);
            };
            let set_layouts = [post.set_layout];
            let descriptor_set = match device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(post.pool).set_layouts(&set_layouts))
            {
                Ok(sets) => sets[0],
                Err(error) => {
                    cleanup_views(device);
                    return Err(error.into());
                }
            };
            let info = [vk::DescriptorImageInfo::default()
                .sampler(post.sampler)
                .image_view(sampled_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info)], &[]);
            let meter_bytes = u64::from(meter_extent.width) * u64::from(meter_extent.height) * 8;
            let cleanup_set = |device: &Device| {
                let _ = device.free_descriptor_sets(post.pool, &[descriptor_set]);
                cleanup_views(device);
            };
            let meter_buffer = match device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(meter_bytes)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            ) {
                Ok(buffer) => buffer,
                Err(error) => {
                    cleanup_set(device);
                    return Err(error.into());
                }
            };
            let buffer_requirements = device.get_buffer_memory_requirements(meter_buffer);
            let meter_memory = match find_readback_memory_type(
                memory_properties,
                buffer_requirements.memory_type_bits,
            )
            .and_then(|index| Ok(device.allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(buffer_requirements.size).memory_type_index(index), None)?))
            {
                Ok(memory) => memory,
                Err(error) => {
                    device.destroy_buffer(meter_buffer, None);
                    cleanup_set(device);
                    return Err(error);
                }
            };
            let mapped = device.bind_buffer_memory(meter_buffer, meter_memory, 0).and_then(|()| {
                device.map_memory(meter_memory, 0, meter_bytes, vk::MemoryMapFlags::empty())
            });
            let meter_ptr = match mapped {
                Ok(pointer) => pointer as *const u16,
                Err(error) => {
                    device.destroy_buffer(meter_buffer, None);
                    device.free_memory(meter_memory, None);
                    cleanup_set(device);
                    return Err(error.into());
                }
            };
            let mut target = Self {
                pool: post.pool,
                image,
                memory,
                attachment_view,
                sampled_view,
                descriptor_set,
                extent,
                mip_levels,
                meter_level,
                meter_extent,
                meter_buffer,
                meter_memory,
                meter_ptr,
                meter_preexposure: None,
                pending_preexposure: None,
                bloom: BloomImage::empty(extent, 2),
                coarse: BloomImage::empty(extent, 8),
            };
            if let Err(error) = target.create_bloom(device, memory_properties, post) {
                target.destroy(device);
                return Err(error);
            }
            Ok(target)
        }
    }

    unsafe fn create_bloom(&mut self, device: &Device, memory_properties: vk::PhysicalDeviceMemoryProperties, post: &PostPipeline) -> RendererResult<()> {
        self.bloom.create(device, memory_properties, post)?;
        self.coarse.create(device, memory_properties, post)?;
        let info = |view| [vk::DescriptorImageInfo::default().sampler(post.sampler).image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let (scene, bloom, coarse) = (info(self.sampled_view), info(self.bloom.view), info(self.coarse.view));
        fn write<'a>(set: vk::DescriptorSet, binding: u32, info: &'a [vk::DescriptorImageInfo; 1]) -> vk::WriteDescriptorSet<'a> {
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(binding).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(info)
        }
        // Camera pass: pyramid + bloom. Fine pass: pyramid + coarse. The
        // coarse pass reads only the pyramid (its binding 1 repeats it).
        device.update_descriptor_sets(&[
            write(self.descriptor_set, 1, &bloom),
            write(self.bloom.set, 0, &scene),
            write(self.bloom.set, 1, &coarse),
            write(self.coarse.set, 0, &scene),
            write(self.coarse.set, 1, &scene),
        ], &[]);
        Ok(())
    }

    /// The bloom passes (coarse, then fine): after `record_resolve`, before
    /// the camera pass samples the result.
    pub unsafe fn record_bloom(&self, device: &Device, command_buffer: vk::CommandBuffer, post: &PostPipeline) {
        self.coarse.record(device, command_buffer, post, 1.0);
        self.bloom.record(device, command_buffer, post, 0.0);
    }

    pub unsafe fn destroy(&self, device: &Device) {
        self.bloom.destroy(device, self.pool);
        self.coarse.destroy(device, self.pool);
        device.unmap_memory(self.meter_memory);
        device.destroy_buffer(self.meter_buffer, None);
        device.free_memory(self.meter_memory, None);
        let _ = device.free_descriptor_sets(self.pool, &[self.descriptor_set]);
        device.destroy_image_view(self.sampled_view, None);
        device.destroy_image_view(self.attachment_view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }

    fn levels(&self, base: u32, count: u32) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(base)
            .level_count(count)
            .layer_count(1)
    }

    /// Level 0 becomes the scene passes' colour attachment. The previous
    /// frame's reads are complete: this output's fence was waited on.
    pub unsafe fn record_begin(&self, device: &Device, command_buffer: vk::CommandBuffer) {
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(self.image)
            .subresource_range(self.levels(0, 1));
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    /// Box-filtered pyramid, metering copy, then every level readable by the
    /// camera pass.
    pub unsafe fn record_resolve(&self, device: &Device, command_buffer: vk::CommandBuffer) {
        let to_source = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(self.image)
            .subresource_range(self.levels(0, 1));
        device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_source]);
        for level in 1..self.mip_levels {
            let to_destination = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(self.image)
                .subresource_range(self.levels(level, 1));
            device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_destination]);
            let source = [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: (self.extent.width >> (level - 1)).max(1) as i32,
                    y: (self.extent.height >> (level - 1)).max(1) as i32,
                    z: 1,
                },
            ];
            let destination = [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: (self.extent.width >> level).max(1) as i32,
                    y: (self.extent.height >> level).max(1) as i32,
                    z: 1,
                },
            ];
            let layers = |mip| vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(mip)
                .layer_count(1);
            let blit = [vk::ImageBlit::default()
                .src_subresource(layers(level - 1))
                .src_offsets(source)
                .dst_subresource(layers(level))
                .dst_offsets(destination)];
            device.cmd_blit_image(command_buffer, self.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &blit, vk::Filter::LINEAR);
            let to_next_source = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(self.image)
                .subresource_range(self.levels(level, 1));
            device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_next_source]);
        }
        let copy = [vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(self.meter_level)
                .layer_count(1))
            .image_extent(vk::Extent3D { width: self.meter_extent.width, height: self.meter_extent.height, depth: 1 })];
        device.cmd_copy_image_to_buffer(command_buffer, self.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            self.meter_buffer, &copy);
        let host_read = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .buffer(self.meter_buffer)
            .size(vk::WHOLE_SIZE)];
        let to_shader = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.image)
            .subresource_range(self.levels(0, self.mip_levels));
        device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(), &[], &host_read, &[to_shader]);
    }

    /// Reads the metering level of the last completed frame. Call only after
    /// this output's fence has signalled.
    ///
    /// Metering follows how the ISS photographs are exposed: for the lit part
    /// of the scene. The key is the 85th percentile of the Earth's luminance
    /// (area-weighted, mildly centre-weighted), so a half-lit globe or a
    /// terminator view is exposed for daylight and the night side falls to
    /// black, while an all-night view opens up for moonlight and lights.
    /// Sunlit sky elements hold the exposure down only when they are a real
    /// part of the frame (over 3 %, like the sunrise band from the ISS): a
    /// thin lit limb around a night globe (~1 %), the Sun or the Moon
    /// saturate like city lights, and the bloom compresses them (post.frag),
    /// as the adapted eye sees the night side and the stars past them. The
    /// 99th percentile used before kept a daylight exposure for that thin
    /// rim. A frame without the Earth is a starfield.
    ///
    /// The Earth's night side is drawn at an exposure of its own (EV 16,
    /// `night_gain` in earth_textured.frag), so its texels do not scale
    /// with the camera's. The shader flags them with a negative coverage
    /// and the meter undoes the gain: read as scene luminance they are the
    /// same at every camera exposure. Read naively they looked a stop darker
    /// for every stop the camera opened, the target chased the exposure,
    /// and once they fell under the subject cut the key jumped to the lit
    /// limb: a sawtooth between EV 8 and 13 every two seconds, with the blue
    /// twilight arc blooming and fading in step.
    pub fn read_meter(&self) -> Option<MeterReading> {
        let preexposure = self.meter_preexposure?;
        let (width, height) = (self.meter_extent.width as usize, self.meter_extent.height as usize);
        let texels = unsafe { std::slice::from_raw_parts(self.meter_ptr, width * height * 4) };
        let mut samples: Vec<MeterSample> = Vec::with_capacity(width * height);
        let mut whole_frame: Vec<f32> = Vec::with_capacity(width * height);
        let mut coverage = 0.0_f64;
        for (index, texel) in texels.chunks_exact(4).enumerate() {
            // Overflowed half floats are the brightest texels in frame, not
            // missing data: skipping them once locked the meter at a night
            // exposure while twilight blew the frame out.
            let [r, g, b, a] = [0, 1, 2, 3].map(|c| {
                let value = crate::sky::f16_to_f32(texel[c]);
                if value.is_infinite() { 65504.0 } else { value }
            });
            if r.is_nan() || g.is_nan() || b.is_nan() || a.is_nan() {
                continue;
            }
            let lit = a >= 0.0;
            let a = a.abs().min(1.0);
            let exposure = if lit { preexposure } else { preexposure.max(NIGHT_SERIES_PREEXPOSURE) };
            coverage += f64::from(a);
            whole_frame.push(((0.2126 * r + 0.7152 * g + 0.0722 * b) / exposure).clamp(1.0e-12, 4.0));
            if a < 0.25 {
                continue;
            }
            let x = ((index % width) as f32 + 0.5) / width as f32 - 0.5;
            let y = ((index / width) as f32 + 0.5) / height as f32 - 0.5;
            let centre = 1.0 - (x * x + y * y);
            let luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / a / exposure;
            samples.push(MeterSample { luminance: luminance.clamp(1.0e-12, 4.0), weight: a * centre, lit });
        }
        let coverage = (coverage / (width * height) as f64) as f32;
        whole_frame.sort_by(|a, b| a.total_cmp(b));
        let p97 = whole_frame
            .get(((whole_frame.len() as f32 * 0.97) as usize).min(whole_frame.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0.0);
        // The band keeps its colours (orange, white, blue) at ~1.5 stops
        // over the key; at 3 stops over it was a featureless white arc.
        let highlight = if p97 > 2.0e-3 { p97 / 3.0 } else { 0.0 };
        let p995 = whole_frame
            .get(((whole_frame.len() as f32 * 0.995) as usize).min(whole_frame.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0.0);
        let total: f32 = samples.iter().map(|s| s.weight).sum();
        if total < 0.02 * (width * height) as f32 {
            // No Earth in frame: the sky has a fixed brightness, so keep
            // the exposure (no flash when the Earth comes back into view).
            return None;
        }
        samples.sort_by(|a, b| a.luminance.total_cmp(&b.luminance));
        // Meter the lit part: an airless night side is black, and with a
        // crescent it took the 85th percentile, so the exposure opened up
        // until the crescent was a white blot. Anything under 1/500 of the
        // brightest lit percent is not part of the subject. The Earth's
        // night side (moonlight, cities, aurora) is a subject in its own
        // right: it stays, and outweighs a thin twilight arc as the night
        // footage's exposure does, with the arc blooming over it.
        let lit: Vec<f32> = samples.iter().filter(|s| s.lit).map(|s| s.luminance).collect();
        if let Some(&reference) = lit.get((lit.len() * 99 / 100).min(lit.len().saturating_sub(1))) {
            samples.retain(|sample| !sample.lit || sample.luminance >= reference * 2.0e-3);
        }
        let total: f32 = samples.iter().map(|s| s.weight).sum();
        // A thin arc may bloom over the night side, but a crescent that is a
        // real part of the globe is the subject: from 1.5 % of the Earth's
        // area to 4 % its median is held to ~1.5 stops over the key. The
        // night key alone once put a wide crescent at EV 17, a white blot.
        // Daylight is found by brightness over the whole frame, not by the
        // lit flag: a thin crescent falls mostly in meter texels it shares
        // with the night side, which read as night or under the coverage
        // cut (8 of 217 Earth texels for a crescent of ~5 %).
        let daylit = &whole_frame[whole_frame.partition_point(|&l| l < DAYLIGHT_FLOOR)..];
        let earth_texels = coverage * (width * height) as f32;
        let lit_share = if earth_texels > 0.0 { daylit.len() as f32 / earth_texels } else { 0.0 };
        let crescent = daylit.get(daylit.len() / 2).map_or(0.0, |median| median / 2.8);
        let blend = ((lit_share - 0.015) / 0.025).clamp(0.0, 1.0);
        let mut accumulated = 0.0;
        let mut key = samples.last().map_or(1.0, |s| s.luminance);
        let mut key_lit = samples.last().is_none_or(|s| s.lit);
        for sample in &samples {
            accumulated += sample.weight;
            if accumulated >= 0.85 * total {
                key = sample.luminance;
                key_lit = sample.lit;
                break;
            }
        }
        // A thin sunlit band ten stops and more over the key: the twilight
        // arc at sunrise and sunset in a frame metered for the night side.
        // Exposed only once it filled 3 % of the frame, the arc sat 2^5-2^7
        // over white, a thick white band that swallowed the Sun. The meter's
        // texels each average ~30x30 pixels, so a band a few pixels thick
        // reads several times dimmer than it is: its brightest half percent
        // is held at the key, which leaves the arc's core a few stops over
        // white and its outer layers coloured, a thin line as in ISS
        // footage. The controller applies it only with the Sun near the
        // limb in view: a twilit limb in a night view stays overexposed, as
        // the footage's night exposure keeps the aurora and lights bright.
        let band = if p995 > 1024.0 * key { p995 } else { 0.0 };
        if std::env::var_os("EARTH_NATIVE_METER_DEBUG").is_some() {
            eprintln!("meter: pre={preexposure:.3e} key={key:.3e} lit={key_lit} p97={p97:.3e} highlight={highlight:.3e} p995={p995:.3e} band={band:.3e} lit_share={lit_share:.3} crescent={crescent:.3e} blend={blend:.2} earth_weight={total:.1} texels={}", width * height);
        }
        let metered = key;
        // Log-space blend: a partial share closes down by part of the stops.
        let key = key.max(highlight);
        let key = if crescent > key { key * (crescent / key).powf(blend) } else { key };
        Some(MeterReading {
            log2_luminance: key.log2(),
            coverage,
            sunlit: if key_lit { (1.0 - (key / metered).log2()).clamp(0.0, 1.0) } else { 0.0 },
            band_log2: (band > key).then(|| band.log2()),
        })
    }
}

/// One bloom render target (a fraction of the output's size) and the
/// descriptor set its pass samples through.
struct BloomImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    extent: vk::Extent2D,
    set: vk::DescriptorSet,
}

impl BloomImage {
    fn empty(extent: vk::Extent2D, divisor: u32) -> Self {
        Self {
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            extent: vk::Extent2D { width: extent.width.div_ceil(divisor).max(1), height: extent.height.div_ceil(divisor).max(1) },
            set: vk::DescriptorSet::null(),
        }
    }

    unsafe fn create(&mut self, device: &Device, memory_properties: vk::PhysicalDeviceMemoryProperties, post: &PostPipeline) -> RendererResult<()> {
        self.image = device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(HDR_FORMAT)
                .extent(vk::Extent3D { width: self.extent.width, height: self.extent.height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )?;
        let requirements = device.get_image_memory_requirements(self.image);
        let index = find_memory_type(memory_properties, requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        self.memory = device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(index), None)?;
        device.bind_image_memory(self.image, self.memory, 0)?;
        self.view = device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(self.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(HDR_FORMAT)
                .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1)),
            None,
        )?;
        let set_layouts = [post.set_layout];
        self.set = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default().descriptor_pool(post.pool).set_layouts(&set_layouts))?[0];
        Ok(())
    }

    unsafe fn record(&self, device: &Device, command_buffer: vk::CommandBuffer, post: &PostPipeline, pass: f32) {
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let to_attachment = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(self.image)
            .subresource_range(range);
        device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::DependencyFlags::empty(), &[], &[], &[to_attachment]);
        let attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(self.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: self.extent };
        device.cmd_begin_rendering(command_buffer, &vk::RenderingInfo::default()
            .render_area(area).layer_count(1).color_attachments(&attachment));
        device.cmd_set_viewport(command_buffer, 0, &[vk::Viewport {
            x: 0.0, y: 0.0, width: self.extent.width as f32, height: self.extent.height as f32,
            min_depth: 0.0, max_depth: 1.0,
        }]);
        device.cmd_set_scissor(command_buffer, 0, &[area]);
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, post.bloom_pipeline);
        device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, post.layout, 0, &[self.set], &[]);
        let constants = PostFrame { tone: [pass, 0.0, 0.0, 0.0], ..PostFrame::default() };
        device.cmd_push_constants(command_buffer, post.layout, vk::ShaderStageFlags::FRAGMENT, 0,
            std::slice::from_raw_parts((&constants as *const PostFrame).cast::<u8>(), size_of::<PostFrame>()));
        device.cmd_draw(command_buffer, 3, 1, 0, 0);
        device.cmd_end_rendering(command_buffer);
        let to_sampled = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.image)
            .subresource_range(range);
        device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[to_sampled]);
    }

    unsafe fn destroy(&self, device: &Device, pool: vk::DescriptorPool) {
        if self.set != vk::DescriptorSet::null() {
            let _ = device.free_descriptor_sets(pool, &[self.set]);
        }
        device.destroy_image_view(self.view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}
