//! BC7 page encoding on the GPU: the same Intel ISPC kernel and "alpha basic"
//! settings as the CPU path (intel_tex_2), ported to a WGSL compute shader by
//! the `block_compression` crate. Pages are stacked vertically in one texture
//! so each page's 4x4 blocks come back contiguous and row-major, exactly the
//! CPU path's payload layout.

use std::time::Duration;

use block_compression::{BC7Settings, CompressionVariant, GpuBlockCompressor};
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, Device, Extent3d, MapMode,
    Origin3d, PollType, Queue, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView,
};

use crate::Result;

pub struct Encoder {
    pub adapter_name: String,
    pub pages_per_batch: usize,
    page: u32,
    device: Device,
    queue: Queue,
    compressor: GpuBlockCompressor,
    texture: Texture,
    view: TextureView,
    blocks: Buffer,
    staging: Buffer,
}

impl Encoder {
    /// `page` is the padded page edge in pixels (a multiple of 4).
    pub fn new(page: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;
        let adapter_name = adapter.get_info().name;
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("earth-bake"),
            required_limits: limits.clone(),
            ..Default::default()
        }))?;
        let pages_per_batch = (limits.max_texture_dimension_2d / page) as usize;
        let page_bytes = CompressionVariant::BC7(BC7Settings::alpha_basic()).blocks_byte_size(page, page) as u64;
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("pages"),
            size: Extent3d { width: page, height: page * pages_per_batch as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let size = page_bytes * pages_per_batch as u64;
        let blocks = device.create_buffer(&BufferDescriptor {
            label: Some("blocks"),
            size,
            usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("staging"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let compressor = GpuBlockCompressor::new(device.clone(), queue.clone());
        Ok(Self { adapter_name, pages_per_batch, page, device, queue, compressor, texture, view, blocks, staging })
    }

    /// Encode `count` (<= pages_per_batch) RGBA8 pages stacked in `pixels`;
    /// returns their BC7 blocks, page after page.
    pub fn compress(&mut self, pixels: &[u8], count: usize) -> Result<Vec<u8>> {
        assert!(count <= self.pages_per_batch);
        let (page, height) = (self.page, self.page * count as u32);
        assert_eq!(pixels.len(), (page * height * 4) as usize);
        self.queue.write_texture(
            TexelCopyTextureInfo { texture: &self.texture, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            pixels,
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(page * 4), rows_per_image: Some(height) },
            Extent3d { width: page, height, depth_or_array_layers: 1 },
        );
        let variant = CompressionVariant::BC7(BC7Settings::alpha_basic());
        let bytes = variant.blocks_byte_size(page, height) as u64;
        self.compressor.add_compression_task(variant, &self.view, page, height, &self.blocks, None, None);
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("bc7") });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("bc7"), timestamp_writes: None });
            self.compressor.compress(&mut pass);
        }
        encoder.copy_buffer_to_buffer(&self.blocks, 0, &self.staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let slice = self.staging.slice(..bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| sender.send(result).unwrap());
        self.device.poll(PollType::Wait { submission_index: None, timeout: Some(Duration::from_secs(120)) })?;
        receiver.recv()??;
        let out = slice.get_mapped_range()?.to_vec();
        self.staging.unmap();
        Ok(out)
    }
}
