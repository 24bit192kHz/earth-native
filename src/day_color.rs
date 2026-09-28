//! Optional fixed-resolution day-colour previews for the stage-three renderer.
//!
//! The production path uses tiled `.earthvt` containers.  Until that path is
//! connected, the offline baker can export the two source hemispheres as small
//! raw BGRA8 previews.  Keeping this loader strict prevents an arbitrary image
//! file from silently becoming wallpaper input, while retaining the procedural
//! renderer as the fallback when both variables are unset.

use std::{
    env,
    ffi::OsString,
    fs::{self, File},
    path::{Path, PathBuf},
};

use memmap2::{Mmap, MmapOptions};

use crate::star_panorama::StarPanoramaMip;
use serde::Deserialize;
use thiserror::Error;

pub const DAY_COLOR_EAST_ENV: &str = "EARTH_NATIVE_DAY_COLOR_EAST";
pub const DAY_COLOR_WEST_ENV: &str = "EARTH_NATIVE_DAY_COLOR_WEST";
pub const NIGHT_EMISSION_ENV: &str = "EARTH_NATIVE_NIGHT_EMISSION";
pub const CLOUDS_A_ENV: &str = "EARTH_NATIVE_CLOUDS_A";
pub const CLOUDS_BA_ENV: &str = "EARTH_NATIVE_CLOUDS_BA";
pub const TILING_NOISE_ENV: &str = "EARTH_NATIVE_TILING_NOISE";
pub const DESERT_CLOUD_MASK_ENV: &str = "EARTH_NATIVE_DESERT_CLOUD_MASK";
pub const HEIGHT_ENV: &str = "EARTH_NATIVE_HEIGHT";
pub const SURFACE_NORMAL_EAST_ENV: &str = "EARTH_NATIVE_SURFACE_NORMAL_EAST";
pub const SURFACE_NORMAL_WEST_ENV: &str = "EARTH_NATIVE_SURFACE_NORMAL_WEST";
pub const MOON_ENV: &str = "EARTH_NATIVE_MOON";
pub const JUPITER_ENV: &str = "EARTH_NATIVE_JUPITER";
pub const MERCURY_ENV: &str = "EARTH_NATIVE_MERCURY";
pub const MARS_ENV: &str = "EARTH_NATIVE_MARS";
pub const SATURN_ENV: &str = "EARTH_NATIVE_SATURN";
pub const SATURN_RING_ENV: &str = "EARTH_NATIVE_SATURN_RING";
pub const VENUS_ENV: &str = "EARTH_NATIVE_VENUS";
pub const URANUS_ENV: &str = "EARTH_NATIVE_URANUS";
pub const NEPTUNE_ENV: &str = "EARTH_NATIVE_NEPTUNE";

const BYTES_PER_PIXEL: u64 = 4;
const SIDECAR_SCHEMA_VERSION: u32 = 1;

/// The two source hemispheres that make up the native 80K day-colour input.
pub struct DayColorHemispheres {
    east: PreviewTexture,
    west: PreviewTexture,
}

impl DayColorHemispheres {
    pub fn east(&self) -> &PreviewTexture {
        &self.east
    }

    pub fn west(&self) -> &PreviewTexture {
        &self.west
    }
}

/// The paired 80K surface-normal sources. They use the same virtual-page
/// layout as the day-colour hemispheres but are sampled as linear normal data.
pub struct SurfaceNormalHemispheres {
    east: PreviewTexture,
    west: PreviewTexture,
}

impl SurfaceNormalHemispheres {
    pub fn east(&self) -> &PreviewTexture {
        &self.east
    }

    pub fn west(&self) -> &PreviewTexture {
        &self.west
    }
}

/// A fixed-resolution preview of the reference night-emission control map.
///
/// The source's red and green channels are consumed by the Earth material;
/// it is not a display-ready night panorama. Keeping it separate from the
/// hemisphere pair makes that contract explicit at the renderer boundary.
pub struct NightEmissionPreview {
    texture: PreviewTexture,
}

impl NightEmissionPreview {
    pub fn texture(&self) -> &PreviewTexture {
        &self.texture
    }
}

/// The linear cloud-density and packed control sources exported from the
/// reference material.  Both files are required because the analytical cloud
/// shell combines them at the same surface coordinates.
pub struct CloudPreviews {
    a: PreviewTexture,
    ba: PreviewTexture,
}

impl CloudPreviews {
    pub fn a(&self) -> &PreviewTexture {
        &self.a
    }

    pub fn ba(&self) -> &PreviewTexture {
        &self.ba
    }
}

/// The small sRGB tiling-noise source sampled by the reference cloud material
/// to modulate each authored cloud layer's formation.
pub struct TilingNoisePreview {
    texture: PreviewTexture,
}

impl TilingNoisePreview {
    pub fn texture(&self) -> &PreviewTexture {
        &self.texture
    }
}

/// The linear terrain permission mask that prevents cloud opacity from
/// spilling uniformly across authored desert regions.
pub struct DesertCloudMaskPreview {
    texture: PreviewTexture,
}

impl DesertCloudMaskPreview {
    pub fn texture(&self) -> &PreviewTexture {
        &self.texture
    }
}

/// The linear packed R/G terrain-height control source used by the reference
/// cloud material's mountain permission branch.
pub struct HeightPreview {
    texture: PreviewTexture,
}

impl HeightPreview {
    pub fn texture(&self) -> &PreviewTexture {
        &self.texture
    }
}

/// Optional authored Moon albedo. Unlike the Earth material inputs, it is a
/// display-ready equirectangular texture used only by the far-away Moon pass.
pub struct MoonPreview {
    texture: PreviewTexture,
}

impl MoonPreview {
    pub fn texture(&self) -> &PreviewTexture {
        &self.texture
    }
}

/// Optional authored planet albedo. A display-ready 2:1 equirectangular map
/// rendered as the planet's surface when the body is switched to it. It is
/// independent of the Earth preview set so switching bodies never wedges the
/// renderer on an unrelated diagnostic material input.
pub struct PlanetPreview {
    texture: PreviewTexture,
}

impl PlanetPreview {
    pub fn texture(&self) -> &PreviewTexture {
        &self.texture
    }
}

/// Optional authored Saturn ring. A radial alpha map (not 2:1) sampled as a
/// flat annular ring around Saturn's equator.
pub struct RingPreview {
    texture: PreviewTexture,
}

impl RingPreview {
    pub fn texture(&self) -> &PreviewTexture {
        &self.texture
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreviewColorSpace {
    Srgb,
    Linear,
}

impl PreviewColorSpace {
    fn metadata_name(self) -> &'static str {
        match self {
            Self::Srgb => "srgb",
            Self::Linear => "linear",
        }
    }
}

/// A validated raw BGRA8 payload held open only until its Vulkan upload.
pub struct PreviewTexture {
    payload_path: PathBuf,
    file: File,
    width: u32,
    height: u32,
    payload_bytes: u64,
    equirectangular: bool,
    block_format: Option<BlockFormat>,
    mips: Vec<StarPanoramaMip>,
}

/// Offline block-compressed previews (NASA pipeline, `bcn.py`). Each carries
/// its full mip chain, so the renderer uploads blocks without a GPU mip bake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockFormat {
    Bc1,
    Bc3,
    Bc4,
    /// Two-channel (relief normals).
    Bc5,
}

impl BlockFormat {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "bc1" => Some(Self::Bc1),
            "bc3" => Some(Self::Bc3),
            "bc4" => Some(Self::Bc4),
            "bc5" => Some(Self::Bc5),
            _ => None,
        }
    }

    pub const fn bytes_per_block(self) -> u64 {
        match self {
            Self::Bc1 | Self::Bc4 => 8,
            Self::Bc3 | Self::Bc5 => 16,
        }
    }
}

impl std::fmt::Debug for PreviewTexture {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreviewTexture")
            .field("payload_path", &self.payload_path)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("payload_bytes", &self.payload_bytes)
            .finish_non_exhaustive()
    }
}

impl PreviewTexture {
    pub fn extent(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// `None` for raw BGRA8 previews.
    pub fn block_format(&self) -> Option<BlockFormat> {
        self.block_format
    }

    /// Precomputed mip levels of a block-compressed preview (empty for BGRA8).
    pub fn mips(&self) -> &[StarPanoramaMip] {
        &self.mips
    }

    pub fn map_payload(&self) -> Result<MappedPreviewTexture, DayColorError> {
        let metadata = self
            .file
            .metadata()
            .map_err(|source| DayColorError::PayloadMetadata {
                path: self.payload_path.clone(),
                source,
            })?;
        validate_file_length(&self.payload_path, metadata.len(), self.payload_bytes)?;
        let length = usize::try_from(self.payload_bytes)
            .map_err(|_| DayColorError::PayloadTooLarge(self.payload_bytes))?;
        let bytes =
            unsafe { MmapOptions::new().len(length).map(&self.file) }.map_err(|source| {
                DayColorError::MapPayload {
                    path: self.payload_path.clone(),
                    source,
                }
            })?;
        Ok(MappedPreviewTexture { bytes })
    }

    pub fn map_canonical_payload(&self, east: bool) -> Result<Vec<u8>, DayColorError> {
        let mapped = self.map_payload()?;
        if self.equirectangular {
            return Ok(mapped.bytes().to_vec());
        }
        let (width, height) = self.extent();
        let width = width as usize;
        let height = height as usize;
        let source = mapped.bytes();
        let mut bytes = vec![0; source.len()];
        for y in 0..height {
            let mesh_v = (y as f32 + 0.5) / height as f32;
            for x in 0..width {
                let local_u = (x as f32 + 0.5) / width as f32;
                let mesh_u = if east { 0.5 + local_u * 0.5 } else { local_u * 0.5 };
                let virtual_u = mesh_u * 10.0;
                let virtual_v = mesh_v * 5.0 + 1.0;
                let block_x = (virtual_u.floor() as usize) % 5;
                let block_y = (virtual_v.floor() as usize) % 5;
                let source_block_y = (5 - block_y) % 5;
                let source_x = (((block_x as f32 + virtual_u.fract()) / 5.0) * width as f32)
                    .floor()
                    .min((width - 1) as f32) as usize;
                let source_y = (((source_block_y as f32 + virtual_v.fract()) / 5.0) * height as f32)
                    .floor()
                    .min((height - 1) as f32) as usize;
                let source_offset = (source_y * width + source_x) * 4;
                let output_offset = (y * width + x) * 4;
                bytes[output_offset..output_offset + 4]
                    .copy_from_slice(&source[source_offset..source_offset + 4]);
            }
        }
        repair_canonical_page_edges(&mut bytes, width, height);
        Ok(bytes)
    }
}

fn repair_canonical_page_edges(bytes: &mut [u8], width: usize, height: usize) {
    const PAGE_COUNT: usize = 5;
    const CHANNEL_COUNT: usize = 4;
    if width == 0 || height < PAGE_COUNT || bytes.len() != width * height * CHANNEL_COUNT {
        return;
    }

    let original = bytes.to_vec();
    let depth = (height / PAGE_COUNT / 256).clamp(1, 2);
    for boundary in 1..PAGE_COUNT {
        let boundary_row = boundary * height / PAGE_COUNT;
        let current_reference_row = boundary_row - depth - 1;
        let next_reference_row = boundary_row + depth;
        for offset in 0..depth {
            let amount = (offset + 1) as u32;
            let divisor = (depth + 1) as u32;
            let current_row = boundary_row - 1 - offset;
            let next_row = boundary_row + offset;
            for x in 0..width {
                for channel in 0..CHANNEL_COUNT {
                    let current_reference = original[(current_reference_row * width + x) * CHANNEL_COUNT + channel] as u32;
                    let next_reference = original[(next_reference_row * width + x) * CHANNEL_COUNT + channel] as u32;
                    let midpoint = (current_reference + next_reference + 1) / 2;
                    let current_value = (midpoint * (divisor - amount) + current_reference * amount + divisor / 2) / divisor;
                    let next_value = (midpoint * (divisor - amount) + next_reference * amount + divisor / 2) / divisor;
                    bytes[(current_row * width + x) * CHANNEL_COUNT + channel] = current_value as u8;
                    bytes[(next_row * width + x) * CHANNEL_COUNT + channel] = next_value as u8;
                }
            }
        }
    }
}

pub struct MappedPreviewTexture {
    bytes: Mmap,
}

impl MappedPreviewTexture {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Load no preview when both variables are absent.  Supplying just one
/// hemisphere is an error because it would create a seam at the date line.
pub fn load_optional_day_color_hemispheres() -> Result<Option<DayColorHemispheres>, DayColorError> {
    let east = env::var_os(DAY_COLOR_EAST_ENV);
    let west = env::var_os(DAY_COLOR_WEST_ENV);
    match (east, west) {
        (None, None) => Ok(None),
        (Some(east), Some(west)) => {
            let east = load_preview_texture(DAY_COLOR_EAST_ENV, east, PreviewColorSpace::Srgb)?;
            let west = load_preview_texture(DAY_COLOR_WEST_ENV, west, PreviewColorSpace::Srgb)?;
            if east.extent() != west.extent() {
                return Err(DayColorError::MismatchedExtent {
                    east: east.extent(),
                    west: west.extent(),
                });
            }
            Ok(Some(DayColorHemispheres { east, west }))
        }
        _ => Err(DayColorError::IncompleteHemispherePair),
    }
}

/// Load both optional linear surface-normal hemispheres, or neither. They
/// deliberately share the day-colour pair's all-or-nothing seam contract.
pub fn load_optional_surface_normal_hemispheres(
    day_colours_present: bool,
) -> Result<Option<SurfaceNormalHemispheres>, DayColorError> {
    let east = env::var_os(SURFACE_NORMAL_EAST_ENV);
    let west = env::var_os(SURFACE_NORMAL_WEST_ENV);
    match (east, west) {
        (None, None) => Ok(None),
        (Some(east), Some(west)) => {
            if !day_colours_present {
                return Err(DayColorError::SurfaceNormalsWithoutDayColour);
            }
            let east = load_preview_texture(
                SURFACE_NORMAL_EAST_ENV,
                east,
                PreviewColorSpace::Linear,
            )?;
            let west = load_preview_texture(
                SURFACE_NORMAL_WEST_ENV,
                west,
                PreviewColorSpace::Linear,
            )?;
            if east.extent() != west.extent() {
                return Err(DayColorError::MismatchedSurfaceNormalExtent {
                    east: east.extent(),
                    west: west.extent(),
                });
            }
            Ok(Some(SurfaceNormalHemispheres { east, west }))
        }
        _ => Err(DayColorError::IncompleteSurfaceNormalPair),
    }
}

/// Load the optional sRGB night-emission preview.  It is only meaningful with
/// the day-colour pair, so reject a partial material configuration instead of
/// silently presenting a misleading night-only Earth.
pub fn load_optional_night_emission(
    day_colours_present: bool,
) -> Result<Option<NightEmissionPreview>, DayColorError> {
    let Some(payload) = env::var_os(NIGHT_EMISSION_ENV) else {
        return Ok(None);
    };
    if !day_colours_present {
        return Err(DayColorError::NightEmissionWithoutDayColour);
    }
    Ok(Some(NightEmissionPreview {
        texture: load_preview_texture(NIGHT_EMISSION_ENV, payload, PreviewColorSpace::Srgb)?,
    }))
}

/// Load both linear cloud-source previews, or neither.  A partial set would
/// make the analytical shell react differently to the same reference asset
/// configuration, so it is rejected before Vulkan resource creation.
pub fn load_optional_cloud_previews(
    day_colours_present: bool,
) -> Result<Option<CloudPreviews>, DayColorError> {
    let a = env::var_os(CLOUDS_A_ENV);
    let ba = env::var_os(CLOUDS_BA_ENV);
    match (a, ba) {
        (None, None) => Ok(None),
        (Some(a), Some(ba)) => {
            if !day_colours_present {
                return Err(DayColorError::CloudPreviewsWithoutDayColour);
            }
            let a = load_preview_texture(CLOUDS_A_ENV, a, PreviewColorSpace::Linear)?;
            let ba = load_preview_texture(CLOUDS_BA_ENV, ba, PreviewColorSpace::Linear)?;
            if a.block_format().is_none() && a.extent() != ba.extent() {
                return Err(DayColorError::MismatchedCloudExtent {
                    a: a.extent(),
                    ba: ba.extent(),
                });
            }
            Ok(Some(CloudPreviews { a, ba }))
        }
        _ => Err(DayColorError::IncompleteCloudPair),
    }
}

/// Load the optional reference tiling noise only with the cloud-source pair.
/// When it is absent, Vulkan binds a white fallback so the preview material
/// remains usable while preserving the unmodulated cloud-layer contract.
pub fn load_optional_tiling_noise(
    cloud_previews_present: bool,
) -> Result<Option<TilingNoisePreview>, DayColorError> {
    let Some(payload) = env::var_os(TILING_NOISE_ENV) else {
        return Ok(None);
    };
    if !cloud_previews_present {
        return Err(DayColorError::TilingNoiseWithoutCloudPreviews);
    }
    Ok(Some(TilingNoisePreview {
        texture: load_preview_texture(TILING_NOISE_ENV, payload, PreviewColorSpace::Srgb)?,
    }))
}

/// Load the linear desert-cloud permission map only with the cloud previews.
/// The shader binds a black fallback when this diagnostic export is absent so
/// existing preview sessions retain their previous cloud-coverage behavior.
pub fn load_optional_desert_cloud_mask(
    cloud_previews_present: bool,
) -> Result<Option<DesertCloudMaskPreview>, DayColorError> {
    let Some(payload) = env::var_os(DESERT_CLOUD_MASK_ENV) else {
        return Ok(None);
    };
    if !cloud_previews_present {
        return Err(DayColorError::DesertCloudMaskWithoutCloudPreviews);
    }
    Ok(Some(DesertCloudMaskPreview {
        texture: load_preview_texture(
            DESERT_CLOUD_MASK_ENV,
            payload,
            PreviewColorSpace::Linear,
        )?,
    }))
}

/// Load the packed terrain-height preview only with the cloud previews. The
/// same asset later feeds surface parallax, but this first consumer is the
/// cloud terrain permission branch.
pub fn load_optional_height_preview(
    cloud_previews_present: bool,
) -> Result<Option<HeightPreview>, DayColorError> {
    let Some(payload) = env::var_os(HEIGHT_ENV) else {
        return Ok(None);
    };
    if !cloud_previews_present {
        return Err(DayColorError::HeightWithoutCloudPreviews);
    }
    Ok(Some(HeightPreview {
        texture: load_preview_texture(HEIGHT_ENV, payload, PreviewColorSpace::Linear)?,
    }))
}

/// Load the authored Moon albedo when supplied. It is independent of the
/// Earth preview set so a celestial body cannot silently disappear because an
/// unrelated diagnostic material input is omitted.
pub fn load_optional_moon_preview() -> Result<Option<MoonPreview>, DayColorError> {
    let Some(payload) = env::var_os(MOON_ENV) else {
        return Ok(None);
    };
    let texture = load_preview_texture(MOON_ENV, payload, PreviewColorSpace::Srgb)?;
    let (width, height) = texture.extent();
    if width != height.saturating_mul(2) {
        return Err(DayColorError::MoonNotTwoToOne { width, height });
    }
    Ok(Some(MoonPreview { texture }))
}

/// Load an authored planet albedo (2:1 equirectangular) when supplied.
/// Independent of the Earth preview set, like the Moon, so a planet never
/// depends on an unrelated Earth diagnostic input.
fn load_optional_planet(
    variable: &'static str,
    payload: Option<OsString>,
) -> Result<Option<PlanetPreview>, DayColorError> {
    let Some(payload) = payload else {
        return Ok(None);
    };
    let texture = load_preview_texture(variable, payload, PreviewColorSpace::Srgb)?;
    let (width, height) = texture.extent();
    if width != height.saturating_mul(2) {
        return Err(DayColorError::MoonNotTwoToOne { width, height });
    }
    Ok(Some(PlanetPreview { texture }))
}

/// Resident-on-demand albedo maps for every non-Earth body, indexed by
/// `Body::id()`. Each is optional; a missing map renders as a black globe.
#[derive(Default)]
pub struct PlanetPreviews([Option<PlanetPreview>; crate::body::Body::COUNT]);

impl PlanetPreviews {
    pub fn get(&self, body: crate::body::Body) -> Option<&PlanetPreview> {
        self.0[body.id() as usize].as_ref()
    }

    pub fn any(&self) -> bool {
        self.0.iter().any(Option::is_some)
    }
}

/// Load every supplied body albedo (`EARTH_NATIVE_<BODY>`; the Moon body
/// reuses the sky Moon's `EARTH_NATIVE_MOON`).
pub fn load_optional_planet_previews() -> Result<PlanetPreviews, DayColorError> {
    use crate::body::Body;
    let mut previews = PlanetPreviews::default();
    for (body, variable) in [
        (Body::Jupiter, JUPITER_ENV),
        (Body::Mercury, MERCURY_ENV),
        (Body::Mars, MARS_ENV),
        (Body::Saturn, SATURN_ENV),
        (Body::Venus, VENUS_ENV),
        (Body::Uranus, URANUS_ENV),
        (Body::Neptune, NEPTUNE_ENV),
        (Body::Moon, MOON_ENV),
    ] {
        previews.0[body.id() as usize] = load_optional_planet(variable, env::var_os(variable))?;
    }
    Ok(previews)
}

/// Load the authored Saturn ring alpha map when supplied. Unlike the planet
/// albedos this is a radial strip, not a 2:1 equirectangular map.
pub fn load_optional_saturn_ring_preview() -> Result<Option<RingPreview>, DayColorError> {
    let Some(payload) = env::var_os(SATURN_RING_ENV) else {
        return Ok(None);
    };
    let texture = load_preview_texture(SATURN_RING_ENV, payload, PreviewColorSpace::Srgb)?;
    Ok(Some(RingPreview { texture }))
}

pub(crate) fn load_preview_texture(
    variable: &'static str,
    payload: OsString,
    color_space: PreviewColorSpace,
) -> Result<PreviewTexture, DayColorError> {
    if payload.is_empty() {
        return Err(DayColorError::EmptyPath(variable));
    }
    let payload_path = PathBuf::from(payload);
    let metadata_path = default_sidecar_path(&payload_path);
    let metadata_bytes =
        fs::read(&metadata_path).map_err(|source| DayColorError::ReadMetadata {
            path: metadata_path.clone(),
            source,
        })?;
    let metadata =
        serde_json::from_slice::<PreviewMetadata>(&metadata_bytes).map_err(|source| {
            DayColorError::InvalidMetadata {
                path: metadata_path.clone(),
                source,
            }
        })?;
    let mips = validate_metadata(&metadata, &metadata_path, color_space)?;
    let file = File::open(&payload_path).map_err(|source| DayColorError::OpenPayload {
        path: payload_path.clone(),
        source,
    })?;
    let file_metadata = file
        .metadata()
        .map_err(|source| DayColorError::PayloadMetadata {
            path: payload_path.clone(),
            source,
        })?;
    if !file_metadata.is_file() {
        return Err(DayColorError::NotAFile { path: payload_path });
    }
    validate_file_length(&payload_path, file_metadata.len(), metadata.payload_bytes)?;
    Ok(PreviewTexture {
        payload_path,
        file,
        width: metadata.width,
        height: metadata.height,
        payload_bytes: metadata.payload_bytes,
        equirectangular: metadata.layout.as_deref() == Some("equirectangular"),
        block_format: BlockFormat::parse(&metadata.pixel_format),
        mips,
    })
}

fn default_sidecar_path(payload_path: &Path) -> PathBuf {
    let mut value = payload_path.as_os_str().to_owned();
    value.push(".json");
    PathBuf::from(value)
}

fn validate_file_length(path: &Path, actual: u64, expected: u64) -> Result<(), DayColorError> {
    if actual == expected {
        Ok(())
    } else {
        Err(DayColorError::ByteCount {
            path: path.to_owned(),
            expected,
            actual,
        })
    }
}

fn validate_metadata(
    metadata: &PreviewMetadata,
    path: &Path,
    color_space: PreviewColorSpace,
) -> Result<Vec<StarPanoramaMip>, DayColorError> {
    let unsupported = || DayColorError::UnsupportedMetadata {
        path: path.to_owned(),
    };
    let block_format = BlockFormat::parse(&metadata.pixel_format);
    if metadata.layout.as_deref().is_some_and(|layout| layout != "equirectangular")
        || metadata.schema_version != SIDECAR_SCHEMA_VERSION
        || metadata.asset != "earth_native_preview_texture"
        || (metadata.pixel_format != "bgra8" && block_format.is_none())
        || metadata.color_space != color_space.metadata_name()
        || metadata.origin != "top_left"
        || metadata.width == 0
        || (color_space == PreviewColorSpace::Srgb
            && metadata._rgb_filter.as_deref() != Some("linear-light-srgb-v1"))
        || metadata.height == 0
    {
        return Err(unsupported());
    }
    if let Some(format) = block_format {
        // The 5x5 page remap in map_canonical_payload works on raw texels only.
        if metadata.layout.as_deref() != Some("equirectangular") {
            return Err(unsupported());
        }
        return validate_block_mips(metadata, format).ok_or_else(unsupported);
    }
    if metadata.mips.is_some() {
        return Err(unsupported());
    }
    let row_stride = u64::from(metadata.width)
        .checked_mul(BYTES_PER_PIXEL)
        .ok_or_else(|| DayColorError::UnsupportedMetadata {
            path: path.to_owned(),
        })?;
    let payload_bytes = row_stride
        .checked_mul(u64::from(metadata.height))
        .ok_or_else(|| DayColorError::UnsupportedMetadata {
            path: path.to_owned(),
        })?;
    if metadata.row_stride_bytes != row_stride || metadata.payload_bytes != payload_bytes {
        return Err(DayColorError::UnsupportedMetadata {
            path: path.to_owned(),
        });
    }
    Ok(Vec::new())
}

/// Mips must be whole 4x4 blocks, successive halvings of the base extent, and
/// tightly concatenated so the declared payload is exactly their sum.
fn validate_block_mips(metadata: &PreviewMetadata, format: BlockFormat) -> Option<Vec<StarPanoramaMip>> {
    let mips = metadata.mips.as_ref()?;
    if mips.is_empty() || mips.len() > 16 {
        return None;
    }
    let mut offset = 0u64;
    let (mut width, mut height) = (metadata.width, metadata.height);
    let mut layout = Vec::with_capacity(mips.len());
    for mip in mips {
        if mip.width != width || mip.height != height || width % 4 != 0 || height % 4 != 0
            || mip.byte_offset != offset
        {
            return None;
        }
        layout.push(StarPanoramaMip { byte_offset: offset, width, height });
        let blocks = u64::from(width / 4) * u64::from(height / 4);
        offset = offset.checked_add(blocks.checked_mul(format.bytes_per_block())?)?;
        width /= 2;
        height /= 2;
    }
    (offset == metadata.payload_bytes).then_some(layout)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MipMetadata {
    byte_offset: u64,
    width: u32,
    height: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PreviewMetadata {
    #[serde(default)]
    layout: Option<String>,
    schema_version: u32,
    asset: String,
    width: u32,
    height: u32,
    row_stride_bytes: u64,
    payload_bytes: u64,
    pixel_format: String,
    color_space: String,
    origin: String,
    #[serde(rename = "source_object_path")]
    _source_object_path: String,
    #[serde(rename = "source_width")]
    _source_width: u32,
    #[serde(rename = "source_height")]
    _source_height: u32,
    #[serde(rename = "source_blocks")]
    _source_blocks: u32,
    #[serde(rename = "filter")]
    _filter: String,
    #[serde(rename = "rgb_filter", default)]
    _rgb_filter: Option<String>,
    #[serde(default)]
    mips: Option<Vec<MipMetadata>>,
}

#[derive(Debug, Error)]
pub enum DayColorError {
    #[error("{0} cannot be empty")]
    EmptyPath(&'static str),
    #[error("set both {DAY_COLOR_EAST_ENV} and {DAY_COLOR_WEST_ENV}, or neither")]
    IncompleteHemispherePair,
    #[error("set both {SURFACE_NORMAL_EAST_ENV} and {SURFACE_NORMAL_WEST_ENV}, or neither")]
    IncompleteSurfaceNormalPair,
    #[error("set {SURFACE_NORMAL_EAST_ENV} and {SURFACE_NORMAL_WEST_ENV} only together with {DAY_COLOR_EAST_ENV} and {DAY_COLOR_WEST_ENV}")]
    SurfaceNormalsWithoutDayColour,
    #[error("set {NIGHT_EMISSION_ENV} only together with {DAY_COLOR_EAST_ENV} and {DAY_COLOR_WEST_ENV}")]
    NightEmissionWithoutDayColour,
    #[error("set both {CLOUDS_A_ENV} and {CLOUDS_BA_ENV}, or neither")]
    IncompleteCloudPair,
    #[error("set {CLOUDS_A_ENV} and {CLOUDS_BA_ENV} only together with {DAY_COLOR_EAST_ENV} and {DAY_COLOR_WEST_ENV}")]
    CloudPreviewsWithoutDayColour,
    #[error("set {TILING_NOISE_ENV} only together with {CLOUDS_A_ENV} and {CLOUDS_BA_ENV}")]
    TilingNoiseWithoutCloudPreviews,
    #[error("set {DESERT_CLOUD_MASK_ENV} only together with {CLOUDS_A_ENV} and {CLOUDS_BA_ENV}")]
    DesertCloudMaskWithoutCloudPreviews,
    #[error("set {HEIGHT_ENV} only together with {CLOUDS_A_ENV} and {CLOUDS_BA_ENV}")]
    HeightWithoutCloudPreviews,
    #[error("day-colour hemisphere previews must have equal dimensions; east is {east:?}, west is {west:?}")]
    MismatchedExtent { east: (u32, u32), west: (u32, u32) },
    #[error("surface-normal hemisphere previews must have equal dimensions; east is {east:?}, west is {west:?}")]
    MismatchedSurfaceNormalExtent { east: (u32, u32), west: (u32, u32) },
    #[error("cloud previews must have equal dimensions; A is {a:?}, BA is {ba:?}")]
    MismatchedCloudExtent { a: (u32, u32), ba: (u32, u32) },
    #[error("Moon preview must be a 2:1 equirectangular map; got {width}x{height}")]
    MoonNotTwoToOne { width: u32, height: u32 },
    #[error("could not read day-colour metadata {path}: {source}")]
    ReadMetadata {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("day-colour metadata {path} is invalid JSON: {source}")]
    InvalidMetadata {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("preview metadata {path} is not a supported BGRA8/BC1/BC3/BC4 source preview; sRGB previews must declare rgb_filter=linear-light-srgb-v1")]
    UnsupportedMetadata { path: PathBuf },
    #[error("could not open day-colour payload {path}: {source}")]
    OpenPayload {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not read metadata for day-colour payload {path}: {source}")]
    PayloadMetadata {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("day-colour payload {path} is not a regular file")]
    NotAFile { path: PathBuf },
    #[error("day-colour payload {path} has {actual} bytes; expected {expected}")]
    ByteCount {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("day-colour payload is too large to map: {0} bytes")]
    PayloadTooLarge(u64),
    #[error("could not map day-colour payload {path}: {source}")]
    MapPayload {
        path: PathBuf,
        source: std::io::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::{
        default_sidecar_path, repair_canonical_page_edges, validate_metadata, PreviewColorSpace,
        PreviewMetadata,
    };
    use std::path::Path;

    fn metadata(color_space: &str, rgb_filter: Option<&str>) -> PreviewMetadata {
        PreviewMetadata {
            layout: None,
            schema_version: 1,
            asset: "earth_native_preview_texture".to_owned(),
            width: 2,
            height: 1,
            row_stride_bytes: 8,
            payload_bytes: 8,
            pixel_format: "bgra8".to_owned(),
            color_space: color_space.to_owned(),
            origin: "top_left".to_owned(),
            _source_object_path: "/Game/Test.Texture".to_owned(),
            _source_width: 2,
            _source_height: 1,
            _source_blocks: 1,
            _filter: "test".to_owned(),
            _rgb_filter: rgb_filter.map(str::to_owned),
            mips: None,
        }
    }

    #[test]
    fn nasa_layout_is_explicit_and_unknown_layouts_are_rejected() {
        let mut source = metadata("linear", None);
        source.layout = Some("equirectangular".to_owned());
        assert!(validate_metadata(&source, Path::new("nasa.json"), PreviewColorSpace::Linear).is_ok());
        source.layout = Some("unknown-projection".to_owned());
        assert!(validate_metadata(&source, Path::new("nasa.json"), PreviewColorSpace::Linear).is_err());
    }

    #[test]
    fn sidecar_is_appended_to_raw_payload_path() {
        assert_eq!(
            default_sidecar_path(Path::new("preview.bgra")),
            Path::new("preview.bgra.json")
        );
    }

    #[test]
    fn corrected_srgb_filter_is_required() {
        let path = Path::new("preview.bgra.json");
        assert!(validate_metadata(
            &metadata("srgb", Some("linear-light-srgb-v1")),
            path,
            PreviewColorSpace::Srgb,
        )
        .is_ok());
        assert!(validate_metadata(
            &metadata("srgb", None),
            path,
            PreviewColorSpace::Srgb,
        )
        .is_err());
    }

    #[test]
    fn legacy_linear_preview_remains_valid() {
        assert!(validate_metadata(
            &metadata("linear", None),
            Path::new("preview.bgra.json"),
            PreviewColorSpace::Linear,
        )
        .is_ok());
    }

    #[test]
    fn canonical_page_edges_are_stitched_without_extra_runtime_samples() {
        let mut bytes = Vec::new();
        for row in 0..10_u8 {
            bytes.extend_from_slice(&[if row < 2 { 0 } else { 100 }; 4]);
        }

        repair_canonical_page_edges(&mut bytes, 1, 10);

        assert_eq!(bytes[4], 25);
        assert_eq!(bytes[8], 75);
    }
}
