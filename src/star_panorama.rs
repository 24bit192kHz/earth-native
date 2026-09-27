//! Direct, GPU-ready input for the pinned star panorama.
//!
//! The payload is a tightly packed raw `bgra8` file or concatenated BC1 mip
//! chain. It is deliberately not an image format: the runtime does no image
//! decoding or color conversion before staging it into a Vulkan image. A JSON
//! sidecar is required so an Unreal export can state the byte layout and
//! equirectangular convention without relying on renderer-specific environment
//! dimensions.
//!
//! Set `EARTH_NATIVE_STAR_PANORAMA` to the raw payload path. By default its
//! sidecar is the same path with `.json` appended, for example
//! `stars.bgra` and `stars.bgra.json`. Set
//! `EARTH_NATIVE_STAR_PANORAMA_METADATA` only when the sidecar has another
//! name. No panorama configuration, or a missing optional payload/sidecar,
//! selects the procedural fallback.
//!
//! Sidecar schema version 1:
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "asset": "star_panorama",
//!   "width": 16384,
//!   "height": 8192,
//!   "row_stride_bytes": 65536,
//!   "payload_bytes": 536870912,
//!   "pixel_format": "bgra8",
//!   "mip_count": 1,
//!   "color_space": "srgb",
//!   "projection": "equirectangular",
//!   "origin": "top_left",
//!   "longitude_at_u0_degrees": -180,
//!   "latitude_at_v0_degrees": 90
//! }
//! ```
//!
//! Pixel `(0, 0)` is the top-left pixel, U wraps at the antimeridian, and the
//! image center is longitude zero. The sidecar intentionally has no Display-P3
//! or implicit-color-management mode: export the exact sRGB bytes consumed by
//! the Vulkan sRGB image. Unreal's native BGRA8 source order is preserved all
//! the way through the transfer; the Vulkan image format supplies the channel
//! interpretation without a CPU conversion.

use std::{
    env,
    ffi::OsString,
    fs::{self, File},
    path::{Path, PathBuf},
};

use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;
use thiserror::Error;

pub const STAR_PANORAMA_ENV: &str = "EARTH_NATIVE_STAR_PANORAMA";
pub const STAR_PANORAMA_METADATA_ENV: &str = "EARTH_NATIVE_STAR_PANORAMA_METADATA";
const SIDECAR_SCHEMA_VERSION: u32 = 1;
const BYTES_PER_PIXEL: u64 = 4;
const BC1_BYTES_PER_BLOCK: u64 = 8;

/// The validated sRGB GPU texture format of a star panorama.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StarPanoramaFormat {
    Bgra8Srgb,
    Bc1Srgb,
}

/// The dimensions and payload offset of one tightly packed mip level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StarPanoramaMip {
    pub byte_offset: u64,
    pub width: u32,
    pub height: u32,
}

/// A validated raw panorama file held open until its one-time Vulkan upload.
///
/// Keeping the descriptor and file metadata rather than a `Vec<u8>` means a
/// native 16K source does not remain resident in CPU memory after staging.
pub struct StarPanorama {
    payload_path: PathBuf,
    metadata_path: PathBuf,
    file: File,
    width: u32,
    height: u32,
    payload_bytes: u64,
    format: StarPanoramaFormat,
    mips: Vec<StarPanoramaMip>,
}

impl std::fmt::Debug for StarPanorama {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StarPanorama")
            .field("payload_path", &self.payload_path)
            .field("metadata_path", &self.metadata_path)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("payload_bytes", &self.payload_bytes)
            .field("format", &self.format)
            .field("mips", &self.mips)
            .finish_non_exhaustive()
    }
}

impl StarPanorama {
    pub fn extent(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn format(&self) -> StarPanoramaFormat {
        self.format
    }

    pub fn mips(&self) -> &[StarPanoramaMip] {
        &self.mips
    }

    #[cfg(test)]
    pub fn payload_path(&self) -> &Path {
        &self.payload_path
    }

    #[cfg(test)]
    pub fn metadata_path(&self) -> &Path {
        &self.metadata_path
    }

    /// Map the exact validated payload for the immediate staging copy.
    ///
    /// The returned mapping must not outlive the upload copy. It is not retained
    /// by the renderer after the device-local image has been populated.
    pub fn map_payload(&self) -> Result<MappedStarPanorama, StarPanoramaError> {
        let metadata =
            self.file
                .metadata()
                .map_err(|source| StarPanoramaError::PayloadMetadata {
                    path: self.payload_path.clone(),
                    source,
                })?;
        validate_file_length(&self.payload_path, metadata.len(), self.payload_bytes)?;
        let length = usize::try_from(self.payload_bytes)
            .map_err(|_| StarPanoramaError::PayloadTooLarge(self.payload_bytes))?;
        let bytes =
            unsafe { MmapOptions::new().len(length).map(&self.file) }.map_err(|source| {
                StarPanoramaError::MapPayload {
                    path: self.payload_path.clone(),
                    source,
                }
            })?;
        if bytes.len() != length {
            return Err(StarPanoramaError::ByteCount {
                path: self.payload_path.clone(),
                expected: self.payload_bytes,
                actual: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            });
        }
        Ok(MappedStarPanorama { bytes })
    }
}

/// A short-lived, read-only mapping used for the direct Vulkan staging copy.
pub struct MappedStarPanorama {
    bytes: Mmap,
}

impl MappedStarPanorama {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Load an optional production panorama. The renderer remains usable without
/// assets while the offline export has not generated the raw payload yet.
pub fn load_optional_star_panorama() -> Result<Option<StarPanorama>, StarPanoramaError> {
    let Some(payload_path) = env::var_os(STAR_PANORAMA_ENV) else {
        return Ok(None);
    };
    if payload_path.is_empty() {
        return Err(StarPanoramaError::EmptyPath(STAR_PANORAMA_ENV));
    }
    let payload_path = PathBuf::from(payload_path);
    let metadata_path = match env::var_os(STAR_PANORAMA_METADATA_ENV) {
        Some(path) if path.is_empty() => {
            return Err(StarPanoramaError::EmptyPath(STAR_PANORAMA_METADATA_ENV))
        }
        Some(path) => PathBuf::from(path),
        None => default_sidecar_path(&payload_path),
    };
    load_optional_from_paths(&payload_path, &metadata_path)
}

fn load_optional_from_paths(
    payload_path: &Path,
    metadata_path: &Path,
) -> Result<Option<StarPanorama>, StarPanoramaError> {
    match load_from_paths(payload_path, metadata_path) {
        Ok(panorama) => Ok(Some(panorama)),
        Err(error) if error.is_missing_optional_asset() => Ok(None),
        Err(error) => Err(error),
    }
}

fn load_from_paths(
    payload_path: &Path,
    metadata_path: &Path,
) -> Result<StarPanorama, StarPanoramaError> {
    let sidecar_bytes = match fs::read(metadata_path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Err(StarPanoramaError::MissingAsset {
                path: metadata_path.to_owned(),
            })
        }
        Err(source) => {
            return Err(StarPanoramaError::ReadMetadata {
                path: metadata_path.to_owned(),
                source,
            })
        }
    };
    let metadata =
        serde_json::from_slice::<StarPanoramaMetadata>(&sidecar_bytes).map_err(|source| {
            StarPanoramaError::InvalidMetadata {
                path: metadata_path.to_owned(),
                source,
            }
        })?;
    let layout = validate_metadata(metadata, metadata_path)?;
    let file = match File::open(payload_path) {
        Ok(file) => file,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Err(StarPanoramaError::MissingAsset {
                path: payload_path.to_owned(),
            })
        }
        Err(source) => {
            return Err(StarPanoramaError::OpenPayload {
                path: payload_path.to_owned(),
                source,
            })
        }
    };
    let payload_metadata =
        file.metadata()
            .map_err(|source| StarPanoramaError::PayloadMetadata {
                path: payload_path.to_owned(),
                source,
            })?;
    if !payload_metadata.is_file() {
        return Err(StarPanoramaError::NotAFile {
            path: payload_path.to_owned(),
        });
    }
    validate_file_length(payload_path, payload_metadata.len(), layout.payload_bytes)?;
    Ok(StarPanorama {
        payload_path: payload_path.to_owned(),
        metadata_path: metadata_path.to_owned(),
        file,
        width: layout.width,
        height: layout.height,
        payload_bytes: layout.payload_bytes,
        format: layout.format,
        mips: layout.mips,
    })
}

fn default_sidecar_path(payload_path: &Path) -> PathBuf {
    let mut value: OsString = payload_path.as_os_str().to_owned();
    value.push(".json");
    PathBuf::from(value)
}

fn validate_file_length(path: &Path, actual: u64, expected: u64) -> Result<(), StarPanoramaError> {
    if actual == expected {
        Ok(())
    } else {
        Err(StarPanoramaError::ByteCount {
            path: path.to_owned(),
            expected,
            actual,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StarPanoramaMetadata {
    schema_version: u32,
    #[serde(rename = "asset")]
    _asset: AssetKind,
    width: u32,
    height: u32,
    row_stride_bytes: u64,
    payload_bytes: u64,
    pixel_format: PixelFormat,
    #[serde(default = "default_mip_count")]
    mip_count: u32,
    #[serde(rename = "color_space")]
    _color_space: ColorSpace,
    #[serde(rename = "projection")]
    _projection: Projection,
    #[serde(rename = "origin")]
    _origin: Origin,
    longitude_at_u0_degrees: i16,
    latitude_at_v0_degrees: i16,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AssetKind {
    StarPanorama,
}

#[derive(Debug, Deserialize)]
enum PixelFormat {
    #[serde(rename = "bgra8")]
    Bgra8,
    #[serde(rename = "bc1")]
    Bc1,
}

fn default_mip_count() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
enum ColorSpace {
    #[serde(rename = "srgb")]
    Srgb,
}

#[derive(Debug, Deserialize)]
enum Projection {
    #[serde(rename = "equirectangular")]
    Equirectangular,
}

#[derive(Debug, Deserialize)]
enum Origin {
    #[serde(rename = "top_left")]
    TopLeft,
}

struct ValidatedLayout {
    width: u32,
    height: u32,
    payload_bytes: u64,
    format: StarPanoramaFormat,
    mips: Vec<StarPanoramaMip>,
}

fn validate_metadata(
    metadata: StarPanoramaMetadata,
    metadata_path: &Path,
) -> Result<ValidatedLayout, StarPanoramaError> {
    let StarPanoramaMetadata {
        schema_version,
        _asset: _,
        width,
        height,
        row_stride_bytes,
        payload_bytes,
        pixel_format,
        mip_count,
        _color_space: _,
        _projection: _,
        _origin: _,
        longitude_at_u0_degrees,
        latitude_at_v0_degrees,
    } = metadata;
    if schema_version != SIDECAR_SCHEMA_VERSION {
        return Err(StarPanoramaError::UnsupportedSchemaVersion {
            path: metadata_path.to_owned(),
            found: schema_version,
        });
    }
    if width == 0 || height == 0 {
        return Err(StarPanoramaError::ZeroExtent {
            path: metadata_path.to_owned(),
        });
    }
    if u64::from(width) != u64::from(height) * 2 {
        return Err(StarPanoramaError::NotTwoToOne {
            path: metadata_path.to_owned(),
            width,
            height,
        });
    }
    let (format, expected_stride, expected_payload_bytes, mips) = match pixel_format {
        PixelFormat::Bgra8 => {
            if mip_count != 1 {
                return Err(StarPanoramaError::MipCount {
                    path: metadata_path.to_owned(),
                    format: StarPanoramaFormat::Bgra8Srgb,
                    found: mip_count,
                    maximum: 1,
                });
            }
            let stride = u64::from(width)
                .checked_mul(BYTES_PER_PIXEL)
                .ok_or_else(|| StarPanoramaError::LayoutOverflow {
                    path: metadata_path.to_owned(),
                })?;
            let payload = stride.checked_mul(u64::from(height)).ok_or_else(|| {
                StarPanoramaError::LayoutOverflow {
                    path: metadata_path.to_owned(),
                }
            })?;
            (
                StarPanoramaFormat::Bgra8Srgb,
                stride,
                payload,
                vec![StarPanoramaMip {
                    byte_offset: 0,
                    width,
                    height,
                }],
            )
        }
        PixelFormat::Bc1 => {
            if width % 4 != 0 || height % 4 != 0 {
                return Err(StarPanoramaError::Bc1BaseExtent {
                    path: metadata_path.to_owned(),
                    width,
                    height,
                });
            }
            let full_mip_count = u32::BITS - width.max(height).leading_zeros();
            if mip_count == 0 || mip_count > full_mip_count {
                return Err(StarPanoramaError::MipCount {
                    path: metadata_path.to_owned(),
                    format: StarPanoramaFormat::Bc1Srgb,
                    found: mip_count,
                    maximum: full_mip_count,
                });
            }
            let stride = u64::from(width.div_ceil(4)) * BC1_BYTES_PER_BLOCK;
            let mut mips = Vec::with_capacity(mip_count as usize);
            let mut mip_width = width;
            let mut mip_height = height;
            let mut offset = 0_u64;
            for _ in 0..mip_count {
                mips.push(StarPanoramaMip {
                    byte_offset: offset,
                    width: mip_width,
                    height: mip_height,
                });
                let mip_bytes = u64::from(mip_width.div_ceil(4))
                    .checked_mul(u64::from(mip_height.div_ceil(4)))
                    .and_then(|blocks| blocks.checked_mul(BC1_BYTES_PER_BLOCK))
                    .ok_or_else(|| StarPanoramaError::LayoutOverflow {
                        path: metadata_path.to_owned(),
                    })?;
                offset = offset.checked_add(mip_bytes).ok_or_else(|| {
                    StarPanoramaError::LayoutOverflow {
                        path: metadata_path.to_owned(),
                    }
                })?;
                mip_width = (mip_width / 2).max(1);
                mip_height = (mip_height / 2).max(1);
            }
            (StarPanoramaFormat::Bc1Srgb, stride, offset, mips)
        }
    };
    if row_stride_bytes != expected_stride {
        return Err(StarPanoramaError::RowStride {
            path: metadata_path.to_owned(),
            expected: expected_stride,
            actual: row_stride_bytes,
        });
    }
    if payload_bytes != expected_payload_bytes {
        return Err(StarPanoramaError::PayloadLength {
            path: metadata_path.to_owned(),
            expected: expected_payload_bytes,
            actual: payload_bytes,
        });
    }
    if longitude_at_u0_degrees != -180 || latitude_at_v0_degrees != 90 {
        return Err(StarPanoramaError::UnsupportedOrientation {
            path: metadata_path.to_owned(),
            longitude_at_u0_degrees,
            latitude_at_v0_degrees,
        });
    }
    Ok(ValidatedLayout {
        width,
        height,
        payload_bytes,
        format,
        mips,
    })
}

#[derive(Debug, Error)]
pub enum StarPanoramaError {
    #[error("{0} cannot be empty")]
    EmptyPath(&'static str),
    #[error("optional star panorama asset is absent: {path}")]
    MissingAsset { path: PathBuf },
    #[error("could not read star panorama metadata {path}: {source}")]
    ReadMetadata {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid star panorama metadata {path}: {source}")]
    InvalidMetadata {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("star panorama metadata {path} uses unsupported schema version {found}")]
    UnsupportedSchemaVersion { path: PathBuf, found: u32 },
    #[error("star panorama metadata {path} has a zero-sized extent")]
    ZeroExtent { path: PathBuf },
    #[error(
        "star panorama metadata {path} must be a 2:1 equirectangular image, got {width}x{height}"
    )]
    NotTwoToOne {
        path: PathBuf,
        width: u32,
        height: u32,
    },
    #[error("star panorama metadata {path} has row stride {actual}; expected tightly packed stride {expected}")]
    RowStride {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("star panorama metadata {path} declares {actual} payload bytes; expected {expected}")]
    PayloadLength {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("star panorama metadata {path} has BC1 base extent {width}x{height}; both dimensions must be multiples of 4")]
    Bc1BaseExtent {
        path: PathBuf,
        width: u32,
        height: u32,
    },
    #[error("star panorama metadata {path} requests {found} mip levels for {format:?}; expected 1..={maximum}")]
    MipCount {
        path: PathBuf,
        format: StarPanoramaFormat,
        found: u32,
        maximum: u32,
    },
    #[error("star panorama metadata {path} has unsupported orientation: U=0 longitude {longitude_at_u0_degrees}, V=0 latitude {latitude_at_v0_degrees}; expected -180 and 90")]
    UnsupportedOrientation {
        path: PathBuf,
        longitude_at_u0_degrees: i16,
        latitude_at_v0_degrees: i16,
    },
    #[error("star panorama metadata {path} overflows the raw image layout")]
    LayoutOverflow { path: PathBuf },
    #[error("could not open raw star panorama {path}: {source}")]
    OpenPayload {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not inspect raw star panorama {path}: {source}")]
    PayloadMetadata {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("raw star panorama path is not a regular file: {path}")]
    NotAFile { path: PathBuf },
    #[error("raw star panorama {path} has {actual} bytes; expected {expected}")]
    ByteCount {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("raw star panorama needs {0} bytes, which cannot be addressed on this host")]
    PayloadTooLarge(u64),
    #[error("could not memory-map raw star panorama {path}: {source}")]
    MapPayload {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl StarPanoramaError {
    fn is_missing_optional_asset(&self) -> bool {
        matches!(self, Self::MissingAsset { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_test_directory(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("wall clock must be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "earth-native-star-panorama-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn valid_sidecar(payload_bytes: u64) -> String {
        format!(
            r#"{{
  "schema_version": 1,
  "asset": "star_panorama",
  "width": 2,
  "height": 1,
  "row_stride_bytes": 8,
  "payload_bytes": {payload_bytes},
  "pixel_format": "bgra8",
  "color_space": "srgb",
  "projection": "equirectangular",
  "origin": "top_left",
  "longitude_at_u0_degrees": -180,
  "latitude_at_v0_degrees": 90
}}"#
        )
    }

    fn bc1_sidecar(width: u32, height: u32, mip_count: u32, payload_bytes: u64) -> String {
        let row_stride_bytes = width.div_ceil(4) * 8;
        format!(
            r#"{{
  "schema_version": 1,
  "asset": "star_panorama",
  "width": {width},
  "height": {height},
  "row_stride_bytes": {row_stride_bytes},
  "payload_bytes": {payload_bytes},
  "pixel_format": "bc1",
  "mip_count": {mip_count},
  "color_space": "srgb",
  "projection": "equirectangular",
  "origin": "top_left",
  "longitude_at_u0_degrees": -180,
  "latitude_at_v0_degrees": 90
}}"#
        )
    }

    fn validate_sidecar(sidecar: &str) -> Result<ValidatedLayout, StarPanoramaError> {
        let metadata = serde_json::from_str(sidecar).expect("deserialize sidecar");
        validate_metadata(metadata, Path::new("stars.raw.json"))
    }

    #[test]
    fn direct_raw_panorama_maps_exact_validated_bytes() {
        let directory = unique_test_directory("valid");
        fs::create_dir_all(&directory).expect("create test directory");
        let payload = directory.join("stars.bgra");
        let sidecar = default_sidecar_path(&payload);
        let bytes = [2, 6, 18, 255, 3, 7, 19, 255];
        fs::write(&payload, bytes).expect("write raw payload");
        fs::write(&sidecar, valid_sidecar(bytes.len() as u64)).expect("write sidecar");

        let panorama = load_from_paths(&payload, &sidecar).expect("validate raw panorama");
        assert_eq!(panorama.extent(), (2, 1));
        assert_eq!(panorama.format(), StarPanoramaFormat::Bgra8Srgb);
        assert_eq!(
            panorama.mips(),
            &[StarPanoramaMip {
                byte_offset: 0,
                width: 2,
                height: 1,
            }]
        );
        assert_eq!(panorama.payload_path(), payload);
        assert_eq!(panorama.metadata_path(), sidecar);
        assert_eq!(panorama.map_payload().expect("map payload").bytes(), bytes);

        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn bc1_single_mip_layout_is_tightly_packed() {
        let layout = validate_sidecar(&bc1_sidecar(8, 4, 1, 16)).expect("validate BC1");

        assert_eq!(layout.format, StarPanoramaFormat::Bc1Srgb);
        assert_eq!(layout.payload_bytes, 16);
        assert_eq!(
            layout.mips,
            vec![StarPanoramaMip {
                byte_offset: 0,
                width: 8,
                height: 4,
            }]
        );
    }

    #[test]
    fn bc1_multi_mip_layout_uses_concatenated_block_offsets() {
        let layout = validate_sidecar(&bc1_sidecar(16, 8, 4, 96)).expect("validate BC1 mips");

        assert_eq!(
            layout.mips,
            vec![
                StarPanoramaMip {
                    byte_offset: 0,
                    width: 16,
                    height: 8,
                },
                StarPanoramaMip {
                    byte_offset: 64,
                    width: 8,
                    height: 4,
                },
                StarPanoramaMip {
                    byte_offset: 80,
                    width: 4,
                    height: 2,
                },
                StarPanoramaMip {
                    byte_offset: 88,
                    width: 2,
                    height: 1,
                },
            ]
        );
    }

    #[test]
    fn bc1_rejects_invalid_payload_length_and_mip_count() {
        assert!(matches!(
            validate_sidecar(&bc1_sidecar(8, 4, 1, 15)),
            Err(StarPanoramaError::PayloadLength {
                expected: 16,
                actual: 15,
                ..
            })
        ));
        for mip_count in [0, 5] {
            assert!(matches!(
                validate_sidecar(&bc1_sidecar(8, 4, mip_count, 16)),
                Err(StarPanoramaError::MipCount {
                    found,
                    maximum: 4,
                    ..
                }) if found == mip_count
            ));
        }
    }

    #[test]
    fn raw_payload_length_must_match_the_explicit_sidecar() {
        let directory = unique_test_directory("short");
        fs::create_dir_all(&directory).expect("create test directory");
        let payload = directory.join("stars.bgra");
        let sidecar = default_sidecar_path(&payload);
        fs::write(&payload, [0_u8; 7]).expect("write short payload");
        fs::write(&sidecar, valid_sidecar(8)).expect("write sidecar");

        assert!(matches!(
            load_from_paths(&payload, &sidecar),
            Err(StarPanoramaError::ByteCount {
                expected: 8,
                actual: 7,
                ..
            })
        ));

        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn sidecar_rejects_non_srgb_or_unknown_pixel_formats() {
        let directory = unique_test_directory("metadata");
        fs::create_dir_all(&directory).expect("create test directory");
        let payload = directory.join("stars.bgra");
        let sidecar = default_sidecar_path(&payload);
        fs::write(&payload, [0_u8; 8]).expect("write payload");
        for invalid in [
            valid_sidecar(8).replace("\"bgra8\"", "\"rgba8\""),
            valid_sidecar(8).replace("\"srgb\"", "\"display_p3\""),
        ] {
            fs::write(&sidecar, invalid).expect("write invalid sidecar");
            assert!(matches!(
                load_from_paths(&payload, &sidecar),
                Err(StarPanoramaError::InvalidMetadata { .. })
            ));
        }

        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn absent_optional_payload_or_sidecar_selects_the_fallback() {
        let directory = unique_test_directory("missing");
        let payload = directory.join("stars.bgra");
        let sidecar = default_sidecar_path(&payload);
        assert!(load_optional_from_paths(&payload, &sidecar)
            .expect("missing optional asset is not an error")
            .is_none());
    }

    #[test]
    fn default_sidecar_keeps_the_raw_payload_extension() {
        assert_eq!(
            default_sidecar_path(Path::new("assets/stars.bgra")),
            PathBuf::from("assets/stars.bgra.json")
        );
    }
}
