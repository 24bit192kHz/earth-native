//! Validated production asset bundle contract.
//!
//! A bundle is intentionally a small JSON index beside immutable, GPU-ready
//! payloads. Validation happens before Wayland/Vulkan startup so production
//! can never silently combine a partial bundle with preview variables.

use memmap2::MmapOptions;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use crate::earthvt::{EarthVt, TextureChannel};

pub const BUNDLE_ENV: &str = "EARTH_NATIVE_ASSET_BUNDLE";
pub const QUALITY_ENV: &str = "EARTH_NATIVE_RENDER_QUALITY";
pub const EARTHVT_ENV: &str = "EARTH_NATIVE_EARTHVT";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderQuality {
    Production,
    Compat,
}

impl RenderQuality {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "production" => Ok(Self::Production),
            "compat" => Ok(Self::Compat),
            value => Err(format!(
                "{QUALITY_ENV} must be production or compat, got {value:?}"
            )),
        }
    }
    pub fn from_environment() -> Result<Self, String> {
        Self::parse(env::var(QUALITY_ENV).as_deref().unwrap_or("compat"))
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::Production => "production",
            Self::Compat => "compat",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleManifest {
    pub schema_version: u32,
    pub bundle_id: String,
    pub earthvt: Artifact,
    pub artifacts: Vec<Artifact>,
    pub source_asset_ids: Vec<String>,
    pub channel_transforms: serde_json::Map<String, serde_json::Value>,
    pub compressor: Compressor,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub role: String,
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Compressor {
    pub name: String,
    pub version: String,
}

#[derive(Clone, Debug)]
pub struct ValidatedBundle {
    pub manifest_path: PathBuf,
    pub earthvt_path: PathBuf,
    pub artifact_count: usize,
    pub channels: Vec<TextureChannel>,
}

const REQUIRED_CHANNELS: &[TextureChannel] = &[
    TextureChannel::DayColor,
    TextureChannel::SurfaceNormal,
    TextureChannel::NightEmission,
    TextureChannel::CityCloudGlow,
    TextureChannel::WaterMask,
    TextureChannel::CloudDensity,
    TextureChannel::CloudNormal,
    TextureChannel::Height,
    TextureChannel::AuroraMask,
    TextureChannel::AuroraColor,
    TextureChannel::LightningMask,
    TextureChannel::StarPanorama,
];

pub fn validate(path: impl AsRef<Path>) -> Result<ValidatedBundle, String> {
    let manifest_path = path
        .as_ref()
        .canonicalize()
        .map_err(|e| format!("bundle manifest: {e}"))?;
    let bytes = fs::read(&manifest_path).map_err(|e| format!("bundle manifest: {e}"))?;
    let manifest: BundleManifest =
        serde_json::from_slice(&bytes).map_err(|e| format!("bundle manifest JSON: {e}"))?;
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported bundle schema {}",
            manifest.schema_version
        ));
    }
    if manifest.bundle_id.trim().is_empty() || manifest.source_asset_ids.is_empty() {
        return Err("bundle id and source_asset_ids are required".into());
    }
    if manifest.channel_transforms.is_empty() {
        return Err("bundle channel_transforms are required".into());
    }
    if manifest.compressor.name != "ispc_texcomp" || manifest.compressor.version.trim().is_empty() {
        return Err("bundle must pin ispc_texcomp and its version".into());
    }
    let root = manifest_path
        .parent()
        .ok_or("bundle has no parent directory")?;
    let mut all = manifest.artifacts.clone();
    all.push(manifest.earthvt.clone());
    for artifact in &all {
        if artifact.sha256.len() != 64 || !artifact.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("invalid SHA-256 for {}", artifact.role));
        }
        let candidate = root.join(&artifact.path);
        let canonical = candidate
            .canonicalize()
            .map_err(|e| format!("{}: {e}", artifact.path))?;
        if !canonical.starts_with(root) {
            return Err(format!(
                "artifact escapes bundle directory: {}",
                artifact.path
            ));
        }
        verify_sha256(&canonical, &artifact.sha256)
            .map_err(|e| format!("{}: {e}", artifact.path))?;
    }
    let earthvt_path = root.join(&manifest.earthvt.path);
    let earthvt_file = File::open(&earthvt_path).map_err(|e| format!("earthvt: {e}"))?;
    let mapped =
        unsafe { MmapOptions::new().map(&earthvt_file) }.map_err(|e| format!("earthvt: {e}"))?;
    let container = EarthVt::parse(&mapped).map_err(|e| format!("earthvt: {e}"))?;
    let mut channels = container
        .layers()
        .iter()
        .map(|l| l.channel)
        .collect::<Vec<_>>();
    channels.sort_by_key(|c| c.raw());
    channels.dedup();
    for required in REQUIRED_CHANNELS {
        if !channels.contains(required) {
            return Err(format!("earthvt missing required channel {required:?}"));
        }
    }
    Ok(ValidatedBundle {
        manifest_path,
        earthvt_path,
        artifact_count: all.len(),
        channels,
    })
}

fn verify_sha256(path: &Path, expected: &str) -> Result<(), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let bytes = reader.read(&mut buffer).map_err(|e| e.to_string())?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    let actual = format!("{:x}", hasher.finalize());
    (actual.eq_ignore_ascii_case(expected))
        .then_some(())
        .ok_or_else(|| format!("checksum mismatch (expected {expected}, got {actual})"))
}

pub fn configure_environment() -> Result<(RenderQuality, Option<ValidatedBundle>), String> {
    let quality = RenderQuality::from_environment()?;
    let bundle_path = env::var_os(BUNDLE_ENV);
    let bundle = match bundle_path {
        Some(path) => Some(validate(path)?),
        None if quality == RenderQuality::Production => {
            return Err(format!("{QUALITY_ENV}=production requires {BUNDLE_ENV}"))
        }
        None => None,
    };
    if let Some(bundle) = &bundle {
        env::set_var(EARTHVT_ENV, &bundle.earthvt_path);
    }
    Ok((quality, bundle))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quality_is_closed() {
        assert_eq!(
            RenderQuality::parse("production"),
            Ok(RenderQuality::Production)
        );
        assert!(RenderQuality::parse("ultra").is_err());
    }
    #[test]
    fn required_channels_are_multichannel() {
        assert!(REQUIRED_CHANNELS.len() >= 12);
    }
    #[test]
    fn sha256_validation_streams_the_expected_digest() {
        let path = std::env::temp_dir().join(format!("earth-native-sha256-{}", std::process::id()));
        fs::write(&path, b"abc").unwrap();
        let result = verify_sha256(
            &path,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
        fs::remove_file(path).unwrap();
        assert!(result.is_ok());
    }
}
