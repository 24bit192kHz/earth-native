#![allow(dead_code)] // Shares the runtime format parser for pack-time validation.

#[path = "../earthvt.rs"]
mod earthvt;

use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use earthvt::{
    full_mip_count, EarthVt, LayerDescriptor, PixelFormat, TextureChannel, TileKey, GUTTER_SIZE,
    HEADER_BYTES, INDEX_ENTRY_BYTES, LAYER_ENTRY_BYTES, LAYER_FLAG_CLAMP_Y, LAYER_FLAG_SRGB,
    LAYER_FLAG_WRAP_X, MAGIC, PAYLOAD_ALIGNMENT, TILE_SIZE, VERSION,
};
use memmap2::MmapOptions;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    layers: Vec<ManifestLayer>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestLayer {
    id: u16,
    channel: String,
    format: String,
    base_width: u32,
    base_height: u32,
    #[serde(default)]
    srgb: bool,
    #[serde(default)]
    wrap_x: bool,
    #[serde(default)]
    clamp_y: bool,
    tiles: Vec<ManifestTile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestTile {
    mip: u16,
    x: u32,
    y: u32,
    path: PathBuf,
    #[serde(default)]
    content_hash: u32,
}

struct PreparedLayer {
    descriptor: LayerDescriptor,
    tiles: Vec<PreparedTile>,
}

struct PreparedTile {
    key: TileKey,
    path: PathBuf,
    bytes: u64,
    content_hash: u32,
    payload_offset: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("earthvt-pack: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let manifest_path = arguments
        .next()
        .ok_or("usage: earthvt-pack MANIFEST.json OUTPUT.earthvt")?;
    let output_path = arguments
        .next()
        .ok_or("usage: earthvt-pack MANIFEST.json OUTPUT.earthvt")?;
    if arguments.next().is_some() {
        return Err("usage: earthvt-pack MANIFEST.json OUTPUT.earthvt".into());
    }
    pack(Path::new(&manifest_path), Path::new(&output_path))
}

fn pack(manifest_path: &Path, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_parent = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let manifest: Manifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
    let mut layers = prepare_layers(manifest, manifest_parent)?;
    let index_count = layers.iter().try_fold(0_u32, |total, layer| {
        total
            .checked_add(u32::try_from(layer.tiles.len()).map_err(|_| "too many tiles")?)
            .ok_or("tile index count overflow")
    })?;
    let layer_table_offset = HEADER_BYTES as u64;
    let index_offset = layer_table_offset
        .checked_add(
            u64::try_from(layers.len())?
                .checked_mul(LAYER_ENTRY_BYTES as u64)
                .ok_or("layer table overflow")?,
        )
        .ok_or("index offset overflow")?;
    let payload_offset = align(
        index_offset
            .checked_add(
                u64::from(index_count)
                    .checked_mul(INDEX_ENTRY_BYTES as u64)
                    .ok_or("index table overflow")?,
            )
            .ok_or("payload offset overflow")?,
        PAYLOAD_ALIGNMENT,
    )?;
    let mut next_payload = payload_offset;
    for layer in &mut layers {
        for tile in &mut layer.tiles {
            tile.payload_offset = next_payload;
            next_payload = next_payload
                .checked_add(tile.bytes)
                .ok_or("payload size overflow")?;
        }
    }
    let temporary = output_path.with_extension(format!("earthvt.tmp.{}", std::process::id()));
    let write_result = write_container(
        &temporary,
        &layers,
        index_count,
        layer_table_offset,
        index_offset,
        payload_offset,
        next_payload,
    );
    match write_result {
        Ok(()) => {
            let packed_file = File::open(&temporary)?;
            let packed = unsafe { MmapOptions::new().map(&packed_file)? };
            EarthVt::parse(&packed)?;
            drop(packed);
            drop(packed_file);
            fs::rename(&temporary, output_path)?;
            println!(
                "packed {} layers, {} tiles, {} bytes into {}",
                layers.len(),
                index_count,
                next_payload,
                output_path.display()
            );
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error)
        }
    }
}

fn prepare_layers(
    manifest: Manifest,
    base: &Path,
) -> Result<Vec<PreparedLayer>, Box<dyn std::error::Error>> {
    if manifest.layers.is_empty() || manifest.layers.len() > usize::from(u16::MAX) {
        return Err("manifest must contain between one and 65535 layers".into());
    }
    let mut layers = Vec::with_capacity(manifest.layers.len());
    for layer in manifest.layers {
        let channel = parse_channel(&layer.channel)?;
        let format = parse_format(&layer.format)?;
        if !channel.accepts_format(format) {
            return Err(format!("{channel:?} cannot use {format:?}").into());
        }
        let mip_count = full_mip_count(layer.base_width, layer.base_height)
            .ok_or("layer dimensions must be nonzero")?;
        let flags = (u32::from(layer.srgb) * LAYER_FLAG_SRGB)
            | (u32::from(layer.wrap_x) * LAYER_FLAG_WRAP_X)
            | (u32::from(layer.clamp_y) * LAYER_FLAG_CLAMP_Y);
        let descriptor = LayerDescriptor {
            id: layer.id,
            channel,
            format,
            mip_count,
            base_width: layer.base_width,
            base_height: layer.base_height,
            first_index: 0,
            index_count: 0,
            flags,
        };
        let expected_count = descriptor
            .expected_tile_count()
            .ok_or("tile count overflow")?;
        let mut tiles = Vec::with_capacity(layer.tiles.len());
        for tile in layer.tiles {
            let key = TileKey::new(layer.id, tile.mip, tile.x, tile.y);
            if !descriptor.contains_key(key) {
                return Err(format!("tile {key:?} is outside its mip grid").into());
            }
            let path = if tile.path.is_absolute() {
                tile.path
            } else {
                base.join(tile.path)
            };
            let bytes = fs::metadata(&path)?.len();
            if bytes != format.encoded_tile_bytes() {
                return Err(format!(
                    "{} has {bytes} bytes; {:?} tiles require {} bytes",
                    path.display(),
                    format,
                    format.encoded_tile_bytes()
                )
                .into());
            }
            tiles.push(PreparedTile {
                key,
                path,
                bytes,
                content_hash: tile.content_hash,
                payload_offset: 0,
            });
        }
        tiles.sort_by_key(|tile| tile.key);
        if u32::try_from(tiles.len())? != expected_count {
            return Err(format!(
                "layer {} has {} tiles but a complete mip pyramid requires {expected_count}",
                layer.id,
                tiles.len()
            )
            .into());
        }
        for (index, tile) in tiles.iter().enumerate() {
            if index > 0 && tiles[index - 1].key == tile.key {
                return Err(
                    format!("layer {} contains duplicate tile {:?}", layer.id, tile.key).into(),
                );
            }
            let expected = expected_key_at(&descriptor, index as u32)?;
            if tile.key != expected {
                return Err(format!(
                    "layer {} is missing {:?} or contains unexpected {:?}",
                    layer.id, expected, tile.key
                )
                .into());
            }
        }
        layers.push(PreparedLayer { descriptor, tiles });
    }
    layers.sort_by_key(|layer| layer.descriptor.id);
    for index in 1..layers.len() {
        if layers[index - 1].descriptor.id == layers[index].descriptor.id {
            return Err("layer ids must be unique".into());
        }
    }
    let mut first_index = 0_u32;
    for layer in &mut layers {
        layer.descriptor.first_index = first_index;
        layer.descriptor.index_count = u32::try_from(layer.tiles.len())?;
        first_index = first_index
            .checked_add(layer.descriptor.index_count)
            .ok_or("tile index count overflow")?;
    }
    Ok(layers)
}

fn expected_key_at(
    layer: &LayerDescriptor,
    ordinal: u32,
) -> Result<TileKey, Box<dyn std::error::Error>> {
    let mut remaining = ordinal;
    for mip in 0..layer.mip_count {
        let (tiles_x, tiles_y) = layer.tile_grid(mip).ok_or("invalid mip grid")?;
        let count = tiles_x
            .checked_mul(tiles_y)
            .ok_or("mip tile count overflow")?;
        if remaining < count {
            return Ok(TileKey::new(
                layer.id,
                mip,
                remaining % tiles_x,
                remaining / tiles_x,
            ));
        }
        remaining -= count;
    }
    Err("tile ordinal is outside complete mip pyramid".into())
}

#[allow(clippy::too_many_arguments)]
fn write_container(
    output: &Path,
    layers: &[PreparedLayer],
    index_count: u32,
    layer_table_offset: u64,
    index_offset: u64,
    payload_offset: u64,
    file_bytes: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(output)?;
    file.set_len(file_bytes)?;
    let mut header = [0_u8; HEADER_BYTES];
    header[..8].copy_from_slice(&MAGIC);
    put_u16(&mut header, 8, VERSION);
    put_u16(&mut header, 10, HEADER_BYTES as u16);
    put_u16(&mut header, 16, TILE_SIZE);
    put_u16(&mut header, 18, GUTTER_SIZE);
    put_u16(&mut header, 20, u16::try_from(layers.len())?);
    put_u16(&mut header, 22, LAYER_ENTRY_BYTES as u16);
    put_u16(&mut header, 24, INDEX_ENTRY_BYTES as u16);
    put_u32(&mut header, 28, index_count);
    put_u64(&mut header, 32, layer_table_offset);
    put_u64(&mut header, 40, index_offset);
    put_u64(&mut header, 48, payload_offset);
    put_u64(&mut header, 56, file_bytes);
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header)?;

    file.seek(SeekFrom::Start(layer_table_offset))?;
    for layer in layers {
        let mut entry = [0_u8; LAYER_ENTRY_BYTES];
        put_u16(&mut entry, 0, layer.descriptor.id);
        put_u16(&mut entry, 2, layer.descriptor.channel.raw());
        put_u16(&mut entry, 4, layer.descriptor.format.raw());
        put_u16(&mut entry, 6, layer.descriptor.mip_count);
        put_u32(&mut entry, 8, layer.descriptor.base_width);
        put_u32(&mut entry, 12, layer.descriptor.base_height);
        put_u32(&mut entry, 16, layer.descriptor.first_index);
        put_u32(&mut entry, 20, layer.descriptor.index_count);
        put_u32(&mut entry, 24, layer.descriptor.flags);
        file.write_all(&entry)?;
    }

    file.seek(SeekFrom::Start(index_offset))?;
    for layer in layers {
        for tile in &layer.tiles {
            let mut entry = [0_u8; INDEX_ENTRY_BYTES];
            put_u16(&mut entry, 0, tile.key.layer);
            put_u16(&mut entry, 2, tile.key.mip);
            put_u32(&mut entry, 4, tile.key.x);
            put_u32(&mut entry, 8, tile.key.y);
            put_u64(&mut entry, 16, tile.payload_offset);
            put_u64(&mut entry, 24, tile.bytes);
            put_u32(&mut entry, 32, tile.content_hash);
            file.write_all(&entry)?;
        }
    }

    let mut buffer = [0_u8; 1024 * 1024];
    for layer in layers {
        for tile in &layer.tiles {
            file.seek(SeekFrom::Start(tile.payload_offset))?;
            let mut input = BufReader::new(File::open(&tile.path)?);
            let mut remaining = tile.bytes;
            while remaining > 0 {
                let requested = usize::try_from(remaining.min(buffer.len() as u64))?;
                input.read_exact(&mut buffer[..requested])?;
                file.write_all(&buffer[..requested])?;
                remaining -= requested as u64;
            }
        }
    }
    file.sync_all()?;
    Ok(())
}

fn parse_channel(value: &str) -> Result<TextureChannel, Box<dyn std::error::Error>> {
    match value {
        "day_color" => Ok(TextureChannel::DayColor),
        "surface_normal" => Ok(TextureChannel::SurfaceNormal),
        "night_emission" => Ok(TextureChannel::NightEmission),
        "water_mask" => Ok(TextureChannel::WaterMask),
        "cloud_density" => Ok(TextureChannel::CloudDensity),
        "cloud_normal" => Ok(TextureChannel::CloudNormal),
        "height" => Ok(TextureChannel::Height),
        "star_panorama" => Ok(TextureChannel::StarPanorama),
        "city_cloud_glow" => Ok(TextureChannel::CityCloudGlow),
        "aurora_mask" => Ok(TextureChannel::AuroraMask),
        "aurora_color" => Ok(TextureChannel::AuroraColor),
        "lightning_mask" => Ok(TextureChannel::LightningMask),
        _ => Err(format!("unknown texture channel {value:?}").into()),
    }
}

fn parse_format(value: &str) -> Result<PixelFormat, Box<dyn std::error::Error>> {
    match value {
        "bc1" => Ok(PixelFormat::Bc1),
        "bc7" => Ok(PixelFormat::Bc7),
        "bc5" => Ok(PixelFormat::Bc5),
        "bc4" => Ok(PixelFormat::Bc4),
        "r16" => Ok(PixelFormat::R16),
        _ => Err(format!("unknown pixel format {value:?}").into()),
    }
}

fn align(value: u64, alignment: u64) -> Result<u64, Box<dyn std::error::Error>> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| "container offset overflow".into())
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_is_stable() {
        assert_eq!(align(96, 16).unwrap(), 96);
        assert_eq!(align(97, 16).unwrap(), 112);
    }

    #[test]
    fn canonical_order_is_mip_row_then_column() {
        let layer = LayerDescriptor {
            id: 1,
            channel: TextureChannel::DayColor,
            format: PixelFormat::Bc7,
            mip_count: full_mip_count(512, 256).unwrap(),
            base_width: 512,
            base_height: 256,
            first_index: 0,
            index_count: 0,
            flags: LAYER_FLAG_SRGB | LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y,
        };
        assert_eq!(
            expected_key_at(&layer, 0).unwrap(),
            TileKey::new(1, 0, 0, 0)
        );
        assert_eq!(
            expected_key_at(&layer, 1).unwrap(),
            TileKey::new(1, 0, 1, 0)
        );
        assert_eq!(
            expected_key_at(&layer, 2).unwrap(),
            TileKey::new(1, 1, 0, 0)
        );
    }
}
