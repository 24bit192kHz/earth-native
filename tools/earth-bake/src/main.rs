//! earth-bake: NASA/GEBCO global sources -> earth-native GPU textures.
//!
//!   earth-bake day-vt --bmng DIR --month 200409 --gebco DIR --out day.earthvt [--width 65536]
//!
//! Blue Marble Next Generation "world" 500 m tiles (A1..D2, 21600^2 pixels
//! each covering 90 x 90 degrees; land surface and shallow water without
//! baked relief shading or bathymetry) are area-resampled in linear light to
//! a power-of-two grid (each source tile becomes a (width/4)^2 block), given a
//! water mask in alpha, mip-mapped in linear light and BC7-encoded into
//! 256 px pages with 4 px gutters: a one-layer (DayColor) .earthvt v1.
//!
//! The width must be a power of two times 256 with height width/2, so the
//! page grid halves exactly at every level: this is what the renderer's page
//! table (and its shader addressing) assumes.

#[allow(dead_code)]
#[path = "../../../src/earthvt.rs"]
mod earthvt;

use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{BufReader, Seek, SeekFrom, Write},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use earthvt::{
    full_mip_count, EarthVt, LayerDescriptor, PixelFormat, TextureChannel, GUTTER_SIZE, HEADER_BYTES,
    INDEX_ENTRY_BYTES, LAYER_ENTRY_BYTES, LAYER_FLAG_CLAMP_Y, LAYER_FLAG_SRGB, LAYER_FLAG_WRAP_X, MAGIC,
    PADDED_TILE_SIZE, PAYLOAD_ALIGNMENT, TILE_SIZE, VERSION,
};
use rayon::prelude::*;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const SOURCE_TILE: usize = 21_600;
const COLUMNS: [char; 4] = ['A', 'B', 'C', 'D'];

fn main() {
    if let Err(error) = run() {
        eprintln!("earth-bake: {error}");
        std::process::exit(1);
    }
}

struct Args(Vec<String>);

impl Args {
    fn value(&self, name: &str) -> Option<&str> {
        self.0.iter().position(|a| a == name).and_then(|i| self.0.get(i + 1)).map(String::as_str)
    }
    fn required(&self, name: &str) -> Result<&str> {
        self.value(name).ok_or_else(|| format!("missing {name}").into())
    }
}

fn run() -> Result<()> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let command = arguments.first().cloned().unwrap_or_default();
    let args = Args(arguments);
    match command.as_str() {
        "day-vt" => {
            let width: usize = args.value("--width").unwrap_or("65536").parse()?;
            if width % 256 != 0 || !(width / 256).is_power_of_two() {
                return Err("--width must be a power of two times 256".into());
            }
            bake_day_vt(
                Path::new(args.required("--bmng")?),
                args.required("--month")?,
                args.value("--gebco").map(Path::new),
                Path::new(args.required("--out")?),
                width,
            )
        }
        "gray-bc4" => {
            let width: usize = args.required("--width")?.parse()?;
            let grid: Vec<usize> = args.required("--grid")?.split('x').map(str::parse).collect::<std::result::Result<_, _>>()?;
            let tiles: Vec<PathBuf> = args.required("--tiles")?.split(',').map(PathBuf::from).collect();
            if grid.len() != 2 || tiles.len() != grid[0] * grid[1] {
                return Err("--grid CxR must match the number of --tiles (row-major)".into());
            }
            bake_gray_bc4(&tiles, grid[0], grid[1], width, args.required("--color-space")?,
                args.required("--name")?, Path::new(args.required("--out")?))
        }
        "static-vt" => bake_static_vt(Path::new(args.required("--textures")?), args.value("--tail-width").unwrap_or("4096").parse()?),
        "relief-bc5" => {
            let width: usize = args.required("--width")?.parse()?;
            let exaggeration: f32 = args.value("--exaggeration").unwrap_or("1.0").parse()?;
            bake_relief_bc5(Path::new(args.required("--gebco")?), width, exaggeration, Path::new(args.required("--out")?))
        }
        _ => Err("usage: earth-bake day-vt --bmng DIR --month YYYYMM [--gebco DIR] --out FILE [--width 65536]\n       earth-bake gray-bc4 --tiles A,B,... --grid CxR --width W --color-space srgb|linear --name NASA/x --out FILE.bc4\n       earth-bake static-vt --textures DIR [--tail-width 4096]".into()),
    }
}

// ---------------------------------------------------------------- colour

fn srgb_to_linear_table() -> [f32; 256] {
    let mut table = [0.0; 256];
    for (i, value) in table.iter_mut().enumerate() {
        let c = i as f32 / 255.0;
        *value = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
    }
    table
}

fn linear_to_srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

// ---------------------------------------------------------------- sources

fn load_png_rgb(path: &Path) -> Result<Vec<u8>> {
    let mut decoder = png::Decoder::new(BufReader::with_capacity(8 << 20, File::open(path)?));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let mut buffer = vec![0; reader.output_buffer_size().ok_or("PNG too large")?];
    let info = reader.next_frame(&mut buffer)?;
    if info.width as usize != SOURCE_TILE || info.height as usize != SOURCE_TILE {
        return Err(format!("{}: expected {SOURCE_TILE}^2, got {}x{}", path.display(), info.width, info.height).into());
    }
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => return Err(format!("{}: unsupported colour type {other:?}", path.display()).into()),
    };
    buffer.truncate(info.buffer_size());
    if channels == 4 {
        buffer = buffer.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    }
    Ok(buffer)
}

fn gebco_path(dir: &Path, column: usize, row: usize) -> PathBuf {
    let west = [-180, -90, 0, 90][column];
    let (north, south) = if row == 0 { ("90.0", "0.0") } else { ("0.0", "-90.0") };
    dir.join(format!("gebco_2026_n{north}_s{south}_w{west}.0_e{}.0_geotiff.tif", west + 90))
}

fn load_gebco(path: &Path) -> Result<Vec<i16>> {
    let mut decoder = tiff::decoder::Decoder::new(BufReader::with_capacity(8 << 20, File::open(path)?))?
        .with_limits(tiff::decoder::Limits::unlimited());
    let (width, height) = decoder.dimensions()?;
    if width as usize != SOURCE_TILE || height as usize != SOURCE_TILE {
        return Err(format!("{}: expected {SOURCE_TILE}^2, got {width}x{height}", path.display()).into());
    }
    match decoder.read_image()? {
        tiff::decoder::DecodingResult::I16(data) => Ok(data),
        _ => Err(format!("{}: expected 16-bit signed elevations", path.display()).into()),
    }
}

/// Water classification at 500 m. Blue Marble paints open ocean with a flat
/// (2, 5, 20) fill and inland water near-black; shallow banks and turbid
/// coasts keep their real colour, so below-sea-level pixels count as water
/// unless they look like bright desert (Qattara, Death Valley) or ice.
fn is_water(rgb: [u8; 3], elevation: Option<i16>) -> bool {
    let [r, g, b] = rgb.map(i32::from);
    let dark_fill = r.max(g).max(b) <= 24 && b >= r;
    if dark_fill {
        return true;
    }
    // Bright sea ice and salt flats are not open water (no specular glint).
    let bright = r > 150 && g > 150 && b > 150;
    match elevation {
        Some(e) if e < -5 => !(r > 120 && r > b + 30) && !bright,
        _ => false,
    }
}

// ---------------------------------------------------------------- resampling

/// Linear-light RGBA (alpha = water fraction) area resample of one source
/// tile into a `size` square, written into `target` (stride `target_width`)
/// at (`x0`, `y0`).
fn resample_tile(rgb: &[u8], elevation: Option<&[i16]>, size: usize, target: &mut [u8], target_width: usize, x0: usize, y0: usize) {
    let lut = srgb_to_linear_table();
    let ratio = SOURCE_TILE as f64 / size as f64;
    let rows: Vec<Vec<[u8; 4]>> = (0..size)
        .into_par_iter()
        .map(|ty| {
            let s0 = ty as f64 * ratio;
            let s1 = (ty + 1) as f64 * ratio;
            let mut accumulator = vec![[0.0f32; 4]; SOURCE_TILE];
            let mut sy = s0.floor() as usize;
            while (sy as f64) < s1 && sy < SOURCE_TILE {
                let weight = ((sy + 1) as f64).min(s1) - (sy as f64).max(s0);
                let w = (weight / ratio) as f32;
                let row = &rgb[sy * SOURCE_TILE * 3..(sy + 1) * SOURCE_TILE * 3];
                for sx in 0..SOURCE_TILE {
                    let p = [row[sx * 3], row[sx * 3 + 1], row[sx * 3 + 2]];
                    let water = is_water(p, elevation.map(|e| e[sy * SOURCE_TILE + sx]));
                    let a = &mut accumulator[sx];
                    a[0] += lut[p[0] as usize] * w;
                    a[1] += lut[p[1] as usize] * w;
                    a[2] += lut[p[2] as usize] * w;
                    a[3] += if water { w } else { 0.0 };
                }
                sy += 1;
            }
            (0..size)
                .map(|tx| {
                    let s0 = tx as f64 * ratio;
                    let s1 = (tx + 1) as f64 * ratio;
                    let mut sum = [0.0f32; 4];
                    let mut sx = s0.floor() as usize;
                    while (sx as f64) < s1 && sx < SOURCE_TILE {
                        let w = ((((sx + 1) as f64).min(s1) - (sx as f64).max(s0)) / ratio) as f32;
                        for c in 0..4 {
                            sum[c] += accumulator[sx][c] * w;
                        }
                        sx += 1;
                    }
                    [linear_to_srgb(sum[0]), linear_to_srgb(sum[1]), linear_to_srgb(sum[2]), (sum[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8]
                })
                .collect()
        })
        .collect();
    for (ty, row) in rows.into_iter().enumerate() {
        let offset = ((y0 + ty) * target_width + x0) * 4;
        for (tx, p) in row.into_iter().enumerate() {
            target[offset + tx * 4..offset + tx * 4 + 4].copy_from_slice(&p);
        }
    }
}

struct Image {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

/// 2x2 box in linear light (colour) / linearly (alpha).
fn half(image: &Image) -> Image {
    let lut = srgb_to_linear_table();
    let width = (image.width / 2).max(1);
    let height = (image.height / 2).max(1);
    let mut rgba = vec![0u8; width * height * 4];
    rgba.par_chunks_mut(width * 4).enumerate().for_each(|(y, out)| {
        for x in 0..width {
            let mut sum = [0.0f32; 4];
            let mut n = 0.0;
            for dy in 0..2 {
                for dx in 0..2 {
                    let sx = (x * 2 + dx).min(image.width - 1);
                    let sy = (y * 2 + dy).min(image.height - 1);
                    let p = &image.rgba[(sy * image.width + sx) * 4..(sy * image.width + sx) * 4 + 4];
                    sum[0] += lut[p[0] as usize];
                    sum[1] += lut[p[1] as usize];
                    sum[2] += lut[p[2] as usize];
                    sum[3] += p[3] as f32;
                    n += 1.0;
                }
            }
            out[x * 4] = linear_to_srgb(sum[0] / n);
            out[x * 4 + 1] = linear_to_srgb(sum[1] / n);
            out[x * 4 + 2] = linear_to_srgb(sum[2] / n);
            out[x * 4 + 3] = (sum[3] / n + 0.5) as u8;
        }
    });
    Image { width, height, rgba }
}

impl Image {
    fn texel(&self, x: i64, y: i64) -> [u8; 4] {
        let x = x.rem_euclid(self.width as i64) as usize;
        let y = y.clamp(0, self.height as i64 - 1) as usize;
        let i = (y * self.width + x) * 4;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }

    /// Bilinear at continuous texel coordinates (x wraps, y clamps).
    fn bilinear(&self, x: f64, y: f64) -> [u8; 4] {
        let lut = srgb_to_linear_table();
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
        let mut sum = [0.0f32; 4];
        for (dx, dy, w) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
            let p = self.texel(x0 as i64 + dx, y0 as i64 + dy);
            sum[0] += lut[p[0] as usize] * w;
            sum[1] += lut[p[1] as usize] * w;
            sum[2] += lut[p[2] as usize] * w;
            sum[3] += p[3] as f32 * w;
        }
        [linear_to_srgb(sum[0]), linear_to_srgb(sum[1]), linear_to_srgb(sum[2]), (sum[3] + 0.5) as u8]
    }
}

// ---------------------------------------------------------------- baking

fn bake_day_vt(bmng: &Path, month: &str, gebco: Option<&Path>, out: &Path, width: usize) -> Result<()> {
    let started = Instant::now();
    let height = width / 2;
    let block = width / 4;
    eprintln!("earth-bake: {width}x{height} day VT from Blue Marble {month}");
    let mut base = Image { width, height, rgba: vec![0u8; width * height * 4] };
    for row in 0..2 {
        for column in 0..4 {
            let name = format!("{}{}", COLUMNS[column], row + 1);
            let path = bmng.join(format!("world.{month}.3x21600x21600.{name}.png"));
            let t = Instant::now();
            let rgb = load_png_rgb(&path)?;
            let elevation = match gebco {
                Some(dir) => Some(load_gebco(&gebco_path(dir, column, row))?),
                None => None,
            };
            resample_tile(&rgb, elevation.as_deref(), block, &mut base.rgba, width, column * block, row * block);
            eprintln!("  tile {name}: {:.1?}", t.elapsed());
        }
    }
    let mip_count = full_mip_count(width as u32, height as u32).ok_or("bad size")?;
    let mut levels = vec![base];
    while levels.last().map_or(false, |l| l.width > 1 || l.height > 1) {
        let next = half(levels.last().unwrap());
        levels.push(next);
    }
    assert_eq!(levels.len(), mip_count as usize);
    eprintln!("  mips: {:.1?}", started.elapsed());
    if let Some(level) = levels.iter().find(|l| l.width <= 4096) {
        let preview = out.with_extension("preview.png");
        write_png(&preview, level)?;
        eprintln!("  preview {}", preview.display());
    }

    let layer = LayerDescriptor {
        id: 1,
        channel: TextureChannel::DayColor,
        format: PixelFormat::Bc7,
        mip_count,
        base_width: width as u32,
        base_height: height as u32,
        first_index: 0,
        index_count: 0,
        flags: LAYER_FLAG_SRGB | LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y,
    };
    let tiles0 = (width / TILE_SIZE as usize, height / TILE_SIZE as usize);
    let mut keys = Vec::new();
    for mip in 0..mip_count {
        let (tx, ty) = layer.tile_grid(mip).ok_or("grid")?;
        // The shader addresses pages with max(tiles0 >> mip, 1).
        let shader = ((tiles0.0 >> mip).max(1) as u32, (tiles0.1 >> mip).max(1) as u32);
        if (tx, ty) != shader {
            return Err(format!("mip {mip}: format grid {tx}x{ty} != shader grid {shader:?}").into());
        }
        for y in 0..ty {
            for x in 0..tx {
                keys.push((mip, x, y));
            }
        }
    }
    let tile_bytes = PixelFormat::Bc7.encoded_tile_bytes();
    let layer_table_offset = HEADER_BYTES as u64;
    let index_offset = layer_table_offset + LAYER_ENTRY_BYTES as u64;
    let payload_offset = (index_offset + keys.len() as u64 * INDEX_ENTRY_BYTES as u64).div_ceil(PAYLOAD_ALIGNMENT) * PAYLOAD_ALIGNMENT;
    let file_bytes = payload_offset + keys.len() as u64 * tile_bytes;
    let temporary = out.with_extension("earthvt.partial");
    let _ = fs::remove_file(&temporary);
    let mut file = OpenOptions::new().create_new(true).read(true).write(true).open(&temporary)?;
    file.set_len(file_bytes)?;

    let padded = PADDED_TILE_SIZE as usize;
    let gutter = GUTTER_SIZE as i64;
    let done = AtomicUsize::new(0);
    let settings = intel_tex_2::bc7::alpha_basic_settings();
    let hashes: Vec<u32> = keys
        .par_iter()
        .enumerate()
        .map(|(ordinal, &(mip, x, y))| -> Result<u32> {
            let image = &levels[mip as usize];
            let (tx, ty) = layer.tile_grid(mip).unwrap();
            let mut pixels = vec![0u8; padded * padded * 4];
            let exact = image.width == tx as usize * 256 && image.height == ty as usize * 256;
            for j in 0..padded {
                for i in 0..padded {
                    let p = if exact {
                        image.texel(x as i64 * 256 + i as i64 - gutter, y as i64 * 256 + j as i64 - gutter)
                    } else {
                        // Coarse levels: the page spans the map; stretch.
                        let u = (x as f64 + (i as f64 - gutter as f64 + 0.5) / 256.0) / tx as f64;
                        let v = (y as f64 + (j as f64 - gutter as f64 + 0.5) / 256.0) / ty as f64;
                        image.bilinear(u * image.width as f64 - 0.5, v * image.height as f64 - 0.5)
                    };
                    pixels[(j * padded + i) * 4..(j * padded + i) * 4 + 4].copy_from_slice(&p);
                }
            }
            let surface = intel_tex_2::RgbaSurface { data: &pixels, width: padded as u32, height: padded as u32, stride: padded as u32 * 4 };
            let payload = intel_tex_2::bc7::compress_blocks(&settings, &surface);
            if payload.len() as u64 != tile_bytes {
                return Err(format!("BC7 payload {} != {tile_bytes}", payload.len()).into());
            }
            file.write_all_at(&payload, payload_offset + ordinal as u64 * tile_bytes)?;
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 2000 == 0 {
                eprintln!("  {n}/{} pages ({:.1?})", keys.len(), started.elapsed());
            }
            Ok(fnv1a(&payload))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut header = [0u8; HEADER_BYTES];
    header[..8].copy_from_slice(&MAGIC);
    put16(&mut header, 8, VERSION);
    put16(&mut header, 10, HEADER_BYTES as u16);
    put16(&mut header, 16, TILE_SIZE);
    put16(&mut header, 18, GUTTER_SIZE);
    put16(&mut header, 20, 1);
    put16(&mut header, 22, LAYER_ENTRY_BYTES as u16);
    put16(&mut header, 24, INDEX_ENTRY_BYTES as u16);
    put32(&mut header, 28, keys.len() as u32);
    put64(&mut header, 32, layer_table_offset);
    put64(&mut header, 40, index_offset);
    put64(&mut header, 48, payload_offset);
    put64(&mut header, 56, file_bytes);
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header)?;
    let mut entry = [0u8; LAYER_ENTRY_BYTES];
    put16(&mut entry, 0, layer.id);
    put16(&mut entry, 2, layer.channel.raw());
    put16(&mut entry, 4, layer.format.raw());
    put16(&mut entry, 6, layer.mip_count);
    put32(&mut entry, 8, layer.base_width);
    put32(&mut entry, 12, layer.base_height);
    put32(&mut entry, 16, 0);
    put32(&mut entry, 20, keys.len() as u32);
    put32(&mut entry, 24, layer.flags);
    file.write_all(&entry)?;
    let mut index = Vec::with_capacity(keys.len() * INDEX_ENTRY_BYTES);
    for (ordinal, &(mip, x, y)) in keys.iter().enumerate() {
        let mut e = [0u8; INDEX_ENTRY_BYTES];
        put16(&mut e, 0, layer.id);
        put16(&mut e, 2, mip);
        put32(&mut e, 4, x);
        put32(&mut e, 8, y);
        put64(&mut e, 16, payload_offset + ordinal as u64 * tile_bytes);
        put64(&mut e, 24, tile_bytes);
        put32(&mut e, 32, hashes[ordinal]);
        index.extend_from_slice(&e);
    }
    file.seek(SeekFrom::Start(index_offset))?;
    file.write_all(&index)?;
    file.sync_all()?;
    drop(file);
    let mapped = unsafe { memmap2::Mmap::map(&File::open(&temporary)?)? };
    EarthVt::parse(&mapped).map_err(|e| format!("validation: {e}"))?;
    drop(mapped);
    fs::rename(&temporary, out)?;
    eprintln!("earth-bake: wrote {} ({} pages, {:.2} GB) in {:.1?}", out.display(), keys.len(), file_bytes as f64 / 1e9, started.elapsed());
    Ok(())
}

fn write_png(path: &Path, image: &Image) -> Result<()> {
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(File::create(path)?), image.width as u32, image.height as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&image.rgba)?;
    Ok(())
}


// ---------------------------------------------------------------- static VT

/// One block-compressed source map (a `.bc4`/`.bc5` with its mip chain).
struct BlockMap {
    bytes: memmap2::Mmap,
    meta: serde_json::Value,
    block_bytes: usize,
    /// (width, height, byte offset) per mip.
    mips: Vec<(usize, usize, usize)>,
}

impl BlockMap {
    fn open(path: &Path) -> Result<Self> {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".json");
        let meta: serde_json::Value = serde_json::from_slice(&fs::read(&sidecar)?)?;
        let block_bytes = match meta["pixel_format"].as_str() {
            Some("bc4") => 8,
            Some("bc5") => 16,
            other => return Err(format!("{}: unsupported pixel format {other:?}", path.display()).into()),
        };
        let mips = meta["mips"].as_array().ok_or("sidecar has no mips")?.iter().map(|m| -> Result<_> {
            Ok((m["width"].as_u64().ok_or("mip width")? as usize, m["height"].as_u64().ok_or("mip height")? as usize,
                m["byte_offset"].as_u64().ok_or("mip offset")? as usize))
        }).collect::<Result<Vec<_>>>()?;
        let bytes = unsafe { memmap2::Mmap::map(&File::open(path)?)? };
        Ok(Self { bytes, meta, block_bytes, mips })
    }

    /// One 264x264 tile (66x66 blocks, a one-block gutter) of `mip`, copied
    /// block for block: longitude wraps, latitude and small levels clamp.
    fn tile(&self, mip: usize, x: u32, y: u32) -> Vec<u8> {
        let (width, height, offset) = self.mips[mip.min(self.mips.len() - 1)];
        let (bw, bh) = ((width / 4).max(1), (height / 4).max(1));
        let blocks = PADDED_TILE_SIZE as usize / 4;
        let mut out = Vec::with_capacity(blocks * blocks * self.block_bytes);
        for j in 0..blocks as i64 {
            let by = (y as i64 * 64 + j - 1).clamp(0, bh as i64 - 1) as usize;
            for i in 0..blocks as i64 {
                let bx = (x as i64 * 64 + i - 1).rem_euclid(bw as i64).min(bw as i64 - 1) as usize;
                let at = offset + (by * bw + bx) * self.block_bytes;
                out.extend_from_slice(&self.bytes[at..at + self.block_bytes]);
            }
        }
        out
    }

    /// The mips from `tail_width` down, as a standalone map (its sidecar
    /// rebased), for the always-resident part the renderer samples beyond
    /// the virtual texture.
    fn write_tail(&self, tail_width: usize, out: &Path) -> Result<()> {
        let first = self.mips.iter().position(|m| m.0 <= tail_width).ok_or("no mip at the tail width")?;
        let base = self.mips[first].2;
        let end = self.meta["payload_bytes"].as_u64().ok_or("payload_bytes")? as usize;
        let temporary = out.with_extension("partial");
        fs::write(&temporary, &self.bytes[base..end])?;
        let mut meta = self.meta.clone();
        let (width, height, _) = self.mips[first];
        meta["width"] = width.into();
        meta["height"] = height.into();
        meta["row_stride_bytes"] = ((width / 4) * self.block_bytes).into();
        meta["payload_bytes"] = (end - base).into();
        meta["mips"] = self.mips[first..].iter()
            .map(|&(w, h, o)| serde_json::json!({"byte_offset": o - base, "width": w, "height": h}))
            .collect();
        let mut sidecar = out.as_os_str().to_owned();
        sidecar.push(".json");
        fs::write(&sidecar, serde_json::to_vec_pretty(&meta)?)?;
        fs::rename(&temporary, out)?;
        Ok(())
    }
}

/// `earth-static.earthvt`: the 32K night lights, NASA cloud map and GEBCO
/// relief normals as one three-layer virtual texture (their blocks re-tiled,
/// not re-encoded), plus `*-tail` maps of the levels from `tail_width` down.
/// The renderer streams only the three finest levels and keeps the tails
/// resident: ~1.4 GB of fully resident VRAM becomes ~0.1 GB.
fn bake_static_vt(textures: &Path, tail_width: usize) -> Result<()> {
    let started = Instant::now();
    let sources = [
        ("night.bc4", TextureChannel::NightEmission, PixelFormat::Bc4, "night-tail.bc4"),
        ("clouds.bc4", TextureChannel::CloudDensity, PixelFormat::Bc4, "clouds-tail.bc4"),
        ("relief.bc5", TextureChannel::SurfaceNormal, PixelFormat::Bc5, "relief-tail.bc5"),
    ];
    let maps = sources.iter().map(|(name, ..)| BlockMap::open(&textures.join(name))).collect::<Result<Vec<_>>>()?;
    let (width, height, _) = maps[0].mips[0];
    if maps.iter().any(|m| m.mips[0].0 != width || m.mips[0].1 != height) {
        return Err("night, clouds and relief must share one size".into());
    }
    let mip_count = full_mip_count(width as u32, height as u32).ok_or("bad size")?;
    let mut layers = Vec::new();
    let mut keys = Vec::new();
    for (ordinal, (&(_, channel, format, _), _)) in sources.iter().zip(&maps).enumerate() {
        let mut layer = LayerDescriptor {
            id: ordinal as u16 + 1, channel, format, mip_count,
            base_width: width as u32, base_height: height as u32,
            first_index: keys.len() as u32, index_count: 0,
            flags: LAYER_FLAG_WRAP_X | LAYER_FLAG_CLAMP_Y,
        };
        for mip in 0..mip_count {
            let (tx, ty) = layer.tile_grid(mip).ok_or("grid")?;
            for y in 0..ty {
                for x in 0..tx {
                    keys.push((ordinal, layer.id, mip, x, y));
                }
            }
        }
        layer.index_count = keys.len() as u32 - layer.first_index;
        layers.push(layer);
    }
    let layer_table_offset = HEADER_BYTES as u64;
    let index_offset = layer_table_offset + (LAYER_ENTRY_BYTES * layers.len()) as u64;
    let payload_offset = (index_offset + keys.len() as u64 * INDEX_ENTRY_BYTES as u64).div_ceil(PAYLOAD_ALIGNMENT) * PAYLOAD_ALIGNMENT;
    let mut offsets = Vec::with_capacity(keys.len());
    let mut cursor = payload_offset;
    for &(ordinal, ..) in &keys {
        offsets.push(cursor);
        cursor += sources[ordinal].2.encoded_tile_bytes();
    }
    let file_bytes = cursor;
    let out = textures.join("earth-static.earthvt");
    let temporary = out.with_extension("earthvt.partial");
    let _ = fs::remove_file(&temporary);
    let mut file = OpenOptions::new().create_new(true).read(true).write(true).open(&temporary)?;
    file.set_len(file_bytes)?;
    let hashes: Vec<u32> = keys.par_iter().zip(&offsets).map(|(&(ordinal, _, mip, x, y), &offset)| -> Result<u32> {
        let payload = maps[ordinal].tile(mip as usize, x, y);
        file.write_all_at(&payload, offset)?;
        Ok(fnv1a(&payload))
    }).collect::<Result<Vec<_>>>()?;

    let mut header = [0u8; HEADER_BYTES];
    header[..8].copy_from_slice(&MAGIC);
    put16(&mut header, 8, VERSION);
    put16(&mut header, 10, HEADER_BYTES as u16);
    put16(&mut header, 16, TILE_SIZE);
    put16(&mut header, 18, GUTTER_SIZE);
    put16(&mut header, 20, layers.len() as u16);
    put16(&mut header, 22, LAYER_ENTRY_BYTES as u16);
    put16(&mut header, 24, INDEX_ENTRY_BYTES as u16);
    put32(&mut header, 28, keys.len() as u32);
    put64(&mut header, 32, layer_table_offset);
    put64(&mut header, 40, index_offset);
    put64(&mut header, 48, payload_offset);
    put64(&mut header, 56, file_bytes);
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header)?;
    for layer in &layers {
        let mut entry = [0u8; LAYER_ENTRY_BYTES];
        put16(&mut entry, 0, layer.id);
        put16(&mut entry, 2, layer.channel.raw());
        put16(&mut entry, 4, layer.format.raw());
        put16(&mut entry, 6, layer.mip_count);
        put32(&mut entry, 8, layer.base_width);
        put32(&mut entry, 12, layer.base_height);
        put32(&mut entry, 16, layer.first_index);
        put32(&mut entry, 20, layer.index_count);
        put32(&mut entry, 24, layer.flags);
        file.write_all(&entry)?;
    }
    let mut index = Vec::with_capacity(keys.len() * INDEX_ENTRY_BYTES);
    for (ordinal, &(source, id, mip, x, y)) in keys.iter().enumerate() {
        let mut e = [0u8; INDEX_ENTRY_BYTES];
        put16(&mut e, 0, id);
        put16(&mut e, 2, mip);
        put32(&mut e, 4, x);
        put32(&mut e, 8, y);
        put64(&mut e, 16, offsets[ordinal]);
        put64(&mut e, 24, sources[source].2.encoded_tile_bytes());
        put32(&mut e, 32, hashes[ordinal]);
        index.extend_from_slice(&e);
    }
    file.seek(SeekFrom::Start(index_offset))?;
    file.write_all(&index)?;
    file.sync_all()?;
    drop(file);
    let mapped = unsafe { memmap2::Mmap::map(&File::open(&temporary)?)? };
    EarthVt::parse(&mapped).map_err(|e| format!("validation: {e}"))?;
    drop(mapped);
    fs::rename(&temporary, &out)?;
    for (map, &(.., tail)) in maps.iter().zip(&sources) {
        map.write_tail(tail_width, &textures.join(tail))?;
    }
    eprintln!("earth-bake: wrote {} ({} tiles, {:.2} GB) and {tail_width}-wide tails in {:.1?}",
        out.display(), keys.len(), file_bytes as f64 / 1e9, started.elapsed());
    Ok(())
}

fn fnv1a(bytes: &[u8]) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    for &b in bytes {
        hash ^= u32::from(b);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

fn put16(b: &mut [u8], o: usize, v: u16) {
    b[o..o + 2].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn put64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}

// ---------------------------------------------------------------- gray BC4

fn load_gray(path: &Path) -> Result<(usize, usize, Vec<u8>)> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if extension == "png" {
        let mut decoder = png::Decoder::new(BufReader::with_capacity(8 << 20, File::open(path)?));
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = decoder.read_info()?;
        let mut buffer = vec![0; reader.output_buffer_size().ok_or("PNG too large")?];
        let info = reader.next_frame(&mut buffer)?;
        buffer.truncate(info.buffer_size());
        let channels = buffer.len() / (info.width as usize * info.height as usize);
        let gray = if channels == 1 { buffer } else { buffer.chunks_exact(channels).map(|p| p[0]).collect() };
        return Ok((info.width as usize, info.height as usize, gray));
    }
    let mut decoder = tiff::decoder::Decoder::new(BufReader::with_capacity(8 << 20, File::open(path)?))?
        .with_limits(tiff::decoder::Limits::unlimited());
    let (width, height) = decoder.dimensions()?;
    let pixels = (width as usize) * (height as usize);
    let gray = match decoder.read_image()? {
        tiff::decoder::DecodingResult::U8(data) => {
            let channels = data.len() / pixels;
            if channels == 1 { data } else { data.chunks_exact(channels).map(|p| p[0]).collect() }
        }
        _ => return Err(format!("{}: expected 8-bit samples", path.display()).into()),
    };
    Ok((width as usize, height as usize, gray))
}

/// Area resample of an 8-bit map (sRGB-coded values averaged in linear
/// light when `srgb`) into `target` at (x0, y0) with size bw x bh.
fn resample_gray(source: &[u8], sw: usize, sh: usize, bw: usize, bh: usize, srgb: bool,
    target: &mut [u8], target_width: usize, x0: usize, y0: usize) {
    let lut = srgb_to_linear_table();
    let decode = |v: u8| if srgb { lut[v as usize] } else { v as f32 / 255.0 };
    let encode = |v: f32| if srgb { linear_to_srgb(v) } else { (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8 };
    let rx = sw as f64 / bw as f64;
    let ry = sh as f64 / bh as f64;
    let rows: Vec<Vec<u8>> = (0..bh).into_par_iter().map(|ty| {
        let (s0, s1) = (ty as f64 * ry, (ty + 1) as f64 * ry);
        let mut accumulator = vec![0.0f32; sw];
        let mut sy = s0.floor() as usize;
        while (sy as f64) < s1 && sy < sh {
            let w = (((sy + 1) as f64).min(s1) - (sy as f64).max(s0)) / ry;
            let row = &source[sy * sw..(sy + 1) * sw];
            for (a, &v) in accumulator.iter_mut().zip(row) {
                *a += decode(v) * w as f32;
            }
            sy += 1;
        }
        (0..bw).map(|tx| {
            let (s0, s1) = (tx as f64 * rx, (tx + 1) as f64 * rx);
            let mut sum = 0.0f32;
            let mut sx = s0.floor() as usize;
            while (sx as f64) < s1 && sx < sw {
                sum += accumulator[sx] * ((((sx + 1) as f64).min(s1) - (sx as f64).max(s0)) / rx) as f32;
                sx += 1;
            }
            encode(sum)
        }).collect()
    }).collect();
    for (ty, row) in rows.into_iter().enumerate() {
        let offset = (y0 + ty) * target_width + x0;
        target[offset..offset + bw].copy_from_slice(&row);
    }
}

fn half_gray(source: &[u8], width: usize, height: usize, srgb: bool) -> Vec<u8> {
    let lut = srgb_to_linear_table();
    let decode = |v: u8| if srgb { lut[v as usize] } else { v as f32 / 255.0 };
    let encode = |v: f32| if srgb { linear_to_srgb(v) } else { (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8 };
    let (w, h) = (width / 2, height / 2);
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, value) in row.iter_mut().enumerate() {
            let i = y * 2 * width + x * 2;
            let sum = decode(source[i]) + decode(source[i + 1]) + decode(source[i + width]) + decode(source[i + width + 1]);
            *value = encode(sum * 0.25);
        }
    });
    out
}

fn bc4(gray: &[u8], width: usize, height: usize) -> Vec<u8> {
    const STRIP: usize = 64;
    let strips: Vec<Vec<u8>> = (0..height.div_ceil(STRIP)).into_par_iter().map(|strip| {
        let y0 = strip * STRIP;
        let rows = STRIP.min(height - y0);
        let surface = intel_tex_2::RSurface { data: &gray[y0 * width..(y0 + rows) * width], width: width as u32, height: rows as u32, stride: width as u32 };
        intel_tex_2::bc4::compress_blocks(&surface)
    }).collect();
    strips.concat()
}

fn bake_gray_bc4(tiles: &[PathBuf], columns: usize, rows: usize, width: usize, color_space: &str, name: &str, out: &Path) -> Result<()> {
    let started = Instant::now();
    let srgb = match color_space { "srgb" => true, "linear" => false, _ => return Err("--color-space srgb|linear".into()) };
    let height = width / 2;
    if width % (4 * columns) != 0 || height % (4 * rows) != 0 {
        return Err("size must split into whole BC4 blocks per source tile".into());
    }
    let (bw, bh) = (width / columns, height / rows);
    let mut base = vec![0u8; width * height];
    let mut source_size = (0, 0);
    for (index, path) in tiles.iter().enumerate() {
        let t = Instant::now();
        let (sw, sh, gray) = load_gray(path)?;
        source_size = (sw * columns, sh * rows);
        resample_gray(&gray, sw, sh, bw, bh, srgb, &mut base, width, (index % columns) * bw, (index / columns) * bh);
        eprintln!("  {} ({sw}x{sh}): {:.1?}", path.display(), t.elapsed());
    }
    let mut payload = Vec::new();
    let mut mips = Vec::new();
    let (mut w, mut h, mut level) = (width, height, base);
    loop {
        mips.push(format!("{{\"byte_offset\": {}, \"width\": {w}, \"height\": {h}}}", payload.len()));
        payload.extend_from_slice(&bc4(&level, w, h));
        if w / 2 < 4 || h / 2 < 4 || w % 8 != 0 || h % 8 != 0 || mips.len() == 16 {
            break;
        }
        level = half_gray(&level, w, h, srgb);
        w /= 2;
        h /= 2;
    }
    fs::write(out, &payload)?;
    let sidecar = format!(
        "{{\n  \"schema_version\": 1,\n  \"asset\": \"earth_native_preview_texture\",\n  \"width\": {width},\n  \"height\": {height},\n  \"row_stride_bytes\": {},\n  \"payload_bytes\": {},\n  \"pixel_format\": \"bc4\",\n  \"color_space\": \"{color_space}\",\n  \"origin\": \"top_left\",\n  \"source_object_path\": \"{name}\",\n  \"source_width\": {},\n  \"source_height\": {},\n  \"source_blocks\": 1,\n  \"filter\": \"area\",\n  \"rgb_filter\": \"linear-light-srgb-v1\",\n  \"layout\": \"equirectangular\",\n  \"mips\": [\n    {}\n  ]\n}}\n",
        width / 4 * 8, payload.len(), source_size.0, source_size.1, mips.join(",\n    "));
    let mut sidecar_path = out.as_os_str().to_owned();
    sidecar_path.push(".json");
    fs::write(sidecar_path, sidecar)?;
    eprintln!("earth-bake: wrote {} ({width}x{height}, {} mips, {:.0} MB) in {:.1?}", out.display(), mips.len(), payload.len() as f64 / 1e6, started.elapsed());
    Ok(())
}

// ---------------------------------------------------------------- relief BC5

/// Area-averaged GEBCO elevation (m, oceans at 0) on a width x width/2 grid.
fn gebco_heights(dir: &Path, width: usize) -> Result<Vec<f32>> {
    let height = width / 2;
    let block = width / 4;
    let mut grid = vec![0.0f32; width * height];
    for row in 0..2 {
        for column in 0..4 {
            let t = Instant::now();
            let path = gebco_path(dir, column, row);
            let source = load_gebco(&path)?;
            let ratio = SOURCE_TILE as f64 / block as f64;
            let rows: Vec<Vec<f32>> = (0..block).into_par_iter().map(|ty| {
                let (s0, s1) = (ty as f64 * ratio, (ty + 1) as f64 * ratio);
                let mut accumulator = vec![0.0f32; SOURCE_TILE];
                let mut sy = s0.floor() as usize;
                while (sy as f64) < s1 && sy < SOURCE_TILE {
                    let w = ((((sy + 1) as f64).min(s1) - (sy as f64).max(s0)) / ratio) as f32;
                    for (a, &e) in accumulator.iter_mut().zip(&source[sy * SOURCE_TILE..(sy + 1) * SOURCE_TILE]) {
                        // The sea surface is flat: no bathymetry shading.
                        *a += (e as f32).max(0.0) * w;
                    }
                    sy += 1;
                }
                (0..block).map(|tx| {
                    let (s0, s1) = (tx as f64 * ratio, (tx + 1) as f64 * ratio);
                    let mut sum = 0.0;
                    let mut sx = s0.floor() as usize;
                    while (sx as f64) < s1 && sx < SOURCE_TILE {
                        sum += accumulator[sx] * ((((sx + 1) as f64).min(s1) - (sx as f64).max(s0)) / ratio) as f32;
                        sx += 1;
                    }
                    sum
                }).collect()
            }).collect();
            for (ty, values) in rows.into_iter().enumerate() {
                let offset = (row * block + ty) * width + column * block;
                grid[offset..offset + block].copy_from_slice(&values);
            }
            eprintln!("  {}: {:.1?}", path.display(), t.elapsed());
        }
    }
    Ok(grid)
}

/// Unit surface normals (east, north components as 0.5 + 0.5 n) from a
/// height grid, central differences on the sphere (x wraps, y clamps).
fn slopes_rg(heights: &[f32], width: usize, height: usize, exaggeration: f32) -> Vec<u8> {
    const RADIUS_M: f64 = 6_378_137.0;
    let dy_m = RADIUS_M * std::f64::consts::PI / height as f64;
    let mut out = vec![0u8; width * height * 2];
    out.par_chunks_mut(width * 2).enumerate().for_each(|(y, row)| {
        let latitude = (0.5 - (y as f64 + 0.5) / height as f64) * std::f64::consts::PI;
        let dx_m = (RADIUS_M * latitude.cos() * 2.0 * std::f64::consts::PI / width as f64).max(100.0);
        let north = heights[y.saturating_sub(1) * width..][..width].as_ptr();
        let south = heights[(y + 1).min(height - 1) * width..][..width].as_ptr();
        let span_y = if y == 0 || y == height - 1 { 1.0 } else { 2.0 };
        for x in 0..width {
            let (h_east, h_west) = (heights[y * width + (x + 1) % width], heights[y * width + (x + width - 1) % width]);
            let (h_north, h_south) = unsafe { (*north.add(x), *south.add(x)) };
            let slope_east = (h_east - h_west) as f64 / (2.0 * dx_m) * exaggeration as f64;
            let slope_north = (h_north - h_south) as f64 / (span_y * dy_m) * exaggeration as f64;
            let length = (slope_east * slope_east + slope_north * slope_north + 1.0).sqrt();
            let (nx, ny) = (-slope_east / length, -slope_north / length);
            row[x * 2] = ((nx * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            row[x * 2 + 1] = ((ny * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    });
    out
}

fn half_heights(heights: &[f32], width: usize, height: usize) -> Vec<f32> {
    let (w, h) = (width / 2, height / 2);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, value) in row.iter_mut().enumerate() {
            let i = y * 2 * width + x * 2;
            *value = 0.25 * (heights[i] + heights[i + 1] + heights[i + width] + heights[i + width + 1]);
        }
    });
    out
}

fn bc5(rg: &[u8], width: usize, height: usize) -> Vec<u8> {
    const STRIP: usize = 64;
    let strips: Vec<Vec<u8>> = (0..height.div_ceil(STRIP)).into_par_iter().map(|strip| {
        let y0 = strip * STRIP;
        let rows = STRIP.min(height - y0);
        let surface = intel_tex_2::RgSurface { data: &rg[y0 * width * 2..(y0 + rows) * width * 2], width: width as u32, height: rows as u32, stride: width as u32 * 2 };
        intel_tex_2::bc5::compress_blocks(&surface)
    }).collect();
    strips.concat()
}

fn bake_relief_bc5(gebco: &Path, width: usize, exaggeration: f32, out: &Path) -> Result<()> {
    let started = Instant::now();
    let height = width / 2;
    let mut heights = gebco_heights(gebco, width)?;
    let mut payload = Vec::new();
    let mut mips = Vec::new();
    let (mut w, mut h) = (width, height);
    loop {
        mips.push(format!("{{\"byte_offset\": {}, \"width\": {w}, \"height\": {h}}}", payload.len()));
        payload.extend_from_slice(&bc5(&slopes_rg(&heights, w, h, exaggeration), w, h));
        if w / 2 < 4 || h / 2 < 4 || w % 8 != 0 || h % 8 != 0 || mips.len() == 16 {
            break;
        }
        heights = half_heights(&heights, w, h);
        w /= 2;
        h /= 2;
    }
    fs::write(out, &payload)?;
    let sidecar = format!(
        "{{\n  \"schema_version\": 1,\n  \"asset\": \"earth_native_preview_texture\",\n  \"width\": {width},\n  \"height\": {height},\n  \"row_stride_bytes\": {},\n  \"payload_bytes\": {},\n  \"pixel_format\": \"bc5\",\n  \"color_space\": \"linear\",\n  \"origin\": \"top_left\",\n  \"source_object_path\": \"GEBCO_2026/relief-normals-x{exaggeration}\",\n  \"source_width\": 86400,\n  \"source_height\": 43200,\n  \"source_blocks\": 1,\n  \"filter\": \"area\",\n  \"rgb_filter\": \"linear-light-srgb-v1\",\n  \"layout\": \"equirectangular\",\n  \"mips\": [\n    {}\n  ]\n}}\n",
        width / 4 * 16, payload.len(), mips.join(",\n    "));
    let mut sidecar_path = out.as_os_str().to_owned();
    sidecar_path.push(".json");
    fs::write(sidecar_path, sidecar)?;
    eprintln!("earth-bake: wrote {} ({width}x{height} relief, {} mips, {:.0} MB) in {:.1?}", out.display(), mips.len(), payload.len() as f64 / 1e6, started.elapsed());
    Ok(())
}
