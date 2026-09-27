use std::{fs, path::PathBuf};

use jpeg_encoder::{ColorType, Encoder};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::{mpsc, OnceLock};

struct CaptureJob {
    name: String,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    order: PixelOrder,
    frame: serde_json::Value,
}

/// At most two queued images plus one encoding job; no unbounded capture memory.
pub fn submit_frame(name: &str, width: u32, height: u32, pixels: Vec<u8>,
    order: PixelOrder, frame: serde_json::Value) -> Result<bool, Box<dyn std::error::Error>> {
    static WRITER: OnceLock<mpsc::SyncSender<CaptureJob>> = OnceLock::new();
    let sender = WRITER.get_or_init(|| {
        let (sender, receiver) = mpsc::sync_channel::<CaptureJob>(2);
        std::thread::spawn(move || {
            while let Ok(job) = receiver.recv() {
                let result = write_jpeg(&job.name, job.width, job.height, &job.pixels, job.order)
                    .and_then(|path| {
                        write_metadata(&path, job.width, job.height, &job.pixels, &job.frame)?;
                        Ok(path)
                    });
                match result {
                    Ok(path) => eprintln!("earth-native: completed capture {}", path.display()),
                    Err(error) => {
                        eprintln!("earth-native: capture failed: {error}");
                        let path = output_path(&job.name).with_extension("json");
                        let temporary = path.with_extension("json.tmp");
                        let _ = fs::write(&temporary, serde_json::json!({"error": error.to_string()}).to_string());
                        let _ = fs::rename(temporary, path);
                    }
                }
            }
        });
        sender
    });
    match sender.try_send(CaptureJob { name: name.to_owned(), width, height, pixels, order, frame }) {
        Ok(()) => Ok(true),
        Err(mpsc::TrySendError::Full(_)) => Ok(false),
        Err(mpsc::TrySendError::Disconnected(_)) => Err("capture worker stopped".into()),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelOrder {
    Bgra,
    Rgba,
}

pub fn write_jpeg(
    output_name: &str,
    width: u32,
    height: u32,
    pixels: &[u8],
    order: PixelOrder,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let pixel_count = width
        .checked_mul(height)
        .ok_or("debug capture dimensions overflow")? as usize;
    let expected = pixel_count
        .checked_mul(4)
        .ok_or("debug capture byte count overflows")?;
    if pixels.len() != expected {
        return Err(format!(
            "debug capture expected {expected} bytes, received {}",
            pixels.len()
        )
        .into());
    }
    let width = u16::try_from(width).map_err(|_| "debug capture width exceeds JPEG limits")?;
    let height = u16::try_from(height).map_err(|_| "debug capture height exceeds JPEG limits")?;

    // jpeg-encoder consumes 4-byte BGRA/RGBA directly (alpha ignored), so the
    // ~25 MB RGB swizzle temp is deleted; channel math is identical.
    let color = match order {
        PixelOrder::Bgra => ColorType::Bgra,
        PixelOrder::Rgba => ColorType::Rgba,
    };
    let mut encoded = Vec::new();
    Encoder::new(&mut encoded, 95).encode(pixels, width, height, color)?;
    let path = output_path(output_name);
    let temporary = path.with_extension("jpg.tmp");
    fs::write(&temporary, &encoded)?;
    fs::rename(temporary, &path)?;
    Ok(path)
}

/// The sidecar is the completion marker: consumers never inspect a partial frame.
pub fn write_metadata(
    image: &std::path::Path,
    width: u32,
    height: u32,
    pixels: &[u8],
    frame: &impl Serialize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut minimum = 255_u8;
    let mut maximum = 0_u8;
    let mut sum = 0_u64;
    let mut nonblack = 0_u64;
    for pixel in pixels.chunks_exact(4) {
        nonblack += u64::from(pixel[..3].iter().any(|&v| v > 3));
        for &v in &pixel[..3] {
            minimum = minimum.min(v);
            maximum = maximum.max(v);
            sum += u64::from(v);
        }
    }
    let count = u64::from(width) * u64::from(height);
    let metadata = serde_json::json!({
        "image": image, "width": width, "height": height,
        "pixel_min": minimum, "pixel_max": maximum,
        "mean_channel": sum as f64 / (count * 3).max(1) as f64,
        "nonblack_fraction": nonblack as f64 / count.max(1) as f64,
        "raw_pixel_sha256": format!("{:x}", Sha256::digest(pixels)),
        "frame": frame,
    });
    let path = image.with_extension("json");
    let temporary = image.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(&metadata)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub fn output_path(output_name: &str) -> PathBuf {
    let mut safe_name = output_name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if safe_name.is_empty() {
        safe_name.push_str("output");
    }
    PathBuf::from(format!("/tmp/earth-native-{safe_name}.jpg"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_path_is_monitor_named_and_confined_to_tmp() {
        assert_eq!(
            output_path("DP-1"),
            PathBuf::from("/tmp/earth-native-DP-1.jpg")
        );
        assert_eq!(
            output_path("../../unsafe name"),
            PathBuf::from("/tmp/earth-native-______unsafe_name.jpg")
        );
    }

    #[test]
    fn bgra_conversion_produces_a_jpeg() {
        let pixels = [0_u8, 0, 255, 255, 0, 255, 0, 255];
        let mut encoded = Vec::new();
        Encoder::new(&mut encoded, 90)
            .encode(&pixels, 2, 1, ColorType::Bgra)
            .unwrap();
        assert_eq!(&encoded[..2], &[0xff, 0xd8]);
        assert_eq!(&encoded[encoded.len() - 2..], &[0xff, 0xd9]);
    }

    #[test]
    fn direct_four_channel_jpeg_matches_rgb_conversion() {
        let rgb = [255, 0, 0, 0, 255, 0, 0, 0, 255, 37, 128, 219];
        let mut reference = Vec::new();
        Encoder::new(&mut reference, 95).encode(&rgb, 2, 2, ColorType::Rgb).unwrap();
        for color in [ColorType::Bgra, ColorType::Rgba] {
            let pixels: Vec<u8> = rgb.chunks_exact(3).enumerate().flat_map(|(i, p)| {
                let alpha = (i * 63) as u8;
                match color {
                    ColorType::Bgra => [p[2], p[1], p[0], alpha],
                    _ => [p[0], p[1], p[2], alpha],
                }
            }).collect();
            let mut actual = Vec::new();
            Encoder::new(&mut actual, 95).encode(&pixels, 2, 2, color).unwrap();
            assert_eq!(actual, reference);
        }
    }

    #[test]
    fn capture_worker_publishes_image_before_completion_metadata() {
        let name = format!("test-worker-{}", std::process::id());
        let path = output_path(&name);
        let metadata = path.with_extension("json");
        let _ = fs::remove_file(&metadata);
        assert!(submit_frame(&name, 2, 1, vec![0, 0, 255, 255, 0, 255, 0, 255],
            PixelOrder::Bgra, serde_json::json!({"unix_seconds": 123})).unwrap());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !metadata.exists() {
            assert!(std::time::Instant::now() < deadline, "capture worker timed out");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let report: serde_json::Value = serde_json::from_slice(&fs::read(&metadata).unwrap()).unwrap();
        assert_eq!(report["pixel_max"], 255);
        assert_eq!(report["nonblack_fraction"], 1.0);
        assert_eq!(report["frame"]["unix_seconds"], 123);
        assert_eq!(&fs::read(&path).unwrap()[..2], &[0xff, 0xd8]);
        fs::remove_file(path).unwrap();
        fs::remove_file(metadata).unwrap();
    }

    #[test]
    fn capture_rejects_wrong_pixel_length_and_overflow() {
        assert!(write_jpeg("invalid", 2, 1, &[0; 4], PixelOrder::Rgba).is_err());
        assert!(write_jpeg("invalid", u32::MAX, u32::MAX, &[], PixelOrder::Bgra).is_err());
    }
}
