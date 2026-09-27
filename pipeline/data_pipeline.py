# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=11", "numpy>=2", "eccodes>=2.38", "OpenEXR>=3.3"]
# ///
"""NASA texture preparation and bounded NOAA GFS weather updates.

Downloads are data, never code. Everything is written under the data
directory ($EARTH_NATIVE_DATA_DIR, default $XDG_DATA_HOME/earth-native),
which the renderer discovers on its own.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageFilter

import bcn

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("EARTH_NATIVE_DATA_DIR")
            or Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "earth-native")
RAW = DATA / "sources"
GITHUB_REV = "11ebb4ee043715aefbba6aeec8a61746fad67fa7"
LIBRARY = f"https://raw.githubusercontent.com/nasa/NASA-3D-Resources/{GITHUB_REV}/Images%20and%20Textures/"
SOURCES = {
    "day": ("https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73801/world.topo.bathy.200409.3x21600x10800.jpg", "NASA Blue Marble Next Generation, September 2004, topography and bathymetry, 21600x10800; monthly MODIS composite"),
    "night": ("https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144897/BlackMarble_2016_3km_gray_geo.tif", "NASA Earth Observatory / VIIRS Black Marble 2016, 3 km grayscale lights only; static composite"),
    "height": ("https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_21600x10800.png", "NASA Visible Earth / GEBCO 2008 elevation visualization"),
    "cloud-detail": ("https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57747/cloud_combined_8192.tif", "NASA Blue Marble cloud composite, 8192x4096; static historical clouds, not live observations"),
    "moon": ("https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_16bit_srgb_4k.tif", "NASA SVS CGI Moon Kit, LRO WAC/LOLA, December 2025 version"),
}
SSS = "https://www.solarsystemscope.com/textures/download/"
SSS_CREDIT = "Solar System Scope 8K texture, CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/); derived from NASA imagery with tuned color"
for key in ["jupiter", "mars", "saturn", "mercury"]:
    SOURCES[key] = (SSS + f"8k_{key}.jpg", f"{SSS_CREDIT} ({key})")
# Venus as seen in visible light is its cloud deck, not the radar surface.
SOURCES["venus"] = (SSS + "4k_venus_atmosphere.jpg", f"{SSS_CREDIT} (Venus cloud tops)")
for key in ["uranus", "neptune"]:
    SOURCES[key] = (SSS + f"2k_{key}.jpg", f"{SSS_CREDIT} ({key}, Voyager 2 derived)")
SOURCES["stars"] = ("https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_16k.exr",
                    "NASA SVS Deep Star Maps 2020 (Hipparcos-2, Tycho-2, Gaia DR2), 16K HDR, J2000 equatorial")
LARGE_SOURCES = {"stars"}
Image.MAX_IMAGE_PIXELS = 250_000_000


def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".incoming-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2) + "\n").encode())


def download(url, limit=100_000_000, byte_range=None):
    headers = {"User-Agent": "earth-native-nasa-data/1.0"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    with urlopen(Request(url, headers=headers), timeout=25) as response:
        if byte_range and (response.status != 206 or response.headers.get("Content-Range", "").split("/")[0] != f"bytes {byte_range[0]}-{byte_range[1]}"):
            raise ValueError("server did not honor the exact GRIB byte range")
        if int(response.headers.get("Content-Length", "0")) > limit:
            raise ValueError("download exceeds size limit")
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError("download exceeds size limit")
    if byte_range and len(content) != byte_range[1] - byte_range[0] + 1:
        raise ValueError("incomplete GRIB byte range")
    return content


def file_sha256(path):
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 22):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def download_to_file(url, path, limit=2_000_000_000):
    """Stream a large source to disk atomically (no in-memory copy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".incoming-" + path.name)
    request = Request(url, headers={"User-Agent": "earth-native-nasa-data/1.0"})
    with urlopen(request, timeout=60) as response, open(temporary, "wb") as stream:
        total = 0
        while chunk := response.read(1 << 22):
            total += len(chunk)
            if total > limit:
                raise ValueError("download exceeds size limit")
            stream.write(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def star_panorama(directory, exr_path):
    """16K HDR star map -> sRGB BC1 with a full mip chain for the star pass.

    Radiance is scaled so the brightest ~0.002 % of texels (the naked-eye
    stars' cores) reach white, then sRGB-encoded: faint Gaia stars and the
    Milky Way keep their relative brightness instead of a pre-clipped JPEG's.
    """
    import OpenEXR
    with OpenEXR.File(str(exr_path)) as exr:
        channels = exr.channels()
        rgb = next((channels[key].pixels for key in ("RGB", "RGBA") if key in channels), None)
        if rgb is None:
            rgb = np.dstack([channels[c].pixels for c in ("R", "G", "B")])
    rgb = np.asarray(rgb, dtype=np.float32)[:, :, :3]
    height, width = rgb.shape[:2]
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    white = float(np.percentile(luminance[::4, ::4], 99.998))
    srgb = bcn.linear_to_srgb(np.clip(rgb / max(white, 1e-12), 0.0, 1.0))
    image = np.round(srgb * 255).astype(np.uint8)
    del rgb, luminance, srgb
    payload, mips = bcn.encode_full_chain(image, "bc1", [0, 1, 2])
    path = directory / "stars.bc1"
    atomic_bytes(path, payload)
    atomic_json(Path(str(path) + ".json"), {
        "schema_version": 1, "asset": "star_panorama", "width": width, "height": height,
        "row_stride_bytes": ((width + 3) // 4) * 8, "payload_bytes": len(payload),
        "pixel_format": "bc1", "mip_count": len(mips), "color_space": "srgb",
        "projection": "equirectangular", "origin": "top_left",
        "longitude_at_u0_degrees": -180, "latitude_at_v0_degrees": 90})
    return {"file": path.name, "sha256": hashlib.sha256(payload).hexdigest(), "width": width, "height": height}


def fetch():
    for name, (url, credit) in SOURCES.items():
        path = RAW / name
        record = path.with_suffix(".json")
        if path.exists() and record.exists():
            info = json.loads(record.read_text())
            if info["url"] == url and file_sha256(path)[0] == info["sha256"]:
                print(f"{name}: verified cached source", flush=True)
                continue
        if name in LARGE_SOURCES:
            download_to_file(url, path)
            content_hash, size = file_sha256(path)
            atomic_json(record, {"url": url, "credit": credit, "sha256": content_hash,
                                 "downloaded_utc": datetime.now(timezone.utc).isoformat(), "bytes": size})
            print(f"{name}: downloaded {size / 1e6:.1f} MB", flush=True)
            continue
        content = download(url)
        with Image.open(io.BytesIO(content)) as source:
            source.verify()
        atomic_bytes(path, content)
        atomic_json(record, {"url": url, "credit": credit, "sha256": hashlib.sha256(content).hexdigest(),
                             "downloaded_utc": datetime.now(timezone.utc).isoformat(), "bytes": len(content)})
        print(f"{name}: downloaded {len(content) / 1e6:.1f} MB", flush=True)


def resize_rgb(image, size):
    """Downsample colour in linear light, alpha/data separately."""
    image = image.convert("RGB")
    if image.size == size:
        return np.asarray(image).copy()
    channels = []
    for channel in image.split():
        srgb = np.asarray(channel, dtype=np.float32) / 255
        linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        linear = np.asarray(Image.fromarray(linear).resize(size, Image.Resampling.LANCZOS))
        linear = np.clip(linear, 0, 1)
        srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
        channels.append(np.round(srgb * 255).astype(np.uint8))
    return np.stack(channels, axis=-1)



def wrap_seam(rgb, width=8):
    """Cross-fade the antimeridian columns to a common value.

    Wrapped sampling blends column W-1 with column 0; when the source edges
    disagree (e.g. Mercury's 74-mean gap) the step shows as a vertical seam.
    Both edges ramp to their per-row common mean over ``width`` columns, so
    the wraparound join is exact while interior pixels are untouched.
    """
    rgb = rgb.copy()
    left = rgb[:, :width].astype(np.float32)
    right = rgb[:, -width:].astype(np.float32)
    common = ((left + right) / 2).mean(axis=1, keepdims=True)
    ramp = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    rgb[:, :width] = np.round(common * (1 - ramp) + left * ramp).astype(np.uint8)
    rgb[:, -width:] = np.round(right * (1 - ramp) + common * ramp).astype(np.uint8)
    return rgb


def polar_resample(image):
    """Low-pass each equirectangular row along longitude by 1/cos(latitude).

    Near a pole the whole row is a tiny circle, so source detail along U
    becomes radial pinwheel streaks. A wrapped box of 1/cos(lat) texels gives
    every row the equator's ground resolution; rows below ~60 deg are untouched.
    """
    out = image.copy()
    height, width = image.shape[:2]
    for y in range(height):
        latitude = np.radians(90.0 - (y + 0.5) * 180.0 / height)
        k = int(round(min(1.0 / max(np.cos(latitude), 1e-6), width / 2)))
        if k < 2:
            continue
        row = image[y].astype(np.float32)
        padded = np.concatenate((row[-(k // 2) - 1:], row, row[:k]), axis=0)
        csum = np.cumsum(padded, axis=0)
        start = np.arange(width)
        out[y] = np.round((csum[start + k] - csum[start]) / k).astype(image.dtype)
    return out


def periodic_noise(size, beta=2.0, seed=80):
    """Tileable fractal noise (1/f^beta spectrum on a torus), uint8 gray.

    Replaces the Unreal-pack tiling noise so the shipped data is free of
    third-party assets; periodic by construction, so REPEAT sampling is seamless.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fft2(rng.standard_normal((size, size)))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0
    field = np.real(np.fft.ifft2(spectrum / radius ** (beta / 2)))
    field = (field - field.min()) / (field.max() - field.min())
    return np.round(field * 255).astype(np.uint8)


def preview(directory, name, rgb, color_space="linear", alpha=None):
    height, width = rgb.shape[:2]
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb[:, :, ::-1]
    rgba[:, :, 3] = 255 if alpha is None else alpha
    path = directory / f"{name}.bgra"
    payload = rgba.tobytes()
    atomic_bytes(path, payload)
    metadata = {"schema_version": 1, "asset": "earth_native_preview_texture", "width": width,
                "height": height, "row_stride_bytes": width * 4, "payload_bytes": len(payload),
                "pixel_format": "bgra8", "color_space": color_space, "origin": "top_left",
                "source_object_path": f"NASA/{name}", "source_width": width, "source_height": height,
                "source_blocks": 1, "filter": "lanczos", "rgb_filter": "linear-light-srgb-v1",
                "layout": "equirectangular"}
    atomic_json(Path(str(path) + ".json"), metadata)
    return {"file": path.name, "sha256": hashlib.sha256(payload).hexdigest(), "width": width, "height": height}


def compressed(directory, name, image, pixel_format, color_space="linear", srgb_channels=()):
    """Block-compressed preview with a precomputed mip chain (see bcn.py)."""
    if image.shape[0] % 4 or image.shape[1] % 4:
        rgba = image.ndim == 3 and image.shape[2] == 4
        return preview(directory, name, image[..., :3] if rgba else image, color_space,
                       image[..., 3] if rgba else None)
    payload, mips = bcn.encode_mipped(image, pixel_format, list(srgb_channels))
    path = directory / f"{name}.{pixel_format}"
    atomic_bytes(path, payload)
    height, width = image.shape[:2]
    metadata = {"schema_version": 1, "asset": "earth_native_preview_texture", "width": width,
                "height": height, "row_stride_bytes": (width // 4) * (8 if pixel_format != "bc3" else 16),
                "payload_bytes": len(payload), "pixel_format": pixel_format, "color_space": color_space,
                "origin": "top_left", "source_object_path": f"NASA/{name}", "source_width": width,
                "source_height": height, "source_blocks": 1, "filter": "lanczos",
                "rgb_filter": "linear-light-srgb-v1", "layout": "equirectangular", "mips": mips}
    atomic_json(Path(str(path) + ".json"), metadata)
    return {"file": path.name, "sha256": hashlib.sha256(payload).hexdigest(), "width": width, "height": height}


def build(width):
    fetch()
    size = (width, width // 2)
    directory = Path(tempfile.mkdtemp(prefix=f"textures-{width}-", dir=DATA))
    assets = {}
    with Image.open(RAW / "day") as source:
        day = polar_resample(resize_rgb(source, size))
    # The GEBCO grayscale visualization is suitable for relief, not calibrated DEM metres.
    with Image.open(RAW / "height") as source:
        height_map = np.asarray(source.convert("L").resize(size, Image.Resampling.LANCZOS)).copy()
    # Spectral water classification is derived, not an official shoreline dataset.
    # BMNG bathymetry tints shallow shelves cyan, so the old blue-dominance
    # test alone missed them; GEBCO's zero-elevation ocean catches the rest.
    water = (((day[:, :, 2].astype(np.float32) > day[:, :, 0] * 1.35) &
              (day[:, :, 2].astype(np.float32) > day[:, :, 1] * 1.08)) | (height_map <= 1)).astype(np.uint8) * 255
    for name, part in [("west", slice(0, width // 2)), ("east", slice(width // 2, width))]:
        rgba = np.dstack((day[:, part], water[:, part]))
        assets[f"day-{name}"] = compressed(directory, f"day-{name}", rgba, "bc3", "srgb", (0, 1, 2))
    assets["height"] = compressed(directory, "height", np.asarray(Image.fromarray(height_map).resize((4096, 2048), Image.Resampling.LANCZOS)), "bc4")
    # Unexaggerated angular slopes from the relief visualization; no invented high-frequency mountains.
    h = np.asarray(Image.fromarray(height_map).resize((2048, 1024), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.4
    dy = np.gradient(h, axis=0) * 0.8
    normals = np.stack((-dx, -dy, np.ones_like(h)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.round((normals * 0.5 + 0.5) * 255).astype(np.uint8)
    for name, part in [("west", slice(0, 1024)), ("east", slice(1024, 2048))]:
        assets[f"normal-{name}"] = preview(directory, f"normal-{name}", normals[:, part])
    del day, water, normals, h, dx, dy
    # Grayscale Black Marble: lights on true black, no cartographic background
    # to subtract. The renderer colours them from intensity.
    with Image.open(RAW / "night") as source:
        night = polar_resample(resize_rgb(source.convert("L").convert("RGB"), size)[:, :, 0])
    assets["night"] = compressed(directory, "night", night, "bc4", "srgb", (0,))
    del night
    with Image.open(RAW / "cloud-detail") as source:
        clouds = np.asarray(source.convert("L").resize(size, Image.Resampling.LANCZOS)).copy()
    clouds = polar_resample(clouds)
    assets["clouds"] = compressed(directory, "clouds", wrap_seam(np.dstack([clouds] * 3))[:, :, 0], "bc4")
    del clouds
    for name in ["moon", "jupiter", "mercury", "mars", "saturn", "venus", "uranus", "neptune"]:
        with Image.open(RAW / name) as source:
            # Solar System Scope ships Mercury in tuned true color; keep it.
            # Planets may use full 8K width (8192x4096); Moon's 4K source caps itself.
            target_width = min(8192, source.width)
            target_width -= target_width % 2
            colour = wrap_seam(resize_rgb(source, (target_width, target_width // 2)))
        assets[name] = compressed(directory, name, colour, "bc1", "srgb", (0, 1, 2))
    # Ancillary authored control maps are replaced by neutral values in the direct data shader.
    for name, value in [("desert", 255)]:
        assets[name] = preview(directory, name, np.full((4, 4), value, dtype=np.uint8), "srgb" if name == "noise" else "linear")
    assets["tiling-noise"] = preview(directory, "tiling-noise", periodic_noise(256), "srgb")
    assets["cloud-empty"] = preview(directory, "cloud-empty", np.zeros((4, 4), dtype=np.uint8), alpha=np.zeros((4, 4), dtype=np.uint8))
    # NASA/JPL main-ring boundaries; a modeled optical-depth strip, not a photograph.
    radius = np.linspace(74658, 136775, 4096)
    tau = np.where(radius < 92000, 0.12, np.where(radius < 117580, 1.2, np.where(radius < 122170, 0.035, 0.5)))
    tau *= 0.85 + 0.15 * np.sin(radius / 95) * np.sin(radius / 37)
    tau[(radius > 133410) & (radius < 133740)] = 0.001
    tau[(radius > 136480) & (radius < 136520)] = 0.001
    ring_rgb = np.broadcast_to(np.array([218, 211, 195], dtype=np.uint8), (1, 4096, 3))
    assets["saturn-ring"] = preview(directory, "saturn-ring", ring_rgb, "srgb", np.round((1 - np.exp(-tau))[None, :] * 255).astype(np.uint8))
    assets["stars"] = star_panorama(directory, RAW / "stars")
    provenance = {"schema_version": 1, "source": "NASA", "assets": assets,
                  "sources": {name: json.loads((RAW / name).with_suffix(".json").read_text()) for name in SOURCES},
                  "notes": ["Static historical mosaics, not current imagery", "Water mask and terrain normals are derived approximations",
                            "GEBCO visualization is not a calibrated DEM",
                            "Solar System Scope planet maps are CC-BY 4.0 and keep their published colour calibration"]}
    atomic_json(directory / "provenance.json", provenance)
    atomic_json(DATA / "active.json", {"textures": directory.name})
    print(f"NASA textures ready: {directory}")


def select_ranges(index):
    rows = [line.split(":") for line in index.splitlines() if line]
    wanted = {("TCDC", "entire atmosphere"): "cloud", ("CAPE", "surface"): "cape", ("PRATE", "surface"): "rain"}
    wanted[("CWAT", "entire atmosphere (considered as a single layer)")] = "water"
    found = {}
    for row, following in zip(rows, rows[1:]):
        name = wanted.get(tuple(row[3:5]))
        if name:
            found[name] = (int(row[1]), int(following[1]) - 1)
    if set(found) != set(wanted.values()):
        raise ValueError("GFS cycle is missing a required field")
    return found


def decode_grib(payload, expected_utc=None):
    import eccodes
    handle = eccodes.codes_new_from_message(payload)
    try:
        width = eccodes.codes_get(handle, "Ni")
        height = eccodes.codes_get(handle, "Nj")
        if (width, height) != (1440, 721) or eccodes.codes_get(handle, "gridType") != "regular_ll":
            raise ValueError("unexpected GFS geometry")
        if eccodes.codes_get(handle, "latitudeOfFirstGridPointInDegrees") != 90 or eccodes.codes_get(handle, "longitudeOfFirstGridPointInDegrees") != 0:
            raise ValueError("unexpected GFS origin")
        if eccodes.codes_get(handle, "iScansNegatively") or eccodes.codes_get(handle, "jScansPositively"):
            raise ValueError("unexpected GFS scanning order")
        if expected_utc is not None:
            actual = str(eccodes.codes_get(handle, "validityDate")) + f"{eccodes.codes_get(handle, 'validityTime'):04d}"
            if actual != expected_utc.strftime("%Y%m%d%H%M"):
                raise ValueError("GFS payload timestamp disagrees with cycle")
        values = eccodes.codes_get_values(handle).reshape(height, width)
        if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 1e9:
            raise ValueError("GFS contains missing or nonfinite values")
        # Nodes at 0/.25 deg -> pixel centres at -179.875/...; periodic longitude.
        values = 0.5 * (values + np.roll(values, -1, axis=1))
        values = 0.5 * (values[:-1] + values[1:])
        return np.roll(values, width // 2, axis=1)
    finally:
        eccodes.codes_release(handle)


def aurora_field():
    url = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
    product = json.loads(download(url, limit=3_000_000))
    coordinates = np.asarray(product["coordinates"], dtype=np.float64)
    if coordinates.shape[1:] != (3,) or len(coordinates) < 64000 or not np.all(np.isfinite(coordinates)):
        raise ValueError("invalid OVATION coordinates")
    lon, lat, probability = coordinates.T
    if np.any((lon < 0) | (lon >= 360) | (lat < -90) | (lat > 90) | (probability < 0) | (probability > 100)):
        raise ValueError("OVATION coordinate out of range")
    keys = (90 - lat) * 360 + lon
    if np.any(lon != np.floor(lon)) or np.any(lat != np.floor(lat)) or len(np.unique(keys)) != len(keys):
        raise ValueError("OVATION coordinates must be unique integer-degree nodes")
    grid = np.zeros((181, 360), dtype=np.float32)
    grid[(90 - lat).astype(int), lon.astype(int)] = probability / 100
    grid = np.roll(grid, 180, axis=1)
    field = np.asarray(Image.fromarray(grid).resize((1440, 720), Image.Resampling.BILINEAR))
    forecast = int(datetime.fromisoformat(product["Forecast Time"].replace("Z", "+00:00")).timestamp())
    return np.round(field * 255).astype(np.uint8), forecast


def weather():
    now = datetime.now(timezone.utc)
    cycle = now.replace(hour=now.hour // 6 * 6, minute=0, second=0, microsecond=0)
    existing_path = DATA / "weather" / "current.json"
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else {}
    last_error = None
    for offset in range(9):
        candidate = cycle - timedelta(hours=6 * offset)
        base = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{candidate:%Y%m%d}/{candidate:%H}/atmos/gfs.t{candidate:%H}z.pgrb2.0p25.f000"
        try:
            index = download(base + ".idx", limit=200_000).decode()
            ranges = select_ranges(index)
            if existing.get("valid_unix_utc") == int(candidate.timestamp()) and existing.get("packing_version") == 3:
                old = Path(existing["texture"]).read_bytes()
                if hashlib.sha256(old).hexdigest() != existing["sha256"]:
                    raise ValueError("cached weather checksum mismatch")
                packed = np.frombuffer(old, np.uint8).reshape(720, 1440, 4)[:, :, 2::-1].copy()
            else:
                fields = {name: decode_grib(download(base, limit=10_000_000, byte_range=span), candidate) for name, span in ranges.items()}
                # Extinction for a mixed ice/liquid column; do not render thin
                # 100%-coverage cirrus as an opaque white cloud deck.
                cloud = np.clip(fields["cloud"] / 100, 0, 1) * (1 - np.exp(-75 * np.maximum(fields["water"], 0)))
                # Smooth grid-cell boundaries once offline, with periodic longitude.
                padded = np.pad(np.round(cloud * 255).astype(np.uint8), ((0, 0), (3, 3)), mode="wrap")
                cloud = np.asarray(Image.fromarray(padded).filter(ImageFilter.GaussianBlur(0.7)))[:, 3:-3] / 255
                cape = np.clip(fields["cape"] / 4000, 0, 1)
                rain = np.clip(np.log1p(np.maximum(fields["rain"], 0) * 3600) / math.log(51), 0, 1)
                packed = np.round(np.stack((cloud, cape, rain), axis=-1) * 255).astype(np.uint8)
            try:
                aurora, aurora_utc = aurora_field()
            except (OSError, ValueError, KeyError) as error:
                print(f"OVATION unavailable, aurora disabled: {error}", file=sys.stderr, flush=True)
                aurora, aurora_utc = np.zeros((720, 1440), dtype=np.uint8), 0
            generation = DATA / "weather" / f"{candidate:%Y%m%dT%H}-{int(now.timestamp())}"
            asset = preview(generation, "fields", packed, alpha=aurora)
            metadata = {"schema_version": 1, "packing_version": 3, "source": "NOAA-GFS", "kind": "model-analysis", "url": base,
                        "valid_unix_utc": int(candidate.timestamp()), "downloaded_unix_utc": int(now.timestamp()),
                        "aurora_unix_utc": aurora_utc, "aurora_source": "NOAA-SWPC-OVATION-forecast",
                        "texture": str(generation / asset["file"]), "sha256": asset["sha256"],
                        "width": 1440, "height": 720, "lightning": "simulated-from-CAPE-and-precipitation",
                        "channels": {"r": "cloud fraction * (1-exp(-75 * column cloud water kg/m2))", "g": "CAPE / 4000 J/kg", "b": "log1p(precipitation mm/h) / log(51)", "a": "OVATION aurora probability / 100 percent"}}
            atomic_json(existing_path, metadata)
            snapshots = sorted(path for path in existing_path.parent.iterdir()
                               if path.is_dir() and re.fullmatch(r"\d{8}T\d{2}-\d{10}", path.name))
            for obsolete in snapshots[:-8]:
                if {path.name for path in obsolete.iterdir()} == {"fields.bgra", "fields.bgra.json"}:
                    (obsolete / "fields.bgra").unlink()
                    (obsolete / "fields.bgra.json").unlink()
                    obsolete.rmdir()
            print(f"Weather ready: NOAA GFS {candidate.isoformat()}", flush=True)
            return metadata
        except (HTTPError, OSError, ValueError) as error:
            last_error = error
            print(f"GFS {candidate:%Y-%m-%d %H}: {error}", file=sys.stderr, flush=True)
    raise RuntimeError(f"No complete recent GFS cycle; retaining last good data: {last_error}")


def notify_renderer():
    path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "earth-native.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(10)
        stream.connect(str(path))
        stream.sendall(b"weather reload\n")
        response = stream.recv(4096).decode().strip()
        if not response.startswith("ok"):
            raise RuntimeError(response)
        print(response, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "build", "weather", "watch", "test", "verify"])
    parser.add_argument("--width", type=int, choices=[4096, 8192], default=8192)
    args = parser.parse_args()
    if args.command == "verify":
        from weather_verify import verify
        verify()
    elif args.command == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT), "test_data_pipeline.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    elif args.command == "fetch":
        fetch()
    elif args.command == "build":
        build(args.width)
    elif args.command == "weather":
        weather()
    else:
        while True:
            try:
                weather()
                notify_renderer()
            except Exception as error:
                print(f"Weather update failed; retaining last good data: {error}", file=sys.stderr, flush=True)
            time.sleep(1800)


if __name__ == "__main__":
    main()
