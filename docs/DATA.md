# Texture pack and live data

The renderer reads a texture pack from its data directory (first match):
`$EARTH_NATIVE_DATA_DIR`, `$XDG_DATA_HOME/earth-native`,
`~/.local/share/earth-native`, `<prefix>/share/earth-native`,
`/usr/local/share/earth-native`, `/usr/share/earth-native`.
`active.json` names the texture set inside it; `weather/current.json` is the
live weather manifest written by the feed.

## Contents (8K build, ≈ 250 MB on disk)

| File | Format | Source |
| --- | --- | --- |
| `day-east.bc3`, `day-west.bc3` | BC3 sRGB + water mask, 2×4096² | NASA Blue Marble NG, Sept. 2004, topo + bathymetry |
| `night.bc4` | BC4 (sRGB-coded) 8192×4096 | NASA Black Marble 2016, grayscale lights |
| `clouds.bc4` | BC4 8192×4096 | NASA Blue Marble cloud composite |
| `height.bc4`, `normal-*.bgra` | BC4 / BGRA | GEBCO 2008 via NASA Visible Earth (derived normals) |
| `moon.bc1` | BC1 sRGB 4096×2048 | NASA SVS CGI Moon Kit (LRO) |
| `mercury.bc1` … `neptune.bc1` | BC1 sRGB, 8K/4K/2K | Solar System Scope (CC BY 4.0) |
| `saturn-ring.bgra` | radial strip | NASA/JPL ring boundaries, modelled optical depth |
| `stars.bc1` | BC1 sRGB 16384×8192, full mips | NASA SVS Deep Star Maps 2020 (16K HDR) |
| `tiling-noise.bgra` | 256² periodic fractal noise | generated |
| `provenance.json` | — | every source URL, SHA-256, download time and credit |

All maps are block-compressed offline with precomputed linear-light mip
chains (`pipeline/bcn.py`), so the GPU uploads blocks directly: with the
8K pack the renderer needs about 350 MiB of VRAM (the 500 m set
below ~560 MiB with its static virtual texture, ~2 GB without). Equirectangular rows are low-passed
along longitude by 1/cos(latitude) (`polar_resample`) so the poles do not
pinwheel.

## High-resolution Earth (500 m)

When present, these replace the 8K Earth maps (`src/data_dir.rs` finds them
in the data directory). They are baked offline by `tools/earth-bake`
(Rust, ISPC BC7/BC4/BC5 encoders):

| File | Format | Source |
| --- | --- | --- |
| `vt/earth-day-MM.earthvt` | BC7 virtual texture, 65536×32768, 256 px pages + 4 px gutters, water mask in alpha, ≈ 3 GB per month | NASA Blue Marble NG 500 m monthly "world" (2004, no baked relief), 8 tiles of 21600², water from GEBCO 2026 |
| `vt/earth-day.earthvt` | as above, one file for every month | Sentinel-2 cloudless 2016 (EOX) land, BMNG water mask (wettest month) and ocean (darkest month), BMNG January where Sentinel-2 has no data |
| `textures/night.bc4` | BC4 32768×16384, 13 mips | NASA Black Marble 2016 500 m grayscale |
| `textures/clouds.bc4` | BC4 32768×16384 | NASA Blue Marble cloud composite (1 km) |
| `textures/relief.bc5` | BC5 normals 32768×16384 | GEBCO 2026 15″ grid |

```sh
cargo build --release --manifest-path tools/earth-bake/Cargo.toml
earth-bake day-vt --bmng DIR --month 200409 --gebco DIR --out earth-day-09.earthvt
earth-bake rgba-vt --raw earth.rgba --out earth-day.earthvt   # any 65536x32768 RGBA, alpha = water
earth-bake gray-bc4 --tiles A1,B1,C1,D1,A2,B2,C2,D2 --grid 4x2 --width 32768 \
    --color-space srgb --name NASA/night --out night.bc4
earth-bake relief-bc5 --gebco DIR --width 32768 --out relief.bc5
```


`--gpu` (day-vt, rgba-vt) runs the same ISPC BC7 kernel as a wgpu compute
shader (the `block_compression` crate) instead of on the CPU cores: equal
quality, about 1.5x faster on an RTX 3080 Ti against 32 CPU threads.

With these, stream the three finest levels of the night lights, cloud map
and relief instead of keeping 1.4 GB of them resident:

```sh
earth-bake static-vt --textures <texture set>   # ~2 s: re-tiles the blocks, no re-encode
```

This writes `earth-static.earthvt` (three layers, 256 px tiles; 1.5 GB on
disk) and `night-tail.bc4`, `clouds-tail.bc4`, `relief-tail.bc5` (the levels
from 4096 px down, 23 MB, always resident). When all four are present the
renderer streams the rest on demand (`EARTH_NATIVE_STATIC_VT_BUDGET_MB`,
default 96) and blends between the two nearest levels; the full-size maps
are no longer loaded (VRAM 2.0 -> 0.85 GB with the 500 m set).

Sources: the Blue Marble NG "world" tiles
(`world.2004MM.3x21600x21600.{A1..D2}.png`, NASA Visible Earth, one record
per month), the Black Marble 2016 gray tiles
(`BlackMarble_2016_{A1..D2}_geo_gray.tif`, Earth Observatory record 144897)
and the GEBCO 2026 grid. None of this is in the release packs: a month is
~1.5-2 GB of PNG in and ~3 GB out.

Put the virtual textures in `<data dir>/vt/` (or the texture set itself) and
the `.bc4`/`.bc5` files, with their `.json` sidecars, in the active texture
set, next to the 8K maps they replace. The renderer loads the current
month's virtual texture, else the nearest baked month; `earth-native status`
reports its page cache as `vt_vram_mb`.

## Rebuilding the pack

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r pipeline/requirements.txt
python3 pipeline/data_pipeline.py build          # downloads ~600 MB of sources once
```

Sources are cached in `<data dir>/sources` and re-verified by SHA-256.

## Live weather, clouds, aerosol, sea ice and aurora

`pipeline/data_pipeline.py watch` (the `earth-native-weather` service)
refreshes every 30 minutes:

- **NOAA GFS 0.25° analysis** (only the needed GRIB2 messages are fetched by
  byte range): cloud fraction × cloud water, CAPE and precipitation (where
  lightning flashes), and low cloud.
- **NOAA GMGSI** hourly geostationary mosaics (visible and 10.7 µm infrared)
  for observed cloud cover, against clear-sky composites kept in
  `weather/clearsky.npz`. The visible image is stitched from five satellites
  that see the same ground from different sides, so its brightness steps at
  their seams (up to 0.4 in albedo at 93° E, where Himawari looks toward the
  Sun across Meteosat's sea); the step is measured across each seam and
  removed from the brighter side. A missing satellite segment arrives as
  counts of 255 (infrared) or 0 (visible) rather than the declared fill
  value; both are treated as missing, and the gap is filled from the GFS
  cloud field, feathered into the observation over ~2.5° so it leaves no
  seam.
- **NOAA GEFS-Aerosols analysis** (GOCART, 0.25°): aerosol optical depth at
  440, 550 and 645 nm.
- **EUMETSAT OSI SAF** daily sea-ice concentration (OSI-401, 10 km polar
  stereographic grids, both hemispheres, via MET Norway THREDDS).
- **NOAA SWPC OVATION** aurora probability, which places the auroral oval.

It writes two checksummed textures and tells the renderer to reload them:
`fields.bgra` (1440×720: R cloud, G CAPE, B precipitation, A aurora) and
`clouds.bgra` (4096×2048: R cloud cover, G sea-ice concentration,
B (Ångström exponent + 0.5)/3, A √(τ₅₅₀/4)). Each source is optional: without
it the renderer falls back to the NASA cloud map, the climatological haze or
open oceans, and aurora and lightning stay off.

`data_pipeline.py aurora-preview --kp 7` swaps tonight's OVATION oval for a
synthetic storm of that Kp (Feldstein-Starkov-style boundaries in magnetic
latitude and local time), labelled `SYNTHETIC` in the manifest, to review the
aurora rendering when the real oval is quiet. The next feed update (at most
30 minutes) or `data_pipeline.py weather` restores the observation.
