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
chains (`pipeline/bcn.py`), so the GPU uploads blocks directly: the whole
Earth view needs about 270 MiB of VRAM. Equirectangular rows are low-passed
along longitude by 1/cos(latitude) (`polar_resample`) so the poles do not
pinwheel.

## Rebuilding the pack

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r pipeline/requirements.txt
python3 pipeline/data_pipeline.py build          # downloads ~600 MB of sources once
```

Sources are cached in `<data dir>/sources` and re-verified by SHA-256.

## Live weather and aurora

`pipeline/data_pipeline.py watch` (the `earth-native-weather` service)
refreshes every 30 minutes:

- **NOAA GFS 0.25° analysis** (only the needed GRIB2 messages are fetched by
  byte range): cloud fraction × cloud water, CAPE and precipitation, which
  drive where lightning flashes.
- **NOAA SWPC OVATION** aurora probability, which places the auroral oval.

It writes a checksummed 1440×720 field texture and tells the renderer to
reload it. Without the feed, aurora and lightning simply stay off.
