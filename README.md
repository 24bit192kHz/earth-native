<div align="center">

# earth-native

**The real Earth — and every planet — live on your desktop.**
A physically based, native Vulkan desktop background for **Wayland** and **Xorg**,
driven by NASA imagery, live NOAA data and exact ephemerides.

![Earth over Africa, rendered by earth-native](docs/previews/earth-day.jpg)

[Features](#features) · [Gallery](#gallery) · [Install](#install) · [Use](#use) · [How it works](docs/PHYSICS.md) · [Data](docs/DATA.md)

</div>

---

## Features

- **The sky, right now.** Sun, Moon and planets come from Astronomy Engine and
  match JPL Horizons to the arc-minute; day and night, the terminator, lunar
  phase, and even **solar eclipse shadows** are where they really are.
- **NASA imagery.** Blue Marble Next Generation surface, Black Marble city
  lights, the Blue Marble cloud map, LRO Moon, and a **16K Hipparcos/Tycho/Gaia
  star map** — calibrated against real DSCOVR/EPIC photos of Earth.
- **Real atmosphere.** Rayleigh scattering with published optical depths,
  aerosols, orange twilight at the terminator, a thin blue limb, Cox–Munk
  sunglint on the oceans, moonlit clouds and night-side airglow.
- **Live space weather.** A volumetric **aurora** placed by NOAA's OVATION
  forecast, with green and red emission layers and moving curtains; lightning
  flashes where NOAA GFS reports thunderstorms.
- **Every planet.** Mercury, Venus, Mars, Jupiter, Saturn (with rings),
  Uranus and Neptune plus the Moon — each with its IAU axis and rotation,
  real flattening and its own limb-darkening law.
- **Built to sit in the background.** ~270 MiB VRAM (block-compressed maps),
  ~0.3 ms of GPU time per frame, motion-gated redraws at 15 fps, idle CPU
  priority, one analytic fullscreen triangle per pass.
- **Native everywhere.** Wayland via wlr-layer-shell (Hyprland, Sway, KDE
  Plasma, niri, river, labwc, Wayfire…) and Xorg via an EWMH desktop window on
  every monitor. Multi-monitor desktops share one continuous camera.

## Gallery

| | | |
|:-:|:-:|:-:|
| ![Night side over Asia with aurora](docs/previews/earth-night.jpg) | ![Half-lit Earth at the equinox](docs/previews/earth-twilight.jpg) | ![2024-04-08 total solar eclipse](docs/previews/earth-eclipse.jpg) |
| Night side, city lights and aurora | Half-lit Earth at the equinox | The 8 April 2024 eclipse shadow |
| ![Aurora over the night side](docs/previews/earth-aurora.jpg) | ![Sunglint over the Atlantic](docs/previews/earth-glint.jpg) | ![The Moon](docs/previews/moon.jpg) |
| Aurora over the night side | Sunglint | The Moon |
| ![Mercury](docs/previews/mercury.jpg) | ![Venus](docs/previews/venus.jpg) | ![Mars](docs/previews/mars.jpg) |
| Mercury | Venus | Mars |
| ![Jupiter](docs/previews/jupiter.jpg) | ![Saturn](docs/previews/saturn.jpg) | ![Uranus](docs/previews/uranus.jpg) |
| Jupiter | Saturn (rings as in October 2017) | Uranus |
| ![Neptune](docs/previews/neptune.jpg) | | |
| Neptune | | |

Every image is a direct Vulkan readback from the renderer (`earth-native capture_frame`), not a screenshot.

## Install

**Requirements:** a Vulkan 1.3 GPU and driver (Mesa or NVIDIA), and a
Wayland compositor with wlr-layer-shell or any Xorg desktop. GNOME on Wayland
has no layer-shell; use the GNOME Xorg session there.

### Release bundle (recommended)

```sh
curl -LO https://github.com/24bit192kHz/earth-native/releases/latest/download/earth-native-x86_64-linux.tar.zst
tar -xf earth-native-x86_64-linux.tar.zst
cd earth-native-x86_64-linux && ./install.sh
```

This installs the binary to `~/.local/bin`, the texture pack to
`~/.local/share/earth-native`, and a systemd user service that starts with
your graphical session (plus an XDG autostart entry as a fallback). The live
weather/aurora feed runs in its own small Python environment; skip it with
`./install.sh --no-weather`.

Hyprland or Sway without a systemd session target: add
`exec-once = earth-native start` (Hyprland) or `exec earth-native start` (Sway).

### Arch Linux

`packaging/PKGBUILD` builds from source and installs the release texture pack
system-wide: `makepkg -si` in that directory.

### From source

```sh
git clone https://github.com/24bit192kHz/earth-native && cd earth-native
./install.sh            # needs cargo and glslc (shaderc); downloads no textures
```

Then get the texture pack from the latest release
(`earth-native-data-*.tar.zst`, extract into `~/.local/share/earth-native`)
or build it yourself: see [docs/DATA.md](docs/DATA.md).

## Use

```sh
earth-native status                  # one-line state: outputs, fps, GPU time, data ages
earth-native body saturn             # earth moon mercury venus mars jupiter saturn uranus neptune
earth-native control                 # take the monitor under the cursor for interaction
earth-native time unix 1712600280    # jump to a moment (here: the 2024 eclipse); `time live` to return
earth-native camera 180 20 8         # fixed yaw/pitch/distance; `camera live` to follow the ISS
earth-native capture_frame           # write JPEG + JSON readbacks of every output
earth-native stop
```

In control mode: drag to orbit, scroll or Q/E to zoom, arrow keys to orbit,
**Ctrl+←/→** to tour the planets, Esc to hand the desktop back.

| Variable | Effect |
| --- | --- |
| `EARTH_NATIVE_BACKEND` | `wayland` or `x11` (default: Wayland if available) |
| `EARTH_NATIVE_BODY` | body at startup |
| `EARTH_NATIVE_FOCUS_OUTPUT` | monitor the globe is centred on (default: the largest) |
| `EARTH_NATIVE_DATA_DIR` | texture pack location |

## How it works

A single Rust binary renders straight to each monitor's swapchain: stars
first, then the body as one fullscreen triangle whose fragment shader
ray-intersects the exact ellipsoid and evaluates the lighting. The physics,
constants and the verification against JPL Horizons and DSCOVR/EPIC are in
[docs/PHYSICS.md](docs/PHYSICS.md); the data sources and the texture
pipeline are in [docs/DATA.md](docs/DATA.md).

## Credits

Imagery courtesy of NASA (Earth Observatory, Goddard SVS, Visible Earth),
NOAA (GFS, SWPC) and Solar System Scope (CC BY 4.0); ephemerides by
Astronomy Engine; SGP4 by Daniel Warner. Full list and licences in
[CREDITS.md](CREDITS.md). Code: MIT.
