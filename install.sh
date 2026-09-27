#!/usr/bin/env bash
# earth-native installer (per user, no root).
#
# From a release bundle it installs the prebuilt binary and texture pack;
# from a source checkout it builds with cargo first. Everything lands in
#   ~/.local/bin/earth-native
#   ~/.local/share/earth-native/   (textures, weather pipeline, venv)
#   ~/.config/systemd/user/        (session services)
#   ~/.config/autostart/           (fallback for non-systemd sessions)
#
# Usage: ./install.sh [--no-weather] [--no-enable] [--no-autostart]
set -euo pipefail

weather=1 enable=1 autostart=1
for arg in "$@"; do
    case "$arg" in
        --no-weather) weather=0 ;;
        --no-enable) enable=0 ;;
        --no-autostart) autostart=0 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) printf 'unknown option: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bin_dir="${HOME}/.local/bin"
data_dir="${XDG_DATA_HOME:-$HOME/.local/share}/earth-native"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
autostart_dir="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }

# --- requirements ---------------------------------------------------------
libraries="$(ldconfig -p 2>/dev/null || true)"
if ! grep -q 'libvulkan\.so\.1' <<<"$libraries"; then
    warn "libvulkan.so.1 not found: install your distro's Vulkan loader and GPU driver (e.g. vulkan-icd-loader + mesa/nvidia-utils)"
fi

# --- binary ---------------------------------------------------------------
if [[ -x "$here/bin/earth-native" ]]; then
    binary="$here/bin/earth-native"
elif [[ -f "$here/Cargo.toml" ]]; then
    command -v cargo >/dev/null || { echo "cargo is required to build from source (https://rustup.rs)" >&2; exit 1; }
    command -v glslc >/dev/null || { echo "glslc (shaderc) is required to build the shaders" >&2; exit 1; }
    say "building release binary"
    cargo build --release --manifest-path "$here/Cargo.toml"
    binary="$here/target/release/earth-native"
else
    echo "run this from a release bundle or a source checkout" >&2; exit 1
fi
say "installing $bin_dir/earth-native"
install -Dm755 "$binary" "$bin_dir/earth-native"

# --- textures -------------------------------------------------------------
mkdir -p "$data_dir"
if [[ -d "$here/share/earth-native" ]]; then
    say "installing texture pack to $data_dir"
    cp -r "$here/share/earth-native/." "$data_dir/"
elif ! compgen -G "$data_dir/*/day-east.bc3" >/dev/null; then
    warn "no texture pack found. Download earth-native-data-*.tar.zst from the GitHub release and extract it into $data_dir, or build it with: python3 $data_dir/pipeline/data_pipeline.py build"
fi
mkdir -p "$data_dir/pipeline"
cp "$here"/pipeline/*.py "$here/pipeline/requirements.txt" "$data_dir/pipeline/"

# --- live weather + aurora (optional) ---------------------------------------
if (( weather )); then
    if command -v python3 >/dev/null; then
        say "setting up the NOAA weather/aurora feed (Python venv)"
        if python3 -m venv "$data_dir/venv" \
            && "$data_dir/venv/bin/pip" install --quiet --upgrade pip \
            && "$data_dir/venv/bin/pip" install --quiet numpy pillow eccodes; then
            :
        else
            warn "weather feed dependencies failed to install; aurora and lightning stay off (rerun later)"
            weather=0
        fi
    else
        warn "python3 not found; skipping the live weather/aurora feed"
        weather=0
    fi
fi

# --- session integration ----------------------------------------------------
mkdir -p "$unit_dir"
install -m644 "$here/packaging/earth-native.service" "$unit_dir/"
(( weather )) && install -m644 "$here/packaging/earth-native-weather.service" "$unit_dir/"
if (( autostart )); then
    mkdir -p "$autostart_dir"
    install -m644 "$here/packaging/earth-native.desktop" "$autostart_dir/"
fi
if command -v systemctl >/dev/null && systemctl --user daemon-reload 2>/dev/null; then
    if (( enable )); then
        systemctl --user enable earth-native.service >/dev/null
        (( weather )) && systemctl --user enable earth-native-weather.service >/dev/null
        if systemctl --user is-active --quiet graphical-session.target; then
            systemctl --user restart earth-native.service
            say "started (systemd user session)"
        else
            "$bin_dir/earth-native" start >/dev/null && say "started"
        fi
        (( weather )) && systemctl --user restart earth-native-weather.service
    fi
fi

cat <<EOF

earth-native is installed.
  status:   earth-native status
  control:  earth-native control        (drag to orbit, scroll/Q/E zoom, Ctrl+←/→ switch planet, Esc release)
  planets:  earth-native body {earth|moon|mercury|venus|mars|jupiter|saturn|uranus|neptune}
  stop:     earth-native stop
Sway/Hyprland without a systemd session target: add  exec-once = earth-native start
(Hyprland) or  exec earth-native start  (Sway) to your config.
EOF
case ":$PATH:" in *":$bin_dir:"*) ;; *) warn "$bin_dir is not on your PATH" ;; esac
