#!/usr/bin/env bash
# Remove earth-native. Keeps the texture pack unless --purge is given.
set -euo pipefail
purge=0
[[ "${1:-}" == "--purge" ]] && purge=1
data_dir="${XDG_DATA_HOME:-$HOME/.local/share}/earth-native"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if command -v systemctl >/dev/null; then
    systemctl --user disable --now earth-native-weather.service earth-native.service 2>/dev/null || true
fi
"$HOME/.local/bin/earth-native" stop 2>/dev/null || true
rm -f "$HOME/.local/bin/earth-native" \
      "$unit_dir/earth-native.service" "$unit_dir/earth-native-weather.service" \
      "${XDG_CONFIG_HOME:-$HOME/.config}/autostart/earth-native.desktop"
command -v systemctl >/dev/null && systemctl --user daemon-reload 2>/dev/null || true
rm -rf "$data_dir/pipeline" "$data_dir/venv"
if (( purge )); then
    rm -rf "$data_dir" "${XDG_CACHE_HOME:-$HOME/.cache}/earth-native"
    echo "earth-native removed, including textures."
else
    echo "earth-native removed. Textures kept in $data_dir (use --purge to delete)."
fi
