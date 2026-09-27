"""Native GPU captures of current modeled storms, aurora, and atmospheric limbs."""
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from data_pipeline import DATA, ROOT

LAUNCHER = ROOT.parent / "Tools/earth-native"
DESTINATION = Path("/tmp/earth-native-weather-verification")


def run(*args):
    text = subprocess.check_output([str(LAUNCHER), *map(str, args)], text=True, timeout=45).strip()
    if text.startswith("error"):
        raise RuntimeError(text)
    return text


def capture(name):
    result = json.loads(run("capture_frame"))
    chosen = next(value for value in result["captures"] if value["image"].endswith("DP-2.jpg"))
    assert chosen["frame"]["uniforms"]["nasa_materials"]
    assert chosen["frame"]["uniforms"]["weather_valid_unix_utc"] > 0
    assert chosen["nonblack_fraction"] > 0.001
    path = DESTINATION / f"{name}.jpg"
    shutil.copy2(chosen["image"], path)
    chosen["preview"] = str(path)
    return chosen


def blue_limb_pixels(frame):
    rgb = np.asarray(Image.open(frame["preview"])).astype(np.int16)
    height, width = rgb.shape[:2]
    uniform = frame["frame"]["uniforms"]
    viewport = frame["frame"]["viewport"]
    canvas = uniform["canvas"]
    x = viewport["x"] + (np.arange(width, dtype=np.float32) + 0.5) / width * viewport["width"]
    y = viewport["y"] + (np.arange(height, dtype=np.float32) + 0.5) / height * viewport["height"]
    x = 2 * (x - uniform["focus_x"]) / canvas["width"] * uniform["tan_half_fov_x"]
    y = 2 * (uniform["focus_y"] - y) / canvas["height"] * uniform["tan_half_fov_y"]
    camera = np.array(uniform["camera_position"])
    b = (np.dot(camera, uniform["forward"]) + x[None, :] * np.dot(camera, uniform["right"])
         + y[:, None] * np.dot(camera, uniform["up"])) / np.sqrt(1 + x[None, :] ** 2 + y[:, None] ** 2)
    miss_squared = np.dot(camera, camera) - b * b
    radius = frame["frame"]["scene_surface_radius"]
    annulus = (b < 0) & (miss_squared > radius ** 2) & (miss_squared < (radius * (1 + 100 / 6378.137)) ** 2)
    blue = (rgb[:, :, 2] > rgb[:, :, 0] + 5) & (rgb[:, :, 2] > 20)
    return int((annulus & blue).sum())


def verify():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    report = {"captures": {}, "lightning": "modeled, not observed strikes"}
    try:
        run("restart")
        subprocess.run(["systemctl", "--user", "stop", "earth-native-weather.service"], check=True)
        product = json.loads((DATA / "weather/current.json").read_text())
        run("weather", "reload")
        report["weather"] = product
        fields = np.fromfile(product["texture"], np.uint8).reshape(720, 1440, 4)
        storm = fields[:, :, 1].astype(float) * fields[:, :, 0] * fields[:, :, 2]
        storm[:180] = 0
        storm[540:] = 0
        y, x = np.unravel_index(np.argmax(storm), storm.shape)
        latitude, longitude = 90 - (y + 0.5) / 4, (x + 0.5) / 4 - 180
        report["storm_location_degrees"] = [latitude, longitude]
        utc = int(time.time())
        run("camera", 180 - longitude, -latitude, 2.2)
        run("celestial", "sun", longitude + 180, -latitude)
        hashes = set()
        baseline = None
        for i in range(12):
            run("time", "unix", utc + i)
            frame = capture(f"storm-{i:02d}")
            rgb = np.asarray(Image.open(frame["preview"]))
            baseline = rgb.copy() if baseline is None else np.minimum(baseline, rgb)
            hashes.add(frame["raw_pixel_sha256"])
            report["captures"][f"storm-{i:02d}"] = frame
        flash_counts = []
        for i in range(12):
            rgb = np.asarray(Image.open(DESTINATION / f"storm-{i:02d}.jpg")).astype(np.int16)
            # Sun, camera, weather and terrain are fixed and the entire main
            # output is Earth. White lightning need not remain blue after extinction.
            change = np.max(rgb - baseline, axis=2)
            flash_counts.append(int((change > 20).sum()))
        report["lightning_variable_bright_pixels"] = flash_counts
        print(f"Storm sequence: {len(hashes)} distinct frames, flash pixels {flash_counts}", flush=True)
        strongest = int(np.argmax(flash_counts))
        shutil.copy2(DESTINATION / f"storm-{strongest:02d}.jpg", DESTINATION / "lightning.jpg")
        run("camera", 180, -78, 4.5)
        run("celestial", "sun", 0, -90)
        report["captures"]["aurora"] = capture("aurora")
        run("camera", 180, 0, 7.3)
        run("celestial", "sun", 65, 0)
        report["captures"]["atmosphere"] = capture("atmosphere")
        report["blue_limb_pixels_outside_surface"] = blue_limb_pixels(report["captures"]["atmosphere"])
        assert report["blue_limb_pixels_outside_surface"] > 20, "atmosphere is missing outside the surface"
        run("camera", 180, -8, 1.8)
        run("celestial", "sun", 0, 0)
        report["captures"]["close-clouds"] = capture("close-clouds")
        run("camera", 180, 0, 7.3)
        run("celestial", "moon", 192, 4)
        report["captures"]["moon"] = capture("moon")
        assert len(hashes) > 1, "storm animation is frozen"
        assert max(flash_counts) > 10 and max(flash_counts) > min(flash_counts), "no visible lightning variation"
        report["result"] = "passed"
    finally:
        report["restored_status"] = run("restart")
        assert "control=none" in report["restored_status"]
        (DESTINATION / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(DESTINATION / "report.json")
