import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import bcn
import data_pipeline as pipeline


class DataPipelineTests(unittest.TestCase):
    def test_grib_ranges_are_exact_and_all_fields_required(self):
        index = "\n".join([
            "1:0:d=2026090512:TCDC:entire atmosphere:anl:",
            "2:100:d=2026090512:CAPE:surface:anl:",
            "3:400:d=2026090512:PRATE:surface:anl:",
            "4:600:d=2026090512:CWAT:entire atmosphere (considered as a single layer):anl:",
            "5:900:d=2026090512:TMP:surface:anl:"])
        self.assertEqual(pipeline.select_ranges(index), {"cloud": (0, 99), "cape": (100, 399), "rain": (400, 599), "water": (600, 899)})
        with self.assertRaises(ValueError):
            pipeline.select_ranges("\n".join(index.splitlines()[:3]))

    def test_resize_averages_colour_in_linear_light(self):
        image = Image.fromarray(np.array([[[0, 0, 0], [255, 255, 255]]], np.uint8))
        result = pipeline.resize_rgb(image, (1, 1))
        self.assertTrue(185 <= int(result[0, 0, 0]) <= 190)

    def test_preview_is_bgra_top_left_with_independent_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            record = pipeline.preview(path, "sample", np.array([[[10, 20, 30]]], np.uint8), alpha=np.array([[42]], np.uint8))
            self.assertEqual((path / record["file"]).read_bytes(), bytes([30, 20, 10, 42]))
            metadata = json.loads((path / "sample.bgra.json").read_text())
            self.assertEqual(metadata["layout"], "equirectangular")
            self.assertEqual(metadata["payload_bytes"], 4)

    def test_failed_atomic_write_preserves_last_good_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            pipeline.atomic_bytes(path, b"previous")
            with patch("data_pipeline.os.replace", side_effect=OSError("test failure")):
                with self.assertRaises(OSError):
                    pipeline.atomic_bytes(path, b"incomplete")
            self.assertEqual(path.read_bytes(), b"previous")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_range_download_rejects_ignored_range(self):
        response = io.BytesIO(b"abcd")
        response.status = 200
        response.headers = {"Content-Length": "4"}
        with patch("data_pipeline.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "honor"):
                pipeline.download("https://test.invalid/", byte_range=(0, 3))

    def test_grib_nodes_become_geographic_pixel_centres(self):
        import eccodes
        handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        try:
            for key, value in {"Ni": 1440, "Nj": 721, "latitudeOfFirstGridPointInDegrees": 90,
                               "longitudeOfFirstGridPointInDegrees": 0, "latitudeOfLastGridPointInDegrees": -90,
                               "longitudeOfLastGridPointInDegrees": 359.75, "iDirectionIncrementInDegrees": 0.25,
                               "jDirectionIncrementInDegrees": 0.25}.items():
                eccodes.codes_set(handle, key, value)
            values = np.broadcast_to(np.arange(1440, dtype=float), (721, 1440))
            eccodes.codes_set_values(handle, values.ravel())
            decoded = pipeline.decode_grib(eccodes.codes_get_message(handle))
            self.assertEqual(decoded.shape, (720, 1440))
            self.assertAlmostEqual(decoded[0, 0], 720.5)
            self.assertAlmostEqual(decoded[0, 720], 0.5)
        finally:
            eccodes.codes_release(handle)

    def test_aurora_rejects_invalid_probability(self):
        payload = {"coordinates": [[0, 0, 101]] * 64000, "Forecast Time": "2026-09-05T12:00:00Z"}
        with patch("data_pipeline.download", return_value=json.dumps(payload).encode()):
            with self.assertRaisesRegex(ValueError, "out of range"):
                pipeline.aurora_field()

    def test_block_compression_round_trips_and_mips_tile_whole_blocks(self):
        rng = np.random.default_rng(1)
        gray = np.clip(np.linspace(0, 255, 64)[None, :] + rng.normal(0, 4, (32, 64)), 0, 255).astype(np.uint8)
        blocks = np.frombuffer(bcn.encode_bc4(gray), dtype=np.uint8).reshape(-1, 8).astype(np.int64)
        e0, e1 = blocks[:, 0], blocks[:, 1]
        bits = sum(blocks[:, 2 + i] << (8 * i) for i in range(6))
        index = np.stack([(bits >> (3 * t)) & 7 for t in range(16)], axis=1)
        weights = np.array([0, 7, 1, 2, 3, 4, 5, 6])
        palette = ((7 - weights)[None] * e0[:, None] + weights[None] * e1[:, None]) / 7
        decoded = np.take_along_axis(palette, index, 1).reshape(8, 16, 4, 4).transpose(0, 2, 1, 3).reshape(32, 64)
        self.assertLess(np.abs(decoded - gray).max(), 6)
        rgba = rng.integers(0, 256, (16, 32, 4), dtype=np.uint8)
        payload, mips = bcn.encode_mipped(rgba, "bc3", [0, 1, 2])
        self.assertEqual([(m["width"], m["height"]) for m in mips], [(32, 16), (16, 8), (8, 4)])
        self.assertEqual(len(payload), sum(m["width"] * m["height"] for m in mips))
        self.assertEqual([m["byte_offset"] for m in mips], [0, 512, 640])

    def test_polar_rows_are_low_passed_to_equatorial_ground_resolution(self):
        image = np.zeros((8, 16), dtype=np.uint8)
        image[:, ::2] = 255
        out = pipeline.polar_resample(image)
        # Rows 3/4 straddle the equator (cos ~ 1): untouched.
        np.testing.assert_array_equal(out[3:5], image[3:5])
        # The polar rows lose the single-texel pattern that turns into streaks.
        self.assertLess(out[0].max() - out[0].min(), 60)
        self.assertEqual(out.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
