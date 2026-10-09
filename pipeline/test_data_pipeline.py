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
    def test_cloud_observation_survives_failed_updates_and_generation_cleanup(self):
        from datetime import datetime, timezone, timedelta
        now = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
        index = "\n".join([
            "1:0:d=x:TCDC:entire atmosphere:anl:", "2:100:d=x:CAPE:surface:anl:",
            "3:400:d=x:PRATE:surface:anl:",
            "4:600:d=x:CWAT:entire atmosphere (considered as a single layer):anl:",
            "5:900:d=x:LCDC:low cloud layer:anl:", "6:1000:d=x:TMP:surface:anl:"])
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "DATA", Path(directory)), \
                patch.object(pipeline, "CLOUD_SIZE", (4, 2)), patch.object(pipeline, "datetime") as clock, \
                patch.object(pipeline, "download", return_value=index.encode()), \
                patch.object(pipeline, "aurora_field", return_value=(np.zeros((720, 1440), np.uint8), 0)), \
                patch.object(pipeline, "aerosol_field", side_effect=OSError("offline")), \
                patch.object(pipeline, "sea_ice_field", side_effect=OSError("offline")), \
                patch.object(pipeline, "live_clouds", side_effect=OSError("offline")):
            root = Path(directory) / "weather"
            old = root / "20261005T12-1791199800"
            fields = pipeline.preview(old, "fields", np.zeros((720, 1440, 3), np.uint8))
            clouds = pipeline.preview(old, "clouds", np.ones((2, 4, 3), np.uint8))
            np.save(root / "gfs-low-cloud.npy", np.zeros((720, 1440), np.float16))
            observation = int((now - timedelta(hours=1)).timestamp())
            existing = dict(packing_version=4, valid_unix_utc=int(now.timestamp()),
                            texture=str(old / fields["file"]), sha256=fields["sha256"],
                            clouds_texture=str(old / clouds["file"]), clouds_sha256=clouds["sha256"],
                            clouds_unix_utc=observation, aerosol_unix_utc=observation - 3600)
            pipeline.atomic_json(root / "current.json", existing)
            for minutes in (0, 30, 60):
                clock.now.return_value = now + timedelta(minutes=minutes)
                result = pipeline.weather()
                self.assertEqual(result["clouds_texture"], existing["clouds_texture"])
                self.assertEqual(result["clouds_unix_utc"], observation)
                self.assertEqual(result["aerosol_unix_utc"], observation - 3600)
                self.assertTrue(Path(result["clouds_texture"]).is_file())
            self.assertEqual(len(list(root.glob("20261005T12-*"))), 3)  # two GFS + retained clouds
            clock.now.return_value = now + timedelta(hours=5)  # six-hour observation limit
            expired = pipeline.weather()
            self.assertNotIn("clouds_texture", expired)
            self.assertFalse(old.exists())

    def test_corrupt_or_missing_cloud_payload_is_not_retained(self):
        from datetime import datetime, timezone
        now = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "CLOUD_SIZE", (4, 2)):
            root = Path(directory)
            cloud = pipeline.preview(root, "clouds", np.zeros((2, 4, 3), np.uint8))
            existing = dict(packing_version=4, clouds_unix_utc=int(now.timestamp()),
                            clouds_texture=str(root / cloud["file"]), clouds_sha256=cloud["sha256"])
            self.assertTrue(pipeline.retained_cloud_metadata(existing, now))
            (root / cloud["file"]).write_bytes(bytes([1]) * 32)
            self.assertEqual(pipeline.retained_cloud_metadata(existing, now), {})
            (root / cloud["file"]).unlink()
            self.assertEqual(pipeline.retained_cloud_metadata(existing, now), {})

    def test_gmgsi_missing_segments_are_not_cloud(self):
        # A missing satellite segment arrives as 255 (infrared) or 0
        # (visible), not as the -9999 fill value; both must become NaN.
        ir = pipeline.gmgsi_counts(np.array([[-9999.0, 255.0, 253.0, 120.0]]), "LW")
        self.assertTrue(np.isnan(ir[0, 0]) and np.isnan(ir[0, 1]))
        self.assertEqual(list(ir[0, 2:]), [253.0, 120.0])
        vis = pipeline.gmgsi_counts(np.array([[0.0, 1.0, 255.0]]), "VIS")
        self.assertTrue(np.isnan(vis[0, 0]))
        self.assertEqual(list(vis[0, 1:]), [1.0, 255.0])

    def test_gmgsi_seams_lie_midway_between_the_satellites(self):
        # The sector boundaries measured in the 2026-10 mosaics.
        longitudes = sorted(column / 10 - 180 for column in pipeline.gmgsi_seams(3600))
        for found, measured in zip(longitudes, [-178.2, -106.1, -37.7, 22.7, 93.1]):
            self.assertAlmostEqual(found, measured, delta=0.15)

    def test_satellites_are_matched_across_a_seam(self):
        # 93 E, 2026-10-02 10:05 UTC: Himawari looks toward the Sun and reads
        # clear sea 0.3 brighter than Meteosat. 1 degree per pixel, so the
        # 20 degree reach is 20 columns.
        height, width = 90, 360
        seam = pipeline.gmgsi_seams(width)[3]
        hazy_east = np.full((height, width), 0.05)
        hazy_east[:, seam:seam + 60] += 0.3
        balanced = pipeline.balance_satellites(hazy_east)
        row = height // 2
        self.assertLess(abs(balanced[row, seam] - balanced[row, seam - 1]), 0.01)
        self.assertAlmostEqual(balanced[row, seam - 5], 0.05)        # the clearer side stays as it is
        self.assertAlmostEqual(balanced[row, seam + 30], 0.35)       # beyond the reach the haze is the satellite's own
        self.assertTrue(np.all(np.diff(balanced[row, seam:seam + 20]) >= 0))   # and returns smoothly
        # The same when the west side is the brighter one.
        hazy_west = np.full((height, width), 0.05)
        hazy_west[:, seam - 60:seam] += 0.3
        balanced = pipeline.balance_satellites(hazy_west)
        self.assertLess(abs(balanced[row, seam] - balanced[row, seam - 1]), 0.01)
        self.assertAlmostEqual(balanced[row, seam + 5], 0.05)
        self.assertAlmostEqual(balanced[row, seam - 30], 0.35)

    def test_the_seam_is_found_where_it_really_is(self):
        # The cut at 93 E sits a pixel east of the satellites' midpoint; the
        # clear column west of it must not be treated as part of the haze.
        height, width = 90, 360
        nominal = pipeline.gmgsi_seams(width)[3]
        field = np.full((height, width), 0.05)
        field[:, nominal + 1:nominal + 61] += 0.3
        balanced = pipeline.balance_satellites(field)
        row = height // 2
        self.assertAlmostEqual(balanced[row, nominal], 0.05)
        self.assertAlmostEqual(balanced[row, nominal - 1], 0.05)
        self.assertLess(abs(balanced[row, nominal + 1] - 0.05), 0.01)

    def test_satellite_balance_ignores_gaps_and_leaves_alone_what_matches(self):
        height, width = 90, 360
        seam = pipeline.gmgsi_seams(width)[3]
        night_east = np.full((height, width), 0.05)
        night_east[:, seam:seam + 60] = np.nan                     # no visible image on one side
        np.testing.assert_array_equal(pipeline.balance_satellites(night_east), night_east)
        noise = np.random.default_rng(3).normal(0.05, 0.02, (720, width))
        self.assertLess(np.abs(pipeline.balance_satellites(noise) - noise).max(), 0.03)
        cloud_on_both_sides = np.full((height, width), 0.05)
        cloud_on_both_sides[20:40, seam - 15:seam + 15] = 0.9      # one cloud straddling the seam
        np.testing.assert_allclose(pipeline.balance_satellites(cloud_on_both_sides), cloud_on_both_sides, atol=1e-9)

    def test_grib_ranges_are_exact_and_all_fields_required(self):
        index = "\n".join([
            "1:0:d=2026090512:TCDC:entire atmosphere:anl:",
            "2:100:d=2026090512:CAPE:surface:anl:",
            "3:400:d=2026090512:PRATE:surface:anl:",
            "4:600:d=2026090512:CWAT:entire atmosphere (considered as a single layer):anl:",
            "5:900:d=2026090512:LCDC:low cloud layer:anl:",
            "6:1000:d=2026090512:TMP:surface:anl:"])
        self.assertEqual(pipeline.select_ranges(index), {"cloud": (0, 99), "cape": (100, 399), "rain": (400, 599),
                                                         "water": (600, 899), "low": (900, 999)})
        with self.assertRaises(ValueError):
            pipeline.select_ranges("\n".join(index.splitlines()[:3]))

    def test_polar_stereographic_matches_osi_saf_grid_corners(self):
        # OSI SAF 10 km grids: the pole is the origin, and on the true-scale
        # parallel (70 deg) the radius is a cos(phi) / sqrt(1 - e^2 sin^2 phi).
        south = "+proj=stere +a=6378273 +b=6356889.44891 +lat_0=-90 +lat_ts=-70 +lon_0=0"
        north = "+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=-45"
        x, y = pipeline._polar_stereographic(np.array([-90.0]), np.array([0.0]), south)
        self.assertLess(abs(x[0]) + abs(y[0]), 1e-6)
        a, b, phi = 6378.273, 6356.88944891, np.radians(70.0)
        radius = a * np.cos(phi) / np.sqrt(1 - (1 - (b / a) ** 2) * np.sin(phi) ** 2)
        x, y = pipeline._polar_stereographic(np.array([-70.0]), np.array([0.0]), south)
        self.assertAlmostEqual(x[0], 0.0, places=6)
        self.assertAlmostEqual(y[0], radius, places=6)    # south grid: +y toward 0 deg E
        x, y = pipeline._polar_stereographic(np.array([70.0]), np.array([-45.0]), north)
        self.assertAlmostEqual(x[0], 0.0, places=6)
        self.assertAlmostEqual(y[0], -radius, places=6)   # north grid: -y toward lon_0

    def test_storm_preview_oval_is_nightside_and_in_both_hemispheres(self):
        from datetime import datetime, timezone
        oval = pipeline.storm_oval(datetime(2026, 9, 28, 0, 0, tzinfo=timezone.utc), 7) / 255
        lat = 90 - (np.arange(720) + 0.5) * 0.25
        column = lambda lon: oval[:, int((lon + 180) * 4)]
        # Midnight at 0 deg E (UTC 00:00) against noon at 180 deg.
        self.assertGreater(column(0.0)[lat > 0].max(), 2 * column(179.0)[lat > 0].max())
        self.assertGreater(oval[lat < 0].max(), 0.4)
        self.assertEqual(oval[np.abs(lat) < 35].max(), 0)

    def test_storms_move_onto_observed_cold_tops_near_model_instability(self):
        packed = np.zeros((720, 1440, 3), np.uint8)
        packed[340:350, 700:710, 1] = 60          # model instability (CAPE ~940 J/kg)
        convection = np.zeros((720, 1440), np.float32)
        convection[344, 703] = 1.0                # an observed cold top inside it
        convection[100, 100] = 1.0                # one where the model is stable
        steered = pipeline.steer_storms(packed, convection)
        self.assertGreaterEqual(steered[344, 703, 0], 0.8 * 255)
        self.assertGreaterEqual(steered[344, 703, 2], 0.5 * 255)
        self.assertEqual(steered[100, 100].tolist(), [0, 0, 0])
        # A pure function of the raw model: steering twice changes nothing more.
        np.testing.assert_array_equal(pipeline.steer_storms(packed, convection), steered)
        np.testing.assert_array_equal(pipeline.steer_storms(packed, None), packed)

    def test_deep_convection_is_the_coldest_tropical_tops(self):
        lat = 90 - (np.arange(2048) + 0.5) * 180 / 2048
        ir = np.full((2048, 4096), 100.0)
        ir_cold = np.full((2048, 4096), 60.0)
        ir[1000:1010, 2000:2010] = 220.0          # an anvil near the equator
        ir[100:110, 2000:2010] = 220.0            # cold polar air, not a storm
        field = pipeline.deep_convection(ir, ir_cold, lat)
        self.assertEqual(field.shape, (720, 1440))
        self.assertGreater(field[1005 * 720 // 2048, 2005 * 1440 // 4096], 0.5)
        self.assertEqual(field[105 * 720 // 2048, 2005 * 1440 // 4096], 0.0)
        self.assertEqual(field[360, 100], 0.0)

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
