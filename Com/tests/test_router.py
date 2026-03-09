"""Unit tests for saved-image routing."""

import os
from pathlib import Path
import shutil
import sys
import unittest
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app_config
from image_router import route_saved_image


class FakeInfoLabel:
    def __init__(self):
        self.last_text = None

    def config(self, **kwargs):
        self.last_text = kwargs.get("text")


class FakeScanCtrl:
    def __init__(self, active: bool):
        self._active = active
        self.save_calls = []

    def is_active(self):
        return self._active

    def save_image(self, name, data):
        self.save_calls.append((name, data))
        return "fake/scan/path.jpg"


class FakeApp:
    def __init__(self, aiming_active: bool, scan_active: bool):
        self._aiming_active = aiming_active
        self.scan_ctrl = FakeScanCtrl(scan_active)
        self.info_label = FakeInfoLabel()
        self.pointing_calls = []
        self.preview_calls = []

    def _on_pointing_image_received(self, name, data):
        self.pointing_calls.append((name, data))

    def _set_preview(self, data):
        self.preview_calls.append(data)


class RouterTest(unittest.TestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._run_dir = None
        scratch_root = PROJECT_ROOT / "tests" / "_scratch"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self._run_dir = scratch_root / f"router_test_{uuid.uuid4().hex}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self._run_dir)

    def tearDown(self):
        try:
            os.chdir(self._old_cwd)
        finally:
            if self._run_dir is not None:
                shutil.rmtree(self._run_dir, ignore_errors=True)

    def test_pointing_route(self):
        app = FakeApp(aiming_active=True, scan_active=False)
        name = "pointing_led_on_1.jpg"
        data = b"pointing-data"

        route_saved_image(app, name, data)

        self.assertEqual(len(app.pointing_calls), 1)
        self.assertEqual(app.pointing_calls[0], (name, data))
        self.assertEqual(len(app.scan_ctrl.save_calls), 0)
        self.assertEqual(len(app.preview_calls), 1)
        self.assertEqual(app.preview_calls[0], data)
        self.assertFalse((app_config.SAVE_DIR / name).exists())

    def test_scan_route(self):
        app = FakeApp(aiming_active=False, scan_active=True)
        name = "scan_20260101_000000_t+00_p+000_led_on.jpg"
        data = b"scan-data"

        route_saved_image(app, name, data)

        self.assertEqual(len(app.pointing_calls), 0)
        self.assertEqual(len(app.scan_ctrl.save_calls), 1)
        self.assertEqual(app.scan_ctrl.save_calls[0], (name, data))
        self.assertEqual(len(app.preview_calls), 1)
        self.assertEqual(app.preview_calls[0], data)
        self.assertFalse((app_config.SAVE_DIR / name).exists())

    def test_default_save_route(self):
        app = FakeApp(aiming_active=False, scan_active=False)
        name = "snap_20260101_000000.jpg"
        data = b"default-save-data"

        route_saved_image(app, name, data)

        saved_path = app_config.SAVE_DIR / name
        self.assertTrue(saved_path.exists())
        self.assertEqual(saved_path.read_bytes(), data)
        self.assertEqual(app.info_label.last_text, f"💾 저장됨: {name}")
        self.assertEqual(len(app.pointing_calls), 0)
        self.assertEqual(len(app.scan_ctrl.save_calls), 0)
        self.assertEqual(len(app.preview_calls), 1)
        self.assertEqual(app.preview_calls[0], data)


if __name__ == "__main__":
    unittest.main()
