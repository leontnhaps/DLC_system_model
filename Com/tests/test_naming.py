"""Unit tests for image-name parsing."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from naming import parse_image_name


class NamingParserTest(unittest.TestCase):
    def test_preview_name(self):
        parsed = parse_image_name("_preview_123.jpg")
        self.assertEqual(parsed.get("kind"), "preview")

    def test_scan_led_on_name(self):
        parsed = parse_image_name("scan_20260101_000000_t+00_p+000_led_on.jpg")
        self.assertEqual(parsed.get("kind"), "scan")
        self.assertEqual(parsed.get("tilt"), 0)
        self.assertEqual(parsed.get("pan"), 0)
        self.assertEqual(parsed.get("led_state"), "on")

    def test_scan_led_off_name(self):
        parsed = parse_image_name("scan_20260101_000000_t-30_p+180_led_off.jpg")
        self.assertEqual(parsed.get("kind"), "scan")
        self.assertEqual(parsed.get("tilt"), -30)
        self.assertEqual(parsed.get("pan"), 180)
        self.assertEqual(parsed.get("led_state"), "off")

    def test_pointing_name(self):
        parsed = parse_image_name("pointing_led_on_1.jpg")
        self.assertEqual(parsed.get("kind"), "pointing")
        self.assertEqual(parsed.get("label"), "led_on")
        self.assertEqual(parsed.get("iteration"), 1)

    def test_other_name(self):
        parsed = parse_image_name("snap_20260101_000000.jpg")
        self.assertEqual(parsed.get("kind"), "other")


if __name__ == "__main__":
    unittest.main()
