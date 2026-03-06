"""Wrapper import tests for led_filter compatibility modules."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return name in sys.modules


def _inject_optional_dependency_stubs():
    if not _has_module("cv2") and "cv2" not in sys.modules:
        sys.modules["cv2"] = types.ModuleType("cv2")

    if not _has_module("numpy") and "numpy" not in sys.modules:
        sys.modules["numpy"] = types.ModuleType("numpy")


_inject_optional_dependency_stubs()


class VisionLedFilterWrapperImportTest(unittest.TestCase):
    def test_wrapper_and_impl_exports_are_identical(self):
        import led_filter
        from vision import led_filter as vision_led_filter

        self.assertTrue(hasattr(led_filter, "__all__"))
        self.assertIsInstance(led_filter.__all__, list)
        self.assertGreater(len(led_filter.__all__), 0)

        for name in led_filter.__all__:
            self.assertTrue(hasattr(led_filter, name), f"wrapper missing: {name}")
            self.assertTrue(hasattr(vision_led_filter, name), f"impl missing: {name}")
            self.assertIs(
                getattr(led_filter, name),
                getattr(vision_led_filter, name),
                f"object mismatch: {name}",
            )


if __name__ == "__main__":
    unittest.main()
