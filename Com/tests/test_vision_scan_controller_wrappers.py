"""Wrapper import tests for scan_controller compatibility modules."""

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
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.__path__ = []
        sys.modules["numpy"] = numpy_stub

    if not _has_module("numpy.linalg") and "numpy.linalg" not in sys.modules:
        numpy_mod = sys.modules.get("numpy")
        if numpy_mod is None:
            numpy_mod = types.ModuleType("numpy")
            numpy_mod.__path__ = []
            sys.modules["numpy"] = numpy_mod
        linalg_stub = types.ModuleType("numpy.linalg")

        def _norm(*args, **kwargs):
            return 0.0

        linalg_stub.norm = _norm
        numpy_mod.linalg = linalg_stub
        sys.modules["numpy.linalg"] = linalg_stub

    if not _has_module("scipy") and "scipy" not in sys.modules:
        scipy_stub = types.ModuleType("scipy")
        scipy_stub.__path__ = []
        sys.modules["scipy"] = scipy_stub

    if not _has_module("scipy.optimize") and "scipy.optimize" not in sys.modules:
        scipy_mod = sys.modules.get("scipy")
        if scipy_mod is None:
            scipy_mod = types.ModuleType("scipy")
            scipy_mod.__path__ = []
            sys.modules["scipy"] = scipy_mod
        optimize_stub = types.ModuleType("scipy.optimize")

        def _linear_sum_assignment(*args, **kwargs):
            return [], []

        optimize_stub.linear_sum_assignment = _linear_sum_assignment
        scipy_mod.optimize = optimize_stub
        sys.modules["scipy.optimize"] = optimize_stub


_inject_optional_dependency_stubs()


class VisionScanControllerWrapperImportTest(unittest.TestCase):
    def test_wrapper_and_impl_are_same_object(self):
        import scan_controller
        from vision import scan_controller as vision_scan_controller

        self.assertTrue(hasattr(scan_controller, "ScanController"))
        self.assertTrue(hasattr(vision_scan_controller, "ScanController"))
        self.assertIs(scan_controller.ScanController, vision_scan_controller.ScanController)


if __name__ == "__main__":
    unittest.main()
