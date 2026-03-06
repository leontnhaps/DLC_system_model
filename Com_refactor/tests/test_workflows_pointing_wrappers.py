"""Wrapper import tests for pointing workflow compatibility modules."""

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


class WorkflowsPointingWrapperImportTest(unittest.TestCase):
    def test_wrapper_and_impl_identity(self):
        import pointing_handler
        from workflows import pointing_workflow as wf_pointing

        self.assertIs(pointing_handler.PointingHandlerMixin, wf_pointing.PointingHandlerMixin)
        self.assertEqual(
            pointing_handler.CONVERGENCE_TOL_PX, wf_pointing.CONVERGENCE_TOL_PX
        )
        self.assertEqual(
            pointing_handler.LASER_DIFF_THRESHOLD, wf_pointing.LASER_DIFF_THRESHOLD
        )


if __name__ == "__main__":
    unittest.main()
