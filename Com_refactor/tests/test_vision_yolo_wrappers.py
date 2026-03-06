"""Wrapper import tests for yolo_utils compatibility modules."""

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
        cv2_stub = types.ModuleType("cv2")
        dnn_stub = types.ModuleType("cv2.dnn")

        def _nms_boxes(*args, **kwargs):
            return []

        dnn_stub.NMSBoxes = _nms_boxes
        cv2_stub.dnn = dnn_stub
        sys.modules["cv2"] = cv2_stub
        sys.modules["cv2.dnn"] = dnn_stub

    if not _has_module("numpy") and "numpy" not in sys.modules:
        sys.modules["numpy"] = types.ModuleType("numpy")


_inject_optional_dependency_stubs()


class VisionYoloWrapperImportTest(unittest.TestCase):
    def test_root_yolo_utils_import(self):
        import yolo_utils

        self.assertTrue(hasattr(yolo_utils, "non_max_suppression"))
        self.assertTrue(hasattr(yolo_utils, "predict_with_tiling"))
        self.assertTrue(hasattr(yolo_utils, "YOLOProcessor"))

    def test_wrapper_and_impl_are_same_objects(self):
        import yolo_utils
        from vision import yolo_utils as vision_yolo_utils

        self.assertIs(
            yolo_utils.non_max_suppression,
            vision_yolo_utils.non_max_suppression,
        )
        self.assertIs(
            yolo_utils.predict_with_tiling,
            vision_yolo_utils.predict_with_tiling,
        )
        self.assertIs(yolo_utils.YOLOProcessor, vision_yolo_utils.YOLOProcessor)


if __name__ == "__main__":
    unittest.main()
