"""Compatibility wrapper for YOLO utilities."""

import vision.yolo_utils as _impl
from vision.yolo_utils import *  # noqa: F401,F403

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
for _name in ("non_max_suppression", "predict_with_tiling", "YOLOProcessor"):
    if _name not in __all__:
        __all__.append(_name)
