"""Compatibility wrapper for scan controller."""

import vision.scan_controller as _impl
from vision.scan_controller import *  # noqa: F401,F403

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
if "ScanController" not in __all__:
    __all__.append("ScanController")
