"""Compatibility wrapper for pointing workflow."""

import workflows.pointing_workflow as _impl
from workflows.pointing_workflow import *  # noqa: F401,F403

if not hasattr(_impl, "CONVERGENCE_TOL_PX"):
    _impl.CONVERGENCE_TOL_PX = getattr(_impl, "CONVERGENCE_TOL_PX_X", None)

CONVERGENCE_TOL_PX = _impl.CONVERGENCE_TOL_PX

_REQUIRED_EXPORTS = [
    "PointingHandlerMixin",
    "CENTERING_GAIN_PAN",
    "CENTERING_GAIN_TILT",
    "CONVERGENCE_TOL_PX",
    "OBJECT_SIZE_CM",
    "TARGET_OFFSET_CM",
    "LASER_DIFF_THRESHOLD",
]

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
for _name in _REQUIRED_EXPORTS:
    if _name not in __all__:
        __all__.append(_name)
