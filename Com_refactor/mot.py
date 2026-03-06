"""Compatibility wrapper for MOT utilities."""

import vision.mot as _impl
from vision.mot import *  # noqa: F401,F403

__all__ = [name for name in dir(_impl) if not name.startswith("_")]
for _name in ("get_feature_vector", "calc_cosine_similarity", "ObjectTracker"):
    if _name not in __all__:
        __all__.append(_name)
