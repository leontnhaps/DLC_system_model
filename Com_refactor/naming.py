"""Utilities for parsing image file names."""

from __future__ import annotations

import os
import re


_PREVIEW_RE = re.compile(r"^_preview_(\d+)(?:\.[^.]+)?$")
_SCAN_RE = re.compile(
    r"^(?P<session>.+)_t(?P<tilt>[+-]?\d+)_p(?P<pan>[+-]?\d+)_led_(?P<led_state>on|off)(?:\.[^.]+)?$"
)
_POINTING_ITER_RE = re.compile(r"^(?P<label>.+)_(?P<iteration>\d+)$")


def parse_image_name(name: str) -> dict:
    """Parse image naming conventions used by the GUI client.

    Returns only metadata; it never raises and falls back to {"kind": "other"}.
    """
    try:
        if not isinstance(name, str):
            return {"kind": "other"}

        base = os.path.basename(name.strip())
        if not base:
            return {"kind": "other"}

        if base.startswith("_preview_"):
            match = _PREVIEW_RE.match(base)
            timestamp = int(match.group(1)) if match else None
            return {"kind": "preview", "timestamp": timestamp}

        scan_match = _SCAN_RE.match(base)
        if scan_match:
            return {
                "kind": "scan",
                "session": scan_match.group("session"),
                "tilt": int(scan_match.group("tilt")),
                "pan": int(scan_match.group("pan")),
                "led_state": scan_match.group("led_state"),
            }

        stem = os.path.splitext(base)[0]
        if stem.startswith("pointing_"):
            remainder = stem[len("pointing_") :]
            iter_match = _POINTING_ITER_RE.match(remainder)
            if iter_match:
                return {
                    "kind": "pointing",
                    "label": iter_match.group("label"),
                    "iteration": int(iter_match.group("iteration")),
                }
            return {"kind": "pointing", "label": remainder}

        return {"kind": "other"}
    except Exception:
        return {"kind": "other"}
