#!/usr/bin/env python3
"""
Shared LED color filter helpers for Scan / Pointing / Scheduling.
"""

import cv2
import numpy as np


DEFAULT_LED_FILTER_PARAMS = {
    "r_min": 60,
    "g_min": 60,
    "b_min": 60,
    "rg_min": 10,
    "rb_min": 10,
    "gr_min": 10,
    "gb_min": 10,
    "br_min": 10,
    "bg_min": 10,
    "min_pixels": 50,
}


def get_default_led_filter_params():
    """Return a copy of default LED filter params."""
    return dict(DEFAULT_LED_FILTER_PARAMS)


def led_score_to_bits(score, threshold=0):
    """
    Convert per-channel raw scores into independent 3-bit LED states.

    Bit order is fixed as ``R,B,G`` to match the target firmware mapping.
    A channel is considered ON only when it is both positive and above the
    configured threshold so that threshold=0 still keeps zero-count channels OFF.
    """
    raw = {
        "R": max(0, int((score or {}).get("R", 0))),
        "G": max(0, int((score or {}).get("G", 0))),
        "B": max(0, int((score or {}).get("B", 0))),
    }
    try:
        threshold = max(0, int(round(float(threshold))))
    except Exception:
        threshold = 0

    states = {
        "R": raw["R"] > 0 and raw["R"] >= threshold,
        "B": raw["B"] > 0 and raw["B"] >= threshold,
        "G": raw["G"] > 0 and raw["G"] >= threshold,
    }
    bits = f"{int(states['R'])}{int(states['B'])}{int(states['G'])}"
    return {
        "bits": bits,
        "states": dict(states),
        "threshold": int(threshold),
        "raw": dict(raw),
    }


def expand_led_roi_from_bbox(bbox, img_shape, top_ratio=1.0 / 3.0):
    """
    Build LED ROI around the top band of the YOLO bbox.
    The ROI spans one top-ratio band above the bbox and includes the
    same-height band inside the upper part of the bbox.
    bbox: (x, y, w, h)
    output: (x, y, w, h) clamped to image boundary.
    """
    if bbox is None:
        return None

    h_img, w_img = img_shape[:2]
    x, y, w, h = [int(round(v)) for v in bbox]
    if w <= 0 or h <= 0:
        return None

    x1 = max(0, x)
    x2 = min(w_img, x + w)
    led_h = max(1, int(round(h * float(top_ratio))))
    y2 = min(h_img, y + led_h)
    y1 = max(0, y - led_h)

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def _mask_counts_rgb(img_bgr, params):
    """Count RGB-dominant pixels with threshold rules."""
    if img_bgr is None or img_bgr.size == 0:
        return {"R": 0, "G": 0, "B": 0}

    b, g, r = cv2.split(img_bgr)
    b16 = b.astype(np.int16)
    g16 = g.astype(np.int16)
    r16 = r.astype(np.int16)

    mask_r = (
        (r16 >= int(params["r_min"]))
        & ((r16 - g16) >= int(params["rg_min"]))
        & ((r16 - b16) >= int(params["rb_min"]))
    )
    mask_g = (
        (g16 >= int(params["g_min"]))
        & ((g16 - r16) >= int(params["gr_min"]))
        & ((g16 - b16) >= int(params["gb_min"]))
    )
    mask_b = (
        (b16 >= int(params["b_min"]))
        & ((b16 - r16) >= int(params["br_min"]))
        & ((b16 - g16) >= int(params["bg_min"]))
    )

    return {
        "R": int(np.count_nonzero(mask_r)),
        "G": int(np.count_nonzero(mask_g)),
        "B": int(np.count_nonzero(mask_b)),
    }


def classify_from_on_off(img_on, img_off, bbox, params=None, top_ratio=1.0 / 3.0):
    """
    Classify LED color using ON/OFF pair on expanded ROI.

    Returns:
      pred: "R" | "G" | "B" | "NONE"
      score: {"R": int, "G": int, "B": int}  (on_count - off_count, floored at 0)
      roi: (x, y, w, h) or None
      on_counts: {"R": int, "G": int, "B": int}
      off_counts: {"R": int, "G": int, "B": int}
    """
    if params is None:
        params = DEFAULT_LED_FILTER_PARAMS

    roi = expand_led_roi_from_bbox(bbox, img_on.shape, top_ratio=top_ratio)
    if roi is None:
        zero = {"R": 0, "G": 0, "B": 0}
        return "NONE", dict(zero), None, dict(zero), dict(zero)

    x, y, w, h = roi
    roi_on = img_on[y:y + h, x:x + w]
    roi_off = img_off[y:y + h, x:x + w]

    on_counts = _mask_counts_rgb(roi_on, params)
    off_counts = _mask_counts_rgb(roi_off, params)

    score = {
        "R": max(0, on_counts["R"] - off_counts["R"]),
        "G": max(0, on_counts["G"] - off_counts["G"]),
        "B": max(0, on_counts["B"] - off_counts["B"]),
    }

    best_color = max(score, key=score.get)
    min_pixels = int(params.get("min_pixels", 0))
    if score[best_color] < min_pixels:
        pred = "NONE"
    else:
        pred = best_color

    return pred, score, roi, on_counts, off_counts


def classify_bits_from_on_off(img_on, img_off, bbox, params=None, top_ratio=1.0 / 3.0, threshold=None):
    """
    Classify LED bits using an ON/OFF pair on the expanded ROI.

    Returns:
      bits: "000" ~ "111"
      score: {"R": int, "G": int, "B": int}
      roi: (x, y, w, h) or None
      states: {"R": bool, "B": bool, "G": bool}
      on_counts/off_counts: original dominant-pixel counts
    """
    if params is None:
        params = DEFAULT_LED_FILTER_PARAMS
    pred, score, roi, on_counts, off_counts = classify_from_on_off(
        img_on,
        img_off,
        bbox,
        params=params,
        top_ratio=top_ratio,
    )
    _ = pred  # legacy compatibility helper already computed this.
    bit_result = led_score_to_bits(score, threshold=params.get("min_pixels", 0) if threshold is None else threshold)
    return bit_result["bits"], score, roi, bit_result["states"], on_counts, off_counts


def classify_from_single_roi(img_bgr, roi, params=None):
    """
    Classify LED color from a single frame using a precomputed ROI.

    Args:
      img_bgr: BGR frame
      roi: (x, y, w, h)
      params: filter params

    Returns:
      pred: "R" | "G" | "B" | "NONE"
      score: {"R": int, "G": int, "B": int}
      roi_clamped: (x, y, w, h) or None
    """
    if params is None:
        params = DEFAULT_LED_FILTER_PARAMS
    if img_bgr is None or img_bgr.size == 0 or roi is None:
        return "NONE", {"R": 0, "G": 0, "B": 0}, None

    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = [int(round(v)) for v in roi]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + max(0, w))
    y2 = min(h_img, y + max(0, h))
    if x2 <= x1 or y2 <= y1:
        return "NONE", {"R": 0, "G": 0, "B": 0}, None

    roi_clamped = (x1, y1, x2 - x1, y2 - y1)
    roi_img = img_bgr[y1:y2, x1:x2]
    score = _mask_counts_rgb(roi_img, params)

    best_color = max(score, key=score.get)
    min_pixels = int(params.get("min_pixels", 0))
    pred = best_color if score[best_color] >= min_pixels else "NONE"
    return pred, score, roi_clamped


def classify_bits_from_single_roi(img_bgr, roi, params=None, threshold=None):
    """
    Classify LED bits from a single frame using a precomputed ROI.

    Returns:
      bits: "000" ~ "111"
      score: {"R": int, "G": int, "B": int}
      roi_clamped: (x, y, w, h) or None
      states: {"R": bool, "B": bool, "G": bool}
    """
    if params is None:
        params = DEFAULT_LED_FILTER_PARAMS
    _pred, score, roi_clamped = classify_from_single_roi(img_bgr, roi, params=params)
    bit_result = led_score_to_bits(score, threshold=params.get("min_pixels", 0) if threshold is None else threshold)
    return bit_result["bits"], score, roi_clamped, bit_result["states"]
