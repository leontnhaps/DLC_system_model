#!/usr/bin/env python3
"""
Find the last Phase 3 plus/base laser ON/OFF pairs in a pointing capture folder
and visualize diff-based weighted centroids.

Centroid rule:
- use intensity-weighted centroid on laser gray_diff >= 30
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from tkinter import Tk, filedialog


PHASE3_PATTERN = re.compile(
    r"iter_(?P<iter>\d+)_phase3_(?P<phase3>\d+)_(?P<tag>plus|base|minus)_(?P<kind>laser|led)_(?P<state>on|off)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)
DIFF_EXIST_THRESHOLD = 30
LED_DIFF_EXCLUDE_THRESHOLD = 50
CLICK_ROI_SIZE = 360
DEFAULT_CLICK_ROI_CENTER = (1375, 780)
FIXED_REFERENCE_POINT = (1360, 760)


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resize_to_fit(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    height, width = img.shape[:2]
    if width <= 0 or height <= 0:
        return img
    scale = min(float(max_w) / float(width), float(max_h) / float(height), 1.0)
    if scale >= 0.999:
        return img
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def choose_folder(root: Tk) -> Path | None:
    path = filedialog.askdirectory(parent=root, title="Select pointing capture folder")
    return Path(path) if path else None


def find_last_phase3_pairs(folder: Path) -> tuple[int, dict[str, dict[str, dict[str, Path]]]]:
    grouped: dict[int, dict[str, dict[str, dict[str, Path]]]] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = PHASE3_PATTERN.match(path.name)
        if not match:
            continue
        iter_no = int(match.group("iter"))
        tag = match.group("tag").lower()
        kind = match.group("kind").lower()
        state = match.group("state").lower()
        grouped.setdefault(iter_no, {}).setdefault(tag, {}).setdefault(kind, {})[state] = path

    if not grouped:
        raise RuntimeError("No Phase 3 laser ON/OFF files were found in this folder.")

    valid_iters = [
        iter_no
        for iter_no, tag_map in grouped.items()
        if "plus" in tag_map
        and "base" in tag_map
        and "laser" in tag_map["plus"]
        and "led" in tag_map["plus"]
        and "laser" in tag_map["base"]
        and "led" in tag_map["base"]
        and "on" in tag_map["plus"]["laser"]
        and "off" in tag_map["plus"]["laser"]
        and "on" in tag_map["plus"]["led"]
        and "off" in tag_map["plus"]["led"]
        and "on" in tag_map["base"]["laser"]
        and "off" in tag_map["base"]["laser"]
        and "on" in tag_map["base"]["led"]
        and "off" in tag_map["base"]["led"]
    ]
    if not valid_iters:
        raise RuntimeError("No complete plus/base LED/LASER ON/OFF set was found.")

    last_iter = max(valid_iters)
    return last_iter, grouped[last_iter]


def compute_diff_response(img_on: np.ndarray, img_off: np.ndarray) -> np.ndarray:
    gray_on = cv2.cvtColor(img_on, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_off = cv2.cvtColor(img_off, cv2.COLOR_BGR2GRAY).astype(np.float32)
    delta_pos = np.clip(gray_on - gray_off, 0.0, None)
    return np.clip(delta_pos, 0.0, 255.0).astype(np.uint8)


def compute_diff_heatmap(gray_u8: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)


def compute_weighted_centroid(
    gray_u8: np.ndarray,
    threshold: int = DIFF_EXIST_THRESHOLD,
    exclude_mask: np.ndarray | None = None,
    roi_box: tuple[int, int, int, int] | None = None,
) -> tuple[tuple[int, int] | None, int]:
    valid_mask = gray_u8 >= int(threshold)
    if exclude_mask is not None:
        valid_mask &= ~(exclude_mask > 0)
    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        roi_mask = np.zeros_like(valid_mask, dtype=bool)
        roi_mask[y1:y2, x1:x2] = True
        valid_mask &= roi_mask

    ys, xs = np.nonzero(valid_mask)
    if len(xs) <= 0:
        return None, 0

    weights = gray_u8[ys, xs].astype(np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return None, int(len(xs))

    cx = int(round(float(np.sum(xs * weights) / weight_sum)))
    cy = int(round(float(np.sum(ys * weights) / weight_sum)))
    return (cx, cy), int(len(xs))


def compute_diff_visualization(
    gray_u8: np.ndarray,
    threshold: int = DIFF_EXIST_THRESHOLD,
    exclude_mask: np.ndarray | None = None,
    roi_box: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int] | None, int]:
    heatmap = compute_diff_heatmap(gray_u8)
    centroid, pixel_count = compute_weighted_centroid(
        gray_u8,
        threshold=threshold,
        exclude_mask=exclude_mask,
        roi_box=roi_box,
    )
    return heatmap, centroid, pixel_count


def compute_midpoint(
    a: tuple[int, int] | None,
    b: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if a is None or b is None:
        return None
    return (
        int(round((float(a[0]) + float(b[0])) / 2.0)),
        int(round((float(a[1]) + float(b[1])) / 2.0)),
    )


def build_center_roi(
    image_shape: tuple[int, ...],
    center: tuple[int, int],
    size: int = CLICK_ROI_SIZE,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    half = int(round(float(size) / 2.0))
    cx = int(center[0])
    cy = int(center[1])
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(width, x1 + int(size))
    y2 = min(height, y1 + int(size))
    x1 = max(0, x2 - int(size))
    y1 = max(0, y2 - int(size))
    return (int(x1), int(y1), int(x2), int(y2))


def draw_marker(
    canvas: np.ndarray,
    pos: tuple[int, int] | None,
    color: tuple[int, int, int],
    size: int = 30,
) -> None:
    if pos is None:
        return
    x, y = pos
    outer_size = int(size) + 10
    mid_size = int(size) + 4
    ring_radius = max(8, int(size // 2))

    # dark shadow so the marker survives bright backgrounds
    cv2.drawMarker(canvas, (x, y), (0, 0, 0), cv2.MARKER_TILTED_CROSS, outer_size, 8)
    cv2.circle(canvas, (x, y), ring_radius + 3, (0, 0, 0), 6)

    # white outline for contrast on dark regions and heatmaps
    cv2.drawMarker(canvas, (x, y), (255, 255, 255), cv2.MARKER_TILTED_CROSS, mid_size, 5)
    cv2.circle(canvas, (x, y), ring_radius, (255, 255, 255), 4)

    # final colored marker
    cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_TILTED_CROSS, size, 2)
    cv2.circle(canvas, (x, y), ring_radius - 2, color, 2)
    cv2.circle(canvas, (x, y), 4, color, -1)


def draw_fixed_reference_marker(
    canvas: np.ndarray,
    pos: tuple[int, int] = FIXED_REFERENCE_POINT,
) -> None:
    x, y = int(pos[0]), int(pos[1])
    cv2.drawMarker(canvas, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 72, 12)
    cv2.drawMarker(canvas, (x, y), (0, 0, 0), cv2.MARKER_CROSS, 58, 8)
    cv2.circle(canvas, (x, y), 18, (255, 255, 255), 6)
    cv2.circle(canvas, (x, y), 14, (0, 0, 0), 4)


def draw_roi_box(
    canvas: np.ndarray,
    roi_box: tuple[int, int, int, int] | None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    if roi_box is None:
        return
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color, 2)


def apply_diff_presence_mask(
    img: np.ndarray,
    gray_diff: np.ndarray,
    threshold: int = DIFF_EXIST_THRESHOLD,
    exclude_mask: np.ndarray | None = None,
    roi_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    canvas = np.zeros_like(img)
    valid_mask = gray_diff >= int(threshold)
    if exclude_mask is not None:
        valid_mask &= ~(exclude_mask > 0)
    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        roi_mask = np.zeros_like(valid_mask, dtype=bool)
        roi_mask[y1:y2, x1:x2] = True
        valid_mask &= roi_mask
    canvas[valid_mask] = img[valid_mask]
    return canvas


def compute_led_diff_exclude_mask(
    img_led_on: np.ndarray,
    img_led_off: np.ndarray,
    threshold: int = LED_DIFF_EXCLUDE_THRESHOLD,
) -> np.ndarray:
    led_diff = compute_diff_response(img_led_on, img_led_off)
    return (led_diff > int(threshold)).astype(np.uint8) * 255


def draw_plus_overlay(
    plus_img: np.ndarray,
    plus_centroid: tuple[int, int] | None,
    plus_pixels: int,
    source_name: str,
    roi_box: tuple[int, int, int, int] | None = None,
    metric_label: str = "gray diff >= 30 weighted centroid",
) -> np.ndarray:
    canvas = plus_img.copy()
    draw_roi_box(canvas, roi_box)
    draw_fixed_reference_marker(canvas)
    draw_marker(canvas, plus_centroid, (0, 255, 255), 30)
    lines = [
        f"Phase3 plus | {metric_label}",
        f"source={source_name}",
        f"plus diff_pixels={plus_pixels}",
        "yellow = plus weighted centroid",
        "black cross = fixed ref (1355, 770)",
        "green box = clicked/default 300x300 ROI",
    ]
    y = 36
    for line in lines:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 30
    return canvas


def draw_base_overlay(
    base_img: np.ndarray,
    plus_centroid: tuple[int, int] | None,
    base_centroid: tuple[int, int] | None,
    avg_centroid: tuple[int, int] | None,
    base_pixels: int,
    source_name: str,
    roi_box: tuple[int, int, int, int] | None = None,
    metric_label: str = "gray diff >= 30 weighted centroid",
) -> np.ndarray:
    canvas = base_img.copy()
    draw_roi_box(canvas, roi_box)
    draw_fixed_reference_marker(canvas)
    draw_marker(canvas, plus_centroid, (0, 255, 255), 28)
    draw_marker(canvas, base_centroid, (255, 255, 0), 32)
    draw_marker(canvas, avg_centroid, (255, 0, 255), 36)

    lines = [
        f"Phase3 base | {metric_label}",
        f"source={source_name}",
        f"base diff_pixels={base_pixels}",
        "yellow = plus weighted centroid",
        "cyan = base weighted centroid",
        "magenta = avg(plus, base)",
        "black cross = fixed ref (1355, 770)",
        "green box = clicked/default 300x300 ROI",
    ]
    y = 36
    for line in lines:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 30
    return canvas


def draw_heatmap_overlay(
    heatmap: np.ndarray,
    plus_centroid: tuple[int, int] | None = None,
    base_centroid: tuple[int, int] | None = None,
    avg_centroid: tuple[int, int] | None = None,
    roi_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    canvas = heatmap.copy()
    draw_roi_box(canvas, roi_box)
    draw_fixed_reference_marker(canvas)
    draw_marker(canvas, plus_centroid, (0, 255, 255), 28)
    draw_marker(canvas, base_centroid, (255, 255, 0), 32)
    draw_marker(canvas, avg_centroid, (255, 0, 255), 36)
    return canvas


def draw_threshold_plot(
    masked_img: np.ndarray,
    tag: str,
    plus_centroid: tuple[int, int] | None = None,
    base_centroid: tuple[int, int] | None = None,
    avg_centroid: tuple[int, int] | None = None,
    roi_box: tuple[int, int, int, int] | None = None,
    metric_label: str = "laser diff",
) -> np.ndarray:
    canvas = masked_img.copy()
    draw_roi_box(canvas, roi_box)
    draw_fixed_reference_marker(canvas)
    draw_marker(canvas, plus_centroid, (0, 255, 255), 28)
    draw_marker(canvas, base_centroid, (255, 255, 0), 32)
    draw_marker(canvas, avg_centroid, (255, 0, 255), 36)
    cv2.putText(
        canvas,
        f"{tag} | only {metric_label} >= {DIFF_EXIST_THRESHOLD}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        "left click: set 300x300 ROI, right click: clear, r: reset all",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    return canvas


def draw_threshold_plot_with_exclusion(
    masked_img: np.ndarray,
    tag: str,
    plus_centroid: tuple[int, int] | None = None,
    base_centroid: tuple[int, int] | None = None,
    avg_centroid: tuple[int, int] | None = None,
    exclude_mask: np.ndarray | None = None,
    roi_box: tuple[int, int, int, int] | None = None,
    metric_label: str = "laser diff",
) -> np.ndarray:
    canvas = masked_img.copy()
    draw_roi_box(canvas, roi_box)
    draw_fixed_reference_marker(canvas)
    if exclude_mask is not None and np.count_nonzero(exclude_mask) > 0:
        contours, _ = cv2.findContours(exclude_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2)
    draw_marker(canvas, plus_centroid, (0, 255, 255), 28)
    draw_marker(canvas, base_centroid, (255, 255, 0), 32)
    draw_marker(canvas, avg_centroid, (255, 0, 255), 36)
    cv2.putText(
        canvas,
        f"{tag} | {metric_label} >= {DIFF_EXIST_THRESHOLD}, excluding LED diff > {LED_DIFF_EXCLUDE_THRESHOLD}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        "left click: set 300x300 ROI, right click: clear, r: reset all",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    return canvas


def make_side_by_side(left: np.ndarray, right: np.ndarray, title: str) -> tuple[np.ndarray, dict[str, int]]:
    height = max(left.shape[0], right.shape[0])
    left_pad = cv2.copyMakeBorder(left, 0, height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    right_pad = cv2.copyMakeBorder(right, 0, height - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    gap = np.zeros((height, 24, 3), dtype=np.uint8)
    canvas = np.concatenate([left_pad, gap, right_pad], axis=1)
    cv2.putText(canvas, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    meta = {
        "left_w": int(left.shape[1]),
        "left_h": int(left.shape[0]),
        "gap_w": int(gap.shape[1]),
        "right_w": int(right.shape[1]),
        "right_h": int(right.shape[0]),
        "pair_w": int(canvas.shape[1]),
        "pair_h": int(canvas.shape[0]),
    }
    return canvas, meta


def map_click_to_side(
    x: int,
    y: int,
    view_shape: tuple[int, int, int],
    pair_meta: dict[str, int],
) -> tuple[str | None, tuple[int, int] | None]:
    view_h, view_w = view_shape[:2]
    if view_w <= 0 or view_h <= 0:
        return None, None

    scale = min(float(view_w) / float(pair_meta["pair_w"]), float(view_h) / float(pair_meta["pair_h"]))
    if scale <= 0.0:
        return None, None

    orig_x = float(x) / scale
    orig_y = float(y) / scale

    left_w = float(pair_meta["left_w"])
    left_h = float(pair_meta["left_h"])
    gap_w = float(pair_meta["gap_w"])
    right_w = float(pair_meta["right_w"])
    right_h = float(pair_meta["right_h"])

    if 0.0 <= orig_x < left_w and 0.0 <= orig_y < left_h:
        return "plus", (int(round(orig_x)), int(round(orig_y)))

    right_x0 = left_w + gap_w
    if right_x0 <= orig_x < right_x0 + right_w and 0.0 <= orig_y < right_h:
        return "base", (int(round(orig_x - right_x0)), int(round(orig_y)))

    return None, None


def main():
    root = Tk()
    root.withdraw()

    folder = choose_folder(root)
    if folder is None:
        print("[INFO] Folder selection cancelled.")
        return

    last_iter, pairs = find_last_phase3_pairs(folder)
    print(f"Selected folder : {folder}")
    print(f"Last Phase 3 iter: {last_iter}")

    plus_laser_on_path = pairs["plus"]["laser"]["on"]
    plus_laser_off_path = pairs["plus"]["laser"]["off"]
    plus_led_on_path = pairs["plus"]["led"]["on"]
    plus_led_off_path = pairs["plus"]["led"]["off"]
    base_laser_on_path = pairs["base"]["laser"]["on"]
    base_laser_off_path = pairs["base"]["laser"]["off"]
    base_led_on_path = pairs["base"]["led"]["on"]
    base_led_off_path = pairs["base"]["led"]["off"]

    plus_laser_on = imread_unicode(plus_laser_on_path)
    plus_laser_off = imread_unicode(plus_laser_off_path)
    plus_led_on = imread_unicode(plus_led_on_path)
    plus_led_off = imread_unicode(plus_led_off_path)
    base_laser_on = imread_unicode(base_laser_on_path)
    base_laser_off = imread_unicode(base_laser_off_path)
    base_led_on = imread_unicode(base_led_on_path)
    base_led_off = imread_unicode(base_led_off_path)
    if any(
        img is None
        for img in (
            plus_laser_on,
            plus_laser_off,
            plus_led_on,
            plus_led_off,
            base_laser_on,
            base_laser_off,
            base_led_on,
            base_led_off,
        )
    ):
        raise RuntimeError("Failed to load one or more plus/base LED/LASER images.")

    plus_diff_u8 = compute_diff_response(plus_laser_on, plus_laser_off)
    base_diff_u8 = compute_diff_response(base_laser_on, base_laser_off)
    plus_led_exclude_mask = compute_led_diff_exclude_mask(plus_led_on, plus_led_off)
    base_led_exclude_mask = compute_led_diff_exclude_mask(base_led_on, base_led_off)

    print(f"PLUS LASER ON  : {plus_laser_on_path}")
    print(f"PLUS LASER OFF : {plus_laser_off_path}")
    print(f"PLUS LED ON    : {plus_led_on_path}")
    print(f"PLUS LED OFF   : {plus_led_off_path}")
    print(f"BASE LASER ON  : {base_laser_on_path}")
    print(f"BASE LASER OFF : {base_laser_off_path}")
    print(f"BASE LED ON    : {base_led_on_path}")
    print(f"BASE LED OFF   : {base_led_off_path}")

    screen_w = max(640, int(root.winfo_screenwidth() * 0.92))
    screen_h = max(480, int(root.winfo_screenheight() * 0.82))
    window_names = [
        "Phase3 Plus Base Centroids + Avg Point",
        "Phase3 Plus Base Heatmaps + Avg Point",
        "Phase3 Laser Diff >= Threshold Regions",
        "Phase3 Laser Diff >= Threshold Regions Excluding LED Diff > 50",
    ]
    default_plus_roi = build_center_roi(plus_laser_on.shape, DEFAULT_CLICK_ROI_CENTER, size=CLICK_ROI_SIZE)
    default_base_roi = build_center_roi(base_laser_on.shape, DEFAULT_CLICK_ROI_CENTER, size=CLICK_ROI_SIZE)
    state: dict[str, object] = {
        "plus_roi_box": default_plus_roi,
        "base_roi_box": default_base_roi,
        "window_meta": {},
    }

    def render() -> None:
        plus_roi_box = state.get("plus_roi_box")
        base_roi_box = state.get("base_roi_box")

        plus_heatmap_base, plus_centroid, plus_pixels = compute_diff_visualization(
            plus_diff_u8,
            roi_box=plus_roi_box,
        )
        base_heatmap_base, base_centroid, base_pixels = compute_diff_visualization(
            base_diff_u8,
            roi_box=base_roi_box,
        )
        avg_centroid = compute_midpoint(plus_centroid, base_centroid)
        plus_threshold_plot = apply_diff_presence_mask(plus_laser_on, plus_diff_u8, roi_box=plus_roi_box)
        base_threshold_plot = apply_diff_presence_mask(base_laser_on, base_diff_u8, roi_box=base_roi_box)
        plus_excluded_centroid, plus_excluded_pixels = compute_weighted_centroid(
            plus_diff_u8,
            exclude_mask=plus_led_exclude_mask,
            roi_box=plus_roi_box,
        )
        base_excluded_centroid, base_excluded_pixels = compute_weighted_centroid(
            base_diff_u8,
            exclude_mask=base_led_exclude_mask,
            roi_box=base_roi_box,
        )
        avg_excluded_centroid = compute_midpoint(plus_excluded_centroid, base_excluded_centroid)
        plus_threshold_excluded_plot = apply_diff_presence_mask(
            plus_laser_on,
            plus_diff_u8,
            exclude_mask=plus_led_exclude_mask,
            roi_box=plus_roi_box,
        )
        base_threshold_excluded_plot = apply_diff_presence_mask(
            base_laser_on,
            base_diff_u8,
            exclude_mask=base_led_exclude_mask,
            roi_box=base_roi_box,
        )
        print(f"PLUS centroid  : {plus_centroid} diff_pixels={plus_pixels} (threshold={DIFF_EXIST_THRESHOLD})")
        print(
            f"PLUS excl LED>50: centroid={plus_excluded_centroid} diff_pixels={plus_excluded_pixels} "
            f"excluded={int(np.count_nonzero(plus_led_exclude_mask))}"
        )
        print(f"BASE centroid  : {base_centroid} diff_pixels={base_pixels} (threshold={DIFF_EXIST_THRESHOLD})")
        print(
            f"BASE excl LED>50: centroid={base_excluded_centroid} diff_pixels={base_excluded_pixels} "
            f"excluded={int(np.count_nonzero(base_led_exclude_mask))}"
        )
        print(f"AVG centroid   : {avg_centroid}")
        print(f"AVG excl LED>50: {avg_excluded_centroid}")
        print(f"ROI plus/base  : {plus_roi_box} / {base_roi_box}")

        plus_overlay = draw_plus_overlay(
            plus_laser_on,
            plus_centroid,
            plus_pixels,
            plus_laser_on_path.name,
            roi_box=plus_roi_box,
            metric_label="gray diff >= 30 weighted centroid",
        )
        base_overlay = draw_base_overlay(
            base_laser_on,
            plus_centroid,
            base_centroid,
            avg_centroid,
            base_pixels,
            base_laser_on_path.name,
            roi_box=base_roi_box,
            metric_label="gray diff >= 30 weighted centroid",
        )
        plus_heatmap = draw_heatmap_overlay(
            plus_heatmap_base,
            plus_centroid=plus_centroid,
            roi_box=plus_roi_box,
        )
        base_heatmap = draw_heatmap_overlay(
            base_heatmap_base,
            plus_centroid=plus_centroid,
            base_centroid=base_centroid,
            avg_centroid=avg_centroid,
            roi_box=base_roi_box,
        )
        plus_threshold_view = draw_threshold_plot(
            plus_threshold_plot,
            "Phase3 plus",
            plus_centroid=plus_centroid,
            roi_box=plus_roi_box,
            metric_label="gray diff",
        )
        base_threshold_view = draw_threshold_plot(
            base_threshold_plot,
            "Phase3 base",
            plus_centroid=plus_centroid,
            base_centroid=base_centroid,
            avg_centroid=avg_centroid,
            roi_box=base_roi_box,
            metric_label="gray diff",
        )
        plus_threshold_excluded_view = draw_threshold_plot_with_exclusion(
            plus_threshold_excluded_plot,
            "Phase3 plus",
            plus_centroid=plus_excluded_centroid,
            exclude_mask=plus_led_exclude_mask,
            roi_box=plus_roi_box,
            metric_label="gray diff",
        )
        base_threshold_excluded_view = draw_threshold_plot_with_exclusion(
            base_threshold_excluded_plot,
            "Phase3 base",
            plus_centroid=plus_excluded_centroid,
            base_centroid=base_excluded_centroid,
            avg_centroid=avg_excluded_centroid,
            exclude_mask=base_led_exclude_mask,
            roi_box=base_roi_box,
            metric_label="gray diff",
        )
        overlay_pair, overlay_meta = make_side_by_side(
            plus_overlay,
            base_overlay,
            f"Last Phase3 plus/base centroids + avg point | iter {last_iter}",
        )
        heatmap_pair, heatmap_meta = make_side_by_side(
            plus_heatmap,
            base_heatmap,
            f"Last Phase3 plus/base diff heatmaps + avg point | iter {last_iter}",
        )
        threshold_pair, threshold_meta = make_side_by_side(
            plus_threshold_view,
            base_threshold_view,
            f"Last Phase3 laser diff >= {DIFF_EXIST_THRESHOLD} regions | iter {last_iter}",
        )
        threshold_excluded_pair, threshold_excluded_meta = make_side_by_side(
            plus_threshold_excluded_view,
            base_threshold_excluded_view,
            f"Last Phase3 laser diff >= {DIFF_EXIST_THRESHOLD} regions excluding LED diff > {LED_DIFF_EXCLUDE_THRESHOLD} | iter {last_iter}",
        )

        overlay_view = resize_to_fit(overlay_pair, screen_w, screen_h)
        heatmap_view = resize_to_fit(heatmap_pair, screen_w, screen_h)
        threshold_view_fit = resize_to_fit(threshold_pair, screen_w, screen_h)
        threshold_excluded_view_fit = resize_to_fit(threshold_excluded_pair, screen_w, screen_h)

        cv2.imshow(window_names[0], overlay_view)
        cv2.imshow(window_names[1], heatmap_view)
        cv2.imshow(window_names[2], threshold_view_fit)
        cv2.imshow(window_names[3], threshold_excluded_view_fit)

        state["window_meta"] = {
            window_names[0]: {"pair_meta": overlay_meta, "view_shape": overlay_view.shape},
            window_names[1]: {"pair_meta": heatmap_meta, "view_shape": heatmap_view.shape},
            window_names[2]: {"pair_meta": threshold_meta, "view_shape": threshold_view_fit.shape},
            window_names[3]: {"pair_meta": threshold_excluded_meta, "view_shape": threshold_excluded_view_fit.shape},
        }

    def on_mouse(event: int, x: int, y: int, flags: int, window_name: str) -> None:
        meta = state.get("window_meta", {}).get(window_name)
        if meta is None:
            return
        side, local_pos = map_click_to_side(
            x,
            y,
            meta["view_shape"],
            meta["pair_meta"],
        )
        if side is None or local_pos is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            image_shape = plus_laser_on.shape if side == "plus" else base_laser_on.shape
            state[f"{side}_roi_box"] = build_center_roi(image_shape, local_pos, size=CLICK_ROI_SIZE)
            render()
        elif event == cv2.EVENT_RBUTTONDOWN:
            state[f"{side}_roi_box"] = None
            render()

    render()
    for window_name in window_names:
        cv2.setMouseCallback(window_name, on_mouse, window_name)

    print(
        f"[INFO] Default ROI center: {DEFAULT_CLICK_ROI_CENTER}, size={CLICK_ROI_SIZE} "
        f"-> plus/base {default_plus_roi} / {default_base_roi}"
    )
    print("[INFO] Left click: set 300x300 ROI for clicked side.")
    print("[INFO] Right click: clear ROI for clicked side. Press 'r' to reset all, 'q' or ESC to close.")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            state["plus_roi_box"] = default_plus_roi
            state["base_roi_box"] = default_base_roi
            render()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
