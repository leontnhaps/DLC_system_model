#!/usr/bin/env python3
"""
Visualize an 8cm x 8cm area centered 11.75cm below the detected film center.

Flow:
1. Load LED ON / LED OFF images
2. Compute absolute diff
3. Detect the film on the diff image using tiled YOLO inference
4. Estimate px/cm from the real film size (5.5cm x 5.5cm)
5. Draw the target area on screen
"""

import argparse
import csv
import re
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np

from yolo_utils import YOLOProcessor, predict_with_tiling


REAL_FILM_SIZE_CM = 5.5
AREA_SIDE_CM = 8.0
AREA_CENTER_OFFSET_CM = 11.75

YOLO_TILE_ROWS = 2
YOLO_TILE_COLS = 3
YOLO_TILE_OVERLAP = 0.15
YOLO_CONF_THRESHOLD = 0.20
YOLO_IOU_THRESHOLD = 0.45
YOLO_SCORE_FILTER = 0.50
RESPONSE_COLORS = [
    (255, 0, 0),    # Blue
    (255, 255, 0),  # Cyan
    (0, 255, 0),    # Green
    (0, 255, 255),  # Yellow
    (0, 165, 255),  # Orange
    (0, 0, 255),    # Red
]
RESPONSE_VALUE_BINS = [0, 43, 86, 129, 172, 215, 256]


def imread_unicode(filepath):
    """Read an image safely even when the path contains non-ASCII characters."""
    try:
        stream = np.fromfile(filepath, dtype=np.uint8)
        return cv2.imdecode(stream, cv2.IMREAD_COLOR)
    except Exception as exc:
        print(f"[ERROR] Failed to read image: {exc}")
        return None


def resize_for_display(img, max_width=1600, max_height=900):
    """Resize for display while preserving aspect ratio."""
    if img is None:
        return None

    height, width = img.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    if scale >= 1.0:
        return img

    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def select_image_pair():
    """Select LED ON and LED OFF images using a single Tk root."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    initial_dir = Path(__file__).resolve().parent
    filetypes = (("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*"))

    path_on = filedialog.askopenfilename(
        initialdir=str(initial_dir),
        title="1. Select LED ON image",
        filetypes=filetypes,
        parent=root,
    )
    if not path_on:
        root.destroy()
        return None, None

    path_off = filedialog.askopenfilename(
        initialdir=str(Path(path_on).parent),
        title="2. Select LED OFF image",
        filetypes=filetypes,
        parent=root,
    )
    root.destroy()
    if not path_off:
        return None, None
    return path_on, path_off


def select_input_directory():
    """Select a directory that contains LED ON/OFF image pairs."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected_dir = filedialog.askdirectory(
        initialdir=str(Path(__file__).resolve().parents[2]),
        title="Select a folder containing LED ON/OFF image pairs",
        parent=root,
    )
    root.destroy()
    return selected_dir or None


def select_weights_path(default_path=None):
    """Use a default weights path when present, otherwise ask the user."""
    if default_path and Path(default_path).exists():
        return str(default_path)

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    weights_path = filedialog.askopenfilename(
        initialdir=str(Path(__file__).resolve().parents[2]),
        title="Select YOLO weights (.pt)",
        filetypes=(("PyTorch Weights", "*.pt"), ("All Files", "*.*")),
        parent=root,
    )
    root.destroy()
    return weights_path or None


def resolve_default_weights():
    """Pick the most likely diff-model weights if present in the repo root."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "yolov11m_diff.pt",
        repo_root / "yolov11n_diff.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_pair_key(path, base_dir):
    """Normalize LED/LASER ON/OFF filenames into a common measurement key."""
    relative_parent = path.parent.relative_to(base_dir)
    stem_key = re.sub(r"(?:led|laser)_(?:on|off)", "pair", path.stem, flags=re.IGNORECASE)
    return relative_parent.as_posix(), stem_key.lower()


def find_measurement_groups(base_dir):
    """
    Find measurement groups recursively.

    Expected patterns include names like:
    - iter_1_led_on.jpg / iter_1_led_off.jpg
    - pointing_phase3_led_on_11_11_base.jpg / pointing_phase3_led_off_11_11_base.jpg
    - pointing_phase3_laser_on_11_11_base.jpg / pointing_phase3_laser_off_11_11_base.jpg
    """
    base_path = Path(base_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    groups = {}

    for path in base_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in image_exts:
            continue

        stem_lower = path.stem.lower()
        if not any(token in stem_lower for token in ("led_on", "led_off", "laser_on", "laser_off")):
            continue

        key = normalize_pair_key(path, base_path)
        entry = groups.setdefault(
            key,
            {
                "relative_parent": path.parent.relative_to(base_path),
                "base_name": re.sub(r"(?:led|laser)_(?:on|off)", "pair", path.stem, flags=re.IGNORECASE),
                "led_on": None,
                "led_off": None,
                "laser_on": None,
                "laser_off": None,
            },
        )

        if "led_on" in stem_lower:
            entry["led_on"] = path
        elif "led_off" in stem_lower:
            entry["led_off"] = path
        elif "laser_on" in stem_lower:
            entry["laser_on"] = path
        elif "laser_off" in stem_lower:
            entry["laser_off"] = path

    valid_groups = [
        group for group in groups.values()
        if group["led_on"] is not None and group["led_off"] is not None
    ]
    valid_groups.sort(key=lambda item: (str(item["relative_parent"]).lower(), item["base_name"].lower()))
    return valid_groups


def detect_film(diff_img, weights_path):
    """Run tiled YOLO detection and keep every box above the system cutoff."""
    yolo = YOLOProcessor()
    model = yolo.get_model(weights_path)
    if model is None:
        raise RuntimeError("YOLO model could not be loaded.")

    device = yolo.get_device()
    boxes, scores, classes = predict_with_tiling(
        model,
        diff_img,
        rows=YOLO_TILE_ROWS,
        cols=YOLO_TILE_COLS,
        overlap=YOLO_TILE_OVERLAP,
        conf=YOLO_CONF_THRESHOLD,
        iou=YOLO_IOU_THRESHOLD,
        device=device,
    )

    if boxes:
        keep = [idx for idx, score in enumerate(scores) if score >= YOLO_SCORE_FILTER]
        if keep:
            boxes = [boxes[idx] for idx in keep]
            scores = [scores[idx] for idx in keep]
            classes = [classes[idx] for idx in keep]
        else:
            boxes, scores, classes = [], [], []

    if not boxes:
        return None

    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        detections.append({
            "box": box,
            "score": float(score),
            "class_id": int(cls),
        })

    detections.sort(key=lambda item: (-item["score"], -(item["box"][2] * item["box"][3])))
    return detections


def compute_area_geometry(box):
    """Convert the detected film size into an 8cm area 11.75cm below the center."""
    x, y, w, h = box

    film_center_x = x + w / 2.0
    film_center_y = y + h / 2.0

    px_per_cm_x = w / REAL_FILM_SIZE_CM
    px_per_cm_y = h / REAL_FILM_SIZE_CM
    px_per_cm = (px_per_cm_x + px_per_cm_y) / 2.0

    area_center_x = film_center_x
    area_center_y = film_center_y + AREA_CENTER_OFFSET_CM * px_per_cm

    area_side_px = AREA_SIDE_CM * px_per_cm
    half_side_px = area_side_px / 2.0

    area_x1 = int(round(area_center_x - half_side_px))
    area_y1 = int(round(area_center_y - half_side_px))
    area_x2 = int(round(area_center_x + half_side_px))
    area_y2 = int(round(area_center_y + half_side_px))

    return {
        "film_center": (film_center_x, film_center_y),
        "px_per_cm_x": px_per_cm_x,
        "px_per_cm_y": px_per_cm_y,
        "px_per_cm": px_per_cm,
        "area_center": (area_center_x, area_center_y),
        "area_side_px": area_side_px,
        "area_box": (area_x1, area_y1, area_x2, area_y2),
    }


def clip_area_box(area_box, shape):
    """Clip an area box to image bounds."""
    height, width = shape[:2]
    x1, y1, x2, y2 = area_box
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def compute_laser_response(laser_on_img, laser_off_img, geometries):
    """Compute positive laser response and per-area metrics."""
    if laser_on_img is None or laser_off_img is None:
        return None, []

    if laser_on_img.shape != laser_off_img.shape:
        raise RuntimeError(f"Laser image size mismatch: ON={laser_on_img.shape}, OFF={laser_off_img.shape}")

    gray_on = cv2.cvtColor(laser_on_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_off = cv2.cvtColor(laser_off_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    delta_pos = np.clip(gray_on - gray_off, 0.0, None)

    per_area_metrics = []
    for geometry in geometries:
        x1, y1, x2, y2 = clip_area_box(geometry["area_box"], delta_pos.shape)
        roi = delta_pos[y1:y2, x1:x2]
        positive = roi[roi > 0]

        if positive.size == 0:
            metrics = {
                "mean_delta": 0.0,
                "max_delta": 0.0,
                "core_delta": 0.0,
            }
        else:
            top_count = max(1, int(np.ceil(positive.size * 0.10)))
            top_values = np.partition(positive, -top_count)[-top_count:]
            metrics = {
                "mean_delta": float(np.mean(positive)),
                "max_delta": float(np.max(positive)),
                "core_delta": float(np.mean(top_values)),
            }

        per_area_metrics.append(metrics)

    return delta_pos, per_area_metrics


def draw_laser_response_overlay(base_img, geometries, delta_pos, metrics, panel_title):
    """Color the brightness change inside each area box using response levels."""
    canvas = base_img.copy()
    area_mask = np.zeros(delta_pos.shape, dtype=np.uint8)

    for geometry in geometries:
        x1, y1, x2, y2 = clip_area_box(geometry["area_box"], delta_pos.shape)
        if x2 > x1 and y2 > y1:
            area_mask[y1:y2, x1:x2] = 255

    delta_area = delta_pos[area_mask > 0]
    global_max = float(np.max(delta_area)) if delta_area.size > 0 else 0.0

    if delta_area.size > 0:
        color_layer = np.zeros_like(canvas)
        for idx in range(len(RESPONSE_VALUE_BINS) - 1):
            low = RESPONSE_VALUE_BINS[idx]
            high = RESPONSE_VALUE_BINS[idx + 1]
            selection = (area_mask > 0) & (delta_pos >= low) & (delta_pos < high)
            color_layer[selection] = RESPONSE_COLORS[idx]

        blended = cv2.addWeighted(canvas, 0.55, color_layer, 0.45, 0)
        canvas[area_mask > 0] = blended[area_mask > 0]

    for idx, (geometry, metric) in enumerate(zip(geometries, metrics), start=1):
        x1, y1, x2, y2 = clip_area_box(geometry["area_box"], delta_pos.shape)
        center = (
            int(round(geometry["area_center"][0])),
            int(round(geometry["area_center"][1])),
        )
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.drawMarker(canvas, center, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
        label = f"#{idx} mean={metric['mean_delta']:.1f} core={metric['core_delta']:.1f} max={metric['max_delta']:.1f}"
        label_y = max(28, y1 - 8)
        cv2.putText(canvas, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    legend_lines = [
        panel_title,
        "Laser response = max(gray(laser_on) - gray(laser_off), 0)",
        "Color scale uses absolute response bins on 0..255",
        f"Global area max delta = {global_max:.1f}",
    ]
    y_text = 35
    for line in legend_lines:
        cv2.putText(canvas, line, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y_text += 32

    legend_y = y_text + 10
    legend_labels = ["0-42", "43-85", "86-128", "129-171", "172-214", "215-255"]
    x_cursor = 20
    for color, label in zip(RESPONSE_COLORS, legend_labels):
        cv2.rectangle(canvas, (x_cursor, legend_y), (x_cursor + 28, legend_y + 20), color, -1)
        cv2.rectangle(canvas, (x_cursor, legend_y), (x_cursor + 28, legend_y + 20), (255, 255, 255), 1)
        cv2.putText(canvas, label, (x_cursor + 36, legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        x_cursor += 130

    return canvas


def draw_overlay(base_img, annotations, title):
    """Draw all detections above threshold with their computed area boxes."""
    canvas = base_img.copy()
    color_cycle = [
        ((0, 255, 0), (0, 165, 255)),
        ((255, 0, 0), (255, 255, 0)),
        ((255, 0, 255), (0, 255, 255)),
        ((0, 128, 255), (255, 128, 0)),
        ((128, 255, 0), (255, 0, 128)),
    ]

    for idx, item in enumerate(annotations, start=1):
        detection = item["detection"]
        geometry = item["geometry"]
        box_color, area_color = color_cycle[(idx - 1) % len(color_cycle)]

        x, y, w, h = detection["box"]
        area_x1, area_y1, area_x2, area_y2 = geometry["area_box"]

        film_center = (
            int(round(geometry["film_center"][0])),
            int(round(geometry["film_center"][1])),
        )
        area_center = (
            int(round(geometry["area_center"][0])),
            int(round(geometry["area_center"][1])),
        )

        cv2.rectangle(canvas, (x, y), (x + w, y + h), box_color, 3)
        cv2.drawMarker(
            canvas,
            film_center,
            box_color,
            markerType=cv2.MARKER_CROSS,
            markerSize=26,
            thickness=2,
        )

        cv2.rectangle(canvas, (area_x1, area_y1), (area_x2, area_y2), area_color, 3)
        cv2.drawMarker(
            canvas,
            area_center,
            area_color,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=28,
            thickness=2,
        )
        cv2.line(canvas, film_center, area_center, (255, 255, 255), 2)

        label = f"#{idx} score={detection['score']:.3f}"
        label_y = max(30, y - 10)
        cv2.putText(canvas, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)

    info_lines = [
        title,
        f"detections(score >= {YOLO_SCORE_FILTER:.2f}) = {len(annotations)}",
        f"offset={AREA_CENTER_OFFSET_CM:.2f}cm, area={AREA_SIDE_CM:.2f}cm x {AREA_SIDE_CM:.2f}cm",
        "Each detection has its own film center and 8cm x 8cm area box.",
    ]

    y_text = 35
    for line in info_lines:
        cv2.putText(canvas, line, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        y_text += 34

    return canvas


def make_comparison_view(diff_img, detection, geometry, laser_on_img=None, laser_panel=None):
    """Create a side-by-side result view for easier inspection."""
    annotations = [
        {"detection": det, "geometry": geo}
        for det, geo in zip(detection, geometry)
    ]
    overlay_diff = draw_overlay(diff_img, annotations, "DIFF + area overlay")
    panels = [overlay_diff]
    if laser_on_img is not None:
        overlay_laser_on = draw_overlay(laser_on_img, annotations, "Laser ON + area overlay")
        panels.append(overlay_laser_on)
    if laser_panel is not None:
        panels.append(laser_panel)
    return np.hstack(panels)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize an area below the detected solar cell film.")
    parser.add_argument("--dir", dest="input_dir", help="Directory containing LED ON/OFF image pairs")
    parser.add_argument("--on", dest="path_on", help="Path to LED ON image")
    parser.add_argument("--off", dest="path_off", help="Path to LED OFF image")
    parser.add_argument("--laser-on", dest="path_laser_on", help="Optional path to LASER ON image")
    parser.add_argument("--laser-off", dest="path_laser_off", help="Optional path to LASER OFF image")
    parser.add_argument("--weights", help="Path to YOLO weights (.pt)")
    parser.add_argument("--save", help="Optional output image path")
    return parser.parse_args()


def process_measurement(led_on_path, led_off_path, weights_path, laser_on_path=None, laser_off_path=None):
    """Process one measurement group and return area geometry plus optional laser response."""
    img_led_on = imread_unicode(str(led_on_path))
    img_led_off = imread_unicode(str(led_off_path))
    if img_led_on is None or img_led_off is None:
        raise RuntimeError("Failed to load the selected images.")

    if img_led_on.shape != img_led_off.shape:
        raise RuntimeError(f"Image size mismatch: ON={img_led_on.shape}, OFF={img_led_off.shape}")

    led_diff_img = cv2.absdiff(img_led_on, img_led_off)

    detections = detect_film(led_diff_img, weights_path)
    if detections is None:
        raise RuntimeError("No film was detected on the diff image.")

    geometries = [compute_area_geometry(detection["box"]) for detection in detections]
    laser_panel = None
    laser_metrics = []
    img_laser_on = None

    if laser_on_path and laser_off_path:
        img_laser_on = imread_unicode(str(laser_on_path))
        img_laser_off = imread_unicode(str(laser_off_path))
        if img_laser_on is None or img_laser_off is None:
            raise RuntimeError("Failed to load the selected laser images.")
        delta_pos, laser_metrics = compute_laser_response(img_laser_on, img_laser_off, geometries)
        laser_panel = draw_laser_response_overlay(
            led_diff_img,
            geometries,
            delta_pos,
            laser_metrics,
            panel_title="(LED)DIFF + response overlay",
        )

    result = make_comparison_view(
        led_diff_img,
        detections,
        geometries,
        laser_on_img=img_laser_on,
        laser_panel=laser_panel,
    )
    return {
        "led_diff_img": led_diff_img,
        "detections": detections,
        "geometries": geometries,
        "laser_metrics": laser_metrics,
        "result": result,
    }


def save_batch_result(output_root, input_root, pair, result_img):
    """Save one batch result while preserving the input folder structure."""
    relative_parent = pair["relative_parent"]
    save_dir = Path(output_root) / relative_parent
    save_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{pair['base_name']}_area.jpg"
    output_path = save_dir / output_name
    cv2.imwrite(str(output_path), result_img)
    return output_path


def save_batch_summary(output_root, rows):
    """Save a compact CSV summary for all processed pairs."""
    summary_path = Path(output_root) / "solarcell_area_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "status",
                "pair_name",
                "det_index",
                "relative_parent",
                "led_on",
                "led_off",
                "laser_on",
                "laser_off",
                "score",
                "bbox",
                "px_per_cm_x",
                "px_per_cm_y",
                "px_per_cm_avg",
                "film_center",
                "area_center",
                "area_box",
                "laser_mean_delta",
                "laser_core_delta",
                "laser_max_delta",
                "saved_result",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def run_batch_mode(input_dir, weights_path):
    """Process all LED ON/OFF pairs found under the selected directory."""
    base_dir = Path(input_dir)
    groups = find_measurement_groups(base_dir)
    if not groups:
        print(f"[INFO] No LED ON/OFF groups found under: {base_dir}")
        return

    output_root = base_dir / "_solarcell_area_results"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(groups)} measurement groups under: {base_dir}")
    print(f"[INFO] Results will be saved to: {output_root}")

    summary_rows = []
    success_count = 0

    for index, group in enumerate(groups, start=1):
        rel_parent_str = str(group["relative_parent"]) if str(group["relative_parent"]) != "." else ""
        pair_label = group["base_name"]
        print(f"[{index}/{len(groups)}] Processing: {pair_label}")

        row = {
            "status": "error",
            "pair_name": pair_label,
            "det_index": "",
            "relative_parent": rel_parent_str,
            "led_on": str(group["led_on"]),
            "led_off": str(group["led_off"]),
            "laser_on": str(group["laser_on"]) if group["laser_on"] else "",
            "laser_off": str(group["laser_off"]) if group["laser_off"] else "",
            "score": "",
            "bbox": "",
            "px_per_cm_x": "",
            "px_per_cm_y": "",
            "px_per_cm_avg": "",
            "film_center": "",
            "area_center": "",
            "area_box": "",
            "laser_mean_delta": "",
            "laser_core_delta": "",
            "laser_max_delta": "",
            "saved_result": "",
            "message": "",
        }

        try:
            processed = process_measurement(
                group["led_on"],
                group["led_off"],
                weights_path,
                laser_on_path=group["laser_on"],
                laser_off_path=group["laser_off"],
            )
            detections = processed["detections"]
            geometries = processed["geometries"]
            laser_metrics = processed["laser_metrics"]
            saved_path = save_batch_result(output_root, base_dir, group, processed["result"])
            for det_index, (detection, geometry) in enumerate(zip(detections, geometries), start=1):
                laser_metric = laser_metrics[det_index - 1] if det_index - 1 < len(laser_metrics) else {}
                summary_rows.append({
                    **row,
                    "status": "ok",
                    "det_index": det_index,
                    "score": f"{detection['score']:.6f}",
                    "bbox": str(detection["box"]),
                    "px_per_cm_x": f"{geometry['px_per_cm_x']:.6f}",
                    "px_per_cm_y": f"{geometry['px_per_cm_y']:.6f}",
                    "px_per_cm_avg": f"{geometry['px_per_cm']:.6f}",
                    "film_center": f"({geometry['film_center'][0]:.2f}, {geometry['film_center'][1]:.2f})",
                    "area_center": f"({geometry['area_center'][0]:.2f}, {geometry['area_center'][1]:.2f})",
                    "area_box": str(geometry["area_box"]),
                    "laser_mean_delta": f"{laser_metric.get('mean_delta', 0.0):.6f}",
                    "laser_core_delta": f"{laser_metric.get('core_delta', 0.0):.6f}",
                    "laser_max_delta": f"{laser_metric.get('max_delta', 0.0):.6f}",
                    "saved_result": str(saved_path),
                })
            success_count += 1
        except Exception as exc:
            row["message"] = str(exc)
            print(f"    [ERROR] {exc}")
            summary_rows.append(row)

    summary_path = save_batch_summary(output_root, summary_rows)
    print(f"[INFO] Batch complete: {success_count}/{len(groups)} succeeded")
    print(f"[INFO] Summary saved to: {summary_path}")


def run_single_mode(path_on, path_off, weights_path, save_path=None, laser_on_path=None, laser_off_path=None):
    """Process one pair and show the visualization window."""
    processed = process_measurement(
        path_on,
        path_off,
        weights_path,
        laser_on_path=laser_on_path,
        laser_off_path=laser_off_path,
    )
    detections = processed["detections"]
    geometries = processed["geometries"]
    laser_metrics = processed["laser_metrics"]
    result = processed["result"]

    print("=== Solar Cell Area Result ===")
    print(f"LED ON : {path_on}")
    print(f"LED OFF: {path_off}")
    if laser_on_path and laser_off_path:
        print(f"LASER ON : {laser_on_path}")
        print(f"LASER OFF: {laser_off_path}")
    print(f"YOLO   : {weights_path}")
    print(f"Detections >= {YOLO_SCORE_FILTER:.2f}: {len(detections)}")
    for det_index, (detection, geometry) in enumerate(zip(detections, geometries), start=1):
        laser_metric = laser_metrics[det_index - 1] if det_index - 1 < len(laser_metrics) else {}
        print(f"-- Detection #{det_index}")
        print(f"   box          : {detection['box']}")
        print(f"   score        : {detection['score']:.4f}")
        print(f"   px/cm (x)    : {geometry['px_per_cm_x']:.4f}")
        print(f"   px/cm (y)    : {geometry['px_per_cm_y']:.4f}")
        print(f"   px/cm (avg)  : {geometry['px_per_cm']:.4f}")
        print(f"   film center  : ({geometry['film_center'][0]:.2f}, {geometry['film_center'][1]:.2f})")
        print(f"   area center  : ({geometry['area_center'][0]:.2f}, {geometry['area_center'][1]:.2f})")
        print(f"   area box     : {geometry['area_box']}")
        if laser_metric:
            print(f"   laser mean   : {laser_metric.get('mean_delta', 0.0):.4f}")
            print(f"   laser core   : {laser_metric.get('core_delta', 0.0):.4f}")
            print(f"   laser max    : {laser_metric.get('max_delta', 0.0):.4f}")

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        print(f"Saved result to: {output_path}")

    preview = resize_for_display(result)
    cv2.imshow("Solar Cell Area", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    weights_path = select_weights_path(args.weights or resolve_default_weights())
    if not weights_path:
        print("[INFO] YOLO weights selection cancelled.")
        return

    if args.input_dir:
        run_batch_mode(args.input_dir, weights_path)
        return

    path_on = args.path_on
    path_off = args.path_off

    if not path_on and not path_off:
        selected_dir = select_input_directory()
        if selected_dir:
            run_batch_mode(selected_dir, weights_path)
            return

    if not path_on or not path_off:
        path_on, path_off = select_image_pair()
    if not path_on or not path_off:
        print("[INFO] Image selection cancelled.")
        return

    run_single_mode(
        path_on,
        path_off,
        weights_path,
        save_path=args.save,
        laser_on_path=args.path_laser_on,
        laser_off_path=args.path_laser_off,
    )


if __name__ == "__main__":
    main()
