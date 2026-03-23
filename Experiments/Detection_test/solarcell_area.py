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
    """Normalize LED ON/OFF filenames into a common pair key."""
    relative_parent = path.parent.relative_to(base_dir)
    stem_key = re.sub(r"led_(?:on|off)", "led_pair", path.stem, flags=re.IGNORECASE)
    return relative_parent.as_posix(), stem_key.lower()


def find_led_pairs(base_dir):
    """
    Find LED ON/OFF pairs recursively.

    Expected patterns include names like:
    - iter_1_led_on.jpg / iter_1_led_off.jpg
    - pointing_phase3_led_on_11_11_base.jpg / pointing_phase3_led_off_11_11_base.jpg
    """
    base_path = Path(base_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs = {}

    for path in base_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in image_exts:
            continue

        stem_lower = path.stem.lower()
        if "led_on" not in stem_lower and "led_off" not in stem_lower:
            continue

        key = normalize_pair_key(path, base_path)
        entry = pairs.setdefault(
            key,
            {
                "relative_parent": path.parent.relative_to(base_path),
                "base_name": re.sub(r"led_(?:on|off)", "led_pair", path.stem, flags=re.IGNORECASE),
                "on": None,
                "off": None,
            },
        )

        if "led_on" in stem_lower:
            entry["on"] = path
        elif "led_off" in stem_lower:
            entry["off"] = path

    valid_pairs = [
        pair for pair in pairs.values()
        if pair["on"] is not None and pair["off"] is not None
    ]
    valid_pairs.sort(key=lambda item: (str(item["relative_parent"]).lower(), item["base_name"].lower()))
    return valid_pairs


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


def make_comparison_view(img_on, diff_img, detection, geometry):
    """Create a side-by-side result view for easier inspection."""
    annotations = [
        {"detection": det, "geometry": geo}
        for det, geo in zip(detection, geometry)
    ]
    overlay_on = draw_overlay(img_on, annotations, "LED ON + area overlay")
    overlay_diff = draw_overlay(diff_img, annotations, "DIFF + area overlay")
    return np.hstack([overlay_on, overlay_diff])


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize an area below the detected solar cell film.")
    parser.add_argument("--dir", dest="input_dir", help="Directory containing LED ON/OFF image pairs")
    parser.add_argument("--on", dest="path_on", help="Path to LED ON image")
    parser.add_argument("--off", dest="path_off", help="Path to LED OFF image")
    parser.add_argument("--weights", help="Path to YOLO weights (.pt)")
    parser.add_argument("--save", help="Optional output image path")
    return parser.parse_args()


def process_pair(path_on, path_off, weights_path):
    """Process one LED ON/OFF pair and return result images plus metrics."""
    img_on = imread_unicode(str(path_on))
    img_off = imread_unicode(str(path_off))
    if img_on is None or img_off is None:
        raise RuntimeError("Failed to load the selected images.")

    if img_on.shape != img_off.shape:
        raise RuntimeError(f"Image size mismatch: ON={img_on.shape}, OFF={img_off.shape}")

    diff_img = cv2.absdiff(img_on, img_off)

    detections = detect_film(diff_img, weights_path)
    if detections is None:
        raise RuntimeError("No film was detected on the diff image.")

    geometries = [compute_area_geometry(detection["box"]) for detection in detections]
    result = make_comparison_view(img_on, diff_img, detections, geometries)
    return diff_img, detections, geometries, result


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
                "score",
                "bbox",
                "px_per_cm_x",
                "px_per_cm_y",
                "px_per_cm_avg",
                "film_center",
                "area_center",
                "area_box",
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
    pairs = find_led_pairs(base_dir)
    if not pairs:
        print(f"[INFO] No LED ON/OFF pairs found under: {base_dir}")
        return

    output_root = base_dir / "_solarcell_area_results"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(pairs)} LED ON/OFF pairs under: {base_dir}")
    print(f"[INFO] Results will be saved to: {output_root}")

    summary_rows = []
    success_count = 0

    for index, pair in enumerate(pairs, start=1):
        rel_parent_str = str(pair["relative_parent"]) if str(pair["relative_parent"]) != "." else ""
        pair_label = pair["base_name"]
        print(f"[{index}/{len(pairs)}] Processing: {pair_label}")

        row = {
            "status": "error",
            "pair_name": pair_label,
            "det_index": "",
            "relative_parent": rel_parent_str,
            "led_on": str(pair["on"]),
            "led_off": str(pair["off"]),
            "score": "",
            "bbox": "",
            "px_per_cm_x": "",
            "px_per_cm_y": "",
            "px_per_cm_avg": "",
            "film_center": "",
            "area_center": "",
            "area_box": "",
            "saved_result": "",
            "message": "",
        }

        try:
            _, detections, geometries, result = process_pair(pair["on"], pair["off"], weights_path)
            saved_path = save_batch_result(output_root, base_dir, pair, result)
            for det_index, (detection, geometry) in enumerate(zip(detections, geometries), start=1):
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
                    "saved_result": str(saved_path),
                })
            success_count += 1
        except Exception as exc:
            row["message"] = str(exc)
            print(f"    [ERROR] {exc}")
            summary_rows.append(row)

    summary_path = save_batch_summary(output_root, summary_rows)
    print(f"[INFO] Batch complete: {success_count}/{len(pairs)} succeeded")
    print(f"[INFO] Summary saved to: {summary_path}")


def run_single_mode(path_on, path_off, weights_path, save_path=None):
    """Process one pair and show the visualization window."""
    _, detections, geometries, result = process_pair(path_on, path_off, weights_path)

    print("=== Solar Cell Area Result ===")
    print(f"LED ON : {path_on}")
    print(f"LED OFF: {path_off}")
    print(f"YOLO   : {weights_path}")
    print(f"Detections >= {YOLO_SCORE_FILTER:.2f}: {len(detections)}")
    for det_index, (detection, geometry) in enumerate(zip(detections, geometries), start=1):
        print(f"-- Detection #{det_index}")
        print(f"   box          : {detection['box']}")
        print(f"   score        : {detection['score']:.4f}")
        print(f"   px/cm (x)    : {geometry['px_per_cm_x']:.4f}")
        print(f"   px/cm (y)    : {geometry['px_per_cm_y']:.4f}")
        print(f"   px/cm (avg)  : {geometry['px_per_cm']:.4f}")
        print(f"   film center  : ({geometry['film_center'][0]:.2f}, {geometry['film_center'][1]:.2f})")
        print(f"   area center  : ({geometry['area_center'][0]:.2f}, {geometry['area_center'][1]:.2f})")
        print(f"   area box     : {geometry['area_box']}")

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

    run_single_mode(path_on, path_off, weights_path, args.save)


if __name__ == "__main__":
    main()
