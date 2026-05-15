#!/usr/bin/env python3
"""
Phase 3 solarcell response score debugger.

Flow:
1. Select LED ON image
2. Select LED OFF image
3. Select LASER ON image
4. Select LASER OFF image
5. Select YOLO weights if auto-detection fails
6. Compute the same Phase 3 solarcell-area response metrics used by the app

This script mirrors the production logic closely:
- object detection from LED ON/OFF diff
- center ROI X-band filtering
- px/cm estimation from bbox size
- 8cm x 8cm area box 11.75cm below target center
- response_mean / response_core / response_max from LASER ON/OFF diff
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
from tkinter import Tk, filedialog


REPO_ROOT = Path(__file__).resolve().parents[2]
COM_ROOT = REPO_ROOT / "Com"
for candidate in (str(COM_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from yolo_utils import YOLOProcessor  # noqa: E402


OBJECT_SIZE_CM = 5.5
PHASE3_TARGET_BELOW_CM = 11.75
PHASE3_AREA_SIDE_CM = 8.0
PHASE3_RESPONSE_TOP_RATIO = 0.10
PHASE23_CENTER_ROI_SIZE_PX = 800


@dataclass
class DetectionResult:
    bbox: tuple[int, int, int, int] | None
    all_bboxes: list[tuple[int, int, int, int]]
    diff_img: np.ndarray
    center_roi_box: tuple[int, int, int, int]


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    suffix = path.suffix.lower() or ".png"
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def choose_file(root: Tk, title: str) -> Path | None:
    path = filedialog.askopenfilename(
        parent=root,
        title=title,
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*")],
    )
    return Path(path) if path else None


def choose_weights(root: Tk) -> Path | None:
    for candidate in (
        REPO_ROOT / "yolov11m_diff.pt",
        REPO_ROOT / "yolov11n_diff.pt",
        COM_ROOT / "yolov11m_diff.pt",
        COM_ROOT / "yolov11n_diff.pt",
    ):
        if candidate.exists():
            return candidate

    path = filedialog.askopenfilename(
        parent=root,
        title="Select YOLO weights (.pt)",
        filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")],
    )
    return Path(path) if path else None


def resize_to_fit(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    height, width = img.shape[:2]
    if width <= 0 or height <= 0:
        return img
    scale = min(float(max_w) / float(width), float(max_h) / float(height), 1.0)
    if scale >= 0.999:
        return img
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def get_center_roi_box(shape: tuple[int, ...], size_px: int = PHASE23_CENTER_ROI_SIZE_PX) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    half = max(1, int(round(float(size_px) / 2.0)))
    cx = width // 2
    x1 = max(0, cx - half)
    x2 = min(width, cx + half)
    return (x1, 0, x2, height)


def point_in_box(px: float, py: float, box: tuple[int, int, int, int] | None) -> bool:
    if box is None:
        return True
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def detect_target_bbox(
    img_led_on: np.ndarray,
    img_led_off: np.ndarray,
    yolo: YOLOProcessor,
) -> DetectionResult:
    diff = cv2.absdiff(img_led_on, img_led_off)
    results = yolo.detect(diff, conf=0.20, iou=0.45)
    use_results = [r for r in results if len(r) >= 6 and float(r[4]) >= 0.5] or results

    all_bboxes: list[tuple[int, int, int, int]] = []
    candidates = []
    center_roi_box = get_center_roi_box(diff.shape)
    height, width = diff.shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0

    for item in use_results:
        x1, y1, x2, y2, conf, cls_id = item
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        if w <= 0.0 or h <= 0.0:
            continue
        cx = float(x1) + (w / 2.0)
        cy = float(y1) + (h / 2.0)
        bbox = (int(round(x1)), int(round(y1)), int(round(w)), int(round(h)))
        all_bboxes.append(bbox)
        dist = float(((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5)
        candidates.append(
            {
                "bbox": bbox,
                "dist": dist,
                "in_center_roi": point_in_box(cx, cy, center_roi_box),
            }
        )

    if candidates:
        selectable = [item for item in candidates if item["in_center_roi"]] or candidates
        selectable.sort(key=lambda item: item["dist"])
        return DetectionResult(
            bbox=selectable[0]["bbox"],
            all_bboxes=all_bboxes,
            diff_img=diff,
            center_roi_box=center_roi_box,
        )

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 30:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + (w / 2.0)
            cy = y + (h / 2.0)
            bbox = (int(x), int(y), int(w), int(h))
            all_bboxes = [bbox]
            if point_in_box(cx, cy, center_roi_box):
                return DetectionResult(
                    bbox=bbox,
                    all_bboxes=all_bboxes,
                    diff_img=diff,
                    center_roi_box=center_roi_box,
                )

    return DetectionResult(
        bbox=None,
        all_bboxes=all_bboxes,
        diff_img=diff,
        center_roi_box=center_roi_box,
    )


def build_area_geometry(
    bbox: tuple[int, int, int, int],
) -> dict[str, float | tuple[int, int, int, int]]:
    bx, by, bw, bh = [float(v) for v in bbox]
    obj_cx = bx + (bw / 2.0)
    obj_cy = by + (bh / 2.0)
    px_per_cm_x = bw / OBJECT_SIZE_CM
    px_per_cm_y = bh / OBJECT_SIZE_CM
    px_per_cm = max(1.0, (px_per_cm_x + px_per_cm_y) / 2.0)

    target_x = float(obj_cx)
    target_y = float(obj_cy + (PHASE3_TARGET_BELOW_CM * px_per_cm))
    area_side_px = float(PHASE3_AREA_SIDE_CM * px_per_cm)
    half_side_px = area_side_px / 2.0
    area_box = (
        int(round(target_x - half_side_px)),
        int(round(target_y - half_side_px)),
        int(round(target_x + half_side_px)),
        int(round(target_y + half_side_px)),
    )
    return {
        "target_x": float(target_x),
        "target_y": float(target_y),
        "px_per_cm_x": float(px_per_cm_x),
        "px_per_cm_y": float(px_per_cm_y),
        "px_per_cm": float(px_per_cm),
        "area_side_px": float(area_side_px),
        "area_box": area_box,
    }


def clip_area_box(area_box: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x1, y1, x2, y2 = area_box
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def compute_response_metrics(
    img_laser_on: np.ndarray,
    img_laser_off: np.ndarray,
    area_box: tuple[int, int, int, int],
) -> tuple[np.ndarray, dict[str, float | tuple[int, int] | tuple[int, int, int, int] | None]]:
    gray_on = cv2.cvtColor(img_laser_on, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_off = cv2.cvtColor(img_laser_off, cv2.COLOR_BGR2GRAY).astype(np.float32)
    delta_pos = np.clip(gray_on - gray_off, 0.0, None)
    delta_u8 = np.clip(delta_pos, 0.0, 255.0).astype(np.uint8)

    x1, y1, x2, y2 = clip_area_box(area_box, delta_u8.shape)
    if x2 <= x1 or y2 <= y1:
        return delta_u8, {
            "mean_delta": 0.0,
            "core_delta": 0.0,
            "max_delta": 0.0,
            "sum_delta": 0.0,
            "peak_pos": None,
            "clipped_area_box": (x1, y1, x2, y2),
        }

    roi = delta_pos[y1:y2, x1:x2]
    positive = roi[roi > 0]

    if positive.size <= 0:
        return delta_u8, {
            "mean_delta": 0.0,
            "core_delta": 0.0,
            "max_delta": 0.0,
            "sum_delta": 0.0,
            "peak_pos": None,
            "clipped_area_box": (x1, y1, x2, y2),
        }

    top_count = max(1, int(np.ceil(float(positive.size) * PHASE3_RESPONSE_TOP_RATIO)))
    top_values = np.partition(positive, -top_count)[-top_count:]
    peak_local = np.unravel_index(np.argmax(roi), roi.shape)
    peak_pos = (int(x1 + peak_local[1]), int(y1 + peak_local[0]))
    return delta_u8, {
        "mean_delta": float(np.mean(positive)),
        "core_delta": float(np.mean(top_values)),
        "max_delta": float(np.max(positive)),
        "sum_delta": float(np.sum(positive)),
        "peak_pos": peak_pos,
        "clipped_area_box": (x1, y1, x2, y2),
    }


def draw_overlay(
    base_img: np.ndarray,
    detection: DetectionResult,
    geometry: dict[str, float | tuple[int, int, int, int]],
    metrics: dict[str, float | tuple[int, int] | tuple[int, int, int, int] | None],
    title_lines: list[str],
) -> np.ndarray:
    canvas = base_img.copy()
    rx1, ry1, rx2, ry2 = detection.center_roi_box
    cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
    cv2.putText(canvas, f"CENTER ROI X {PHASE23_CENTER_ROI_SIZE_PX}px", (rx1, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    for bx, by, bw, bh in detection.all_bboxes:
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (100, 100, 100), 1)

    if detection.bbox is not None:
        bx, by, bw, bh = detection.bbox
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
        cv2.putText(canvas, "Selected BBox", (bx, max(20, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    x1, y1, x2, y2 = metrics.get("clipped_area_box", geometry["area_box"])
    cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(canvas, "Solarcell Area 8cm x 8cm", (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    tx = int(round(float(geometry["target_x"])))
    ty = int(round(float(geometry["target_y"])))
    cv2.drawMarker(canvas, (tx, ty), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 28, 2)

    peak_pos = metrics.get("peak_pos")
    if peak_pos is not None:
        px, py = peak_pos
        cv2.drawMarker(canvas, (int(px), int(py)), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)

    y = 32
    for line in title_lines:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        y += 28

    return canvas


def main():
    root = Tk()
    root.withdraw()

    led_on_path = choose_file(root, "Select Phase3 LED ON image")
    if led_on_path is None:
        print("[INFO] LED ON image selection cancelled.")
        return
    led_off_path = choose_file(root, "Select Phase3 LED OFF image")
    if led_off_path is None:
        print("[INFO] LED OFF image selection cancelled.")
        return
    laser_on_path = choose_file(root, "Select Phase3 LASER ON image")
    if laser_on_path is None:
        print("[INFO] LASER ON image selection cancelled.")
        return
    laser_off_path = choose_file(root, "Select Phase3 LASER OFF image")
    if laser_off_path is None:
        print("[INFO] LASER OFF image selection cancelled.")
        return

    weights_path = choose_weights(root)
    if weights_path is None:
        print("[INFO] YOLO weights selection cancelled.")
        return

    img_led_on = imread_unicode(led_on_path)
    img_led_off = imread_unicode(led_off_path)
    img_laser_on = imread_unicode(laser_on_path)
    img_laser_off = imread_unicode(laser_off_path)
    if any(img is None for img in (img_led_on, img_led_off, img_laser_on, img_laser_off)):
        raise RuntimeError("One or more selected images could not be loaded.")

    yolo = YOLOProcessor()
    if yolo.get_model(str(weights_path)) is None:
        raise RuntimeError(f"Failed to load YOLO weights: {weights_path}")

    detection = detect_target_bbox(img_led_on, img_led_off, yolo)
    if detection.bbox is None:
        raise RuntimeError("No target bbox detected from LED ON/OFF diff.")

    geometry = build_area_geometry(detection.bbox)
    response_u8, metrics = compute_response_metrics(
        img_laser_on=img_laser_on,
        img_laser_off=img_laser_off,
        area_box=geometry["area_box"],
    )

    print("=== Phase 3 Solarcell Score ===")
    print(f"LED ON  : {led_on_path}")
    print(f"LED OFF : {led_off_path}")
    print(f"LASER ON: {laser_on_path}")
    print(f"LASER OFF: {laser_off_path}")
    print(f"YOLO weights: {weights_path}")
    print(f"Selected bbox: {detection.bbox}")
    print(
        "px_per_cm: x={:.4f}, y={:.4f}, avg={:.4f}".format(
            float(geometry["px_per_cm_x"]),
            float(geometry["px_per_cm_y"]),
            float(geometry["px_per_cm"]),
        )
    )
    print(f"Target point: ({float(geometry['target_x']):.2f}, {float(geometry['target_y']):.2f})")
    print(f"Area side px: {float(geometry['area_side_px']):.2f}")
    print(f"Area box: {metrics['clipped_area_box']}")
    print(
        "Response mean/core/max/sum: {:.4f} / {:.4f} / {:.4f} / {:.4f}".format(
            float(metrics["mean_delta"]),
            float(metrics["core_delta"]),
            float(metrics["max_delta"]),
            float(metrics["sum_delta"]),
        )
    )
    print("mean: area 안의 양수 반응 평균")
    print("core: area 안의 가장 밝은 상위 10% 평균")
    print("max : area 안의 최대 반응값")
    print("sum : area 안의 양수 반응 전체 합")

    overlay = draw_overlay(
        base_img=img_laser_on,
        detection=detection,
        geometry=geometry,
        metrics=metrics,
        title_lines=[
            "Phase 3 Solarcell Score Debug",
            f"mean={float(metrics['mean_delta']):.2f} core={float(metrics['core_delta']):.2f} max={float(metrics['max_delta']):.2f} sum={float(metrics['sum_delta']):.1f}",
            f"px/cm={float(geometry['px_per_cm']):.2f} area={float(geometry['area_side_px']):.1f}px",
        ],
    )

    response_color = cv2.applyColorMap(response_u8, cv2.COLORMAP_JET)
    save_dir = laser_on_path.parent
    base_name = laser_on_path.stem
    overlay_path = save_dir / f"{base_name}_phase3_score_overlay.png"
    heatmap_path = save_dir / f"{base_name}_phase3_response_heatmap.png"
    diff_path = save_dir / f"{base_name}_phase3_response_gray.png"
    if not imwrite_unicode(overlay_path, overlay):
        raise RuntimeError(f"Failed to save overlay image: {overlay_path}")
    if not imwrite_unicode(heatmap_path, response_color):
        raise RuntimeError(f"Failed to save heatmap image: {heatmap_path}")
    if not imwrite_unicode(diff_path, response_u8):
        raise RuntimeError(f"Failed to save gray response image: {diff_path}")
    print(f"Saved overlay : {overlay_path}")
    print(f"Saved heatmap : {heatmap_path}")
    print(f"Saved graymap : {diff_path}")

    screen_w = max(640, int(root.winfo_screenwidth() * 0.92))
    screen_h = max(480, int(root.winfo_screenheight() * 0.82))
    overlay_view = resize_to_fit(overlay, screen_w, screen_h)
    response_view = resize_to_fit(response_color, screen_w, screen_h)

    cv2.imshow("Phase3 Solarcell Score", overlay_view)
    cv2.imshow("Phase3 Response Heatmap", response_view)
    print("[INFO] Press any key on an OpenCV window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
