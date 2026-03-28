#!/usr/bin/env python3
"""
Overlay final target ROIs stored in a scan CSV on a selected image.

Flow:
1. Select one scan detections CSV
2. Read unique final ROI per track_id from final_led_roi_* columns
3. Select one image file
4. Draw all final ROIs on that image

Controls:
  D : choose another image
  Q / ESC : quit
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tkinter import Tk, filedialog


@dataclass
class FinalROI:
    track_id: int
    x: int
    y: int
    w: int
    h: int
    src_w: int | None
    src_h: int | None
    final_pan: float | None
    final_tilt: float | None


def _read_image(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _ask_open_file(root: Tk, title: str, filetypes) -> Path | None:
    try:
        root.deiconify()
        root.attributes("-topmost", True)
        root.lift()
        root.update_idletasks()
        root.withdraw()
        root.update()
    except Exception:
        pass

    selected = filedialog.askopenfilename(
        parent=root,
        title=title,
        filetypes=filetypes,
    )
    return Path(selected) if selected else None


def _select_csv_file(root: Tk) -> Path | None:
    return _ask_open_file(
        root,
        title="Select scan detections CSV",
        filetypes=[("CSV Files", "*.csv")],
    )


def _select_image_file(root: Tk) -> Path | None:
    return _ask_open_file(
        root,
        title="Select one image file",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")],
    )


def _parse_int(row: dict, key: str) -> int | None:
    raw = row.get(key, "")
    if raw in ("", None):
        return None
    try:
        return int(round(float(raw)))
    except Exception:
        return None


def _parse_float(row: dict, key: str) -> float | None:
    raw = row.get(key, "")
    if raw in ("", None):
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _load_final_rois(csv_path: Path):
    final_rois: dict[int, FinalROI] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = _parse_int(row, "track_id")
            if track_id is None:
                continue

            x = _parse_int(row, "final_led_roi_x")
            y = _parse_int(row, "final_led_roi_y")
            w = _parse_int(row, "final_led_roi_w")
            h = _parse_int(row, "final_led_roi_h")
            if x is None or y is None or w is None or h is None or w <= 0 or h <= 0:
                continue

            src_w = _parse_int(row, "final_led_roi_src_w")
            src_h = _parse_int(row, "final_led_roi_src_h")
            if src_w is None:
                src_w = _parse_int(row, "W")
            if src_h is None:
                src_h = _parse_int(row, "H")

            final_rois[track_id] = FinalROI(
                track_id=track_id,
                x=x,
                y=y,
                w=w,
                h=h,
                src_w=src_w,
                src_h=src_h,
                final_pan=_parse_float(row, "final_pan_deg"),
                final_tilt=_parse_float(row, "final_tilt_deg"),
            )

    return dict(sorted(final_rois.items()))


def _track_color(track_id: int):
    palette = (
        (0, 255, 255),
        (0, 200, 0),
        (255, 160, 0),
        (255, 0, 180),
        (0, 180, 255),
        (180, 255, 0),
        (255, 80, 80),
        (180, 80, 255),
        (100, 255, 255),
        (255, 255, 100),
    )
    return palette[track_id % len(palette)]


def _resize_for_view(img, max_w=1600, max_h=950):
    h, w = img.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale >= 1.0:
        return img
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _draw_final_rois(base_img, final_rois: dict[int, FinalROI], image_path: Path, focus_track_id: int | None = None):
    canvas = base_img.copy()
    img_h, img_w = canvas.shape[:2]
    visible_count = 0

    cv2.putText(
        canvas,
        image_path.name,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        (
            f"final ROI count: {len(final_rois)}"
            if focus_track_id is None
            else f"focus ID: {focus_track_id} / total={len(final_rois)}"
        ),
        (18, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )

    if not final_rois:
        cv2.putText(
            canvas,
            "No final_led_roi_* found in CSV.",
            (18, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 180, 255),
            2,
        )
        return canvas

    for track_id, roi in final_rois.items():
        if focus_track_id is not None and int(track_id) != int(focus_track_id):
            continue
        src_w = roi.src_w if roi.src_w and roi.src_w > 0 else img_w
        src_h = roi.src_h if roi.src_h and roi.src_h > 0 else img_h
        sx = float(img_w) / float(src_w)
        sy = float(img_h) / float(src_h)

        x1 = int(round(roi.x * sx))
        y1 = int(round(roi.y * sy))
        x2 = int(round((roi.x + roi.w) * sx))
        y2 = int(round((roi.y + roi.h) * sy))

        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w - 1, x2))
        y2 = max(0, min(img_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        color = _track_color(track_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.circle(canvas, ((x1 + x2) // 2, (y1 + y2) // 2), 4, color, -1)
        visible_count += 1

        if roi.final_pan is not None and roi.final_tilt is not None:
            label = f"ID {track_id} ({roi.final_pan:.1f}, {roi.final_tilt:.1f})"
        else:
            label = f"ID {track_id}"
        label_y = y1 - 8 if y1 >= 22 else y2 + 20
        cv2.putText(
            canvas,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    if focus_track_id is not None:
        cv2.putText(
            canvas,
            f"single overlay mode | showing {visible_count} ROI",
            (18, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 255),
            2,
        )

    return canvas


def main():
    root = Tk()
    root.withdraw()
    try:
        csv_path = _select_csv_file(root)
        if csv_path is None:
            print("No CSV selected.")
            return

        final_rois = _load_final_rois(csv_path)
        print(f"Loaded CSV: {csv_path}")
        print(f"Loaded final ROI count: {len(final_rois)}")
        for track_id, roi in final_rois.items():
            print(
                f"  ID {track_id}: roi=({roi.x},{roi.y},{roi.w},{roi.h}) "
                f"src=({roi.src_w},{roi.src_h})"
            )

        image_path = _select_image_file(root)
        if image_path is None:
            print("No image selected.")
            return
        print(f"Initial image: {image_path}")

        window_name = "Final ROI Overlay Viewer"
        ordered_track_ids = list(final_rois.keys())
        focus_index = None

        while True:
            img = _read_image(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                view = np.zeros((300, 900, 3), dtype=np.uint8)
                cv2.putText(
                    view,
                    f"Failed to load: {image_path.name}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            else:
                focus_track_id = None if focus_index is None or not ordered_track_ids else ordered_track_ids[focus_index]
                view = _draw_final_rois(img, final_rois, image_path, focus_track_id=focus_track_id)

            view = _resize_for_view(view)
            footer = np.zeros((54, view.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                footer,
                "A: next ROI overlay | D: choose another image | Q/ESC: quit",
                (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (220, 220, 220),
                2,
            )
            panel = np.vstack([view, footer])
            cv2.imshow(window_name, panel)

            key = cv2.waitKeyEx(0)
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("a"), ord("A")):
                if ordered_track_ids:
                    if focus_index is None:
                        focus_index = 0
                    else:
                        focus_index = (focus_index + 1) % len(ordered_track_ids)
                    print(f"Overlay focus -> ID {ordered_track_ids[focus_index]}")
                continue
            if key in (2555904, ord("d"), ord("D")):
                new_image_path = _select_image_file(root)
                if new_image_path is not None:
                    image_path = new_image_path
                    print(f"Selected image: {image_path}")
                continue

        cv2.destroyAllWindows()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
