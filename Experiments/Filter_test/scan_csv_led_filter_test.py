#!/usr/bin/env python3
"""
Scheduling-compatible LED probe viewer based on final CSV ROIs.

Flow:
1. Select one scan detections CSV
2. Read final_led_roi_* values per track_id
3. Select one image file
4. Cycle CSV ROIs and run classify_from_single_roi() on the selected image

Controls:
  R : next ROI
  E : previous ROI
  I : choose another image
  S : save filter params
  +/- : zoom in/out
  Q / ESC : quit
"""

from __future__ import annotations

import csv
import json
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

from led_filter import classify_from_single_roi, get_default_led_filter_params  # noqa: E402


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


class LEDSingleROIProbe:
    def __init__(self):
        self.root = Tk()
        self.root.withdraw()

        self.csv_path: Path | None = None
        self.image_path: Path | None = None
        self.current_image = None
        self.current_image_path: Path | None = None

        self.final_rois: dict[int, FinalROI] = {}
        self.ordered_track_ids: list[int] = []
        self.roi_index = 0

        self.main_window = "LED CSV ROI Probe"
        self.ctrl_window = "LED Probe Controls"
        self.mask_window = "RGB Masks"
        self.zoom_window = "ROI Zoom"

        self.zoom_scale = 10
        self._view_scale_x = 1.0
        self._view_scale_y = 1.0

        self.params = get_default_led_filter_params()
        self.params["rg_min"] = 170
        self.params["min_pixels"] = 200

    def _ask_open_file(self, title: str, filetypes) -> Path | None:
        try:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.lift()
            self.root.update_idletasks()
            self.root.withdraw()
            self.root.update()
        except Exception:
            pass

        selected = filedialog.askopenfilename(
            parent=self.root,
            title=title,
            filetypes=filetypes,
        )
        return Path(selected) if selected else None

    def _select_csv_file(self) -> Path | None:
        return self._ask_open_file(
            title="Select scan detections CSV",
            filetypes=[("CSV Files", "*.csv")],
        )

    def _select_image_file(self) -> Path | None:
        return self._ask_open_file(
            title="Select one image file",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")],
        )

    @staticmethod
    def _parse_int(row: dict, key: str) -> int | None:
        raw = row.get(key, "")
        if raw in ("", None):
            return None
        try:
            return int(round(float(raw)))
        except Exception:
            return None

    @staticmethod
    def _parse_float(row: dict, key: str) -> float | None:
        raw = row.get(key, "")
        if raw in ("", None):
            return None
        try:
            return float(raw)
        except Exception:
            return None

    @staticmethod
    def _read_image(path: Path):
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _load_final_rois(self, csv_path: Path):
        final_rois: dict[int, FinalROI] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                track_id = self._parse_int(row, "track_id")
                if track_id is None:
                    continue

                x = self._parse_int(row, "final_led_roi_x")
                y = self._parse_int(row, "final_led_roi_y")
                w = self._parse_int(row, "final_led_roi_w")
                h = self._parse_int(row, "final_led_roi_h")
                if x is None or y is None or w is None or h is None or w <= 0 or h <= 0:
                    continue

                src_w = self._parse_int(row, "final_led_roi_src_w")
                src_h = self._parse_int(row, "final_led_roi_src_h")
                if src_w is None:
                    src_w = self._parse_int(row, "W")
                if src_h is None:
                    src_h = self._parse_int(row, "H")

                final_rois[track_id] = FinalROI(
                    track_id=track_id,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    src_w=src_w,
                    src_h=src_h,
                    final_pan=self._parse_float(row, "final_pan_deg"),
                    final_tilt=self._parse_float(row, "final_tilt_deg"),
                )

        self.final_rois = dict(sorted(final_rois.items()))
        self.ordered_track_ids = list(self.final_rois.keys())
        self.roi_index = 0

    def load_inputs(self):
        self.csv_path = self._select_csv_file()
        if self.csv_path is None:
            print("No CSV selected.")
            return False

        self._load_final_rois(self.csv_path)
        if not self.final_rois:
            print(f"No final_led_roi_* values found in CSV: {self.csv_path}")
            return False

        print(f"Loaded CSV: {self.csv_path}")
        print(f"Loaded final ROI count: {len(self.final_rois)}")
        for track_id, roi in self.final_rois.items():
            print(
                f"  ID {track_id}: roi=({roi.x},{roi.y},{roi.w},{roi.h}) "
                f"src=({roi.src_w},{roi.src_h})"
            )

        self.image_path = self._select_image_file()
        if self.image_path is None:
            print("No image selected.")
            return False

        print(f"Initial image: {self.image_path}")
        return True

    def _create_trackbars(self):
        cv2.namedWindow(self.ctrl_window, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("R Min", self.ctrl_window, int(self.params["r_min"]), 255, self._on_change)
        cv2.createTrackbar("G Min", self.ctrl_window, int(self.params["g_min"]), 255, self._on_change)
        cv2.createTrackbar("B Min", self.ctrl_window, int(self.params["b_min"]), 255, self._on_change)
        cv2.createTrackbar("R-G Min", self.ctrl_window, int(self.params["rg_min"]), 255, self._on_change)
        cv2.createTrackbar("R-B Min", self.ctrl_window, int(self.params["rb_min"]), 255, self._on_change)
        cv2.createTrackbar("G-R Min", self.ctrl_window, int(self.params["gr_min"]), 255, self._on_change)
        cv2.createTrackbar("G-B Min", self.ctrl_window, int(self.params["gb_min"]), 255, self._on_change)
        cv2.createTrackbar("B-R Min", self.ctrl_window, int(self.params["br_min"]), 255, self._on_change)
        cv2.createTrackbar("B-G Min", self.ctrl_window, int(self.params["bg_min"]), 255, self._on_change)
        cv2.createTrackbar("Min Pixels", self.ctrl_window, int(self.params["min_pixels"]), 20000, self._on_change)

    def _read_trackbar_values(self):
        self.params["r_min"] = cv2.getTrackbarPos("R Min", self.ctrl_window)
        self.params["g_min"] = cv2.getTrackbarPos("G Min", self.ctrl_window)
        self.params["b_min"] = cv2.getTrackbarPos("B Min", self.ctrl_window)
        self.params["rg_min"] = cv2.getTrackbarPos("R-G Min", self.ctrl_window)
        self.params["rb_min"] = cv2.getTrackbarPos("R-B Min", self.ctrl_window)
        self.params["gr_min"] = cv2.getTrackbarPos("G-R Min", self.ctrl_window)
        self.params["gb_min"] = cv2.getTrackbarPos("G-B Min", self.ctrl_window)
        self.params["br_min"] = cv2.getTrackbarPos("B-R Min", self.ctrl_window)
        self.params["bg_min"] = cv2.getTrackbarPos("B-G Min", self.ctrl_window)
        self.params["min_pixels"] = cv2.getTrackbarPos("Min Pixels", self.ctrl_window)

    def _on_change(self, _=None):
        self._render()

    @staticmethod
    def _resize_for_view(img, max_w=1400, max_h=850):
        h, w = img.shape[:2]
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        if scale >= 1.0:
            return img
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def _get_current_roi(self):
        if not self.ordered_track_ids:
            return None
        track_id = self.ordered_track_ids[self.roi_index % len(self.ordered_track_ids)]
        return self.final_rois.get(track_id)

    def _scale_roi_to_image(self, roi: FinalROI, img_shape):
        img_h, img_w = img_shape[:2]
        src_w = roi.src_w if roi.src_w and roi.src_w > 0 else img_w
        src_h = roi.src_h if roi.src_h and roi.src_h > 0 else img_h
        sx = float(img_w) / float(src_w)
        sy = float(img_h) / float(src_h)

        x = int(round(roi.x * sx))
        y = int(round(roi.y * sy))
        w = int(round(roi.w * sx))
        h = int(round(roi.h * sy))
        return (x, y, max(1, w), max(1, h))

    @staticmethod
    def _expand_roi_downward(roi_rect, img_shape):
        if roi_rect is None:
            return None
        img_h, img_w = img_shape[:2]
        x, y, w, h = [int(round(v)) for v in roi_rect]
        if w <= 0 or h <= 0:
            return None

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + (2 * h))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2 - x1, y2 - y1)

    def _make_rgb_masks(self, roi_img):
        if roi_img is None or roi_img.size == 0:
            zero = np.zeros((1, 1), dtype=np.uint8)
            return zero, zero, zero

        b, g, r = cv2.split(roi_img)
        b16 = b.astype(np.int16)
        g16 = g.astype(np.int16)
        r16 = r.astype(np.int16)

        mask_r = (
            (r16 >= int(self.params["r_min"]))
            & ((r16 - g16) >= int(self.params["rg_min"]))
            & ((r16 - b16) >= int(self.params["rb_min"]))
        ).astype(np.uint8) * 255
        mask_g = (
            (g16 >= int(self.params["g_min"]))
            & ((g16 - r16) >= int(self.params["gr_min"]))
            & ((g16 - b16) >= int(self.params["gb_min"]))
        ).astype(np.uint8) * 255
        mask_b = (
            (b16 >= int(self.params["b_min"]))
            & ((b16 - r16) >= int(self.params["br_min"]))
            & ((b16 - g16) >= int(self.params["bg_min"]))
        ).astype(np.uint8) * 255

        return mask_r, mask_g, mask_b

    def _render_mask_panel(self, roi_img):
        if roi_img is None or roi_img.size == 0:
            blank = np.zeros((160, 720, 3), dtype=np.uint8)
            cv2.putText(blank, "No ROI available", (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(blank, "CSV final_led_roi_* is used as the ROI source", (16, 118),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow(self.mask_window, blank)
            return

        mask_r, mask_g, mask_b = self._make_rgb_masks(roi_img)
        mr = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)
        mg = cv2.cvtColor(mask_g, cv2.COLOR_GRAY2BGR)
        mb = cv2.cvtColor(mask_b, cv2.COLOR_GRAY2BGR)

        for img, label, color in (
            (mr, "R Mask", (0, 0, 255)),
            (mg, "G Mask", (0, 255, 0)),
            (mb, "B Mask", (255, 0, 0)),
        ):
            cv2.putText(img, label, (6, min(img.shape[0] - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        target_h = min(260, mr.shape[0]) if mr.shape[0] > 0 else 1
        scale = target_h / float(max(1, mr.shape[0]))
        if abs(scale - 1.0) > 1e-6:
            target_w = max(1, int(round(mr.shape[1] * scale)))
            mr = cv2.resize(mr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            mg = cv2.resize(mg, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            mb = cv2.resize(mb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        panel = np.hstack([mr, mg, mb])
        cv2.imshow(self.mask_window, panel)

    def _render_roi_zoom(self, img, roi_rect, pred, score, track_id):
        if roi_rect is None:
            blank = np.zeros((180, 760, 3), dtype=np.uint8)
            cv2.putText(blank, "No ROI available", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(blank, "Check final_led_roi_* values in the selected CSV", (16, 116),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow(self.zoom_window, blank)
            return

        x, y, w, h = roi_rect
        roi = img[y:y + h, x:x + w]
        if roi.size == 0:
            return

        z = max(1, int(self.zoom_scale))
        zoom = cv2.resize(roi, (w * z, h * z), interpolation=cv2.INTER_NEAREST)
        zh, zw = zoom.shape[:2]
        cv2.rectangle(zoom, (0, 0), (zw - 1, zh - 1), (0, 255, 255), 1)
        cv2.line(zoom, (zw // 2, 0), (zw // 2, zh - 1), (0, 255, 255), 1)
        cv2.line(zoom, (0, zh // 2), (zw - 1, zh // 2), (0, 255, 255), 1)

        overlay = np.zeros((max(190, zh + 78), max(zw, 600), 3), dtype=np.uint8)
        overlay[:zh, :zw] = zoom
        info_1 = f"Track ID={track_id} | ROI x={x} y={y} w={w} h={h} | zoom x{z}"
        info_2 = f"Pred={pred} | R={score['R']} G={score['G']} B={score['B']}"
        info_3 = "ROI source: CSV final_led_roi_*"
        cv2.putText(overlay, info_1, (10, zh + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, info_2, (10, zh + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(overlay, info_3, (10, zh + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.imshow(self.zoom_window, overlay)

    def _save_params(self):
        out_dir = Path(__file__).resolve().parent
        out = {
            "csv_path": str(self.csv_path) if self.csv_path is not None else None,
            "image_path": str(self.image_path) if self.image_path is not None else None,
            "current_track_id": self._get_current_roi().track_id if self._get_current_roi() is not None else None,
            "params": dict(self.params),
            "zoom_scale": int(self.zoom_scale),
        }
        out_path = out_dir / "led_single_roi_probe_params.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved params: {out_path}")

    def _load_current_image(self):
        if self.image_path is None:
            self.current_image = None
            self.current_image_path = None
            return None
        if self.current_image_path != self.image_path:
            self.current_image = self._read_image(self.image_path)
            self.current_image_path = self.image_path
        return self.current_image

    @staticmethod
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
        )
        return palette[track_id % len(palette)]

    def _render(self):
        self._read_trackbar_values()

        img = self._load_current_image()
        if img is None or self.image_path is None:
            print(f"Failed to read image: {self.image_path}")
            return

        current_roi = self._get_current_roi()
        if current_roi is None:
            print("No current ROI available.")
            return

        roi_seed_original = self._scale_roi_to_image(current_roi, img.shape)
        roi_seed = self._expand_roi_downward(roi_seed_original, img.shape)
        pred, score, roi_clamped = classify_from_single_roi(img, roi_seed, params=self.params)

        vis = img.copy()
        track_id = current_roi.track_id
        color = self._track_color(track_id)

        title_1 = f"CSV: {self.csv_path.name if self.csv_path else '-'}"
        title_2 = (
            f"Track {track_id} [{self.roi_index + 1}/{len(self.ordered_track_ids)}] | "
            f"Pred={pred} | R={score['R']} G={score['G']} B={score['B']}"
        )
        title_3 = (
            f"Image: {self.image_path.name} | "
            f"Final ROI src={current_roi.src_w}x{current_roi.src_h}"
        )

        cv2.putText(vis, title_1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
        cv2.putText(vis, title_2, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
        cv2.putText(vis, title_3, (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2)
        cv2.putText(
            vis,
            "ROI test mode: CSV ROI + same height downward | controls fully drive final Pred",
            (12, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            2,
        )
        cv2.putText(
            vis,
            "Keys: R/E ROI next/prev | I image | S save | +/- zoom | Q quit",
            (12, 146),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            2,
        )

        if current_roi.final_pan is not None and current_roi.final_tilt is not None:
            pose_text = f"Final pan/tilt = ({current_roi.final_pan:.1f}, {current_roi.final_tilt:.1f})"
            cv2.putText(vis, pose_text, (12, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

        color_map = {"R": (0, 0, 255), "G": (0, 255, 0), "B": (255, 0, 0), "NONE": (180, 180, 180)}
        cv2.rectangle(vis, (12, 188), (60, 236), color_map.get(pred, (180, 180, 180)), -1)
        cv2.rectangle(vis, (12, 188), (60, 236), (255, 255, 255), 2)
        cv2.putText(vis, pred, (72, 223), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color_map.get(pred, (180, 180, 180)), 2)

        if roi_seed_original is not None:
            ox, oy, ow, oh = roi_seed_original
            cv2.rectangle(vis, (ox, oy), (ox + ow, oy + oh), (140, 140, 140), 1)
            cv2.putText(
                vis,
                "orig",
                (ox, max(16, oy - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (140, 140, 140),
                1,
            )

        if roi_clamped is not None:
            x, y, w, h = roi_clamped
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                vis,
                f"ROI ID {track_id}",
                (x, max(16, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
            )
            roi_img = img[y:y + h, x:x + w]
        else:
            cv2.putText(
                vis,
                "ROI is outside image bounds after scaling/clamp.",
                (12, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 180, 255),
                2,
            )
            roi_img = None

        vis_show = self._resize_for_view(vis)
        self._view_scale_x = vis_show.shape[1] / float(vis.shape[1])
        self._view_scale_y = vis_show.shape[0] / float(vis.shape[0])

        cv2.imshow(self.main_window, vis_show)
        self._render_roi_zoom(img, roi_clamped, pred, score, track_id)
        self._render_mask_panel(roi_img)

    def _select_new_image(self):
        new_path = self._select_image_file()
        if new_path is not None:
            self.image_path = new_path
            self.current_image = None
            self.current_image_path = None
            print(f"Selected image: {self.image_path}")
            self._render()

    def run(self):
        if not self.load_inputs():
            return

        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.mask_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.zoom_window, cv2.WINDOW_NORMAL)
        self._create_trackbars()
        self._render()

        print("=" * 84)
        print("LED CSV ROI Probe")
        print("ROI source: CSV final_led_roi_* columns")
        print("Test ROI: original CSV ROI height expanded 2x downward")
        print("Classification path: classify_from_single_roi() from Com/vision/led_filter.py")
        print("No hidden fallback is applied in this test tool.")
        print("Final Pred is driven only by the current LED Probe Controls values.")
        print("Keys: R=next ROI, E=prev ROI, I=choose another image, S=save params, +/-=zoom, Q/ESC=quit")
        print("=" * 84)

        try:
            while True:
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    if self.ordered_track_ids:
                        self.roi_index = (self.roi_index + 1) % len(self.ordered_track_ids)
                        self._render()
                elif key in (ord("e"), ord("E")):
                    if self.ordered_track_ids:
                        self.roi_index = (self.roi_index - 1) % len(self.ordered_track_ids)
                        self._render()
                elif key in (ord("i"), ord("I")):
                    self._select_new_image()
                elif key == ord("s"):
                    self._save_params()
                elif key in (ord("+"), ord("=")):
                    self.zoom_scale = min(20, self.zoom_scale + 1)
                    self._render()
                elif key in (ord("-"), ord("_")):
                    self.zoom_scale = max(1, self.zoom_scale - 1)
                    self._render()
                elif key == ord(" "):
                    self._render()
        finally:
            cv2.destroyAllWindows()
            try:
                self.root.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    LEDSingleROIProbe().run()
