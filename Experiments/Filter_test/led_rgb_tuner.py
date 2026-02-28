#!/usr/bin/env python3
import json
import pathlib
import datetime
import cv2
import numpy as np
from tkinter import Tk, filedialog


class LEDRGBTuner:
    def __init__(self):
        self.image_paths = []
        self.index = 0
        self.current_image = None
        self.current_path = None
        self.main_window = "LED RGB Tuner"
        self.ctrl_window = "RGB Filter Controls"
        self.mask_window = "RGB Masks"
        self.zoom_window = "ROI Zoom"
        self.roi_width = 100
        self.roi_height = 30
        self.zoom_scale = 8
        self.roi_rect = None  # (x, y, w, h) in original-image coordinates
        self._view_scale_x = 1.0
        self._view_scale_y = 1.0
        self.params = {
            "r_min": 60,
            "g_min": 60,
            "b_min": 60,
            "rg_min": 10,
            "rb_min": 40,
            "gr_min": 10,
            "gb_min": 10,
            "br_min": 40,
            "bg_min": 40,
            "min_pixels": 0,
        }

    def load_folder(self):
        root = Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="LED 테스트 이미지 폴더 선택")
        if not folder:
            return False

        base = pathlib.Path(folder)
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        paths = []
        for p in base.rglob("*"):
            if p.suffix.lower() in exts:
                paths.append(p)
        paths.sort()

        if not paths:
            print("No image files found.")
            return False

        self.image_paths = paths
        self.index = 0
        print(f"Loaded {len(paths)} images from: {base}")
        return True

    def _read_image(self, path):
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def _create_trackbars(self):
        cv2.namedWindow(self.ctrl_window, cv2.WINDOW_NORMAL)

        cv2.createTrackbar("R Min", self.ctrl_window, self.params["r_min"], 255, self._on_change)
        cv2.createTrackbar("G Min", self.ctrl_window, self.params["g_min"], 255, self._on_change)
        cv2.createTrackbar("B Min", self.ctrl_window, self.params["b_min"], 255, self._on_change)

        cv2.createTrackbar("R-G Min", self.ctrl_window, self.params["rg_min"], 255, self._on_change)
        cv2.createTrackbar("R-B Min", self.ctrl_window, self.params["rb_min"], 255, self._on_change)
        cv2.createTrackbar("G-R Min", self.ctrl_window, self.params["gr_min"], 255, self._on_change)
        cv2.createTrackbar("G-B Min", self.ctrl_window, self.params["gb_min"], 255, self._on_change)
        cv2.createTrackbar("B-R Min", self.ctrl_window, self.params["br_min"], 255, self._on_change)
        cv2.createTrackbar("B-G Min", self.ctrl_window, self.params["bg_min"], 255, self._on_change)

        cv2.createTrackbar("Min Pixels", self.ctrl_window, self.params["min_pixels"], 20000, self._on_change)

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

    def _make_roi_rect(self, cx, cy, img_w, img_h):
        roi_w = min(self.roi_width, img_w)
        roi_h = min(self.roi_height, img_h)
        x = int(cx - roi_w // 2)
        y = int(cy - roi_h // 2)
        x = max(0, min(img_w - roi_w, x))
        y = max(0, min(img_h - roi_h, y))
        return (x, y, roi_w, roi_h)

    def _on_mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.current_image is None:
            return

        h, w = self.current_image.shape[:2]
        sx = self._view_scale_x if self._view_scale_x > 0 else 1.0
        sy = self._view_scale_y if self._view_scale_y > 0 else 1.0

        ox = int(round(x / sx))
        oy = int(round(y / sy))
        ox = max(0, min(w - 1, ox))
        oy = max(0, min(h - 1, oy))

        self.roi_rect = self._make_roi_rect(ox, oy, w, h)
        rx, ry, rw, rh = self.roi_rect
        print(f"ROI set: x={rx}, y={ry}, w={rw}, h={rh}")
        self._render()

    def _detect(self, img, roi_rect=None):
        work = img
        if roi_rect is not None:
            x, y, w, h = roi_rect
            work = img[y:y + h, x:x + w]

        b, g, r = cv2.split(work)

        # Channel dominance + brightness floor
        mask_r = (
            (r >= self.params["r_min"])
            & ((r.astype(np.int16) - g.astype(np.int16)) >= self.params["rg_min"])
            & ((r.astype(np.int16) - b.astype(np.int16)) >= self.params["rb_min"])
        )
        mask_g = (
            (g >= self.params["g_min"])
            & ((g.astype(np.int16) - r.astype(np.int16)) >= self.params["gr_min"])
            & ((g.astype(np.int16) - b.astype(np.int16)) >= self.params["gb_min"])
        )
        mask_b = (
            (b >= self.params["b_min"])
            & ((b.astype(np.int16) - r.astype(np.int16)) >= self.params["br_min"])
            & ((b.astype(np.int16) - g.astype(np.int16)) >= self.params["bg_min"])
        )

        mask_r = mask_r.astype(np.uint8) * 255
        mask_g = mask_g.astype(np.uint8) * 255
        mask_b = mask_b.astype(np.uint8) * 255

        cnt_r = int(np.count_nonzero(mask_r))
        cnt_g = int(np.count_nonzero(mask_g))
        cnt_b = int(np.count_nonzero(mask_b))

        min_pixels = self.params["min_pixels"]
        pred = "NONE"
        best = max(cnt_r, cnt_g, cnt_b)
        if best >= min_pixels:
            if best == cnt_r:
                pred = "R"
            elif best == cnt_g:
                pred = "G"
            else:
                pred = "B"

        return pred, (cnt_r, cnt_g, cnt_b), (mask_r, mask_g, mask_b)

    def _resize_for_view(self, img, max_w=1400, max_h=850):
        h, w = img.shape[:2]
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        if scale >= 1.0:
            return img
        nw = int(w * scale)
        nh = int(h * scale)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def _render_roi_zoom(self, img):
        if self.roi_rect is None:
            blank = np.zeros((140, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Click main image to set ROI (100x30)", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(blank, "Use +/- to change zoom", (12, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
            cv2.imshow(self.zoom_window, blank)
            return

        x, y, w, h = self.roi_rect
        roi = img[y:y + h, x:x + w]
        if roi.size == 0:
            return

        z = max(1, int(self.zoom_scale))
        zoom = cv2.resize(roi, (w * z, h * z), interpolation=cv2.INTER_NEAREST)

        zh, zw = zoom.shape[:2]
        cx = zw // 2
        cy = zh // 2
        cv2.line(zoom, (cx, 0), (cx, zh - 1), (0, 255, 255), 1)
        cv2.line(zoom, (0, cy), (zw - 1, cy), (0, 255, 255), 1)

        cv2.rectangle(zoom, (0, 0), (zw - 1, zh - 1), (0, 255, 255), 1)
        info = f"x={x} y={y} w={w} h={h} | zoom x{z}"
        cv2.putText(zoom, info, (8, max(18, min(28, zh - 6))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(self.zoom_window, zoom)

    def _render(self):
        if not self.image_paths:
            return
        self._read_trackbar_values()

        path = self.image_paths[self.index]
        if self.current_path != path:
            self.current_image = self._read_image(path)
            self.current_path = path

        if self.current_image is None:
            print(f"Failed to read image: {path}")
            return

        img = self.current_image
        h_img, w_img = img.shape[:2]
        if self.roi_rect is not None:
            cx = self.roi_rect[0] + self.roi_rect[2] // 2
            cy = self.roi_rect[1] + self.roi_rect[3] // 2
            self.roi_rect = self._make_roi_rect(cx, cy, w_img, h_img)

        pred, (cnt_r, cnt_g, cnt_b), masks = self._detect(img, roi_rect=self.roi_rect)

        vis = img.copy()
        text1 = f"[{self.index + 1}/{len(self.image_paths)}] {path.name}"
        roi_text = "FULL" if self.roi_rect is None else f"{self.roi_rect[2]}x{self.roi_rect[3]}"
        text2 = (
            f"Pred: {pred}  |  R={cnt_r}  G={cnt_g}  B={cnt_b}  |  "
            f"ROI={roi_text}  |  MinPixels={self.params['min_pixels']}"
        )
        cv2.putText(vis, text1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(vis, text2, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Prediction color marker
        color_map = {"R": (0, 0, 255), "G": (0, 255, 0), "B": (255, 0, 0), "NONE": (180, 180, 180)}
        cv2.rectangle(vis, (12, 70), (60, 118), color_map[pred], -1)
        cv2.rectangle(vis, (12, 70), (60, 118), (255, 255, 255), 2)

        if self.roi_rect is not None:
            x, y, w, h = self.roi_rect
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(vis, "ROI", (x, max(16, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        vis_show = self._resize_for_view(vis)
        self._view_scale_x = vis_show.shape[1] / float(vis.shape[1])
        self._view_scale_y = vis_show.shape[0] / float(vis.shape[0])
        cv2.imshow(self.main_window, vis_show)
        self._render_roi_zoom(img)

        # Mask panel
        mask_r, mask_g, mask_b = masks
        mr = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)
        mg = cv2.cvtColor(mask_g, cv2.COLOR_GRAY2BGR)
        mb = cv2.cvtColor(mask_b, cv2.COLOR_GRAY2BGR)

        h_mask, w_mask = mr.shape[:2]
        # ROI가 작을 때 라벨이 과도하게 커지는 문제 방지
        label_scale = max(0.35, min(0.7, h_mask / 320.0))
        label_thick = 1 if h_mask < 180 else 2
        label_y = min(h_mask - 8, max(18, int(24 * (h_mask / 260.0))))
        cv2.putText(mr, "R Mask", (6, label_y), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 255), label_thick)
        cv2.putText(mg, "G Mask", (6, label_y), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 255, 0), label_thick)
        cv2.putText(mb, "B Mask", (6, label_y), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 0, 0), label_thick)

        # ROI(예: 100x30)는 업스케일하지 않고, 큰 이미지만 축소
        target_h = min(260, h_mask)
        scale = target_h / float(h_mask)
        if abs(scale - 1.0) > 1e-6:
            tw = max(1, int(w_mask * scale))
            mr = cv2.resize(mr, (tw, target_h), interpolation=cv2.INTER_NEAREST)
            mg = cv2.resize(mg, (tw, target_h), interpolation=cv2.INTER_NEAREST)
            mb = cv2.resize(mb, (tw, target_h), interpolation=cv2.INTER_NEAREST)

        panel = np.hstack([mr, mg, mb])
        cv2.imshow(self.mask_window, panel)

    def _save_params(self):
        out_dir = pathlib.Path(__file__).resolve().parent
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"led_rgb_params_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.params, f, ensure_ascii=False, indent=2)
        print(f"Saved params: {out_path}")

    def run(self):
        if not self.load_folder():
            return

        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.mask_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.zoom_window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.main_window, self._on_mouse)
        self._create_trackbars()
        self._render()

        print("=" * 72)
        print("LED RGB Tuner")
        print("Keys: n/d=next, p/a=prev, s=save params, c=clear ROI, +/-=zoom, q/ESC=quit")
        print("Mouse: Left click on image -> set 100x30 ROI centered at click (W x H)")
        print("ROI mode: click to use 100x30 ROI, press c to return FULL-image mode.")
        print("=" * 72)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("n"), ord("d")):
                self.index = (self.index + 1) % len(self.image_paths)
                self._render()
            elif key in (ord("p"), ord("a")):
                self.index = (self.index - 1) % len(self.image_paths)
                self._render()
            elif key == ord("s"):
                self._save_params()
            elif key == ord("c"):
                self.roi_rect = None
                print("ROI cleared (FULL image mode).")
                self._render()
            elif key in (ord("+"), ord("=")):
                self.zoom_scale = min(20, self.zoom_scale + 1)
                print(f"Zoom scale: x{self.zoom_scale}")
                self._render()
            elif key in (ord("-"), ord("_")):
                self.zoom_scale = max(1, self.zoom_scale - 1)
                print(f"Zoom scale: x{self.zoom_scale}")
                self._render()
            elif key == ord("r"):
                self._render()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    LEDRGBTuner().run()
