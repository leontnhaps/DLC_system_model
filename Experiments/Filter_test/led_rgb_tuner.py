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
        self.params = {
            "r_min": 130,
            "g_min": 130,
            "b_min": 130,
            "rg_min": 25,
            "rb_min": 25,
            "gr_min": 25,
            "gb_min": 25,
            "br_min": 25,
            "bg_min": 25,
            "min_pixels": 60,
            "morph_kernel": 3,
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
        cv2.createTrackbar("Morph K", self.ctrl_window, self.params["morph_kernel"], 21, self._on_change)

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
        k = max(1, cv2.getTrackbarPos("Morph K", self.ctrl_window))
        if k % 2 == 0:
            k += 1
        self.params["morph_kernel"] = k

    def _on_change(self, _=None):
        self._render()

    def _morph(self, mask):
        k = self.params["morph_kernel"]
        kernel = np.ones((k, k), np.uint8)
        out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
        return out

    def _detect(self, img):
        b, g, r = cv2.split(img)

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

        mask_r = self._morph((mask_r.astype(np.uint8) * 255))
        mask_g = self._morph((mask_g.astype(np.uint8) * 255))
        mask_b = self._morph((mask_b.astype(np.uint8) * 255))

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
        pred, (cnt_r, cnt_g, cnt_b), masks = self._detect(img)

        vis = img.copy()
        text1 = f"[{self.index + 1}/{len(self.image_paths)}] {path.name}"
        text2 = f"Pred: {pred}  |  R={cnt_r}  G={cnt_g}  B={cnt_b}  |  MinPixels={self.params['min_pixels']}"
        cv2.putText(vis, text1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(vis, text2, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Prediction color marker
        color_map = {"R": (0, 0, 255), "G": (0, 255, 0), "B": (255, 0, 0), "NONE": (180, 180, 180)}
        cv2.rectangle(vis, (12, 70), (60, 118), color_map[pred], -1)
        cv2.rectangle(vis, (12, 70), (60, 118), (255, 255, 255), 2)

        vis_show = self._resize_for_view(vis)
        cv2.imshow(self.main_window, vis_show)

        # Mask panel
        mask_r, mask_g, mask_b = masks
        mr = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)
        mg = cv2.cvtColor(mask_g, cv2.COLOR_GRAY2BGR)
        mb = cv2.cvtColor(mask_b, cv2.COLOR_GRAY2BGR)

        cv2.putText(mr, "R Mask", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(mg, "G Mask", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(mb, "B Mask", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        h, w = mr.shape[:2]
        target_h = 260
        scale = target_h / float(h)
        tw = max(1, int(w * scale))
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
        self._create_trackbars()
        self._render()

        print("=" * 72)
        print("LED RGB Tuner")
        print("Keys: n/d=next, p/a=prev, s=save params, q/ESC=quit")
        print("ROI is NOT used in this tool (full-image filter tuning).")
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
            elif key == ord("r"):
                self._render()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    LEDRGBTuner().run()
