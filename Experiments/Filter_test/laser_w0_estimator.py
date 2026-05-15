#!/usr/bin/env python3
"""
Estimate laser beam waist (w0) from LASER ON/OFF images.

Method:
- read LASER ON and LASER OFF images
- compute positive grayscale diff: gray_on - gray_off
- threshold the diff to isolate the beam spot
- compute weighted centroid and weighted covariance
- convert weighted sigma to Gaussian 1/e^2 beam radii by w = 2 * sigma
- report wx, wy and w0 = sqrt(wx * wy)

Notes:
- Without calibration, results are reported in pixels.
- If PIXEL_SIZE_UM is set, the script also reports micrometers / millimeters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np
from tkinter import Tk, filedialog


DIFF_THRESHOLD = 30.0
PIXEL_SIZE_UM = None  # Example: 1.55 for 1.55 um / px. Keep None to report only in px.
DISPLAY_MAX_W = 1800
DISPLAY_MAX_H = 980


@dataclass
class W0Estimate:
    centroid_x: float
    centroid_y: float
    sigma_major: float
    sigma_minor: float
    w_major: float
    w_minor: float
    w0: float
    angle_deg: float
    diff_pixels: int
    diff_sum: float


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, img)
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


def resize_to_fit(img: np.ndarray, max_w: int = DISPLAY_MAX_W, max_h: int = DISPLAY_MAX_H) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return img
    scale = min(float(max_w) / float(w), float(max_h) / float(h), 1.0)
    if scale >= 0.999:
        return img
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def normalize_to_u8(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    if gray.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    max_val = float(np.max(gray))
    if max_val <= 1e-9:
        return np.zeros_like(gray, dtype=np.uint8)
    scaled = np.clip((gray / max_val) * 255.0, 0.0, 255.0)
    return scaled.astype(np.uint8)


def compute_w0_from_on_off(
    img_on: np.ndarray,
    img_off: np.ndarray,
    diff_threshold: float = DIFF_THRESHOLD,
) -> tuple[W0Estimate | None, np.ndarray, np.ndarray]:
    if img_on is None or img_off is None:
        return None, np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.uint8)

    if img_on.shape[:2] != img_off.shape[:2]:
        img_off = cv2.resize(img_off, (img_on.shape[1], img_on.shape[0]), interpolation=cv2.INTER_AREA)

    gray_on = cv2.cvtColor(img_on, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_off = cv2.cvtColor(img_off, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = gray_on - gray_off
    positive_diff = np.maximum(diff, 0.0)
    mask = positive_diff >= float(diff_threshold)

    if not np.any(mask):
        return None, positive_diff, mask.astype(np.uint8) * 255

    ys, xs = np.nonzero(mask)
    weights = positive_diff[ys, xs].astype(np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return None, positive_diff, mask.astype(np.uint8) * 255

    x_mean = float(np.sum(xs * weights) / weight_sum)
    y_mean = float(np.sum(ys * weights) / weight_sum)

    dx = xs.astype(np.float64) - x_mean
    dy = ys.astype(np.float64) - y_mean
    cov_xx = float(np.sum(weights * dx * dx) / weight_sum)
    cov_yy = float(np.sum(weights * dy * dy) / weight_sum)
    cov_xy = float(np.sum(weights * dx * dy) / weight_sum)
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    sigma_major = float(math.sqrt(float(eigvals[0])))
    sigma_minor = float(math.sqrt(float(eigvals[1])))
    w_major = 2.0 * sigma_major
    w_minor = 2.0 * sigma_minor
    w0 = float(math.sqrt(max(w_major, 0.0) * max(w_minor, 0.0)))
    angle_deg = float(math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0])))

    estimate = W0Estimate(
        centroid_x=x_mean,
        centroid_y=y_mean,
        sigma_major=sigma_major,
        sigma_minor=sigma_minor,
        w_major=w_major,
        w_minor=w_minor,
        w0=w0,
        angle_deg=angle_deg,
        diff_pixels=int(mask.sum()),
        diff_sum=weight_sum,
    )
    return estimate, positive_diff, mask.astype(np.uint8) * 255


def draw_cross(img: np.ndarray, center: tuple[int, int], color: tuple[int, int, int], arm: int = 22) -> None:
    cx, cy = [int(v) for v in center]
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), (0, 0, 0), 5)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), (0, 0, 0), 5)
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), (255, 255, 255), 3)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), (255, 255, 255), 3)
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), color, 2)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), color, 2)
    cv2.circle(img, (cx, cy), 4, color, -1)


def render_overlay(
    base_img: np.ndarray,
    positive_diff: np.ndarray,
    mask_u8: np.ndarray,
    estimate: W0Estimate | None,
) -> tuple[np.ndarray, np.ndarray]:
    overlay = base_img.copy()
    heatmap_base = normalize_to_u8(positive_diff)
    heatmap = cv2.applyColorMap(heatmap_base, cv2.COLORMAP_JET)

    if estimate is not None:
        cx = int(round(estimate.centroid_x))
        cy = int(round(estimate.centroid_y))
        draw_cross(overlay, (cx, cy), (0, 255, 255), arm=26)
        draw_cross(heatmap, (cx, cy), (255, 255, 255), arm=26)

        axes = (
            max(1, int(round(estimate.w_major))),
            max(1, int(round(estimate.w_minor))),
        )
        angle = float(estimate.angle_deg)
        cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, (0, 255, 0), 2)
        cv2.ellipse(heatmap, (cx, cy), axes, angle, 0, 360, (255, 255, 255), 2)

        text_lines = [
            f"Diff threshold = {DIFF_THRESHOLD:.1f}",
            f"centroid = ({estimate.centroid_x:.1f}, {estimate.centroid_y:.1f})",
            f"sigma major/minor = {estimate.sigma_major:.2f} / {estimate.sigma_minor:.2f} px",
            f"w major/minor = {estimate.w_major:.2f} / {estimate.w_minor:.2f} px",
            f"w0 = sqrt(wx*wy) = {estimate.w0:.2f} px",
            f"angle = {estimate.angle_deg:.2f} deg",
            f"diff_pixels = {estimate.diff_pixels}, diff_sum = {estimate.diff_sum:.1f}",
        ]
        if PIXEL_SIZE_UM:
            um_per_px = float(PIXEL_SIZE_UM)
            text_lines.extend(
                [
                    f"w major/minor = {estimate.w_major * um_per_px:.2f} / {estimate.w_minor * um_per_px:.2f} um",
                    f"w0 = {estimate.w0 * um_per_px:.2f} um ({(estimate.w0 * um_per_px) / 1000.0:.4f} mm)",
                ]
            )

        for idx, line in enumerate(text_lines):
            y = 36 + (idx * 28)
            cv2.putText(overlay, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4)
            cv2.putText(overlay, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    else:
        text = f"No beam pixels above diff threshold {DIFF_THRESHOLD:.1f}"
        cv2.putText(overlay, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(overlay, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    return overlay, np.hstack([heatmap, mask_bgr])


def main() -> int:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    laser_on_path = choose_file(root, "Select LASER ON image")
    if laser_on_path is None:
        print("Cancelled: LASER ON image not selected.")
        return 1

    laser_off_path = choose_file(root, "Select LASER OFF image")
    if laser_off_path is None:
        print("Cancelled: LASER OFF image not selected.")
        return 1

    img_on = imread_unicode(laser_on_path)
    img_off = imread_unicode(laser_off_path)
    if img_on is None or img_off is None:
        print("Failed to read LASER ON/OFF images.")
        return 1

    estimate, positive_diff, mask_u8 = compute_w0_from_on_off(img_on, img_off, diff_threshold=DIFF_THRESHOLD)
    overlay, heatmap_and_mask = render_overlay(img_on, positive_diff, mask_u8, estimate)

    stem = laser_on_path.stem
    save_dir = laser_on_path.parent
    overlay_path = save_dir / f"{stem}_w0_overlay.png"
    heatmap_path = save_dir / f"{stem}_w0_heatmap_mask.png"
    diff_path = save_dir / f"{stem}_w0_diff_gray.png"

    imwrite_unicode(overlay_path, overlay)
    imwrite_unicode(heatmap_path, heatmap_and_mask)
    imwrite_unicode(diff_path, normalize_to_u8(positive_diff))

    print(f"LASER ON : {laser_on_path}")
    print(f"LASER OFF: {laser_off_path}")
    print(f"Saved overlay : {overlay_path}")
    print(f"Saved heatmap : {heatmap_path}")
    print(f"Saved diff    : {diff_path}")

    if estimate is not None:
        print(f"centroid      : ({estimate.centroid_x:.2f}, {estimate.centroid_y:.2f})")
        print(f"sigma major   : {estimate.sigma_major:.4f} px")
        print(f"sigma minor   : {estimate.sigma_minor:.4f} px")
        print(f"w major       : {estimate.w_major:.4f} px")
        print(f"w minor       : {estimate.w_minor:.4f} px")
        print(f"w0            : {estimate.w0:.4f} px")
        print(f"angle         : {estimate.angle_deg:.2f} deg")
        print(f"diff_pixels   : {estimate.diff_pixels}")
        print(f"diff_sum      : {estimate.diff_sum:.4f}")
        if PIXEL_SIZE_UM:
            w0_um = estimate.w0 * float(PIXEL_SIZE_UM)
            print(f"w0            : {w0_um:.4f} um")
            print(f"w0            : {w0_um / 1000.0:.6f} mm")
    else:
        print(f"No valid beam region found above diff threshold {DIFF_THRESHOLD:.1f}.")

    cv2.imshow("Laser w0 Overlay", resize_to_fit(overlay))
    cv2.imshow("Laser w0 Heatmap + Mask", resize_to_fit(heatmap_and_mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
