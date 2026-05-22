#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Gaussian beam intensity values for solar-cell polygons.

Input:
    - selected CSV file, usually solarcell_points.csv
    - selected image folder containing the files listed in the CSV

Pattern:
    laser ON/OFF row
    laser ON/OFF row
    PV row with 4 points
    PV row with 4 points
    ...
    next laser ON/OFF row
    next laser ON/OFF row
    PV row with 4 points
    ...

PV row:
    point_count == 4 and status == "done"

Laser row:
    otherwise

Output:
    - backup CSV
    - solarcell_intensity_cal.csv
    - laser_debug/
    - integral_debug/
"""

from __future__ import annotations

import math
import shutil
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd


# =========================================================
# User settings
# =========================================================

CSV_NAME = "solarcell_points.csv"
OUTPUT_CSV_NAME = "solarcell_intensity_cal.csv"

SORT_BY_FILENAME = True

# 레이저 중심 검출 파라미터
LASER_DIFF_THRESHOLD = 40
MIN_LASER_AREA = 5
LASER_BLUR_KSIZE = 5

# Debug 저장 옵션
SAVE_LASER_DEBUG = True
SAVE_INTEGRAL_DEBUG = True
SAVE_HEATMAP_DEBUG = True
HEATMAP_ALPHA = 0.35

# Gaussian beam shape model
# I_px(u,v) = I0 * exp(-2 * (u^2 / w_u^2 + v^2 / w_v^2))
DEFAULT_BEAM_PARAM = {
    "w_u": 118.1416,
    "w_v": 124.5135,
    "I0": 1.0,
}

# group_id -> distance mapping
# group_id 1 = 첫 번째 laser ON/OFF pair 이후 PV들
# group_id 2 = 두 번째 laser ON/OFF pair 이후 PV들
# 기본적으로 distance = group_id + 1 로 넣어둠. 필요하면 수정.
BEAM_PARAMS = {
    1: {"distance": 2, "w_u": 118.1416, "w_v": 124.5135, "I0": 1.0},
    2: {"distance": 3, "w_u": 118.1416, "w_v": 124.5135, "I0": 1.0},
    3: {"distance": 4, "w_u": 118.1416, "w_v": 124.5135, "I0": 1.0},
    4: {"distance": 5, "w_u": 118.1416, "w_v": 124.5135, "I0": 1.0},
    5: {"distance": 6, "w_u": 118.1416, "w_v": 124.5135, "I0": 1.0},
    6: {"distance": 7, "w_u": 118.1416, "w_v": 124.5135, "I0": 1.0},
}

# BEAM_PARAMS에 없는 group도 기본 shape로 계산할지 여부
# True: group_id가 7, 8...이어도 같은 w_u/w_v로 계산
# False: missing_beam_param 처리
USE_DEFAULT_BEAM_FOR_MISSING_GROUP = True


OUTPUT_COLUMNS = [
    "group_id",
    "distance",
    "laser_on_file",
    "laser_off_file",
    "laser_cx",
    "laser_cy",
    "w_u",
    "w_v",
    "I0",
    "cell_cx",
    "cell_cy",
    "offset_x_px",
    "offset_y_px",
    "offset_r_px",
    "cell_area_px",
    "beam_integral_sum",
    "beam_integral_mean",
    "beam_integral_norm_total",
    "beam_integral_norm_peak",
    "integral_status",
]


# =========================================================
# Basic IO
# =========================================================

def select_csv_file() -> Path | None:
    """Ask the user to select the solar-cell point CSV file."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            title="Select solarcell_points.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()

    return Path(path) if path else None


def select_image_folder() -> Path | None:
    """Ask the user to select the image folder."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        folder = filedialog.askdirectory(title="Select image folder containing original images")
    finally:
        root.destroy()

    return Path(folder) if folder else None


def load_image_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to decode image: {path}")
    return img


def save_image_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        ext = ".jpg"

    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"Failed to encode image: {path}")

    encoded.tofile(str(path))


# =========================================================
# Row classification
# =========================================================

def safe_float(value, default=np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        s = str(value).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def safe_int(value, default=0) -> int:
    try:
        if pd.isna(value):
            return default
        s = str(value).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def is_pv_row(row: pd.Series) -> bool:
    status = str(row.get("status", "")).strip().lower()
    point_count = safe_int(row.get("point_count", 0), default=0)

    if status != "done":
        return False
    if point_count != 4:
        return False

    coord_cols = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    for col in coord_cols:
        val = safe_float(row.get(col, np.nan))
        if not np.isfinite(val):
            return False

    return True


def is_laser_row(row: pd.Series) -> bool:
    return not is_pv_row(row)


def get_polygon_points(row: pd.Series) -> np.ndarray | None:
    pts = []
    for i in range(1, 5):
        x = safe_float(row.get(f"x{i}", np.nan))
        y = safe_float(row.get(f"y{i}", np.nan))
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        pts.append([x, y])

    pts = np.array(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        return None

    area = abs(cv2.contourArea(pts))
    if area < 1.0:
        return None

    return pts


# =========================================================
# Grouping
# =========================================================

def looks_like_off(filename: str) -> bool:
    name = filename.lower()
    return (
        "off" in name
        or "laseroff" in name
        or "laser_off" in name
        or "led_off" in name
    )


def looks_like_on(filename: str) -> bool:
    name = filename.lower()
    return (
        "on" in name
        or "laseron" in name
        or "laser_on" in name
        or "led_on" in name
    )


def order_laser_pair(file_a: str, file_b: str) -> tuple[str, str]:
    """
    Return laser_on_file, laser_off_file.
    absdiff 기반이라 순서가 바뀌어도 중심 검출은 가능하지만,
    출력 CSV 가독성을 위해 이름에 on/off가 있으면 정렬한다.
    """
    a_on = looks_like_on(file_a)
    a_off = looks_like_off(file_a)
    b_on = looks_like_on(file_b)
    b_off = looks_like_off(file_b)

    if a_on and b_off:
        return file_a, file_b
    if b_on and a_off:
        return file_b, file_a

    # 구분 불가하면 시간순 첫 번째를 ON, 두 번째를 OFF로 둔다.
    return file_a, file_b


def build_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    CSV 행을 순서대로 보면서:
    - laser row 2개 연속 -> 새로운 group 시작
    - 이후 PV row -> 현재 group에 배정
    """
    groups = {}
    pending_laser_indices = []
    current_group_id = None
    group_id = 0

    for idx, row in df.iterrows():
        filename = str(row.get("filename", "")).strip()

        if is_pv_row(row):
            if current_group_id is None:
                df.at[idx, "integral_status"] = "no_laser_pair"
                continue

            df.at[idx, "group_id"] = current_group_id
            df.at[idx, "laser_on_file"] = groups[current_group_id]["laser_on_file"]
            df.at[idx, "laser_off_file"] = groups[current_group_id]["laser_off_file"]
            groups[current_group_id]["pv_indices"].append(idx)
            continue

        # laser row
        pending_laser_indices.append(idx)
        df.at[idx, "integral_status"] = "laser_row"

        if len(pending_laser_indices) == 2:
            group_id += 1
            a_idx, b_idx = pending_laser_indices
            pending_laser_indices = []

            file_a = str(df.at[a_idx, "filename"]).strip()
            file_b = str(df.at[b_idx, "filename"]).strip()
            laser_on_file, laser_off_file = order_laser_pair(file_a, file_b)

            groups[group_id] = {
                "group_id": group_id,
                "laser_on_file": laser_on_file,
                "laser_off_file": laser_off_file,
                "laser_row_indices": [a_idx, b_idx],
                "pv_indices": [],
                "laser_cx": np.nan,
                "laser_cy": np.nan,
                "laser_status": "not_computed",
            }

            current_group_id = group_id

            for laser_idx in [a_idx, b_idx]:
                df.at[laser_idx, "group_id"] = group_id
                df.at[laser_idx, "laser_on_file"] = laser_on_file
                df.at[laser_idx, "laser_off_file"] = laser_off_file

    if len(pending_laser_indices) == 1:
        idx = pending_laser_indices[0]
        print(f"[WARN] Last single laser row without pair: {df.at[idx, 'filename']}")

    return df, groups


# =========================================================
# Beam model
# =========================================================

def get_beam_param(group_id: int) -> dict | None:
    if group_id in BEAM_PARAMS:
        return dict(BEAM_PARAMS[group_id])

    if USE_DEFAULT_BEAM_FOR_MISSING_GROUP:
        p = dict(DEFAULT_BEAM_PARAM)
        p["distance"] = group_id + 1
        return p

    return None


def gaussian_beam_intensity(
    W: int,
    H: int,
    laser_cx: float,
    laser_cy: float,
    w_u: float,
    w_v: float,
    I0: float,
) -> np.ndarray:
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    u = X - np.float32(laser_cx)
    v = Y - np.float32(laser_cy)

    I = np.float32(I0) * np.exp(
        -2.0 * ((u ** 2) / np.float32(w_u ** 2) + (v ** 2) / np.float32(w_v ** 2))
    )

    return I.astype(np.float32)


def polygon_mask(W: int, H: int, points: np.ndarray) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    pts_i = np.round(points).astype(np.int32)
    pts_i[:, 0] = np.clip(pts_i[:, 0], 0, W - 1)
    pts_i[:, 1] = np.clip(pts_i[:, 1], 0, H - 1)
    cv2.fillPoly(mask, [pts_i], 255)
    return mask > 0


# =========================================================
# Laser center detection
# =========================================================

def detect_laser_center(img_a: np.ndarray, img_b: np.ndarray):
    """
    Return:
        laser_cx, laser_cy, diff_bgr, mask
    """
    if img_a is None or img_b is None:
        return None, None, None, None

    if img_a.shape[:2] != img_b.shape[:2]:
        print(f"[WARN] Laser pair image size mismatch: {img_a.shape[:2]} vs {img_b.shape[:2]}")
        return None, None, None, None

    diff = cv2.absdiff(img_a, img_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    if LASER_BLUR_KSIZE and LASER_BLUR_KSIZE >= 3:
        k = LASER_BLUR_KSIZE
        if k % 2 == 0:
            k += 1
        gray_blur = cv2.GaussianBlur(gray, (k, k), 0)
    else:
        gray_blur = gray

    _, mask = cv2.threshold(gray_blur, LASER_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # morphology로 작은 노이즈 제거
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= MIN_LASER_AREA:
            valid_contours.append((area, c))

    if valid_contours:
        valid_contours.sort(key=lambda t: t[0], reverse=True)
        _, largest = valid_contours[0]
        M = cv2.moments(largest)
        if abs(M["m00"]) > 1e-9:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return float(cx), float(cy), diff, mask

    # fallback: threshold 영역 밝기 가중 무게중심
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        weights = gray_blur[ys, xs].astype(np.float64)
        if np.sum(weights) > 0:
            cx = np.sum(xs * weights) / np.sum(weights)
            cy = np.sum(ys * weights) / np.sum(weights)
            return float(cx), float(cy), diff, mask

    # fallback 2: 전체 gray에서 가장 밝은 픽셀 근방
    max_val = gray_blur.max()
    if max_val > LASER_DIFF_THRESHOLD:
        ys, xs = np.where(gray_blur == max_val)
        if len(xs) > 0:
            return float(np.mean(xs)), float(np.mean(ys)), diff, mask

    return None, None, diff, mask


# =========================================================
# Integral computation
# =========================================================

def contour_centroid(points: np.ndarray) -> tuple[float, float] | None:
    M = cv2.moments(points.astype(np.float32))
    if abs(M["m00"]) < 1e-9:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return float(cx), float(cy)


def compute_integral_for_row(
    row: pd.Series,
    group_info: dict,
    beam_param: dict,
    folder: Path,
) -> tuple[dict, dict | None]:
    """
    Returns:
        result_dict, debug_info
    """
    filename = str(row.get("filename", "")).strip()
    image_path = folder / filename

    try:
        img = load_image_unicode(image_path)
    except Exception as exc:
        print(f"[ERROR] PV image load error: {filename} | {exc}")
        return {"integral_status": "image_load_error"}, None

    H, W = img.shape[:2]

    points = get_polygon_points(row)
    if points is None:
        return {"integral_status": "bad_polygon"}, None

    cell_center = contour_centroid(points)
    if cell_center is None:
        return {"integral_status": "bad_polygon"}, None

    cell_cx, cell_cy = cell_center

    laser_cx = group_info.get("laser_cx", np.nan)
    laser_cy = group_info.get("laser_cy", np.nan)

    if not np.isfinite(laser_cx) or not np.isfinite(laser_cy):
        return {"integral_status": "bad_laser_center"}, None

    w_u = float(beam_param["w_u"])
    w_v = float(beam_param["w_v"])
    I0 = float(beam_param["I0"])

    if w_u <= 0 or w_v <= 0 or I0 <= 0:
        return {"integral_status": "missing_beam_param"}, None

    mask = polygon_mask(W, H, points)
    cell_area_px = int(mask.sum())
    if cell_area_px <= 0:
        return {"integral_status": "bad_polygon"}, None

    intensity = gaussian_beam_intensity(W, H, laser_cx, laser_cy, w_u, w_v, I0)

    beam_integral_sum = float(np.sum(intensity[mask], dtype=np.float64))
    beam_integral_mean = float(beam_integral_sum / max(cell_area_px, 1))

    total_sum = float(np.sum(intensity, dtype=np.float64))
    beam_integral_norm_total = float(beam_integral_sum / total_sum) if total_sum > 0 else np.nan
    beam_integral_norm_peak = float(beam_integral_mean / I0) if I0 > 0 else np.nan

    offset_x_px = float(cell_cx - laser_cx)
    offset_y_px = float(cell_cy - laser_cy)
    offset_r_px = float(math.sqrt(offset_x_px ** 2 + offset_y_px ** 2))

    result = {
        "group_id": group_info["group_id"],
        "distance": beam_param.get("distance", np.nan),
        "laser_on_file": group_info["laser_on_file"],
        "laser_off_file": group_info["laser_off_file"],
        "laser_cx": laser_cx,
        "laser_cy": laser_cy,
        "w_u": w_u,
        "w_v": w_v,
        "I0": I0,
        "cell_cx": cell_cx,
        "cell_cy": cell_cy,
        "offset_x_px": offset_x_px,
        "offset_y_px": offset_y_px,
        "offset_r_px": offset_r_px,
        "cell_area_px": cell_area_px,
        "beam_integral_sum": beam_integral_sum,
        "beam_integral_mean": beam_integral_mean,
        "beam_integral_norm_total": beam_integral_norm_total,
        "beam_integral_norm_peak": beam_integral_norm_peak,
        "integral_status": "done",
    }

    debug_info = {
        "image": img,
        "points": points,
        "mask": mask,
        "intensity": intensity,
        "laser_cx": laser_cx,
        "laser_cy": laser_cy,
        "cell_cx": cell_cx,
        "cell_cy": cell_cy,
        "result": result,
    }

    return result, debug_info


# =========================================================
# Debug drawing
# =========================================================

def draw_text_with_outline(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_cross(
    img: np.ndarray,
    x: float,
    y: float,
    color: tuple[int, int, int],
    size: int = 35,
    thickness: int = 2,
) -> None:
    xi, yi = int(round(x)), int(round(y))
    cv2.line(img, (xi - size, yi), (xi + size, yi), color, thickness, cv2.LINE_AA)
    cv2.line(img, (xi, yi - size), (xi, yi + size), color, thickness, cv2.LINE_AA)
    cv2.circle(img, (xi, yi), 6, color, -1, cv2.LINE_AA)


def draw_laser_debug(
    out_path: Path,
    diff: np.ndarray | None,
    mask: np.ndarray | None,
    laser_cx: float | None,
    laser_cy: float | None,
    group_info: dict,
    beam_param: dict | None,
) -> None:
    if diff is None:
        return

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    vis_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    vis = cv2.cvtColor(vis_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if mask is not None:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr[:, :, 1] = np.maximum(mask_bgr[:, :, 1], mask)
        vis = cv2.addWeighted(vis, 0.75, mask_bgr, 0.25, 0)

    if laser_cx is not None and laser_cy is not None:
        draw_cross(vis, laser_cx, laser_cy, (0, 255, 0), size=45, thickness=3)
        draw_text_with_outline(
            vis,
            f"laser=({laser_cx:.1f},{laser_cy:.1f})",
            (30, 80),
            scale=0.8,
            color=(0, 255, 0),
        )

    gid = group_info["group_id"]
    dist = beam_param.get("distance", np.nan) if beam_param else np.nan

    draw_text_with_outline(vis, f"Group {gid} | distance={dist}", (30, 35), scale=0.8)
    draw_text_with_outline(vis, f"ON : {group_info['laser_on_file']}", (30, 125), scale=0.6)
    draw_text_with_outline(vis, f"OFF: {group_info['laser_off_file']}", (30, 155), scale=0.6)

    save_image_unicode(out_path, vis)


def make_heatmap_overlay(image: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    norm = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX)
    norm_u8 = norm.astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1.0 - HEATMAP_ALPHA, heatmap, HEATMAP_ALPHA, 0)
    return overlay


def draw_integral_debug(out_path: Path, debug_info: dict) -> None:
    img = debug_info["image"].copy()
    points = debug_info["points"]
    intensity = debug_info["intensity"]
    laser_cx = debug_info["laser_cx"]
    laser_cy = debug_info["laser_cy"]
    cell_cx = debug_info["cell_cx"]
    cell_cy = debug_info["cell_cy"]
    result = debug_info["result"]

    if SAVE_HEATMAP_DEBUG:
        vis = make_heatmap_overlay(img, intensity)
    else:
        vis = img

    pts_i = np.round(points).astype(np.int32)
    cv2.polylines(vis, [pts_i], isClosed=True, color=(0, 255, 255), thickness=3)

    # polygon points
    for i, (x, y) in enumerate(pts_i, start=1):
        cv2.circle(vis, (int(x), int(y)), 7, (0, 255, 255), -1)
        draw_text_with_outline(vis, str(i), (int(x) + 8, int(y) - 8), scale=0.7, color=(0, 255, 255))

    # laser center
    draw_cross(vis, laser_cx, laser_cy, (0, 255, 0), size=45, thickness=3)
    draw_text_with_outline(vis, "LASER", (int(laser_cx) + 15, int(laser_cy) - 15), scale=0.7, color=(0, 255, 0))

    # cell center
    draw_cross(vis, cell_cx, cell_cy, (0, 0, 255), size=35, thickness=3)
    draw_text_with_outline(vis, "CELL", (int(cell_cx) + 15, int(cell_cy) - 15), scale=0.7, color=(0, 0, 255))

    # connection line
    cv2.line(
        vis,
        (int(round(laser_cx)), int(round(laser_cy))),
        (int(round(cell_cx)), int(round(cell_cy))),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    draw_text_with_outline(
        vis,
        f"sum={result['beam_integral_sum']:.3f} | norm_total={result['beam_integral_norm_total']:.6f}",
        (30, 40),
        scale=0.8,
        color=(255, 255, 255),
    )
    draw_text_with_outline(
        vis,
        f"offset_r={result['offset_r_px']:.2f}px | area={result['cell_area_px']}px",
        (30, 80),
        scale=0.8,
        color=(255, 255, 255),
    )
    draw_text_with_outline(
        vis,
        f"group={result['group_id']} | distance={result['distance']} | w=({result['w_u']:.2f},{result['w_v']:.2f})",
        (30, 120),
        scale=0.7,
        color=(255, 255, 255),
    )

    save_image_unicode(out_path, vis)


# =========================================================
# Main
# =========================================================

def init_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # 매번 새로 계산하기 위해 결과 컬럼 초기화
    for col in OUTPUT_COLUMNS:
        df[col] = np.nan

    df["integral_status"] = ""
    return df


def fill_common_group_values(df: pd.DataFrame, idx, group_info: dict, beam_param: dict | None) -> None:
    gid = group_info["group_id"]
    df.at[idx, "group_id"] = gid
    df.at[idx, "laser_on_file"] = group_info["laser_on_file"]
    df.at[idx, "laser_off_file"] = group_info["laser_off_file"]

    if np.isfinite(group_info.get("laser_cx", np.nan)):
        df.at[idx, "laser_cx"] = group_info["laser_cx"]
    if np.isfinite(group_info.get("laser_cy", np.nan)):
        df.at[idx, "laser_cy"] = group_info["laser_cy"]

    if beam_param is not None:
        df.at[idx, "distance"] = beam_param.get("distance", np.nan)
        df.at[idx, "w_u"] = beam_param.get("w_u", np.nan)
        df.at[idx, "w_v"] = beam_param.get("w_v", np.nan)
        df.at[idx, "I0"] = beam_param.get("I0", np.nan)


def main() -> None:
    csv_path = select_csv_file()
    if csv_path is None:
        print("[INFO] No CSV selected.")
        return

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    image_folder = select_image_folder()
    if image_folder is None:
        print("[INFO] No image folder selected.")
        return

    if not image_folder.exists():
        print(f"[ERROR] Image folder not found: {image_folder}")
        return

    # Keep the original variable name used by lower-level helper calls.
    folder = image_folder
    output_dir = csv_path.parent

    print("=" * 70)
    print("[INFO] Solar-cell intensity calculation")
    print(f"[INFO] CSV file    : {csv_path}")
    print(f"[INFO] Image folder: {image_folder}")
    print(f"[INFO] Output dir  : {output_dir}")
    print("=" * 70)

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[ERROR] Failed to read CSV: {csv_path} | {exc}")
        return

    required = ["filename", "width", "height", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "point_count", "status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        return

    # 원본 순서 보존용
    df["_original_order"] = np.arange(len(df))

    if SORT_BY_FILENAME:
        df["filename"] = df["filename"].astype(str)
        df = df.sort_values("filename", kind="stable").reset_index(drop=True)
        print("[INFO] Rows sorted by filename.")
    else:
        df = df.reset_index(drop=True)
        print("[INFO] Rows kept in CSV order.")

    # backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = output_dir / f"solarcell_points_backup_{ts}.csv"
    try:
        shutil.copy2(csv_path, backup_path)
        print(f"[BACKUP] {backup_path}")
    except Exception as exc:
        print(f"[WARN] Failed to create backup: {exc}")

    df = init_output_columns(df)

    laser_debug_dir = output_dir / "laser_debug"
    integral_debug_dir = output_dir / "integral_debug"
    laser_debug_dir.mkdir(exist_ok=True)
    integral_debug_dir.mkdir(exist_ok=True)

    # grouping
    df, groups = build_groups(df)
    print(f"[INFO] Detected groups: {len(groups)}")

    total_laser_rows = 0
    total_pv_rows = 0
    done_count = 0
    fail_count = 0

    # group별 레이저 중심 계산
    for gid, group_info in groups.items():
        beam_param = get_beam_param(gid)

        if beam_param is None:
            print(f"[WARN] Group {gid}: missing beam parameter")
            for idx in group_info["laser_row_indices"] + group_info["pv_indices"]:
                fill_common_group_values(df, idx, group_info, None)
                if idx in group_info["pv_indices"]:
                    df.at[idx, "integral_status"] = "missing_beam_param"
                    fail_count += 1
            continue

        on_file = group_info["laser_on_file"]
        off_file = group_info["laser_off_file"]

        try:
            img_on = load_image_unicode(folder / on_file)
            img_off = load_image_unicode(folder / off_file)
        except Exception as exc:
            print(f"[ERROR] Group {gid}: laser image load error | {exc}")
            for idx in group_info["laser_row_indices"] + group_info["pv_indices"]:
                fill_common_group_values(df, idx, group_info, beam_param)
                if idx in group_info["pv_indices"]:
                    df.at[idx, "integral_status"] = "image_load_error"
                    fail_count += 1
            continue

        laser_cx, laser_cy, diff, mask = detect_laser_center(img_on, img_off)

        if laser_cx is not None and laser_cy is not None:
            group_info["laser_cx"] = laser_cx
            group_info["laser_cy"] = laser_cy
            group_info["laser_status"] = "done"
        else:
            group_info["laser_status"] = "bad_laser_center"

        # laser row에도 공통 정보 채우기
        for idx in group_info["laser_row_indices"]:
            total_laser_rows += 1
            fill_common_group_values(df, idx, group_info, beam_param)
            df.at[idx, "integral_status"] = "laser_row"

        if SAVE_LASER_DEBUG:
            laser_debug_path = laser_debug_dir / f"group_{gid:03d}_laser_center_debug.jpg"
            try:
                draw_laser_debug(laser_debug_path, diff, mask, laser_cx, laser_cy, group_info, beam_param)
            except Exception as exc:
                print(f"[WARN] Group {gid}: failed to save laser debug | {exc}")

        if group_info["laser_status"] != "done":
            print(f"[WARN] Group {gid}: bad laser center")
            for idx in group_info["pv_indices"]:
                total_pv_rows += 1
                fill_common_group_values(df, idx, group_info, beam_param)
                df.at[idx, "integral_status"] = "bad_laser_center"
                fail_count += 1
            continue

        print(
            f"Group {gid} | distance={beam_param.get('distance', np.nan)} "
            f"| w=({beam_param['w_u']:.4f},{beam_param['w_v']:.4f}) px "
            f"| laser=({laser_cx:.1f},{laser_cy:.1f}) "
            f"| PV rows={len(group_info['pv_indices'])}"
        )

        # PV row별 적분 계산
        for idx in group_info["pv_indices"]:
            total_pv_rows += 1
            fill_common_group_values(df, idx, group_info, beam_param)

            row = df.loc[idx]
            result, debug_info = compute_integral_for_row(row, group_info, beam_param, folder)

            for key, value in result.items():
                if key in df.columns:
                    df.at[idx, key] = value

            if result.get("integral_status") == "done":
                done_count += 1

                if SAVE_INTEGRAL_DEBUG and debug_info is not None:
                    filename = str(row.get("filename", "")).strip()
                    out_name = f"{Path(filename).stem}_integral_debug.jpg"
                    out_path = integral_debug_dir / out_name
                    try:
                        draw_integral_debug(out_path, debug_info)
                    except Exception as exc:
                        print(f"[WARN] Failed to save integral debug for {filename} | {exc}")
            else:
                fail_count += 1
                print(f"[WARN] {row.get('filename', '')}: {result.get('integral_status')}")

    # group에 속하지 않은 PV row count/status 정리
    for idx, row in df.iterrows():
        if is_pv_row(row) and str(df.at[idx, "integral_status"]).strip() in ["", "nan"]:
            total_pv_rows += 1
            df.at[idx, "integral_status"] = "no_laser_pair"
            fail_count += 1

    # 원본 순서로 되돌릴지 여부
    # 결과 해석 편하게 filename 시간순 유지하려면 아래 2줄 주석 처리
    # df = df.sort_values("_original_order", kind="stable").reset_index(drop=True)

    if "_original_order" in df.columns:
        df = df.drop(columns=["_original_order"])

    output_path = output_dir / OUTPUT_CSV_NAME

    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[ERROR] Failed to save output CSV: {output_path} | {exc}")
        return

    print("\n" + "=" * 70)
    print("[DONE] Solar-cell intensity calculation finished")
    print(f"Total rows       : {len(df)}")
    print(f"Laser rows       : {total_laser_rows}")
    print(f"PV rows          : {total_pv_rows}")
    print(f"Integral done    : {done_count}")
    print(f"Integral failed  : {fail_count}")
    print(f"Output CSV       : {output_path}")
    print(f"Laser debug dir  : {laser_debug_dir}")
    print(f"Integral debug dir: {integral_debug_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()