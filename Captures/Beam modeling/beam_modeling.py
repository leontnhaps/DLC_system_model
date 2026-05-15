import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# User-editable parameters
# =========================
roi_size = 400
corner_size = 30
overwrite_existing = False

SHUTTER_LIST = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "Beam modeling")
if not os.path.isdir(BASE_DIR):
    BASE_DIR = SCRIPT_DIR

CSV_PATH = os.path.join(SCRIPT_DIR, "beam_parameters_all.csv")
DEBUG_ROOT = SCRIPT_DIR

RESULT_COLUMNS = [
    "folder",
    "x_index",
    "distance_m",
    "pair_index",
    "shutter_us",
    "on_path",
    "off_path",
    "click_x",
    "click_y",
    "roi_center_x_ref",
    "roi_center_y_ref",
    "roi_x1",
    "roi_y1",
    "total_intensity",
    "centroid_x",
    "centroid_y",
    "centroid_x_full",
    "centroid_y_full",
    "sigma_x",
    "sigma_y",
    "D_x",
    "D_y",
    "w_x",
    "w_y",
    "aspect_ratio",
    "FWHM_x",
    "FWHM_y",
    "I_upper",
    "I_lower",
    "R_asym",
    "I_peak",
    "saturation_ratio",
    "valid",
    "fail_reason",
]


def select_image_file(title):
    """Compatibility placeholder; batch mode does not use manual file selection."""
    raise RuntimeError("Batch mode does not use select_image_file().")


def load_image_gray(path):
    """Load an image as grayscale float64.

    cv2.imread can fail on Windows paths containing Korean/non-ASCII text.
    np.fromfile + cv2.imdecode reads the bytes first, then lets OpenCV decode.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception as exc:
        raise FileNotFoundError(f"Failed to read image bytes: {path}") from exc
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img.astype(np.float64)


def get_user_click(image, title):
    """Compatibility placeholder; batch mode detects ROI centers automatically."""
    raise RuntimeError("Batch mode does not use get_user_click().")


def crop_roi(image, center_x, center_y, roi_size):
    """Crop a ROI centered at the detected point, clipped to image bounds."""
    h, w = image.shape[:2]
    half = int(round(float(roi_size) / 2.0))
    cx = int(round(float(center_x)))
    cy = int(round(float(center_y)))

    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ROI. Detected center may be outside the image.")
    return image[y1:y2, x1:x2].copy(), int(x1), int(y1)


def estimate_background_from_corners(roi, corner_size):
    """Estimate background mean/std using four corner patches."""
    h, w = roi.shape[:2]
    cs = int(max(1, min(corner_size, h, w)))
    corners = [
        roi[:cs, :cs],
        roi[:cs, max(0, w - cs):w],
        roi[max(0, h - cs):h, :cs],
        roi[max(0, h - cs):h, max(0, w - cs):w],
    ]
    samples = np.concatenate([c.reshape(-1) for c in corners if c.size > 0])
    if samples.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(samples)), float(np.std(samples))


def _natural_numeric_key(path):
    name = os.path.basename(path)
    stem, _ext = os.path.splitext(name)
    chunks = []
    current = ""
    is_digit = None
    for ch in stem:
        ch_is_digit = ch.isdigit()
        if is_digit is None or ch_is_digit == is_digit:
            current += ch
        else:
            chunks.append(int(current) if is_digit else current.lower())
            current = ch
        is_digit = ch_is_digit
    if current:
        chunks.append(int(current) if is_digit else current.lower())
    return chunks + [name.lower()]


def list_numeric_folders(base_dir):
    folders = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.isdigit():
            folders.append((int(name), path))
    return sorted(folders, key=lambda item: item[0])


def list_image_files(folder_path):
    files = []
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            files.append(path)
    return sorted(files, key=_natural_numeric_key)


def detect_reference_roi_center(reference_image):
    """Detect the brightest laser spot using a component + weighted centroid method."""
    image = np.asarray(reference_image, dtype=np.float64)
    if image.size == 0:
        raise ValueError("empty_reference_image")

    blur = cv2.GaussianBlur(image.astype(np.float32), (5, 5), 0)
    finite = blur[np.isfinite(blur)]
    if finite.size == 0:
        raise ValueError("invalid_reference_image")

    max_val = float(np.max(finite))
    if max_val <= 0:
        raise ValueError("reference_image_has_no_bright_signal")

    percentile_t = float(np.percentile(finite, 99.5))
    ratio_t = max_val * 0.55
    threshold = max(percentile_t, ratio_t)
    mask = (blur >= threshold).astype(np.uint8)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_label = None
    best_score = -np.inf
    min_area = 5

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component_mask = labels == label
        component_values = image[component_mask]
        total_intensity = float(np.sum(component_values))
        peak = float(np.max(component_values)) if component_values.size else 0.0
        score = total_intensity + peak * area
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        y, x = np.unravel_index(int(np.argmax(blur)), blur.shape)
        return float(x), float(y), "fallback_argmax"

    component_mask = labels == best_label
    yy, xx = np.indices(image.shape, dtype=np.float64)
    weights = np.where(component_mask, np.clip(image, 0.0, None), 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        y, x = np.unravel_index(int(np.argmax(blur)), blur.shape)
        return float(x), float(y), "fallback_argmax_zero_component_weight"

    cx = float(np.sum(xx * weights) / weight_sum)
    cy = float(np.sum(yy * weights) / weight_sum)
    return cx, cy, "component_weighted_centroid"


def _base_result(
    folder,
    on_path,
    off_path,
    x_index,
    pair_index,
    shutter_us,
    roi_center_x_ref,
    roi_center_y_ref,
    roi_x1=np.nan,
    roi_y1=np.nan,
    valid=False,
    fail_reason="",
):
    distance_m = 0.195 + 0.45 * int(x_index)
    result = {col: np.nan for col in RESULT_COLUMNS}
    result.update(
        {
            "folder": str(folder),
            "x_index": int(x_index),
            "distance_m": float(distance_m),
            "pair_index": int(pair_index),
            "shutter_us": int(shutter_us),
            "on_path": str(on_path),
            "off_path": str(off_path),
            "click_x": np.nan,
            "click_y": np.nan,
            "roi_center_x_ref": float(roi_center_x_ref),
            "roi_center_y_ref": float(roi_center_y_ref),
            "roi_x1": roi_x1,
            "roi_y1": roi_y1,
            "valid": bool(valid),
            "fail_reason": str(fail_reason),
        }
    )
    return result


def process_pair(
    on_path,
    off_path,
    x_index,
    roi_size,
    corner_size,
    pair_index=0,
    shutter_us=np.nan,
    roi_center=None,
    folder=None,
):
    """Process one ON/OFF image pair using a fixed reference ROI center."""
    folder = str(folder if folder is not None else x_index)
    if roi_center is None:
        raise ValueError("roi_center_is_required_in_batch_mode")

    roi_center_x_ref, roi_center_y_ref = float(roi_center[0]), float(roi_center[1])
    base = _base_result(
        folder=folder,
        on_path=on_path,
        off_path=off_path,
        x_index=x_index,
        pair_index=pair_index,
        shutter_us=shutter_us,
        roi_center_x_ref=roi_center_x_ref,
        roi_center_y_ref=roi_center_y_ref,
    )

    try:
        on_gray = load_image_gray(on_path)
        off_gray = load_image_gray(off_path)
        on_roi, roi_x1, roi_y1 = crop_roi(on_gray, roi_center_x_ref, roi_center_y_ref, roi_size)
        off_roi, off_x1, off_y1 = crop_roi(off_gray, roi_center_x_ref, roi_center_y_ref, roi_size)
    except Exception as exc:
        base["fail_reason"] = f"load_or_crop_failed:{exc}"
        return base, _empty_debug()

    common_h = min(on_roi.shape[0], off_roi.shape[0])
    common_w = min(on_roi.shape[1], off_roi.shape[1])
    on_roi = on_roi[:common_h, :common_w]
    off_roi = off_roi[:common_h, :common_w]
    roi_x1 = min(int(roi_x1), int(off_x1))
    roi_y1 = min(int(roi_y1), int(off_y1))

    i_diff = np.clip(on_roi - off_roi, 0.0, None)
    bg_mean, bg_std = estimate_background_from_corners(i_diff, corner_size)
    if not np.isfinite(bg_mean) or not np.isfinite(bg_std):
        threshold = np.nan
        i_clean = np.zeros_like(i_diff)
        fail_reason = "invalid_background_estimate"
    else:
        threshold = bg_mean + 3.0 * bg_std
        i_clean = np.where(i_diff > threshold, i_diff, 0.0)
        fail_reason = ""

    total_intensity = float(np.sum(i_clean))
    roi_area = float(max(1, on_roi.size))
    saturation_ratio = float(np.count_nonzero(on_roi >= 250.0) / roi_area)
    i_peak = float(np.max(i_clean)) if i_clean.size else 0.0

    debug = {
        "on_roi": on_roi,
        "off_roi": off_roi,
        "diff": i_diff,
        "clean": i_clean,
        "norm": np.zeros_like(i_clean),
        "threshold": threshold,
        "bg_mean": bg_mean,
        "bg_std": bg_std,
        "centroid_x": np.nan,
        "centroid_y": np.nan,
    }

    base.update(
        {
            "roi_x1": int(roi_x1),
            "roi_y1": int(roi_y1),
            "total_intensity": total_intensity,
            "I_peak": i_peak,
            "saturation_ratio": saturation_ratio,
        }
    )

    if fail_reason:
        base["fail_reason"] = fail_reason
        return base, debug
    if total_intensity <= 0.0 or not np.isfinite(total_intensity):
        base["fail_reason"] = "zero_total_intensity_after_threshold"
        return base, debug

    i_norm = i_clean / total_intensity
    debug["norm"] = i_norm

    yy, xx = np.indices(i_norm.shape, dtype=np.float64)
    centroid_x = float(np.sum(xx * i_norm))
    centroid_y = float(np.sum(yy * i_norm))
    sigma_x = float(np.sqrt(np.sum(((xx - centroid_x) ** 2) * i_norm)))
    sigma_y = float(np.sqrt(np.sum(((yy - centroid_y) ** 2) * i_norm)))
    d_x = 4.0 * sigma_x
    d_y = 4.0 * sigma_y
    w_x = 2.0 * sigma_x
    w_y = 2.0 * sigma_y
    aspect_ratio = float(d_y / d_x) if d_x > 0 else np.nan

    upper_mask = yy < centroid_y
    lower_mask = yy >= centroid_y
    i_upper = float(np.sum(i_norm[upper_mask]))
    i_lower = float(np.sum(i_norm[lower_mask]))
    r_asym = float(i_lower / i_upper) if i_upper > 0 else np.nan

    if i_peak > 0:
        half_mask = i_clean >= (0.5 * i_peak)
        if np.any(half_mask):
            ys, xs = np.where(half_mask)
            fwhm_x = float(xs.max() - xs.min() + 1)
            fwhm_y = float(ys.max() - ys.min() + 1)
        else:
            fwhm_x = np.nan
            fwhm_y = np.nan
    else:
        fwhm_x = np.nan
        fwhm_y = np.nan

    debug["centroid_x"] = centroid_x
    debug["centroid_y"] = centroid_y

    base.update(
        {
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "centroid_x_full": float(roi_x1 + centroid_x),
            "centroid_y_full": float(roi_y1 + centroid_y),
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "D_x": d_x,
            "D_y": d_y,
            "w_x": w_x,
            "w_y": w_y,
            "aspect_ratio": aspect_ratio,
            "FWHM_x": fwhm_x,
            "FWHM_y": fwhm_y,
            "I_upper": i_upper,
            "I_lower": i_lower,
            "R_asym": r_asym,
            "valid": True,
            "fail_reason": "",
        }
    )
    return base, debug


def _empty_debug():
    z = np.zeros((1, 1), dtype=np.float64)
    return {
        "on_roi": z,
        "off_roi": z,
        "diff": z,
        "clean": z,
        "norm": z,
        "threshold": np.nan,
        "bg_mean": np.nan,
        "bg_std": np.nan,
        "centroid_x": np.nan,
        "centroid_y": np.nan,
    }


def save_debug_figure(result, debug, output_dir=DEBUG_ROOT):
    """Save a 4-panel debug figure for one processed pair."""
    x_index = int(result["x_index"])
    pair_index = int(result["pair_index"])
    shutter_us = int(result["shutter_us"])
    out_dir = os.path.join(output_dir, f"x_{x_index}", f"pair_{pair_index}_sh{shutter_us}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "beam_debug.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()

    im0 = axes[0].imshow(debug["on_roi"], cmap="gray")
    axes[0].set_title("ON ROI grayscale")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(debug["off_roi"], cmap="gray")
    axes[1].set_title("OFF ROI grayscale")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(debug["clean"], cmap="inferno")
    axes[2].set_title(f"Clean diff (T={debug['threshold']:.3f})")
    _draw_centroid_marker(axes[2], debug)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(debug["norm"], cmap="viridis")
    axes[3].set_title("Normalized beam shape")
    _draw_centroid_marker(axes[3], debug)
    fig.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"x_index={x_index}, distance={float(result['distance_m']):.3f} m, "
        f"pair={pair_index}, shutter={shutter_us} us, valid={bool(result['valid'])}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _draw_centroid_marker(ax, debug):
    cx = debug.get("centroid_x", np.nan)
    cy = debug.get("centroid_y", np.nan)
    if np.isfinite(cx) and np.isfinite(cy):
        ax.plot([cx], [cy], marker="+", markersize=16, markeredgewidth=2.5, color="cyan")


def save_summary_plots(df_all):
    """Save distance summary plots.

    total_intensity is camera pixel intensity after subtraction, not absolute
    optical power. It should only be compared across images with the same
    shutter/gain.
    """
    if df_all is None or df_all.empty:
        return

    df = df_all.copy()
    for col in ("distance_m", "D_x", "D_y", "w_x", "w_y", "R_asym", "total_intensity", "shutter_us"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_df = df[df.get("valid", False).astype(bool)].copy() if "valid" in df.columns else df
    valid_df = valid_df.sort_values(["distance_m", "shutter_us"])

    width_df = valid_df.dropna(subset=["distance_m", "D_x", "D_y"])
    if not width_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(width_df["distance_m"], width_df["D_x"], label="D_x", alpha=0.8)
        ax.scatter(width_df["distance_m"], width_df["D_y"], label="D_y", alpha=0.8)
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Second-moment diameter [px]")
        ax.set_title("Beam width vs distance")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(SCRIPT_DIR, "beam_width_vs_distance.png"), dpi=150)
        plt.close(fig)

    waist_df = valid_df.dropna(subset=["distance_m", "w_x", "w_y"])
    if not waist_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(waist_df["distance_m"], waist_df["w_x"], label="w_x", alpha=0.8)
        ax.scatter(waist_df["distance_m"], waist_df["w_y"], label="w_y", alpha=0.8)
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("2*sigma beam width [px]")
        ax.set_title("w_x/w_y vs distance")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(SCRIPT_DIR, "beam_w_vs_distance.png"), dpi=150)
        plt.close(fig)

    asym_df = valid_df.dropna(subset=["distance_m", "R_asym"])
    if not asym_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(asym_df["distance_m"], asym_df["R_asym"], alpha=0.8)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("R_asym = I_lower / I_upper")
        ax.set_title("Asymmetry vs distance")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(SCRIPT_DIR, "asymmetry_vs_distance.png"), dpi=150)
        plt.close(fig)

    intensity_df = valid_df.dropna(subset=["distance_m", "total_intensity"])
    if not intensity_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(intensity_df["distance_m"], intensity_df["total_intensity"], alpha=0.8)
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Total camera intensity [a.u.]")
        ax.set_title("Total intensity vs distance")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(SCRIPT_DIR, "total_intensity_vs_distance.png"), dpi=150)
        plt.close(fig)


def append_or_save_csv(df_result, csv_path, overwrite_existing):
    """Save all batch results to CSV, with optional x_index overwrite behavior."""
    df_result = df_result.reindex(columns=RESULT_COLUMNS)

    if overwrite_existing and os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        if "x_index" in df_old.columns:
            new_x = set(pd.to_numeric(df_result["x_index"], errors="coerce").dropna().astype(int).tolist())
            old_x = pd.to_numeric(df_old["x_index"], errors="coerce")
            df_old = df_old.loc[~old_x.isin(new_x)].copy()
        df_all = pd.concat([df_old, df_result], ignore_index=True).reindex(columns=RESULT_COLUMNS)
        df_all.to_csv(csv_path, mode="w", header=True, index=False)
        return df_all

    if os.path.exists(csv_path):
        df_result.to_csv(csv_path, mode="a", header=False, index=False)
        return pd.read_csv(csv_path)

    df_result.to_csv(csv_path, mode="w", header=True, index=False)
    return df_result.copy()


def process_folder(x_index, folder_path):
    files = list_image_files(folder_path)
    folder_name = os.path.basename(folder_path)
    results = []
    debug_paths = []

    if not files:
        print(f"[WARN] Folder {folder_name}: no image files, skipped.")
        return results, debug_paths
    if len(files) % 2 != 0:
        print(f"[WARN] Folder {folder_name}: odd image count ({len(files)}), last file ignored.")
        files = files[:-1]
    if not files:
        return results, debug_paths

    half = len(files) // 2
    on_files = files[:half]
    off_files = files[half:half * 2]
    pair_count = min(len(on_files), len(off_files), len(SHUTTER_LIST))

    if half > len(SHUTTER_LIST):
        print(
            f"[WARN] Folder {folder_name}: {half} pairs exceed SHUTTER_LIST length "
            f"({len(SHUTTER_LIST)}). Processing first {pair_count} pairs."
        )

    try:
        reference = load_image_gray(on_files[0])
        roi_cx, roi_cy, method = detect_reference_roi_center(reference)
        print(f"[INFO] Folder {folder_name}: ROI center=({roi_cx:.1f}, {roi_cy:.1f}) via {method}")
    except Exception as exc:
        print(f"[ERROR] Folder {folder_name}: failed to detect reference ROI center: {exc}")
        for pair_index in range(pair_count):
            result = _base_result(
                folder=folder_name,
                on_path=on_files[pair_index],
                off_path=off_files[pair_index],
                x_index=x_index,
                pair_index=pair_index,
                shutter_us=SHUTTER_LIST[pair_index],
                roi_center_x_ref=np.nan,
                roi_center_y_ref=np.nan,
                valid=False,
                fail_reason=f"reference_roi_detection_failed:{exc}",
            )
            results.append(result)
        return results, debug_paths

    for pair_index in range(pair_count):
        shutter_us = SHUTTER_LIST[pair_index]
        result, debug = process_pair(
            on_path=on_files[pair_index],
            off_path=off_files[pair_index],
            x_index=x_index,
            roi_size=roi_size,
            corner_size=corner_size,
            pair_index=pair_index,
            shutter_us=shutter_us,
            roi_center=(roi_cx, roi_cy),
            folder=folder_name,
        )
        results.append(result)
        debug_paths.append(save_debug_figure(result, debug, output_dir=DEBUG_ROOT))
        status = "OK" if bool(result.get("valid")) else f"FAIL:{result.get('fail_reason')}"
        print(f"[PAIR] x={x_index} pair={pair_index} shutter={shutter_us}us {status}")

    return results, debug_paths


def main():
    """Batch process all numeric folders."""
    numeric_folders = list_numeric_folders(BASE_DIR)
    all_results = []
    all_debug_paths = []

    print("Batch beam shape extraction from numeric folders")
    print(f"Base dir: {BASE_DIR}")
    print("ON/OFF rule: sorted files first half = ON, second half = OFF")
    print("distance_m = 0.195 + 0.45 * x_index")

    for x_index, folder_path in numeric_folders:
        results, debug_paths = process_folder(x_index, folder_path)
        all_results.extend(results)
        all_debug_paths.extend(debug_paths)

    if not all_results:
        print("No results were processed.")
        return

    df_result = pd.DataFrame(all_results).reindex(columns=RESULT_COLUMNS)
    df_all = append_or_save_csv(df_result, CSV_PATH, overwrite_existing)
    save_summary_plots(df_all)

    processed_folders = len([1 for _x, _path in numeric_folders])
    processed_pairs = len(df_result)
    failed_pairs = int((~df_result["valid"].astype(bool)).sum()) if "valid" in df_result.columns else 0

    print("\nBatch results:")
    print(df_result)

    print("\nD_x and D_y are second-moment beam diameters in pixels.")
    print("w_x and w_y are 2*sigma beam widths in pixels.")
    print("R_asym > 1 means lower-side intensity is stronger than upper-side intensity.")
    print("total_intensity is the sum of camera pixel intensity after background subtraction, not absolute optical power.")
    print("total_intensity should only be compared across images acquired with the same shutter/gain.")
    print("I_norm-based D_x, D_y, and R_asym can be used for relative beam shape analysis.")
    print("A high saturation_ratio means intensity fitting may be unreliable.")
    print("If the ROI does not contain the full beam, D_x and D_y can be underestimated.")
    print("This code extracts relative beam shape parameters from camera images, not absolute optical power.")

    print("\nSummary:")
    print(f"Processed folders: {processed_folders}")
    print(f"Processed pairs: {processed_pairs}")
    print(f"Failed pairs: {failed_pairs}")
    print(f"CSV saved to: {CSV_PATH}")
    print(f"Debug figures saved under: {DEBUG_ROOT}\\x_*/pair_*_sh*/beam_debug.png")


if __name__ == "__main__":
    main()
