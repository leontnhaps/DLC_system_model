#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regression comparison with 5 holdout points for solar-cell intensity data.

This script reads two CSV files:
    1) SUM CSV  : uses beam_integral_sum
    2) MEAN CSV : uses beam_integral_mean

For each CSV:
    1. valid data are selected.
    2. x-axis range is divided into 5 intervals.
    3. one data point is selected from each interval as holdout data.
    4. regression is performed using all remaining data.
    5. two regression modes are performed:
        - normalized-x regression using S_n
        - raw-x regression without S_n normalization
    6. three plots are saved for each mode:
        - train regression plot
        - holdout-only comparison plot
        - all-data plot with 5 holdout points highlighted

Models:
    1) Linear
    2) Quadratic
    3) Shifted quadratic monotone
    4) Exponential growth

Important:
    - No above-curve CSV files are generated.
    - Normalized mode uses S_n internally for numerical stability.
    - Raw mode uses the original S value directly, without S_n normalization.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# =========================================================
# Settings
# =========================================================

V_COL = "vol_load"

SUM_X_COL = "beam_integral_sum"
MEAN_X_COL = "beam_integral_mean"

AMBIENT_VOLTAGE_V = 0.0175
CLIP_NEGATIVE_VOLTAGE = True

ROBUST_LOSS = "soft_l1"
F_SCALE = 0.08

HOLDOUT_COUNT = 5
MIN_TRAIN_ROWS = 5

DPI = 170


# =========================================================
# File select
# =========================================================

def _ask_csv(title: str) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
    finally:
        root.destroy()

    return Path(path) if path else None


def select_sum_csv_file() -> Path | None:
    return _ask_csv("Select SUM CSV with vol_load and beam_integral_sum")


def select_mean_csv_file() -> Path | None:
    return _ask_csv("Select MEAN CSV with vol_load and beam_integral_mean")


# =========================================================
# Metrics
# =========================================================

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan

    yt = y_true[mask]
    yp = y_pred[mask]

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))

    if ss_tot <= 1e-12:
        return np.nan

    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan

    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


# =========================================================
# Normalization
# =========================================================

def make_x_normalization(x_train: np.ndarray) -> tuple[float, float]:
    x_min = float(np.min(x_train))
    x_max = float(np.max(x_train))
    x_scale = x_max - x_min

    if x_scale <= 1e-12:
        x_scale = 1.0

    return x_min, x_scale


def normalize_x(x: np.ndarray, x_min: float, x_scale: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x_min) / x_scale


# =========================================================
# Models
#   - normalized mode: t = S_n
#   - raw mode       : t = original S value
# =========================================================

def model_linear_t(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    V = m*S_n + b
    p = [m, b]
    """
    m, b = p
    return m * t + b


def model_quadratic_t(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    V = a*S_n^2 + b*S_n + c
    p = [a, b, c]
    """
    a, b, c = p
    return a * t**2 + b * t + c


def model_shifted_quadratic_t(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    V = a*(S_n - h)^2 + c
    p = [a, h, c]
    """
    a, h, c = p
    return a * (t - h) ** 2 + c


def model_exponential_growth_t(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Normalized mode: V = A*(exp(k*S_n) - 1) + c
    Raw mode       : V = A*(exp(k*S)   - 1) + c
    p = [A, k, c]

    np.clip is only used to prevent numerical overflow during optimization.
    """
    A, k, c = p
    z = np.clip(k * t, -60.0, 60.0)
    return A * (np.exp(z) - 1.0) + c


def predict_fit(fit: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if fit.get("use_x_normalization", True):
        t = normalize_x(x, fit["x_min"], fit["x_scale"])
    else:
        t = x

    p = fit["params"]
    model = fit["model"]

    if model == "linear":
        return model_linear_t(t, p)

    if model == "quadratic":
        return model_quadratic_t(t, p)

    if model == "shifted_quadratic":
        return model_shifted_quadratic_t(t, p)

    if model == "exponential_growth":
        return model_exponential_growth_t(t, p)

    raise ValueError(f"Unknown model: {model}")


# =========================================================
# Fit helpers
# =========================================================

def _clip_initial(p0: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    eps = 1e-9
    return np.minimum(np.maximum(p0, lower + eps), upper - eps)


def _regression_x(
    x_train: np.ndarray,
    x_min: float,
    x_scale: float,
    use_x_normalization: bool,
) -> np.ndarray:
    if use_x_normalization:
        return normalize_x(x_train, x_min, x_scale)
    return np.asarray(x_train, dtype=float)


def _raw_x_stats(t: np.ndarray) -> tuple[float, float, float]:
    t = np.asarray(t, dtype=float)
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    t_range = max(t_max - t_min, 1e-12)
    t_abs = max(abs(t_min), abs(t_max), 1.0)
    return t_min, t_range, t_abs


def _make_fit_dict(
    name: str,
    model: str,
    params: np.ndarray,
    x_min: float,
    x_scale: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    success: bool,
    use_x_normalization: bool,
) -> dict:
    fit = {
        "name": name,
        "model": model,
        "params": np.asarray(params, dtype=float),
        "x_min": float(x_min),
        "x_scale": float(x_scale),
        "success": bool(success),
        "use_x_normalization": bool(use_x_normalization),
        "x_mode": "normalized_Sn" if use_x_normalization else "raw_S",
        "x_symbol": "S_n" if use_x_normalization else "S",
    }

    y_pred_train = predict_fit(fit, x_train)

    fit["train_r2"] = r2_score(y_train, y_pred_train)
    fit["train_rmse"] = rmse(y_train, y_pred_train)

    fit["holdout_r2"] = np.nan
    fit["holdout_rmse"] = np.nan

    return fit


def fit_linear_robust(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_min: float,
    x_scale: float,
    use_x_normalization: bool = True,
) -> dict:
    t = _regression_x(x_train, x_min, x_scale, use_x_normalization)

    try:
        m0, b0 = np.polyfit(t, y_train, 1)
    except Exception:
        m0, b0 = 1.0, float(np.mean(y_train))

    lower = np.array([-50.0, -2.0], dtype=float)
    upper = np.array([50.0, 2.0], dtype=float)
    p0 = _clip_initial(np.array([m0, b0], dtype=float), lower, upper)

    res = least_squares(
        lambda p: model_linear_t(t, p) - y_train,
        p0,
        bounds=(lower, upper),
        loss=ROBUST_LOSS,
        f_scale=F_SCALE,
        max_nfev=20000,
    )

    return _make_fit_dict(
        name="linear",
        model="linear",
        params=res.x,
        x_min=x_min,
        x_scale=x_scale,
        x_train=x_train,
        y_train=y_train,
        success=res.success,
        use_x_normalization=use_x_normalization,
    )


def fit_quadratic_robust(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_min: float,
    x_scale: float,
    use_x_normalization: bool = True,
) -> dict:
    t = _regression_x(x_train, x_min, x_scale, use_x_normalization)

    try:
        a0, b0, c0 = np.polyfit(t, y_train, 2)
        a0 = max(float(a0), 1e-6)
    except Exception:
        a0, b0, c0 = 1.0, 0.0, float(np.min(y_train))

    lower = np.array([0.0, -50.0, -2.0], dtype=float)
    upper = np.array([50.0, 50.0, 2.0], dtype=float)
    p0 = _clip_initial(np.array([a0, b0, c0], dtype=float), lower, upper)

    res = least_squares(
        lambda p: model_quadratic_t(t, p) - y_train,
        p0,
        bounds=(lower, upper),
        loss=ROBUST_LOSS,
        f_scale=F_SCALE,
        max_nfev=20000,
    )

    return _make_fit_dict(
        name="quadratic",
        model="quadratic",
        params=res.x,
        x_min=x_min,
        x_scale=x_scale,
        x_train=x_train,
        y_train=y_train,
        success=res.success,
        use_x_normalization=use_x_normalization,
    )


def fit_shifted_quadratic_robust(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_min: float,
    x_scale: float,
    use_x_normalization: bool = True,
) -> dict:
    t = _regression_x(x_train, x_min, x_scale, use_x_normalization)

    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    y_range = max(y_max - y_min, 1e-6)
    t_min, t_range, _ = _raw_x_stats(t)

    if use_x_normalization:
        # h <= 0 means the vertex is placed left of the normalized data range.
        p0 = np.array([y_range, -0.05, y_min], dtype=float)
        lower = np.array([0.0, -5.0, -2.0], dtype=float)
        upper = np.array([50.0, 0.0, 2.0], dtype=float)
    else:
        # Raw mode: V = a*(S - h)^2 + c.
        # h is constrained to the left of the measured S range to keep monotone growth.
        p0 = np.array([y_range / (t_range**2), t_min - 0.05 * t_range, y_min], dtype=float)
        lower = np.array([0.0, t_min - 5.0 * t_range, -2.0], dtype=float)
        upper = np.array([50.0 / (t_range**2), t_min, 2.0], dtype=float)

    p0 = _clip_initial(p0, lower, upper)

    res = least_squares(
        lambda p: model_shifted_quadratic_t(t, p) - y_train,
        p0,
        bounds=(lower, upper),
        loss=ROBUST_LOSS,
        f_scale=F_SCALE,
        max_nfev=20000,
    )

    return _make_fit_dict(
        name="shifted_quadratic",
        model="shifted_quadratic",
        params=res.x,
        x_min=x_min,
        x_scale=x_scale,
        x_train=x_train,
        y_train=y_train,
        success=res.success,
        use_x_normalization=use_x_normalization,
    )


def fit_exponential_growth_robust(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_min: float,
    x_scale: float,
    use_x_normalization: bool = True,
) -> dict:
    t = _regression_x(x_train, x_min, x_scale, use_x_normalization)

    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    y_range = max(y_max - y_min, 1e-6)
    _, t_range, t_abs = _raw_x_stats(t)

    if use_x_normalization:
        k0 = 1.0
        k_upper = 10.0
    else:
        # Raw S can be large, so k must be much smaller to avoid exp overflow.
        k0 = 1.0 / t_abs
        k_upper = 10.0 / t_abs

    p0 = np.array([y_range, k0, y_min], dtype=float)

    lower = np.array([0.0, 0.0, -2.0], dtype=float)
    upper = np.array([5.0, k_upper, 2.0], dtype=float)
    p0 = _clip_initial(p0, lower, upper)

    res = least_squares(
        lambda p: model_exponential_growth_t(t, p) - y_train,
        p0,
        bounds=(lower, upper),
        loss=ROBUST_LOSS,
        f_scale=F_SCALE,
        max_nfev=30000,
    )

    return _make_fit_dict(
        name="exponential_growth",
        model="exponential_growth",
        params=res.x,
        x_min=x_min,
        x_scale=x_scale,
        x_train=x_train,
        y_train=y_train,
        success=res.success,
        use_x_normalization=use_x_normalization,
    )


def fit_all_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_holdout: np.ndarray,
    y_holdout: np.ndarray,
    use_x_normalization: bool = True,
) -> list[dict]:
    x_min, x_scale = make_x_normalization(x_train)

    fits = [
        fit_linear_robust(x_train, y_train, x_min, x_scale, use_x_normalization),
        fit_quadratic_robust(x_train, y_train, x_min, x_scale, use_x_normalization),
        fit_shifted_quadratic_robust(x_train, y_train, x_min, x_scale, use_x_normalization),
        fit_exponential_growth_robust(x_train, y_train, x_min, x_scale, use_x_normalization),
    ]

    for fit in fits:
        y_pred_holdout = predict_fit(fit, x_holdout)
        fit["holdout_r2"] = r2_score(y_holdout, y_pred_holdout)
        fit["holdout_rmse"] = rmse(y_holdout, y_pred_holdout)

    return fits


# =========================================================
# Data preparation
# =========================================================

def prepare_dataframe(csv_path: Path, x_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if x_col not in df.columns:
        raise ValueError(f"Missing column: {x_col}")

    if V_COL not in df.columns:
        raise ValueError(f"Missing column: {V_COL}")

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[V_COL] = pd.to_numeric(df[V_COL], errors="coerce")

    df["ambient_voltage_V"] = AMBIENT_VOLTAGE_V
    df["vol_corr"] = df[V_COL] - AMBIENT_VOLTAGE_V

    if CLIP_NEGATIVE_VOLTAGE:
        df["vol_corr"] = df["vol_corr"].clip(lower=0.0)

    valid = df[x_col].notna()
    valid &= df["vol_corr"].notna()
    valid &= df[x_col] > 0

    if "integral_status" in df.columns:
        valid &= df["integral_status"].astype(str).str.lower().eq("done")

    df["valid_for_fit"] = valid

    df_valid = df[valid].copy()

    min_required = HOLDOUT_COUNT + MIN_TRAIN_ROWS
    if len(df_valid) < min_required:
        raise ValueError(
            f"Not enough valid rows: {len(df_valid)} < {min_required}. "
            f"Need at least {HOLDOUT_COUNT} holdout + {MIN_TRAIN_ROWS} train rows."
        )

    return df, df_valid


# =========================================================
# Holdout selection
# =========================================================

def select_holdout_indices_by_x_bins(
    df_valid: pd.DataFrame,
    x_col: str,
    n_bins: int = HOLDOUT_COUNT,
) -> list:
    """
    Divide x range into n_bins intervals and select one point per interval.

    If an interval is empty, fallback selection fills the remaining holdout
    points using evenly spaced sorted x positions.
    """
    if len(df_valid) < n_bins:
        raise ValueError(f"Not enough valid rows for {n_bins} holdout points.")

    temp = df_valid[[x_col]].copy()
    temp["_orig_index"] = df_valid.index
    temp = temp.sort_values(x_col).reset_index(drop=True)

    x = temp[x_col].to_numpy(dtype=float)
    orig_indices = temp["_orig_index"].to_numpy()

    x_min = float(np.min(x))
    x_max = float(np.max(x))

    selected: list = []

    if x_max <= x_min + 1e-12:
        positions = np.linspace(0, len(temp) - 1, n_bins).round().astype(int)
        for pos in positions:
            idx = orig_indices[pos]
            if idx not in selected:
                selected.append(idx)
        return selected[:n_bins]

    edges = np.linspace(x_min, x_max, n_bins + 1)

    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        center = (lo + hi) / 2.0

        if i == n_bins - 1:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)

        candidate_positions = np.where(mask)[0]

        if len(candidate_positions) == 0:
            continue

        available_positions = [
            pos for pos in candidate_positions
            if orig_indices[pos] not in selected
        ]

        if not available_positions:
            continue

        best_pos = min(
            available_positions,
            key=lambda pos: abs(float(x[pos]) - center)
        )
        selected.append(orig_indices[best_pos])

    # Fallback: fill missing holdout points with evenly spaced sorted data
    if len(selected) < n_bins:
        target_positions = np.linspace(0, len(temp) - 1, n_bins).round().astype(int)

        for pos in target_positions:
            idx = orig_indices[pos]
            if idx not in selected:
                selected.append(idx)
            if len(selected) == n_bins:
                break

    # Final fallback: scan all rows
    if len(selected) < n_bins:
        for idx in orig_indices:
            if idx not in selected:
                selected.append(idx)
            if len(selected) == n_bins:
                break

    if len(selected) != n_bins:
        raise ValueError(f"Failed to select {n_bins} holdout points.")

    return selected


# =========================================================
# Output cleanup
# =========================================================

def cleanup_old_outputs(out_dir: Path, out_dir_name: str) -> None:
    """
    Remove legacy outputs from previous version so that old files do not confuse results.
    """
    if not out_dir.exists():
        return

    legacy_patterns = [
        "points_above_*.csv",
        f"{out_dir_name}_plot.png",
    ]

    for pattern in legacy_patterns:
        for path in out_dir.glob(pattern):
            try:
                path.unlink()
                print(f"[CLEAN] Removed old file: {path.name}")
            except Exception as exc:
                print(f"[WARN] Could not remove old file {path}: {exc}")


# =========================================================
# Plot helpers
# =========================================================

def _x_line_from_data(x_train: np.ndarray, x_holdout: np.ndarray) -> np.ndarray:
    x_all = np.concatenate([x_train, x_holdout])
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))

    if x_max <= x_min + 1e-12:
        x_max = x_min + 1.0

    margin = 0.03 * (x_max - x_min)
    return np.linspace(x_min - margin, x_max + margin, 500)


def _plot_fit_lines(x_line: np.ndarray, fits: list[dict]) -> None:
    for fit in fits:
        y_line = predict_fit(fit, x_line)
        mode_suffix = "" if fit.get("use_x_normalization", True) else " (raw S)"

        if fit["name"] == "linear":
            plt.plot(
                x_line,
                y_line,
                linestyle=":",
                linewidth=2.3,
                label=(
                    f"linear{mode_suffix} | train R²={fit['train_r2']:.4f}, "
                    f"test RMSE={fit['holdout_rmse']:.4f}"
                ),
            )

        elif fit["name"] == "quadratic":
            plt.plot(
                x_line,
                y_line,
                linestyle="-",
                linewidth=2.3,
                label=(
                    f"quadratic{mode_suffix} | train R²={fit['train_r2']:.4f}, "
                    f"test RMSE={fit['holdout_rmse']:.4f}"
                ),
            )

        elif fit["name"] == "shifted_quadratic":
            plt.plot(
                x_line,
                y_line,
                linestyle="-",
                linewidth=2.6,
                label=(
                    f"shifted quadratic{mode_suffix} | train R²={fit['train_r2']:.4f}, "
                    f"test RMSE={fit['holdout_rmse']:.4f}"
                ),
            )

        elif fit["name"] == "exponential_growth":
            plt.plot(
                x_line,
                y_line,
                linestyle="--",
                linewidth=2.6,
                label=(
                    f"exponential{mode_suffix} | train R²={fit['train_r2']:.4f}, "
                    f"test RMSE={fit['holdout_rmse']:.4f}"
                ),
            )


def _save_current_plot(out_path: Path) -> None:
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_train_regression(
    out_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_holdout: np.ndarray,
    x_col: str,
    prefix: str,
    fits: list[dict],
) -> None:
    plt.figure(figsize=(9.2, 6.3))

    plt.scatter(
        x_train,
        y_train,
        s=42,
        alpha=0.8,
        label="Train data"
    )

    x_line = _x_line_from_data(x_train, x_holdout)
    _plot_fit_lines(x_line, fits)

    plt.xlabel(x_col)
    plt.ylabel("Corrected load voltage [V]")
    plt.title(f"{prefix.upper()}: regression using training data only")
    _save_current_plot(out_path)


def plot_holdout_compare(
    out_path: Path,
    x_train: np.ndarray,
    x_holdout: np.ndarray,
    y_holdout: np.ndarray,
    x_col: str,
    prefix: str,
    fits: list[dict],
) -> None:
    plt.figure(figsize=(9.2, 6.3))

    plt.scatter(
        x_holdout,
        y_holdout,
        s=95,
        marker="X",
        label="Test data"
    )

    x_line = _x_line_from_data(x_train, x_holdout)
    _plot_fit_lines(x_line, fits)

    plt.xlabel(x_col)
    plt.ylabel("Corrected load voltage [V]")
    plt.title(f"{prefix.upper()}: 5 holdout data vs trained regression curves")
    _save_current_plot(out_path)


def plot_all_highlight_holdout(
    out_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_holdout: np.ndarray,
    y_holdout: np.ndarray,
    x_col: str,
    prefix: str,
    fits: list[dict],
) -> None:
    plt.figure(figsize=(9.2, 6.3))

    plt.scatter(
        x_train,
        y_train,
        s=38,
        alpha=0.65,
        label="Train data"
    )

    plt.scatter(
        x_holdout,
        y_holdout,
        s=105,
        marker="X",
        label="Test data"
    )

    x_line = _x_line_from_data(x_train, x_holdout)
    _plot_fit_lines(x_line, fits)

    plt.xlabel("Beam Intensity Mean")
    plt.ylabel("Corrected load voltage [V]")
    plt.title(f"{prefix.upper()}: all data with 5 holdout points highlighted")
    _save_current_plot(out_path)


# =========================================================
# Save result CSV
# =========================================================

def add_predictions_to_dataframe(
    df: pd.DataFrame,
    x_col: str,
    holdout_indices: list,
    fits: list[dict],
) -> pd.DataFrame:
    df = df.copy()

    df["fit_split"] = ""
    df.loc[df["valid_for_fit"], "fit_split"] = "train"
    df.loc[holdout_indices, "fit_split"] = "holdout"

    valid_mask = df["valid_for_fit"]
    x_all_valid = df.loc[valid_mask, x_col].to_numpy(dtype=float)

    for fit in fits:
        model_name = fit["name"]
        pred_col = f"V_pred_{model_name}"
        residual_col = f"V_residual_{model_name}"

        df[pred_col] = np.nan
        df.loc[valid_mask, pred_col] = predict_fit(fit, x_all_valid)
        df[residual_col] = df["vol_corr"] - df[pred_col]

    return df


def save_holdout_csv(
    out_path: Path,
    df_result: pd.DataFrame,
    x_col: str,
    holdout_indices: list,
) -> None:
    holdout_df = df_result.loc[holdout_indices].copy()

    preferred_cols = [
        "filename",
        "fit_split",
        "group_id",
        "distance",
        x_col,
        V_COL,
        "ambient_voltage_V",
        "vol_corr",
        "offset_x_px",
        "offset_y_px",
        "offset_r_px",
        "cell_area_px",
        "beam_integral_sum",
        "beam_integral_mean",
        "V_pred_linear",
        "V_residual_linear",
        "V_pred_quadratic",
        "V_residual_quadratic",
        "V_pred_shifted_quadratic",
        "V_residual_shifted_quadratic",
        "V_pred_exponential_growth",
        "V_residual_exponential_growth",
    ]

    cols = []
    seen = set()

    for col in preferred_cols:
        if col in holdout_df.columns and col not in seen:
            cols.append(col)
            seen.add(col)

    holdout_df.to_csv(out_path, columns=cols, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out_path}")


# =========================================================
# Summary
# =========================================================

def save_summary(
    out_path: Path,
    csv_path: Path,
    x_col: str,
    prefix: str,
    n_total: int,
    n_valid: int,
    n_train: int,
    n_holdout: int,
    holdout_indices: list,
    fits: list[dict],
) -> None:
    lines = []
    use_norm = fits[0].get("use_x_normalization", True) if fits else True
    mode_title = "normalized S_n" if use_norm else "raw S without S_n normalization"
    x_symbol = "S_n" if use_norm else "S"

    lines.append("=" * 70)
    lines.append(f"{prefix.upper()} regression summary with 5 holdout points ({mode_title})")
    lines.append("=" * 70)
    lines.append(f"Input CSV       : {csv_path}")
    lines.append(f"X column        : {x_col}")
    lines.append(f"Y column        : corrected voltage = {V_COL} - {AMBIENT_VOLTAGE_V}")
    lines.append(f"Total rows      : {n_total}")
    lines.append(f"Valid rows      : {n_valid}")
    lines.append(f"Training rows   : {n_train}")
    lines.append(f"Holdout rows    : {n_holdout}")
    lines.append(f"Holdout indices : {list(holdout_indices)}")
    lines.append("")
    lines.append("[Method]")
    lines.append("The valid data range on the x-axis was divided into 5 intervals.")
    lines.append("One data point was selected from each interval as holdout data.")
    lines.append("The 5 holdout points were excluded from fitting.")
    lines.append("Regression was performed using the remaining training data.")
    lines.append("")
    if use_norm:
        lines.append("[Internal x normalization]")
        lines.append("For numerical stability, regression uses normalized x:")
        lines.append("S_n = (S - S_min_train) / (S_max_train - S_min_train)")
        lines.append("Plots still use the original x-axis values.")
    else:
        lines.append("[No internal x normalization]")
        lines.append("Regression uses the original x value directly:")
        lines.append("S = original beam integral value")
        lines.append("No S_n = (S - S_min)/(S_max - S_min) transformation is used.")
    lines.append("")
    lines.append("[Model results]")
    lines.append("")

    for fit in fits:
        p = fit["params"]

        lines.append("-" * 70)
        lines.append(f"Model: {fit['name']}")
        lines.append(f"success       : {fit['success']}")
        lines.append(f"S_min_train   : {fit['x_min']:.10f}")
        lines.append(f"S_scale_train : {fit['x_scale']:.10f}")
        lines.append(f"train R^2     : {fit['train_r2']:.10f}")
        lines.append(f"train RMSE    : {fit['train_rmse']:.10f}")
        lines.append(f"holdout R^2   : {fit['holdout_r2']:.10f}")
        lines.append(f"holdout RMSE  : {fit['holdout_rmse']:.10f}")

        if fit["name"] == "linear":
            lines.append(f"Equation      : V = m*{x_symbol} + b")
            lines.append(f"m = {p[0]:.10f}")
            lines.append(f"b = {p[1]:.10f}")

        elif fit["name"] == "quadratic":
            lines.append(f"Equation      : V = a*{x_symbol}^2 + b*{x_symbol} + c")
            lines.append(f"a = {p[0]:.10f}")
            lines.append(f"b = {p[1]:.10f}")
            lines.append(f"c = {p[2]:.10f}")

        elif fit["name"] == "shifted_quadratic":
            lines.append(f"Equation      : V = a*({x_symbol} - h)^2 + c")
            lines.append(f"a = {p[0]:.10f}")
            lines.append(f"h = {p[1]:.10f}")
            lines.append(f"c = {p[2]:.10f}")

        elif fit["name"] == "exponential_growth":
            lines.append(f"Equation      : V = A*(exp(k*{x_symbol}) - 1) + c")
            lines.append(f"A = {p[0]:.10f}")
            lines.append(f"k = {p[1]:.10f}")
            lines.append(f"c = {p[2]:.10f}")

        lines.append("")

    sorted_by_holdout = sorted(
        fits,
        key=lambda f: np.nan_to_num(f["holdout_rmse"], nan=np.inf),
    )

    lines.append("[Best model by holdout RMSE]")
    for rank, fit in enumerate(sorted_by_holdout, start=1):
        lines.append(
            f"{rank}. {fit['name']}: "
            f"holdout RMSE={fit['holdout_rmse']:.10f}, "
            f"train R^2={fit['train_r2']:.10f}"
        )

    lines.append("=" * 70)

    out_path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"[SAVE] {out_path}")


def save_outputs_for_fit_mode(
    out_dir: Path,
    out_dir_name: str,
    file_tag: str,
    plot_prefix: str,
    csv_path: Path,
    x_col: str,
    df: pd.DataFrame,
    df_valid: pd.DataFrame,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    holdout_indices: list,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_holdout: np.ndarray,
    y_holdout: np.ndarray,
    fits: list[dict],
) -> None:
    result_csv_path = out_dir / f"{out_dir_name}_{file_tag}_results.csv"
    holdout_csv_path = out_dir / f"{out_dir_name}_{file_tag}_holdout_points.csv"
    summary_path = out_dir / f"{out_dir_name}_{file_tag}_summary.txt"

    train_plot_path = out_dir / f"{out_dir_name}_{file_tag}_train_regression_plot.png"
    holdout_plot_path = out_dir / f"{out_dir_name}_{file_tag}_holdout_compare_plot.png"
    all_plot_path = out_dir / f"{out_dir_name}_{file_tag}_all_highlight_holdout_plot.png"

    df_result = add_predictions_to_dataframe(
        df=df,
        x_col=x_col,
        holdout_indices=holdout_indices,
        fits=fits,
    )

    df_result.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {result_csv_path}")

    save_holdout_csv(
        out_path=holdout_csv_path,
        df_result=df_result,
        x_col=x_col,
        holdout_indices=holdout_indices,
    )

    plot_train_regression(
        out_path=train_plot_path,
        x_train=x_train,
        y_train=y_train,
        x_holdout=x_holdout,
        x_col=x_col,
        prefix=plot_prefix,
        fits=fits,
    )

    plot_holdout_compare(
        out_path=holdout_plot_path,
        x_train=x_train,
        x_holdout=x_holdout,
        y_holdout=y_holdout,
        x_col=x_col,
        prefix=plot_prefix,
        fits=fits,
    )

    plot_all_highlight_holdout(
        out_path=all_plot_path,
        x_train=x_train,
        y_train=y_train,
        x_holdout=x_holdout,
        y_holdout=y_holdout,
        x_col=x_col,
        prefix=plot_prefix,
        fits=fits,
    )

    save_summary(
        out_path=summary_path,
        csv_path=csv_path,
        x_col=x_col,
        prefix=plot_prefix,
        n_total=len(df),
        n_valid=len(df_valid),
        n_train=len(train_df),
        n_holdout=len(holdout_df),
        holdout_indices=holdout_indices,
        fits=fits,
    )

    print("")
    print(f"[RESULT: {file_tag}]")
    for fit in fits:
        print(
            f"{fit['name']:22s} | "
            f"train R²={fit['train_r2']:.4f} | "
            f"train RMSE={fit['train_rmse']:.4f} | "
            f"holdout RMSE={fit['holdout_rmse']:.4f}"
        )


# =========================================================
# One run
# =========================================================

def run_regression_with_holdout(
    csv_path: Path,
    x_col: str,
    out_dir_name: str,
    prefix: str,
) -> None:
    print("=" * 70)
    print(f"[{prefix.upper()}] Regression with 5 holdout points")
    print(f"Input CSV : {csv_path}")
    print(f"X column  : {x_col}")
    print("=" * 70)

    df, df_valid = prepare_dataframe(csv_path, x_col)

    holdout_indices = select_holdout_indices_by_x_bins(
        df_valid=df_valid,
        x_col=x_col,
        n_bins=HOLDOUT_COUNT,
    )

    train_df = df_valid[~df_valid.index.isin(holdout_indices)].copy()
    holdout_df = df_valid[df_valid.index.isin(holdout_indices)].copy()

    if len(train_df) < MIN_TRAIN_ROWS:
        raise ValueError(f"Not enough training rows: {len(train_df)} < {MIN_TRAIN_ROWS}")

    x_train = train_df[x_col].to_numpy(dtype=float)
    y_train = train_df["vol_corr"].to_numpy(dtype=float)

    x_holdout = holdout_df[x_col].to_numpy(dtype=float)
    y_holdout = holdout_df["vol_corr"].to_numpy(dtype=float)

    print(f"[INFO] valid rows   : {len(df_valid)}")
    print(f"[INFO] train rows   : {len(train_df)}")
    print(f"[INFO] holdout rows : {len(holdout_df)}")
    print(f"[INFO] holdout idx  : {list(holdout_indices)}")

    # 1) Existing style: normalized S_n regression
    fits_norm = fit_all_models(
        x_train=x_train,
        y_train=y_train,
        x_holdout=x_holdout,
        y_holdout=y_holdout,
        use_x_normalization=True,
    )

    # 2) Added style: raw S regression without S_n normalization
    fits_raw = fit_all_models(
        x_train=x_train,
        y_train=y_train,
        x_holdout=x_holdout,
        y_holdout=y_holdout,
        use_x_normalization=False,
    )

    out_dir = csv_path.parent / out_dir_name
    out_dir.mkdir(exist_ok=True)
    cleanup_old_outputs(out_dir, out_dir_name)

    save_outputs_for_fit_mode(
        out_dir=out_dir,
        out_dir_name=out_dir_name,
        file_tag="norm_Sn",
        plot_prefix=f"{prefix} / normalized S_n",
        csv_path=csv_path,
        x_col=x_col,
        df=df,
        df_valid=df_valid,
        train_df=train_df,
        holdout_df=holdout_df,
        holdout_indices=holdout_indices,
        x_train=x_train,
        y_train=y_train,
        x_holdout=x_holdout,
        y_holdout=y_holdout,
        fits=fits_norm,
    )

    save_outputs_for_fit_mode(
        out_dir=out_dir,
        out_dir_name=out_dir_name,
        file_tag="raw_S",
        plot_prefix=f"{prefix} / raw S",
        csv_path=csv_path,
        x_col=x_col,
        df=df,
        df_valid=df_valid,
        train_df=train_df,
        holdout_df=holdout_df,
        holdout_indices=holdout_indices,
        x_train=x_train,
        y_train=y_train,
        x_holdout=x_holdout,
        y_holdout=y_holdout,
        fits=fits_raw,
    )

    print("")
    print("[DONE]")
    print(f"Output folder      : {out_dir}")
    print(f"Normalized outputs : {out_dir_name}_norm_Sn_*.png/csv/txt")
    print(f"Raw-S outputs      : {out_dir_name}_raw_S_*.png/csv/txt")


# =========================================================
# Main
# =========================================================

def main() -> None:
    sum_csv_path = select_sum_csv_file()
    if sum_csv_path is None:
        print("[INFO] No SUM CSV selected.")
        return

    mean_csv_path = select_mean_csv_file()
    if mean_csv_path is None:
        print("[INFO] No MEAN CSV selected.")
        return

    try:
        run_regression_with_holdout(
            csv_path=sum_csv_path,
            x_col=SUM_X_COL,
            out_dir_name="regression_sum",
            prefix="sum",
        )
    except Exception as exc:
        print(f"[ERROR] SUM regression failed: {exc}")

    print("")
    print("#" * 70)
    print("")

    try:
        run_regression_with_holdout(
            csv_path=mean_csv_path,
            x_col=MEAN_X_COL,
            out_dir_name="regression_mean",
            prefix="mean",
        )
    except Exception as exc:
        print(f"[ERROR] MEAN regression failed: {exc}")


if __name__ == "__main__":
    main()
