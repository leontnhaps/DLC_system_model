#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze PV voltage/electrical power using both beam_integral_sum and
beam_integral_mean.

Input:
    intensity_cal_vol.csv
    또는 filename, vol_load, beam_integral_sum, beam_integral_mean 컬럼이 있는 CSV

Outputs:
    efficiency_sum.csv
    efficiency_mean.csv
    efficiency_sum_mean_analysis/
        beam_sum_correct_voltage.png
        beam_sum_electrical_power.png
        beam_mean_correct_voltage.png
        beam_mean_electrical_power.png
        efficiency_sum_summary.txt
        efficiency_mean_summary.txt

No regression/fitting curves are drawn in this script. The generated CSV files
are intended for later regression or curve-fitting analysis.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Settings
# =========================================================

OUT_DIR_NAME = "efficiency_sum_mean_analysis"
SUM_OUTPUT_CSV_NAME = "efficiency_sum.csv"
MEAN_OUTPUT_CSV_NAME = "efficiency_mean.csv"

SUM_COL = "beam_integral_sum"
MEAN_COL = "beam_integral_mean"
V_COL = "vol_load"

R_LOAD_OHM = 100.0
AMBIENT_VOLTAGE_V = 0.0175
CLIP_NEGATIVE_VOLTAGE = True

MIN_INTEGRAL_FOR_FIT = 0.0
MIN_VOLTAGE_FOR_FIT = 0.0

DPI = 170


# =========================================================
# File select
# =========================================================

def select_csv_file() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            title="Select intensity_cal_vol.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()

    return Path(path) if path else None


# =========================================================
# Utility
# =========================================================

def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")


def to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_min(s: pd.Series) -> float:
    return float(s.min(skipna=True)) if s.notna().any() else np.nan


def safe_max(s: pd.Series) -> float:
    return float(s.max(skipna=True)) if s.notna().any() else np.nan


def safe_mean(s: pd.Series) -> float:
    return float(s.mean(skipna=True)) if s.notna().any() else np.nan


# =========================================================
# Data preparation
# =========================================================

def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    require_columns(df, [V_COL, SUM_COL, MEAN_COL])

    df[V_COL] = to_numeric_series(df, V_COL)
    df[SUM_COL] = to_numeric_series(df, SUM_COL)
    df[MEAN_COL] = to_numeric_series(df, MEAN_COL)

    optional_numeric_cols = [
        "distance",
        "group_id",
        "offset_x_px",
        "offset_y_px",
        "offset_r_px",
        "cell_area_px",
        "beam_integral_norm_total",
        "beam_integral_norm_peak",
    ]
    for col in optional_numeric_cols:
        if col in df.columns:
            df[col] = to_numeric_series(df, col)

    # Voltage correction
    df["ambient_voltage_V"] = AMBIENT_VOLTAGE_V
    df["vol_corr"] = df[V_COL] - AMBIENT_VOLTAGE_V
    if CLIP_NEGATIVE_VOLTAGE:
        df["vol_corr"] = df["vol_corr"].clip(lower=0.0)

    # Current and electrical power using load resistor
    df["R_load_ohm"] = R_LOAD_OHM
    df["I_calc_A"] = df["vol_corr"] / R_LOAD_OHM
    df["I_calc_mA"] = df["I_calc_A"] * 1000.0
    df["P_elec_W"] = (df["vol_corr"] ** 2) / R_LOAD_OHM
    df["P_elec_mW"] = df["P_elec_W"] * 1000.0

    # Valid base condition
    valid_base = df[V_COL].notna() & df["vol_corr"].notna()
    valid_base &= df["vol_corr"] > MIN_VOLTAGE_FOR_FIT
    if "integral_status" in df.columns:
        valid_base &= df["integral_status"].astype(str).str.lower().eq("done")

    df["valid_voltage_row"] = valid_base
    return df


def make_efficiency_dataframe(df: pd.DataFrame, integral_col: str, label: str) -> pd.DataFrame:
    out = df.copy()

    x = out[integral_col].replace(0, np.nan)
    out[f"rel_V_per_{label}"] = out["vol_corr"] / x
    out[f"rel_P_per_{label}"] = out["P_elec_W"] / x

    max_x = out[integral_col].max(skipna=True)
    max_v = out["vol_corr"].max(skipna=True)
    max_p = out["P_elec_W"].max(skipna=True)

    out[f"{label}_norm_by_max"] = (
        out[integral_col] / max_x if pd.notna(max_x) and max_x > 0 else np.nan
    )
    out["vol_corr_norm_by_max"] = (
        out["vol_corr"] / max_v if pd.notna(max_v) and max_v > 0 else np.nan
    )
    out["P_elec_norm_by_max"] = (
        out["P_elec_W"] / max_p if pd.notna(max_p) and max_p > 0 else np.nan
    )

    valid = out["valid_voltage_row"].copy()
    valid &= out[integral_col].notna()
    valid &= out[integral_col] > MIN_INTEGRAL_FOR_FIT

    out[f"valid_for_fit_{label}"] = valid
    return out


# =========================================================
# Plotting
# =========================================================

def save_scatter_plot(
    out_path: Path,
    df_valid: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    x = pd.to_numeric(df_valid[x_col], errors="coerce")
    y = pd.to_numeric(df_valid[y_col], errors="coerce")
    mask = x.notna() & y.notna()

    plt.figure(figsize=(7.6, 5.4))
    plt.scatter(x[mask], y[mask], s=38, alpha=0.82, label="Measured data")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()


# =========================================================
# Summary
# =========================================================

def make_summary_text(df_all: pd.DataFrame, df_valid: pd.DataFrame, integral_col: str, label: str) -> str:
    x = df_valid[integral_col].to_numpy(float)
    v = df_valid["vol_corr"].to_numpy(float)
    p = df_valid["P_elec_W"].to_numpy(float)

    lines = []
    lines.append("=" * 70)
    lines.append(f"PV efficiency summary using {integral_col}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Voltage column             : {V_COL} [V]")
    lines.append(f"Integral column            : {integral_col}")
    lines.append(f"Load resistor              : {R_LOAD_OHM:.6g} ohm")
    lines.append(f"Ambient voltage subtracted : {AMBIENT_VOLTAGE_V:.6g} V")
    lines.append(f"Regression/fitting         : not applied in this script")
    lines.append("")
    lines.append(f"Total rows                 : {len(df_all)}")
    lines.append(f"Valid rows                 : {len(df_valid)}")
    lines.append("")

    if len(df_valid) > 0:
        lines.append("[Input intensity]")
        lines.append(f"{integral_col} min/max/mean : "
                     f"{safe_min(df_valid[integral_col]):.8g} / "
                     f"{safe_max(df_valid[integral_col]):.8g} / "
                     f"{safe_mean(df_valid[integral_col]):.8g}")
        lines.append("")

        lines.append("[Corrected voltage]")
        lines.append(f"vol_load min/max/mean      : "
                     f"{safe_min(df_valid[V_COL]):.8g} / "
                     f"{safe_max(df_valid[V_COL]):.8g} / "
                     f"{safe_mean(df_valid[V_COL]):.8g} V")
        lines.append(f"vol_corr min/max/mean      : "
                     f"{safe_min(df_valid['vol_corr']):.8g} / "
                     f"{safe_max(df_valid['vol_corr']):.8g} / "
                     f"{safe_mean(df_valid['vol_corr']):.8g} V")
        lines.append("")

        lines.append("[Electrical output]")
        lines.append(f"I_calc min/max/mean        : "
                     f"{safe_min(df_valid['I_calc_mA']):.8g} / "
                     f"{safe_max(df_valid['I_calc_mA']):.8g} / "
                     f"{safe_mean(df_valid['I_calc_mA']):.8g} mA")
        lines.append(f"P_elec min/max/mean        : "
                     f"{safe_min(df_valid['P_elec_mW']):.8g} / "
                     f"{safe_max(df_valid['P_elec_mW']):.8g} / "
                     f"{safe_mean(df_valid['P_elec_mW']):.8g} mW")
        lines.append("")

        lines.append("[Correlation only]")
        lines.append(f"Pearson r: {label} vs vol_corr : {pearson_corr(x, v):.8g}")
        lines.append(f"Pearson r: {label} vs P_elec   : {pearson_corr(x, p):.8g}")
        lines.append("")

    lines.append("[Generated files]")
    lines.append(f"CSV     : efficiency_{label}.csv")
    lines.append(f"Summary : efficiency_{label}_summary.txt")
    lines.append("")
    return "\n".join(lines)


# =========================================================
# Main
# =========================================================

def main() -> None:
    csv_path = select_csv_file()
    if csv_path is None:
        print("[INFO] No CSV selected.")
        return

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    print("=" * 70)
    print("[INFO] PV efficiency analysis: sum + mean")
    print(f"[INFO] Input CSV: {csv_path}")
    print("=" * 70)

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[ERROR] Failed to read CSV: {exc}")
        return

    try:
        base_df = prepare_base_dataframe(df)
        sum_df = make_efficiency_dataframe(base_df, SUM_COL, "sum")
        mean_df = make_efficiency_dataframe(base_df, MEAN_COL, "mean")
    except Exception as exc:
        print(f"[ERROR] Failed to prepare data: {exc}")
        return

    output_dir = csv_path.parent
    analysis_dir = output_dir / OUT_DIR_NAME
    analysis_dir.mkdir(exist_ok=True)

    sum_csv_path = output_dir / SUM_OUTPUT_CSV_NAME
    mean_csv_path = output_dir / MEAN_OUTPUT_CSV_NAME

    try:
        sum_df.to_csv(sum_csv_path, index=False, encoding="utf-8-sig")
        mean_df.to_csv(mean_csv_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[ERROR] Failed to save output CSVs: {exc}")
        return

    sum_valid = sum_df[sum_df["valid_for_fit_sum"]].copy()
    mean_valid = mean_df[mean_df["valid_for_fit_mean"]].copy()

    try:
        save_scatter_plot(
            analysis_dir / "beam_sum_correct_voltage.png",
            sum_valid,
            SUM_COL,
            "vol_corr",
            "beam_integral_sum",
            "Corrected load voltage [V]",
            "beam_integral_sum vs Corrected voltage",
        )
        save_scatter_plot(
            analysis_dir / "beam_sum_electrical_power.png",
            sum_valid,
            SUM_COL,
            "P_elec_W",
            "beam_integral_sum",
            "Electrical power [W]",
            "beam_integral_sum vs Electrical power",
        )
        save_scatter_plot(
            analysis_dir / "beam_mean_correct_voltage.png",
            mean_valid,
            MEAN_COL,
            "vol_corr",
            "beam_integral_mean",
            "Corrected load voltage [V]",
            "beam_integral_mean vs Corrected voltage",
        )
        save_scatter_plot(
            analysis_dir / "beam_mean_electrical_power.png",
            mean_valid,
            MEAN_COL,
            "P_elec_W",
            "beam_integral_mean",
            "Electrical power [W]",
            "beam_integral_mean vs Electrical power",
        )
    except Exception as exc:
        print(f"[ERROR] Failed to save plots: {exc}")
        return

    sum_summary_path = analysis_dir / "efficiency_sum_summary.txt"
    mean_summary_path = analysis_dir / "efficiency_mean_summary.txt"

    try:
        sum_summary_path.write_text(
            make_summary_text(sum_df, sum_valid, SUM_COL, "sum"),
            encoding="utf-8",
        )
        mean_summary_path.write_text(
            make_summary_text(mean_df, mean_valid, MEAN_COL, "mean"),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[ERROR] Failed to save summary files: {exc}")
        return

    print("")
    print("[DONE]")
    print(f"Total rows       : {len(base_df)}")
    print(f"Valid sum rows   : {len(sum_valid)}")
    print(f"Valid mean rows  : {len(mean_valid)}")
    print(f"Sum CSV          : {sum_csv_path}")
    print(f"Mean CSV         : {mean_csv_path}")
    print(f"Analysis dir     : {analysis_dir}")


if __name__ == "__main__":
    main()
