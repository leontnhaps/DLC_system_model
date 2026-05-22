#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot only: SUM / MEAN beam integral vs voltage/power

Input:
    1) SUM CSV  : efficiency_sum.csv 또는 beam_integral_sum, vol_load가 있는 CSV
    2) MEAN CSV : efficiency_mean.csv 또는 beam_integral_mean, vol_load가 있는 CSV

Output folder:
    plot_sum_mean/

Saved images:
    1) sum_corrected_voltage.png
    2) sum_electrical_power.png
    3) mean_corrected_voltage.png
    4) mean_electrical_power.png
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Settings
# =========================================================

SUM_X_COL = "beam_integral_sum"
MEAN_X_COL = "beam_integral_mean"
V_COL = "vol_load"

R_LOAD_OHM = 100.0
AMBIENT_VOLTAGE_V = 0.0175
CLIP_NEGATIVE_VOLTAGE = True

OUT_DIR_NAME = "plot_sum_mean"
DPI = 170


# =========================================================
# File select
# =========================================================

def ask_csv(title: str) -> Path | None:
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


# =========================================================
# Data preparation
# =========================================================

def prepare_dataframe(csv_path: Path, x_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if x_col not in df.columns:
        raise ValueError(f"Missing column: {x_col}")

    if V_COL not in df.columns:
        raise ValueError(f"Missing column: {V_COL}")

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[V_COL] = pd.to_numeric(df[V_COL], errors="coerce")

    # Corrected voltage
    df["ambient_voltage_V"] = AMBIENT_VOLTAGE_V
    df["vol_corr"] = df[V_COL] - AMBIENT_VOLTAGE_V

    if CLIP_NEGATIVE_VOLTAGE:
        df["vol_corr"] = df["vol_corr"].clip(lower=0.0)

    # Electrical power
    df["R_load_ohm"] = R_LOAD_OHM
    df["P_elec_W"] = (df["vol_corr"] ** 2) / R_LOAD_OHM
    df["P_elec_mW"] = df["P_elec_W"] * 1000.0

    # Valid data
    valid = df[x_col].notna()
    valid &= df["vol_corr"].notna()
    valid &= df[x_col] > 0

    if "integral_status" in df.columns:
        valid &= df["integral_status"].astype(str).str.lower().eq("done")

    df["valid_for_plot"] = valid

    return df


# =========================================================
# Plot functions
# =========================================================

def save_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    valid_df = df[df["valid_for_plot"]].copy()

    if len(valid_df) == 0:
        print(f"[WARN] No valid data for plot: {out_path.name}")
        return

    x = valid_df[x_col].to_numpy(dtype=float)
    y = valid_df[y_col].to_numpy(dtype=float)

    plt.figure(figsize=(8.8, 6.1))
    plt.scatter(x, y, s=42, alpha=0.8, label="Measured data")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()

    print(f"[SAVE] {out_path}")


def run_plot(csv_path: Path, x_col: str, prefix: str, out_dir: Path) -> None:
    print("=" * 70)
    print(f"[{prefix.upper()}] Plot only")
    print(f"Input CSV : {csv_path}")
    print(f"X column  : {x_col}")
    print("=" * 70)

    df = prepare_dataframe(csv_path, x_col)

    valid_count = int(df["valid_for_plot"].sum())
    print(f"[INFO] Valid rows: {valid_count}")

    # 1. Beam integral vs corrected voltage
    save_scatter_plot(
        df=df,
        x_col=x_col,
        y_col="vol_corr",
        out_path=out_dir / f"{prefix}_corrected_voltage.png",
        title=f"{prefix.upper()}: Beam integral vs corrected voltage",
        xlabel=x_col,
        ylabel="Corrected load voltage [V]",
    )

    # 2. Beam integral vs electrical power
    save_scatter_plot(
        df=df,
        x_col=x_col,
        y_col="P_elec_mW",
        out_path=out_dir / f"{prefix}_electrical_power.png",
        title=f"{prefix.upper()}: Beam integral vs electrical power",
        xlabel=x_col,
        ylabel="Electrical power [mW]",
    )


# =========================================================
# Main
# =========================================================

def main() -> None:
    sum_csv_path = ask_csv("Select SUM CSV: efficiency_sum.csv")
    if sum_csv_path is None:
        print("[INFO] No SUM CSV selected.")
        return

    mean_csv_path = ask_csv("Select MEAN CSV: efficiency_mean.csv")
    if mean_csv_path is None:
        print("[INFO] No MEAN CSV selected.")
        return

    out_dir = sum_csv_path.parent / OUT_DIR_NAME
    out_dir.mkdir(exist_ok=True)

    try:
        run_plot(
            csv_path=sum_csv_path,
            x_col=SUM_X_COL,
            prefix="sum",
            out_dir=out_dir,
        )
    except Exception as exc:
        print(f"[ERROR] SUM plot failed: {exc}")

    print("")
    print("#" * 70)
    print("")

    try:
        run_plot(
            csv_path=mean_csv_path,
            x_col=MEAN_X_COL,
            prefix="mean",
            out_dir=out_dir,
        )
    except Exception as exc:
        print(f"[ERROR] MEAN plot failed: {exc}")

    print("")
    print("[DONE]")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()