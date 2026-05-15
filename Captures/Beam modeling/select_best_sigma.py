#!/usr/bin/env python3
"""
select_best_sigma.py

Purpose
-------
Read beam_parameters_all.csv and select one representative beam-shape row
per x_index for estimating sigma_x, sigma_y, w_x, and w_y.

Recommended use:
    python select_best_sigma.py

Optional:
    python select_best_sigma.py --input beam_parameters_all.csv --sat-limit 0.005 --i-peak-min 50 --exclude-x 1

Selection rule
--------------
For each x_index:
1. valid == True
2. x_index not in exclude list
3. sigma_x/sigma_y/w_x/w_y exist
4. strict candidates:
   - saturation_ratio <= sat_limit
   - I_peak >= i_peak_min
5. choose the strict candidate with the largest shutter_us.
6. If no strict candidate exists, choose the valid row with the minimum saturation_ratio
   and mark it as relaxed.

Outputs
-------
- beam_parameters_sigma_best.csv
- beam_sigma_summary.csv
- selected_sigma_vs_distance.png
- selected_w_vs_distance.png
- selected_quality_vs_distance.png
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SAT_LIMIT = 0.005
DEFAULT_I_PEAK_MIN = 50.0
DEFAULT_EXCLUDE_X = "1"
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select representative sigma rows from beam_parameters_all.csv"
    )
    parser.add_argument(
        "--input",
        default=str(SCRIPT_DIR / "beam_parameters_all.csv"),
        help="Input CSV path. Default: beam_parameters_all.csv next to this script.",
    )
    parser.add_argument(
        "--output-best",
        default="beam_parameters_sigma_best.csv",
        help="Output CSV for selected best rows.",
    )
    parser.add_argument(
        "--output-summary",
        default="beam_sigma_summary.csv",
        help="Output CSV for representative sigma/w summary.",
    )
    parser.add_argument(
        "--sat-limit",
        type=float,
        default=DEFAULT_SAT_LIMIT,
        help=f"Saturation ratio limit for strict selection. Default: {DEFAULT_SAT_LIMIT}",
    )
    parser.add_argument(
        "--i-peak-min",
        type=float,
        default=DEFAULT_I_PEAK_MIN,
        help=f"Minimum I_peak for strict selection. Default: {DEFAULT_I_PEAK_MIN}",
    )
    parser.add_argument(
        "--exclude-x",
        default=DEFAULT_EXCLUDE_X,
        help='Comma-separated x_index values to exclude. Default: "1"',
    )
    parser.add_argument(
        "--out-dir",
        default=str(SCRIPT_DIR),
        help="Directory for output CSVs and plots. Default: this script directory.",
    )
    return parser.parse_args()


def parse_exclude_x(text):
    if text is None or str(text).strip() == "":
        return set()

    values = set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError:
            print(f"[WARN] Ignoring invalid exclude-x token: {token}")
    return values


def to_bool_series(series):
    """Robust bool conversion for valid column loaded from CSV."""
    if series.dtype == bool:
        return series

    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes", "y", "ok"]
    )


def load_all_results(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "x_index",
        "distance_m",
        "pair_index",
        "shutter_us",
        "sigma_x",
        "sigma_y",
        "w_x",
        "w_y",
        "D_x",
        "D_y",
        "I_peak",
        "saturation_ratio",
        "valid",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )

    numeric_cols = [
        "x_index",
        "distance_m",
        "pair_index",
        "shutter_us",
        "sigma_x",
        "sigma_y",
        "w_x",
        "w_y",
        "D_x",
        "D_y",
        "I_peak",
        "saturation_ratio",
        "total_intensity",
        "FWHM_x",
        "FWHM_y",
        "R_asym",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["valid_bool"] = to_bool_series(df["valid"])

    # Remove exact duplicate rows if the batch script was run more than once.
    duplicate_keys = [
        col
        for col in ["x_index", "pair_index", "shutter_us", "on_path", "off_path"]
        if col in df.columns
    ]
    if duplicate_keys:
        before = len(df)
        df = df.drop_duplicates(subset=duplicate_keys, keep="last").copy()
        after = len(df)
        if after < before:
            print(f"[INFO] Removed duplicate rows: {before - after}")

    return df


def select_best_sigma_rows(df_all, sat_limit, i_peak_min, exclude_x):
    """
    Select one representative row per x_index.

    Strict rule:
        valid
        saturation_ratio <= sat_limit
        I_peak >= i_peak_min
        choose max shutter_us

    Relaxed fallback:
        choose minimum saturation_ratio among valid rows.
    """
    best_rows = []
    selection_log = []

    grouped = df_all.groupby("x_index", sort=True)

    for x_index, group in grouped:
        try:
            x_int = int(x_index)
        except Exception:
            continue

        if x_int in exclude_x:
            selection_log.append(
                {
                    "x_index": x_int,
                    "status": "excluded",
                    "reason": "x_index is in exclude list",
                }
            )
            continue

        g = group.copy()

        valid_mask = (
            g["valid_bool"]
            & np.isfinite(g["sigma_x"])
            & np.isfinite(g["sigma_y"])
            & np.isfinite(g["w_x"])
            & np.isfinite(g["w_y"])
            & np.isfinite(g["saturation_ratio"])
            & np.isfinite(g["I_peak"])
            & np.isfinite(g["shutter_us"])
        )

        g_valid = g.loc[valid_mask].copy()

        if g_valid.empty:
            selection_log.append(
                {
                    "x_index": x_int,
                    "status": "failed",
                    "reason": "no valid rows with finite sigma/w/quality columns",
                }
            )
            continue

        strict = g_valid.loc[
            (g_valid["saturation_ratio"] <= sat_limit)
            & (g_valid["I_peak"] >= i_peak_min)
        ].copy()

        if not strict.empty:
            strict = strict.sort_values(
                ["shutter_us", "I_peak", "total_intensity" if "total_intensity" in strict.columns else "I_peak"],
                ascending=[False, False, False],
            )
            row = strict.iloc[0].copy()
            row["sigma_selection"] = "strict"
            row["sigma_selection_reason"] = (
                f"sat<={sat_limit}, I_peak>={i_peak_min}, selected max shutter"
            )
        else:
            relaxed = g_valid.sort_values(
                ["saturation_ratio", "I_peak", "shutter_us"],
                ascending=[True, False, False],
            )
            row = relaxed.iloc[0].copy()
            row["sigma_selection"] = "relaxed"
            row["sigma_selection_reason"] = (
                "no strict candidate; selected minimum saturation valid row"
            )

        best_rows.append(row)

        selection_log.append(
            {
                "x_index": x_int,
                "status": str(row["sigma_selection"]),
                "distance_m": row.get("distance_m", np.nan),
                "shutter_us": row.get("shutter_us", np.nan),
                "I_peak": row.get("I_peak", np.nan),
                "saturation_ratio": row.get("saturation_ratio", np.nan),
                "sigma_x": row.get("sigma_x", np.nan),
                "sigma_y": row.get("sigma_y", np.nan),
                "w_x": row.get("w_x", np.nan),
                "w_y": row.get("w_y", np.nan),
                "reason": str(row["sigma_selection_reason"]),
            }
        )

    if not best_rows:
        return pd.DataFrame(), pd.DataFrame(selection_log)

    df_best = pd.DataFrame(best_rows)
    df_best = df_best.sort_values("x_index").reset_index(drop=True)

    return df_best, pd.DataFrame(selection_log)


def build_summary(df_best, sat_limit, i_peak_min, exclude_x):
    if df_best.empty:
        return pd.DataFrame()

    summary = {
        "n_selected": int(len(df_best)),
        "n_strict": int((df_best["sigma_selection"] == "strict").sum()),
        "n_relaxed": int((df_best["sigma_selection"] == "relaxed").sum()),
        "excluded_x": ",".join(map(str, sorted(exclude_x))),
        "sat_limit": float(sat_limit),
        "i_peak_min": float(i_peak_min),
    }

    stats_cols = ["sigma_x", "sigma_y", "w_x", "w_y", "D_x", "D_y"]

    for col in stats_cols:
        values = pd.to_numeric(df_best[col], errors="coerce").dropna()
        summary[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
        summary[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        summary[f"{col}_median"] = float(values.median()) if len(values) else np.nan
        summary[f"{col}_min"] = float(values.min()) if len(values) else np.nan
        summary[f"{col}_max"] = float(values.max()) if len(values) else np.nan

        mean_val = summary[f"{col}_mean"]
        std_val = summary[f"{col}_std"]
        if np.isfinite(mean_val) and mean_val != 0 and np.isfinite(std_val):
            summary[f"{col}_cv_percent"] = float(100.0 * std_val / mean_val)
        else:
            summary[f"{col}_cv_percent"] = np.nan

    # Model-ready values
    summary["model_sigma_x_px"] = summary["sigma_x_mean"]
    summary["model_sigma_y_px"] = summary["sigma_y_mean"]
    summary["model_w_x_px"] = summary["w_x_mean"]
    summary["model_w_y_px"] = summary["w_y_mean"]

    return pd.DataFrame([summary])


def save_plots(df_best, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df_best.empty:
        return

    df = df_best.copy()
    df = df.sort_values("distance_m")

    # Selected sigma vs distance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["distance_m"], df["sigma_x"], "o-", label="sigma_x")
    ax.plot(df["distance_m"], df["sigma_y"], "o-", label="sigma_y")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Selected sigma [px]")
    ax.set_title("Selected sigma vs distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "selected_sigma_vs_distance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVE] {path}")

    # Selected w vs distance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["distance_m"], df["w_x"], "o-", label="w_x = 2 sigma_x")
    ax.plot(df["distance_m"], df["w_y"], "o-", label="w_y = 2 sigma_y")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Selected w [px]")
    ax.set_title("Selected beam radius w vs distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "selected_w_vs_distance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVE] {path}")

    # Quality plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["distance_m"], df["I_peak"], "o-", label="I_peak")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("I_peak [camera a.u.]")
    ax.set_title("Selected I_peak vs distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "selected_quality_vs_distance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVE] {path}")

    # Saturation plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["distance_m"], df["saturation_ratio"], "o-", label="saturation_ratio")
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Saturation ratio")
    ax.set_title("Selected saturation ratio vs distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "selected_saturation_vs_distance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVE] {path}")


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = SCRIPT_DIR / input_path

    output_best = Path(args.output_best)
    if not output_best.is_absolute():
        output_best = out_dir / output_best

    output_summary = Path(args.output_summary)
    if not output_summary.is_absolute():
        output_summary = out_dir / output_summary

    exclude_x = parse_exclude_x(args.exclude_x)

    print("[INFO] Best sigma selector")
    print(f"[INFO] Input CSV: {input_path}")
    print(f"[INFO] Output dir: {out_dir.resolve()}")
    print(f"[INFO] sat_limit={args.sat_limit}")
    print(f"[INFO] i_peak_min={args.i_peak_min}")
    print(f"[INFO] exclude_x={sorted(exclude_x)}")

    df_all = load_all_results(input_path)

    df_best, df_log = select_best_sigma_rows(
        df_all=df_all,
        sat_limit=args.sat_limit,
        i_peak_min=args.i_peak_min,
        exclude_x=exclude_x,
    )

    if df_best.empty:
        print("[ERROR] No best sigma rows were selected.")
        if not df_log.empty:
            log_path = out_dir / "beam_sigma_selection_log.csv"
            df_log.to_csv(log_path, index=False)
            print(f"[SAVE] Selection log: {log_path}")
        return

    df_summary = build_summary(
        df_best=df_best,
        sat_limit=args.sat_limit,
        i_peak_min=args.i_peak_min,
        exclude_x=exclude_x,
    )

    df_best.to_csv(output_best, index=False)
    df_summary.to_csv(output_summary, index=False)

    log_path = out_dir / "beam_sigma_selection_log.csv"
    df_log.to_csv(log_path, index=False)

    save_plots(df_best, out_dir)

    print(f"\n[SAVE] Best rows CSV: {output_best}")
    print(f"[SAVE] Summary CSV: {output_summary}")
    print(f"[SAVE] Selection log: {log_path}")

    print("\n[SELECTED ROWS]")
    display_cols = [
        "x_index",
        "distance_m",
        "shutter_us",
        "I_peak",
        "saturation_ratio",
        "sigma_x",
        "sigma_y",
        "w_x",
        "w_y",
        "sigma_selection",
    ]
    print(df_best[display_cols].to_string(index=False))

    print("\n[SUMMARY]")
    summary_row = df_summary.iloc[0]
    print(f"n_selected      = {summary_row['n_selected']}")
    print(f"sigma_x_mean    = {summary_row['sigma_x_mean']:.4f} px")
    print(f"sigma_y_mean    = {summary_row['sigma_y_mean']:.4f} px")
    print(f"w_x_mean        = {summary_row['w_x_mean']:.4f} px")
    print(f"w_y_mean        = {summary_row['w_y_mean']:.4f} px")
    print(f"sigma_x_cv      = {summary_row['sigma_x_cv_percent']:.2f} %")
    print(f"sigma_y_cv      = {summary_row['sigma_y_cv_percent']:.2f} %")

    print("\n[MODEL FORM]")
    print("I_px(u,v) = exp(-2 * ((u^2 / w_x^2) + (v^2 / w_y^2)))")
    print(f"Use w_x = {summary_row['w_x_mean']:.4f} px")
    print(f"Use w_y = {summary_row['w_y_mean']:.4f} px")


if __name__ == "__main__":
    main()
