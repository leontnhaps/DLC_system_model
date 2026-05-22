#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add vol_load values to solarcell_intensity_cal.csv by matching filename.

Input:
    1) solarcell_intensity_cal.csv
    2) optional source CSV containing filename and vol_load

Output:
    intensity_cal_vol.csv

Matching rule:
    - Exact filename match first
    - Basename-only matching is used, so full paths in filename are okay.

This file also contains an embedded vol_load dictionary extracted from the CSV
provided during code preparation. If you do not select a source CSV, the embedded
mapping is used.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import pandas as pd


OUTPUT_CSV_NAME = "intensity_cal_vol.csv"

# Embedded fallback values from the provided CSV.
VOL_LOAD_BY_FILENAME = {
    'snap_20260515_194600.jpg': 1.042,
    'snap_20260515_194710.jpg': 0.1,
    'snap_20260515_194814.jpg': 0.116,
    'snap_20260515_194905.jpg': 0.072,
    'snap_20260515_194957.jpg': 0.266,
    'snap_20260515_195346.jpg': 0.248,
    'snap_20260515_195605.jpg': 0.655,
    'snap_20260515_195738.jpg': 0.93,
    'snap_20260515_200001.jpg': 0.06,
    'snap_20260515_200148.jpg': 0.886,
    'snap_20260515_200219.jpg': 0.023,
    'snap_20260515_200520.jpg': 0.09,
    'snap_20260515_200635.jpg': 0.775,
    'snap_20260515_200735.jpg': 0.755,
    'snap_20260515_200938.jpg': 0.194,
    'snap_20260515_201013.jpg': 0.073,
    'snap_20260515_201433.jpg': 1.248,
    'snap_20260515_201550.jpg': 0.071,
    'snap_20260515_201630.jpg': 0.661,
    'snap_20260515_201744.jpg': 0.388,
    'snap_20260515_201858.jpg': 0.674,
    'snap_20260515_202144.jpg': 0.108,
    'snap_20260515_202240.jpg': 0.546,
    'snap_20260515_202616.jpg': 1.118,
    'snap_20260515_203054.jpg': 0.547,
    'snap_20260515_203612.jpg': 0.886,
    'snap_20260515_203704.jpg': 0.238,
    'snap_20260515_203737.jpg': 0.054,
    'snap_20260515_204242.jpg': 0.349,
    'snap_20260515_204803.jpg': 0.74,
    'snap_20260515_205028.jpg': 0.088,
    'snap_20260515_205208.jpg': 0.025,
    'snap_20260515_205550.jpg': 0.079,
    'snap_20260515_205738.jpg': 0.02,
    'snap_20260515_205846.jpg': 0.113,
    'snap_20260515_210326.jpg': 0.947,
    'snap_20260515_210645.jpg': 0.825,
    'snap_20260515_210710.jpg': 0.026,
    'snap_20260515_211010.jpg': 0.076,
    'snap_20260515_211318.jpg': 0.045,
    'snap_20260515_211636.jpg': 0.8,
    'snap_20260515_211733.jpg': 0.135,
    'snap_20260515_211923.jpg': 0.021,
    'snap_20260515_212150.jpg': 0.073,
    'snap_20260515_212723.jpg': 0.207,
    'snap_20260515_212748.jpg': 0.02,
    'snap_20260515_212808.jpg': 0.037,
}


def select_csv_file(title: str) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return Path(path) if path else None


def normalize_filename(value) -> str:
    """Normalize a CSV filename cell for matching."""
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("\\", "/")
    if not text:
        return ""
    return Path(text).name


def build_mapping_from_csv(csv_path: Path) -> dict[str, float]:
    """Read filename/vol_load pairs from a CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required = ["filename", "vol_load"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Source CSV missing required columns: {missing}")

    mapping: dict[str, float] = {}
    duplicate_count = 0

    for _, row in df.iterrows():
        name = normalize_filename(row.get("filename"))
        if not name:
            continue

        vol = pd.to_numeric(row.get("vol_load"), errors="coerce")
        if pd.isna(vol):
            continue

        if name in mapping:
            duplicate_count += 1

        mapping[name] = float(vol)

    if duplicate_count:
        print(f"[WARN] Duplicate filename rows found in voltage source: {duplicate_count}. Last value was used.")

    return mapping


def build_embedded_mapping() -> dict[str, float]:
    return {normalize_filename(k): float(v) for k, v in VOL_LOAD_BY_FILENAME.items()}


def add_vol_load(intensity_df: pd.DataFrame, mapping: dict[str, float]) -> pd.DataFrame:
    df = intensity_df.copy()

    if "filename" not in df.columns:
        raise ValueError("Intensity CSV missing required column: filename")

    # Replace old vol_load if it exists, because this script is the voltage merge step.
    if "vol_load" in df.columns:
        df = df.drop(columns=["vol_load"])

    df["_filename_key"] = df["filename"].map(normalize_filename)
    df["vol_load"] = df["_filename_key"].map(mapping)
    df["vol_load_match"] = df["vol_load"].notna()

    df = df.drop(columns=["_filename_key"])
    return df


def main() -> None:
    intensity_path = select_csv_file("Select solarcell_intensity_cal.csv")
    if intensity_path is None:
        print("[INFO] No intensity CSV selected.")
        return

    if not intensity_path.exists():
        print(f"[ERROR] Intensity CSV not found: {intensity_path}")
        return

    source_path = select_csv_file(
        "Select source CSV with filename and vol_load. Cancel to use embedded values."
    )

    print("=" * 70)
    print("[INFO] Add vol_load to intensity CSV")
    print(f"[INFO] Intensity CSV: {intensity_path}")

    try:
        if source_path is not None:
            print(f"[INFO] Voltage source: {source_path}")
            mapping = build_mapping_from_csv(source_path)
        else:
            print("[INFO] Voltage source: embedded dictionary")
            mapping = build_embedded_mapping()
    except Exception as exc:
        print(f"[ERROR] Failed to build voltage mapping: {exc}")
        return

    if not mapping:
        print("[ERROR] No valid vol_load mapping was found.")
        return

    try:
        intensity_df = pd.read_csv(intensity_path, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[ERROR] Failed to read intensity CSV: {exc}")
        return

    try:
        out_df = add_vol_load(intensity_df, mapping)
    except Exception as exc:
        print(f"[ERROR] Failed to add vol_load: {exc}")
        return

    output_path = intensity_path.parent / OUTPUT_CSV_NAME
    try:
        out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[ERROR] Failed to save output CSV: {exc}")
        return

    total = len(out_df)
    matched = int(out_df["vol_load_match"].sum())

    print("")
    print("[DONE]")
    print(f"Total rows      : {total}")
    print(f"Matched vol_load: {matched}")
    print(f"Unmatched rows  : {total - matched}")
    print(f"Output CSV      : {output_path}")


if __name__ == "__main__":
    main()
