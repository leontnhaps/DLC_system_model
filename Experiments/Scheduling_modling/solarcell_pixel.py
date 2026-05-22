#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from scipy.optimize import least_squares


# =========================================================
# CSV 파일 직접 선택: File Dialog 방식
# =========================================================
def select_csv_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        file_path = filedialog.askopenfilename(
            title="Select solarcell_points.csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
    finally:
        root.destroy()

    if not file_path:
        return None

    return Path(file_path)


# =========================================================
# 한 행에서 사각형 4변 평균 길이 계산
# =========================================================
def calc_rectangle_side_mean(row):
    points = []

    for i in range(1, 5):
        x = row.get(f"x{i}")
        y = row.get(f"y{i}")

        if pd.isna(x) or pd.isna(y):
            return np.nan, [np.nan, np.nan, np.nan, np.nan]

        points.append((float(x), float(y)))

    side_lengths = []

    for p1, p2 in zip(points, points[1:] + points[:1]):
        x1, y1 = p1
        x2, y2 = p2
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        side_lengths.append(length)

    mean_side = float(np.mean(side_lengths))

    return mean_side, side_lengths


# =========================================================
# 성능 지표
# =========================================================
def calc_r2_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return float(r2), float(rmse)


# =========================================================
# Regression Models
# =========================================================
def model_linear(x, p):
    """
    y = ax + b
    p = [a, b]
    """
    a, b = p
    return a * x + b


def model_quadratic(x, p):
    """
    y = ax^2 + bx + c
    p = [a, b, c]
    """
    a, b, c = p
    return a * x**2 + b * x + c


def model_shifted_quadratic_decreasing(x, p):
    """
    y = a(x - h)^2 + c

    h를 x 데이터 범위 오른쪽에 두면,
    데이터 구간 안에서는 감소하는 2차 곡선 형태가 됨.
    p = [a, h, c]
    """
    a, h, c = p
    return a * (x - h) ** 2 + c


def model_exponential_decay(x, p):
    """
    y = A * exp(-k*x) + c
    p = [A, k, c]
    """
    A, k, c = p
    return A * np.exp(-k * x) + c


# =========================================================
# Fitting
# =========================================================
def fit_model(name, model_func, x, y, p0, lower, upper):
    p0 = np.asarray(p0, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    eps = 1e-9
    p0 = np.minimum(np.maximum(p0, lower + eps), upper - eps)

    res = least_squares(
        lambda p: model_func(x, p) - y,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=10.0,
        max_nfev=30000,
    )

    y_pred = model_func(x, res.x)
    r2, rmse = calc_r2_rmse(y, y_pred)

    return {
        "name": name,
        "params": res.x,
        "success": res.success,
        "y_pred": y_pred,
        "r2": r2,
        "rmse": rmse,
    }


def fit_all_models(x, y):
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_range = max(y_max - y_min, 1e-6)

    # 1) Linear 초기값
    lin_a0, lin_b0 = np.polyfit(x, y, 1)

    # 2) Quadratic 초기값
    quad_a0, quad_b0, quad_c0 = np.polyfit(x, y, 2)

    fits = []

    # Linear: y = ax + b
    fits.append(
        fit_model(
            name="linear",
            model_func=model_linear,
            x=x,
            y=y,
            p0=[lin_a0, lin_b0],
            lower=[-1e5, -1e5],
            upper=[1e5, 1e5],
        )
    )

    # Quadratic: y = ax^2 + bx + c
    fits.append(
        fit_model(
            name="quadratic",
            model_func=model_quadratic,
            x=x,
            y=y,
            p0=[quad_a0, quad_b0, quad_c0],
            lower=[-1e5, -1e5, -1e5],
            upper=[1e5, 1e5, 1e5],
        )
    )

    # Shifted quadratic decreasing: y = a(x-h)^2 + c
    # h >= x_max로 제한해서 데이터 구간에서 감소 형태 유도
    fits.append(
        fit_model(
            name="shifted_quadratic_decreasing",
            model_func=model_shifted_quadratic_decreasing,
            x=x,
            y=y,
            p0=[y_range / max((x_max - x_min) ** 2, 1e-6), x_max + 1.0, y_min],
            lower=[0.0, x_max, -1e5],
            upper=[1e5, x_max + 100.0, 1e5],
        )
    )

    # Exponential decay: y = A exp(-kx) + c
    fits.append(
        fit_model(
            name="exponential_decay",
            model_func=model_exponential_decay,
            x=x,
            y=y,
            p0=[y_range, 0.5, y_min],
            lower=[0.0, 0.0, -1e5],
            upper=[1e5, 100.0, 1e5],
        )
    )

    return fits


def predict_by_fit(fit, x):
    name = fit["name"]
    p = fit["params"]

    if name == "linear":
        return model_linear(x, p)

    if name == "quadratic":
        return model_quadratic(x, p)

    if name == "shifted_quadratic_decreasing":
        return model_shifted_quadratic_decreasing(x, p)

    if name == "exponential_decay":
        return model_exponential_decay(x, p)

    raise ValueError(f"Unknown model: {name}")


def equation_text(fit):
    name = fit["name"]
    p = fit["params"]

    if name == "linear":
        a, b = p
        return f"y = {a:.6f}x + {b:.6f}"

    if name == "quadratic":
        a, b, c = p
        return f"y = {a:.6f}x² + {b:.6f}x + {c:.6f}"

    if name == "shifted_quadratic_decreasing":
        a, h, c = p
        return f"y = {a:.6f}(x - {h:.6f})² + {c:.6f}"

    if name == "exponential_decay":
        A, k, c = p
        return f"y = {A:.6f} exp(-{k:.6f}x) + {c:.6f}"

    return ""


def latex_text(fit):
    name = fit["name"]
    p = fit["params"]

    if name == "linear":
        a, b = p
        return rf"y = {a:.6f}x + {b:.6f}"

    if name == "quadratic":
        a, b, c = p
        return rf"y = {a:.6f}x^2 + {b:.6f}x + {c:.6f}"

    if name == "shifted_quadratic_decreasing":
        a, h, c = p
        return rf"y = {a:.6f}(x - {h:.6f})^2 + {c:.6f}"

    if name == "exponential_decay":
        A, k, c = p
        return rf"y = {A:.6f}e^{{-{k:.6f}x}} + {c:.6f}"

    return ""


# =========================================================
# Main
# =========================================================
def main():
    csv_path = select_csv_file()

    if csv_path is None:
        print("[INFO] CSV 파일이 선택되지 않았습니다.")
        return

    print(f"[INFO] Selected CSV: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    group_id = 0
    in_group = False
    detail_rows = []

    # =========================================================
    # 1. skipped 기준 그룹 분리 + 사각형 한 변 평균 계산
    # =========================================================
    for idx, row in df.iterrows():
        status = str(row.get("status", "")).strip().lower()

        if status == "skipped":
            if in_group:
                group_id += 1
                in_group = False
            continue

        point_count = row.get("point_count", 0)

        try:
            point_count = int(point_count)
        except:
            continue

        if point_count != 4:
            continue

        if not in_group:
            in_group = True

        mean_side, sides = calc_rectangle_side_mean(row)

        if not np.isfinite(mean_side):
            continue

        detail_rows.append({
            "group": group_id + 1,
            "filename": row.get("filename", ""),
            "side_1_px": sides[0],
            "side_2_px": sides[1],
            "side_3_px": sides[2],
            "side_4_px": sides[3],
            "mean_side_px": mean_side,
        })

    detail_df = pd.DataFrame(detail_rows)

    if detail_df.empty:
        print("[ERROR] 계산 가능한 사각형 데이터가 없습니다.")
        return

    # =========================================================
    # 2. 그룹별 평균 계산
    # =========================================================
    summary_df = (
        detail_df
        .groupby("group", as_index=False)
        .agg(
            image_count=("filename", "count"),
            mean_side_px=("mean_side_px", "mean"),
            std_side_px=("mean_side_px", "std"),
            min_side_px=("mean_side_px", "min"),
            max_side_px=("mean_side_px", "max"),
        )
    )

    summary_df["std_side_px"] = summary_df["std_side_px"].fillna(0.0)

    # =========================================================
    # 3. 그룹별 n 부여: 15, 14, ..., 5
    #    x = 0.45 * n [m]
    #    y = mean_side_px
    # =========================================================
    summary_df = summary_df.sort_values("group").reset_index(drop=True)

    summary_df["n"] = 15 - summary_df.index
    summary_df["x_m"] = 0.45 * summary_df["n"]
    summary_df["y_mean_side_px"] = summary_df["mean_side_px"]

    if summary_df["n"].iloc[-1] != 5:
        print(
            "[WARN] 마지막 그룹의 n이 5가 아닙니다. "
            f"현재 마지막 n = {summary_df['n'].iloc[-1]}"
        )
        print("[WARN] 그룹 개수가 11개인지 확인하세요.")

    x = summary_df["x_m"].to_numpy(dtype=float)
    y = summary_df["y_mean_side_px"].to_numpy(dtype=float)

    # =========================================================
    # 4. 여러 모델 회귀
    # =========================================================
    fits = fit_all_models(x, y)

    for fit in fits:
        pred_col = f"y_pred_{fit['name']}"
        res_col = f"residual_{fit['name']}"
        summary_df[pred_col] = fit["y_pred"]
        summary_df[res_col] = y - fit["y_pred"]

    best_fit = min(fits, key=lambda f: f["rmse"])

    # =========================================================
    # 5. 출력
    # =========================================================
    print("\n==============================")
    print("Group-wise rectangle side length")
    print("==============================")

    for _, row in summary_df.iterrows():
        print(
            f"Group {int(row['group']):02d} | "
            f"n={int(row['n']):02d} | "
            f"x={row['x_m']:.2f} m | "
            f"N={int(row['image_count'])} | "
            f"mean side={row['mean_side_px']:.3f} px"
        )

    print("\n==============================")
    print("Regression Results")
    print("==============================")

    for fit in fits:
        print(f"\nModel   : {fit['name']}")
        print(f"success : {fit['success']}")
        print(f"Equation: {equation_text(fit)}")
        print(f"R²      : {fit['r2']:.6f}")
        print(f"RMSE    : {fit['rmse']:.6f} px")
        print(f"LaTeX   : {latex_text(fit)}")

    print("\n==============================")
    print("Best Model")
    print("==============================")
    print(f"Best model by RMSE: {best_fit['name']}")
    print(f"Equation: {equation_text(best_fit)}")
    print(f"R²      : {best_fit['r2']:.6f}")
    print(f"RMSE    : {best_fit['rmse']:.6f} px")

    # =========================================================
    # 6. 저장
    # =========================================================
    out_dir = csv_path.parent

    detail_out = out_dir / "rectangle_side_length_detail.csv"
    summary_out = out_dir / "rectangle_side_length_summary_with_multi_regression.csv"
    regression_txt_out = out_dir / "rectangle_side_length_multi_regression.txt"
    plot_out = out_dir / "rectangle_side_length_multi_regression_plot.png"

    detail_df.to_csv(detail_out, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

    with open(regression_txt_out, "w", encoding="utf-8-sig") as f:
        f.write("=" * 70 + "\n")
        f.write("Rectangle Side Length Multi-Regression\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Input CSV : {csv_path}\n\n")
        f.write("Group setting:\n")
        f.write("  Group 1 starts from n = 15\n")
        f.write("  Last group is expected to be n = 5\n")
        f.write("  x = 0.45 * n [m]\n")
        f.write("  y = group mean side length [px]\n\n")

        f.write("[Regression Models]\n\n")

        for fit in fits:
            f.write("-" * 70 + "\n")
            f.write(f"Model   : {fit['name']}\n")
            f.write(f"success : {fit['success']}\n")
            f.write(f"Equation: {equation_text(fit)}\n")
            f.write(f"LaTeX   : {latex_text(fit)}\n")
            f.write(f"R^2     : {fit['r2']:.10f}\n")
            f.write(f"RMSE    : {fit['rmse']:.10f} px\n")
            f.write(f"params  : {fit['params']}\n\n")

        f.write("[Best model by RMSE]\n")
        f.write(f"Model   : {best_fit['name']}\n")
        f.write(f"Equation: {equation_text(best_fit)}\n")
        f.write(f"LaTeX   : {latex_text(best_fit)}\n")
        f.write(f"R^2     : {best_fit['r2']:.10f}\n")
        f.write(f"RMSE    : {best_fit['rmse']:.10f} px\n\n")

        f.write("[Data]\n")
        f.write(summary_df.to_string(index=False))

    # =========================================================
    # 7. Plot 저장
    # =========================================================
    x_line = np.linspace(np.min(x), np.max(x), 500)

    plt.figure(figsize=(9.5, 6.3))

    plt.scatter(
        x,
        y,
        s=75,
        label="Group mean side length"
    )

    for _, row in summary_df.iterrows():
        plt.text(
            row["x_m"],
            row["y_mean_side_px"],
            f" n={int(row['n'])}",
            fontsize=9,
            ha="left",
            va="bottom"
        )

    for fit in fits:
        y_line = predict_by_fit(fit, x_line)

        if fit["name"] == "linear":
            linestyle = ":"
            linewidth = 2.2
            label = f"Linear | R²={fit['r2']:.4f}, RMSE={fit['rmse']:.2f}"

        elif fit["name"] == "quadratic":
            linestyle = "-"
            linewidth = 2.2
            label = f"Quadratic | R²={fit['r2']:.4f}, RMSE={fit['rmse']:.2f}"

        elif fit["name"] == "shifted_quadratic_decreasing":
            linestyle = "-."
            linewidth = 2.4
            label = f"Shifted quadratic | R²={fit['r2']:.4f}, RMSE={fit['rmse']:.2f}"

        elif fit["name"] == "exponential_decay":
            linestyle = "--"
            linewidth = 2.6
            label = f"Exponential decay | R²={fit['r2']:.4f}, RMSE={fit['rmse']:.2f}"

        else:
            linestyle = "-"
            linewidth = 2.0
            label = fit["name"]

        plt.plot(
            x_line,
            y_line,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label
        )

    plt.xlabel("Distance x = 0.45n [m]")
    plt.ylabel("Mean side length [px]")
    plt.title("Regression of Solar Cell Side Length vs Distance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_out, dpi=170)
    plt.close()

    print("\n[SAVE]")
    print(f"Detail CSV      : {detail_out}")
    print(f"Summary CSV     : {summary_out}")
    print(f"Regression TXT  : {regression_txt_out}")
    print(f"Plot PNG        : {plot_out}")


if __name__ == "__main__":
    main()