from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# User-editable settings
# ============================================================

NODE_COUNT = 4
MONTE_CARLO_RUNS = 100

DISTANCE_MIN_M = 5.0
DISTANCE_MAX_M = 10.0

BASE_TIME_MIN = 60.0          # 1 hour 기준
FRAMES_PER_HOUR = 15          # 1시간당 15 frame

FRAME_TIME_MIN = BASE_TIME_MIN / FRAMES_PER_HOUR
FRAME_TIME_SEC = FRAME_TIME_MIN * 60.0

MAX_SIM_TIME_MIN = 10.0 * 24.0 * 60.0     # 10 days [min]
FRAME_COUNT = int(MAX_SIM_TIME_MIN / FRAME_TIME_MIN)

DEATH_THRESHOLD = 0.125
EPSILON = 1e-9


# ============================================================
# Shaking condition settings
# ============================================================

# Original: 조준오차 없음
# Shaking 0-5px: x/y 방향 각각 -5~+5 px 범위에서 랜덤 offset 생성
SHAKING_CONDITIONS = [
    ("Original", 0.0),
    ("Shaking 0-5px", 5.0),
]


# ============================================================
# Energy-based charging model
# ============================================================

LOAD_RESISTANCE_OHM = 100.0      # [ohm], same as experiment
CHARGING_EFFICIENCY = 5.0        # charging power scale factor처럼 사용 중


# ============================================================
# Li-ion battery-based energy model
# ============================================================

BATTERY_CAPACITY_MAH = 1000.0      # [mAh]
BATTERY_V_FULL = 4.2               # [V]
BATTERY_V_EMPTY = 3.0              # [V]
BATTERY_V_AVG = (BATTERY_V_FULL + BATTERY_V_EMPTY) / 2.0

BATTERY_CAPACITY_AH = BATTERY_CAPACITY_MAH / 1000.0

# Battery energy [J]
# 1000 mAh = 1 Ah, average voltage = 3.6 V
# E = Ah * V * 3600 = 1 * 3.6 * 3600 = 12960 J
STORAGE_ENERGY_CAPACITY_J = (
    BATTERY_CAPACITY_AH * BATTERY_V_AVG * 3600.0
)

# Node power consumption
# 50 mW = 0.05 W
NODE_POWER_CONSUMPTION_W = 0.05

# Normalized battery consumption
CONSUMPTION_PER_FRAME = (
    NODE_POWER_CONSUMPTION_W * FRAME_TIME_SEC
) / STORAGE_ENERGY_CAPACITY_J

CONSUMPTION_PER_SEC = (
    NODE_POWER_CONSUMPTION_W
) / STORAGE_ENERGY_CAPACITY_J


# ============================================================
# Initial condition and random seed
# ============================================================

INITIAL_ENERGY = 1.0             # every node starts from 100%

RANDOM_SEED = 42

USE_DISTANCE_CLAMP_FOR_PIXEL_MODEL = False


# ============================================================
# Laser map parameters
# ============================================================

# The map is centered at the image center.
# Beam parameters from image-based Gaussian fitting.
LASER_MAP_SIZE_PX = 512
LASER_W_U_PX = 118.14
LASER_W_V_PX = 124.51
LASER_PEAK_INTENSITY = 1.0
LASER_APERTURE_RADIUS_PX: Optional[float] = None


# ============================================================
# Output paths
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent

OUTPUT_SUMMARY_CSV_PATH = OUTPUT_DIR / "first_node_death_summary_conditions.csv"
OUTPUT_TRIAL_CSV_PATH = OUTPUT_DIR / "first_node_death_trials_conditions.csv"
OUTPUT_NODE_CSV_PATH = OUTPUT_DIR / "node_conditions_with_shaking.csv"

OUTPUT_PLOT_ORIGINAL_PATH = OUTPUT_DIR / "first_node_death_original.png"
OUTPUT_PLOT_SHAKING_PATH = OUTPUT_DIR / "first_node_death_shaking_0_5px.png"
OUTPUT_PLOT_COMPARE_PATH = OUTPUT_DIR / "first_node_death_compare_conditions.png"

Y_AXIS_CROP_MIN = 3500.0

OUTPUT_PLOT_ORIGINAL_CROPPED_PATH = OUTPUT_DIR / "first_node_death_original_ymin_3500.png"
OUTPUT_PLOT_SHAKING_CROPPED_PATH = OUTPUT_DIR / "first_node_death_shaking_0_5px_ymin_3500.png"
OUTPUT_PLOT_COMPARE_CROPPED_PATH = OUTPUT_DIR / "first_node_death_compare_conditions_ymin_3500.png"


@dataclass
class Node:
    distance_m: float
    initial_energy: float
    energy: float
    pv_side_px: float
    pv_area_px: float
    offset_x_px: float
    offset_y_px: float
    received_intensity_S: float
    predicted_voltage: float


# ============================================================
# PV pixel-size model
# ============================================================

def _distance_for_pixel_model(distance_m: float) -> float:
    if not USE_DISTANCE_CLAMP_FOR_PIXEL_MODEL:
        return distance_m
    return float(np.clip(distance_m, DISTANCE_MIN_M, DISTANCE_MAX_M))


def pv_side_length_px(distance_m: float) -> float:
    d = _distance_for_pixel_model(distance_m)
    return float(574.689046 * np.exp(-0.575443 * d) + 61.848215)


def pv_area_px(distance_m: float) -> float:
    side = pv_side_length_px(distance_m)
    return side ** 2


# ============================================================
# Laser intensity and received intensity
# ============================================================

def compute_laser_intensity_map() -> np.ndarray:
    """
    Build a centered elliptical Gaussian laser intensity map.

        I(u,v) = I0 * exp[-2 * (u^2 / w_u^2 + v^2 / w_v^2)]

    where:
        w_u = 118.14 px
        w_v = 124.51 px
    """
    size = int(LASER_MAP_SIZE_PX)
    axis = np.arange(size, dtype=float) - (size - 1) / 2.0
    uu, vv = np.meshgrid(axis, axis)

    intensity = LASER_PEAK_INTENSITY * np.exp(
        -2.0 * (
            (uu ** 2) / (LASER_W_U_PX ** 2)
            + (vv ** 2) / (LASER_W_V_PX ** 2)
        )
    )

    if LASER_APERTURE_RADIUS_PX is not None:
        radius_sq = uu ** 2 + vv ** 2
        aperture_sq = LASER_APERTURE_RADIUS_PX ** 2
        intensity = np.where(radius_sq <= aperture_sq, intensity, 0.0)

    return intensity


def compute_received_intensity_S(
    distance_m: float,
    pv_side_px: float,
    intensity_map: Optional[np.ndarray] = None,
    offset_x_px: float = 0.0,
    offset_y_px: float = 0.0,
) -> float:
    """
    Compute raw S_i.

    Original:
        offset_x_px = 0, offset_y_px = 0

    Shaking 0-5px:
        PV-cell center is shifted from the laser center by offset_x/y.
    """
    if intensity_map is None:
        intensity_map = compute_laser_intensity_map()

    _ = distance_m

    map_h, map_w = intensity_map.shape

    laser_center_x = (map_w - 1) / 2.0
    laser_center_y = (map_h - 1) / 2.0

    # Shaking 적용: PV 중심을 laser 중심에서 offset만큼 이동
    pv_center_x = laser_center_x + offset_x_px
    pv_center_y = laser_center_y + offset_y_px

    side_px_int = max(1, int(round(pv_side_px)))
    x0 = int(round(pv_center_x - side_px_int / 2.0))
    y0 = int(round(pv_center_y - side_px_int / 2.0))
    x1 = x0 + side_px_int
    y1 = y0 + side_px_int

    x0_clip = max(0, x0)
    y0_clip = max(0, y0)
    x1_clip = min(map_w, x1)
    y1_clip = min(map_h, y1)

    if x0_clip >= x1_clip or y0_clip >= y1_clip:
        return 0.0

    pv_laser_intersection = intensity_map[y0_clip:y1_clip, x0_clip:x1_clip]
    received_sum = float(np.sum(pv_laser_intersection))

    # Divide by the modeled PV area, not by the clipped image area.
    return received_sum / max(pv_side_px ** 2, EPSILON)


def predict_charging_voltage(raw_S: float) -> float:
    """
    Shifted quadratic regression using raw S_i.

    Negative predictions are treated as no effective charging voltage
    and clipped to zero.
    """
    voltage = 1.5128 * (raw_S - 0.0131) ** 2 - 0.0275
    return max(0.0, float(voltage))


# ============================================================
# Node generation
# ============================================================

def generate_trial_distances(rng: np.random.Generator) -> np.ndarray:
    """
    Generate node distances for one Monte Carlo trial.

    These distances are shared by all shaking conditions and all schemes
    inside the same trial.
    """
    return rng.uniform(DISTANCE_MIN_M, DISTANCE_MAX_M, size=NODE_COUNT).astype(float)


def generate_pointing_offsets(
    rng: np.random.Generator,
    max_offset_px: float,
) -> np.ndarray:
    """
    Generate pointing offsets.

    max_offset_px = 0:
        Original condition, all offsets are zero.

    max_offset_px = 5:
        offset_x and offset_y are sampled from U(-5, +5) px.

    Important:
        Same offsets are shared by all schemes inside the same trial
        and same condition.
    """
    if max_offset_px <= 0.0:
        return np.zeros((NODE_COUNT, 2), dtype=float)

    return rng.uniform(
        low=-max_offset_px,
        high=max_offset_px,
        size=(NODE_COUNT, 2),
    ).astype(float)


def generate_nodes_from_geometry(
    distances_m: np.ndarray,
    offsets_xy_px: np.ndarray,
    intensity_map: Optional[np.ndarray] = None,
) -> list[Node]:
    """
    Generate nodes using fixed distances and fixed pointing offsets.

    This guarantees fair comparison:
    all schemes receive the same nodes inside each trial and condition.
    """
    if intensity_map is None:
        intensity_map = compute_laser_intensity_map()

    nodes: list[Node] = []

    for idx in range(NODE_COUNT):
        distance_m = float(distances_m[idx])
        offset_x_px = float(offsets_xy_px[idx, 0])
        offset_y_px = float(offsets_xy_px[idx, 1])

        initial_energy = INITIAL_ENERGY

        side_px = pv_side_length_px(distance_m)
        area_px = side_px ** 2

        raw_S = compute_received_intensity_S(
            distance_m=distance_m,
            pv_side_px=side_px,
            intensity_map=intensity_map,
            offset_x_px=offset_x_px,
            offset_y_px=offset_y_px,
        )

        voltage = predict_charging_voltage(raw_S)

        nodes.append(
            Node(
                distance_m=distance_m,
                initial_energy=initial_energy,
                energy=initial_energy,
                pv_side_px=side_px,
                pv_area_px=area_px,
                offset_x_px=offset_x_px,
                offset_y_px=offset_y_px,
                received_intensity_S=raw_S,
                predicted_voltage=voltage,
            )
        )

    return nodes


# ============================================================
# Battery coefficients
# ============================================================

def compute_battery_state_3bit(energies: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(8.0 * energies).astype(int), 0, 7)


def compute_battery_coefficient_B(
    energies: np.ndarray,
    alive_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute B_i(k) from the current energy state.

    The paper-style notation can use b_i(k-1), but for this frame simulation
    the current frame state E_i(k) is used.
    """
    coeff = np.zeros_like(energies, dtype=float)
    alive_count = int(np.sum(alive_mask))
    if alive_count == 0:
        return coeff

    d_bits = compute_battery_state_3bit(energies)
    battery_shortage = (8.0 - d_bits) / 8.0
    shortage_alive = battery_shortage[alive_mask]
    shortage_sum = float(np.sum(shortage_alive))

    if shortage_sum <= EPSILON:
        coeff[alive_mask] = 1.0 / alive_count
    else:
        coeff[alive_mask] = shortage_alive / shortage_sum

    return coeff


def compute_charging_efficiency_coefficient_C(
    voltages: np.ndarray,
    alive_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute charging-efficiency coefficient C_i.

        C_i = (1 / (V_i^2 + eps)) / sum_j(1 / (V_j^2 + eps))

    Lower voltage means lower charging power, so the corresponding node receives
    a larger charging-efficiency coefficient.
    """
    coeff = np.zeros_like(voltages, dtype=float)
    alive_count = int(np.sum(alive_mask))
    if alive_count == 0:
        return coeff

    voltage_sq_alive = voltages[alive_mask] ** 2
    inverse_voltage_sq = 1.0 / (voltage_sq_alive + EPSILON)
    inverse_sum = float(np.sum(inverse_voltage_sq))

    if inverse_sum <= EPSILON:
        coeff[alive_mask] = 1.0 / alive_count
    else:
        coeff[alive_mask] = inverse_voltage_sq / inverse_sum

    return coeff


# ============================================================
# Time allocation schemes
# ============================================================

def allocate_time_no_charging(alive_mask: np.ndarray) -> np.ndarray:
    """
    No charging baseline.

    No node receives any charging time.
    """
    return np.zeros_like(alive_mask, dtype=float)


def allocate_time_round_robin(alive_mask: np.ndarray) -> np.ndarray:
    times = np.zeros_like(alive_mask, dtype=float)
    alive_count = int(np.sum(alive_mask))
    if alive_count > 0:
        times[alive_mask] = FRAME_TIME_SEC / alive_count
    return times


def allocate_time_battery_aware(
    energies: np.ndarray,
    alive_mask: np.ndarray,
) -> np.ndarray:
    battery_coeff = compute_battery_coefficient_B(energies, alive_mask)
    return FRAME_TIME_SEC * battery_coeff


def allocate_time_proposed(
    energies: np.ndarray,
    voltages: np.ndarray,
    alive_mask: np.ndarray,
) -> np.ndarray:
    battery_coeff = compute_battery_coefficient_B(energies, alive_mask)
    efficiency_coeff = compute_charging_efficiency_coefficient_C(voltages, alive_mask)

    scores = battery_coeff * efficiency_coeff
    score_sum = float(np.sum(scores))

    if score_sum <= EPSILON:
        return allocate_time_round_robin(alive_mask)

    return FRAME_TIME_SEC * scores / score_sum


# ============================================================
# Battery update and scheme simulation
# ============================================================

def compute_charging_power_w(voltages: np.ndarray) -> np.ndarray:
    """
    Convert predicted charging voltage to charging power.

    Since the voltage was measured across a 100-ohm load resistor:

        P_chg = scale * V^2 / R
    """
    voltages = np.asarray(voltages, dtype=float)
    return CHARGING_EFFICIENCY * (voltages ** 2) / LOAD_RESISTANCE_OHM


def _node_arrays(nodes: list[Node]) -> tuple[np.ndarray, np.ndarray]:
    energies = np.array([node.energy for node in nodes], dtype=float)
    voltages = np.array([node.predicted_voltage for node in nodes], dtype=float)
    return energies, voltages


def simulate_scheme_once(nodes: list[Node], scheme: str) -> float:
    energies, voltages = _node_arrays(nodes)

    if np.any(energies <= DEATH_THRESHOLD):
        return 0.0

    elapsed_sec = 0.0
    charging_power_w = compute_charging_power_w(voltages)

    for _frame_idx in range(FRAME_COUNT):
        alive_mask = energies > DEATH_THRESHOLD

        # 1) 스케줄링 계수와 충전 시간은 frame마다 한 번만 계산
        if scheme == "No Charging":
            charging_times_frame = allocate_time_no_charging(alive_mask)
        elif scheme == "Round Robin":
            charging_times_frame = allocate_time_round_robin(alive_mask)
        elif scheme == "Battery-aware":
            charging_times_frame = allocate_time_battery_aware(energies, alive_mask)
        elif scheme == "Proposed":
            charging_times_frame = allocate_time_proposed(energies, voltages, alive_mask)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        # 2) frame 내부에서는 충전 비율이 고정됨
        charging_fraction = charging_times_frame / FRAME_TIME_SEC

        # 3) normalized battery 변화율 [1/sec]
        # dE/dt = (-P_con + P_chg * u_i) / E_bat
        energy_rate_per_sec = (
            -NODE_POWER_CONSUMPTION_W
            + charging_power_w * charging_fraction
        ) / STORAGE_ENERGY_CAPACITY_J

        # 4) frame 끝까지 갔을 때의 배터리 상태
        energies_next = energies + energy_rate_per_sec * FRAME_TIME_SEC

        # 5) frame 중간에 death threshold를 넘는지 정확히 계산
        crossing_mask = (
            (energies > DEATH_THRESHOLD)
            & (energies_next <= DEATH_THRESHOLD)
            & (energy_rate_per_sec < 0)
        )

        if np.any(crossing_mask):
            time_to_death_sec = (
                (energies[crossing_mask] - DEATH_THRESHOLD)
                / (-energy_rate_per_sec[crossing_mask])
            )

            return (elapsed_sec + float(np.min(time_to_death_sec))) / 60.0

        # 6) frame 끝 상태 반영
        energies = np.clip(energies_next, 0.0, 1.0)
        elapsed_sec += FRAME_TIME_SEC

    return MAX_SIM_TIME_MIN


# ============================================================
# Monte Carlo, output, and plot
# ============================================================

def run_monte_carlo() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    intensity_map = compute_laser_intensity_map()

    schemes = ["No Charging", "Round Robin", "Battery-aware", "Proposed"]

    trial_records = []
    node_records = []

    for trial_idx in range(MONTE_CARLO_RUNS):
        # 같은 trial에서는 Original과 Shaking 조건이 동일한 거리 조건을 사용
        distances_m = generate_trial_distances(rng)

        for condition_name, max_offset_px in SHAKING_CONDITIONS:
            # 같은 trial + 같은 condition 안에서는 모든 scheme이 동일한 offset 사용
            offsets_xy_px = generate_pointing_offsets(rng, max_offset_px)

            base_nodes = generate_nodes_from_geometry(
                distances_m=distances_m,
                offsets_xy_px=offsets_xy_px,
                intensity_map=intensity_map,
            )

            # 확인용 node condition 기록 저장
            for node_idx, node in enumerate(base_nodes, start=1):
                node_records.append(
                    {
                        "trial": trial_idx + 1,
                        "condition": condition_name,
                        "max_offset_px": max_offset_px,
                        "node": node_idx,
                        "distance_m": node.distance_m,
                        "offset_x_px": node.offset_x_px,
                        "offset_y_px": node.offset_y_px,
                        "pv_side_px": node.pv_side_px,
                        "pv_area_px": node.pv_area_px,
                        "received_intensity_S": node.received_intensity_S,
                        "predicted_voltage": node.predicted_voltage,
                    }
                )

            for scheme in schemes:
                # Fairness condition:
                # 같은 trial + 같은 condition에서는 모든 scheme이
                # 동일한 distance, offset, voltage 조건을 사용한다.
                nodes_for_scheme = deepcopy(base_nodes)
                death_time_min = simulate_scheme_once(nodes_for_scheme, scheme)

                trial_records.append(
                    {
                        "trial": trial_idx + 1,
                        "condition": condition_name,
                        "max_offset_px": max_offset_px,
                        "scheme": scheme,
                        "first_node_death_time_min": death_time_min,
                    }
                )

    trial_df = pd.DataFrame(trial_records)
    node_df = pd.DataFrame(node_records)

    summary_records = []

    for condition_name, max_offset_px in SHAKING_CONDITIONS:
        for scheme in schemes:
            values = trial_df.loc[
                (trial_df["condition"] == condition_name)
                & (trial_df["scheme"] == scheme),
                "first_node_death_time_min",
            ].to_numpy(dtype=float)

            std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

            summary_records.append(
                {
                    "condition": condition_name,
                    "max_offset_px": max_offset_px,
                    "scheme": scheme,
                    "mean_first_node_death_time_min": float(np.mean(values)),
                    "std_first_node_death_time_min": std_value,
                    "min_first_node_death_time_min": float(np.min(values)),
                    "max_first_node_death_time_min": float(np.max(values)),
                }
            )

    summary_df = pd.DataFrame(summary_records)
    return summary_df, trial_df, node_df


def save_results(
    summary_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    node_df: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(
        OUTPUT_SUMMARY_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    trial_df.to_csv(
        OUTPUT_TRIAL_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    node_df.to_csv(
        OUTPUT_NODE_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def _plot_single_condition(
    summary_df: pd.DataFrame,
    condition_name: str,
    output_path: Path,
    y_min: float | None = None,
) -> None:
    subset = summary_df.loc[summary_df["condition"] == condition_name].copy()

    scheme_order = ["No Charging", "Round Robin", "Battery-aware", "Proposed"]
    subset["scheme"] = pd.Categorical(
        subset["scheme"],
        categories=scheme_order,
        ordered=True,
    )
    subset = subset.sort_values("scheme")

    plt.figure(figsize=(9.0, 5.2))

    x = np.arange(len(subset))
    means = subset["mean_first_node_death_time_min"].to_numpy(dtype=float)
    stds = subset["std_first_node_death_time_min"].to_numpy(dtype=float)

    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=["#999999", "#4C78A8", "#59A14F", "#F28E2B"],
        edgecolor="#222222",
        linewidth=0.8,
    )

    plt.xticks(x, subset["scheme"].astype(str).tolist())
    plt.ylabel("average first node death time [min]")
    if y_min is not None:
        plt.ylim(bottom=y_min)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_results(summary_df: pd.DataFrame) -> None:
    # 1) 기존 조건 plot
    _plot_single_condition(
        summary_df,
        condition_name="Original",
        output_path=OUTPUT_PLOT_ORIGINAL_PATH,
    )
    _plot_single_condition(
        summary_df,
        condition_name="Original",
        output_path=OUTPUT_PLOT_ORIGINAL_CROPPED_PATH,
        y_min=Y_AXIS_CROP_MIN,
    )

    # 2) 흔든 조건 plot
    _plot_single_condition(
        summary_df,
        condition_name="Shaking 0-5px",
        output_path=OUTPUT_PLOT_SHAKING_PATH,
    )
    _plot_single_condition(
        summary_df,
        condition_name="Shaking 0-5px",
        output_path=OUTPUT_PLOT_SHAKING_CROPPED_PATH,
        y_min=Y_AXIS_CROP_MIN,
    )

    # 3) 조건 비교용 grouped bar plot
    schemes = ["No Charging", "Round Robin", "Battery-aware", "Proposed"]
    conditions = [name for name, _ in SHAKING_CONDITIONS]

    x = np.arange(len(schemes))
    bar_width = 0.35

    plt.figure(figsize=(10.0, 5.4))

    colors = {
        "Original": "#4C78A8",
        "Shaking 0-5px": "#F28E2B",
    }

    for cond_idx, condition in enumerate(conditions):
        means = []
        stds = []

        for scheme in schemes:
            row = summary_df.loc[
                (summary_df["condition"] == condition)
                & (summary_df["scheme"] == scheme)
            ]

            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(row["mean_first_node_death_time_min"].iloc[0]))
                stds.append(float(row["std_first_node_death_time_min"].iloc[0]))

        offset = (cond_idx - 0.5) * bar_width

        plt.bar(
            x + offset,
            means,
            width=bar_width,
            yerr=stds,
            capsize=5,
            label=condition,
            color=colors.get(condition, None),
            edgecolor="#222222",
            linewidth=0.8,
        )

    plt.xticks(x, schemes)
    plt.ylabel("average first node death time [min]")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_COMPARE_PATH, dpi=200)
    plt.ylim(bottom=Y_AXIS_CROP_MIN)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_COMPARE_CROPPED_PATH, dpi=200)
    plt.close()


def main() -> None:
    summary_df, trial_df, node_df = run_monte_carlo()

    save_results(summary_df, trial_df, node_df)
    plot_results(summary_df)

    print("\n========== Summary ==========")
    print(summary_df.to_string(index=False))

    print(f"\n[SAVE] {OUTPUT_SUMMARY_CSV_PATH}")
    print(f"[SAVE] {OUTPUT_TRIAL_CSV_PATH}")
    print(f"[SAVE] {OUTPUT_NODE_CSV_PATH}")
    print(f"[SAVE] {OUTPUT_PLOT_ORIGINAL_PATH}")
    print(f"[SAVE] {OUTPUT_PLOT_ORIGINAL_CROPPED_PATH}")
    print(f"[SAVE] {OUTPUT_PLOT_SHAKING_PATH}")
    print(f"[SAVE] {OUTPUT_PLOT_SHAKING_CROPPED_PATH}")
    print(f"[SAVE] {OUTPUT_PLOT_COMPARE_PATH}")
    print(f"[SAVE] {OUTPUT_PLOT_COMPARE_CROPPED_PATH}")


if __name__ == "__main__":
    main()
