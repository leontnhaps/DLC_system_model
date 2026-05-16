#!/usr/bin/env python3
"""
Plot the normalized beam intensity formula:

    I_px(u, v) = exp[-2 * (u^2 / 118.1416^2 + v^2 / 124.5135^2)]

Run:
    python plot_ipx_formula.py

Output:
    ipx_formula_heatmap.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

U_RADIUS = 118.1416
V_RADIUS = 124.5135


def i_px(u, v):
    """Evaluate I_px(u, v) on scalars or numpy arrays."""
    return np.exp(-2.0 * ((u / U_RADIUS) ** 2 + (v / V_RADIUS) ** 2))


def main():
    # Plot a little beyond the 1/e^2 radii so the Gaussian shape is visible.
    u = np.linspace(-2.5 * U_RADIUS, 2.5 * U_RADIUS, 500)
    v = np.linspace(-2.5 * V_RADIUS, 2.5 * V_RADIUS, 500)
    uu, vv = np.meshgrid(u, v)
    intensity = i_px(uu, vv)

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)

    image = ax.imshow(
        intensity,
        extent=[u.min(), u.max(), v.min(), v.max()],
        origin="lower",
        cmap="inferno",
        aspect="equal",
    )
    contour = ax.contour(
        uu,
        vv,
        intensity,
        levels=[np.exp(-2), 0.5, 0.8],
        colors=["cyan", "white", "lime"],
        linewidths=[2.0, 1.2, 1.2],
    )
    ax.clabel(contour, inline=True, fmt="I = %.3f", fontsize=9)

    ax.set_title(
        r"$I_{px}(u,v)=\exp\left[-2\left(\frac{u^2}{118.1416^2}"
        r"+\frac{v^2}{124.5135^2}\right)\right]$"
    )
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("normalized intensity")

    output_path = SCRIPT_DIR / "ipx_formula_heatmap.png"
    fig.savefig(output_path, dpi=200)
    plt.show()

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
