"""Shared figure style (Biophysical Journal single/double column sizes)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SINGLE_COL = (3.35, 2.6)
DOUBLE_COL = (7.2, 2.7)

# one colour per detachment rate, used consistently across figures
KOFF_COLORS = {0.001: "tab:red", 0.01: "tab:green", 0.1: "tab:orange",
               1.0: "tab:blue", 10.0: "tab:purple"}


def koff_color(k_off):
    return KOFF_COLORS.get(round(float(k_off), 6), "black")


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
        "lines.markersize": 4,
        "legend.frameon": False,
        "mathtext.fontset": "dejavuserif",
    })


def sci(x):
    """Format a number as LaTeX, e.g. 1e-05 -> $10^{-5}$, 2e-9 -> $2\\times10^{-9}$."""
    m, e = f"{x:.0e}".split("e")
    return rf"$10^{{{int(e)}}}$" if m == "1" else rf"${m}\times10^{{{int(e)}}}$"


def panel_label(ax, text, x=-0.16, y=1.02):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)


def save(fig, out_dir, name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved {out / name}.png / .pdf")
