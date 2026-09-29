"""Figure S5: global vs local criteria for adding tail fibers (analytical).

Sign maps over (k_off/k_on, M) at fixed k_off:
  global  Delta = <T_ads(M)> - <T_ads(1)>      (is multivalency better than one fiber?)
  local   Delta = <T_ads(M+1)> - <T_ads(M)>    (does one more fiber help?)
Blue: Delta < 0 (favourable); red: Delta > 0 (unfavourable).
Top row: low host density; bottom row: high host density.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from phage_search import theory  # noqa: E402
from phage_search.config import figure_parser, load_config  # noqa: E402
from phage_search.plotting import save, setup_style  # noqa: E402


def log_edges(x):
    lx = np.log10(x)
    mid = 0.5 * (lx[1:] + lx[:-1])
    return 10 ** np.concatenate([[2 * lx[0] - mid[0]], mid, [2 * lx[-1] - mid[-1]]])


def main():
    args = figure_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    f = cfg["figures"]["figS5"]
    tp = theory.params_from_config(cfg)
    ratios = np.logspace(np.log10(f["ratio_min"]), np.log10(f["ratio_max"]), f["n_ratio"])
    M = np.arange(1, f["M_max"] + 1)
    cmap = ListedColormap(["#2563eb", "#ffffff", "#dc2626"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.6), sharex=True, sharey=True)
    for row, (eta_B, name) in enumerate(((f["eta_B_low_density"], "low density"),
                                         (f["eta_B_high_density"], "high density"))):
        glob = np.zeros((len(M), len(ratios)))
        loc = np.zeros_like(glob)
        for j, r in enumerate(ratios):
            kw = dict(tp, k_off=f["k_off"], k_on=f["k_off"] / r, eta_B=eta_B)
            T = theory.mean_adsorption_time(np.arange(1, f["M_max"] + 2), **kw)
            glob[:, j] = np.sign(T[:-1] - T[0])
            loc[:, j] = np.sign(T[1:] - T[:-1])
        for col, (S, title) in enumerate(((glob, r"Global: $T_M - T_1$"),
                                          (loc, r"Local: $T_{M+1} - T_M$"))):
            ax = axes[row, col]
            ax.pcolormesh(log_edges(ratios), np.arange(0.5, f["M_max"] + 1), S, cmap=cmap, norm=norm)
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(title, fontsize=9)
            if row == 1:
                ax.set_xlabel(r"$k_{\mathrm{off}}/k_{\mathrm{on}}$")
            if col == 0:
                ax.set_ylabel(f"$M$ ({name})")
    axes[0, 1].legend(handles=[Patch(color="#2563eb", label=r"$\Delta<0$"),
                               Patch(color="#dc2626", label=r"$\Delta>0$")],
                      loc="upper left", frameon=True)
    fig.tight_layout()
    save(fig, args.out, "figS5_global_local")


if __name__ == "__main__":
    main()
