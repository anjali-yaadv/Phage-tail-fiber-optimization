"""Figure 5: optimal tail fiber number N* vs k_off/k_on (analytical, Eqs. 5-6).

N* minimises <T_tot>(N) over integers 1..N_max. Each curve has a fixed k_off;
k_on = k_off / ratio varies along the x axis. Stars mark k_on = 1, i.e. the
parameter pairs whose full <T_tot>(N) curves are shown in Fig 4c, d.
(a) high host density, (b) low host density.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from phage_search import theory  # noqa: E402
from phage_search.config import figure_parser, load_config  # noqa: E402
from phage_search.plotting import DOUBLE_COL, koff_color, panel_label, save, sci, setup_style  # noqa: E402


def optimal_curve(tp, k_off, ratios, eta_B, N_max):
    return np.array([theory.optimal_N(N_max, **dict(tp, k_off=k_off, k_on=k_off / r, eta_B=eta_B))
                     for r in ratios])


def main():
    args = figure_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    f = cfg["figures"]["fig5"]
    tp = theory.params_from_config(cfg)
    ratios = np.logspace(np.log10(f["ratio_min"]), np.log10(f["ratio_max"]), f["n_ratio"])
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL, sharey=True)
    for ax, eta_B, name, lab in ((axes[0], f["eta_B_high_density"], "High density", "a)"),
                                 (axes[1], f["eta_B_low_density"], "Low density", "b)")):
        for k_off in f["k_off"]:
            c = koff_color(k_off)
            Nstar = optimal_curve(tp, k_off, ratios, eta_B, f["N_max"])
            ax.plot(ratios, Nstar, "o-", ms=3, color=c, label=rf"$k_{{\mathrm{{off}}}}={k_off:g}$")
            star = k_off / cfg["model"]["k_on"]          # the k_on = 1 point
            ax.plot(star, theory.optimal_N(f["N_max"], **dict(tp, k_off=k_off, eta_B=eta_B)), "*",
                    ms=9, color=c, mec="black", zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$k_{\mathrm{off}}/k_{\mathrm{on}}$")
        ax.text(0.04, 0.93, f"{name} (" + r"$\eta_B=$" + sci(eta_B) + ")", transform=ax.transAxes, va="top")
        panel_label(ax, lab)
    axes[0].set_ylabel(r"$N^*$")
    axes[1].legend(loc="center left")
    fig.tight_layout()
    save(fig, args.out, "fig5_optimal_N")


if __name__ == "__main__":
    main()
