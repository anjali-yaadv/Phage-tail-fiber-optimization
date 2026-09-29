"""Figure 6: effect of thermal COM fluctuations on the adsorption time.

Mean adsorption time <T_ads> vs N for two fiber reaches r_s, comparing a fixed
capture radius r_t with the fluctuation-enlarged radius
r_eff(n) = max(r_t, r_exp(n)) (Eqs. reff1, reff), where n is the number of
attached fibers. r_exp uses fiber length ell = ell_over_rs * r_s with M
segments (config: thermal). A larger patch (L = 20) keeps the target area
fraction small. The underlying distributions are shown in Fig S2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from phage_search import analysis, datasets, theory  # noqa: E402
from phage_search.config import figure_parser, load_config  # noqa: E402
from phage_search.plotting import SINGLE_COL, save, setup_style  # noqa: E402

CAPTURE_COLORS = {"fixed": "tab:orange", "thermal": "tab:green"}
MARKERS = ["o", "^", "s", "D"]
LINES = ["-", ":", "--", "-."]


def mean_T(cfg, N, r_s, capture, n_traj, L, resimulate):
    m = cfg["model"]
    df, _ = datasets.adsorption(cfg, N, m["k_off"], n_traj, r_s=r_s, L=L, capture=capture,
                                resimulate=resimulate)
    df = df[df.adsorbed]
    T = analysis.adsorption_times(df.surface_time, df.n_detach, m["eta_B"],
                                  theory.p_return(m["a_cell"], m["r_t"]), cfg["simulation"]["seed"])
    return T


def main():
    args = figure_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    f = cfg["figures"]["fig6"]
    n_traj = 50 if args.quick else f["n_traj"]
    Ns = [3, 6] if args.quick else f["N"]
    setup_style()

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    for i, r_s in enumerate(f["r_s"]):
        for capture in ("fixed", "thermal"):
            means = [mean_T(cfg, N, r_s, capture, n_traj, f["L"], args.resimulate).mean() for N in Ns]
            ax.plot(Ns, means, marker=MARKERS[i], ls=LINES[i], color=CAPTURE_COLORS[capture])
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\langle T_{\mathrm{ads}}\rangle\ (k_{\mathrm{on}}^{-1})$")
    h1 = [Line2D([], [], color="gray", marker=MARKERS[i], ls=LINES[i], label=rf"$r_s={r:g}$")
          for i, r in enumerate(f["r_s"])]
    h2 = [Line2D([], [], color=c, lw=4, label=lab)
          for lab, c in ((r"$r_t$", CAPTURE_COLORS["fixed"]), (r"$r_{\mathrm{eff}}$", CAPTURE_COLORS["thermal"]))]
    leg = ax.legend(handles=h1, loc="lower left")
    ax.add_artist(leg)
    ax.legend(handles=h2, loc="lower right")
    fig.tight_layout()
    save(fig, args.out, "fig6_thermal_fluctuations" + ("_quick" if args.quick else ""))


if __name__ == "__main__":
    main()
