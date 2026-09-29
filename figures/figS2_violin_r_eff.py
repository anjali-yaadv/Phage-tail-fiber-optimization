"""Figure S2: adsorption-time distributions with fixed r_t vs thermal r_eff.

One panel per fiber reach r_s; violins show the distribution of T_ads for each
N, with the median (line) and mean (marker). Uses the same simulations as Fig 6.
Adsorption at T = 0 (landing directly within the capture radius) is drawn at 10^-2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from fig6_thermal_fluctuations import CAPTURE_COLORS, mean_T  # noqa: E402
from phage_search.config import figure_parser, load_config  # noqa: E402
from phage_search.plotting import panel_label, save, setup_style  # noqa: E402


def main():
    args = figure_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    f = cfg["figures"]["figS2"]
    n_traj = 50 if args.quick else f["n_traj"]
    Ns = [3, 6] if args.quick else f["N"]
    setup_style()

    fig, axes = plt.subplots(1, len(f["r_s"]), figsize=(7.2, 2.6))
    for ax, r_s, lab in zip(np.atleast_1d(axes), f["r_s"], "abcdef"):
        for j, (capture, dx) in enumerate((("fixed", -0.18), ("thermal", 0.18))):
            data = [mean_T(cfg, N, r_s, capture, n_traj, f["L"], args.resimulate) for N in Ns]
            logs = [np.log10(np.clip(d, 1e-2, None)) for d in data]
            pos = np.array(Ns) + dx
            parts = ax.violinplot(logs, positions=pos, widths=0.35, showextrema=False, showmedians=True)
            for b in parts["bodies"]:
                b.set_facecolor(CAPTURE_COLORS[capture])
                b.set_alpha(0.45)
            parts["cmedians"].set_color("black")
            ax.plot(pos, [np.log10(d.mean()) for d in data], "os"[j], ms=3,
                    color=CAPTURE_COLORS[capture], mec="black", mew=0.5)
        ax.set_xticks(Ns)
        ax.set_xlabel(r"$N$")
        ax.set_title(rf"$r_s={r_s:g}$", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: rf"$10^{{{v:g}}}$"))
        panel_label(ax, lab + ")")
    np.atleast_1d(axes)[0].set_ylabel(r"$T_{\mathrm{ads}}\ (k_{\mathrm{on}}^{-1})$")
    np.atleast_1d(axes)[-1].legend(handles=[Patch(color=CAPTURE_COLORS["fixed"], label=r"$r_t$"),
                                            Patch(color=CAPTURE_COLORS["thermal"], label=r"$r_{\mathrm{eff}}$")],
                                   loc="lower right")
    fig.tight_layout()
    save(fig, args.out, "figS2_violin_r_eff" + ("_quick" if args.quick else ""))


if __name__ == "__main__":
    main()
