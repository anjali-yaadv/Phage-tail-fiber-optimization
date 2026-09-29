"""Figure S4: surface exploration vs tail fiber number for two fiber reaches.

The L x L surface is divided into an m x m lattice; for trajectories that stay
attached for the whole window t_window, we count the distinct lattice cells
visited by the COM. k_off << k_on so that most trajectories stay attached.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from phage_search import datasets  # noqa: E402
from phage_search.analysis import summary_stats  # noqa: E402
from phage_search.config import figure_parser, load_config  # noqa: E402
from phage_search.plotting import panel_label, save, setup_style  # noqa: E402


def main():
    args = figure_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    f = cfg["figures"]["figS4"]
    n_surv = 20 if args.quick else f["n_survivors"]
    t_window = 200.0 if args.quick else f["t_window"]
    Ns = [2, 4, 6] if args.quick else f["N"]
    setup_style()

    fig, axes = plt.subplots(1, len(f["r_s"]), figsize=(5.6, 2.4))
    for ax, r_s, lab in zip(np.atleast_1d(axes), f["r_s"], "abcd"):
        stats = [summary_stats(datasets.coverage(cfg, N, f["k_off"], r_s, f["m"], t_window, n_surv,
                                                 resimulate=args.resimulate)[0].cells_visited)
                 for N in Ns]
        ax.errorbar(Ns, [s["mean"] for s in stats], yerr=[s["sem"] for s in stats],
                    fmt="o-", ms=3, capsize=2, label=rf"$r_s={r_s:g}$")
        ax.set_xlabel(r"$N$")
        ax.legend(loc="upper right")
        panel_label(ax, lab + ")", x=-0.3)
    np.atleast_1d(axes)[0].set_ylabel("Unique lattice cells visited")
    fig.tight_layout()
    save(fig, args.out, "figS4_lattice_coverage" + ("_quick" if args.quick else ""))


if __name__ == "__main__":
    main()
