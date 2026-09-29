"""Figure 3: validation of the hit-time and detachment-time theory.

(a) Mean hit time <T_h> (times k_off) vs N, simulated with the last fiber
    unable to detach, compared with Eq. 1 / Supplement Eq. S6 using the
    simulated D_eff from Fig 1c (data/processed/D_eff_vs_N.csv).
(b) Mean complete-detachment time <T_d> vs N, compared with Eq. 2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phage_search import datasets, theory  # noqa: E402
from phage_search.analysis import summary_stats  # noqa: E402
from phage_search.config import REPO_ROOT, figure_parser, load_config  # noqa: E402
from phage_search.plotting import DOUBLE_COL, koff_color, panel_label, save, setup_style  # noqa: E402


def main():
    args = figure_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    f, m, th = cfg["figures"]["fig3"], cfg["model"], cfg["theory"]
    quick = args.quick
    setup_style()
    D_table = pd.read_csv(REPO_ROOT / "data" / "processed" / "D_eff_vs_N.csv")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    # (a) hit time --------------------------------------------------------------
    for k in ([0.1] if quick else f["hit_k_off"]):
        Ns, sim, err, thy = [], [], [], []
        for N in ([3, 5] if quick else f["hit_N"]):
            df, _ = datasets.hit_times(cfg, N, k, 30 if quick else f["n_traj"],
                                       resimulate=args.resimulate)
            s = summary_stats(df.hit_time)
            row = D_table[(D_table.N == N) & np.isclose(D_table.k_off, k) & np.isclose(D_table.r_s, m["r_s"])]
            Ns.append(N)
            sim.append(s["mean"])
            err.append(s["sem"])
            thy.append(theory.mean_hit_time(row.D_eff.item(), m["L"], m["r_t"], m["n_targets"],
                                            th["C_torus"]) if len(row) else np.nan)
        c = koff_color(k)
        axa.errorbar(Ns, np.array(sim) * k, yerr=np.array(err) * k, fmt="o", capsize=3, color=c,
                     label=rf"$k_{{\mathrm{{off}}}}={k:g}$")
        axa.plot(Ns, np.array(thy) * k, "--", color=c)
    axa.set_ylim(0, None)
    axa.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axa.set_xlabel(r"$N$")
    axa.set_ylabel(r"$\langle T_h\rangle \times k_{\mathrm{off}}$")
    axa.legend(loc="upper left")
    panel_label(axa, "a)")

    # (b) detachment time ---------------------------------------------------------
    for k in f["detach_k_off"]:
        N_list = f["detach_N"]
        if quick:   # large N at small k_off takes very long without numba
            N_list = [N for N in N_list if theory.detachment_rate(N, m["k_on"], k) > 1e-3]
        Ns, sim, err = [], [], []
        for N in N_list:
            df, _ = datasets.detach_times(cfg, N, k, 200 if quick else f["n_traj"],
                                          resimulate=args.resimulate)
            s = summary_stats(df.detach_time)
            Ns.append(N)
            sim.append(s["mean"])
            err.append(s["sem"])
        c = koff_color(k)
        axb.errorbar(Ns, sim, yerr=err, fmt="o", capsize=3, color=c,
                     label=rf"$k_{{\mathrm{{off}}}}={k:g}$")
        N_th = np.array(f["detach_N"])
        axb.plot(N_th, 1.0 / theory.detachment_rate(N_th, m["k_on"], k), "--", color=c)
    axb.set_yscale("log")
    axb.set_xlabel(r"$N$")
    axb.set_ylabel(r"$\langle T_d\rangle$")
    axb.legend(loc="upper left")
    panel_label(axb, "b)")

    fig.tight_layout()
    save(fig, args.out, "fig3_hit_detach_validation" + ("_quick" if quick else ""))


if __name__ == "__main__":
    main()
