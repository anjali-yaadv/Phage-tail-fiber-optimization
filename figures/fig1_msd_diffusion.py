"""Figure 1 (b, c): surface mobility decreases with tail fiber number.

(b) Mean-squared displacement of the phage COM vs time for several N
    (k_off = 0.1, r_s = 1), log-log with a linear inset.
(c) Effective diffusion constant D_eff / k_off vs N for several k_off.

Panel (a) is a BioRender schematic and is not produced by code.

By default panel (c) uses the D_eff values behind the published figure
(data/processed/D_eff_vs_N.csv); with --resimulate it is recomputed from
new MSD simulations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402

from phage_search import datasets  # noqa: E402
from phage_search.config import REPO_ROOT, figure_parser, load_config  # noqa: E402
from phage_search.plotting import DOUBLE_COL, koff_color, panel_label, save, setup_style  # noqa: E402


def main():
    args = figure_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    f, m = cfg["figures"]["fig1"], cfg["model"]
    n_traj = 50 if args.quick else f["n_traj"]
    msd_N = [3, 6, 10] if args.quick else f["msd_N"]
    setup_style()

    fig, (axb, axc) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    # (b) MSD curves ---------------------------------------------------------
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(msd_N)))
    ins = inset_axes(axb, width="38%", height="35%", loc="upper left", borderpad=1.2)
    for N, c in zip(msd_N, colors):
        df, _ = datasets.msd(cfg, N, f["msd_k_off"], m["r_s"], n_traj, f["n_points"],
                             n_bootstrap=f["n_bootstrap"], bootstrap_fraction=f["bootstrap_fraction"],
                             resimulate=args.resimulate)
        # show times at which at least 10% of the trajectories are still attached
        ok = (df.time > 0) & (df.msd_mean > 0) & (df.n_attached >= max(10, 0.1 * n_traj))
        t, y, e = df.time[ok], df.msd_mean[ok], df.msd_sem[ok]
        axb.loglog(t, y, color=c, lw=1.2, label=f"$N={N}$")
        axb.fill_between(t, y - e, y + e, color=c, alpha=0.2, lw=0)
        ins.plot(t, y, color=c, lw=1)
    ins.set_xlim(0, 1000)
    ins.set_ylim(0, None)
    ins.tick_params(labelsize=6)
    axb.set_xlabel(r"$t\ (k_{\mathrm{on}}^{-1})$")
    axb.set_ylabel(r"$\langle r^2 \rangle$")
    axb.legend(loc="lower right", ncol=2, fontsize=7)
    panel_label(axb, "b)")

    # (c) D_eff / k_off vs N -------------------------------------------------
    if args.resimulate or args.quick:
        rows = []
        k_list = [0.1] if args.quick else f["D_k_off"]
        N_list = [3, 5, 8] if args.quick else f["D_N"]
        for k in k_list:
            for N in N_list:
                _, info = datasets.msd(cfg, N, k, m["r_s"], n_traj, f["n_points"],
                                       n_bootstrap=f["n_bootstrap"],
                                       bootstrap_fraction=f["bootstrap_fraction"],
                                       resimulate=args.resimulate)
                rows.append(dict(N=N, k_off=k, D_eff=info["D_eff"], D_eff_sem=info["D_eff_sem"]))
        D = pd.DataFrame(rows)
    else:
        D = pd.read_csv(REPO_ROOT / "data" / "processed" / "D_eff_vs_N.csv")
        D = D[np.isclose(D.r_s, m["r_s"]) & D.k_off.isin(f["D_k_off"]) & D.N.isin(f["D_N"])]

    for k, g in D.groupby("k_off"):
        g = g.sort_values("N")
        axc.errorbar(g.N, g.D_eff / k, yerr=g.D_eff_sem / k, fmt="o", ms=3, capsize=2,
                     color=koff_color(k), label=rf"$k_{{\mathrm{{off}}}}={k:g}$")
    axc.set_ylim(0, None)
    axc.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axc.set_xlabel(r"$N$")
    axc.set_ylabel(r"$D_{\mathrm{eff}}/k_{\mathrm{off}}$")
    axc.legend(loc="upper right")
    panel_label(axc, "c)")

    fig.subplots_adjust(wspace=0.3)
    save(fig, args.out, "fig1_msd_diffusion" + ("_quick" if args.quick else ""))


if __name__ == "__main__":
    main()
