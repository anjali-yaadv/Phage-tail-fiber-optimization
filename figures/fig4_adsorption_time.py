"""Figure 4: adsorption time from simulations and from the renewal model.

(a) Distribution of adsorption times P_ads (start: one fiber attached) for two
    k_off; points = simulation, dashed = renewal model (Eq. 3) with analytical
    lambda_h (Eq. 1, D_eff scaling of Fig 1c) and lambda_d (Eq. 2).
(b) Mean adsorption time <T_ads> vs N; points = simulation mean, band =
    interquartile range, dashed = Eq. 5.
(c, d) Mean total time <T_tot> = <T_ads> + 1/eta_B vs N from Eqs. 5-6 for
    high (c) and low (d) host density. Hatched: below the bulk-search time 1/eta_B.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phage_search import analysis, datasets, theory  # noqa: E402
from phage_search.config import REPO_ROOT, figure_parser, load_config  # noqa: E402
from phage_search.plotting import koff_color, panel_label, save, sci, setup_style  # noqa: E402


def main():
    args = figure_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    f, m = cfg["figures"]["fig4"], cfg["model"]
    quick = args.quick
    n_traj = 100 if quick else f["n_traj"]
    tp = theory.params_from_config(cfg)
    p_r = theory.p_return(m["a_cell"], m["r_t"])
    p_l = theory.p_land(m["r_t"], m["L"], m["n_targets"])
    seed = cfg["simulation"]["seed"]
    # simulated D_eff of Fig 1c where available, scaling law otherwise
    D_table = pd.read_csv(REPO_ROOT / "data" / "processed" / "D_eff_vs_N.csv")
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    axa, axb, axc, axd = axes.ravel()

    # (a) distribution --------------------------------------------------------
    N = f["dist_N"]
    for k in f["dist_k_off"]:
        df, _ = datasets.adsorption(cfg, N, k, n_traj, resimulate=args.resimulate)
        df = df[df.adsorbed]
        T = analysis.adsorption_times(df.surface_time, df.n_detach, m["eta_B"], p_r, seed)
        edges = np.unique(np.logspace(np.log10(max(1.0, T[T > 0].min())), np.log10(T.max()), 15))
        x, y, e = analysis.pdf_with_errors(T, edges)
        ok = y > 0
        c = koff_color(k)
        axa.errorbar(x[ok], y[ok], yerr=e[ok], fmt="o", ms=3, capsize=1, color=c,
                     label=rf"$k_{{\mathrm{{off}}}}={k:g}$")
        D = theory.D_eff_for_comparison(N, **dict(tp, k_off=k), table=D_table)
        lam_h = 1.0 / theory.mean_hit_time(D, m["L"], m["r_t"], m["n_targets"], tp["C_torus"])
        lam_d = theory.detachment_rate(N, m["k_on"], k)
        t = np.linspace(0, 1.2 * T.max(), 4000)
        axa.plot(t, theory.renewal_distribution(t, lam_h, lam_d, m["eta_B"], p_r, p_l, tp["direct_landing_after_return"]), "--", color=c)
    axa.set_yscale("log")
    axa.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axa.set_ylim(1e-8, None)
    axa.set_xlabel(r"$T_{\mathrm{ads}}$")
    axa.set_ylabel(r"$P_{\mathrm{ads}}$")
    axa.legend()
    panel_label(axa, "a)")

    # (b) mean vs N -----------------------------------------------------------
    for k in ([0.1] if quick else f["mean_k_off"]):
        Ns = [3, 5] if quick else f["mean_N"]
        stats = []
        for N in Ns:
            df, _ = datasets.adsorption(cfg, N, k, n_traj, resimulate=args.resimulate)
            df = df[df.adsorbed]
            stats.append(analysis.summary_stats(
                analysis.adsorption_times(df.surface_time, df.n_detach, m["eta_B"], p_r, seed)))
        c = koff_color(k)
        axb.fill_between(Ns, [s["p25"] for s in stats], [s["p75"] for s in stats],
                         color=c, alpha=0.15, lw=0)
        axb.plot(Ns, [s["mean"] for s in stats], "o", color=c, label=rf"$k_{{\mathrm{{off}}}}={k:g}$")
        N_th = np.arange(min(Ns), max(Ns) + 1)
        D = [theory.D_eff_for_comparison(n, **dict(tp, k_off=k), table=D_table) for n in N_th]
        axb.plot(N_th, theory.mean_adsorption_time(N_th, **dict(tp, k_off=k), D_eff=D), "--", color=c)
    axb.set_yscale("log")
    axb.set_xlabel(r"$N$")
    axb.set_ylabel(r"$\langle T_{\mathrm{ads}}\rangle$")
    axb.legend(loc="lower right")
    panel_label(axb, "b)")

    # (c, d) theory: total time from the bulk ---------------------------------
    N_th = np.arange(1, f["theory_N_max"] + 1)
    for ax, eta_B, lab in ((axc, f["eta_B_high_density"], "c)"), (axd, f["eta_B_low_density"], "d)")):
        for ratio in f["theory_k_off_over_k_on"]:
            k_off = ratio * m["k_on"]
            T = theory.mean_total_time(N_th, **dict(tp, k_off=k_off, eta_B=eta_B))
            ax.plot(N_th, T, "o-", ms=3, lw=1.5, color=koff_color(k_off),
                    label=rf"$k_{{\mathrm{{off}}}}/k_{{\mathrm{{on}}}}={ratio:g}$")
        ax.axhspan(1e-30, 1.0 / eta_B, facecolor="none", hatch="///", edgecolor="gray", lw=0, alpha=0.5)
        ax.set_ylim(0.5 / eta_B, None)
        ax.set_yscale("log")
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$\langle T_{\mathrm{tot}}\rangle$")
        ax.text(0.97, 0.05, r"$\eta_B=$" + sci(eta_B), transform=ax.transAxes, ha="right")
        panel_label(ax, lab)
    axd.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    for ax in (axb, axc, axd):
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    save(fig, args.out, "fig4_adsorption_time" + ("_quick" if quick else ""))


if __name__ == "__main__":
    main()
