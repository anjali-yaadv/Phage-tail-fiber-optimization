"""Figure S3: effect of the fiber reach r_s.

(a) D_eff vs N for several r_s (MSD simulations, default k_off), log-log.
(b) Adsorption-time distributions P_ads for several r_s at the default N;
    dashed = renewal model with D_eff ~ r_s^2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from phage_search import analysis, datasets, theory  # noqa: E402
from phage_search.config import REPO_ROOT, figure_parser, load_config  # noqa: E402
from phage_search.plotting import DOUBLE_COL, panel_label, save, setup_style  # noqa: E402


def main():
    args = figure_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    f, m = cfg["figures"]["figS3"], cfg["model"]
    f1 = cfg["figures"]["fig1"]
    quick = args.quick
    n_traj = 50 if quick else f["n_traj"]
    tp = theory.params_from_config(cfg)
    p_r = theory.p_return(m["a_cell"], m["r_t"])
    p_l = theory.p_land(m["r_t"], m["L"], m["n_targets"])
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    setup_style()

    fig, (axa, axb) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    # (a) D_eff vs N for several r_s ---------------------------------------------
    for r_s, c in zip(f["D_r_s"], colors):
        Ns = [3, 6, 10] if quick else f["D_N"]
        D = [datasets.msd(cfg, N, m["k_off"], r_s, n_traj, f1["n_points"],
                          n_bootstrap=f1["n_bootstrap"], bootstrap_fraction=f1["bootstrap_fraction"],
                          resimulate=args.resimulate)[1]["D_eff"] for N in Ns]
        axa.loglog(Ns, D, "o-", color=c, label=rf"$r_s={r_s:g}$")
    axa.set_xticks(Ns)
    axa.set_xticklabels([str(n) for n in Ns])
    axa.minorticks_off()
    axa.set_xlabel(r"$N$")
    axa.set_ylabel(r"$D_{\mathrm{eff}}$")
    axa.legend()
    panel_label(axa, "a)")

    # (b) P_ads for several r_s ---------------------------------------------------
    N = m["N"]
    for r_s, c in zip(f["dist_r_s"], colors):
        df, _ = datasets.adsorption(cfg, N, m["k_off"], n_traj, r_s=r_s, resimulate=args.resimulate)
        df = df[df.adsorbed]
        T = analysis.adsorption_times(df.surface_time, df.n_detach, m["eta_B"], p_r,
                                      cfg["simulation"]["seed"])
        edges = np.unique(np.logspace(np.log10(max(1.0, T[T > 0].min())), np.log10(T.max()), 15))
        x, y, e = analysis.pdf_with_errors(T, edges)
        ok = y > 0
        axb.errorbar(x[ok], y[ok], yerr=e[ok], fmt="o", ms=3, capsize=1, color=c, label=rf"$r_s={r_s:g}$")
        D = theory.D_eff_for_comparison(N, **dict(tp, r_s=r_s),
                                        table=pd.read_csv(REPO_ROOT / "data" / "processed" / "D_eff_vs_N.csv"))
        lam_h = 1.0 / theory.mean_hit_time(D, m["L"], m["r_t"], m["n_targets"], tp["C_torus"])
        lam_d = theory.detachment_rate(N, m["k_on"], m["k_off"])
        t = np.linspace(0, 1.2 * T.max(), 4000)
        axb.plot(t, theory.renewal_distribution(t, lam_h, lam_d, m["eta_B"], p_r, p_l, tp["direct_landing_after_return"]), "--", color=c)
    axb.set_yscale("log")
    axb.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axb.set_xlabel(r"$T_{\mathrm{ads}}\ (k_{\mathrm{on}}^{-1})$")
    axb.set_ylabel(r"$P_{\mathrm{ads}}$")
    axb.legend()
    panel_label(axb, "b)")

    fig.tight_layout()
    save(fig, args.out, "figS3_tradeoff_rs" + ("_quick" if quick else ""))


if __name__ == "__main__":
    main()
