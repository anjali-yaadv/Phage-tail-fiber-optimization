"""Run the model for any parameter set (beyond the paper figures).

Defaults come from config/parameters.yaml (`model` and `simulation`);
anything given on the command line replaces them. Results are saved in
data/simulated/<kind>/ (CSV + JSON with the exact parameters) and a short
summary is printed.

Examples
--------
  python simulate.py msd        --N 6 --k_off 0.05 --r_s 2 --n_traj 500
  python simulate.py hit        --N 4 --k_off 0.1
  python simulate.py detach     --N 5 --k_off 0.5
  python simulate.py adsorption --N 6 --k_off 0.1 --capture thermal --eta_B 1e-7
  python simulate.py coverage   --N 4 --k_off 0.01 --r_s 2
  python simulate.py theory     --k_off 0.1 --eta_B 1e-9            # <T_ads>, <T_tot>, N*
  python simulate.py adsorption --N 6 --set model.n_targets=3        # any config value
"""
import argparse

import numpy as np
import pandas as pd

from phage_search import analysis, datasets, theory
from phage_search.config import REPO_ROOT, load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["msd", "hit", "detach", "adsorption", "coverage", "theory"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any value in the parameter file")
    for name in ("k_on", "k_off", "r_s", "r_t", "L", "eta_B", "a_cell"):
        ap.add_argument(f"--{name}", type=float, help=f"model.{name}")
    ap.add_argument("--N", type=int, help="model.N (number of tail fibers)")
    ap.add_argument("--n_targets", type=int, help="model.n_targets")
    ap.add_argument("--n_traj", type=int, help="number of trajectories (default: simulation.n_traj)")
    ap.add_argument("--seed", type=int, help="simulation.seed")
    ap.add_argument("--capture", choices=["fixed", "thermal"], default="fixed",
                    help="adsorption: fixed r_t or thermal r_eff(n)")
    ap.add_argument("--n_points", type=int, default=2000, help="msd: stored time points")
    ap.add_argument("--t_max", type=float, help="msd: trajectory length (default simulation.t_max_msd)")
    ap.add_argument("--m", type=int, default=50, help="coverage: lattice size m x m")
    ap.add_argument("--t_window", type=float, default=2000.0, help="coverage: time window")
    ap.add_argument("--N_max", type=int, default=20, help="theory: largest N")
    ap.add_argument("--resimulate", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    for name in ("k_on", "k_off", "r_s", "r_t", "L", "eta_B", "a_cell", "N", "n_targets"):
        if getattr(a, name) is not None:
            cfg["model"][name] = getattr(a, name)
    if a.seed is not None:
        cfg["simulation"]["seed"] = a.seed
    m = cfg["model"]
    n_traj = a.n_traj or cfg["simulation"]["n_traj"]
    print("parameters:", ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))

    if a.kind == "theory":
        tp = theory.params_from_config(cfg)
        N = np.arange(1, a.N_max + 1)
        Tads = theory.mean_adsorption_time(N, **tp)
        print(f"\n{'N':>3} {'<T_ads>':>12} {'<T_tot>':>12}")
        for n, t in zip(N, Tads):
            print(f"{n:>3} {t:12.4g} {t + 1 / m['eta_B']:12.4g}")
        print(f"\noptimal N* (1..{a.N_max}) = {theory.optimal_N(a.N_max, **tp)}")
        return

    if a.kind == "msd":
        df, info = datasets.msd(cfg, m["N"], m["k_off"], m["r_s"], n_traj, a.n_points, a.t_max,
                                resimulate=a.resimulate)
        print(f"D_eff = {info['D_eff']:.4g} +- {info['D_eff_sem']:.2g}")
    elif a.kind == "hit":
        df, info = datasets.hit_times(cfg, m["N"], m["k_off"], n_traj, resimulate=a.resimulate)
        s = analysis.summary_stats(df.hit_time)
        print(f"<T_h> = {s['mean']:.4g} +- {s['sem']:.2g}   (median {s['median']:.4g})")
    elif a.kind == "detach":
        df, info = datasets.detach_times(cfg, m["N"], m["k_off"], n_traj, resimulate=a.resimulate)
        s = analysis.summary_stats(df.detach_time)
        print(f"<T_d> = {s['mean']:.4g} +- {s['sem']:.2g}   theory (Eq. 2) = "
              f"{1 / theory.detachment_rate(m['N'], m['k_on'], m['k_off']):.4g}")
    elif a.kind == "adsorption":
        df, info = datasets.adsorption(cfg, m["N"], m["k_off"], n_traj, capture=a.capture,
                                       resimulate=a.resimulate)
        ok = df[df.adsorbed]
        T = analysis.adsorption_times(ok.surface_time, ok.n_detach, m["eta_B"],
                                      theory.p_return(m["a_cell"], m["r_t"]), cfg["simulation"]["seed"])
        s = analysis.summary_stats(T)
        print(f"adsorbed {len(ok)}/{len(df)};  <T_ads> = {s['mean']:.4g} +- {s['sem']:.2g}  "
              f"(median {s['median']:.4g}, IQR {s['p25']:.3g}-{s['p75']:.3g})")
        if a.capture == "fixed":
            tp = theory.params_from_config(cfg)
            D = theory.D_eff_for_comparison(m["N"], **tp, table=pd.read_csv(
                REPO_ROOT / "data" / "processed" / "D_eff_vs_N.csv"))
            print(f"theory (Eq. 5, D_eff = {D:.3g}): {theory.mean_adsorption_time(m['N'], **tp, D_eff=D):.4g}")
        else:
            print("(Eq. 5 assumes a fixed capture radius r_t; no theory line for --capture thermal)")
    elif a.kind == "coverage":
        df, info = datasets.coverage(cfg, m["N"], m["k_off"], m["r_s"], a.m, a.t_window,
                                     min(n_traj, 500), resimulate=a.resimulate)
        s = analysis.summary_stats(df.cells_visited)
        print(f"cells visited = {s['mean']:.4g} +- {s['sem']:.2g}  (survival {info['survival_fraction']:.3f})")
    print("data saved in data/simulated/" + a.kind + "/")


if __name__ == "__main__":
    main()
