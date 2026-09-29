"""Run a simulation for one parameter set and save/load the result.

Each function returns the data for one parameter set. Results are stored in
`data/simulated/<kind>/` as a CSV file plus a JSON file holding the exact
parameters. If a file with identical parameters already exists it is loaded
instead of re-simulating (use `resimulate=True` to force a new run).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, simulation, theory
from .config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "simulated"


def _fmt(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def _paths(kind, params, label_keys):
    blob = json.dumps(params, sort_keys=True, default=float).encode()
    h = hashlib.sha1(blob).hexdigest()[:8]
    label = "_".join(f"{k}{_fmt(params[k])}" for k in label_keys).replace(".", "p")
    stem = f"{kind}_{label}_{h}"
    return DATA_DIR / kind / f"{stem}.csv", DATA_DIR / kind / f"{stem}.json"


def _cached(kind, params, label_keys, compute, resimulate):
    csv, meta = _paths(kind, params, label_keys)
    if csv.exists() and meta.exists() and not resimulate:
        return pd.read_csv(csv), json.loads(meta.read_text())
    print(f"  simulating {kind}: " + ", ".join(f"{k}={_fmt(params[k])}" for k in label_keys)
          + f" ({params.get('n_traj', params.get('n_survivors'))} trajectories)", flush=True)
    t0 = time.time()
    df, extra = compute()
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    info = {"parameters": params, **extra,
            "runtime_s": round(time.time() - t0, 1),
            "numba": simulation.HAVE_NUMBA,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    meta.write_text(json.dumps(info, indent=2, default=float))
    return df, info


# -----------------------------------------------------------------------------
def msd(cfg, N, k_off, r_s, n_traj, n_points, t_max=None, n_bootstrap=20,
        bootstrap_fraction=0.1, resimulate=False):
    """Mean MSD curve and fitted D_eff for one (N, k_off, r_s)."""
    m, s = cfg["model"], cfg["simulation"]
    p = dict(N=int(N), k_on=m["k_on"], k_off=float(k_off), r_s=float(r_s),
             t_max=float(t_max or s["t_max_msd"]), n_points=int(n_points),
             n_traj=int(n_traj), seed=int(s["seed"]),
             n_bootstrap=int(n_bootstrap), bootstrap_fraction=float(bootstrap_fraction))

    def compute():
        runs = simulation.run_trajectories(
            "msd", (p["N"], p["k_on"], p["k_off"], p["r_s"], p["t_max"], p["n_points"]),
            p["n_traj"], p["seed"], s["n_workers"])
        t = runs[0][0]
        r2 = np.array([r[1] for r in runs])
        mean, sem, n_alive = analysis.mean_msd(r2)
        D, D_err = analysis.D_with_error(t, r2, p["n_bootstrap"], p["bootstrap_fraction"], p["seed"])
        df = pd.DataFrame({"time": t, "msd_mean": mean, "msd_sem": sem, "n_attached": n_alive})
        return df, {"D_eff": D, "D_eff_sem": D_err}

    return _cached("msd", p, ["N", "k_off", "r_s"], compute, resimulate)


def hit_times(cfg, N, k_off, n_traj, L=None, r_s=None, resimulate=False):
    """Hit times with the last fiber unable to detach (pure surface search)."""
    m, s = cfg["model"], cfg["simulation"]
    p = dict(N=int(N), k_on=m["k_on"], k_off=float(k_off), r_s=float(r_s or m["r_s"]),
             r_t=m["r_t"], L=float(L or m["L"]), n_targets=int(m["n_targets"]),
             t_max=float(s["t_max_adsorption"]), n_traj=int(n_traj), seed=int(s["seed"]))

    def compute():
        out = simulation.run_trajectories(
            "hit", (p["N"], p["k_on"], p["k_off"], p["r_s"], p["r_t"], p["L"], p["n_targets"], p["t_max"]),
            p["n_traj"], p["seed"], s["n_workers"])
        return pd.DataFrame({"hit_time": out}), {}

    return _cached("hit", p, ["N", "k_off", "r_s", "L"], compute, resimulate)


def detach_times(cfg, N, k_off, n_traj, resimulate=False):
    """Complete-detachment times starting from one attached fiber."""
    m, s = cfg["model"], cfg["simulation"]
    p = dict(N=int(N), k_on=m["k_on"], k_off=float(k_off),
             t_max=float(s["t_max_adsorption"]), n_traj=int(n_traj), seed=int(s["seed"]))

    def compute():
        out = simulation.run_trajectories("detach", (p["N"], p["k_on"], p["k_off"], p["t_max"]),
                                          p["n_traj"], p["seed"], s["n_workers"])
        return pd.DataFrame({"detach_time": out}), {}

    return _cached("detach", p, ["N", "k_off"], compute, resimulate)


def adsorption(cfg, N, k_off, n_traj, r_s=None, L=None, capture="fixed", resimulate=False):
    """Full adsorption runs (surface part only; bulk time is added in analysis).

    capture = "fixed"   : capture radius r_t
    capture = "thermal" : capture radius r_eff(n) = max(r_t, r_exp(n)) (Eq. reff)
    """
    m, s, th = cfg["model"], cfg["simulation"], cfg["thermal"]
    p = dict(N=int(N), k_on=m["k_on"], k_off=float(k_off), r_s=float(r_s or m["r_s"]),
             r_t=m["r_t"], L=float(L or m["L"]), n_targets=int(m["n_targets"]),
             capture=capture, t_max=float(s["t_max_adsorption"]),
             n_traj=int(n_traj), seed=int(s["seed"]))
    if capture == "thermal":
        p.update(M=th["M"], ell_over_rs=th["ell_over_rs"], D_3d=th["D_3d"])
        radii = theory.r_eff_by_n(p["N"], p["k_on"], p["k_off"], p["r_s"], p["r_t"],
                                  th["M"], th["ell_over_rs"], th["D_3d"])
    elif capture == "fixed":
        radii = np.full(p["N"] + 1, p["r_t"])
    else:
        raise ValueError("capture must be 'fixed' or 'thermal'")

    def compute():
        out = simulation.run_trajectories(
            "adsorption", (p["N"], p["k_on"], p["k_off"], p["r_s"], p["L"], p["n_targets"],
                           p["r_t"], radii, p["t_max"]),
            p["n_traj"], p["seed"], s["n_workers"])
        df = pd.DataFrame(out, columns=["adsorbed", "surface_time", "n_detach",
                                        "first_cycle_time", "first_cycle_outcome"])
        return df, {"capture_radius_by_n_attached": radii.tolist()}

    return _cached("adsorption", p, ["N", "k_off", "r_s", "L", "capture"], compute, resimulate)


def coverage(cfg, N, k_off, r_s, m_cells, t_window, n_survivors, max_tries_factor=100,
             resimulate=False):
    """Distinct lattice cells visited, for trajectories that stay attached for t_window."""
    mod, s = cfg["model"], cfg["simulation"]
    p = dict(N=int(N), k_on=mod["k_on"], k_off=float(k_off), r_s=float(r_s), L=float(mod["L"]),
             m=int(m_cells), t_window=float(t_window), n_survivors=int(n_survivors),
             seed=int(s["seed"]))

    def compute():
        kept, tried, batch = [], 0, max(4 * p["n_survivors"], 50)
        while len(kept) < p["n_survivors"] and tried < max_tries_factor * p["n_survivors"]:
            out = simulation.run_trajectories(
                "coverage", (p["N"], p["k_on"], p["k_off"], p["r_s"], p["L"], p["m"], p["t_window"]),
                batch, p["seed"] + tried, s["n_workers"])
            tried += batch
            kept += [v for v, ok in out if ok]
        kept = kept[: p["n_survivors"]]
        return pd.DataFrame({"cells_visited": kept}), {"n_tried": tried,
                                                       "survival_fraction": len(kept) / tried}

    return _cached("coverage", p, ["N", "k_off", "r_s"], compute, resimulate)
