"""Quick consistency checks (run with:  python -m pytest tests/).

They compare the simulation with exact results and check the theory
functions against each other; together they take well under a minute.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phage_search import simulation, theory  # noqa: E402
from phage_search.config import load_config  # noqa: E402


def test_config_loads_and_overrides():
    cfg = load_config(overrides=["model.k_off=0.5", "figures.fig4.n_traj=10"])
    assert cfg["model"]["k_off"] == 0.5
    assert cfg["figures"]["fig4"]["n_traj"] == 10


def test_reattachment_stays_within_reach():
    np.random.seed(1)
    r_s, L = 1.0, 10.0
    x = np.array([5.0, 5.5, 0.0, 0.0])
    y = np.array([5.0, 5.2, 0.0, 0.0])
    att = np.array([1, 1, 0, 0])
    for _ in range(200):
        a = att.copy()
        xx, yy = x.copy(), y.copy()
        cx, cy = simulation.center_of_mass(xx, yy, a, L)
        assert simulation.reattach_fiber(2, xx, yy, a, r_s, L)
        assert np.hypot(xx[2] - cx, yy[2] - cy) <= r_s + 1e-12


def test_center_of_mass_across_periodic_boundary():
    x = np.array([9.9, 0.1])
    y = np.array([5.0, 5.0])
    cx, _ = simulation.center_of_mass(x, y, np.array([1, 1]), 10.0)
    assert min(cx, 10 - cx) < 1e-9          # midpoint is at the boundary, not at 5


def test_detachment_time_matches_birth_death_theory():
    for N, k_off in [(3, 1.0), (4, 0.5)]:
        t = np.array(simulation.run_trajectories("detach", (N, 1.0, k_off, 1e9), 4000, 7, 1))
        expected = 1.0 / theory.detachment_rate(N, 1.0, k_off)
        assert abs(t.mean() - expected) < 4 * t.std() / np.sqrt(len(t))


def test_renewal_distribution_is_normalised_and_has_eq5_mean():
    h, d, e, pr, pl = 7e-5, 2e-3, 1e-5, 100 / 100.2, np.pi * 0.04 / 100
    t = np.logspace(-3, 9, 200001)
    for flag in (True, False):
        P = theory.renewal_distribution(t, h, d, e, pr, pl, flag)
        assert abs(trapezoid(P, t) - 1) < 1e-3
        pl_s = pl if flag else (1 - pr) * pl
        T5 = (1 + d * (1 - pr) / e) / (h + pl_s * d)
        assert abs(trapezoid(P * t, t) / T5 - 1) < 1e-3


def test_single_fiber_formula():
    cfg = load_config()
    tp = theory.params_from_config(cfg)
    pr = theory.p_return(tp["a_cell"], tp["r_t"])
    pl = theory.p_land(tp["r_t"], tp["L"], tp["n_targets"])
    expected = (1 / pl) * (1 / tp["k_off"] + (1 - pr) / tp["eta_B"])
    assert np.isclose(theory.mean_adsorption_time(1, **tp), expected)


def test_r_exp_reproduces_supplement_S1():
    # T4-like phage: ell = 150 nm, M = 2, D = 4 um^2/s, tau_e = 1e-3 s / 6  ->  ~44 nm for n = 2
    r = theory.r_exp(2, 150.0, 2, 4e6, 1e-3 / 6)
    assert 40 < r < 47
