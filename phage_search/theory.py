"""Analytical results of the paper (equation numbers refer to the manuscript).

All functions take plain numbers/arrays; `params_from_config` collects the
values from the parameter file so that scripts can call e.g.

    T = mean_total_time(N, **params_from_config(cfg))
"""
from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
#  Building blocks
# -----------------------------------------------------------------------------
def D_eff_scaling(N, k_on, k_off, r_s, D0, r_s_ref=1.0, alpha=1.0):
    """Effective surface diffusion constant, D_eff ~ k_off r_s^2 / N^alpha (Fig 1c).

    D_eff = D0 (r_s/r_s_ref)^2 * k_on k_off / (k_on + k_off) * N^(-alpha)
    """
    N = np.asarray(N, dtype=float)
    return D0 * (r_s / r_s_ref) ** 2 * k_on * k_off / (k_on + k_off) * N ** (-alpha)


def mean_hit_time(D_eff, L, r_t, n_targets=1, C_torus=0.0):
    """Mean time for 2D diffusion on an L x L torus to reach a receptor.

    One receptor (Eq. 1, Supplement Eq. S5):
        <T_h> = L^2 / (2 pi D_eff) * [ln(L / r_t) + C_torus]
    n_t > 1 well-separated receptors (Supplement Eq. S7):
        <T_h> = L^2 / (4 pi D_eff n_t r_t^2) * [ln(L^2 / (n_t pi r_t^2)) + gamma - 1/4]
    """
    D_eff = np.asarray(D_eff, dtype=float)
    if n_targets == 1:
        return L**2 / (2.0 * np.pi * D_eff) * (np.log(L / r_t) + C_torus)
    phi = n_targets * np.pi * r_t**2 / L**2
    return L**2 / (4.0 * np.pi * D_eff * n_targets * r_t**2) * (np.log(1.0 / phi) + np.euler_gamma - 0.25)


def D_eff_for_comparison(N, k_on, k_off, r_s, D0, r_s_ref=1.0, alpha=1.0, table=None, **_):
    """D_eff used when theory is compared with simulations (Figs 3a, 4a, 4b, S3b).

    Uses the simulated value from `table` (data/processed/D_eff_vs_N.csv, the
    Fig 1c data) when it contains this (N, k_off, r_s); otherwise the scaling law.
    """
    if table is not None:
        row = table[(table.N == N) & np.isclose(table.k_off, k_off) & np.isclose(table.r_s, r_s)
                    & np.isclose(table.k_on, k_on)]
        if len(row):
            return float(row.D_eff.iloc[0])
    return float(D_eff_scaling(N, k_on, k_off, r_s, D0, r_s_ref, alpha))


def detachment_rate(N, k_on, k_off):
    """lambda_d = 1/<T_d>: complete detachment starting from one attached fiber (Eq. 2)."""
    N = np.asarray(N, dtype=float)
    return N * k_on / ((1.0 + k_on / k_off) ** N - 1.0)


def p_return(a_cell, r_t):
    """Probability p_r = a / (a + r_t) of returning to the same cell after detachment."""
    return a_cell / (a_cell + r_t)


def p_land(r_t, L, n_targets=1):
    """Probability p_l = n_t pi r_t^2 / L^2 of landing directly on a receptor."""
    return n_targets * np.pi * r_t**2 / L**2


# -----------------------------------------------------------------------------
#  Mean adsorption time (Eqs. 5, 6 and Supplement "Derivation of the mean adsorption time")
# -----------------------------------------------------------------------------
def mean_adsorption_time(N, k_on, k_off, eta_B, r_s, r_t, L, n_targets, a_cell,
                         D0, r_s_ref=1.0, alpha=1.0, C_torus=0.0,
                         direct_landing_after_return=True, D_eff=None, **_):
    """<T_ads(N)>: mean time from landing on a cell to adsorption.

    N >= 2 (general form of Eq. 5):
        <T_ads> = [1 + lambda_d (1 - p_r) / eta_B] / [lambda_h + p_l lambda_d]
    N = 1 (Eq. 6):
        <T_ads> = (1 / p_l) [1/k_off + (1 - p_r) / eta_B]

    `D_eff` may be given explicitly (e.g. measured values); otherwise the
    scaling law `D_eff_scaling` is used.

    direct_landing_after_return: if True (paper derivation, and what the
    simulation does), a phage that returns to the same cell can also land
    directly on a receptor, so the denominator contains p_l lambda_d. If False,
    direct landing is only possible after a bulk search, giving
    (1 - p_r) p_l lambda_d (the version used for the published Figs 4c,d, 5, S5).
    """
    N_arr = np.atleast_1d(np.asarray(N, dtype=float))
    pr = p_return(a_cell, r_t)
    pl = p_land(r_t, L, n_targets)
    if D_eff is None:
        D = D_eff_scaling(N_arr, k_on, k_off, r_s, D0, r_s_ref, alpha)
    else:
        D = np.broadcast_to(np.asarray(D_eff, dtype=float), N_arr.shape)
    lam_h = 1.0 / mean_hit_time(D, L, r_t, n_targets, C_torus)
    lam_d = detachment_rate(N_arr, k_on, k_off)
    pl_surface = pl if direct_landing_after_return else (1.0 - pr) * pl
    T = (1.0 + lam_d * (1.0 - pr) / eta_B) / (lam_h + pl_surface * lam_d)
    T1 = (1.0 / pl) * (1.0 / k_off + (1.0 - pr) / eta_B)
    T = np.where(N_arr == 1, T1, T)
    return T if np.ndim(N) else float(T[0])


def mean_total_time(N, eta_B, **kw):
    """<T_tot> = <T_ads> + 1/eta_B: starting from a free phage in the bulk (Figs 4c,d, 5)."""
    return mean_adsorption_time(N, eta_B=eta_B, **kw) + 1.0 / eta_B


def optimal_N(N_max, **kw):
    """N* minimising <T_tot> over integers 1..N_max (Fig 5)."""
    N = np.arange(1, N_max + 1)
    return int(N[np.argmin(mean_total_time(N, **kw))])


# -----------------------------------------------------------------------------
#  Full adsorption-time distribution from the renewal equation (Eq. 3, Fig 4a)
# -----------------------------------------------------------------------------
def renewal_distribution(t, lam_h, lam_d, eta_B, p_r, p_l, direct_landing_after_return=True):
    """Adsorption-time density P_ads(t) from the renewal model (Eq. 3), exact.

    Surface phase: competing exponentials,
        P_h(t) = lam_h exp(-(lam_h + lam_d) t),  P_d(t) = lam_d exp(-(lam_h + lam_d) t)
    After detachment: same cell with probability p_r (no delay), otherwise a
    bulk search P_s(t) = eta_B exp(-eta_B t); on landing, direct adsorption
    with probability p_l, otherwise the surface phase restarts:
        P_ads = P_h + P_d * Q * [p_l delta + (1 - p_l) P_ads],  Q = p_r delta + (1 - p_r) P_s
    (* = convolution). With direct_landing_after_return=False, direct landing
    is only possible after a bulk search (see `mean_adsorption_time`).

    All kernels are exponential, so the Laplace transform of P_ads is a ratio
    of a linear and a quadratic polynomial in s, and P_ads(t) is a sum of two
    exponentials (a fast surface-search mode and a slow mode involving bulk
    searches). They are computed here in closed form.
    """
    t = np.asarray(t, dtype=float)
    h, d, e = lam_h, lam_d, eta_B
    a = h + d
    q, l = p_r, p_l
    lr = p_l if direct_landing_after_return else 0.0
    # P_ads(s) = (n1 s + n0) / (s^2 + b1 s + b0)
    n1 = h + d * q * lr
    n0 = h * e + d * q * lr * e + d * (1 - q) * e * l
    b1 = a + e - d * q * (1 - lr)
    b0 = a * e - d * q * (1 - lr) * e - d * (1 - q) * e * (1 - l)
    disc = np.sqrt(b1 * b1 - 4.0 * b0)
    s2 = -0.5 * (b1 + disc)                 # fast root
    s1 = b0 / s2                            # numerically stable: s1 * s2 = b0
    c1 = (n1 * s1 + n0) / (s1 - s2)
    c2 = (n1 * s2 + n0) / (s2 - s1)
    return c1 * np.exp(s1 * t) + c2 * np.exp(s2 * t)


# -----------------------------------------------------------------------------
#  Thermal fluctuations of the COM (Eqs. reff1, reff; Supplement S1)
# -----------------------------------------------------------------------------
def tau_e(n, N, k_on, k_off):
    """Mean time until the number of attached fibers changes: 1/[(N - n) k_on + n k_off]."""
    n = np.asarray(n, dtype=float)
    return 1.0 / ((N - n) * k_on + n * k_off)


def r_exp(n, ell, M, D, tau):
    """RMS thermal displacement of the COM with n fibers attached (Eq. reff1).

    r_exp = sqrt( 2 M kappa^2 / (3 n) * (1 - exp(-6 n D tau / (M kappa^2))) ),  kappa = ell / M
    """
    n = np.asarray(n, dtype=float)
    kappa = ell / M
    Mk2 = M * kappa**2
    return np.sqrt(2.0 * Mk2 / (3.0 * n) * (1.0 - np.exp(-6.0 * n * D * tau / Mk2)))


def r_eff_by_n(N, k_on, k_off, r_s, r_t, M, ell_over_rs, D_3d):
    """Capture radius r_eff(n) = max(r_t, r_exp(n)) for n = 0..N attached fibers (Eq. reff).

    Index n of the returned array is the radius used when n fibers are
    attached (index 0 is never used by the simulation).
    """
    n = np.arange(1, N + 1)
    r = r_exp(n, ell_over_rs * r_s, M, D_3d, tau_e(n, N, k_on, k_off))
    return np.concatenate([[r_t], np.maximum(r_t, r)])


# -----------------------------------------------------------------------------
#  Convenience
# -----------------------------------------------------------------------------
def params_from_config(cfg, **changes):
    """Model + theory parameters as keyword arguments for the functions above."""
    p = dict(cfg["model"])
    p.update({k: cfg["theory"][k] for k in ("D0", "r_s_ref", "alpha", "C_torus",
                                             "direct_landing_after_return")})
    p.pop("N", None)
    p.update(changes)
    return p
