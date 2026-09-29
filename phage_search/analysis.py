"""Turning raw simulation output into the quantities shown in the figures."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress


# -----------------------------------------------------------------------------
#  MSD -> D_eff  (Fig 1c, Fig S3a)
# -----------------------------------------------------------------------------
def mean_msd(r2_all):
    """Ensemble mean, SEM and number of contributing trajectories at each time."""
    r2_all = np.asarray(r2_all, dtype=float)
    n = np.sum(np.isfinite(r2_all), axis=0)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(r2_all, axis=0)
        sem = np.nanstd(r2_all, axis=0, ddof=1) / np.sqrt(n)
    return mean, sem, n


def bootstrap_msd_curves(r2_all, n_bootstrap=20, fraction=0.1, seed=0):
    """Mean MSD curves of `n_bootstrap` resamples, each using `fraction` of the trajectories."""
    rng = np.random.default_rng(seed)
    r2_all = np.asarray(r2_all, dtype=float)
    size = max(1, int(np.ceil(fraction * len(r2_all))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.array([np.nanmean(r2_all[rng.choice(len(r2_all), size)], axis=0)
                         for _ in range(n_bootstrap)])


def diffusive_window(t, msd):
    """Time window where the MSD grows linearly (local log-log slope ~ 1)."""
    ok = np.isfinite(t) & np.isfinite(msd) & (t > 0) & (msd > 0)
    t, msd = t[ok], msd[ok]
    if len(t) < 100:
        return None
    lt, lm = np.log10(t), np.log10(msd)
    w = max(50, len(t) // 20)
    step = max(5, w // 10)
    centres, slopes, r2 = [], [], []
    for s in range(0, len(t) - w, step):
        res = linregress(lt[s:s + w], lm[s:s + w])
        centres.append(10 ** lt[s:s + w].mean())
        slopes.append(res.slope)
        r2.append(res.rvalue**2)
    centres, slopes, r2 = map(np.asarray, (centres, slopes, r2))
    good = (np.abs(slopes - 1) < 0.15) & (r2 > 0.95)
    if not good.any():
        good = (np.abs(slopes - 1) < 0.3) & (r2 > 0.9)
    if not good.any():
        return t[len(t) // 4], t[3 * len(t) // 4]
    start = max(centres[good].min() * 0.8, t[len(t) // 10])
    end = min(centres[good].max() * 1.2, t[-len(t) // 10])
    return start, end


def fit_D(t, msd):
    """Fit MSD = 4 D t (+ C) in the diffusive window; returns (D, info)."""
    t = np.asarray(t, dtype=float)
    msd = np.asarray(msd, dtype=float)
    win = diffusive_window(t, msd)
    if win is None:
        return np.nan, {"error": "too few valid points"}
    m = (t >= win[0]) & (t <= win[1]) & np.isfinite(msd)
    if m.sum() < 30:
        return np.nan, {"error": "too few points in diffusive window"}
    tf, yf = t[m], msd[m]
    fits = []
    for name, f, p0 in (("no_intercept", lambda x, D: 4 * D * x, [1e-3]),
                        ("with_intercept", lambda x, D, C: 4 * D * x + C, [1e-3, 0.0])):
        popt, _ = curve_fit(f, tf, yf, p0=p0)
        pred = f(tf, *popt)
        r2 = 1 - np.sum((yf - pred) ** 2) / np.sum((yf - yf.mean()) ** 2)
        fits.append((r2, popt[0], name))
    r2, D, name = max(fits)
    info = {"t_start": win[0], "t_end": win[1], "r_squared": r2, "method": name}
    return (D if D > 0 else np.nan), info


def D_with_error(t, r2_all, n_bootstrap=20, fraction=0.1, seed=0):
    """D_eff from the bootstrap MSD curves: mean and standard error (Fig 1c error bars)."""
    boots = bootstrap_msd_curves(r2_all, n_bootstrap, fraction, seed)
    Ds = np.array([fit_D(t, b)[0] for b in boots])
    Ds = Ds[np.isfinite(Ds)]
    if len(Ds) == 0:
        return np.nan, np.nan
    return Ds.mean(), Ds.std(ddof=1) / np.sqrt(len(Ds)) if len(Ds) > 1 else np.nan


# -----------------------------------------------------------------------------
#  Adsorption times (Figs 4, 6, S2, S3b)
# -----------------------------------------------------------------------------
def adsorption_times(surface_time, n_detach, eta_B, p_r, seed=0, from_bulk=False):
    """Add the bulk-search time between surface visits to simulated runs.

    After each complete detachment the phage returns to the same cell with
    probability p_r (no delay); otherwise it spends an Exp(eta_B) time in the
    bulk. With `from_bulk=True` an initial bulk search is added as well
    (<T_tot> in the paper).
    """
    rng = np.random.default_rng(seed)
    surface_time = np.asarray(surface_time, dtype=float)
    n_detach = np.asarray(n_detach, dtype=int)
    n_bulk = rng.binomial(n_detach, 1.0 - p_r) + (1 if from_bulk else 0)
    bulk = np.zeros_like(surface_time)
    pos = n_bulk > 0
    bulk[pos] = rng.gamma(shape=n_bulk[pos], scale=1.0 / eta_B)
    return surface_time + bulk


def summary_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {k: np.nan for k in ("n", "mean", "sem", "median", "p10", "p25", "p75", "p90")}
    sem = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    p10, p25, med, p75, p90 = np.percentile(x, [10, 25, 50, 75, 90])
    return {"n": len(x), "mean": x.mean(), "sem": sem, "median": med,
            "p10": p10, "p25": p25, "p75": p75, "p90": p90}


def pdf_with_errors(x, edges):
    """Histogram density with binomial standard errors."""
    x = np.asarray(x, dtype=float)
    counts, edges = np.histogram(x, bins=edges)
    width = np.diff(edges)
    n = len(x)
    p = counts / n
    dens = p / width
    err = np.sqrt(p * (1 - p) / n) / width
    return 0.5 * (edges[1:] + edges[:-1]), dens, err
