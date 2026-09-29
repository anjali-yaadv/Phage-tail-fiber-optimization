"""Stochastic (Gillespie) simulation of a multivalent phage walking on a cell surface.

Model (paper, "Stochastic Model Framework")
--------------------------------------------
* The phage has N tail fibers. An attached fiber detaches at rate k_off; a
  detached fiber reattaches at rate k_on (only while at least one fiber is
  still attached).
* The phage centre of mass (COM) is the mean position of the attached fibers.
* A reattaching fiber lands inside the angular sector between its two
  neighbouring attached fibers (neighbours in the ring of fibers), at a
  distance drawn uniformly from [0, r_s] around the COM. With only one other
  fiber attached, the sector is the full circle.
* The surface is an L x L patch with periodic boundaries. Receptors are
  non-overlapping discs of radius r_t; adsorption happens when the COM enters one.
* A search starts with one fiber attached at a random position. When all
  fibers are detached, the surface cycle ends ("complete detachment").

All functions are written so that numba can compile them. numba is optional:
without it the same code runs as plain Python (identical results, much slower).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - fallback path
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

# outcome codes of one surface cycle
TIMEOUT, ADSORBED, DETACHED = 0, 1, 2


# -----------------------------------------------------------------------------
#  Geometry helpers
# -----------------------------------------------------------------------------
@njit(cache=True)
def _delta(d, L):
    """Minimum-image displacement on a periodic box (L <= 0 means an open plane)."""
    if L > 0.0:
        return d - L * np.round(d / L)
    return d


@njit(cache=True)
def _wrap(x, L):
    if L > 0.0:
        return x % L
    return x


@njit(cache=True)
def center_of_mass(x, y, att, L):
    """COM of the attached fibers (att == 1).

    On a periodic box the positions are averaged as minimum-image offsets from
    the first attached fiber, which is exact while the fibers span less than L/2.
    """
    ref = -1
    for i in range(att.size):
        if att[i] == 1:
            ref = i
            break
    if ref < 0:
        return np.nan, np.nan
    sx = 0.0
    sy = 0.0
    n = 0
    for i in range(att.size):
        if att[i] == 1:
            sx += _delta(x[i] - x[ref], L)
            sy += _delta(y[i] - y[ref], L)
            n += 1
    return _wrap(x[ref] + sx / n, L), _wrap(y[ref] + sy / n, L)


@njit(cache=True)
def reattach_fiber(i, x, y, att, r_s, L):
    """Reattach detached fiber i inside the sector between its attached neighbours."""
    n = att.size
    left = (i - 1) % n
    while left != i and att[left] == 0:
        left = (left - 1) % n
    if left == i:          # no other fiber attached: cannot reattach
        return False
    right = (i + 1) % n
    while att[right] == 0:
        right = (right + 1) % n

    cx, cy = center_of_mass(x, y, att, L)
    if left == right:
        a0 = 0.0
        width = 2.0 * np.pi
    else:
        a_left = np.arctan2(_delta(y[left] - cy, L), _delta(x[left] - cx, L))
        a_right = np.arctan2(_delta(y[right] - cy, L), _delta(x[right] - cx, L))
        a0 = a_left
        width = (a_right - a_left) % (2.0 * np.pi)

    theta = a0 + width * np.random.random()
    r = r_s * np.random.random()
    x[i] = _wrap(cx + r * np.cos(theta), L)
    y[i] = _wrap(cy + r * np.sin(theta), L)
    att[i] = 1
    return True


@njit(cache=True)
def place_targets(n_targets, L, r_t):
    """Random, non-overlapping receptor centres on the periodic patch."""
    xt = np.empty(n_targets)
    yt = np.empty(n_targets)
    k = 0
    attempts = 0
    while k < n_targets and attempts < 100000:
        attempts += 1
        xn = np.random.random() * L
        yn = np.random.random() * L
        ok = True
        for j in range(k):
            dx = _delta(xn - xt[j], L)
            dy = _delta(yn - yt[j], L)
            if dx * dx + dy * dy < (2.0 * r_t) ** 2:
                ok = False
                break
        if ok:
            xt[k] = xn
            yt[k] = yn
            k += 1
    return xt[:k], yt[:k]


@njit(cache=True)
def _on_target(cx, cy, xt, yt, r, L):
    for k in range(xt.size):
        dx = _delta(cx - xt[k], L)
        dy = _delta(cy - yt[k], L)
        if dx * dx + dy * dy < r * r:
            return True
    return False


# -----------------------------------------------------------------------------
#  Gillespie building blocks
# -----------------------------------------------------------------------------
@njit(cache=True)
def _init_one_attached(N, x0, y0):
    x = np.zeros(N)
    y = np.zeros(N)
    att = np.zeros(N, dtype=np.int64)
    i = np.random.randint(0, N)
    att[i] = 1
    x[i] = x0
    y[i] = y0
    return x, y, att


@njit(cache=True)
def _rates(att, k_on, k_off, protect_last):
    n_att = 0
    for i in range(att.size):
        n_att += att[i]
    rate_off = k_off * n_att
    if protect_last and n_att == 1:
        rate_off = 0.0     # last fiber may not detach (pure surface search)
    rate_on = k_on * (att.size - n_att) if n_att > 0 else 0.0
    return rate_off, rate_on, n_att


@njit(cache=True)
def _apply_event(x, y, att, rate_off, rate_on, n_att, r_s, L):
    """Execute one detachment or reattachment; return the new number attached."""
    if np.random.random() * (rate_off + rate_on) < rate_off:
        k = np.random.randint(0, n_att)
        for i in range(att.size):
            if att[i] == 1:
                if k == 0:
                    att[i] = 0
                    return n_att - 1
                k -= 1
    else:
        k = np.random.randint(0, att.size - n_att)
        for i in range(att.size):
            if att[i] == 0:
                if k == 0:
                    reattach_fiber(i, x, y, att, r_s, L)
                    return n_att + 1
                k -= 1
    return n_att


# -----------------------------------------------------------------------------
#  1. MSD trajectory (Fig 1b, c; Fig S3a) - open plane, no receptors
# -----------------------------------------------------------------------------
@njit(cache=True)
def msd_trajectory(N, k_on, k_off, r_s, t_max, n_points, seed):
    """Squared COM displacement r^2(t) on sample times linspace(0, t_max, n_points).

    Starts with one fiber attached. If all fibers detach, the remaining samples
    are NaN (the ensemble MSD is then averaged over attached trajectories).
    Returns (times, r2, detach_time) with detach_time = inf if never detached.
    """
    np.random.seed(seed)
    times = np.linspace(0.0, t_max, n_points)
    r2 = np.full(n_points, np.nan)
    x, y, att = _init_one_attached(N, 0.0, 0.0)
    cx, cy = center_of_mass(x, y, att, 0.0)
    x0, y0 = cx, cy
    r2[0] = 0.0
    idx = 1
    t = 0.0
    while idx < n_points:
        rate_off, rate_on, n_att = _rates(att, k_on, k_off, False)
        total = rate_off + rate_on
        t_next = t + np.random.exponential(1.0 / total)
        # the configuration is constant on [t, t_next): record those samples
        while idx < n_points and times[idx] < t_next:
            r2[idx] = (cx - x0) ** 2 + (cy - y0) ** 2
            idx += 1
        t = t_next
        n_att = _apply_event(x, y, att, rate_off, rate_on, n_att, r_s, 0.0)
        if n_att == 0:
            return times, r2, t
        cx, cy = center_of_mass(x, y, att, 0.0)
    return times, r2, np.inf


# -----------------------------------------------------------------------------
#  2. Hit time without detachment (Fig 3a)
# -----------------------------------------------------------------------------
@njit(cache=True)
def hit_time(N, k_on, k_off, r_s, r_t, L, n_targets, t_max, seed):
    """Time for the COM to reach a receptor when the phage cannot fully detach."""
    np.random.seed(seed)
    xt, yt = place_targets(n_targets, L, r_t)
    x, y, att = _init_one_attached(N, np.random.random() * L, np.random.random() * L)
    cx, cy = center_of_mass(x, y, att, L)
    if _on_target(cx, cy, xt, yt, r_t, L):
        return 0.0
    t = 0.0
    while t < t_max:
        rate_off, rate_on, n_att = _rates(att, k_on, k_off, True)
        t += np.random.exponential(1.0 / (rate_off + rate_on))
        _apply_event(x, y, att, rate_off, rate_on, n_att, r_s, L)
        cx, cy = center_of_mass(x, y, att, L)
        if _on_target(cx, cy, xt, yt, r_t, L):
            return t
    return np.inf


# -----------------------------------------------------------------------------
#  3. Complete-detachment time (Fig 3b)
# -----------------------------------------------------------------------------
@njit(cache=True)
def detachment_time(N, k_on, k_off, t_max, seed):
    """Time until all N fibers are detached, starting from one attached fiber.

    Complete detachment depends only on the number of attached fibers (a
    birth-death process with rates n k_off and (N - n) k_on), so fiber
    positions are not needed here.
    """
    np.random.seed(seed)
    n = 1
    t = 0.0
    while t < t_max:
        rate_off = k_off * n
        rate_on = k_on * (N - n)
        total = rate_off + rate_on
        t += np.random.exponential(1.0 / total)
        if np.random.random() * total < rate_off:
            n -= 1
            if n == 0:
                return t
        else:
            n += 1
    return np.inf


# -----------------------------------------------------------------------------
#  4. Full adsorption process (Figs 4a, 4b, 6, S2, S3b)
# -----------------------------------------------------------------------------
@njit(cache=True)
def surface_cycle(N, k_on, k_off, r_s, L, xt, yt, r_target_by_n, t_max):
    """One visit to a cell surface, starting from a random landing point.

    r_target_by_n[n] is the capture radius when n fibers are attached
    (all equal to r_t for a fixed receptor size; r_eff(n) with thermal
    fluctuations). Returns (outcome, duration).
    """
    x, y, att = _init_one_attached(N, np.random.random() * L, np.random.random() * L)
    cx, cy = center_of_mass(x, y, att, L)
    if _on_target(cx, cy, xt, yt, r_target_by_n[1], L):
        return ADSORBED, 0.0          # direct landing on a receptor
    t = 0.0
    while True:
        rate_off, rate_on, n_att = _rates(att, k_on, k_off, False)
        t += np.random.exponential(1.0 / (rate_off + rate_on))
        if t >= t_max:
            return TIMEOUT, t_max
        n_att = _apply_event(x, y, att, rate_off, rate_on, n_att, r_s, L)
        if n_att == 0:
            return DETACHED, t
        cx, cy = center_of_mass(x, y, att, L)
        if _on_target(cx, cy, xt, yt, r_target_by_n[n_att], L):
            return ADSORBED, t


@njit(cache=True)
def adsorption_run(N, k_on, k_off, r_s, L, n_targets, r_t, r_target_by_n, t_max, seed):
    """Repeat surface cycles until adsorption.

    After each complete detachment the phage lands again at a random position
    (receptors are re-drawn, which is equivalent). The time spent in the bulk
    between cycles is NOT simulated here; it is added afterwards from eta_B and
    p_r (see `analysis.adsorption_times`), so eta_B can be changed without
    re-running the simulation.

    Returns (adsorbed, surface_time, n_detachments, first_cycle_time, first_cycle_outcome).
    """
    np.random.seed(seed)
    surface_time = 0.0
    n_detach = 0
    first_time = 0.0
    first_outcome = TIMEOUT
    while surface_time < t_max:
        xt, yt = place_targets(n_targets, L, r_t)
        outcome, dt = surface_cycle(N, k_on, k_off, r_s, L, xt, yt, r_target_by_n,
                                    t_max - surface_time)
        if n_detach == 0 and first_outcome == TIMEOUT:
            first_time = dt
            first_outcome = outcome
        surface_time += dt
        if outcome == ADSORBED:
            return True, surface_time, n_detach, first_time, first_outcome
        if outcome == TIMEOUT:
            break
        n_detach += 1
    return False, surface_time, n_detach, first_time, first_outcome


# -----------------------------------------------------------------------------
#  5. Surface coverage on a lattice (Fig S4)
# -----------------------------------------------------------------------------
@njit(cache=True)
def lattice_coverage(N, k_on, k_off, r_s, L, m, t_window, seed):
    """Number of distinct cells of an m x m lattice visited by the COM in t_window.

    Starts with two attached fibers (one at the origin, one within reach) so
    that the reattachment sector is defined. Returns (cells_visited, survived);
    survived is False if the phage fully detached within the window.
    """
    np.random.seed(seed)
    x = np.zeros(N)
    y = np.zeros(N)
    att = np.zeros(N, dtype=np.int64)
    att[0] = 1
    att[1] = 1
    theta = 2.0 * np.pi * np.random.random()
    r = r_s * np.random.random()
    x[1] = _wrap(r * np.cos(theta), L)
    y[1] = _wrap(r * np.sin(theta), L)
    visited = np.zeros((m, m), dtype=np.int64)
    cell = L / m

    cx, cy = center_of_mass(x, y, att, L)
    visited[min(int(cx / cell), m - 1), min(int(cy / cell), m - 1)] = 1
    t = 0.0
    while True:
        rate_off, rate_on, n_att = _rates(att, k_on, k_off, False)
        t += np.random.exponential(1.0 / (rate_off + rate_on))
        if t >= t_window:
            break
        n_att = _apply_event(x, y, att, rate_off, rate_on, n_att, r_s, L)
        if n_att == 0:
            return int(visited.sum()), False
        cx, cy = center_of_mass(x, y, att, L)
        visited[min(int(cx / cell), m - 1), min(int(cy / cell), m - 1)] = 1
    return int(visited.sum()), True


# -----------------------------------------------------------------------------
#  Running many trajectories (optionally in parallel)
# -----------------------------------------------------------------------------
_KERNELS = {
    "msd": msd_trajectory,
    "hit": hit_time,
    "detach": detachment_time,
    "adsorption": adsorption_run,
    "coverage": lattice_coverage,
}


def _run_chunk(kind, args, seeds):
    fn = _KERNELS[kind]
    return [fn(*args, int(s)) for s in seeds]


def run_trajectories(kind, args, n_traj, base_seed, n_workers=0):
    """Run `n_traj` independent trajectories of simulation `kind`.

    `args` are the arguments of the kernel without the final `seed`; trajectory
    j uses seed base_seed + j, so every run is reproducible.
    """
    seeds = np.arange(base_seed, base_seed + n_traj)
    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    if n_workers == 1 or n_traj < 4 * n_workers:
        return _run_chunk(kind, args, seeds)
    chunks = np.array_split(seeds, 4 * n_workers)
    out = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(_run_chunk, [kind] * len(chunks), [args] * len(chunks), chunks):
            out.extend(res)
    return out
