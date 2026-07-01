import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import csv
import pandas as pd
from scipy.optimize import curve_fit
import json
from pathlib import Path

# ----------------------------
# Helpers for saving/structure
# ----------------------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def make_tag(n_legs, k_off, eta_B):
    return f"legs_{n_legs}_koff_{k_off:.4f}_etaB_{eta_B:.2e}"

def compute_histogram(series, bins=50):
    series = np.asarray(series, dtype=float)
    if series.size == 0:
        return np.array([]), np.array([])
    hist, edges = np.histogram(series, bins=bins, density=True)
    centers = 0.5*(edges[:-1] + edges[1:])
    return centers, hist

def open_loggers(base_dir, tag):
    """Open CSV writers for sims/cycles/events; return (handles, writers)."""
    out_dir = os.path.join(base_dir, tag)
    ensure_dir(out_dir)

    sims_f   = open(os.path.join(out_dir, "sims.csv"),   "w", newline="")
    cycles_f = open(os.path.join(out_dir, "cycles.csv"), "w", newline="")
    events_f = open(os.path.join(out_dir, "events.csv"), "w", newline="")

    sims_w   = csv.writer(sims_f)
    cycles_w = csv.writer(cycles_f)
    events_w = csv.writer(events_f)

    sims_w.writerow(["sim_id","win","total_time","n_cycles","n_detach_events","n_win_events"])
    cycles_w.writerow(["sim_id","cycle_id","outcome","cycle_time"])
    events_w.writerow(["sim_id","cycle_id","event_type","duration","cum_time"])

    return (sims_f, cycles_f, events_f), (sims_w, cycles_w, events_w)

def close_loggers(handles):
    for f in handles:
        f.close()

def save_required_data(script_dir, folder_name, params, results, bins=50):
    """
    Saves per-condition, analysis-ready files:
      - events_long.csv     (one row per event type + time; flattened)
      - distributions.csv   (histograms for detach, win_cycle, total_win)
      - manifest.json       (parameters + metadata)
    Also appends/creates:
      - summary.csv         (one row per param combo across all runs)
    """
    out_dir = os.path.join(script_dir, folder_name, make_tag(params['n_legs'], params['k_off'], params['eta_B']))
    ensure_dir(out_dir)

    # 1) Long-format events (flattened legacy-friendly)
    events_path = os.path.join(out_dir, "events_long.csv")
    with open(events_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["event_type", "time"])
        for t in results['all_detach_times']:
            w.writerow(["detach", t])
        for t in results['all_win_times']:
            w.writerow(["win_cycle", t])
        for t in results['total_win_times']:
            w.writerow(["total_win", t])

    # 2) Histograms
    c_det, h_det = compute_histogram(results['all_detach_times'], bins=bins)
    c_win, h_win = compute_histogram(results['all_win_times'], bins=bins)
    c_tot, h_tot = compute_histogram(results['total_win_times'], bins=bins)

    dists_path = os.path.join(out_dir, "distributions.csv")
    with open(dists_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["kind", "bin_center", "density"])
        for x, y in zip(c_det, h_det):
            w.writerow(["detach", x, y])
        for x, y in zip(c_win, h_win):
            w.writerow(["win_cycle", x, y])
        for x, y in zip(c_tot, h_tot):
            w.writerow(["total_win", x, y])

    # 3) Append compact summary
    row = [
        params['n_legs'], params['k_off'], params['eta_B'],
        results['win_rate'], results['mean_total_win_time'],
        results['mean_detach_time'], results['mean_win_time'],
        results['mean_cycles'], results['wins'],
        len(results['all_detach_times']), len(results['all_win_times'])
    ]
    summary_header = [
        'n_legs', 'k_off', 'eta_B', 'win_rate', 'mean_total_win_time',
        'mean_detach_time', 'mean_win_time', 'mean_cycles', 'total_wins',
        'n_detach_events', 'n_win_events'
    ]
    summary_dir = os.path.join(script_dir, folder_name)
    ensure_dir(summary_dir)
    summary_file = os.path.join(summary_dir, "summary.csv")
    file_exists = os.path.exists(summary_file)

    with open(summary_file, 'a', newline='') as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(summary_header)
        w.writerow(row)

    # 4) Manifest
    manifest = {
        "n_legs": params['n_legs'],
        "k_off": params['k_off'],
        "eta_B": params['eta_B'],
        "radius": params.get('radius', None),
        "n_targets": params.get('n_targets', None),
        "max_sim_time": params.get('max_sim_time', None),
        "num_simulations": params.get('num_simulations', None),
        "seed": params.get('seed', None),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

# ========== CORE SIM FUNCTIONS ==========
@njit
def wrap(x, L):
    return x % L

@njit
def periodic_com(x, y, state, L):
    mask = state == 0
    count = np.sum(mask)
    if count == 0:
        return 0.0, 0.0
    x_phase = np.exp(2j * np.pi * x[mask] / L)
    y_phase = np.exp(2j * np.pi * y[mask] / L)
    x_mean = (np.angle(np.mean(x_phase)) / (2 * np.pi)) * L
    y_mean = (np.angle(np.mean(y_phase)) / (2 * np.pi)) * L
    return wrap(x_mean, L), wrap(y_mean, L)

@njit
def pbc_dist(x1, x2, L):
    delta = x1 - x2
    return delta - L * np.round(delta / L)

@njit
def reattach_leg(i, x, y, state, r_min, r_max, L):
    n_legs = len(state)
    left = (i - 1) % n_legs
    while left != i and state[left] == 1:
        left = (left - 1) % n_legs
    right = (i + 1) % n_legs
    while right != i and state[right] == 1:
        right = (right + 1) % n_legs

    # If both neighbors are i, no other legs are attached -> cannot reattach
    if left == i or right == i:
        return False

    com_x, com_y = periodic_com(x, y, state, L)
    dx_left = pbc_dist(x[left], com_x, L)
    dy_left = pbc_dist(y[left], com_y, L)
    dx_right = pbc_dist(x[right], com_x, L)
    dy_right = pbc_dist(y[right], com_y, L)

    angle_left = np.arctan2(dy_left, dx_left)
    angle_right = np.arctan2(dy_right, dx_right)

    if left == right:
        ang_min, ang_max = 0.0, 2 * np.pi
    else:
        delta = (angle_right - angle_left) % (2 * np.pi)
        ang_min, ang_max = angle_left, angle_left + delta

    theta = np.random.uniform(ang_min, ang_max)
    r = np.random.uniform(r_min, r_max)

    x[i] = wrap(com_x + r * np.cos(theta), L)
    y[i] = wrap(com_y + r * np.sin(theta), L)
    state[i] = 0
    return True

@njit
def init_hexapod(n_legs, radius, k_on, k_off, L):
    x = np.zeros(n_legs, dtype=np.float64)
    y = np.zeros(n_legs, dtype=np.float64)
    state = np.ones(n_legs, dtype=np.int32)
    attached_leg = np.random.randint(0, n_legs)
    x[attached_leg] = np.random.uniform(0, L)
    y[attached_leg] = np.random.uniform(0, L)
    state[attached_leg] = 0
    return x, y, state

@njit
def generate_targets(n_targets, L, min_dist):
    x_target = np.empty(n_targets, dtype=np.float64)
    y_target = np.empty(n_targets, dtype=np.float64)
    for i in range(n_targets):
        attempts = 0
        while attempts < 1000:
            x_new = np.random.uniform(0, L)
            y_new = np.random.uniform(0, L)
            too_close = False
            for j in range(i):
                dx = x_new - x_target[j]
                dy = y_new - y_target[j]
                if dx*dx + dy*dy < min_dist*min_dist:
                    too_close = True
                    break
            if not too_close:
                x_target[i] = x_new
                y_target[i] = y_new
                break
            attempts += 1
    return x_target, y_target

@njit
def single_surface_search(n_legs, radius, k_on, k_off, r_min, r_max, x, y, state, L,
                          x_target, y_target, n_targets, target_radius, max_time):
    """
    Run one cycle on a single bacterial surface.
    Returns: (outcome, cycle_time, n_events)
      outcome: 0=timeout, 1=win_cycle, 2=detach
    """
    t = 0.0
    n_events = 0
    while t < max_time:
        rates = np.empty(n_legs, dtype=np.float64)
        for i in range(n_legs):
            rates[i] = k_off if state[i] == 0 else k_on
        a_total = np.sum(rates)
        if a_total == 0:
            break
        dt = np.random.exponential(1 / a_total)
        t += dt
        n_events += 1

        # pick event
        rv = np.random.rand() * a_total
        cumsum = 0.0
        chosen = -1
        for i in range(n_legs):
            cumsum += rates[i]
            if rv <= cumsum:
                chosen = i
                break

        if state[chosen] == 0:
            # detach this leg
            state[chosen] = 1
        else:
            # attempt attach if any leg attached remains
            if np.any(state == 0):
                ok = reattach_leg(chosen, x, y, state, r_min, r_max, L)
                if not ok:
                    return 2, t, n_events  # complete detachment

        if np.all(state == 1):
            return 2, t, n_events  # all detached

        # win check
        if np.any(state == 0):
            com_x, com_y = periodic_com(x, y, state, L)
            for i in range(n_targets):
                dx = pbc_dist(com_x, x_target[i], L)
                dy = pbc_dist(com_y, y_target[i], L)
                if dx*dx + dy*dy < target_radius*target_radius:
                    return 1, t, n_events  # win
    return 0, t, n_events  # timeout

# --------- RUN ONE SIM WITH STREAMING LOGGING ----------
def run_one_sim_and_log(sim_id,
                        n_legs, radius, k_on, k_off, eta_B,
                        L, n_targets, target_radius,
                        max_sim_time,
                        sims_writer, cycles_writer, events_writer):
    """
    Returns: (win, total_time, n_cycles, n_detach_events, n_win_events, 
              detach_times_list, win_times_list)
    """
    total_time = 0.0
    n_cycles = 0
    n_detach_events = 0
    n_win_events = 0
    
    # NEW: Track times for this simulation
    detach_times_list = []
    win_times_list = []

    min_dist = 2 * target_radius
    x_target, y_target = generate_targets(n_targets, L, min_dist)

    while total_time < max_sim_time:
        n_cycles += 1
        cycle_id = n_cycles

        x, y, state = init_hexapod(n_legs, radius, k_on, k_off, L)
        outcome, cycle_time, _ = single_surface_search(
            n_legs, radius, k_on, k_off, 0.0, 2.0*radius,
            x, y, state, L, x_target, y_target, n_targets, target_radius,
            max_sim_time - total_time
        )
        total_time += cycle_time

        label = "timeout" if outcome == 0 else ("win_cycle" if outcome == 1 else "detach")
        cycles_writer.writerow([sim_id, cycle_id, label, cycle_time])

        # NEW: Record the cycle time
        if outcome == 1:  # win_cycle
            n_win_events += 1
            win_times_list.append(cycle_time)
            sims_writer.writerow([sim_id, True, total_time, n_cycles, n_detach_events, n_win_events])
            return True, total_time, n_cycles, n_detach_events, n_win_events, detach_times_list, win_times_list

        elif outcome == 2:  # detach
            n_detach_events += 1
            detach_times_list.append(cycle_time)
            x_target, y_target = generate_targets(n_targets, L, min_dist)

        else:  # timeout
            sims_writer.writerow([sim_id, False, total_time, n_cycles, n_detach_events, n_win_events])
            return False, total_time, n_cycles, n_detach_events, n_win_events, detach_times_list, win_times_list

    sims_writer.writerow([sim_id, False, total_time, n_cycles, n_detach_events, n_win_events])
    return False, total_time, n_cycles, n_detach_events, n_win_events, detach_times_list, win_times_list

# --------- PARAMETER ANALYSIS + LEGACY SAVES ----------
def run_parameter_analysis_with_distributions(n_legs, k_off, num_simulations, radius, n_targets, target_radius, eta_B,
                                              script_dir, folder_name, max_sim_time=500000):
    """
    Runs many simulations, writes structured logs (sims/cycles/events) per condition,
    and returns aggregated arrays for legacy distributions file.
    """

    # Open structured loggers for this (n_legs, k_off, eta_B)
    base_dir = os.path.join(script_dir, folder_name)
    tag = make_tag(n_legs, k_off, eta_B)
    handles, writers = open_loggers(base_dir, tag)
    sims_w, cycles_w, events_w = writers

    # Collect for legacy distributions
    all_detach_times = []
    all_win_times = []
    total_win_times = []
    cycle_counts = []
    wins = 0

    print(f"Starting analysis for n_legs={n_legs}, k_off={k_off}")

    for sim in range(num_simulations):
        if sim % 200 == 0:
            print(f"  Simulation {sim}/{num_simulations}")

        win, total_time, n_cycles, n_detach_events, n_win_events, detach_list, win_list = run_one_sim_and_log(
            sim, n_legs, radius, k_on, k_off, eta_B, L, n_targets, target_radius,
            max_sim_time, sims_w, cycles_w, events_w
        )

        # NEW: Extend the arrays directly
        all_detach_times.extend(detach_list)
        all_win_times.extend(win_list)

        if win:
            wins += 1
            total_win_times.append(total_time)

        cycle_counts.append(n_cycles)

    # Close structured logs
    close_loggers(handles)


    # Summary stats
    win_rate = wins / num_simulations if num_simulations else 0.0
    mean_detach_time = float(np.mean(all_detach_times)) if all_detach_times else 0.0
    mean_win_time = float(np.mean(all_win_times)) if all_win_times else 0.0
    mean_total_win_time = float(np.mean(total_win_times)) if total_win_times else 0.0
    mean_cycles = float(np.mean(cycle_counts)) if cycle_counts else 0.0

    # Save legacy distributions for compatibility
    # --- Save separate time-distribution files (vertical format) ---
    dist_dir = os.path.join(script_dir, folder_name, "time_distributions")
    ensure_dir(dist_dir)

    # 1) DETACH TIMES (one column)
    detach_file = os.path.join(dist_dir, f"detach_times_legs_{n_legs}_koff_{k_off:.4f}_r_s_{radius:.2f}.csv")
    pd.DataFrame({"detach_time": all_detach_times}).to_csv(detach_file, index=False)

    # 2) WIN CYCLE TIMES (one column)
    win_file = os.path.join(dist_dir, f"win_times_legs_{n_legs}_koff_{k_off:.4f}_r_s_{radius:.2f}.csv")
    pd.DataFrame({"win_cycle_time": all_win_times}).to_csv(win_file, index=False)   

    # 3) TOTAL WIN TIMES (one column)
    total_file = os.path.join(dist_dir, f"total_win_times_legs_{n_legs}_koff_{k_off:.4f}_r_s_{radius:.2f}.csv")
    pd.DataFrame({"total_win_time": total_win_times}).to_csv(total_file, index=False)


    # Legacy per-sim outcome summary (sourced from sims.csv)
    outcomes_file = os.path.join(script_dir, f"{folder_name}/simulation_outcomes_legs_{n_legs}_koff_{k_off:.4f}_r_s_{radius:.2f}.csv")
    # We can copy sims.csv into this legacy filename
    sims_src = os.path.join(base_dir, tag, "sims.csv")
    with open(sims_src, "r") as src, open(outcomes_file, "w", newline="") as dst:
        dst.write(src.read())

    print(f"  Win rate: {win_rate:.3f}")
    print(f"  Detach events: {len(all_detach_times)}, Win events: {len(all_win_times)}")
    print(f"  Mean detach time: {mean_detach_time:.1f}, Mean win time: {mean_win_time:.1f}")

    # Pack results dict for save_required_data
    results = {
        'wins': wins,
        'all_detach_times': all_detach_times,
        'all_win_times': all_win_times,
        'total_win_times': total_win_times,
        'cycle_counts': cycle_counts,
        'simulation_outcomes': None,  # structured version is in sims/cycles/events
        'win_rate': win_rate,
        'mean_detach_time': mean_detach_time,
        'mean_win_time': mean_win_time,
        'mean_total_win_time': mean_total_win_time,
        'mean_cycles': mean_cycles,
    }
    return results


def _fit_exp_log_linear(times, bins=20):
    """Fit A*exp(-λt) by linear regression on log(hist)."""
    t = np.asarray(times, float)
    t = t[np.isfinite(t)]
    if len(t) < 3:
        return None
    # histogram in density mode
    hist, edges = np.histogram(t, bins=bins, density=True)
    centers = 0.5*(edges[:-1]+edges[1:])
    mask = hist > 0
    x, y = centers[mask], hist[mask]
    ly = np.log(y)
    # simple linear fit: log(y) = a - λ t
    A = np.vstack([x, np.ones_like(x)]).T
    coeff, *_ = np.linalg.lstsq(A, ly, rcond=None)
    slope, intercept = coeff
    lam = -slope
    A_fit = np.exp(intercept)
    # fitted curve
    t_fit = np.linspace(min(x), max(x), 200)
    y_fit = A_fit*np.exp(-lam*t_fit)
    return lam, A_fit, t_fit, y_fit


def plot_time_distributions(script_dir, folder_name, n_legs_list, k_off_list):
    """Plot detach/win time histograms with exponential fits."""
    fig, axes = plt.subplots(2, len(k_off_list), figsize=(5*len(k_off_list), 8))
    if len(k_off_list) == 1:
        axes = axes.reshape(-1, 1)

    for j, k_off in enumerate(k_off_list):
        for n_legs in n_legs_list:
            path = os.path.join(script_dir, f"{folder_name}/time_distributions_legs_{n_legs}_koff_{k_off:.4f}_r_s_{radius:.2f}.csv")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                rows = list(csv.reader(f))
            det = np.array([float(x) for x in rows[0][1:] if x])
            win = np.array([float(x) for x in rows[1][1:] if x])

            # --- DETACH ---
            if len(det) > 5:
                lam, A_fit, t_fit, y_fit = _fit_exp_log_linear(det, bins=25)
                axes[0, j].hist(det, bins=25, density=True, alpha=0.6, label=f'{n_legs} legs')
                axes[0, j].plot(t_fit, y_fit, '--', lw=2, label=f'λ={lam:.3f}, ⟨t⟩={1/lam:.2f}')

            # --- WIN ---
            if len(win) > 5:
                lam, A_fit, t_fit, y_fit = _fit_exp_log_linear(win, bins=25)
                axes[1, j].hist(win, bins=25, density=True, alpha=0.6, label=f'{n_legs} legs')
                axes[1, j].plot(t_fit, y_fit, '--', lw=2, label=f'λ={lam:.3f}, ⟨t⟩={1/lam:.2f}')

        axes[0, j].set_title(f"Detachment Times (k_off={k_off})")
        axes[0, j].set_xlabel("Detachment time")
        axes[0, j].set_ylabel("Probability density")
        axes[0, j].set_yscale('log')
        axes[0, j].legend()
        axes[0, j].grid(alpha=0.3)

        axes[1, j].set_title(f"Win Times (k_off={k_off})")
        axes[1, j].set_xlabel("Win time")
        axes[1, j].set_ylabel("Probability density")
        axes[1, j].set_yscale('log')
        axes[1, j].legend()
        axes[1, j].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()  # forces window to render before script ends
    #plt.close(fig)

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Parameters
    radius = 0.5
    k_on = 1
    L = 20
    n_targets = 1
    eta_B = 0.00001
    target_radius = 0.2

    # Simulation parameters
    num_simulations = 2000   # increase for final runs
    n_legs_array = [ 3, 4, 5,6, 7, 8]
    k_off_array = [0.1]
    max_sim_time = 20000000.0

    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = f"r_s_{radius}_r_t_{target_radius}_plotting_etaB_{eta_B:.2e}_n_targets_{n_targets}"
    ensure_dir(os.path.join(script_dir, folder_name))

    print(f"Running detailed phage analysis with time distributions")
    print(f"Parameters: eta_B = {eta_B:.2e}, n_targets = {n_targets}")
    print(f"Simulations per condition: {num_simulations}")

    all_results = []

    for n_legs in n_legs_array:
        for k_off in k_off_array:
            results = run_parameter_analysis_with_distributions(
                n_legs, k_off, num_simulations, radius, n_targets, target_radius,
                eta_B, script_dir, folder_name, max_sim_time=max_sim_time
            )

            # Save tidy, analysis-ready data per condition (manifest, flattened events, histograms, summary row)
            params = {
                'n_legs': n_legs,
                'k_off': k_off,
                'eta_B': eta_B,
                'radius': radius,
                'target_radius': target_radius,
                'n_targets': n_targets,
                'max_sim_time': max_sim_time,
                'num_simulations': num_simulations,
            }
            save_required_data(script_dir, folder_name, params, results, bins=50)

            # retain in-memory summary (optional)
            results['n_legs'] = n_legs
            results['k_off']  = k_off
            results['eta_B']  = eta_B
            all_results.append(results)

    # Global summary (legacy)
    summary_file = os.path.join(script_dir, f"{folder_name}/summary_results.csv")
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'n_legs', 'k_off', 'eta_B', 'win_rate', 'mean_total_win_time',
            'mean_detach_time', 'mean_win_time', 'mean_cycles', 'total_wins',
            'n_detach_events', 'n_win_events'
        ])
        for result in all_results:
            writer.writerow([
                result['n_legs'], result['k_off'], result['eta_B'],
                result['win_rate'], result['mean_total_win_time'],
                result['mean_detach_time'], result['mean_win_time'],
                result['mean_cycles'], result['wins'],
                len(result['all_detach_times']), len(result['all_win_times'])
            ])

    print(f"\nAnalysis complete. Results saved in {folder_name}/")

    # Quick-look plots (detach & win cycle PDFs)
    plot_time_distributions(script_dir, folder_name, n_legs_array, k_off_array)
