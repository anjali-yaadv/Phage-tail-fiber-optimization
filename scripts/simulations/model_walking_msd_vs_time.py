import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time

# ========== OPTIMIZED CORE FUNCTIONS ==========

@njit
def reattach_leg_fast(i, x, y, state, r_min, r_max):
    """Optimized reattachment - same logic but faster"""
    n_legs = len(state)
    
    # Find neighbors
    left = (i - 1) % n_legs
    while left != i and state[left] == 1:
        left = (left - 1) % n_legs
    
    right = (i + 1) % n_legs  
    while right != i and state[right] == 1:
        right = (right + 1) % n_legs
        
    if left == i or right == i:
        raise RuntimeError("No other leg attached")
    
    # COM calculation (excluding leg i)
    com_x = com_y = 0.0
    count = 0
    for j in range(n_legs):
        if j != i and state[j] == 0:
            com_x += x[j]
            com_y += y[j] 
            count += 1
    com_x /= count
    com_y /= count
    
    # Angles and wedge
    angle_left = np.arctan2(y[left] - com_y, x[left] - com_x)
    angle_right = np.arctan2(y[right] - com_y, x[right] - com_x)
    
    if left == right:
        ang_min, ang_max = 0.0, 2*np.pi
    else:
        delta = (angle_right - angle_left) % (2*np.pi)
        ang_min, ang_max = angle_left, angle_left + delta
    
    # Sample position
    theta = np.random.uniform(ang_min, ang_max)
    r = np.random.uniform(r_min, r_max)
    
    x[i] = com_x + r * np.cos(theta)
    y[i] = com_y + r * np.sin(theta)
    state[i] = 0

@njit
def init_equilibrium_fast(n_legs, radius, k_on, k_off):
    """Fast equilibrium initialization"""
    x = np.zeros(n_legs, dtype=np.float64)
    y = np.zeros(n_legs, dtype=np.float64)
    state = np.ones(n_legs, dtype=np.int32)
    
    # Equilibrium number of legs
    p_attach = k_on / (k_on + k_off)
    n_attached = np.random.binomial(n_legs, p_attach)
    if n_attached == 0:
        n_attached = 1  # Must have at least one
    
    # Place first leg at origin
    x[0] = 0.0
    y[0] = 0.0
    state[0] = 0
    
    # Attach remaining legs
    for i in range(1, min(n_attached, n_legs)):
        state[i] = 0
        reattach_leg_fast(i, x, y, state, 0, 2*radius)
    
    return x, y, state

@njit
def com_fast(x, y, state):
    """Fast COM calculation"""
    com_x = com_y = 0.0
    count = 0
    for i in range(len(state)):
        if state[i] == 0:
            com_x += x[i]
            com_y += y[i]
            count += 1
    if count > 0:
        return com_x / count, com_y / count
    return 0.0, 0.0

@njit
def gillespie_sim_optimized(n_legs, radius, k_on, k_off, sim_time, 
                           num_points, seed):
    """Optimized Gillespie simulation"""
    np.random.seed(seed)
    
    x, y, state = init_equilibrium_fast(n_legs, radius, k_on, k_off)
    
    t = 0.0
    detach_time = np.inf
    
    # Pre-allocate arrays for speed
    time_points = np.linspace(0, sim_time, num_points)
    msd_values = np.full(num_points, np.nan)
    
    # Initial values
    com_x, com_y = com_fast(x, y, state)
    msd_values[0] = com_x*com_x + com_y*com_y
    
    time_idx = 1
    r_min, r_max = 0, 2*radius
    
    while t < sim_time and time_idx < num_points:
        # Calculate rates
        n_attached = np.sum(state == 0)
        n_detached = n_legs - n_attached
        
        rate_detach = k_off * n_attached
        rate_attach = k_on * n_detached if n_attached > 0 else 0
        total_rate = rate_detach + rate_attach
        
        if total_rate == 0:
            break
            
        # Time step
        dt = np.random.exponential(1.0 / total_rate)
        t += dt
        
        # Choose event
        if np.random.random() < rate_detach / total_rate:
            # Detachment
            attached = np.where(state == 0)[0]
            if len(attached) > 0:
                chosen = attached[np.random.randint(len(attached))]
                state[chosen] = 1
        else:
            # Attachment  
            detached = np.where(state == 1)[0]
            if len(detached) > 0 and np.sum(state == 0) > 0:
                chosen = detached[np.random.randint(len(detached))]
                reattach_leg_fast(chosen, x, y, state, r_min, r_max)
        
        # Check for complete detachment
        if np.all(state == 1):
            detach_time = t
            break
        
        # Update MSD at time points
        while time_idx < num_points and time_points[time_idx] <= t:
            com_x, com_y = com_fast(x, y, state)
            msd_values[time_idx] = com_x*com_x + com_y*com_y
            time_idx += 1
    
    return time_points, msd_values, detach_time

# ========== DATA MANAGEMENT ==========

class OptimizedDataManager:
    def __init__(self, base_dir="phage_diffusion_analysis"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
    
    def setup_directories(self):
        """Create organized directory structure for focused k_off values"""
        dirs = [
            "data/raw_msd", 
            "data/processed", 
            "data/metadata",
            "data/raw_msd/k_off_0.01",  # Very low k_off
        ]
        for d in dirs:
            (self.base_dir / d).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory structure created at: {self.base_dir.absolute()}")
    
    def get_filename(self, n_legs, k_off, k_on, r_span):
        return f"msd_nlegs_{n_legs}_koff_{k_off:.3f}_kon_{k_on:.3f}_rspa_{r_span:.2f}.csv"
    
    def save_msd_batch(self, results, metadata):
        """Save multiple MSD results efficiently in organized subdirectories"""
        saved_files = []
        
        for result in results:
            k_off = result['k_off']
            
            # Create k_off specific subdirectory
            k_off_dir = self.base_dir / "data/raw_msd" / f"k_off_{k_off:.2f}"
            k_off_dir.mkdir(exist_ok=True)
            
            filename = self.get_filename(result['n_legs'], result['k_off'], 
                                       result['k_on'], result['r_span'])
            filepath = k_off_dir / filename
            
            # Save MSD data
            df = pd.DataFrame({
                'time': result['time'],
                'mean_msd': result['mean_msd'],
                'sem_msd': result['sem_msd']
            })
            df.to_csv(filepath, index=False)
            
            # Save metadata - convert numpy types to Python types for JSON
            meta_file = filepath.with_suffix('.json')
            result_meta = {**metadata}
            
            # Convert numpy types to native Python types
            for key, value in result['params'].items():
                if hasattr(value, 'item'):  # numpy scalar
                    result_meta[key] = value.item()
                else:
                    result_meta[key] = value
            
            try:
                with open(meta_file, 'w') as f:
                    json.dump(result_meta, f, indent=2)
            except TypeError as e:
                print(f"Warning: Could not save metadata for {filename}: {e}")
            
            saved_files.append(filepath)
            print(f"✓ Saved: {filepath}")
        
        return saved_files

# ========== PARALLELIZED SIMULATION ==========

def run_single_parameter_set(params):
    """Run simulation for single parameter combination - for multiprocessing"""
    n_legs, k_off, k_on, r_span, n_sims, sim_time, num_points = params
    
    print(f"Starting: N={n_legs}, k_off={k_off:.3f}")
    start_time = time.time()
    
    # Run multiple simulations
    all_msd = []
    detach_times = []
    
    for sim in range(n_sims):
        seed = hash((n_legs, k_off, sim)) % (2**31)  # Reproducible seeds
        time_points, msd_values, detach_time = gillespie_sim_optimized(
            n_legs, r_span, k_on, k_off, sim_time, num_points, seed
        )
        
        all_msd.append(msd_values)
        if detach_time < np.inf:
            detach_times.append(detach_time)
    
    # Calculate statistics with better handling of empty data
    all_msd = np.array(all_msd)
    
    # Check if we have valid data
    n_valid_sims = np.sum(~np.isnan(all_msd).all(axis=1))
    if n_valid_sims < 10:
        print(f"Warning: Only {n_valid_sims} valid simulations for N={n_legs}, k_off={k_off}")
    
    # Calculate mean and SEM with proper handling of NaNs
    with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
        mean_msd = np.nanmean(all_msd, axis=0)
        n_valid_per_timepoint = np.sum(~np.isnan(all_msd), axis=0)
        
        # Only calculate SEM where we have at least 2 valid points
        sem_msd = np.full_like(mean_msd, np.nan)
        mask = n_valid_per_timepoint >= 2
        if np.any(mask):
            sem_msd[mask] = np.nanstd(all_msd[:, mask], axis=0) / np.sqrt(n_valid_per_timepoint[mask])
    
    elapsed = time.time() - start_time
    print(f"Completed: N={n_legs}, k_off={k_off:.3f} in {elapsed:.1f}s")
    
    return {
        'params': {
            'n_legs': int(n_legs),  # Convert to native Python int
            'k_off': float(k_off),  # Convert to native Python float
            'k_on': float(k_on),
            'r_span': float(r_span),
            'n_simulations': int(n_sims),
            'sim_time': float(sim_time),
            'n_valid_simulations': int(n_valid_sims),
            'mean_detach_time': float(np.mean(detach_times)) if detach_times else float('nan')
        },
        'time': time_points,
        'mean_msd': mean_msd,
        'sem_msd': sem_msd,
        'n_legs': int(n_legs),
        'k_off': float(k_off),
        'k_on': float(k_on),
        'r_span': float(r_span)
    }

def run_optimized_parameter_sweep():
    """Run full parameter sweep with parallelization and immediate saving"""
    
    # Parameters - Focused sweep
    n_legs_range = np.arange(3,11)  # 2-10 legs (9 values)
    k_off_range = [0.01]  # 5 focused k_off values
    k_on = 1.0
    r_span = 0.5
    n_sims = 5000  # Good statistics
    sim_time = 10000  # Long enough for good MSD curves
    num_points = 20000  # Good resolution
    
    # Initialize data manager
    dm = OptimizedDataManager()
    
    # Create parameter combinations
    param_combinations = []
    for k_off in k_off_range:
        for n_legs in n_legs_range:
            param_combinations.append(
                (n_legs, k_off, k_on, r_span, n_sims, sim_time, num_points)
            )
    
    print(f"Running {len(param_combinations)} parameter combinations")
    print(f"Using {mp.cpu_count()-1} CPU cores")
    print("Files will be saved immediately after each parameter set completes")
    print()
    
    # Metadata for all runs
    base_metadata = {
        'timestamp': datetime.now().isoformat(),
        'total_combinations': len(param_combinations),
        'notes': 'Optimized parallel run with immediate saving'
    }
    
    saved_files = []
    start_time = time.time()
    
    # Process all combinations with immediate saving
    print("Processing all parameter combinations...")
    
    # Run all simulations in parallel
    with ProcessPoolExecutor(max_workers=mp.cpu_count()-1) as executor:
        # Submit all jobs
        future_to_params = {
            executor.submit(run_single_parameter_set, params): params 
            for params in param_combinations
        }
        
        # Process results as they complete
        from concurrent.futures import as_completed
        completed = 0
        
        for future in as_completed(future_to_params):
            result = future.result()
            
            # Save immediately when each job completes
            single_result_list = [result]
            batch_saved = dm.save_msd_batch(single_result_list, base_metadata)
            saved_files.extend(batch_saved)
            completed += 1
            
            # Progress update
            elapsed = time.time() - start_time
            if completed < len(param_combinations):
                eta = (elapsed / completed) * (len(param_combinations) - completed)
                print(f"✓ Completed {completed}/{len(param_combinations)} - ETA: {eta/60:.4f} min")
            else:
                print(f"✓ Completed {completed}/{len(param_combinations)} - DONE!")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Saved {len(saved_files)} files")
    
    return saved_files

# ========== QUICK VALIDATION ==========

def quick_speed_test():
    """Test speed improvements"""
    print("Speed test: 1000 simulations, N=4 legs")
    
    start = time.time()
    time_points, msd_values, detach_time = gillespie_sim_optimized(
        4, 0.5, 1.0, 0.5, 1000, 100, 42
    )
    elapsed = time.time() - start
    
    print(f"Single simulation: {elapsed:.3f}s")
    print(f"Estimated time for 1000 sims: {elapsed * 1000:.1f}s")

if __name__ == "__main__":
    print("=== PHAGE DIFFUSION PARAMETER SWEEP ===")
    print("Focused parameter sweep:")
    print("• N_legs: 2-10 (9 values)")
    print("• k_off: 0.01, 0.1, 0.5, 1.0, 2.0 (5 values)")
    print(f"• Total combinations: 9 × 5 = 45")
    print("• This will take approximately 15-30 minutes")
    print()
    
    # Initialize data manager (creates directories)
    dm = OptimizedDataManager()
    print("✓ Directory structure ready")
    
    # Run full sweep
    saved_files = run_optimized_parameter_sweep()
    
    print("\n=== SWEEP COMPLETED ===")
    print(f"Generated {len(saved_files)} MSD datasets")
    print("\nFile structure:")
    print("phage_diffusion_analysis/")
    print("├── data/raw_msd/")
    print("│   ├── k_off_0.1/")  
    print("│   ├── k_off_0.2/")
    print("│   ├── ... (organized by k_off value)")
    print("│   └── k_off_2.0/")
    print("└── data/processed/ (for analysis results)")
    print("\nNext steps:")
    print("1. Run windowing analysis to extract D_eff values") 
    print("2. Generate publication figures")
    print("3. Analyze scaling relationships")