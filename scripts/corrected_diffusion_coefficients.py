import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import pandas as pd
from pathlib import Path

def identify_diffusive_regime_visual(time, msd, plot_analysis=False):
    """
    Better method to identify diffusive regime by analyzing the derivative and local slopes
    """
    # Remove any invalid data points
    valid_mask = np.isfinite(time) & np.isfinite(msd) & (time > 0) & (msd > 0)
    time_clean = time[valid_mask]
    msd_clean = msd[valid_mask]
    
    if len(time_clean) < 100:
        return None, None, None
    
    # Convert to log space for slope analysis
    log_time = np.log10(time_clean)
    log_msd = np.log10(msd_clean)
    
    # Calculate local slopes using different window sizes
    window_size = max(50, len(time_clean) // 20)  # Adaptive window size
    step_size = max(5, window_size // 10)
    
    slopes = []
    centers = []
    r_squared_vals = []
    time_centers = []
    
    for start in range(0, len(log_time) - window_size, step_size):
        end = start + window_size
        t_win = log_time[start:end]
        msd_win = log_msd[start:end]
        
        # Linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = linregress(t_win, msd_win)
        
        slopes.append(slope)
        centers.append(np.mean(t_win))
        time_centers.append(10**np.mean(t_win))  # Convert back to linear time
        r_squared_vals.append(r_value**2)
    
    slopes = np.array(slopes)
    centers = np.array(centers)
    time_centers = np.array(time_centers)
    r_squared_vals = np.array(r_squared_vals)
    
    '''if plot_analysis:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Local slopes vs time
        ax1.semilogx(time_centers, slopes, 'bo-', markersize=4)
        ax1.axhline(1.0, color='r', linestyle='--', linewidth=2, label='Diffusive (slope=1)')
        ax1.axhspan(0.85, 1.15, alpha=0.2, color='green', label='Good diffusive region')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Local slope in log-log')
        ax1.set_title('Local Slope Analysis')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: R-squared values
        ax2.semilogx(time_centers, r_squared_vals, 'go-', markersize=4)
        ax2.axhline(0.95, color='r', linestyle='--', linewidth=2, label='Good fit threshold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('R² of local fits')
        ax2.set_title('Quality of Local Linear Fits')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()'''
    
    # Find good diffusive regions (slope ≈ 1 and high R²)
    good_slope = (slopes > 0.85) & (slopes < 1.15)
    good_r2 = r_squared_vals > 0.95
    good_regions = good_slope & good_r2
    
    if not np.any(good_regions):
        # Relax criteria if no perfect regions found
        good_slope = (slopes > 0.7) & (slopes < 1.3)
        good_r2 = r_squared_vals > 0.9
        good_regions = good_slope & good_r2
        
    if not np.any(good_regions):
        print("Warning: No clear diffusive region found, using middle portion")
        # Use middle 50% of data as fallback
        start_idx = len(time_clean) // 4
        end_idx = 3 * len(time_clean) // 4
        return time_clean[start_idx], time_clean[end_idx], np.median(slopes)
    
    # Find the best continuous region
    good_times = time_centers[good_regions]
    good_slopes_subset = slopes[good_regions]
    
    # Use a generous range around the best regions
    start_time = np.min(good_times) * 0.8  # Start a bit earlier
    end_time = np.max(good_times) * 1.2    # End a bit later
    
    # Make sure we don't exceed data bounds
    start_time = max(start_time, time_clean[len(time_clean)//10])  # Don't start too early
    end_time = min(end_time, time_clean[-len(time_clean)//10])     # Don't end too late
    
    avg_slope = np.mean(good_slopes_subset)
    
    return start_time, end_time, avg_slope

def extract_diffusion_coefficient_improved(time, msd, plot_diagnostic=False, title_info=""):
    """
    Improved diffusion coefficient extraction
    """
    
    # Clean data
    valid_mask = np.isfinite(time) & np.isfinite(msd) & (time > 0) & (msd > 0)
    time_clean = time[valid_mask]
    msd_clean = msd[valid_mask]
    
    if len(time_clean) < 100:
        return np.nan, {'error': 'insufficient_data'}
    
    # Find diffusive regime
    start_time, end_time, avg_slope = identify_diffusive_regime_visual(
        time_clean, msd_clean, plot_analysis=plot_diagnostic
    )
    
    if start_time is None:
        return np.nan, {'error': 'no_diffusive_regime_found'}
    
    # Extract fit region
    fit_mask = (time_clean >= start_time) & (time_clean <= end_time)
    t_fit = time_clean[fit_mask]
    msd_fit = msd_clean[fit_mask]
    
    if len(t_fit) < 30:
        return np.nan, {'error': 'insufficient_fit_points', 'n_points': len(t_fit)}
    
    # Define fitting functions
    def linear_no_intercept(t, D):
        return 4.0 * D * t
    
    def linear_with_intercept(t, D, C):
        return 4.0 * D * t + C
    
    # Try different fitting approaches
    results = []
    
    try:
        # Method 1: No intercept
        popt1, _ = curve_fit(linear_no_intercept, t_fit, msd_fit)
        D1 = popt1[0]
        pred1 = linear_no_intercept(t_fit, D1)
        r2_1 = 1 - np.sum((msd_fit - pred1)**2) / np.sum((msd_fit - np.mean(msd_fit))**2)
        results.append(('no_intercept', D1, 0.0, r2_1, pred1))
        
        # Method 2: With intercept
        popt2, _ = curve_fit(linear_with_intercept, t_fit, msd_fit)
        D2, C2 = popt2
        pred2 = linear_with_intercept(t_fit, D2, C2)
        r2_2 = 1 - np.sum((msd_fit - pred2)**2) / np.sum((msd_fit - np.mean(msd_fit))**2)
        results.append(('with_intercept', D2, C2, r2_2, pred2))
        
        # Method 3: Weighted fit (more weight to later times)
        weights = np.sqrt(t_fit) / np.max(np.sqrt(t_fit))  # Normalized weights
        popt3, _ = curve_fit(linear_with_intercept, t_fit, msd_fit, sigma=1.0/weights)
        D3, C3 = popt3
        pred3 = linear_with_intercept(t_fit, D3, C3)
        r2_3 = 1 - np.sum((msd_fit - pred3)**2) / np.sum((msd_fit - np.mean(msd_fit))**2)
        results.append(('weighted', D3, C3, r2_3, pred3))
        
    except Exception as e:
        return np.nan, {'error': f'fitting_failed: {str(e)}'}
    
    # Choose best result (highest R²)
    results.sort(key=lambda x: x[3], reverse=True)
    best_method, D_best, C_best, r2_best, pred_best = results[0]
    
    # Sanity check
    if D_best <= 0:
        return np.nan, {'error': 'negative_diffusion_coefficient'}
    
    '''# Optional diagnostic plot
    if plot_diagnostic:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Diffusion Analysis: {title_info}', fontsize=14)
        
        # Plot 1: Full data with fit region
        axes[0,0].loglog(time_clean, msd_clean, 'b-', alpha=0.7, linewidth=1, label='MSD data')
        axes[0,0].loglog(t_fit, msd_fit, 'ro', markersize=3, alpha=0.8, label='Fit region')
        axes[0,0].loglog(t_fit, pred_best, 'k--', linewidth=2, 
                        label=f'Best fit: D={D_best:.3g}')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('MSD')
        axes[0,0].set_title('Log-Log View')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Linear scale
        axes[0,1].plot(time_clean, msd_clean, 'b-', alpha=0.7, linewidth=1, label='MSD data')
        axes[0,1].plot(t_fit, msd_fit, 'ro', markersize=2, alpha=0.8, label='Fit region')
        axes[0,1].plot(t_fit, pred_best, 'k--', linewidth=2, 
                      label=f'{best_method} (R²={r2_best:.3f})')
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('MSD')
        axes[0,1].set_title('Linear Scale')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: All fitting methods comparison
        colors = ['red', 'green', 'blue']
        for i, (method, D, C, r2, pred) in enumerate(results):
            axes[1,0].plot(t_fit, pred, color=colors[i], linewidth=2, 
                          label=f'{method}: D={D:.3g}, R²={r2:.3f}')
        axes[1,0].plot(t_fit, msd_fit, 'ko', markersize=2, alpha=0.6, label='Data')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('MSD')
        axes[1,0].set_title('Method Comparison')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Residuals
        residuals = msd_fit - pred_best
        axes[1,1].plot(t_fit, residuals, 'go', markersize=3, alpha=0.7)
        axes[1,1].axhline(0, color='k', linestyle='--', alpha=0.7)
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Residuals')
        axes[1,1].set_title('Fit Residuals')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    '''
    # Return results
    quality_metrics = {
        'D_eff': D_best,
        'C_intercept': C_best,
        'r_squared': r2_best,
        't_start': float(start_time),
        't_end': float(end_time),
        'n_fit_points': len(t_fit),
        'method_used': best_method,
        'avg_slope_log': avg_slope if avg_slope else np.nan,
        'all_methods': {method: {'D': D, 'C': C, 'r2': r2} for method, D, C, r2, _ in results},
        'error': None
    }
    
    return D_best, quality_metrics

# Modified version of your existing function to use the improved method
def process_single_msd_file_improved(filepath, plot_fit=False):
    """Process a single MSD file with improved diffusion extraction"""
    
    # Parse filename (your existing code)
    filename = filepath.stem
    parts = filename.split('_')
    
    try:
        nlegs_idx = parts.index('nlegs'); n_legs = int(parts[nlegs_idx + 1])
        koff_idx  = parts.index('koff');  k_off  = float(parts[koff_idx + 1])
        kon_idx   = parts.index('kon');   k_on   = float(parts[kon_idx + 1])
        rspa_idx  = parts.index('rspa');  r_span = float(parts[rspa_idx + 1])   
    except (ValueError, IndexError) as e:
        print(f"Error parsing filename {filename}: {e}")
        return None
    
    # Load data (your existing code)
    try:
        df = pd.read_csv(filepath)
        time = df['time'].to_numpy(dtype=float)
        msd = df['mean_msd'].to_numpy(dtype=float)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
    
    # Use improved extraction method
    title_info = f"N={n_legs}, k_off={k_off}"
    D_eff, quality = extract_diffusion_coefficient_improved(
        time, msd, plot_diagnostic=plot_fit, title_info=title_info
    )
    
    # Compile results (your existing structure)
    result = {
        'n_legs': n_legs,
        'k_off': k_off,
        'k_on': k_on,
        'r_span': r_span,
        'D_eff': D_eff,
        **quality,
        'filepath': str(filepath)
    }
    
    return result

# Test with your existing pipeline
def test_improved_method_on_your_data(base_dir="phage_diffusion_analysis", 
                                     ):
    """Test the improved method on a few of your MSD files"""
    
    base_path = Path(base_dir)
    raw_msd_dir = base_path / "data" / "raw_msd"
    
    if not raw_msd_dir.exists():
        print(f"Error: Directory {raw_msd_dir} does not exist!")
        return None
    
    all_msd_files = list(raw_msd_dir.glob("**/*.csv"))
    test_files = all_msd_files
    
    print(f"Testing improved method on {len(test_files)} files:")
    
    for filepath in test_files:
        print(f"\nProcessing: {filepath.name}")
        result = process_single_msd_file_improved(filepath, plot_fit=False)
        
        if result and not np.isnan(result['D_eff']):
            print(f"✓ D_eff = {result['D_eff']:.3g}")
            print(f"  Method: {result['method_used']}")
            print(f"  R² = {result['r_squared']:.3f}")
            print(f"  Fit region: t ∈ [{result['t_start']:.1f}, {result['t_end']:.1f}]")
            print(f"  Fit points: {result['n_fit_points']}")
        else:
            print(f"✗ Failed: {result.get('error', 'unknown') if result else 'file error'}")


# ---------------- Batch runner ----------------
def extract_all_diffusion_coefficients(base_dir="phage_diffusion_analysis",
                                       save_results=True,
                                       plot_examples=True):
    base = Path(base_dir)
    raw_dir = base/"data"/"raw_msd"
    files = list(raw_dir.glob("k_off_*/*.csv"))

    print(f"Found {len(files)} MSD files")

    results, failed = [], []
    for i,f in enumerate(files):
        res = process_single_msd_file_improved(f, plot_fit=(plot_examples and i<3))
        if res is None or np.isnan(res['D_eff']):
            failed.append(res)
        else:
            results.append(res)
            print(f"✓ {f.name}: D={res['D_eff']:.3g}, R²={res['r_squared']:.3f}")

    results_df, failed_df = pd.DataFrame(results), pd.DataFrame(failed)
    if save_results:
        proc_dir = base/"data"/"processed"; proc_dir.mkdir(exist_ok=True)
        results_df.to_csv(proc_dir/"diffusion_coefficients.csv", index=False)
    return results_df, failed_df

try:
    import scienceplots
    plt.style.use(['science', 'nature'])
    # Disable LaTeX if not available
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif"
    })
    print("Using scienceplots styling (without LaTeX)")
except ImportError:
    print("Warning: scienceplots not available, using default matplotlib styling")
    plt.style.use('default')
except Exception as e:
    print(f"Warning: Could not apply scienceplots styling ({e}), using default matplotlib styling")
    plt.style.use('default')
plot_params ={
    # Resolution / output
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": (3.35, 2.5),  # Single column width
    
    # Fonts
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    
    # Axes and spines
    "axes.linewidth": 0.8,
    
    # Tick sizes
    "xtick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "xtick.minor.size": 2,
    "xtick.minor.width": 0.6,
    "ytick.major.size": 3.5,
    "ytick.major.width": 0.8,
    "ytick.minor.size": 2,
    "ytick.minor.width": 0.6,
    
    # Tick direction
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.markersize": 3,  # Smaller marker size
    
    # Layout
    "figure.autolayout": True
}


plt.rcParams.update(plot_params)


# ---------------- Main ----------------
def main():
    df, failed = extract_all_diffusion_coefficients(plot_examples=True)
    if df is None or len(df)==0: return
    print(df.head())
    fig, ax = plt.subplots()  # small compact size like Biophys J
    results_df = df
    markers = {0.001:'o', 0.01:'s', 0.05:'^', 0.1:'D'}
    colors = {0.001:'C0', 0.01:'C1', 0.05:'C2', 0.1:'C3'}
    for koff, grp in results_df.groupby("k_off"):
        grp = grp.sort_values("n_legs")
        grp["D_eff"] = grp["D_eff"] * ( 1 + grp["k_off"])/ grp["k_off"]  # Normalize by mean k_off for better comparison
        ax.plot(grp['n_legs'], grp['D_eff'], 
                marker=markers[koff], 
                color=colors[koff],  
                linestyle= 'None',
                label=rf"$k_{{off}} = {koff}$")
        
        #Include error bars as well

    # Axis labels
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$D$")
    #ax.set_ylim(0, None)
    #ax.set_yscale('log')  # Log scale for better visibility of trends
    #ax.set_xscale('log')  # Log scale for better visibility of trends
    # Legend inside plot, top-left
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    plt.ylim(0, None)
    # Axis styling
    #ax.tick_params(axis='both', which='major', labelsize=9)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("D_vsN_for_k_off.png", dpi=600, bbox_inches="tight")
    plt.show()

if __name__=="__main__":
    main()
