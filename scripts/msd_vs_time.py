import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


def plot_msd_for_koff(k_off, base_dir="phage_diffusion_analysis", save_figure=True):
    """
    Plot MSD vs time for all n_legs values at a specific k_off value.
    Main panel: log-log MSD vs time
    Inset: linear MSD vs time
    
    Parameters:
    - k_off: k_off value to plot (e.g., 0.1, 0.5, 1.0)
    - base_dir: base directory containing data
    - save_figure: whether to save the figure
    """
    
    base_path = Path(base_dir)
    data_dir = base_path / "data/raw_msd/msd_vs_time_plot"
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist!")
        print("Available directories:")
        raw_msd_dir = base_path / "data/raw_msd"
        if raw_msd_dir.exists():
            for d in raw_msd_dir.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
        return
    
    msd_files = list(data_dir.glob("msd_nlegs_*.csv"))
    if not msd_files:
        print(f"No MSD files found in {data_dir}")
        return
    
    file_data = []
    for file_path in msd_files:
        filename = file_path.stem
        parts = filename.split('_')
        try:
            nlegs_idx = parts.index('nlegs')
            n_legs = int(parts[nlegs_idx + 1])
            file_data.append((n_legs, file_path))
        except (ValueError, IndexError):
            print(f"Warning: Could not parse n_legs from filename {filename}")
            continue
    
    file_data.sort(key=lambda x: x[0])
    print(f"Found {len(file_data)} MSD files for k_off = {k_off}")
    print("N_legs order:", [x[0] for x in file_data])
    # Create figure (main log-log axis)
    fig, ax = plt.subplots() 
    colors = plt.cm.viridis(np.linspace(0, 1, len(file_data)))
    
    n_legs_data = []
    
    for i, (n_legs, file_path) in enumerate(file_data):
        df = pd.read_csv(file_path)
        time = df['time'].values
        mean_msd = df['mean_msd'].values
        sem_msd = df['sem_msd'].values
        
        valid_mask = ~np.isnan(mean_msd) & ~np.isnan(sem_msd) & (mean_msd > 0)
        time_clean = time[valid_mask]
        msd_clean = mean_msd[valid_mask]
        sem_clean = sem_msd[valid_mask]
        
        if len(time_clean) == 0:
            print(f"Warning: No valid data for n_legs = {n_legs}")
            continue
        
        color = colors[i % len(colors)]
        
        # Main log-log plot
        ax.loglog(time_clean, msd_clean, '-', color=color, lw=2,
                  label=f'N = {n_legs}', alpha=0.9)
        ax.fill_between(time_clean, msd_clean - sem_clean, msd_clean + sem_clean,
                        color=color, alpha=0.2)
        
        n_legs_data.append({
            'n_legs': n_legs,
            'time': time_clean,
            'msd': msd_clean,
            'sem': sem_clean
        })
    
    # Format main log-log axis
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle r^2 \rangle$')
    #ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(bottom = 1e-2)
    # Reference slope=1 guide line
    if n_legs_data:
        all_times = np.concatenate([d['time'] for d in n_legs_data])
        t_ref = np.array([all_times.min(), all_times.max()])
        msd_ref = t_ref / t_ref[0]
    
    # --- Inset linear plot ---
    ax_inset = inset_axes(ax, width="38%", height="35%", loc='upper left', borderpad=2)
    for i, d in enumerate(n_legs_data):
        color = colors[i % len(colors)]
        ax_inset.plot(d['time'], d['msd'], '-', color=color, lw=1.5)
        ax_inset.fill_between(d['time'], d['msd'] - d['sem'], d['msd'] + d['sem'],
                              color=color, alpha=0.2)
        ax_inset.set_xlim(right = 1000)
        ax_inset.set_ylim(top = 20)
    ax_inset.grid(True, alpha=0.2)
    
    # Legend outside inset
    ax.legend(loc='lower right', frameon=False)
    figures_dir = base_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    filename = f"msd_vs_time_koff_{k_off:.2f}.png"
    plt.savefig(filename, dpi=600)


    
    plt.show()
    return n_legs_data





def plot_multiple_koff_comparison(k_off_list, base_dir="phage_diffusion_analysis"):
    """
    Compare MSD curves across different k_off values for a specific n_legs
    """
    
    base_path = Path(base_dir)
    
    # Choose a representative n_legs value (e.g., 4)
    n_legs_target = 4
    
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(k_off_list)))
    
    for i, k_off in enumerate(k_off_list):
        data_dir = base_path / "data/raw_msd/k_off_0.1"# / f"k_off_{k_off:.1f}"
        
        # Find file for target n_legs
        msd_file = data_dir / f"msd_nlegs_{n_legs_target}_koff_{k_off:.1f}_kon_1.000_rspa_0.50.csv"
        
        if not msd_file.exists():
            print(f"Warning: File not found for k_off={k_off}, n_legs={n_legs_target}")
            continue
        
        # Load data
        df = pd.read_csv(msd_file)
        time = df['time'].values
        mean_msd = df['mean_msd'].values
        sem_msd = df['sem_msd'].values
        
        # Clean data
        valid_mask = ~np.isnan(mean_msd) & (mean_msd > 0)
        time_clean = time[valid_mask]
        msd_clean = mean_msd[valid_mask]
        sem_clean = sem_msd[valid_mask]
        
        if len(time_clean) == 0:
            continue
            
        color = colors[i]
        
        # Linear plot
        ax1.plot(time_clean, msd_clean, '-', color=color, linewidth=2, 
                label=f'k_off = {k_off}', alpha=0.8)
        ax1.fill_between(time_clean, msd_clean - sem_clean, msd_clean + sem_clean, 
                        color=color, alpha=0.2)
        
        # Log-log plot
        ax2.loglog(time_clean, msd_clean, '-', color=color, linewidth=2, 
                  label=f'k_off = {k_off}', alpha=0.8)
        ax2.fill_between(time_clean, msd_clean - sem_clean, msd_clean + sem_clean, 
                        color=color, alpha=0.2)
    
    # Format plots
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean Square Displacement ⟨r²⟩')
    ax1.set_title(f'MSD vs Time - k_off Comparison (N = {n_legs_target})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mean Square Displacement ⟨r²⟩')
    ax2.set_title(f'MSD vs Time - Log Scale (N = {n_legs_target})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save comparison figure
    figures_dir = base_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    filename = f"msd_koff_comparison_nlegs_{n_legs_target}.pdf"
    filepath = figures_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison figure saved: {filepath}")
    
    plt.show()

def main():
    """Main function with examples"""
    
    print("=== MSD vs Time Plotting ===")
    print()
    
    # Available k_off values
    k_off_values = [0.1]
    
    print("Available k_off values:", k_off_values)
    print()
    
    # Example 1: Plot for specific k_off
    k_off_to_plot = 0.1  # Change this to plot different k_off values
    print(f"Plotting MSD curves for k_off = {k_off_to_plot}")
    n_legs_data = plot_msd_for_koff(k_off_to_plot)
    
    print()
    
    # Example 2: Compare across k_off values
    print("Creating k_off comparison plot...")
    plot_multiple_koff_comparison(k_off_values)
    
    print()
    print("=== Plotting Complete ===")
    print("Check the 'figures' directory for saved plots!")

if __name__ == "__main__":
    # Quick plot for specific k_off - change this value as needed
    k_off_to_plot = 0.1 # Try 0.01, 0.1, 0.5, 1.0, or 2.0
    
    print(f"Plotting MSD vs time for k_off = {k_off_to_plot}")
    plot_msd_for_koff(k_off_to_plot)
    
    # Uncomment to run full comparison
    # main()