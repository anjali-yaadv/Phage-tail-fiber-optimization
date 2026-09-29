"""Produce every figure of the paper in output/.

    python make_all_figures.py            # full statistics (hours; needs numba)
    python make_all_figures.py --quick    # few trajectories, checks the pipeline in minutes
    python make_all_figures.py fig4 fig5  # only some figures

Extra options (e.g. --set model.eta_B=1e-7) are passed on to every script.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = {
    "fig1": "fig1_msd_diffusion.py",
    "fig3": "fig3_hit_detach_validation.py",
    "fig4": "fig4_adsorption_time.py",
    "fig5": "fig5_optimal_N.py",
    "fig6": "fig6_thermal_fluctuations.py",
    "figS1": "figS1_r_exp.py",
    "figS2": "figS2_violin_r_eff.py",
    "figS3": "figS3_tradeoff_rs.py",
    "figS4": "figS4_lattice_coverage.py",
    "figS5": "figS5_global_local.py",
}


def main():
    chosen = [a for a in sys.argv[1:] if a in FIGURES] or list(FIGURES)
    passthrough = [a for a in sys.argv[1:] if a not in FIGURES]
    failed = []
    for key in chosen:
        print(f"\n=== {key} ===", flush=True)
        r = subprocess.run([sys.executable, str(HERE / "figures" / FIGURES[key]), *passthrough])
        if r.returncode:
            failed.append(key)
    print("\nfailed: " + ", ".join(failed) if failed else "\nall figures written to output/")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
