# Phage tail fiber optimization

Code for **"Multivalent Surface Search Dynamics Shape Bacteriophage Adsorption Efficiency: A Stochastic Model of Tail Fiber Optimization"** (Yadav et al.).

A phage is modelled as a multivalent walker whose *N* tail fibers bind to (rate `k_on`) and unbind from (rate `k_off`) a bacterial surface. The model is used to study how *N*, the binding kinetics, the fiber reach and the host density set the time until the phage finds a receptor. This repository contains:

- the stochastic (Gillespie) simulation,
- the analytical results (hit time, detachment rate, renewal model, optimal *N*, thermal capture radius),
- one script per figure of the paper.

Every figure can be regenerated from scratch, and any parameter can be changed.

---

## 1. Installation

Python ≥ 3.9 and the packages in `requirements.txt`:

| package | used for |
|---|---|
| `numpy`, `scipy`, `pandas` | numerics, fitting, data files |
| `matplotlib` | figures |
| `pyyaml` | reading the parameter file |
| `numba` (strongly recommended) | compiles the simulation loops, about 100× faster |

```bash
git clone https://github.com/anjali-yaadv/Phage-tail-fiber-optimization.git
cd Phage-tail-fiber-optimization
python -m pip install -r requirements.txt
```

The code also runs without `numba` (same model, plain Python), which is fine for the `--quick` mode and the analytical figures. For the full statistics, install `numba`.

## 2. Quick start

```bash
python make_all_figures.py --quick     # every figure with few trajectories (minutes), to check the setup
python make_all_figures.py             # every figure with the statistics used in the paper
python figures/fig5_optimal_N.py       # a single figure
python -m pytest tests/                # consistency checks (< 1 min)
```

Figures are written to `output/` (PNG and PDF). Simulation results are stored in `data/simulated/` and reused on the next run; add `--resimulate` to run them again.

## 3. Where the parameters are defined

**All parameter values are in one file: [`config/parameters.yaml`](config/parameters.yaml).** No script contains hard-coded model parameters.

- **`model`** is the *single default parameter set* of the paper (Table 1). Every figure starts from it.
- **`theory`** holds constants of the analytical model.
- **`thermal`** holds the parameters of the thermal-fluctuation estimate (Fig. 6, S2).
- **`simulation`** sets the number of trajectories, the time limits and the random seed.
- **`figures.<name>`** lists, for each figure, only what that figure changes relative to `model` (for example a sweep over *N* or `k_off`).

So there is one parameter set; each figure is produced by varying one or two parameters of it, as listed in Section 4.

### Default parameter set (`model`, Table 1 of the paper)

All quantities are dimensionless. Time is in units of 1/`k_on` (`k_on` = 1 corresponds to about 10³ s⁻¹), and 1 length unit is 10 nm.

| key | symbol | default | meaning |
|---|---|---|---|
| `k_on` | k_on | 1 | attachment rate of a detached fiber |
| `k_off` | k_off | 0.1 | detachment rate of an attached fiber |
| `N` | N | 6 | number of tail fibers |
| `r_s` | r_s | 1 (10 nm) | fiber reach: a fiber reattaches at a distance drawn uniformly from [0, r_s] around the centre of mass (COM) |
| `r_t` | r_t | 0.2 (2 nm) | receptor radius: adsorption when the COM comes within r_t of a receptor |
| `L` | L | 10 (0.1 µm) | side of the periodic L × L surface patch |
| `n_targets` | n_t | 1 | receptors on the patch |
| `eta_B` | ηB | 10⁻⁵ | encounter rate with a host in the bulk (1/ηB = mean bulk search time) |
| `a_cell` | a | 100 (1 µm) | cell radius; probability to return to the same cell p_r = a/(a + r_t) |

### Analytical model (`theory`) and thermal fluctuations (`thermal`)

| key | default | meaning |
|---|---|---|
| `D0`, `r_s_ref`, `alpha` | 0.1, 1, 1 | D_eff(N) = D0 (r_s/r_s_ref)² k_on k_off/(k_on + k_off) N^(−alpha), the scaling of Fig. 1c |
| `C_torus` | −1.2 | geometric constant in ⟨T_h⟩ = L²/(2π D_eff) [ln(L/r_t) + C_torus] (Suppl. Eq. S6) |
| `direct_landing_after_return` | true | whether a phage returning to the same cell can land directly on a receptor (see Section 6) |
| `M` | 2 | Gaussian-chain segments per fiber (Eq. reff1) |
| `ell_over_rs` | 1.414 | fiber length ℓ = 1.414 r_s |
| `D_3d` | 40 | 3D diffusion constant of the phage, 4 µm²/s in simulation units |

## 4. Figures

Each script's docstring explains the panels. The table lists what is varied relative to the default set, and the settings in `figures.<name>`.

| figure | script | what is varied | type |
|---|---|---|---|
| Fig. 1b, c | `figures/fig1_msd_diffusion.py` | N = 3–10; k_off = 0.001, 0.01, 0.1 | simulation (MSD) |
| Fig. 3 | `figures/fig3_hit_detach_validation.py` | N; k_off = 0.01, 0.1 (a) and 0.1, 1 (b) | simulation + theory |
| Fig. 4 | `figures/fig4_adsorption_time.py` | k_off; N; k_off/k_on; ηB = 10⁻⁵, 10⁻⁹ | simulation + theory |
| Fig. 5 | `figures/fig5_optimal_N.py` | k_off/k_on = 10⁻⁴–10²; k_off = 0.1, 1, 10; ηB | theory |
| Fig. 6 | `figures/fig6_thermal_fluctuations.py` | r_s = 2, 10; N = 3–8; fixed r_t vs r_eff; L = 20 | simulation |
| Fig. S1 | `figures/figS1_r_exp.py` | n, ℓ, M (physical units) | theory |
| Fig. S2 | `figures/figS2_violin_r_eff.py` | r_s = 1, 2, 10; N = 3–8; fixed r_t vs r_eff; L = 20 | simulation |
| Fig. S3 | `figures/figS3_tradeoff_rs.py` | r_s | simulation + theory |
| Fig. S4 | `figures/figS4_lattice_coverage.py` | N = 2–10; r_s = 0.2, 2; k_off = 0.01 | simulation |
| Fig. S5 | `figures/figS5_global_local.py` | k_off/k_on; M = 1–10; ηB | theory |

Figs. 1a and 2 are schematics made in BioRender.

Fig. 1c and the theory curves of Figs. 3a and 4 use the simulated D_eff values stored in `data/processed/D_eff_vs_N.csv`, which are the data behind Fig. 1c. Run `fig1_msd_diffusion.py --resimulate` to regenerate them from new simulations.

**Run time.** Figs. 5, S1 and S5 are analytical and take seconds. With `numba`, the simulation figures take minutes to a few hours each at full statistics. The slowest cases are those with very long adsorption times (small k_off and large N), for example the N = 8, k_off = 0.1 detachment times in Fig. 3b (⟨T_d⟩ ≈ 3 × 10⁷).

## 5. Changing parameters

Any value in the parameter file can be overridden from the command line with `--set`, without editing the file:

```bash
# Fig. 5 for a lower bulk density and a larger receptor
python figures/fig5_optimal_N.py --set figures.fig5.eta_B_low_density=1e-11 --set model.r_t=0.5

# Fig. 4 with a different default reach and fewer trajectories
python figures/fig4_adsorption_time.py --set model.r_s=2 --set figures.fig4.n_traj=500
```

Other options: `--config my_parameters.yaml` uses a different parameter file, and `--out folder` changes where figures go.

For parameter sets beyond the paper, `simulate.py` runs any single case and prints a summary:

```bash
python simulate.py msd        --N 6 --k_off 0.05 --r_s 2 --n_traj 500     # MSD and D_eff
python simulate.py hit        --N 4 --k_off 0.1                            # hit time, no detachment
python simulate.py detach     --N 5 --k_off 0.5                            # complete-detachment time
python simulate.py adsorption --N 6 --k_off 0.1 --capture thermal          # full adsorption process
python simulate.py coverage   --N 4 --k_off 0.01 --r_s 2                   # lattice coverage
python simulate.py theory     --k_off 0.1 --eta_B 1e-9                     # <T_ads>(N), <T_tot>(N), N*
```

The functions can also be used directly from Python:

```python
from phage_search import theory, datasets
from phage_search.config import load_config

cfg = load_config()                                   # defaults
tp = theory.params_from_config(cfg, k_off=0.5)        # change what you need
print(theory.mean_total_time([1, 2, 5, 10], **tp))    # Eqs. 5-6
df, info = datasets.adsorption(cfg, N=6, k_off=0.1, n_traj=200)
```

## 6. Model details as implemented

- **Fiber dynamics.** Each attached fiber detaches at rate `k_off`. Each detached fiber reattaches at rate `k_on` while at least one fiber is attached. A reattaching fiber lands in the angular sector between its two attached neighbours (neighbours in the ring of *N* fibers), at a distance uniform in [0, `r_s`] from the COM. The COM is the mean position of the attached fibers.
- **Start of a search.** One fiber is attached at a random position on the L × L periodic patch with randomly placed, non-overlapping receptors.
- **Adsorption** happens when the COM is within the capture radius of a receptor: `r_t`, or r_eff(n) = max(r_t, r_exp(n)) with *n* attached fibers when thermal fluctuations are included (Eqs. reff1, reff). A landing directly on a receptor counts as adsorption at time 0.
- **Complete detachment** ends a surface visit. The phage then returns to the same cell with probability p_r (no delay) or searches the bulk for an Exp(ηB) time. In either case it lands at a random point of a surface.
- **Bulk time.** The simulation stores the time spent on surfaces and the number of detachments of each run. The bulk-search times are added afterwards (`analysis.adsorption_times`), so ηB and p_r can be changed without re-simulating.
- **MSD simulations** (Fig. 1, S3a) run on an open plane without receptors. Trajectories that fully detach stop contributing, and the MSD is averaged over the trajectories still attached.
- **Hit-time simulations** (Fig. 3a) forbid detachment of the last attached fiber (pure surface search). **Detachment-time simulations** (Fig. 3b) only need the number of attached fibers (a birth–death process).
- **Analytical mean adsorption time** (Eq. 5 and Supplement): ⟨T_ads⟩ = [1 + λ_d (1 − p_r)/ηB] / [λ_h + p_l λ_d] for N ≥ 2, and Eq. 6 for N = 1. The switch `theory.direct_landing_after_return: false` replaces p_l λ_d by (1 − p_r) p_l λ_d, i.e. direct landing only after a bulk search.
- **Renewal distribution** (Fig. 4a, S3b): all kernels are exponential, so P_ads(t) is evaluated exactly as a sum of two exponentials (`theory.renewal_distribution`).

## 7. Reproducibility

Trajectory *j* of a parameter set uses the random seed `simulation.seed + j`. Each result in `data/simulated/` has a JSON file next to it with the exact parameters, the number of trajectories, whether `numba` was used, and the run time.

`numba` and NumPy use different random-number generators, so runs with and without `numba` give statistically equivalent but not identical numbers.

## 8. Repository layout

```
config/parameters.yaml     all parameter values (default set + per-figure changes)
phage_search/              the model, as a Python package
    simulation.py          Gillespie simulation kernels
    theory.py              analytical results
    analysis.py            MSD fits, adsorption-time assembly, statistics
    datasets.py            run one parameter set and cache it in data/simulated/
    config.py, plotting.py parameter loading, figure style
figures/                   one script per paper figure
simulate.py                command-line runs for any parameter set
make_all_figures.py        all figures in one go
data/processed/            D_eff values behind Fig. 1c
data/simulated/            simulation results (created when you run the scripts)
tests/                     consistency checks
```

## Citation

If you use this code, please cite the paper (reference to be added on publication).

## License

To be added by the authors.
