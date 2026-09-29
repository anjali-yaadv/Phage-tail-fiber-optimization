"""Stochastic model of multivalent phage surface search (Yadav et al.).

Modules
-------
config      read config/parameters.yaml and command-line overrides
simulation  Gillespie simulation kernels (numba-accelerated when available)
theory      analytical results (hit time, detachment rate, renewal model, r_eff)
analysis    MSD fits, adsorption-time assembly, statistics
datasets    run + cache one parameter set in data/simulated/
plotting    shared figure style
"""
from . import analysis, config, datasets, simulation, theory  # noqa: F401

__all__ = ["analysis", "config", "datasets", "simulation", "theory"]
