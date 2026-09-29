"""Load the parameter file and apply command-line overrides.

Every script reads parameters through `load_config`, so `config/parameters.yaml`
is the single source of truth. Values can be overridden without editing the
file:

    python figures/fig4_adsorption_time.py --set model.eta_B=1e-7 --set figures.fig4.n_traj=500
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "parameters.yaml"


def _set_dotted(cfg: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"Unknown parameter group '{k}' in '{dotted_key}'")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"Unknown parameter '{dotted_key}' (check config/parameters.yaml)")
    node[keys[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Read the YAML file and apply `key=value` overrides (values parsed as YAML)."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def model_params(cfg: dict, **changes) -> dict:
    """The default parameter set (`model`) with some values replaced."""
    p = copy.deepcopy(cfg["model"])
    for k, v in changes.items():
        if k not in p:
            raise KeyError(f"'{k}' is not a model parameter")
        p[k] = v
    return p


def figure_parser(description: str) -> argparse.ArgumentParser:
    """Common command-line options for all figure scripts."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None, help="parameter file (default: config/parameters.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a parameter, e.g. --set model.k_off=0.05 (repeatable)")
    ap.add_argument("--quick", action="store_true",
                    help="few trajectories, for testing the pipeline (not publication quality)")
    ap.add_argument("--resimulate", action="store_true",
                    help="ignore saved simulation data and run the simulations again")
    ap.add_argument("--out", default=str(REPO_ROOT / "output"), help="folder for the figure files")
    return ap
