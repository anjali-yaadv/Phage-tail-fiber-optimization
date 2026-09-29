"""Figure S1: thermal exploration radius r_exp in physical units (Eq. reff1).

(a) r_exp vs number of attached fibers n, for fiber length ell and several M.
(b) r_exp vs fiber length ell for n = N attached fibers.
Parameters (config: figures.figS1): D = 4 um^2/s (T4 in water),
tau_e = 1e-3 s / N, N = 6. Values below r_t (2 nm) are set to r_t.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from phage_search import theory  # noqa: E402
from phage_search.config import figure_parser, load_config  # noqa: E402
from phage_search.plotting import DOUBLE_COL, panel_label, save, setup_style  # noqa: E402


def main():
    args = figure_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    f = cfg["figures"]["figS1"]
    r_t_nm = cfg["model"]["r_t"] * 10.0        # 1 length unit = 10 nm
    D = f["D_um2_per_s"] * 1e6                 # nm^2 / s
    tau = f["tau_e_s"] / f["N"]
    setup_style()

    fig, (axa, axb) = plt.subplots(1, 2, figsize=DOUBLE_COL)
    n = np.arange(f["n_range"][0], f["n_range"][1] + 1)
    ell = np.linspace(*f["ell_range_nm"], 100)
    for M in f["M"]:
        axa.plot(n, np.maximum(r_t_nm, theory.r_exp(n, f["ell_nm"], M, D, tau)), label=f"$M={M}$")
        axb.plot(ell, np.maximum(r_t_nm, theory.r_exp(f["N"], ell, M, D, tau)), label=f"$M={M}$")
    axa.set_xlabel(r"$n$")
    axa.set_ylabel(r"$r_{\mathrm{exp}}$ (nm)")
    axb.set_xlabel(r"$\ell$ (nm)")
    axb.set_ylabel(r"$r_{\mathrm{exp}}$ (nm)")
    for ax, lab in ((axa, "a)"), (axb, "b)")):
        ax.legend()
        panel_label(ax, lab)
    fig.tight_layout()
    save(fig, args.out, "figS1_r_exp")


if __name__ == "__main__":
    main()
