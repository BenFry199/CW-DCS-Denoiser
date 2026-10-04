"""
Minimal use case: denoise a batch of noisy g2(tau) curves and plot the result.

Uses synthetic curves (examples/synthetic_dcs.py) so it runs with no data files.
To use your own data, replace `taus` and `g2_noisy` with your lag times
(seconds) and measured g2 curves, shape (..., n_tau).

Run:  python examples/denoise_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))          # for synthetic_dcs
from synthetic_dcs import example_curves

from dcs_denoiser import load_denoiser


def main():
    # 1) data: one clean curve and 50 noisy realisations of it
    taus, g2_clean, g2_noisy = example_curves(n_repeats=50, rho=2.5, aDb=1e-8, beta=0.5)

    # 2) load the bundled model and denoise -- every curve along the last axis
    #    is denoised independently, so any leading shape works
    denoiser = load_denoiser()
    print(denoiser)
    g2_denoised = denoiser.denoise(g2_noisy, taus)

    # 3) compare against the known clean curve
    rmse_noisy = np.sqrt(np.mean((g2_noisy - g2_clean) ** 2))
    rmse_denoised = np.sqrt(np.mean((g2_denoised - g2_clean) ** 2))
    print(f"RMSE vs clean curve:  noisy {rmse_noisy:.4f}   denoised {rmse_denoised:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, curves, title in ((axes[0], g2_noisy, "Noisy"), (axes[1], g2_denoised, "Denoised")):
        ax.semilogx(taus, curves[:10].T, lw=0.8, alpha=0.6, color="tab:gray")
        ax.semilogx(taus, g2_clean, lw=1.8, color="tab:red", label="clean (ground truth)")
        ax.set_title(f"{title} (10 of {len(curves)} curves)")
        ax.set_xlabel(r"$\tau$ (s)")
        ax.legend(frameon=False)
    axes[0].set_ylabel(r"$g_2(\tau)$")
    axes[0].set_ylim(0.9, 1.7)
    fig.tight_layout()

    out = Path(__file__).parent / "denoise_example.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
