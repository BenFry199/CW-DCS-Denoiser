"""
Synthetic DCS curves for the examples -- so they run without any measured data.

The forward model is the standard semi-infinite homogeneous correlation
diffusion solution with Brownian motion. The noise is a simplified
shot-noise-like model (larger at short lags, where correlator bins are
narrow). It is for illustration only and is NOT the noise model the denoiser
was trained with.
"""

from __future__ import annotations

import numpy as np


def semi_infinite_g2(taus, rho=2.5, aDb=1e-8, beta=0.5, mua=0.1, musp=10.0,
                     n=1.33, wavelength_nm=785.0):
    """g2(tau) for a semi-infinite homogeneous medium, Brownian motion.

    Units: ``taus`` in s, ``rho`` in cm, ``aDb`` in cm^2/s, ``mua`` / ``musp``
    in cm^-1.
    """
    taus = np.asarray(taus, dtype=np.float64)
    k0 = 2.0 * np.pi * n / (wavelength_nm * 1e-7)                 # cm^-1
    r_eff = -1.440 / n**2 + 0.710 / n + 0.668 + 0.0636 * n
    z0 = 1.0 / musp
    zb = 2.0 * (1.0 + r_eff) / (3.0 * musp * (1.0 - r_eff))
    r1 = np.sqrt(rho**2 + z0**2)
    rb = np.sqrt(rho**2 + (z0 + 2.0 * zb)**2)

    def G1(tau):
        K = np.sqrt(3.0 * mua * musp + 6.0 * musp**2 * k0**2 * aDb * tau)
        return np.exp(-K * r1) / r1 - np.exp(-K * rb) / rb

    g1 = G1(taus) / G1(0.0)
    return 1.0 + beta * g1**2


def add_noise(g2, taus, beta=0.5, count_rate=2e4, t_int=1.0, rng=None):
    """Add simplified shot-noise-like Gaussian noise to clean g2 curves.

    ``sigma(tau) = sqrt(1 + (g2 - 1)) / (count_rate * sqrt(T_bin * t_int))``
    with the bin width ``T_bin`` taken equal to ``tau`` (a multi-tau
    correlator). ``g2`` may have any leading dimensions; noise is independent
    per point.
    """
    rng = np.random.default_rng(rng)
    g2 = np.asarray(g2, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    sigma = np.sqrt(1.0 + (g2 - 1.0)) / (count_rate * np.sqrt(taus * t_int))
    return g2 + rng.normal(size=g2.shape) * sigma


def example_curves(n_repeats=50, n_tau=128, seed=0, **forward_kwargs):
    """A lag-time grid, one clean curve, and ``n_repeats`` noisy copies of it.

    Returns ``(taus, g2_clean, g2_noisy)`` with shapes ``(n_tau,)``,
    ``(n_tau,)`` and ``(n_repeats, n_tau)``.
    """
    taus = np.logspace(-7, -2, n_tau)                             # 0.1 us .. 10 ms
    g2_clean = semi_infinite_g2(taus, **forward_kwargs)
    noisy = add_noise(np.broadcast_to(g2_clean, (n_repeats, n_tau)), taus, rng=seed)
    return taus, g2_clean, noisy
