"""
Check that an install reproduces the published model's output.

reference_io.npz holds synthetic noisy curves and the output the released
checkpoint produced for them. Run:  pytest   (or: python tests/test_reference.py)
"""

from pathlib import Path

import numpy as np

from dcs_denoiser import load_denoiser

REF = Path(__file__).parent / "reference_io.npz"


def test_matches_reference_output():
    with np.load(REF) as d:
        taus, g2_noisy, expected = d["taus"], d["g2_noisy"], d["g2_denoised"]
    got = load_denoiser(device="cpu").denoise(g2_noisy, taus)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_zero_lag_and_nonfinite_curves_pass_through():
    with np.load(REF) as d:
        taus, g2_noisy = d["taus"], d["g2_noisy"]
    denoiser = load_denoiser(device="cpu")
    ref = denoiser.denoise(g2_noisy, taus)

    # a leading tau = 0 bin is returned unchanged and does not affect the rest
    taus0 = np.concatenate([[0.0], taus])
    g20 = np.concatenate([np.full((len(g2_noisy), 1), 1.5), g2_noisy], axis=1)
    out0 = denoiser.denoise(g20, taus0)
    np.testing.assert_array_equal(out0[:, 0], 1.5)
    np.testing.assert_allclose(out0[:, 1:], ref, atol=1e-5)

    # a curve with a NaN is returned untouched; the others are still denoised
    bad = g2_noisy.copy()
    bad[0, 3] = np.nan
    out = denoiser.denoise(bad, taus)
    np.testing.assert_array_equal(out[0], bad[0])
    np.testing.assert_allclose(out[1:], ref[1:], atol=1e-5)


if __name__ == "__main__":
    test_matches_reference_output()
    test_zero_lag_and_nonfinite_curves_pass_through()
    print("all reference checks passed")
