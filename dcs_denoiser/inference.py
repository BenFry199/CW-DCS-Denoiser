"""Load the bundled DCS denoiser and apply it to g2(tau) curves.

    from dcs_denoiser import load_denoiser

    denoiser = load_denoiser()
    g2_clean = denoiser.denoise(g2, taus)     # g2: (..., n_tau), taus: (n_tau,) in seconds

Everything the network needs besides the curves themselves -- the LSTM shape and
the lag-time normalisation constants fit on the training set -- is stored inside
the checkpoint file, so there is nothing else to keep in sync.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from .model import Generalised_LSTMDenoiser

WEIGHTS_DIR = Path(__file__).parent / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "dcs_lstm_denoiser_v1.pt"


def normalise_taus(real_taus, taus_min, taus_max, low=0.1, high=1.0, eps=1e-12):
    """Log-normalise lag times into ``[low, high]``.

    ``log10(tau + eps)`` is min-max scaled with ``taus_min`` / ``taus_max`` (the
    ``tau_norm`` bounds fit on the training set), then mapped onto
    ``[low, high]``. ``eps``, ``low`` and ``high`` must match the values used at
    training time; ``DCSDenoiser`` reads all of them from the checkpoint.

    Args:
        real_taus: raw lag times in seconds (torch tensor).
        taus_min, taus_max: min / max of ``log10(tau)`` from ``tau_norm``.
        low, high: target range (default [0.1, 1.0]).

    Returns:
        Tensor, same shape as ``real_taus``, values in ``[low, high]`` for lag
        times inside the training range.
    """
    logged = torch.log10(real_taus + eps)
    lognormed = (logged - taus_min) / (taus_max - taus_min)
    return lognormed * (high - low) + low


def _resolve_device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCSDenoiser:
    """A loaded denoiser: the network plus the tau normalisation it was trained with.

    Build one with ``load_denoiser()`` rather than directly.

    Attributes
    ----------
    model : Generalised_LSTMDenoiser
        The network, in eval mode on ``device``.
    tau_norm : (float, float)
        ``(log10_tau_min, log10_tau_max)`` fit on the training set.
    tau_range_s : (float, float)
        The same bounds in seconds -- the lag-time range the network was
        trained on.
    meta : dict
        Provenance recorded when the checkpoint was packaged.
    """

    def __init__(self, model, tau_norm, device, tau_scale=(0.1, 1.0), tau_eps=1e-12, meta=None):
        self.model = model
        self.tau_norm = (float(tau_norm[0]), float(tau_norm[1]))
        self.tau_scale = (float(tau_scale[0]), float(tau_scale[1]))
        self.tau_eps = float(tau_eps)
        self.device = device
        self.meta = dict(meta or {})

    @property
    def tau_range_s(self):
        return (10.0 ** self.tau_norm[0], 10.0 ** self.tau_norm[1])

    def __repr__(self):
        lo, hi = self.tau_range_s
        return (f"DCSDenoiser(device={self.device}, "
                f"trained tau range=[{lo:.3g}, {hi:.3g}] s)")

    def denoise(self, g2, taus, batch_size=4096):
        """Denoise intensity autocorrelation curves g2(tau).

        Parameters
        ----------
        g2 : array-like, shape (..., n_tau)
            Normalised intensity autocorrelation curves (decaying from
            ``1 + beta`` towards 1 -- NOT ``g2 - 1``). Any number of leading
            dimensions; every curve along the last axis is denoised
            independently.
        taus : array-like, shape (n_tau,)
            Lag times in SECONDS, shared by every curve, strictly increasing.
        batch_size : int, default 4096
            Curves per forward pass. Lower it if you run out of memory.

        Returns
        -------
        numpy.ndarray, float64, same shape as ``g2``
            The denoised g2(tau) curves.

        Notes
        -----
        * Lags with ``tau <= 0`` (e.g. a correlator's zero-lag bin) cannot be
          log-normalised. They are not shown to the network and are returned
          unchanged.
        * A curve containing any non-finite value is returned unchanged.
        * Lags outside the training range (see ``tau_range_s``) raise a
          warning: the network is extrapolating there.
        * The output is always >= 1, because the network predicts ``g2 - 1``
          through a softplus.
        """
        g2 = np.asarray(g2, dtype=np.float64)
        taus = np.asarray(taus, dtype=np.float64).ravel()
        if g2.ndim == 0 or g2.shape[-1] != taus.size:
            raise ValueError(
                f"last axis of g2 {g2.shape} must match the number of lag times ({taus.size})")

        flat = g2.reshape(-1, taus.size)                 # (curves, tau)
        out = flat.copy()

        tau_ok = np.isfinite(taus) & (taus > 0)
        if not tau_ok.any():
            raise ValueError("no positive, finite lag times in `taus`")
        taus_used = taus[tau_ok]
        if np.any(np.diff(taus_used) <= 0):
            raise ValueError("`taus` must be strictly increasing")
        self._warn_if_out_of_range(taus_used)

        sub = flat[:, tau_ok]
        curve_ok = np.isfinite(sub).all(axis=1)
        idx = np.flatnonzero(curve_ok)
        if idx.size == 0:
            return out.reshape(g2.shape)

        n_tau = taus_used.size
        tau_t = torch.from_numpy(taus_used.astype(np.float32)).view(1, n_tau, 1)
        tau_t = normalise_taus(tau_t, *self.tau_norm, *self.tau_scale, eps=self.tau_eps)

        cols = np.flatnonzero(tau_ok)
        with torch.inference_mode():
            for s in range(0, idx.size, batch_size):
                rows = idx[s:s + batch_size]
                sig = torch.from_numpy((sub[rows] - 1.0).astype(np.float32)).unsqueeze(-1)  # (b, L, 1)
                x = torch.cat([tau_t.expand(len(rows), n_tau, 1), sig], dim=-1).to(self.device)
                lengths = torch.full((len(rows),), n_tau, dtype=torch.long)                # no padding
                pred = self.model(x, lengths)                                               # (b, 1, L), g2 - 1
                out[np.ix_(rows, cols)] = pred.squeeze(1).cpu().numpy().astype(np.float64) + 1.0

        return out.reshape(g2.shape)

    __call__ = denoise

    def _warn_if_out_of_range(self, taus):
        log_t = np.log10(taus)
        tol = 1e-3
        n_out = int(((log_t < self.tau_norm[0] - tol) | (log_t > self.tau_norm[1] + tol)).sum())
        if n_out:
            lo, hi = self.tau_range_s
            warnings.warn(
                f"{n_out} of {taus.size} lag times fall outside the range the denoiser was "
                f"trained on ([{lo:.3g}, {hi:.3g}] s); it is extrapolating there. Check that "
                f"`taus` is in seconds.", stacklevel=3)


def load_denoiser(weights=None, device=None):
    """Load a packaged denoiser checkpoint.

    Parameters
    ----------
    weights : str or Path, optional
        Checkpoint to load. Defaults to the one bundled with this package.
    device : str or torch.device, optional
        Where to run inference. Defaults to CUDA when available, else CPU.

    Returns
    -------
    DCSDenoiser
    """
    path = Path(weights) if weights is not None else DEFAULT_WEIGHTS
    if not path.is_file():
        raise FileNotFoundError(f"denoiser checkpoint not found: {path}")
    device = _resolve_device(device)

    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" not in ckpt or "tau_norm" not in ckpt:
        raise ValueError(
            f"{path} is not a packaged dcs_denoiser checkpoint (expected 'state_dict', 'arch' "
            f"and 'tau_norm' keys). Raw training checkpoints must first be packaged with "
            f"scripts/package_weights.py.")

    model = Generalised_LSTMDenoiser(dropout=0.0, **ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return DCSDenoiser(model, ckpt["tau_norm"], device,
                       tau_scale=ckpt.get("tau_scale", (0.1, 1.0)),
                       tau_eps=ckpt.get("tau_eps", 1e-12),
                       meta=ckpt.get("meta"))


_default = {}


def denoise(g2, taus, denoiser=None, batch_size=4096):
    """One-call convenience: ``denoise(g2, taus)``.

    Uses the bundled checkpoint, loaded once and reused on later calls, unless a
    ``DCSDenoiser`` from ``load_denoiser()`` is passed. See
    ``DCSDenoiser.denoise`` for the argument and return details.
    """
    if denoiser is None:
        if "denoiser" not in _default:
            _default["denoiser"] = load_denoiser()
        denoiser = _default["denoiser"]
    return denoiser.denoise(g2, taus, batch_size=batch_size)
