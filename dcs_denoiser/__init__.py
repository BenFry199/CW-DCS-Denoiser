"""LSTM denoiser for diffuse correlation spectroscopy (DCS) g2(tau) curves."""

from .inference import DCSDenoiser, denoise, load_denoiser, normalise_taus
from .model import Generalised_LSTMDenoiser

__version__ = "1.0.0"

__all__ = [
    "DCSDenoiser",
    "Generalised_LSTMDenoiser",
    "denoise",
    "load_denoiser",
    "normalise_taus",
]
