# dcs-denoiser

A trained neural network that denoises diffuse correlation spectroscopy (DCS) intensity autocorrelation curves, g2(τ).

You give it noisy g2 curves and their lag times; it returns denoised g2 curves of the same shape. The lag-time grid is an input to the network, so one model works across correlator schemes.

```python
from dcs_denoiser import load_denoiser

denoiser = load_denoiser()
g2_clean = denoiser.denoise(g2, taus)   # g2: (..., n_tau), taus: (n_tau,) in seconds
```

## Installation

Requires Python 3.9 or later, NumPy and PyTorch 2.0 or later. The model weights (about 3 MB) are included in the package.

```bash
pip install git+https://github.com/BenFry199/CW-DCS-Denoiser.git
```

Or, to also run the examples and tests, clone the repository and install it in editable mode:

```bash
git clone https://github.com/BenFry199/CW-DCS-Denoiser.git
cd CW-DCS-Denoiser
pip install -e ".[examples,test]"
```

## Usage

```python
import numpy as np
from dcs_denoiser import load_denoiser

denoiser = load_denoiser()                 # bundled weights; CUDA if available, else CPU
g2_denoised = denoiser.denoise(g2, taus)
```

For a single call without keeping a model object, use `denoise`, which loads the bundled model once and reuses it:

```python
from dcs_denoiser import denoise
g2_denoised = denoise(g2, taus)
```

A complete worked example on synthetic curves is in [examples/denoise_example.ipynb](examples/denoise_example.ipynb), with a script version in [examples/denoise_example.py](examples/denoise_example.py).

### Inputs

| Argument | Shape | Description |
|---|---|---|
| `g2` | `(..., n_tau)` | Normalised intensity autocorrelation curves, decaying from `1 + β` towards 1. Pass plain g2, **not** `g2 − 1`. |
| `taus` | `(n_tau,)` | Lag times in **seconds**, strictly increasing, shared by every curve. |
| `batch_size` | integer | Curves per forward pass (default 4096). Lower it if you run out of memory. |

`g2` can have any number of leading dimensions, for example `(time, channel, n_tau)`. Each curve along the last axis is denoised independently.

### Output

A float64 NumPy array with the same shape as `g2`.

### Behaviour to be aware of

- **Units.** `taus` must be in seconds. If lag times fall outside the range the model was trained on, a warning is raised.
- **Zero lag.** A `tau = 0` bin (or any non-positive lag) cannot be log-normalised. It is not shown to the network and is returned unchanged.
- **Non-finite curves.** A curve containing NaN or inf is returned unchanged.
- **Output floor.** The output is always ≥ 1, because the network predicts `g2 − 1` through a softplus.
- **Noise statistics.** Denoised curves no longer carry the original measurement noise. Do not apply noise-weighted fits that assume it.

### Choosing the device

```python
denoiser = load_denoiser(device="cpu")     # or "cuda", "cuda:1", ...
```

## Valid range

The model was trained on lag times from 5×10⁻⁸ s to 5 s. Outside that range it is extrapolating.

<!-- TODO: fill in the training distribution from the paper, so users can judge whether their data are in range:
     source-detector separations, mua and musp ranges, beta range, aDb range, noise model and noise levels,
     and whether training curves were homogeneous or layered. -->

## Model

| | |
|---|---|
| Architecture | Bidirectional LSTM with a per-lag linear head and softplus output |
| Input channels | 2: log-normalised lag time, and `g2 − 1` |
| Hidden units | 96 per direction |
| Layers | 4 |
| Parameters | 745,153 |
| Lag-time normalisation | `log10(τ)` min-max scaled with bounds [-7.30091, 0.69885], mapped to [0.1, 1.0] |

The weights file, [dcs_denoiser/weights/dcs_lstm_denoiser_v1.pt](dcs_denoiser/weights/dcs_lstm_denoiser_v1.pt), holds the network weights, the architecture and the lag-time normalisation constants together. `load_denoiser` reads all three, so the model cannot be paired with the wrong normalisation.

If you want the network itself rather than the wrapper, it is `dcs_denoiser.Generalised_LSTMDenoiser`, a standard `torch.nn.Module`. Its docstring describes the raw tensor inputs, and `dcs_denoiser.normalise_taus` builds the lag-time channel.


## Checking your installation

The tests compare the model's output on a stored set of synthetic curves against the output of the released model:

```bash
pytest
```

Without pytest: `python tests/test_reference.py`.

## Citation

If you use this model or code, please cite it.

<!-- TODO: replace with the paper reference once it is available, and update CITATION.cff to match. -->

```
Fry, B., Mesquita, R,. C., dcs-denoiser: an LSTM denoiser for diffuse correlation spectroscopy (version 1.0.0). 2026.
https://github.com/BenFry199/CW-DCS-Denoiser
```

A paper describing the model is in preparation. Once it is published, please cite the paper as well.

## Licence

- **Code** is released under the [MIT Licence](LICENSE). Copies and substantial portions must keep the copyright notice.
- **Model weights** (`dcs_denoiser/weights/`) are released under [Creative Commons Attribution 4.0 International](LICENSE-WEIGHTS) (CC BY 4.0). You may use, share and adapt them, including commercially, provided you give credit: name the author, link to this repository and the licence, cite the work as given under [Citation](#citation), and say if you changed the weights.

Copyright © 2026 Ben Fry.
