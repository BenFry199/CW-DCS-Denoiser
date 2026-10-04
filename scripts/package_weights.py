"""
Package a raw training checkpoint into the single file dcs_denoiser loads.

A training run saves a bare state_dict, and the tau normalisation bounds live in
the separate preprocessing .npz the run was trained on. This script combines
them -- weights, LSTM shape, tau_norm and provenance -- into one checkpoint, so
the published model cannot be paired with the wrong normalisation.

Only needed when publishing a new model; users of the package never run it.

Run:
    python scripts/package_weights.py \
        --checkpoint .../run_04_seed1982049089/model_best.pt \
        --preproc-npz .../preproc_020926.npz \
        --out dcs_denoiser/weights/dcs_lstm_denoiser_v1.pt
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch


def infer_lstm_arch(state_dict):
    """Recover (input_size, hidden_size, num_layers, bidirectional) from a
    Generalised_LSTMDenoiser state_dict."""
    ih0 = state_dict["lstm.weight_ih_l0"]
    hidden_size = ih0.shape[0] // 4          # LSTM stacks 4 gate matrices
    input_size = ih0.shape[1]
    bidirectional = "lstm.weight_ih_l0_reverse" in state_dict
    layer_ids = [int(k.split("_l")[1].split("_")[0])
                 for k in state_dict if k.startswith("lstm.weight_ih_l")]
    return dict(input_size=int(input_size), hidden_size=int(hidden_size),
                num_layers=max(layer_ids) + 1, bidirectional=bidirectional)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="raw training checkpoint (a state_dict .pt/.pth)")
    ap.add_argument("--preproc-npz", required=True, type=Path,
                    help="preprocessing .npz the checkpoint was trained on (holds 'tau_norm')")
    ap.add_argument("--out", required=True, type=Path, help="packaged checkpoint to write")
    args = ap.parse_args()

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    arch = infer_lstm_arch(state_dict)
    with np.load(args.preproc_npz, allow_pickle=True) as d:
        tau_min, tau_max = (float(v) for v in np.asarray(d["tau_norm"]).ravel()[:2])

    # Plain tensors / floats / strings only, so the file loads with weights_only=True.
    ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "arch": arch,
        "tau_norm": [tau_min, tau_max],      # log10(seconds)
        "tau_scale": [0.1, 1.0],             # normalise_taus low / high
        "tau_eps": 1e-12,                    # normalise_taus log floor
        "meta": {
            "source_run": args.checkpoint.parent.name,
            "source_checkpoint": args.checkpoint.name,
            "source_checkpoint_sha256": sha256(args.checkpoint),
            "preproc_file": args.preproc_npz.name,
            "n_parameters": int(sum(v.numel() for v in state_dict.values())),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)

    print(f"wrote  {args.out}")
    print(f"arch   {arch}")
    print(f"tau_norm [{tau_min:.5f}, {tau_max:.5f}]  ->  "
          f"[{10 ** tau_min:.3g}, {10 ** tau_max:.3g}] s")
    for k, v in ckpt["meta"].items():
        print(f"{k}: {v}")
    print(f"packaged sha256: {sha256(args.out)}")


if __name__ == "__main__":
    main()
