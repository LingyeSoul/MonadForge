"""Slim an RSD student .pth to an ema-only .safetensors for release/ComfyUI.

The training ckpt (`rsd_student_*.pth`) bundles both `ema` and `student` branches
(~955 MB) plus the arch metadata (`noise_mode` / `noise_channels`). Inference uses
ONLY the `ema` weights, so this drops the `student` branch (halves the file to
~478 MB) and rewrites as safetensors (no pickle, faster mmap load).

safetensors stores tensors only, so the arch metadata — which the loader needs to
rebuild StochasticUNet's widened first conv — is carried in the file's string
metadata header (all values stringified; the node reads them back via
`safetensors.safe_open(...).metadata()`).

    python export_student_safetensors.py --ckpt output/sr/rsd/rsd_student_final.pth \
        --out rsd_student_12k.safetensors

Then upload `rsd_student_12k.safetensors` to huggingface.co/sorryhyun/Distilled-ResShift-4x.
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE.parents[1] / "output" / "sr" / "rsd" / "rsd_student_final.pth"),
                    help="training ckpt with an 'ema' key")
    ap.add_argument("--out", default="rsd_student_12k.safetensors",
                    help="output safetensors path")
    ap.add_argument("--branch", choices=["ema", "student"], default="ema",
                    help="which branch to export (ema = the deployable 1-step generator)")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    if args.branch not in sd:
        raise SystemExit(f"{args.branch!r} not in ckpt keys: {list(sd)}")
    weights = sd[args.branch]

    # safetensors requires contiguous, non-aliased tensors; clone to be safe.
    tensors = {k: v.contiguous().clone() for k, v in weights.items()}

    # Arch metadata the node needs to rebuild the widened first conv. safetensors
    # metadata values must be strings; the loader casts noise_channels back to int.
    meta = {
        "noise_mode": str(sd.get("noise_mode", "concat")),
        "noise_channels": str(sd.get("noise_channels", 1)),
        # Which `lq` conditioning the student was distilled under (rsd_models.make_cond_lq).
        # Absent from pre-2026-07-11 ckpts -> latent, matching the loader's own fallback.
        # Feeding the wrong one is off-manifold and renders neon green, so it must ride
        # along with the weights rather than be guessed by the consumer.
        "cond_lq": str(sd.get("cond_lq", "latent")),
        "step": str(sd.get("step", "")),
        "branch": args.branch,
        "arch": "resshift_swinunet_stochastic",
        "format": "pt",  # lets safetensors.torch.load_file infer the framework
    }

    save_file(tensors, args.out, metadata=meta)
    mb = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out} ({mb:.1f} MB, {len(tensors)} tensors) meta={meta}")


if __name__ == "__main__":
    main()
