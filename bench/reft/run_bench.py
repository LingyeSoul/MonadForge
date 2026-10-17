#!/usr/bin/env python3
"""ReFT Phase-1' runner — baseline-reuse + register-synergy arms.

Trains the bench arms through the REAL trainer (train.py, same data pipeline /
loss stack as the reference checkpoint) with the bench-local network module
``bench.reft.reft_network`` — zero live-tree changes. The plain-LoRA reference
is NOT retrained: ``output/ckpt/bench_sincos_half_plain.safetensors`` (rank-16
plain LoRA, sincos/* @ sample_ratio 0.5, 4 epochs, lr 5e-5 — its
`.snapshot.toml` is the recipe this runner pins, because configs/methods/
lora.toml has since drifted to lr 2e-5 / 8 epochs).

Arms (see bench/reft/plan.md §Phase-1'):

  reft64        ReFT-only, d=64 last_8, all tokens (~2.1M params)
  reg36         registers-only control, K=36 @ block 8 (~74k params)
  reg36_reft64  registers + ReFT confined to the K register positions,
                same ReFT footprint as reft64 (last_8, all >= insert_block)

Each arm inherits `--method lora` so the aux-loss stack (REPA, masked loss,
shuffled caption variants) matches the baseline run; every lora-family net
kwarg (rank/alpha/hydra/t-mask toggles) is ignored by the bench module.

Run (from anima_lora/; expects the sincos caches from the baseline run)::

    uv run python bench/reft/run_bench.py [--arms reft64 reg36 reg36_reft64]
        [--dry_run] [--force] [--skip_eval]

Sequential single-GPU runs (~baseline cost each). If a daemon training job is
live, run this after it drains — the runner does not queue.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

BASELINE = "output/ckpt/bench_sincos_half_plain.safetensors"

# Training-loop knobs pinned from the baseline's .snapshot.toml (method-TOML
# drift protection). Everything else inherits --method lora / --preset default.
RECIPE = [
    "--method", "lora",
    "--preset", "default",
    "--network_module", "bench.reft.reft_network",
    "--path_pattern", "sincos/*",
    "--sample_ratio", "0.5",
    "--learning_rate", "5e-5",
    "--max_train_epochs", "4",
    "--save_every_n_epochs", "4",
    "--checkpointing_epochs", "4",
    "--caption_dropout_rate", "0.1",
]

ARMS: dict[str, list[str]] = {
    "reft64": [
        "reft_dim=64",
        "reft_layers=last_8",
        "reft_positions=all",
        "num_registers=0",
    ],
    "reg36": [
        "reft_layers=none",
        "num_registers=36",
        "insert_block=8",
    ],
    "reg36_reft64": [
        "reft_dim=64",
        "reft_layers=last_8",
        "reft_positions=registers",
        "num_registers=36",
        "insert_block=8",
    ],
}


def ckpt_path(arm: str) -> Path:
    return REPO_ROOT / "output" / "ckpt" / f"bench_sincos_half_{arm}.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--force", action="store_true", help="Retrain existing arms.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--label", default="reft_phase1p")
    args = ap.parse_args()

    if not (REPO_ROOT / BASELINE).exists():
        sys.exit(f"baseline reference missing: {BASELINE}")

    for arm in args.arms:
        out = ckpt_path(arm)
        if out.exists() and not args.force:
            print(f"[{arm}] exists, skipping: {out.name}", flush=True)
            continue
        cmd = [
            sys.executable,
            "train.py",
            *RECIPE,
            "--output_name", out.stem,
            "--network_args", *ARMS[arm],
        ]
        print(f"[{arm}] {' '.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    if args.skip_eval:
        return
    adapters = [BASELINE] + [
        str(ckpt_path(a).relative_to(REPO_ROOT)) for a in args.arms
    ]
    missing = [a for a in adapters if not (REPO_ROOT / a).exists()]
    if missing:
        sys.exit(f"eval skipped, checkpoints missing: {missing}")
    cmd = [
        sys.executable,
        "bench/reft/eval_cmmd.py",
        "--adapters", *adapters,
        "--label", args.label,
    ]
    print(f"[eval] {' '.join(cmd)}", flush=True)
    if not args.dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
