#!/usr/bin/env python3
"""A colorize LoRA as one daemon job: cache → train → eval of the final LoRA.

Each stage is a subprocess and the run stops at the first that fails, so a broken
cache never trains. ``colorize_cache.py`` builds ``post_image_dataset/moss/<name>``;
the LoRA is ``moss_colorize_3k_ocr``'s recipe: lr 5e-5, 3 epochs, the page alone
on 30 % of steps, of the rest the OCR caption 70 % and a sparse caption 30 %. The
OCR captions run up to ~2.2k text rows, so training starts on the worst case
(``--fit_report``) with expandable segments.

    make daemon-run ARGS="--queue --stall-timeout 1800 project/moss_lora/src/colorize_run.py --name <name>"
"""

import argparse
import os
import subprocess
import sys

from library.env import resolve_under_home
from library.moss.requests import TrainRequest

SRC = "project/moss_lora/src"


def run(argv, env=None):
    print(f"[colorize] {' '.join(argv)}", flush=True)
    subprocess.run(
        [sys.executable, *argv],
        check=True,
        cwd=resolve_under_home("."),
        env={**os.environ, **(env or {})},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=1667, help="illustration pairs")
    ap.add_argument("--manga", type=int, default=1667, help="manga pages")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=10000)
    args = ap.parse_args()
    cache = f"post_image_dataset/moss/{args.name}"
    out = f"output/ckpt_moss/{args.name}.safetensors"

    run(
        [
            f"{SRC}/colorize_cache.py",
            *("--out", cache, "--n", str(args.n), "--manga", str(args.manga)),
        ]
    )
    train = TrainRequest(
        cache=cache,
        output=out,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=5e-5,
        save_every_epochs=1,
        plain_caption_rate=0.3,
        caption_variant_rate=0.3,
        activation_memory_budget=0.15,
        blocks_to_swap=28,
        fit_report=True,
    )
    run(
        ["scripts/moss/train.py", *train.to_argv()],
        env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    run([f"{SRC}/colorize_eval.py", "--lora", out, "--cache", cache])


if __name__ == "__main__":
    main()
