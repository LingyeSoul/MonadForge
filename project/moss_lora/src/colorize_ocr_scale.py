#!/usr/bin/env python3
"""``moss_colorize_3k`` with the page's OCR in the caption, as one daemon job:
cache → train → eval of the final LoRA (``colorize_ocr.md``).

The cache is ``colorize_ocr.py`` over ``moss_colorize_3k`` (its pairs, latents,
refs and 12-page holdout; fp8 TE): ``embeds`` is the layout caption, the three
``embeds_v{k}`` the tag-dropped captions with the OCR text + box clause. The
LoRA is ``colorize_scale.py``'s request with ``caption_variant_rate 0.3``, so
captioned steps take the layout 70 % of the time. The OCR captions run up to
~2.2k text rows (3k: ~1.2k), so training starts on the worst case
(``--fit_report``) with expandable segments: the first run OOMed in epoch 2 with
1.9 GB reserved but unallocated.

    make daemon-run ARGS="--queue --stall-timeout 1800 project/moss_lora/src/colorize_ocr_scale.py"
"""

import argparse
import os
import subprocess
import sys

from library.env import resolve_under_home
from library.moss.requests import TrainRequest

SRC = "project/moss_lora/src"


def run(argv, env=None):
    print(f"[ocr_scale] {' '.join(argv)}", flush=True)
    subprocess.run(
        [sys.executable, *argv],
        check=True,
        cwd=resolve_under_home("."),
        env={**os.environ, **(env or {})},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="moss_colorize_3k_ocr")
    ap.add_argument("--src", default="post_image_dataset/moss/moss_colorize_3k")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=10000)
    args = ap.parse_args()
    cache = f"post_image_dataset/moss/{args.name}"
    out = f"output/ckpt_moss/{args.name}.safetensors"

    run(
        [
            f"{SRC}/colorize_ocr.py",
            *("--src", args.src, "--holdout_from", args.src, "--out", cache),
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
    run(
        [
            f"{SRC}/colorize_eval.py",
            *("--lora", out, "--cache", cache, "--te_precision", "fp8"),
        ]
    )


if __name__ == "__main__":
    main()
