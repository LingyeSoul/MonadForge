#!/usr/bin/env python3
"""Run a colorize LoRA as one daemon job: cache → train → eval of the final LoRA.

Each stage is a subprocess; the run stops at the first that fails.
``scripts/moss/colorize_cache.py`` builds ``post_image_dataset/moss/<name>``; the
training recipe is ``moss_colorize_3k_ocr``'s (``configs/moss/i2i/colorize/train.toml``).
OCR captions run up to ~2.2k text rows, so training starts on the worst case
(``--fit_report``) with expandable segments.

    make daemon-run ARGS="--queue --stall-timeout 1800 project/moss_lora/src/colorize_run.py --name <name>"
"""

import argparse
import subprocess
import sys

from library.env import resolve_under_home
from library.moss.requests import ColorizeCacheRequest, TrainRequest

SRC = "project/moss_lora/src"


def run(argv):
    print(f"[colorize] {' '.join(argv)}", flush=True)
    subprocess.run([sys.executable, *argv], check=True, cwd=resolve_under_home("."))


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

    cache_req = ColorizeCacheRequest(out=cache, n=args.n, manga=args.manga)
    run([cache_req.SCRIPT, *cache_req.to_argv()])
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
        expandable_segments=True,
    )
    run([train.SCRIPT, *train.to_argv()])
    run([f"{SRC}/colorize_eval.py", "--lora", out, "--cache", cache])


if __name__ == "__main__":
    main()
