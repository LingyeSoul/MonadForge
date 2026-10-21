#!/usr/bin/env python3
"""The colorize scaling run as one daemon job: cache → train → eval per epoch save.

Each stage is a subprocess and the run stops at the first that fails, so a broken
cache never trains. The cache links ``colorize_te`` and ``colorize_manga`` (the
same seed-0 draw, so their pairs and the 12-page holdout carry over) and encodes
the rest with tag-dropped caption variants; the LoRA trains on the variants only,
as Anima's colorize recipe (``docs/proposal/moss_i2i.md``).

    make daemon-run ARGS="--queue --stall-timeout 1800 project/moss_lora/src/colorize_scale.py"
"""

import argparse
import subprocess
import sys

from library.env import resolve_under_home
from library.moss.requests import TrainRequest

SRC = "project/moss_lora/src"


def run(argv):
    print(f"[scale] {' '.join(argv)}", flush=True)
    subprocess.run([sys.executable, *argv], check=True, cwd=resolve_under_home("."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="moss_colorize_3k")
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
            *("--n", str(args.n), "--manga", str(args.manga)),
            *("--link_from", "post_image_dataset/moss/colorize_te"),
            *("--link_from", "post_image_dataset/moss/colorize_manga"),
            *("--out", cache, "--ref_scale", "2", "--variants", "4"),
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
        caption_variant_rate=1.0,
        activation_memory_budget=0.15,
        blocks_to_swap=28,
    )
    run(["scripts/moss/train.py", *train.to_argv()])
    epoch_dir = f"output/ckpt_moss/{args.name}"
    loras = [f"{epoch_dir}/{args.name}_e{e}.safetensors" for e in range(1, args.epochs)]
    for lora in [*loras, out]:
        run([f"{SRC}/colorize_eval.py", "--lora", lora, "--cache", cache])


if __name__ == "__main__":
    main()
