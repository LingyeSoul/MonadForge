#!/usr/bin/env python3
"""Run a LoRA training job in-process, driving AnimaTrainer directly.

`make lora` (→ `python tasks.py lora`) shells out to:

    accelerate launch train.py --method lora --preset default

That's the supported path, and the only one for multi-GPU. On a single GPU you
can skip the launcher and drive the trainer from Python — useful for embedding
training in a larger script, a notebook, or a custom sweep. This example
reproduces train.py's __main__ block:

    setup_parser() + populate_schema()  →  parse  →  read_config_from_file  →
    AnimaTrainer().train(args)

Prereqs: `make download-models` and `make preprocess` (training reads only the
cached latents/embeddings under post_image_dataset/lora/).

    python examples/04_train_lora.py
    python examples/04_train_lora.py --max_train_epochs 8 --network_dim 32

Any extra argv is forwarded verbatim to the trainer (same override semantics as
the CLI), so method settings still win over preset on overlap.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python examples/<script>.py`: put the repo root on sys.path so
# `import train` / `library` resolve. Still run from the repo root — configs/
# and the dataset cache are resolved relative to the CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import (
    AnimaTrainer,
    build_network_extras,
    setup_parser,
    verify_command_line_training_args,
)
from library.config import schema as config_schema

# `read_config_from_file` is re-exported from the `anima_lora` front door
# (AnimaTrainer / setup_parser stay on `train` — they're not part of the façade).
from anima_lora import read_config_from_file


def build_training_args(extra_argv: list[str]):
    """Reproduce train.py's argument assembly for a given method+preset."""
    argv = ["--method", "lora", "--preset", "default", *extra_argv]

    parser = setup_parser()
    # populate_schema adds the config-driven flags (incl. the network_module
    # str-extras that create_network reads); without it the routing keys are
    # missing from the namespace.
    config_schema.populate_schema(parser, extras=build_network_extras())

    args = parser.parse_args(argv)
    verify_command_line_training_args(args)
    # Applies the base→preset→method merge, then layers CLI overrides on top
    # (that's how `--network_dim 32` wins over the method file). We pass `argv`
    # explicitly so the override layer is driven by *our* list — not the process
    # sys.argv. (Default argv=None preserves the CLI behaviour for train.py.)
    args = read_config_from_file(args, parser, argv=argv)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility
    return args


def main() -> None:
    # Everything after this script's name is forwarded to the trainer as
    # overrides, e.g. `--max_train_epochs 8 --network_dim 32`.
    args = build_training_args(sys.argv[1:])
    AnimaTrainer().train(args)


if __name__ == "__main__":
    main()
