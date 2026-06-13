#!/usr/bin/env python3
"""From a method config to a built network to an in-process training run.

Three progressive parts — each builds on the previous, each opt-in via a flag so
the cheap part runs by default and the GPU-heavy parts stay explicit:

  1. load_method_preset() — the config merge chain (default; no GPU, no weights)
     base.toml → presets.toml[<preset>] → methods/<method>.toml → (CLI on top).
     This is what `train.py` does before it touches a single weight. The LoRA
     family is routed by a three-axis surface — use_moe_style / route_per_layer /
     router_source — which is just three keys in the merged dict; we print them.

  2. create_network()  (--build-network) — turn the resolved config into a live
     LoRA network bound to the DiT. Needs the DiT checkpoint.

  3. AnimaTrainer().train()  (--train) — actually run the training loop in-process.
     `make lora` (→ `python tasks.py lora`) shells out to
     `accelerate launch train.py --method lora --preset default`; that's the
     supported path and the only one for multi-GPU. On a single GPU you can skip
     the launcher and drive the trainer from Python — useful for embedding
     training in a larger script, a notebook, or a custom sweep.

    python examples/02_config_and_train.py --method lora --preset default
    python examples/02_config_and_train.py --method lora --build-network
    python examples/02_config_and_train.py --train --max_train_epochs 8 --network_dim 32

Part 3 prereq: `make download-models` and `make preprocess` (training reads only
the cached latents/embeddings under post_image_dataset/lora/). Any extra argv is
forwarded verbatim to the trainer (same override semantics as the CLI), so method
settings still win over preset on overlap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python examples/<script>.py`: put the repo root on sys.path so
# `import library` / `train` resolve. Model/config paths resolve against the
# repo home regardless of CWD (set ANIMA_HOME for a relocated checkout); only the
# output paths below are written relative to your CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# `load_method_preset` is re-exported from the `anima_lora` front door.
from anima_lora import load_method_preset

# Keys that make up the LoRA routing/shape surface — the values an adapter
# author cares about. (Full merged dict has ~150 keys spanning optimizer,
# dataset, logging, etc.)
NETWORK_KEYS = (
    "network_module",
    "network_dim",
    "network_alpha",
    "network_dropout",
    "use_moe_style",
    "route_per_layer",
    "router_source",
)


def show_config(method: str, preset: str) -> dict:
    """Part 1 — merge + print the network-relevant keys with provenance."""
    merged, provenance = load_method_preset(method, preset, return_provenance=True)

    print(f"\nmethod={method!r}  preset={preset!r}")
    print("-" * 72)
    for k in NETWORK_KEYS:
        src = provenance.get(k, "(unset → code default)")
        print(f"  {k:16} = {merged.get(k)!r:30}  ← {src}")
    print("-" * 72)
    print(f"  ({len(merged)} keys total in the merged config)\n")
    return merged


def build_network(merged: dict):
    """Part 2 — instantiate the network against the real DiT.

    Mirrors how train.py wires the adapter: the resolved routing keys are
    forwarded as **kwargs to the network module's create_network().
    """
    import torch

    from library.anima import weights as anima_weights
    from networks.lora_anima import create_network

    # We only need the raw DiT weights to bind a fresh network to — no LoRA / no
    # adapters to attach. So skip the namespace-driven load_dit_model and call the
    # explicit-argument primitive directly (as its docstring advises, and as
    # examples/04_load_models.py does). attn_mode "torch" is the portable default.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = anima_weights.load_anima_model(
        device=device,
        dit_path="models/diffusion_models/anima-base-v1.0.safetensors",
        attn_mode="torch",
        loading_device=device,
        dit_weight_dtype=torch.bfloat16,
    )

    # Forward the routing surface (skip None — let create_network use its own
    # defaults) plus any other str-valued knobs the module reads from kwargs.
    routing = {
        k: merged[k]
        for k in NETWORK_KEYS
        if merged.get(k) is not None and k != "network_module"
    }
    network = create_network(
        multiplier=1.0,
        network_dim=merged.get("network_dim"),
        network_alpha=merged.get("network_alpha"),
        vae=None,
        text_encoders=[],
        unet=unet,
        **{
            k: v
            for k, v in routing.items()
            if k not in ("network_dim", "network_alpha")
        },
    )
    n_params = sum(p.numel() for p in network.parameters())
    print(f"built {type(network).__name__}: {n_params:,} trainable params")
    return network


def run_training(method: str, preset: str, extra_argv: list[str]) -> None:
    """Part 3 — reproduce train.py's __main__ block and run the trainer.

    setup_parser() + populate_schema()  →  parse  →  read_config_from_file  →
    AnimaTrainer().train(args)
    """
    from train import (
        AnimaTrainer,
        build_network_extras,
        setup_parser,
        verify_command_line_training_args,
    )
    from library.config import schema as config_schema

    # `read_config_from_file` is re-exported from the `anima_lora` front door
    # (AnimaTrainer / setup_parser stay on `train` — not part of the façade).
    from anima_lora import read_config_from_file

    argv = ["--method", method, "--preset", preset, *extra_argv]

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

    AnimaTrainer().train(args)


def main() -> None:
    # parse_known_args so anything extra (e.g. --max_train_epochs 8) is forwarded
    # verbatim to the trainer in part 3, exactly like the CLI override layer.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", default="lora")
    p.add_argument("--preset", default="default")
    p.add_argument(
        "--build-network",
        action="store_true",
        help="part 2: also instantiate the network (loads the DiT — slow)",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="part 3: run the training loop in-process (needs preprocessed cache)",
    )
    opts, extra_argv = p.parse_known_args()

    if opts.train:
        # Training assembles + merges the config itself (via train.py's parser),
        # so go straight to the trainer with method/preset + forwarded overrides.
        run_training(opts.method, opts.preset, extra_argv)
        return

    merged = show_config(opts.method, opts.preset)
    if opts.build_network:
        build_network(merged)


if __name__ == "__main__":
    main()
