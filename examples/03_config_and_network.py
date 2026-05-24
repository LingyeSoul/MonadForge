#!/usr/bin/env python3
"""The building blocks: resolve a method config, then build a LoRA network.

Two parts:

  1. load_method_preset() — the config merge chain
     base.toml → presets.toml[<preset>] → methods/<method>.toml → (CLI on top).
     Runs anywhere; no GPU, no model files. This is what `train.py` does before
     it touches a single weight.

  2. create_network() — turn the resolved config into a live LoRA network bound
     to the DiT. Needs the DiT checkpoint, so it's opt-in behind --build-network.

The LoRA family is routed by a three-axis surface — use_moe_style /
route_per_layer / router_source — which is just three keys in the merged dict.
Print them to see what a given method+preset actually selects.

    python examples/03_config_and_network.py --method lora --preset default
    python examples/03_config_and_network.py --method lora --build-network
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python examples/<script>.py`: put the repo root on sys.path so
# `import library` / `inference` resolve. Still run from the repo root —
# configs/ and models/ paths are resolved relative to the CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library.config.io import load_method_preset

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

    import inference
    from library.inference.models import load_dit_model
    from networks.lora_anima import create_network

    # A minimal args namespace just to drive the DiT loader.
    args = inference.parse_args(
        [
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--text_encoder",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--prompt",
            "x",  # unused; satisfies the parser's "need a prompt" check
            "--save_path",
            "/tmp/unused.png",
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    unet = load_dit_model(args, device, torch.bfloat16)

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", default="lora")
    p.add_argument("--preset", default="default")
    p.add_argument(
        "--build-network",
        action="store_true",
        help="also instantiate the network (loads the DiT — slow, needs weights)",
    )
    opts = p.parse_args()

    merged = show_config(opts.method, opts.preset)
    if opts.build_network:
        build_network(merged)


if __name__ == "__main__":
    main()
