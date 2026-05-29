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
# `import library` / `inference` resolve. Model/config paths resolve against the
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
    # examples/05 does). attn_mode "torch" is the portable default.
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
