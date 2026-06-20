#!/usr/bin/env python3
"""Bake a LoRA adapter into the base DiT and save as a new safetensors file.

Thin CLI shell over ``library.anima.merge`` — see that module for the merge
semantics (what's bakeable, the load→build→merge→save path).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from library.anima.merge import (  # noqa: E402
    DTYPE_MAP,
    NonBakeableError,
    merge_adapter_into_dit,
    pick_latest_adapter,
)
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bake a LoRA adapter into the base DiT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adapter_dir",
        type=Path,
        default=Path("output/ckpt"),
        help="Directory to pick the latest adapter from (ignored if --adapter is set).",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Explicit adapter .safetensors path (overrides --adapter_dir).",
    )
    parser.add_argument(
        "--dit",
        type=Path,
        default=Path("models/diffusion_models/anima-base-v1.0.safetensors"),
        help="Base DiT safetensors.",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="LoRA strength to bake in.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Defaults to <adapter-stem>_merged.safetensors next to the adapter.",
    )
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the merge math.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Drop unsupported keys (Hydra moe / postfix / prefix) and bake the rest. "
        "The merged DiT will not reproduce those components.",
    )
    parser.add_argument(
        "--network_module",
        default="networks.lora_anima",
        help="Network module providing create_network_from_weights.",
    )
    args = parser.parse_args()

    adapter = args.adapter or pick_latest_adapter(args.adapter_dir)

    try:
        merge_adapter_into_dit(
            adapter=adapter,
            dit=args.dit,
            out=args.out,
            multiplier=args.multiplier,
            dtype=DTYPE_MAP[args.dtype],
            device=args.device,
            allow_partial=args.allow_partial,
            network_module=args.network_module,
        )
    except NonBakeableError as e:
        logger.error(str(e))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
