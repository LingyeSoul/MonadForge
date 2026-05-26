"""Shared Anima-loading helpers for bench/ scripts.

`bench/_common.py` owns the result envelope. This module owns the
*bench-facing* CLI surface (`add_common_args`) and re-exports the run
harness that now lives in `library/`:

- ``build_anima`` / ``AnimaBundle`` → ``library.runtime.harness`` (the
  DiT + adapter loader with the compile-after-apply ordering).
- ``discover_bucketed_samples`` → ``library.io.cache`` (next to the other
  cache-discovery helpers).

They were promoted out of bench so ``bench`` / ``scripts`` / ``preprocess``
and the low-level probes share one harness instead of copying it. The
re-exports here keep ``from bench._anima import build_anima, …`` working.

The compile-after-apply ordering is the load-bearing invariant:
``torch.compile`` traces the adapter's monkey-patched forward, so
``compile_blocks`` MUST run after ``network.apply_to`` + ``load_weights``.
See ``library.runtime.harness`` for the full sequence.

Usage::

    from bench._anima import add_common_args, build_anima, discover_bucketed_samples

    p = argparse.ArgumentParser()
    p.add_argument("--dit", required=True)
    p.add_argument("--adapter", default=None)
    add_common_args(p)            # injects --label/--seed/--device/--dtype/
                                  # --attn_mode/--gradient_checkpointing/
                                  # --cpu_offload_checkpointing/--compile/--compile_mode
    args = p.parse_args()

    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    bucket, picks = discover_bucketed_samples(
        Path("post_image_dataset/lora"), args.bucket, args.num_samples, args.seed
    )

All helpers are opt-in. A bench that doesn't load the DiT (e.g. an
analytical simulator) simply doesn't import this module. A bench that
needs two DiTs (e.g. ``bench/fm_vr_headroom``) calls ``build_anima``
twice with explicit ``dit_path=`` overrides.
"""

from __future__ import annotations

import argparse

import torch

# Re-exports — the harness moved into library/ (see module docstring). Imported
# here so existing `from bench._anima import build_anima` call sites don't churn.
from library.io.cache import discover_bucketed_samples  # noqa: F401
from library.runtime.cli import add_device_args
from library.runtime.device import str_to_dtype
from library.runtime.harness import AnimaBundle, build_anima  # noqa: F401

__all__ = [
    "add_common_args",
    "resolve_dtype",
    "build_anima",
    "AnimaBundle",
    "discover_bucketed_samples",
]


# ---------------------------------------------------------------------------
# Common argparse surface.
# ---------------------------------------------------------------------------


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    include_label: bool = True,
    include_seed: bool = True,
    include_device: bool = True,
    include_dtype: bool = True,
    include_model: bool = True,
    include_checkpointing: bool = True,
    include_compile: bool = True,
) -> argparse.ArgumentParser:
    """Inject the bench-common CLI surface.

    All groups are individually opt-out so a bench can skip what doesn't
    apply (e.g. a CPU-only analytical script can drop ``include_device``).
    ``--device`` / ``--dtype`` delegate to
    ``library.runtime.cli.add_device_args`` (the shared compute flag group).

    Flags injected at defaults:
        --label             free-form run label, fed to make_run_dir
        --seed              int, default 0
        --device            "cuda" | "cpu" | "cuda:N", default "cuda" if available
        --dtype             bf16|fp16|fp32, default bf16
        --attn_mode         flash|torch|..., default "flash"
        --gradient_checkpointing  bool flag
        --cpu_offload_checkpointing  bool flag
        --compile           bool flag — torch.compile DiT blocks
        --compile_mode      str, default None (inductor default)
    """
    if include_label:
        parser.add_argument(
            "--label",
            type=str,
            default=None,
            help="Free-form label appended to the run directory name.",
        )
    if include_seed:
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="RNG seed for sample discovery and noise draws.",
        )
    add_device_args(
        parser, include_device=include_device, include_dtype=include_dtype
    )
    if include_model:
        parser.add_argument(
            "--attn_mode",
            type=str,
            default="flash",
            help="Attention backend (flash, torch, ...). Default: flash.",
        )
    if include_checkpointing:
        parser.add_argument(
            "--gradient_checkpointing",
            action="store_true",
            help="Enable activation checkpointing on the DiT. Trades ~30%% "
            "compute for ~4-5x smaller activation footprint. Required for "
            "benches that backward through the full DiT at high resolutions.",
        )
        parser.add_argument(
            "--cpu_offload_checkpointing",
            action="store_true",
            help="With --gradient_checkpointing, additionally CPU-offload "
            "the checkpointed activations. Further VRAM savings at higher "
            "compute cost.",
        )
    if include_compile:
        parser.add_argument(
            "--compile",
            action="store_true",
            help="torch.compile each DiT block (via DiT.compile_blocks). First "
            "batch pays the compile cost (~30-60s); subsequent batches run "
            "faster. compile_blocks runs AFTER adapter apply_to + load_weights "
            "so the LoRA monkey-patches are part of the compiled graph.",
        )
        parser.add_argument(
            "--compile_mode",
            type=str,
            default=None,
            help="Optional inductor mode for compile_blocks (e.g. "
            "'reduce-overhead'). Leave unset for the default.",
        )
    return parser


def resolve_dtype(name: str) -> torch.dtype:
    """Map a --dtype string to a torch dtype. Raises ValueError on unknown.

    Delegates to ``library.runtime.device.str_to_dtype`` (the canonical home,
    also exported as ``anima_lora.str_to_dtype``) so the bench harness shares one
    source of truth for the mapping.
    """
    return str_to_dtype(name)
