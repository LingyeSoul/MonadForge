"""Shared Anima-loading helpers for bench/ scripts.

`bench/_common.py` owns the result envelope. This module owns the
*model-side* boilerplate every DiT-loading bench rewrites: argparse
surface, DiT + adapter loading in the correct order, bucketed sample
discovery from `post_image_dataset/lora`-style cache layouts.

The compile-after-apply ordering is the load-bearing invariant:
``torch.compile`` traces the adapter's monkey-patched forward, so
``compile_blocks`` MUST run after ``network.apply_to`` + ``load_weights``.
Open-coding this in every bench means newcomers either skip ``--compile``
entirely or call it in the wrong order; ``build_anima`` exists to
remove the choice.

Usage::

    from bench._anima import add_common_args, build_anima, discover_bucketed_samples

    p = argparse.ArgumentParser()
    p.add_argument("--dit", required=True)
    p.add_argument("--adapter", default=None)
    add_common_args(p)            # injects --label/--seed/--device/--dtype/
                                  # --attn_mode/--gradient_checkpointing/
                                  # --cpu_offload_checkpointing/--compile/--compile_mode
    args = p.parse_args()

    anima, network = build_anima(args, adapter=args.adapter, train_mode=False)
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
import glob
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("bench._anima")

# --dtype choices. The string→torch.dtype mapping itself lives in the canonical
# library helper (library.runtime.device.str_to_dtype); resolve_dtype delegates
# to it so bench doesn't carry a parallel table.
_DTYPE_CHOICES = ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32")


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
    if include_device:
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        parser.add_argument(
            "--device",
            type=str,
            default=default_device,
            help="Compute device. Default: cuda if available, else cpu.",
        )
    if include_dtype:
        parser.add_argument(
            "--dtype",
            type=str,
            choices=_DTYPE_CHOICES,
            default="bf16",
            help="Model dtype. bf16 is the production default.",
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
    from library.runtime.device import str_to_dtype

    return str_to_dtype(name)


# ---------------------------------------------------------------------------
# DiT + adapter loading.
# ---------------------------------------------------------------------------


@dataclass
class AnimaBundle:
    """Container for a built Anima model + optional adapter.

    Returned by ``build_anima``. ``network`` is ``None`` when no adapter
    was requested. ``device`` / ``dtype`` are the resolved torch values
    (callers that need them on the device side don't have to re-resolve).
    """

    anima: object  # anima_models.Anima — typed as object to avoid heavy import
    network: Optional[object]  # networks.lora_anima.network.LoRANetwork or None
    device: torch.device
    dtype: torch.dtype


def build_anima(
    args: argparse.Namespace,
    *,
    dit_path: str | None = None,
    adapter: str | None = None,
    train_mode: bool = False,
    network_requires_grad: bool = True,
    multiplier: float = 1.0,
) -> AnimaBundle:
    """Load the DiT (+ optional adapter) with the correct ordering.

    Sequence — read the inline comments before changing:
        1. ``load_anima_model`` → DiT on device, weights cast to ``dtype``.
        2. ``anima.requires_grad_(False)`` + ``reset_mod_guidance``.
        3. If ``adapter``: ``create_network_from_weights`` → ``apply_to``
           → ``load_weights`` → ``network.to(device, dtype)``.
        4. If ``--gradient_checkpointing``: enable it (gated by
           ``anima.training`` — so train mode below must come after).
        5. ``anima.train()`` / ``anima.eval()`` per ``train_mode``. Network
           same mode (the training-time LoRA forward path is gated on
           ``network.training`` — T-LoRA mask, fp32 bottleneck, etc.).
        6. **``compile_blocks`` last** — the adapter monkey-patches must
           already be installed or torch.compile traces the wrong forward.

    Arguments:
        args: argparse namespace populated by ``add_common_args``. Reads
            ``device``, ``dtype``, ``attn_mode``, ``gradient_checkpointing``,
            ``cpu_offload_checkpointing``, ``compile``, ``compile_mode``.
        dit_path: Path to the base DiT safetensors. Defaults to
            ``args.dit`` if the bench's argparse exposed one.
        adapter: Optional adapter safetensors path. When set, the adapter
            is loaded and applied with ``multiplier`` as the apply-time
            scale.
        train_mode: If True, both anima + network are put in train mode.
            Required for any bench that calls ``backward()`` — the LoRA
            training-path forward, T-LoRA mask, and gradient checkpointing
            are all gated on ``self.training``.
        network_requires_grad: When ``train_mode=True`` and ``adapter`` is
            set, controls whether adapter params have ``requires_grad=True``
            (default) or are frozen.
        multiplier: Adapter forward-time multiplier. ``set_multiplier(0.0)``
            can flip it later to recover the base-model output.

    Returns:
        ``AnimaBundle(anima, network, device, dtype)``.
    """
    # Late imports — the bench/_anima module should import cheaply even on
    # CPU-only smoke runs that never load a DiT.
    from library.anima import weights as anima_utils

    device = torch.device(getattr(args, "device", "cuda"))
    dtype = resolve_dtype(getattr(args, "dtype", "bf16"))
    attn_mode = getattr(args, "attn_mode", "flash")

    if dit_path is None:
        dit_path = getattr(args, "dit", None)
    if dit_path is None:
        raise SystemExit(
            "build_anima: no DiT path. Pass dit_path= explicitly or expose "
            "--dit in your argparse."
        )

    log.info(f"loading base DiT: {dit_path}")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=dit_path,
        attn_mode=attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device, dtype=dtype).requires_grad_(False)
    anima.reset_mod_guidance()

    network = None
    if adapter is not None:
        log.info(f"loading adapter:  {adapter}")
        # Late import — adapter machinery has its own load-time cost.
        from networks.lora_anima.factory import create_network_from_weights

        network, _sd = create_network_from_weights(
            multiplier,
            adapter,
            None,  # ae (unused for bench)
            None,  # text_encoders (unused for bench)
            anima,
            for_inference=not train_mode,
        )
        network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
        info = network.load_weights(adapter)
        log.info(f"adapter loaded — {info}")

        network.to(device=device, dtype=dtype)
        if train_mode and network_requires_grad:
            network.requires_grad_(True)
        else:
            network.requires_grad_(False)
        anima.requires_grad_(False)  # always — DiT stays frozen in bench

        trainable = [p for p in network.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        if train_mode and network_requires_grad:
            if n_train == 0:
                raise SystemExit(
                    "build_anima: adapter loaded with train_mode=True but "
                    "no trainable parameters were detected. Check the "
                    "checkpoint."
                )
            log.info(
                f"adapter trainable params: {n_train:,} ({len(trainable)} tensors)"
            )

    # Grad checkpointing is gated on anima.training (see models.py); set the
    # flag here but its effect requires train_mode below.
    if getattr(args, "gradient_checkpointing", False):
        cpu_off = getattr(args, "cpu_offload_checkpointing", False)
        suffix = " (cpu offload)" if cpu_off else ""
        log.info(f"enabling gradient checkpointing{suffix}")
        anima.enable_gradient_checkpointing(cpu_offload=cpu_off)

    if train_mode:
        anima.train()
        if network is not None:
            network.train()
    else:
        anima.eval()
        if network is not None:
            network.eval()

    # COMPILE LAST. Adapter monkey-patches must be installed first or
    # torch.compile traces the wrong forward.
    if getattr(args, "compile", False):
        mode = getattr(args, "compile_mode", None)
        log.info(
            f"compiling DiT blocks{' (mode=' + mode + ')' if mode else ''} "
            "— first batch pays ~30-60s compile cost"
        )
        anima.compile_blocks(mode=mode)

    return AnimaBundle(anima=anima, network=network, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Bucketed sample discovery.
# ---------------------------------------------------------------------------

_RES_RE = re.compile(r"_(\d{3,5})x(\d{3,5})_anima\.npz$")


def discover_bucketed_samples(
    data_dir: Path,
    bucket: str | None,
    num_samples: int,
    seed: int,
    *,
    allow_replace: bool = False,
) -> tuple[str, list[tuple[str, str, str, str]]]:
    """Scan ``data_dir`` for (latent npz, TE sidecar) pairs grouped by bucket.

    Filename convention: ``{stem}_{Wpix}x{Hpix}_anima.npz`` paired with
    ``{stem}_anima_te.safetensors``. Items without a matching TE sidecar
    are skipped. ``latents_{WxH}`` keys inside the npz define the bucket
    string.

    Args:
        data_dir: e.g. ``Path("post_image_dataset/lora")``.
        bucket: Bucket string like ``"128x192"`` (latent dims, not pixel
            dims). If None, the most populous bucket is chosen.
        num_samples: How many samples to return.
        seed: For the np.random.choice.
        allow_replace: If True and the pool is smaller than
            ``num_samples``, resample with replacement (logs a warning).
            If False (default), raises.

    Returns:
        ``(chosen_bucket, [(stem, latent_key, npz_path, te_path), ...])``.

    Raises:
        SystemExit: if no pairs are found, the requested bucket is empty,
            or the pool is too small and ``allow_replace=False``.
    """
    npz_paths = sorted(glob.glob(str(data_dir / "*_anima.npz")))
    if not npz_paths:
        raise SystemExit(f"no `*_anima.npz` in {data_dir}")

    by_bucket: dict[str, list[tuple[str, str, str, str]]] = {}
    for p in npz_paths:
        name = Path(p).name
        m = _RES_RE.search(name)
        if not m:
            continue
        stem = name[: m.start()]
        te = data_dir / f"{stem}_anima_te.safetensors"
        if not te.exists():
            continue
        with np.load(p) as z:
            for k in z.keys():
                if k.startswith("latents_"):
                    bk = k.removeprefix("latents_")
                    by_bucket.setdefault(bk, []).append((stem, k, p, str(te)))
                    break

    if not by_bucket:
        raise SystemExit("no paired (latent, TE) samples found")

    chosen = bucket or max(by_bucket, key=lambda k: len(by_bucket[k]))
    if chosen not in by_bucket:
        top = sorted(((k, len(v)) for k, v in by_bucket.items()), key=lambda x: -x[1])[
            :5
        ]
        raise SystemExit(f"bucket {chosen!r} not found. Top buckets: {top}")

    pool = by_bucket[chosen]
    if len(pool) < num_samples:
        if not allow_replace:
            raise SystemExit(
                f"bucket {chosen!r} has {len(pool)} samples; need {num_samples}. "
                f"Pass allow_replace=True to resample with replacement."
            )
        log.warning(
            f"bucket {chosen!r} has {len(pool)} samples; resampling with "
            f"replacement to reach {num_samples}."
        )

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=num_samples, replace=(len(pool) < num_samples))
    return chosen, [pool[i] for i in idx]
