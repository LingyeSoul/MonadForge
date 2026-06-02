"""Shared DiT + adapter run harness.

The *model-side* boilerplate every DiT-loading tool rewrites: load the DiT,
attach an optional adapter, and apply ``torch.compile`` — all in the one
ordering the pipeline actually requires. Promoted out of ``bench/`` (where it
was ``bench/_anima.py::build_anima``) so ``bench`` / ``scripts`` / ``preprocess``
and low-level probes share a single harness instead of copying it.

The compile-after-apply ordering is the load-bearing invariant:
``torch.compile`` traces the adapter's monkey-patched forward, so
``compile_blocks`` MUST run after ``network.apply_to`` + ``load_weights``.
Open-coding this means newcomers either skip ``--compile`` entirely or call it
in the wrong order; ``build_anima`` exists to remove the choice.

Usage::

    from library.runtime.harness import build_anima

    bundle = build_anima(args, dit_path=..., adapter=..., train_mode=False)
    anima, network = bundle.anima, bundle.network

``build_anima`` reads its knobs off an argparse ``Namespace`` (``device`` /
``dtype`` / ``attn_mode`` / ``gradient_checkpointing`` /
``cpu_offload_checkpointing`` / ``compile`` / ``compile_mode``); the matching
parser surface lives in ``library.runtime.cli.add_device_args`` +
``bench._anima.add_common_args``. Callers without a parser can pass a plain
``argparse.Namespace(**kwargs)``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Optional

import torch

log = logging.getLogger("library.runtime.harness")


@dataclass
class AnimaBundle:
    """Container for a built Anima model + optional adapter.

    Returned by ``build_anima``. ``network`` is ``None`` when no adapter
    was requested. ``device`` / ``dtype`` are the resolved torch values
    (callers that need them on the device side don't have to re-resolve).
    """

    anima: object  # library.anima Anima — typed as object to avoid heavy import
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
        args: argparse namespace populated by ``add_device_args`` /
            ``bench._anima.add_common_args``. Reads ``device``, ``dtype``,
            ``attn_mode``, ``gradient_checkpointing``,
            ``cpu_offload_checkpointing``, ``compile``, ``compile_mode``.
        dit_path: Path to the base DiT safetensors. Defaults to
            ``args.dit`` if the caller's argparse exposed one.
        adapter: Optional adapter safetensors path. When set, the adapter
            is loaded and applied with ``multiplier`` as the apply-time
            scale.
        train_mode: If True, both anima + network are put in train mode.
            Required for any caller that calls ``backward()`` — the LoRA
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
    # Late imports — this module should import cheaply even on CPU-only smoke
    # runs that never load a DiT.
    from library.anima import weights as anima_utils
    from library.runtime.device import str_to_dtype

    device = torch.device(getattr(args, "device", "cuda"))
    dtype = str_to_dtype(getattr(args, "dtype", "bf16"))
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
            None,  # ae (unused for harness callers)
            None,  # text_encoders (unused for harness callers)
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
        anima.requires_grad_(False)  # always — DiT stays frozen in the harness

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


@dataclass
class InferenceBundle:
    """Everything a probe needs to drive (and hook) a real ``generate()`` call.

    Where :class:`AnimaBundle` stops at *DiT + adapter*, this carries the full
    inference set — text encoder, VAE, the resolved ``GenerationSettings`` — plus
    the ``shared_models`` dict already primed with ``model`` / ``text_encoder`` /
    ``conds_cache``. Hand ``shared_models`` straight to
    ``library.inference.generate(args, gen_settings, shared_models)``; because the
    DiT is pre-loaded into ``shared_models["model"]``, ``generate()`` reuses *this*
    instance — so any forward-hook / monkeypatch you install on ``bundle.model``
    before generating is live during sampling (the trick batch inference uses, now
    a first-class seam). ``vae`` is ``None`` when the bundle was built with
    ``with_vae=False`` or no ``--vae`` path.
    """

    model: object  # the loaded DiT (also stashed in shared_models["model"])
    vae: Optional[object]  # AutoencoderKLQwenImage, or None
    text_encoder: object  # on CPU; generate()/prepare_text_inputs moves it
    gen_settings: object  # library.inference.GenerationSettings
    shared_models: dict  # {"model", "text_encoder", "conds_cache"} -> generate()
    args: argparse.Namespace  # the namespace the bundle was built from
    device: torch.device

    def generate(self, args: Optional[argparse.Namespace] = None):
        """Run ``library.inference.generate`` reusing this bundle's loaded models.

        Defaults to the namespace the bundle was built from; pass a per-call
        ``args`` (e.g. a different prompt/seed) to override. Returns the latent
        tensor ``generate()`` produces.
        """
        from library.inference import generate as _generate

        return _generate(args or self.args, self.gen_settings, self.shared_models)


def build_inference_bundle(
    args: argparse.Namespace,
    device: torch.device | str | None = None,
    *,
    with_vae: bool = True,
) -> InferenceBundle:
    """Assemble the text-encoder + DiT (+ optional VAE) set for a generation.

    The inference-side counterpart to :func:`build_anima`. ``inference.main()``
    open-codes this sequence (load text encoder, load DiT, stash it in
    ``shared_models["model"]`` so ``generate()`` reuses the instance, load the
    VAE); a bench/probe that observes or perturbs a *real* generation had to
    reverse-engineer it. This bundles it once.

    Sequence:
        1. ``get_generation_settings(args)`` → resolved device (cuda-else-cpu).
        2. ``load_shared_models(args)`` → text encoder on CPU; add ``conds_cache``.
        3. ``load_dit_model(args, device, bf16)`` → DiT, stashed in
           ``shared_models["model"]`` so ``generate()`` reuses it (the hook seam).
        4. If ``with_vae`` and ``args.vae`` is set: ``load_vae(..., bf16, eval)``.

    Args:
        args: a fully-defaulted namespace (``inference.parse_args`` /
            ``GenerationRequest.to_args()`` / ``build_default_args``). Reads
            ``vae`` / ``text_encoder`` / ``dit`` / adapter + sampler knobs.
        device: optional explicit device; when given it's written back to
            ``args.device`` so every downstream loader agrees. ``None`` resolves
            cuda-else-cpu via ``get_generation_settings``.
        with_vae: load the VAE (needed only to decode latents → pixels). A
            latent-space probe can pass ``False`` to skip the load.

    Returns:
        :class:`InferenceBundle` — pass ``.shared_models`` to ``generate()`` or
        call ``.generate()``.
    """
    # Late imports — keep this module import-cheap on CPU-only smoke runs, and
    # avoid an import-time edge into the inference engine.
    from library.inference import (
        get_generation_settings,
        load_dit_model,
        load_shared_models,
    )

    if device is not None:
        # Pin it on the namespace so get_generation_settings + load_dit_model all
        # resolve to the same device (mirrors inference.main()).
        args.device = str(device) if not isinstance(device, str) else device

    gen_settings = get_generation_settings(args)
    resolved_device = gen_settings.device

    shared_models = load_shared_models(args)  # text encoder on CPU
    shared_models["conds_cache"] = {}

    anima = load_dit_model(args, resolved_device, torch.bfloat16)
    # Stash so generate() reuses *this* instance — the only seam for hooking the
    # DiT before the sampler loop runs.
    shared_models["model"] = anima

    vae = None
    vae_path = getattr(args, "vae", None)
    if with_vae and vae_path:
        from library.models.qwen_vae import load_vae

        vae = load_vae(
            vae_path,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=getattr(args, "vae_chunk_size", None),
            disable_cache=getattr(args, "vae_disable_cache", False),
            dtype=torch.bfloat16,
            eval=True,
        )
    elif with_vae:
        log.warning(
            "build_inference_bundle(with_vae=True) but args.vae is unset; "
            "bundle.vae is None (latent decode will be unavailable)."
        )

    return InferenceBundle(
        model=anima,
        vae=vae,
        text_encoder=shared_models["text_encoder"],
        gen_settings=gen_settings,
        shared_models=shared_models,
        args=args,
        device=resolved_device,
    )


# --- Training-side build helpers -------------------------------------------
#
# ``build_anima`` above owns the *inference / existing-adapter* path: it loads a
# checkpoint with ``create_network_from_weights`` + ``load_weights``. The
# distillation trainers (``scripts/distill_{mod,spd,turbo}.py``) instead build a
# *fresh, untrained* network (or train an in-model MLP), each with its own
# ordering of freeze / optimizer / per-forward swap toggles — so they can't call
# ``build_anima`` wholesale. These three composable helpers factor out the parts
# that were copied verbatim across all three (the block-swap placement, the
# dynamo-cache-bump + ``compile_blocks``, and the grad-checkpoint toggle) without
# imposing a single ordering. Call them in whatever order your trainer needs;
# the compile-after-monkey-patch invariant still applies — run
# ``compile_dit_blocks`` only after the network's ``apply_to``.


def place_dit_for_training(
    anima: object, device: torch.device, *, blocks_to_swap: int = 0
) -> None:
    """Move a (frozen-base) DiT onto ``device`` for a training run.

    With block swap on, the swapped blocks stay on CPU and ride the
    forward+backward swap hooks while everything else moves to ``device``;
    without it the whole model moves. This arms the *training* swap path (two
    block movements per step) — distinct from the inference placement
    ``build_anima`` does. Call before ``compile_dit_blocks`` / ``train()``.
    """
    if blocks_to_swap > 0:
        anima.enable_block_swap(blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.switch_block_swap_for_training()  # forward+backward block movement
    else:
        anima.to(device)


def compile_dit_blocks(
    anima: object,
    *,
    enabled: bool = True,
    cache_size_limit: int = 64,
    backend: str = "inductor",
    mode: Optional[str] = None,
) -> None:
    """``torch.compile`` each ``Block._forward`` for a distillation/training run.

    ``compile_blocks`` turns on native-shape flattening (every aspect bucket
    runs at its real token count, no padding → no flash pad-leak into the
    target) and traces one block graph per distinct token count. Distillation
    pools span more than the 2 ``CONSTANT_TOKEN_BUCKETS`` families, so pre-raise
    the dynamo cache to ``cache_size_limit`` (``compile_blocks``' own ``max()``
    won't lower it) so each shape traces instead of falling back to eager
    mid-warmup. No-op when ``enabled`` is False.

    COMPILE LAST — install the adapter / network monkey-patches first, or
    torch.compile traces the wrong forward (the invariant ``build_anima``
    encodes).
    """
    if not enabled:
        return
    import torch._dynamo as _dynamo

    _dynamo.config.cache_size_limit = max(
        _dynamo.config.cache_size_limit, cache_size_limit
    )
    anima.compile_blocks(backend, mode=mode)


def enable_training_grad_ckpt(anima: object, *, enabled: bool) -> None:
    """Toggle unsloth CPU-offload gradient checkpointing for a training run.

    Recomputes block activations in backward, offloading saved tensors to CPU
    between forward/backward. The model must stay in ``train()`` mode —
    ``Block.forward`` gates checkpointing on ``self.training``. Logs and no-ops
    when ``enabled`` is False.
    """
    if enabled:
        anima.enable_gradient_checkpointing(unsloth_offload=True)
        log.info("gradient checkpointing: on (unsloth CPU offload)")
    else:
        log.info("gradient checkpointing: off")
