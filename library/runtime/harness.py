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
    dynamic_seq: bool = False,
    n_token_families: Optional[int] = None,
    seq_range: Optional[tuple] = None,
) -> None:
    """``torch.compile`` each ``Block._forward`` for a distillation/training run.

    ``compile_blocks`` turns on native-shape flattening (every aspect bucket
    runs at its real token count, no padding → no flash pad-leak into the
    target) and traces one block graph per distinct token count. Distillation
    pools span more than the 2 ``CONSTANT_TOKEN_BUCKETS`` families, so pre-raise
    the dynamo cache to ``cache_size_limit`` (``compile_blocks``' own ``max()``
    won't lower it) so each shape traces instead of falling back to eager
    mid-warmup. No-op when ``enabled`` is False.

    ``dynamic_seq`` (mirrors the LoRA-training ``--compile_dynamic_seq`` path)
    collapses the per-token-count block graphs to a single graph by marking only
    the seq-length axis dynamic. ``seq_range`` bounds that symbolic axis and
    ``n_token_families`` sizes the dynamo cache budget over the active tiers;
    both default to the canonical 1024 table inside ``compile_blocks`` when
    ``None``.

    COMPILE LAST — install the adapter / network monkey-patches first, or
    torch.compile traces the wrong forward (the invariant ``build_anima``
    encodes).
    """
    if not enabled:
        return
    from library.runtime.dynamo import pin_dynamo_limit

    # Pin the canonical .default (not a context-local override) so the wider
    # distillation-pool budget survives into the backward compile context.
    pin_dynamo_limit("recompile_limit", cache_size_limit)
    anima.compile_blocks(
        backend,
        mode=mode,
        n_token_families=n_token_families,
        dynamic_seq=dynamic_seq,
        seq_range=seq_range,
    )


def compile_signature(
    *,
    n_token_families: Optional[int],
    seq_range: Optional[tuple],
    dynamic_seq: bool,
    backend: str = "inductor",
    mode: Optional[str] = None,
) -> str:
    """Canonical signature string for ``maybe_clear_stale_compile_cache``.

    Every compile entry point (``train.py``, ``scripts/distill_turbo``) must
    build the marker signature through this one formatter so equivalent compile
    configs serialize identically — a formatting drift between callers would
    thrash-wipe the shared inductor cache on every entry-point switch. ``mode``
    is normalized so the two "inductor default" spellings (``None`` and ``""``)
    don't read as a signature change.
    """
    return (
        f"families={n_token_families};seq_range={seq_range};"
        f"dynamic_seq={dynamic_seq};backend={backend};mode={mode or None}"
    )


# Original torch.compile cache base, captured on the first isolate_compile_cache
# call so repeated calls (or a different signature later in the same process)
# re-derive from the same root instead of nesting per-signature dirs.
_compile_cache_base: Optional[str] = None


def isolate_compile_cache(signature: str) -> str:
    """Route this run's torch.compile caches to a per-signature directory.

    The persistent compile caches (``FxGraphCache`` AND ``AOTAutogradCache``,
    both rooted at ``TORCHINDUCTOR_CACHE_DIR``) key on the FX graph but NOT on
    the ``mark_dynamic`` value range, so processes compiled with different
    seq-range bounds poison each other through the shared default cache dir.
    Concretely: inference/bench runs compile the block graph with the canonical
    1024-table default range and deposit entries whose stored guards are floored
    at ``seq >= 4032``; a later multi-tier training run marks ``[3000, 4200]``,
    and if its first compile's example batch happens to be ≥4032 tokens, the
    stale entry's guard evaluates TRUE at that hint — AOTAutogradCache accepts
    the hit and re-asserts the narrow guard into the fresh ShapeEnv
    (``autograd_cache.py::evaluate_guards``), which then contradicts the wider
    mark constraint → ``ConstraintViolationError`` (instead of a cache miss).
    Hint-dependent, which is why it strikes "sometimes": a sub-4032 first batch
    evaluates the guard False and misses cleanly.

    Wiping the shared dir (the previous approach) can't fix this: inference
    re-deposits default-range entries between training runs. Instead, point
    ``TORCHINDUCTOR_CACHE_DIR`` at a per-signature subdir of the original cache
    root — every entry inside was compiled under the SAME seq bounds, so guard
    replay is always consistent. Same-signature reruns keep their warm cache
    (and unlike the wipe, switching tier sets back and forth no longer
    re-compiles from scratch each time). Inference/bench keep the default dir.

    Must run BEFORE the first ``torch.compile`` trace in the process (torch
    reads the env var lazily per cache access). Build ``signature`` via
    ``compile_signature``. Returns the directory used.
    """
    global _compile_cache_base
    import hashlib
    import os

    if _compile_cache_base is None:
        base = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if not base:
            try:
                from torch._inductor.runtime.cache_dir_utils import default_cache_dir

                base = default_cache_dir()
            except Exception:  # noqa: BLE001 — torch internals move across versions
                import getpass
                import tempfile

                base = os.path.join(
                    tempfile.gettempdir(), f"torchinductor_{getpass.getuser()}"
                )
        _compile_cache_base = base

    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    target = os.path.join(_compile_cache_base, f"anima-sig-{digest}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = target
    log.info(f"torch.compile cache isolated per compile signature: {target}")
    log.info(f"compile signature: {signature}")
    return target


def _apply_activation_memory_budget(
    budget: float, *, grad_ckpt: bool, logger: logging.Logger = log
) -> None:
    """Cap the AOT min-cut partitioner's saved-for-backward set.

    The 2026-06-10 custom-autograd removal silently grew that set: the old
    ``LoRADownProjectFn`` was an explicit autograd boundary (save bf16 x +
    weight, recompute casts in backward), and once the rank path became plain
    traceable ops the partitioner chose to save ~0.8 GB more intermediates per
    step — first-step OOM on 16 GB at 4200 tokens without grad-ckpt.
    ``budget < 1.0`` makes it recompute cheap intermediates instead; 0.85
    reproduces the pre-removal footprint at identical step time (1.02 vs
    1.01 s/it, bench 2026-06-10).

    Must be set BEFORE the block compile — partitioning happens at
    first-forward compile, and this is a plain module attr (no ContextVar
    revert, unlike dynamo's recompile_limit). Skipped under gradient
    checkpointing: the budget repartitions the joint graph, so checkpoint's
    recompute pass can select a different graph than forward →
    ``CheckpointError`` (saved-vs-recomputed metadata mismatch, torch
    #166926). Ckpt already minimizes saved activations, so the cap buys
    nothing there.
    """
    if budget < 1.0 and not grad_ckpt:
        import torch._functorch.config as _functorch_config

        _functorch_config.activation_memory_budget = budget
        logger.info(
            "torch.compile activation_memory_budget = %.3g "
            "(partitioner recomputes cheap intermediates in backward)",
            budget,
        )
    elif budget < 1.0:
        logger.info(
            "activation_memory_budget ignored: incompatible with "
            "gradient_checkpointing (and redundant under it)"
        )


def compile_blocks_for_training(
    unet: object,
    network: object,
    *,
    backend: str,
    mode: Optional[str] = None,
    n_token_families: Optional[int] = None,
    seq_range: Optional[tuple] = None,
    dynamic_seq: bool = False,
    activation_memory_budget: float = 1.0,
    grad_ckpt: bool = False,
    logger: logging.Logger = log,
) -> None:
    """The LoRA-training (``train.py``) compile sequence, post ``apply_to``.

    Native-shape flattening + per-block torch.compile. COMPILE LAST — run only
    after ``network.apply_to`` + ``load_weights`` so dynamo traces the
    adapter's monkey-patched Linear forwards, not the bare DiT (the invariant
    ``build_anima`` encodes). ``compile_blocks`` turns on the flatten (one
    block graph per token-count family) and raises the dynamo cache-size
    budget itself, so unlike :func:`compile_dit_blocks_for_pool` no explicit
    ``recompile_limit`` pin is needed here.

    Sequence:
      1. :func:`_apply_activation_memory_budget` — partitioner cap, skipped
         under grad-ckpt (see its docstring for the history + CheckpointError
         interaction).
      2. ``isolate_compile_cache(compile_signature(...))`` — per-signature
         persistent-cache dir so a stale seq-range guard (e.g. an inference
         run's canonical 4032-floored range) can't poison this run's wider
         dynamic-seq marks with a ConstraintViolationError. Same signature →
         warm cache reuse.
      3. ``unet.compile_blocks(...)`` with the caller-derived token budget
         (``train.py::_derive_token_budget`` — the buckets the dataset
         actually populated, not ``args.target_res``).
      4. ``network.compile_cond_stream(...)`` when the adapter exposes it:
         EasyControl's patched ``Block.forward`` routes the active cond path
         through ``_two_stream_inner``, bypassing the just-compiled
         ``block._forward`` — so step 3 never reaches the cond stream (incl.
         the cond LoRA projections). Same backend/mode, same
         compile-after-apply ordering.
    """
    _apply_activation_memory_budget(
        activation_memory_budget, grad_ckpt=grad_ckpt, logger=logger
    )
    isolate_compile_cache(
        compile_signature(
            n_token_families=n_token_families,
            seq_range=seq_range,
            dynamic_seq=dynamic_seq,
            backend=backend,
            mode=mode,
        )
    )
    unet.compile_blocks(
        backend,
        mode=mode,
        n_token_families=n_token_families,
        dynamic_seq=dynamic_seq,
        seq_range=seq_range,
    )
    if hasattr(network, "compile_cond_stream"):
        network.compile_cond_stream(
            backend,
            mode=mode,
            n_token_families=n_token_families,
            dynamic_seq=dynamic_seq,
            seq_range=seq_range,
        )


@dataclass
class PoolCompileResult:
    """What :func:`compile_dit_blocks_for_pool` derived, for caller-side logging.

    ``n_token_families`` / ``seq_range`` are ``None`` on the static (per-shape)
    path; ``n_shapes`` is always the distinct-token-count tally (≥1).
    """

    n_shapes: int
    n_token_families: Optional[int]
    seq_range: Optional[tuple]


def compile_dit_blocks_for_pool(
    anima: object,
    token_counts,
    *,
    enabled: bool = True,
    dynamic_seq: bool = True,
    backend: str = "inductor",
    mode: Optional[str] = None,
    activation_memory_budget: float = 1.0,
    grad_ckpt: bool = False,
    cache_size_limit: Optional[int] = None,
    logger: logging.Logger = log,
) -> PoolCompileResult:
    """Partitioner budget → per-signature cache isolation → block compile, for a
    self-describing distillation pool.

    The verbatim compile-setup sequence ``distill_spd`` / ``distill_mod`` shared —
    and the home where the cross-cutting compile fixes belong (dynamo cache
    sizing, the AOTAutogradCache guard-poisoning isolation, the ``_functorch``
    partitioner budget). The caller derives ``token_counts`` — the distinct
    ``(W//patch)*(H//patch)`` counts its cached pool populates, coupled to each
    trainer's stage/synth pool logic, so it stays at the call site — and this
    owns everything after:

      1. ``n_shapes`` and, under ``dynamic_seq``, the symbolic-seq
         ``n_token_families`` / ``seq_range`` (one graph bounded by the pool's
         real token range; else ``None`` → per-shape static graphs).
      2. the partitioner ``activation_memory_budget`` — skipped + logged under
         ``grad_ckpt`` (repartitioning the joint graph trips CheckpointError,
         torch #166926); ckpt already minimizes saved activations.
      3. ``isolate_compile_cache(compile_signature(...))`` — a per-signature
         cache dir so a stale seq-range guard can't poison this run
         (ConstraintViolationError; see :func:`isolate_compile_cache`).
      4. ``compile_dit_blocks`` with the dynamo cache sized to ``2*n_shapes + 8``
         (override via ``cache_size_limit``).

    Returns the derived :class:`PoolCompileResult` (computed even when ``enabled``
    is False, so callers can still log it). Run AFTER the network ``apply_to`` —
    the compile-after-monkey-patch invariant ``build_anima`` encodes.
    """
    counts = {int(c) for c in token_counts}
    n_shapes = max(1, len(counts))
    if dynamic_seq and counts:
        # ONE symbolic-seq graph for every shape: mark_dynamic on the seq axis,
        # bounded by the pool's real token range.
        n_token_families: Optional[int] = n_shapes
        seq_range: Optional[tuple] = (min(counts), max(counts))
    else:
        # Static path: one graph per distinct token count, each at its real
        # token count (no padding). compile_blocks keys per token count.
        n_token_families = None
        seq_range = None
    result = PoolCompileResult(n_shapes, n_token_families, seq_range)

    if not enabled:
        return result

    # (2) Partitioner saved-activation cap — see _apply_activation_memory_budget
    # for the ordering constraint and the grad-ckpt CheckpointError interaction.
    _apply_activation_memory_budget(
        activation_memory_budget, grad_ckpt=grad_ckpt, logger=logger
    )

    # (3) Isolate the persistent compile caches per signature — entries compiled
    # under different seq-range bounds otherwise poison this run's dynamic-seq
    # marks (AOTAutogradCache replays a stale narrow guard → ConstraintViolationError).
    # Same signature → warm-cache reuse shared with the other training entry points.
    isolate_compile_cache(
        compile_signature(
            n_token_families=n_token_families,
            seq_range=seq_range,
            dynamic_seq=dynamic_seq,
            backend=backend,
            mode=mode,
        )
    )
    # (4) Compile each Block._forward (compile-after-apply invariant).
    compile_dit_blocks(
        anima,
        enabled=True,
        cache_size_limit=(
            2 * n_shapes + 8 if cache_size_limit is None else cache_size_limit
        ),
        backend=backend,
        mode=mode,
        dynamic_seq=dynamic_seq,
        n_token_families=n_token_families,
        seq_range=seq_range,
    )
    return result


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
