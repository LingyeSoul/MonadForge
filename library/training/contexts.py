"""Shared training/validation context dataclasses.

``TrainCtx``/``ValCtx`` are frozen bundles built once at the top of
``train()`` and threaded through per-step / per-batch methods on the trainer
plus the loop runner in :mod:`library.training.loop`. ``RuntimeState`` is the
mutable counterpart -- per-run feature state that methods mutate as training
progresses. All live here (rather than in ``train.py``) so ``loop.py`` and any
future trainer entrypoints can import them directly instead of receiving them
as injected class parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from accelerate import Accelerator


@dataclass(frozen=True)
class TrainCtx:
    """Training-wide state built once near the top of ``train()`` and passed to
    per-step / per-batch methods instead of 15-arg parameter lists. Fields here
    are fixed for the whole training run -- per-call values (epoch, global_step,
    progress_bar, logging keys, …) stay explicit at call sites."""

    args: Any
    accelerator: Accelerator
    network: Any
    unet: Any
    vae: Any
    text_encoders: list
    noise_scheduler: Any
    text_encoding_strategy: Any
    tokenize_strategy: Any
    vae_dtype: torch.dtype
    weight_dtype: torch.dtype
    train_text_encoder: bool
    train_unet: bool
    optimizer_eval_fn: Callable
    optimizer_train_fn: Callable
    is_tracking: bool


@dataclass(frozen=True)
class ValCtx:
    """Validation-wide state fixed for the entire training run. The per-call
    val_loss_recorder (step vs epoch) stays explicit since it differs per call
    site; everything else here is shared."""

    dataloader: Any
    sigmas: list
    steps: int
    total_steps: int
    train_loss_recorder: Any
    original_t_min: float
    original_t_max: float
    # The val DatasetGroup itself. Held so CMMD-style validation can enumerate
    # held-out items (absolute_path, caption, bucket_reso, text_encoder_outputs_npz)
    # for paired sample generation against the cached PE reference pool.
    dataset_group: Any = None


@dataclass
class RuntimeState:
    """Per-run mutable state that's threaded across trainer methods.

    Unlike the frozen ``*Ctx`` bundles above, these fields are mutated as
    training progresses. Grouped together so the lifecycle of each feature's
    state is documented in one place rather than scattered as bare attributes.
    """

    # Per-step aux dict -- adapters' ``extra_forwards`` returns are merged
    # here in ``get_noise_pred_and_target`` and consumed by the loss composer
    # in ``_process_batch_inner``.
    extras_for_step: dict = field(default_factory=dict)
    # EMA λ state, mutated by the flow_matching_vr loss handler each step. The
    # "frozen reference" for the AsymFlow §5.2 control variate is just the
    # trainable DiT with ``network.set_multiplier(0)`` — see the VR block in
    # ``get_noise_pred_and_target``.
    vr: dict = field(default_factory=lambda: {"lambda_ema": None})
    # T5("") crossattn sidecar (shape ``(1, S, 1024)`` bf16 on device).
    # Populated by ``_ensure_uncond_crossattn`` when caption dropout is
    # enabled; consumed by ``prepare_text_conds`` so dropped rows match
    # Anima's CFG-uncond inference path instead of falling back to zeros.
    uncond_crossattn_1: torch.Tensor | None = None
    # Set during dataset prep from subset.caption_dropout_rate; gates whether
    # ``_ensure_uncond_crossattn`` actually stages the sidecar.
    caption_dropout_enabled: bool = False
