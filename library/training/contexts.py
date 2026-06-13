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


@dataclass(frozen=True)
class DatasetBundle:
    """Return of ``AnimaTrainer._prepare_dataset``: the train/val dataset groups
    plus the shared collator and the ``multiprocessing.Value`` step/epoch
    counters the collator and loop read."""

    train_group: Any
    val_group: Any
    current_epoch: Any  # multiprocessing.Value("i", ...)
    current_step: Any
    collator: Any
    use_user_config: bool
    use_dreambooth_method: bool


@dataclass(frozen=True)
class NetworkBundle:
    """Return of ``AnimaTrainer._create_and_apply_network``: the built+applied
    adapter network and the train/eval flags derived while applying it."""

    network: Any
    net_kwargs: dict
    train_unet: bool
    train_text_encoder: bool


@dataclass(frozen=True)
class OptimizerBundle:
    """Return of ``AnimaTrainer._setup_optimizer_and_dataloader``: optimizer (+
    its name/args and train/eval fns), LR schedule, and the train/val
    dataloaders, all pre-``accelerator.prepare``."""

    optimizer: Any
    optimizer_name: str
    optimizer_args: Any
    optimizer_train_fn: Callable
    optimizer_eval_fn: Callable
    text_encoder_lr: Any
    lr_descriptions: Any
    train_dataloader: Any
    val_dataloader: Any
    lr_scheduler: Any


@dataclass(frozen=True)
class AcceleratedBundle:
    """Return of ``AnimaTrainer._prepare_with_accelerator``: the
    ``accelerator.prepare``-wrapped network/optimizer/dataloaders/scheduler plus
    the re-resolved model handles (the wrapped DiT, the cast/prepared text
    encoders, and the chosen unet weight dtype)."""

    network: Any
    optimizer: Any
    train_dataloader: Any
    val_dataloader: Any
    lr_scheduler: Any
    training_model: Any
    unet: Any
    text_encoders: list
    text_encoder: Any
    unet_weight_dtype: torch.dtype


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
