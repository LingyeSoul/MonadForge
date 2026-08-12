"""Training-loop orchestration.

Owns the per-epoch / per-step body that used to live inline in
``AnimaTrainer.train()``. The entrypoint is :func:`run_training_loop`, which
takes a built :class:`LoopState` plus the trainer instance so override hooks
(``process_batch``, ``on_step_start``, ``sample_images``,
``generate_step_logs``, ``step_logging``, ``epoch_logging``) keep working
unchanged. The validation pass lives in :mod:`library.training.validation`.

State that used to be on ``self`` for cross-call signaling —
``_last_router_H_postfix``, ``_cudagraph_mark_step``, ``_hydra_warmup_step``,
``_adapters`` — stays on the trainer; this module reads them through the
``trainer`` handle.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm

from library import train_util
from library.datasets import LossRecorder
from library.runtime.device import clean_memory_on_device
from library.training.checkpoints import CheckpointSaver
from library.training.contexts import TrainCtx, ValCtx
from library.training.log_dispatch import dispatch_wandb_extras, effective_lr
from library.training.method_adapter import StepCtx
from library.training.metrics import MetricContext, collect_metrics
from library.training.stage_schedule import (
    StageRuntimePlan,
    apply_active_subsets_to_dataset,
    log_stage_switch,
    progress_from_steps,
    resolve_stage_index,
)
from library.training.state import capture_rng_state
from library.training.stop import StopController
from library.training.validation import run_validation
from library.training.wandb_metrics import (
    GradientHistogramCollector,
    SystemMetricsCollector,
    WeightSnapshotCollector,
)

logger = logging.getLogger(__name__)

# Liveness early check (issues.md P1.1): late enough that warmups / partial
# sidecar coverage have had a chance to fire at least once, early enough that
# a silently-dead feature aborts a strict run in minutes instead of hours.
LIVENESS_EARLY_CHECK_STEP = 25


def release_text_encoder_handles(text_encoder, text_encoders):
    """Remove text-encoder references from the loop-facing state.

    Rebinding a loop-local variable is insufficient because the original list
    can still be aliased by the caller or an accelerator bundle. Clear it before
    discarding the scalar handle so cached-only encoders can be reclaimed.
    """
    if isinstance(text_encoders, list):
        text_encoders.clear()
    return None, []


@dataclass
class LoopState:
    """Bundles every local that used to live in ``train()``'s for-epoch scope.

    Most fields are constants for the run; ``global_step``, ``profile_started``,
    ``profile_range``, ``initial_step``, and ``text_encoder(s)`` are mutated
    during the loop. ``current_epoch`` / ``current_step`` are mp.Value handles
    shared with :class:`CheckpointSaver` for state persistence.
    """

    args: Any
    accelerator: Accelerator
    train_ctx: TrainCtx
    val_ctx: ValCtx
    saver: CheckpointSaver

    network: Any
    unet: Any
    text_encoder: Any
    text_encoders: list
    vae: Any
    tokenizers: Any
    training_model: Any
    train_dataloader: Any
    optimizer: Any
    lr_scheduler: Any
    lr_descriptions: Optional[list]
    optimizer_train_fn: Callable
    optimizer_eval_fn: Callable
    weight_dtype: Any
    unet_weight_dtype: Any

    current_epoch: Any  # mp.Value
    current_step: Any  # mp.Value
    num_train_epochs: int
    epoch_to_start: int
    initial_step: int

    metadata: dict
    is_tracking: bool
    progress_bar: Any
    loss_recorder: LossRecorder
    val_step_loss_recorder: LossRecorder
    val_epoch_loss_recorder: LossRecorder

    validation_steps: int

    profile_range: Optional[tuple]
    on_step_start_for_network: Callable

    stage_plan: Optional[StageRuntimePlan] = None
    stage_index: int = -1
    stage_batch_cursor: int = 0
    stage_loader_generator_state: Any = None
    outer_epoch_index: int = 0

    # Cooperative daemon/CLI stop state.  The flag is inspected only after a
    # complete optimizer step so an interrupted run never saves half a gradient
    # accumulation window.
    stop_controller: Optional[StopController] = None
    stop_requested: bool = False
    micro_batch_offset: int = 0
    at_optimizer_boundary: bool = True

    global_step: int = 0
    profile_started: bool = False

    def checkpoint_runtime_state(self) -> dict[str, Any]:
        # Keep the provider tolerant of lightweight test/embedding owners that
        # predate the explicit cursor fields.  The saver supplies its shared
        # epoch/step defaults when a field is omitted; real LoopState instances
        # always expose every field below.
        runtime_state: dict[str, Any] = {}
        if hasattr(self, "global_step"):
            runtime_state["global_step"] = int(self.global_step)
        if hasattr(self, "current_epoch"):
            runtime_state["current_epoch"] = int(
                getattr(self.current_epoch, "value", self.current_epoch)
            )
        if hasattr(self, "micro_batch_offset"):
            runtime_state["micro_batch_offset"] = int(self.micro_batch_offset)
        runtime_state["rng_state"] = capture_rng_state()
        args = getattr(self, "args", None)
        if args is not None:
            runtime_state["config_signature"] = getattr(args, "config_signature", None)
            runtime_state["dataset_signature"] = getattr(args, "dataset_signature", None)
        if self.stage_plan is None:
            return runtime_state
        runtime_state.update({
            "stage_index": self.stage_index,
            "stage_batch_cursor": self.stage_batch_cursor,
            "stage_outer_epoch": self.outer_epoch_index,
        })
        if self.stage_loader_generator_state is not None:
            runtime_state["stage_loader_generator_state"] = (
                self.stage_loader_generator_state.tolist()
            )
        return runtime_state


def build_loop_state(
    trainer,
    *,
    args,
    accelerator: Accelerator,
    saver: CheckpointSaver,
    network,
    unet,
    text_encoder,
    text_encoders,
    vae,
    tokenizers,
    training_model,
    train_dataloader,
    val_dataloader,
    val_dataset_group,
    optimizer,
    lr_scheduler,
    lr_descriptions,
    optimizer_train_fn,
    optimizer_eval_fn,
    weight_dtype,
    unet_weight_dtype,
    vae_dtype,
    text_encoding_strategy,
    tokenize_strategy,
    train_text_encoder,
    train_unet,
    current_epoch,
    current_step,
    num_train_epochs,
    epoch_to_start,
    initial_step,
    metadata,
    stage_plan=None,
    initial_global_step=None,
    stage_batch_cursor=0,
    stage_loader_generator_state=None,
    stop_controller: Optional[StopController] = None,
) -> LoopState:
    """Build :class:`LoopState`. Mirrors the pre-loop setup that used to sit
    between ``_prepare_with_accelerator()`` and the for-epoch loop in
    ``train()``: noise scheduler, trackers, loss recorders, optional text
    encoder eviction, ``--sample_at_first``, train/val ctx construction,
    progress bar, profiler parsing.
    """
    noise_scheduler = trainer.get_noise_scheduler(args, accelerator.device)

    train_util.init_trackers(accelerator, args, "network_train")

    # Initialize wandb-enhanced collectors when wandb tracker is active.
    # Stored on trainer for _log_step to pick up.
    trainer._wandb_collectors = {}
    _has_wandb = "wandb" in [t.name for t in accelerator.trackers]
    if _has_wandb:
        trainer._wandb_collectors["sys"] = SystemMetricsCollector()
        _grad_freq = int(getattr(args, "log_every_n_steps", 1) or 1) * 50
        trainer._wandb_collectors["grad"] = GradientHistogramCollector(freq=_grad_freq)
        trainer._wandb_collectors["weight"] = WeightSnapshotCollector()

    loss_recorder = LossRecorder()
    val_step_loss_recorder = LossRecorder()
    val_epoch_loss_recorder = LossRecorder()

    if hasattr(accelerator.unwrap_model(network), "on_step_start"):
        on_step_start_for_network = accelerator.unwrap_model(network).on_step_start
    else:

        def on_step_start_for_network(*args, **kwargs):
            return None

    if trainer.is_text_encoder_not_needed_for_training(args):
        logger.info("text_encoder is not needed for training. deleting to save memory.")
        text_encoder, text_encoders = release_text_encoder_handles(
            text_encoder, text_encoders
        )
        gc.collect()
        clean_memory_on_device(accelerator.device)

    optimizer_eval_fn()
    trainer.sample_images(
        accelerator,
        args,
        0,
        0,
        accelerator.device,
        vae,
        tokenizers,
        text_encoder,
        unet,
        network=network,
    )
    optimizer_train_fn()
    is_tracking = len(accelerator.trackers) > 0
    if is_tracking:
        accelerator.log({}, step=0)

    train_ctx = TrainCtx(
        args=args,
        accelerator=accelerator,
        network=network,
        unet=unet,
        vae=vae,
        text_encoders=text_encoders,
        noise_scheduler=noise_scheduler,
        text_encoding_strategy=text_encoding_strategy,
        tokenize_strategy=tokenize_strategy,
        vae_dtype=vae_dtype,
        weight_dtype=weight_dtype,
        train_text_encoder=train_text_encoder,
        train_unet=train_unet,
        optimizer_eval_fn=optimizer_eval_fn,
        optimizer_train_fn=optimizer_train_fn,
        is_tracking=is_tracking,
    )

    # Resume skip prelude: fast-forward global_step before tqdm so the bar
    # total is sized right, and consume per-epoch skip credit so
    # skip_first_batches has the right first-epoch offset.
    global_step = int(initial_global_step) if initial_global_step is not None else 0
    if initial_global_step is None and initial_step > 0:
        global_step = initial_step // args.gradient_accumulation_steps
        for skip_epoch in range(epoch_to_start):
            logger.info(
                f"skipping epoch {skip_epoch + 1} because initial_step "
                f"(multiplied) is {initial_step}"
            )
            initial_step -= len(train_dataloader)

    logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
    _ts_parts = [f"timestep_sampling={args.timestep_sampling}"]
    if args.timestep_sampling in ("sigmoid", "shift", "flux_shift"):
        _ts_parts.append(f"sigmoid_scale={args.sigmoid_scale}")
        _ts_parts.append(f"sigmoid_bias={getattr(args, 'sigmoid_bias', 0.0)}")
    if args.timestep_sampling in ("shift", "flux_shift"):
        _ts_parts.append(f"discrete_flow_shift={args.discrete_flow_shift}")
    if (
        getattr(args, "t_min", None) is not None
        or getattr(args, "t_max", None) is not None
    ):
        _ts_parts.append(
            f"σ∈[{getattr(args, 't_min', None)}, {getattr(args, 't_max', None)}]"
        )
    logger.info("sigma sampling: " + ", ".join(_ts_parts))
    for i, t_enc in enumerate(text_encoders):
        params_itr = t_enc.parameters()
        params_itr.__next__()
        params_itr.__next__()  # CLIP first two params are embeddings
        param_3rd = params_itr.__next__()
        logger.info(
            f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}"
        )

    clean_memory_on_device(accelerator.device)

    # Use the ABSOLUTE step range so the bar shows ``global_step /
    # max_train_steps`` on both a fresh run and a resume. The training loop
    # calls ``progress_bar.update(1)`` once per optimizer step and
    # ``global_step`` is seeded with the resume offset above, so starting the
    # iterator at ``initial=global_step`` keeps tqdm's internal ``n`` in lock
    # step with ``global_step``. Without this, the bar sized itself to
    # ``max_train_steps - global_step`` (the remaining count) while the JSONL
    # progress sink logged the absolute ``global_step`` — the WebUI's two
    # metric channels (stdout tqdm parser vs JSONL) then raced on ``step`` /
    # ``total_steps`` and produced impossible reads like ``172/138``.
    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )

    validation_steps = (
        min(args.max_validation_steps, len(val_dataloader))
        if args.max_validation_steps is not None
        else len(val_dataloader)
    )
    # Fixed sigma values across the schedule: 0.1 near-clean / fine detail,
    # 0.4 mid / bulk structure, 0.7 high noise / coarse denoising.
    validation_sigmas = (
        args.validation_sigmas
        if args.validation_sigmas is not None
        else [0.1, 0.4, 0.7]
    )
    val_ctx = ValCtx(
        dataloader=val_dataloader,
        sigmas=validation_sigmas,
        steps=validation_steps,
        total_steps=validation_steps * len(validation_sigmas),
        train_loss_recorder=loss_recorder,
        original_t_min=args.t_min,
        original_t_max=args.t_max,
        dataset_group=val_dataset_group,
    )

    # nsys workflow: --profile_steps START-END toggles the cuda profiler API
    # around the requested step window. Wrap the launch with
    #   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
    # so nsys only records that window.
    profile_range = trainer._parse_profile_steps(args)

    return LoopState(
        args=args,
        accelerator=accelerator,
        train_ctx=train_ctx,
        val_ctx=val_ctx,
        saver=saver,
        network=network,
        unet=unet,
        text_encoder=text_encoder,
        text_encoders=text_encoders,
        vae=vae,
        tokenizers=tokenizers,
        training_model=training_model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_descriptions=lr_descriptions,
        optimizer_train_fn=optimizer_train_fn,
        optimizer_eval_fn=optimizer_eval_fn,
        weight_dtype=weight_dtype,
        unet_weight_dtype=unet_weight_dtype,
        current_epoch=current_epoch,
        current_step=current_step,
        num_train_epochs=num_train_epochs,
        epoch_to_start=epoch_to_start,
        initial_step=initial_step,
        metadata=metadata,
        is_tracking=is_tracking,
        progress_bar=progress_bar,
        loss_recorder=loss_recorder,
        val_step_loss_recorder=val_step_loss_recorder,
        val_epoch_loss_recorder=val_epoch_loss_recorder,
        validation_steps=validation_steps,
        profile_range=profile_range,
        on_step_start_for_network=on_step_start_for_network,
        stage_plan=stage_plan,
        stage_batch_cursor=stage_batch_cursor,
        stage_loader_generator_state=stage_loader_generator_state,
        stop_controller=stop_controller,
        global_step=global_step,
    )


def _log_checkpoint_artifact(trainer, args, state: LoopState, epoch: int) -> None:
    """Log the latest checkpoint as a wandb artifact, if wandb is active."""
    _wb = getattr(trainer, "_wandb_collectors", {})
    if not _wb or not getattr(args, "_wandb_log_artifact", False):
        return
    try:
        import wandb

        if not wandb.run:
            return
        # Construct the expected checkpoint path using the same helpers
        # the CheckpointSaver uses internally.
        from library.training.checkpoints import (
            get_epoch_ckpt_name,
            get_step_ckpt_name,
        )

        ext = "." + args.save_model_as
        ckpt_name = get_epoch_ckpt_name(args, ext, epoch + 1)
        ckpt_path = os.path.join(args.output_dir, ckpt_name)
        if not os.path.isfile(ckpt_path):
            ckpt_name = get_step_ckpt_name(args, ext, state.global_step)
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
        if os.path.isfile(ckpt_path):
            artifact = wandb.Artifact(
                name=f"{args.output_name or 'model'}_step_{state.global_step}",
                type="model",
            )
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)
    except Exception as e:
        logger.warning(f"WandB checkpoint artifact logging failed: {e}")


def run_training_loop(trainer, state: LoopState) -> None:
    """Run the full for-epoch training loop and the post-loop end-of-training
    metadata write. Mutates ``state.global_step``, profiler bookkeeping, and
    the metadata dict; the per-checkpoint saves go through ``state.saver``.
    """
    args = state.args
    accelerator = state.accelerator
    _maybe_apply_stage_schedule(state)

    for epoch in range(state.epoch_to_start, state.num_train_epochs):
        state.outer_epoch_index = epoch
        if not (epoch == state.epoch_to_start and state.initial_step > 0):
            state.stage_batch_cursor = 0
        accelerator.print(f"\nepoch {epoch + 1}/{state.num_train_epochs}\n")
        state.current_epoch.value = epoch + 1
        state.metadata["ss_epoch"] = str(epoch + 1)

        accelerator.unwrap_model(state.network).on_epoch_start(
            state.text_encoder, state.unet
        )

        _run_epoch_steps(trainer, state, epoch)
        if state.stop_requested:
            break
        schedule_finished = (
            state.stage_plan is not None and state.global_step >= args.max_train_steps
        )
        _run_epoch_validation(trainer, state, epoch)
        _log_epoch_average(trainer, state, epoch)
        _run_adapter_epoch_hooks(trainer, state)

        # WandB weight snapshot at epoch boundary
        _wb = getattr(trainer, "_wandb_collectors", {})
        if _wb:
            weight_collector: WeightSnapshotCollector = _wb.get("weight")
            if weight_collector and weight_collector.should_collect(
                epoch_boundary=True
            ):
                _unwrapped = accelerator.unwrap_model(state.network)
                extras = weight_collector.collect(_unwrapped)
                dispatch_wandb_extras(accelerator, extras, state.global_step)

        accelerator.wait_for_everyone()

        state.optimizer_eval_fn()
        state.saver.maybe_save_epoch(
            state.network, state.global_step, epoch, state.num_train_epochs
        )
        state.saver.maybe_save_resumable(
            state.network, state.global_step, epoch, state.num_train_epochs
        )

        # Log checkpoint as wandb artifact when enabled
        _log_checkpoint_artifact(trainer, args, state, epoch)

        trainer.sample_images(
            accelerator,
            args,
            epoch + 1,
            state.global_step,
            accelerator.device,
            state.vae,
            state.tokenizers,
            state.text_encoder,
            state.unet,
            network=state.network,
        )
        state.optimizer_train_fn()

        if schedule_finished:
            break

    if not state.stop_requested:
        _audit_liveness(trainer, state, where="run end")

    state.metadata["ss_training_finished_at"] = str(time.time())


def _maybe_apply_stage_schedule(state: LoopState) -> None:
    """Switch dataset membership and rebuild the prepared DataLoader."""
    args = state.args
    plan = state.stage_plan
    if plan is None:
        return
    if plan.dataset_group is None or not plan.dataloader_kwargs:
        raise RuntimeError("stage_schedule lost its dataset or DataLoader factory")

    stages = plan.stages
    progress = progress_from_steps(state.global_step, int(args.max_train_steps or 1))
    next_index = resolve_stage_index(stages, progress)
    if next_index == state.stage_index:
        return

    active = {stages[next_index].subset_index}
    if not apply_active_subsets_to_dataset(plan.dataset_group, active):
        raise RuntimeError(
            "stage_schedule switch produced an empty dataset "
            f"(stage={next_index + 1}, subset_indices={sorted(active or [])}, "
            f"step={state.global_step}/{args.max_train_steps})"
        )

    raw_dataloader = torch.utils.data.DataLoader(
        plan.dataset_group,
        shuffle=True,
        **plan.dataloader_kwargs,
    )
    state.train_dataloader = state.accelerator.prepare_data_loader(raw_dataloader)
    if state.initial_step <= 0:
        state.stage_batch_cursor = 0
    state.stage_index = next_index
    state.metadata["ss_stage_index"] = str(next_index)
    stage = stages[next_index]
    log_stage_switch(stage, next_index, state.global_step, args.max_train_steps)
    state.accelerator.print(
        f"[stage] {next_index + 1}/{len(stages)} {stage.name or ''} "
        f"dataset={stage.subset_index} @ step "
        f"{state.global_step}/{args.max_train_steps}"
    )


def _run_epoch_steps(trainer, state: LoopState, epoch: int) -> None:
    """Inner per-step loop: walk the dataloader, execute the accumulate
    scope, run sample / save / log / step-validation ticks.

    A stage boundary may occur inside an epoch. In that case the current loader
    is abandoned, membership is switched, and iteration continues with the new
    prepared loader without resetting optimizer or scheduler state.
    """
    args = state.args
    accelerator = state.accelerator
    generator = (
        state.stage_plan.loader_generator if state.stage_plan is not None else None
    )
    if (
        state.initial_step > 0
        and generator is not None
        and state.stage_loader_generator_state is not None
    ):
        generator.set_state(state.stage_loader_generator_state)

    skipped_dataloader = None
    resume_batch_offset = max(0, int(state.initial_step))
    if resume_batch_offset > 0:
        skipped_dataloader = accelerator.skip_first_batches(
            state.train_dataloader, resume_batch_offset - 1
        )
        state.initial_step = 1

    # ``step`` is the zero-based micro-batch index within the current epoch.
    # Keep it absolute when a resumed loader is used; otherwise a later
    # interrupted save would report offset=1/2 again and lose the explicit
    # cursor promised by train_state.json.
    step = resume_batch_offset - 2 if resume_batch_offset > 0 else -1
    while state.global_step < args.max_train_steps:
        _maybe_apply_stage_schedule(state)
        loader = skipped_dataloader or state.train_dataloader
        if skipped_dataloader is None and generator is not None:
            state.stage_loader_generator_state = generator.get_state()
        skipped_dataloader = None
        stage_at_start = state.stage_index

        for batch in loader:
            step += 1
            state.current_step.value = state.global_step
            state.micro_batch_offset = max(0, step + 1)
            if state.initial_step > 0:
                state.initial_step -= 1
                continue

            _profiler_step_begin(state)
            state.at_optimizer_boundary = False
            loss = _run_step(trainer, state, batch)
            if state.stage_plan is not None:
                state.stage_batch_cursor += 1
            _profiler_step_end(state)

            keys_scaled, mean_norm, maximum_norm, max_mean_logs = _maybe_scale_norm(
                state
            )

            if accelerator.sync_gradients:
                state.progress_bar.update(1)
                state.global_step += 1
                state.current_step.value = state.global_step
                state.at_optimizer_boundary = True
                if state.global_step == LIVENESS_EARLY_CHECK_STEP:
                    _audit_liveness(
                        trainer, state, where=f"step {state.global_step} early check"
                    )
                # Stop only after the optimizer has committed a complete
                # update.  Save the semantic cursor and Accelerate payload
                # before returning so a daemon stop can resume exactly here.
                controller = getattr(state, "stop_controller", None)
                if controller is not None and controller.requested:
                    state.stop_requested = True
                    state.optimizer_eval_fn()
                    state.saver.save_interrupt_state(
                        state.network, state.global_step, epoch + 1
                    )
                    return

                # Publish recovery state only after the complete optimizer
                # update/scheduler step in _run_step has committed.
                state.saver.maybe_save_rolling_state(state.global_step)
                _sample_at_step(trainer, state)
                state.saver.maybe_save_step(state.network, state.global_step, epoch)
                _log_checkpoint_artifact(trainer, args, state, epoch)
                state.optimizer_train_fn()

            _log_step(
                trainer,
                state,
                loss=loss,
                step=step,
                epoch=epoch,
                keys_scaled=keys_scaled,
                mean_norm=mean_norm,
                maximum_norm=maximum_norm,
                max_mean_logs=max_mean_logs,
            )

            _maybe_run_step_validation(trainer, state, epoch)

            if state.global_step >= args.max_train_steps:
                return

            if stage_at_start >= 0 and state.stage_plan is not None:
                stages = state.stage_plan.stages
                progress = progress_from_steps(
                    state.global_step, int(args.max_train_steps or 1)
                )
                if resolve_stage_index(stages, progress) != state.stage_index:
                    break
        else:
            return


def _should_check_loss_finite(state: LoopState) -> bool:
    """Return True on the same cadence that would sync loss to host for logging."""
    if not state.accelerator.sync_gradients:
        return False
    log_every = max(1, int(getattr(state.args, "log_every_n_steps", 1) or 1))
    next_step = state.global_step + 1
    return (next_step % log_every == 0) or (next_step >= state.args.max_train_steps)


def _debug_finite_enabled(args) -> bool:
    value = getattr(args, "debug_finite_checks", False)
    env = os.environ.get("ANIMA_DEBUG_FINITE")
    if env is not None:
        return env.lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _check_loss_finite(
    loss: torch.Tensor, *, mixed_precision: Optional[str] = None
) -> None:
    """Fail fast before backward when the scalar loss is non-finite.

    This intentionally syncs the scalar predicate to Python, so callers should
    run it only on log cadence rather than every micro-step.
    """
    if torch.isfinite(loss.detach()).all():
        return
    hint = ""
    if mixed_precision == "fp16":
        hint = (
            " fp16 autocast is active; inspect batch/σ, residual-range guards, "
            "and the fp16-safe FinalLayer projection path."
        )
    raise FloatingPointError(
        "non-finite training loss before backward; aborting to avoid logging "
        "NaN averages or saving an invalid checkpoint." + hint
    )


def _check_trainable_grads_finite(network) -> None:
    bad: list[str] = []
    for name, param in network.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if not torch.isfinite(grad.detach()).all():
            bad.append(f"{name}: dtype={grad.dtype} shape={tuple(grad.shape)}")
            if len(bad) >= 8:
                break
    if bad:
        raise FloatingPointError(
            "non-finite trainable gradients after backward: " + "; ".join(bad)
        )


def _run_step(trainer, state: LoopState, batch) -> torch.Tensor:
    """The accumulate-scope body: on_step_start hooks, cudagraph mark, forward,
    backward gating, sync_gradients hooks (hydra warmup, grad capture, clip),
    optimizer step + zero_grad. Returns the loss (detached or live)."""
    args = state.args
    accelerator = state.accelerator
    network = state.network

    with accelerator.accumulate(state.training_model):
        state.on_step_start_for_network(state.text_encoder, state.unet)

        trainer.on_step_start(state.train_ctx, batch, is_train=True)

        # Clear last-step gate/σ refs + memoized router-stats caches before the
        # next forward. Unconditional: cudagraph needs it (lingering refs into
        # the cudagraph pool block reclamation → demotes to eager), and eager
        # needs it so per-step memoized stats invalidate instead of freezing at
        # their first values.
        net_unwrapped = accelerator.unwrap_model(network)
        if hasattr(net_unwrapped, "clear_step_caches"):
            net_unwrapped.clear_step_caches()

        # CUDAGraphs need an explicit iteration boundary before the forward
        # every step; without it the "pending, uninvoked backwards" fast-path
        # check fails and cudagraphs silently fall back to eager.
        if trainer._cudagraph_mark_step:
            torch.compiler.cudagraph_mark_step_begin()

        if state.profile_started:
            torch.cuda.nvtx.range_push("forward")
        loss = trainer.process_batch(state.train_ctx, batch, is_train=True)
        if _should_check_loss_finite(state) or _debug_finite_enabled(args):
            _check_loss_finite(
                loss,
                mixed_precision=getattr(state.accelerator, "mixed_precision", None),
            )
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

        if state.profile_started:
            torch.cuda.nvtx.range_push("backward")
        accelerator.backward(loss)
        # Block-swap backward hooks enqueue device transfers.  Do not let an
        # optimizer step (or a cooperative stop checkpoint) observe stale
        # parameters; wait here so worker exceptions reach the training thread.
        unet_for_swap = state.unet
        try:
            unet_for_swap = accelerator.unwrap_model(unet_for_swap)
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("could not unwrap DiT for block-swap wait: %s", exc)
        wait_for_swap = getattr(unet_for_swap, "wait_for_block_swap", None)
        if callable(wait_for_swap):
            wait_for_swap()
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

        # Post-backward adapter hook (before clip/step) — injects extra grad
        # contributions that can't share the primary backward, e.g. soft-tokens
        # gradient-cached contrastive negatives under active block swapping.
        trainer.run_after_backward(state.train_ctx)
        if _debug_finite_enabled(args):
            _check_trainable_grads_finite(accelerator.unwrap_model(network))

        if accelerator.sync_gradients:
            net_unwrapped = accelerator.unwrap_model(network)
            # Snapshot Hydra up-weight grad norms before zero_grad wipes them
            # (metric ``hydra_up_grad`` reads it later). Pre-clip so magnitudes
            # aren't distorted by the global rescale. Log-cadence only;
            # global_step increments below so predict the post-increment value.
            _log_every = max(1, int(getattr(args, "log_every_n_steps", 1) or 1))
            _will_log_after = state.is_tracking and (
                ((state.global_step + 1) % _log_every == 0)
                or ((state.global_step + 1) >= args.max_train_steps)
            )
            if _will_log_after and hasattr(net_unwrapped, "capture_up_grad_stats"):
                net_unwrapped.capture_up_grad_stats()
            if args.max_grad_norm != 0.0:
                params_to_clip = accelerator.unwrap_model(
                    network
                ).get_trainable_params()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

        if state.profile_started:
            torch.cuda.nvtx.range_push("optimizer")
        state.optimizer.step()
        state.lr_scheduler.step()
        state.optimizer.zero_grad(set_to_none=True)
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

    return loss


def _profiler_step_begin(state: LoopState) -> None:
    if (
        state.profile_range
        and state.global_step == state.profile_range[0]
        and not state.profile_started
    ):
        state.accelerator.print(f"\n[profiler] starting at step {state.global_step}")
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        state.profile_started = True

    if state.profile_started:
        torch.cuda.nvtx.range_push(f"step={state.global_step}")


def _profiler_step_end(state: LoopState) -> None:
    if state.profile_started:
        torch.cuda.nvtx.range_pop()
    if state.profile_started and state.global_step >= state.profile_range[1]:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        state.accelerator.print(f"\n[profiler] stopped at step {state.global_step}")
        state.accelerator.print(
            "[profiler] open the .nsys-rep with the Nsight Systems GUI\n"
        )
        state.profile_started = False
        state.profile_range = None  # don't re-trigger
        # Hard-exit so the launcher exits and nsys finalizes the report.
        # sys.exit(0) hangs in interpreter shutdown (DataLoader workers +
        # NCCL/CUDA atexit handlers wait on futexes); the profile buffer is
        # already flushed by the preceding synchronize() + cuProfilerStop.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _maybe_scale_norm(state: LoopState):
    args = state.args
    if args.scale_weight_norms:
        keys_scaled, mean_norm, maximum_norm = state.accelerator.unwrap_model(
            state.network
        ).apply_max_norm_regularization(
            args.scale_weight_norms, state.accelerator.device
        )
        max_mean_logs = {
            "Keys Scaled": keys_scaled,
            "Average key norm": mean_norm,
        }
        return keys_scaled, mean_norm, maximum_norm, max_mean_logs
    return None, None, None, {}


def _sample_at_step(trainer, state: LoopState) -> None:
    state.optimizer_eval_fn()
    trainer.sample_images(
        state.accelerator,
        state.args,
        None,
        state.global_step,
        state.accelerator.device,
        state.vae,
        state.tokenizers,
        state.text_encoder,
        state.unet,
        network=state.network,
    )


def _log_step(
    trainer,
    state: LoopState,
    *,
    loss,
    step: int,
    epoch: int,
    keys_scaled,
    mean_norm,
    maximum_norm,
    max_mean_logs,
) -> float:
    args = state.args
    log_every = max(1, int(getattr(args, "log_every_n_steps", 1) or 1))
    # Gate on sync_gradients: with gradient_accumulation_steps > 1 this hook
    # fires per micro-batch but global_step advances only on sync; without the
    # gate log_every_n_steps gives the same answer for every micro-batch,
    # bursting N tracker writes then N silent ones.
    should_log_step = state.accelerator.sync_gradients and (
        (state.global_step % log_every == 0)
        or (state.global_step >= args.max_train_steps)
    )

    # Only trigger the D2H sync (loss.detach().item()) on log cadence.
    # Between log steps the tqdm postfix shows the last-cached loss value —
    # the WebUI polls at ~300ms intervals so log_every_n_steps granularity
    # is more than sufficient for a responsive UI.  This eliminates a
    # per-step GPU→CPU synchronisation point that stalls the pipeline.
    _unwrapped_net = state.accelerator.unwrap_model(state.network)
    # Track whether a *real* loss/lr has been observed yet. The caches below
    # default to 0.0, and ``state.progress_bar.update(1)`` (the refresh that
    # flushes the postfix to stderr) runs at the TOP of the sync_gradients
    # block — BEFORE this function — so the first logged step's postfix is
    # whatever ``set_postfix`` staged on the PREVIOUS step. On step 1 that
    # staged value is the uninitialized 0.0 default, which the WebUI's stdout
    # parser then reads as a real ``avr_loss=0, lr=0`` data point and plots as
    # a zero on the loss/LR curves. Suppressing the fields until a real value
    # exists keeps the zero off the chart instead of injecting a phantom one.
    _have_real = getattr(trainer, "_postfix_has_real", False)
    _cached_avr = getattr(trainer, "_last_postfix_avr", 0.0)
    _cached_lr = getattr(trainer, "_last_postfix_lr", 0.0)
    if should_log_step:
        current_loss = loss.detach().item()
        state.loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
        _cached_avr = state.loss_recorder.moving_average
        trainer._last_postfix_avr = _cached_avr
        lrs = state.lr_scheduler.get_last_lr()
        _cached_lr = lrs[0] if lrs else 0.0
        # Prodigy / D-Adaptation are adaptive: the user-set ``lr`` (e.g. 1.0)
        # is only a multiplier on the optimizer's internal distance estimate
        # ``d``; the *effective* learning rate actually applied to params is
        # ``d * lr``. ``d`` starts tiny and grows each step, so reporting the
        # base ``lr`` alone makes the dashboard show a flat 1.0 while the real
        # lr rises from ~1e-6. Report the effective value here so the tqdm
        # postfix (and the WebUI's stdout parser) tracks the real lr. The base
        # lr is still recorded under ``lr/base`` by ``generate_step_logs``.
        _cached_lr = effective_lr(
            getattr(args, "optimizer_type", None), state.optimizer, _cached_lr
        )
        trainer._last_postfix_lr = _cached_lr
        trainer._postfix_has_real = True
        _have_real = True

    # Lightweight tqdm postfix update every step using cached values. Until the
    # first real loss/lr lands, omit the fields entirely (see _have_real note
    # above) so tqdm never renders a phantom ``avr_loss=0, lr=0``.
    logs: dict = {}
    if _have_real:
        logs["avr_loss"] = _cached_avr
        logs["lr"] = _cached_lr
    # Refresh router_H only on log cadence — get_router_entropy does a full
    # get_router_stats compute (D2H syncs) wasted on the progress-bar postfix;
    # tqdm shows a harmlessly-stale cached value between log steps.
    if getattr(_unwrapped_net, "_use_hydra", False) and should_log_step:
        _router_H = _unwrapped_net.get_router_entropy()
        if _router_H is not None:
            trainer._last_router_H_postfix = _router_H
    _router_H_cached = getattr(trainer, "_last_router_H_postfix", None)
    if _router_H_cached is not None:
        logs["router_H"] = f"{_router_H_cached:.3f}"
    state.progress_bar.set_postfix(refresh=False, **{**max_mean_logs, **logs})

    # The Phase-0 progress sink (GUI / daemon progress bar tails progress.jsonl).
    # Only write on log cadence to avoid per-step synchronous disk I/O that
    # stalls the training loop — the WebUI polls at ~300ms intervals so a
    # log_every_n_steps cadence (typically every 2 steps) is more than fast
    # enough for a responsive UI. When tracking is active the step_logging
    # call below already feeds the sink via dispatch_logs.
    # NOTE: this branch only fires on log cadence (should_log_step gated), by
    # which point the caches hold real values — but guard with _have_real
    # anyway so a future code path can't re-introduce a phantom-zero step event.
    progress_sink = getattr(trainer, "progress_sink", None)
    if (
        should_log_step
        and _have_real
        and not state.is_tracking
        and progress_sink is not None
    ):
        progress_sink.log(logs, global_step=state.global_step, epoch=epoch + 1)

    if state.is_tracking and should_log_step:
        logs = trainer.generate_step_logs(
            args,
            current_loss,
            _cached_avr,
            state.lr_scheduler,
            state.lr_descriptions,
            state.optimizer,
            keys_scaled,
            mean_norm,
            maximum_norm,
            None,  # mean_grad_norm — not tracked here
            None,  # mean_combined_norm — not tracked here
        )
        producers = [_unwrapped_net, *trainer._adapters]
        # Ledger is a MetricProducer — liveness/<name> coverage is the live
        # view of the run-end LIVENESS audit.
        _ledger = getattr(trainer, "_liveness", None)
        if _ledger is not None:
            producers.append(_ledger)
        logs.update(
            collect_metrics(
                producers,
                MetricContext(args=args, network=_unwrapped_net),
            )
        )
        trainer.step_logging(state.accelerator, logs, state.global_step, epoch + 1)

        # WandB enhanced metrics — system stats go into the regular log dict;
        # gradient histograms are wandb-only objects dispatched separately.
        _wb = getattr(trainer, "_wandb_collectors", {})
        if _wb:
            sys_collector: SystemMetricsCollector = _wb.get("sys")
            if sys_collector:
                logs.update(sys_collector.collect())
                # Re-dispatch with system metrics included (wandb only gets
                # the extra keys; tensorboard already received the base set).
                dispatch_wandb_extras(
                    state.accelerator, sys_collector.collect(), state.global_step
                )
            grad_collector: GradientHistogramCollector = _wb.get("grad")
            if grad_collector and grad_collector.should_collect(state.global_step):
                extras = grad_collector.collect(_unwrapped_net)
                dispatch_wandb_extras(state.accelerator, extras, state.global_step)

    return _cached_avr


def _maybe_run_step_validation(trainer, state: LoopState, epoch: int) -> None:
    args = state.args
    should_validate_step = (
        args.validate_every_n_steps is not None
        and state.global_step % args.validate_every_n_steps == 0
    )
    if (
        state.accelerator.sync_gradients
        and state.validation_steps > 0
        and should_validate_step
    ):
        run_validation(
            trainer,
            state.train_ctx,
            state.val_ctx,
            val_loss_recorder=state.val_step_loss_recorder,
            epoch=epoch,
            global_step=state.global_step,
            progress_bar=state.progress_bar,
            progress_desc="validation steps",
            postfix_label="val_avg_loss",
            log_avg_key="loss/validation/step_average",
            log_div_key="loss/validation/step_divergence",
            logging_fn=trainer.step_logging,
        )


def _run_epoch_validation(trainer, state: LoopState, epoch: int) -> None:
    args = state.args
    should_validate_epoch = (
        (epoch + 1) % args.validate_every_n_epochs == 0
        if args.validate_every_n_epochs is not None
        else True
    )
    if should_validate_epoch and len(state.val_ctx.dataloader) > 0:
        run_validation(
            trainer,
            state.train_ctx,
            state.val_ctx,
            val_loss_recorder=state.val_epoch_loss_recorder,
            epoch=epoch,
            global_step=state.global_step,
            progress_bar=state.progress_bar,
            progress_desc="epoch validation steps",
            postfix_label="val_epoch_avg_loss",
            log_avg_key="loss/validation/epoch_average",
            log_div_key="loss/validation/epoch_divergence",
            logging_fn=trainer.epoch_logging,
        )


def _log_epoch_average(trainer, state: LoopState, epoch: int) -> None:
    if not state.is_tracking:
        return
    logs = {"loss/epoch_average": state.loss_recorder.moving_average}
    trainer.epoch_logging(state.accelerator, logs, state.global_step, epoch + 1)


def _audit_liveness(trainer, state: LoopState, *, where: str) -> None:
    """Liveness audit (issues.md P1.1): a configured-ON aux loss that never
    consumed its aux input is a silent baseline — flag it loudly.

    Reads the trainer-owned ``LivenessLedger`` that the per-step composer
    feeds (``train.py`` threads it through ``build_loss_composer``). Dead
    features ERROR-log with the greppable ``LIVENESS:`` prefix (main process
    only — counts are per-rank but a dead dispatch is dead on every rank);
    ``--liveness_strict`` escalates to a hard abort, evaluated on each rank's
    own ledger so distributed runs fail together instead of hanging.
    """
    ledger = getattr(trainer, "_liveness", None)
    if ledger is None:
        return
    if state.accelerator.is_main_process:
        dead = ledger.audit(where=where)
    else:
        dead = ledger.dead_features()
    if dead and bool(getattr(state.args, "liveness_strict", False)):
        raise RuntimeError(
            f"LIVENESS: configured-but-dead feature(s) at {where}: "
            f"{', '.join(dead)} — aborting (--liveness_strict)"
        )


def _run_adapter_epoch_hooks(trainer, state: LoopState) -> None:
    """Per-method end-of-epoch hooks (IP-Adapter diagnostic dump, …).
    Main process only — adapters that need cross-rank reduction should do
    that internally."""
    if not (trainer._adapters and state.accelerator.is_main_process):
        return
    epoch_end_ctx = StepCtx(
        args=state.args,
        accelerator=state.accelerator,
        network=state.network,
        weight_dtype=state.weight_dtype,
    )
    for adapter in trainer._adapters:
        adapter.on_epoch_end(epoch_end_ctx)
