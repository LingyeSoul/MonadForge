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
from library.training.method_adapter import StepCtx
from library.training.metrics import MetricContext, collect_metrics
from library.training.validation import run_validation
from library.training.wandb_metrics import (
    GradientHistogramCollector,
    SystemMetricsCollector,
    WeightSnapshotCollector,
)
from library.training.log_dispatch import dispatch_wandb_extras

logger = logging.getLogger(__name__)


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

    global_step: int = 0
    profile_started: bool = False


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
        for t_enc in text_encoders:
            del t_enc
        text_encoders = []
        text_encoder = None
        gc.collect()
        clean_memory_on_device(accelerator.device)

    # --sample_at_first
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

    # Skip prelude: when resuming with skip_until_initial_step, fast-forward
    # the global_step counter before tqdm so the bar total is sized correctly,
    # and consume per-epoch skip credit so dataloader.skip_first_batches has
    # the right offset on the first epoch only. Runs before the dtype log so
    # the log order matches the original train() body.
    global_step = 0
    if initial_step > 0:
        global_step = initial_step // args.gradient_accumulation_steps
        for skip_epoch in range(epoch_to_start):
            logger.info(
                f"skipping epoch {skip_epoch + 1} because initial_step "
                f"(multiplied) is {initial_step}"
            )
            initial_step -= len(train_dataloader)

    logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
    for i, t_enc in enumerate(text_encoders):
        params_itr = t_enc.parameters()
        params_itr.__next__()  # skip the first parameter
        params_itr.__next__()  # skip the second parameter (CLIP first two are embeddings)
        param_3rd = params_itr.__next__()
        logger.info(
            f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}"
        )

    clean_memory_on_device(accelerator.device)

    progress_bar = tqdm(
        range(args.max_train_steps - global_step),
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )

    validation_steps = (
        min(args.max_validation_steps, len(val_dataloader))
        if args.max_validation_steps is not None
        else len(val_dataloader)
    )
    # Validate at fixed sigma values across the schedule:
    # 0.1 = near-clean / fine detail, 0.4 = mid / bulk structure,
    # 0.7 = high noise / coarse denoising (early inference steps).
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

    for epoch in range(state.epoch_to_start, state.num_train_epochs):
        accelerator.print(f"\nepoch {epoch + 1}/{state.num_train_epochs}\n")
        state.current_epoch.value = epoch + 1
        state.metadata["ss_epoch"] = str(epoch + 1)

        # network.train() is invoked here
        accelerator.unwrap_model(state.network).on_epoch_start(
            state.text_encoder, state.unet
        )

        _run_epoch_steps(trainer, state, epoch)
        _run_epoch_validation(trainer, state, epoch)
        _log_epoch_average(trainer, state, epoch)
        _run_adapter_epoch_hooks(trainer, state)

        # WandB weight snapshot at epoch boundary
        _wb = getattr(trainer, "_wandb_collectors", {})
        if _wb:
            weight_collector: WeightSnapshotCollector = _wb.get("weight")
            if weight_collector and weight_collector.should_collect(epoch_boundary=True):
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
        )
        state.optimizer_train_fn()

    state.metadata["ss_training_finished_at"] = str(time.time())


def _run_epoch_steps(trainer, state: LoopState, epoch: int) -> None:
    """Inner per-step loop: walk the dataloader, execute the accumulate
    scope, run sample / save / log / step-validation ticks."""
    args = state.args
    accelerator = state.accelerator

    skipped_dataloader = None
    if state.initial_step > 0:
        skipped_dataloader = accelerator.skip_first_batches(
            state.train_dataloader, state.initial_step - 1
        )
        state.initial_step = 1

    for step, batch in enumerate(skipped_dataloader or state.train_dataloader):
        state.current_step.value = state.global_step
        if state.initial_step > 0:
            state.initial_step -= 1
            continue

        _profiler_step_begin(state)

        loss = _run_step(trainer, state, batch)

        _profiler_step_end(state)

        keys_scaled, mean_norm, maximum_norm, max_mean_logs = _maybe_scale_norm(state)

        if accelerator.sync_gradients:
            state.progress_bar.update(1)
            state.global_step += 1
            _sample_at_step(trainer, state)
            state.saver.maybe_save_step(state.network, state.global_step, epoch)
            _log_checkpoint_artifact(trainer, args, state, epoch)
            state.optimizer_train_fn()

        avr_loss = _log_step(
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
            break


def _run_step(trainer, state: LoopState, batch) -> torch.Tensor:
    """The accumulate-scope body: on_step_start hooks, cudagraph mark, forward,
    backward gating, sync_gradients hooks (hydra warmup, grad capture, clip),
    optimizer step + zero_grad. Returns the loss (detached or live)."""
    args = state.args
    accelerator = state.accelerator
    network = state.network

    with accelerator.accumulate(state.training_model):
        state.on_step_start_for_network(state.text_encoder, state.unet)

        # preprocess batch for each model
        trainer.on_step_start(state.train_ctx, batch, is_train=True)

        # Clear last-step gate/σ tensor refs + memoized router-stats caches
        # before the next forward. Called unconditionally — the cudagraph
        # branch below also needs it (lingering refs into the cudagraph
        # memory pool block pool reclamation, demoting the run to eager),
        # and the eager path needs it so per-step memoized stats
        # (``_router_stats_cache`` / ``_chimera_router_stats_cache``) get
        # invalidated each step instead of freezing at their first computed
        # values. Cost is ~60 Python attr writes; stats compute itself is
        # already log-step-gated by callers.
        net_unwrapped = accelerator.unwrap_model(network)
        if hasattr(net_unwrapped, "clear_step_caches"):
            net_unwrapped.clear_step_caches()

        # CUDAGraphs (reduce-overhead / max-autotune) also need an explicit
        # iteration boundary for inductor's cudagraph_trees. Without this
        # call, the "pending, uninvoked backwards" fast-path check fails
        # every step and cudagraphs silently fall back to the eager path —
        # you pay compile latency and keep launch overhead. Must be called
        # before the forward on every step.
        if trainer._cudagraph_mark_step:
            torch.compiler.cudagraph_mark_step_begin()

        if state.profile_started:
            torch.cuda.nvtx.range_push("forward")
        loss = trainer.process_batch(state.train_ctx, batch, is_train=True)
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

        if state.profile_started:
            torch.cuda.nvtx.range_push("backward")
        accelerator.backward(loss)
        if state.profile_started:
            torch.cuda.nvtx.range_pop()

        # Post-backward adapter hook (before clip/step) — injects extra grad
        # contributions that can't share the primary backward, e.g. soft-tokens
        # gradient-cached contrastive negatives under active block swapping.
        trainer.run_after_backward(state.train_ctx)

        if accelerator.sync_gradients:
            net_unwrapped = accelerator.unwrap_model(network)
            # Snapshot Hydra up-weight grad norms before zero_grad wipes them.
            # The metric ``hydra_up_grad`` reads this stash later in the step.
            # Also runs pre-clip so absolute magnitudes aren't distorted by
            # the global rescale (clipping preserves the below/above ratio).
            # Skip on non-log steps — the metric only fires at log cadence,
            # so capturing every step burns kernels whose output is never
            # read. global_step increments below, so predict the
            # post-increment value.
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
        torch.cuda.nvtx.range_pop()  # close per-step NVTX range
    if state.profile_started and state.global_step >= state.profile_range[1]:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        state.accelerator.print(f"\n[profiler] stopped at step {state.global_step}")
        state.accelerator.print(
            "[profiler] open the .nsys-rep with the Nsight Systems GUI\n"
        )
        state.profile_started = False
        state.profile_range = None  # don't re-trigger
        # Hard-exit so the accelerate launcher exits and nsys finalizes the
        # report. sys.exit(0) hangs in interpreter shutdown here (DataLoader
        # workers + NCCL/CUDA atexit handlers all wait on futexes). os._exit
        # skips that cleanup; the profile buffer is already flushed by the
        # preceding cuda.synchronize() + cuProfilerStop, and the .nsys-rep is
        # owned by nsys, not this process, so the hard exit is safe.
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
    # Gate on sync_gradients: with gradient_accumulation_steps > 1 the
    # dataloader fires this hook once per micro-batch but global_step only
    # advances on sync. Without the gate, log_every_n_steps decides the same
    # answer for every micro-batch in an accumulation cycle, producing bursts
    # of N back-to-back tracker writes followed by N silent ones (exactly the
    # "sometimes honored, sometimes ignored" pattern).
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
    _cached_avr = getattr(trainer, "_last_postfix_avr", 0.0)
    _cached_lr = getattr(trainer, "_last_postfix_lr", 0.0)
    if should_log_step:
        current_loss = loss.detach().item()
        state.loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
        _cached_avr = state.loss_recorder.moving_average
        trainer._last_postfix_avr = _cached_avr
        lrs = state.lr_scheduler.get_last_lr()
        _cached_lr = lrs[0] if lrs else 0.0
        trainer._last_postfix_lr = _cached_lr

    # Lightweight tqdm postfix update every step using cached values.
    logs = {"avr_loss": _cached_avr, "lr": _cached_lr}
    # Refresh router_H only on log cadence — get_router_entropy → full
    # get_router_stats compute (with D2H syncs) is wasted if the only
    # consumer is the progress-bar postfix. Cache last value on trainer
    # so tqdm shows a stale value harmlessly between log steps.
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
    progress_sink = getattr(trainer, "progress_sink", None)
    if should_log_step and not state.is_tracking and progress_sink is not None:
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

    # Write a lightweight step event to the JSONL progress sink on
    # log cadence (matching tensorboard write frequency) so the WebUI
    # can tail structured metrics.  Writing every sync_gradients step
    # adds synchronous disk I/O that stalls the GPU pipeline — the
    # WebUI polls at ~300ms intervals so log_every_n_steps cadence is
    # sufficient for a responsive UI.
    if should_log_step:
        _sink = getattr(trainer, "progress_sink", None)
        if _sink is not None:
            _sink.log(
                {"avr_loss": _cached_avr, "lr": logs.get("lr", 0.0)},
                global_step=state.global_step,
                epoch=epoch,
            )

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
