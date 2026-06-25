"""Log-dispatch: fan an assembled ``logs`` dict out to its sinks.

Sends a ``logs`` dict to every configured Accelerate tracker (tensorboard /
wandb / others) plus the Phase-0 :class:`~library.training.progress.ProgressSink`.
This is the *output* end of the metrics pipeline — the values are produced via
the :mod:`library.training.metrics` collector protocol, assembled into a dict,
then handed here. Distinct from :mod:`library.log`, which configures Python's
stdlib console logging.

``AnimaTrainer`` keeps thin ``step_logging`` / ``epoch_logging`` /
``val_logging`` wrappers that call :func:`dispatch_logs` with the trainer's
``progress_sink``.
"""

from __future__ import annotations

from typing import Optional


def is_d_adaptation_optimizer(optimizer_type: Optional[str]) -> bool:
    """True for adaptive optimizers where the user-set ``lr`` is only a multiplier.

    Prodigy (``prodigyopt.Prodigy``) and D-Adaptation (``dadaptation.*``) grow
    an internal distance estimate ``d`` each step and apply ``d * lr`` to the
    parameters. The base ``lr`` is therefore not the effective learning rate —
    reporting it directly (e.g. ``lr=1.0``) makes the dashboard show a flat
    line while the real lr rises from ~1e-6. Case-insensitive.
    """
    if not optimizer_type:
        return False
    ot = optimizer_type.lower()
    return ot.startswith("dadapt") or ot.startswith("prodigy")


def effective_lr(optimizer_type: Optional[str], optimizer, base_lr: float) -> float:
    """Return the learning rate actually applied to parameters.

    For Prodigy / D-Adaptation the effective lr is ``d * lr`` (``d`` is the
    optimizer's per-group distance estimate, updated after each step). For
    every other optimizer the base ``lr`` is the effective lr. Falls back to
    ``base_lr`` if the optimizer exposes no ``d`` field (e.g. before the first
    step) so the postfix never shows ``0.0`` prematurely.
    """
    if not is_d_adaptation_optimizer(optimizer_type):
        return base_lr
    try:
        group = optimizer.param_groups[0]
    except (AttributeError, IndexError):
        return base_lr
    d = group.get("d")
    if d is None:
        return base_lr
    return float(d) * base_lr


def generate_step_logs(
    args,
    current_loss,
    avr_loss,
    lr_scheduler,
    lr_descriptions,
    optimizer=None,
    keys_scaled=None,
    mean_norm=None,
    maximum_norm=None,
    mean_grad_norm=None,
    mean_combined_norm=None,
    *,
    vr_state: Optional[dict] = None,
) -> dict:
    """Assemble the per-step ``logs`` dict (loss, norms, per-group LRs).

    The *input* end of the metrics pipeline — the dict built here is handed to
    :func:`dispatch_logs`. ``vr_state`` is the trainer's ``RuntimeState.vr``
    dict (λ tracking for the variance-reduced FM loss); pass ``None`` when VR
    is off.

    Note on history: the old in-trainer version carried a ``for…else`` whose
    ``else`` block (a legacy ``lr/group{i}`` naming fallback from before
    ``lr_descriptions`` existed) ran unconditionally — a for-loop ``else``
    fires whenever the loop completes without ``break`` — duplicating every LR
    series under a second key set. The loop below is the whole behavior.
    """
    logs = {"loss/current": current_loss, "loss/average": avr_loss}

    if keys_scaled is not None:
        logs["max_norm/keys_scaled"] = keys_scaled
        logs["max_norm/max_key_norm"] = maximum_norm
    if mean_norm is not None:
        logs["norm/avg_key_norm"] = mean_norm
    if mean_grad_norm is not None:
        logs["norm/avg_grad_norm"] = mean_grad_norm
    if mean_combined_norm is not None:
        logs["norm/avg_combined_norm"] = mean_combined_norm

    if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0 and vr_state:
        lambda_ema = vr_state.get("lambda_ema")
        lambda_batch = vr_state.get("lambda_batch")
        if isinstance(lambda_ema, float):
            logs["vr/lambda_ema"] = lambda_ema
        if isinstance(lambda_batch, float):
            logs["vr/lambda_batch"] = lambda_batch

    lrs = lr_scheduler.get_last_lr()
    for i, lr in enumerate(lrs):
        if lr_descriptions is not None:
            lr_desc = lr_descriptions[i]
        else:
            idx = i - (0 if args.network_train_unet_only else -1)
            if idx == -1:
                lr_desc = "textencoder"
            else:
                if len(lrs) > 2:
                    lr_desc = f"group{idx}"
                else:
                    lr_desc = "unet"

        is_d_adapt = is_d_adaptation_optimizer(args.optimizer_type)
        # Prodigy / D-Adaptation apply ``d * lr`` to the params, not the base
        # ``lr`` (``d`` is the optimizer's growing distance estimate). The
        # dashboard reads the plain ``lr/<desc>`` key, so emit the *effective*
        # value there; keep the raw base lr under ``lr/base/<desc>`` for anyone
        # who wants the multiplier, and leave ``lr/d*lr/<desc>`` in place for
        # backward compatibility with existing tensorboard curves.
        if is_d_adapt:
            group = lr_scheduler.optimizers[-1].param_groups[i]
            d = group.get("d")
            if d is not None:
                d_lr = float(d) * float(group["lr"])
                logs[f"lr/{lr_desc}"] = d_lr
                logs[f"lr/base/{lr_desc}"] = lr
                logs[f"lr/d*lr/{lr_desc}"] = d_lr
            else:
                logs[f"lr/{lr_desc}"] = lr
        else:
            logs[f"lr/{lr_desc}"] = lr

        if (
            args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower())
            and optimizer is not None
        ):  # tracking d*lr value of unet.
            d = optimizer.param_groups[0].get("d")
            if d is not None:
                logs["lr/d*lr"] = float(d) * float(
                    optimizer.param_groups[0]["lr"]
                )

    return logs


def dispatch_logs(
    accelerator,
    logs: dict,
    step_value: int,
    global_step: int,
    epoch: int,
    val_step: Optional[int] = None,
    *,
    progress_sink=None,
) -> None:
    """Send ``logs`` to all trackers and the progress sink.

    ``step_value`` is the x-axis for tensorboard; ``global_step`` / ``epoch`` /
    ``val_step`` are attached as fields for wandb. ``progress_sink`` (if given)
    receives the same dict on the main process only.
    """
    tensorboard_tracker = None
    wandb_tracker = None
    other_trackers = []
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tensorboard_tracker = accelerator.get_tracker("tensorboard")
        elif tracker.name == "wandb":
            wandb_tracker = accelerator.get_tracker("wandb")
        else:
            other_trackers.append(accelerator.get_tracker(tracker.name))

    if tensorboard_tracker is not None:
        tensorboard_tracker.log(logs, step=step_value)

    if wandb_tracker is not None:
        logs["global_step"] = global_step
        logs["epoch"] = epoch
        if val_step is not None:
            logs["val_step"] = val_step
        wandb_tracker.log(logs)

    for tracker in other_trackers:
        tracker.log(logs, step=step_value)

    if progress_sink is not None and accelerator.is_main_process:
        progress_sink.log(
            logs, global_step=global_step, epoch=epoch, val_step=val_step
        )


def dispatch_wandb_extras(accelerator, extras: dict, step_value: int) -> None:
    """Send wandb-only objects (Histogram, Image, Table) to the wandb tracker.

    ``extras`` may contain ``wandb.Histogram`` / ``wandb.Image`` / etc.
    that TensorBoard cannot consume, so they are routed exclusively to wandb.
    """
    if not extras:
        return
    wandb_tracker = None
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            wandb_tracker = accelerator.get_tracker("wandb")
            break
    if wandb_tracker is None:
        return
    extras["global_step"] = step_value
    wandb_tracker.log(extras)
