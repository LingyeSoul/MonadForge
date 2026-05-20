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
