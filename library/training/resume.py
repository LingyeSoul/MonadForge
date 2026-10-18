"""Translate persisted optimizer steps into DataLoader resume positions."""

from __future__ import annotations

import math


def normalize_stage_resume_state(
    plan, max_train_steps: int, state: dict, *, global_step: int | None = None
) -> dict:
    """Accept old boundary snapshots without applying the old stage's cursor."""
    saved_step = int(state.get("global_step", state.get("current_step", 0)))
    step = saved_step if global_step is None else global_step
    expected = plan.index_for_step(step, max_train_steps)
    saved = state.get("stage_index")
    if saved is None or int(saved) == expected:
        return state
    previous = plan.index_for_step(max(0, step - 1), max_train_steps)
    if step == saved_step and step > 0 and int(saved) == previous and previous != expected:
        # Older snapshots persisted the completed stage at the transition.
        # Its loader cursor and pre-permutation RNG belong to that old stage.
        import logging

        logging.getLogger(__name__).warning(
            "Normalizing legacy stage-boundary checkpoint at step %s; "
            "starting stage %s with a fresh loader permutation",
            step,
            expected,
        )
        normalized = dict(state, stage_index=expected, stage_batch_cursor=0)
        normalized.pop("stage_loader_generator_state", None)
        return normalized
    raise ValueError(
        "resume stage does not match the active stage_schedule: "
        f"state={saved}, expected={expected}"
    )


def resolve_resume_position(
    global_step: int,
    batches_per_epoch: int,
    gradient_accumulation_steps: int,
) -> tuple[int, int]:
    """Return ``(start_epoch, batch_offset)`` for a saved optimizer step.

    ``train_state.json`` stores optimizer/global steps, while
    ``Accelerator.skip_first_batches`` consumes DataLoader micro-batches.  The
    final optimizer step of an epoch may contain fewer than
    ``gradient_accumulation_steps`` batches, so converting through total
    micro-batches would drift at every non-divisible epoch boundary.
    """
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if batches_per_epoch <= 0:
        raise ValueError("batches_per_epoch must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation_steps
    )
    start_epoch, optimizer_step_offset = divmod(global_step, optimizer_steps_per_epoch)
    batch_offset = optimizer_step_offset * gradient_accumulation_steps
    return start_epoch, batch_offset


def resolve_persisted_resume_position(
    global_step: int,
    batches_per_epoch: int,
    gradient_accumulation_steps: int,
    *,
    current_epoch: int,
    micro_batch_offset: int,
) -> tuple[int, int]:
    """Validate and return an explicit schema-v2 DataLoader cursor.

    ``current_epoch`` is persisted as the trainer's one-based display epoch;
    the returned epoch is zero-based for ``range(epoch_to_start, ...)``.
    Offsets at the exact end of an epoch normalize to the next epoch so epoch
    validation/checkpoint hooks are not replayed after a restart.

    Old state records have no explicit cursor and continue to use
    :func:`resolve_resume_position` instead.
    """

    # Reuse the legacy conversion's validation and step-per-epoch definition so
    # both state generations agree at partial accumulation epoch boundaries.
    derived = resolve_resume_position(
        global_step, batches_per_epoch, gradient_accumulation_steps
    )
    try:
        epoch = int(current_epoch)
        offset = int(micro_batch_offset)
    except (TypeError, ValueError) as exc:
        raise ValueError("persisted resume cursor must contain integers") from exc
    if epoch <= 0:
        raise ValueError("persisted current_epoch must be one-based and positive")
    if offset < 0:
        raise ValueError("persisted micro_batch_offset must be non-negative")

    start_epoch = epoch - 1
    start_epoch += offset // batches_per_epoch
    batch_offset = offset % batches_per_epoch

    # A cooperative state is saved only after sync_gradients.  Inside an epoch,
    # that means the consumed batch count must land on an accumulation boundary;
    # an epoch's short final window normalized to batch_offset=0 above.
    if batch_offset % gradient_accumulation_steps != 0:
        raise ValueError(
            "persisted micro_batch_offset is not an optimizer-step boundary"
        )

    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation_steps
    )
    cursor_step = (
        start_epoch * optimizer_steps_per_epoch
        + batch_offset // gradient_accumulation_steps
    )
    if cursor_step != global_step:
        raise ValueError(
            "persisted epoch/micro-batch cursor does not match global_step "
            f"({cursor_step} != {global_step}; derived={derived})"
        )
    return start_epoch, batch_offset
