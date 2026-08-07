from __future__ import annotations

import pytest

from library.training.resume import (
    resolve_persisted_resume_position,
    resolve_resume_position,
)


def test_resume_at_reported_epoch_boundary():
    assert resolve_resume_position(950, 760, 4) == (5, 0)


def test_resume_mid_epoch_translates_optimizer_steps_to_batches():
    assert resolve_resume_position(953, 760, 4) == (5, 12)


def test_resume_handles_partial_accumulation_epoch_boundary():
    # Ten batches produce three optimizer steps with accumulation=4.  Step 3
    # is the next epoch boundary, not batch 12 of epoch zero.
    assert resolve_resume_position(3, 10, 4) == (1, 0)
    assert resolve_resume_position(4, 10, 4) == (1, 4)


def test_persisted_cursor_normalizes_exact_epoch_end():
    # current_epoch is one-based and the saved offset counts consumed
    # micro-batches.  Ten consumed batches are the end of epoch zero, so the
    # next launch starts at epoch one, batch zero.
    assert resolve_persisted_resume_position(
        3, 10, 4, current_epoch=1, micro_batch_offset=10
    ) == (1, 0)


def test_persisted_cursor_is_used_for_mid_epoch_resume():
    assert resolve_persisted_resume_position(
        4, 10, 4, current_epoch=2, micro_batch_offset=4
    ) == (1, 4)


def test_persisted_cursor_rejects_step_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        resolve_persisted_resume_position(
            4, 10, 4, current_epoch=1, micro_batch_offset=4
        )


@pytest.mark.parametrize(
    ("global_step", "batches_per_epoch", "gradient_accumulation_steps"),
    [(-1, 10, 1), (0, 0, 1), (0, 10, 0)],
)
def test_resume_rejects_invalid_inputs(
    global_step: int,
    batches_per_epoch: int,
    gradient_accumulation_steps: int,
):
    with pytest.raises(ValueError):
        resolve_resume_position(
            global_step, batches_per_epoch, gradient_accumulation_steps
        )
