"""Checkpoint cursors preserve the next stage and the shuffled sample stream."""

from types import SimpleNamespace as NS

import pytest
import torch
from accelerate import Accelerator

from library.training import loop
from library.training.resume import normalize_stage_resume_state
from library.training.stage_schedule import StageRuntimePlan, parse_stage_specs
from library.training.staged_resolution import build_stage_schedule


def _plan():
    return StageRuntimePlan(
        stages=tuple(parse_stage_specs(build_stage_schedule([100 / 3] * 3))),
        dataset_group=object(),
        target_batch_counts=(10, 10, 10),
        full_num_train_images=30,
        full_num_reg_images=0,
        full_dataset_counts=(),
        loader_generator=torch.Generator().manual_seed(17),
    )


def test_boundary_snapshot_owns_next_stage_and_resets_cursor():
    plan = _plan()
    old_generator = plan.loader_generator.get_state()
    torch.randperm(10, generator=plan.loader_generator)
    owner = NS(
        args=NS(max_train_steps=3),
        global_step=1,
        current_epoch=NS(value=1),
        micro_batch_offset=1,
        stage_plan=plan,
        stage_index=0,
        stage_batch_cursor=1,
        outer_epoch_index=0,
        stage_loader_generator_state=old_generator,
    )
    saved = loop.LoopState.checkpoint_runtime_state(owner)
    assert saved["stage_index"] == 1
    assert saved["stage_batch_cursor"] == 0
    assert (
        saved["stage_loader_generator_state"]
        == plan.loader_generator.get_state().tolist()
    )
    assert normalize_stage_resume_state(plan, 3, saved) == saved


def test_legacy_boundary_resets_only_the_completed_stage_cursor():
    plan = _plan()
    old = {
        "global_step": 1,
        "stage_index": 0,
        "stage_batch_cursor": 9,
        "stage_loader_generator_state": [1, 2],
        "stage_outer_epoch": 3,
    }
    normalized = normalize_stage_resume_state(plan, 3, old)
    assert normalized["stage_index"] == 1 and normalized["stage_batch_cursor"] == 0
    assert normalized["stage_outer_epoch"] == 3
    assert "stage_loader_generator_state" not in normalized
    assert old["stage_batch_cursor"] == 9
    with pytest.raises(ValueError, match="resume stage"):
        normalize_stage_resume_state(plan, 30, {"global_step": 15, "stage_index": 0})
    with pytest.raises(ValueError, match="resume stage"):
        normalize_stage_resume_state(
            plan, 30, {"global_step": 5, "stage_index": 0}, global_step=10
        )


@pytest.mark.parametrize("stop_after", [3, 10])
def test_interrupted_loader_matches_remaining_samples_and_next_epoch(
    monkeypatch, stop_after
):
    accelerator = Accelerator(cpu=True)
    for name in (
        "_profiler_step_begin",
        "_profiler_step_end",
        "_sample_at_step",
        "_log_checkpoint_artifact",
        "_log_step",
        "_maybe_run_step_validation",
        "_audit_liveness",
    ):
        monkeypatch.setattr(loop, name, lambda *a, **kw: None)
    monkeypatch.setattr(loop, "_maybe_scale_norm", lambda *a: (None, None, None, {}))

    def run(generator, *, saved=None, stop=None, epoch=0):
        loader = accelerator.prepare_data_loader(
            torch.utils.data.DataLoader(
                list(range(10)), batch_size=1, shuffle=True, generator=generator
            )
        )
        seen, checkpoints = [], []
        offset = saved["micro_batch_offset"] if saved and epoch == 0 else 0
        previous = (
            torch.tensor(saved["dataloader_generator_state"], dtype=torch.uint8)
            if saved
            else None
        )
        if previous is not None:
            generator.set_state(previous)
        state = NS(
            args=NS(max_train_steps=20),
            accelerator=accelerator,
            initial_step=offset,
            train_dataloader=loader,
            stage_plan=None,
            stage_index=-1,
            dataloader_generator=generator,
            dataloader_generator_state=previous,
            global_step=saved["global_step"] if saved else 0,
            current_epoch=NS(value=epoch + 1),
            current_step=NS(value=0),
            outer_epoch_index=epoch,
            micro_batch_offset=0,
            network=object(),
            stop_controller=NS(requested=False),
            pause_watcher=None,
            optimizer_eval_fn=lambda: None,
            optimizer_train_fn=lambda: None,
            progress_bar=NS(update=lambda _: None),
        )

        def step(*args):
            seen.append(args[-1].item())
            if stop is not None and len(seen) == stop:
                state.stop_controller.requested = True
            return torch.tensor(1.0)

        monkeypatch.setattr(loop, "_run_step", step)
        state.saver = NS(
            save_interrupt_state=lambda *a: checkpoints.append(
                loop.LoopState.checkpoint_runtime_state(state)
            ),
            maybe_save_step=lambda *a: None,
            maybe_save_rolling_state=lambda *a: None,
        )
        loop._run_epoch_steps(None, state, epoch)
        return seen, checkpoints

    baseline_generator = torch.Generator().manual_seed(7)
    baseline, _ = run(baseline_generator)
    baseline_next, _ = run(baseline_generator, epoch=1)
    first, [saved] = run(torch.Generator().manual_seed(7), stop=stop_after)
    assert first == baseline[:stop_after]
    next_generator = torch.Generator().manual_seed(7)
    if stop_after < 10:
        remaining, _ = run(next_generator, saved=saved)
        assert first + remaining == baseline
        next_epoch, _ = run(next_generator, epoch=1)
    else:
        assert saved["dataloader_generator_epoch"] == 1
        next_epoch, _ = run(next_generator, saved=saved, epoch=1)
    assert next_epoch == baseline_next
