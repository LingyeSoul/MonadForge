"""Focused tests for the daemon cooperative-stop hard-exit boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_hard_exit_after_cooperative_stop_is_daemon_only(monkeypatch):
    import train

    exits: list[int] = []
    monkeypatch.setattr(train.os, "_exit", exits.append)

    # The daemon injects this environment marker for both Linux and Windows
    # jobs.  A completed cooperative stop must bypass interpreter teardown.
    monkeypatch.setenv("ANIMA_DAEMON_STOP_FILE", "/tmp/test-stop.requested")
    train._hard_exit_after_cooperative_stop(True)
    assert exits == [0]

    # A normal CLI run (or a non-stop return) keeps the ordinary cleanup path.
    monkeypatch.delenv("ANIMA_DAEMON_STOP_FILE")
    train._hard_exit_after_cooperative_stop(True)
    train._hard_exit_after_cooperative_stop(False)
    assert exits == [0]


def test_run_end_is_flushed_before_cooperative_hard_exit(tmp_path, monkeypatch):
    import json

    import train
    from library.training.progress import ProgressSink, run_scope

    progress = tmp_path / "progress.jsonl"
    sink = ProgressSink(
        str(progress), run="stop-order", method="lora", preset="default"
    )
    sink.run_start(total_steps=2, total_epochs=1, pid=123)
    with run_scope(sink, final_step=lambda: 1, stopped=lambda: True):
        pass

    exits: list[int] = []
    monkeypatch.setenv("ANIMA_DAEMON_STOP_FILE", str(tmp_path / "stop.requested"))
    monkeypatch.setattr(train.os, "_exit", exits.append)
    train._hard_exit_after_cooperative_stop(True)

    events = [json.loads(line) for line in progress.read_text().splitlines()]
    assert events[-1]["ev"] == "run_end"
    assert events[-1]["status"] == "stopped"
    assert exits == [0]


def test_cooperative_stop_save_failure_propagates(monkeypatch):
    import library.training.loop as loop_module

    monkeypatch.setattr(loop_module, "_run_step", lambda *_args: object())
    monkeypatch.setattr(loop_module, "_profiler_step_begin", lambda _state: None)
    monkeypatch.setattr(loop_module, "_profiler_step_end", lambda _state: None)
    monkeypatch.setattr(
        loop_module, "_maybe_scale_norm", lambda _state: (None, None, None, {})
    )

    def fail_save(*_args):
        raise OSError("disk full")

    state = SimpleNamespace(
        args=SimpleNamespace(max_train_steps=2),
        accelerator=SimpleNamespace(sync_gradients=True),
        initial_step=0,
        train_dataloader=[object()],
        stage_plan=None,
        stage_index=-1,
        global_step=0,
        current_step=SimpleNamespace(value=0),
        progress_bar=SimpleNamespace(update=lambda _amount: None),
        saver=SimpleNamespace(save_interrupt_state=fail_save),
        network=object(),
        stop_controller=SimpleNamespace(requested=True),
        stop_requested=False,
        optimizer_eval_fn=lambda: None,
    )

    with pytest.raises(OSError, match="disk full"):
        loop_module._run_epoch_steps(object(), state, epoch=0)

    assert state.stop_requested is True
    assert state.at_optimizer_boundary is True
