from __future__ import annotations

import json
import random
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from library.training.state import (
    build_train_state,
    capture_rng_state,
    normalize_train_state,
    restore_rng_state,
    state_is_complete,
    write_complete_marker,
)
from train import _resume_config_signature


def _write_resume_state(path, step: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "train_state.json").write_text(
        json.dumps(build_train_state(global_step=step)), encoding="utf-8"
    )
    write_complete_marker(path)


def _auto_resume_saver(args):
    from library.training.checkpoints import CheckpointSaver

    return CheckpointSaver(
        args=args,
        accelerator=object(),
        save_dtype=None,
        metadata={},
        minimum_metadata={},
        get_sai_model_spec_fn=lambda _args: {},
        current_epoch=SimpleNamespace(value=0),
        current_step=SimpleNamespace(value=0),
    )


def test_legacy_current_step_is_normalized():
    state = normalize_train_state({"current_step": "7", "epoch": "2"})
    assert state["global_step"] == 7
    assert state["current_step"] == 7
    assert state["current_epoch"] == 2
    assert state["micro_batch_offset"] == 0


def test_resume_signature_ignores_daemon_snapshot_path_but_not_training_config():
    base = {
        "config_file": "/tmp/job-a/config.snapshot.toml",
        "output_name": "unit",
        "network_module": "networks.lora_anima",
        "learning_rate": 1e-5,
        "max_train_steps": 200,
        "progress_jsonl": "/tmp/job-a/progress.jsonl",
        "sample_dir": "/tmp/job-a/sample",
        "logging_dir": "/tmp/job-a/logs",
    }
    next_job = dict(base)
    next_job.update(
        {
            "config_file": "/tmp/job-b/config.snapshot.toml",
            "progress_jsonl": "/tmp/job-b/progress.jsonl",
            "sample_dir": "/tmp/job-b/sample",
            "logging_dir": "/tmp/job-b/logs",
        }
    )

    assert _resume_config_signature(SimpleNamespace(**base)) == _resume_config_signature(
        SimpleNamespace(**next_job)
    )

    changed = dict(next_job, learning_rate=2e-5)
    assert _resume_config_signature(SimpleNamespace(**next_job)) != _resume_config_signature(
        SimpleNamespace(**changed)
    )


def test_numpy_rng_round_trip_preserves_sequence():
    np.random.seed(1234)
    encoded = capture_rng_state(include_cuda=False)
    expected = np.random.random(5)
    np.random.seed(999)
    restore_rng_state(encoded)
    np.testing.assert_allclose(np.random.random(5), expected)


def test_python_and_torch_rng_round_trip():
    random.seed(123)
    torch.manual_seed(123)
    encoded = capture_rng_state(include_cuda=False)
    expected_py = [random.random() for _ in range(3)]
    expected_torch = torch.rand(3)
    random.seed(999)
    torch.manual_seed(999)
    restore_rng_state(encoded)
    assert [random.random() for _ in range(3)] == expected_py
    assert torch.equal(torch.rand(3), expected_torch)


def test_complete_marker_distinguishes_partial_state(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps(build_train_state(global_step=3)), encoding="utf-8"
    )
    # Legacy parseable states remain readable before the marker is introduced.
    assert state_is_complete(state_dir)
    write_complete_marker(state_dir)
    assert state_is_complete(state_dir)
    (state_dir / "train_state.json").write_text("{", encoding="utf-8")
    assert not state_is_complete(state_dir)


def test_strict_auto_resume_rejects_unpublished_schema_v2_state(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps(build_train_state(global_step=3)), encoding="utf-8"
    )
    assert state_is_complete(state_dir)
    assert not state_is_complete(state_dir, require_marker=True)
    write_complete_marker(state_dir)
    assert state_is_complete(state_dir, require_marker=True)


def test_auto_resume_uses_newer_normal_state_over_stale_interrupted_state(tmp_path):
    from library.io.output_layout import layout_from_args

    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="unit",
        resume=None,
        max_train_steps=5,
        checkpointing_epochs=None,
        save_state_on_train_end=True,
        auto_resume=True,
    )
    layout = layout_from_args(args)
    _write_resume_state(layout.interrupted_state, 2)
    _write_resume_state(layout.state, 3)

    _auto_resume_saver(args).auto_resume()

    assert args.resume == layout.state
    assert args.skip_until_initial_step is True


def test_auto_resume_prefers_interrupted_state_when_steps_are_equal(tmp_path):
    from library.io.output_layout import layout_from_args

    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="unit",
        resume=None,
        max_train_steps=5,
        checkpointing_epochs=None,
        save_state_on_train_end=True,
        auto_resume=True,
    )
    layout = layout_from_args(args)
    _write_resume_state(layout.interrupted_state, 3)
    _write_resume_state(layout.state, 3)

    _auto_resume_saver(args).auto_resume()

    assert args.resume == layout.interrupted_state


def test_interrupt_save_forces_explicit_interrupted_marker(tmp_path):
    from library.training.checkpoints import CheckpointSaver

    class Accelerator:
        is_main_process = True

        def unwrap_model(self, model):
            return model

        def register_save_state_pre_hook(self, hook):
            self.save_hook = hook

        def register_load_state_pre_hook(self, hook):
            self.load_hook = hook

        def wait_for_everyone(self):
            pass

        def save_state(self, output_dir):
            from pathlib import Path

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.save_hook([self.network], [], output_dir)

    accelerator = Accelerator()
    network = object()
    accelerator.network = network
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="unit",
        save_model_as="safetensors",
    )
    saver = CheckpointSaver(
        args=args,
        accelerator=accelerator,
        save_dtype=None,
        metadata={},
        minimum_metadata={},
        get_sai_model_spec_fn=lambda _args: {},
        current_epoch=SimpleNamespace(value=2),
        current_step=SimpleNamespace(value=6),
    )
    # Even a stale/custom provider value must not override the save interface's
    # authoritative interrupted-state intent.
    saver.set_runtime_state_provider(
        lambda: {
            "global_step": 7,
            "current_epoch": 2,
            "micro_batch_offset": 3,
            "interrupted": False,
        }
    )
    saver.register_hooks(network)

    state_dir = saver.save_interrupt_state(
        network, global_step=7, epoch=2, save_weights=False
    )
    state = json.loads((tmp_path / "unit-interrupted-state" / "train_state.json").read_text())

    assert state_dir == str(tmp_path / "unit-interrupted-state")
    assert state["global_step"] == 7
    assert state["micro_batch_offset"] == 3
    assert state["interrupted"] is True
    assert (tmp_path / "unit-interrupted-state" / "complete.marker").is_file()


def test_explicit_resume_rejects_incomplete_or_mismatched_state(tmp_path):
    from library.training.checkpoints import CheckpointSaver

    class Accelerator:
        is_main_process = True

        def unwrap_model(self, model):
            return model

        def register_save_state_pre_hook(self, hook):
            self.save_hook = hook

        def register_load_state_pre_hook(self, hook):
            self.load_hook = hook

    accelerator = Accelerator()
    network = object()
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="unit",
        config_signature="cfg-good",
        dataset_signature="data-good",
    )
    saver = CheckpointSaver(
        args=args,
        accelerator=accelerator,
        save_dtype=None,
        metadata={},
        minimum_metadata={},
        get_sai_model_spec_fn=lambda _args: {},
        current_epoch=SimpleNamespace(value=1),
        current_step=SimpleNamespace(value=2),
    )
    saver.register_hooks(network)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "train_state.json").write_text(
        json.dumps(
            build_train_state(
                global_step=2,
                config_signature="cfg-good",
                dataset_signature="data-good",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete"):
        accelerator.load_hook([network], str(incomplete))

    mismatched = tmp_path / "mismatched"
    mismatched.mkdir()
    (mismatched / "train_state.json").write_text(
        json.dumps(
            build_train_state(
                global_step=2,
                config_signature="cfg-other",
                dataset_signature="data-good",
            )
        ),
        encoding="utf-8",
    )
    write_complete_marker(mismatched)
    with pytest.raises(ValueError, match="config signature mismatch"):
        accelerator.load_hook([network], str(mismatched))


def test_multirank_state_save_publishes_one_complete_directory(tmp_path):
    """Every rank's payload must survive the single atomic publication."""

    from library.training.checkpoints import _atomic_accelerator_save_state

    barrier = threading.Barrier(2)

    class Accelerator:
        num_processes = 2

        def __init__(self, rank):
            self.process_index = rank
            self.is_main_process = rank == 0

        def wait_for_everyone(self):
            barrier.wait(timeout=5)

        def save_state(self, output_dir):
            from pathlib import Path

            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"rank-{self.process_index}.bin").write_text(
                f"rank {self.process_index}", encoding="ascii"
            )

    target = tmp_path / "state"
    errors = [None, None]

    def run(rank):
        try:
            _atomic_accelerator_save_state(Accelerator(rank), str(target))
        except Exception as exc:  # noqa: BLE001 - collect thread failures
            errors[rank] = exc

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == [None, None]
    assert (target / "rank-0.bin").read_text(encoding="ascii") == "rank 0"
    assert (target / "rank-1.bin").read_text(encoding="ascii") == "rank 1"
    assert (target / "complete.marker").is_file()
    assert (target / ".save-token").is_file()


def test_multirank_state_save_failure_keeps_previous_complete_state(tmp_path):
    """A failed rank must not publish a marker or replace the prior state."""

    from library.training.checkpoints import _atomic_accelerator_save_state

    barrier = threading.Barrier(2)

    class Accelerator:
        num_processes = 2

        def __init__(self, rank):
            self.process_index = rank
            self.is_main_process = rank == 0

        def wait_for_everyone(self):
            barrier.wait(timeout=5)

        def save_state(self, output_dir):
            from pathlib import Path

            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"rank-{self.process_index}.bin").write_text(
                "new", encoding="ascii"
            )
            if self.process_index == 1:
                raise RuntimeError("simulated rank save failure")

    target = tmp_path / "state"
    target.mkdir()
    (target / "old.bin").write_text("old", encoding="ascii")
    (target / ".save-token").write_text("old-token", encoding="ascii")
    write_complete_marker(target)
    errors = [None, None]

    def run(rank):
        try:
            _atomic_accelerator_save_state(Accelerator(rank), str(target))
        except Exception as exc:  # noqa: BLE001 - collect thread failures
            errors[rank] = exc

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(isinstance(error, RuntimeError) for error in errors)
    assert "incomplete state" in str(errors[0]) or "rank 1" in str(errors[0])
    assert (target / "old.bin").read_text(encoding="ascii") == "old"
    assert (target / ".save-token").read_text(encoding="ascii") == "old-token"
    assert (target / "complete.marker").is_file()
