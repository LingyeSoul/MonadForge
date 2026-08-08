"""Regression tests for the WebUI checkpoint resume / wipe logic.

Two bugs surfaced by user feedback:

1. **"Restart from scratch" was a no-op for end-of-training state dirs.**
   ``wipe_checkpoint`` deletes ``<name>-checkpoint-state`` (mid-training, from
   ``checkpointing_epochs``) AND ``<name>-state`` (end-of-training, from
   ``save_state_on_train_end``). But the frontend ``wipeAndTrain`` only
   stripped the ``-checkpoint-state`` suffix when recovering ``output_name``
   from the state dir path, so for an end-state dir it sent a wrong
   ``output_name`` (``"lora-state"`` instead of ``"lora"``) and the backend
   never matched the real dir → training resumed from the breakpoint anyway.

2. (Guarded in ``test_task_service_metrics``) progress display races — kept
   honest here by asserting the absolute-step contract the tqdm fix relies on.

These tests pin the backend behaviour so the frontend's suffix-stripping has a
correct contract to target: both state-dir layouts must be wiped when the
*real* ``output_name`` is passed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.services import config_service as cs  # noqa: E402


@pytest.fixture
def fake_root(tmp_path: Path, monkeypatch):
    """Point ``config_service.ROOT`` at a temp dir so the wipe logic doesn't
    touch the real repo's ``output/`` tree."""
    monkeypatch.setattr(cs, "ROOT", tmp_path)
    return tmp_path


def _write_state(state_dir: Path, *, step: int, epoch: int = 0) -> None:
    """Write a minimal ``train_state.json`` like CheckpointSaver persists."""
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": epoch, "current_step": step}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# find_resumable_checkpoint — prefers checkpoint-state, falls back to -state
# ---------------------------------------------------------------------------


def test_find_resumable_prefers_checkpoint_state(fake_root: Path):
    """When both dirs exist, the mid-training snapshot wins (it's newer)."""
    out_dir = fake_root / "output" / "ckpt"
    _write_state(out_dir / "lora-checkpoint-state", step=50)
    _write_state(out_dir / "lora-state", step=138)

    merged = {
        "output_dir": "output/ckpt",
        "output_name": "lora",
        "checkpointing_epochs": 1,
        "save_state_on_train_end": True,
    }
    state_dir, step = cs.find_resumable_checkpoint(merged)

    assert state_dir.name == "lora-checkpoint-state"
    assert step == 50


def test_find_resumable_falls_back_to_end_state(fake_root: Path):
    """A finished run (only ``-state`` present, no checkpointing_epochs) is
    still resumable via ``save_state_on_train_end``."""
    out_dir = fake_root / "output" / "ckpt"
    _write_state(out_dir / "lora-state", step=138)

    merged = {
        "output_dir": "output/ckpt",
        "output_name": "lora",
        "save_state_on_train_end": True,
    }
    state_dir, step = cs.find_resumable_checkpoint(merged)

    assert state_dir.name == "lora-state"
    assert step == 138


def test_find_resumable_returns_none_without_flags(fake_root: Path):
    """No resume flag set → no resumable dir, even if one is on disk."""
    out_dir = fake_root / "output" / "ckpt"
    _write_state(out_dir / "lora-state", step=138)

    merged = {"output_dir": "output/ckpt", "output_name": "lora"}
    assert cs.find_resumable_checkpoint(merged) is None


# ---------------------------------------------------------------------------
# wipe_checkpoint — must clear BOTH layouts given the real output_name
# ---------------------------------------------------------------------------


def test_wipe_removes_checkpoint_state_dir_and_sidecar(fake_root: Path):
    """Mid-training: dir + sidecar adapter both deleted."""
    out_dir = fake_root / "output" / "ckpt"
    _write_state(out_dir / "lora-checkpoint-state", step=50)
    sidecar = out_dir / "lora-checkpoint.safetensors"
    sidecar.write_text("weights")

    cs.wipe_checkpoint("output/ckpt", "lora")

    assert not (out_dir / "lora-checkpoint-state").exists()
    assert not sidecar.exists()


def test_wipe_removes_end_state_dir(fake_root: Path):
    """Regression for user report: end-of-training ``-state`` dir must also
    be wiped when the real output_name is passed. The frontend used to send a
    mangled name (``"lora-state"``) for this layout, so the dir survived and
    training auto-resumed. The backend itself is correct; this pins it."""
    out_dir = fake_root / "output" / "ckpt"
    _write_state(out_dir / "lora-state", step=138)

    cs.wipe_checkpoint("output/ckpt", "lora")

    assert not (out_dir / "lora-state").exists()


def test_wipe_with_mangled_name_leaves_end_state(fake_root: Path):
    """Documents the pre-fix bug as a negative test: feeding the wrong
    ``output_name`` (``"lora-state"`` — what the buggy frontend sent) makes
    the backend look for ``lora-state-state`` and miss the real dir. This is
    exactly why the frontend suffix-stripping matters."""
    out_dir = fake_root / "output" / "ckpt"
    _write_state(out_dir / "lora-state", step=138)

    cs.wipe_checkpoint("output/ckpt", "lora-state")

    # The real end-state dir is untouched — proves the name matters.
    assert (out_dir / "lora-state").exists()
    assert not (out_dir / "lora-state-state").exists()


def test_wipe_clears_both_layouts_at_once(fake_root: Path):
    """If both a stale checkpoint-state and an end-state exist, one wipe call
    with the real output_name clears both."""
    out_dir = fake_root / "output" / "ckpt"
    _write_state(out_dir / "lora-checkpoint-state", step=50)
    _write_state(out_dir / "lora-state", step=138)

    cs.wipe_checkpoint("output/ckpt", "lora")

    assert not (out_dir / "lora-checkpoint-state").exists()
    assert not (out_dir / "lora-state").exists()


def test_wipe_does_not_touch_final_checkpoint(fake_root: Path):
    """The intentional final product ``<name>.safetensors`` must survive a
    wipe — only the resumable state dirs (and the checkpoint-state sidecar)
    are deleted."""
    out_dir = fake_root / "output" / "ckpt"
    _write_state(out_dir / "lora-state", step=138)
    final = out_dir / "lora.safetensors"
    final.write_text("final weights")

    cs.wipe_checkpoint("output/ckpt", "lora")

    assert final.exists()


def test_wipe_removes_checkpoint_sidecars_for_all_supported_weight_formats(
    fake_root: Path,
):
    out_dir = fake_root / "output" / "ckpt"
    out_dir.mkdir(parents=True)
    for ext in (".safetensors", ".ckpt", ".pt"):
        (out_dir / f"lora-interrupted{ext}").write_text("weights")
        (out_dir / f"lora-checkpoint{ext}").write_text("weights")

    cs.wipe_checkpoint("output/ckpt", "lora")

    for ext in (".safetensors", ".ckpt", ".pt"):
        assert not (out_dir / f"lora-interrupted{ext}").exists()
        assert not (out_dir / f"lora-checkpoint{ext}").exists()
