"""Recovery metadata and JSONL boundaries; no worker or subprocess is started."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from anima_daemon import config, proc
from anima_daemon.jobs import Job, load_all
from anima_daemon.manager import JobManager
from library.runtime import profiles
from library.training.progress import ProgressSink


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    manager = JobManager()
    monkeypatch.setattr(manager, "_refresh_recovery_metadata", lambda job: None)
    monkeypatch.setattr(manager, "_lift_result", lambda job: None)
    return manager


@pytest.mark.parametrize("old_status", ["paused", "ok", "stopped"])
def test_new_process_failure_does_not_reuse_old_terminal(manager, old_status):
    job = Job("rerun", "lora", "default", state="running", legacy=True)
    job.dir.mkdir(parents=True)
    job.progress_path = str(job.dir / "progress.jsonl")
    old = (
        json.dumps({"ev": "run_end", "status": old_status, "final_step": 12}) + "\n"
    ).encode()
    Path(job.progress_path).write_bytes(old)
    manager._begin_progress_run(job)
    manager._finalize_from_exit(job, SimpleNamespace(poll=lambda: 1))
    assert job.state == "error" and job.rc == 1
    assert job.progress_epoch == 1 and job.progress_start_offset == len(old)
    assert Path(job.progress_path).read_bytes() == old


def test_reconcile_does_not_park_from_previous_process_stream(manager, monkeypatch):
    job = Job("orphan", "lora", "default", state="running", legacy=True)
    job.dir.mkdir(parents=True)
    job.progress_path = str(job.dir / "progress.jsonl")
    Path(job.progress_path).write_text('{"ev":"run_end","status":"paused"}\n')
    manager._begin_progress_run(job)
    monkeypatch.setattr(proc, "is_alive", lambda *args: False)
    manager._reconcile()
    assert manager.get(job.id).state == "error"


def test_only_new_terminal_can_park_relaunched_process(manager):
    job = Job("rerun", "lora", "default", state="running", legacy=True)
    job.dir.mkdir(parents=True)
    job.progress_path = str(job.dir / "progress.jsonl")
    Path(job.progress_path).write_text('{"ev":"run_end","status":"ok"}\n')
    manager._begin_progress_run(job)
    with open(job.progress_path, "a") as stream:
        stream.write('{"ev":"run_start","pid":123}\n')
        stream.write(
            '{"ev":"run_end","status":"paused","state_dir":"new-state","final_step":7}\n'
        )
    manager._finalize_from_exit(job, SimpleNamespace(poll=lambda: 0))
    assert job.state == "paused" and job.released
    assert job.resume_state_dir == "new-state"


@pytest.mark.parametrize("tail", [b"\n", b"", b'\n{"ev":"step"'])
def test_daemon_sink_preserves_previous_bytes_and_separates_tail(
    tmp_path, monkeypatch, tail
):
    monkeypatch.setenv("ANIMA_DAEMON_JOB_DIR", str(tmp_path))
    path = tmp_path / "progress.jsonl"
    old = b'{"ev":"sample","path":"old.png"}' + tail
    path.write_bytes(old)
    sink = ProgressSink(str(path), run="new", method="lora", preset="default")
    sink.run_start(total_steps=8, total_epochs=1, pid=123)
    sink.run_end(status="ok", final_step=8)
    data = path.read_bytes()
    assert data.startswith(old)
    new = data[len(old) :].lstrip(b"\n").splitlines()
    assert [json.loads(line)["ev"] for line in new] == ["run_start", "run_end"]


def test_standalone_sink_still_replaces_old_stream(tmp_path, monkeypatch):
    monkeypatch.delenv("ANIMA_DAEMON_JOB_DIR", raising=False)
    path = tmp_path / "progress.jsonl"
    path.write_text('{"ev":"sample","path":"old.png"}\n')
    sink = ProgressSink(str(path), run="new", method="lora", preset=None)
    sink.run_start(total_steps=1, total_epochs=1, pid=123)
    sink.run_end(status="ok", final_step=1)
    assert [json.loads(line)["ev"] for line in path.read_text().splitlines()] == [
        "run_start",
        "run_end",
    ]


@pytest.fixture
def registered_runtime(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    (configs / "custom").mkdir(parents=True)
    path = configs / "custom/runtime.toml"
    path.write_text(f"[runtimes.curation]\npython={json.dumps(sys.executable)}\n")
    original = profiles.resolve_runtime
    registration = JobManager._runtime_registration
    monkeypatch.setattr(
        profiles, "resolve_runtime", lambda name: original(name, configs)
    )
    monkeypatch.setattr(
        JobManager,
        "_runtime_registration",
        staticmethod(lambda name: registration(name, configs_dir=configs)),
    )
    return path


def test_removed_registration_rejects_saved_job_even_if_fallback_path_is_same(
    manager, registered_runtime
):
    job = manager.submit_command(
        label="prepare", argv=["unused"], runtime_profile="curation", start=False
    )
    assert job.runtime_registration["curation"] is True
    registered_runtime.write_text("")
    assert profiles.resolve_runtime("curation").python == Path(job.python_executable)
    with pytest.raises(ValueError, match="removed"):
        manager._validate_job_runtime(job)


def test_pinned_runtime_rejects_invalid_name_and_changed_environment(
    manager, registered_runtime
):
    with pytest.raises(ValueError, match="Unknown runtime profile"):
        manager._resolve_runtime("not-registered", sys.executable, {})
    job = manager.submit_command(
        label="prepare", argv=["unused"], runtime_profile="curation", start=False
    )
    registered_runtime.write_text(
        f'[runtimes.curation]\npython={json.dumps(sys.executable)}\nenv={{MODE="changed"}}\n'
    )
    with pytest.raises(ValueError, match="environment differs"):
        manager._validate_job_runtime(job)


def test_released_requeue_validates_runtime_before_mutating_job(
    manager, registered_runtime, tmp_path
):
    job = manager.submit_command(
        label="prepare", argv=["unused"], runtime_profile="curation", start=False
    )
    job.state, job.released = "paused", True
    job.resume_state_dir = str(tmp_path)
    manager._jobs[job.id] = job
    registered_runtime.write_text("")
    response = manager.unpause_job(job.id)
    assert "removed" in response["error"]
    assert job.state == "paused" and job.released and job.resume_count == 0


def test_launch_revalidates_runtime_without_starting_process(
    manager, registered_runtime, monkeypatch
):
    job = manager.submit_command(
        label="prepare", argv=["unused"], runtime_profile="curation", start=False
    )
    registered_runtime.write_text("")
    monkeypatch.setattr(
        proc, "spawn_detached", lambda *args, **kwargs: pytest.fail("must not spawn")
    )
    manager._launch_and_monitor(job)
    assert job.state == "error" and "removed" in job.error


def test_pinned_interpreter_cannot_silently_follow_changed_registration(
    manager, registered_runtime, tmp_path
):
    job = manager.submit_command(
        label="prepare", argv=["unused"], runtime_profile="curation", start=False
    )
    other = tmp_path / "other-python"
    other.symlink_to(sys.executable)
    registered_runtime.write_text(
        f"[runtimes.curation]\npython={json.dumps(str(other))}\n"
    )
    with pytest.raises(ValueError, match="interpreter differs"):
        manager._validate_job_runtime(job)
    assert job.python_executable == sys.executable


@pytest.mark.parametrize("state", ["paused", "stopped"])
def test_unknown_fields_roundtrip_without_exposing_private_continuation(manager, state):
    raw = {
        "id": "legacy",
        "method": "lora",
        "preset": "default",
        "state": state,
        "paused_at": 12.5,
        "stop_requested": True,
        "release_requested": True,
        "future_payload": {"nested": [1, 2]},
        "continuation": {
            "mode": "extend",
            "token": "private",
            "blueprint": {"secret": 1},
        },
    }
    job = Job.from_dict(raw)
    job.persist()
    restored = load_all()[job.id]
    saved = json.loads((job.dir / "job.json").read_text())
    assert saved["future_payload"] == raw["future_payload"]
    assert saved["continuation"] == raw["continuation"]
    assert restored.state == state and restored.paused_at == 12.5
    assert restored.stop_requested and restored.release_requested
    assert restored.public()["continuation"] == {"mode": "extend"}
    assert restored.public()["future_payload"] == raw["future_payload"]
