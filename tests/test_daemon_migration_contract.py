"""Pure control-plane regression checks; no model or trainer is executed."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from anima_daemon import config, proc
from anima_daemon.jobs import Job, STATE_PAUSED, STATE_RUNNING, STATE_STOPPED
from anima_daemon.manager import JobManager


@pytest.fixture
def isolated_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(
        JobManager,
        "_command_runs_train",
        staticmethod(
            lambda argv: len(argv) > 1 and argv[:2] == ["tasks.py", "lora-gui"]
        ),
    )
    return JobManager()


@pytest.mark.parametrize(
    "name", ["config", "jobs", "manager", "proc", "server", "client", "continuation"]
)
def test_legacy_modules_share_identity(name):
    assert importlib.import_module(f"scripts.daemon.{name}") is importlib.import_module(
        f"anima_daemon.{name}"
    )


@pytest.mark.parametrize("field", ["rc", "returncode"])
def test_job_exit_code_dual_read_preserves_context(field):
    job = Job.from_dict(
        {
            "id": "old",
            "method": "lora",
            "preset": "default",
            field: 0,
            "context": {"model_type": "anima"},
            "recovery_state": "/state",
        }
    )
    assert job.rc == job.returncode == 0
    assert job.public()["rc"] == job.public()["returncode"] == 0
    assert job.recovery_state == "/state"
    assert job.context == {"model_type": "anima"}


def test_reconcile_never_prunes_history(isolated_manager, monkeypatch):
    job = Job("historic", "lora", "default", state="done", ended_at=1)
    job.persist()
    monkeypatch.setattr(config, "JOB_RETENTION_DAYS", 1)
    isolated_manager._reconcile()
    assert config.job_dir(job.id).is_dir()
    assert job.id in isolated_manager._jobs


def test_submission_persists_resolved_runtime_and_context(
    isolated_manager, monkeypatch
):
    from library.runtime import profiles

    monkeypatch.setattr(
        profiles,
        "resolve_runtime",
        lambda name: profiles.RuntimeSpec(
            name, Path(sys.executable), {"RUNTIME_TEST": "on"}
        ),
    )
    job = isolated_manager.submit_command(
        label="curation",
        argv=["tasks.py", "help"],
        context={"model_type": "moss", "workspace_id": "sample"},
        runtime_profile="curation",
        start=False,
        captured_env={"HF_HOME": "/tmp/submitter-hf", "PATH": "not-captured"},
    )
    persisted = json.loads((job.dir / "job.json").read_text())
    assert persisted["python_executable"] == sys.executable
    assert persisted["runtime_env"] == {"RUNTIME_TEST": "on"}
    assert persisted["captured_env"] == {"HF_HOME": "/tmp/submitter-hf"}
    assert persisted["context"]["model_type"] == "moss"


def test_pinned_runtime_environment_survives_submission_and_chain(
    isolated_manager, monkeypatch
):
    from library.runtime import profiles

    monkeypatch.setattr(
        profiles,
        "resolve_runtime",
        lambda name: profiles.RuntimeSpec(
            name, Path(sys.executable), {"RUNTIME_TEST": "pinned"}
        ),
    )
    job = isolated_manager.submit_command(
        label="prepare",
        argv=["tasks.py", "help"],
        python_executable=sys.executable,
        runtime_profile="curation",
        runtime_env={"RUNTIME_TEST": "pinned"},
        context={"model_type": "anima"},
        chain_train={"method": "lora", "preset": "default"},
        start=False,
    )
    isolated_manager._finalize(job, "done", rc=0)
    child = isolated_manager.get(job.chained_job_id)
    assert child.python_executable == sys.executable
    assert child.runtime_profile == "curation"
    assert child.runtime_env == {"RUNTIME_TEST": "pinned"}
    persisted = json.loads((child.dir / "job.json").read_text())
    assert persisted["runtime_env"] == child.runtime_env


@pytest.mark.parametrize("environment", [{"BAD=KEY": "x"}, {"KEY": 1}, []])
def test_runtime_environment_rejects_invalid_mapping(isolated_manager, environment):
    with pytest.raises(ValueError, match="runtime_env"):
        isolated_manager.submit_command(
            label="invalid",
            argv=["tasks.py", "help"],
            python_executable=sys.executable,
            runtime_env=environment,
        )
    assert not isolated_manager._jobs


@pytest.mark.parametrize("model", ["qwen21", "moss"])
def test_independent_model_launch_gets_job_identity_without_anima_stop_protocol(
    isolated_manager, monkeypatch, model
):
    job = Job(
        "line-job", f"{model}-train", "", kind="command", context={"model_type": model}
    )
    captured = {}
    monkeypatch.setattr(isolated_manager, "_build_cmd", lambda job: (["not-run"], {}))

    def spawn(cmd, **kwargs):
        captured.update(kwargs["env"])
        raise OSError("mock launch stopped before process creation")

    monkeypatch.setattr(proc, "spawn_detached", spawn)
    isolated_manager._launch_and_monitor(job)
    assert captured["ANIMA_DAEMON_JOB_DIR"] == str(job.dir)
    assert captured["ANIMA_DAEMON_JOB_ID"] == job.id
    assert "ANIMA_DAEMON_STOP_FILE" not in captured


@pytest.mark.parametrize("model", ["qwen21", "moss"])
def test_other_models_never_get_anima_continue(isolated_manager, model):
    job = Job(
        "other", "lora", "default", state=STATE_STOPPED, context={"model_type": model}
    )
    isolated_manager._jobs[job.id] = job
    assert isolated_manager.continue_job(job.id) is None


def test_legacy_resume_dispatch_is_disjoint(isolated_manager, monkeypatch):
    job = Job("task", "lora", "default", state=STATE_PAUSED)
    isolated_manager._jobs[job.id] = job
    calls = []
    monkeypatch.setattr(
        isolated_manager, "unpause_job", lambda jid: calls.append("unpause")
    )
    monkeypatch.setattr(
        isolated_manager, "continue_job", lambda jid: calls.append("continue")
    )
    isolated_manager.resume_job(job.id)
    job.state = STATE_STOPPED
    isolated_manager.resume_job(job.id)
    assert calls == ["unpause", "continue"]
    job.state = STATE_RUNNING
    job.release_requested = True
    with pytest.raises(ValueError):
        isolated_manager.resume_job(job.id)
    assert calls == ["unpause", "continue"]


def test_frozen_attempt_holds_gpu_slot_released_attempt_does_not(isolated_manager):
    job = Job("paused", "lora", "default", state=STATE_PAUSED)
    isolated_manager._jobs[job.id] = job
    assert isolated_manager.active_job() is job
    assert not isolated_manager._queue_is_idle_locked()
    job.released = True
    assert isolated_manager.active_job() is None
    assert not isolated_manager._queue_is_idle_locked()  # still an active lineage


def test_release_requeue_command_updates_argv_and_keeps_attempt(
    isolated_manager, tmp_path
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    job = Job(
        "released",
        "lora-gui",
        "default",
        kind="command",
        state=STATE_PAUSED,
        released=True,
        resume_state_dir=str(state_dir),
        argv=["tasks.py", "lora-gui", "lora", "--resume", "old"],
        rc=0,
        returncode=0,
    )
    isolated_manager._jobs[job.id] = job
    response = isolated_manager.unpause_job(job.id)
    assert response["job_id"] == "released"
    assert job.argv[-3:] == ["--resume", str(state_dir), "--skip_until_initial_step"]
    assert job.argv.count("--resume") == 1
    assert job.rc is job.returncode is None
    assert isolated_manager._queue.get_nowait() == (0, 0, job.id)


def test_stop_release_paused_is_terminal_without_signals(isolated_manager, monkeypatch):
    job = Job("released", "lora", "default", state=STATE_PAUSED, released=True)
    isolated_manager._jobs[job.id] = job
    monkeypatch.setattr(
        proc, "kill_tree", lambda *a, **kw: pytest.fail("no process to kill")
    )
    isolated_manager.stop(job.id)
    assert job.state == STATE_STOPPED


def test_release_request_on_gui_command_uses_training_protocol(isolated_manager):
    job = Job(
        "gui",
        "lora-gui",
        "default",
        kind="command",
        state=STATE_RUNNING,
        argv=["tasks.py", "lora-gui", "lora"],
    )
    isolated_manager._jobs[job.id] = job
    result = isolated_manager.pause_job(job.id, release_model=True)
    assert result["release_requested"]
    assert json.loads((job.dir / "pause.request").read_text())["release_model"]
