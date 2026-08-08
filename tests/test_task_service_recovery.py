"""Regression tests for daemon-authoritative WebUI task state."""

from __future__ import annotations

import asyncio

import pytest


def _run(coro):
    return asyncio.run(coro)


def test_reconcile_restores_running_daemon_job(monkeypatch):
    import webui.services.task_service as task_module
    from webui.services.task_service import TaskService, TaskState

    svc = TaskService()
    job = {
        "id": "20260714-120000-abcdef",
        "kind": "command",
        "method": "lora-gui",
        "argv": ["tasks.py", "lora-gui", "lokr"],
        "state": "running",
        "pid": 4321,
        "started_at": 1784001600.0,
        "stdout_path": "output/daemon/jobs/job/stdout.log",
        "progress_path": "output/daemon/jobs/job/progress.jsonl",
        "sample_dir": "output/daemon/jobs/job/sample",
    }
    monitor_calls: list[tuple[str, str | None]] = []

    async def list_jobs():
        return [job]

    monkeypatch.setattr(task_module.daemon_client, "list_jobs", list_jobs)
    monkeypatch.setattr(
        svc,
        "_ensure_monitors",
        lambda task, progress_path=None: monitor_calls.append((task.id, progress_path)),
    )

    _run(svc.reconcile_daemon_jobs())

    restored = svc.get_task(job["id"])
    assert restored is not None
    assert restored.state == TaskState.RUNNING
    assert restored.command == "lora-gui"
    assert restored.args == ["lokr"]
    assert restored.pid == 4321
    assert restored.job_id == job["id"]
    assert restored.is_training is True
    assert restored.info()["category"] == "training"
    assert monitor_calls == [(job["id"], job["progress_path"])]


def test_ensure_monitors_is_idempotent(monkeypatch):
    from webui.services.task_service import Task, TaskService, TaskState

    svc = TaskService()
    task = Task(
        id="one-monitor",
        command="lora",
        args=[],
        state=TaskState.PENDING,
        job_id="one-monitor",
        is_training=True,
    )

    async def wait_forever(*_args):
        await asyncio.Event().wait()

    monkeypatch.setattr(svc, "_poll_daemon_job", wait_forever)
    monkeypatch.setattr(svc, "_watch_progress_jsonl", wait_forever)

    async def exercise():
        svc._ensure_monitors(task, "progress.jsonl")
        first_poller = svc._pollers[task.id]
        first_watcher = svc._progress_watchers[task.id]
        svc._ensure_monitors(task, "progress.jsonl")

        assert svc._pollers[task.id] is first_poller
        assert svc._progress_watchers[task.id] is first_watcher
        await svc.close()

    _run(exercise())


def test_queue_poll_recovers_job_missed_at_startup(monkeypatch):
    import webui.services.task_service as task_module
    from webui.services.task_service import TaskService, TaskState

    svc = TaskService()
    job = {
        "id": "late-daemon-job",
        "kind": "command",
        "method": "lora-gui",
        "argv": ["tasks.py", "lora-gui", "lora"],
        "state": "running",
        "pid": 731,
        "submitted_at": 1784001600.0,
        "progress_path": "output/daemon/jobs/late-daemon-job/progress.jsonl",
    }
    monitor_calls: list[tuple[str, str | None]] = []

    async def list_jobs():
        return [job]

    async def health():
        return {"paused": False}

    monkeypatch.setattr(task_module.daemon_client, "list_jobs", list_jobs)
    monkeypatch.setattr(task_module.daemon_client, "health", health)
    monkeypatch.setattr(
        svc,
        "_ensure_monitors",
        lambda task, progress_path=None: monitor_calls.append((task.id, progress_path)),
    )

    status = _run(svc.get_queue_status())

    restored = svc.get_task(job["id"])
    assert status == {"daemon_up": True, "paused": False, "positions": {}}
    assert restored is not None
    assert restored.state == TaskState.RUNNING
    assert restored.pid == 731
    assert monitor_calls == [(job["id"], job["progress_path"])]


def test_monitoring_error_does_not_override_daemon_terminal_state(monkeypatch):
    import webui.services.task_service as task_module
    from webui.services.task_service import Task, TaskService, TaskState

    svc = TaskService()
    task = Task(
        id="job-running",
        command="lora",
        args=[],
        state=TaskState.PENDING,
        job_id="job-running",
    )
    replies = iter(
        [
            {"state": "running", "pid": 123, "stdout_path": "ignored.log"},
            {"state": "done", "rc": 0, "stdout_path": "ignored.log"},
        ]
    )

    async def get_job(_job_id):
        return next(replies)

    async def broken_telemetry(_task):
        raise RuntimeError("telemetry parser failed")

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(task_module.daemon_client, "get_job", get_job)
    monkeypatch.setattr(svc, "_drain_stdout", broken_telemetry)
    monkeypatch.setattr(task_module.asyncio, "sleep", no_sleep)

    _run(svc._poll_daemon_job(task))

    assert task.state == TaskState.SUCCESS
    assert task.exit_code == 0


def test_submit_failure_does_not_leave_fake_failed_task(monkeypatch):
    import webui.services.task_service as task_module
    from webui.services.daemon_client import DaemonError
    from webui.services.task_service import TaskService

    svc = TaskService()

    async def fail_submit(*_args, **_kwargs):
        raise DaemonError("daemon unavailable")

    monkeypatch.setattr(task_module.daemon_client, "submit_command", fail_submit)

    with pytest.raises(DaemonError):
        _run(svc.start_task("lora"))

    assert svc.list_tasks() == []


def test_webui_command_catalog_only_contains_real_cli_commands():
    from tasks import COMMANDS
    from webui.services.task_catalog import COMMAND_CATALOG

    assert set(COMMAND_CATALOG) <= set(COMMANDS)
    assert "easycontrol" in COMMAND_CATALOG
    assert "test-easycontrol" in COMMAND_CATALOG
    assert {
        "sr-prep",
        "sr-phase0",
        "sr-test",
        "sr-build-hr-pool",
        "sr-detect-text",
        "sr-train",
        "sr-rsd-train",
        "sr-rsd-dryrun",
        "sr-rsd-infer",
    } <= set(COMMAND_CATALOG)
    assert "sr-setup" not in COMMAND_CATALOG
    assert "exp-easycontrol" not in COMMAND_CATALOG
    assert "exp-ip-adapter" not in COMMAND_CATALOG


def test_task_api_rejects_removed_command_before_submission():
    from fastapi import HTTPException

    from webui.api.tasks import TaskStartRequest, start_task

    with pytest.raises(HTTPException) as exc_info:
        _run(start_task(TaskStartRequest(command="exp-easycontrol")))

    assert exc_info.value.status_code == 400


def test_task_api_maps_daemon_failure_to_502(monkeypatch):
    from fastapi import HTTPException

    from webui.api import tasks as tasks_api
    from webui.services.daemon_client import DaemonError

    async def fail_start(*_args, **_kwargs):
        raise DaemonError("internal daemon location")

    monkeypatch.setattr(tasks_api.task_service, "start_task", fail_start)

    with pytest.raises(HTTPException) as exc_info:
        _run(tasks_api.start_task(tasks_api.TaskStartRequest(command="lora")))

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Training daemon is unavailable"
