"""Regression tests for daemon-authoritative WebUI task state."""

from __future__ import annotations

import asyncio
import json

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

    async def list_job_groups_page(**_kwargs):
        return {"groups": [svc._physical_job_group(job)], "total": 1}

    monkeypatch.setattr(task_module.daemon_client, "list_job_groups_page", list_job_groups_page)
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


def test_reconcile_restores_stop_requested_job_as_stopping(monkeypatch):
    import webui.services.task_service as task_module
    from webui.services.task_service import TaskService, TaskState

    svc = TaskService()
    job = {
        "id": "20260714-120001-abcdef",
        "kind": "command",
        "method": "lora-gui",
        "argv": ["tasks.py", "lora-gui", "lokr"],
        "state": "running",
        "stop_requested": True,
        "pid": 4322,
        "started_at": 1784001601.0,
        "stdout_path": "output/daemon/jobs/job/stdout.log",
        "progress_path": "output/daemon/jobs/job/progress.jsonl",
        "sample_dir": "output/daemon/jobs/job/sample",
    }

    async def list_job_groups_page(**_kwargs):
        return {"groups": [svc._physical_job_group(job)], "total": 1}

    monkeypatch.setattr(task_module.daemon_client, "list_job_groups_page", list_job_groups_page)
    monkeypatch.setattr(svc, "_ensure_monitors", lambda *_args, **_kwargs: None)

    _run(svc.reconcile_daemon_jobs())

    restored = svc.get_task(job["id"])
    assert restored is not None
    assert restored.state == TaskState.STOPPING
    assert restored.info()["state"] == TaskState.STOPPING.value


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

    async def list_job_groups_page(**_kwargs):
        return {"groups": [svc._physical_job_group(job)], "total": 1}

    async def health():
        return {"paused": False}

    monkeypatch.setattr(task_module.daemon_client, "list_jobs", list_jobs)
    monkeypatch.setattr(task_module.daemon_client, "list_job_groups_page", list_job_groups_page)
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


def test_terminal_task_without_recovery_state_is_not_resumable():
    from webui.services.task_service import TaskService, TaskState

    task = TaskService()._task_from_daemon(
        {
            "id": "failed-without-state",
            "kind": "command",
            "method": "lora-gui",
            "argv": ["tasks.py", "lora-gui", "lora"],
            "state": "error",
            "error": "process exited",
        }
    )

    assert task.state == TaskState.FAILED
    assert task.resumable is False


def test_logical_task_merges_attempt_history_and_overlapping_steps(tmp_path):
    from webui.services.task_service import TaskService

    root_stdout = tmp_path / "root.stdout.log"
    child_stdout = tmp_path / "child.stdout.log"
    root_progress = tmp_path / "root.progress.jsonl"
    child_progress = tmp_path / "child.progress.jsonl"
    root_stdout.write_text("root output\n", encoding="utf-8")
    child_stdout.write_text("child output\n", encoding="utf-8")
    root_progress.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"ev": "run_start", "total_steps": 3},
                {"ev": "step", "global_step": 1, "epoch": 1, "loss/average": 0.9},
                {"ev": "step", "global_step": 2, "epoch": 1, "loss/average": 0.8},
                {"ev": "run_end", "final_step": 2},
            )
        ),
        encoding="utf-8",
    )
    child_progress.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"ev": "run_start", "total_steps": 3},
                {"ev": "step", "global_step": 2, "epoch": 1, "loss/average": 0.7},
                {"ev": "step", "global_step": 3, "epoch": 1, "loss/average": 0.6},
                {"ev": "run_end", "final_step": 3},
            )
        ),
        encoding="utf-8",
    )
    group = {
        "id": "root",
        "root_job_id": "root",
        "current_job_id": "child",
        "kind": "command",
        "method": "lora-gui",
        "argv": ["tasks.py", "lora-gui", "lora"],
        "state": "done",
        "attempts": [
            {
                "id": "root",
                "root_job_id": "root",
                "attempt_index": 0,
                "kind": "command",
                "method": "lora-gui",
                "argv": ["tasks.py", "lora-gui", "lora"],
                "state": "error",
                "stdout_path": str(root_stdout),
                "progress_path": str(root_progress),
            },
            {
                "id": "child",
                "root_job_id": "root",
                "parent_job_id": "root",
                "attempt_index": 1,
                "kind": "command",
                "method": "lora-gui",
                "argv": ["tasks.py", "lora-gui", "lora"],
                "state": "done",
                "recovery_step": 2,
                "rc": 0,
                "stdout_path": str(child_stdout),
                "progress_path": str(child_progress),
            },
        ],
    }

    task = TaskService()._task_from_daemon(group)
    info = task.info()

    assert task.id == "root"
    assert task.job_id == "child"
    assert info["attempt_count"] == 2
    assert [attempt["job_id"] for attempt in info["attempts"]] == ["root", "child"]
    assert task.lines == [
        "[attempt 1/2 · root]",
        "root output",
        "[attempt 2/2 · child from step 2]",
        "child output",
    ]
    metrics = task.parser.metrics
    assert metrics.step_history == [1, 2, 3]
    assert metrics.loss_history == [0.9, 0.7, 0.6]


def test_resume_reuses_logical_task_and_delete_targets_group(monkeypatch):
    import webui.services.task_service as task_module
    from webui.services.task_service import TaskService

    svc = TaskService()
    initial = {
        "id": "root",
        "root_job_id": "root",
        "current_job_id": "root",
        "kind": "command",
        "method": "lora-gui",
        "argv": ["tasks.py", "lora-gui", "lora"],
        "state": "error",
        "recovery_state": "/state/step-2",
        "attempts": [
            {
                "id": "root",
                "root_job_id": "root",
                "attempt_index": 0,
                "kind": "command",
                "method": "lora-gui",
                "argv": ["tasks.py", "lora-gui", "lora"],
                "state": "error",
                "recovery_state": "/state/step-2",
            }
        ],
    }
    task = svc._task_from_daemon(initial)
    svc._tasks[task.id] = task
    resumed_group = {
        **initial,
        "current_job_id": "child",
        "state": "queued",
        "recovery_state": "/state/step-2",
        "attempts": [
            initial["attempts"][0],
            {
                "id": "child",
                "root_job_id": "root",
                "parent_job_id": "root",
                "attempt_index": 1,
                "kind": "command",
                "method": "lora-gui",
                "argv": ["tasks.py", "lora-gui", "lora"],
                "state": "queued",
                "recovery_step": 2,
                "recovery_state": "/state/step-2",
            },
        ],
    }
    deleted: list[str] = []

    async def resume(_job_id):
        return {"root_job_id": "root", "job_id": "child"}

    async def get_job_group(_root_id):
        return resumed_group

    async def delete_job_group(root_id):
        deleted.append(root_id)
        return {"ok": True}

    monkeypatch.setattr(task_module.daemon_client, "resume", resume)
    monkeypatch.setattr(task_module.daemon_client, "get_job_group", get_job_group)
    monkeypatch.setattr(task_module.daemon_client, "delete_job_group", delete_job_group)
    monkeypatch.setattr(svc, "_ensure_monitors", lambda *_args, **_kwargs: None)

    resumed = _run(svc.resume_task("root"))

    assert resumed is task
    assert list(svc._tasks) == ["root"]
    assert task.job_id == "child"
    assert task.info()["attempt_count"] == 2

    task.state = task_module.TaskState.CANCELLED
    assert _run(svc.delete_task("root")) is True
    assert deleted == ["root"]


def test_gui_training_snapshot_pins_variant_and_manifest(monkeypatch):
    import webui.services.config_service as config_service
    from webui.services.task_service import TaskService

    monkeypatch.setattr(
        config_service,
        "merged_gui_variant_preset",
        lambda variant, preset: (
            {"output_name": f"{variant}-{preset}", "max_train_steps": 100},
            {},
        ),
    )

    snapshot = TaskService._training_config_snapshot(
        "lora-gui",
        ["tasks.py", "lora-gui", "lokr", "--preprocess_run", "run/manifest.json"],
        {"PRESET": "low_vram"},
    )

    assert snapshot == {
        "output_name": "lokr-low_vram",
        "max_train_steps": 100,
        "preprocess_run": "run/manifest.json",
    }


def test_gui_epoch_only_snapshot_drops_inherited_step_default(monkeypatch):
    import webui.services.config_service as config_service
    from webui.services.task_service import TaskService

    monkeypatch.setattr(
        config_service,
        "merged_gui_variant_preset",
        lambda variant, preset: (
            {"max_train_steps": 1600, "max_train_epochs": 4, "output_name": "demo"},
            {"max_train_steps": "base", "max_train_epochs": "method"},
        ),
    )

    snapshot = TaskService._training_config_snapshot(
        "lora-gui", ["tasks.py", "lora-gui", "lora"], {}
    )

    assert snapshot is not None
    assert "max_train_steps" not in snapshot
    assert snapshot["max_train_epochs"] == 4


def test_list_tasks_page_uses_daemon_total(monkeypatch):
    import webui.services.task_service as task_module
    from webui.services.task_service import TaskService, TaskState

    svc = TaskService()
    jobs = [
        {
            "id": "history-page-2",
            "kind": "command",
            "method": "lora-gui",
            "argv": ["tasks.py", "lora-gui", "lora"],
            "state": "error",
            "rc": 1,
            "status_detail": "process exited",
        }
    ]

    async def list_job_groups_page(**_kwargs):
        return {
            "groups": [svc._physical_job_group(job) for job in jobs],
            "total": 37,
            "offset": 25,
            "limit": 25,
        }

    monkeypatch.setattr(task_module.daemon_client, "list_job_groups_page", list_job_groups_page)
    monkeypatch.setattr(svc, "_ensure_monitors", lambda *_args, **_kwargs: None)

    items, total = _run(svc.list_tasks_page(state="failed", limit=25, offset=25))

    assert total == 37
    assert len(items) == 1
    assert items[0]["task_id"] == "history-page-2"
    assert items[0]["state"] == TaskState.FAILED.value


def test_list_tasks_page_filters_legacy_unfiltered_daemon(monkeypatch):
    """An older daemon may ignore the state query; keep the UI filter correct."""
    import webui.services.task_service as task_module
    from webui.services.task_service import TaskService

    svc = TaskService()
    unfiltered = [
        {
            "id": "legacy-done",
            "kind": "command",
            "method": "lora-gui",
            "argv": ["tasks.py", "lora-gui", "lora"],
            "state": "done",
        },
        {
            "id": "legacy-running",
            "kind": "command",
            "method": "lora-gui",
            "argv": ["tasks.py", "lora-gui", "lora"],
            "state": "running",
        },
    ]
    calls: list[dict] = []

    async def list_job_groups_page(**_kwargs):
        from webui.services.daemon_client import DaemonError

        raise DaemonError("old daemon")

    async def list_jobs_page(**kwargs):
        calls.append(kwargs)
        if kwargs.get("state"):
            return {"jobs": unfiltered, "total": len(unfiltered)}
        return {"jobs": unfiltered, "total": len(unfiltered)}

    monkeypatch.setattr(task_module.daemon_client, "list_job_groups_page", list_job_groups_page)
    monkeypatch.setattr(task_module.daemon_client, "list_jobs_page", list_jobs_page)
    monkeypatch.setattr(svc, "_ensure_monitors", lambda *_args, **_kwargs: None)

    items, total = _run(svc.list_tasks_page(state="active", limit=25, offset=0))

    assert total == 1
    assert [item["task_id"] for item in items] == ["legacy-running"]
    assert calls == [
        {"state": "queued,running", "offset": 0, "limit": 25, "newest_first": True},
    ]


def test_direct_history_access_hydrates_missing_task(monkeypatch):
    import webui.services.task_service as task_module
    from webui.services.task_service import TaskService

    svc = TaskService()

    def get_job_group_sync(_task_id):
        job = {
            "id": "direct-history",
            "kind": "command",
            "method": "preprocess",
            "argv": ["tasks.py", "preprocess"],
            "state": "done",
            "rc": 0,
            "stdout_path": "/path/that/does/not/exist",
        }
        return svc._physical_job_group(job)

    monkeypatch.setattr(task_module.daemon_client, "get_job_group_sync", get_job_group_sync)

    info = svc.get_task_info("direct-history")

    assert info is not None
    assert info["state"] == "success"


def test_daemon_snapshot_persists_epoch_target_without_step_default(monkeypatch, tmp_path):
    from scripts.daemon import config as daemon_config
    from scripts.daemon.manager import JobManager

    monkeypatch.setattr(daemon_config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(daemon_config, "JOBS_DIR", tmp_path / "jobs")

    manager = JobManager()
    job = manager.submit_command(
        label="lora-gui",
        argv=["tasks.py", "lora-gui", "lora"],
        config_snapshot={"max_train_epochs": 4, "output_name": "demo"},
        start=False,
    )

    assert job.target_steps is None
    assert job.target_epochs == 4


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


def test_cancel_keeps_daemon_poller_alive_until_stopped(monkeypatch):
    """A successful stop request is not terminal until daemon teardown ends."""
    import webui.services.task_service as task_module
    from webui.services.task_service import Task, TaskService, TaskState

    svc = TaskService()
    task = Task(
        id="stop-later",
        command="lora",
        args=[],
        state=TaskState.RUNNING,
        job_id="stop-later",
    )
    svc._tasks[task.id] = task
    replies = iter(
        [
            {"state": "running", "pid": 123, "stdout_path": "ignored.log"},
            {"state": "stopped", "rc": 0, "stdout_path": "ignored.log"},
        ]
    )

    async def stop(_job_id):
        return {"job_id": task.id, "state": "running"}

    async def get_job(_job_id):
        return next(replies)

    async def no_sleep(_seconds):
        return None

    async def no_drain(_task):
        return None

    monkeypatch.setattr(task_module.daemon_client, "stop", stop)
    monkeypatch.setattr(task_module.daemon_client, "get_job", get_job)
    monkeypatch.setattr(task_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(svc, "_safe_drain_stdout", no_drain)

    async def exercise():
        assert await svc.cancel_task(task.id) is True
        assert task.state == TaskState.STOPPING
        await svc._poll_daemon_job(task)

    _run(exercise())
    assert task.state == TaskState.CANCELLED
    assert task.exit_code == 0


def test_cancel_queued_task_reaches_terminal_state(monkeypatch):
    """Queued cancellation is visible immediately without a stale poller."""
    import webui.services.task_service as task_module
    from webui.services.task_service import Task, TaskService, TaskState

    svc = TaskService()
    task = Task(
        id="stop-queued",
        command="preprocess",
        args=[],
        state=TaskState.PENDING,
        job_id="stop-queued",
    )
    svc._tasks[task.id] = task

    async def stop(_job_id):
        return {"job_id": task.id, "state": "stopped"}

    async def get_job(_job_id):
        return {"state": "stopped", "rc": None, "stdout_path": "ignored.log"}

    async def no_drain(_task):
        return None

    monkeypatch.setattr(task_module.daemon_client, "stop", stop)
    monkeypatch.setattr(task_module.daemon_client, "get_job", get_job)
    monkeypatch.setattr(svc, "_safe_drain_stdout", no_drain)

    async def exercise():
        assert await svc.cancel_task(task.id) is True
        assert task.state == TaskState.STOPPING
        await svc._poll_daemon_job(task)

    _run(exercise())
    assert task.state == TaskState.CANCELLED


def test_cancel_does_not_resurrect_terminal_task(monkeypatch):
    """A poller completion racing the stop response remains authoritative."""
    import webui.services.task_service as task_module
    from webui.services.task_service import Task, TaskService, TaskState

    svc = TaskService()
    task = Task(
        id="already-done",
        command="lora",
        args=[],
        state=TaskState.RUNNING,
        job_id="already-done",
    )
    svc._tasks[task.id] = task

    async def stop(_job_id):
        task.state = TaskState.SUCCESS
        return {"job_id": task.id, "state": "done"}

    monkeypatch.setattr(task_module.daemon_client, "stop", stop)

    async def exercise():
        assert await svc.cancel_task(task.id) is True

    _run(exercise())
    assert task.state == TaskState.SUCCESS


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
