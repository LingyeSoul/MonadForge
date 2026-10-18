"""History deletion cannot hide surviving records or race continuation input."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from anima_daemon import config, manager as manager_module
from anima_daemon.jobs import Job
from anima_daemon.manager import JobManager
from anima_daemon.server import _Handler


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    manager = JobManager()
    for job in (
        Job("root", "lora", "default", state="error", root_job_id="root"),
        Job(
            "child",
            "lora",
            "default",
            state="done",
            root_job_id="root",
            parent_job_id="root",
            attempt_index=1,
        ),
    ):
        manager._jobs[job.id] = job
        job.persist()
    return manager


def test_failed_individual_delete_keeps_history(manager, monkeypatch):
    def fail(_path, **_kwargs):
        raise PermissionError("locked history")

    monkeypatch.setattr(manager_module.shutil, "rmtree", fail)
    with pytest.raises(PermissionError):
        manager.delete("root")
    assert manager.get("root") is not None
    assert manager.job_group("root")["attempt_count"] == 2


def test_partial_group_delete_keeps_surviving_lineage_visible(manager, monkeypatch):
    original = manager_module.shutil.rmtree

    def fail_child(path, **kwargs):
        if path.name == "child":
            raise PermissionError("locked child")
        return original(path, **kwargs)

    monkeypatch.setattr(manager_module.shutil, "rmtree", fail_child)
    with pytest.raises(PermissionError):
        manager.delete_group("root")
    assert manager.get("root") is None
    groups, total = manager.list_job_groups_filtered()
    assert total == 1
    assert groups[0]["id"] == "root"
    assert groups[0]["current_job_id"] == "child"
    assert manager.delete_group("missing") is None


@pytest.mark.parametrize("group", [False, True])
def test_deletion_refuses_root_with_continuation_in_progress(manager, group):
    manager._resuming_roots.add("root")
    assert not (manager.delete_group("child") if group else manager.delete("root"))
    assert all(job.dir.exists() for job in manager._jobs.values())
    assert manager.job_group("root")["attempt_count"] == 2


@pytest.mark.parametrize("group", [False, True])
def test_http_delete_reports_continuation_conflict(manager, group):
    manager._resuming_roots.add("root")
    handler = _Handler.__new__(_Handler)
    handler.headers = {}
    handler.server = SimpleNamespace(manager=manager)
    handler.path = "/job-groups/root" if group else "/jobs/root"
    replies = []
    handler._send_json = lambda payload, status=200: replies.append((status, payload))
    handler.do_DELETE()
    assert replies[0][0] == 409
    assert "continuation" in replies[0][1]["error"]
