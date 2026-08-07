"""Deterministic daemon stop/timeout contract tests."""

from __future__ import annotations

import json
import os
import threading
import time

from scripts.daemon import config, proc
from scripts.daemon.jobs import STATE_RUNNING, Job
from scripts.daemon.manager import JobManager


def test_stop_tree_gracefully_force_kills_after_cooperative_deadline(monkeypatch):
    """A non-cooperative process family is terminated, then killed, and the
    return value makes the forced path observable without waiting 30 seconds."""

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self._create_time = 42.0
            self.terminated = False
            self.killed = False

        def create_time(self):
            return self._create_time

        def children(self, recursive=True):
            return []

        def send_signal(self, _signal):
            self.terminated = True

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    parent = FakeProcess(123)
    waits = iter([([parent], [parent]), ([], [parent])])
    monkeypatch.setattr(proc.psutil, "Process", lambda _pid: parent)
    monkeypatch.setattr(proc.psutil, "wait_procs", lambda *_a, **_k: next(waits))
    monkeypatch.setattr(proc.sys, "platform", "win32")

    assert (
        proc.stop_tree_gracefully(
            123, expected_create_time=42.0, grace_seconds=0.01
        )
        is False
    )
    assert parent.terminated is True
    assert parent.killed is True


def test_stop_tree_gracefully_ignores_zombie_after_wait(monkeypatch):
    """An orphaned zombie must not consume the force-kill branch/deadline."""

    class Zombie:
        pid = 123
        status = lambda self: proc.psutil.STATUS_ZOMBIE

        def create_time(self):
            return 42.0

        def children(self, recursive=True):
            return []

        def terminate(self):
            raise AssertionError("zombie should not be terminated")

        def kill(self):
            raise AssertionError("zombie should not be killed")

        def is_running(self):
            return True

    zombie = Zombie()
    monkeypatch.setattr(proc.psutil, "Process", lambda _pid: zombie)
    monkeypatch.setattr(proc.sys, "platform", "win32")
    monkeypatch.setattr(proc.psutil, "wait_procs", lambda *_a, **_k: ([], [zombie]))

    assert (
        proc.stop_tree_gracefully(
            123, expected_create_time=42.0, grace_seconds=0.01
        )
        is True
    )


def test_stop_tree_gracefully_refuses_reused_pid_before_signal(monkeypatch):
    """A stale job record must not signal the process now owning its old PID."""

    class ReusedProcess:
        pid = 123

        def create_time(self):
            return 200.0

        def children(self, recursive=True):
            raise AssertionError("a mismatched root must not be traversed")

        def send_signal(self, _signal):
            raise AssertionError("a reused PID must not receive a signal")

        def terminate(self):
            raise AssertionError("a reused PID must not be terminated")

        def kill(self):
            raise AssertionError("a reused PID must not be killed")

    monkeypatch.setattr(proc.psutil, "Process", lambda _pid: ReusedProcess())
    monkeypatch.setattr(
        proc.os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(
            AssertionError("a reused PID must not reach getpgid")
        ),
    )
    monkeypatch.setattr(proc.sys, "platform", "linux")

    assert (
        proc.stop_tree_gracefully(
            123, expected_create_time=100.0, grace_seconds=0.01
        )
        is True
    )


def test_stop_tree_gracefully_rechecks_identity_before_force_kill(monkeypatch):
    """A PID recycled during the grace period is excluded from force-kill."""

    class ReusedDuringWait:
        pid = 123

        def __init__(self):
            self._create_time = 42.0

        def create_time(self):
            return self._create_time

        def children(self, recursive=True):
            return []

        def status(self):
            return proc.psutil.STATUS_RUNNING

        def is_running(self):
            return True

        def terminate(self):
            raise AssertionError("the replacement process must not be terminated")

        def kill(self):
            raise AssertionError("the replacement process must not be killed")

    parent = ReusedDuringWait()

    def wait_and_reuse(*_args, **_kwargs):
        parent._create_time = 99.0
        return [], [parent]

    monkeypatch.setattr(proc.psutil, "Process", lambda _pid: parent)
    monkeypatch.setattr(proc.psutil, "wait_procs", wait_and_reuse)
    monkeypatch.setattr(proc.sys, "platform", "win32")

    assert (
        proc.stop_tree_gracefully(
            123, expected_create_time=42.0, grace_seconds=0.01
        )
        is True
    )


def test_stop_tree_gracefully_refreshes_descendants_before_force_kill(monkeypatch):
    """A worker spawned during the grace wait must not survive the hard kill."""

    class FakeProcess:
        def __init__(self, pid, create_time):
            self.pid = pid
            self._create_time = create_time
            self.terminated = False
            self.killed = False

        def create_time(self):
            return self._create_time

        def status(self):
            return proc.psutil.STATUS_RUNNING

        def is_running(self):
            return True

        def children(self, recursive=True):
            return []

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    parent = FakeProcess(123, 42.0)
    child = FakeProcess(456, 43.0)
    child_seen = False

    def children(*, recursive=True):
        nonlocal child_seen
        if not child_seen:
            child_seen = True
            return []
        return [child]

    parent.children = children
    processes = {123: parent, 456: child}
    monkeypatch.setattr(proc.psutil, "Process", lambda pid: processes[pid])
    monkeypatch.setattr(
        proc.psutil,
        "wait_procs",
        lambda processes, **_kwargs: ([], list(processes)),
    )
    monkeypatch.setattr(proc.sys, "platform", "win32")

    assert (
        proc.stop_tree_gracefully(
            123, expected_create_time=42.0, grace_seconds=0.01
        )
        is False
    )
    assert child.terminated is True
    assert child.killed is True


def test_kill_tree_refuses_reused_pid(monkeypatch):
    """Immediate force-kill paths use the same persisted identity contract."""

    class ReusedProcess:
        pid = 123

        def create_time(self):
            return 200.0

        def children(self, recursive=True):
            raise AssertionError("a mismatched root must not be traversed")

        def terminate(self):
            raise AssertionError("a reused PID must not be terminated")

        def kill(self):
            raise AssertionError("a reused PID must not be killed")

    monkeypatch.setattr(proc.psutil, "Process", lambda _pid: ReusedProcess())

    proc.kill_tree(123, expected_create_time=100.0, grace_seconds=0.01)


def test_manager_writes_windows_stop_file_and_marks_forced_timeout(tmp_path, monkeypatch):
    """The Windows path has no reliable console signal, so the manager writes
    the per-job stop-file first. A timeout is persisted as ``forced_stop``."""

    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "STOP_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(proc.sys, "platform", "win32")
    stop_call = {}

    def fake_stop(_pid, *, expected_create_time, grace_seconds):
        stop_call.update(
            pid=_pid,
            expected_create_time=expected_create_time,
            grace_seconds=grace_seconds,
        )
        return False

    monkeypatch.setattr(proc, "stop_tree_gracefully", fake_stop)

    job = Job(
        id="windows-stop",
        method="lora",
        preset="default",
        state=STATE_RUNNING,
        pid=123,
        create_time=42.0,
        stop_requested=True,
    )
    job.progress_path = str(job.dir / "progress.jsonl")
    job.persist()

    manager = JobManager()
    monkeypatch.setattr(manager, "_job_runs_train", lambda _job: True)
    manager._stop_job_tree(job)

    stop_file = job.dir / "stop.requested"
    assert stop_file.read_text(encoding="utf-8") == "stop\n"
    assert stop_call == {
        "pid": 123,
        "expected_create_time": 42.0,
        "grace_seconds": 0.01,
    }
    assert job.forced_stop is True
    assert job.status_detail == "stop timeout; process tree force-killed"
    persisted = json.loads((job.dir / "job.json").read_text(encoding="utf-8"))
    assert Job.from_dict(persisted).forced_stop is True


def test_manager_retries_transient_windows_stop_file_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "STOP_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(proc.sys, "platform", "win32")
    monkeypatch.setattr(
        proc,
        "stop_tree_gracefully",
        lambda _pid, *, expected_create_time, grace_seconds: True,
    )
    monkeypatch.setattr("scripts.daemon.manager.time.sleep", lambda _delay: None)

    real_replace = os.replace
    attempts = 0

    def fail_first_replace(source, target):
        nonlocal attempts
        if str(target).endswith("stop.requested"):
            attempts += 1
            if attempts == 1:
                raise PermissionError("transient scanner lock")
        return real_replace(source, target)

    monkeypatch.setattr("scripts.daemon.manager.os.replace", fail_first_replace)

    job = Job(
        id="windows-stop-retry",
        method="lora",
        preset="default",
        state=STATE_RUNNING,
        pid=123,
        create_time=42.0,
        stop_requested=True,
    )
    job.progress_path = str(job.dir / "progress.jsonl")
    job.persist()

    manager = JobManager()
    monkeypatch.setattr(manager, "_job_runs_train", lambda _job: True)
    manager._stop_job_tree(job)

    assert attempts == 2
    assert (job.dir / "stop.requested").read_text(encoding="utf-8") == "stop\n"


def test_monitor_waits_for_stop_worker_before_finalizing(tmp_path, monkeypatch):
    """A dead launcher must not release the queue ahead of live descendants."""

    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "STOP_GRACE_SECONDS", 0.01)
    job = Job(
        id="wait-stop-worker",
        method="lora",
        preset="default",
        state=STATE_RUNNING,
        pid=123,
        create_time=42.0,
        stop_requested=True,
    )
    manager = JobManager()
    stop_done = threading.Event()
    manager._stop_events[job.id] = stop_done
    finalized: list[bool] = []
    monkeypatch.setattr(
        manager,
        "_finalize_from_exit",
        lambda _job, _popen: finalized.append(stop_done.is_set()),
    )

    class Exited:
        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monitor = threading.Thread(
        target=manager._monitor, kwargs={"job": job, "popen": Exited()}
    )
    monitor.start()
    time.sleep(0.05)
    assert finalized == []
    stop_done.set()
    monitor.join(timeout=2)

    assert finalized == [True]
    assert job.id not in manager._stop_events
