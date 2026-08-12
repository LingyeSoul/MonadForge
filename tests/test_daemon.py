"""Phase 1 training daemon: arg builder, job persistence, liveness, and an
end-to-end serial-queue run over the real HTTP surface with fake training
subprocesses.

The fake "trainer" is a tiny ``python -c`` script that writes a well-formed
Phase-0 ``progress.jsonl`` and exits — exercising the spawn → tail → finalize
path without launching torch/accelerate.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import psutil
import pytest

from scripts.daemon import config, gpu, jobs, proc

# Bound at import time so tests that monkeypatch the client module's attribute
# can still build a real (dead) client without recursing into their own patch.
from scripts.daemon.client import DaemonClient as _RealDaemonClient
from scripts.daemon.manager import JobManager
from scripts.daemon.mcp import MCPServer
from scripts.daemon.server import serve
from scripts.tasks._common import build_method_args


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------


def test_build_method_args_basic():
    args = build_method_args("lora", preset="default")
    assert args == ["--method", "lora", "--preset", "default"]


def test_build_method_args_subdir_artist_profile_and_extra():
    args = build_method_args(
        "tlora",
        preset="low_vram",
        methods_subdir="gui-methods",
        extra=["--network_dim", "32"],
        artist="alice",
        profile_steps="3-5",
    )
    assert args[:6] == [
        "--method",
        "tlora",
        "--preset",
        "low_vram",
        "--methods_subdir",
        "gui-methods",
    ]
    assert "--artist_filter" in args and "alice" in args
    assert "--profile_steps" in args and "3-5" in args
    assert args[-2:] == ["--network_dim", "32"]


def test_build_method_args_respects_explicit_overrides():
    # caller already passed --artist_filter in extra → builder must not duplicate
    args = build_method_args(
        "lora", preset="default", extra=["--artist_filter", "bob"], artist="alice"
    )
    assert args.count("--artist_filter") == 1
    assert "alice" not in args


def test_job_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    job = jobs.Job(
        id="j1", method="lora", preset="default", overrides={"network_dim": 16}
    )
    job.progress_path = str(job.dir / "progress.jsonl")
    job.persist()
    loaded = jobs.load_all()
    assert "j1" in loaded
    assert loaded["j1"].method == "lora"
    assert loaded["j1"].overrides == {"network_dim": 16}


def test_filtered_history_can_page_newest_first():
    manager = JobManager.__new__(JobManager)
    manager._lock = threading.RLock()
    manager._jobs = {
        "old": jobs.Job(id="old", method="old", preset="", submitted_at=1.0),
        "new": jobs.Job(id="new", method="new", preset="", submitted_at=2.0),
    }

    page, total = manager.list_jobs_filtered(offset=0, limit=1, newest_first=True)

    assert total == 2
    assert [job.id for job in page] == ["new"]


def test_liveness_pid_create_time():
    me = os.getpid()
    ct = proc.create_time(me)
    assert proc.is_alive(me, ct)
    # wrong create_time → treated as a reused PID, not our process
    assert not proc.is_alive(me, (ct or 0) + 10_000)
    # a definitely-dead pid
    assert not proc.is_alive(2_147_483_000, 123.0)


# --------------------------------------------------------------------------
# end-to-end over the HTTP surface
# --------------------------------------------------------------------------

_FAKE_TRAINER = r"""
import _thread, json, os, signal, sys, threading, time
path, dur = sys.argv[1], float(sys.argv[2])

def interrupt(_signum, _frame):
    raise KeyboardInterrupt

for name in ("SIGTERM", "SIGBREAK"):
    signum = getattr(signal, name, None)
    if signum is not None:
        signal.signal(signum, interrupt)

stop_file = os.environ.get("ANIMA_DAEMON_STOP_FILE")
if os.name == "nt" and stop_file:
    def watch_stop_file():
        while not os.path.exists(stop_file):
            time.sleep(0.05)
        _thread.interrupt_main()

    threading.Thread(target=watch_stop_file, daemon=True).start()

with open(path, "w", buffering=1) as f:
    f.write(json.dumps({"ev": "run_start", "ts": 0.0}) + "\n")
    f.write(json.dumps({"ev": "step", "ts": 0.1, "global_step": 1, "loss": 0.5}) + "\n")
    try:
        deadline = time.time() + dur
        while time.time() < deadline:
            time.sleep(min(0.05, deadline - time.time()))
    except KeyboardInterrupt:
        f.write(json.dumps({"ev": "run_end", "ts": 0.2, "status": "stopped", "final_step": 1}) + "\n")
        raise
    else:
        f.write(json.dumps({"ev": "ckpt", "ts": dur, "global_step": 1, "path": "/tmp/fake.safetensors"}) + "\n")
        f.write(json.dumps({"ev": "run_end", "ts": dur, "status": "ok", "final_step": 1}) + "\n")
"""

# A fake trainer that dies before writing ``run_end`` with a chosen exit code.
# Exercises the ``_finalize_from_exit`` no-run_end branch (the real trainer
# crashes here — e.g. a CUDA SIGABRT — so the only terminal signal is the
# nonzero rc the manager must forward to the WebUI as ``Job.rc``).
_FAKE_TRAINER_CRASH = r"""
import sys
path, rc = sys.argv[1], int(sys.argv[2])
with open(path, "w", buffering=1) as f:
    f.write('{"ev": "run_start", "ts": 0.0}\n')
    f.write('{"ev": "step", "ts": 0.1, "global_step": 1, "loss": 0.5}\n')
sys.exit(rc)
"""

# A cooperative trainer used by the stop contract test. It treats either the
# daemon stop-file or SIGTERM/SIGINT as a request, writes a complete semantic
# state record, and emits run_end(stopped) before exiting cleanly.
_FAKE_TRAINER_COOPERATIVE = r"""
import json, os, signal, sys, time
progress, state_dir, dur = sys.argv[1], sys.argv[2], float(sys.argv[3])
requested = False
def request(*_args):
    global requested
    requested = True
for name in ('SIGTERM', 'SIGINT'):
    sig = getattr(signal, name, None)
    if sig is not None:
        signal.signal(sig, request)
os.makedirs(state_dir, exist_ok=True)
with open(progress, 'w', buffering=1) as f:
    f.write(json.dumps({'ev': 'run_start', 'ts': 0.0}) + '\n')
    f.write(json.dumps({'ev': 'step', 'ts': 0.1, 'global_step': 3}) + '\n')
    deadline = time.time() + dur
    while time.time() < deadline and not requested:
        stop_file = os.environ.get('ANIMA_DAEMON_STOP_FILE')
        if stop_file and os.path.exists(stop_file):
            requested = True
        time.sleep(0.02)
    if requested:
        with open(os.path.join(state_dir, 'train_state.json'), 'w') as sf:
            json.dump({'global_step': 3, 'current_epoch': 1, 'micro_batch_offset': 0}, sf)
        with open(os.path.join(state_dir, '.complete'), 'w') as marker:
            marker.write('ok\n')
        f.write(json.dumps({'ev': 'run_end', 'ts': 1.0, 'status': 'stopped', 'final_step': 3}) + '\n')
        sys.exit(0)
    f.write(json.dumps({'ev': 'run_end', 'ts': dur, 'status': 'ok', 'final_step': 3}) + '\n')
"""


def _fake_build_cmd_crash(self, job):
    rc = int(job.overrides.get("crash_rc", 42))
    cmd = [sys.executable, "-c", _FAKE_TRAINER_CRASH, job.progress_path, str(rc)]
    return cmd, os.environ.copy()


def _fake_build_cmd(self, job):
    if job.overrides.get("cooperative"):
        cmd = [
            sys.executable,
            "-c",
            _FAKE_TRAINER_COOPERATIVE,
            job.progress_path,
            str(job.dir / "resume-state"),
            str(job.overrides.get("duration", 60.0)),
        ]
        return cmd, os.environ.copy()
    dur = float(job.overrides.get("duration", 1.0))
    cmd = [sys.executable, "-c", _FAKE_TRAINER, job.progress_path, str(dur)]
    return cmd, os.environ.copy()


def _wait_until(pred, timeout=20.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    """An in-process daemon (manager + HTTP server) with fake training cmds."""
    from scripts.daemon import client

    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "PIDFILE", tmp_path / "daemon.json")
    monkeypatch.setattr(config, "DAEMON_LOG", tmp_path / "daemon.log")
    monkeypatch.setattr(config, "STOP_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(JobManager, "_build_cmd", _fake_build_cmd)
    # Fake trainers don't touch the GPU; stub the guard so the test doesn't
    # block on whatever real workload happens to hold VRAM on the host.
    monkeypatch.setattr(gpu, "gpu_pids", lambda: set())

    mgr = JobManager()
    mgr.start()
    srv = serve(mgr, port=0)
    t = threading.Thread(
        target=srv.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
    )
    t.start()
    port = srv.server_address[1]
    cl = client.DaemonClient(port)
    assert _wait_until(lambda: cl.health() is not None, timeout=5)
    try:
        yield cl, mgr
    finally:
        srv.request_shutdown(True)
        srv.server_close()


def test_health(daemon):
    cl, _ = daemon
    h = cl.health()
    assert h["ok"] is True
    assert h["active_job"] is None


def test_serial_queue(daemon):
    cl, _ = daemon
    j1 = cl.submit(method="lora", overrides={"duration": 1.0})["job_id"]
    j2 = cl.submit(method="lora", overrides={"duration": 1.0})["job_id"]

    assert _wait_until(lambda: cl.get(j1)["state"] == "done", timeout=15)
    assert _wait_until(lambda: cl.get(j2)["state"] == "done", timeout=15)

    g1, g2 = cl.get(j1), cl.get(j2)
    # serial: the second job can't start before the first ends
    assert g2["started_at"] >= g1["ended_at"] - 0.5
    # ckpt path picked up from the progress stream
    assert g1["ckpt_path"] == "/tmp/fake.safetensors"
    assert g1["latest"]["ev"] == "run_end"


def _isolate_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "PIDFILE", tmp_path / "daemon.json")
    monkeypatch.setattr(config, "DAEMON_LOG", tmp_path / "daemon.log")
    monkeypatch.setattr(gpu, "gpu_pids", lambda: set())


def test_worker_survives_build_exception(tmp_path, monkeypatch):
    """A job whose _build_cmd raises must fail ERROR without killing the worker;
    the next queued job still runs. Regression for the silent-worker-death hang
    that left every later job stuck `queued` forever with no error (the stall
    watchdog only guards *running* jobs, so a never-launched job spins forever)."""
    _isolate_state(tmp_path, monkeypatch)

    def build_or_boom(self, job):
        if job.method == "boom":
            raise RuntimeError("kaboom while building the command")
        return _fake_build_cmd(self, job)

    monkeypatch.setattr(JobManager, "_build_cmd", build_or_boom)
    mgr = JobManager()
    mgr.start()
    try:
        bad = mgr.submit(
            method="boom", preset="default", methods_subdir=None, start=True
        )
        good = mgr.submit(
            method="lora",
            preset="default",
            methods_subdir=None,
            overrides={"duration": 0.2},
            start=True,
        )
        assert _wait_until(lambda: mgr.get(bad.id).state == "error", timeout=10)
        assert _wait_until(lambda: mgr.get(good.id).state == "done", timeout=15)
        assert "launch failed" in (mgr.get(bad.id).error or "")
        assert mgr.worker_alive() is True
    finally:
        mgr.shutdown(kill_jobs=False)


def test_worker_survives_prelaunch_exception(tmp_path, monkeypatch):
    """An exception *before* launch (here in the GPU guard, outside
    _launch_and_monitor's own try) is caught by the worker-loop crash guard:
    the job fails ERROR and the worker keeps draining the queue."""
    _isolate_state(tmp_path, monkeypatch)
    monkeypatch.setattr(JobManager, "_build_cmd", _fake_build_cmd)

    def boom_guard(self, job, **kw):
        if job.method == "boom":
            raise RuntimeError("guard blew up")

    monkeypatch.setattr(JobManager, "_gpu_guard", boom_guard)
    mgr = JobManager()
    mgr.start()
    try:
        bad = mgr.submit(
            method="boom", preset="default", methods_subdir=None, start=True
        )
        good = mgr.submit(
            method="lora",
            preset="default",
            methods_subdir=None,
            overrides={"duration": 0.2},
            start=True,
        )
        assert _wait_until(lambda: mgr.get(bad.id).state == "error", timeout=10)
        assert _wait_until(lambda: mgr.get(good.id).state == "done", timeout=15)
        assert "unexpected error" in (mgr.get(bad.id).error or "")
        assert mgr.worker_alive() is True
    finally:
        mgr.shutdown(kill_jobs=False)


def test_cli_queue_submits_instead_of_launching(daemon, monkeypatch):
    """`train(..., extra=["--queue"])` enqueues on the daemon and returns,
    rather than calling accelerate_launch inline."""
    from scripts.tasks import _common

    cl, _ = daemon
    # Point the CLI's daemon client at the in-process test daemon (train() does
    # a local `from scripts.daemon import client` then calls ensure_daemon).
    import scripts.daemon.client as daemon_client

    monkeypatch.setattr(daemon_client, "ensure_daemon", lambda **kw: cl)
    launched = []
    monkeypatch.setattr(_common, "accelerate_launch", lambda *a: launched.append(a))

    _common.train("tlora", ["--queue"], methods_subdir="gui-methods")

    assert launched == []  # inline path skipped
    jobs_list = cl.list_jobs()
    assert len(jobs_list) == 1
    job = jobs_list[0]
    assert job["method"] == "tlora"
    assert job["methods_subdir"] == "gui-methods"
    assert "--queue" not in job["extra"]


def test_cli_queue_folds_artist_into_extra(daemon, monkeypatch):
    """ARTIST env is folded into the queued job's extra (the daemon's own
    build_method_args doesn't read env vars)."""
    from scripts.tasks import _common

    cl, _ = daemon
    import scripts.daemon.client as daemon_client

    monkeypatch.setattr(daemon_client, "ensure_daemon", lambda **kw: cl)
    monkeypatch.setattr(_common, "accelerate_launch", lambda *a: None)
    monkeypatch.setenv("ARTIST", "alice")

    _common.train("lora", ["--queue"])

    job = cl.list_jobs()[-1]
    assert "--artist_filter" in job["extra"]
    assert "alice" in job["extra"]


def test_stop_running_job(daemon):
    cl, mgr = daemon
    jid = cl.submit(method="lora", overrides={"duration": 60.0})["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "running", timeout=10)
    pid = cl.get(jid)["pid"]
    assert pid and psutil.pid_exists(pid)

    cl.stop(jid)
    assert _wait_until(lambda: cl.get(jid)["state"] == "stopped", timeout=10)
    # tree torn down → the training pid is gone
    assert _wait_until(lambda: not psutil.pid_exists(pid), timeout=5)


def test_stop_queued_job_finalizes_immediately(daemon):
    """Cancelling a job that's still queued behind a running one finalizes it
    *now* (not lazily when the worker eventually dequeues it), so a UI watching
    the job list sees it leave the queue right away."""
    cl, _ = daemon
    # j1 holds the worker for a while; j2 stays queued behind it.
    j1 = cl.submit(method="lora", overrides={"duration": 60.0})["job_id"]
    j2 = cl.submit(method="lora", overrides={"duration": 60.0})["job_id"]
    assert _wait_until(lambda: cl.get(j1)["state"] == "running", timeout=10)
    assert cl.get(j2)["state"] == "queued"

    cl.stop(j2)
    # Finalized immediately while j1 is still running — no need to wait for the
    # worker to reach j2.
    assert _wait_until(lambda: cl.get(j2)["state"] == "stopped", timeout=3)
    assert cl.get(j1)["state"] == "running"  # the running job is untouched

    # The stale FIFO entry is harmless: when the worker eventually dequeues j2's
    # id it skips it (state != queued), never relaunching it.
    cl.stop(j1)
    assert _wait_until(lambda: cl.get(j1)["state"] == "stopped", timeout=10)
    time.sleep(0.5)
    assert cl.get(j2)["state"] == "stopped"


def test_done_job_carries_zero_rc(daemon):
    """A clean ``run_end:ok`` job surfaces its real subprocess exit code (0)
    as ``Job.rc`` — the WebUI's ``Task.exit_code`` mirrors the old
    direct-subprocess design (which got it from ``process.wait()``). Before
    the rc field was added, the WebUI read ``info.get("rc")`` and always saw
    ``None``, masking whether the trainer actually exited cleanly."""
    cl, _ = daemon
    jid = cl.submit(method="lora", overrides={"duration": 0.2})["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "done", timeout=15)
    assert cl.get(jid)["rc"] == 0


def test_late_stop_cannot_override_run_end_ok(tmp_path, monkeypatch):
    """A stop click racing process reaping must not turn run_end:ok into stopped."""
    _isolate_state(tmp_path, monkeypatch)
    job = jobs.Job(
        id="late-stop-ok",
        method="lora",
        preset="default",
        state=jobs.STATE_RUNNING,
        stop_requested=True,
        stop_requested_at=time.time(),
        status_detail="stopping",
        forced_stop=True,
    )
    job.progress_path = str(job.dir / "progress.jsonl")
    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "progress.jsonl").write_text(
        json.dumps({"ev": "run_end", "status": "ok", "final_step": 7}) + "\n",
        encoding="utf-8",
    )

    class Exited:
        def poll(self):
            return 0

    JobManager()._finalize_from_exit(job, Exited())

    assert job.state == jobs.STATE_DONE
    assert job.rc == 0
    assert job.stop_requested is False
    assert job.stop_requested_at is None
    assert job.forced_stop is False
    assert job.status_detail is None


def test_crashed_job_carries_real_rc(daemon, monkeypatch):
    """A trainer that dies before ``run_end`` (CUDA SIGABRT / segfault) leaves
    only a nonzero exit code. The manager must forward that exact rc — not a
    synthesized -1 — so the WebUI sees the real failure code (the old
    direct-subprocess design surfaced it faithfully)."""
    monkeypatch.setattr(JobManager, "_build_cmd", _fake_build_cmd_crash)
    cl, _ = daemon
    jid = cl.submit(method="lora", overrides={"crash_rc": 42})["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "error", timeout=15)
    assert cl.get(jid)["rc"] == 42


def test_stopped_job_carries_kill_rc(daemon):
    """A stopped job keeps the subprocess rc, including a clean cooperative 0."""
    cl, _ = daemon
    jid = cl.submit(method="lora", overrides={"duration": 60.0})["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "running", timeout=10)
    assert _wait_until(lambda: cl.get(jid)["latest"] is not None, timeout=10)
    cl.stop(jid)
    assert _wait_until(lambda: cl.get(jid)["state"] == "stopped", timeout=10)
    # The new stop path first requests a checkpoint-capable exit. Depending on
    # which process in the detached tree observes the signal, rc can be 0 or a
    # platform signal code; it must still be captured rather than synthesized.
    rec = cl.get(jid)
    assert rec["rc"] is not None
    assert rec["latest"]["ev"] == "run_end"
    assert rec["latest"]["status"] == "stopped"


def test_cooperative_stop_persists_state_and_queue_continues(daemon):
    cl, _ = daemon
    first = cl.submit(
        method="lora", overrides={"cooperative": True, "duration": 60.0}
    )["job_id"]
    second = cl.submit(method="lora", overrides={"duration": 0.2})["job_id"]
    assert _wait_until(lambda: cl.get(first)["state"] == "running", timeout=10)

    cl.stop(first)
    assert _wait_until(lambda: cl.get(first)["state"] == "stopped", timeout=10)
    record = cl.get(first)
    assert record["status_detail"] == "cooperative stop completed"
    assert record["rc"] == 0
    state_file = (
        config.JOBS_DIR / first / "resume-state" / "train_state.json"
    )
    assert state_file.is_file()
    events = [
        json.loads(line)
        for line in (config.JOBS_DIR / first / "progress.jsonl").read_text().splitlines()
    ]
    assert events[-1]["ev"] == "run_end"
    assert events[-1]["status"] == "stopped"

    assert _wait_until(lambda: cl.get(second)["state"] == "done", timeout=15)


def test_queue_hold_then_start(daemon):
    """A job submitted with ``start=False`` is enqueued but *held* (the queue is
    paused — health reflects it), and only runs once ``start_queue`` resumes it.
    This is the GUI "add to queue, don't start now" → "Start Queue" flow."""
    cl, _ = daemon
    jid = cl.submit(method="lora", overrides={"duration": 1.0}, start=False)["job_id"]

    assert cl.health()["paused"] is True
    # Held: it stays queued and does not start on its own.
    assert _wait_until(lambda: cl.get(jid)["state"] == "queued", timeout=2)
    time.sleep(0.7)
    assert cl.get(jid)["state"] == "queued"  # still not launched

    cl.start_queue()
    assert cl.health()["paused"] is False
    assert _wait_until(lambda: cl.get(jid)["state"] == "done", timeout=15)


def test_queue_start_true_flushes_held_backlog(daemon):
    """``start=True`` (the main Train/Run button) resumes a paused queue, so a
    job held earlier via ``start=False`` runs too."""
    cl, _ = daemon
    held = cl.submit(method="lora", overrides={"duration": 1.0}, start=False)["job_id"]
    assert cl.health()["paused"] is True

    run_now = cl.submit(method="lora", overrides={"duration": 1.0}, start=True)[
        "job_id"
    ]
    assert cl.health()["paused"] is False
    # Both drain in FIFO order once the gate opens.
    assert _wait_until(lambda: cl.get(held)["state"] == "done", timeout=15)
    assert _wait_until(lambda: cl.get(run_now)["state"] == "done", timeout=15)
    assert cl.get(run_now)["started_at"] >= cl.get(held)["ended_at"] - 0.5


def test_add_to_queue_while_running_auto_advances(daemon):
    """Regression: ``start=False`` ("add to queue") must NOT pause a queue that's
    already playing. Adding a job while another was running used to clear the
    global run gate, so the new job stalled ``queued`` the moment the running one
    finished — the GUI's "infinite loading" report. It should auto-advance on its
    own (cassette-tape behaviour), with no Start Queue press."""
    cl, _ = daemon
    running = cl.submit(method="lora", overrides={"duration": 3.0}, start=True)[
        "job_id"
    ]
    assert _wait_until(lambda: cl.get(running)["state"] == "running", timeout=10)

    # Add-to-queue while a job is running: the gate must stay open.
    queued = cl.submit(method="lora", overrides={"duration": 1.0}, start=False)[
        "job_id"
    ]
    assert cl.health()["paused"] is False

    # Both drain without anyone pressing Start Queue, in FIFO order.
    assert _wait_until(lambda: cl.get(running)["state"] == "done", timeout=20)
    assert _wait_until(lambda: cl.get(queued)["state"] == "done", timeout=20)
    assert cl.get(queued)["started_at"] >= cl.get(running)["ended_at"] - 0.5


def test_pause_does_not_interrupt_running_job(daemon):
    """Pausing the queue holds the *next* launch but never stops a job already
    running."""
    cl, _ = daemon
    running = cl.submit(method="lora", overrides={"duration": 60.0}, start=True)[
        "job_id"
    ]
    queued = cl.submit(method="lora", overrides={"duration": 1.0})["job_id"]
    assert _wait_until(lambda: cl.get(running)["state"] == "running", timeout=10)

    cl.pause_queue()
    assert cl.health()["paused"] is True
    assert cl.get(running)["state"] == "running"  # untouched

    cl.stop(running)
    assert _wait_until(lambda: cl.get(running)["state"] == "stopped", timeout=10)
    # The queued one stays held while paused — it must not advance.
    time.sleep(0.7)
    assert cl.get(queued)["state"] == "queued"
    cl.start_queue()
    assert _wait_until(lambda: cl.get(queued)["state"] == "done", timeout=15)


def test_reconcile_orphan_requeue_adopt(tmp_path, monkeypatch):
    """Boot sweep: dead `running` → orphaned error; `queued` → re-enqueued;
    live `running` → adopted for monitoring."""
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")

    # a `running` job whose process died while the daemon was down
    dead = jobs.Job(
        id="dead",
        method="lora",
        preset="default",
        state=jobs.STATE_RUNNING,
        pid=2_147_483_000,
        create_time=1.0,
    )
    dead.progress_path = str(dead.dir / "progress.jsonl")
    dead.persist()

    # a `queued` job that never started
    pend = jobs.Job(id="pend", method="lora", preset="default", state=jobs.STATE_QUEUED)
    pend.persist()

    # a `running` job that's actually alive (use this test process as the pid)
    me = os.getpid()
    live = jobs.Job(
        id="live",
        method="lora",
        preset="default",
        state=jobs.STATE_RUNNING,
        pid=me,
        create_time=proc.create_time(me),
    )
    live.persist()

    mgr = JobManager()
    mgr._reconcile()  # sweep without starting the worker

    assert mgr.get("dead").state == jobs.STATE_ERROR
    assert mgr.get("dead").status_detail == "orphaned"
    assert mgr._queue.get_nowait() == "pend"  # re-enqueued
    assert "live" in mgr._adopt  # re-attached for monitoring


def test_resume_selects_newest_complete_state_with_matching_run_signatures(
    tmp_path, monkeypatch
):
    from library.training.state import build_train_state, write_complete_marker

    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "daemon")
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "daemon" / "jobs")
    output_dir = tmp_path / "ckpt"
    output_root = output_dir / "unit"
    snapshot = tmp_path / "source.snapshot.toml"
    snapshot.write_text(
        f'output_dir = "{output_dir.as_posix()}"\n'
        'output_name = "unit"\n'
        "max_train_steps = 100\n",
        encoding="utf-8",
    )

    def write_state(name, step, config_signature, dataset_signature, *, complete=True):
        path = output_root / name
        path.mkdir(parents=True)
        (path / "train_state.json").write_text(
            json.dumps(
                build_train_state(
                    global_step=step,
                    config_signature=config_signature,
                    dataset_signature=dataset_signature,
                    job_id="source",
                    root_job_id="source",
                )
            ),
            encoding="utf-8",
        )
        if complete:
            write_complete_marker(path)
        return path

    matching = write_state("unit-rolling-state", 75, "cfg-a", "data-a")
    write_state("unit-checkpoint-state", 90, "cfg-other", "data-a")
    write_state("unit-interrupted-state", 95, "cfg-a", "data-a", complete=False)

    source = jobs.Job(
        id="source",
        method="lora",
        preset="default",
        state=jobs.STATE_ERROR,
        config_file=str(snapshot),
        progress_path=str(tmp_path / "source.progress.jsonl"),
        target_steps=100,
    )
    with open(source.progress_path, "w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "ev": "run_start",
                    "config_signature": "cfg-a",
                    "dataset_signature": "data-a",
                }
            )
            + "\n"
        )

    manager = JobManager()
    manager._jobs[source.id] = source
    manager._refresh_recovery_metadata(source)
    resumed = manager.resume_job(source.id)

    assert source.config_signature == "cfg-a"
    assert source.dataset_signature == "data-a"
    assert source.recovery_state == str(matching)
    assert source.recovery_step == 75
    assert resumed is not None
    assert resumed.recovery_state == str(matching)
    assert resumed.recovery_step == 75
    assert resumed.extra[-2:] == ["--resume", str(matching)]
    assert resumed.root_job_id == source.id
    assert resumed.parent_job_id == source.id
    assert resumed.attempt_index == 1
    group = manager.job_group(source.id)
    assert group is not None
    assert group["id"] == source.id
    assert group["current_job_id"] == resumed.id
    assert group["attempt_count"] == 2

    with pytest.raises(ValueError, match="only the latest attempt"):
        manager.resume_job(source.id)


def test_resume_discovers_raw_output_name_inside_safe_output_root(tmp_path, monkeypatch):
    from library.training.state import build_train_state, write_complete_marker

    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "daemon")
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "daemon" / "jobs")
    output_dir = tmp_path / "ckpt"
    output_root = output_dir / "Tlora-_artist"
    state_dir = output_root / "Tlora-@artist-checkpoint-state"
    state_dir.mkdir(parents=True)
    (state_dir / "train_state.json").write_text(
        json.dumps(
            build_train_state(
                global_step=840,
                config_signature="cfg-special",
                dataset_signature="data-special",
                job_id="special",
                root_job_id="special",
            )
        ),
        encoding="utf-8",
    )
    write_complete_marker(state_dir)
    snapshot = tmp_path / "special.snapshot.toml"
    snapshot.write_text(
        f'output_dir = "{output_dir.as_posix()}"\n'
        'output_name = "Tlora-@artist"\n'
        "max_train_epochs = 12\n",
        encoding="utf-8",
    )
    job = jobs.Job(
        id="special",
        method="lora-gui",
        preset="",
        kind="command",
        argv=["tasks.py", "lora-gui", "tlora"],
        state=jobs.STATE_ERROR,
        config_file=str(snapshot),
        config_signature="cfg-special",
        dataset_signature="data-special",
    )

    candidates = JobManager._state_candidates(job)

    assert candidates == [(840, state_dir)]


def test_recovery_state_requires_matching_logical_root_owner(tmp_path, monkeypatch):
    from library.training.state import build_train_state, write_complete_marker

    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "daemon")
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "daemon" / "jobs")
    output_dir = tmp_path / "ckpt"
    output_root = output_dir / "unit"
    snapshot = tmp_path / "source.snapshot.toml"
    snapshot.write_text(
        f'output_dir = "{output_dir.as_posix()}"\n'
        'output_name = "unit"\n',
        encoding="utf-8",
    )

    def write_state(name: str, **ownership):
        path = output_root / name
        path.mkdir(parents=True)
        (path / "train_state.json").write_text(
            json.dumps(
                build_train_state(
                    global_step=12,
                    config_signature="cfg-a",
                    dataset_signature="data-a",
                    **ownership,
                )
            ),
            encoding="utf-8",
        )
        write_complete_marker(path)
        return path

    mismatched = write_state(
        "unit-interrupted-state",
        job_id="other-attempt",
        root_job_id="other-root",
    )
    unowned = write_state("unit-rolling-state")
    source = jobs.Job(
        id="source",
        method="lora",
        preset="default",
        state=jobs.STATE_ERROR,
        root_job_id="source",
        config_file=str(snapshot),
        config_signature="cfg-a",
        dataset_signature="data-a",
    )

    assert JobManager._state_candidates(source) == []
    assert mismatched.is_dir()
    assert unowned.is_dir()


def test_concurrent_resume_reserves_one_attempt_per_logical_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "daemon")
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "daemon" / "jobs")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "global_step": 12,
                "job_id": "root",
                "root_job_id": "root",
            }
        ),
        encoding="utf-8",
    )

    manager = JobManager()
    root = jobs.Job(
        id="root",
        method="lora",
        preset="default",
        state=jobs.STATE_ERROR,
        root_job_id="root",
    )
    manager._jobs[root.id] = root
    first_in_discovery = threading.Event()
    allow_first_to_continue = threading.Event()
    discovery_calls = 0
    discovery_lock = threading.Lock()

    def candidates(_job):
        nonlocal discovery_calls
        with discovery_lock:
            discovery_calls += 1
            call_number = discovery_calls
        if call_number == 1:
            first_in_discovery.set()
            assert allow_first_to_continue.wait(timeout=5)
        return [(12, state_dir)]

    monkeypatch.setattr(manager, "_state_candidates", candidates)
    results: list[object] = []

    def run_resume():
        try:
            results.append(manager.resume_job(root.id))
        except Exception as exc:  # noqa: BLE001 - assertion captures the public error
            results.append(exc)

    first = threading.Thread(target=run_resume)
    second = threading.Thread(target=run_resume)
    first.start()
    assert first_in_discovery.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    allow_first_to_continue.set()
    first.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    created = [result for result in results if isinstance(result, jobs.Job)]
    errors = [result for result in results if isinstance(result, ValueError)]
    assert len(created) == 1
    assert len(errors) == 1
    assert "resume is already being created" in str(errors[0])
    assert [job.attempt_index for job in manager.lineage(root)] == [0, 1]


def test_resume_allows_interruption_inside_target_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "daemon")
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "daemon" / "jobs")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "global_step": 12,
                "current_epoch": 3,
                "job_id": "root",
                "root_job_id": "root",
                "interrupted": True,
            }
        ),
        encoding="utf-8",
    )

    manager = JobManager()
    root = jobs.Job(
        id="root",
        method="lora",
        preset="default",
        state=jobs.STATE_STOPPED,
        root_job_id="root",
        target_epochs=3,
    )
    manager._jobs[root.id] = root
    monkeypatch.setattr(manager, "_state_candidates", lambda _job: [(12, state_dir)])
    monkeypatch.setattr(manager, "resume", lambda: None)

    resumed = manager.resume_job(root.id)

    assert resumed is not None
    assert resumed.parent_job_id == root.id
    assert resumed.target_epochs == 3
    assert resumed.recovery_step == 12


def test_job_groups_filter_latest_attempt_and_delete_whole_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "daemon")
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "daemon" / "jobs")
    manager = JobManager()
    root = jobs.Job(
        id="root",
        method="lora",
        preset="default",
        state=jobs.STATE_ERROR,
        root_job_id="root",
        submitted_at=1.0,
    )
    child = jobs.Job(
        id="child",
        method="lora",
        preset="default",
        state=jobs.STATE_DONE,
        root_job_id="root",
        parent_job_id="root",
        attempt_index=1,
        submitted_at=2.0,
    )
    for job in (root, child):
        manager._jobs[job.id] = job
        job.persist()

    groups, total = manager.list_job_groups_filtered(
        state=jobs.STATE_DONE, newest_first=True
    )

    assert total == 1
    assert [group["id"] for group in groups] == ["root"]
    assert groups[0]["current_job_id"] == "child"
    assert [attempt["job_id"] for attempt in groups[0]["attempts"]] == [
        "root",
        "child",
    ]
    assert manager.delete_group("child") == ["root", "child"]
    assert not root.dir.exists()
    assert not child.dir.exists()


def test_job_groups_http_list_get_and_delete_whole_chain(daemon):
    cl, manager = daemon
    root = jobs.Job(
        id="http-root",
        method="lora",
        preset="default",
        state=jobs.STATE_ERROR,
        root_job_id="http-root",
        attempt_index=0,
        submitted_at=1.0,
    )
    child = jobs.Job(
        id="http-child",
        method="lora",
        preset="default",
        state=jobs.STATE_DONE,
        root_job_id="http-root",
        parent_job_id="http-root",
        attempt_index=1,
        submitted_at=2.0,
    )
    with manager._lock:
        for job in (root, child):
            manager._jobs[job.id] = job
            job.persist()

    page = cl._request("GET", "/job-groups?state=done&offset=0&limit=1")
    assert page["total"] == 1
    assert page["offset"] == 0
    assert page["limit"] == 1
    assert [group["id"] for group in page["groups"]] == ["http-root"]

    detail = cl.get_job_group("http-child")
    assert detail["root_job_id"] == "http-root"
    assert detail["current_job_id"] == "http-child"
    assert [attempt["job_id"] for attempt in detail["attempts"]] == [
        "http-root",
        "http-child",
    ]

    deleted = cl.delete_job_group("http-child")
    assert deleted == {
        "ok": True,
        "root_job_id": "http-root",
        "job_ids": ["http-root", "http-child"],
    }
    assert manager.get("http-root") is None
    assert manager.get("http-child") is None
    assert not root.dir.exists()
    assert not child.dir.exists()


def test_lineage_fields_survive_job_table_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "daemon")
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "daemon" / "jobs")
    root = jobs.Job(
        id="root",
        method="lora",
        preset="default",
        state=jobs.STATE_ERROR,
        root_job_id="root",
        attempt_index=0,
    )
    child = jobs.Job(
        id="child",
        method="lora",
        preset="default",
        state=jobs.STATE_DONE,
        root_job_id="root",
        parent_job_id="root",
        attempt_index=1,
    )
    root.persist()
    child.persist()

    reloaded = jobs.load_all()
    manager = JobManager()
    manager._jobs = reloaded
    group = manager.job_group("child")

    assert group is not None
    assert group["id"] == "root"
    assert group["current_job_id"] == "child"
    assert [attempt["attempt_index"] for attempt in group["attempts"]] == [0, 1]
    assert reloaded["child"].parent_job_id == "root"


def test_resume_rejects_lineage_with_active_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "daemon")
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "daemon" / "jobs")
    manager = JobManager()
    root = jobs.Job(
        id="root",
        method="lora",
        preset="default",
        state=jobs.STATE_ERROR,
        root_job_id="root",
        attempt_index=0,
    )
    active = jobs.Job(
        id="active",
        method="lora",
        preset="default",
        state=jobs.STATE_RUNNING,
        root_job_id="root",
        parent_job_id="root",
        attempt_index=1,
    )
    manager._jobs = {root.id: root, active.id: active}

    with pytest.raises(ValueError, match="only the latest attempt"):
        manager.resume_job(root.id)
    assert manager.resume_job(active.id) is None
    assert set(manager._jobs) == {"root", "active"}


def test_command_job_build_cmd():
    """A `kind="command"` job builds a plain `python <argv>` call (no
    accelerate launch) and merges its extra_env over the inherited env."""
    job = jobs.Job(
        id="c1",
        method="preprocess",
        preset="",
        kind="command",
        argv=["tasks.py", "preprocess"],
        extra_env={"CAPTION_SHUFFLE_VARIANTS": "7"},
    )
    mgr = JobManager.__new__(JobManager)  # no worker thread
    cmd, env = mgr._build_cmd(job)
    # Command jobs launch under the resolved venv interpreter (windowless on
    # Windows), not necessarily the caller's sys.executable.
    from scripts.daemon.client import venv_python

    assert cmd == [venv_python(windowless=True), "tasks.py", "preprocess"]
    assert "train.py" not in cmd
    assert env["CAPTION_SHUFFLE_VARIANTS"] == "7"
    assert env["PYTHONUNBUFFERED"] == "1"
    # tqdm throttled so stdout.log stays tail-readable (redraws every 10s, not 0.1s)
    assert env["TQDM_MININTERVAL"] == "10"


def test_resumed_training_command_uses_immutable_config_snapshot():
    job = jobs.Job(
        id="resume-command",
        method="lora-gui",
        preset="",
        kind="command",
        argv=["tasks.py", "lora-gui", "lokr", "--resume", "/tmp/state"],
        config_file="/tmp/job/config.snapshot.toml",
        recovery_state="/tmp/state",
        progress_path="/tmp/job/progress.jsonl",
        sample_dir="/tmp/job/sample",
    )
    mgr = JobManager.__new__(JobManager)

    cmd, _env = mgr._build_cmd(job)

    assert cmd.count("--config_file") == 1
    assert cmd[cmd.index("--config_file") + 1] == job.config_file
    assert cmd.count("--resume") == 1


def test_resumed_staged_training_forwards_snapshot_resume_and_daemon_paths():
    """Staged training must use the pinned runtime config on resume.

    The command wrapper accepts only the profile plus daemon-injected
    ``--config_file``/``--resume``/telemetry paths.  This guards the recovery
    path against silently recompiling a profile that may have changed after the
    original job was submitted.
    """
    job = jobs.Job(
        id="resume-staged",
        method="staged-train",
        preset="",
        kind="command",
        argv=["tasks.py", "staged-train", "default", "--resume", "/tmp/state"],
        config_file="/tmp/job/config.snapshot.toml",
        recovery_state="/tmp/state",
        progress_path="/tmp/job/progress.jsonl",
        sample_dir="/tmp/job/sample",
    )
    mgr = JobManager.__new__(JobManager)

    cmd, _env = mgr._build_cmd(job)

    assert cmd.count("--config_file") == 1
    assert cmd[cmd.index("--config_file") + 1] == job.config_file
    assert cmd.count("--resume") == 1
    assert cmd[cmd.index("--resume") + 1] == "/tmp/state"
    assert cmd[cmd.index("--progress_jsonl") + 1] == job.progress_path
    assert cmd[cmd.index("--sample_dir") + 1] == job.sample_dir


def test_command_job_loads_with_train_default():
    """A legacy job.json (written before `kind` existed) loads as a train job."""
    job = jobs.Job.from_dict({"id": "old", "method": "lora", "preset": "default"})
    assert job.kind == "train"
    assert job.argv == [] and job.extra_env == {}


def test_command_training_job_injects_sample_dir(tmp_path, monkeypatch):
    """A training command job (``tasks.py lora …``) gets ``--sample_dir`` and
    ``--progress_jsonl`` injected pointing at its own job dir, so the dashboard
    never replays a previous task's gallery/metrics.

    Regression guard for the per-job sample isolation fix: the daemon must set
    ``job.sample_dir`` (read back by the WebUI over HTTP) AND inject it into
    the train process's argv. Both halves are required — argv alone is
    invisible to the preview API after a WebUI restart.
    """
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    job = jobs.Job(
        id="train-job",
        method="lora",
        preset="default",
        kind="command",
        argv=["tasks.py", "lora"],
    )
    # Mirror what submit() does: assign the per-job paths on the record.
    d = job.dir
    job.progress_path = str(d / "progress.jsonl")
    job.sample_dir = str(d / "sample")

    mgr = JobManager.__new__(JobManager)  # no worker thread
    cmd, _env = mgr._build_cmd(job)

    # Both flags injected, pointing at the job's own dir (not a shared path).
    def _flag_value(argv, flag):
        i = argv.index(flag)
        return argv[i + 1]

    assert _flag_value(cmd, "--progress_jsonl") == str(d / "progress.jsonl")
    assert _flag_value(cmd, "--sample_dir") == str(d / "sample")
    # The WebUI recovers sample_dir from the persisted Job record — assert it
    # round-trips through job.json (the C8 restart-recovery path).
    job.persist()
    loaded = jobs.load_all()["train-job"]
    assert loaded.sample_dir == str(d / "sample")


def test_legacy_job_json_loads_without_sample_dir(tmp_path, monkeypatch):
    """A job.json written before ``sample_dir`` existed must still load —
    ``from_dict`` filters unknown keys, and the new field defaults to None."""
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    (tmp_path / "jobs").mkdir(parents=True)
    legacy = tmp_path / "jobs" / "old" / "job.json"
    legacy.parent.mkdir(parents=True)
    # No sample_dir key — simulates a pre-fix job.json on disk.
    legacy.write_text(
        '{"id": "old", "method": "lora", "preset": "default", "state": "done"}',
        encoding="utf-8",
    )
    loaded = jobs.load_all()["old"]
    assert loaded.sample_dir is None
    assert loaded.method == "lora"


@pytest.fixture
def real_cmd_daemon(tmp_path, monkeypatch):
    """Daemon with the *real* `_build_cmd` (no fake-trainer patch) so command
    jobs actually exec their argv. GPU guard stubbed so the queue never blocks
    on the host's VRAM."""
    from scripts.daemon import client

    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(config, "PIDFILE", tmp_path / "daemon.json")
    monkeypatch.setattr(config, "DAEMON_LOG", tmp_path / "daemon.log")
    monkeypatch.setattr(gpu, "gpu_pids", lambda: set())

    mgr = JobManager()
    mgr.start()
    srv = serve(mgr, port=0)
    t = threading.Thread(
        target=srv.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True
    )
    t.start()
    cl = client.DaemonClient(srv.server_address[1])
    assert _wait_until(lambda: cl.health() is not None, timeout=5)
    try:
        yield cl, mgr
    finally:
        srv.request_shutdown(True)
        srv.server_close()


def test_command_job_end_to_end(real_cmd_daemon):
    """submit_command → detached exec → exit-code finalize (no progress.jsonl),
    with extra_env applied and stdout captured."""
    cl, _ = real_cmd_daemon
    resp = cl.submit_command(
        label="preprocess",
        argv=[
            "-c",
            "import os;print('shuf=' + os.environ['CAPTION_SHUFFLE_VARIANTS'])",
        ],
        extra_env={"CAPTION_SHUFFLE_VARIANTS": "7"},
    )
    jid = resp["job_id"]
    assert resp["state"] == "queued"
    assert _wait_until(lambda: cl.get(jid)["state"] == "done", timeout=15)
    job = cl.get(jid)
    assert job["kind"] == "command"
    assert job["argv"][0] == "-c"
    log = (config.job_dir(jid) / "stdout.log").read_text()
    assert "shuf=7" in log


def test_command_job_missing_argv_rejected(real_cmd_daemon):
    """A command submission without argv is a 400 (urllib raises HTTPError)."""
    import urllib.error

    cl, _ = real_cmd_daemon
    with pytest.raises(urllib.error.HTTPError) as ei:
        cl._request("POST", "/jobs", {"kind": "command", "label": "x"})
    assert ei.value.code == 400


def test_serve_falls_back_when_port_held_by_stranger():
    """A non-anima process on the preferred port → bind an ephemeral one
    instead of failing (``serve_with_fallback``)."""
    import socket

    from scripts.daemon.server import serve_with_fallback

    # A plain listener that never speaks HTTP — stands in for a stranger.
    stranger = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stranger.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    stranger.bind((config.HOST, 0))
    stranger.listen(1)
    held = stranger.getsockname()[1]

    mgr = JobManager.__new__(JobManager)  # serve() doesn't need a started worker
    server = None
    try:
        server = serve_with_fallback(mgr, port=held)
        bound = server.server_address[1]
        assert bound != held  # moved off the contested port
        assert bound != 0
    finally:
        if server is not None:
            server.server_close()
        stranger.close()


def test_serve_defers_to_a_live_sibling_daemon(daemon):
    """If an anima daemon already answers on the port, ``serve_with_fallback``
    re-raises so the second process stands down (no duplicate daemon)."""
    from scripts.daemon.server import serve_with_fallback

    cl, mgr = daemon  # a real in-process daemon is already serving here
    port = cl.port
    with pytest.raises(OSError):
        serve_with_fallback(JobManager.__new__(JobManager), port=port)


# --------------------------------------------------------------------------
# MCP stdio bridge (scripts/daemon/mcp.py)
# --------------------------------------------------------------------------


def _mcp_for(cl):
    """A bridge wired to an in-process daemon client (no pidfile discovery)."""
    return MCPServer(client_factory=lambda: cl, ensure=lambda: cl)


def _call_tool(srv, name, arguments=None, msg_id=1):
    resp = srv.handle(
        {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
    )
    result = resp["result"]
    payload = json.loads(result["content"][0]["text"])
    return result, payload


def _dead_client():
    """A client pointed at a port nothing listens on (health → None fast)."""
    return _RealDaemonClient(port=1)


def test_mcp_initialize_and_tools_list():
    srv = MCPServer(client_factory=_dead_client, ensure=_dead_client)
    resp = srv.handle(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
        }
    )
    res = resp["result"]
    assert res["protocolVersion"] == "2025-06-18"
    assert "tools" in res["capabilities"]
    # notifications get no response
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    tools = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {t["name"] for t in tools["result"]["tools"]}
    assert {
        "submit_training",
        "submit_command",
        "list_jobs",
        "get_job",
        "stop_job",
        "tail_log",
        "pause_queue",
        "start_queue",
        "health",
        "shutdown",
    } <= names
    assert "tail_logs" not in names  # SSE endpoint replaced, not registered
    for t in tools["result"]["tools"]:
        assert t["inputSchema"]["type"] == "object"


def test_mcp_unknown_method_and_tool():
    srv = MCPServer(client_factory=_dead_client, ensure=_dead_client)
    resp = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "nope/nope"})
    assert resp["error"]["code"] == -32601
    result = srv.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "no_such_tool", "arguments": {}},
        }
    )["result"]
    assert result["isError"] is True


def test_mcp_daemon_down_is_reported_not_spawned():
    srv = MCPServer(client_factory=_dead_client, ensure=_dead_client)
    # health degrades gracefully…
    result, payload = _call_tool(srv, "health")
    assert result["isError"] is False
    assert payload["up"] is False
    # …while other passive tools error with a hint instead of booting a daemon
    result = srv.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "list_jobs", "arguments": {}},
        }
    )["result"]
    assert result["isError"] is True
    assert "no daemon is running" in result["content"][0]["text"]


def test_mcp_submit_train_get_stop_roundtrip(daemon):
    cl, _ = daemon
    srv = _mcp_for(cl)

    result, payload = _call_tool(
        srv, "submit_training", {"method": "lora", "overrides": {"duration": 0.5}}
    )
    assert result["isError"] is False
    jid = payload["job_id"]

    def done():
        _, job = _call_tool(srv, "get_job", {"id": jid})
        return job["state"] == "done"

    assert _wait_until(done, timeout=15)
    _, job = _call_tool(srv, "get_job", {"id": jid})
    assert job["latest"]["ev"] == "run_end"

    result, payload = _call_tool(srv, "health")
    assert payload["ok"] is True

    # stopping an already-done job is a clean no-op response, not a crash
    result, payload = _call_tool(srv, "stop_job", {"id": jid})
    assert result["isError"] is False


def test_mcp_get_job_404_is_tool_error(daemon):
    cl, _ = daemon
    result = _mcp_for(cl).handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "get_job", "arguments": {"id": "nope"}},
        }
    )["result"]
    assert result["isError"] is True
    assert "404" in result["content"][0]["text"]


def test_mcp_submit_command_and_tail_log(real_cmd_daemon):
    cl, _ = real_cmd_daemon
    srv = _mcp_for(cl)

    # the bridge injects kind="command" so the daemon doesn't treat it as train
    result, payload = _call_tool(
        srv,
        "submit_command",
        {"label": "echo", "argv": ["-c", "print('hello-mcp')"]},
    )
    assert result["isError"] is False
    jid = payload["job_id"]

    def done():
        _, job = _call_tool(srv, "get_job", {"id": jid})
        return job["state"] == "done"

    assert _wait_until(done, timeout=15)

    result, payload = _call_tool(srv, "tail_log", {"id": jid, "lines": 5})
    assert result["isError"] is False
    assert payload["state"] == "done"
    assert any("hello-mcp" in line for line in payload["lines"])

    # tail_log survives the daemon going away (reads job.json + stdout.log)
    down = MCPServer(client_factory=_dead_client, ensure=_dead_client)
    result, payload = _call_tool(down, "tail_log", {"id": jid})
    assert result["isError"] is False
    assert payload["state"] == "done"
    assert any("hello-mcp" in line for line in payload["lines"])


# --------------------------------------------------------------------------
# daemon-status CLI verb
# --------------------------------------------------------------------------


def test_daemon_status_json(daemon, monkeypatch, capsys):
    import scripts.daemon.client as daemon_client
    from scripts.tasks import daemon as daemon_tasks

    cl, _ = daemon
    monkeypatch.setattr(daemon_client, "DaemonClient", lambda port=None: cl)
    jid = cl.submit(method="lora", overrides={"duration": 0.3})["job_id"]

    daemon_tasks.cmd_daemon_status([])
    out = json.loads(capsys.readouterr().out)
    assert out["up"] is True
    assert out["base_url"] == cl.base
    assert any(j["id"] == jid for j in out["jobs"])
    # compact by default: heavy record fields are stripped…
    assert "argv" not in out["jobs"][0] and "extra_env" not in out["jobs"][0]

    # …and --full restores the raw records
    daemon_tasks.cmd_daemon_status(["--full"])
    full = json.loads(capsys.readouterr().out)
    assert "argv" in full["jobs"][0]


def test_daemon_status_down_exits_1(monkeypatch, capsys):
    import scripts.daemon.client as daemon_client
    from scripts.tasks import daemon as daemon_tasks

    monkeypatch.setattr(daemon_client, "DaemonClient", lambda port=None: _dead_client())
    with pytest.raises(SystemExit) as ei:
        daemon_tasks.cmd_daemon_status([])
    assert ei.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["up"] is False


def test_tail_while_write(tmp_path):
    """progress.jsonl tail-while-write: last_event sees the freshest line even
    as it grows (Windows-strict-locking smoke check)."""
    from scripts.daemon import tail

    p = tmp_path / "progress.jsonl"
    with open(p, "w", buffering=1, encoding="utf-8") as f:
        f.write(json.dumps({"ev": "run_start", "ts": 0.0}) + "\n")
        assert tail.last_event(str(p))["ev"] == "run_start"
        f.write(json.dumps({"ev": "step", "ts": 0.1, "global_step": 5}) + "\n")
        ev = tail.last_event(str(p))
        assert ev["ev"] == "step" and ev["global_step"] == 5
    assert tail.last_ckpt_path(str(p)) is None


# --------------------------------------------------------------------------
# structured progress queries (get_progress) + agent-readable log tails
# --------------------------------------------------------------------------


def test_read_events_filters(tmp_path):
    from scripts.daemon import tail

    p = tmp_path / "progress.jsonl"
    stream = [{"ev": "run_start", "ts": 0.0}]
    for i in range(1, 11):
        stream.append({"ev": "step", "ts": float(i), "global_step": i, "loss": 1.0 / i})
    stream += [
        {"ev": "log", "ts": 10.5, "level": "WARNING", "logger": "x", "msg": "boom"},
        {"ev": "ckpt", "ts": 11.0, "global_step": 10, "path": "/tmp/x.safetensors"},
        {"ev": "run_end", "ts": 12.0, "status": "ok", "final_step": 10},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for ev in stream:
            f.write(json.dumps(ev) + "\n")

    assert len(tail.read_events(str(p))) == len(stream)

    # ev-kind filter
    steps = tail.read_events(str(p), events=["step"])
    assert [e["global_step"] for e in steps] == list(range(1, 11))

    # since_step — step-less events inherit the preceding step
    late = tail.read_events(str(p), since_step=8)
    assert [e["ev"] for e in late] == ["step", "step", "step", "log", "ckpt", "run_end"]

    # every_nth thins step events but always keeps the latest one
    thinned = tail.read_events(str(p), events=["step"], every_nth=4)
    assert [e["global_step"] for e in thinned] == [1, 5, 9, 10]

    # last_n trailing cap
    assert [e["ev"] for e in tail.read_events(str(p), last_n=2)] == ["ckpt", "run_end"]

    # a half-written tail line is skipped, not fatal
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"ev": "step", "global_st')
    assert len(tail.read_events(str(p))) == len(stream)

    # missing / unset path → empty
    assert tail.read_events(None) == []
    assert tail.read_events(str(tmp_path / "nope.jsonl")) == []


def test_progress_endpoint_http(daemon):
    import urllib.error

    cl, _ = daemon
    jid = cl.submit(method="lora", overrides={"duration": 0.2})["job_id"]
    assert _wait_until(lambda: cl.get(jid)["state"] == "done", timeout=15)

    out = cl._request("GET", f"/jobs/{jid}/progress")
    assert out["job_id"] == jid and out["state"] == "done"
    kinds = [e["ev"] for e in out["events"]]
    assert kinds == ["run_start", "step", "ckpt", "run_end"]
    assert out["count"] == 4

    out = cl._request("GET", f"/jobs/{jid}/progress?events=step,run_end&last_n=1")
    assert [e["ev"] for e in out["events"]] == ["run_end"]

    with pytest.raises(urllib.error.HTTPError):
        cl._request("GET", "/jobs/nope/progress")


def test_mcp_get_progress(daemon):
    cl, _ = daemon
    srv = _mcp_for(cl)
    _, payload = _call_tool(
        srv, "submit_training", {"method": "lora", "overrides": {"duration": 0.2}}
    )
    jid = payload["job_id"]

    def done():
        _, job = _call_tool(srv, "get_job", {"id": jid})
        return job["state"] == "done"

    assert _wait_until(done, timeout=15)

    # registered in the catalog (rides in from server.TOOLS)
    tools = srv.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/list"})
    assert "get_progress" in {t["name"] for t in tools["result"]["tools"]}

    result, payload = _call_tool(srv, "get_progress", {"id": jid})
    assert result["isError"] is False
    assert [e["ev"] for e in payload["events"]] == [
        "run_start",
        "step",
        "ckpt",
        "run_end",
    ]

    # filters ride through (comma-string form, as in the manifest schema)
    _, payload = _call_tool(srv, "get_progress", {"id": jid, "events": "step"})
    assert [e["ev"] for e in payload["events"]] == ["step"]

    # …and it survives the daemon going away (reads progress.jsonl from disk)
    down = MCPServer(client_factory=_dead_client, ensure=_dead_client)
    result, payload = _call_tool(down, "get_progress", {"id": jid})
    assert result["isError"] is False
    assert payload["events"][-1]["ev"] == "run_end"

    result, payload = _call_tool(down, "get_progress", {"id": "nope"})
    assert payload.get("error") == "no such job"


def test_tail_lines_collapse_tqdm_redraws(tmp_path):
    """One tqdm bar = one tail line: \\r redraw runs collapse to the final
    rendering instead of flooding the window with bar updates."""
    from scripts.daemon.mcp import _tail_lines

    p = tmp_path / "stdout.log"
    bar = "\r".join(f"caching:  {i}%|██| {i}/100" for i in range(0, 101, 10))
    p.write_text("starting\n" + bar + "\nwarn: thing happened\n\n", encoding="utf-8")
    assert _tail_lines(str(p), 10) == [
        "starting",
        "caching:  100%|██| 100/100",
        "warn: thing happened",
    ]
