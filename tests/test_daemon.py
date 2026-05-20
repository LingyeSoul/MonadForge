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
from scripts.daemon.manager import JobManager
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
import json, sys, time
path, dur = sys.argv[1], float(sys.argv[2])
with open(path, "w", buffering=1) as f:
    f.write(json.dumps({"ev": "run_start", "ts": 0.0}) + "\n")
    f.write(json.dumps({"ev": "step", "ts": 0.1, "global_step": 1, "loss": 0.5}) + "\n")
    time.sleep(dur)
    f.write(json.dumps({"ev": "ckpt", "ts": dur, "global_step": 1, "path": "/tmp/fake.safetensors"}) + "\n")
    f.write(json.dumps({"ev": "run_end", "ts": dur, "status": "ok", "final_step": 1}) + "\n")
"""


def _fake_build_cmd(self, job):
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
    monkeypatch.setattr(
        _common, "accelerate_launch", lambda *a: launched.append(a)
    )

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
    assert cmd == [sys.executable, "tasks.py", "preprocess"]
    assert "train.py" not in cmd
    assert env["CAPTION_SHUFFLE_VARIANTS"] == "7"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_command_job_loads_with_train_default():
    """A legacy job.json (written before `kind` existed) loads as a train job."""
    job = jobs.Job.from_dict({"id": "old", "method": "lora", "preset": "default"})
    assert job.kind == "train"
    assert job.argv == [] and job.extra_env == {}


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
        argv=["-c", "import os;print('shuf=' + os.environ['CAPTION_SHUFFLE_VARIANTS'])"],
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
