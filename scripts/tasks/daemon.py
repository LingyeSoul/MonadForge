"""CLI surface for the local training daemon (``make daemon*``).

Four verbs, mapped to the lifecycle guarantees in ``plan.md`` Phase 1:

    daemon            start (idempotent — no-op if already up), wait /health
    daemon-attach     non-owning viewer; ctrl-C detaches only, training lives on
    daemon-kill       abort the running (or JOB=<id>) job, free GPU; daemon stays up
    daemon-terminate  shut the whole daemon down (active job dies too)

``daemon`` starts the daemon **console-detached** (see ``proc.spawn_detached``),
so the terminal's SIGINT reaches only the foreground group, never the daemon.
``daemon-attach`` is the parent of nothing, so its ctrl-C can't touch training.
Both teardown verbs verify the pidfile's ``(pid, create_time)`` before acting so
they never touch a PID-reused stranger.
"""

from __future__ import annotations

import os
import sys

from scripts.daemon import client as _client
from scripts.daemon import config as _cfg
from scripts.daemon import proc as _proc


def _job_arg(extra) -> str | None:
    """Resolve a job id from ``JOB=<id>`` env or the first positional arg."""
    job = os.environ.get("JOB")
    if not job and extra and not extra[0].startswith("-"):
        job = extra[0]
    return job or None


def cmd_daemon(extra):
    """Start the training daemon (idempotent). Detached + waits for /health."""
    existing = _proc.daemon_alive(_cfg.PIDFILE)
    if existing is not None:
        print(
            f"daemon already running (pid {existing.get('pid')}, "
            f"port {existing.get('port')})."
        )
        return
    try:
        cl = _client.ensure_daemon()
    except RuntimeError as e:
        print(f"failed to start daemon: {e}", file=sys.stderr)
        sys.exit(1)
    health = cl.health() or {}
    print(
        f"daemon up on {cl.base} (pid {health.get('pid')}). "
        f"Logs: {_cfg.DAEMON_LOG}\n"
        "  make daemon-attach        # follow events\n"
        "  make daemon-kill          # abort the running job\n"
        "  make daemon-terminate     # stop the daemon"
    )


def cmd_daemon_attach(extra):
    """Read-only viewer. ``JOB=<id>`` follows that job's stdout; otherwise the
    daemon event stream. Ctrl-C detaches this terminal only — never the daemon
    or the training subprocess (we are the parent of nothing)."""
    if not _client.is_running():
        print("no daemon; `make daemon` to start.", file=sys.stderr)
        sys.exit(1)
    cl = _client.DaemonClient()
    job = _job_arg(extra)
    stream = cl.stream_logs(job) if job else cl.stream_events()
    what = f"job {job}" if job else "daemon events"
    print(f"attached to {what} ({cl.base}) — ctrl-C to detach\n")
    try:
        for line in stream:
            print(line, flush=True)
    except KeyboardInterrupt:
        print("\ndetached (training continues).")
    except Exception as e:  # noqa: BLE001 — socket reset on daemon shutdown, etc.
        print(f"\nstream ended: {e}")


def cmd_daemon_kill(extra):
    """Abort a job; the daemon stays up and advances to the next queued job.
    ``JOB=<id>`` targets a specific job; otherwise the running one."""
    if not _client.is_running():
        print("no daemon running.", file=sys.stderr)
        sys.exit(1)
    cl = _client.DaemonClient()
    job = _job_arg(extra)
    result = cl.stop(job)
    if result.get("error"):
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    print(f"job {result.get('job_id')} → {result.get('state')} (daemon still up).")


def cmd_daemon_terminate(extra):
    """Stop the whole daemon. The active job tree is killed and the GPU freed."""
    if not _client.is_running():
        print("no daemon running.", file=sys.stderr)
        return
    cl = _client.DaemonClient()
    cl.shutdown(kill_jobs=True)
    print("daemon terminated (active job killed, GPU freed, queue discarded).")
