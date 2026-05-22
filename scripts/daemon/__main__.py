"""Daemon process entry point: ``python -m scripts.daemon``.

This is the long-lived, console-detached process that ``make daemon`` spawns.
It takes the single-daemon lock (pidfile ``(pid, create_time)``; a live sibling
that already answers ``/health`` on the port also makes us stand down), binds
the preferred port — falling back to an ephemeral one if a *stranger* holds it
— reconciles ``jobs/`` from any previous run, then serves until
``POST /shutdown``.
"""

from __future__ import annotations

import logging
import os
import sys

from . import config, proc
from .manager import JobManager
from .server import serve_with_fallback


def _setup_logging() -> None:
    config.ensure_state_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],  # stderr → daemon.log file
    )


def main() -> int:
    _setup_logging()
    log = logging.getLogger("anima.daemon")

    port = config.DEFAULT_PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    # Refuse to start a second daemon over a live one.
    existing = proc.daemon_alive(config.PIDFILE)
    if existing is not None:
        log.error(
            "daemon already running (pid %s, port %s); refusing to start",
            existing.get("pid"),
            existing.get("port"),
        )
        return 3

    manager = JobManager()
    manager.start()

    try:
        # Falls back to an ephemeral port if a stranger holds the preferred one,
        # but re-raises (→ exit 3) if a sibling anima daemon already owns it, so
        # a startup race can't produce two daemons. The bound port is written to
        # the pidfile below; clients re-resolve it from there.
        server = serve_with_fallback(manager, port=port)
    except OSError as exc:
        log.error("could not bind 127.0.0.1:%s (%s); another anima daemon?", port, exc)
        manager.shutdown(kill_jobs=False)
        return 3

    proc.write_pidfile(config.PIDFILE, pid=os.getpid(), port=server.server_address[1])
    log.info(
        "anima training daemon up on http://%s:%s (pid %s)",
        config.HOST,
        server.server_address[1],
        os.getpid(),
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        manager.shutdown(kill_jobs=False)
    finally:
        try:
            config.PIDFILE.unlink()
        except OSError:
            pass
        log.info("daemon stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
