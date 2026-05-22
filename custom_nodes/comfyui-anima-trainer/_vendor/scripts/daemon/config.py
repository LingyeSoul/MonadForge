"""Shared paths + constants for the local training daemon.

Single localhost process; one job at a time. State lives under
``output/daemon/`` so it sits beside the checkpoints the jobs produce and is
already covered by the repo's ``output/`` gitignore.

    output/daemon/
      daemon.json           pidfile: {"pid", "create_time", "port"}
      daemon.log            the detached daemon's stdout/stderr
      jobs/<job_id>/
        job.json            persisted Job record (survives daemon restart)
        stdout.log          the training subprocess's captured stdout+stderr
        progress.jsonl      the Phase-0 structured progress stream (we point
                            train.py's --progress_jsonl here)
"""

from __future__ import annotations

import os
from pathlib import Path

# scripts/daemon/config.py -> parents[2] == anima_lora/
ROOT = Path(__file__).resolve().parents[2]

STATE_DIR = ROOT / "output" / "daemon"
JOBS_DIR = STATE_DIR / "jobs"
PIDFILE = STATE_DIR / "daemon.json"
DAEMON_LOG = STATE_DIR / "daemon.log"

# Fixed localhost port — the daemon binds 127.0.0.1 only (non-goal: remote /
# auth). Overridable for tests / odd setups via env.
DEFAULT_PORT = int(os.environ.get("ANIMA_DAEMON_PORT", "8765"))
HOST = "127.0.0.1"


def ensure_state_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id
