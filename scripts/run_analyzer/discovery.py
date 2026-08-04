"""Run discovery — build the unified run index from daemon jobs + log dirs.

Sources:
  - ``output/daemon/jobs/<job_id>/``  (job.json + progress.jsonl + stdout.log + sample/)
  - ``output/logs/<run>_<timestamp>/`` (snapshot.toml + network_train/tfevents)
     → inline (non-daemon) runs, linked via the JSONL ``log_dir`` field.

Non-training jobs (preprocess/mask/…) are filtered out of the index.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from scripts.run_analyzer.sources.jsonl import JsonlRun, parse as parse_jsonl
from scripts.run_analyzer.sources.tensorboard import TbRun, parse as parse_tb
from scripts.run_analyzer.sources.snapshot import Snapshot, parse as parse_snapshot
from scripts.run_analyzer.sources.stdout_log import StdoutRun, parse as parse_stdout

_TRAINING_METHODS = {"lora-gui", "lora", "easycontrol", "exp", "turbo", "exp-spd", "lora8", "lora-8gb"}
_LOG_DIR_RE = re.compile(r"^(.*)_(\d{8})-(\d{4})$")

MONADFORGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOBS_DIR = os.path.join(MONADFORGE_ROOT, "output", "daemon", "jobs")
LOGS_DIR = os.path.join(MONADFORGE_ROOT, "output", "logs")


@dataclass
class Run:
    id: str
    kind: str  # "daemon" | "inline"
    dir: str
    job: Optional[dict] = None
    jsonl: Optional[JsonlRun] = None
    tb: Optional[TbRun] = None
    snapshot: Optional[Snapshot] = None
    stdout: Optional[StdoutRun] = None
    log_dir: Optional[str] = None
    sample_dir: Optional[str] = None

    # derived
    run_name: str = ""
    method: str = ""
    preset: str = ""
    state: str = "unknown"  # running | done | error | stopped | queued | orphan
    submitted_at: Optional[float] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    total_steps: Optional[int] = None
    total_epochs: Optional[int] = None
    ckpt_path: Optional[str] = None
    error: Optional[str] = None
    stop_requested: bool = False
    sources: dict = field(default_factory=dict)


def _is_training_job(job: dict) -> bool:
    method = (job.get("method") or "").lower()
    if method in _TRAINING_METHODS:
        return True
    argv = job.get("argv") or []
    return any("train.py" in str(a) for a in argv)


def _load_job(job_dir: str) -> Optional[dict]:
    p = os.path.join(job_dir, "job.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _scan_daemon_jobs() -> list[Run]:
    runs: list[Run] = []
    if not os.path.isdir(JOBS_DIR):
        return runs
    for name in sorted(os.listdir(JOBS_DIR), reverse=True):
        job_dir = os.path.join(JOBS_DIR, name)
        if not os.path.isdir(job_dir):
            continue
        job = _load_job(job_dir)
        if job is None or not _is_training_job(job):
            continue
        run = Run(
            id=name,
            kind="daemon",
            dir=job_dir,
            job=job,
            jsonl=parse_jsonl(os.path.join(job_dir, "progress.jsonl")),
            stdout=parse_stdout(os.path.join(job_dir, "stdout.log")),
            sample_dir=job.get("sample_dir") or os.path.join(job_dir, "sample"),
            method=(job.get("method") or ""),
            preset=(job.get("extra_env") or {}).get("PRESET", ""),
            state=(job.get("state") or "unknown"),
            submitted_at=job.get("submitted_at"),
            started_at=job.get("started_at"),
            ended_at=job.get("ended_at"),
            ckpt_path=job.get("ckpt_path"),
            error=job.get("error"),
            stop_requested=bool(job.get("stop_requested")),
        )
        jl = run.jsonl
        if jl is not None:
            run.run_name = jl.run or run.run_name
            run.method = jl.method or run.method
            run.preset = jl.preset or run.preset
            run.total_steps = jl.total_steps
            run.total_epochs = jl.total_epochs
            if jl.log_dir:
                run.log_dir = os.path.join(MONADFORGE_ROOT, jl.log_dir)
            if jl.run_end_status == "error" and run.state != "running":
                run.state = "error"
            elif jl.run_end_status == "stopped" and run.state != "running":
                run.state = "stopped"
        run.sources = _source_flags(run)
        runs.append(run)
    return runs


def _scan_inline_logs(known_log_dirs: set) -> list[Run]:
    """Log dirs not owned by any daemon job → inline CLI runs."""
    runs: list[Run] = []
    if not os.path.isdir(LOGS_DIR):
        return runs
    for name in sorted(os.listdir(LOGS_DIR), reverse=True):
        log_dir = os.path.join(LOGS_DIR, name)
        if not os.path.isdir(log_dir) or log_dir in known_log_dirs:
            continue
        snaps = [f for f in os.listdir(log_dir) if f.endswith(".snapshot.toml")]
        if not snaps:
            continue
        run_name = snaps[0][: -len(".snapshot.toml")]
        run = Run(
            id=f"inline-{name}",
            kind="inline",
            dir=log_dir,
            log_dir=log_dir,
            snapshot=parse_snapshot(os.path.join(log_dir, snaps[0])),
            tb=parse_tb(log_dir),
            run_name=run_name,
            state="orphan",
        )
        run.sources = _source_flags(run)
        if run.tb is not None:
            run.total_steps = None
        runs.append(run)
    return runs


def _source_flags(run: Run) -> dict:
    return {
        "jsonl": run.jsonl is not None and bool(run.jsonl.series),
        "tensorboard": run.tb is not None,
        "snapshot": run.snapshot is not None,
        "stdout": run.stdout is not None,
    }


def discover(use_cache: bool = True) -> list[Run]:
    """Build the full run index (daemon jobs first, then inline runs)."""
    runs = _scan_daemon_jobs()
    known = {r.log_dir for r in runs if r.log_dir}
    inline = _scan_inline_logs(known)
    runs.extend(inline)
    # link missing snapshot/tb for daemon runs whose log_dir exists
    for run in runs:
        if run.kind != "daemon":
            continue
        ld = run.log_dir
        if ld is None or not os.path.isdir(ld):
            continue
        if run.snapshot is None:
            snaps = [f for f in os.listdir(ld) if f.endswith(".snapshot.toml")]
            if snaps:
                run.snapshot = parse_snapshot(os.path.join(ld, snaps[0]))
        if run.tb is None:
            run.tb = parse_tb(ld)
        run.sources = _source_flags(run)
    return runs
