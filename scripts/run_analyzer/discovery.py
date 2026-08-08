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

_TRAINING_METHODS = {
    "lora-gui",
    "lora",
    "easycontrol",
    "exp",
    "turbo",
    "exp-spd",
    "lora8",
    "lora-8gb",
}
_LOG_DIR_RE = re.compile(r"^(.*)_(\d{8})-(\d{4})$")

MONADFORGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
JOBS_DIR = os.path.join(MONADFORGE_ROOT, "output", "daemon", "jobs")
LOGS_DIR = os.path.join(MONADFORGE_ROOT, "output", "logs")


@dataclass
class Run:
    id: str
    kind: str  # "daemon" | "inline"
    dir: str
    job: Optional[dict] = None
    jsonl: Optional[JsonlRun] = None
    jsonl_path: Optional[str] = None
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
    if (job.get("kind") or "").lower() == "train":
        return True
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


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.normpath(path)))


def _resolve_log_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    resolved = path if os.path.isabs(path) else os.path.join(MONADFORGE_ROOT, path)
    return os.path.normpath(resolved)


def apply_jsonl_metadata(run: Run, jl: JsonlRun) -> None:
    """Apply structured progress metadata, whose terminal event is authoritative."""

    run.jsonl = jl
    run.run_name = jl.run or run.run_name
    run.method = jl.method or run.method
    run.preset = jl.preset or run.preset
    if jl.total_steps is not None:
        run.total_steps = jl.total_steps
    if jl.total_epochs is not None:
        run.total_epochs = jl.total_epochs
    if jl.log_dir:
        run.log_dir = _resolve_log_dir(jl.log_dir)
    terminal_states = {"ok": "done", "error": "error", "stopped": "stopped"}
    if jl.run_end_status in terminal_states:
        run.state = terminal_states[jl.run_end_status]
        run.stop_requested = jl.run_end_status == "stopped" or run.stop_requested
    if jl.run_end_error and not run.error:
        run.error = jl.run_end_error


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
        jsonl_path = os.path.join(job_dir, "progress.jsonl")
        run = Run(
            id=name,
            kind="daemon",
            dir=job_dir,
            job=job,
            jsonl=parse_jsonl(jsonl_path),
            jsonl_path=jsonl_path,
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
            apply_jsonl_metadata(run, jl)
        run.sources = _source_flags(run)
        runs.append(run)
    return runs


def _scan_inline_logs(known_log_dirs: set) -> list[Run]:
    """Log dirs not owned by any daemon job → inline CLI runs."""
    runs: list[Run] = []
    if not os.path.isdir(LOGS_DIR):
        return runs
    jsonl_by_log_dir: dict[str, tuple[str, JsonlRun]] = {}
    for filename in os.listdir(LOGS_DIR):
        if not filename.endswith(".progress.jsonl"):
            continue
        jsonl_path = os.path.join(LOGS_DIR, filename)
        jl = parse_jsonl(jsonl_path)
        log_dir = _resolve_log_dir(jl.log_dir if jl else None)
        if jl is not None and log_dir:
            jsonl_by_log_dir[_normalized_path(log_dir)] = (jsonl_path, jl)

    normalized_known = {_normalized_path(path) for path in known_log_dirs if path}
    for name in sorted(os.listdir(LOGS_DIR), reverse=True):
        log_dir = os.path.join(LOGS_DIR, name)
        normalized_log_dir = _normalized_path(log_dir)
        if not os.path.isdir(log_dir) or normalized_log_dir in normalized_known:
            continue
        snaps = [f for f in os.listdir(log_dir) if f.endswith(".snapshot.toml")]
        jsonl_entry = jsonl_by_log_dir.get(normalized_log_dir)
        if not snaps and jsonl_entry is None:
            continue
        jsonl_path, jl = jsonl_entry if jsonl_entry else (None, None)
        run_name = (jl.run if jl else None) or (
            snaps[0][: -len(".snapshot.toml")] if snaps else name
        )
        run = Run(
            id=f"inline-{name}",
            kind="inline",
            dir=log_dir,
            log_dir=log_dir,
            jsonl=jl,
            jsonl_path=jsonl_path,
            snapshot=(
                parse_snapshot(os.path.join(log_dir, snaps[0])) if snaps else None
            ),
            tb=parse_tb(log_dir),
            run_name=run_name,
            state="orphan",
        )
        if jl is not None:
            apply_jsonl_metadata(run, jl)
        run.sources = _source_flags(run)
        if run.tb is not None and run.jsonl is None:
            run.total_steps = None
        runs.append(run)
    return runs


def _source_flags(run: Run) -> dict:
    return {
        "jsonl": run.jsonl is not None,
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
