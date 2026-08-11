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
    # Physical daemon attempts, oldest first. Inline runs leave this empty.
    attempts: list["Run"] = field(default_factory=list)


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


def _link_log_sources(run: Run) -> None:
    log_dir = run.log_dir
    if log_dir is None or not os.path.isdir(log_dir):
        run.sources = _source_flags(run)
        return
    if run.snapshot is None:
        snaps = [f for f in os.listdir(log_dir) if f.endswith(".snapshot.toml")]
        if snaps:
            run.snapshot = parse_snapshot(os.path.join(log_dir, snaps[0]))
    if run.tb is None:
        run.tb = parse_tb(log_dir)
    run.sources = _source_flags(run)


def _attempt_sort_key(run: Run) -> tuple[int, float, str]:
    job = run.job or {}
    try:
        attempt_index = int(job.get("attempt_index") or 0)
    except (TypeError, ValueError):
        attempt_index = 0
    submitted = run.submitted_at
    try:
        submitted_value = float(submitted) if submitted is not None else 0.0
    except (TypeError, ValueError):
        submitted_value = 0.0
    return attempt_index, submitted_value, run.id


def _shift_event(event: dict, attempt_id: str, offset: float) -> dict:
    copied = dict(event)
    copied["attempt_id"] = attempt_id
    ts = copied.get("ts")
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        copied["ts"] = float(ts) + offset
    return copied


def _merge_jsonl(attempts: list[Run]) -> Optional[JsonlRun]:
    available = [run for run in attempts if run.jsonl is not None]
    if not available:
        return None

    out = JsonlRun()
    series_by_step: dict[str, dict[int, float]] = {}
    step_times: list[float] = []
    starts = [float(run.started_at) for run in attempts if run.started_at is not None]
    root_started_at = min(starts) if starts else None

    for run in attempts:
        jl = run.jsonl
        if jl is None:
            continue
        if jl.run:
            out.run = jl.run
        if jl.method:
            out.method = jl.method
        if jl.preset:
            out.preset = jl.preset
        if jl.total_steps is not None:
            out.total_steps = jl.total_steps
        if jl.total_epochs is not None:
            out.total_epochs = jl.total_epochs
        if jl.pid is not None:
            out.pid = jl.pid
        if jl.log_dir:
            out.log_dir = jl.log_dir
        out.sampling_enabled = out.sampling_enabled or jl.sampling_enabled

        offset = 0.0
        if root_started_at is not None and run.started_at is not None:
            offset = max(0.0, float(run.started_at) - root_started_at)
        for tag, points in jl.series.items():
            merged = series_by_step.setdefault(tag, {})
            for step, value in points:
                merged[int(step)] = float(value)
        out.gs_epoch.update({int(step): int(epoch) for step, epoch in jl.gs_epoch.items()})
        out.ckpts.extend(_shift_event(event, run.id, offset) for event in jl.ckpts)
        out.samples.extend(_shift_event(event, run.id, offset) for event in jl.samples)
        out.vals.extend(_shift_event(event, run.id, offset) for event in jl.vals)
        out.logs.extend(_shift_event(event, run.id, offset) for event in jl.logs)
        out.error_count += jl.error_count
        for ts in (jl.first_step_ts, jl.last_step_ts):
            if isinstance(ts, (int, float)) and not isinstance(ts, bool):
                step_times.append(float(ts) + offset)

    out.series = {
        tag: [[step, values[step]] for step in sorted(values)]
        for tag, values in series_by_step.items()
    }
    all_steps = sorted({step for values in series_by_step.values() for step in values})
    if all_steps:
        out.first_step = all_steps[0]
        out.last_step = all_steps[-1]
    if step_times:
        out.first_step_ts = min(step_times)
        out.last_step_ts = max(step_times)
    latest_jsonl = attempts[-1].jsonl
    if latest_jsonl is not None:
        out.run_end_status = latest_jsonl.run_end_status
        out.run_end_final_step = latest_jsonl.run_end_final_step
        out.run_end_error = latest_jsonl.run_end_error
        out.run_end_extra = dict(latest_jsonl.run_end_extra)
    return out


def _merge_tensorboard(attempts: list[Run]) -> Optional[TbRun]:
    series_by_step: dict[str, dict[int, float]] = {}
    latest_file: Optional[str] = None
    for run in attempts:
        if run.tb is None:
            continue
        latest_file = run.tb.file or latest_file
        for tag, points in run.tb.series.items():
            merged = series_by_step.setdefault(tag, {})
            for step, value in points:
                merged[int(step)] = float(value)
    if not series_by_step:
        return None
    series = {
        tag: [[step, values[step]] for step in sorted(values)]
        for tag, values in series_by_step.items()
    }
    steps = sorted({step for values in series_by_step.values() for step in values})
    return TbRun(series=series, steps=steps, file=latest_file)


def _merge_stdout(attempts: list[Run]) -> Optional[StdoutRun]:
    if not any(run.stdout is not None for run in attempts):
        return None
    out = StdoutRun()
    for position, run in enumerate(attempts, start=1):
        stdout = run.stdout
        if stdout is None:
            continue
        recovery_step = (run.job or {}).get("recovery_step")
        detail = f" from step {recovery_step}" if recovery_step is not None else ""
        out.lines.append(f"[attempt {position}/{len(attempts)} - {run.id}{detail}]")
        out.lines.extend(stdout.lines)
        out.warnings.extend(stdout.warnings)
        out.errors.extend(stdout.errors)
        out.markers.extend(stdout.markers)
    return out


def merge_daemon_attempts(attempts: list[Run]) -> Run:
    """Aggregate physical daemon attempts into one logical training run."""
    ordered = sorted(attempts, key=_attempt_sort_key)
    current = ordered[-1]
    current_job = current.job or {}
    root_id = str(current_job.get("root_job_id") or ordered[0].id)
    merged_jsonl = _merge_jsonl(ordered)
    merged = Run(
        id=root_id,
        kind="daemon",
        dir=current.dir,
        job=current.job,
        jsonl=merged_jsonl,
        jsonl_path=current.jsonl_path,
        tb=_merge_tensorboard(ordered),
        snapshot=next((run.snapshot for run in reversed(ordered) if run.snapshot), None),
        stdout=_merge_stdout(ordered),
        log_dir=next((run.log_dir for run in reversed(ordered) if run.log_dir), None),
        sample_dir=current.sample_dir,
        run_name=(merged_jsonl.run if merged_jsonl else current.run_name),
        method=(merged_jsonl.method if merged_jsonl else current.method),
        preset=(merged_jsonl.preset if merged_jsonl else current.preset),
        state=current.state,
        submitted_at=min(
            (run.submitted_at for run in ordered if run.submitted_at is not None),
            default=current.submitted_at,
        ),
        started_at=min(
            (run.started_at for run in ordered if run.started_at is not None),
            default=current.started_at,
        ),
        ended_at=current.ended_at,
        total_steps=(merged_jsonl.total_steps if merged_jsonl else current.total_steps),
        total_epochs=(merged_jsonl.total_epochs if merged_jsonl else current.total_epochs),
        ckpt_path=current.ckpt_path,
        error=current.error,
        stop_requested=current.stop_requested,
        attempts=ordered,
    )
    if merged_jsonl is not None:
        apply_jsonl_metadata(merged, merged_jsonl)
    merged.sources = _source_flags(merged)
    return merged


def _scan_daemon_jobs() -> list[Run]:
    physical_runs: list[Run] = []
    if not os.path.isdir(JOBS_DIR):
        return []
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
        _link_log_sources(run)
        physical_runs.append(run)

    grouped: dict[str, list[Run]] = {}
    for run in physical_runs:
        root_id = str((run.job or {}).get("root_job_id") or run.id)
        grouped.setdefault(root_id, []).append(run)
    runs = [merge_daemon_attempts(group) for group in grouped.values()]
    return sorted(
        runs,
        key=lambda run: float(run.submitted_at or 0.0),
        reverse=True,
    )


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
    known = {
        attempt.log_dir
        for run in runs
        for attempt in (run.attempts or [run])
        if attempt.log_dir
    }
    inline = _scan_inline_logs(known)
    runs.extend(inline)
    return runs
