"""The ``Job`` record + its on-disk persistence.

One JSON file per job under ``output/daemon/jobs/<id>/job.json`` so the daemon
survives a restart and can show history. The in-memory job table is the
authority while running; ``persist()`` mirrors each state change to disk, and
``load_all()`` rebuilds the table on boot for the reconciliation sweep.
"""

from __future__ import annotations

import json
import shutil
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import config

# queued → running → {done | error | stopped}
STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_DONE = "done"
STATE_ERROR = "error"
STATE_STOPPED = "stopped"
ACTIVE_STATES = frozenset({STATE_RUNNING, STATE_PAUSED})
TERMINAL_STATES = frozenset({STATE_DONE, STATE_ERROR, STATE_STOPPED})


class JobConflictError(ValueError):
    """A valid job request conflicts with existing persisted training state."""


def new_job_id() -> str:
    """Sortable, collision-resistant id: ``YYYYmmdd-HHMMSS-<6hex>``."""
    import secrets

    return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


@dataclass
class Job:
    id: str
    method: str
    preset: str
    methods_subdir: Optional[str] = None
    config_file: Optional[str] = None
    overrides: dict = field(default_factory=dict)
    extra: list[str] = field(default_factory=list)

    # "train" (accelerate launch train.py) or "command" (plain ``python <argv>``
    # — preprocess/mask, carrying own argv+env, finalized on exit code with no
    # progress.jsonl wiring). ``method`` doubles as the command's display label.
    # Defaulting to "train" keeps pre-field legacy job.json records loading.
    kind: str = "train"
    argv: list[str] = field(default_factory=list)
    extra_env: dict = field(default_factory=dict)
    stall_timeout: float | None = None

    # Auto-chain: a command job carrying a ``chain_train`` spec ({method, preset,
    # methods_subdir}) makes the manager enqueue that train job on success, so a
    # GUI-initiated "preprocess → train" survives the GUI closing (the chain
    # lives in the daemon). ``chained_job_id`` records the follow-on it spawned.
    chain_train: Optional[dict] = None
    chained_job_id: Optional[str] = None

    # Set on a chain-spawned follow-on train job → skip the pre-launch GPU guard:
    # the daemon just ran the preceding step on this same serial queue, so
    # guarding here only races the just-exited step's not-yet-released VRAM.
    from_chain: bool = False

    # One user-visible training may span several physical daemon processes.
    # Each process keeps its own immutable job directory; these additive fields
    # link resumed attempts into one logical task without breaking old records.
    # Legacy job.json files omit them and are treated as independent roots.
    root_job_id: Optional[str] = None
    parent_job_id: Optional[str] = None
    attempt_index: int = 0

    state: str = STATE_QUEUED
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    # The spawned `accelerate launch` process, identified as (pid, create_time)
    # so a reused PID can never be mistaken for our job.
    pid: Optional[int] = None
    create_time: Optional[float] = None

    progress_path: Optional[str] = None
    stdout_path: Optional[str] = None
    # Per-job preview-image dir (``<job_dir>/sample``); injected as
    # ``--sample_dir`` on training jobs so the dashboard never replays a
    # previous task's gallery. Returned to the WebUI on submit so it can find
    # the gallery even after a restart (``_tasks`` is session-only memory).
    sample_dir: Optional[str] = None
    ckpt_path: Optional[str] = None
    # The subprocess's real exit code (``popen.poll()``), captured at finalize
    # time so the WebUI's Task.exit_code mirrors the old direct-subprocess
    # design (which got it straight from ``process.wait()``). ``None`` for
    # queued-cancelled / launch-failed / orphaned paths that never ran a
    # process — ``public()`` exposes it as ``rc``, which the WebUI's
    # ``_finalize_from_daemon`` reads.
    rc: Optional[int] = None

    error: Optional[str] = None
    # Free-text hint for terminal states ("orphaned", "gpu_held_by_unknown", …).
    status_detail: Optional[str] = None
    # Set by an explicit stop request so the monitor records `stopped`, not
    # `error`, when it sees the process vanish. Runtime + persisted.
    stop_requested: bool = False
    stop_requested_at: Optional[float] = None
    forced_stop: bool = False

    # Immutable recovery contract captured at submission time.  These fields
    # are deliberately additive so older job.json records remain readable.
    config_signature: Optional[str] = None
    dataset_signature: Optional[str] = None
    target_steps: Optional[int] = None
    target_epochs: Optional[int] = None
    data_manifest: Optional[str] = None
    legacy: bool = False
    recovery_state: Optional[str] = None
    recovery_step: Optional[int] = None
    terminal_reason: Optional[str] = None
    continuation: Optional[dict] = None

    # Persistent task identity and interpreter selection; defaults retain legacy jobs.
    context: dict = field(default_factory=dict)
    runtime_profile: Optional[str] = None
    runtime_registration: Optional[dict] = None
    python_executable: Optional[str] = None
    runtime_env: dict = field(default_factory=dict)
    captured_env: dict = field(default_factory=dict)
    paused_at: Optional[float] = None
    release_requested: bool = False
    released: bool = False
    resume_state_dir: Optional[str] = None
    resume_count: int = 0
    accelerate_launched: bool = False
    returncode: Optional[int] = None
    result_path: Optional[str] = None
    result_summary: Optional[dict] = None
    # A release-pause can relaunch the same job. Keep the append-only stream,
    # but only events after this launch's byte offset can finalize its process.
    progress_epoch: int = 0
    progress_start_offset: Optional[int] = None

    @property
    def dir(self) -> Path:
        return config.job_dir(self.id)

    def persist(self) -> None:
        d = self.dir
        d.mkdir(parents=True, exist_ok=True)
        target = d / "job.json"
        payload = json.dumps(self._record(), indent=2)
        # Windows scanners can briefly hold the previous record. Keep the
        # replacement atomic while retrying that transient lock.
        for attempt in range(8):
            tmp = d / f".job.json.{os.getpid()}.{time.time_ns()}.tmp"
            try:
                tmp.write_text(payload, encoding="utf-8")
                os.replace(tmp, target)
                return
            except OSError:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt >= 7:
                    raise
                time.sleep(min(0.05 * (attempt + 1), 0.5))

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        code = payload.get("rc")
        if code is None:
            code = payload.get("returncode")
        payload["rc"] = payload["returncode"] = code
        job = cls(**payload)
        # Preserve forward/legacy fields without assigning arbitrary attributes.
        # Keep these outside the dataclass so serialization stays flat.
        job._unknown_fields = {k: v for k, v in data.items() if k not in known}
        return job

    def _record(self) -> dict:
        return {**getattr(self, "_unknown_fields", {}), **asdict(self)}

    def public(self) -> dict:
        """The dict shape returned over HTTP (drops nothing sensitive — this is
        localhost — but keeps the field order stable for clients)."""
        payload = self._record()
        code = self.rc if self.rc is not None else self.returncode
        payload["rc"] = payload["returncode"] = code
        if self.continuation:
            # The full context (blueprint, fingerprints, submit token) stays in
            # job.json; listings only need the display-level summary.
            context = self.continuation
            payload["continuation"] = {
                name: context[name]
                for name in (
                    "mode",
                    "continuation_kind",
                    "budget_key",
                    "target",
                    "target_steps",
                    "source_job_id",
                )
                if name in context
            }
        payload["root_job_id"] = self.root_job_id or self.id
        return payload


def load_all() -> dict[str, Job]:
    """Rebuild the job table from ``jobs/*/job.json`` (boot reconciliation)."""
    out: dict[str, Job] = {}
    if not config.JOBS_DIR.exists():
        return out
    for d in sorted(config.JOBS_DIR.iterdir()):
        f = d / "job.json"
        if not f.is_file():
            continue
        try:
            out[d.name] = Job.from_dict(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
    return out


def _dir_size(d: Path) -> int:
    """Bytes under ``d`` (best-effort — a file that vanishes mid-walk is 0)."""
    total = 0
    for f in d.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def prune_jobs(
    *,
    max_age_days: Optional[float] = None,
    keep_recent: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """Delete the job dirs of terminal jobs older than ``max_age_days``.

    Exposed as ``python -m anima_daemon prune`` (dry-run by default). Startup
    does not call this helper. Stop the daemon before applying a prune so its
    in-memory table cannot later persist and recreate a removed record.

    A dir is a candidate only when its ``job.json`` parses with a terminal
    state (queued/running/paused and unreadable records are always kept), it's
    older than ``max_age_days`` (by ``ended_at`` / ``submitted_at`` / mtime),
    and it's not among the ``keep_recent`` newest terminal jobs.
    ``max_age_days=0`` disables pruning. Returns
    ``{scanned, pruned: [ids], kept, freed_bytes, errors}``; never raises.
    """
    days = config.JOB_RETENTION_DAYS if max_age_days is None else max_age_days
    keep = config.JOB_RETENTION_KEEP if keep_recent is None else keep_recent
    summary: dict = {
        "scanned": 0,
        "pruned": [],
        "kept": 0,
        "freed_bytes": 0,
        "errors": 0,
        "max_age_days": days,
        "keep_recent": keep,
        "dry_run": dry_run,
    }
    if days <= 0 or not config.JOBS_DIR.exists():
        return summary

    now = time.time()
    cutoff = now - days * 86400.0
    candidates: list[tuple[float, Path]] = []  # (age-key timestamp, dir)
    for d in sorted(config.JOBS_DIR.iterdir()):
        f = d / "job.json"
        if not f.is_file():
            continue
        summary["scanned"] += 1
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("state") not in TERMINAL_STATES:
            continue
        ts = data.get("ended_at") or data.get("submitted_at")
        if not isinstance(ts, (int, float)):
            try:
                ts = f.stat().st_mtime
            except OSError:
                continue
        candidates.append((float(ts), d))

    # Newest first, then drop the protected head (terminal jobs only).
    candidates.sort(key=lambda t: t[0], reverse=True)
    for ts, d in candidates[keep:] if keep > 0 else candidates:
        if ts >= cutoff:
            continue
        size = _dir_size(d)
        if not dry_run:
            try:
                shutil.rmtree(d)
            except OSError:
                summary["errors"] += 1
                continue
        summary["pruned"].append(d.name)
        summary["freed_bytes"] += size
    summary["kept"] = len(candidates) - len(summary["pruned"])
    return summary
