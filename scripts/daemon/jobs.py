"""The ``Job`` record + its on-disk persistence.

One JSON file per job under ``output/daemon/jobs/<id>/job.json`` so the daemon
survives a restart and can show history. The in-memory job table is the
authority while running; ``persist()`` mirrors each state change to disk, and
``load_all()`` rebuilds the table on boot for the reconciliation sweep.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import config

# queued → running → {done | error | stopped}
STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_ERROR = "error"
STATE_STOPPED = "stopped"
TERMINAL_STATES = frozenset({STATE_DONE, STATE_ERROR, STATE_STOPPED})


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
    overrides: dict = field(default_factory=dict)
    extra: list[str] = field(default_factory=list)

    # Job kind: "train" (the default — an ``accelerate launch … train.py`` run
    # built from method/preset/overrides/extra) or "command" (a plain
    # ``python <argv>`` task such as preprocess / mask). Command jobs carry
    # their own argv + env and skip the train-specific command building and
    # progress.jsonl wiring; they finalize on exit code. ``method`` doubles as
    # the display label for command jobs. Defaulting to "train" keeps legacy
    # job.json records (written before this field existed) loading correctly.
    kind: str = "train"
    argv: list[str] = field(default_factory=list)
    extra_env: dict = field(default_factory=dict)

    # Daemon-managed auto-chain: when a command job (preprocess) carries a
    # ``chain_train`` spec (``{method, preset, methods_subdir}``), the manager
    # enqueues that training job the moment this one finishes successfully —
    # so a GUI-initiated "preprocess → train" survives the GUI closing (the
    # chain lives in the daemon, not the UI). ``chained_job_id`` records the
    # follow-on it spawned, so a client can hop straight to observing it.
    chain_train: Optional[dict] = None
    chained_job_id: Optional[str] = None

    # Set on the follow-on train job the daemon spawns from a chain_train spec.
    # The pre-launch GPU guard is skipped for these: the daemon itself just ran
    # (and reaped) the preceding preprocess step on this same serial queue, so
    # it already knows nothing external owns the card — guarding here only races
    # against the just-exited preprocess's not-yet-released VRAM. Defaults False
    # so standalone train jobs (and legacy job.json) still guard normally.
    from_chain: bool = False

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
    ckpt_path: Optional[str] = None

    error: Optional[str] = None
    # Free-text hint for terminal states ("orphaned", "gpu_held_by_unknown", …).
    status_detail: Optional[str] = None
    # Set by an explicit stop request so the monitor records `stopped`, not
    # `error`, when it sees the process vanish. Runtime + persisted.
    stop_requested: bool = False

    # ----- persistence -----

    @property
    def dir(self) -> Path:
        return config.job_dir(self.id)

    def persist(self) -> None:
        d = self.dir
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(d / "job.json")  # atomic on POSIX + Windows

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    # ----- views -----

    def public(self) -> dict:
        """The dict shape returned over HTTP (drops nothing sensitive — this is
        localhost — but keeps the field order stable for clients)."""
        return asdict(self)


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
