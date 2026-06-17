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
