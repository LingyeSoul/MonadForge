"""Progress: the line every stage prints, and the daemon stream behind it — stdlib only.

Two halves, and the first is the one every stage owes its caller.

**The printed line.** ``  [done/total] detail`` on stdout is a contract, not a
style: ``gui/jobs.py`` reads it back off the child's stdout for the panel's
progress bar (``frontend/src/state.ts``), and a stage that prints anything else
has no bar. :func:`progress_line` is the one place that format is spelled;
:class:`ProgressBar` counts for a stage that walks its own loop, and
``stages/_progress.py::make_progress`` wraps it for a stage that hands a
``progress(index, total, detail)`` callback to a library function. Every stage
goes through one of the two — a bare ``tqdm`` writes carriage returns the panel
cannot parse and the log keeps as noise.

**The daemon stream.** The rest of this module.

The daemon (``anima_lora/anima_daemon``) runs a curation stage as a *command*
job: it exports ``ANIMA_DAEMON_JOB_DIR``, tails ``<job_dir>/stdout.log``, and
kills a job whose output has frozen past its stall budget (120 s for a command
job). Liveness is the newest mtime of ``stdout.log`` *or*
``<job_dir>/progress.jsonl``, the structured stream its ``get_progress`` reads.
A quiet model load — SAM3, the tagger, a first-run weight download — can sit
past that budget with nothing on stdout, and the watchdog reads a wedge.

This module is the stage's side of that file. With the variable unset every
function is a no-op, so the CLIs, the GUI's runner and a plain shell see no
difference; with it set:

- :func:`step` appends one ``step`` line per ``progress(index, total, detail)``
  call, spelled the way the daemon's reader filters (``ev`` / ``global_step``),
  so ``get_progress`` and its ``since_step`` / ``every_nth`` thinning work on a
  curation job as they do on a train run::

      {"ev": "step", "ts": 1.7e9, "global_step": 41, "total_steps": 900, "detail": "a/1.png"}

- :func:`phase` brackets a quiet stretch (a model load) with ``phase`` lines
  and, while it lasts, a ``heartbeat`` line every :data:`HEARTBEAT_S` from a
  daemon thread, so the file's mtime keeps moving and the load is not a stall::

      {"ev": "phase", "ts": ..., "name": "load sam3", "state": "start"}
      {"ev": "heartbeat", "ts": ..., "phase": "load sam3"}
      {"ev": "phase", "ts": ..., "name": "load sam3", "state": "end", "seconds": 71.2}

Every write is best-effort: a job dir that vanished, a full disk or a
non-serialisable detail is swallowed, because a progress line must never fail
the stage it reports on. The file is append-only and line-buffered, one
``json.dumps`` per line, so a reader can tail it while the stage writes.

The trainer's stream is documented in ``anima_lora/library/training/progress.py``;
its reader is ``anima_daemon/tail.py``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "HEARTBEAT_S",
    "JOB_DIR_ENV",
    "PROGRESS_NAME",
    "Progress",
    "ProgressBar",
    "emit",
    "job_dir",
    "phase",
    "progress_line",
    "progress_path",
    "step",
]

JOB_DIR_ENV = "ANIMA_DAEMON_JOB_DIR"
"""The daemon exports this for every job it spawns: the job's own directory."""

PROGRESS_NAME = "progress.jsonl"
"""The stream's name inside the job dir — the daemon's ``job.progress_path``."""

HEARTBEAT_S = 30.0
"""Seconds between ``heartbeat`` lines inside a :func:`phase`. Well under the
daemon's 120 s command-job budget, and rare enough that a long load costs a
few lines."""

Progress = Callable[[int, int, str], None]
"""The ``progress(index, total, detail)`` callback every stage takes."""


# ---- the printed line ---------------------------------------------------


def progress_line(
    index: int, total: int, detail: str = "", *, every: int = 1, first: bool = False
) -> None:
    """Print ``  [index/total] detail``, thinned, and stream the step regardless.

    The one spelling of the format the GUI parses, and the one thinning rule:
    every ``every``-th line, index 1 when ``first``, and always the last — so a
    run shorter than ``every`` still says it finished. The daemon's
    ``progress.jsonl`` gets every call either way; it does its own thinning.
    """
    if index >= total or (every > 0 and index % every == 0) or (first and index == 1):
        print(f"  [{index}/{total}] {detail}".rstrip(), flush=True)
    step(index, total, detail)


class ProgressBar:
    """A counter over a loop the stage walks itself.

    For the stages that have no ``progress=`` callback to pass — the SAM3 mask
    run, the mask merge, the grouping embedder — so all of them report in the
    one format instead of each reaching for ``tqdm``. :meth:`advance` moves the
    count without saying anything (an image whose outcome is decided further
    down the loop body), :meth:`note` prints where the count stands, and
    :meth:`tick` is the two together, which is what a plain loop wants.
    """

    __slots__ = ("done", "every", "first", "total")

    def __init__(self, total: int, *, every: int = 1, first: bool = False):
        self.total = int(total)
        self.every = int(every)
        self.first = bool(first)
        self.done = 0

    def advance(self, n: int = 1) -> None:
        """``n`` items dealt with, nothing printed yet."""
        self.done += n

    def note(self, detail: str = "") -> None:
        """Print where the count stands, subject to the thinning."""
        progress_line(self.done, self.total, detail, every=self.every, first=self.first)

    def tick(self, detail: str = "", n: int = 1) -> None:
        """:meth:`advance` then :meth:`note` — one line for the ``n`` items it
        moved past (``n > 1`` for a stage that works a batch at a time)."""
        self.advance(n)
        self.note(detail)


# ---- the daemon stream --------------------------------------------------


def job_dir() -> Path | None:
    """The daemon job directory, or ``None`` outside a daemon job."""
    value = os.environ.get(JOB_DIR_ENV)
    return Path(value) if value else None


def progress_path() -> Path | None:
    """``<job_dir>/progress.jsonl``, or ``None`` outside a daemon job."""
    d = job_dir()
    return d / PROGRESS_NAME if d is not None else None


def emit(ev: str, **fields: object) -> bool:
    """Append one ``{"ev": ev, "ts": now, **fields}`` line. Returns whether a
    line was written; never raises."""
    path = progress_path()
    if path is None:
        return False
    try:
        line = json.dumps({"ev": ev, "ts": time.time(), **fields}, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return True
    except (OSError, TypeError, ValueError):
        return False


def step(index: int, total: int, detail: str = "") -> None:
    """One ``step`` line: image ``index`` of ``total`` dealt with. Has the
    :data:`Progress` signature, so it can be a stage's callback outright."""
    emit("step", global_step=int(index), total_steps=int(total), detail=str(detail))


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Bracket a quiet stretch with ``phase`` lines and keep a heartbeat going
    while it lasts. Outside a daemon job the block runs untouched."""
    if progress_path() is None:
        yield
        return
    started = time.monotonic()
    emit("phase", name=name, state="start")
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(HEARTBEAT_S):
            emit("heartbeat", phase=name)

    thread = threading.Thread(
        target=beat, name=f"progress-heartbeat:{name}", daemon=True
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        emit(
            "phase",
            name=name,
            state="end",
            seconds=round(time.monotonic() - started, 3),
        )
