"""Process control for the daemon — spawn detached, kill trees, prove liveness.

Every rule here exists because a training job is a **process tree**
(``accelerate launch → train.py → dataloader workers``), not one PID, and
because PIDs get reused. Route every spawn / kill / liveness check through
psutil so the same code works on Linux and Windows (the daemon must run on
both — ``python tasks.py daemon`` is the Windows alias for ``make daemon``).

This is the ``Popen``-flavored sibling of ``gui/process.py`` (which is
``QProcess``-bound): same snapshot-then-terminate-then-kill tree walk.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import psutil


# --------------------------------------------------------------------------
# liveness — identify a process by (pid, create_time), never PID alone
# --------------------------------------------------------------------------


def create_time(pid: int) -> Optional[float]:
    """``psutil.Process(pid).create_time()`` or ``None`` if the PID is gone."""
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def is_alive(pid: Optional[int], ct: Optional[float], *, tol: float = 1.0) -> bool:
    """True iff ``pid`` exists *and* its create_time matches ``ct``.

    The create_time check is the sole defense against PID reuse — without it a
    recycled PID looks like our still-running job. ``tol`` absorbs the
    sub-second rounding difference between platforms' create_time clocks.
    """
    if pid is None or ct is None:
        return False
    actual = create_time(pid)
    if actual is None:
        return False
    return abs(actual - ct) <= tol


# --------------------------------------------------------------------------
# detached spawn
# --------------------------------------------------------------------------


def spawn_detached(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    env: Optional[dict] = None,
) -> subprocess.Popen:
    """Spawn ``cmd`` detached from this process's console, stdout→file.

    Detaching is what lets a console ctrl-C miss the child:
    ``start_new_session=True`` on POSIX (new session/process group, terminal
    SIGINT only reaches the foreground group), ``CREATE_NO_WINDOW`` on Windows.

    Windows console nuance — why ``CREATE_NO_WINDOW`` *without*
    ``DETACHED_PROCESS``: detaching gives the whole training tree **no console
    at all**, so when ``torch.compile``'s inductor/Triton backend shells out to
    native compilers (``ptxas.exe`` per CUDA kernel, ``cl.exe`` for the C++
    wrapper) with no creation flags, Windows sees "parent has no console" and
    allocates a fresh **visible** console for each — a burst of terminal-window
    flashes on every compile-heavy training start. ``CREATE_NO_WINDOW`` instead
    gives the tree a console that *exists but is hidden*; those compiler
    grandchildren inherit it rather than popping their own. CTRL_C isolation is
    preserved regardless: the daemon runs under ``pythonw`` with no console of
    its own, and a ``CREATE_NO_WINDOW`` child gets its own private hidden
    console, so a stray terminal CTRL_C still can't reach it (and we kill jobs
    via ``kill_tree``, not console events). Stdio still has no usable inherited
    handles, so redirecting to a file stays mandatory — we do it on both
    platforms for uniformity.

    Window suppression on Windows is the *interpreter's* job, not a creation
    flag's: the uv venv ``python.exe`` is a trampoline that re-launches the real
    interpreter, so ``CREATE_NO_WINDOW`` set here doesn't reliably reach the
    child's console. Callers that must stay windowless (the long-lived daemon)
    launch under ``pythonw.exe`` instead (see ``client.venv_python``).
    """
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(stdout_path, "ab", buffering=0)
    kwargs: dict = {
        "cwd": str(cwd),
        "stdout": log,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": env,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(cmd, **kwargs)
    finally:
        # The child has dup'd the fd; our handle is no longer needed.
        log.close()


# --------------------------------------------------------------------------
# tree teardown
# --------------------------------------------------------------------------


def kill_tree(pid: int, *, grace_seconds: float = 5.0) -> None:
    """Terminate ``pid`` and every descendant; SIGKILL survivors after grace.

    Snapshots descendants up-front — children of a dying process get reparented
    and would slip past a re-walk. Safe to call on an already-dead PID.
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    family = [parent]
    try:
        family.extend(parent.children(recursive=True))
    except psutil.NoSuchProcess:
        pass

    for p in family:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    _, alive = psutil.wait_procs(family, timeout=grace_seconds)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


# --------------------------------------------------------------------------
# pidfile — single-daemon lock keyed on (pid, create_time)
# --------------------------------------------------------------------------


def write_pidfile(path: Path, *, pid: int, port: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ct = create_time(pid)
    path.write_text(
        json.dumps({"pid": pid, "create_time": ct, "port": port}),
        encoding="utf-8",
    )


def read_pidfile(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def daemon_alive(path: Path) -> Optional[dict]:
    """Return the pidfile dict iff it points at a live daemon, else ``None``.

    A stale pidfile (process gone, or PID reused by a stranger) reads as not
    alive — the caller is then free to take over the port.
    """
    info = read_pidfile(path)
    if not info:
        return None
    if is_alive(info.get("pid"), info.get("create_time")):
        return info
    return None
