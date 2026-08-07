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
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

import psutil

from library.runtime.compat import prepare_python_child_env

logger = logging.getLogger(__name__)


def _effectively_alive(process) -> bool:
    """Return whether *process* still needs a termination signal.

    ``psutil.wait_procs`` can report an orphaned grandchild as alive while it is
    already a zombie: the daemon cannot reap that process, and sending SIGKILL
    to it only makes the stop path wait until its hard deadline.  Treat zombie
    (and dead) statuses as terminated.  Tiny fake process objects used by unit
    tests may not expose ``status``/``is_running``; those conservatively remain
    alive so the force-kill branch is still exercised.
    """
    try:
        status = process.status()
    except AttributeError:
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    if status in {
        getattr(psutil, "STATUS_ZOMBIE", "zombie"),
        getattr(psutil, "STATUS_DEAD", "dead"),
    }:
        return False
    try:
        return bool(process.is_running())
    except AttributeError:
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _live_after_wait(processes) -> list:
    """Drop processes that exited or became zombies during ``wait_procs``."""
    return [process for process in processes if _effectively_alive(process)]


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


def _process_create_time(process) -> Optional[float]:
    """Read a process object's creation time without propagating races."""
    try:
        # psutil caches Process.create_time() on the object.  Reopen by PID for
        # every check, otherwise a Process object retained across the grace
        # wait could report the original timestamp after its PID was recycled.
        pid = getattr(process, "pid", None)
        probe = psutil.Process(pid) if pid is not None else process
        return float(probe.create_time())
    except (AttributeError, TypeError, ValueError, OSError, psutil.Error):
        # A process can disappear or become inaccessible between the family
        # snapshot and a later terminate/kill call.  Unknown identity is not a
        # reason to send a destructive signal when verification was requested.
        return None


def _identity_matches(
    process,
    expected_create_time: Optional[float],
    *,
    tol: float = 1.0,
) -> bool:
    """Return whether *process* still has the recorded creation time.

    ``None`` is retained as an opt-out for legacy direct callers.  Daemon Job
    paths always pass the persisted value before signalling a process tree.
    """
    if expected_create_time is None:
        return True
    actual = _process_create_time(process)
    try:
        expected = float(expected_create_time)
    except (TypeError, ValueError):
        return False
    return actual is not None and abs(actual - expected) <= tol


def _verified_targets(
    processes: list,
    snapshots: dict[int, Optional[float]],
    *,
    verify_identity: bool,
) -> list:
    """Filter a process list to objects whose PID identity is still stable."""
    if not verify_identity:
        return list(processes)
    targets = []
    for process in processes:
        snapshot = snapshots.get(id(process))
        if snapshot is None or not _identity_matches(process, snapshot):
            logger.warning(
                "skip signalling process %s: PID identity changed or is unavailable",
                getattr(process, "pid", "?"),
            )
            continue
        targets.append(process)
    return targets


def _refresh_live_family(
    pid: int,
    family: list,
    snapshots: dict[int, Optional[float]],
    *,
    expected_create_time: Optional[float],
    verify_identity: bool,
) -> list:
    """Re-enumerate descendants without losing the original family snapshot.

    A cooperative stop can spend tens of seconds inside the training process.
    Launchers and DataLoader workers may create another descendant during that
    interval, so force-killing only the initial snapshot can leave a process
    holding GPU memory after the queue advances.  Newly observed descendants
    are safe to add only while the root still has the persisted identity; the
    original objects remain tracked after the root exits and gets reparented.
    """

    known = {
        (getattr(process, "pid", id(process)), snapshots.get(id(process)))
        for process in family
    }
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        parent = None

    if parent is not None and (
        not verify_identity or _identity_matches(parent, expected_create_time)
    ):
        current = [parent]
        try:
            current.extend(parent.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        for process in current:
            process_pid = getattr(process, "pid", id(process))
            snapshot = (
                expected_create_time
                if process_pid == pid and verify_identity
                else _process_create_time(process)
            )
            if verify_identity and snapshot is None:
                logger.warning(
                    "skip newly observed process %s: PID identity is unavailable",
                    process_pid,
                )
                continue
            identity = (process_pid, snapshot)
            if identity in known:
                continue
            family.append(process)
            snapshots[id(process)] = snapshot
            known.add(identity)

    live = _live_after_wait(family)
    return _verified_targets(live, snapshots, verify_identity=verify_identity)


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
    child_env = dict(os.environ if env is None else env)
    prepare_python_child_env(child_env)
    log = open(stdout_path, "ab", buffering=0)
    kwargs: dict = {
        "cwd": str(cwd),
        "stdout": log,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": child_env,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(cmd, **kwargs)
    finally:
        log.close()  # the child has dup'd the fd; our handle is done


def kill_tree(
    pid: int,
    *,
    expected_create_time: Optional[float] = None,
    grace_seconds: float = 5.0,
) -> None:
    """Terminate ``pid`` and every descendant; SIGKILL survivors after grace.

    Snapshots descendants up-front — children of a dying process get reparented
    and would slip past a re-walk. Safe to call on an already-dead PID. When
    ``expected_create_time`` is supplied, no signal is sent unless the root PID
    still belongs to the recorded process.
    """
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return

    verify_identity = expected_create_time is not None
    if verify_identity and not _identity_matches(parent, expected_create_time):
        logger.warning(
            "refusing to kill pid %s: create_time no longer matches persisted job",
            pid,
        )
        return

    family = [parent]
    try:
        family.extend(parent.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    snapshots = {id(process): _process_create_time(process) for process in family}
    if verify_identity:
        # Keep the persisted value for the root, rather than replacing it with
        # a freshly sampled value after the initial PID-reuse check.
        snapshots[id(parent)] = expected_create_time
        if not _identity_matches(parent, expected_create_time):
            logger.warning(
                "refusing to kill pid %s: create_time changed while taking family snapshot",
                pid,
            )
            return
    targets = _verified_targets(
        family, snapshots, verify_identity=verify_identity
    )

    for p in targets:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    psutil.wait_procs(targets, timeout=grace_seconds)
    alive = _refresh_live_family(
        pid,
        family,
        snapshots,
        expected_create_time=expected_create_time,
        verify_identity=verify_identity,
    )
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def stop_tree_gracefully(
    pid: int,
    *,
    expected_create_time: Optional[float] = None,
    grace_seconds: float = 30.0,
) -> bool:
    """Request cooperative stop, then force-kill the remaining process tree.

    Returns ``True`` when the family exited within the cooperative grace period
    and ``False`` when a forced termination was required.  On Windows the
    daemon writes a stop-file before calling this function; hidden
    ``CREATE_NO_WINDOW`` children do not have a reliable console signal path.
    When ``expected_create_time`` is supplied, a PID mismatch is treated as an
    already-gone job and no signal is sent to the replacement process.
    """

    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return True

    verify_identity = expected_create_time is not None
    if verify_identity and not _identity_matches(parent, expected_create_time):
        logger.warning(
            "refusing to stop pid %s: create_time no longer matches persisted job",
            pid,
        )
        return True

    family = [parent]
    try:
        family.extend(parent.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    snapshots = {id(process): _process_create_time(process) for process in family}
    if verify_identity:
        # The root must continue to match the persisted identity throughout the
        # stop. Descendants use their own creation-time snapshots below.
        snapshots[id(parent)] = expected_create_time
        if not _identity_matches(parent, expected_create_time):
            logger.warning(
                "refusing to stop pid %s: create_time changed while taking family snapshot",
                pid,
            )
            return True

    interrupted = False
    if sys.platform == "win32":
        interrupted = True
    else:
        try:
            # killpg is not mediated by psutil, so recheck immediately before
            # invoking it; otherwise a recycled PID could terminate an
            # unrelated process group.
            if verify_identity and not _identity_matches(parent, expected_create_time):
                logger.warning(
                    "refusing to signal pid %s: create_time changed before SIGTERM",
                    pid,
                )
                return True
            process_group = os.getpgid(pid)
            if process_group == pid:
                os.killpg(process_group, signal.SIGTERM)
            else:
                if verify_identity and not _identity_matches(
                    parent, expected_create_time
                ):
                    logger.warning(
                        "refusing to signal pid %s: create_time changed before SIGTERM",
                        pid,
                    )
                    return True
                parent.send_signal(signal.SIGTERM)
            interrupted = True
        except (OSError, psutil.Error, ValueError):
            try:
                if verify_identity and not _identity_matches(
                    parent, expected_create_time
                ):
                    logger.warning(
                        "refusing to signal pid %s: create_time changed before SIGINT",
                        pid,
                    )
                    return True
                parent.send_signal(signal.SIGINT)
                interrupted = True
            except (OSError, psutil.Error, ValueError):
                pass

    alive = family
    if interrupted:
        psutil.wait_procs(family, timeout=max(0.0, grace_seconds))
        alive = _refresh_live_family(
            pid,
            family,
            snapshots,
            expected_create_time=expected_create_time,
            verify_identity=verify_identity,
        )
    if not alive:
        return True

    for process in alive:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(alive, timeout=5.0)
    alive = _refresh_live_family(
        pid,
        family,
        snapshots,
        expected_create_time=expected_create_time,
        verify_identity=verify_identity,
    )
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if alive:
        logger.warning(
            "force-killed %d process(es) remaining after cooperative stop for pid %s",
            len(alive),
            pid,
        )
    return False


# pidfile — single-daemon lock keyed on (pid, create_time)
def write_pidfile(
    path: Path, *, pid: int, port: int, root: Optional[Path] = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ct = create_time(pid)
    data = {"pid": pid, "create_time": ct, "port": port}
    if root is not None:
        data["root"] = str(root)
    path.write_text(json.dumps(data), encoding="utf-8")


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
