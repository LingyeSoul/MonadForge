"""Windows-quiet subprocess helpers.

On Windows, ``subprocess`` launches of a *console* program (``git``,
``nvidia-smi``, ``powershell`` …) flash a console window on screen unless
``CREATE_NO_WINDOW`` is passed. The daemon's GPU-occupancy poll and the
per-checkpoint ModelSpec git query fire repeatedly, so on Windows users see a
terminal blink several times whenever a checkpoint is written — cosmetic but
alarming.

This is distinct from the *job launcher* (``scripts/daemon/proc.py``), which
spawns the trainer under ``pythonw.exe``: ``CREATE_NO_WINDOW`` doesn't survive
the uv venv ``python.exe`` trampoline re-exec, so that path needs a different
fix. ``CREATE_NO_WINDOW`` *does* work for direct console executables, which is
exactly what these short-lived metadata/probe calls invoke.

Usage::

    subprocess.run([...], **no_window_kwargs())
"""

from __future__ import annotations

import subprocess
import sys


def no_window_kwargs() -> dict:
    """``subprocess`` kwargs that suppress the Windows console-window flash.

    Returns ``{"creationflags": CREATE_NO_WINDOW}`` on Windows, ``{}`` elsewhere
    (so it's a harmless no-op on Linux/macOS). Merge into an existing kwargs
    dict or splat directly into a ``subprocess`` call.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
