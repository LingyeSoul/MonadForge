"""One error mapping for the typed model-line routers.

Qwen-Image-2.1 and MOSS raise the same three families for the same reasons, so
the status codes live in one place: a missing file is 404, bad input (including
an unusable path) is 400, an unreachable daemon is 502. ``label`` names the line
in the daemon message, which used to exist for MOSS only.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException

from webui.services.daemon_client import DaemonError


@contextmanager
def workspace_errors(label: str = "") -> Iterator[None]:
    """Map a workspace failure onto its HTTP status."""
    try:
        yield
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DaemonError as exc:
        detail = daemon_detail(exc)
        if label and isinstance(detail, str):
            detail = f"{label} daemon request failed: {detail}"
        raise HTTPException(status_code=502, detail=detail) from exc


def daemon_detail(exc: Exception) -> object:
    """The response detail for a daemon failure.

    A ``DaemonError`` may carry an i18n key + params (the daemon raises
    user-facing failures with one); the frontend localizes those and falls back
    to the plain message otherwise. Every router that surfaces daemon failures
    uses this shape, so a client never has to branch on the endpoint.
    """
    key = getattr(exc, "key", None)
    if key:
        return {
            "message": str(exc),
            "key": key,
            "params": getattr(exc, "params", {}) or {},
        }
    return str(exc)
