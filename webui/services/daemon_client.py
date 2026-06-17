"""Thin async wrapper over the local training daemon's HTTP surface.

The daemon (``scripts/daemon/``, a localhost serial job queue) owns process
lifecycle; this module is the only place the WebUI should speak HTTP to it.
Keeps ``urllib`` off the hot path by offloading each call to a thread, and
concentrates port discovery / error handling so callers get a small typed API:

    job = await DaemonClient.ensure_started()
    job_id = await client.submit_command("lora", ["--network_dim", "32"])
    info = await client.get_job(job_id)
    await client.stop(job_id)

The daemon client in ``scripts/daemon/client.py`` is the canonical stdlib
implementation (synchronous, used by the CLI / MCP bridge / ComfyUI node). We
import its ``ensure_daemon`` for the auto-start handshake rather than
re-implementing port discovery — but wrap every HTTP call here as ``async`` so
the WebUI's event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Short default for liveness probes; job-state polls can afford the full value.
_HEALTH_TIMEOUT = 3.0
_DEFAULT_TIMEOUT = 15.0


class DaemonError(RuntimeError):
    """Raised when the daemon is unreachable or returns an HTTP error."""


class DaemonClient:
    """Async facade over the daemon's localhost HTTP API.

    Resolves the daemon's base URL (incl. ephemeral port fallback) lazily on
    first use via the canonical pidfile walker in ``scripts.daemon.config`` /
    ``client``. Call :meth:`ensure_started` (or :func:`ensure_daemon_running`)
    once at WebUI boot to auto-spawn a detached daemon if none is running.
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        self._base: Optional[str] = base_url.rstrip("/") if base_url else None

    @property
    def base(self) -> str:
        if self._base is None:
            self._base = self._resolve_base()
        return self._base

    @staticmethod
    def _resolve_base() -> str:
        """Resolve the daemon base URL from its pidfile (handles port drift)."""
        # Lazy import: keep the WebUI importable even if scripts.daemon is
        # mid-rename or the venv lacks optional deps at import time.
        from scripts.daemon import client as _dclient

        port = _dclient._resolve_port()
        return f"http://127.0.0.1:{port}"

    # ── low-level request ──────────────────────────────────────────────

    def _request_sync(
        self,
        method: str,
        path: str,
        *,
        body: Optional[dict] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> Any:
        """Synchronous core HTTP call. Used directly by the tray (no event loop)
        and wrapped by :meth:`_request` for async callers (the WebUI)."""
        url = f"{self.base}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise DaemonError(f"HTTP {e.code} {method} {path}: {detail}") from e
        except urllib.error.URLError as e:
            raise DaemonError(f"daemon unreachable at {url}: {e.reason}") from e

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[dict] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> Any:
        return await asyncio.to_thread(
            self._request_sync, method, path, body=body, timeout=timeout
        )

    # ── lifecycle ──────────────────────────────────────────────────────

    async def health(self) -> dict:
        return await self._request("GET", "/health", timeout=_HEALTH_TIMEOUT)

    def health_sync(self) -> dict:
        return self._request_sync("GET", "/health", timeout=_HEALTH_TIMEOUT)

    async def is_running(self) -> bool:
        try:
            await self.health()
            return True
        except DaemonError:
            return False

    # ── jobs ───────────────────────────────────────────────────────────

    async def list_jobs(self) -> list[dict]:
        return await self._request("GET", "/jobs")

    async def submit_command(
        self,
        argv: list[str],
        *,
        label: str,
        extra_env: Optional[dict] = None,
        chain_train: Optional[dict] = None,
        start: bool = True,
    ) -> dict:
        """Enqueue a ``command`` job (a plain ``python <argv>``).

        The WebUI submits every task (train / preprocess / mask / distill) as a
        command job wrapping ``[python, "tasks.py", command, *args]`` — the
        daemon owns the subprocess + serial queue + GPU guard. Returns the
        daemon's ``{"job_id": …, "state": "queued"}`` response.
        """
        body: dict[str, Any] = {
            "kind": "command",
            "label": label,
            "argv": list(argv),
            "start": start,
        }
        if extra_env:
            body["extra_env"] = extra_env
        if chain_train:
            body["chain_train"] = chain_train
        return await self._request("POST", "/jobs", body=body)

    async def submit_training(
        self,
        method: str,
        *,
        preset: str = "default",
        methods_subdir: Optional[str] = None,
        overrides: Optional[dict] = None,
        extra: Optional[list[str]] = None,
        start: bool = True,
    ) -> dict:
        """Enqueue a native ``train`` job (accelerate launch train.py).

        Used when the WebUI wants the daemon's built-in train path (it passes
        ``--progress_jsonl <job_dir>/progress.jsonl`` automatically) instead of
        a generic command job. Currently the WebUI uses ``submit_command`` for
        everything for parity with the old direct-subprocess shape; this is
        exposed for future direct-to-train submissions.
        """
        body: dict[str, Any] = {
            "method": method,
            "preset": preset,
            "start": start,
        }
        if methods_subdir:
            body["methods_subdir"] = methods_subdir
        if overrides:
            body["overrides"] = overrides
        if extra:
            body["extra"] = list(extra)
        return await self._request("POST", "/jobs", body=body)

    async def get_job(self, job_id: str) -> dict:
        return await self._request("GET", f"/jobs/{job_id}")

    def get_job_sync(self, job_id: str) -> dict:
        return self._request_sync("GET", f"/jobs/{job_id}")

    async def stop(self, job_id: str) -> dict:
        return await self._request("POST", f"/jobs/{job_id}/stop")

    def stop_sync(self, job_id: str) -> dict:
        return self._request_sync("POST", f"/jobs/{job_id}/stop")

    async def tail_log(self, job_id: str, lines: int = 80) -> dict:
        """Last N lines of a job's stdout (on-disk fallback; daemon-down safe)."""
        return await self._request("GET", f"/jobs/{job_id}/progress?last_n={lines}")

    async def get_progress(
        self,
        job_id: str,
        *,
        events: Optional[str] = None,
        since_step: Optional[int] = None,
        every_nth: Optional[int] = None,
        last_n: int = 200,
    ) -> dict:
        """Filtered view of a job's progress.jsonl (training jobs only)."""
        params = [f"last_n={last_n}"]
        if events:
            params.append(f"events={events}")
        if since_step is not None:
            params.append(f"since_step={since_step}")
        if every_nth is not None:
            params.append(f"every_nth={every_nth}")
        return await self._request(
            "GET", f"/jobs/{job_id}/progress?" + "&".join(params)
        )

    # ── queue ──────────────────────────────────────────────────────────

    async def pause_queue(self) -> dict:
        return await self._request("POST", "/queue/pause")

    def pause_queue_sync(self) -> dict:
        return self._request_sync("POST", "/queue/pause")

    async def start_queue(self) -> dict:
        return await self._request("POST", "/queue/start")

    def start_queue_sync(self) -> dict:
        return self._request_sync("POST", "/queue/start")

    async def shutdown(self, *, kill_jobs: bool = True) -> dict:
        return await self._request("POST", "/shutdown", body={"kill_jobs": kill_jobs})

    def shutdown_sync(self, *, kill_jobs: bool = True, timeout: float = _DEFAULT_TIMEOUT) -> dict:
        return self._request_sync("POST", "/shutdown", body={"kill_jobs": kill_jobs}, timeout=timeout)


# Module-level singleton — the WebUI process talks to one daemon.
daemon_client = DaemonClient()


async def ensure_daemon_running(timeout: float = 60.0) -> DaemonClient:
    """Auto-start a detached daemon if none is up; return a live client.

    Thin async wrapper over ``scripts.daemon.client.ensure_daemon`` (which does
    the pidfile discovery + detached spawn + /health ramp). Raises DaemonError
    if the daemon can't be reached within ``timeout``.
    """
    try:
        await asyncio.to_thread(_sync_ensure_daemon, timeout)
    except Exception as e:  # noqa: BLE001 — surface any spawn failure uniformly
        raise DaemonError(f"could not start the training daemon: {e}") from e
    # After ensure_daemon succeeds the singleton resolves to the right port.
    daemon_client._base = None  # force re-resolve (port may have drifted)
    return daemon_client


def _sync_ensure_daemon(timeout: float) -> None:
    from scripts.daemon import client as _dclient

    _dclient.ensure_daemon(timeout=timeout)
