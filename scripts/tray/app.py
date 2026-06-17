"""MonadForge daemon tray — a Windows status indicator + controller.

A small ``pystray`` system-tray app that reflects the local training daemon's
state (idle / running / error / down) and exposes the handful of control
actions a user wants without opening the WebUI: open the WebUI, pause/resume
the queue, stop the active job, restart the daemon, quit.

The tray is a *separate* process from the WebUI and the daemon — it is a
localhost client like any other (it speaks the daemon's HTTP surface via
``webui.services.daemon_client``). Run it with ``pythonw -m scripts.tray`` so
no console window flashes; the daemon itself is started on demand via
``ensure_daemon``.

State polling happens on a background thread (pystray's Win32 backend runs its
own message loop on the main thread); each tick re-resolves daemon health +
the active job and updates the icon/tooltip/menu.
"""

from __future__ import annotations

import logging
import threading
import webbrowser
from typing import Optional

logger = logging.getLogger(__name__)

POLL_INTERVAL = 1.5  # seconds between daemon health/job polls
WEBUI_URL = "http://127.0.0.1:8000"

# Daemon job states we treat as terminal (for "last error" tooltip retention).
_TERMINAL = {"done", "error", "stopped"}


class TrayApp:
    """Owns the pystray Icon, the polling thread, and the menu actions."""

    def __init__(self) -> None:
        # Lazy imports: pystray is Windows-only here and we want a clean error
        # if the module is imported on a platform without the backend.
        import pystray
        from PIL import Image

        self._Image = Image
        self._pystray = pystray
        self._icon: Optional[pystray.Icon] = None
        self._stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        # Last known state for tooltip + flicker animation.
        self._state = "down"
        self._tooltip = "MonadForge — daemon not running"
        self._active_job: Optional[dict] = None
        self._last_error: Optional[str] = None
        self._frame = 0
        self._paused = False

    # ── icon / tooltip ─────────────────────────────────────────────────

    def _render_icon(self):
        from scripts.tray.icons import icon_for

        return icon_for(self._state, frame=self._frame)

    def _refresh_icon(self) -> None:
        if self._icon is None:
            return
        try:
            self._icon.icon = self._render_icon()
            self._icon.title = self._tooltip
        except Exception:  # noqa: BLE001 — never let a render bug kill the tray
            logger.exception("failed to refresh tray icon")

    # ── polling ────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background thread: resolve daemon state + active job each tick."""
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001 — keep polling
                logger.exception("tray poll tick failed")
            # Tick the flicker frame on each loop while running (0↔1).
            if self._state == "running":
                self._frame = 1 - self._frame
            self._stop.wait(POLL_INTERVAL)

    def _tick(self) -> None:
        from webui.services.daemon_client import DaemonError, daemon_client

        # Health first — if the daemon is down, show that and bail.
        try:
            health = daemon_client.health_sync()
        except DaemonError:
            self._state = "down"
            self._tooltip = "MonadForge — daemon not running"
            self._active_job = None
            self._refresh_icon()
            return

        self._paused = bool(health.get("paused"))
        active_id = health.get("active_job")

        if active_id:
            # Resolve the active job for a richer tooltip (label + step if any).
            try:
                job = daemon_client.get_job_sync(active_id)
            except DaemonError:
                job = {}
            self._active_job = job
            self._state = "running"
            label = job.get("method") or job.get("label") or "job"
            step_hint = ""
            latest = (
                (job.get("latest") or {}).get("data")
                if isinstance(job.get("latest"), dict)
                else None
            )
            if isinstance(latest, dict) and "global_step" in latest:
                step_hint = f" (step {latest['global_step']})"
            self._tooltip = f"MonadForge — running {label}{step_hint}"
        else:
            self._active_job = None
            # Retain an error indicator briefly after a job fails, else idle.
            if self._last_error:
                self._state = "error"
                self._tooltip = (
                    f"MonadForge — last job errored: {self._last_error[:48]}"
                )
            else:
                self._state = "idle"
                suffix = " (paused)" if self._paused else ""
                self._tooltip = f"MonadForge — idle{suffix}"
        self._refresh_icon()

    # ── menu actions ───────────────────────────────────────────────────

    def _open_webui(self) -> None:
        try:
            webbrowser.open(WEBUI_URL)
        except Exception:  # noqa: BLE001
            logger.exception("failed to open WebUI")

    def _toggle_queue(self) -> None:
        from webui.services.daemon_client import DaemonError, daemon_client

        try:
            if self._paused:
                daemon_client.start_queue_sync()
            else:
                daemon_client.pause_queue_sync()
        except DaemonError:
            logger.exception("queue toggle failed")

    def _stop_active_job(self) -> None:
        from webui.services.daemon_client import DaemonError, daemon_client

        if not self._active_job:
            return
        job_id = self._active_job.get("id")
        if not job_id:
            return
        try:
            daemon_client.stop_sync(job_id)
        except DaemonError:
            logger.exception("stop job failed")

    def _restart_daemon(self) -> None:
        from webui.services.daemon_client import DaemonError, daemon_client

        try:
            daemon_client.shutdown_sync(kill_jobs=False)
        except DaemonError:
            pass  # was already down — fine
        try:
            from webui.services.daemon_client import ensure_daemon_running

            # ensure_daemon_running is async; run it on a throwaway loop here.
            import asyncio

            asyncio.run(ensure_daemon_running(timeout=30))
        except (DaemonError, RuntimeError):
            logger.exception("daemon restart failed")

    def _quit(self) -> None:
        self._stop.set()
        if self._icon is not None:
            self._icon.stop()

    # ── menu ───────────────────────────────────────────────────────────

    def _build_menu(self):
        MenuItem = self._pystray.MenuItem

        def _state_label(_item):
            return self._tooltip

        def _queue_label(_item):
            return "Resume queue" if self._paused else "Pause queue"

        def _stop_enabled(_item):
            return self._active_job is not None

        return self._pystray.Menu(
            MenuItem(_state_label, None, enabled=False),
            self._pystray.Menu.SEPARATOR,
            MenuItem("Open WebUI", lambda: self._open_webui(), default=True),
            MenuItem(_queue_label, lambda: self._toggle_queue()),
            MenuItem(
                "Stop active job",
                lambda: self._stop_active_job(),
                enabled=_stop_enabled,
            ),
            MenuItem("Restart daemon", lambda: self._restart_daemon()),
            self._pystray.Menu.SEPARATOR,
            MenuItem("Quit", lambda: self._quit()),
        )

    # ── lifecycle ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the polling thread + the pystray icon (blocks until quit)."""
        self._icon = self._pystray.Icon(
            "MonadForge",
            self._render_icon(),
            self._tooltip,
            menu=self._build_menu(),
        )
        self._poll_thread = threading.Thread(
            target=self._poll_loop, name="mf-tray-poll", daemon=True
        )
        self._poll_thread.start()
        # Best-effort: bring the daemon up so the tray shows a live state.
        try:
            from webui.services.daemon_client import ensure_daemon_running
            import asyncio

            asyncio.run(ensure_daemon_running(timeout=30))
        except Exception:  # noqa: BLE001 — tray stays up, just shows 'down'
            logger.warning("could not ensure daemon at tray startup")
        self._icon.run()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    TrayApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
