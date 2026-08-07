"""Cooperative training stop control shared by CLI and daemon jobs."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


class StopController:
    """Set-only stop signal checked at optimizer-step boundaries.

    POSIX jobs receive SIGINT/SIGTERM from the daemon and Windows jobs receive
    a stop-file because hidden ``CREATE_NO_WINDOW`` processes cannot reliably
    consume console control events.  The handlers never raise from a CUDA
    callback; the training loop decides when it is safe to save and exit.
    """

    def __init__(
        self,
        stop_file: str | os.PathLike | None = None,
        *,
        poll_interval: float = 0.1,
        install_signals: bool = True,
    ) -> None:
        self.stop_file = Path(stop_file) if stop_file else None
        self.poll_interval = max(0.02, float(poll_interval))
        self._event = threading.Event()
        self._reason: str | None = None
        self._old_handlers: dict[int, object] = {}
        self._watcher: threading.Thread | None = None
        self._closed = False
        if install_signals:
            self.install_signal_handlers()
        if self.stop_file is not None:
            self._watcher = threading.Thread(
                target=self._watch_stop_file,
                name="monadforge-stop-file",
                daemon=True,
            )
            self._watcher.start()

    @classmethod
    def from_environment(cls, *, install_signals: bool = True) -> "StopController":
        return cls(
            os.environ.get("ANIMA_DAEMON_STOP_FILE"),
            install_signals=install_signals,
        )

    def install_signal_handlers(self, signals: Iterable[int] | None = None) -> None:
        if signals is None:
            signals = [
                getattr(signal, name)
                for name in ("SIGINT", "SIGTERM", "SIGBREAK")
                if hasattr(signal, name)
            ]
        for signum in signals:
            try:
                self._old_handlers[int(signum)] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
            except (ValueError, OSError, RuntimeError):
                # Only the main thread may install handlers; worker/test code
                # should still be able to construct a controller.
                continue

    def _handle_signal(self, signum, _frame) -> None:
        self.request(f"signal:{getattr(signum, 'name', signum)}")

    def _watch_stop_file(self) -> None:
        while not self._closed and not self._event.is_set():
            try:
                if self.stop_file is not None and self.stop_file.exists():
                    self.request("stop-file")
                    return
            except OSError as exc:
                logger.warning("stop-file probe failed: %s", exc)
            time.sleep(self.poll_interval)

    def request(self, reason: str = "requested") -> None:
        if not self._event.is_set():
            self._reason = str(reason)
            self._event.set()
            logger.info("cooperative stop requested (%s)", self._reason)

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def clear(self) -> None:
        self._reason = None
        self._event.clear()

    def close(self) -> None:
        self._closed = True
        for signum, handler in self._old_handlers.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError, RuntimeError):
                pass
        self._old_handlers.clear()

    def __enter__(self) -> "StopController":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def stop_requested_from_env() -> bool:
    """Cheap probe for code that cannot own a controller instance."""

    path = os.environ.get("ANIMA_DAEMON_STOP_FILE")
    return bool(path and Path(path).exists())

