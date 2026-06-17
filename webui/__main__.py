"""Entry point: ``python -m webui [--dev] [--host 0.0.0.0] [--port 8000]``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("webui")


def _ensure_daemon() -> None:
    """Best-effort: bring up the training daemon so the WebUI can submit jobs.

    The WebUI's TaskService submits every task to the daemon (serial queue +
    GPU guard + persistence). If the daemon can't start the WebUI still serves
    config/dataset browsing, but starting a task will surface a clear error.
    Failures here are warnings only — never block the WebUI from booting.
    """
    try:
        from scripts.daemon import client as _dclient

        _dclient.ensure_daemon(timeout=30)
        logger.info("training daemon ready at %s", _dclient.DaemonClient().base)
    except Exception as e:  # noqa: BLE001 — boot must not depend on the daemon
        logger.warning(
            "could not start the training daemon (%s). Tasks will fail to "
            "submit until it is up — run `python tasks.py daemon`.",
            e,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MonadForge WebUI server")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable dev mode (auto-reload, no static file serving)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    _ensure_daemon()

    import uvicorn

    if args.dev:
        uvicorn.run(
            "webui.server:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(Path(__file__).parent)],
        )
    else:
        from webui.server import create_app

        app = create_app(dev=False)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
