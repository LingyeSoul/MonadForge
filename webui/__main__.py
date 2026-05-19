"""Entry point: ``python -m webui [--dev] [--host 0.0.0.0] [--port 8000]``."""

from __future__ import annotations

import argparse
from pathlib import Path


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
