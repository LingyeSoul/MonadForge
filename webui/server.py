"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route

from webui.api import config as config_api
from webui.api import i18n as i18n_api
from webui.api import images as images_api
from webui.api import tasks as tasks_api
from webui.api import ws as ws_api

_DIST_DIR = Path(__file__).parent / "frontend" / "dist"
_INDEX_HTML = _DIST_DIR / "index.html"


_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


def _spa_fallback(request: Request):
    """Return index.html for any non-API, non-asset path."""
    return FileResponse(_INDEX_HTML, media_type="text/html")


def create_app(dev: bool = False) -> FastAPI:
    app = FastAPI(title="MonadForge WebUI", version="0.1.0")

    origins = ["*"] if dev else _ALLOWED_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routers
    app.include_router(config_api.router, prefix="/api/config")
    app.include_router(i18n_api.router, prefix="/api/i18n")
    app.include_router(images_api.router, prefix="/api/images")
    app.include_router(tasks_api.router, prefix="/api/tasks")
    app.include_router(ws_api.router)

    # Serve Vue SPA in production
    if _DIST_DIR.is_dir() and _INDEX_HTML.is_file():
        # Static assets (JS, CSS, fonts) — must be mounted BEFORE the SPA
        # catch-all so /assets/* requests resolve to real files first.
        app.mount(
            "/assets", StaticFiles(directory=str(_DIST_DIR / "assets")), name="assets"
        )

        # SPA catch-all: every other non-API path returns index.html so that
        # vue-router handles client-side routing (/config, /dataset, …).
        spa_app = Starlette(
            routes=[
                Route("/", _spa_fallback),
                Route("/{path:path}", _spa_fallback),
            ]
        )
        app.mount("/", spa_app)

    return app
