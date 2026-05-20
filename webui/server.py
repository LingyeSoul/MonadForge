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
from webui.api import docs as docs_api
from webui.api import files as files_api
from webui.api import i18n as i18n_api
from webui.api import images as images_api
from webui.api import merge as merge_api
from webui.api import preprocess as preprocess_api
from webui.api import system as system_api
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
    # Migrate legacy custom config layout on startup
    from library.config.io import migrate_custom_configs

    migrate_custom_configs()

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
    app.include_router(docs_api.router, prefix="/api/docs")
    app.include_router(files_api.router, prefix="/api/files")
    app.include_router(i18n_api.router, prefix="/api/i18n")
    app.include_router(images_api.router, prefix="/api/images")
    app.include_router(merge_api.router, prefix="/api/merge")
    app.include_router(preprocess_api.router, prefix="/api/preprocess")
    app.include_router(system_api.router, prefix="/api/system")
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
