"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route

from webui.api import config as config_api
from webui.api import distill as distill_api
from webui.api import docs as docs_api
from webui.api import files as files_api
from webui.api import i18n as i18n_api
from webui.api import images as images_api
from webui.api import merge as merge_api
from webui.api import models as models_api
from webui.api import preprocess as preprocess_api
from webui.api import preview as preview_api
from webui.api import system as system_api
from webui.api import staged_resolution as staged_resolution_api
from webui.api import tagger as tagger_api
from webui.api import tasks as tasks_api
from webui.api import ws as ws_api
from webui.services.task_service import task_service

_DIST_DIR = Path(__file__).parent / "frontend" / "dist"
_INDEX_HTML = _DIST_DIR / "index.html"


_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _origin_is_allowed(request: Request, origin: str) -> bool:
    """Accept configured dev origins and the WebUI's actual serving origin."""
    if origin in _ALLOWED_ORIGINS:
        return True
    host = request.headers.get("host")
    if not host:
        return False
    return origin == f"{request.url.scheme}://{host}"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await task_service.reconcile_daemon_jobs()
    try:
        yield
    finally:
        await task_service.close()


def _spa_fallback(request: Request):
    """Serve real files from dist/ first; fall back to index.html for SPA routes."""
    # Strip leading slash and resolve against dist root
    rel = request.url.path.lstrip("/")
    candidate = (_DIST_DIR / rel).resolve()
    # Only serve if the resolved path is inside dist/ (prevent path traversal)
    if candidate.is_file() and str(candidate).startswith(str(_DIST_DIR.resolve())):
        return FileResponse(candidate)
    return FileResponse(_INDEX_HTML, media_type="text/html")


def create_app(dev: bool = False) -> FastAPI:
    # Migrate legacy custom config layout on startup
    from library.config.io import migrate_custom_configs

    migrate_custom_configs()

    app = FastAPI(title="MonadForge WebUI", version="0.1.0", lifespan=_lifespan)

    origins = ["*"] if dev else _ALLOWED_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def enforce_same_origin(request: Request, call_next):
        if request.method.upper() in _UNSAFE_METHODS:
            fetch_site = request.headers.get("sec-fetch-site", "").lower()
            origin = request.headers.get("origin")
            if fetch_site == "cross-site" or (
                origin is not None and not _origin_is_allowed(request, origin)
            ):
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "Cross-origin state-changing requests are forbidden"
                    },
                )
        return await call_next(request)

    # API routers
    app.include_router(config_api.router, prefix="/api/config")
    app.include_router(distill_api.router, prefix="/api/distill")
    app.include_router(docs_api.router, prefix="/api/docs")
    app.include_router(files_api.router, prefix="/api/files")
    app.include_router(i18n_api.router, prefix="/api/i18n")
    app.include_router(images_api.router, prefix="/api/images")
    app.include_router(merge_api.router, prefix="/api/merge")
    app.include_router(models_api.router, prefix="/api/models")
    app.include_router(preprocess_api.router, prefix="/api/preprocess")
    app.include_router(preview_api.router, prefix="/api/preview")
    app.include_router(system_api.router, prefix="/api/system")
    app.include_router(staged_resolution_api.router, prefix="/api/staged-resolution")
    app.include_router(tagger_api.router, prefix="/api/tagger")
    app.include_router(tasks_api.router, prefix="/api/tasks")
    app.include_router(ws_api.router)

    # Serve Vue SPA in production
    if _DIST_DIR.is_dir() and _INDEX_HTML.is_file():
        # Static assets (JS, CSS, fonts) — must be mounted BEFORE the SPA
        # catch-all so /assets/* requests resolve to real files first.
        app.mount(
            "/assets", StaticFiles(directory=str(_DIST_DIR / "assets")), name="assets"
        )

        # SPA catch-all: serves real files from dist/ (favicon.svg, logo.svg,
        # etc.) when they exist, otherwise returns index.html for vue-router
        # client-side routing (/config, /dataset, …).
        spa_app = Starlette(
            routes=[
                Route("/", _spa_fallback),
                Route("/{path:path}", _spa_fallback),
            ]
        )
        app.mount("/", spa_app)

    return app
