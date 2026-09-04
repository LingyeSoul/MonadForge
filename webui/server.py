"""FastAPI application factory."""

from __future__ import annotations

import ipaddress
import os
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


def _extra_allowed_origins() -> list[str]:
    """Extra browser origins accepted via ``ANIMA_WEBUI_ALLOWED_ORIGINS``.

    Cloud-mirror gateways that rewrite ``Host`` without setting any
    ``X-Forwarded-*`` header leave the server no way to reconstruct the
    public URL, so operators list those origins explicitly (comma-separated).
    """
    raw = os.environ.get("ANIMA_WEBUI_ALLOWED_ORIGINS", "")
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


def _peer_may_forward(request: Request) -> bool:
    """Whether the direct peer is allowed to set ``X-Forwarded-*`` headers.

    Only loopback/private peers are trusted: a cloud-mirror gateway connects
    from inside the container's own network, while a client hitting an
    exposed port directly is public — its spoofed forwarded headers must not
    relax the CSRF check.
    """
    if request.client is None:
        return False
    try:
        ip = ipaddress.ip_address(request.client.host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def _serving_origin(request: Request) -> str | None:
    """The origin the browser actually sees, honoring proxy rewrite headers.

    TLS-terminating gateways (cloud port forwarding, local reverse proxies)
    talk plain HTTP to us and often rewrite ``Host``, so the raw
    ``scheme://host`` pair is an internal URL that never matches the
    browser's ``Origin``. Rebuild the browser-visible origin from the
    forwarded headers instead, falling back to the wire values.
    """
    host = request.headers.get("host")
    if not host:
        return None
    scheme = request.url.scheme
    if _peer_may_forward(request):
        # First entry wins: the outermost hop is the one the browser used.
        proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
        fwd_host = request.headers.get("x-forwarded-host", "").split(",")[0].strip()
        if proto:
            scheme = proto
        if fwd_host:
            host = fwd_host
    return f"{scheme}://{host}"


def _origin_is_allowed(request: Request, origin: str, allowed: list[str]) -> bool:
    """Accept configured origins and the WebUI's actual serving origin."""
    return origin in allowed or origin == _serving_origin(request)


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

    allowed_origins = _ALLOWED_ORIGINS + _extra_allowed_origins()

    origins = ["*"] if dev else allowed_origins
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
                origin is not None
                and not _origin_is_allowed(request, origin, allowed_origins)
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
