"""Regression tests for the same-origin CSRF middleware behind port-forwarding
gateways (cloud GPU mirrors: AutoDL 自定义服务, 优云智算, plain reverse proxies).

The gateway terminates TLS and forwards to the container's plain-HTTP
listener, so the server's raw ``scheme://host`` view is an internal URL that
never matches the browser's ``Origin`` — every state-changing request
(including training submission) came back 403. The middleware now rebuilds
the browser-visible origin from ``X-Forwarded-*`` (trusted only from
loopback/private peers, so direct-internet clients can't spoof past CSRF)
and accepts an explicit ``ANIMA_WEBUI_ALLOWED_ORIGINS`` escape hatch for
gateways that rewrite Host without forwarding headers.

Probing strategy: POST to a non-API path that falls through to the SPA
catch-all (405) — anything the router itself returns means the CSRF gate let
it through; the gate's own verdict is always 403. No business endpoint
dependencies, no daemon, no static files.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

PROVIDER_ORIGIN = "https://gpu-8000.example-provider.com"


def _build_app(monkeypatch):
    """create_app() with lifespan daemon calls + config migration stubbed."""

    async def _noop() -> None:
        return None

    monkeypatch.setattr("webui.server.task_service.reconcile_daemon_jobs", _noop)
    monkeypatch.setattr("webui.server.task_service.close", _noop)
    monkeypatch.setattr("library.config.io.migrate_custom_configs", lambda: None)

    from webui.server import create_app

    return create_app()


def _post(app, *, origin=None, headers=None, client_addr=("testclient", 50000)):
    hdrs = dict(headers or {})
    if origin is not None:
        hdrs["Origin"] = origin
    with TestClient(app, client=client_addr) as client:
        return client.post("/api/__origin_probe__", headers=hdrs)


def test_direct_same_origin_allowed(monkeypatch):
    # Browser talks straight to uvicorn: Origin matches scheme://Host.
    resp = _post(_build_app(monkeypatch), origin="http://testserver")
    assert resp.status_code != 403  # passed the gate, reached the router


def test_no_origin_header_allowed(monkeypatch):
    # curl / server-to-server callers send no Origin at all — never blocked.
    resp = _post(_build_app(monkeypatch))
    assert resp.status_code != 403


def test_unknown_cross_origin_rejected(monkeypatch):
    resp = _post(_build_app(monkeypatch), origin=PROVIDER_ORIGIN)
    assert resp.status_code == 403


def test_gateway_forwarded_headers_allowed(monkeypatch):
    # Cloud-mirror gateway: TLS terminates there, private-network peer,
    # browser Origin is the public https URL the gateway serves.
    resp = _post(
        _build_app(monkeypatch),
        origin=PROVIDER_ORIGIN,
        headers={
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "gpu-8000.example-provider.com",
        },
        client_addr=("172.17.0.5", 40000),
    )
    assert resp.status_code != 403


def test_gateway_proto_only_allowed(monkeypatch):
    # Gateway passes the real Host through but rewrites the scheme.
    resp = _post(
        _build_app(monkeypatch),
        origin=PROVIDER_ORIGIN,
        headers={
            "Host": "gpu-8000.example-provider.com",
            "X-Forwarded-Proto": "https",
        },
        client_addr=("127.0.0.1", 40000),
    )
    assert resp.status_code != 403


def test_public_peer_cannot_spoof_forwarded_headers(monkeypatch):
    # Attacker hitting an exposed port directly: public peer + self-set
    # X-Forwarded-* must NOT relax the check. (203.0.113.x won't do here —
    # ipaddress counts TEST-NET ranges as is_private.)
    resp = _post(
        _build_app(monkeypatch),
        origin=PROVIDER_ORIGIN,
        headers={
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "gpu-8000.example-provider.com",
        },
        client_addr=("8.8.8.8", 40000),
    )
    assert resp.status_code == 403


def test_env_allowlist(monkeypatch):
    # Gateways that rewrite Host without any X-Forwarded-* leave nothing to
    # reconstruct from — the operator lists the public origin explicitly.
    monkeypatch.setenv("ANIMA_WEBUI_ALLOWED_ORIGINS", f"garbage,,{PROVIDER_ORIGIN}/")
    resp = _post(_build_app(monkeypatch), origin=PROVIDER_ORIGIN)
    assert resp.status_code != 403


def test_env_allowlist_does_not_open_other_origins(monkeypatch):
    monkeypatch.setenv("ANIMA_WEBUI_ALLOWED_ORIGINS", PROVIDER_ORIGIN)
    resp = _post(_build_app(monkeypatch), origin="https://evil.example.com")
    assert resp.status_code == 403


def test_cors_preflight_accepts_allowlisted_origin(monkeypatch):
    monkeypatch.setenv("ANIMA_WEBUI_ALLOWED_ORIGINS", PROVIDER_ORIGIN)
    with TestClient(_build_app(monkeypatch)) as client:
        resp = client.options(
            "/api/tasks",
            headers={
                "Origin": PROVIDER_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )
    assert resp.headers["access-control-allow-origin"] == PROVIDER_ORIGIN
