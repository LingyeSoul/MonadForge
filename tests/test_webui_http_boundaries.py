"""HTTP input and static-serving regressions, isolated from user configuration."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from webui.api import preprocess


@pytest.fixture
def static_client(tmp_path, monkeypatch):
    from webui import server

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("frontend-index", encoding="utf-8")
    (dist / "logo.svg").write_text("local-logo", encoding="utf-8")
    sibling = tmp_path / "dist-backup"
    sibling.mkdir()
    (sibling / "private.txt").write_text("private-config", encoding="utf-8")
    (dist / "outside.txt").symlink_to(sibling / "private.txt")
    monkeypatch.setattr(server, "_DIST_DIR", dist)
    monkeypatch.setattr(server, "_INDEX_HTML", dist / "index.html")
    monkeypatch.setattr("library.config.io.migrate_custom_configs", lambda: None)
    # No lifespan: do not reconcile or stop the real daemon.
    return TestClient(server.create_app())


@pytest.mark.parametrize("path", ["/%2e%2e/dist-backup/private.txt", "/outside.txt"])
def test_spa_cannot_serve_sibling_or_symlink_outside_dist(static_client, path):
    response = static_client.get(path)
    assert response.status_code == 404
    assert "private-config" not in response.text


def test_spa_serves_assets_and_routes_but_not_unknown_api(static_client):
    assert static_client.get("/logo.svg").text == "local-logo"
    assert static_client.get("/config").text == "frontend-index"
    response = static_client.get("/api/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"]


@pytest.mark.parametrize(
    "patch",
    [
        {"sam": {"threshold": -0.1}},
        {"sam": {"threshold": 1.1}},
        {"sam": {"dilate": -1}},
        {"caption_shuffle_variants": -1},
        {"caption_tag_dropout_rate": 1.1},
        {"mit_text_threshold": -1},
        {"mit_dilate": -1},
    ],
)
def test_preprocess_rejects_invalid_ranges_before_persisting(monkeypatch, patch):
    calls = []
    monkeypatch.setattr(
        preprocess.svc, "save_settings", lambda data: calls.append(data)
    )
    app = FastAPI()
    app.include_router(preprocess.router)
    response = TestClient(app, raise_server_exceptions=False).put(
        "/settings", json=patch
    )
    assert response.status_code == 422
    assert calls == []


def test_preprocess_probability_boundaries_are_valid():
    assert (
        preprocess.PreprocessSettings(
            sam={"threshold": 0, "dilate": 0},
            caption_shuffle_variants=0,
            caption_tag_dropout_rate=1,
            mit_text_threshold=1,
            mit_dilate=0,
        ).caption_tag_dropout_rate
        == 1
    )


@pytest.mark.parametrize("field", ["caption_tag_dropout_rate", "mit_text_threshold"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_preprocess_rejects_nonfinite_probabilities(field, value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        preprocess.PreprocessSettings(**{field: value})


@pytest.fixture
def websocket_app(tmp_path, monkeypatch):
    from webui import server

    monkeypatch.setattr(server, "_DIST_DIR", tmp_path / "missing")
    monkeypatch.setattr("library.config.io.migrate_custom_configs", lambda: None)
    app = server.create_app()
    entries = []

    @app.websocket("/probe")
    async def probe(socket: WebSocket):
        entries.append(True)
        await socket.accept()
        await socket.send_text("connected")
        await socket.close()

    return app, entries


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "https://foreign.example"},
        {"origin": "http://testserver", "sec-fetch-site": "cross-site"},
    ],
)
def test_cross_site_websocket_cannot_subscribe(websocket_app, headers):
    app, entries = websocket_app
    with pytest.raises(WebSocketDisconnect) as failure:
        with TestClient(app).websocket_connect("/probe", headers=headers):
            pass
    assert failure.value.code == 1008
    assert entries == []


@pytest.mark.parametrize(
    "headers",
    [{}, {"origin": "http://testserver"}, {"origin": "http://127.0.0.1:8000"}],
)
def test_same_origin_and_nonbrowser_websocket_are_supported(websocket_app, headers):
    app, entries = websocket_app
    with TestClient(app).websocket_connect("/probe", headers=headers) as connection:
        assert connection.receive_text() == "connected"
    assert entries == [True]
