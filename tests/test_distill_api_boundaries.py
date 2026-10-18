"""Distillation config persistence and HTTP serialization, without training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tomlkit
from fastapi import FastAPI
from fastapi.testclient import TestClient

from webui.api import distill
from webui.services import distill_service as svc


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "_ROOT", tmp_path)
    path = tmp_path / "configs/methods/turbo.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        '# Keep the header\noutput_name = "demo" # name comment\niterations = 10\n'
        "enabled = false\n[optim]\nlr = 0.01 # learning rate\n"
        "[schedule]\nstages = [0.5, 1.0]\n",
        encoding="utf-8",
    )
    app = FastAPI()
    app.include_router(distill.router)
    with TestClient(app) as client:
        yield client, path


def test_method_experimental_flag_survives_http_response(workspace):
    client, _ = workspace
    response = client.get("/methods")
    assert response.status_code == 200
    assert response.json()[0]["experimental"] is True


def test_config_serializes_and_save_preserves_comments_and_numeric_types(workspace):
    client, path = workspace
    response = client.get("/config", params={"method": "turbo"})
    assert response.status_code == 200
    fields = {
        section["name"]: {field["key"]: field["value"] for field in section["fields"]}
        for section in response.json()["sections"]
    }
    assert fields["root"]["enabled"] is False
    assert fields["schedule"]["stages"] == [0.5, 1.0]
    response = client.put(
        "/config?method=turbo",
        json={
            "updates": {
                "root": {"iterations": 12},
                "optim": {"lr": 1},
                "schedule": {"stages": [0.25, 0.5, 1]},
            }
        },
    )
    assert response.status_code == 200
    saved = path.read_text()
    assert "# Keep the header" in saved and "# learning rate" in saved
    doc = tomlkit.parse(saved)
    assert doc["iterations"] == 12 and doc["optim"]["lr"] == 1.0
    assert isinstance(doc["optim"]["lr"], float)
    assert doc["schedule"]["stages"] == [0.25, 0.5, 1.0]


@pytest.mark.parametrize(
    "updates",
    [
        {"missing": {"iterations": 12}},
        {"optim": {"typo_lr": 0.1}},
        {"root": {"optim": {"lr": 0.1}}},
        {"root": {"iterations": "12"}},
        {"root": {"iterations": 12.5}},
        {"root": {"iterations": True}},
        {"root": {"enabled": 1}},
        {"optim": {"lr": False}},
        {"optim": {"lr": float("nan")}},
        {"optim": {"lr": float("inf")}},
        {"schedule": {"stages": [0.5, "1.0"]}},
        {"schedule": {"stages": [0.5, float("nan")]}},
        {"root": {"iterations": 99}, "optim": {"typo_lr": 0.1}},
    ],
)
def test_invalid_updates_return_error_and_do_not_change_file(workspace, updates):
    client, path = workspace
    before = path.read_bytes()
    response = client.put(
        "/config?method=turbo",
        content=json.dumps({"updates": updates}),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400, response.text
    assert "detail" in response.json()
    assert path.read_bytes() == before


def test_atomic_replace_failure_returns_json_error_and_preserves_config(
    workspace, monkeypatch
):
    client, path = workspace
    before = path.read_bytes()
    original_replace = Path.replace

    def fail_replace(source, target):
        if source.suffix == ".tmp":
            raise OSError("fixture: disk replacement failed")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    response = client.put(
        "/config?method=turbo", json={"updates": {"root": {"iterations": 20}}}
    )
    assert response.status_code == 500
    assert "replacement failed" in response.json()["detail"]
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_missing_config_returns_not_found(workspace):
    client, path = workspace
    path.unlink()
    assert client.get("/config?method=turbo").status_code == 404
    assert client.put("/config?method=turbo", json={"updates": {}}).status_code == 404


def test_nonfinite_stored_value_is_reported_before_response_serialization(workspace):
    client, path = workspace
    path.write_text("learning_rate = nan\n")
    response = client.get("/config?method=turbo")
    assert response.status_code == 400
    assert "finite" in response.json()["detail"]


@pytest.mark.parametrize("method", ["turbo", "spd"])
@pytest.mark.parametrize("lang", ["cn", "en", "ja", "ko"])
def test_shipped_distillation_guides_are_available(method, lang):
    assert svc.load_guide(method, lang).strip()


def test_invalid_guide_language_falls_back_without_reading_other_directory(
    tmp_path, monkeypatch
):
    guides = tmp_path / "guides"
    (guides / "en").mkdir(parents=True)
    (guides / "en/turbo.html").write_text("English guide")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside/turbo.html").write_text("Must not be read")
    monkeypatch.setattr(svc, "_GUIDES_DIR", guides)
    assert svc.load_guide("turbo", "../outside") == "English guide"


def test_unknown_guide_method_is_rejected_by_http(workspace):
    client, _ = workspace
    response = client.get("/guide", params={"method": "../escape"})
    assert response.status_code == 400


@pytest.mark.parametrize("method", ["spd", "turbo"])
def test_shipped_config_http_round_trip_uses_exact_field_names(
    tmp_path, monkeypatch, method
):
    relative = svc._DISTILL_METHODS[method]["path"]
    content = (svc._ROOT / relative).read_text(encoding="utf-8")
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(svc, "_ROOT", tmp_path)
    app = FastAPI()
    app.include_router(distill.router)
    with TestClient(app) as client:
        response = client.get("/config", params={"method": method})
        assert response.status_code == 200
        updates = {
            section["name"]: {
                field["key"]: field["value"] for field in section["fields"]
            }
            for section in response.json()["sections"]
        }
        saved = client.put(
            "/config", params={"method": method}, json={"updates": updates}
        )
        assert saved.status_code == 200, saved.text
    assert tomlkit.parse(path.read_text()).unwrap() == tomlkit.parse(content).unwrap()
