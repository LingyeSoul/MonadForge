"""Regression checks for issues found through the configuration workbench."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from webui.api.config import router
from webui.services import config_service


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(config_service, "_SETTINGS_FILE", tmp_path / "settings.json")
    app = FastAPI()
    app.include_router(router, prefix="/api/config")
    with TestClient(app) as test_client:
        yield test_client


def test_wandb_settings_get_is_not_shadowed_by_layer_route(client):
    response = client.get("/api/config/wandb-settings")
    assert response.status_code == 200
    assert response.json()["enabled"] is False


def test_wandb_settings_put_is_not_shadowed_by_layer_route(client):
    response = client.put(
        "/api/config/wandb-settings", json={"enabled": True, "project": "test-project"}
    )
    assert response.status_code == 200
    assert response.json()["project"] == "test-project"
    assert client.get("/api/config/wandb-settings").json()["enabled"] is True


@pytest.mark.parametrize("lang", ["en", "cn", "ja", "ko"])
@pytest.mark.parametrize("method", ["lora", "chimera", "easycontrol"])
def test_method_guides_load_from_shipped_locale_directories(method, lang):
    assert config_service._load_method_guide(method, lang).strip()


def test_guide_locale_directory_takes_priority_with_legacy_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(config_service, "_GUIDES_DIR", tmp_path)
    (tmp_path / "en").mkdir()
    (tmp_path / "lora.en.html").write_text("legacy", encoding="utf-8")
    (tmp_path / "en" / "lora.html").write_text("current", encoding="utf-8")
    assert config_service._read_guide("lora", "en") == "current"
    assert config_service._read_guide("lora", "ja") == "current"
    (tmp_path / "en" / "lora.html").unlink()
    assert config_service._read_guide("lora", "en") == "legacy"


def test_invalid_guide_locale_uses_english():
    assert config_service._read_guide("lora", "../../outside") == config_service._read_guide(
        "lora", "en"
    )
