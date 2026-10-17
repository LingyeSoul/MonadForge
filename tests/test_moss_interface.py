"""MOSS control-plane validation; no tensors, models or worker processes."""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.moss.workflow import parse_workflow
from webui.api.moss import router, task_service
from webui.services import moss_service


def test_moss_http_schema_strict_inputs_and_runtime(monkeypatch):
    calls = []

    async def start(command, args, **kwargs):
        calls.append((command, args, kwargs))
        return SimpleNamespace(id="moss-fixture", command=command)

    monkeypatch.setattr(task_service, "start_task", start)
    app = FastAPI()
    app.include_router(router, prefix="/moss")
    with TestClient(app) as client:
        response = client.get("/moss/schema")
        assert response.status_code == 200
        assert set(response.json()) == {"cache", "train", "generate", "tune"}
        invalid = client.post("/moss/jobs/train", json={"values": {"rank": "16"}})
        assert invalid.status_code == 400
        assert not calls
        submitted = client.post(
            "/moss/jobs/train", json={"values": {"rank": 32}, "runtime_profile": "v100"}
        )
        assert submitted.status_code == 200
    assert calls[0][0] == "moss-train"
    assert calls[0][2] == {
        "context": {"model_type": "moss", "phase": "train"},
        "runtime_profile": "v100",
    }


def test_moss_chain_pins_train_to_cache_output():
    cache, train = parse_workflow(
        [
            "--cache-args",
            json.dumps(["--out", "output/moss/new-cache"]),
            "--train-args",
            json.dumps(["--cache", "output/moss/old-cache"]),
        ]
    )
    assert train.cache == cache.out == "output/moss/new-cache"


def test_moss_manifest_rejects_non_object_before_request_parsing(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(moss_service, "output_path", lambda value: tmp_path)
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        moss_service.read_results(str(tmp_path))
