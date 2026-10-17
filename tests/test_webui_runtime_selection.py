"""Runtime selection endpoints with all job submission and request work mocked."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from library.runtime import profiles
from webui.api import qwen21, system, tasks


@pytest.fixture
def api(monkeypatch):
    submitted = []

    async def start(command, args=None, env=None, **kwargs):
        submitted.append({"command": command, "args": args, "env": env, **kwargs})
        return SimpleNamespace(id="fixture-job", command=command,
                               info=lambda: {"task_id": "fixture-job", "command": command})

    monkeypatch.setattr(tasks, "_COMMAND_DESCRIPTIONS", {"fixture": "fixture"})
    monkeypatch.setattr(tasks.task_service, "start_task", start)
    monkeypatch.setattr(qwen21.task_service, "start_task", start)
    monkeypatch.setattr(qwen21.svc, "request_from_values", lambda *_: SimpleNamespace(to_argv=lambda: ["--fixture"]))
    app = FastAPI()
    app.include_router(tasks.router, prefix="/tasks")
    app.include_router(qwen21.router, prefix="/qwen21")
    app.include_router(system.router, prefix="/system")
    with TestClient(app) as client:
        yield client, submitted


ENDPOINTS = [
    ("/tasks", {"command": "fixture"}),
    ("/qwen21/jobs/cache", {"values": {}}),
    ("/qwen21/chain", {"cache": {"values": {}}, "train": {"values": {}}}),
]


@pytest.mark.parametrize("runtime", ["default", "curation", "v100"])
@pytest.mark.parametrize("endpoint,payload", ENDPOINTS)
def test_named_runtime_is_forwarded_to_existing_task_service(api, runtime, endpoint, payload):
    client, calls = api
    response = client.post(endpoint, json={**payload, "runtime_profile": runtime})
    assert response.status_code == 200, response.text
    assert len(calls) == 1 and calls[0]["runtime_profile"] == runtime
    assert "python_executable" not in calls[0] and "runtime_env" not in calls[0]


@pytest.mark.parametrize("runtime", ["unknown", "/tmp/custom-python", "../../bin/python"])
@pytest.mark.parametrize("endpoint,payload", ENDPOINTS)
def test_browser_cannot_supply_unknown_profile_or_interpreter_path(api, runtime, endpoint, payload):
    client, calls = api
    assert client.post(endpoint, json={**payload, "runtime_profile": runtime}).status_code == 422
    assert calls == []


@pytest.mark.parametrize("field", ["python_executable", "runtime_python", "runtime_env"])
@pytest.mark.parametrize("endpoint,payload", ENDPOINTS)
def test_browser_cannot_override_registered_interpreter_or_environment(api, field, endpoint, payload):
    client, calls = api
    value = {"PYTHONHOME": "/tmp"} if field == "runtime_env" else "/tmp/python"
    assert client.post(endpoint, json={**payload, field: value}).status_code == 422
    assert calls == []


@pytest.mark.parametrize("endpoint,payload", ENDPOINTS)
def test_missing_registration_is_reported_as_an_actionable_error(api, monkeypatch, endpoint, payload):
    client, calls = api

    async def unavailable(*args, **kwargs):
        raise profiles.RuntimeProfileError("V100 runtime is not registered in configs/custom/runtime.toml")

    monkeypatch.setattr(tasks.task_service, "start_task", unavailable)
    monkeypatch.setattr(qwen21.task_service, "start_task", unavailable)
    response = client.post(endpoint, json={**payload, "runtime_profile": "v100"})
    assert response.status_code == 400
    assert "not registered" in response.json()["detail"]
    assert calls == []


def test_runtime_list_reports_individual_errors_without_exposing_environment(api, monkeypatch):
    client, _ = api

    def resolve(name, configs=None):
        if name == "v100":
            raise profiles.RuntimeProfileError("V100 runtime is not registered")
        return profiles.RuntimeSpec(name, Path("/registered/python"), {"PRIVATE_VALUE": "not-for-browser"})

    monkeypatch.setattr(profiles, "resolve_runtime", resolve)
    response = client.get("/system/runtimes")
    assert response.status_code == 200
    rows = response.json()["runtimes"]
    assert [row["name"] for row in rows] == ["default", "curation", "v100"]
    assert rows[0]["available"] and rows[0]["python"] == "/registered/python"
    assert not rows[2]["available"] and "not registered" in rows[2]["error"]
    assert "PRIVATE_VALUE" not in response.text and "not-for-browser" not in response.text


def test_malformed_registration_is_listed_without_loading_torch(tmp_path, monkeypatch):
    import sys
    configs = tmp_path / "configs"
    (configs / "custom").mkdir(parents=True)
    (configs / "custom" / "runtime.toml").write_text("[broken")
    monkeypatch.setattr(profiles, "is_v100_runtime", lambda: False)
    before = set(sys.modules)
    rows = profiles.list_runtime_profiles(configs)
    assert all(not row["available"] and "Cannot read runtime configuration" in row["error"] for row in rows)
    assert "torch" not in set(sys.modules) - before
