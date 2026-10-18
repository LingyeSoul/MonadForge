"""Staged launch requests must preserve the explicitly selected runtime."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from webui.api import staged_resolution


@pytest.mark.parametrize(
    "action,command", [("train", "staged-train"), ("preprocess", "staged-preprocess")]
)
@pytest.mark.parametrize("runtime", ["default", "curation", "v100"])
def test_staged_actions_forward_runtime(monkeypatch, action, command, runtime):
    monkeypatch.setattr(staged_resolution.plans, "load_profile", lambda _name: {})
    submit = AsyncMock(
        return_value=SimpleNamespace(info=lambda: {"task_id": "fixture"})
    )
    monkeypatch.setattr(staged_resolution.task_service, "start_task", submit)
    app = FastAPI()
    app.include_router(staged_resolution.router)
    response = TestClient(app).post(
        f"/profiles/default/{action}", json={"version": 1, "runtime_profile": runtime}
    )
    assert response.status_code == 200
    submit.assert_awaited_once_with(command, ["default"], runtime_profile=runtime)


def test_staged_invalid_runtime_is_rejected_before_submission(monkeypatch):
    submit = AsyncMock()
    monkeypatch.setattr(staged_resolution.task_service, "start_task", submit)
    app = FastAPI()
    app.include_router(staged_resolution.router)
    response = TestClient(app).post(
        "/profiles/default/train", json={"version": 1, "runtime_profile": "missing"}
    )
    assert response.status_code == 422
    submit.assert_not_awaited()
