"""MOSS output validation must protect weights before training starts."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from library.moss.requests import TrainRequest
from library.moss.validation import training_output_paths, validate_request
from webui.api import moss


@pytest.mark.parametrize("endpoint", ["/jobs/train", "/chain"])
def test_json_output_is_rejected_before_browser_submission(monkeypatch, endpoint):
    async def unexpected_submission(*_args, **_kwargs):
        pytest.fail("Invalid output reached the daemon")

    monkeypatch.setattr(moss.task_service, "start_task", unexpected_submission)
    app = FastAPI()
    app.include_router(moss.router)
    train = {"values": {"output": "output/ckpt_moss/run.json"}}
    body = (
        train
        if endpoint == "/jobs/train"
        else {"cache": {"values": {}}, "train": train}
    )
    with TestClient(app) as client:
        response = client.post(endpoint, json=body)
    assert response.status_code == 400
    assert ".safetensors" in response.json()["detail"]


def test_cli_rejects_json_output_before_loading_trainer(monkeypatch):
    from scripts.moss.train import main

    def unexpected_train(_req):
        pytest.fail("Invalid output reached the trainer")

    monkeypatch.setitem(
        sys.modules, "library.moss.train", SimpleNamespace(run_train=unexpected_train)
    )
    monkeypatch.setattr(
        sys, "argv", ["moss-train", "--output", "output/ckpt_moss/run.json"]
    )
    with pytest.raises(ValueError, match=r"must end in \.safetensors"):
        main()


@pytest.mark.parametrize("entry", ["run_train", "train"])
def test_programmatic_training_rejects_json_before_cache_or_model_reads(
    entry, tmp_path
):
    from library.moss import train

    output = tmp_path / "adapter.json"
    output.write_bytes(b"existing report")
    req = TrainRequest(output=str(output), cache=str(tmp_path / "missing-cache"))
    with pytest.raises(ValueError, match=r"must end in \.safetensors"):
        if entry == "run_train":
            train.run_train(req)
        else:
            train.train(req, [])
    assert output.read_bytes() == b"existing report"


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_weight_and_report_alias_is_rejected_without_writing(tmp_path, link):
    output = tmp_path / "adapter.safetensors"
    output.write_bytes(b"existing weight")
    report = output.with_suffix(".json")
    if link == "symlink":
        report.symlink_to(output)
    else:
        report.hardlink_to(output)
    with pytest.raises(ValueError, match="must be different files"):
        validate_request(TrainRequest(output=str(output)))
    assert output.read_bytes() == report.read_bytes() == b"existing weight"


def test_valid_output_keeps_existing_distinct_report(tmp_path):
    output = tmp_path / "adapter.safetensors"
    report = output.with_suffix(".json")
    output.write_bytes(b"existing weight")
    report.write_text('{"steps": 1}', encoding="utf-8")
    validate_request(TrainRequest(output=str(output)))
    assert training_output_paths(str(output)) == (output, report)
    assert output.read_bytes() == b"existing weight"
    assert report.read_text(encoding="utf-8") == '{"steps": 1}'
