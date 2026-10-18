"""Job telemetry with mock training drivers; no model or tensor is constructed."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from library.lines.progress import training_progress
from library.moss.validation import validate_request


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _mock_entry(model_type, driver):
    """Execute only the public wrapper, never import or execute either loop."""
    path = Path(__file__).resolve().parents[1] / f"library/{model_type}/train.py"
    tree = ast.parse(path.read_text())
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_train"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            wrapper,
        ],
        type_ignores=[],
    )
    namespace = {
        "training_progress": training_progress,
        "_run_train": driver,
        "train": lambda req, samples, empty, *, progress: driver(req, progress),
        "resolve_under_home": Path,
        "samples_from_cache": lambda cache: ["synthetic-sample"],
        "validate_request": validate_request,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["run_train"]


def _request(model_type, **values):
    if model_type == "moss":
        from library.moss.requests import TrainRequest
    else:
        from library.qwen21.requests import TrainRequest

        values.pop("caption_dropout", None)
    return TrainRequest(**values)


@pytest.mark.parametrize("model_type", ["qwen21", "moss"])
def test_public_wrapper_writes_job_lifecycle_without_model(
    tmp_path, monkeypatch, model_type
):
    monkeypatch.setenv("ANIMA_DAEMON_JOB_DIR", str(tmp_path))
    output = tmp_path / "mock.safetensors"

    def driver(req, progress):
        progress.start(total_steps=2, total_epochs=1)
        for step in (1, 2):
            progress.step(
                global_step=step,
                epoch=1,
                loss=0.25,
                average_loss=0.3,
                lr=1e-4,
            )
        progress.checkpoint(output)
        return output

    run = _mock_entry(model_type, driver)
    assert (
        run(_request(model_type, output=str(output), cache="unused", caption_dropout=0))
        == output
    )
    events = _read(tmp_path / "progress.jsonl")
    assert [event["ev"] for event in events] == [
        "run_start",
        "step",
        "step",
        "ckpt",
        "run_end",
    ]
    assert events[0]["method"] == f"{model_type}-train"
    assert events[0]["total_steps"] == 2
    assert events[0]["sampling_enabled"] is False
    assert events[1]["loss/average"] == 0.3 and events[1]["lr"] == 1e-4
    assert events[-2]["global_step"] == 2 and events[-2]["path"] == str(output)
    assert events[-1]["status"] == "ok" and events[-1]["final_step"] == 2
    assert not output.exists()  # the mock never writes a checkpoint


@pytest.mark.parametrize("model_type", ["qwen21", "moss"])
@pytest.mark.parametrize(
    "exception,status",
    [(ValueError("preflight failed"), "error"), (KeyboardInterrupt(), "stopped")],
)
def test_wrapper_early_failure_has_terminal_event(
    tmp_path, monkeypatch, model_type, exception, status
):
    monkeypatch.setenv("ANIMA_DAEMON_JOB_DIR", str(tmp_path))

    def driver(req, progress):
        raise exception

    with pytest.raises(type(exception)):
        _mock_entry(model_type, driver)(
            _request(
                model_type, output="mock.safetensors", cache="unused", caption_dropout=0
            )
        )
    events = _read(tmp_path / "progress.jsonl")
    assert [event["ev"] for event in events] == ["run_start", "run_end"]
    assert events[-1]["status"] == status and events[-1]["final_step"] == 0
    if status == "error":
        assert "preflight failed" in events[-1]["error"]


def test_standalone_cli_has_no_job_stream(tmp_path, monkeypatch):
    monkeypatch.delenv("ANIMA_DAEMON_JOB_DIR", raising=False)
    with training_progress("moss", str(tmp_path / "mock.safetensors")) as progress:
        progress.start(total_steps=1, total_epochs=1)
        progress.step(global_step=1, epoch=1, loss=0.1, average_loss=0.1, lr=0.01)
    assert list(tmp_path.iterdir()) == []


def test_moss_output_validation_failure_emits_terminal_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMA_DAEMON_JOB_DIR", str(tmp_path))

    def driver(req, progress):
        pytest.fail("Invalid output must not reach the training driver")

    with pytest.raises(ValueError, match=".safetensors"):
        _mock_entry("moss", driver)(_request("moss", output="invalid.json"))
    events = _read(tmp_path / "progress.jsonl")
    assert [event["ev"] for event in events] == ["run_start", "run_end"]
    assert events[-1]["status"] == "error"
