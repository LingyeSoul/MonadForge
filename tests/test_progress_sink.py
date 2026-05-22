"""Phase 0 structured-progress sink: schema + tail-while-write smoke tests."""

from __future__ import annotations

import argparse
import json
import os

from library.training.progress import ProgressSink, _find_cmmd, _flatten_logs


def _read_events(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_resolve_path_default_derives_to_sibling_logs_dir(tmp_path):
    ckpt = tmp_path / "ckpt"
    args = argparse.Namespace(
        progress_jsonl=None, output_dir=str(ckpt), output_name="my_run"
    )
    resolved = ProgressSink.resolve_path(args)
    # sibling logs/ dir, not the checkpoint dir
    assert resolved == os.path.join(str(tmp_path), "logs", "my_run.progress.jsonl")


def test_resolve_path_disable_tokens():
    base = dict(output_dir="/tmp/x", output_name="r")
    for tok in ("", "  ", "none", "OFF", "None"):
        args = argparse.Namespace(progress_jsonl=tok, **base)
        assert ProgressSink.resolve_path(args) is None


def test_resolve_path_explicit_override(tmp_path):
    explicit = str(tmp_path / "custom.jsonl")
    args = argparse.Namespace(
        progress_jsonl=explicit, output_dir="/ignored", output_name="r"
    )
    assert ProgressSink.resolve_path(args) == explicit


def test_full_lifecycle_schema(tmp_path):
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="run", method="lora", preset="default", t0=0.0)
    sink.run_start(total_steps=100, total_epochs=4, pid=4242)
    sink.log({"loss": 0.5, "lr": 1e-4}, global_step=10, epoch=1)
    sink.log({"loss/val_average": 0.03, "loss/val_cmmd": 0.03}, global_step=10, epoch=1)
    sink.ckpt(global_step=10, path="output/ckpt/run-step10.safetensors")
    sink.run_end(status="ok", final_step=100)

    evs = _read_events(path)
    kinds = [e["ev"] for e in evs]
    assert kinds == ["run_start", "step", "val", "ckpt", "run_end"]

    start = evs[0]
    assert start["run"] == "run" and start["method"] == "lora"
    assert start["total_steps"] == 100 and start["pid"] == 4242

    step = evs[1]
    assert step["global_step"] == 10 and step["epoch"] == 1 and step["loss"] == 0.5

    val = evs[2]
    assert val["cmmd"] == 0.03 and val["global_step"] == 10

    assert evs[3]["path"].endswith("run-step10.safetensors")
    assert evs[4]["status"] == "ok" and evs[4]["final_step"] == 100
    # every line carries an event tag + timestamp
    assert all("ev" in e and "ts" in e for e in evs)


def test_stopped_run_end(tmp_path):
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="r", method=None, preset=None, t0=0.0)
    sink.run_start(total_steps=10, total_epochs=1, pid=1)
    sink.run_end(status="stopped", final_step=3)
    evs = _read_events(path)
    assert evs[-1]["ev"] == "run_end" and evs[-1]["status"] == "stopped"


def test_log_before_run_start_is_noop(tmp_path):
    # Sink not yet opened (no run_start) → log/ckpt must not create the file.
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="r", method=None, preset=None)
    sink.log({"loss": 1.0}, global_step=1, epoch=0)
    sink.ckpt(global_step=1, path="x")
    assert not os.path.exists(path)


def test_tail_while_write(tmp_path):
    # A reader can open + read the file while the sink keeps appending
    # (the concurrency contract the daemon relies on).
    path = str(tmp_path / "run.progress.jsonl")
    sink = ProgressSink(path, run="r", method=None, preset=None, t0=0.0)
    sink.run_start(total_steps=2, total_epochs=1, pid=1)
    sink.log({"loss": 0.1}, global_step=1, epoch=0)
    # read mid-run, before run_end
    mid = _read_events(path)
    assert [e["ev"] for e in mid] == ["run_start", "step"]
    sink.run_end(status="ok", final_step=2)
    assert [e["ev"] for e in _read_events(path)][-1] == "run_end"


def test_flatten_logs_drops_nonscalar():
    flat = _flatten_logs({"loss": 0.5, "ok": True, "name": "x", "arr": [1, 2, 3]})
    assert flat == {"loss": 0.5, "ok": True, "name": "x"}


def test_find_cmmd():
    assert _find_cmmd({"loss/val_cmmd": 0.042, "loss": 1.0}) == 0.042
    assert _find_cmmd({"loss": 1.0}) is None
