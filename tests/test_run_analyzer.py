from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.run_analyzer import discovery, server


def _write_jsonl(path: Path, *events: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def test_training_job_kind_is_authoritative() -> None:
    assert discovery._is_training_job({"kind": "train", "method": "tlora", "argv": []})
    assert not discovery._is_training_job(
        {"kind": "command", "method": "preprocess", "argv": ["tasks.py"]}
    )


def test_inline_progress_stream_is_linked_by_log_dir(tmp_path, monkeypatch) -> None:
    logs_dir = tmp_path / "output" / "logs"
    log_dir = logs_dir / "中文 run_20260808-1200"
    log_dir.mkdir(parents=True)
    jsonl_path = logs_dir / "中文 run.progress.jsonl"
    _write_jsonl(
        jsonl_path,
        {
            "ev": "run_start",
            "run": "中文 run",
            "method": "tlora",
            "preset": "default",
            "total_steps": 10,
            "total_epochs": 2,
            "log_dir": str(log_dir),
        },
        {
            "ev": "step",
            "ts": 1.0,
            "global_step": 1,
            "epoch": 1,
            "loss/current": 0.5,
        },
        {"ev": "run_end", "ts": 2.0, "status": "ok", "final_step": 1},
    )
    monkeypatch.setattr(discovery, "MONADFORGE_ROOT", str(tmp_path))
    monkeypatch.setattr(discovery, "JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(discovery, "LOGS_DIR", str(logs_dir))

    runs = discovery.discover()

    assert len(runs) == 1
    run = runs[0]
    assert run.id == "inline-中文 run_20260808-1200"
    assert run.jsonl_path == str(jsonl_path)
    assert run.sources["jsonl"] is True
    assert run.state == "done"
    assert run.total_steps == 10


def test_freshen_run_reloads_job_and_maps_run_end_to_terminal(tmp_path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    jsonl_path = job_dir / "progress.jsonl"
    _write_jsonl(
        jsonl_path,
        {
            "ev": "run_start",
            "run": "test",
            "method": "lora",
            "total_steps": 4,
            "total_epochs": 1,
        },
        {"ev": "run_end", "ts": 1.0, "status": "ok", "final_step": 4},
    )
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "kind": "train",
                "state": "running",
                "submitted_at": 1.0,
                "started_at": 2.0,
                "ended_at": 3.0,
                "rc": 0,
            }
        ),
        encoding="utf-8",
    )
    run = discovery.Run(
        id="job-1",
        kind="daemon",
        dir=str(job_dir),
        jsonl_path=str(jsonl_path),
        state="running",
    )

    server._freshen_run(run)

    assert run.job is not None
    assert run.ended_at == 3.0
    assert run.state == "done"
    assert run.jsonl is not None and run.jsonl.run_end_final_step == 4


def test_tensorboard_append_changes_index_cache_key(tmp_path, monkeypatch) -> None:
    jobs_dir = tmp_path / "jobs"
    logs_dir = tmp_path / "logs"
    event_path = logs_dir / "run" / "network_train" / "events.out.tfevents.test"
    event_path.parent.mkdir(parents=True)
    jobs_dir.mkdir()
    event_path.write_bytes(b"first")
    monkeypatch.setattr(server, "JOBS_DIR", str(jobs_dir))
    monkeypatch.setattr(server, "LOGS_DIR", str(logs_dir))

    before = server._dir_mtime_key()
    event_path.write_bytes(b"first-second")
    after = server._dir_mtime_key()

    assert before != after


def test_run_hash_round_trips_unicode_and_spaces(tmp_path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    module_path = tmp_path / "dom.mjs"
    shutil.copyfile("scripts/run_analyzer/static/js/core/dom.js", module_path)
    module_uri = module_path.as_uri()
    script = f"""
globalThis.location = {{ hash: '#runs' }};
const {{ parseHash, runHash }} = await import({json.dumps(module_uri)});
const id = 'inline-中文 run';
location.hash = runHash(id);
const parsed = parseHash();
if (parsed.view !== 'run' || parsed.id !== id) process.exit(2);
"""
    result = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            script,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
