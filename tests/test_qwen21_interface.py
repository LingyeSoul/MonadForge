"""Qwen HTTP/configuration integration without model loading or GPU work."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.daemon.jobs import Job
from scripts.daemon.manager import JobManager
from scripts.qwen21.workflow import parse_workflow
from webui.api.qwen21 import router


def test_qwen_http_schema_and_strict_inputs(tmp_path: Path):
    app = FastAPI()
    app.include_router(router, prefix="/qwen21")
    with TestClient(app) as client:
        response = client.get("/qwen21/schema")
        assert response.status_code == 200
        fields = {field["name"]: field for field in response.json()["train"]}
        assert fields["blocks_to_swap"]["value"] is None
        assert fields["activation_reserve_gb"]["value"] is None
        assert fields["alpha"]["value"] is None
        assert fields["compile"]["value"] is True
        assert fields["scheduler"]["kind"] == "str"
        response = client.post(
            "/qwen21/resolve",
            json={
                "values": {
                    "model_dir": str(tmp_path),
                    "text_encoder": str(tmp_path / "te.safetensors"),
                }
            },
        )
        assert response.status_code == 200
        assert response.json()["scheduler"] == str(tmp_path / "scheduler")
        response = client.post("/qwen21/jobs/train", json={"values": {"rank": "16"}})
        assert response.status_code == 400
        assert "rank" in response.json()["detail"]
        response = client.post(
            "/qwen21/jobs/train", json={"values": {"blocks_to_swap": -1}}
        )
        assert response.status_code == 400
        assert "blocks_to_swap" in response.json()["detail"]


def test_default_scheduler_only_follows_dit_source(tmp_path: Path):
    code = """
import json
from library.qwen21.requests import CONFIG_DIR, TrainRequest, model_paths
base = model_paths(TrainRequest())
encoder = model_paths(TrainRequest(text_encoder='te.safetensors', vae='vae.safetensors'))
single = model_paths(TrainRequest(dit='dit.safetensors'))
explicit = model_paths(TrainRequest(dit='dit.safetensors', scheduler='custom.json'))
assert base.scheduler == encoder.scheduler
assert single.scheduler == CONFIG_DIR / 'scheduler.json'
assert explicit.scheduler.name == 'custom.json'
print(json.dumps({'torch_loaded': 'torch' in __import__('sys').modules}))
"""
    env = dict(os.environ, ANIMA_HOME=str(tmp_path), ANIMA_QWEN21_MODEL_DIR="")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout)["torch_loaded"] is False


def test_workflow_links_cache_without_changing_auto_parameters():
    cache_args = ["--out", "output/qwen21/new-cache"]
    train_args = ["--cache", "output/qwen21/old-cache", "--rank", "32"]
    cache, train = parse_workflow(
        ["--cache-args", json.dumps(cache_args), "--train-args", json.dumps(train_args)]
    )
    assert train.cache == cache.out == "output/qwen21/new-cache"
    assert train.rank == 32
    assert train.blocks_to_swap is None
    assert train.activation_reserve_gb is None
    assert train.alpha is None
    assert train_args[1] == "output/qwen21/old-cache"


def test_command_watchdog_override_survives_serialization():
    job = Job(
        id="qwen-interface",
        method="qwen21-train",
        preset="",
        kind="command",
        stall_timeout=900.0,
    )
    restored = Job(**json.loads(json.dumps(dataclasses.asdict(job))))
    assert restored.stall_timeout == 900.0
    restored.started_at = time.time() - 121
    assert JobManager._stall_reason(restored) is None
    restored.started_at = time.time() - 1000
    assert "limit 900s" in JobManager._stall_reason(restored)


def test_qwen_stdout_populates_existing_dashboard_metrics():
    from webui.services.training_log_parser import TrainingLogParser

    parser = TrainingLogParser()
    assert parser.feed("  progress 20/80 epoch 2/8 loss 0.2500 eta 0:01:30")
    assert parser.feed(
        "  step 20/80 loss 0.2400 |g| 0.100 sigma 0.300 lr 1.00e-04 peak 5.00 GB 1.50s/step"
    )
    metrics = parser.metrics.snapshot()
    assert (
        metrics["step"],
        metrics["total_steps"],
        metrics["epoch"],
        metrics["total_epochs"],
    ) == (20, 80, 2, 8)
    assert metrics["avr_loss"] == 0.25
    assert metrics["lr"] == 1e-4
    assert metrics["eta"] == "0:01:30"
    assert metrics["speed"] == "1.50 s/it"
    assert metrics["step_history"] == [20]


def test_command_timeout_crosses_real_http_without_starting_worker(
    tmp_path: Path, monkeypatch
):
    import asyncio
    import threading

    from scripts.daemon import config
    from scripts.daemon.client import DaemonClient
    from scripts.daemon.server import serve
    from webui.services.daemon_client import DaemonClient as AsyncDaemonClient

    monkeypatch.setattr(config, "JOBS_DIR", tmp_path / "jobs")
    manager = JobManager()
    server = serve(manager, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        sync = DaemonClient(port=port)
        asynchronous = AsyncDaemonClient(base_url=f"http://127.0.0.1:{port}")
        submissions = [
            sync.submit_command(
                label="qwen21-cache",
                argv=["-m", "scripts.qwen21.cache", "--help"],
                start=False,
                stall_timeout=900.0,
            ),
            asyncio.run(
                asynchronous.submit_command(
                    ["-m", "scripts.qwen21.generate", "--help"],
                    label="qwen21-generate",
                    start=False,
                    stall_timeout=900.0,
                )
            ),
        ]
        for response in submissions:
            job = manager.get(response["job_id"])
            assert job is not None and job.state == "queued"
            assert job.started_at is None and job.stall_timeout == 900.0
            persisted = json.loads((job.dir / "job.json").read_text())
            assert persisted["stall_timeout"] == 900.0
            assert persisted["argv"][-1] == "--help"
        assert not manager._worker.is_alive()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
