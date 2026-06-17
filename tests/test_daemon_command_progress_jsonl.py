"""Regression tests for daemon command-job training progress wiring.

WebUI submits training through the daemon's generic ``command`` surface for
``tasks.py`` compatibility. Those jobs still need the daemon's per-job
``progress.jsonl`` stream; otherwise train.py falls back to
``output/logs/<output_name>.progress.jsonl``, a cross-run shared file that can
replay stale metrics at task start and leave the live dashboard empty.
"""

from __future__ import annotations

from scripts.daemon.jobs import Job
from scripts.daemon.manager import JobManager


def test_command_training_job_gets_per_job_progress_jsonl(monkeypatch):
    mgr = JobManager()
    job = Job(
        id="20260617-163000-abcdef",
        method="lora-gui",
        preset="",
        kind="command",
        argv=["tasks.py", "lora-gui", "lora-8gb"],
        progress_path="output/daemon/jobs/20260617-163000-abcdef/progress.jsonl",
    )

    monkeypatch.setattr("scripts.daemon.client.venv_python", lambda windowless=False: "pythonw.exe")

    cmd, _env = mgr._build_cmd(job)

    assert "--progress_jsonl" in cmd
    idx = cmd.index("--progress_jsonl")
    assert cmd[idx + 1] == job.progress_path
    assert _env["TQDM_MININTERVAL"] == "0.5"
    assert _env["TQDM_MINITERS"] == "1"


def test_command_non_training_job_does_not_get_progress_jsonl(monkeypatch):
    mgr = JobManager()
    job = Job(
        id="20260617-163000-abcdef",
        method="preprocess",
        preset="",
        kind="command",
        argv=["tasks.py", "preprocess"],
        progress_path="output/daemon/jobs/20260617-163000-abcdef/progress.jsonl",
    )

    monkeypatch.setattr("scripts.daemon.client.venv_python", lambda windowless=False: "pythonw.exe")

    cmd, _env = mgr._build_cmd(job)

    assert "--progress_jsonl" not in cmd


def test_command_training_job_keeps_explicit_progress_jsonl(monkeypatch):
    mgr = JobManager()
    job = Job(
        id="20260617-163000-abcdef",
        method="lora-gui",
        preset="",
        kind="command",
        argv=[
            "tasks.py",
            "lora-gui",
            "lora-8gb",
            "--progress_jsonl",
            "custom.progress.jsonl",
        ],
        progress_path="output/daemon/jobs/20260617-163000-abcdef/progress.jsonl",
    )

    monkeypatch.setattr("scripts.daemon.client.venv_python", lambda windowless=False: "pythonw.exe")

    cmd, _env = mgr._build_cmd(job)

    assert cmd.count("--progress_jsonl") == 1
    idx = cmd.index("--progress_jsonl")
    assert cmd[idx + 1] == "custom.progress.jsonl"
