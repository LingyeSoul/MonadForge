"""CPU-only native curation contracts. Every write stays in pytest tmp_path."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from webui.services import curation_service as svc
from scripts.curation.context import scoped_tools


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    pytest.importorskip("anime_tools.gui.stages")
    monkeypatch.setattr(svc, "ROOT", tmp_path)
    monkeypatch.setattr(svc, "HOME", tmp_path)
    monkeypatch.setattr(svc, "JOBS", tmp_path / "output" / "curation" / "jobs")
    monkeypatch.delenv("ANIME_TOOLS_WORKSPACE", raising=False)
    monkeypatch.delenv("ANIME_TOOLS_MODELS", raising=False)
    import library.runtime.profiles as profiles
    monkeypatch.setattr(profiles, "resolve_runtime", lambda *_: SimpleNamespace(python=Path(sys.executable), env={}))
    root = tmp_path / "image_dataset"
    root.mkdir()
    (root / "one.png").write_bytes(b"fixture image; no image decode is requested")
    (root / "one.txt").write_text("1girl, blue hair", encoding="utf-8")
    return tmp_path


def test_all_eleven_official_schemas_and_ordered_mask_steps(workspace):
    schema = svc.get_schema()
    assert {stage["id"] for stage in schema["stages"]} == {
        "resize", "autotag", "position", "correct", "drop_groups", "audit",
        "ocr", "groups", "masks_sam", "masks_merge", "export",
    }
    plan = svc.prepare("masks_sam", {}, rel="one.png")
    assert [step["stage"] for step in plan["steps"]] == ["resize", "masks_sam", "masks_merge"]
    assert plan["scope"] == "one.png"
    for step in plan["steps"][:2]:
        assert "one" in " ".join(step["argv"])
    assert plan["report_path"] is None  # The official SAM stage writes masks only.
    assert Path(plan["steps"][0]["report_path"]).is_relative_to(Path(plan["job_dir"]))


def test_master_edit_is_overlay_and_caption_grammar_round_trips(workspace):
    source = workspace / "image_dataset" / "one.txt"
    original = source.read_bytes()
    text = '1girl. On the left, blue hair. Japanese text reads as "Hello, world.".'
    saved = svc.caption("one.png", "master", text)
    assert source.read_bytes() == original
    assert (workspace / "workspace" / "master" / "one.txt").read_text() == text
    parsed = svc.parse(text)
    assert svc.compose(parsed["flat_tags"], parsed["clauses"])["text"] == text
    assert saved["text"] == text


def test_registration_and_symlink_boundaries(workspace, tmp_path):
    with pytest.raises(ValueError):
        svc.save_roots({"master": str(workspace / "image_dataset" / "overlay")})
    with pytest.raises(ValueError):
        svc.item("../one.png")
    external = tmp_path.parent / f"{tmp_path.name}-external"
    external.mkdir()
    (external / "private.txt").write_text("private")
    (workspace / "image_dataset" / "escape.txt").symlink_to(external / "private.txt")
    with pytest.raises(ValueError, match="outside"):
        svc.file_path(str(workspace / "image_dataset" / "escape.txt"))
    with pytest.raises(ValueError, match="escapes"):
        svc.dataset()
    with pytest.raises(ValueError):
        svc.save_settings({"dataset": {"src": str(external)}})


def test_snapshot_is_frozen_and_daemon_receives_one_command(workspace, monkeypatch):
    calls = []

    async def start(command, args, **kwargs):
        calls.append((command, args, kwargs))
        return SimpleNamespace(id="daemon-fixture")

    task = SimpleNamespace(start_task=start, get_task=lambda _: None)
    monkeypatch.setitem(sys.modules, "webui.services.task_service", SimpleNamespace(task_service=task))
    job = asyncio.run(svc.submit("correct", {}, False, "one.png"))
    assert len(calls) == 1 and calls[0][0] == "curation-stage"
    assert calls[0][2]["runtime_profile"] == "curation"
    snapshot = svc._load_job(job["id"])
    assert [step["stage"] for step in snapshot["steps"]] == ["resize", "correct"]
    old_roots = snapshot["roots"].copy()
    svc.save_roots({"src": str(workspace / "other-source")})
    assert svc._load_job(job["id"])["roots"] == old_roots
    path = Path(snapshot["job_dir"]) / "snapshot.json"
    modified = json.loads(path.read_text())
    modified["scope"] = "another.png"
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="snapshot changed"):
        svc._load_job(job["id"])


def test_export_apply_requires_review_and_rejects_source_overlap(workspace):
    with pytest.raises(ValueError, match="preview"):
        svc.prepare("export", {}, True)
    with pytest.raises(ValueError, match="outside"):
        svc.prepare("export", {"out": str(workspace / "image_dataset")})


def test_context_restores_environment_and_dataset_reader_on_error(workspace, monkeypatch):
    monkeypatch.setenv("ANIME_TOOLS_HOME", "/old/home")
    import os
    with scoped_tools(workspace) as loaded:
        original = loaded.D.load_settings
    with pytest.raises(RuntimeError):
        with scoped_tools(workspace, {"dataset": {}}) as loaded:
            assert loaded.D.load_settings is not original
            raise RuntimeError("fixture")
    assert loaded.D.load_settings is original
    assert os.environ["ANIME_TOOLS_HOME"] == "/old/home"
