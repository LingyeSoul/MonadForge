"""Path-boundary regressions for request-supplied paths.

Every endpoint that hands a caller-supplied path to file I/O must keep it inside
the project root (or a root named by ``ANIMA_ALLOW_OUTSIDE_ROOTS``). These tests
pin the four sinks that used to accept an arbitrary absolute path — sample
prompts, preprocess run status, checkpoint wipe and archive extraction — plus the
shared sandbox helper they now go through.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from webui.services import config_service as config
from webui.services import image_service as images
from webui.services import preprocess_service as preprocess
from webui.services.sandbox import OUTSIDE_ROOTS_ENV, allowed_roots, resolve_within


@pytest.fixture(autouse=True)
def _no_extra_roots(monkeypatch):
    """The escape hatch must be opt-in; a stray env var must not leak in."""
    monkeypatch.delenv(OUTSIDE_ROOTS_ENV, raising=False)


# ── sandbox.resolve_within ──────────────────────────────────────


def test_relative_path_resolves_under_root(tmp_path):
    assert resolve_within(tmp_path, "sub/file.txt") == tmp_path / "sub" / "file.txt"


def test_absolute_path_inside_root_is_allowed(tmp_path):
    inside = tmp_path / "data" / "prompts.txt"
    assert resolve_within(tmp_path, str(inside)) == inside


@pytest.mark.parametrize(
    "value",
    ["/etc/passwd", "../outside.txt", "sub/../../outside.txt"],
)
def test_outside_paths_are_refused(tmp_path, value):
    with pytest.raises(ValueError):
        resolve_within(tmp_path, value)


def test_symlink_escape_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        resolve_within(root, "link/secret.txt")


def test_suffix_rule_allows_txt_and_bare_names(tmp_path):
    assert resolve_within(tmp_path, "prompts.txt", suffixes=(".txt",))
    assert resolve_within(tmp_path, "prompts", suffixes=(".txt",))
    with pytest.raises(ValueError):
        resolve_within(tmp_path, "configs/base.toml", suffixes=(".txt",))


def test_extra_roots_env_is_the_only_escape_hatch(tmp_path, monkeypatch):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError):
        resolve_within(root, str(outside / "prompts.txt"))
    monkeypatch.setenv(OUTSIDE_ROOTS_ENV, str(outside))
    assert allowed_roots() == (outside.resolve(),)
    assert resolve_within(root, str(outside / "prompts.txt")) == outside / "prompts.txt"


# ── sample prompts (read / overwrite) ───────────────────────────


def test_sample_prompts_refuses_paths_outside_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    for path in ("/etc/passwd", "../prompts.txt", "configs/base.toml"):
        with pytest.raises(ValueError):
            config.read_sample_prompts(path)
        with pytest.raises(ValueError):
            config.write_sample_prompts(path, [{"prompt": "x"}])


def test_sample_prompts_round_trip_inside_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    config.write_sample_prompts(
        "prompts.txt", [{"prompt": "1girl", "negative_prompt": "bad"}]
    )
    assert (tmp_path / "prompts.txt").read_text(
        encoding="utf-8"
    ).strip() == "1girl --n bad"
    assert config.read_sample_prompts("prompts.txt") == [
        {"prompt": "1girl", "negative_prompt": "bad"}
    ]


def test_sample_prompts_api_maps_refusal_to_400(tmp_path, monkeypatch):
    from webui.api.config import router

    monkeypatch.setattr(config, "ROOT", tmp_path)
    app = FastAPI()
    app.include_router(router, prefix="/config")
    client = TestClient(app)

    assert (
        client.get("/config/sample-prompts", params={"path": "/etc/passwd"}).status_code
        == 400
    )
    assert (
        client.put(
            "/config/sample-prompts",
            json={"path": "configs/base.toml", "entries": [{"prompt": "x"}]},
        ).status_code
        == 400
    )
    ok = client.put(
        "/config/sample-prompts",
        json={"path": "prompts.txt", "entries": [{"prompt": "x"}]},
    )
    assert ok.status_code == 200
    assert (tmp_path / "prompts.txt").is_file()


# ── preprocess run status ───────────────────────────────────────


def test_run_status_refuses_manifest_outside_run_roots(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    (runs / "src" / "cfg").mkdir(parents=True)
    manifest = runs / "src" / "cfg" / "manifest.json"
    # ``status != "ready"`` keeps the summary path-only: a complete manifest
    # would be validated against the run's real directories.
    manifest.write_text(json.dumps({"status": "draft"}), encoding="utf-8")
    outside = tmp_path / "elsewhere.json"
    outside.write_text(json.dumps({"status": "draft"}), encoding="utf-8")

    monkeypatch.setattr(preprocess, "_run_roots", lambda *a, **k: [runs])

    inside = preprocess.get_run_status(manifest)
    assert inside["status"] == "draft"
    assert inside["manifest"] == str(manifest)
    # A directory is expanded to its manifest.json.
    assert preprocess.get_run_status(manifest.parent)["manifest"] == str(manifest)
    with pytest.raises(ValueError):
        preprocess.get_run_status(outside)
    with pytest.raises(ValueError):
        preprocess.get_run_status("/etc/passwd")


# ── checkpoint wipe ─────────────────────────────────────────────


def test_wipe_checkpoint_refuses_absolute_path_outside_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        config.wipe_checkpoint(str(tmp_path.parent / "elsewhere"), "lora")


# ── archive extraction ──────────────────────────────────────────


def test_extract_target_must_stay_inside_root(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "ROOT", tmp_path)
    assert (
        images._resolve_extract_target("datasets/new") == tmp_path / "datasets" / "new"
    )
    assert images._resolve_extract_target(str(tmp_path / "abs")) == tmp_path / "abs"
    for bad in ("/etc", str(tmp_path.parent / "elsewhere"), "../elsewhere"):
        with pytest.raises(ValueError):
            images._resolve_extract_target(bad)


def test_extract_target_still_rejects_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "ROOT", tmp_path)
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        images._resolve_extract_target("file.txt")


# ── webui command surface ───────────────────────────────────────


def test_vendor_sync_is_not_browser_reachable():
    from webui.services.task_catalog import COMMAND_CATALOG

    assert "vendor-sync" not in COMMAND_CATALOG
    assert "update" in COMMAND_CATALOG


# ── event-loop safety ───────────────────────────────────────────


def test_ws_hydrates_the_task_off_the_event_loop():
    """``subscribe`` is loop-bound; the blocking daemon lookup must not be."""
    import inspect

    from webui.services.task_service import task_service

    assert inspect.iscoroutinefunction(task_service.ensure_task_loaded)
    assert not inspect.iscoroutinefunction(task_service.subscribe)
