from __future__ import annotations

import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

from scripts.tasks import downloads


def test_danbooru_tags_download_url_points_to_source_repo():
    assert downloads.DANBOORU_TAGS_URLS == (
        "https://raw.githubusercontent.com/Localsmile/danbooru_KR_wiki_tag_search/main/danbooru_tags_classified.csv",
    )


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self) -> bytes:
        return self._payload


def test_download_danbooru_tags_writes_models_file(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_PATH", dest)
    monkeypatch.setattr(
        downloads, "DANBOORU_TAGS_URLS", ("https://example.test/tags.csv",)
    )
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda _req, timeout=60: _FakeResponse(
            b"name,category,post_count,description\n1girl,0,1,test\n"
        ),
    )

    downloads.cmd_download_danbooru_tags([])

    assert dest.read_text(encoding="utf-8").startswith("name,category")


def test_download_danbooru_tags_skips_existing_without_force(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    dest.parent.mkdir(parents=True)
    dest.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_PATH", dest)
    called = False

    def _fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        return _FakeResponse(BytesIO().read())

    monkeypatch.setattr(downloads.urllib.request, "urlopen", _fail_if_called)

    downloads.cmd_download_danbooru_tags([])

    assert not called
    assert dest.read_text(encoding="utf-8") == "existing"


def test_download_danbooru_tags_failure_names_source_repo(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "danbooru_tags_classified.csv"
    monkeypatch.setattr(downloads, "DANBOORU_TAGS_PATH", dest)
    monkeypatch.setattr(
        downloads, "DANBOORU_TAGS_URLS", ("https://example.test/tags.csv",)
    )
    monkeypatch.setattr(
        downloads.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("nope")),
    )

    with pytest.raises(SystemExit) as exc:
        downloads.cmd_download_danbooru_tags([])

    assert "Localsmile/danbooru_KR_wiki_tag_search" in str(exc.value)


def _no_hf_run(*_args, **_kwargs):
    raise AssertionError("hf CLI must not run under --source modelscope")


def test_anima_modelscope_source_uses_snapshot_and_moves(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "run", _no_hf_run)
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def fake_snapshot(repo_id, local_dir, *, filenames=None, include=None, force=False):
        calls.append((repo_id, str(local_dir), tuple(filenames or ())))
        models = Path(local_dir)
        for name in filenames or ():
            p = models / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")

    monkeypatch.setattr(downloads.ms_download, "snapshot_download", fake_snapshot)

    downloads.cmd_download_anima(["--source", "modelscope"])

    models = tmp_path / "models"
    assert calls and calls[0][0] == "circlestone-labs/Anima"
    assert "split_files/diffusion_models/anima-base-v1.0.safetensors" in calls[0][2]
    assert (models / "diffusion_models" / "anima-base-v1.0.safetensors").exists()
    assert (models / "text_encoders" / "qwen_3_06b_base.safetensors").exists()
    assert (models / "vae" / "qwen_image_vae.safetensors").exists()
    assert not (models / "split_files").exists()


def test_sam3_modelscope_env_routes_without_flag(tmp_path, monkeypatch):
    """``ANIMA_DOWNLOAD_SOURCE=modelscope`` alone must select the ms branch —
    no ``--source`` flag needed."""
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setenv("ANIMA_DOWNLOAD_SOURCE", "modelscope")
    monkeypatch.setattr(downloads, "run", _no_hf_run)
    monkeypatch.setattr(
        downloads.ms_download,
        "snapshot_download",
        lambda *a, **k: None,
    )

    downloads.cmd_download_sam3([])


def test_anima_modelscope_env_routes_without_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setenv("ANIMA_DOWNLOAD_SOURCE", "modelscope")
    monkeypatch.setattr(downloads, "run", _no_hf_run)
    called = {}

    def fake_snapshot(repo_id, local_dir, *, filenames=None, include=None, force=False):
        called["repo"] = repo_id
        called["force"] = force

    monkeypatch.setattr(downloads.ms_download, "snapshot_download", fake_snapshot)

    downloads.cmd_download_anima([])

    assert called["repo"] == "circlestone-labs/Anima"
    assert called["force"] is False


def test_anima_modelscope_force_passes_through(tmp_path, monkeypatch):
    """``--force`` must reach the ms snapshot instead of being dropped."""
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "run", _no_hf_run)
    called = {}

    def fake_snapshot(repo_id, local_dir, *, filenames=None, include=None, force=False):
        called["force"] = force

    monkeypatch.setattr(downloads.ms_download, "snapshot_download", fake_snapshot)

    downloads.cmd_download_anima(["--source", "modelscope", "--force"])

    assert called["force"] is True


def test_sam3_modelscope_force_passes_through(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "run", _no_hf_run)
    called = {}

    def fake_snapshot(repo_id, local_dir, *, force=False, **_k):
        called["force"] = force

    monkeypatch.setattr(downloads.ms_download, "snapshot_download", fake_snapshot)

    downloads.cmd_download_sam3(["--source", "modelscope", "--force"])

    assert called["force"] is True


def test_pe_core_modelscope_force_passes_through(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "run", _no_hf_run)
    called = {}

    def fake_fetch_file(*, ms_repo, filename, local_dir, what="asset", force=False):
        called["force"] = force
        return Path(local_dir) / filename

    monkeypatch.setattr(downloads.ms_download, "fetch_file", fake_fetch_file)

    downloads.cmd_download_pe(["--source", "modelscope", "--force"])

    assert called["force"] is True


def test_anima_default_source_still_uses_hf(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    seen: list[list[str]] = []
    monkeypatch.setattr(downloads, "run", lambda cmd, **_k: seen.append(cmd))

    downloads.cmd_download_anima([])

    assert seen and seen[0][:3] == ["hf", "download", "circlestone-labs/Anima"]


def test_pe_spatial_modelscope_without_mirror_skips(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    monkeypatch.setattr(downloads, "run", _no_hf_run)

    downloads.cmd_download_pe_spatial(["--source", "modelscope"])

    out = capsys.readouterr().out
    assert "no ModelScope mirror" in out
    assert not (tmp_path / "models" / "pe" / "PE-Spatial-B16-512.pt").exists()


def test_download_ms_requires_repo_id():
    with pytest.raises(SystemExit, match="usage"):
        downloads.cmd_download_ms([])
    with pytest.raises(SystemExit, match="org/name"):
        downloads.cmd_download_ms(["justname"])


def test_download_ms_rejects_unknown_option(tmp_path, monkeypatch):
    """A typo'd flag must fail loudly, not silently become a repo id/file."""
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="unknown option --localdir"):
        downloads.cmd_download_ms(["org/name", "--localdir", "x"])


def test_download_ms_passes_repo_and_default_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "ROOT", tmp_path)
    calls: dict = {}

    def fake_snapshot(repo_id, local_dir, *, include=None, force=False, **_k):
        calls["args"] = (repo_id, str(local_dir), include, force)

    monkeypatch.setattr(downloads.ms_download, "snapshot_download", fake_snapshot)

    downloads.cmd_download_ms(["org/name", "--include", "*.safetensors"])

    repo_id, local_dir, include, force = calls["args"]
    assert repo_id == "org/name"
    assert local_dir == str(tmp_path / "models" / "name")
    assert include == ["*.safetensors"]
    assert force is False

    downloads.cmd_download_ms(["org/other", "--local-dir", str(tmp_path / "x")])
    assert calls["args"][1] == str(tmp_path / "x")
