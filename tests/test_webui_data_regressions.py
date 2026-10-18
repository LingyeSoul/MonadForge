"""Data-boundary regressions; all writes and task submissions are isolated."""

from __future__ import annotations

import asyncio
import io
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from webui.services import image_service as images


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "one.png").write_bytes(b"image fixture")
    (root / "one.txt").write_text(
        "1girl, smile. On the left, blue hair.", encoding="utf-8"
    )
    monkeypatch.setattr(images, "ROOT", tmp_path)
    monkeypatch.setattr(
        images,
        "get_path_overrides",
        lambda: {
            "source_image_dir": str(root),
            "resized_image_dir": str(root),
            "lora_cache_dir": str(root),
        },
    )
    return root


def test_batch_append_preserves_subject_binding(dataset):
    from library.captioning.position_clauses import parse_caption

    result = images.batch_update_captions(
        str(dataset), ["one.png"], "append", tag="outdoors"
    )
    assert result["updated"] == 1
    parsed = parse_caption((dataset / "one.txt").read_text())
    assert parsed.flat_tags == ("1girl", "smile", "outdoors")
    assert parsed.clauses[0].tags == ("blue hair",)


def test_batch_remove_sees_tag_before_clause_delimiter(dataset):
    images.batch_update_captions(
        str(dataset), ["one.png"], "remove", tag="smile, blue hair"
    )
    assert (dataset / "one.txt").read_text() == "1girl. On the left."


def test_tag_index_preserves_clause_tags(dataset):
    assert images.build_tag_index(str(dataset))["tags"] == {
        "1girl": 1,
        "smile": 1,
        "blue hair": 1,
    }


def test_caption_replace_failure_preserves_existing_text(dataset, monkeypatch):
    old = (dataset / "one.txt").read_bytes()
    real_replace = Path.replace

    def fail_caption(self, target):
        if Path(target) == dataset / "one.txt":
            raise OSError("simulated disk error")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_caption)
    with pytest.raises(OSError, match="disk error"):
        images.save_caption(str(dataset), "one.png", "replacement")
    assert (dataset / "one.txt").read_bytes() == old
    assert not list(dataset.glob("*.tmp"))


def test_caption_history_preserves_previous_versions(dataset):
    original = (dataset / "one.txt").read_text()
    images.save_caption(str(dataset), "one.png", "second")
    images.save_caption(str(dataset), "one.png", "second")
    images.save_caption(str(dataset), "one.png", "third")
    assert [row["text"] for row in images.get_versions(str(dataset), "one.png")] == [
        "second",
        original,
    ]


def test_explicit_dataset_symlink_is_browsable(dataset):
    alias = dataset.parent / "alias"
    alias.symlink_to(dataset, target_is_directory=True)
    assert images.resolve_image_path(str(alias), "one.png") == dataset / "one.png"


def test_image_resolver_rejects_non_image_file(dataset):
    assert images.resolve_image_path(str(dataset), "one.txt") is None


@pytest.mark.parametrize("suffix", [".txt", ".txt.history.jsonl"])
def test_caption_sidecar_cannot_write_outside_dataset(dataset, suffix):
    external = dataset.parent / "external"
    external.write_text("private")
    sidecar = (dataset / "one.png").with_suffix(suffix)
    sidecar.unlink(missing_ok=True)
    sidecar.symlink_to(external)
    with pytest.raises(ValueError, match="outside"):
        images.save_caption(str(dataset), "one.png", "replacement")
    assert external.read_text() == "private"


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "oops"},
        {"action": "replace", "find": ""},
        {"action": "replace", "find": "[", "use_regex": True},
    ],
)
def test_invalid_batch_request_rejected_before_any_write(dataset, payload):
    from webui.api.images import router

    app = FastAPI()
    app.include_router(router, prefix="/images")
    old = (dataset / "one.txt").read_bytes()
    response = TestClient(app).put(
        "/images/batch-caption",
        json={
            "directory": str(dataset),
            "paths": ["one.png"],
            **payload,
        },
    )
    assert response.status_code in (400, 422)
    assert (dataset / "one.txt").read_bytes() == old


def test_corrupt_zip_member_does_not_truncate_existing_file(dataset, tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        output.writestr("one.txt", "fresh archive caption")
    contents = archive.read_bytes().replace(
        b"fresh archive caption", b"wrong archive caption"
    )
    archive.write_bytes(contents)
    old = (dataset / "one.txt").read_bytes()
    with pytest.raises(zipfile.BadZipFile):
        images.extract_archive(archive, str(dataset))
    assert (dataset / "one.txt").read_bytes() == old
    assert not list(dataset.glob("*.tmp"))


def test_zip_traversal_and_existing_symlink_escape_are_skipped(dataset, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "one.txt").write_text("private")
    (dataset / "escape").symlink_to(external, target_is_directory=True)
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside.txt", "bad")
        output.writestr("escape/one.txt", "bad")
        output.writestr("new/two.txt", "good")
    result = images.extract_archive(archive, str(dataset))
    assert result["extracted"] == 1 and result["skipped"] == 2
    assert (external / "one.txt").read_text() == "private"
    assert (dataset / "new/two.txt").read_text() == "good"


def test_tar_member_read_failure_does_not_truncate_existing_file(
    dataset, tmp_path, monkeypatch
):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as output:
        entry = tarfile.TarInfo("one.txt")
        entry.size = 3
        output.addfile(entry, io.BytesIO(b"new"))
    old = (dataset / "one.txt").read_bytes()

    def fail_read(*args):
        raise OSError("archive read interrupted")

    monkeypatch.setattr(tarfile.ExFileObject, "read", fail_read)
    result = images.extract_archive(archive, str(dataset))
    assert result["extracted"] == 0 and result["skipped"] == 1
    assert (dataset / "one.txt").read_bytes() == old
    assert not list(dataset.glob("*.tmp"))


def test_tar_extracts_regular_files_and_skips_links(dataset, tmp_path):
    archive = tmp_path / "data.tar"
    with tarfile.open(archive, "w") as output:
        entry = tarfile.TarInfo("nested/new.txt")
        entry.size = 3
        output.addfile(entry, io.BytesIO(b"new"))
        link = tarfile.TarInfo("escape.txt")
        link.type = tarfile.SYMTYPE
        link.linkname = "../private.txt"
        output.addfile(link)
    result = images.extract_archive(archive, str(dataset))
    assert result["extracted"] == 1 and result["skipped"] == 1
    assert (dataset / "nested/new.txt").read_text() == "new"
    assert not (dataset / "escape.txt").is_symlink()


def test_upload_read_failure_cleans_temporary_file(tmp_path, monkeypatch):
    from webui.api import images as api

    original_temporary = api.tempfile.NamedTemporaryFile
    monkeypatch.setattr(
        api.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: original_temporary(dir=tmp_path, **kwargs),
    )

    async def fail_read(*args):
        raise OSError("upload interrupted")

    upload = SimpleNamespace(filename="data.zip", read=fail_read)
    with pytest.raises(OSError, match="upload interrupted"):
        asyncio.run(api.upload_archive(upload, str(tmp_path / "out")))
    assert list(tmp_path.iterdir()) == []


def test_tagger_uses_explicit_model_for_every_image(dataset, monkeypatch):
    from webui.services import wd_tagger_service as svc

    calls = []
    monkeypatch.setattr(
        svc,
        "get_settings",
        lambda: {
            "model_name": svc.DEFAULT_MODEL,
            "threshold": 0.35,
            "trigger_word": "",
        },
    )
    monkeypatch.setattr(svc, "_load_model", lambda model: None)

    def predict(path, threshold, *, model_name=None):
        calls.append(model_name)
        return {"rating": "general", "general_tags": [], "character_tags": []}

    monkeypatch.setattr(svc, "predict", predict)
    events = list(
        svc.tag_directory(
            str(dataset), skip_existing=False, model_name="wd-swinv2-tagger-v3"
        )
    )
    assert events[-1]["tagged"] == 1
    assert calls == ["wd-swinv2-tagger-v3"]


def test_tagger_preprocessing_composites_transparency(monkeypatch):
    import torch
    from PIL import Image
    from webui.services import wd_tagger_service as svc

    pixels = []

    def transform(image):
        pixels.append(image.getpixel((0, 0)))
        return torch.zeros((3, 1, 1))

    monkeypatch.setattr(svc, "_transform", transform)
    svc._preprocess(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
    assert pixels == [(255, 255, 255)]


@pytest.mark.parametrize(
    "payload", [{"threshold": -1}, {"threshold": 2}, {"model_name": "unknown"}]
)
@pytest.mark.parametrize("endpoint", ["/tag", "/settings"])
def test_invalid_tagger_requests_do_not_start_model_or_persist_settings(
    monkeypatch, payload, endpoint
):
    from webui.api import tagger

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid requests must fail before reaching the service")

    monkeypatch.setattr(tagger.svc, "tag_directory", unexpected)
    monkeypatch.setattr(tagger.svc, "save_settings", unexpected)
    app = FastAPI()
    app.include_router(tagger.router)
    client = TestClient(app)
    response = (
        client.post(endpoint, json={"directory": "unused", **payload})
        if endpoint == "/tag"
        else client.put(endpoint, json=payload)
    )
    assert response.status_code == 422


def test_model_deletion_refuses_resume_state_contents(tmp_path):
    from webui.services.model_service import delete_model

    state = tmp_path / "run-state"
    state.mkdir()
    checkpoint = state / "model.safetensors"
    checkpoint.write_bytes(b"resume state")
    deleted, error = delete_model(str(checkpoint))
    assert not deleted and "trainer" in error
    assert checkpoint.read_bytes() == b"resume state"


def test_curation_preference_failure_does_not_submit_job(monkeypatch, tmp_path):
    from webui.services import curation_service as svc

    monkeypatch.setattr(
        svc, "prepare", lambda *args: {"values": {}, "job_dir": str(tmp_path)}
    )

    def fail_preferences(*args):
        raise OSError("preference write failed")

    calls = []

    async def submit(context):
        calls.append(context)
        return {"id": "started"}

    monkeypatch.setattr(svc, "remember_values", fail_preferences)
    monkeypatch.setattr(svc, "_submit", submit)
    with pytest.raises(OSError, match="preference write failed"):
        asyncio.run(svc.submit("correct", {}))
    assert calls == []
