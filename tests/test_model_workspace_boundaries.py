"""Model workspace boundaries using temporary files and mocked submissions only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from webui.api import moss, qwen21
from webui.services import (
    moss_service,
    moss_workspace,
    qwen21_service,
    qwen21_workspace,
)


@pytest.mark.parametrize("api", [qwen21, moss])
def test_unknown_action_field_is_rejected_before_submission(api, monkeypatch):
    async def unexpected_submission(*_args, **_kwargs):
        pytest.fail("Invalid input reached task submission")

    monkeypatch.setattr(api.task_service, "start_task", unexpected_submission)
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        response = client.post("/jobs/train", json={"values": {"rnak": 64}})
    assert response.status_code == 400
    assert "rnak" in response.json()["detail"]


@pytest.mark.parametrize("api", [qwen21, moss])
@pytest.mark.parametrize("endpoint", ["/jobs/train", "/chain"])
def test_runtime_name_is_validated_before_submission(api, endpoint, monkeypatch):
    async def unexpected_submission(*_args, **_kwargs):
        pytest.fail("Unknown runtime reached task submission")

    monkeypatch.setattr(api.task_service, "start_task", unexpected_submission)
    app = FastAPI()
    app.include_router(api.router)
    payload = (
        {"values": {}}
        if endpoint == "/jobs/train"
        else {"cache": {"values": {}}, "train": {"values": {}}}
    )
    with TestClient(app) as client:
        response = client.post(endpoint, json={**payload, "runtime_profile": "missing"})
    assert response.status_code == 422


def write_tensor_file(path: Path, tensors: dict, *, metadata=None, payload=b""):
    header = json.dumps({"__metadata__": metadata or {}, **tensors}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)


@pytest.mark.parametrize(
    ("tensors", "payload"),
    [
        ({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 0]}}, b""),
        ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [1, 5]}}, b"12345"),
        ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, b"12345"),
        (
            {
                "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "y": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            },
            b"1234",
        ),
    ],
    ids=["missing-payload", "gap", "trailing-payload", "overlap"],
)
def test_header_rejects_invalid_tensor_storage(tmp_path, tensors, payload):
    path = tmp_path / "invalid.safetensors"
    write_tensor_file(path, tensors, payload=payload)
    with pytest.raises(ValueError):
        qwen21_workspace.read_header(path)


@pytest.mark.parametrize("workspace", [qwen21_workspace, moss_workspace])
def test_cache_inspection_rejects_missing_tensor_payload(tmp_path, workspace):
    if workspace is qwen21_workspace:
        write_tensor_file(
            tmp_path / "sample.latent.safetensors",
            {"latents": {"dtype": "F32", "shape": [4, 64], "data_offsets": [0, 0]}},
            metadata={"latent_h": "2", "latent_w": "2"},
        )
        write_tensor_file(
            tmp_path / "sample.te.safetensors",
            {
                "prompt_embeds": {
                    "dtype": "BF16",
                    "shape": [1, 4096],
                    "data_offsets": [0, 0],
                },
                "prompt_embeds_mask": {
                    "dtype": "I64",
                    "shape": [1],
                    "data_offsets": [0, 0],
                },
            },
        )
    else:
        write_tensor_file(
            tmp_path / "sample_16x16.moss_latent.safetensors",
            {"latent": {"dtype": "F32", "shape": [128, 1, 1], "data_offsets": [0, 0]}},
            metadata={"width": "16", "height": "16"},
        )
        write_tensor_file(
            tmp_path / "sample.moss_te.safetensors",
            {"embeds": {"dtype": "BF16", "shape": [1, 4096], "data_offsets": [0, 0]}},
        )
    status = workspace.inspect_cache(str(tmp_path))
    assert not status.ready
    assert status.pairs == 0
    assert status.errors


def test_header_accepts_valid_scalar_empty_and_regular_tensors(tmp_path):
    path = tmp_path / "valid.safetensors"
    write_tensor_file(
        path,
        {
            "scalar": {"dtype": "I64", "shape": [], "data_offsets": [0, 8]},
            "empty": {"dtype": "F32", "shape": [0, 2], "data_offsets": [8, 8]},
            "value": {"dtype": "BF16", "shape": [2], "data_offsets": [8, 12]},
        },
        payload=b"\0" * 12,
    )
    assert set(qwen21_workspace.read_header(path).tensors) == {
        "scalar",
        "empty",
        "value",
    }


def test_qwen_manifest_cannot_point_at_another_job(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen21_service, "resolve_under_home", lambda value: tmp_path / value
    )
    root = tmp_path / "output/qwen21"
    first, second = root / "first", root / "second"
    first.mkdir(parents=True)
    second.mkdir()
    (second / "manifest.json").write_text(
        json.dumps({"images": [], "size": "32x32", "steps": 1, "lora": ""})
    )
    (first / "manifest.json").symlink_to(second / "manifest.json")
    with pytest.raises(ValueError, match="manifest|Manifest"):
        qwen21_service.read_results(str(first))


@pytest.mark.parametrize("workspace", [qwen21_workspace, moss_workspace])
def test_failed_profile_replace_keeps_original_and_removes_temporary(
    tmp_path, monkeypatch, workspace
):
    monkeypatch.setattr(workspace, "resolve_under_home", lambda value: tmp_path / value)
    original = workspace.save_profile("existing", workspace.read_profile("default"))
    path = workspace.profile_path("existing")
    original_bytes = path.read_bytes()
    modified = original.model_copy(deep=True)
    modified.train["rank"] = 64

    def failed_replace(*_args, **_kwargs):
        raise OSError("fixture: replacement failed")

    # Profiles are written through config_storage's staged writer, so the
    # injection point is Path.replace (the shared writer's commit step).
    monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(OSError, match="replacement failed"):
        workspace.save_profile("existing", modified)
    assert path.read_bytes() == original_bytes
    assert not [item for item in path.parent.iterdir() if item.name.endswith(".tmp")]
    assert workspace.read_profile("existing") == original


@pytest.mark.parametrize("workspace", [qwen21_workspace, moss_workspace])
def test_profile_round_trip_preserves_nullable_auto_values_and_ignores_legacy_fields(
    tmp_path, monkeypatch, workspace
):
    monkeypatch.setattr(workspace, "resolve_under_home", lambda value: tmp_path / value)
    profile = workspace.read_profile("default")
    profile.train.update(
        {"alpha": None, "retired_field": "old profile value", "rank": 48}
    )
    normalized = workspace.save_profile("copy", profile)
    restored = workspace.read_profile("copy")
    assert restored == normalized
    assert restored.train["alpha"] is None
    assert restored.train["rank"] == 48
    assert "retired_field" not in restored.train


@pytest.mark.parametrize("svc", [qwen21_service, moss_service])
def test_browser_output_paths_cannot_escape_model_root(tmp_path, monkeypatch, svc):
    monkeypatch.setattr(svc, "resolve_under_home", lambda value: tmp_path / value)
    model_name = "qwen21" if svc is qwen21_service else "moss"
    root = tmp_path / "output" / model_name
    root.mkdir(parents=True)
    (root / "external").symlink_to(tmp_path, target_is_directory=True)
    for value in ("output/anima/result", f"output/{model_name}/external/escaped"):
        with pytest.raises(ValueError):
            svc.output_path(value)
    assert svc.output_path(f"output/{model_name}/new-job") == root / "new-job"


@pytest.mark.parametrize("workspace", [qwen21_workspace, moss_workspace])
def test_valid_cache_is_ready_but_link_to_another_directory_is_not(tmp_path, workspace):
    cache = tmp_path / "cache"
    cache.mkdir()
    if workspace is qwen21_workspace:
        latent_path = cache / "sample.latent.safetensors"
        text_path = cache / "sample.te.safetensors"
        latent = {
            "latents": {"dtype": "F32", "shape": [4, 64], "data_offsets": [0, 1024]}
        }
        metadata = {"latent_h": "2", "latent_w": "2"}
        text = {
            "prompt_embeds": {
                "dtype": "BF16",
                "shape": [1, 4096],
                "data_offsets": [0, 8192],
            },
            "prompt_embeds_mask": {
                "dtype": "I64",
                "shape": [1],
                "data_offsets": [8192, 8200],
            },
        }
        latent_size, text_size = 1024, 8200
    else:
        latent_path = cache / "sample_16x16.moss_latent.safetensors"
        text_path = cache / "sample.moss_te.safetensors"
        latent = {
            "latent": {"dtype": "F32", "shape": [128, 1, 1], "data_offsets": [0, 512]}
        }
        metadata = {"width": "16", "height": "16"}
        text = {
            "embeds": {"dtype": "BF16", "shape": [1, 4096], "data_offsets": [0, 8192]}
        }
        latent_size, text_size = 512, 8192
    write_tensor_file(
        latent_path, latent, metadata=metadata, payload=b"\0" * latent_size
    )
    write_tensor_file(text_path, text, payload=b"\0" * text_size)
    status = workspace.inspect_cache(str(cache))
    assert status.ready and status.pairs == 1
    moved = tmp_path / text_path.name
    text_path.rename(moved)
    text_path.symlink_to(moved)
    assert not workspace.inspect_cache(str(cache)).ready


def test_moss_report_rejects_symlink_outside_output_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        moss_service, "resolve_under_home", lambda value: tmp_path / value
    )
    root = tmp_path / "output/moss"
    root.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text('{"private": "not a task result"}')
    (root / "train.json").symlink_to(outside)

    async def job(_task_id):
        return {
            "method": "moss-train",
            "kind": "command",
            "argv": [
                "tasks.py",
                "moss-train",
                "--output",
                str(root / "train.safetensors"),
            ],
        }

    monkeypatch.setattr(moss.daemon_client, "get_job", job)
    app = FastAPI()
    app.include_router(moss.router)
    with TestClient(app) as client:
        response = client.get("/tasks/test/report")
    assert response.status_code == 400
    assert "not a task result" not in response.text
