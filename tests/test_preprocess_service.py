"""WebUI preprocess service — ``target_res`` read/write round-trip.

Regression: the preprocess view used to expose a vestigial ``resize_resolution``
scalar that the free-fit resize step silently ignores (``library/preprocess/
images.py`` drops the ``--resolution`` value). The real knob is the multi-scale
``target_res`` tier list. These tests pin the service-layer behaviour that
replaced it: reading the live config value and persisting it back to
``configs/preprocess.toml`` without clobbering the other user-owned keys.
"""

from __future__ import annotations

import json

import toml

from webui.services import preprocess_service as svc


def test_normalize_target_res_drops_invalid_and_sorts():
    assert svc._normalize_target_res([1024, 896, 896]) == [896, 1024]
    assert svc._normalize_target_res([1024, 999, 1337]) == [1024]  # 999/1337 ignored
    assert svc._normalize_target_res("768, 1024") == [768, 1024]
    assert svc._normalize_target_res(512) == [512]
    assert svc._normalize_target_res([]) == [1024]  # never an empty tier set
    assert svc._normalize_target_res(None) == [1024]


def test_get_target_res_reads_live_config_not_path_only_projection(monkeypatch):
    """``config_service.get_path_overrides`` projects only the five dataset-path
    keys, so it would drop ``target_res``. ``get_target_res`` must read the full
    merged chain instead — otherwise the UI silently falls back to [1024]."""
    monkeypatch.delenv("METHOD", raising=False)

    captured: dict = {}

    def fake_load(preset, method, methods_subdir):
        captured["args"] = (preset, method, methods_subdir)
        return {"target_res": [1024, 896]}

    monkeypatch.setattr("library.config.io.load_path_overrides", fake_load)

    assert svc.get_target_res() == [896, 1024]
    # No METHOD env → the default "methods" subdir is used (not gui-methods).
    assert captured["args"][2] == "methods"


def test_save_target_res_preserves_other_keys(tmp_path, monkeypatch):
    """Writing ``target_res`` must round-trip the file so the other user-owned
    knobs (freefit_max_ratio, caption_*, …) survive untouched."""
    pp = tmp_path / "preprocess.toml"
    pp.write_text(
        "target_res = [1024, 896]\nfreefit_max_ratio = 4.0\n"
        "caption_shuffle_variants = 4\nmin_pixels = 250000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(svc, "PREPROCESS_TOML", pp)
    monkeypatch.setattr(svc, "CONFIGS_DIR", tmp_path)

    result = svc.save_target_res([1024, 1536])
    assert result == [1024, 1536]

    data = toml.loads(pp.read_text(encoding="utf-8"))
    assert data["target_res"] == [1024, 1536]
    # The other keys are preserved.
    assert data["freefit_max_ratio"] == 4.0
    assert data["caption_shuffle_variants"] == 4
    assert data["min_pixels"] == 250000


def test_save_target_res_does_not_clobber_corrupt_file(tmp_path, monkeypatch):
    """A corrupt ``preprocess.toml`` must raise (NOT be silently rewritten
    from an empty dict), otherwise the other user-owned keys are wiped with
    no signal to the caller. The corrupt content must be left on disk so the
    user can recover it."""
    pp = tmp_path / "preprocess.toml"
    pp.write_text("target_res = [this is not valid toml >>>", encoding="utf-8")
    monkeypatch.setattr(svc, "PREPROCESS_TOML", pp)
    monkeypatch.setattr(svc, "CONFIGS_DIR", tmp_path)

    import pytest

    with pytest.raises(toml.TomlDecodeError):
        svc.save_target_res([1024])
    # The corrupt bytes are untouched — no partial/empty write landed.
    assert "this is not valid toml" in pp.read_text(encoding="utf-8")


def test_save_target_res_atomic_write(tmp_path, monkeypatch):
    """The write must be atomic (temp file + os.replace): no ``*.tmp`` leftover
    after a successful save, and the final file parses back to the saved
    value."""
    pp = tmp_path / "preprocess.toml"
    monkeypatch.setattr(svc, "PREPROCESS_TOML", pp)
    monkeypatch.setattr(svc, "CONFIGS_DIR", tmp_path)

    assert svc.save_target_res([1024, 896]) == [896, 1024]

    assert not (tmp_path / "preprocess.toml.tmp").exists()
    data = toml.loads(pp.read_text(encoding="utf-8"))
    assert data["target_res"] == [896, 1024]


def test_multires_per_image_round_trips_without_clobbering_tiers(tmp_path, monkeypatch):
    pp = tmp_path / "preprocess.toml"
    pp.write_text(
        "target_res = [512, 1024]\nfreefit_max_ratio = 4.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(svc, "PREPROCESS_TOML", pp)
    monkeypatch.setattr(svc, "CONFIGS_DIR", tmp_path)

    assert svc.save_multires_per_image(True) is True
    data = toml.loads(pp.read_text(encoding="utf-8"))
    assert data["multires_per_image"] is True
    assert data["target_res"] == [512, 1024]
    assert data["freefit_max_ratio"] == 4.0


def test_get_multires_per_image_reads_config_chain(monkeypatch):
    monkeypatch.delenv("METHOD", raising=False)
    monkeypatch.setattr(
        "library.config.io.load_path_overrides",
        lambda preset, method, methods_subdir: {"multires_per_image": True},
    )

    assert svc.get_multires_per_image() is True


def test_settings_carries_target_res_not_resize_resolution(monkeypatch):
    """``get_settings`` surfaces ``target_res`` and no longer mentions the dead
    ``resize_resolution`` scalar."""
    monkeypatch.delenv("METHOD", raising=False)
    monkeypatch.setattr(
        "library.config.io.load_path_overrides",
        lambda preset, method, methods_subdir: {"target_res": [1024, 896]},
    )
    monkeypatch.setattr(svc, "_load_sam", dict)
    monkeypatch.setattr(svc, "_load_gui_settings", dict)

    s = svc.get_settings()
    assert s["target_res"] == [896, 1024]
    assert s["multires_per_image"] is False
    assert s["mit_ctd_gate"] is True
    assert "resize_resolution" not in s


def test_ctd_gate_round_trips_gui_settings(tmp_path, monkeypatch):
    settings_file = tmp_path / "webui_settings.json"
    sam_yaml = tmp_path / "sam_mask.yaml"
    monkeypatch.setattr(svc, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(svc, "SAM_YAML", sam_yaml)
    monkeypatch.setattr(svc, "get_target_res", lambda: [1024])
    monkeypatch.setattr(svc, "get_multires_per_image", lambda: False)
    monkeypatch.setattr(svc, "_load_sam", dict)

    saved = svc.save_settings({"sam": {}, "mit_ctd_gate": False})

    assert saved["mit_ctd_gate"] is False
    assert json.loads(settings_file.read_text(encoding="utf-8"))["mit_ctd_gate"] is False


def test_preprocess_api_rejects_multires_with_one_tier():
    import pytest
    from pydantic import ValidationError

    from webui.api.preprocess import PreprocessSettings

    with pytest.raises(ValidationError, match="at least two target_res tiers"):
        PreprocessSettings(target_res=[1024], multires_per_image=True)


def test_preprocess_api_rejects_unknown_target_tier():
    import pytest
    from pydantic import ValidationError

    from webui.api.preprocess import PreprocessSettings

    with pytest.raises(ValidationError, match="unsupported target_res"):
        PreprocessSettings(target_res=[999, 1024])


def test_run_listing_status_and_manifest_paths(tmp_path, monkeypatch):
    from library.preprocess.runs import resolve_preprocess_run

    source = tmp_path / "nested" / "source"
    source.mkdir(parents=True)
    post = tmp_path / "post"
    run = resolve_preprocess_run(
        source,
        {"target_res": [1024], "caption_shuffle_variants": 4},
        post_image_dataset=post,
    )
    (run.resized_dir / "a.png").write_bytes(b"png")
    (run.lora_dir / "a_anima.npz").write_bytes(b"npz")
    run.write_manifest(
        status="ready",
        artifacts={"resized": 1, "latents": 1, "te": 0, "pe": 0},
        updated_at=123.0,
    )

    monkeypatch.setattr(svc, "ROOT", tmp_path)
    monkeypatch.setattr(
        svc,
        "get_path_overrides",
        lambda preset="default", variant=None: {
            "source_image_dir": str(source),
            "resized_image_dir": str(post / "resized"),
            "lora_cache_dir": str(post / "lora"),
            "conditioning_data_dir": str(tmp_path / "conditioning"),
            "conditioning_resized_dir": str(post / "cond_resized"),
        },
    )

    listed = svc.list_runs(source=str(source))
    assert [item["manifest"] for item in listed] == [str(run.manifest_path.resolve())]
    assert listed[0]["status"] == "ready"
    assert listed[0]["artifacts"]["latents"] == 1

    status = svc.get_status(manifest=str(run.manifest_path))
    assert status["resized"] == 1
    assert status["cache"]["latents"] == 1
    assert status["preprocess_run"]["complete"] is True

    paths = svc.get_paths(manifest=str(run.manifest_path))
    assert paths["source_image_dir"] == str(source.resolve())
    assert paths["resized_image_dir"] == str(run.resized_dir)
    assert paths["lora_cache_dir"] == str(run.lora_dir)


def test_manifest_paths_reject_incomplete_run(tmp_path):
    import pytest

    from library.preprocess.runs import PreprocessRunError, resolve_preprocess_run

    source = tmp_path / "source"
    source.mkdir()
    run = resolve_preprocess_run(
        source, {"target_res": [1024]}, post_image_dataset=tmp_path / "post"
    )
    run.write_manifest(status="failed", error="worker failed")

    with pytest.raises(PreprocessRunError, match="Incomplete preprocess manifest"):
        svc.get_paths(manifest=str(run.manifest_path))
