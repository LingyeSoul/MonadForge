"""`write_gen_manifest` — the inference-side half of the daemon result-envelope
lift (proposal Phase 1a). Under a daemon spawn it drops a generation manifest +
`result_path.json` pointer; inline it's a no-op so `python inference.py` stays
standalone.
"""

from __future__ import annotations

import argparse
import json

from library.inference import write_gen_manifest


def _args(**kw) -> argparse.Namespace:
    base = dict(
        lora_weight="output/ckpt/foo.safetensors",
        infer_steps=28,
        guidance_scale=4.0,
        sampler="euler",
        save_path="output/tests",
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_inline_is_noop(monkeypatch):
    monkeypatch.delenv("ANIMA_DAEMON_JOB_DIR", raising=False)
    assert write_gen_manifest(_args(), ["a.png", "b.png"]) is None


def test_daemon_writes_manifest_and_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMA_DAEMON_JOB_DIR", str(tmp_path))
    images = ["output/tests/x.png", "output/tests/y.png"]
    mp = write_gen_manifest(_args(), images)

    manifest = json.loads((tmp_path / "gen_manifest.json").read_text())
    assert manifest["label"] == "foo"  # derived from the adapter stem
    assert manifest["metrics"]["n_images"] == 2
    assert manifest["metrics"]["infer_steps"] == 28
    assert manifest["images"] == images

    pointer = json.loads((tmp_path / "result_path.json").read_text())
    assert pointer["path"] == mp


def test_label_falls_back_to_base_without_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMA_DAEMON_JOB_DIR", str(tmp_path))
    write_gen_manifest(_args(lora_weight=None), [])
    manifest = json.loads((tmp_path / "gen_manifest.json").read_text())
    assert manifest["label"] == "base"
    assert manifest["metrics"]["n_images"] == 0


def test_label_uses_first_of_stacked_adapters(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMA_DAEMON_JOB_DIR", str(tmp_path))
    write_gen_manifest(_args(lora_weight=["output/ckpt/bar.safetensors", "b.st"]), [])
    manifest = json.loads((tmp_path / "gen_manifest.json").read_text())
    assert manifest["label"] == "bar"


def test_explicit_label_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMA_DAEMON_JOB_DIR", str(tmp_path))
    write_gen_manifest(_args(), [], label="my-sweep")
    manifest = json.loads((tmp_path / "gen_manifest.json").read_text())
    assert manifest["label"] == "my-sweep"
