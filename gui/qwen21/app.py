"""Qwen-Image-2.1 LoRA window — the ``gui.lines`` window over ``library.qwen21``.

Sidecars ``scripts/qwen21/{cache,train,generate}.py``. A Test is one job: the
generate sidecar renders every ``multipliers`` scale from one loaded model.
"""

from __future__ import annotations

import json
from pathlib import Path

from gui.lines.window import LineSpec, run
from gui.qwen21.strings import FIELDS_CN, FIELDS_CN_GENERATE, UI
from library.qwen21.requests import (
    CacheRequest,
    GenerateRequest,
    TrainRequest,
    resolve_model_dir,
)
from library.qwen21.scan import cache_counts, scan


def _test_images(run_dir: Path) -> list[tuple[Path, float, str]]:
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return []
    entries = json.loads(manifest.read_text(encoding="utf-8"))["images"]
    return [(run_dir / e["file"], e["multiplier"], e["prompt"]) for e in entries]


SPEC = LineSpec(
    key="qwen21",
    cache_cls=CacheRequest,
    train_cls=TrainRequest,
    test_cls=GenerateRequest,
    model_index=lambda model_dir, _variant: (
        resolve_model_dir(model_dir) / "model_index.json"
    ),
    resolve_model_dir=resolve_model_dir,
    scan=scan,
    cache_counts=cache_counts,
    test_runs=lambda req: [req],
    test_images=_test_images,
    ui=UI,
    fields_cn=FIELDS_CN,
    fields_cn_test=FIELDS_CN_GENERATE,
)


def main() -> None:
    run(SPEC)
