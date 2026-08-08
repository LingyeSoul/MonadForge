from __future__ import annotations

from pathlib import Path

from library.preprocess.runs import resolve_preprocess_run


def test_caption_index_stage_writes_into_active_run(monkeypatch, tmp_path: Path):
    from scripts.tasks import preprocess

    source = tmp_path / "source"
    source.mkdir()
    run = resolve_preprocess_run(
        source,
        {"target_res": [1024], "path_pattern": "*"},
        post_image_dataset=tmp_path / "post",
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(preprocess, "run", lambda command: calls.append(command))

    with preprocess._use_preprocess_run(run):
        preprocess.cmd_caption_index([])

    command = calls[0]
    assert command[1] == "scripts/preprocess/build_caption_index.py"
    assert command[command.index("--src") + 1] == str(run.source_dir)
    assert command[command.index("--out") + 1] == str(run.caption_index_path)
