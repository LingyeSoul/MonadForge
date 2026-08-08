from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from library.preprocess.runs import resolve_preprocess_run


def _completed_run(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    return resolve_preprocess_run(
        source,
        {"target_res": [1024], "caption_shuffle_variants": 4},
        post_image_dataset=tmp_path / "post",
    )


def test_apply_preprocess_run_pins_all_training_cache_scalars(tmp_path):
    import train

    run = _completed_run(tmp_path)
    args = Namespace(preprocess_run=str(run.manifest_path))

    selected = train._apply_preprocess_run(args)

    assert selected == run
    assert args.resized_image_dir == str(run.resized_dir)
    assert args.lora_cache_dir == str(run.lora_dir)
    assert args.text_cache_dir == str(run.lora_dir)
    assert args.multires_image_dir == str(run.multires_dir)
    assert args.conditioning_data_dir == str(run.conditioning_data_dir)
    assert args.conditioning_resized_dir == str(run.conditioning_resized_dir)
    assert args.caption_index_path == str(run.caption_index_path)


def test_apply_preprocess_run_rejects_incomplete_manifest(tmp_path):
    import train

    run = _completed_run(tmp_path)
    run.write_manifest(status="failed", error="interrupted")

    with pytest.raises(ValueError, match="invalid --preprocess_run manifest"):
        train._apply_preprocess_run(Namespace(preprocess_run=str(run.manifest_path)))


def test_validate_preprocess_run_rejects_mixed_cache_root(tmp_path):
    import train

    run = _completed_run(tmp_path)
    group = SimpleNamespace(
        datasets=[
            SimpleNamespace(
                subsets=[
                    SimpleNamespace(
                        image_dir=run.resized_dir,
                        cache_dir=tmp_path / "legacy" / "lora",
                        cond_cache_dir=None,
                    )
                ]
            )
        ]
    )

    with pytest.raises(ValueError, match="cache mixing detected"):
        train._validate_preprocess_dataset_paths(group, run)


def test_validate_preprocess_run_rejects_mixed_text_cache_root(tmp_path):
    import train

    run = _completed_run(tmp_path)
    group = SimpleNamespace(
        datasets=[
            SimpleNamespace(
                subsets=[
                    SimpleNamespace(
                        image_dir=run.resized_dir,
                        cache_dir=run.lora_dir,
                        text_cache_dir=tmp_path / "legacy" / "text",
                        cond_cache_dir=None,
                    )
                ]
            )
        ]
    )

    with pytest.raises(ValueError, match="text_cache_dir"):
        train._validate_preprocess_dataset_paths(group, run)


def test_validate_preprocess_run_allows_text_cache_in_run(tmp_path):
    import train

    run = _completed_run(tmp_path)
    group = SimpleNamespace(
        datasets=[
            SimpleNamespace(
                subsets=[
                    SimpleNamespace(
                        image_dir=run.resized_dir,
                        cache_dir=run.lora_dir,
                        text_cache_dir=run.lora_dir / "text",
                        cond_cache_dir=None,
                    )
                ]
            )
        ]
    )

    train._validate_preprocess_dataset_paths(group, run)


def test_no_preprocess_run_preserves_legacy_args():
    import train

    args = Namespace(
        preprocess_run=None,
        resized_image_dir="legacy/resized",
        lora_cache_dir="legacy/lora",
    )

    assert train._apply_preprocess_run(args) is None
    assert args.resized_image_dir == "legacy/resized"
    assert args.lora_cache_dir == "legacy/lora"
