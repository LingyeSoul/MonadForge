from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from library.anima.strategy import AnimaTextEncoderOutputsCachingStrategy


def _write(path, *, adapter: bool, complete: bool = True, randomized: bool = False):
    data = {"caption_dropout_rate": torch.tensor(0.0)}
    if adapter:
        data["crossattn_emb_v0"] = torch.zeros(2, 4)
        if complete:
            data["t5_attn_mask_v0"] = torch.ones(2, dtype=torch.int32)
    else:
        data["prompt_embeds_v0"] = torch.zeros(2, 4)
        if complete:
            data["attn_mask_v0"] = torch.ones(2, dtype=torch.int32)
            data["t5_input_ids_v0"] = torch.zeros(2, dtype=torch.long)
            data["t5_attn_mask_v0"] = torch.ones(2, dtype=torch.int32)
    data["num_variants"] = torch.tensor(1)
    if randomized:
        # Keep the v-family valid and add the writer's randomized marker, but
        # intentionally omit r1 so the preflight must reject the incomplete
        # cache before the loader can select it.
        data["num_randomized"] = torch.tensor(1)
    save_file(data, str(path))


def test_skip_check_still_validates_required_plain_keys(tmp_path):
    path = tmp_path / "cache.safetensors"
    _write(path, adapter=False, complete=False)
    strategy = AnimaTextEncoderOutputsCachingStrategy(True, 1, True)
    assert not strategy.is_disk_cached_outputs_expected(str(path))


def test_adapter_cache_mode_mismatch_fails_before_loader(tmp_path):
    path = tmp_path / "cache.safetensors"
    _write(path, adapter=True)
    strategy = AnimaTextEncoderOutputsCachingStrategy(True, 1, True)
    with pytest.raises(RuntimeError, match="cache_llm_adapter_outputs=true"):
        strategy.is_disk_cached_outputs_expected(str(path))


def test_plain_cache_mode_mismatch_suggests_false_or_reprocess(tmp_path):
    path = tmp_path / "cache.safetensors"
    _write(path, adapter=False)
    strategy = AnimaTextEncoderOutputsCachingStrategy(
        True, 1, True, cache_llm_adapter_outputs=True
    )
    with pytest.raises(RuntimeError, match="cache_llm_adapter_outputs=false"):
        strategy.is_disk_cached_outputs_expected(str(path))


def test_randomized_marker_requires_complete_r_family_even_when_skip_check(tmp_path):
    path = tmp_path / "cache.safetensors"
    _write(path, adapter=False, randomized=True)
    strategy = AnimaTextEncoderOutputsCachingStrategy(
        True, 1, True, use_randomized_caption_variants=True
    )
    assert not strategy.is_disk_cached_outputs_expected(str(path))


def test_unused_randomized_family_does_not_block_plain_training(tmp_path):
    path = tmp_path / "cache.safetensors"
    _write(path, adapter=False, randomized=True)
    strategy = AnimaTextEncoderOutputsCachingStrategy(True, 1, True)
    assert strategy.is_disk_cached_outputs_expected(str(path))
