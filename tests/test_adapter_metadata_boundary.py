"""Reject a mismatched model family using only safetensors metadata."""

from __future__ import annotations

import pytest

from library.moss.lora import LoRANetwork as MossNetwork
from library.qwen21.lora import LoRANetwork as QwenNetwork


@pytest.mark.parametrize(
    "loader,family", [(MossNetwork, "Qwen-Image-2.1"), (QwenNetwork, "MOSS-Image")]
)
def test_wrong_family_rejected_before_tensor_reads(monkeypatch, loader, family):
    class Header:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def metadata(self):
            return {"base_model": family}

        def keys(self):
            pytest.fail("cross-family weights must be rejected before tensor reads")

    monkeypatch.setattr("safetensors.safe_open", lambda *a, **kw: Header())
    with pytest.raises(ValueError, match="base_model"):
        loader.from_file(object(), "/not-read/model.safetensors")
