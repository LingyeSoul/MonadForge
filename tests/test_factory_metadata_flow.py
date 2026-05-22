"""Metadata-flow regression tests for ``create_network_from_weights``.

``load_file()`` discards safetensors ``__metadata__``, so a caller that
pre-loads tensors and passes ``weights_sd=`` used to silently drop the
three-axis routing stamps (ss_use_moe_style / ss_route_per_layer /
ss_router_source) and trip the "missing three-axis stamps" raise in
``LoRANetworkCfg.from_weights`` — blaming the checkpoint for a call-site fault.

These tests pin the de-footgun: metadata reaches the cfg via the explicit
``metadata=`` channel, via ``file=`` even when ``weights_sd=`` is also given,
via the plain ``file=`` path, and that the bare ``weights_sd=`` case raises an
error that names the real cause.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from safetensors.torch import save_file

from networks.lora_anima.factory import create_network_from_weights


# Class name must be "Block" to match LoRANetwork.ANIMA_TARGET_REPLACE_MODULE.
class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)


class _TinyDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()


_LORA = "lora_unet_block_proj"
_RANK = 4
_NUM_EXPERTS = 3
_MOE_META = {
    "ss_use_moe_style": "shared_A",
    "ss_route_per_layer": "True",
    "ss_router_source": "input",
}


def _moe_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic Hydra-moe (shared_A, per-layer input router) state dict.

    Only the key shapes are sniffed by the factory — the tensors are never
    loaded here (``create_network_from_weights`` returns before any
    ``load_state_dict``), so random values are fine.
    """
    return {
        f"{_LORA}.lora_down.weight": torch.randn(_RANK, 8),
        f"{_LORA}.lora_up_weight": torch.randn(_NUM_EXPERTS, 8, _RANK),
        f"{_LORA}.router.weight": torch.randn(_NUM_EXPERTS, _RANK),
        f"{_LORA}.alpha": torch.tensor(float(_RANK)),
    }


def _build(**kwargs):
    network, _sd = create_network_from_weights(
        multiplier=1.0,
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
        for_inference=True,
        **kwargs,
    )
    return network


def _assert_axes(network) -> None:
    assert network.cfg.use_moe_style == "shared_A"
    assert network.cfg.route_per_layer is True
    assert network.cfg.router_source == "input"


def test_metadata_kwarg_lands_three_axes():
    """Explicit ``metadata=`` carries the stamps even with a pre-loaded sd."""
    net = _build(file=None, weights_sd=_moe_state_dict(), metadata=dict(_MOE_META))
    _assert_axes(net)


def test_file_path_reads_metadata(tmp_path):
    """Regression: the plain ``file=`` path still reads the stamps."""
    path = tmp_path / "moe.safetensors"
    save_file(_moe_state_dict(), str(path), metadata=_MOE_META)
    net = _build(file=str(path), weights_sd=None)
    _assert_axes(net)


def test_file_recovers_metadata_when_weights_supplied(tmp_path):
    """``file=`` + ``weights_sd=`` together must still recover the stamps.

    Previously the read was gated on ``weights_sd is None`` so a caller that
    supplied both lost the metadata anyway.
    """
    path = tmp_path / "moe.safetensors"
    save_file(_moe_state_dict(), str(path), metadata=_MOE_META)
    net = _build(file=str(path), weights_sd=_moe_state_dict())
    _assert_axes(net)


def test_bare_weights_sd_raises_actionable_error():
    """No metadata, no file → loud error naming load_file / metadata=."""
    import pytest

    with pytest.raises(RuntimeError) as exc:
        _build(file=None, weights_sd=_moe_state_dict())
    msg = str(exc.value)
    assert "three-axis" in msg
    assert "load_file" in msg
    assert "metadata=" in msg
