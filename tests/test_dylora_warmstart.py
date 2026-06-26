"""Warm-start / inference round-trip regression tests for DyLoRA.

Pins two bugs found during review of the DyLoRA registry wiring:

Bug 3 (real): ``LoRANetworkCfg.from_weights`` accepted ``dylora_unit`` /
``dylora_algo`` but never set ``use_dylora``, so the cfg fell to its dataclass
default ``use_dylora=False``. ``network.py`` keys the per-module ``unit``/
``algo`` injection off ``cfg.use_dylora`` → the knobs never reached the module,
and a warm-started DyLoRA checkpoint silently used ``unit=1``.

Bug 2 (latent): ``save_weights`` guarded the ``ss_network_spec`` stamp behind
``if metadata:`` (truthiness), so an empty dict dropped the stamp and
``create_network_from_weights`` could no longer route the checkpoint back to
DyLoRA (its ``lora_down``/``lora_up`` keys are indistinguishable from plain
LoRA, so routing relies entirely on the metadata stamp).

These round-trips build a synthetic DyLoRA checkpoint, route it through
``create_network_from_weights``, and assert the knobs survive.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from networks.lora_anima.factory import create_network_from_weights


# Class name must be exactly "Block" to match LoRANetwork.ANIMA_TARGET_REPLACE_MODULE
# (create_modules matches module.__class__.__name__ against the target list).
class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)


class _TinyDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = Block()


_LORA = "lora_unet_block_proj"
_RANK = 4
_UNIT = 4  # must divide _RANK; non-default so a regression to unit=1 is caught.
_ALGO = "switch"


def _dylora_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic DyLoRA state dict.

    DyLoRA's on-disk keys are identical to plain LoRA (``lora_down.weight`` /
    ``lora_up.weight``), so the factory CANNOT key-sniff it — routing depends
    entirely on the ``ss_network_spec="dylora"`` metadata stamp.
    """
    return {
        f"{_LORA}.lora_down.weight": torch.randn(_RANK, 8),
        f"{_LORA}.lora_up.weight": torch.randn(8, _RANK),
        f"{_LORA}.alpha": torch.tensor(float(_RANK)),
    }


def _dylora_metadata() -> dict[str, str]:
    return {
        "ss_network_spec": "dylora",
        "ss_dylora_unit": str(_UNIT),
        "ss_dylora_algo": _ALGO,
    }


def _build(**kwargs):
    network, _sd = create_network_from_weights(
        multiplier=1.0,
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
        **kwargs,
    )
    return network


def test_dylora_warmstart_preserves_unit_and_algo():
    """Bug 3 regression: a warm-started DyLoRA checkpoint must carry
    ``unit``/``algo`` into the reconstructed modules.

    Before the fix, ``cfg.use_dylora`` stayed ``False`` on the load path, the
    per-module ``extra_kwargs`` injection (``network.py: if cfg.use_dylora
    and ...``) short-circuited, and every DyLoRAModule was built with the
    default ``unit=1``.
    """
    net = _build(file=None, weights_sd=_dylora_state_dict(), metadata=_dylora_metadata())

    # The cfg selector must be threaded through (this is what gates the
    # extra_kwargs injection in network.py).
    assert net.cfg.use_dylora is True, (
        "from_weights dropped use_dylora — modules will be built without unit/algo"
    )
    assert net.cfg.dylora_unit == _UNIT
    assert net.cfg.dylora_algo == _ALGO

    # And the knobs must actually reach the modules.
    assert len(net.unet_loras) >= 1, "no unet_loras constructed"
    mod = net.unet_loras[0]
    assert mod.unit == _UNIT, f"DyLoRA module unit={mod.unit}, expected {_UNIT} (injection failed)"
    assert mod.algo == _ALGO, f"DyLoRA module algo={mod.algo!r}, expected {_ALGO!r}"


def test_dylora_warmstart_routes_via_metadata_stamp():
    """Routing depends on ``ss_network_spec="dylora"``: without it the factory
    cannot distinguish a DyLoRA checkpoint from plain LoRA (identical keys)
    and falls back to the LoRA spec. This test documents that contract by
    confirming the stamp IS what selects DyLoRA."""
    net = _build(file=None, weights_sd=_dylora_state_dict(), metadata=_dylora_metadata())
    assert net._network_spec.name == "dylora"
    # The constructed module must be a DyLoRAModule, not a plain LoRAModule.
    from networks.lora_modules import DyLoRAModule

    assert isinstance(net.unet_loras[0], DyLoRAModule), (
        f"expected DyLoRAModule, got {type(net.unet_loras[0]).__name__} "
        "(ss_network_spec stamp not honored?)"
    )


def test_save_weights_stamps_spec_even_on_empty_metadata(tmp_path):
    """Bug 2 regression: ``save_weights`` must stamp ``ss_network_spec`` even
    when the caller passes an empty dict (previously the ``if metadata:``
    truthiness guard swallowed it).

    A library consumer calling ``network.save_weights(f, dt, {})`` used to
    produce a checkpoint that silently degrades to plain LoRA on reload.
    """
    from networks.lora_anima.factory import create_network

    # Build a fresh DyLoRA network (train-time create path) so save_weights
    # has a real network with a populated _network_spec. create_network returns
    # a single LoRANetwork (unlike create_network_from_weights → tuple).
    net = create_network(
        multiplier=1.0,
        network_dim=_RANK,
        network_alpha=_RANK,
        vae=None,
        text_encoders=[],
        unet=_TinyDiT(),
        use_dylora="true",
        dylora_unit=str(_UNIT),
    )

    # Save with an EMPTY metadata dict — the previous guard dropped the stamp.
    out = tmp_path / "dylora.safetensors"
    net.save_weights(str(out), torch.float32, metadata={})

    # Reload via create_network_from_weights — must route back to DyLoRA.
    reloaded = _build(file=str(out), weights_sd=None)
    assert reloaded._network_spec.name == "dylora", (
        "save_weights with empty metadata did not stamp ss_network_spec — "
        "checkpoint can no longer be routed back to DyLoRA"
    )
    # And the unit survives the round-trip.
    assert reloaded.cfg.dylora_unit == _UNIT
