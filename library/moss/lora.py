"""MOSS-Image's adapters: ``library/lines/adapters.py`` bound to this DiT.

Default targets are the six bias-free linears of every block,
``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`` (42 × 6 = 252). The fused
``ffn_up_gate`` gets one adapter over its full 2·ffn_dim output, so the hidden
and the gate halves share the rank.
"""

from __future__ import annotations

from library.lines.adapters import (  # noqa: F401  (the line's adapter surface)
    ALGOS,
    DOWN_INITS,
    SAVED_DTYPES,
    LoHaAdapter,
    LoKrAdapter,
    LoRAAdapter,
    channel_scale,
    load_channel_stats,
)
from library.lines.adapters import LoRANetwork as _LoRANetwork
from library.moss.requests import TrainRequest

DEFAULT_TARGETS = TrainRequest.TARGETS


class LoRANetwork(_LoRANetwork):
    BASE_MODEL = "MOSS-Image"
    BLOCKS = "layers"
    DEFAULT_TARGETS = DEFAULT_TARGETS


load_network = LoRANetwork.from_file
load_for_inference = LoRANetwork.for_inference


def ref_timestep(path) -> float | None:
    """The ``ref_timestep`` a reference LoRA trained with; None = the image's
    (stamped negative, or absent: the LoRAs from before the flag)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        value = (handle.metadata() or {}).get("ref_timestep")
    return None if value is None or float(value) < 0 else float(value)
