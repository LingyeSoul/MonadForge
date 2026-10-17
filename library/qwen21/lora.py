"""Qwen-Image-2.1's adapters: ``library/lines/adapters.py`` bound to this DiT.

Default targets are the single-stream block's attention projections plus the
SwiGLU's three (``requests.DEFAULT_TARGETS``).
"""

from __future__ import annotations

from library.lines.adapters import (  # noqa: F401  (the line's adapter surface)
    ALGOS,
    DOWN_INITS,
    SAVED_DTYPES,
    LoHaAdapter,
    LoKrAdapter,
    LoRAAdapter,
)
from library.lines.adapters import LoRANetwork as _LoRANetwork
from library.qwen21.requests import DEFAULT_TARGETS


class LoRANetwork(_LoRANetwork):
    BASE_MODEL = "Qwen-Image-2.1"
    BLOCKS = "transformer_blocks"
    DEFAULT_TARGETS = DEFAULT_TARGETS


load_network = LoRANetwork.from_file
