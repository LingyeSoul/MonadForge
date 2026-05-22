"""Build the kwargs dict (and rewrite ``crossattn_emb``) for the DiT forward.

One concern sits here:

* **Postfix injection** — for networks with ``append_postfix``, splice the
  learned postfix vectors onto the cached T5 embedding and pool the *real*
  text BEFORE the splice so modulation guidance only sees real text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(slots=True)
class ForwardKwargs:
    """Outputs of ``build_forward_kwargs``.

    ``crossattn_emb`` is the (possibly postfix-extended) embedding; ``kw``
    is the dict of additional kwargs to pass to the DiT call.
    """

    crossattn_emb: torch.Tensor
    kw: dict[str, Any]
    has_postfix: bool


def build_forward_kwargs(
    *,
    network: Any,
    crossattn_emb: torch.Tensor,
    t5_attn_mask: Optional[torch.Tensor],
    timesteps: torch.Tensor,
) -> ForwardKwargs:
    has_postfix = hasattr(network, "append_postfix")
    kw: dict[str, Any] = {}

    if has_postfix:
        # Pool text BEFORE injection so modulation guidance sees only real text.
        kw["pooled_text_override"] = crossattn_emb.max(dim=1).values
        seqlens = t5_attn_mask.sum(dim=-1).to(torch.int32)
        crossattn_emb = network.append_postfix(
            crossattn_emb, seqlens, timesteps=timesteps
        )

    return ForwardKwargs(crossattn_emb=crossattn_emb, kw=kw, has_postfix=has_postfix)
