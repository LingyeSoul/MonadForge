"""Cross-cutting conditioning side-channels for the denoise-loop runners.

Every denoise-loop variant — the standard loop in ``generation.generate_body``
and the ``--spectrum`` / ``--spd`` runners in ``networks/`` — threads the same
block of side-channel args: adapter routing (P-GRAFT, soft-tokens), CFG/SNR
corrections (DCW, SMC-CFG), and the pooled-text override. These are orthogonal
to each sampler's own knobs (Spectrum's window/Chebyshev params, SPD's
resolution stages), so bundling them keeps the runner signatures focused on
what is actually sampler-specific.

``generate_body`` builds one ``SamplerSideChannels`` and hands it to whichever
runner is active. Adding a new side-channel means one field here plus the
``from_args`` build site — not a new kwarg in every runner signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:  # pragma: no cover
    from library.inference.corrections.smc_cfg import SMCCFGState


@dataclass(frozen=True)
class SamplerSideChannels:
    """Conditioning side-channels shared across all denoise-loop variants."""

    pgraft_network: Any = None
    lora_cutoff_step: Optional[int] = None
    pooled_text_pos: Optional[torch.Tensor] = None
    pooled_text_neg: Optional[torch.Tensor] = None
    dcw: bool = False
    dcw_lambda: float = -0.015
    dcw_schedule: str = "one_minus_sigma"
    dcw_band_mask: str = "LL"
    dcw_calibrator: Any = None
    smc_cfg: "Optional[SMCCFGState]" = None
    soft_tokens_net: Any = None
    soft_tokens_embed_seqlens: Optional[torch.Tensor] = None
    soft_tokens_neg_seqlens: Optional[torch.Tensor] = None

    @classmethod
    def from_args(
        cls,
        args,
        *,
        pgraft_network: Any = None,
        lora_cutoff_step: Optional[int] = None,
        pooled_text_pos: Optional[torch.Tensor] = None,
        pooled_text_neg: Optional[torch.Tensor] = None,
        dcw_calibrator: Any = None,
        smc_cfg: "Optional[SMCCFGState]" = None,
        soft_tokens_net: Any = None,
        soft_tokens_embed_seqlens: Optional[torch.Tensor] = None,
        soft_tokens_neg_seqlens: Optional[torch.Tensor] = None,
    ) -> "SamplerSideChannels":
        """Build from parsed CLI ``args`` plus the runtime objects ``generate_body``
        already holds. The DCW scalar defaults live here so the two runner call
        sites don't each repeat the ``getattr(args, ...)`` block.
        """
        return cls(
            pgraft_network=pgraft_network,
            lora_cutoff_step=lora_cutoff_step,
            pooled_text_pos=pooled_text_pos,
            pooled_text_neg=pooled_text_neg,
            dcw=getattr(args, "dcw", False),
            dcw_lambda=getattr(args, "dcw_lambda", -0.015),
            dcw_schedule=getattr(args, "dcw_schedule", "one_minus_sigma"),
            dcw_band_mask=getattr(args, "dcw_band_mask", "LL"),
            dcw_calibrator=dcw_calibrator,
            smc_cfg=smc_cfg,
            soft_tokens_net=soft_tokens_net,
            soft_tokens_embed_seqlens=soft_tokens_embed_seqlens,
            soft_tokens_neg_seqlens=soft_tokens_neg_seqlens,
        )
