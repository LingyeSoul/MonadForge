# Refs:
#   https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
#   https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import contextlib
import random

import torch
from library.log import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


def _absorb_channel_scale(
    weight: torch.Tensor, channel_scale: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """SmoothQuant-style channel-scale absorption into a Linear's input columns.

    Mutates ``weight`` ([out, in]) so ``W[:, c] *= s_norm[c]`` and returns
    ``inv_scale = 1 / s_norm`` (caller applies ``x * inv_scale`` at forward).
    Output is unchanged; the point is to rebalance per-column gradient magnitudes
    so each column's ``∂L/∂W[:,c]`` no longer scales with ``|x[c]|^2``.
    See ``_archive/bench/channel_stats/channel_dominance_analysis.md``.
    """
    assert channel_scale.ndim == 1, (
        f"channel_scale must be 1D, got shape {tuple(channel_scale.shape)}"
    )
    assert channel_scale.shape[0] == weight.shape[1], (
        f"channel_scale length {channel_scale.shape[0]} does not match "
        f"weight in_features {weight.shape[1]}"
    )
    s = channel_scale.detach().to(dtype=torch.float32).clamp_min(eps)
    s = s / s.mean().clamp_min(eps)
    with torch.no_grad():
        weight.mul_(s.to(weight).unsqueeze(0))
    # inv_scale must track ``weight``'s device so the forward multiply and the
    # save-time bake never straddle cuda/cpu (``s`` is seeded CPU from calib).
    # fp32 storage is intentional — only the device moves.
    return (1.0 / s).to(weight.device).contiguous()


class BaseLoRAModule(torch.nn.Module):
    """Shared scaffolding: alpha→scale, multiplier, dropouts, channel_scale,
    timestep masking, ``apply_to`` monkey-patching. Subclasses own ``forward``."""

    supports_conv2d: bool = False

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d" and not self.supports_conv2d:
            raise ValueError(f"{type(self).__name__} does not support Conv2d")

        self.lora_dim = lora_dim
        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        # Runtime knob set by LoRANetwork after construction. Keeps constructor
        # signatures stable across LoRA variants while letting V100/fp16 training
        # preserve LoRA rank-path signal in fp32.
        self.fp32_compute = False

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self._has_channel_scale = False
        # Default all-ones mask → identity multiply; every forward can apply
        # `lx * self._timestep_mask` unconditionally (no None-vs-Tensor guard
        # under torch.compile). T-LoRA rebinds via LoRANetwork.set_timestep_mask.
        self.register_buffer(
            "_timestep_mask",
            torch.ones(1, lora_dim, dtype=torch.float32),
            persistent=False,
        )
        self.enabled = True

    def _register_channel_scale(
        self,
        target_weight: torch.Tensor,
        channel_scale,
        *,
        linear_only: bool = True,
    ) -> None:
        if channel_scale is None:
            return
        if linear_only and target_weight.dim() != 2:
            raise ValueError(
                "channel_scale is only supported for Linear LoRA modules, "
                f"got weight with dim {target_weight.dim()}"
            )
        inv_scale = _absorb_channel_scale(target_weight, channel_scale)
        self.register_buffer("inv_scale", inv_scale, persistent=True)
        self._has_channel_scale = True

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _skip_module(self) -> bool:
        return (
            self.module_dropout is not None
            and self.training
            and random.random() < self.module_dropout
        )

    def _rebalance(self, x: torch.Tensor) -> torch.Tensor:
        # inv_scale stays fp32 in storage (calibration precision); cast at the
        # multiply site so ``bf16 × fp32 → bf16`` instead of promoting to fp32.
        if not self._has_channel_scale:
            return x
        return x * self.inv_scale.to(device=x.device, dtype=x.dtype)

    def _apply_rank_dropout(self, lx: torch.Tensor):
        if self.rank_dropout is not None and self.training:
            mask = (
                torch.rand((lx.size(0), self.lora_dim), device=lx.device)
                > self.rank_dropout
            )
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            lx = lx * mask
            return lx, self.scale * (1.0 / (1.0 - self.rank_dropout))
        return lx, self.scale

    def _rank_compute_dtype(
        self, org_forwarded: torch.Tensor, default_dtype: torch.dtype | None = None
    ) -> torch.dtype:
        """Return the LoRA rank-path compute dtype for the current forward.

        Default remains the frozen base's compute dtype. The opt-in fp32 path is
        intentionally narrow: training only, and only when the base produced fp16.
        BF16/Ampere users keep the long-tested bf16 LoRA behavior unchanged.
        """
        base_dtype = default_dtype or org_forwarded.dtype
        if self.training and self.fp32_compute and org_forwarded.dtype == torch.float16:
            return torch.float32
        return base_dtype

    def _rank_autocast_context(self, x: torch.Tensor, work: torch.dtype):
        if work == torch.float32 and self.training and self.fp32_compute:
            return torch.autocast(device_type=x.device.type, enabled=False)
        return contextlib.nullcontext()

    # Forward scaffold (template method). The invariant chain — enable/fuse
    # short-circuit, eval delegation, module dropout, dtype policy, T-LoRA
    # gate, dropout / rank-dropout, residual add — lives here ONCE; the
    # standard two-GEMM variants (LoRA, OrthoInit, StepExpert) supply only the
    # down / gate / up rank computation via the hooks below.
    #
    # Variants whose forward genuinely differs — the frozen-basis Cayley
    # modules (OrthoLoRA / OrthoHydra: a single batched Cayley solve shared
    # between down and up, ``work`` = basis dtype) and the router-gated MoE
    # modules (Hydra / StackedExperts / Chimera: the "gate" is a routing
    # tensor consumed inside the up-projection, not an elementwise ``lx``
    # multiply) — keep their own ``forward`` and do NOT call this scaffold.

    def forward(self, x):
        if not self.enabled or getattr(self, "_fused", False):
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if not self.training:
            # Inference runs full rank at every t (T-LoRA is training-only) —
            # each variant's eval path is its own simplest form.
            return org_forwarded + self._eval_delta(x, org_forwarded)

        if self._skip_module():
            return org_forwarded

        # THE dtype policy, stated once. By default rank GEMMs run in the model
        # COMPUTE dtype (``org_forwarded.dtype`` — what the frozen base just
        # produced = the autocast/model dtype), NOT the activation dtype: ``x``
        # arrives fp32 from the AdaLN ``nn.LayerNorm`` under autocast(bf16).
        # Keying off ``x`` left the rank path fp32 and made ``_rebalance``
        # allocate a full fp32 activation (``x * inv_scale``) — the OOM the
        # per-channel-scaling path hit — for zero numeric gain on bf16 (autocast
        # re-casts the GEMM to bf16 regardless). V100/fp16 is the exception: the
        # optional fp32_compute flag disables autocast around the tiny rank path
        # so LoRA updates do not lose signal to fp16 underflow/rounding.
        work = self._rank_compute_dtype(org_forwarded)
        with self._rank_autocast_context(x, work):
            x_lora = self._rebalance(x.to(work))
            lx = self._down(x_lora, work)
            lx = self._gate(lx, work)
            if self.dropout is not None:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)
            lx, scale = self._apply_rank_dropout(lx)
            lx = self._up(lx.to(work), work)
        return org_forwarded + (lx * self.multiplier * scale).to(org_forwarded.dtype)

    def _gate(self, lx: torch.Tensor, work: torch.dtype) -> torch.Tensor:
        """Default T-LoRA gate: ``lx * mask``. The fp32 mask promotes ``lx``;
        the ``_up`` hook casts back to ``work``. Inherited unchanged by every
        scaffold variant unless it gates the rank latent differently (e.g.
        OrthoInit folds in its ``lambda_layer``)."""
        return lx * self._timestep_mask

    def _down(self, x_lora: torch.Tensor, work: torch.dtype) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} uses the forward scaffold but does not "
            "implement _down"
        )

    def _up(self, lx: torch.Tensor, work: torch.dtype) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} uses the forward scaffold but does not "
            "implement _up"
        )

    def _eval_delta(self, x: torch.Tensor, org_forwarded: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} uses the forward scaffold but does not "
            "implement _eval_delta"
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
