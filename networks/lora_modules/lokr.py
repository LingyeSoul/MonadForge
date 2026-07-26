"""LyCORIS LoKr backend with MonadForge lifecycle adapters.

The factorization, initialization, bypass forward, dropout, and canonical
state-dict layout come directly from ``lycoris-lora``. This wrapper only
bridges the small lifecycle surface used by :class:`LoRANetwork`.
"""

from __future__ import annotations

from typing import Mapping

import torch
from lycoris.functional.lokr import (
    bypass_forward_diff as lycoris_lokr_bypass_forward_diff,
)
from lycoris.functional.lokr import diff_weight as lycoris_lokr_diff_weight
from lycoris.modules.lokr import LokrModule as LycorisLokrModule

from networks.lora_modules.custom_autograd import eager_lokr_residual


class LoKRModule(LycorisLokrModule):
    """Official LyCORIS LoKr with the MonadForge adapter protocol.

    ``full_factor`` is the historical MonadForge config name for LyCORIS'
    ``full_matrix`` option. Linear LoKr always uses LyCORIS bypass mode so the
    Kronecker delta is evaluated as a sequence of small linear operations
    rather than materializing a full DiT-sized matrix on every forward.
    """

    supports_conv2d = False

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        channel_scale=None,
        decompose_both: bool = False,
        lokr_factor: int = -1,
        full_factor: bool = False,
    ) -> None:
        if channel_scale is not None:
            raise ValueError(
                "LyCORIS LoKr does not support MonadForge channel scaling; "
                "channel_scaling_alpha is disabled for LoKr networks"
            )

        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=float(dropout or 0.0),
            rank_dropout=float(rank_dropout or 0.0),
            module_dropout=float(module_dropout or 0.0),
            decompose_both=decompose_both,
            factor=lokr_factor,
            full_matrix=full_factor,
            bypass_mode=True,
        )

        self.org_module_ref = self.org_module
        self.enabled = True
        self._fused = False
        self.fp32_compute = False
        self.use_custom_down_autograd = False

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if not self.enabled or self._fused:
            return self.org_forward(x, *args, **kwargs)
        if (
            self.module_dropout
            and self.training
            and torch.rand(1) < self.module_dropout
        ):
            return self.org_forward(x, *args, **kwargs)

        # LyCORIS 3.4.0's bypass_forward passes only ``multiplier`` and omits
        # ``self.scale``. Its regular forward and get_weight both include the
        # scale. Keep the memory-efficient official bypass operations while
        # restoring that required alpha/rank factor.
        base = self.org_forward(x, *args, **kwargs)
        if self.training and self.fp32_compute and base.dtype == torch.float16:
            with torch.autocast(device_type=x.device.type, enabled=False):
                if (
                    self.use_custom_down_autograd
                    and not self.dropout
                    and not torch.compiler.is_compiling()
                    and torch.is_grad_enabled()
                ):
                    return self._eager_fp32_bypass_residual(base, x)
                delta = self._fp32_bypass_forward_diff(x)
        else:
            delta = self.bypass_forward_diff(x, scale=self.multiplier * self.scale)
        return base + delta.to(base.dtype)

    def _eager_fp32_bypass_residual(
        self,
        base: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Merge a chunked FP32 LoKr bypass into the fresh base output."""
        return eager_lokr_residual(
            base,
            x,
            self.lokr_w1 if self.use_w1 else None,
            None if self.use_w1 else self.lokr_w1_a,
            None if self.use_w1 else self.lokr_w1_b,
            self.lokr_w2 if self.use_w2 else None,
            None if self.use_w2 else self.lokr_w2_a,
            None if self.use_w2 else self.lokr_w2_b,
            self.scalar,
            self.multiplier * self.scale,
        )

    def _fp32_bypass_forward_diff(self, x: torch.Tensor) -> torch.Tensor:
        """Run the official linear LoKr bypass with fp32 rank operands."""

        def fp32(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(dtype=torch.float32)

        w1 = fp32(self.lokr_w1 if self.use_w1 else None)
        w1a = fp32(None if self.use_w1 else self.lokr_w1_a)
        w1b = fp32(None if self.use_w1 else self.lokr_w1_b)
        w2 = fp32(self.lokr_w2 if self.use_w2 else None)
        w2a = fp32(None if self.use_w2 else self.lokr_w2_a)
        w2b = fp32(None if self.use_w2 else self.lokr_w2_b)
        rank = (
            self.lokr_w1_a.shape[1]
            if not self.use_w1
            else self.lokr_w2_a.shape[1]
            if not self.use_w2
            else 1
        )
        gamma = float(self.scale) * rank
        delta = lycoris_lokr_bypass_forward_diff(
            x.to(dtype=torch.float32),
            None,
            w1,
            w1a,
            w1b,
            w2,
            w2a,
            w2b,
            None,
            gamma=gamma,
        )
        return self.drop(delta * self.multiplier * self.scalar.float())

    def get_weight(self, multiplier=None, shape=None) -> torch.Tensor:
        """Return the official LyCORIS delta with a local multiplier.

        LyCORIS internally calls ``get_weight(shape)`` from regular rebuild
        mode. LoKr is pinned to bypass mode here, but accepting that call shape
        keeps inherited utilities such as ``apply_max_norm`` correct.
        """
        if isinstance(multiplier, (tuple, torch.Size)):
            shape = multiplier
            multiplier = 1.0
        elif multiplier is None:
            multiplier = self.multiplier
        target_shape = self.shape if shape is None else shape
        was_training = self.training
        if was_training and self.rank_dropout:
            self.train(False)
        try:
            weight = LycorisLokrModule.get_weight(self, target_shape)
        finally:
            if was_training and self.rank_dropout:
                self.train(True)
        return weight.float() * float(multiplier)

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        target_shape = self.shape if shape is None else shape
        diff = self.get_weight(multiplier=multiplier, shape=target_shape)
        if device is not None:
            diff = diff.to(device)
        return diff, None

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        diff = self.get_diff_weight(multiplier, shape, device)[0]
        return self.org_weight.to(diff) + diff, None

    @staticmethod
    def _state_tensor(
        state_dict: Mapping[str, torch.Tensor],
        canonical: str,
        legacy: str | None = None,
    ) -> torch.Tensor | None:
        value = state_dict.get(canonical)
        if value is None and legacy is not None:
            value = state_dict.get(legacy)
        return value

    def _reconstruct_delta_from_sd(
        self, state_dict: Mapping[str, torch.Tensor], device
    ) -> torch.Tensor:
        """Rebuild a checkpoint delta through LyCORIS' functional API."""

        def prepare(value: torch.Tensor | None) -> torch.Tensor | None:
            return (
                None if value is None else value.to(device=device, dtype=torch.float32)
            )

        w1 = prepare(self._state_tensor(state_dict, "lokr_w1"))
        w1a = prepare(self._state_tensor(state_dict, "lokr_w1_a", "w1a"))
        w1b = prepare(self._state_tensor(state_dict, "lokr_w1_b", "w1b"))
        w2 = prepare(self._state_tensor(state_dict, "lokr_w2"))
        w2a = prepare(self._state_tensor(state_dict, "lokr_w2_a", "w2a"))
        w2b = prepare(self._state_tensor(state_dict, "lokr_w2_b", "w2b"))
        alpha = state_dict.get("alpha", self.alpha)
        gamma = float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
        return lycoris_lokr_diff_weight(w1, w1a, w1b, w2, w2a, w2b, None, gamma=gamma)

    def merge_to(self, state_dict, dtype=None, device=None):
        """Merge a per-module state-dict slice using official LoKr math."""
        with torch.no_grad():
            weight = self.org_module_ref[0].weight
            target_device = weight.device if device is None else device
            target_dtype = weight.dtype if dtype is None else dtype
            delta = self._reconstruct_delta_from_sd(state_dict, target_device)
            merged = weight.data.float() + delta * self.multiplier
            weight.data.copy_(merged.to(device=weight.device, dtype=target_dtype))

    def fuse_weight(self) -> None:
        if self._fused:
            return
        module = self.org_module_ref[0]
        module.weight.data.add_(self.get_weight().to(module.weight))
        self._fused = True

    def unfuse_weight(self) -> None:
        if not self._fused:
            return
        module = self.org_module_ref[0]
        module.weight.data.sub_(self.get_weight().to(module.weight))
        self._fused = False
