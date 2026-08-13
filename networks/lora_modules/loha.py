"""LyCORIS LoHa backend with MonadForge lifecycle adapters.

The Hadamard-product factorization, initialization, dropout semantics, and
canonical ``hada_*`` state-dict layout come directly from ``lycoris-lora``.
This wrapper only bridges the small lifecycle surface used by
:class:`LoRANetwork`.
"""

from __future__ import annotations

from typing import Mapping

import torch
from lycoris.functional.loha import diff_weight as lycoris_loha_diff_weight
from lycoris.modules.loha import LohaModule as LycorisLohaModule

from networks.lora_modules.base import merge_lora_residual, preserve_lora_output_dtype


class LoHaModule(LycorisLohaModule):
    """Official LyCORIS LoHa with the MonadForge adapter protocol.

    ΔW = (hada_w1_a @ hada_w1_b) ⊙ (hada_w2_a @ hada_w2_b) · (alpha/rank) —
    effective rank up to r² from two rank-r factor pairs. Linear LoHa runs in
    LyCORIS bypass mode; note the Hadamard product cannot factor through the
    input, so bypass still materializes ΔW per forward (it just skips the
    base-weight read/subtract of the rebuild path).

    Two LyCORIS 3.4.0 quirks this wrapper routes around:

    * Unlike LoKr, LoHa's official bypass already applies ``self.scale`` (it
      goes through ``get_weight``), so ``forward`` passes ``scale=multiplier``
      only — copying the LoKr wrapper's ``multiplier * self.scale`` fix here
      would double-scale.
    * ``get_diff_weight``/``get_merged_weight`` multiply ``self.scale`` a
      second time on top of ``get_weight`` — both are overridden.

    ``lycoris.functional.loha.bypass_forward_diff`` is broken as shipped
    (passes ``gamma`` positionally into a 6-target unpack) — never call it;
    the fp32 path uses ``diff_weight`` + ``self.op`` instead.
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
    ) -> None:
        if channel_scale is not None:
            raise ValueError(
                "LyCORIS LoHa does not support MonadForge channel scaling; "
                "channel_scaling_alpha is disabled for LoHa networks"
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
            bypass_mode=True,
        )

        self.org_module_ref = self.org_module
        self.enabled = True
        self._fused = False
        self.fp32_compute = False

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if not self.enabled or self._fused:
            return self.org_forward(x, *args, **kwargs)
        if (
            self.module_dropout
            and self.training
            and torch.rand(1) < self.module_dropout
        ):
            return preserve_lora_output_dtype(
                self.org_forward(x, *args, **kwargs),
                preserve_fp32=self.fp32_compute,
            )

        base = self.org_forward(x, *args, **kwargs)
        if self.training and self.fp32_compute and base.dtype == torch.float16:
            with torch.autocast(device_type=x.device.type, enabled=False):
                delta = self._fp32_bypass_forward_diff(x)
        else:
            # The official bypass bakes ``self.scale`` in via ``get_weight``
            # — multiplier only.
            delta = self.bypass_forward_diff(x, scale=self.multiplier)
        return merge_lora_residual(
            base,
            delta,
            preserve_fp32=self.training and self.fp32_compute,
        )

    def bypass_forward_diff(self, x, scale=1):
        # Byte-equal to the official implementation except for the
        # NON-virtual ``get_weight`` call: the wrapper's ``get_weight``
        # override is a merge/fuse surface (fp32 output, rank_dropout
        # suppressed) and must not leak into the training forward, where
        # LoHa's rank_dropout lives inside the official ``get_weight``.
        diff_weight = (
            LycorisLohaModule.get_weight(self, self.shape) * self.scalar * scale
        )
        return self.drop(self.op(x, diff_weight, **self.kw_dict))

    def _fp32_bypass_forward_diff(self, x: torch.Tensor) -> torch.Tensor:
        """Rebuild the official LoHa delta with fp32 rank operands.

        Mirrors ``bypass_forward_diff`` minus the training-mode rank_dropout
        (matching the LoKr wrapper's fp32 branch, which also bypasses it).
        ``functional.loha.diff_weight`` treats ``gamma`` as the FULL scale
        (alpha/rank) — unlike functional lokr, which divides by rank
        internally.
        """
        w1a = self.hada_w1_a.to(dtype=torch.float32)
        w1b = self.hada_w1_b.to(dtype=torch.float32)
        w2a = self.hada_w2_a.to(dtype=torch.float32)
        w2b = self.hada_w2_b.to(dtype=torch.float32)
        gamma = torch.tensor(float(self.scale), dtype=torch.float32, device=x.device)
        diff_w = lycoris_loha_diff_weight(w1b, w1a, w2b, w2a, None, None, gamma=gamma)
        delta = self.op(x.to(dtype=torch.float32), diff_w, **self.kw_dict)
        return self.drop(delta * self.multiplier * self.scalar.float())

    def get_weight(self, multiplier=None, shape=None) -> torch.Tensor:
        """Return the official LyCORIS delta with a local multiplier.

        LyCORIS internally calls ``get_weight(shape)`` (e.g. from
        ``apply_max_norm``); accepting that call shape keeps inherited
        utilities correct. Training-mode rank_dropout is disabled around the
        rebuild so merge/fuse paths stay deterministic.
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
            weight = LycorisLohaModule.get_weight(self, target_shape)
        finally:
            if was_training and self.rank_dropout:
                self.train(True)
        return weight.float() * float(multiplier)

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        # LyCORIS 3.4.0's implementation multiplies ``self.scale`` on top of a
        # ``get_weight`` that already applied it — this override scales once.
        target_shape = self.shape if shape is None else shape
        diff = self.get_weight(multiplier=multiplier, shape=target_shape)
        if device is not None:
            diff = diff.to(device)
        return diff, None

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        diff = self.get_diff_weight(multiplier, shape, device)[0]
        return self.org_weight.to(diff) + diff, None

    def _reconstruct_delta_from_sd(
        self, state_dict: Mapping[str, torch.Tensor], device
    ) -> torch.Tensor:
        """Rebuild a checkpoint delta through LyCORIS' functional API."""

        def prepare(value: torch.Tensor | None) -> torch.Tensor | None:
            return (
                None if value is None else value.to(device=device, dtype=torch.float32)
            )

        w1a = prepare(state_dict.get("hada_w1_a"))
        w1b = prepare(state_dict.get("hada_w1_b"))
        w2a = prepare(state_dict.get("hada_w2_a"))
        w2b = prepare(state_dict.get("hada_w2_b"))
        if w1a is None or w1b is None or w2a is None or w2b is None:
            raise KeyError(
                f"LoHa slice for {self.lora_name!r} is missing one of "
                "hada_w1_a/hada_w1_b/hada_w2_a/hada_w2_b"
            )
        t1 = prepare(state_dict.get("hada_t1"))
        t2 = prepare(state_dict.get("hada_t2"))
        alpha = state_dict.get("alpha", self.alpha)
        alpha_f = (
            float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
        )
        rank = int(w1b.size(0))
        # functional loha's gamma is the full alpha/rank scale, applied once.
        gamma = torch.tensor(alpha_f / rank, dtype=torch.float32, device=device)
        return lycoris_loha_diff_weight(w1b, w1a, w2b, w2a, t1, t2, gamma=gamma)

    def merge_to(self, state_dict, dtype=None, device=None):
        """Merge a per-module state-dict slice using official LoHa math."""
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
