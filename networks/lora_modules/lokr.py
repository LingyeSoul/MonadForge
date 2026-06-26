# LoKR (Low-Rank Kronecker product) — ΔW = kron(w1, w2) * scale.
# Each factor can be full or further low-rank decomposed (w1 = w1a @ w1b).
# Based on LyCORIS LokrModule (KohakuBlueleaf/LyCORIS) adapted to
# BaseLoRAModule scaffold. Linear-only (Conv2d deferred).

import math
from typing import Dict

import torch
import torch.nn.functional as F

from networks.lora_modules.base import BaseLoRAModule


def _factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """Split *dimension* into ``(m, n)`` where ``m * n == dimension`` and ``m <= n``.

    If *factor* > 0 and divides *dimension*, use it directly. Otherwise find
    the divisor pair whose sum is smallest (closest to square). ``factor == -1``
    searches all divisors.
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


class LoKRModule(BaseLoRAModule):
    """LoKR adapter: ΔW = kron(w1, w2) * scale.

    Each factor is either a full parameter or a low-rank pair
    (``w1a @ w1b`` / ``w2a @ w2b``). ``decompose_both`` controls whether
    both factors are low-rank; ``factor`` controls the dimension split.

    Supports Linear only (``supports_conv2d = False``).
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
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
        )

        out_dim = org_module.out_features
        in_dim = org_module.in_features

        out_a, out_b = _factorization(out_dim, lokr_factor)
        in_a, in_b = _factorization(in_dim, lokr_factor)
        # shape: ΔW = (out_a*out_b, in_a*in_b) via kron((out_a, in_a), (out_b, in_b))

        # --- Factor 1 (smaller): shape (out_a, in_a) ---
        if decompose_both and lora_dim < max(out_a, in_a) / 2:
            self.w1a = torch.nn.Parameter(torch.empty(out_a, lora_dim))
            self.w1b = torch.nn.Parameter(torch.empty(lora_dim, in_a))
            self._use_w1 = False
        else:
            self.lokr_w1 = torch.nn.Parameter(torch.empty(out_a, in_a))
            self._use_w1 = True

        # --- Factor 2 (larger): shape (out_b, in_b) ---
        if lora_dim < max(out_b, in_b) / 2:
            self.w2a = torch.nn.Parameter(torch.empty(out_b, lora_dim))
            self.w2b = torch.nn.Parameter(torch.empty(lora_dim, in_b))
            self._use_w2 = False
        else:
            self.lokr_w2 = torch.nn.Parameter(torch.empty(out_b, in_b))
            self._use_w2 = True

        # --- Init (matches LyCORIS LokrModule, use_scalar=False) ---
        # Zero-start comes entirely from nulling w2: ΔW = kron(w1, 0) = 0.
        # w1 stays kaiming on all paths (full: lokr_w1; decomposed: w1a AND
        # w1b both kaiming) so its gradient flows from step 0. w2 is the null
        # branch — full lokr_w2 and the decomposed chain-end w2b are zeroed;
        # only w2a is kaiming, so the first update only touches w2a's partner
        # (w2b) and the factor unblocks on step 2.
        if self._use_w1:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.w1a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.w1b, a=math.sqrt(5))

        if self._use_w2:
            torch.nn.init.zeros_(self.lokr_w2)
        else:
            torch.nn.init.kaiming_uniform_(self.w2a, a=math.sqrt(5))
            torch.nn.init.zeros_(self.w2b)

        # LoKR cannot absorb channel_scale into w1: ΔW = kron(w1, w2) where
        # w1's in-axis is ``in_a`` (a factor of ``in_features``), so the
        # full-length ``channel_scale`` (length ``in_features``) has no clean
        # Kron decomposition. Standard LoRA absorbs it into ``lora_down`` whose
        # columns equal ``in_features``; LoKR's Kronecker structure breaks that.
        # Register ``inv_scale`` for forward-time input rebalancing; full
        # materialized deltas (get_weight/merge/fuse) apply the equivalent input
        # column scaling after kron reconstruction, not in the factors.
        if channel_scale is not None:
            self._register_lokr_inv_scale(channel_scale, in_dim)

        self.org_module_ref = [org_module]
        self._fused = False

    def _register_lokr_inv_scale(
        self, channel_scale: torch.Tensor, in_features: int, eps: float = 1e-12
    ) -> None:
        """Register ``inv_scale`` for LoKR without factor absorption.

        Mirrors the mean-normalize step of ``_absorb_channel_scale`` so the
        saved/resumed ``inv_scale`` matches the calibration convention, but
        skips the in-place ``W[:,c] *= s[c]`` step (impossible under the Kron
        factorization). Forward ``_rebalance(x)`` applies ``x * inv_scale``;
        full materialized deltas apply the same effect as input-column scaling.
        """
        assert channel_scale.ndim == 1, (
            f"channel_scale must be 1D, got shape {tuple(channel_scale.shape)}"
        )
        assert channel_scale.shape[0] == in_features, (
            f"channel_scale length {channel_scale.shape[0]} does not match "
            f"LoKR in_features {in_features}"
        )
        s = channel_scale.detach().to(dtype=torch.float32).clamp_min(eps)
        s = s / s.mean().clamp_min(eps)
        # fp32 CPU storage — device follows the module on load (matches
        # ``_absorb_channel_scale``'s convention).
        inv_scale = (1.0 / s).contiguous()
        self.register_buffer("inv_scale", inv_scale, persistent=True)
        self._has_channel_scale = True

    # --- Factor reconstruction helpers ---

    def _get_w1(self) -> torch.Tensor:
        if self._use_w1:
            return self.lokr_w1
        return self.w1a @ self.w1b

    def _get_w2(self) -> torch.Tensor:
        if self._use_w2:
            return self.lokr_w2
        return self.w2a @ self.w2b

    def _apply_inv_scale_to_full_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Apply input-column scaling to a materialized full LoKR delta."""
        if not self._has_channel_scale:
            return delta
        return delta * self.inv_scale.to(delta).unsqueeze(0)

    def get_weight(self, multiplier=None) -> torch.Tensor:
        """Return the LoKR delta as a full weight tensor (out, in)."""
        if multiplier is None:
            multiplier = self.multiplier
        w1 = self._get_w1().to(torch.float)
        w2 = self._get_w2().to(torch.float)
        # LoKR cannot absorb channel_scale into the factors, but a materialized
        # full delta can still represent the forward-time ``x * inv_scale`` as
        # input-column scaling: x @ (delta * inv_scale[None, :]).T.
        delta = self._apply_inv_scale_to_full_delta(torch.kron(w1, w2))
        return delta * self.scale * multiplier

    # --- Forward: override the base scaffold (LoKR doesn't decompose as
    # down→gate→up; it computes a full delta-weight and applies it). ---

    def forward(self, x):
        if not self.enabled or getattr(self, "_fused", False):
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if not self.training:
            return org_forwarded + self._eval_delta(x, org_forwarded)

        if self._skip_module():
            return org_forwarded

        work = self._rank_compute_dtype(org_forwarded)
        with self._rank_autocast_context(x, work):
            x_r = self._rebalance(x.to(work))

            w1 = self._get_w1().to(work)
            w2 = self._get_w2().to(work)

            out_a = w1.shape[0]
            in_a = w1.shape[1]
            out_b = w2.shape[0]
            in_b = w2.shape[1]

            # Efficient Kronecker-structured forward (no full kron materialization):
            # for row-major x reshaped to X (..., in_a, in_b),
            #   (x @ kron(w1, w2).T).reshape(..., out_a, out_b)
            #     == w1 @ X @ w2.T
            # where the flattened output order is (out_a, out_b), matching
            # torch.kron(w1, w2)'s row layout.
            x_mat = x_r.reshape(*x_r.shape[:-1], in_a, in_b)
            delta = torch.einsum(
                "oi,...ij,bj->...ob", w1.to(x_mat), x_mat, w2.to(x_mat)
            )
            lx = delta.reshape(*x_r.shape[:-1], out_a * out_b)

            if self.dropout is not None:
                lx = F.dropout(lx, p=self.dropout)

        return org_forwarded + (lx * self.multiplier * self.scale).to(org_forwarded.dtype)

    def _eval_delta(self, x, org_forwarded):
        x_r = self._rebalance(x)
        w1 = self._get_w1().to(x_r)
        w2 = self._get_w2().to(x_r)
        out_a, in_a = w1.shape
        out_b, in_b = w2.shape
        x_mat = x_r.reshape(*x_r.shape[:-1], in_a, in_b)
        delta = torch.einsum("oi,...ij,bj->...ob", w1, x_mat, w2)
        lx = delta.reshape(*x_r.shape[:-1], out_a * out_b)
        return lx * self.multiplier * self.scale

    # --- Merge / Fuse (same pattern as LoRAModule) ---

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w = weight.data.float()
            delta = self._reconstruct_delta_from_sd(sd, device)
            w += self.multiplier * delta * self.scale
            weight.data.copy_(w.to(dtype))

    def _reconstruct_delta_from_sd(self, sd, device) -> torch.Tensor:
        """Rebuild the full kron delta from a state-dict slice."""
        use_w1 = "lokr_w1" in sd
        use_w2 = "lokr_w2" in sd

        if use_w1:
            w1 = sd["lokr_w1"].to(torch.float).to(device)
        else:
            w1a = sd["w1a"].to(torch.float).to(device)
            w1b = sd["w1b"].to(torch.float).to(device)
            w1 = w1a @ w1b

        if use_w2:
            w2 = sd["lokr_w2"].to(torch.float).to(device)
        else:
            w2a = sd["w2a"].to(torch.float).to(device)
            w2b = sd["w2b"].to(torch.float).to(device)
            w2 = w2a @ w2b

        delta = torch.kron(w1, w2)
        if "inv_scale" in sd:
            # State-dict merge materializes the full delta, so express
            # forward-time ``x * inv_scale`` as input-column scaling here.
            inv_scale = sd["inv_scale"].to(torch.float).to(device)
            delta = delta * inv_scale.unsqueeze(0)
        return delta

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data -= delta
        self._fused = False

    # --- Save pipeline ---

    def distill_save_state_dict(self, prefix: str, state_dict: Dict[str, torch.Tensor]) -> None:
        """Write LoKR factors into *state_dict* under *prefix*."""
        if self._use_w1:
            state_dict[f"{prefix}.lokr_w1"] = self.lokr_w1.data.cpu()
        else:
            state_dict[f"{prefix}.w1a"] = self.w1a.data.cpu()
            state_dict[f"{prefix}.w1b"] = self.w1b.data.cpu()

        if self._use_w2:
            state_dict[f"{prefix}.lokr_w2"] = self.lokr_w2.data.cpu()
        else:
            state_dict[f"{prefix}.w2a"] = self.w2a.data.cpu()
            state_dict[f"{prefix}.w2b"] = self.w2b.data.cpu()

        state_dict[f"{prefix}.alpha"] = self.alpha.cpu()

        if self._has_channel_scale:
            state_dict[f"{prefix}.inv_scale"] = self.inv_scale.cpu()
