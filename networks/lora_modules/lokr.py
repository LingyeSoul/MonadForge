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

        self._register_channel_scale(
            self.w1a if not self._use_w1 else self.lokr_w1,
            channel_scale,
            linear_only=True,
        )

        self.org_module_ref = [org_module]
        self._fused = False

    # --- Factor reconstruction helpers ---

    def _get_w1(self) -> torch.Tensor:
        if self._use_w1:
            return self.lokr_w1
        return self.w1a @ self.w1b

    def _get_w2(self) -> torch.Tensor:
        if self._use_w2:
            return self.lokr_w2
        return self.w2a @ self.w2b

    def get_weight(self, multiplier=None) -> torch.Tensor:
        """Return the LoKR delta as a full weight tensor (out, in)."""
        if multiplier is None:
            multiplier = self.multiplier
        w1 = self._get_w1().to(torch.float)
        w2 = self._get_w2().to(torch.float)
        # Undo channel absorption so the merged delta applies to raw inputs.
        if self._has_channel_scale:
            w1 = w1 * self.inv_scale.to(w1).unsqueeze(0)
        delta = torch.kron(w1, w2) * self.scale * multiplier
        return delta

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
            # kron(A, B) @ x  ==  B @ X_mat @ A^T  (reshape x to (in_a, in_b))
            x_mat = x_r.reshape(*x_r.shape[:-1], in_a, in_b)
            # w2: (out_b, in_b)  x_mat: (..., in_a, in_b) → h: (..., in_a, out_b)
            h = torch.einsum("...ij,oj->...io", x_mat, w2.to(x_mat))
            # w1: (out_a, in_a)  h: (..., in_a, out_b) → out: (..., out_a, out_b)
            delta = torch.einsum("...ia,oa->...oa", h, w1.to(h))
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
        h = torch.einsum("...ij,oj->...io", x_mat, w2)
        delta = torch.einsum("...ia,oa->...oa", h, w1)
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

        if "inv_scale" in sd:
            inv_scale = sd["inv_scale"].to(torch.float).to(device)
            w1 = w1 * inv_scale.unsqueeze(0)

        return torch.kron(w1, w2)

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
