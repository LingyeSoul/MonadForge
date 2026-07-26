# GLoKr: Kronecker-factored delta (LoKr layout) + BoRA bi-dimensional
# weight decomposition (arXiv 2412.06441).
#
# The Kronecker split mirrors LyCORIS LoKr exactly (same ``factorization``,
# same init, same alpha/scale conventions), but the module is native — no
# LyCORIS wrapper — because BoRA turns the forward into a *weight replacement*
# (the whole merged matrix is re-normalized), which the additive bypass path
# cannot express:
#
#     ΔW  = kron(W1, W2) · scale                      (LoKr delta)
#     V^r = (W0 + ΔW) / ‖(W0 + ΔW)‖_row               (row-normalize)
#     H   = m_row ⊙ V^r                               (trainable row magnitudes)
#     W'  = m_col ⊙ H / ‖H‖_col                       (col-normalize + col magnitudes)
#     y   = x @ (W0 + multiplier·(W' − W0))^T + b
#
# ``m_row``/``m_col`` init to W0's row/col norms, and ΔW = 0 at init (zero-init
# w2 leg), so W' == W0 exactly at step 0. Norms are detached from the autograd
# graph (official DoRA practice — magnitudes and factors still receive
# gradients through the numerators). The multiplier lerps toward the base
# weight (LyCORIS weight-decompose convention) because W' is non-linear in ΔW.

import math
from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from lycoris.functional.general import factorization

from networks.lora_modules.base import BaseLoRAModule

_NORM_EPS = 1e-8


class GLoKRModule(BaseLoRAModule):
    supports_conv2d = False

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
        channel_scale=None,
        glokr_factor: int = -1,
        decompose_both: bool = False,
        full_factor: bool = False,
        rs_lora: bool = False,
        bora: bool = True,
    ) -> None:
        if channel_scale is not None:
            raise ValueError(
                "GLoKr does not support MonadForge channel scaling — the "
                "Kronecker factor's input axis is a *factor* of in_features, so "
                "a full-length channel scale cannot be absorbed. Set "
                "channel_scaling_alpha=0."
            )
        if dropout:
            raise ValueError(
                "GLoKr does not implement elementwise dropout on the merged "
                "weight path; set dropout=0 (module_dropout is supported)."
            )
        if rank_dropout:
            raise ValueError(
                "GLoKr does not implement rank_dropout (the Kronecker rank axis "
                "is re-normalized by BoRA, so LyCORIS' row-drop semantics do "
                "not transfer); set rank_dropout=0."
            )

        # Full-full layout forces unit scale (LyCORIS convention) — do it
        # BEFORE super() so the persisted ``alpha`` buffer matches the layout.
        in_dim = org_module.in_features
        out_dim = org_module.out_features
        factor = int(glokr_factor)
        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)

        self.use_w1 = not (
            decompose_both and lora_dim < max(out_l, in_m) / 2 and not full_factor
        )
        self.use_w2 = full_factor or not (lora_dim < max(out_k, in_n) / 2)
        if self.use_w1 and self.use_w2:
            alpha = lora_dim  # both factors full → scale must be 1

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

        self.glokr_factor = factor
        self.rs_lora = bool(rs_lora)
        self.bora = bool(bora)
        if self.rs_lora and not (self.use_w1 and self.use_w2):
            # LyCORIS rs convention (lycoris/modules/lokr.py): live scale is
            # alpha/sqrt(r) while the persisted alpha buffer stores
            # alpha·(r/r_factor) = alpha·sqrt(r), so every alpha/rank consumer
            # (merge, metadata-stripped rebuilds) recovers the same scale from
            # tensors alone — the checkpoint is self-describing and the
            # ss_glokr_rs_lora stamp is provenance only.
            self.scale = float(alpha) / math.sqrt(lora_dim)
            self.alpha = torch.tensor(self.scale * lora_dim)

        # Init mirrors LyCORIS 3.4.0 LokrModule (use_scalar=False path):
        # kaiming_uniform(a=√5) on every non-zero factor, zero on the w2 chain
        # end so ΔW = 0 at step 0.
        if self.use_w1:
            self.glokr_w1 = torch.nn.Parameter(torch.empty(out_l, in_m))
            torch.nn.init.kaiming_uniform_(self.glokr_w1, a=math.sqrt(5))
        else:
            self.glokr_w1_a = torch.nn.Parameter(torch.empty(out_l, lora_dim))
            self.glokr_w1_b = torch.nn.Parameter(torch.empty(lora_dim, in_m))
            torch.nn.init.kaiming_uniform_(self.glokr_w1_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.glokr_w1_b, a=math.sqrt(5))

        if self.use_w2:
            self.glokr_w2 = torch.nn.Parameter(torch.empty(out_k, in_n))
            torch.nn.init.zeros_(self.glokr_w2)
        else:
            self.glokr_w2_a = torch.nn.Parameter(torch.empty(out_k, lora_dim))
            self.glokr_w2_b = torch.nn.Parameter(torch.empty(lora_dim, in_n))
            torch.nn.init.kaiming_uniform_(self.glokr_w2_a, a=math.sqrt(5))
            torch.nn.init.zeros_(self.glokr_w2_b)

        if self.bora:
            w0 = org_module.weight.detach().float()
            self.bora_m_row = torch.nn.Parameter(
                w0.norm(dim=1, keepdim=True).clamp_min(_NORM_EPS)
            )
            self.bora_m_col = torch.nn.Parameter(
                w0.norm(dim=0, keepdim=True).clamp_min(_NORM_EPS)
            )

        self.org_module_ref = [org_module]
        # BaseLoRAModule registered org_module as a submodule; drop it now so
        # the frozen base weight never leaks into state_dict() (merge/save
        # paths read state dicts BEFORE apply_to would have deleted it).
        # apply_to below goes through org_module_ref instead.
        del self.org_module
        self._fused = False
        self._fuse_delta: Optional[torch.Tensor] = None

    def apply_to(self):
        module = self.org_module_ref[0]
        self.org_forward = module.forward
        module.forward = self.forward

    # ------------------------------------------------------------------ math

    def _delta_weight(self, gate_rank: bool) -> torch.Tensor:
        """kron(W1, W2) · scale in fp32. ``gate_rank`` applies the T-LoRA mask
        to w2's rank axis (training only; identity when the mask is all-ones;
        no rank axis exists on full factors)."""
        w1 = self.glokr_w1 if self.use_w1 else self.glokr_w1_a @ self.glokr_w1_b
        if self.use_w2:
            w2 = self.glokr_w2
        else:
            w2a = self.glokr_w2_a
            if gate_rank:
                w2a = w2a * self._timestep_mask.to(w2a.dtype)
            w2 = w2a @ self.glokr_w2_b
        return torch.kron(w1.float(), w2.float()) * self.scale

    def _bora_compose(self, w: torch.Tensor) -> torch.Tensor:
        """BoRA two-step normalization: row-normalize → m_row → col-normalize
        → m_col. Norms detached (DoRA memory convention)."""
        row_norm = w.norm(dim=1, keepdim=True).clamp_min(_NORM_EPS).detach()
        h = self.bora_m_row.float() * (w / row_norm)
        col_norm = h.norm(dim=0, keepdim=True).clamp_min(_NORM_EPS).detach()
        return self.bora_m_col.float() * (h / col_norm)

    def _effective_weight(self, w0: torch.Tensor, gate_rank: bool) -> torch.Tensor:
        """W0 + multiplier·(W' − W0) in fp32 (W' = merged/decomposed weight)."""
        w0f = w0.detach().float()
        merged = w0f + self._delta_weight(gate_rank)
        if self.bora:
            merged = self._bora_compose(merged)
        if self.multiplier != 1.0:
            merged = w0f + self.multiplier * (merged - w0f)
        return merged

    # --------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if not self.enabled or self._fused:
            return self.org_forward(x, *args, **kwargs)
        if self._skip_module():
            return self.org_forward(x, *args, **kwargs)
        base = self.org_module_ref[0]
        w_eff = self._effective_weight(base.weight, gate_rank=self.training)
        return F.linear(x, w_eff.to(base.weight.dtype), base.bias)

    # --------------------------------------------------- merge / fuse / save

    def get_weight(self, multiplier: Optional[float] = None) -> torch.Tensor:
        """Effective delta ``W_eff − W0`` (fp32, org weight shape) — the
        additive form every merge/fuse consumer expects. Non-linear in the
        multiplier when BoRA is on (lerp toward W0, LyCORIS wd convention)."""
        if multiplier is None:
            multiplier = self.multiplier
        w0f = self.org_module_ref[0].weight.detach().float()
        merged = w0f + self._delta_weight(gate_rank=False)
        if self.bora:
            merged = self._bora_compose(merged)
        return float(multiplier) * (merged - w0f)

    def _reconstruct_from_sd(
        self, sd: Mapping[str, torch.Tensor], device
    ) -> torch.Tensor:
        """Rebuild the checkpoint's ΔW (pre-BoRA, fp32) from a per-module
        state-dict slice; magnitudes are read separately by the caller."""

        def get(name: str) -> Optional[torch.Tensor]:
            v = sd.get(name)
            return None if v is None else v.to(device=device, dtype=torch.float32)

        rank = None
        w1 = get("glokr_w1")
        if w1 is None:
            w1a, w1b = get("glokr_w1_a"), get("glokr_w1_b")
            if w1a is None or w1b is None:
                raise KeyError(
                    f"{self.lora_name}: incomplete glokr w1 factor in state dict"
                )
            rank = w1a.size(1)
            w1 = w1a @ w1b
        w2 = get("glokr_w2")
        if w2 is None:
            w2a, w2b = get("glokr_w2_a"), get("glokr_w2_b")
            if w2a is None or w2b is None:
                raise KeyError(
                    f"{self.lora_name}: incomplete glokr w2 factor in state dict"
                )
            rank = w2a.size(1)
            w2 = w2a @ w2b

        alpha = sd.get("alpha", self.alpha)
        alpha = float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
        # Always alpha/rank: the rs_lora convention pre-folds sqrt(r) into the
        # persisted alpha (see __init__), so no external flag is needed here.
        scale = 1.0 if rank is None else alpha / rank  # full-full → unit scale
        return torch.kron(w1, w2) * scale

    def merge_to(self, sd: Dict[str, torch.Tensor], dtype, device) -> None:
        """Bake a checkpoint slice into the base weight (replacement + lerp)."""
        with torch.no_grad():
            weight = self.org_module_ref[0].weight
            target_device = weight.device if device is None else device
            target_dtype = weight.dtype if dtype is None else dtype
            w0f = weight.data.float().to(target_device)
            merged = w0f + self._reconstruct_from_sd(sd, target_device)
            m_row = sd.get("bora_m_row")
            m_col = sd.get("bora_m_col")
            if m_row is not None and m_col is not None:
                row_norm = merged.norm(dim=1, keepdim=True).clamp_min(_NORM_EPS)
                h = m_row.to(merged) * (merged / row_norm)
                col_norm = h.norm(dim=0, keepdim=True).clamp_min(_NORM_EPS)
                merged = m_col.to(merged) * (h / col_norm)
            elif self.bora:
                raise KeyError(
                    f"{self.lora_name}: module built with bora=True but the "
                    "state dict has no bora_m_row/bora_m_col keys"
                )
            if self.multiplier != 1.0:
                merged = w0f + self.multiplier * (merged - w0f)
            weight.data.copy_(merged.to(device=weight.device, dtype=target_dtype))

    def fuse_weight(self) -> None:
        """Bake the live delta into the base weight. Keeps the delta stashed
        (one org-shaped tensor) so ``unfuse_weight`` is exact — the BoRA
        normalization is not invertible from the fused weight alone."""
        if self._fused:
            return
        module = self.org_module_ref[0]
        delta = self.get_weight().to(module.weight.dtype)
        module.weight.data += delta
        self._fuse_delta = delta
        self._fused = True

    def unfuse_weight(self) -> None:
        if not self._fused:
            return
        module = self.org_module_ref[0]
        module.weight.data -= self._fuse_delta.to(
            device=module.weight.device, dtype=module.weight.dtype
        )
        self._fuse_delta = None
        self._fused = False
