# FeRA: Frequency-Energy Constrained Routing for diffusion adaptation.
#
# Faithful port of Yin et al., arXiv:2511.17979 (FeRA/ in repo root) adapted to
# Anima's pipeline. Distinct from the FEI-router-on-Hydra variant living in
# ``networks/lora_modules/hydra.py``:
#
#   * Each expert has its own independent ``(lora_down, lora_up)`` pair —
#     **no** shared-A pooling like Hydra. Matches the author's reference at
#     FeRA/fera/layer.py.
#   * A **single global router** consumes the per-batch FEI of ``z_t`` and
#     emits one ``(B, num_experts)`` gate that every adapted Linear reuses
#     for this step. Hydra routes per-Linear from its own input.
#   * Targets all matched Linears (default: attention proj + MLP). The
#     2-band FEI-on-Hydra variant is regex-restricted to MLP layers.
#
# Attachment + memory profile:
#   - Original ``nn.Linear`` modules stay in the DiT module tree. We
#     monkey-patch their ``forward`` (LoRA-family pattern from
#     ``networks/lora_modules/base.py``) rather than replacing the
#     instance. **Critical for block-swap**: the offloader walks
#     ``named_modules()`` to find weights; module replacement (with the
#     base hidden behind ``object.__setattr__``) silently pins the
#     entire frozen DiT base on GPU and OOMs at modest VRAM.
#   - Experts are stored as **stacked Parameters** per FeRALinear
#     (``lora_down: (E, r, in)``, ``lora_up: (E, out, r)``) and consumed
#     by two ``einsum`` calls per forward (down + up). Saves one
#     ``(..., E, r)`` activation for backward instead of E full
#     ``(..., D_out)`` — Hydra-style. Mathematically identical to the
#     author's ``Σ_k w_k · U_k @ D_k @ x``, but ~50× less per-Linear
#     activation memory at default ``E=3, r=8`` on Anima MLP shapes.
#
# σ_low scaling follows the rule
# ``σ_low = min(H_lat, W_lat) / fei_sigma_low_div`` (default ``4.0``,
# picked by the 2026-05-13 dataset sweep) rather than the paper's
# pixel-domain constant ``min(H, W)/128`` — the latter is dataset-specific
# (see ``project_fera_probe_2band_decision`` and ``library/runtime/fei.py``).
# ``num_bands`` defaults to 3 (paper) but can be set to 2 (Anima-validated).
#
# FECL (frequency-energy consistency loss) is exposed via
# ``compute_fecl_loss`` but **not** wired into ``train.py``'s loss composer
# yet — it needs a second no-grad DiT forward per step. Default
# ``fecl_weight = 0.0`` keeps it disabled; opt-in is a follow-up bench.

from __future__ import annotations

import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.log import setup_logging
from library.runtime.fei import gaussian_blur_2d
from library.training.metrics import MetricContext

setup_logging()
logger = logging.getLogger(__name__)


# Author's reference defaults: attn q/k/v + output. Anima uses fused
# ``qkv_proj`` for self-attn and ``q_proj``/``kv_proj`` for cross-attn, plus
# ``output_proj`` and the ``mlp.layer{1,2}`` Linears. Default regex below
# covers all of them — override via ``fera_target_modules`` in TOML.
_DEFAULT_TARGET_REGEX = (
    r".*\.(qkv_proj|q_proj|kv_proj|output_proj|layer[12])$"
)

# ComfyUI-compatible prefix (same as ``LoRANetwork.LORA_PREFIX_ANIMA``).
LORA_PREFIX_ANIMA = "lora_unet"


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────


class FrequencyEnergyIndicator(nn.Module):
    """Multi-band soft-simplex energy on a 4D latent.

    Adapts ``FeRA/fera/utils.py`` to Anima: bucket-adaptive σ_low
    (``min(H_lat, W_lat) / fei_sigma_low_div``) instead of the paper's
    pixel-domain ``min(H, W) / 128``. Bands are Laplacian-style
    differences over a Gaussian pyramid. Returns ``(B, num_bands)`` on the
    simplex.

    bf16-safe: promotes ``z`` to fp32 internally (DoG + squared norm can
    underflow at small energies).
    """

    def __init__(self, num_bands: int = 3, fei_sigma_low_div: float = 4.0):
        super().__init__()
        if num_bands < 2:
            raise ValueError(f"num_bands must be >= 2, got {num_bands}")
        self.num_bands = int(num_bands)
        self.fei_sigma_low_div = float(fei_sigma_low_div)

    def _band_sigmas(self, h_lat: int, w_lat: int) -> List[float]:
        """``num_bands`` σ scales doubling outward from σ_low.

        Author uses ``[2**k for k in range(num_bands)]`` scaled by ``κ``.
        We instead anchor σ_low to ``min(H_lat, W_lat) / fei_sigma_low_div``
        (default ``4.0`` from the 2026-05-13 dataset sweep) and double
        from there. Result: same ratio structure as paper, but
        bucket-invariant in latent coordinates.
        """
        sigma_low = float(min(h_lat, w_lat)) / self.fei_sigma_low_div
        return [sigma_low * (2.0**k) for k in range(self.num_bands - 1)]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, H, W) — caller squeezes any singleton T.
        z = z.float()
        h_lat, w_lat = int(z.shape[-2]), int(z.shape[-1])
        sigmas = self._band_sigmas(h_lat, w_lat)

        # Gaussian pyramid: z, LP(σ_low), LP(2σ_low), ...
        pyramid = [z]
        for s in sigmas:
            pyramid.append(gaussian_blur_2d(pyramid[-1], s))

        # Bands (high → low): differences of adjacent pyramid levels,
        # then the coarsest LP as the residual low-band.
        bands = [pyramid[k] - pyramid[k + 1] for k in range(self.num_bands - 1)]
        bands.append(pyramid[-1])

        energies = torch.stack(
            [b.pow(2).flatten(1).sum(-1) for b in bands], dim=-1
        )  # (B, num_bands), ordered [high ... low]
        return energies / energies.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class SoftFrequencyRouter(nn.Module):
    """Linear → ReLU → Linear → softmax/τ on FEI simplex.

    Mirrors ``FeRA/fera/model.py::SoftFrequencyRouter``. The final layer is
    zero-init so step-0 routing is uniform across experts — combined with
    zero-init ``lora_up`` this guarantees the FeRA contribution is exactly
    zero at the first optimizer step (clean residual baseline).
    """

    def __init__(
        self,
        num_bands: int,
        num_experts: int,
        hidden_dim: int = 64,
        tau: float = 0.7,
    ):
        super().__init__()
        self.tau = float(tau)
        self.net = nn.Sequential(
            nn.Linear(num_bands, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        # Uniform-at-init: zero the output layer so softmax(0/τ) = 1/E.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, e_t: torch.Tensor) -> torch.Tensor:
        # e_t: (B, num_bands) fp32. Logits in fp32; cast to caller dtype later.
        logits = self.net(e_t.float())
        return F.softmax(logits / self.tau, dim=-1)


class FeRALinear(nn.Module):
    """Adapter sidecar for one ``nn.Linear`` — stacked low-rank experts +
    per-step routing weights.

    Attachment mirrors the LoRA family (``networks/lora_modules/base.py``):
    we **monkey-patch the parent Linear's ``forward``** instead of replacing
    the module. This keeps the original Linear inside the DiT's module
    tree so block-swap (``library/runtime/offloading.py`` walks
    ``named_modules()``) and ``.to(device)`` see its weights normally.
    Replacing the module hides those weights from the offloader, which
    pins the entire frozen DiT base on GPU regardless of
    ``blocks_to_swap`` and OOMs at modest VRAM budgets.

    Forward uses a **single down + single up matmul** stacked over experts
    via ``einsum`` (Hydra-style), instead of looping over per-expert
    ``nn.Linear`` modules. Mathematically equivalent to the author's
    ``Σ_k w_k · U_k @ D_k @ x`` but only saves one ``(..., E, r)``
    activation for backward instead of ``E × (..., D_out)``. Cuts
    per-layer activation memory by ``D_out / (E · r)`` — ~50× at MLP
    layer1 for ``E=3, r=8`` on Anima.

    ``set_routing_weights(None)`` short-circuits to the frozen base —
    used by the FECL base-prediction pass.

    ``ortho=True`` swaps the free ``(lora_down, lora_up)`` Parameters for a
    PSOFT-style parameterization: frozen ``Q_basis (r, in)`` /
    ``P_basis (out, r)`` from the base Linear's top-r SVD, plus
    per-expert skew seeds ``S_q, S_p (E, r, r)`` Cayley-rotated to
    orthogonal matrices, plus per-expert diagonal scale ``λ (E, r)``.
    Each expert's effective ΔW is
    ``P_basis @ cayley(S_p_k) @ diag(λ_k) @ cayley(S_q_k) @ Q_basis``.
    Experts share the same singular-bundle (no disjoint slicing) so they
    all live in W's high-energy subspace — symmetry between experts is
    broken by small random init on ``S_p, S_q`` (Kaiming-analog). The
    independent-A invariant from the author paper is preserved at the
    *rotation* level: each expert owns its own ``(S_q_k, S_p_k, λ_k)``
    and is free to rotate within the shared frozen basis.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        num_experts: int,
        rank: int,
        alpha: float,
        lora_name: str,
        ortho: bool = False,
        ortho_init_std: float = 0.02,
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.num_experts = int(num_experts)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        self.lora_name = str(lora_name)
        self.ortho = bool(ortho)
        self.ortho_init_std = float(ortho_init_std)

        if self.ortho:
            # PSOFT-style parameterization with shared bases across experts.
            # See class docstring for the design rationale.
            init_device = "cuda" if torch.cuda.is_available() else "cpu"
            W = base_layer.weight.data.float().to(init_device)
            q = min(self.rank + 6, min(W.shape))
            U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)
            # U: (out, q), V: (in, q) — V is returned directly (not Vh).
            P_init = U[:, : self.rank].clone().contiguous()  # (out, r)
            Q_init = V[:, : self.rank].T.clone().contiguous()  # (r, in)
            del U, _S_vals, V, W

            # Frozen shared bases — define the subspace; per-expert Cayley
            # rotations move within it.
            self.register_buffer("P_basis", P_init.cpu())  # (out, r)
            self.register_buffer("Q_basis", Q_init.cpu())  # (r, in)

            # Per-expert trainable skew seeds. Random init (Kaiming-analog
            # symmetry breaking) — with deterministic SVD init and zero λ,
            # zero-init S would leave every expert bit-identical and the
            # global router would have no gradient signal to differentiate
            # them. ``ortho_init_std`` controls how far each expert starts
            # from identity rotation.
            self.S_p = nn.Parameter(
                torch.randn(self.num_experts, self.rank, self.rank)
                * self.ortho_init_std
            )
            self.S_q = nn.Parameter(
                torch.randn(self.num_experts, self.rank, self.rank)
                * self.ortho_init_std
            )

            # Per-expert diagonal scale — zero-init → ΔW = 0 at init even
            # though S is non-zero. Same convention as standard LoRA's
            # zero-init ``lora_up``.
            self.lambda_layer = nn.Parameter(
                torch.zeros(self.num_experts, self.rank)
            )

            # Pre-allocated identity for batched Cayley solves; allocating
            # ``torch.eye`` per forward emits 2 tiny kernels per module per
            # step under compile.
            self.register_buffer(
                "_eye_r",
                torch.eye(self.rank, dtype=torch.float32),
                persistent=False,
            )
        else:
            # Stacked expert weights — single-matmul-friendly layout.
            #   lora_down: (E, r, in)  — Kaiming on each (r, in) slice
            #   lora_up:   (E, out, r) — zero-init (matches author LoRAExpert)
            # Each Parameter has a flat 3D shape; we keep them as the trainable
            # surface so ``state_dict`` and the optimizer see ``lora_down`` /
            # ``lora_up`` directly without an inner ModuleList.
            self.lora_down = nn.Parameter(
                torch.empty(self.num_experts, self.rank, self.in_features)
            )
            self.lora_up = nn.Parameter(
                torch.zeros(self.num_experts, self.out_features, self.rank)
            )
            for k in range(self.num_experts):
                nn.init.kaiming_uniform_(self.lora_down[k], a=math.sqrt(5))

        # Transient reference to the parent Linear — held only until
        # ``apply_to()`` monkey-patches its forward, then dropped so the
        # frozen base lives solely in the DiT module tree (not as a child
        # of this adapter).
        self.org_module = base_layer

        # Per-Linear cache of the active routing weights for this step.
        # Set once per DiT forward by ``FeRANetwork.prepare_forward``;
        # the same tensor reference is shared by every FeRALinear in the
        # network so a single write propagates to all sites.
        self._routing_weights: Optional[torch.Tensor] = None
        self._multiplier: float = 1.0

    def apply_to(self) -> None:
        """LoRA-style monkey-patch — keep ``org_module`` in place inside the
        DiT and redirect its forward through us.

        After this returns, ``parent.<child>`` is still the original
        ``nn.Linear`` instance; ``parent.<child>.forward`` is ``self.forward``
        (bound to this FeRALinear). The Linear's parameters stay reachable
        via ``named_modules()`` so block-swap can offload them.
        """
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module  # release ownership of the frozen base

    def set_routing_weights(self, weights: Optional[torch.Tensor]) -> None:
        self._routing_weights = weights

    def set_multiplier(self, multiplier: float) -> None:
        self._multiplier = float(multiplier)

    def _cayley_effective(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-expert ``(Q_eff, P_eff)`` via batched Cayley solve.

        Cayley transform: ``R = (I + A)^{-1} (I - A)``, ``A = S - S^T``.
        ``S_q`` and ``S_p`` are stacked into one ``(2E, r, r)`` solve so a
        single LU + TRSM launch covers every expert's both sides at once
        (the same trick ``OrthoLoRAExpModule`` uses for its 2×r×r solve).

        Returns:
            Q_eff: (E, r, in)  — rotated input basis per expert.
            P_eff: (E, out, r) — rotated output basis per expert.
        """
        # (E, r, r) on top of (E, r, r) -> (2E, r, r).
        skew = torch.cat([self.S_q, self.S_p], dim=0)
        A = skew - skew.transpose(-2, -1)
        eye = self._eye_r  # (r, r); broadcasts across the leading 2E axis
        R = torch.linalg.solve(eye + A, eye - A)  # (2E, r, r)
        E = self.num_experts
        R_q = R[:E]  # (E, r, r)
        R_p = R[E:]  # (E, r, r)
        # Q_eff[k] = R_q[k] @ Q_basis  -> (E, r, in)
        Q_eff = torch.einsum("ekj,ji->eki", R_q, self.Q_basis)
        # P_eff[k] = P_basis @ R_p[k]  -> (E, out, r)
        P_eff = torch.einsum("oj,ejr->eor", self.P_basis, R_p)
        return Q_eff, P_eff

    @torch.no_grad()
    def get_distilled_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the plain-FeRA ``(lora_down, lora_up)`` factors.

        In non-ortho mode the trained Parameters are returned directly.
        In ortho mode the Cayley-rotated bases are distilled into the
        same ``(E, r, in)`` / ``(E, out, r)`` layout the inference path
        expects — with ``λ`` folded into ``lora_up`` so that
        ``Σ_e w_e · up_e @ down_e`` reproduces the ortho forward exactly.
        Used at save time so disk files remain inference / ComfyUI
        compatible regardless of whether the trainer used ortho mode.
        """
        if not self.ortho:
            return self.lora_down, self.lora_up
        Q_eff, P_eff = self._cayley_effective()
        # Fold λ into lora_up along the rank axis. lambda_layer: (E, r);
        # P_eff: (E, out, r); broadcast on the out axis. The broadcast
        # result isn't contiguous (the λ side has stride 0 on the out
        # axis); ``safetensors.save`` rejects non-contiguous tensors so
        # we materialize both factors before returning.
        lam = self.lambda_layer.view(self.num_experts, 1, self.rank).to(P_eff.dtype)
        up = (P_eff * lam).contiguous()
        return Q_eff.contiguous(), up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.org_forward(x)
        w = self._routing_weights
        if w is None or self._multiplier == 0.0:
            return base_out

        if self.ortho:
            # Ortho forward — low-dim factored path.
            #
            # The materialized form is
            #   adapter = Σ_k w_k · P_eff_k · diag(λ_k) · Q_eff_k · x
            #   Q_eff_k = R_q_k @ Q_basis,  P_eff_k = P_basis @ R_p_k
            # Since (P_basis, Q_basis) are frozen and shared across experts,
            # we factor them to the boundary and route entirely in r-dim:
            #   adapter = P_basis @ [ Σ_k w_k · R_p_k · diag(λ_k) · R_q_k
            #                         · (Q_basis @ x) ]
            # vs the naive expansion this:
            #   - removes the (E, r, in) and (E, out, r) intermediates that
            #     autograd would otherwise pin until backward (~tens of MB
            #     across Anima's adapted Linears under grad checkpointing).
            #   - cuts the dominant per-token FLOPs from E·r·in + E·r·out
            #     to (r·in + r·out) + 2·E·r² — roughly E× cheaper at default
            #     (E, r) = (3, 4) since the (in→r) projection now runs once.
            # Mathematically bit-equivalent to the naive form; only the
            # autograd graph shape and FLOP count change.
            skew = torch.cat([self.S_q, self.S_p], dim=0)
            A = skew - skew.transpose(-2, -1)
            eye = self._eye_r
            R = torch.linalg.solve(eye + A, eye - A)  # (2E, r, r)
            E = self.num_experts
            R_q = R[:E]  # (E, r, r)
            R_p = R[E:]  # (E, r, r)

            compute_dtype = self.P_basis.dtype  # follow OrthoLoRA Exp convention
            x_c = x if x.dtype == compute_dtype else x.to(compute_dtype)

            # Down boundary: project x through frozen Q_basis ONCE (shared
            # across experts). (..., in) → (..., r).
            x_proj = torch.einsum("...i,ji->...j", x_c, self.Q_basis)

            # Per-expert R_q rotation in r-dim. (..., r) → (..., E, r).
            lx = torch.einsum("...j,eij->...ei", x_proj, R_q)

            # Per-expert λ scaling — mirrors OrthoLoRA Exp placement
            # (pre-gates).
            lx = lx * self.lambda_layer.to(compute_dtype)

            # Gate weighting — identical broadcasting to non-ortho.
            B = w.shape[0]
            n_mid = lx.ndim - 3
            view_shape = (B,) + (1,) * n_mid + (E, 1)
            lx = lx * w.view(view_shape).to(compute_dtype)

            # Per-expert R_p rotation + sum-over-experts in one einsum.
            # (E, r, r) · (..., E, r) → (..., r).
            mid = torch.einsum("ejr,...er->...j", R_p, lx)

            # Up boundary: project through frozen P_basis ONCE (shared).
            # (out, r) · (..., r) → (..., out).
            adapter = torch.einsum("oj,...j->...o", self.P_basis, mid)
            adapter = adapter * (self.scale * self._multiplier)
            return base_out + adapter.to(base_out.dtype)

        # x: (..., in). Anima's Linears see (B, T, in); some adapters use
        # (B, ..., in). einsum's "..." handles either.
        compute_dtype = self.lora_down.dtype
        x_c = x if x.dtype == compute_dtype else x.to(compute_dtype)

        # Single batched down projection over all experts:
        #   (..., in) @ (E, r, in)^T  ->  (..., E, r)
        # Saved for backward: ONE tensor of shape (..., E, r). The naive
        # author loop saves ``E × (..., out)`` activations here.
        lx = torch.einsum("...i,eri->...er", x_c, self.lora_down)

        # Per-batch gate weighting. ``w`` is (B, E); ``lx`` is (B, ..., E, r).
        # Broadcast w as (B, 1, ..., 1, E, 1).
        B = w.shape[0]
        E = w.shape[1]
        n_mid = lx.ndim - 3  # dims between batch and E (e.g. token dim T)
        view_shape = (B,) + (1,) * n_mid + (E, 1)
        lx = lx * w.view(view_shape).to(compute_dtype)

        # Single batched up projection over all experts:
        #   (..., E, r) @ (E, out, r)^T  ->  (..., out)
        adapter = torch.einsum("...er,eor->...o", lx, self.lora_up)
        adapter = adapter * (self.scale * self._multiplier)

        return base_out + adapter.to(base_out.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Save-format converters (fused training-side ↔ split ComfyUI-side)
# ─────────────────────────────────────────────────────────────────────────────
#
# Anima's training-side DiT uses fused projections on self-attention
# (``qkv_proj`` — one Linear emitting ``[Q | K | V]`` along the output
# axis) and cross-attention KV (``kv_proj`` — emitting ``[K | V]``).
# ComfyUI's cosmos backbone (``comfy/comfy/ldm/cosmos/predict2.py``) uses
# separate ``q_proj`` / ``k_proj`` / ``v_proj`` Linears, so a FeRA file
# saved against the fused names doesn't resolve in ComfyUI.
#
# We resolve this by writing the **split** layout on disk:
#
#   * ``save_weights`` always emits split prefixes (q/k/v for self_attn,
#     k/v for cross_attn).
#   * ``load_state_dict`` recognizes either layout and re-fuses on the
#     fly so the training-side ``FeRALinear`` (which adapts the fused
#     base Linear) receives a single stacked Parameter.
#
# Math: the fused output is laid out ``[Q | K | V]`` along the last axis
# (training-side ``Attention.compute_qkv`` does
# ``qkv.unflatten(-1, (3, n_heads, head_dim)).unbind(dim=-3)`` which is
# row-major). Splitting ``lora_up: (E, 3·inner, r)`` along dim=1 into
# three ``(E, inner, r)`` chunks therefore matches each split Linear's
# own ``lora_up`` exactly. ``lora_down: (E, r, in)`` is shared across
# the three (input space is common), so each split prefix gets an
# identical copy. Disk overhead vs fused: ~2× ``lora_down`` for qkv and
# ~1× for kv, both negligible against the dominant ``lora_up`` term.


_FUSED_QKV_SUFFIX = "_qkv_proj"
_FUSED_KV_SUFFIX = "_kv_proj"
_SPLIT_QKV_SUFFIXES = ("_q_proj", "_k_proj", "_v_proj")
_SPLIT_KV_SUFFIXES = ("_k_proj", "_v_proj")


def _split_fused_state_dict(
    sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Convert fused ``qkv_proj`` / ``kv_proj`` FeRA entries to split
    ``q_proj`` / ``k_proj`` / ``v_proj`` entries (ComfyUI-compatible).

    Passes through every key that isn't a fused FeRA pair unchanged
    (router params, ``output_proj`` / MLP entries, cross_attn ``q_proj``).
    """
    out: Dict[str, torch.Tensor] = {}
    consumed: set = set()

    # First pass: find fused prefixes by their lora_down/up pair.
    fused_prefixes = {
        k.rsplit(".", 1)[0]
        for k in sd
        if k.endswith(".lora_down") or k.endswith(".lora_up")
    }

    for prefix in sorted(fused_prefixes):
        if prefix.endswith(_FUSED_QKV_SUFFIX):
            base = prefix[: -len(_FUSED_QKV_SUFFIX)]
            down = sd[f"{prefix}.lora_down"]
            up = sd[f"{prefix}.lora_up"]
            three_inner = up.shape[1]
            if three_inner % 3 != 0:
                raise ValueError(
                    f"{prefix}: lora_up out dim {three_inner} not divisible by 3"
                )
            inner = three_inner // 3
            chunks = (up[:, 0:inner, :], up[:, inner : 2 * inner, :], up[:, 2 * inner :, :])
            for suffix, up_chunk in zip(_SPLIT_QKV_SUFFIXES, chunks):
                out[f"{base}{suffix}.lora_down"] = down.clone()
                out[f"{base}{suffix}.lora_up"] = up_chunk.clone().contiguous()
            consumed.add(f"{prefix}.lora_down")
            consumed.add(f"{prefix}.lora_up")
        elif prefix.endswith(_FUSED_KV_SUFFIX):
            base = prefix[: -len(_FUSED_KV_SUFFIX)]
            down = sd[f"{prefix}.lora_down"]
            up = sd[f"{prefix}.lora_up"]
            two_inner = up.shape[1]
            if two_inner % 2 != 0:
                raise ValueError(
                    f"{prefix}: lora_up out dim {two_inner} not divisible by 2"
                )
            inner = two_inner // 2
            chunks = (up[:, 0:inner, :], up[:, inner : 2 * inner, :])
            for suffix, up_chunk in zip(_SPLIT_KV_SUFFIXES, chunks):
                out[f"{base}{suffix}.lora_down"] = down.clone()
                out[f"{base}{suffix}.lora_up"] = up_chunk.clone().contiguous()
            consumed.add(f"{prefix}.lora_down")
            consumed.add(f"{prefix}.lora_up")

    for key, value in sd.items():
        if key in consumed:
            continue
        out[key] = value
    return out


def _fuse_split_state_dict(
    sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Inverse of ``_split_fused_state_dict`` — re-fuse split q/k/v FeRA
    entries back into ``qkv_proj`` / ``kv_proj`` for the training-side
    fused DiT.

    Detection is structural: a ``{base}_self_attn_q_proj`` prefix paired
    with its ``_k_proj`` and ``_v_proj`` siblings is fused into a single
    ``{base}_self_attn_qkv_proj``. A ``{base}_cross_attn_k_proj`` paired
    with ``_v_proj`` is fused into ``{base}_cross_attn_kv_proj`` (the
    cross_attn ``q_proj`` is identical in both formats and passes
    through untouched).

    Idempotent: a state dict already in fused form has no q/k/v triplet
    siblings, so this function returns it unchanged.
    """
    out: Dict[str, torch.Tensor] = {}
    consumed: set = set()

    prefixes = {
        k.rsplit(".", 1)[0]
        for k in sd
        if k.endswith(".lora_down") or k.endswith(".lora_up")
    }

    for prefix in sorted(prefixes):
        # self_attn — q + k + v triplet.
        if prefix.endswith("_self_attn_q_proj"):
            base = prefix[: -len("_q_proj")]
            q, k, v = f"{base}_q_proj", f"{base}_k_proj", f"{base}_v_proj"
            if k in prefixes and v in prefixes:
                up_q = sd[f"{q}.lora_up"]
                up_k = sd[f"{k}.lora_up"]
                up_v = sd[f"{v}.lora_up"]
                # lora_down is duplicated across the three; take q's
                # canonical copy (validate as a courtesy).
                down = sd[f"{q}.lora_down"]
                if not torch.equal(down, sd[f"{k}.lora_down"]):
                    logger.warning(
                        f"FeRA fuse: {k}.lora_down differs from {q}.lora_down — "
                        "using q's; downstream may diverge from a clean split."
                    )
                fused_up = torch.cat([up_q, up_k, up_v], dim=1).contiguous()
                qkv = f"{base}_qkv_proj"
                out[f"{qkv}.lora_down"] = down
                out[f"{qkv}.lora_up"] = fused_up
                for p in (q, k, v):
                    consumed.add(f"{p}.lora_down")
                    consumed.add(f"{p}.lora_up")
        # cross_attn — k + v pair (q stays as-is).
        elif prefix.endswith("_cross_attn_k_proj"):
            base = prefix[: -len("_k_proj")]
            k, v = f"{base}_k_proj", f"{base}_v_proj"
            if v in prefixes:
                up_k = sd[f"{k}.lora_up"]
                up_v = sd[f"{v}.lora_up"]
                down = sd[f"{k}.lora_down"]
                if not torch.equal(down, sd[f"{v}.lora_down"]):
                    logger.warning(
                        f"FeRA fuse: {v}.lora_down differs from {k}.lora_down — "
                        "using k's."
                    )
                fused_up = torch.cat([up_k, up_v], dim=1).contiguous()
                kv = f"{base}_kv_proj"
                out[f"{kv}.lora_down"] = down
                out[f"{kv}.lora_up"] = fused_up
                consumed.add(f"{k}.lora_down")
                consumed.add(f"{k}.lora_up")
                consumed.add(f"{v}.lora_down")
                consumed.add(f"{v}.lora_up")

    for key, value in sd.items():
        if key in consumed:
            continue
        out[key] = value
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Network surface
# ─────────────────────────────────────────────────────────────────────────────


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    rank = network_dim if network_dim is not None else 4
    alpha = network_alpha if network_alpha is not None else float(rank)

    num_experts = int(kwargs.get("fera_num_experts", 3))
    num_bands = int(kwargs.get("fera_num_bands", 3))
    router_tau = float(kwargs.get("fera_router_tau", 0.7))
    router_hidden = int(kwargs.get("fera_router_hidden", 64))
    fei_sigma_low_div = float(kwargs.get("fei_sigma_low_div", 4.0))
    fecl_weight = float(kwargs.get("fera_fecl_weight", 0.0))
    target_modules = str(kwargs.get("fera_target_modules", _DEFAULT_TARGET_REGEX))
    ortho_raw = kwargs.get("fera_ortho", False)
    ortho = ortho_raw if isinstance(ortho_raw, bool) else str(ortho_raw).lower() == "true"
    ortho_init_std = float(kwargs.get("fera_ortho_init_std", 0.02))

    network = FeRANetwork(
        unet=unet,
        rank=rank,
        alpha=alpha,
        multiplier=multiplier,
        num_experts=num_experts,
        num_bands=num_bands,
        router_tau=router_tau,
        router_hidden=router_hidden,
        fei_sigma_low_div=fei_sigma_low_div,
        fecl_weight=fecl_weight,
        target_modules_regex=target_modules,
        ortho=ortho,
        ortho_init_std=ortho_init_std,
    )
    return network


def create_network_from_weights(
    multiplier,
    file,
    ae,
    text_encoders,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # Pull stamped hyperparams from safetensors metadata when available.
    meta: Dict[str, str] = {}
    if file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            meta = dict(f.metadata() or {})

    def _meta(key: str, default):
        v = meta.get(f"ss_{key}")
        if v is None:
            return default
        if isinstance(default, bool):
            return str(v).lower() == "true"
        if isinstance(default, int):
            return int(v)
        if isinstance(default, float):
            return float(v)
        return v

    rank = _meta("fera_rank", int(kwargs.get("network_dim", 4)))
    alpha = _meta("fera_alpha", float(kwargs.get("network_alpha", rank)))
    num_experts = _meta("fera_num_experts", int(kwargs.get("fera_num_experts", 3)))
    num_bands = _meta("fera_num_bands", int(kwargs.get("fera_num_bands", 3)))
    router_tau = _meta("fera_router_tau", float(kwargs.get("fera_router_tau", 0.7)))
    router_hidden = _meta(
        "fera_router_hidden", int(kwargs.get("fera_router_hidden", 64))
    )
    # Legacy fallback ``8.0`` (not ``4.0``) when loading a checkpoint that
    # has *neither* an ``ss_fei_sigma_low_div`` metadata stamp *nor* a
    # caller-supplied kwarg: such checkpoints predate the 2026-05-13
    # sweep and were trained at div=8. New checkpoints always stamp the
    # value used at train time, so this fallback only fires for old
    # metadata-less files.
    fei_sigma_low_div = _meta(
        "fei_sigma_low_div", float(kwargs.get("fei_sigma_low_div", 8.0))
    )
    fecl_weight = _meta("fera_fecl_weight", float(kwargs.get("fera_fecl_weight", 0.0)))
    target_modules = _meta(
        "fera_target_modules", str(kwargs.get("fera_target_modules", _DEFAULT_TARGET_REGEX))
    )
    # Inference defaults to the distilled (lora_down, lora_up) path even when
    # the checkpoint was trained ortho — distillation is bit-faithful and
    # avoids the per-step Cayley solve. Resume callers can override via
    # ``fera_ortho=true`` in kwargs to rehydrate native S/λ Parameters.
    ortho_kw = kwargs.get("fera_ortho", False)
    if isinstance(ortho_kw, bool):
        ortho = ortho_kw
    else:
        ortho = str(ortho_kw).lower() == "true"
    ortho_init_std = float(kwargs.get("fera_ortho_init_std", 0.02))

    network = FeRANetwork(
        unet=unet,
        rank=int(rank),
        alpha=float(alpha),
        multiplier=multiplier,
        num_experts=int(num_experts),
        num_bands=int(num_bands),
        router_tau=float(router_tau),
        router_hidden=int(router_hidden),
        fei_sigma_low_div=float(fei_sigma_low_div),
        fecl_weight=float(fecl_weight),
        target_modules_regex=str(target_modules),
        ortho=bool(ortho),
        ortho_init_std=float(ortho_init_std),
    )
    return network, weights_sd


class FeRANetwork(nn.Module):
    """Author-faithful FeRA: independent-A experts + one global router.

    Attaches as a Module that owns the router and every ``FeRALinear``.
    ``apply_to`` does the in-place ``nn.Linear`` → ``FeRALinear`` swap on
    the DiT (text encoder is left untouched by default — author paper
    targets the UNet only).
    """

    def __init__(
        self,
        unet: nn.Module,
        rank: int,
        alpha: float,
        *,
        multiplier: float = 1.0,
        num_experts: int = 3,
        num_bands: int = 3,
        router_tau: float = 0.7,
        router_hidden: int = 64,
        fei_sigma_low_div: float = 4.0,
        fecl_weight: float = 0.0,
        target_modules_regex: str = _DEFAULT_TARGET_REGEX,
        ortho: bool = False,
        ortho_init_std: float = 0.02,
    ):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.multiplier = float(multiplier)
        self.num_experts = int(num_experts)
        self.num_bands = int(num_bands)
        self.router_tau = float(router_tau)
        self.router_hidden = int(router_hidden)
        self.fei_sigma_low_div = float(fei_sigma_low_div)
        self.fecl_weight = float(fecl_weight)
        self.target_modules_regex = str(target_modules_regex)
        self.ortho = bool(ortho)
        self.ortho_init_std = float(ortho_init_std)

        self.fei_indicator = FrequencyEnergyIndicator(
            num_bands=self.num_bands, fei_sigma_low_div=self.fei_sigma_low_div
        )
        self.router = SoftFrequencyRouter(
            num_bands=self.num_bands,
            num_experts=self.num_experts,
            hidden_dim=self.router_hidden,
            tau=self.router_tau,
        )

        # Scan the DiT for target Linears now so the param count is known
        # before ``apply_to`` actually performs the swap. Construction order
        # mirrors postfix.py: build all sub-modules in ``__init__`` so
        # ``state_dict`` already has the right shape pre-apply.
        self._planned: List[Tuple[nn.Module, str, nn.Linear, str]] = []
        self._scan_targets(unet)

        # ModuleDict keyed by lora_name → FeRALinear; built in apply_to.
        self.fera_layers: nn.ModuleDict = nn.ModuleDict()

        # Most recent gates produced by the router this step. Useful for
        # telemetry / FECL.
        self._last_gates: Optional[torch.Tensor] = None
        # Last FEI we computed (for diagnostics + FECL).
        self._last_fei: Optional[torch.Tensor] = None

        # Router-stat accumulators, drained by ``metrics()`` once per log
        # step. Lazily allocated on first ``prepare_forward`` so they land
        # on the same device as the gates (no D2H syncs in the hot path).
        # Single-batch reads are noisy at FeRA's typical B=1 — averaging
        # across all forwards between drains gives a usable curve.
        self._stat_h_per_sum: Optional[torch.Tensor] = None    # scalar fp32
        self._stat_margin_sum: Optional[torch.Tensor] = None   # scalar fp32
        self._stat_gate_sum: Optional[torch.Tensor] = None     # (E,) fp32
        self._stat_argmax_count: Optional[torch.Tensor] = None  # (E,) fp32
        self._stat_n: int = 0  # samples accumulated since last drain

        ortho_str = (
            f"ortho=True (init_std={self.ortho_init_std:g}, "
            f"shared SVD bases + Cayley rotation per expert)"
            if self.ortho
            else "ortho=False (free A/B per expert)"
        )
        logger.info(
            f"FeRANetwork: target_modules={self.target_modules_regex!r} "
            f"matched {len(self._planned)} Linears in DiT — "
            f"{self.num_experts} experts × rank {self.rank} each, "
            f"router({self.num_bands} bands → {self.router_hidden} → "
            f"{self.num_experts}, τ={self.router_tau:.2f}), "
            f"σ_low = min(H,W)/{self.fei_sigma_low_div:.1f}, "
            f"fecl_weight={self.fecl_weight}, {ortho_str}"
        )

    # ---- target scan + apply -------------------------------------------------

    def _scan_targets(self, unet: nn.Module) -> None:
        """Enumerate ``(parent_module, child_name, child_linear, lora_name)``.

        ``lora_name`` is ``lora_unet_<dotted path with . → _>`` so saved
        checkpoint keys stay readable and follow the same convention as
        ``networks/lora_modules``.

        Skips ``llm_adapter.*`` modules to mirror the inference-side
        filter in ``custom_nodes/comfyui-hydralora/fera.py``. FeRA
        contributions on llm_adapter Linears can't be replayed at
        ComfyUI inference — the router pre-hook fires on
        ``diffusion_model.forward`` only, so a CFG-doubled (B=2) gate
        from the previous step gets broadcast into the B=1 text-only
        llm_adapter forward and crashes ``Attention.forward``. Training
        them in the first place just bloats the checkpoint with dead
        keys (see ``project_fera_llm_adapter_stale_gates``).
        """
        pattern = re.compile(self.target_modules_regex)
        for module_name, module in unet.named_modules():
            for child_name, child in module.named_children():
                if not isinstance(child, nn.Linear):
                    continue
                full = f"{module_name}.{child_name}" if module_name else child_name
                # Strip torch.compile wrapper if any
                full = full.replace("_orig_mod.", "")
                if full.startswith("llm_adapter.") or full == "llm_adapter":
                    continue
                if not pattern.fullmatch(full):
                    continue
                lora_name = f"{LORA_PREFIX_ANIMA}.{full}".replace(".", "_")
                self._planned.append((module, child_name, child, lora_name))

    def apply_to(
        self,
        text_encoders,
        unet,
        apply_text_encoder: bool = False,
        apply_unet: bool = True,
    ) -> None:
        if not apply_unet:
            logger.warning("FeRANetwork.apply_to: apply_unet=False is a no-op")
            return
        if apply_text_encoder:
            logger.warning(
                "FeRANetwork.apply_to: text-encoder targeting not implemented "
                "(author paper targets UNet only); skipping"
            )

        for _parent, _child_name, original_linear, lora_name in self._planned:
            fera_layer = FeRALinear(
                base_layer=original_linear,
                num_experts=self.num_experts,
                rank=self.rank,
                alpha=self.alpha,
                lora_name=lora_name,
                ortho=self.ortho,
                ortho_init_std=self.ortho_init_std,
            )
            fera_layer.set_multiplier(self.multiplier)
            # Monkey-patch the original Linear's forward in place — the
            # Linear stays in its parent's _modules so block-swap and
            # ``.to(device)`` see its weights. See FeRALinear.apply_to.
            fera_layer.apply_to()
            self.fera_layers[lora_name] = fera_layer

        logger.info(
            f"FeRA: patched {len(self.fera_layers)} Linears (base modules "
            f"remain in DiT tree for block-swap compatibility)"
        )

    # ---- per-step routing ----------------------------------------------------

    @torch.no_grad()
    def _push_gates(self, weights: Optional[torch.Tensor]) -> None:
        for layer in self.fera_layers.values():
            layer.set_routing_weights(weights)

    def prepare_forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """Compute FEI on ``z_t``, run the router, broadcast gates.

        Call once per DiT forward — before ``set_hydra_sigma`` would fire
        in the existing pipeline. Squeezes a singleton temporal dim if the
        caller hands a 5D Anima latent.

        Returns the ``(B, num_experts)`` routing weights so callers can
        log / record them if useful.
        """
        if z_t.dim() == 5:
            z_t = z_t.squeeze(2)
        fei = self.fei_indicator(z_t)  # (B, num_bands), fp32
        # Router runs in fp32 — gate weights cast in FeRALinear.forward.
        gates = self.router(fei)
        self._last_fei = fei.detach()
        self._last_gates = gates.detach()
        self._update_router_stats(self._last_gates)
        self._push_gates(gates)
        return gates

    @torch.no_grad()
    def _update_router_stats(self, gates: torch.Tensor) -> None:
        """Fold one batch of router gates into the metric accumulators.

        Stays GPU-resident — the only D2H sync happens in ``metrics()``
        when the values are drained for logging. ``gates`` is ``(B, E)``
        fp32, post-softmax.
        """
        g = gates.float()
        B, E = g.shape
        if self._stat_h_per_sum is None:
            dev = g.device
            self._stat_h_per_sum = torch.zeros((), device=dev, dtype=torch.float32)
            self._stat_margin_sum = torch.zeros((), device=dev, dtype=torch.float32)
            self._stat_gate_sum = torch.zeros(E, device=dev, dtype=torch.float32)
            self._stat_argmax_count = torch.zeros(E, device=dev, dtype=torch.float32)
        p = g.clamp_min(1e-12)
        # Per-sample entropy (B,), summed; normalization to log(E) deferred
        # to drain so this stays one fused kernel.
        self._stat_h_per_sum += -(p * p.log()).sum(-1).sum()
        # Top-1 / top-2 margin per sample, summed.
        top2 = p.topk(2, dim=-1).values
        self._stat_margin_sum += (top2[..., 0] - top2[..., 1]).sum()
        # Soft load (mean gate) and hard usage (argmax histogram), summed.
        self._stat_gate_sum += g.sum(0)
        self._stat_argmax_count += F.one_hot(g.argmax(-1), num_classes=E).float().sum(0)
        self._stat_n += int(B)

    def _reset_router_stats(self) -> None:
        if self._stat_h_per_sum is not None:
            self._stat_h_per_sum.zero_()
            self._stat_margin_sum.zero_()
            self._stat_gate_sum.zero_()
            self._stat_argmax_count.zero_()
        self._stat_n = 0

    def metrics(self, ctx: MetricContext) -> Dict[str, float]:
        """Drain router-stat accumulators into a log dict.

        Emits per-sample-averaged keys matching Hydra's ``hydra/router_*``
        convention so TensorBoard plots line up across method comparisons.
        Resets the accumulators so the next log window starts fresh.
        Returns ``{}`` if no batches have been routed since the last drain
        (e.g. metric collection ran before the first forward).
        """
        if self._stat_n == 0 or self._stat_h_per_sum is None:
            return {}
        n = float(self._stat_n)
        E = int(self.num_experts)
        log_E = math.log(E) if E > 1 else 1.0
        # Single packed [H_per_sample, margin] D2H, then the (E,) vectors.
        scalar_pack = torch.stack(
            [self._stat_h_per_sum / n, self._stat_margin_sum / n]
        ).cpu()
        gate_mean = (self._stat_gate_sum / n).cpu()
        usage_mean = (self._stat_argmax_count / n).cpu()
        # Collapse detector: entropy of mean-gate, normalized to [0, 1].
        gm = gate_mean.clamp_min(1e-12)
        h_collapse = float((-(gm * gm.log()).sum() / log_E).item())

        out: Dict[str, float] = {
            "fera/router_entropy": float(scalar_pack[0]) / log_E,
            "fera/router_entropy_collapse": h_collapse,
            "fera/router_margin": float(scalar_pack[1]),
        }
        for i in range(E):
            out[f"fera/expert_usage/{i}"] = float(usage_mean[i])
            out[f"fera/load/{i}"] = float(gate_mean[i])
        self._reset_router_stats()
        return out

    def clear_routing(self) -> None:
        """Drop routing weights (and step caches) — used at the end of an
        inference loop or before a base-pass FECL forward."""
        self._push_gates(None)
        self._last_fei = None
        self._last_gates = None

    def clear_step_caches(self) -> None:
        """Hook called between training steps (``library/training/loop.py``)
        to drop per-step tensor references — see postfix.py for the
        cudagraph rationale."""
        self._last_fei = None
        self._last_gates = None

    # ---- FECL (optional aux loss; opt-in via fecl_weight > 0) ---------------

    def compute_fecl_loss(
        self,
        z_base: torch.Tensor,
        z_fera: torch.Tensor,
        z_target: torch.Tensor,
    ) -> torch.Tensor:
        """Frequency-Energy Consistency Loss, paper Eq. (10), **unscaled**.

        Bandwise consistency between adapter correction ``δ = z_fera -
        z_base`` and residual ``r = z_fera - z_target``, weighted by the
        residual's per-band energy share. Drops to a single scalar when
        ``num_bands == 2`` (only two ratios that sum to 1 — the loss
        becomes content-free), so 2-band defaults should keep
        ``fecl_weight = 0``; bench at 3 bands if revisiting.

        Returns a 0-dim scalar **without** the ``fecl_weight`` multiplier
        — the loss registry handler in ``library/training/losses.py``
        applies the scaling so the weight lives in one place
        (matches ``_soft_tokens_contrastive_loss`` / ``_repa_loss``).
        """
        # 4D promote (FEI indicator path).
        def _to4(x):
            return x.squeeze(2) if x.dim() == 5 else x

        z_base = _to4(z_base).float()
        z_fera = _to4(z_fera).float()
        z_target = _to4(z_target).float()

        delta = z_fera - z_base
        resid = z_fera - z_target

        # Reuse FEI's pyramid: per-band component tensors (high → low).
        def _bands(z: torch.Tensor) -> List[torch.Tensor]:
            h_lat, w_lat = int(z.shape[-2]), int(z.shape[-1])
            sigmas = self.fei_indicator._band_sigmas(h_lat, w_lat)
            pyr = [z]
            for s in sigmas:
                pyr.append(gaussian_blur_2d(pyr[-1], s))
            comps = [pyr[k] - pyr[k + 1] for k in range(self.num_bands - 1)]
            comps.append(pyr[-1])
            return comps

        delta_bands = _bands(delta)
        resid_bands = _bands(resid)

        eps = 1e-8
        d_total = delta.flatten(1).pow(2).sum(-1).sqrt().clamp_min(eps)
        r_total = resid.flatten(1).pow(2).sum(-1).sqrt().clamp_min(eps)

        # Per-band weights (residual share, paper Eq. 10).
        r_band_e = torch.stack(
            [b.flatten(1).pow(2).sum(-1) for b in resid_bands], dim=-1
        )
        r_share = r_band_e / r_band_e.sum(-1, keepdim=True).clamp_min(eps)

        loss = z_target.new_zeros(z_target.shape[0])
        for k in range(self.num_bands):
            d_band = delta_bands[k].flatten(1).pow(2).sum(-1).sqrt()
            r_band = resid_bands[k].flatten(1).pow(2).sum(-1).sqrt()
            term = (d_band / d_total - r_band / r_total).pow(2)
            loss = loss + r_share[:, k] * term

        return loss.mean()

    # ---- training-side surface (matches what train.py expects) --------------

    def prepare_network(self, args) -> None:
        # Hook called once after construction. Nothing to do — kept for
        # surface parity with LoRANetwork / PostfixNetwork.
        return

    def enable_gradient_checkpointing(self) -> None:
        # Frozen base + tiny LoRA paths — checkpointing the experts isn't
        # a meaningful win. Left as a no-op (postfix.py does the same).
        return

    def prepare_grad_etc(self, text_encoder, unet) -> None:
        # The DiT is frozen by the trainer before adapter attach; we only
        # need to enable grads on our own params (router + per-Linear
        # lora_down / lora_up stacked Parameters). The base Linears
        # remain in the DiT tree under the trainer's existing freeze.
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet) -> None:
        self.train()

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        lr = unet_lr or default_lr
        params = [{"params": list(self.parameters()), "lr": lr}]
        return params, ["fera"]

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr=None):
        params, _ = self.prepare_optimizer_params_with_multiple_te_lrs(
            text_encoder_lr, unet_lr, default_lr
        )
        return params

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = float(multiplier)
        for layer in self.fera_layers.values():
            layer.set_multiplier(self.multiplier)

    def is_mergeable(self) -> bool:
        # In principle each expert is a plain LoRA — but a router-mixed
        # output isn't a single ΔW. Fold only after picking a routing
        # snapshot, which is not the typical inference path. Refuse for
        # now; revisit if a static-gate inference mode is needed.
        return False

    # ---- save / load --------------------------------------------------------

    def save_weights(self, file, dtype, metadata):
        dtype = dtype or torch.bfloat16

        state_dict = {}
        # Router params — keep at fp32 for safety; they're tiny.
        for k, v in self.router.state_dict().items():
            state_dict[f"router.{k}"] = v.detach().clone().cpu().float()
        # Per-layer stacked expert weights (fused names as they live in
        # the training-side DiT).
        # ``lora_down``: (E, r, in)   ``lora_up``: (E, out, r)
        # Under ortho mode, ``get_distilled_weights`` rebuilds these from
        # ``(Q_basis, P_basis, S_q, S_p, λ)`` so the saved file stays
        # inference / ComfyUI compatible without any loader changes. The
        # native ortho Parameters are *also* saved (below) so training
        # resume can rehydrate the Cayley state.
        for lora_name, layer in self.fera_layers.items():
            down, up = layer.get_distilled_weights()
            state_dict[f"{lora_name}.lora_down"] = (
                down.detach().clone().cpu().to(dtype)
            )
            state_dict[f"{lora_name}.lora_up"] = (
                up.detach().clone().cpu().to(dtype)
            )
            if layer.ortho:
                # Native ortho keys for training resume. Stored in fp32 so
                # Cayley solve precision survives a round-trip (S/λ are
                # tiny in absolute terms — bf16 quantization would shift
                # the implied rotation noticeably at rank 8).
                state_dict[f"{lora_name}.S_p"] = (
                    layer.S_p.detach().clone().cpu().float()
                )
                state_dict[f"{lora_name}.S_q"] = (
                    layer.S_q.detach().clone().cpu().float()
                )
                state_dict[f"{lora_name}.lambda_layer"] = (
                    layer.lambda_layer.detach().clone().cpu().float()
                )
                state_dict[f"{lora_name}.P_basis"] = (
                    layer.P_basis.detach().clone().cpu().float()
                )
                state_dict[f"{lora_name}.Q_basis"] = (
                    layer.Q_basis.detach().clone().cpu().float()
                )

        # ComfyUI's cosmos backbone uses split q/k/v Linears (not fused
        # qkv_proj / kv_proj), so we always write the split layout on
        # disk. ``load_state_dict`` re-fuses transparently when this same
        # file is loaded back into the training-side DiT.
        # The ortho native keys (``.S_p``, ``.S_q``, ``.lambda_layer``,
        # ``.P_basis``, ``.Q_basis``) pass through ``_split_fused_state_dict``
        # untouched — it only rewrites ``.lora_down`` / ``.lora_up`` pairs.
        state_dict = _split_fused_state_dict(state_dict)

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library.training.hashing import precalculate_safetensors_hashes

            if metadata is None:
                metadata = {}
            metadata["ss_network_module"] = "networks.methods.fera"
            metadata["ss_network_spec"] = "fera"
            metadata["ss_fera_rank"] = str(self.rank)
            metadata["ss_fera_alpha"] = str(self.alpha)
            metadata["ss_fera_num_experts"] = str(self.num_experts)
            metadata["ss_fera_num_bands"] = str(self.num_bands)
            metadata["ss_fera_router_tau"] = str(self.router_tau)
            metadata["ss_fera_router_hidden"] = str(self.router_hidden)
            metadata["ss_fei_sigma_low_div"] = str(self.fei_sigma_low_div)
            metadata["ss_fera_fecl_weight"] = str(self.fecl_weight)
            metadata["ss_fera_target_modules"] = self.target_modules_regex
            metadata["ss_fera_ortho"] = str(self.ortho)
            metadata["ss_fera_ortho_init_std"] = str(self.ortho_init_std)

            model_hash, legacy_hash = precalculate_safetensors_hashes(
                state_dict, metadata
            )
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, strict=False)
        if info.missing_keys:
            logger.warning(
                f"FeRA: missing keys on load: {info.missing_keys[:5]}..."
            )
        if info.unexpected_keys:
            logger.warning(
                f"FeRA: unexpected keys on load: {info.unexpected_keys[:5]}..."
            )
        return info

    def load_state_dict(self, state_dict, strict: bool = True):
        # Re-fuse split q/k/v entries back into fused qkv_proj / kv_proj
        # for the training-side DiT (which has fused projections).
        # Idempotent on already-fused dicts, so this is safe for both
        # ComfyUI-format files (split on disk) and any legacy fused
        # files still around.
        state_dict = _fuse_split_state_dict(dict(state_dict))

        # Translate flat per-Linear keys into the ModuleDict path so
        # ``nn.Module.load_state_dict`` is happy. Router keys
        # (``router.*``) pass through untouched. Both the distilled keys
        # (``.lora_down`` / ``.lora_up``) and the native ortho keys
        # (``.S_p`` / ``.S_q`` / ``.lambda_layer`` / ``.P_basis`` /
        # ``.Q_basis``) get the same ``fera_layers.`` prefix added — the
        # underlying nn.Module then accepts whichever subset matches the
        # current network's parameterization (strict=False tolerates the
        # leftovers when loading across mode boundaries).
        _LAYER_SUFFIXES = (
            ".lora_down",
            ".lora_up",
            ".S_p",
            ".S_q",
            ".lambda_layer",
            ".P_basis",
            ".Q_basis",
        )
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("router."):
                remapped[key] = value
                continue
            if any(key.endswith(s) for s in _LAYER_SUFFIXES):
                remapped[f"fera_layers.{key}"] = value
            else:
                remapped[key] = value
        return super().load_state_dict(remapped, strict=strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):  # type: ignore[override]
        # Inverse of load_state_dict's remap so save_weights / external
        # consumers see flat per-Linear keys (``{lora_name}.lora_down`` /
        # ``{lora_name}.lora_up``) without the ``fera_layers.`` prefix.
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        out: Dict[str, torch.Tensor] = {}
        flat_prefix = f"{prefix}fera_layers."
        for key, value in sd.items():
            if key.startswith(flat_prefix):
                rest = key[len(flat_prefix) :]
                # rest = "{lora_name}.lora_down" or "{lora_name}.lora_up"
                out[f"{prefix}{rest}"] = value
                continue
            out[key] = value
        return out
