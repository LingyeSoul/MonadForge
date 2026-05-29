# ChimeraHydra: dual-pool additive MoE with TWO Cayley A's per Linear.
#
# Two independent HydraLoRAs (Tian et al., NeurIPS'24; arXiv:2404.19245) glued
# at the residual — that's the chimera. The content half routes K_c B-heads
# off the pooled post-LLM-adapter ``crossattn_emb`` (prompt content) via the
# network-level ContentRouter. The frequency half routes K_f B-heads off the
# network-level FreqRouter fed FEI of z_t (FeRA, arXiv:2511.17979). T-LoRA's
# rank mask (Liu et al.; TimeStep Master, arXiv:2503.07416) modulates the
# content half only — the freq half stays full-rank at every t, giving an
# asymmetric "core expert always on" / "context expert rank-modulated" split
# inspired by TimeStep Master's asymmetric mixture.
#
# Per Linear:
#
#     A_c = Cayley(S_q_c) · Q_basis_c          (r, in)   — content latent
#     A_f = Cayley(S_q_f) · Q_basis_f          (r, in)   — freq    latent
#
#     B_c[k] = P_bases_c[k] · Cayley(S_p_c[k]) (out, r)  k = 0..K_c-1
#     B_f[j] = P_bases_f[j] · Cayley(S_p_f[j]) (out, r)  j = 0..K_f-1
#
#     Δy = Σ_c π_c[c] · B_c[c] (A_c x · λ_c · mask_t(σ))     ◄ content branch
#        + Σ_f π_f[f] · B_f[f] (A_f x · λ_f)                  ◄ freq    branch
#
# SVD partition gives free orthogonality on BOTH sides:
#   * Top 2r right-singular vectors of W: first r → Q_basis_c, next r →
#     Q_basis_f. Q_basis_c.row_space ⊥ Q_basis_f.row_space.
#   * Top (K_c+K_f)·r left-singular vectors of W: first K_c·r partitioned
#     into (K_c, out, r) → P_bases_c, next K_f·r partitioned into
#     (K_f, out, r) → P_bases_f. Every P_bases_c[k].col_space ⊥ every
#     P_bases_f[j].col_space.
#
# Both pools are network-routed and centered-gate by construction (the only
# shipped configuration): each pool's gate is recentered to ``π − 1/K`` in the
# forward and λ_c/λ_f start at ``lambda_init`` (>0). ΔW = 0 at init (base
# preserved exactly) while each pool's disjoint per-expert P-subspaces still
# feed its router a nonzero step-0 gradient. Both routers live at the network
# level (``ContentRouter`` / ``FreqRouter`` in ``LoRANetwork``) and write π_c /
# π_f into the slot-assigned buffers below; there is no per-Linear router.

import logging
from typing import Dict, List, Optional

import torch

from networks.attn_fuse import match_fused_spec
from networks.lora_modules.base import BaseLoRAModule, _absorb_channel_scale
from networks.lora_modules.custom_autograd import lora_down_project
from networks.lora_modules.lora import defuse_standard_qkv

logger = logging.getLogger(__name__)


class _ChimeraRoutingMixin:
    """Shared dual-pool routing plumbing for the train + inference modules.

    Both chimera classes carry two gate buffers — ``_content_routing_weights``
    (π_c, written by the network-level ContentRouter) and
    ``_freq_routing_weights`` (π_f, written by the network-level FreqRouter) —
    and combine the two pools' up-projections identically. The slot-assign
    contract (NO ``.detach()``, NO ``.copy_()`` — the buffer must carry the
    router's ``grad_fn`` so ``∂L/∂π`` reaches the router parameters) lives here
    once, mirroring ``router_state._set_routing_weights`` and the matching
    methods on ``HydraLoRAModule``.
    """

    def _register_routing_buffers(self, K_c: int, K_f: int) -> None:
        # Uniform 1/K placeholders, overwritten per-step by the network-level
        # routers via direct slot assignment. Non-persistent — re-derived on
        # construction.
        self.register_buffer(
            "_content_routing_weights",
            torch.full((1, K_c), 1.0 / max(K_c, 1), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_freq_routing_weights",
            torch.full((1, K_f), 1.0 / max(K_f, 1), dtype=torch.float32),
            persistent=False,
        )
        # Cached (B, K_c+K_f) gate read by ``LoRANetwork._get_chimera_balance_
        # loss`` (slices at K_c into independent per-pool Switch losses).
        self._last_gate = None

    def set_freq_routing_weights(self, weights: torch.Tensor) -> None:
        """Slot-assign the freq pool's gates (preserves grad_fn)."""
        self._freq_routing_weights = self._coerce_gate(
            weights, self._freq_routing_weights
        )

    def clear_freq_routing_weights(self) -> None:
        """Reset to uniform 1/K_f without rebinding the pointer."""
        K_f = int(self._freq_routing_weights.shape[-1])
        self._freq_routing_weights.fill_(1.0 / max(K_f, 1))

    def set_content_routing_weights(self, weights: torch.Tensor) -> None:
        """Slot-assign π_c from the network-level ContentRouter (preserves
        grad_fn — same contract as :meth:`set_freq_routing_weights`)."""
        self._content_routing_weights = self._coerce_gate(
            weights, self._content_routing_weights
        )

    def clear_content_routing_weights(self) -> None:
        K_c = int(self._content_routing_weights.shape[-1])
        self._content_routing_weights.fill_(1.0 / max(K_c, 1))

    @staticmethod
    def _coerce_gate(weights: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
        w = weights.to(dtype=buf.dtype, device=buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        return w

    @staticmethod
    def _broadcast_gate(pi: torch.Tensor, batch: int) -> torch.Tensor:
        """(K,) / (1, K) → (batch, K); already-batched passes through."""
        if pi.dim() == 1:
            pi = pi.unsqueeze(0)
        if pi.shape[0] == 1 and batch > 1:
            pi = pi.expand(batch, -1)
        return pi

    def _content_gate_raw(self, batch: int) -> torch.Tensor:
        """π_c broadcast to the batch, fp32 (the einsum casts at its boundary)."""
        return self._broadcast_gate(self._content_routing_weights, batch).float()

    def _freq_gate_raw(self, batch: int) -> torch.Tensor:
        """π_f broadcast to the batch in its native dtype (grad_fn intact)."""
        return self._broadcast_gate(self._freq_routing_weights, batch)

    def _full_gate(self, pi_c_raw: torch.Tensor) -> torch.Tensor:
        """(B, K_c+K_f) raw (uncentered) simplex for the per-pool balance loss.

        ``LoRANetwork._get_chimera_balance_loss`` slices at
        ``num_experts_content`` to recover the two halves.
        """
        pi_f = self._broadcast_gate(self._freq_routing_weights, pi_c_raw.shape[0])
        pi_f = pi_f.to(pi_c_raw.dtype)
        if pi_f.shape[0] != pi_c_raw.shape[0]:
            pi_f = pi_f.expand(pi_c_raw.shape[0], -1)
        return torch.cat([pi_c_raw, pi_f], dim=-1)

    @staticmethod
    def _center(pi: torch.Tensor, num_experts: int) -> torch.Tensor:
        """Recenter a gate to ``π − 1/K`` (NOT in-place — preserves grad_fn so
        the FreqRouter / ContentRouter still receive gradient). A uniform gate
        then contributes exactly 0, keeping ΔW=0 at init."""
        return pi - (1.0 / num_experts)

    @staticmethod
    def _combine_up(lx_c, lx_f, comb_c, comb_f, scale):
        """Cat both pools along the rank axis → ONE bmm (only one full
        hidden-size tensor live at a time). ``scale`` is shared (rank_dropout
        returns the same scalar for both pools), so it folds out of the sum.
        """
        orig_shape = lx_c.shape
        B = orig_shape[0]
        lx_cat = torch.cat(
            [lx_c.reshape(B, -1, orig_shape[-1]), lx_f.reshape(B, -1, orig_shape[-1])],
            dim=-1,
        )  # (B, L, 2r)
        comb_cat = torch.cat([comb_c, comb_f], dim=-1)  # (B, out, 2r)
        out = torch.bmm(lx_cat, comb_cat.transpose(1, 2))
        return (out * scale).reshape(*orig_shape[:-1], -1)


class ChimeraHydraLoRAModule(_ChimeraRoutingMixin, BaseLoRAModule):
    """ChimeraHydra training-time module: two Cayley A's, two B-pools, both
    network-routed (content via ContentRouter, freq via FreqRouter).

    Concretely two HydraLoRAs in parallel — same shape on each side, the
    content half routed by the pooled ``crossattn_emb`` and the freq half by
    FEI(z_t). T-LoRA's rank mask is folded into the content branch's effective
    P only — the freq branch keeps full rank at every t (TimeStep Master-style
    asymmetric pool).

    The shared SVD of the base weight gives both pools their bases: distinct
    singular-vector slices on each side ⇒ A_c.row_space ⊥ A_f.row_space and
    B_c[*].col_space ⊥ B_f[*].col_space, structurally, at step 0. Cayley
    rotates each within its assigned subspace.

    Save distills Cayley → free-form per pool (see
    :meth:`distill_save_state_dict` / :meth:`build_moe_state_dict` below);
    load rebuilds a ``ChimeraHydraInferenceModule`` rather than
    re-instantiating this class.
    """

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
        num_experts_content: int = 3,
        num_experts_freq: int = 3,
        channel_scale=None,
        lambda_init: float = 0.0,
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

        if num_experts_content <= 0 or num_experts_freq <= 0:
            raise ValueError(
                f"ChimeraHydra requires both pools non-empty: "
                f"K_c={num_experts_content}, K_f={num_experts_freq}"
            )

        K_c = int(num_experts_content)
        K_f = int(num_experts_freq)
        r = int(lora_dim)

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.num_experts_content = K_c
        self.num_experts_freq = K_f
        self.num_experts = K_c + K_f
        self.in_dim = in_dim

        # SVD partition. Each pool wants:
        #   * its own r right-singular vectors → Q_basis_{c,f} (r, in)
        #   * its own pool-size·r left-singular vectors → P_bases_{c,f}
        #     (K_*, out, r)
        # Take a single low-rank SVD with q big enough to cover both pools.
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        W = org_module.weight.data.float().to(init_device)
        target_left = (K_c + K_f) * r  # need this many U columns
        target_right = 2 * r  # need this many V columns
        max_cols = min(W.shape)
        target = max(target_left, target_right)
        disjoint = target <= max_cols
        q = min(target + 6, max_cols) if disjoint else min(r + 6, max_cols)
        U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)

        if disjoint:
            # Right-singular split: V has shape (in, q). Top r → content,
            # next r → freq. Both are subsets of the same SVD basis so
            # V[:, :r].T @ V[:, r:2r] = 0 (orthonormal columns).
            Q_basis_c = V[:, :r].T.clone().contiguous()  # (r, in)
            Q_basis_f = V[:, r : 2 * r].T.clone().contiguous()  # (r, in)

            # Left-singular split: U has shape (out, q). First K_c·r →
            # content P stack, next K_f·r → freq P stack. Within each
            # stack columns are reshape-partitioned into pool-size disjoint
            # slices — same trick OrthoHydra uses (see ortho.py docstring),
            # giving B_c[k]^T B_c[k'] = 0 for k≠k' and B_f[j]^T B_f[j'] = 0
            # for j≠j'. Across pools, B_c[k]^T B_f[j] = 0 by SVD ortho.
            U_c = U[:, : K_c * r].reshape(out_dim, K_c, r)
            P_bases_c_init = U_c.permute(1, 0, 2).clone().contiguous()
            U_f = U[:, K_c * r : (K_c + K_f) * r].reshape(out_dim, K_f, r)
            P_bases_f_init = U_f.permute(1, 0, 2).clone().contiguous()
        else:
            # Narrow-layer fallback: replicate top-r slice into each pool.
            # Pool-orthogonality is lost; both pools rely on the Cayley
            # rotations diverging during training.
            logger.warning(
                f"{lora_name}: min(out={out_dim}, in={in_dim})={max_cols} < "
                f"max(K_c+K_f, 2)·r = {target}; falling back to shared "
                "SVD slice (pools start identical, rely on Cayley divergence)."
            )
            Q_shared = V[:, :r].T.clone().contiguous()
            Q_basis_c = Q_shared.clone()
            Q_basis_f = Q_shared.clone()
            P_shared = U[:, :r].clone().contiguous()
            P_bases_c_init = P_shared.unsqueeze(0).expand(K_c, -1, -1).contiguous()
            P_bases_f_init = P_shared.unsqueeze(0).expand(K_f, -1, -1).contiguous()
        del U, _S_vals, V, W
        self._disjoint_basis = disjoint

        # Frozen subspace bases (one per pool).
        self.register_buffer("Q_basis_c", Q_basis_c.cpu())
        self.register_buffer("Q_basis_f", Q_basis_f.cpu())
        self.register_buffer("P_bases_c", P_bases_c_init.cpu())  # (K_c, out, r)
        self.register_buffer("P_bases_f", P_bases_f_init.cpu())  # (K_f, out, r)

        # Cayley(0) = I → at init each effective basis equals its frozen
        # buffer. Per-pool S parameters are independent.
        self.S_q_c = torch.nn.Parameter(torch.zeros(r, r))
        self.S_q_f = torch.nn.Parameter(torch.zeros(r, r))
        self.S_p_c = torch.nn.Parameter(torch.zeros(K_c, r, r))
        self.S_p_f = torch.nn.Parameter(torch.zeros(K_f, r, r))

        # Per-pool λ: independent magnitudes through training (no shared
        # scaling). Centered-gate starts λ at ``lambda_init`` (>0): the
        # recentered gate keeps ΔW=0 at init while feeding both routers a
        # step-0 gradient.
        lam0 = float(lambda_init)
        self.lambda_c = torch.nn.Parameter(torch.full((1, r), lam0))
        self.lambda_f = torch.nn.Parameter(torch.full((1, r), lam0))

        # Channel-scale absorption: SmoothQuant-style x rebalance happens
        # ONCE at the input (via inv_scale), then both A_c and A_f need
        # their input columns pre-scaled to compensate. _register_channel_
        # _scale handles Q_basis_c + registers inv_scale; we then manually
        # apply the same column-scale to Q_basis_f.
        if channel_scale is not None:
            self._register_channel_scale(self.Q_basis_c, channel_scale)
            _absorb_channel_scale(self.Q_basis_f, channel_scale)

        # Frozen bases → bf16 (saved-for-backward halved). Cayley solve
        # stays fp32 (orthogonality invariant: R^T R = I to ~1e-7 fp32 vs
        # ~1e-2 bf16 per OrthoLoRA rationale).
        self.Q_basis_c = self.Q_basis_c.to(torch.bfloat16)
        self.Q_basis_f = self.Q_basis_f.to(torch.bfloat16)
        self.P_bases_c = self.P_bases_c.to(torch.bfloat16)
        self.P_bases_f = self.P_bases_f.to(torch.bfloat16)

        # Default off; the factory flips this to True when
        # ``use_custom_down_autograd=true`` is in the config (see
        # ``factory.py``). Forward branches on the flag — when on, both
        # pools' down-projects go through ``lora_down_project`` and the
        # rebalanced ``x_lora`` (B, L, in) is not materialized per Linear.
        self.use_custom_down_autograd = False

        # Pre-allocated identity for the batched Cayley solve. (E_c + E_f
        # + 2) skew-symmetric matrices share one fp32 LU+TRSM call.
        self.register_buffer(
            "_eye_r",
            torch.eye(r, dtype=torch.float32),
            persistent=False,
        )

        self._register_routing_buffers(K_c, K_f)

    @staticmethod
    def _cayley(S: torch.Tensor) -> torch.Tensor:
        """R = (I - A)(I + A)^{-1}, A = S - S^T. 2D or batched 3D.

        Kept for save-time SVD distillation in :meth:`distill_save_state_dict`;
        forward uses a batched solve over the cat'd skew stack.
        """
        A = S - S.transpose(-2, -1)
        r = A.shape[-1]
        eye = torch.eye(r, device=A.device, dtype=A.dtype)
        if A.dim() == 3:
            eye = eye.unsqueeze(0).expand_as(A)
        return torch.linalg.solve(eye + A, eye - A)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        work = self.P_bases_c.dtype  # bf16

        # One batched (2 + K_c + K_f, r, r) Cayley solve covers both A's
        # and both B-pools' rotations. Single LU+TRSM kernel launch.
        skew = torch.cat(
            [
                self.S_q_c.unsqueeze(0),
                self.S_q_f.unsqueeze(0),
                self.S_p_c,
                self.S_p_f,
            ],
            dim=0,
        )
        A = skew - skew.transpose(-2, -1)
        R = torch.linalg.solve(self._eye_r + A, self._eye_r - A)
        K_c = self.num_experts_content
        K_f = self.num_experts_freq
        R_q_c = R[0].to(work)
        R_q_f = R[1].to(work)
        R_p_c = R[2 : 2 + K_c].to(work)
        R_p_f = R[2 + K_c : 2 + K_c + K_f].to(work)

        Q_eff_c = R_q_c @ self.Q_basis_c  # (r, in)
        Q_eff_f = R_q_f @ self.Q_basis_f  # (r, in)

        # Single rank-cat down-projection for both pools. The two pools share
        # the same input ``x`` but have distinct ``Q_eff``; running them as
        # two separate matmuls makes backward materialize TWO ``(B, L, in)``
        # ``grad_x`` tensors that autograd then sums. On wide-input Linears
        # (``mlp.layer2``, in=8192) that doubled fp32 transient cost ~62 MiB
        # /module → ~1.7 GiB across 28 blocks. Concatenating ``Q_eff`` along
        # the rank axis computes ``grad_x`` ONCE; the split is a free view.
        # Bit-identical to the per-pool calls — see
        # ``test_chimera_down_proj_rank_cat_matches_separate``.
        r = self.lora_dim
        Q_eff_cat = torch.cat([Q_eff_c, Q_eff_f], dim=0)  # (2r, in)
        if self.use_custom_down_autograd and self.training:
            # ``ScaledLoRADownProjectFn`` folds ``inv_scale`` into ``Q_eff_cat``
            # at the fp32 matmul (both pools share the same per-input-channel
            # ``inv_scale``), so no rebalanced ``(B, L, in)`` bf16 activation is
            # materialized; saved-for-backward aliases the same ``x`` the
            # original Linear already pinned. See the allclose contract in
            # ``test_chimera_channel_scale_flag_on_matches_legacy_gradients``.
            inv = self.inv_scale if self._has_channel_scale else None
            lx_down_cat = lora_down_project(x, Q_eff_cat, inv).to(work)
        else:
            x_lora = self._rebalance(x.to(work))
            lx_down_cat = torch.nn.functional.linear(x_lora, Q_eff_cat)
        lx_c = lx_down_cat[..., :r]
        lx_f = lx_down_cat[..., r:]

        # Content gate: broadcast π_c from the network-level ContentRouter
        # (slot-assigned with grad_fn intact). Cache the RAW (uncentered)
        # simplex for the per-pool balance loss BEFORE recentering.
        pi_c = self._content_gate_raw(lx_c.shape[0])  # (B, K_c) fp32
        if self.training:
            # Plain STORE_ATTR — see HydraLoRAModule.forward for the rationale;
            # @compiler.disable would force a graph break and explode
            # saved-for-backward memory under torch.compile.
            self._last_gate = self._full_gate(pi_c)
        pi_c = self._center(pi_c, K_c)

        # λ application + T-LoRA mask (content only). Freq branch keeps
        # full rank at every t — by construction the freq pool's job is
        # coarse-stage / high-σ refinement which T-LoRA's argument says
        # WANTS the full rank (TimeStep Master-style asymmetric mixture).
        lx_c = lx_c * self.lambda_c.to(work) * self._timestep_mask.to(work)
        lx_f = lx_f * self.lambda_f.to(work)

        if self.dropout is not None and self.training:
            lx_c = torch.nn.functional.dropout(lx_c, p=self.dropout)
            lx_f = torch.nn.functional.dropout(lx_f, p=self.dropout)

        lx_c, scale_c = self._apply_rank_dropout(lx_c)
        lx_f, scale_f = self._apply_rank_dropout(lx_f)

        # Per-pool gate-weighted P_combined; one bmm per pool over the
        # B/L axis. Cast π at the einsum boundary so bf16 × fp32 doesn't
        # promote P_combined back to fp32 (would inflate saved activation).
        P_eff_c = self.P_bases_c @ R_p_c  # (K_c, out, r)
        P_eff_f = self.P_bases_f @ R_p_f  # (K_f, out, r)

        pi_c_w = pi_c.to(work)
        pi_f_w = self._center(self._freq_gate_raw(lx_c.shape[0]), K_f).to(work)

        P_combined_c = torch.einsum("bc,cor->bor", pi_c_w, P_eff_c)
        P_combined_f = torch.einsum("bf,for->bor", pi_f_w, P_eff_f)

        out = self._combine_up(lx_c, lx_f, P_combined_c, P_combined_f, scale_c)

        return org_forwarded + (out * self.multiplier).to(org_forwarded.dtype)

    def regularization(self):
        """No-op: Cayley guarantees orthogonality structurally on both A's."""
        zero = torch.tensor(0.0, device=self.S_p_c.device)
        return zero, zero

    # ------------------------------------------------------------------
    # Save-pipeline hooks: dual-A distill + per-pool MoE writer.
    # ------------------------------------------------------------------

    @classmethod
    def distill_save_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        dtype: Optional[torch.dtype],
    ) -> None:
        """Chimera training-form → free-form per-pool (Cayley → lora_{down,up}).

        Mutates ``state_dict`` in place. Discriminator: co-located
        ``.S_q_c`` + ``.S_q_f`` keys (chimera is the only variant with
        per-pool ``_c`` / ``_f`` suffixes — never collides with the other
        ortho converters). Runs FIRST in the save pipeline so subsequent
        converters see a chimera-free state_dict.

        Per pool, distill the Cayley-rotated SVD layout into free-form
        (``.lora_down_{c,f}.weight``, ``.lora_up_{c,f}_weight``). The MoE
        writer in :meth:`build_moe_state_dict` then expands the stacked
        per-pool ups into per-expert ``.lora_ups_{c,f}.{i}.weight`` keys
        and per-component q/k/v splits.
        """
        prefixes = set()
        for key in list(state_dict.keys()):
            if not key.endswith(".S_q_c"):
                continue
            prefix = key[: -len(".S_q_c")]
            if state_dict.get(f"{prefix}.S_q_f") is None:
                continue
            prefixes.add(prefix)

        for prefix in prefixes:
            S_q_c = state_dict[f"{prefix}.S_q_c"]
            S_q_f = state_dict[f"{prefix}.S_q_f"]
            S_p_c = state_dict[f"{prefix}.S_p_c"]  # (K_c, r, r)
            S_p_f = state_dict[f"{prefix}.S_p_f"]  # (K_f, r, r)
            Q_basis_c = state_dict[f"{prefix}.Q_basis_c"]
            Q_basis_f = state_dict[f"{prefix}.Q_basis_f"]
            P_bases_c = state_dict[f"{prefix}.P_bases_c"]  # (K_c, out, r)
            P_bases_f = state_dict[f"{prefix}.P_bases_f"]  # (K_f, out, r)
            lam_c = state_dict[f"{prefix}.lambda_c"]
            lam_f = state_dict[f"{prefix}.lambda_f"]
            alpha = state_dict.get(f"{prefix}.alpha")
            save_dtype = dtype if dtype is not None else P_bases_c.dtype

            R_q_c = cls._cayley(S_q_c.float())
            R_q_f = cls._cayley(S_q_f.float())
            R_p_c = cls._cayley(S_p_c.float())
            R_p_f = cls._cayley(S_p_f.float())
            Q_eff_c = R_q_c @ Q_basis_c.float()  # (r, in)
            Q_eff_f = R_q_f @ Q_basis_f.float()
            P_eff_c = P_bases_c.float() @ R_p_c  # (K_c, out, r)
            P_eff_f = P_bases_f.float() @ R_p_f

            def _split(P_eff, Q_eff, lam):
                lam_1d = lam.squeeze(0).float()
                lam_sqrt = lam_1d.abs().sqrt()
                lam_sign = lam_1d.sign()
                lora_down = (
                    (Q_eff * lam_sqrt.unsqueeze(1)).to(save_dtype).cpu().contiguous()
                )
                lora_up_weight = (
                    (P_eff * (lam_sqrt * lam_sign).unsqueeze(0).unsqueeze(0))
                    .to(save_dtype)
                    .cpu()
                    .contiguous()
                )
                return lora_down, lora_up_weight

            lora_down_c, lora_up_c_weight = _split(P_eff_c, Q_eff_c, lam_c)
            lora_down_f, lora_up_f_weight = _split(P_eff_f, Q_eff_f, lam_f)

            for suffix in (
                "S_q_c",
                "S_q_f",
                "S_p_c",
                "S_p_f",
                "Q_basis_c",
                "Q_basis_f",
                "P_bases_c",
                "P_bases_f",
                "lambda_c",
                "lambda_f",
            ):
                state_dict.pop(f"{prefix}.{suffix}", None)

            state_dict[f"{prefix}.lora_down_c.weight"] = lora_down_c
            state_dict[f"{prefix}.lora_up_c_weight"] = lora_up_c_weight
            state_dict[f"{prefix}.lora_down_f.weight"] = lora_down_f
            state_dict[f"{prefix}.lora_up_f_weight"] = lora_up_f_weight
            if alpha is not None:
                state_dict[f"{prefix}.alpha"] = alpha

    @staticmethod
    def build_moe_state_dict(
        state_dict: Dict[str, torch.Tensor],
        dtype: Optional[torch.dtype],
    ) -> Dict[str, torch.Tensor]:
        """Build the ``*_chimera.safetensors`` payload.

        Expects :meth:`distill_save_state_dict` to have already run.

        Two transforms:
          1. Expand stacked ``.lora_up_c_weight (K_c, out, r)`` →
             per-expert ``.lora_ups_c.{i}.weight``; same for ``_f``.
          2. Per-pool fused-qkv defuse on attention prefixes. Both pools
             share the prefix (chimera = one module per Linear), so when
             the prefix ends in a fused frag we split BOTH pools'
             (lora_down + ups stack) per component. ``alpha`` / ``inv_scale``
             clone into each split component.

        Top-level ``content_router.*`` / ``freq_router.*`` keys pass through
        untouched (they don't carry a ``lora_unet_*`` prefix and don't match
        any fused frag suffix).

        After the per-pool split, the remaining fused-qkv prefixes are
        the OrthoLoRA fallbacks for attention projections excluded
        from ``router_targets`` (already distilled to plain LoRA by
        :meth:`OrthoLoRAModule.distill_save_state_dict`). Run them
        through the shared :func:`defuse_standard_qkv` so they emerge
        in the split q/k/v layout that ComfyUI's cosmos backbone
        expects — otherwise they surface as ``lora key not loaded``
        warnings at load time.
        """
        sd: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            v = v.detach().clone().to("cpu")
            if k.endswith(".lora_up_c_weight"):
                prefix = k.removesuffix(".lora_up_c_weight")
                for i in range(v.size(0)):
                    sd[f"{prefix}.lora_ups_c.{i}.weight"] = v[i]
            elif k.endswith(".lora_up_f_weight"):
                prefix = k.removesuffix(".lora_up_f_weight")
                for j in range(v.size(0)):
                    sd[f"{prefix}.lora_ups_f.{j}.weight"] = v[j]
            else:
                sd[k] = v

        # Per-pool q/k/v split. Detect by either pool's down key (both
        # should be present per chimera prefix; iterating one set is
        # sufficient).
        fused_groups: List[tuple] = []
        for key in list(sd.keys()):
            if not key.endswith(".lora_down_c.weight"):
                continue
            prefix = key.removesuffix(".lora_down_c.weight")
            spec = match_fused_spec(prefix)
            if spec is not None:
                fused_groups.append((prefix, spec))

        for prefix, spec in fused_groups:
            suffixes = spec.component_letters
            n = len(suffixes)
            down_c = sd.pop(f"{prefix}.lora_down_c.weight")
            down_f = sd.pop(f"{prefix}.lora_down_f.weight")
            alpha = sd.pop(f"{prefix}.alpha", None)
            inv_scale = sd.pop(f"{prefix}.inv_scale", None)

            ups_c_keys = sorted(
                (
                    k
                    for k in list(sd.keys())
                    if k.startswith(f"{prefix}.lora_ups_c.") and k.endswith(".weight")
                ),
                key=lambda k: int(
                    k.removeprefix(f"{prefix}.lora_ups_c.").removesuffix(".weight")
                ),
            )
            ups_f_keys = sorted(
                (
                    k
                    for k in list(sd.keys())
                    if k.startswith(f"{prefix}.lora_ups_f.") and k.endswith(".weight")
                ),
                key=lambda k: int(
                    k.removeprefix(f"{prefix}.lora_ups_f.").removesuffix(".weight")
                ),
            )
            ups_c = [sd.pop(k) for k in ups_c_keys]
            ups_f = [sd.pop(k) for k in ups_f_keys]
            ups_c_chunked = [u.chunk(n, dim=0) for u in ups_c]
            ups_f_chunked = [u.chunk(n, dim=0) for u in ups_f]

            base_prefix = prefix.removesuffix(spec.fused_frag)
            for ci, letter in enumerate(suffixes):
                new_prefix = base_prefix + spec.component_frag(letter)
                sd[f"{new_prefix}.lora_down_c.weight"] = down_c.clone()
                sd[f"{new_prefix}.lora_down_f.weight"] = down_f.clone()
                for ei, u_chunks in enumerate(ups_c_chunked):
                    sd[f"{new_prefix}.lora_ups_c.{ei}.weight"] = (
                        u_chunks[ci].contiguous().clone()
                    )
                for ei, u_chunks in enumerate(ups_f_chunked):
                    sd[f"{new_prefix}.lora_ups_f.{ei}.weight"] = (
                        u_chunks[ci].contiguous().clone()
                    )
                if alpha is not None:
                    sd[f"{new_prefix}.alpha"] = alpha.clone()
                if inv_scale is not None:
                    sd[f"{new_prefix}.inv_scale"] = inv_scale.clone()

        # Plain-LoRA leg defuse on any remaining fused attention prefixes.
        defuse_standard_qkv(sd)

        if dtype is not None:
            sd = {k: v.to(dtype) for k, v in sd.items()}
        return sd


class ChimeraHydraInferenceModule(_ChimeraRoutingMixin, BaseLoRAModule):
    """Free-form inference form of ChimeraHydra, loaded from a distilled
    ``*_chimera.safetensors``.

    Mirrors the training class's per-Linear shape but with explicit
    per-pool (lora_down, stacked lora_up) instead of Cayley-rotated SVD
    bases — produced by :meth:`ChimeraHydraLoRAModule.distill_save_state_dict`
    at save time.

    Buffer / parameter inventory:
      * ``lora_down_c.weight`` (r, in)        — content A
      * ``lora_up_c_weight``  (K_c, out, r)   — content B stack
      * ``lora_down_f.weight`` (r, in)        — freq A
      * ``lora_up_f_weight``  (K_f, out, r)   — freq B stack
      * ``_content_routing_weights`` (1, K_c) buffer — slot-written by the
        network-level ContentRouter
      * ``_freq_routing_weights`` (1, K_f) buffer    — slot-written by the
        network-level FreqRouter

    Both pools' gates are recentered to ``π − 1/K`` before the combine
    (λ is folded symmetrically into the saved ups, so the centered combine
    reproduces the trained forward exactly). No T-LoRA mask is applied at
    inference (consistent with all other LoRA-family inference modules — see
    ``[[project_tlora_inference_full_rank]]``).
    """

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
        num_experts_content: int = 3,
        num_experts_freq: int = 3,
        channel_scale=None,
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

        K_c = int(num_experts_content)
        K_f = int(num_experts_freq)
        r = int(lora_dim)
        in_dim = org_module.in_features
        out_dim = org_module.out_features

        self.num_experts_content = K_c
        self.num_experts_freq = K_f
        self.num_experts = K_c + K_f
        self.in_dim = in_dim

        # Free-form down-projections (one per pool). Initialized empty;
        # actual weights overwritten by load_state_dict.
        self.lora_down_c = torch.nn.Linear(in_dim, r, bias=False)
        self.lora_down_f = torch.nn.Linear(in_dim, r, bias=False)
        # Stacked B's, fused (K_*, out, r). Loader expands per-expert
        # ``.lora_ups_*.{i}.weight`` into these stacks before calling
        # load_state_dict — see factory.create_network_from_weights.
        self.lora_up_c_weight = torch.nn.Parameter(torch.zeros(K_c, out_dim, r))
        self.lora_up_f_weight = torch.nn.Parameter(torch.zeros(K_f, out_dim, r))

        if channel_scale is not None:
            self._register_channel_scale(self.lora_down_c.weight.data, channel_scale)
            _absorb_channel_scale(self.lora_down_f.weight.data, channel_scale)

        self._register_routing_buffers(K_c, K_f)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        x_lora = self._rebalance(x)
        x_f32 = x_lora.float()
        lx_c = torch.nn.functional.linear(x_f32, self.lora_down_c.weight.float())
        lx_f = torch.nn.functional.linear(x_f32, self.lora_down_f.weight.float())

        # Centered-gate parity with training: subtract per-pool 1/K. λ is
        # already baked into the saved ups, so this reproduces the trained
        # ``Σ (π[k] - 1/K)·B[k]`` combine exactly.
        pi_c = self._center(
            self._content_gate_raw(lx_c.shape[0]), self.num_experts_content
        )
        pi_f = self._center(
            self._freq_gate_raw(lx_c.shape[0]).float(), self.num_experts_freq
        )

        if self.dropout is not None and self.training:
            lx_c = torch.nn.functional.dropout(lx_c, p=self.dropout)
            lx_f = torch.nn.functional.dropout(lx_f, p=self.dropout)
        lx_c, scale_c = self._apply_rank_dropout(lx_c)
        lx_f, scale_f = self._apply_rank_dropout(lx_f)

        # Gate-weighted up projection per pool.
        comb_c = torch.einsum(
            "bc,cor->bor", pi_c.float(), self.lora_up_c_weight.float()
        )
        comb_f = torch.einsum(
            "bf,for->bor", pi_f.float(), self.lora_up_f_weight.float()
        )

        out = self._combine_up(lx_c, lx_f, comb_c, comb_f, scale_c)

        return org_forwarded + (out * self.multiplier).to(org_forwarded.dtype)
