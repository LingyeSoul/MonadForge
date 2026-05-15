# ChimeraHydra: dual-pool additive MoE with TWO Cayley A's per Linear.
#
# Two independent HydraLoRAs (Tian et al., NeurIPS'24; arXiv:2404.19245) glued
# at the residual — that's the chimera. The content half routes K_c B-heads
# off pooled rank-R text features (HydraLoRA's per-layer router on lx). The
# frequency half routes K_f B-heads off the network-level FreqRouter fed FEI
# of z_t (FeRA, arXiv:2511.17979). T-LoRA's rank mask (Liu et al.;
# TimeStep Master, arXiv:2503.07416) modulates the content half only — the
# freq half stays full-rank at every t, giving an asymmetric "core expert
# always on" / "context expert rank-modulated" split inspired by TimeStep
# Master's asymmetric mixture.
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
#   This is strictly stronger than the prior 1-A chimera, which gave only
#   output-side orthogonality (B-pool subspaces) while sharing one A.

import logging

import torch

from networks.lora_modules.base import BaseLoRAModule, _absorb_channel_scale

logger = logging.getLogger(__name__)


class ChimeraHydraLoRAExpModule(BaseLoRAModule):
    """ChimeraHydra training-time module: two Cayley A's, two B-pools,
    one per-Linear content router, one shared freq buffer.

    Concretely two HydraLoRAs in parallel — the content half is the
    HydraLoRA paper's "1 A → many Bs + per-layer router on lx", the freq
    half is the same shape but routed by the network-level FreqRouter
    (built in ``LoRANetwork``) reading FEI(z_t). T-LoRA's rank mask is
    folded into the content branch's effective P only — the freq branch
    keeps full rank at every t (TimeStep Master-style asymmetric pool).

    The shared SVD of the base weight gives both pools their bases:
    distinct singular-vector slices on each side ⇒ A_c.row_space ⊥
    A_f.row_space and B_c[*].col_space ⊥ B_f[*].col_space, structurally,
    at step 0. Cayley rotates each within its assigned subspace.

    Save distills Cayley → free-form per pool (see
    ``networks/lora_save.py::_convert_chimera_dual_a_to_hydra``); load
    rebuilds a ``ChimeraHydraInferenceModule`` rather than re-instantiating
    this class.
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
            P_bases_c_init = (
                P_shared.unsqueeze(0).expand(K_c, -1, -1).contiguous()
            )
            P_bases_f_init = (
                P_shared.unsqueeze(0).expand(K_f, -1, -1).contiguous()
            )
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

        # Per-pool λ: ΔW=0 at step 0 (zero-init) and the two pools have
        # independent magnitudes through training (no shared scaling).
        self.lambda_c = torch.nn.Parameter(torch.zeros(1, r))
        self.lambda_f = torch.nn.Parameter(torch.zeros(1, r))

        # Per-Linear content router: pooled rank-R lx_c → K_c. The freq
        # router lives at the network level (one FreqRouter shared across
        # all chimera Linears) and writes π_f via the slot-assigned
        # ``_freq_routing_weights`` buffer below.
        self.router = torch.nn.Linear(r, K_c, bias=True)
        with torch.no_grad():
            torch.nn.init.normal_(self.router.weight, std=0.01)
            self.router.bias.zero_()

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

        # Custom autograd path is the OrthoLoRA-style "store bf16 lx"
        # trick; both pools' down-projects are independent so it composes,
        # but the parent OrthoHydra had use_custom_down_autograd=False as
        # default and chimera follows suit (kept off until a bench
        # confirms the activation-budget win is real for dual-A).
        self.use_custom_down_autograd = False

        # Pre-allocated identity for the batched Cayley solve. (E_c + E_f
        # + 2) skew-symmetric matrices share one fp32 LU+TRSM call.
        self.register_buffer(
            "_eye_r",
            torch.eye(r, dtype=torch.float32),
            persistent=False,
        )

        # Freq pool's gate buffer. Uniform 1/K_f placeholder; the
        # network-level FreqRouter overwrites via direct slot assignment
        # in ``set_freq_routing_weights`` (NO .detach(), NO .copy_() —
        # grad_fn must survive so ∂L/∂π_f reaches the FreqRouter
        # parameters). Non-persistent — re-derived on construction.
        placeholder = torch.full(
            (1, K_f), 1.0 / max(K_f, 1), dtype=torch.float32
        )
        self.register_buffer("_freq_routing_weights", placeholder, persistent=False)

        # Cached gate (B, K_c+K_f) for the per-pool balance loss.
        # _last_gate is read by ``LoRANetwork._get_chimera_balance_loss``
        # which slices at K_c into independent Switch losses per pool.
        self._last_gate = None

    @staticmethod
    def _cayley(S: torch.Tensor) -> torch.Tensor:
        """R = (I - A)(I + A)^{-1}, A = S - S^T. 2D or batched 3D.

        Kept for save-time SVD distillation in ``networks/lora_save.py``;
        forward uses a batched solve over the cat'd skew stack.
        """
        A = S - S.transpose(-2, -1)
        r = A.shape[-1]
        eye = torch.eye(r, device=A.device, dtype=A.dtype)
        if A.dim() == 3:
            eye = eye.unsqueeze(0).expand_as(A)
        return torch.linalg.solve(eye + A, eye - A)

    def set_freq_routing_weights(self, weights: torch.Tensor) -> None:
        """Slot-assign the freq pool's gates (preserves grad_fn).

        Direct slot assignment (NO .detach(), NO .copy_()) — the buffer
        must carry the FreqRouter's grad_fn so ∂L_denoise/∂π_f reaches
        FreqRouter parameters. Mirrors ``router_state._set_routing_weights``
        and ``HydraLoRAModule.set_freq_routing_weights``.
        """
        buf = self._freq_routing_weights
        w = weights.to(dtype=buf.dtype, device=buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        self._freq_routing_weights = w

    def clear_freq_routing_weights(self) -> None:
        """Reset to uniform 1/K_f without rebinding the pointer."""
        K_f = int(self._freq_routing_weights.shape[-1])
        self._freq_routing_weights.fill_(1.0 / max(K_f, 1))

    def _compute_content_gate(self, lx_c: torch.Tensor) -> torch.Tensor:
        """RMS-pool lx_c over the sequence axis (matches HydraLoRA), then
        per-Linear router → softmax → π_c (B, K_c).

        Pooling on lx_c (NOT lx_f or x): the content router's job is to
        partition CONTENT B-heads, so the load-bearing signal is the
        content-side latent. Pooling lx_f would cross-couple the two
        pools and defeat the chimera's input-separation argument.
        """
        if lx_c.dim() >= 3:
            B = lx_c.shape[0]
            pooled = lx_c.reshape(B, -1, lx_c.shape[-1]).pow(2).mean(dim=1).sqrt()
        else:
            pooled = lx_c
        pooled = pooled.to(self.router.weight.dtype)
        logits = self.router(pooled)  # (B, K_c)
        return torch.softmax(logits, dim=-1)

    def _full_gate(self, pi_c: torch.Tensor) -> torch.Tensor:
        """Construct the (B, K_c+K_f) gate cached in ``_last_gate`` for
        the per-pool balance loss. ``LoRANetwork._get_chimera_balance_loss``
        slices at ``num_experts_content`` to get the two halves.
        """
        pi_f = self._freq_routing_weights
        if pi_f.dim() == 1:
            pi_f = pi_f.unsqueeze(0)
        pi_f = pi_f.to(pi_c.dtype).expand(pi_c.shape[0], -1)
        return torch.cat([pi_c, pi_f], dim=-1)

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

        x_lora = self._rebalance(x.to(work))
        lx_c = torch.nn.functional.linear(x_lora, Q_eff_c)
        lx_f = torch.nn.functional.linear(x_lora, Q_eff_f)

        # Content router pools lx_c (pre-λ; zero-init λ would zero the
        # router input at step 0 and freeze the router gradient).
        pi_c = self._compute_content_gate(lx_c)  # (B, K_c) fp32
        if self.training:
            # Plain STORE_ATTR — see HydraLoRAModule.forward for the
            # rationale; @compiler.disable would force a graph break and
            # explode saved-for-backward memory under compile_mode=full.
            self._last_gate = self._full_gate(pi_c)

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
        pi_f = self._freq_routing_weights
        if pi_f.dim() == 1:
            pi_f = pi_f.unsqueeze(0)
        pi_f_w = pi_f.to(work).expand(pi_c_w.shape[0], -1)

        P_combined_c = torch.einsum("bc,cor->bor", pi_c_w, P_eff_c)
        P_combined_f = torch.einsum("bf,for->bor", pi_f_w, P_eff_f)

        orig_shape = lx_c.shape
        B = orig_shape[0]
        lx_c_3d = lx_c.reshape(B, -1, orig_shape[-1])
        lx_f_3d = lx_f.reshape(B, -1, orig_shape[-1])
        out_c = torch.bmm(lx_c_3d, P_combined_c.transpose(1, 2))
        out_f = torch.bmm(lx_f_3d, P_combined_f.transpose(1, 2))
        out = (out_c * scale_c + out_f * scale_f).reshape(*orig_shape[:-1], -1)

        return org_forwarded + (out * self.multiplier).to(org_forwarded.dtype)

    def regularization(self):
        """No-op: Cayley guarantees orthogonality structurally on both A's."""
        zero = torch.tensor(0.0, device=self.S_p_c.device)
        return zero, zero


class ChimeraHydraInferenceModule(BaseLoRAModule):
    """Free-form inference form of ChimeraHydra, loaded from a distilled
    ``*_chimera.safetensors``.

    Mirrors the training class's per-Linear shape but with explicit
    per-pool (lora_down, stacked lora_up) instead of Cayley-rotated SVD
    bases — produced by ``_convert_chimera_dual_a_to_hydra`` at save time.

    Buffer / parameter inventory:
      * ``lora_down_c.weight`` (r, in)        — content A
      * ``lora_up_c_weight``  (K_c, out, r)  — content B stack
      * ``router.weight``      (K_c, r)       — content router
      * ``router.bias``        (K_c,)
      * ``lora_down_f.weight`` (r, in)        — freq A
      * ``lora_up_f_weight``  (K_f, out, r)  — freq B stack
      * ``_freq_routing_weights`` (1, K_f) buffer  — slot-written by
        the network-level FreqRouter

    No T-LoRA mask is applied at inference (consistent with all other
    LoRA-family inference modules — see
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
        self.lora_up_c_weight = torch.nn.Parameter(
            torch.zeros(K_c, out_dim, r)
        )
        self.lora_up_f_weight = torch.nn.Parameter(
            torch.zeros(K_f, out_dim, r)
        )
        # Content router: identical shape to HydraLoRAModule's K_c-narrowed
        # router (see hydra.py for the chimera-load contract).
        self.router = torch.nn.Linear(r, K_c, bias=True)

        if channel_scale is not None:
            self._register_channel_scale(self.lora_down_c.weight.data, channel_scale)
            _absorb_channel_scale(self.lora_down_f.weight.data, channel_scale)

        placeholder = torch.full(
            (1, K_f), 1.0 / max(K_f, 1), dtype=torch.float32
        )
        self.register_buffer("_freq_routing_weights", placeholder, persistent=False)
        self._last_gate = None

    def set_freq_routing_weights(self, weights: torch.Tensor) -> None:
        """Slot-assign the freq pool's gates. Same protocol as the
        training class — see that docstring."""
        buf = self._freq_routing_weights
        w = weights.to(dtype=buf.dtype, device=buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        self._freq_routing_weights = w

    def clear_freq_routing_weights(self) -> None:
        K_f = int(self._freq_routing_weights.shape[-1])
        self._freq_routing_weights.fill_(1.0 / max(K_f, 1))

    def _compute_content_gate(self, lx_c: torch.Tensor) -> torch.Tensor:
        if lx_c.dim() >= 3:
            B = lx_c.shape[0]
            pooled = lx_c.reshape(B, -1, lx_c.shape[-1]).pow(2).mean(dim=1).sqrt()
        else:
            pooled = lx_c
        pooled = pooled.to(self.router.weight.dtype)
        return torch.softmax(self.router(pooled), dim=-1)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        x_lora = self._rebalance(x)
        lx_c = torch.nn.functional.linear(
            x_lora.float(), self.lora_down_c.weight.float()
        )
        lx_f = torch.nn.functional.linear(
            x_lora.float(), self.lora_down_f.weight.float()
        )

        pi_c = self._compute_content_gate(lx_c)  # (B, K_c)
        pi_f = self._freq_routing_weights
        if pi_f.dim() == 1:
            pi_f = pi_f.unsqueeze(0)
        pi_f = pi_f.to(pi_c.dtype).expand(pi_c.shape[0], -1)

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

        orig_shape = lx_c.shape
        B = orig_shape[0]
        lx_c_3d = lx_c.reshape(B, -1, orig_shape[-1])
        lx_f_3d = lx_f.reshape(B, -1, orig_shape[-1])
        out_c = torch.bmm(lx_c_3d, comb_c.transpose(1, 2))
        out_f = torch.bmm(lx_f_3d, comb_f.transpose(1, 2))
        out = (out_c * scale_c + out_f * scale_f).reshape(*orig_shape[:-1], -1)

        return org_forwarded + (out * self.multiplier).to(org_forwarded.dtype)
