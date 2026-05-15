# ChimeraHydra: dual-pool additive MoE on the OrthoHydra Cayley parameterization.
#
# See ``docs/proposal/chimera_hydra.md``. Two pools of B-heads share one A per
# adapted Linear — a **content** pool routed by the local rank-R router (input:
# pooled ``lx``), and a **frequency** pool routed by the network-level
# ``FreqRouter`` (input: FEI + sinusoidal-σ features). The full gate
# ``[π_c | π_f]`` is fed into the existing OrthoHydra einsum, so the
# additive composition ``Σ π_c · B_c (Ax) + Σ π_f · B_f (Ax)`` falls out of
# the same shared code path — pools are disjoint by name only.
#
# T-LoRA composition: ``use_timestep_mask`` applies the rank mask to the
# **content branch only** — the freq branch sees full rank at every t (high-σ
# steps are exactly where the freq pool wants coarse-stage capacity, while
# the content pool is the layout/identity memorization risk surface).
# Implemented with two bmm calls.

import torch

from networks.lora_modules.custom_autograd import lora_down_project
from networks.lora_modules.ortho import OrthoHydraLoRAExpModule


class ChimeraHydraLoRAExpModule(OrthoHydraLoRAExpModule):
    """OrthoHydra split into a content pool (``num_experts_content``) and a
    frequency pool (``num_experts_freq``). The local router still produces
    ``π_c`` over the content slice; the network-level ``FreqRouter`` writes
    ``π_f`` over the freq slice through a separate shared buffer
    ``_freq_routing_weights``.

    Total experts ``E = K_c + K_f`` keeps the OrthoHydra ``P_bases (E, out, r)``
    + ``S_p (E, r, r)`` layout untouched — only the gate is constructed
    from two disjoint sources. By construction the first ``K_c`` slices of
    the SVD column space (``V[:, :K_c·r]``) belong to the content pool and
    the next ``K_f`` slices to the freq pool (sequential disjoint slicing
    is OrthoHydra's existing behaviour at ``num_experts = E``).
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
        if num_experts_content <= 0 or num_experts_freq <= 0:
            raise ValueError(
                f"ChimeraHydra requires both pools to be non-empty: "
                f"K_c={num_experts_content}, K_f={num_experts_freq}"
            )

        # Build the parent OrthoHydra with E = K_c + K_f experts and the
        # local content router on rank-R only. The freq router owns the σ/FEI
        # axes by design (§"Why HydraLoRA's auto-specialization argument gets
        # stronger"), so we pass 0 for those feature dims here — even though
        # the network still wires per-sample σ for the FreqRouter's input.
        # σ-band partition is off (incompatible with the broadcast freq
        # router taking ownership of the σ axis).
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            num_experts=num_experts_content + num_experts_freq,
            channel_scale=channel_scale,
            sigma_feature_dim=0,
            fei_feature_dim=0,
            specialize_experts_by_sigma_buckets=False,
            num_sigma_buckets=1,
            sigma_bucket_boundaries=None,
            use_global_router=False,
        )
        self.num_experts_content = int(num_experts_content)
        self.num_experts_freq = int(num_experts_freq)

        # Re-narrow ``self.router`` from E → K_c so its output IS π_c
        # directly (no slicing in the hot path). The parent already
        # initialised a Linear(r, E); replace with the right-shaped one
        # and re-do the small-std init.
        in_features = self.router.in_features
        self.router = torch.nn.Linear(in_features, self.num_experts_content, bias=True)
        with torch.no_grad():
            torch.nn.init.normal_(self.router.weight, std=0.01)
            self.router.bias.zero_()

        # FreqRouter (network-level) broadcasts (B, K_f) into this buffer
        # via the same direct-slot-assignment protocol GlobalRouter uses
        # for FeRA (see ``router_state._set_routing_weights``). Uniform
        # 1/K_f placeholder; LoRANetwork.set_freq_routing_weights overwrites.
        placeholder = torch.full(
            (1, self.num_experts_freq),
            1.0 / max(self.num_experts_freq, 1),
            dtype=torch.float32,
        )
        self.register_buffer("_freq_routing_weights", placeholder, persistent=False)

    def _compute_gate(self, lx: torch.Tensor) -> torch.Tensor:
        """Construct ``gate = cat([π_c, π_f], dim=-1)`` over the full E pool.

        ``π_c`` is the per-layer router over pooled ``lx`` (rank-R only —
        σ/FEI deliberately excluded so content cannot become the time router).
        ``π_f`` is broadcast by the network-level FreqRouter through
        ``_freq_routing_weights``. The concatenated gate flows into the
        OrthoHydra einsum/bmm path; additive composition of the two pools
        is therefore identical math to single-pool routing with a partitioned
        gate vector.
        """
        # Pool rank-R input (RMS over the sequence axis — matches the
        # parent's policy, see hydra._compute_gate for the rationale).
        if lx.dim() >= 3:
            B = lx.shape[0]
            pooled = lx.reshape(B, -1, lx.shape[-1]).pow(2).mean(dim=1).sqrt()
        else:
            pooled = lx
        pooled = pooled.to(self.router.weight.dtype)
        logits_c = self.router(pooled)  # (B, K_c)
        pi_c = torch.softmax(logits_c, dim=-1)

        # π_f arrives pre-softmaxed from FreqRouter; broadcast to match B.
        pi_f = self._freq_routing_weights
        if pi_f.dim() == 1:
            pi_f = pi_f.unsqueeze(0)
        pi_f = pi_f.to(pi_c.dtype).expand(pi_c.shape[0], -1)

        return torch.cat([pi_c, pi_f], dim=-1)  # (B, K_c + K_f)

    def set_freq_routing_weights(self, weights: torch.Tensor) -> None:
        """Slot-assign the freq router's gates (preserves grad_fn).

        Direct slot assignment (NO .detach(), NO .copy_()) — the buffer
        must carry the FreqRouter's grad_fn so ∂L/∂π_f reaches the
        FreqRouter's parameters. Mirrors ``router_state._set_routing_weights``.
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

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        work = self.P_bases.dtype  # bf16 — bases + downstream activations

        # One batched (E+1, r, r) solve covers R_q + all R_p[e]. Cayley stays
        # fp32; boundary cast feeds R into the bf16 basis matmuls.
        skew = torch.cat([self.S_q.unsqueeze(0), self.S_p], dim=0)
        A = skew - skew.transpose(-2, -1)
        R = torch.linalg.solve(self._eye_r + A, self._eye_r - A)
        R_q = R[0].to(work)
        R_p = R[1:].to(work)
        Q_eff = R_q @ self.Q_basis  # bf16

        if self.use_custom_down_autograd and self.training:
            inv_scale = self.inv_scale if self._has_channel_scale else None
            # lora_down_project upcasts internally; bring lx back to work
            # dtype so saved-for-backward activation downstream is bf16.
            lx = lora_down_project(x, Q_eff, inv_scale).to(work)
        else:
            x_lora = self._rebalance(x.to(work))
            lx = torch.nn.functional.linear(x_lora, Q_eff)

        # Pool pre-λ (zero-init λ would zero the router input at step 0).
        gate = self._compute_gate(lx)  # (B, K_c + K_f) — fp32 from the router
        if self.training:
            # Plain STORE_ATTR — see HydraLoRAModule.forward. Balance loss
            # reads this and splits across the two pools. Keep native dtype.
            self._last_gate = gate

        # Apply λ once; T-LoRA mask is folded into ``P_combined_c`` below so
        # the freq pool keeps full rank at every t (rationale: docs/proposal/
        # chimera_hydra.md §T-LoRA integration). λ stays a fp32 Parameter
        # (Adam state precision); cast at the multiply.
        lx_scaled = lx * self.lambda_layer.to(work)

        if self.dropout is not None and self.training:
            lx_scaled = torch.nn.functional.dropout(lx_scaled, p=self.dropout)
        lx_scaled, scale = self._apply_rank_dropout(lx_scaled)

        P_eff = self.P_bases @ R_p  # (E, out, r) bf16
        K_c = self.num_experts_content

        # Cast gate at the einsum boundary so bf16 × fp32 doesn't promote
        # P_combined back to fp32 and inflate the saved activation.
        gate_w = gate.to(work)
        gate_c = gate_w[..., :K_c]  # (B, K_c)
        gate_f = gate_w[..., K_c:]  # (B, K_f)
        P_eff_c = P_eff[:K_c]
        P_eff_f = P_eff[K_c:]

        P_combined_c = torch.einsum("bc,cor->bor", gate_c, P_eff_c)
        P_combined_f = torch.einsum("bf,for->bor", gate_f, P_eff_f)

        # Fold the T-LoRA mask into the content branch's P_combined on the
        # rank axis (broadcast (1,1,r) over (B,out,r)) instead of allocating
        # a separate mask-scaled lx_c. Lets us add the two pools and issue
        # one bmm (saves a kernel + the duplicate (B,L,r) saved-for-backward
        # activation). Path stays shape-static under torch.compile because
        # the mask buffer is always-a-Tensor (see base.py).
        P_combined_c = P_combined_c * self._timestep_mask.to(work).view(1, 1, -1)
        P_combined = P_combined_c + P_combined_f

        orig_shape = lx_scaled.shape
        B = orig_shape[0]
        lx_3d = lx_scaled.reshape(B, -1, orig_shape[-1])
        out = torch.bmm(lx_3d, P_combined.transpose(1, 2)).reshape(
            *orig_shape[:-1], -1
        )

        lora_out = out * self.multiplier * scale
        return org_forwarded + lora_out.to(org_forwarded.dtype)
