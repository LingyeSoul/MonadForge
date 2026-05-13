# StackedExpertsLoRAModule: independent-A multi-LoRA experts gated from a
# broadcast routing buffer.
#
# Distinct from HydraLoRAModule's shared-A (pooled lora_down) + per-expert
# lora_up design — each expert here owns its own (lora_down, lora_up) pair.
# The independent-A invariant is the defining trait of the FeRA paper
# (Yin et al., arXiv:2511.17979): every expert has a free input basis, so
# they can specialize on disjoint sub-features instead of competing inside
# a shared pooled subspace.
#
# Gates come from outside via the `_routing_weights` buffer; this module
# does NOT carry its own router. The router lives one level up — owned by
# ``LoRANetwork`` when ``cfg.route_per_layer=False`` (global router on
# FEI(z_t) or σ-features). The per-Linear router combination
# (``route_per_layer=True``) is reserved for a future extension layered on
# top of this base.
#
# Two parameterization modes:
#
#   * Free (``ortho=False``): ``lora_down_weight (E, r, in)`` Kaiming-init
#     per slice; ``lora_up_weight (E, out, r)`` zero-init. Standard
#     "ΔW=0 at step 0" LoRA convention.
#
#   * Ortho (``ortho=True``): PSOFT-style independent rotations within a
#     shared SVD basis. Frozen ``P_basis (out, r)`` and ``Q_basis (r, in)``
#     from the base Linear's top-r SVD; per-expert trainable skew seeds
#     ``S_q (E, r, r)``, ``S_p (E, r, r)`` Cayley-rotated; per-expert
#     diagonal ``lambda_layer (E, r)``. Each expert's effective ΔW is
#     ``P_basis @ cayley(S_p_e) @ diag(λ_e) @ cayley(S_q_e) @ Q_basis``.
#     Experts share the singular bundle (no disjoint slicing) — symmetry is
#     broken at the rotation level via small random init on ``S_q/S_p``
#     (size ``ortho_init_std``), so the global router has gradient signal
#     to differentiate experts from step 0.
#
# Activation-memory profile: stacked Parameters + two einsum boundaries
# save one ``(..., E, r)`` activation for backward instead of E full
# ``(..., out)`` activations from a per-expert loop. ~50× less per-Linear
# autograd memory than the naive loop at typical ``(E, r) = (3, 8)`` on
# Anima MLP shapes.

import math

import torch

from networks.lora_modules.base import BaseLoRAModule


# ─────────────────────────────────────────────────────────────────────────────
# Shared-buffer aliasing helpers (parallel _set_fei / _set_sigma protocol)
# ─────────────────────────────────────────────────────────────────────────────


def _register_routing_weights_buffer(
    module: torch.nn.Module, num_experts: int
) -> None:
    """Register a pointer-stable ``_routing_weights`` buffer.

    Initialised to uniform ``1/E`` so the forward gate-weighting branch can
    run unconditionally — no None-vs-Tensor guard fires under ``compile``.
    ``LoRANetwork.set_routing_weights`` rebinds it across every participating
    module via the same shared-buffer aliasing protocol used by ``set_fei``
    and ``set_sigma`` (see ``[[project_set_sigma_aliasing_bug]]``).
    """
    placeholder = torch.full(
        (1, num_experts),
        1.0 / max(int(num_experts), 1),
        dtype=torch.float32,
    )
    module.register_buffer("_routing_weights", placeholder, persistent=False)


def _copy_or_rebind_buffer(
    module: torch.nn.Module, name: str, value: torch.Tensor
) -> None:
    """Same-shape ``copy_`` (preserves pointer) or replace + reassign."""
    buf = getattr(module, name)
    if buf.shape == value.shape and buf.device == value.device:
        buf.copy_(value.to(buf.dtype))
    else:
        setattr(module, name, value.to(buf.dtype).clone())


def _set_routing_weights(
    module: torch.nn.Module, weights: torch.Tensor
) -> None:
    """Replace the ``_routing_weights`` buffer with the live router output.

    Direct slot reassignment (NOT in-place ``.copy_()``) and no ``.detach()`` —
    the buffer must carry the router's autograd ``grad_fn`` so ``∂L/∂α`` flows
    back to the ``GlobalRouter`` parameters. This is the gradient path the
    FeRA paper relies on (eq. 6-7, 11): ``α_t = softmax(g_φ(e_t)/τ)`` appears
    as a live multiplier in ``y_t = Σ_m α_{t,m} E_m(z_t)``, so plain
    ``L_denoise`` backprop trains the router with no special mechanism.
    """
    buf = module._routing_weights
    w = weights.to(dtype=buf.dtype, device=buf.device)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    module._routing_weights = w


def _clear_routing_weights(module: torch.nn.Module) -> None:
    """Reset gates to uniform ``1/E`` without rebinding the pointer."""
    E = int(module._routing_weights.shape[-1])
    module._routing_weights.fill_(1.0 / max(E, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────────────────


class StackedExpertsLoRAModule(BaseLoRAModule):
    """Independent-A multi-expert LoRA, gated from a broadcast buffer.

    Trainable parameters (free mode, ``ortho=False``)::

        lora_down_weight: (E, r, in)  — Kaiming per slice
        lora_up_weight:   (E, out, r) — zero-init

    Trainable parameters (ortho mode, ``ortho=True``)::

        S_q:           (E, r, r) — per-expert input-basis skew seed
        S_p:           (E, r, r) — per-expert output-basis skew seed
        lambda_layer:  (E, r)    — per-expert diagonal scale (zero-init)

    Frozen buffers (ortho mode)::

        P_basis: (out, r)   — top-r left singular vectors of the base weight
        Q_basis: (r, in)    — top-r right singular vectors (transposed)
        _eye_r:  (r, r)     — pre-allocated identity for the batched Cayley solve

    Shared buffers (both modes)::

        _routing_weights: (1, E) placeholder, rebound by
                          ``LoRANetwork.set_routing_weights`` to ``(B, E)``.

    Compose with T-LoRA via the inherited ``_timestep_mask`` buffer; the
    ``(1, r)`` mask broadcasts across the expert axis so every expert sees
    the same effective rank schedule per timestep. ``rank_dropout`` is not
    supported on the stacked layout in v1 — the base helper assumes a 2D/3D
    /4D ``lx`` shape and our forward produces 4D ``(B, L, E, r)``.
    """

    # 2D Linear bases only — Conv2d is unsupported by design (Anima's
    # adapted targets are projection Linears).
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
        num_experts: int = 3,
        channel_scale=None,
        ortho: bool = False,
        ortho_init_std: float = 0.02,
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

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = int(num_experts)
        self.ortho = bool(ortho)
        self.ortho_init_std = float(ortho_init_std)

        if self.ortho:
            # Top-r SVD of the base Linear; randomized lowrank is fast and
            # near-machine-precision on the kept slice (cf. OrthoLoRAExp).
            init_device = "cuda" if torch.cuda.is_available() else "cpu"
            W = org_module.weight.data.float().to(init_device)
            q = min(self.lora_dim + 6, min(W.shape))
            U, _S_vals, V = torch.svd_lowrank(W, q=q, niter=2)
            P_init = U[:, : self.lora_dim].clone().contiguous()  # (out, r)
            Q_init = V[:, : self.lora_dim].T.clone().contiguous()  # (r, in)
            del U, _S_vals, V, W

            self.register_buffer("P_basis", P_init.cpu())  # (out, r)
            self.register_buffer("Q_basis", Q_init.cpu())  # (r, in)

            # Random Kaiming-analog init on the skew seeds. With deterministic
            # SVD init and zero λ, zero-init S would leave every expert
            # bit-identical and the global router would have no gradient
            # signal to differentiate them. ``ortho_init_std`` controls how
            # far each expert starts from identity rotation.
            self.S_p = torch.nn.Parameter(
                torch.randn(self.num_experts, self.lora_dim, self.lora_dim)
                * self.ortho_init_std
            )
            self.S_q = torch.nn.Parameter(
                torch.randn(self.num_experts, self.lora_dim, self.lora_dim)
                * self.ortho_init_std
            )

            # Zero-init diagonal — ΔW = 0 at step 0 even though S is non-zero.
            self.lambda_layer = torch.nn.Parameter(
                torch.zeros(self.num_experts, self.lora_dim)
            )

            # Pre-allocated identity for the batched Cayley solve. Allocating
            # ``torch.eye`` per forward emits 2 tiny kernels per module per
            # step under compile.
            self.register_buffer(
                "_eye_r",
                torch.eye(self.lora_dim, dtype=torch.float32),
                persistent=False,
            )

            # SmoothQuant-style input rebalance absorbs into the frozen
            # ``Q_basis``. (out, r) ↦ P, (r, in) ↦ Q with scale on input cols.
            if channel_scale is not None:
                self._register_channel_scale(self.Q_basis, channel_scale)
        else:
            # Stacked-Parameter layout — one tensor per side, indexed by
            # expert on the leading axis. Saves an ``(..., E, r)``
            # activation for backward vs E full ``(..., out)`` from a
            # per-expert ``ModuleList`` loop. Mathematically equivalent to
            # ``Σ_e w_e · U_e @ D_e @ x``.
            self.lora_down_weight = torch.nn.Parameter(
                torch.empty(self.num_experts, self.lora_dim, in_dim)
            )
            self.lora_up_weight = torch.nn.Parameter(
                torch.zeros(self.num_experts, out_dim, self.lora_dim)
            )
            for k in range(self.num_experts):
                torch.nn.init.kaiming_uniform_(
                    self.lora_down_weight[k], a=math.sqrt(5)
                )

            # Absorb the same channel rebalance into each expert's down
            # slice — they share input space, so the absorption is uniform
            # across experts. ``_register_channel_scale`` registers
            # ``inv_scale`` once; repeat calls overwrite it with the same
            # value (idempotent) and mutate each slice in-place.
            if channel_scale is not None:
                for k in range(self.num_experts):
                    self._register_channel_scale(
                        self.lora_down_weight[k], channel_scale
                    )

        # Pointer-stable routing-weights buffer. ``LoRANetwork.set_routing_weights``
        # rebinds it once per step from the GlobalRouter output.
        _register_routing_weights_buffer(self, self.num_experts)

    # ------------------------------------------------------------------ API
    def set_routing_weights(self, weights: torch.Tensor) -> None:
        _set_routing_weights(self, weights)

    def clear_routing_weights(self) -> None:
        _clear_routing_weights(self)

    # ----------------------------------------------------- ortho-mode helper
    def _cayley_rotations(self):
        """Batched Cayley over both sides: ``R = (I + A)^{-1}(I - A)``,
        ``A = S - Sᵀ``.

        Stacks ``S_q`` and ``S_p`` into one ``(2E, r, r)`` solve so a single
        LU + TRSM launch covers every expert's both rotations at once —
        mirrors the ``OrthoLoRAExpModule`` 2×r×r trick, extended over the
        expert axis.

        Returns:
            R_q: ``(E, r, r)``
            R_p: ``(E, r, r)``
        """
        E = self.num_experts
        skew = torch.cat([self.S_q.float(), self.S_p.float()], dim=0)
        A = skew - skew.transpose(-2, -1)
        R = torch.linalg.solve(self._eye_r + A, self._eye_r - A)  # (2E, r, r)
        return R[:E], R[E:]

    # ----------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded
        if self._skip_module():
            return org_forwarded

        # ``_routing_weights`` is always a Tensor (uniform ``1/E`` placeholder
        # until ``set_routing_weights`` rebinds it). Shape ``(B, E)``;
        # broadcast to ``(B, 1, ..., 1, E, 1)`` to multiply into the
        # ``(B, ..., E, r)`` rank-level activations.
        w = self._routing_weights

        if self.ortho:
            compute_dtype = self.P_basis.dtype  # follow OrthoLoRA Exp convention
            x_lora = self._rebalance(x.to(compute_dtype))

            R_q, R_p = self._cayley_rotations()  # both (E, r, r), fp32
            R_q = R_q.to(compute_dtype)
            R_p = R_p.to(compute_dtype)

            # Down boundary: project x through frozen Q_basis once (shared
            # across experts). (..., in) → (..., r).
            x_proj = torch.nn.functional.linear(x_lora, self.Q_basis)

            # Per-expert R_q rotation in r-dim. (..., r) → (..., E, r).
            lx = torch.einsum("...j,eij->...ei", x_proj, R_q)

            # Per-expert diagonal scaling. ``lambda_layer (E, r)`` broadcasts
            # over the (..., ) prefix; lx[b, ..., e, r] *= λ[e, r].
            lx = lx * self.lambda_layer.to(compute_dtype)

            # T-LoRA mask: ``(1, r)`` broadcasts over ``(..., E, r)`` — applied
            # uniformly across experts (mask is rank-only, not expert-aware).
            lx = lx * self._timestep_mask

            # Standard dropout on the rank intermediate (shape-agnostic).
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            # Gate weighting. w (B, E) → (B, 1, ..., 1, E, 1) to broadcast
            # over the middle dims.
            B = w.shape[0]
            n_mid = lx.ndim - 3
            view_shape = (B,) + (1,) * n_mid + (self.num_experts, 1)
            lx = lx * w.view(view_shape).to(compute_dtype)

            # Per-expert R_p rotation + sum-over-experts in one einsum.
            # (E, r, r) · (..., E, r) → (..., r).
            mid = torch.einsum("ejr,...er->...j", R_p, lx)

            # Up boundary: project through frozen P_basis once (shared).
            adapter = torch.nn.functional.linear(mid, self.P_basis)
        else:
            # Bottleneck matmul in fp32 (bf16 storage policy — matches the
            # ``Hydra`` free-mode convention in ``networks/lora_modules/hydra.py``).
            x_lora = self._rebalance(x)

            # Single batched down projection over all experts:
            #   (..., in) @ (E, r, in)^T  →  (..., E, r)
            # Saves ONE (..., E, r) activation for backward instead of
            # E × (..., out) from a per-expert loop.
            lx = torch.einsum(
                "...i,eri->...er",
                x_lora.float(),
                self.lora_down_weight.float(),
            )

            # T-LoRA mask + dropout, same order as OrthoHydra.
            lx = lx * self._timestep_mask
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            # Gate weighting — same broadcast as the ortho path.
            B = w.shape[0]
            n_mid = lx.ndim - 3
            view_shape = (B,) + (1,) * n_mid + (self.num_experts, 1)
            lx = lx * w.view(view_shape).float()

            # Single batched up projection over all experts:
            #   (..., E, r) @ (E, out, r)^T  →  (..., out)
            adapter = torch.einsum(
                "...er,eor->...o", lx, self.lora_up_weight.float()
            )

        lora_out = adapter * self.multiplier * self.scale
        return org_forwarded + lora_out.to(org_forwarded.dtype)

    def regularization(self):
        """No-op: Cayley structurally guarantees orthogonality in ortho mode;
        free mode has no orthogonality constraint."""
        device = (
            self.S_p.device if self.ortho else self.lora_down_weight.device
        )
        zero = torch.tensor(0.0, device=device)
        return zero, zero
