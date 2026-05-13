# HydraLoRA: MoE-style multi-head LoRA with layer-local routing.

import math
from typing import List, Optional

import torch

from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.custom_autograd import lora_down_project


# freqs only depend on (half_dim, device) and never change once built; caching
# keeps `_sigma_sinusoidal_features` from emitting a fresh `arange` + `exp`
# kernel on every call (one per module per step under set_sigma).
_FREQS_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _sigma_sinusoidal_features(
    sigma: torch.Tensor, sigma_feature_dim: int
) -> torch.Tensor:
    """Sinusoidal σ features matching the DiT t_embedder functional form.

    Shared helper (also used by postfix-sigma, inlined there for historical
    self-containedness). Kept here so HydraLoRAModule / OrthoHydraLoRAExpModule
    can reuse the identical spectrum without cross-module coupling.
    """
    t = sigma.flatten().float()
    half_dim = sigma_feature_dim // 2
    key = (half_dim, t.device)
    freqs = _FREQS_CACHE.get(key)
    if freqs is None:
        exponent = (
            -math.log(10000)
            * torch.arange(half_dim, dtype=torch.float32, device=t.device)
            / max(half_dim, 1)
        )
        freqs = torch.exp(exponent)
        _FREQS_CACHE[key] = freqs
    angles = t[:, None] * freqs[None, :]  # [B, half_dim]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


def _register_sigma_feature_cache(
    module: torch.nn.Module, sigma_feature_dim: int
) -> None:
    """Register pointer-stable sigma buffers for router conditioning."""
    module.register_buffer(
        "_sigma", torch.zeros(1, dtype=torch.float32), persistent=False
    )
    if sigma_feature_dim <= 0:
        return
    zero_feat = _sigma_sinusoidal_features(module._sigma, sigma_feature_dim)
    module.register_buffer("_sigma_features", zero_feat, persistent=False)


def _copy_or_rebind_buffer(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    buf = getattr(module, name)
    if buf.shape == value.shape and buf.device == value.device:
        buf.copy_(value.to(buf.dtype))
    else:
        setattr(module, name, value.to(buf.dtype).clone())


def _set_sigma_feature_cache(
    module: torch.nn.Module,
    sigmas: torch.Tensor,
    sigma_features: torch.Tensor | None = None,
) -> None:
    """Update per-module sigma state without changing buffer pointers per step."""
    sigmas = sigmas.detach()
    _copy_or_rebind_buffer(module, "_sigma", sigmas)
    if getattr(module, "sigma_feature_dim", 0) <= 0:
        return
    if sigma_features is None:
        sigma_features = _sigma_sinusoidal_features(sigmas, module.sigma_feature_dim)
    _copy_or_rebind_buffer(module, "_sigma_features", sigma_features.detach())


def _clear_sigma_feature_cache(module: torch.nn.Module) -> None:
    module._sigma.zero_()
    if getattr(module, "sigma_feature_dim", 0) > 0:
        zero_feat = _sigma_sinusoidal_features(module._sigma, module.sigma_feature_dim)
        _copy_or_rebind_buffer(module, "_sigma_features", zero_feat)


def _register_fei_feature_cache(
    module: torch.nn.Module, fei_feature_dim: int
) -> None:
    """Register pointer-stable FEI buffer for router conditioning.

    Always registers ``_fei`` (zero placeholder, shape ``(1, max(fei_feature_dim, 1))``)
    so the cat path in ``_compute_gate`` has a Tensor to read pre-set_fei,
    same pattern as ``_sigma`` / ``_sigma_features``. When
    ``fei_feature_dim == 0`` the placeholder is still registered (cheap, 1
    float) but unused — keeps ``Module._apply`` parity with the σ side.
    """
    width = max(int(fei_feature_dim), 1)
    module.register_buffer(
        "_fei", torch.zeros(1, width, dtype=torch.float32), persistent=False
    )


def _set_fei_feature_cache(module: torch.nn.Module, fei: torch.Tensor) -> None:
    """Update per-module FEI state without changing buffer pointers per step."""
    fei = fei.detach()
    _copy_or_rebind_buffer(module, "_fei", fei)


def _clear_fei_feature_cache(module: torch.nn.Module) -> None:
    module._fei.zero_()


def _register_sigma_band_partition(
    module: torch.nn.Module,
    num_experts: int,
    num_sigma_buckets: int,
    sigma_bucket_boundaries: Optional[List[float]] = None,
) -> None:
    """Register σ-partition buffers: ``_expert_band`` (E,) and ``_sigma_edges``
    (B-1,).

    Layout is **interleaved**: expert ``e`` belongs to band
    ``e mod num_sigma_buckets``. With sequential SVD slicing in OrthoHydra,
    interleaving gives every band a representative spread of singular slices
    instead of binding band 0 to the top slice and band B-1 to the bottom.

    ``sigma_bucket_boundaries`` is an optional length-(B+1) list of σ edges
    starting at 0.0 and ending at 1.0; the interior B-1 cuts are stored as
    a buffer for ``torch.bucketize``. When ``None``, defaults to uniform
    ``linspace(0, 1, B+1)`` (equivalent to the previous ``(σ * B).floor()``
    rule for typical σ ∈ [0, 1)). Caller validates length and bounds.

    At forward time, samples whose σ lands in band ``b`` only see experts
    with ``_expert_band == b`` (others masked to ``-inf`` before softmax).
    """
    band = torch.arange(num_experts, dtype=torch.long) % num_sigma_buckets
    module.register_buffer("_expert_band", band, persistent=False)
    if sigma_bucket_boundaries is None:
        edges = torch.linspace(0.0, 1.0, num_sigma_buckets + 1)
    else:
        edges = torch.tensor(list(sigma_bucket_boundaries), dtype=torch.float32)
    interior = edges[1:-1].contiguous()
    module.register_buffer("_sigma_edges", interior, persistent=False)
    module._sigma_num_buckets = int(num_sigma_buckets)


def _apply_sigma_band_mask(
    logits: torch.Tensor,
    sigma: torch.Tensor,
    expert_band: torch.Tensor,
    sigma_edges: torch.Tensor,
) -> torch.Tensor:
    """Mask out-of-band expert logits with -inf before softmax.

    ``logits``: (B, E). ``sigma``: (B,) per-sample σ ∈ [0, 1] (may broadcast
    from (1,) if ``set_sigma`` hasn't fired this forward — caller-side
    invariant). ``expert_band``: (E,) long, registered by
    ``_register_sigma_band_partition``. ``sigma_edges``: (B-1,) interior cuts
    consumed by ``torch.bucketize`` (right=False default → σ exactly at an
    edge maps to the upper bucket). Returns logits with out-of-band positions
    set to ``-inf`` so softmax produces 0 there and renormalises across the
    in-band experts.
    """
    num_buckets = int(sigma_edges.numel()) + 1
    bucket_ids = torch.bucketize(sigma.float(), sigma_edges).clamp(
        0, num_buckets - 1
    )
    if bucket_ids.shape[0] == 1 and logits.shape[0] > 1:
        bucket_ids = bucket_ids.expand(logits.shape[0])
    in_band = bucket_ids[:, None] == expert_band[None, :]  # (B, E) bool
    return logits.masked_fill(~in_band, float("-inf"))


class HydraLoRAModule(BaseLoRAModule):
    """
    HydraLoRA: MoE-style multi-head LoRA with layer-local routing.
    Shared lora_down captures common features; per-expert lora_up heads specialize.
    Each module owns its own router that reads the layer input and emits per-sample gates,
    so specialization is learned per-layer rather than globally.
    Reference: docs/methods/hydra-lora.md

    Optional σ-conditional routing (Track B, timestep-hydra.md): when
    ``sigma_feature_dim > 0``, ``sinusoidal(σ)`` is **concatenated** to the
    pooled rank-R router input, making ``self.router`` a
    ``Linear(r + sigma_feat, E)``. The σ-feature columns of the router
    weight are zero-init, so step-0 behavior is identical to the no-σ router
    and σ-dependence only emerges as those columns accumulate gradient.

    Why direct-input rather than the previous additive-bias ``sigma_mlp``: a
    bias-only σ path's gradient is ``dL/d logits · d_sigma_feat``, which
    vanishes whenever experts are undifferentiated (all ``score_e`` near
    equal → ``dL/d logit_e ≈ 0``). Feeding σ into the router's input avoids
    this chicken-and-egg problem — the σ columns train alongside the
    content columns on the same chain rule, so σ routing emerges as soon as
    the router learns anything at all.

    ``use_global_router`` (shared_A + ``route_per_layer=False`` path): the
    per-layer router is dropped and gates are read from a ``_routing_weights``
    buffer broadcast in from the network-level ``GlobalRouter``. The shared
    ``lora_down`` + per-expert ``lora_up`` layout is unchanged — only the
    gating source moves. σ-band partition is incompatible with this path
    (the partition rewrites local logits and there are no local logits to
    rewrite). Balance loss is silently inert in this mode: ``_last_gate``
    is the detached buffer broadcast, so ``get_balance_loss`` contributes a
    constant with no gradient signal back to the GlobalRouter.
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
        num_experts=4,
        channel_scale=None,
        sigma_feature_dim: int = 0,
        expert_init_std: float = 0.0,
        specialize_experts_by_sigma_buckets: bool = False,
        num_sigma_buckets: int = 1,
        sigma_bucket_boundaries: Optional[List[float]] = None,
        fei_feature_dim: int = 0,
        use_global_router: bool = False,
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

        self.num_experts = num_experts
        self.in_dim = in_dim

        # Shared down projection
        self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

        # Fused per-expert up projections: (num_experts, out_dim, lora_dim).
        # Zero-init keeps ΔW = 0 at step 0 (LoRA-safe). Per-expert symmetry is
        # broken by `expert_warmup_ratio` (random per-step expert-gradient
        # masking — see LoRANetwork.step_expert_warmup) for production runs.
        # `expert_init_std` is a paper-baseline knob (Tian et al. NeurIPS'24
        # original mitigation): a tiny Gaussian perturb on `lora_up_weight`
        # that gives the router distinct directions to latch onto at init.
        # Production training should leave it at 0.0.
        self.lora_up_weight = torch.nn.Parameter(
            torch.zeros(num_experts, out_dim, self.lora_dim)
        )
        if expert_init_std > 0.0:
            torch.nn.init.normal_(self.lora_up_weight, mean=0.0, std=expert_init_std)

        self.use_global_router = bool(use_global_router)
        # Local router: reads [pooled rank-R signal | sinusoidal(σ)] → per-sample
        # expert gates. Operating in rank-R space (not raw in_dim) is load-bearing:
        # raw DiT inputs have 80–96× DC-bias outlier channels and ~4096 tokens, so
        # mean-pooling raw inputs collapsed the signal to near-constant DC noise and
        # left the router with no trainable gradient (see docs/methods/hydra-lora.md
        # §Fixes). `lora_down` is trained jointly, so signal-carrying directions
        # accumulate here and there are no large outliers to saturate softmax in bf16.
        # Under ``use_global_router`` the per-layer router is skipped entirely;
        # gates arrive via the ``_routing_weights`` shared buffer instead.
        if self.use_global_router:
            self.sigma_feature_dim = 0
            self.fei_feature_dim = 0
        else:
            self.sigma_feature_dim = int(sigma_feature_dim)
            # FEI router input (FeRA-style content-aware routing). When
            # ``fei_feature_dim > 0``, the per-sample 2-band FEI ``[B, fei_dim]``
            # is concatenated to the router input alongside the rank-R signal
            # and (optional) sinusoidal(σ). All three sources can coexist; the
            # toggle blocks in ``configs/methods/lora.toml`` keep them in
            # opposition for clean A/B. Default ``fei_dim=2`` = the raw simplex
            # ``(e_low, e_high)`` from ``library.runtime.fei.compute_fei_2band``
            # (bench-validated 2-band collapse — see
            # ``[[project_fera_probe_2band_decision]]``).
            self.fei_feature_dim = int(fei_feature_dim)
            router_in_dim = (
                self.lora_dim + self.sigma_feature_dim + self.fei_feature_dim
            )
            self.router = torch.nn.Linear(router_in_dim, num_experts, bias=True)
            # Split init: small-std on the pooled rank-R columns, zeros on the
            # σ- and FEI-feature columns. Step 0 gate is then identical to the
            # σ=off, FEI=off router, and σ/FEI influence emerges only as those
            # columns train.
            with torch.no_grad():
                self.router.weight.zero_()
                torch.nn.init.normal_(
                    self.router.weight[:, : self.lora_dim], std=0.01
                )
                self.router.bias.zero_()

        self._register_channel_scale(self.lora_down.weight.data, channel_scale)

        # Opt-in: save bf16 x instead of retaining fp32 x_lora for backward.
        # Set externally by the network factory when use_custom_down_autograd
        # is enabled. Applies to the shared down projection; router + gate-
        # weighted up projection take the legacy path.
        self.use_custom_down_autograd = False

        self._last_gate = None  # (B, num_experts), cached each forward for balance loss
        # σ tensor; always a Tensor (never None) so the sinusoidal branch in
        # _compute_gate can run unconditionally without a None-vs-Tensor guard.
        # Registered as a non-persistent buffer so ``.to(device)`` moves the
        # placeholder along with the module — otherwise a pre-``set_sigma``
        # forward would fail with a CPU/GPU device mismatch in ``torch.cat``.
        # ``LoRANetwork.set_sigma`` rebinds it to the step's (B,) timesteps
        # before every forward, so the placeholder is only used if set_sigma
        # is somehow skipped.
        _register_sigma_feature_cache(self, self.sigma_feature_dim)
        # Same register-on-init rationale as ``_sigma`` — keep ``_fei`` a Tensor
        # at all times so ``_compute_gate``'s cat path needs no None-vs-Tensor
        # guard. ``set_fei`` rebinds it once the network's shared FEI buffer
        # is wired in (see ``LoRANetwork._wire_shared_fei_buffers``).
        _register_fei_feature_cache(self, self.fei_feature_dim)
        # Under ``use_global_router``, ``_routing_weights`` is the per-step
        # gate broadcast in by ``LoRANetwork.set_routing_weights`` (same
        # protocol as ``StackedExpertsLoRAModule``). Uniform ``1/E`` placeholder
        # so the forward branch runs unconditionally without a None guard.
        if self.use_global_router:
            self.register_buffer(
                "_routing_weights",
                torch.full(
                    (1, num_experts),
                    1.0 / max(int(num_experts), 1),
                    dtype=torch.float32,
                ),
                persistent=False,
            )
        # Hard σ-band expert partition (Track C). Independent of σ-feature
        # router — when on, the E experts are split into ``num_sigma_buckets``
        # bands of ``E // num_sigma_buckets`` each; out-of-band logits are
        # masked to -inf before softmax so a sample at σ in band b can only
        # route to the experts in that band. Soft routing still operates
        # within each band. Validated upstream (network constructor).
        # Incompatible with ``use_global_router`` — the partition rewrites
        # local logits and there are no local logits to rewrite.
        if specialize_experts_by_sigma_buckets and self.use_global_router:
            raise ValueError(
                "specialize_experts_by_sigma_buckets is incompatible with "
                "use_global_router=True (no per-layer logits to mask). Pick "
                "one: per-layer σ partition, or network-level GlobalRouter."
            )
        self._sigma_band_partition: bool = bool(specialize_experts_by_sigma_buckets)
        if self._sigma_band_partition:
            _register_sigma_band_partition(
                self, num_experts, num_sigma_buckets, sigma_bucket_boundaries
            )
        # Expert-warmup gradient masking. Single buffer holding the per-expert
        # grad-scale (1.0 = full gradient, 0.0 = stop-grad). Default all-ones
        # makes the forward branch a no-op (``up*1 + up.detach()*0 == up``),
        # so the branch is applied unconditionally — no Python-bool guard for
        # dynamo to recompile on. LoRANetwork.step_expert_warmup flips values
        # in-place; buffer mutations are tracked as dynamic by dynamo, no
        # recompile per step.
        self.register_buffer(
            "_expert_grad_mask",
            torch.ones(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def _compute_gate(self, lx: torch.Tensor) -> torch.Tensor:
        """Pool the rank-R `lora_down` output over the sequence dim, optionally
        concatenate sinusoidal(σ), run router, softmax.

        RMS (L2-norm) pool per channel: ``sqrt(mean(lx**2))``. Unlike mean pool,
        RMS does not cancel zero-mean activations by √N, so the pooled vector
        retains sample-level content over long sequences (L≈4096). Raw DiT inputs
        have DC-bias outliers that would break this aggregator in bf16, but
        rank-R space (post `lora_down`) is bounded by ``‖lora_down‖·‖x‖`` and
        has no such outliers, so RMS is safe here (see
        ``docs/methods/hydra-lora.md`` §Fixes).

        When ``sigma_feature_dim > 0``, ``sinusoidal(σ)`` is concatenated to
        the pooled vector. ``self._sigma`` always holds a tensor (zero
        placeholder at init, step σ after ``set_sigma``), so the branch on
        "σ set vs not" is gone — the router-input shape stays constant and
        there is no None-vs-Tensor guard to recompile on.

        Under ``use_global_router`` the gate is the broadcast
        ``_routing_weights`` buffer (``(1,E)`` placeholder → ``(B,E)`` after
        ``set_routing_weights``); ``lx`` is ignored, and pool/cat/router are
        all skipped.
        """
        if self.use_global_router:
            B = lx.shape[0] if lx.dim() >= 1 else 1
            w = self._routing_weights
            if w.dim() == 1:
                w = w.unsqueeze(0)
            return w.to(lx.dtype).expand(B, -1)
        if lx.dim() >= 3:
            B = lx.shape[0]
            pooled = lx.reshape(B, -1, lx.shape[-1]).pow(2).mean(dim=1).sqrt()
        else:
            pooled = lx
        # lx is fp32 (bottleneck policy) but router weights follow the adapter's
        # storage dtype (bf16 at inference) — align before matmul.
        pooled = pooled.to(self.router.weight.dtype)
        parts = [pooled]
        if self.sigma_feature_dim > 0:
            sigma_feat = self._sigma_features.to(pooled.dtype)
            # Broadcast placeholder (shape (1, D) before first set_sigma) to
            # batch size. Once set_sigma has run, _sigma matches pooled.shape[0]
            # and the expand is a no-op.
            sigma_feat = sigma_feat.expand(pooled.shape[0], -1)
            parts.append(sigma_feat)
        if self.fei_feature_dim > 0:
            # Same broadcast rule as σ: ``_fei`` placeholder is shape
            # ``(1, fei_dim)`` until ``set_fei`` rebinds it to ``(B, fei_dim)``.
            fei_feat = self._fei.to(pooled.dtype)
            fei_feat = fei_feat.expand(pooled.shape[0], -1)
            parts.append(fei_feat)
        router_in = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        logits = self.router(router_in)  # (B, num_experts)
        if self._sigma_band_partition:
            logits = _apply_sigma_band_mask(
                logits, self._sigma, self._expert_band, self._sigma_edges
            )
        return torch.softmax(logits, dim=-1)

    def set_sigma(
        self, sigmas: torch.Tensor, sigma_features: torch.Tensor | None = None
    ) -> None:
        _set_sigma_feature_cache(self, sigmas, sigma_features)

    def clear_sigma(self) -> None:
        _clear_sigma_feature_cache(self)

    def set_fei(self, fei: torch.Tensor) -> None:
        _set_fei_feature_cache(self, fei)

    def clear_fei(self) -> None:
        _clear_fei_feature_cache(self)

    def set_routing_weights(self, weights: torch.Tensor) -> None:
        # Direct buffer reassignment (no ``.detach()``, no in-place ``.copy_()``)
        # so the router's autograd ``grad_fn`` survives. See FeRA eq. 6-7, 11:
        # gates ``α_t`` appear as live multipliers in ``y_t = Σ α_{t,m} E_m(z_t)``,
        # so plain ``L_denoise`` backprop is the router's gradient path.
        if not getattr(self, "use_global_router", False):
            return
        buf = self._routing_weights
        w = weights.to(dtype=buf.dtype, device=buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        self._routing_weights = w

    def clear_routing_weights(self) -> None:
        if not getattr(self, "use_global_router", False):
            return
        E = int(self._routing_weights.shape[-1])
        self._routing_weights.fill_(1.0 / max(E, 1))

    def forward(self, x):
        # Policy: bf16 storage, fp32 for the bottleneck matmuls. See
        # LoRAModule.forward for rationale. Gate/router stays in autocast
        # dtype — softmax over num_experts is fine in bf16 with the
        # small-std router init.
        org_forwarded = self.org_forward(x)

        if not self.enabled:
            return org_forwarded

        if self._skip_module():
            return org_forwarded

        # per-channel input rebalancing (SmoothQuant-style, see LoRAModule.forward)
        if self.use_custom_down_autograd and self.training:
            inv_scale = self.inv_scale if self._has_channel_scale else None
            lx = lora_down_project(x, self.lora_down.weight, inv_scale)
        else:
            x_lora = self._rebalance(x)
            lx = torch.nn.functional.linear(
                x_lora.float(), self.lora_down.weight.float()
            )

        # Layer-local routing: gate is computed from the rank-R signal *before*
        # timestep masking / dropout — those are training-time perturbations and
        # the gate should behave identically at train and inference.
        gate = self._compute_gate(lx)  # (B, num_experts)
        if self.training:
            # Cache for network-level balance loss. Plain STORE_ATTR inline
            # (not a @torch.compiler.disable helper): under compile_mode=full
            # a disabled helper forces a graph break at every LoRA module's
            # forward, which splits the AOT-autograd compiled region into
            # many small segments and explodes saved-for-backward activation
            # memory (observed OOM at 56 MoE + 140 OrthoLoRAExp modules,
            # T4-class budget). The None↔Tensor guard on _last_gate is
            # prophylactic only — Phase A1 cleared the real recompile sources
            # and trial.log shows no _last_gate guard firing in practice.
            self._last_gate = gate

        # timestep-dependent rank masking (T-LoRA compatibility). Mask is a
        # per-module all-ones buffer by default (neutral); set_timestep_mask
        # reassigns each module to a shared live-updated mask when T-LoRA is
        # active. Always applied — no None-vs-Tensor guard to recompile on.
        if self.training:
            lx = lx * self._timestep_mask

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        # Expert-warmup masking: keep full MoE inference (all experts contribute
        # via the learned gate) but let gradient flow only into the randomly-
        # chosen expert's up-weight slice during warmup. Breaks the cold-start
        # deadlock where zero-init experts receive identical gradients under a
        # near-uniform router. Applied unconditionally — outside warmup the
        # mask is all-ones, so ``up*1 + up.detach()*0`` collapses to ``up``
        # (autograd-equivalent). No Python-bool guard means no dynamo recompile
        # at the warmup→post-warmup transition.
        up_weight = self.lora_up_weight
        expert_mask = self._expert_grad_mask.to(up_weight.dtype).view(-1, 1, 1)
        up_weight = (
            up_weight * expert_mask + up_weight.detach() * (1.0 - expert_mask)
        )

        # Gate-weighted combined weight per batch element: (B, out_dim, lora_dim)
        combined = torch.einsum(
            "be,eod->bod", gate.float(), up_weight.float()
        )
        # Apply: lx is (B, ..., lora_dim), combined is (B, out_dim, lora_dim)
        orig_shape = lx.shape
        B = orig_shape[0]
        lx_3d = lx.reshape(B, -1, orig_shape[-1])  # (B, *, lora_dim)
        out = torch.bmm(lx_3d, combined.transpose(1, 2))  # (B, *, out_dim)
        out = out.reshape(*orig_shape[:-1], -1)  # restore prefix dims

        return org_forwarded + (out * self.multiplier * scale).to(
            org_forwarded.dtype
        )
