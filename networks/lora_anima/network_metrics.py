# Metrics / diagnostics mixin for LoRANetwork.
#
# Pure read-side machinery — balance loss, router stats, up-weight grad-norm
# diagnostics, ortho regularization, and the log-step ``metrics`` aggregator.
# Split out of network.py to keep the assembly/runtime core readable; these
# methods only ever read instance state (``self.cfg``, ``self.unet_loras``,
# the router handles, the per-step caches) that network.py owns. Mixed into
# ``LoRANetwork`` so every method is still ``network.<method>()``.

import math
from typing import Dict, List, Optional, Union

import torch

from library.training.metrics import MetricContext


class _NetworkMetricsMixin:
    def step_balance_loss_warmup(self, global_step: int, max_train_steps: int) -> None:
        """Activate the MoE load-balance penalty once training crosses warmup.

        Step function: ``_balance_loss_weight`` holds at 0 during the first
        ``_balance_loss_warmup_ratio`` of steps, then flips to
        ``_balance_loss_target_weight`` — letting the router specialize before
        the penalty kicks in, then preventing single-expert collapse after.
        No-op unless both attrs are attached (hydra post_init) and ratio > 0.
        """
        target = float(getattr(self, "_balance_loss_target_weight", 0.0) or 0.0)
        ratio = float(getattr(self, "_balance_loss_warmup_ratio", 0.0) or 0.0)
        if ratio <= 0.0 or max_train_steps <= 0 or target <= 0.0:
            return
        warmup_steps = int(max_train_steps * ratio)
        self._balance_loss_weight = 0.0 if global_step < warmup_steps else target

    @staticmethod
    def _switch_balance(gate: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer balance: E · Σ_i frac_i · mean_gate_i. Scalar."""
        num_experts = gate.shape[-1]
        expert_idx = gate.argmax(dim=-1)  # (B,)
        frac = torch.zeros(num_experts, device=gate.device, dtype=gate.dtype)
        frac.scatter_add_(0, expert_idx, torch.ones_like(expert_idx, dtype=gate.dtype))
        frac = frac / gate.shape[0]
        gate_mean = gate.mean(dim=0)  # (num_experts,)
        return num_experts * (frac * gate_mean).sum()

    def get_balance_loss(self) -> torch.Tensor:
        """Switch-Transformer load-balancing loss averaged over HydraLoRA modules.

        Global term aggregates gates over the full batch. When σ-conditional
        routing is on, also adds a per-σ-bucket term so global balance can't
        mask per-bucket collapse. Chimera dispatches to the dual-pool variant
        (independent content/freq switch losses — a single combined term could
        flatten one pool while concentrating the other).
        """
        # Chimera: dual-pool branch. Computed entirely separately from the
        # legacy single-pool / σ-bucket aggregation since the weights and
        # accumulation are independent per pool.
        if getattr(self, "_use_chimera_hydra", False):
            return self._get_chimera_balance_loss()

        total = None
        per_bucket_total = None
        count = 0
        per_bucket_count = 0

        sigma = self._last_sigma  # (B,) or None
        num_buckets = self.cfg.num_sigma_buckets
        bucket_w = float(self.cfg.per_bucket_balance_weight or 0.0)
        want_per_bucket = (
            self.cfg.router_source == "sigma"
            and sigma is not None
            and num_buckets > 1
            and bucket_w > 0.0
        )
        if want_per_bucket:
            thresholds = torch.linspace(0.0, 1.0, num_buckets + 1, device=sigma.device)[
                1:-1
            ]
            bucket_ids = torch.bucketize(sigma.float(), thresholds)  # (B,) in [0, N)

        for lora in self.unet_loras + self.text_encoder_loras:
            gate = getattr(lora, "_last_gate", None)
            if gate is None:
                continue
            term = self._switch_balance(gate)
            total = term if total is None else total + term
            count += 1

            if want_per_bucket and getattr(lora, "sigma_feature_dim", 0) > 0:
                # Only penalize per-bucket collapse on modules that actually
                # have σ-conditional routing capacity to collapse.
                module_bucket_sum = None
                module_bucket_count = 0
                for b in range(num_buckets):
                    mask = bucket_ids == b
                    if int(mask.sum()) < 2:
                        # Not enough samples to meaningfully measure balance
                        # in this bucket on this step; skip.
                        continue
                    bterm = self._switch_balance(gate[mask])
                    module_bucket_sum = (
                        bterm
                        if module_bucket_sum is None
                        else module_bucket_sum + bterm
                    )
                    module_bucket_count += 1
                if module_bucket_sum is not None:
                    per_bucket_total = (
                        module_bucket_sum / module_bucket_count
                        if per_bucket_total is None
                        else per_bucket_total + module_bucket_sum / module_bucket_count
                    )
                    per_bucket_count += 1

        if total is None:
            return torch.tensor(0.0)
        out = total / count
        if per_bucket_total is not None and per_bucket_count > 0:
            out = out + bucket_w * (per_bucket_total / per_bucket_count)
        return out

    def _get_chimera_balance_loss(self) -> torch.Tensor:
        """Dual-pool switch loss for the chimera path.

        Each module's gate is ``(B, K_c + K_f)``: ``[:K_c]`` is π_c, the rest
        π_f. Compute Switch balance independently per slice, average across
        modules, combine with ``_balance_w_content`` / ``_balance_w_freq``.
        Warmup is asymmetric: the content pool rides ``_balance_loss_weight``
        (held at 0 during warmup); the freq pool bypasses warmup since the
        FreqRouter has its own symmetry-breaker (FEI input + init_std), so its
        balance pressure fires from step 0. Trainer consumes this directly.
        """
        K_c_default = int(getattr(self.cfg, "num_experts_content", 0))
        total_c = None
        total_f = None
        count_c = 0
        count_f = 0
        for lora in self.unet_loras + self.text_encoder_loras:
            gate = getattr(lora, "_last_gate", None)
            if gate is None:
                continue
            K_c = int(getattr(lora, "num_experts_content", K_c_default))
            if K_c <= 0:
                continue
            gate_c = gate[..., :K_c]
            gate_f = gate[..., K_c:]
            if gate_c.shape[-1] > 1:
                term_c = self._switch_balance(gate_c)
                total_c = term_c if total_c is None else total_c + term_c
                count_c += 1
            if gate_f.shape[-1] > 1:
                term_f = self._switch_balance(gate_f)
                total_f = term_f if total_f is None else total_f + term_f
                count_f += 1

        if total_c is None and total_f is None:
            return torch.tensor(0.0)
        w_c = float(getattr(self, "_balance_w_content", 0.0) or 0.0)
        w_f = float(getattr(self, "_balance_w_freq", 0.0) or 0.0)
        # Warmup gate applies to the CONTENT pool only; freq fires from step 0.
        outer = float(getattr(self, "_balance_loss_weight", 0.0) or 0.0)
        out = torch.tensor(0.0)
        if total_c is not None and count_c > 0:
            out = out + outer * w_c * (total_c / count_c)
        if total_f is not None and count_f > 0:
            out = out + w_f * (total_f / count_f)
        return out

    def get_router_entropy(self) -> Optional[float]:
        """Mean per-sample normalized entropy of hydra router gates, averaged
        across modules. None when no hydra module cached a gate this step. Thin
        wrapper over :meth:`get_router_stats` for the progress-bar postfix;
        prefer ``get_router_stats`` for logging. Chimera returns the mean of
        the per-pool entropies (each already normalized to [0, 1]).
        """
        if getattr(self, "_use_chimera_hydra", False):
            cstats = self.get_chimera_router_stats()
            if not cstats:
                return None
            parts = [
                cstats[k] for k in ("content_entropy", "freq_entropy") if k in cstats
            ]
            if not parts:
                return None
            return sum(parts) / len(parts)
        stats = self.get_router_stats()
        return stats.get("entropy_mean") if stats else None

    def get_router_stats(
        self,
    ) -> Dict[str, Union[float, List[float], List[List[float]], List[int]]]:
        """Per-step router diagnostics aggregated across hydra modules.

        Returns entropy (mean + p05/p50/p95, normalized to [0,1] over reachable
        support), top1-top2 margin, per-expert argmax usage, and — when σ was
        set and ``num_sigma_buckets > 1`` — per-σ-bucket usage + bucket counts.
        Empty dict when no hydra module cached a gate this step.

        Vectorized: gates with matching E are stacked into one ``(M, B, E)``
        tensor reduced in a single pass per metric (~10 launches regardless of
        module count vs ~500 for the per-module loop — see
        ``docs/optimizations/nsys_analysis_0503.md``). Memoized on
        ``_router_stats_cache``, invalidated by ``clear_step_caches``.
        """
        if self._router_stats_cache is not None:
            return self._router_stats_cache

        # Collect gates with matching expert count. Modules with mismatched E
        # are skipped (aggregating usage vectors of different length isn't
        # meaningful) — same policy as the previous per-module loop.
        gates: List[torch.Tensor] = []
        E_ref: Optional[int] = None
        for lora in self.unet_loras + self.text_encoder_loras:
            gate = getattr(lora, "_last_gate", None)
            if gate is None:
                continue
            E = gate.shape[-1]
            if E <= 1:
                continue
            if E_ref is None:
                E_ref = E
            elif E != E_ref:
                continue
            gates.append(gate)

        if not gates:
            return {}

        g = torch.stack(gates, dim=0)  # (M, B, E)
        M, B, E = g.shape

        sigma = self._last_sigma  # (B,) or None
        num_buckets = int(self.cfg.num_sigma_buckets)
        want_per_bucket = sigma is not None and num_buckets > 1
        # When ``specialize_experts_by_sigma_buckets`` is on, each sample can
        # only route to its band's ``E / num_buckets`` experts (others masked
        # to -inf pre-softmax). Normalizing entropy by ``log(E)`` then caps
        # the achievable max at ``log(experts_per_band) / log(E)`` (e.g. 0.44
        # for E=12, num_buckets=4) — making "uniform within band" look like
        # collapse. Normalize by the actual reachable support instead.
        band_partition_active = bool(
            self.cfg.specialize_experts_by_sigma_buckets and num_buckets > 1
        )
        effective_E = (E // num_buckets) if band_partition_active else E
        norm = math.log(effective_E) if effective_E > 1 else 1.0

        p = g.float().clamp_min(1e-12)
        # (M,) per-module mean entropy, normalized to [0, 1] over reachable support
        H_per_module = -(p * p.log()).sum(dim=-1).mean(dim=-1) / norm
        # (M, B, 2) top-2 in one batched topk → (M,) mean margin
        top2 = p.topk(2, dim=-1).values
        margin_per_module = (top2[..., 0] - top2[..., 1]).mean(dim=-1)
        # Argmax usage: one_hot + sum → (M, E) histograms in one pass
        expert_idx = g.argmax(dim=-1)  # (M, B)
        usage_per_module = torch.nn.functional.one_hot(expert_idx, num_classes=E).to(
            g.dtype
        ).sum(dim=1) / float(B)  # (M, E)

        H_per_module = H_per_module.detach()
        q_probs = torch.tensor(
            [0.05, 0.5, 0.95], device=H_per_module.device, dtype=H_per_module.dtype
        )
        q = torch.quantile(H_per_module, q_probs)  # (3,)
        # Single packed summary: [mean_H, p05, p50, p95, margin_mean]. One DtoH.
        summary = torch.stack(
            [H_per_module.mean(), q[0], q[1], q[2], margin_per_module.detach().mean()]
        ).cpu()
        usage_mean = usage_per_module.detach().mean(dim=0).cpu().tolist()
        out: Dict[str, Union[float, List[float], List[List[float]], List[int]]] = {
            "entropy_mean": float(summary[0]),
            "entropy_p05": float(summary[1]),
            "entropy_p50": float(summary[2]),
            "entropy_p95": float(summary[3]),
            "margin_mean": float(summary[4]),
            "expert_usage": usage_mean,
        }

        if want_per_bucket and sigma is not None:
            thresholds = torch.linspace(0.0, 1.0, num_buckets + 1, device=sigma.device)[
                1:-1
            ]
            bucket_ids = torch.bucketize(sigma.float(), thresholds).clamp(
                0, num_buckets - 1
            )  # (B,)
            bucket_counts_t = torch.zeros(
                num_buckets, device=sigma.device, dtype=torch.long
            )
            bucket_counts_t.scatter_add_(
                0, bucket_ids, torch.ones_like(bucket_ids, dtype=torch.long)
            )
            # Per-bucket argmax frequency, normalized within each bucket so
            # each row sums to ~1 (or 0 for empty buckets). Flat scatter_add
            # over (M, num_buckets * E) avoids a per-module loop.
            bucket_ids_dev = bucket_ids.to(expert_idx.device)
            flat_idx = bucket_ids_dev[None, :] * E + expert_idx  # (M, B)
            bu = torch.zeros(M, num_buckets * E, device=g.device, dtype=g.dtype)
            bu.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=g.dtype))
            bu = bu.view(M, num_buckets, E)
            bc = bucket_counts_t.to(g.dtype).clamp_min(1).view(1, num_buckets, 1)
            bucket_usage_mean = (bu / bc).detach().mean(dim=0).cpu().tolist()
            out["expert_usage_per_bucket"] = bucket_usage_mean
            out["bucket_counts"] = bucket_counts_t.cpu().tolist()

        self._router_stats_cache = out
        return out

    def get_chimera_router_stats(
        self,
    ) -> Dict[str, Union[float, List[float]]]:
        """Per-pool router diagnostics for chimera dual-pool routing.

        Chimera's ``_last_gate`` is ``cat([π_c, π_f])`` summing to 2, so a
        single argmax over the concat is doubly misleading. Report each pool
        independently with **mean gates** (see
        ``[[project_fera_expert_usage_mean_gates]]``): content π_c aggregated
        across modules; freq π_f read once from ``freq_router._last_gates``
        (broadcast identically to every module). Entropy is normalized per pool
        by ``log(K_pool)``. Empty dict on non-chimera nets / no cached gate.
        """
        if not getattr(self, "_use_chimera_hydra", False):
            return {}
        if self._chimera_router_stats_cache is not None:
            return self._chimera_router_stats_cache

        out: Dict[str, Union[float, List[float]]] = {}
        K_c_default = int(getattr(self.cfg, "num_experts_content", 0))

        # --- Content pool: aggregate π_c across modules ----------------------
        pi_c_list: List[torch.Tensor] = []
        K_c_ref: Optional[int] = None
        for lora in self.unet_loras + self.text_encoder_loras:
            gate = getattr(lora, "_last_gate", None)
            if gate is None:
                continue
            K_c = int(getattr(lora, "num_experts_content", K_c_default))
            if K_c <= 0:
                continue
            if K_c_ref is None:
                K_c_ref = K_c
            elif K_c != K_c_ref:
                continue
            pi_c_list.append(gate[..., :K_c])

        if pi_c_list and K_c_ref is not None and K_c_ref > 1:
            pi_c = torch.stack(pi_c_list, dim=0).float().clamp_min(1e-12)  # (M, B, K_c)
            norm_c = math.log(K_c_ref)
            H_c_per_mod = -(pi_c * pi_c.log()).sum(dim=-1).mean(dim=-1) / norm_c
            top2_c = pi_c.topk(2, dim=-1).values
            margin_c_per_mod = (top2_c[..., 0] - top2_c[..., 1]).mean(dim=-1)
            usage_c = pi_c.mean(dim=(0, 1))  # (K_c,)
            summary_c = torch.stack(
                [H_c_per_mod.mean().detach(), margin_c_per_mod.mean().detach()]
            ).cpu()
            out["content_entropy"] = float(summary_c[0])
            out["content_margin"] = float(summary_c[1])
            out["content_usage"] = usage_c.detach().cpu().tolist()

        # --- Freq pool: single broadcast tensor from FreqRouter --------------
        fr = getattr(self, "freq_router", None)
        pi_f = fr._last_gates if fr is not None else None
        if pi_f is not None and pi_f.dim() == 2 and pi_f.shape[-1] > 1:
            K_f = int(pi_f.shape[-1])
            pf = pi_f.float().clamp_min(1e-12)
            norm_f = math.log(K_f)
            H_f = (-(pf * pf.log()).sum(dim=-1).mean() / norm_f).detach()
            top2_f = pf.topk(2, dim=-1).values
            margin_f = (top2_f[..., 0] - top2_f[..., 1]).mean().detach()
            usage_f = pf.mean(dim=0).detach()
            summary_f = torch.stack([H_f, margin_f]).cpu()
            out["freq_entropy"] = float(summary_f[0])
            out["freq_margin"] = float(summary_f[1])
            out["freq_usage"] = usage_f.cpu().tolist()

        self._chimera_router_stats_cache = out
        return out

    def capture_up_grad_stats(self) -> None:
        """Snapshot per-expert grad-norm on Hydra up-weights (+ OrthoHydra S_p).

        Diagnoses the T-LoRA × σ-bucket interaction: a high-σ-band expert only
        fires where T-LoRA clamps rank to ``min_rank``, so rank columns
        ``[min_rank, R)`` of its ``lora_up`` accumulate near-zero grad (dead
        capacity). Splits the L2 norm at the ``min_rank`` boundary, also emits
        per-σ-band sums, OrthoHydra S_p totals + cross-expert coupling, and the
        per-Linear router grad. Sum-of-squares are stashed on-device (the D2H
        is deferred to ``get_up_grad_stats``); must run between
        ``accelerator.backward(loss)`` and ``optimizer.zero_grad``.
        """
        if not getattr(self, "_use_hydra", False):
            self._last_up_grad_stats = {}
            return

        use_tlora = bool(self.cfg.use_timestep_mask)
        min_rank = int(self.cfg.min_rank) if use_tlora else 0
        max_rank = int(self.cfg.lora_dim)
        # Clamp min_rank: a misconfig like min_rank > lora_dim would make the
        # "above" slice empty and the "below" slice full-rank, silently turning
        # the diagnostic into a no-op. Pin to [0, R].
        min_rank = max(0, min(min_rank, max_rank))
        has_tlora_split = use_tlora and 0 < min_rank < max_rank

        # Collect grads first; reduce in a few fused passes at the end (the
        # naive per-module loop stalled the post-backward boundary by hundreds
        # of ms on log steps — see docs/optimizations/nsys_analysis_0503.md).
        up_grads: List[torch.Tensor] = []  # each (E, out_i, R)
        sp_grads: List[torch.Tensor] = []  # each (E, r, r)
        expert_band_ref: Optional[torch.Tensor] = None
        # Per-layer router-weight grad sum-of-squares (centered-gate diagnostic:
        # confirms router logits get nonzero gradient at step 0). Skipped under
        # the network-level GlobalRouter — no per-Linear ``router`` then.
        router_grad_sq: Optional[torch.Tensor] = None

        for lora in self.unet_loras + self.text_encoder_loras:
            up = getattr(lora, "lora_up_weight", None)
            sp = getattr(lora, "S_p", None)
            up_grad = up.grad if isinstance(up, torch.nn.Parameter) else None
            sp_grad = sp.grad if isinstance(sp, torch.nn.Parameter) else None
            rtr = getattr(lora, "router", None)
            if isinstance(rtr, torch.nn.Linear) and rtr.weight.grad is not None:
                g2 = rtr.weight.grad.detach().float().square().sum()
                router_grad_sq = g2 if router_grad_sq is None else router_grad_sq + g2
            if up_grad is not None:
                up_grads.append(up_grad.detach())
            if sp_grad is not None and sp_grad.dim() == 3:
                # (E, r, r) — OrthoHydra rotation generator. No clean rank-col
                # split (Cayley couples all entries), so we report total only.
                # Plain OrthoLoRA's S_p is (r, r) with no expert axis — skipped
                # by the dim==3 check, since this diagnostic is per-expert.
                sp_grads.append(sp_grad.detach())
            if expert_band_ref is None:
                band = getattr(lora, "_expert_band", None)
                if band is not None:
                    expert_band_ref = band.detach()

        if not up_grads and not sp_grads:
            self._last_up_grad_stats = {}
            return

        total_per_exp: Optional[torch.Tensor] = None
        below_per_exp: Optional[torch.Tensor] = None
        above_per_exp: Optional[torch.Tensor] = None
        sp_total_per_exp: Optional[torch.Tensor] = None
        device_ref: Optional[torch.device] = None

        if up_grads:
            # All entries share E and R; only out_i varies. Cat along the out
            # axis into one (E, sum_out, R) tensor and reduce in one pass.
            big_up = torch.cat(up_grads, dim=1).float()
            sq_up = big_up.square()
            total_per_exp = sq_up.sum(dim=(1, 2))
            device_ref = total_per_exp.device
            if has_tlora_split:
                # Slices are views into ``sq_up``; sum along (out, rank-slice).
                below_per_exp = sq_up[:, :, :min_rank].sum(dim=(1, 2))
                above_per_exp = sq_up[:, :, min_rank:].sum(dim=(1, 2))

        sp_coupling: Optional[torch.Tensor] = None
        if sp_grads:
            # All entries share (E, r, r). Stack into (M, E, r, r) and reduce
            # over modules + r×r in one pass.
            big_sp = torch.stack(sp_grads, dim=0).float()
            sp_total_per_exp = big_sp.square().sum(dim=(0, 2, 3))
            if device_ref is None:
                device_ref = sp_total_per_exp.device

            # Cross-expert coupling of the rotation-generator gradients. The
            # P_e subspaces are Euclidean-orthogonal by construction (so the
            # structural ``ortho/subspace_overlap`` is a constant), but the
            # loss can still push two experts the same way — that shows up as
            # aligned ∂L/∂S_p[e]. Mean |cosine| over expert pairs and modules:
            # the loss-geometry coupling the gradient-Gram is after, read off
            # parameter grads (no activation hooks / no pre-forward arming).
            E_ = big_sp.shape[1]
            if E_ > 1:
                flat = big_sp.reshape(big_sp.shape[0], E_, -1)  # (M, E, r*r)
                flat = torch.nn.functional.normalize(flat, dim=-1, eps=1e-12)
                cos = torch.matmul(flat, flat.transpose(1, 2)).abs()  # (M, E, E)
                diag = cos.diagonal(dim1=1, dim2=2).sum(dim=1)  # (M,)
                off = (cos.sum(dim=(1, 2)) - diag) / (E_ * (E_ - 1))  # (M,)
                sp_coupling = off.mean().reshape(1)

        # Stash on-device tensors only — the D2H happens in
        # ``get_up_grad_stats`` so non-log steps avoid the
        # ``cudaStreamSynchronize`` that .cpu().tolist() forces.
        out: Dict[str, object] = {
            "min_rank": [float(min_rank)],
            "num_buckets": [float(self.cfg.num_sigma_buckets)],
        }
        if total_per_exp is not None:
            out["total"] = total_per_exp
        if below_per_exp is not None and above_per_exp is not None:
            out["below"] = below_per_exp
            out["above"] = above_per_exp
        if sp_total_per_exp is not None:
            out["sp_total"] = sp_total_per_exp
        if sp_coupling is not None:
            out["sp_coupling"] = sp_coupling
        if router_grad_sq is not None:
            out["router_grad_sq"] = router_grad_sq.reshape(1)

        # Per-band aggregation: scatter the per-expert sum-of-squares along
        # _expert_band. Only meaningful when σ-bucket partition is active —
        # otherwise the band assignment is undefined and per-band rows would
        # be misleading.
        if (
            expert_band_ref is not None
            and bool(self.cfg.specialize_experts_by_sigma_buckets)
            and int(self.cfg.num_sigma_buckets) > 1
        ):
            B = int(self.cfg.num_sigma_buckets)
            band = expert_band_ref.to(device_ref)

            def _scatter_to_band(per_exp: torch.Tensor) -> torch.Tensor:
                buf = torch.zeros(B, device=per_exp.device, dtype=per_exp.dtype)
                buf.scatter_add_(0, band, per_exp)
                return buf

            if total_per_exp is not None:
                out["total_band"] = _scatter_to_band(total_per_exp)
            if below_per_exp is not None and above_per_exp is not None:
                out["below_band"] = _scatter_to_band(below_per_exp)
                out["above_band"] = _scatter_to_band(above_per_exp)
            if sp_total_per_exp is not None:
                out["sp_total_band"] = _scatter_to_band(sp_total_per_exp)

        self._last_up_grad_stats = out

    def get_up_grad_stats(self) -> Dict[str, List[float]]:
        """Materialize the on-device stash from ``capture_up_grad_stats``.

        D2H is deferred to here so non-log steps don't pay the sync — the
        capture must run between backward and zero_grad (when ``.grad`` is
        live), but the metric only consumes the result on log steps.
        """
        raw = self._last_up_grad_stats
        if not raw:
            return {}
        materialized: Dict[str, List[float]] = {}
        for k, v in raw.items():
            if torch.is_tensor(v):
                materialized[k] = v.detach().cpu().tolist()
            else:
                materialized[k] = list(v)  # type: ignore[arg-type]
        return materialized

    def get_ortho_regularization(self) -> torch.Tensor:
        """Sum orthogonality regularization from all OrthoLoRA and ReFT modules."""
        total_reg = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "regularization"):
                p_reg, q_reg = lora.regularization()
                total_reg = total_reg + p_reg + q_reg
                count += 1
        for reft in self.text_encoder_refts + self.unet_refts:
            total_reg = total_reg + reft.regularization()
            count += 1
        return total_reg / max(count, 1)

    def metrics(self, ctx: MetricContext) -> dict[str, float]:
        """Emit log-step keys owned by the LoRA network.

        Covers ortho regularization, hydra balance loss, router stats, and
        hydra up-weight grad-norm diagnostics. Each block returns nothing
        if its driver is off (``_ortho_reg_weight == 0``, ``_use_hydra ==
        False``, etc.) so the cost on inactive paths is one attr check.
        """
        out: dict[str, float] = {}

        # Ortho regularization magnitude.
        ortho_w = float(getattr(self, "_ortho_reg_weight", 0.0) or 0.0)
        if ortho_w > 0.0:
            v = self.get_ortho_regularization()
            if torch.is_tensor(v):
                v = v.detach().item()
            out["reg/ortho"] = float(v)
            out["reg/ortho_weighted"] = float(ortho_w * v)

        # Hydra balance loss magnitude.
        bal_w = float(getattr(self, "_balance_loss_weight", 0.0) or 0.0)
        if bal_w > 0.0:
            v = self.get_balance_loss()
            if torch.is_tensor(v):
                v = v.detach().item()
            out["reg/balance"] = float(v)
            out["reg/balance_weighted"] = float(bal_w * v)

        if not getattr(self, "_use_hydra", False):
            return out

        # Router diagnostics. Chimera takes a different path because its
        # ``_last_gate`` is a concat of two independent softmaxes, so the
        # argmax-histogram aggregation under ``hydra/*`` is doubly misleading
        # (sums to 1 instead of 2; biased toward whichever pool has higher
        # init variance — see ``get_chimera_router_stats`` docstring).
        if getattr(self, "_use_chimera_hydra", False):
            cstats = self.get_chimera_router_stats()
            if cstats:
                if "content_entropy" in cstats:
                    out["chimera/content_entropy"] = float(cstats["content_entropy"])
                    out["chimera/content_margin"] = float(cstats["content_margin"])
                    for i, v in enumerate(cstats.get("content_usage", [])):
                        out[f"chimera/content_usage/{i}"] = float(v)
                if "freq_entropy" in cstats:
                    out["chimera/freq_entropy"] = float(cstats["freq_entropy"])
                    out["chimera/freq_margin"] = float(cstats["freq_margin"])
                    for i, v in enumerate(cstats.get("freq_usage", [])):
                        out[f"chimera/freq_usage/{i}"] = float(v)
        else:
            stats = self.get_router_stats()
            if stats:
                out["hydra/router_entropy"] = float(stats["entropy_mean"])
                out["hydra/router_entropy_p05"] = float(stats["entropy_p05"])
                out["hydra/router_entropy_p50"] = float(stats["entropy_p50"])
                out["hydra/router_entropy_p95"] = float(stats["entropy_p95"])
                out["hydra/router_margin"] = float(stats["margin_mean"])
                for i, v in enumerate(stats.get("expert_usage", [])):
                    out[f"hydra/expert_usage/{i}"] = float(v)
                for b, row in enumerate(stats.get("expert_usage_per_bucket", [])):
                    for i, v in enumerate(row):
                        out[f"hydra/expert_usage_b{b}/{i}"] = float(v)
                for b, c in enumerate(stats.get("bucket_counts", [])):
                    out[f"hydra/bucket_count/{b}"] = float(c)

        # Hydra up-weight grad norms by rank region and σ-band.
        up = self.get_up_grad_stats()
        if up:
            eps = 1e-12

            def _emit_per_expert(prefix: str, sq: list[float]) -> None:
                for i, v in enumerate(sq):
                    out[f"hydra/up_grad/{prefix}/exp{i}"] = float(v) ** 0.5

            def _emit_per_band(prefix: str, sq: list[float]) -> None:
                for b, v in enumerate(sq):
                    out[f"hydra/up_grad/{prefix}/band{b}"] = float(v) ** 0.5

            if "total" in up:
                _emit_per_expert("total", up["total"])
            if "below" in up and "above" in up:
                _emit_per_expert("below", up["below"])
                _emit_per_expert("above", up["above"])
                for i, (b_, a_) in enumerate(zip(up["below"], up["above"])):
                    out[f"hydra/up_grad/above_below_ratio/exp{i}"] = float(
                        a_
                    ) ** 0.5 / (float(b_) ** 0.5 + eps)
            if "sp_total" in up:
                _emit_per_expert("sp_total", up["sp_total"])
            if "total_band" in up:
                _emit_per_band("total", up["total_band"])
            if "below_band" in up and "above_band" in up:
                _emit_per_band("below", up["below_band"])
                _emit_per_band("above", up["above_band"])
                for b, (bv, av) in enumerate(zip(up["below_band"], up["above_band"])):
                    out[f"hydra/up_grad/above_below_ratio/band{b}"] = float(
                        av
                    ) ** 0.5 / (float(bv) ** 0.5 + eps)
            if "sp_total_band" in up:
                _emit_per_band("sp_total", up["sp_total_band"])
            # OrthoHydra centered-gate diagnostics.
            if up.get("router_grad_sq"):
                out["ortho/router_grad_norm"] = float(up["router_grad_sq"][0]) ** 0.5
            if up.get("sp_coupling"):
                out["ortho/grad_coupling"] = float(up["sp_coupling"][0])

        # Structural guardrail: cross-expert column-space overlap of the
        # OrthoHydra ``P_bases``. Constant by construction (~0 disjoint, ~1
        # narrow-layer fallback); a drift off this baseline flags a basis/init
        # regression. Cheap (cached per module) — see OrthoHydra.subspace_overlap.
        ov_vals = [
            lora.subspace_overlap()
            for lora in self.unet_loras + self.text_encoder_loras
            if hasattr(lora, "subspace_overlap")
        ]
        if ov_vals:
            out["ortho/subspace_overlap"] = float(sum(ov_vals) / len(ov_vals))

        # GlobalRouter stats — for stacked-experts + per-network routing
        # (plan2 §three-axis-config). Mirrors the per-Linear hydra keys but
        # under the ``fera/`` namespace so dashboards can compare across
        # variants. ``_last_gates`` is populated by ``GlobalRouter.forward``;
        # absent (None) outside of a step that fired the router.
        if (
            self.global_router is not None
            and self.global_router._last_gates is not None
        ):
            gates = self.global_router._last_gates  # (B, E) detached
            if gates.dim() == 2 and gates.shape[1] > 1:
                g = gates.float().clamp_min(1e-12)
                E = int(g.shape[-1])
                # Per-batch normalized entropy.
                norm = math.log(E)
                H = -(g * g.log()).sum(dim=-1).mean() / norm
                # Top1-Top2 margin (confidence).
                top2 = g.topk(2, dim=-1).values
                margin = (top2[..., 0] - top2[..., 1]).mean()
                # Per-expert mean gate weight. argmax-histogram breaks exact
                # ties to index 0 and misreports a uniform router as
                # "100% expert 0"; mean(gates) reflects the actual soft
                # distribution and still sums to 1.
                usage = g.mean(dim=0)
                summary = torch.stack([H.detach(), margin.detach()]).cpu()
                out["fera/router_entropy"] = float(summary[0])
                out["fera/router_margin"] = float(summary[1])
                for i, v in enumerate(usage.detach().cpu().tolist()):
                    out[f"fera/expert_usage/{i}"] = float(v)

        return out
