"""APEX condition-space shifting.

Implements c_fake = A*c + b (Eq. 9 of the APEX paper, arXiv:2604.12322).
The shifted condition is fed into the same DiT under the fake branch to
provide an endogenous adversarial reference for the real branch — no
discriminator, no precomputed teacher, no architectural change.

Three parameterizations:
  scalar  : A = a*I, b = beta*1                (2 params)
  diag    : A = diag(a), b                     (2D params)
  full    : A in R^{DxD}, b                    (D^2 + D params)

Default init follows Table 7 of the paper: a = -1.0, b = 0.5 — a moderate
negative scaling that sits inside the stable region observed empirically.
Phase 0 on a 2D toy reproduced convergence to this neighborhood on its own
(a -> -1.08, b -> 0.62) so it is a safe starting point for Phase 1.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from library.datasets.base import LossRecorder
from library.training.method_adapter import (
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
)

logger = logging.getLogger(__name__)


class ConditionShift(nn.Module):
    MODES = ("scalar", "diag", "full")

    def __init__(
        self,
        dim: int,
        mode: str = "scalar",
        init_a: float = -1.0,
        init_b: float = 0.5,
        freeze_b: bool = False,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(
                f"ConditionShift mode must be one of {self.MODES}, got {mode!r}"
            )
        self.dim = int(dim)
        self.mode = mode

        if mode == "scalar":
            self.a = nn.Parameter(torch.tensor(float(init_a)))
            self.b = nn.Parameter(torch.tensor(float(init_b)))
        elif mode == "diag":
            self.a = nn.Parameter(torch.full((self.dim,), float(init_a)))
            self.b = nn.Parameter(torch.full((self.dim,), float(init_b)))
        else:  # full
            self.A = nn.Parameter(float(init_a) * torch.eye(self.dim))
            self.b = nn.Parameter(torch.full((self.dim,), float(init_b)))

        # b lives in the softmax-null subspace under RMSNorm cross-attn — gradient
        # on b is large (broadcast across tokens) but mostly invisible end-to-end,
        # so under sign-flip init (init_b=0) the optimizer drains capacity into b
        # while a barely moves. See docs/experimental/apex-0506.md.
        if freeze_b:
            self.b.requires_grad_(False)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Apply the shift to a cross-attention embedding tensor.

        Args:
            c: [B, S, D] text conditioning.

        Returns:
            [B, S, D] shifted conditioning c_fake, matching c.dtype.
        """
        dt = c.dtype
        if self.mode == "scalar":
            return self.a.to(dt) * c + self.b.to(dt)
        if self.mode == "diag":
            return c * self.a.to(dt).view(1, 1, -1) + self.b.to(dt).view(1, 1, -1)
        # full
        return c @ self.A.to(dt).t() + self.b.to(dt).view(1, 1, -1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, mode={self.mode}"


# ----------------------------------------------------------------- warm-start helper


def promote_warmstart_to_merge(args) -> None:
    """For APEX, rewire ``--network_weights`` from "trainable adapter" to
    "merge into DiT at load time, then train a fresh same-rank adapter."

    Keeping the warm-start LoRA as a runtime adapter via
    ``dim_from_weights`` costs ~1.5 GB more peak VRAM than baking it into
    the base — empirically OOMs a 16 GB card at rank 32 even though the
    bench's cold-start at rank 64 fits. Rewriting silently lets the user
    keep ``network_weights = "..."`` + ``dim_from_weights = true`` in the
    APEX config without manually picking a rank or a multiplier — both
    are inferred from the saved file's first ``lora_down`` (rank) and
    ``alpha`` (network_alpha). Multiplier defaults to 1.0.

    No-op outside APEX. Skips if ``--lora_path`` is already set (caller
    already chose merge), or if ``--network_weights`` isn't set
    (cold-start APEX).
    """
    method = getattr(args, "method", "") or ""
    if not (method == "apex" or method.startswith("apex_")):
        return
    if not getattr(args, "network_weights", None):
        return
    if getattr(args, "lora_path", None):
        return  # caller already wired the merge path explicitly

    path = args.network_weights
    if not str(path).endswith(".safetensors"):
        return  # rank inference below assumes safetensors layout

    from safetensors import safe_open

    rank: Optional[int] = None
    alpha: Optional[float] = None
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith(".lora_down.weight") and rank is None:
                rank = int(f.get_tensor(key).shape[0])
            elif key.endswith(".alpha") and alpha is None:
                alpha = float(f.get_tensor(key).item())
            if rank is not None and alpha is not None:
                break
    if rank is None:
        logger.warning(
            f"APEX warm-start auto-merge: no lora_down keys found in "
            f"{path}; leaving --network_weights path intact."
        )
        return
    if alpha is None:
        alpha = float(rank)

    args.lora_path = path
    if getattr(args, "lora_multiplier", None) is None:
        args.lora_multiplier = 1.0
    args.network_dim = rank
    args.network_alpha = alpha
    args.network_weights = None
    args.dim_from_weights = False
    logger.info(
        f"APEX warm-start auto-merge: baking {path} into DiT base "
        f"(multiplier={args.lora_multiplier}); fresh APEX adapter at "
        f"rank={rank}, alpha={alpha}."
    )


# ----------------------------------------------------------------- trainer integration


class ApexMethodAdapter(MethodAdapter):
    """APEX self-adversarial distillation (arXiv:2604.12322 §3) integration
    with the trainer adapter dispatch.

    Setup: enforce the warmup-or-warmstart precondition (cold-start regresses,
    proposal §7.3).
    Per step: increment the step counter consumed by the rampup schedule and
    surfaced to TensorBoard via ``metrics``.
    Extra forwards: run the two fake-branch DiT calls (eq. 11 + eq. 23) and
    one stop-gradient call for the L_mix target. The inner mixing coefficient
    used to construct ``T_mix_v`` is ramped 0 → ``apex_lambda`` over the
    warmup+rampup window — at ramp start ``T_mix = x`` so L_mix is pure FM,
    providing the cold-start bootstrap signal without a separate L_sup term.
    No-op outside training and on steps where the network has no
    ``apex_condition_shift`` attached.
    """

    name = "apex"

    def __init__(self) -> None:
        self.step = 0  # exposed via metrics() under "apex/lam_*"
        # Per-step state stashed by ``extra_forwards`` for the deferred fake-
        # branch forward in ``extra_forwards_fake``. Cleared at the start of
        # every step (whether or not the split path is used) so a stale
        # ``c_fake`` graph never leaks across iterations.
        self._fake_state: dict | None = None
        # Last computed schedule values, surfaced via metrics() so the
        # TB layer can plot warmup/rampup live. Updated in extra_forwards.
        self._last_lam_inner_eff: float = 0.0
        self._last_lam_f_eff: float = 0.0
        # Realized anchor fraction this step (per-element lam_inner=0 mask).
        # Differs from args.apex_anchor_ratio when B*ratio doesn't round
        # cleanly or the schedule is still in warmup (lam_inner_eff=0 → no
        # anchor needed). Surfaced via metrics so anchor_ratio's effect is
        # visible end-to-end.
        self._last_anchor_frac: float = 0.0
        # MSE(v_fake_sg, F_real) — key degeneracy detector. Near-zero means the
        # shifted condition produces the same DiT output as the unshifted one,
        # so L_mix collapses to the trivial self-consistency fixed point and
        # the adversarial signal vanishes. None until the first train step.
        self._last_v_fake_divergence: float | None = None
        # Fraction of Forward 2 t-shifted timesteps that hit the [0.02, 0.98]
        # clamp this step. Only meaningful when --apex_dt != 0; surfaced via
        # metrics() so the combined-3F lane can verify Δt = −0.05 stays well
        # below the upper-clamp threshold (the artifact that degenerated ~5%
        # of probe steps at +Δt). None for shipped APEX (apex_dt = 0).
        self._last_dt_clamp_frac: float | None = None
        # Cumulative-since-start moving averages for the noisy adapter scalars
        # — point-in-time loss_mix / loss_fake / v_fake_divergence are jittery
        # under typical batch sizes (per-step CV ~50%), and the cumulative avg
        # tracks trend much better. Same LossRecorder semantics as `loss/average`
        # so the TB curves are directly comparable. Recorders are also stashed
        # on the network in ``on_network_built`` so losses.py can update them
        # at the same site that writes ``_last_apex_{mix,fake}_value``.
        self._mix_recorder = LossRecorder()
        self._fake_recorder = LossRecorder()
        self._v_fake_div_recorder = LossRecorder()

    def on_network_built(self, ctx: SetupCtx) -> None:
        args = ctx.args
        has_warmup = (
            int(getattr(args, "apex_warmup_steps", 0) or 0) > 0
            or float(getattr(args, "apex_warmup_ratio", 0.0) or 0.0) > 0.0
        )
        # promote_warmstart_to_merge nulls args.network_weights and rewires the
        # warm-start to args.lora_path (baked into DiT). Either form counts.
        has_weights = (
            getattr(args, "network_weights", None) is not None
            or getattr(args, "lora_path", None) is not None
        )
        if not (has_warmup or has_weights):
            raise ValueError(
                "APEX training requires either --apex_warmup_ratio > 0 "
                "(or --apex_warmup_steps > 0) or --network_weights <path> "
                "(warm-start). Cold-start training is known to regress vs. "
                "plain FM on the one-step objective; see proposal.md §7.3."
            )
        # Temporal-2F precondition: t-shift is the only perturbation source in
        # this lane (no ConditionShift, no L_fake), so apex_dt == 0 reduces the
        # whole method to plain FM with one wasted extra forward. Refuse explicitly
        # rather than silently spending compute on an inert config.
        if bool(getattr(args, "apex_temporal_only", False)):
            apex_dt = float(getattr(args, "apex_dt", 0.0) or 0.0)
            if apex_dt == 0.0:
                raise ValueError(
                    "--apex_temporal_only requires --apex_dt != 0. With Δt=0, "
                    "Forward 2 is bit-identical to Forward 1 and v_fake_sg "
                    "collapses to F_real — L_mix degenerates to plain FM. "
                    "Recommended: --apex_dt -0.05. See docs/experimental/apex-0508.md."
                )
        # Expose loss-side recorders to ``library/training/losses.py`` so the
        # _apex_mix_loss / _apex_fake_loss sites can append every step.
        if ctx.network is not None:
            ctx.network._apex_mix_recorder = self._mix_recorder
            ctx.network._apex_fake_recorder = self._fake_recorder

    def on_step_start(self, ctx: StepCtx, batch, *, is_train: bool) -> None:
        # Counts at process_batch granularity; aligns with global_step under
        # gradient_accumulation_steps=1 and is close enough under larger
        # accumulation.
        if is_train:
            self.step += 1
        # Drop any stash from a previous step before this step's forwards run.
        # Without this, a step that skips the split path (validation, missing
        # crossattn_emb, network without ConditionShift) would inherit stale
        # tensors keyed for an earlier batch.
        self._fake_state = None

    def wants_split_backward(self, *, is_train: bool) -> bool:
        # APEX runs two grad-tracked DiT forwards per step that share no
        # autograd graph (forward 3's input is built from ``model_pred.detach()``).
        # Split-backward roughly halves peak activation memory; only useful at
        # train time — validation has no fake branch.
        return is_train

    def extra_forwards(self, ctx: StepCtx, primary: ForwardArtifacts):
        if not primary.is_train:
            return None
        network = ctx.network
        args = ctx.args
        # temporal-2F runs without ConditionShift; every other lane requires it.
        temporal_only = bool(getattr(args, "apex_temporal_only", False))
        if not temporal_only and getattr(network, "apex_condition_shift", None) is None:
            return None
        if primary.crossattn_emb is None:
            return None  # APEX needs the cross-attn embedding (shifted or not)

        anima = primary.anima_call
        noisy_model_input = primary.noisy_model_input  # 5D
        model_pred = primary.model_pred  # 5D
        timesteps = primary.timesteps
        crossattn_emb = primary.crossattn_emb
        padding_mask = primary.padding_mask
        kw = primary.forward_kwargs

        # Forward-2 condition. Shipped APEX / combined-3F shift c via the
        # learned affine; temporal-2F leaves c alone and relies on the t-shift
        # alone for the perturbation.
        if temporal_only:
            c_for_fake = crossattn_emb  # no ConditionShift
        else:
            # Shifted condition (grad flows into ConditionShift via L_fake —
            # the graph is preserved across the inline real-branch backward
            # because nothing in the real branch depends on c_fake's autograd
            # chain). Detached for the no_grad fake-target call below.
            c_fake = network.apex_condition_shift(crossattn_emb)
            c_for_fake = c_fake.detach()

        # (1) Fake-target forward — stop-gradient target for L_mix.
        # Shipped APEX (Δt=0, c=c_fake): paper §3.2 "v_fake := sg(F(x_t, t, c_fake))".
        # combined-3F (Δt≠0, c=c_fake): orthogonal-in-output-space perturbation
        # — c-shift and t-shift each carry distinct supervision (apex-0508.md).
        # temporal-2F (Δt≠0, c=c): t-shift only — drops L_fake's cold-start
        # poisoning risk at the cost of fake-trajectory self-distillation.
        # Clamp keeps t_eff in the sampler's training range; avoids the t=0.98
        # upper-edge artifact that degenerated ~5% of probe steps under +Δt.
        apex_dt = float(getattr(args, "apex_dt", 0.0) or 0.0)
        if apex_dt != 0.0:
            t_shifted = (timesteps.float() + apex_dt).clamp(0.02, 0.98)
            self._last_dt_clamp_frac = float(
                ((t_shifted == 0.02) | (t_shifted == 0.98)).float().mean().item()
            )
            t_eff = t_shifted.to(timesteps.dtype)
        else:
            self._last_dt_clamp_frac = None
            t_eff = timesteps
        with torch.no_grad():
            v_fake_sg = anima(
                noisy_model_input,
                t_eff,
                c_for_fake,
                padding_mask=padding_mask,
                **kw,
            )
            # Degeneracy detector for c-shift lanes; for temporal-2F this
            # reduces to MSE(F(x_t, t+Δt, c), F(x_t, t, c)) — the t-shift's
            # output-space magnitude (probe table 1 metric). Either way, near-
            # zero means the perturbation is invisible to the network.
            self._last_v_fake_divergence = float(
                (v_fake_sg - model_pred.detach()).pow(2).mean().item()
            )
            self._v_fake_div_recorder.add(
                epoch=0, step=self.step, loss=self._last_v_fake_divergence
            )

        # Build x_fake / x_fake_t / target_fake only for 3-forward lanes.
        # temporal-2F drops L_fake and Forward 3 entirely.
        if not temporal_only:
            # Endpoint predictor (Eq. 11): x_fake = x_t - t * F_real, sg.
            t_bcast = timesteps.view(-1, 1, 1, 1, 1).to(model_pred.dtype)
            with torch.no_grad():
                x_fake = noisy_model_input - t_bcast * model_pred.detach()
            # Fresh noise + fresh t for the fake OT trajectory.
            z_fake = torch.randn_like(x_fake)
            t_fake = torch.rand(
                noisy_model_input.shape[0],
                device=noisy_model_input.device,
                dtype=timesteps.dtype,
            )
            t_fake_bcast = t_fake.view(-1, 1, 1, 1, 1).to(model_pred.dtype)
            x_fake_t = t_fake_bcast * z_fake + (1.0 - t_fake_bcast) * x_fake
            # target_fake = z_fake - x_fake (OT velocity on the fake traj)
            target_fake = z_fake - x_fake

        # Resolve warmup/rampup schedule. inner-lambda controls T_mix's
        # supervision/adversarial blend; lam_f_eff gates the L_fake outer
        # weight on the fake branch.
        from library.training.apex_loss import apex_schedule_weights

        total_steps = int(getattr(args, "max_train_steps", 0) or 0)
        warmup_abs = int(getattr(args, "apex_warmup_steps", 0) or 0)
        rampup_abs = int(getattr(args, "apex_rampup_steps", 0) or 0)
        if warmup_abs <= 0:
            warmup_abs = int(
                float(getattr(args, "apex_warmup_ratio", 0.0) or 0.0) * total_steps
            )
        if rampup_abs <= 0:
            rampup_abs = int(
                float(getattr(args, "apex_rampup_ratio", 0.0) or 0.0) * total_steps
            )
        lam_inner_eff, lam_f_eff = apex_schedule_weights(
            step=self.step,
            warmup_steps=warmup_abs,
            rampup_steps=rampup_abs,
            lam_inner_target=float(getattr(args, "apex_lambda", 0.5)),
            lam_f_target=float(getattr(args, "apex_lambda_p", 1.0)),
        )
        # temporal-2F has no Forward 3 → L_fake is structurally absent.
        # Force the outer gate to 0 so apex_fake's loss path returns zeros
        # regardless of the configured apex_lambda_p (which has no effect here).
        if temporal_only:
            lam_f_eff = 0.0
        self._last_lam_inner_eff = float(lam_inner_eff)
        self._last_lam_f_eff = float(lam_f_eff)

        # Per-element anchor mask: a fraction of the batch gets lam_inner=0
        # (T_mix = v_data → pure FM). EMF Theorem 4.3's validity condition
        # ‖u_{t->t} − u_t‖ → 0 must hold *throughout* training, not just at
        # init — a permanent anchor keeps the FM signal alive past rampup at
        # zero extra forwards. Off (anchor_ratio=0.0) by default.
        B = noisy_model_input.shape[0]
        anchor_ratio = float(getattr(args, "apex_anchor_ratio", 0.0) or 0.0)
        if anchor_ratio > 0.0 and lam_inner_eff > 0.0:
            n_anchor = int(round(B * anchor_ratio))
            n_anchor = max(0, min(B, n_anchor))
            if n_anchor > 0:
                # Random subset: avoids correlating anchor identity with batch
                # position (which would interact with bucket ordering).
                perm = torch.randperm(B, device=noisy_model_input.device)
                anchor_idx = perm[:n_anchor]
                anchor_mask_b = torch.zeros(
                    B, device=noisy_model_input.device, dtype=model_pred.dtype
                )
                anchor_mask_b[anchor_idx] = 1.0  # 1 = anchored (lam=0)
            else:
                anchor_mask_b = torch.zeros(
                    B, device=noisy_model_input.device, dtype=model_pred.dtype
                )
        else:
            anchor_mask_b = torch.zeros(
                B, device=noisy_model_input.device, dtype=model_pred.dtype
            )
        # Per-element effective lam_inner: lam_inner_eff for non-anchor,
        # 0 for anchor. Broadcast to T_mix_v's [B,C,1,H,W] layout.
        lam_inner_per = (
            lam_inner_eff * (1.0 - anchor_mask_b)
        ).view(-1, 1, 1, 1, 1)
        anchor_frac = float(anchor_mask_b.mean().item())
        self._last_anchor_frac = anchor_frac

        # T_mix_v in velocity space (Eq. 23 after Prop. 3 conversion):
        #   T_mix_v = (1-lam_inner)*v_data + lam_inner*v_fake_sg
        # At lam_inner=0 (warmup or anchor), T_mix_v == v_data, so L_mix
        # collapses to pure FM — the bootstrap / permanent-anchor signal.
        v_data_5d = (primary.noise - primary.latents).unsqueeze(2)  # [B,C,1,H,W]
        T_mix_v = (
            (1.0 - lam_inner_per) * v_data_5d + lam_inner_per * v_fake_sg
        ).detach()

        # Stash everything ``extra_forwards_fake`` needs to run forward 3 after
        # the trainer has freed forward 1's autograd graph. temporal-2F skips
        # this entirely — _fake_state stays None so extra_forwards_fake no-ops
        # and the trainer never schedules Forward 3.
        if temporal_only:
            self._fake_state = None
        else:
            self._fake_state = {
                "anima_call": anima,
                "x_fake_t": x_fake_t,
                "t_fake": t_fake,
                "c_fake": c_fake,
                "padding_mask": padding_mask,
                "kw": kw,
                "target_fake": target_fake,
                "lam_f_eff": lam_f_eff,
            }

        # Real-branch aux only — apex_fake is computed in compose_fake_branch
        # after extra_forwards_fake fills in F_fake_on_fake_xt below. lam_c
        # is constant (paper Eq. 25 outer L_mix weight); the schedule's job
        # is to ramp the inner lambda into T_mix_v above, not to gate L_mix.
        return {
            "apex": {
                "T_mix_v": T_mix_v.squeeze(2),
                "lam_inner_eff": lam_inner_eff,
                "lam_f_eff": lam_f_eff,
            }
        }

    def extra_forwards_fake(self, ctx: StepCtx):
        state = self._fake_state
        if state is None:
            return None
        # Single-shot: clear the stash so a buggy double-call can't
        # double-backward forward 3's graph.
        self._fake_state = None

        args = ctx.args
        # Forward 3: F_fake_on_fake_xt at (x_fake_t, t_fake, c_fake). Grad
        # flows back through both LoRA and ConditionShift via L_fake.
        F_fake_on_fake_xt = state["anima_call"](
            state["x_fake_t"],
            state["t_fake"],
            state["c_fake"],
            padding_mask=state["padding_mask"],
            **state["kw"],
        )

        # Weighting for the L_fake term at its own timestep. Imported lazily so
        # this module doesn't pull in library.anima at definition time.
        from library.anima.training import compute_loss_weighting_for_anima

        weighting_fake = compute_loss_weighting_for_anima(
            weighting_scheme=args.weighting_scheme, sigmas=state["t_fake"]
        )

        return {
            "apex": {
                "F_fake_on_fake_xt": F_fake_on_fake_xt.squeeze(2),
                "target_fake": state["target_fake"].squeeze(2),
                "weighting_fake": weighting_fake,
                "t_fake": state["t_fake"],
                "lam_f_eff": state["lam_f_eff"],
            }
        }

    def metrics(self, ctx) -> dict[str, float]:
        """Emit log-step keys owned by APEX.

        Adapter-private scalars (warmup/rampup eff, v_fake divergence) are
        read from ``self``. Loss values stashed by ``library/training/losses.py``
        and the ``ConditionShift`` parameters live on the network (the loss
        code that writes them has no adapter handle); read them through
        ``ctx.network``.
        """
        method = getattr(ctx.args, "method", None) or ""
        if not (method == "apex" or method.startswith("apex_")):
            return {}
        # temporal-2F has no Forward 3; suppress L_fake / lam_f_eff scalars so
        # TB doesn't grow flat-zero traces. v_fake_divergence stays — under
        # temporal-2F it equals MSE(F(x_t, t+Δt, c), F(x_t, t, c)), the t-shift's
        # output-space magnitude (probe table 1 metric).
        temporal_only = bool(getattr(ctx.args, "apex_temporal_only", False))
        out: dict[str, float] = {
            "apex/lam_inner_eff": float(self._last_lam_inner_eff),
            "apex/anchor_frac": float(self._last_anchor_frac),
        }
        if not temporal_only:
            out["apex/lam_f_eff"] = float(self._last_lam_f_eff)
        if self._last_v_fake_divergence is not None:
            out["apex/v_fake_divergence"] = float(self._last_v_fake_divergence)
        if len(self._v_fake_div_recorder.loss_list) > 0:
            out["apex/v_fake_divergence_avg"] = float(
                self._v_fake_div_recorder.moving_average
            )
        if self._last_dt_clamp_frac is not None:
            out["apex/dt_clamp_frac"] = float(self._last_dt_clamp_frac)

        network = ctx.network
        mix_v = getattr(network, "_last_apex_mix_value", None)
        if mix_v is not None:
            out["apex/loss_mix"] = float(mix_v)
        if len(self._mix_recorder.loss_list) > 0:
            out["apex/loss_mix_avg"] = float(self._mix_recorder.moving_average)
        if not temporal_only:
            fake_v = getattr(network, "_last_apex_fake_value", None)
            if fake_v is not None:
                out["apex/loss_fake"] = float(fake_v)
            if len(self._fake_recorder.loss_list) > 0:
                out["apex/loss_fake_avg"] = float(self._fake_recorder.moving_average)

        cs = getattr(network, "apex_condition_shift", None)
        if cs is not None:
            mode = getattr(cs, "mode", None)
            if mode == "scalar":
                out["apex/cs_a"] = float(cs.a.detach().item())
                out["apex/cs_b"] = float(cs.b.detach().item())
            elif mode == "diag":
                out["apex/cs_a_norm"] = float(cs.a.detach().norm().item())
                out["apex/cs_b_norm"] = float(cs.b.detach().norm().item())
            elif mode == "full":
                out["apex/cs_A_norm"] = float(cs.A.detach().norm().item())
                out["apex/cs_b_norm"] = float(cs.b.detach().norm().item())
        return out
