"""Turbo distillation main loop — DP-DMD (diversity-preserved DMD).

Usage:
    python -m scripts.distill_turbo.distill [--config configs/methods/turbo.toml] ...

The math walkthrough lives in :mod:`scripts.distill_turbo`; this file is the
per-step orchestrator (teacher K-step anchor → diversity-supervised first step →
DMD-refined N-step student rollout → fake/critic update → save).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from library.anima import weights as anima_utils
from library.anima.models import Anima
from library.datasets.distill import CachedDataset
from library.inference.sampling import get_timesteps_sigmas
from library.inference.uncond import (
    default_uncond_path,
    load_uncond_crossattn,
    uncond_for_batch,
)
from library.runtime.harness import (
    compile_dit_blocks,
    enable_training_grad_ckpt,
    place_dit_for_training,
)
from networks.methods.turbo_dmd import TurboDMDNetwork

from .config import (
    build_argparser,
    load_turbo_config,
    resolve_config,
    snapshot_toml_text,
    tb_config_text,
)
from .metrics import TurboMetrics, tqdm_postfix, write_scalars
from .primitives import (
    PadCache,
    make_collate,
    make_scheduler,
    renoise,
    sample_t,
)
from .warmup import run_fake_warmup

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _step_tag(step: int) -> str:
    """Human checkpoint suffix: 1000 -> ``1k``, 8000 -> ``8k``, else raw count.

    Matches the hand-rolled ``_1k`` / ``_500`` naming the runs already use.
    """
    return f"{step // 1000}k" if step % 1000 == 0 else str(step)


def mean_var_kl(
    x: torch.Tensor, mu_t: torch.Tensor | float, sigma2_t: torch.Tensor | float
) -> torch.Tensor:
    """Mean-variance regularizer (lever B / paper Eq. 7, arXiv:2511.22677).

    KL of each generated image's per-image Gaussian ``N(μ_i, σ²_i)`` toward the
    real-latent target ``N(μ_t, σ²_t)``, averaged over the batch:

        L_mv = (1/B) Σ_i ½[ (σ_i² + (μ_i − μ_t)²)/σ_t² − 1 − log(σ_i²/σ_t²) ]

    Differentiable in ``x`` (= ``x_pred``), so it backprops into the student and
    directly clamps the **variance inflation** that *is* the over-bake's
    oversaturation (§3.2). Stats are per-image over (C, H, W); full-frame even
    under masked loss (paper-faithful — the reg is a global distribution clamp).
    """
    B = x.shape[0]
    flat = x.reshape(B, -1)
    mu_i = flat.mean(dim=1)
    var_i = flat.var(dim=1, unbiased=False)
    kl = 0.5 * (
        (var_i + (mu_i - mu_t) ** 2) / sigma2_t
        - 1.0
        - torch.log((var_i / sigma2_t).clamp_min(1e-8))
    )
    return kl.mean()


def calibrate_mean_var(
    dataloader: torch.utils.data.DataLoader,
    *,
    max_batches: int = 0,
    norm_floor: float = 0.05,
) -> tuple[float, float]:
    """Exact one-pass global mean/variance of the real cached latents.

    The Eq.7 reg target is a static dataset statistic — a single scalar
    ``(μ, σ²)`` over every latent element — so we measure it directly rather
    than EMA-tracking it during training. Accumulates count / sum / sum-of-
    squares in fp64 (population variance, ``unbiased=False`` — matching the
    per-image stats in :func:`mean_var_kl`) for numerical stability across the
    whole pool. ``max_batches <= 0`` scans the full dataset; a positive cap
    trades a little exactness for I/O (the global scalar converges fast).
    Returns ``(μ_t, σ²_t)`` with σ²_t floored at ``norm_floor²``.
    """
    n = 0
    s = 0.0
    s2 = 0.0
    seen = 0
    for batch in dataloader:
        # Batch layout mirrors the training loop: masked adds a trailing mask.
        latents = batch[1].double()
        flat = latents.reshape(-1)
        n += flat.numel()
        s += float(flat.sum())
        s2 += float((flat * flat).sum())
        seen += 1
        if max_batches > 0 and seen >= max_batches:
            break
    if n == 0:
        raise RuntimeError(
            "mean-variance calibration scanned 0 latents — empty dataloader "
            "(check data_dir / curation keep_list / drop_last vs batch_size)."
        )
    mu = s / n
    var = max(s2 / n - mu * mu, norm_floor**2)
    logger.info(
        f"mean-variance calibration: scanned {seen} batches / {n:,} latent "
        f"elements → μ_t={mu:.6g}, σ²_t={var:.6g}"
    )
    return mu, var


def main():
    args = build_argparser().parse_args()
    cfg = resolve_config(args, load_turbo_config(args.config))

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ---------------- Model ----------------
    logger.info(f"loading DiT: {cfg.dit_path}")
    model: Anima = anima_utils.load_anima_model(
        device,
        cfg.dit_path,
        attn_mode=cfg.attn_mode,
        loading_device="cpu" if cfg.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )

    # Block swap (per-forward prepare hook done at each forward call below), then
    # compile each block._forward (native-shape flatten, one graph per token count;
    # the pool spans more than the 2 CONSTANT_TOKEN_BUCKETS families).
    place_dit_for_training(model, device, blocks_to_swap=cfg.blocks_to_swap)
    compile_dit_blocks(model, enabled=cfg.torch_compile, mode="")
    enable_training_grad_ckpt(model, enabled=cfg.grad_ckpt)

    # ---------------- LoRA stacks ----------------
    turbo = TurboDMDNetwork(
        unet=model,
        student_rank=cfg.student_rank,
        fake_rank=cfg.fake_rank,
        student_alpha=cfg.student_alpha,
        fake_alpha=cfg.fake_alpha,
        use_custom_down_autograd=cfg.use_custom_down_autograd,
    )
    turbo.freeze_dit()
    turbo.student.to(device=device, dtype=dtype)
    turbo.fake.to(device=device, dtype=dtype)
    # `model.training` gates grad-ckpt inside block.forward; toggled per
    # forward in `_forward` below so no_grad teacher/fake forwards don't
    # incur grad-ckpt setup cost. Initial state set by the first call.

    n_student = sum(p.numel() for p in turbo.student_params())
    n_fake = sum(p.numel() for p in turbo.fake_params())
    logger.info(f"trainable: student={n_student:,}  fake={n_fake:,}")

    # ---------------- Optimizers ----------------
    student_opt = torch.optim.AdamW(
        turbo.student_params(),
        lr=cfg.student_lr,
        weight_decay=cfg.weight_decay,
        fused=torch.cuda.is_available(),
    )
    fake_opt = torch.optim.AdamW(
        turbo.fake_params(),
        lr=cfg.fake_lr,
        weight_decay=cfg.weight_decay,
        fused=torch.cuda.is_available(),
    )

    student_sched = make_scheduler(student_opt, cfg.iterations, cfg.student_lr)
    # The fake optimizer takes ``iterations · fake_steps_per_student_step``
    # updates in the main loop plus ``fake_warmup_steps`` head-start updates
    # BEFORE it (the head-start is now counted in fake updates directly, NOT
    # scaled by the cadence — see warmup.py). The fake scheduler is stepped
    # through both phases, so its total update count — and hence the ``0.02·total``
    # LR warmup span — is sized over the same total. The fake LR warmup therefore
    # overlaps the head-start (the fake enters the main loop already calibrated
    # AND at full LR), and the cosine still lands at the end of the main loop.
    # The student schedule is independent: ``0.02·iterations``, no head-start offset.
    fake_sched = make_scheduler(
        fake_opt,
        cfg.iterations * cfg.fake_steps_per_student_step + cfg.fake_warmup_steps,
        cfg.fake_lr,
    )

    # ---------------- Dataset ----------------
    dataset = CachedDataset(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        sample_ratio=cfg.sample_ratio,
        mask_dir=cfg.mask_dir if cfg.use_masked_loss else None,
    )
    if cfg.single_prompt_idx is not None:
        # Phase 0 overfit — wrap as a 1-sample list so the dataloader cycles it.
        # The "N samples from ..." line above is CachedDataset.__init__'s own
        # log, fired BEFORE this slice; we re-log post-slice so the live
        # dataset state is unambiguous in the run log.
        pinned_idx = cfg.single_prompt_idx % len(dataset.samples)
        only = dataset.samples[pinned_idx]
        dataset.samples = [only]
        latent_stem = os.path.basename(only[0])
        logger.info(
            f"single-prompt overfit mode: pinned to idx={cfg.single_prompt_idx} "
            f"(post-slice len(dataset)={len(dataset)}, latent={latent_stem})"
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # bucket-grouped
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=make_collate(cfg.use_masked_loss),
    )

    # ---------------- Logging ----------------
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    writer = None
    if not cfg.no_log:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log = Path(cfg.log_dir) / run_name
        run_log.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_log))
        writer.add_text("config", tb_config_text(cfg))
        logger.info(f"TB logs -> {run_log}")

        # Mirror the resolved config into the run log dir so the timestamped
        # run becomes a self-contained record of "this run + the config that
        # produced it" — the turbo analogue of train.py's .snapshot.toml.
        snapshot_path = run_log / f"{cfg.output_name}.snapshot.toml"
        try:
            snapshot_path.write_text(
                snapshot_toml_text(cfg, source_config=args.config),
                encoding="utf-8",
            )
            logger.info(f"Config snapshot written: {snapshot_path}")
        except OSError as e:
            logger.warning(f"Could not write config snapshot to {snapshot_path}: {e}")

    # ---------------- Forward helper ----------------
    pad_cache = PadCache(dtype)
    # CFG-uncond cross-attention input. Anima's inference path uses the T5("")
    # embedding (real BOS/EOS/sentinel tokens nonzero; only padding zeroed) —
    # passing a fully-zero tensor here is fed-out-of-distribution and the
    # resulting `v_real_uncond_ca` is a meaningless direction that, amplified
    # at (α-1)=3×, drives the student off-manifold (saturated white output).
    # Staged by `make preprocess-te` (or `make distill-prep`); shared with
    # the mod-guidance distill (`library/inference/uncond.py`).
    uncond_path = str(default_uncond_path())
    uncond_base = load_uncond_crossattn(uncond_path, device=device, dtype=dtype)
    logger.info(
        f"loaded T5('') uncond sidecar: {uncond_path}  shape={tuple(uncond_base.shape)}"
    )

    def _forward(view: str, x: torch.Tensor, t_b: torch.Tensor, c: torch.Tensor, *, no_grad: bool):
        """Switch view, prepare block swap, run forward.

        ``x`` is (B, 16, H, W); we unsqueeze to (B, 16, 1, H, W) inside.

        Per-forward CPU prep is the GPU-idle window between launches —
        ``set_view`` short-circuits when already in ``view`` (see
        ``TurboDMDNetwork.set_view``), and the cudagraph step-begin marker
        is hoisted to once per outer step in the loop below.

        The DiT is frozen (``freeze_dit`` in ``__init__``), so ``model.training``
        is left at its post-construction value (``True``) for the whole run —
        grad-ckpt (``cfg.grad_ckpt``, default on) is gated on ``self.training``
        inside ``Block.forward``, so it stays armed without a per-forward toggle.
        We deliberately do NOT flip train/eval per forward: the no_grad teacher/
        fake forwards build no backward graph regardless, and the recursive
        submodule walk a per-forward toggle triggered was the dominant
        per-forward CPU stall. Grad-ckpt's recompute only bites on the grad-
        bearing student/fake-update forwards.
        """
        turbo.set_view(view)
        if model.blocks_to_swap:
            # free_cache=False: base DiT is frozen, LoRA shapes are constant,
            # block swap moves params at identical shape, and static 4096 tokens
            # pins activation sizes — the allocator reaches a steady state within
            # a few steps and per-forward empty_cache() is pure sync +
            # refragmentation overhead.
            model.prepare_block_swap_before_forward(free_cache=False)
        pad = pad_cache.get(x)
        x_in = x.unsqueeze(2)  # add temporal dim
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx, torch.autocast("cuda", dtype=dtype):
            return model.forward_mini_train_dit(
                x_in, t_b, c, padding_mask=pad, skip_pooled_text_proj=True
            )

    # ---------------- DP-DMD setup ----------------
    # The student/teacher Euler grids are static (token-count-invariant flow-
    # matching σ schedule), so build them once. Both span σ: 1 → 0; the student
    # has `student_steps + 1` points, the teacher anchor grid `teacher_anchor_steps
    # + 1`. `sigmas[i] - sigmas[i+1]` is the Euler dt for step i.
    student_sigmas = (
        get_timesteps_sigmas(cfg.student_steps, cfg.flow_shift, "cpu")[1].tolist()
    )
    teacher_anchor_sigmas = (
        get_timesteps_sigmas(cfg.teacher_anchor_steps, cfg.flow_shift, "cpu")[1]
        .tolist()
    )
    # Continuous time at the anchor (incoming σ after k_anchor teacher steps).
    # `v_target = (ε − z_tk)/(1 − t_k)` — a σ mismatch here silently mis-scales
    # the diversity target (proposal §6.3), so it's read from the teacher grid,
    # not the student grid.
    t_k_anchor = float(teacher_anchor_sigmas[cfg.k_anchor])
    logger.info(
        f"DP-DMD grids: student σ={['%.3f' % s for s in student_sigmas]}, "
        f"anchor t_k={t_k_anchor:.4f} (teacher step {cfg.k_anchor}/"
        f"{cfg.teacher_anchor_steps})"
    )

    def _teacher_cfg_velocity(x, t_b, c_cond, c_null):
        """CFG-guided teacher velocity ``v_u + α·(v_c − v_u)`` (no grad, fp32).

        Used by the DP-DMD K-step anchor rollout. At ``teacher_cfg == 1`` the
        uncond forward is skipped (single forward).
        """
        v_c = _forward("teacher", x, t_b, c_cond, no_grad=True).squeeze(2)
        if cfg.teacher_cfg == 1.0:
            return v_c.float()
        v_u = _forward("teacher", x, t_b, c_null, no_grad=True).squeeze(2)
        return v_u.float() + cfg.teacher_cfg * (v_c.float() - v_u.float())

    # ---------------- Mean-variance reg target (lever B / S2) ----------------
    # Real-data stats the Eq.7 KL pulls each generated image toward. Either
    # pinned (cfg.mv_sigma2_t > 0) or measured EXACTLY in a one-pass scan over
    # the real latents (cfg.mv_sigma2_t <= 0). The target is a static dataset
    # statistic — a single global (μ, σ²) over all latent elements — so an exact
    # pre-pass strictly beats a running EMA: no decay lag, no batch-to-batch
    # wobble, deterministic, and free in the hot loop. Computed in fp64 via the
    # numerically-stable count/sum/sumsq route over the same `latents` the
    # dataloader yields (the REAL training latents — NOT teacher-synthetic, since
    # the reg is a shield against the teacher's variance inflation). Runs BEFORE
    # the fake head-start: it's model-independent, so doing it first fails fast on
    # an empty/misconfigured pool instead of after burning warmup compute.
    mv_auto = cfg.mean_var_weight > 0.0 and cfg.mv_sigma2_t <= 0.0
    mv_tgt_mu = cfg.mv_mu_t
    mv_tgt_var = cfg.mv_sigma2_t
    if cfg.mean_var_weight > 0.0:
        if mv_auto:
            mv_tgt_mu, mv_tgt_var = calibrate_mean_var(
                dataloader,
                max_batches=cfg.mv_calib_batches,
                norm_floor=cfg.norm_floor,
            )
        logger.info(
            "mean-variance reg ON (lever B / Eq.7): weight="
            f"{cfg.mean_var_weight}, target="
            + (
                f"measured μ_t={mv_tgt_mu:.6g}, σ²_t={mv_tgt_var:.6g} "
                f"(exact, over real latents)"
                if mv_auto
                else f"fixed μ_t={mv_tgt_mu}, σ²_t={mv_tgt_var}"
            )
        )

    # ---------------- Fake (critic) head-start ----------------
    data_iter = iter(dataloader)
    data_iter = run_fake_warmup(
        warmup_steps=cfg.fake_warmup_steps,
        turbo=turbo,
        forward_fn=_forward,
        data_iter=data_iter,
        dataloader=dataloader,
        fake_opt=fake_opt,
        fake_sched=fake_sched,
        grad_clip=cfg.grad_clip,
        t_distribution=cfg.t_distribution,
        sigmoid_scale=cfg.sigmoid_scale,
        device=device,
        dtype=dtype,
        log_interval=cfg.log_interval,
        writer=writer,
    )

    # ---------------- Training loop ----------------
    logger.info(f"starting DP-DMD training: {cfg.iterations} iterations")
    progress = tqdm(range(cfg.iterations), desc="turbo")
    metrics = TurboMetrics(device)

    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        if cfg.use_masked_loss:
            _idx, latents, crossattn_emb, _pooled, mask = batch
            # float (not bf16): the student loss is assembled in fp32. [B,1,H,W]
            # broadcasts over the [B,16,H,W] grad signal.
            mask = mask.to(device, dtype=torch.float32, non_blocking=True)
        else:
            _idx, latents, crossattn_emb, _pooled = batch
            mask = None

        latents = latents.to(device, dtype=dtype, non_blocking=True)
        crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
        B = latents.shape[0]

        # One step-begin marker per training step (not per forward).
        # ``compile_blocks(mode="default")`` doesn't enable cudagraphs, so this
        # is semantically a no-op today, but it's the right cadence if/when
        # the script switches to ``mode="reduce-overhead"``.
        torch.compiler.cudagraph_mark_step_begin()

        # ============ DP-DMD student update ============
        # No single t / x_t: the student rolls the full N-step Euler grid from
        # pure noise ε. Step 1 is supervised toward a teacher K-step anchor
        # (diversity) and detached, then DMD refines x_θ over steps 2..N
        # (quality). See dpdmd.md §3.2.
        eps = torch.randn_like(latents)  # shared start for anchor + student

        # --- teacher K-step CFG anchor (no grad) → v_target ---
        c_null = uncond_for_batch(uncond_base, crossattn_emb)
        z = eps
        for i in range(cfg.k_anchor):
            s_i = teacher_anchor_sigmas[i]
            s_next = teacher_anchor_sigmas[i + 1]
            t_b = torch.full((B,), s_i, device=device, dtype=dtype)
            v = _teacher_cfg_velocity(z, t_b, crossattn_emb, c_null)
            z = (z.float() - (s_i - s_next) * v).to(dtype)
        # Average velocity ε→z_tk over [t_k, 1]; this is exactly the target
        # for the student's t=1 first step (Euler x_next = x − dt·v_first).
        v_target = ((eps.float() - z.float()) / (1.0 - t_k_anchor)).detach()

        # --- student N-step rollout; step-0 div-supervised + (optionally) detached ---
        # Split forward/backward: under `detach_after_first` the step-0 forward
        # graph is severed from the steps 2..N DMD chain, so we backward the
        # diversity term IMMEDIATELY and free step-0's activations BEFORE the DMD
        # chain builds its own graph. Peak student-forward activations then stay at
        # ONE forward (the longer of {step-0, the N-1 DMD chain}) instead of holding
        # both concurrently — the two losses share no graph, so a single combined
        # backward was only ever paying 2× activation memory for nothing. This is
        # what lets grad-ckpt be optional at small N: the "unroll" it tames is N-1
        # steps deep on the DMD chain, NOT the (now-freed) step-0 graph. With the
        # detach OFF (A/B), the graphs are entangled and we MUST keep one combined
        # backward (the diversity term is folded in at assembly below).
        split_bwd = cfg.detach_after_first

        x = eps
        x.requires_grad_()  # grad-ckpt needs a grad-requiring forward input
        # Step 0: the diversity-supervised first step (grad → v_first).
        s0, s0_next = student_sigmas[0], student_sigmas[1]
        t_b = torch.full((B,), s0, device=device, dtype=dtype)
        v_first = _forward("student", x, t_b, crossattn_emb, no_grad=False).squeeze(2)
        x = x - (s0 - s0_next) * v_first
        div_loss_t = nn.functional.mse_loss(v_first.float(), v_target)
        if split_bwd:
            # Load-bearing stop-grad: the DMD reverse-KL from steps 2..N must NOT
            # flow back into the diversity mapping (their Fig 5). Backward the
            # diversity term against the step-0 graph now (accumulates into the
            # student .grad buffers), then re-leaf so the DMD chain gets a fresh
            # grad-ckpt root. The optimizer step waits for the DMD backward below.
            (cfg.div_weight * div_loss_t).backward()
            x = x.detach().requires_grad_()

        # Steps 2..N: the DMD-refined rollout (grad flows to x_pred).
        for i in range(1, cfg.student_steps):
            s_i = student_sigmas[i]
            s_next = student_sigmas[i + 1]
            t_b = torch.full((B,), s_i, device=device, dtype=dtype)
            v = _forward(
                "student", x, t_b, crossattn_emb, no_grad=False
            ).squeeze(2)
            x = x - (s_i - s_next) * v
        x_pred = x  # = x_θ (B,16,H,W); v_student aliases v_first for metrics
        v_student = v_first

        # --- DMD on x_θ (steps 2..N), against teacher + fake ---
        # The real score is CFG-GUIDED (v_u + α·(v_c − v_u)), NOT cond-only.
        # Guidance rides the single DMD real score — exactly like the reference
        # `compute_dmd_loss` (dpdmd/train_sd35_dpdmd.py:118-129, teacher cat
        # cond+uncond → v_u + scale·(v_c−v_u)). Without it v_real≈v_fake (both
        # unguided cond preds collapse, dm_cos≈0.9999) and the quality gradient
        # is noise. The fake stays cond-only, matching the reference.
        tau_dm = torch.rand(B, device=device, dtype=dtype)
        eps_dm = torch.randn_like(x_pred)
        x_renoised_dm = renoise(x_pred.detach(), tau_dm, eps_dm)
        v_real_cond_dm = _teacher_cfg_velocity(
            x_renoised_dm, tau_dm, crossattn_emb, c_null
        )
        v_fake_cond_dm = _forward(
            "fake", x_renoised_dm, tau_dm, crossattn_emb, no_grad=True
        ).squeeze(2)
        delta_dm = v_real_cond_dm - v_fake_cond_dm

        tau_dm_e = tau_dm.view(B, 1, 1, 1).float()
        grad_dm = tau_dm_e * delta_dm.float()
        if cfg.dm_x0_norm:
            denom = (
                (tau_dm_e * v_real_cond_dm.float())
                .abs()
                .mean(dim=(1, 2, 3), keepdim=True)
                .clamp_min(cfg.norm_floor)
            )
            grad_dm = grad_dm / denom
        grad_signal = grad_dm.detach()

        # --- assemble: DMD surrogate on x_θ (+ optional mean-var) ---
        # The diversity term was already backwarded above when split_bwd; otherwise
        # it rides this combined backward (graphs still entangled). grad_clip below
        # runs once on the ACCUMULATED .grad (div + DMD), so the clipped norm is the
        # full student gradient either way.
        if mask is not None:
            loss_dmd = (grad_signal * x_pred.float() * mask).mean()
        else:
            loss_dmd = (grad_signal * x_pred.float()).mean()
        loss_student = loss_dmd

        mv_loss = torch.zeros((), device=device)
        if cfg.mean_var_weight > 0.0:
            mv_loss = mean_var_kl(x_pred.float(), mv_tgt_mu, mv_tgt_var)
            loss_student = loss_student + cfg.mean_var_weight * mv_loss

        if not split_bwd:
            loss_student = loss_student + cfg.div_weight * div_loss_t

        loss_student.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                turbo.student_params(), max_norm=cfg.grad_clip
            )
        student_opt.step()
        student_opt.zero_grad(set_to_none=True)
        student_sched.step()

        # --- 5. FAKE UPDATE ---
        # Run ``fake_steps_per_student_step`` inner updates against the same
        # x_pred.detach(), resampling (τ_fake, ε_fake) each iteration so each
        # inner step sees a different rung of the flow-matching forward path.
        # Standard DMD2 practice: keep the fake's regression target ahead of
        # the student's moving x_pred distribution.
        x_pred_d = x_pred.detach()
        fake_loss_sum = torch.zeros((), device=device)
        for _ in range(cfg.fake_steps_per_student_step):
            tau_fake = sample_t(
                B, distribution=cfg.t_distribution,
                sigmoid_scale=cfg.sigmoid_scale, device=device, dtype=dtype,
            )
            eps_fake = torch.randn_like(x_pred_d)
            x_t_fake = renoise(x_pred_d, tau_fake, eps_fake).requires_grad_()
            v_fake = _forward(
                "fake", x_t_fake, tau_fake, crossattn_emb, no_grad=False
            ).squeeze(2)
            target_v_fake = eps_fake - x_pred_d  # flow-matching target
            fake_loss = nn.functional.mse_loss(v_fake.float(), target_v_fake.float())
            fake_loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    turbo.fake_params(), max_norm=cfg.grad_clip
                )
            fake_opt.step()
            fake_opt.zero_grad(set_to_none=True)
            fake_sched.step()
            fake_loss_sum = fake_loss_sum + fake_loss.detach()
        fake_loss_mean_t = fake_loss_sum / cfg.fake_steps_per_student_step

        # --- logging accumulators (all GPU-side; flushed below every log_interval
        # in one stacked .tolist() so per-step CUDA syncs go to zero) ---
        metrics.accumulate_per_step(
            fake_loss_mean_t=fake_loss_mean_t,
            grad_signal=grad_signal,
            delta_dm=delta_dm,
            x_pred=x_pred,
            v_student=v_student,
            tau_dm_e=tau_dm_e,
            v_real_cond_dm=v_real_cond_dm,
            v_fake_cond_dm=v_fake_cond_dm,
            mv_loss=mv_loss,
        )
        metrics.add_div(div_loss_t)

        if (step + 1) % cfg.log_interval == 0:
            m = metrics.flush(cfg.log_interval)
            if writer is not None:
                write_scalars(writer, m, step + 1)
                writer.add_scalar(
                    "train/student_lr", student_sched.get_last_lr()[0], step + 1
                )
                writer.add_scalar("train/fake_lr", fake_sched.get_last_lr()[0], step + 1)
            # tqdm postfix at log_interval cadence (per-step would re-introduce
            # the syncs we just eliminated). First log_interval steps show no
            # postfix — harmless.
            progress.set_postfix(**tqdm_postfix(m))
            metrics.reset()

        # --- save ---
        # Every save_every checkpoint is kept under a step-tagged name (no
        # overwrite, so the whole training trajectory survives for eyeballing);
        # the final step also writes the canonical bare `{output_name}` that
        # inference / merge / `make test` look for.
        if (step + 1) % cfg.save_every == 0 or (step + 1) == cfg.iterations:
            n = step + 1
            is_final = n == cfg.iterations
            metadata = {
                "ss_turbo_objective": "dpdmd",
                "ss_turbo_student_rank": str(cfg.student_rank),
                "ss_turbo_student_steps": str(cfg.student_steps),
                "ss_turbo_teacher_cfg": str(cfg.teacher_cfg),
                "ss_turbo_step": str(n),
                "ss_turbo_k_anchor": str(cfg.k_anchor),
                "ss_turbo_div_weight": str(cfg.div_weight),
            }
            save_names = [f"{cfg.output_name}_{_step_tag(n)}"]
            if is_final:
                save_names.append(cfg.output_name)  # canonical bare name
            for name in save_names:
                save_path = str(Path(cfg.output_dir) / f"{name}.safetensors")
                turbo.save_student(save_path, dtype=torch.bfloat16, metadata=metadata)
                logger.info(f"saved checkpoint: {save_path}")

    if writer is not None:
        writer.close()
    logger.info("turbo distillation complete.")


if __name__ == "__main__":
    main()
