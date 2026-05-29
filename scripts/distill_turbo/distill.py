"""Turbo distillation main loop — single-call DMD2.

Usage:
    python -m scripts.distill_turbo.distill [--config configs/methods/turbo.toml] ...

The math walkthrough lives in :mod:`scripts.distill_turbo`; this file is the
per-step orchestrator (sample t, student/CA/DM/fake forwards, optimizer steps,
save).
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

from .ca_band import apply_ca_band_deficit
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
    sample_t_above,
)
from .warmup import run_fake_warmup

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    compile_dit_blocks(model, enabled=cfg.torch_compile, mode="reduce-overhead")
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
    # Fake gets ``fake_steps_per_student_step`` updates per outer iteration, plus
    # ``fake_warmup_steps`` head-start iterations BEFORE the main loop. The fake
    # scheduler is stepped through both phases, so its total update count — and
    # hence the ``0.02·total`` LR warmup span — is computed over
    # ``iterations + fake_warmup_steps``. The fake LR warmup therefore overlaps
    # the head-start (the fake enters the main loop already calibrated AND at
    # full LR), and the cosine still lands at the end of the main loop. The
    # student schedule is independent: ``0.02·iterations``, no head-start offset.
    fake_sched = make_scheduler(
        fake_opt,
        (cfg.iterations + cfg.fake_warmup_steps) * cfg.fake_steps_per_student_step,
        cfg.fake_lr,
    )

    # ---------------- Dataset ----------------
    # Curation gate (item 5): load the keep_list stems from `make exp-turbo-prep`
    # and pass them to the reader, which drops every stem not in the cut.
    keep_list: set[str] | None = None
    if cfg.use_prep_list:
        prep_path = Path(cfg.prep_list_path)
        if not prep_path.exists():
            raise FileNotFoundError(
                f"use_prep_list=true but keep_list missing: {prep_path}. "
                f"Run `make exp-turbo-prep` first (or set use_prep_list=false)."
            )
        import json

        keep_list = set(json.loads(prep_path.read_text())["kept"])
        logger.info(f"curation gate ON: {len(keep_list)} stems from {prep_path}")

    dataset = CachedDataset(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        sample_ratio=cfg.sample_ratio,
        mask_dir=cfg.mask_dir if cfg.use_masked_loss else None,
        keep_list=keep_list,
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

        The DiT is frozen (``freeze_dit`` in ``__init__``) and grad-ckpt is
        off in this script's default path, so ``model.training`` is left at
        whatever it was post-construction — toggling it per forward only
        gated grad-ckpt setup that isn't active here, and the recursive
        submodule walk it triggered was the dominant per-forward CPU stall.
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
        fake_steps_per_student_step=cfg.fake_steps_per_student_step,
        grad_clip=cfg.grad_clip,
        t_distribution=cfg.t_distribution,
        sigmoid_scale=cfg.sigmoid_scale,
        device=device,
        dtype=dtype,
        log_interval=cfg.log_interval,
        writer=writer,
    )

    # ---------------- Training loop ----------------
    logger.info(f"starting DMD2 training: {cfg.iterations} iterations")
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

        # Sample generator-t on CPU so the do_ca skip-check below stays sync-free
        # (proposal R5: skip CA when t is very late — collapsed interval → noisy
        # grad). Mid-step .item() on a device tensor would drain the CUDA
        # pipeline between the student forward and CA branch.
        if cfg.t_distribution == "uniform":
            t_cpu = torch.rand(B, dtype=torch.float32)
        else:
            t_cpu = torch.sigmoid(cfg.sigmoid_scale * torch.randn(B, dtype=torch.float32))
        do_ca = bool((t_cpu < cfg.tau_ca_skip_above_t).any().item())  # CPU op
        t = t_cpu.to(device=device, dtype=dtype, non_blocking=True)

        # Build x_t = (1-t)·x_0 + t·ε.
        eps = torch.randn_like(latents)
        t_e = t.view(B, 1, 1, 1)
        x_t = (
            (1.0 - t_e) * latents + t_e * eps
        ).requires_grad_()  # requires_grad for grad-ckpt

        # --- 1. STUDENT FORWARD (grad to student) ---
        v_student = _forward("student", x_t, t, crossattn_emb, no_grad=False)
        # v_student: (B, 16, 1, H, W). Drop temporal dim for arithmetic.
        v_student = v_student.squeeze(2)
        x_pred = x_t.squeeze(2) - t_e * v_student  # (B, 16, H, W), grad-bearing

        # --- 2. CA BRANCH (no grad, teacher × 2) ---
        band_diag = None
        if do_ca:
            tau_ca = sample_t_above(t.float(), min_gap=cfg.tau_ca_min_gap).to(dtype)
            eps_ca = torch.randn_like(x_pred)
            x_renoised_ca = renoise(x_pred.detach(), tau_ca, eps_ca)
            v_real_cond_ca = _forward(
                "teacher", x_renoised_ca, tau_ca, crossattn_emb, no_grad=True
            ).squeeze(2)
            c_null = uncond_for_batch(uncond_base, crossattn_emb)
            v_real_uncond_ca = _forward(
                "teacher", x_renoised_ca, tau_ca, c_null, no_grad=True
            ).squeeze(2)
            delta_cfg = v_real_cond_ca - v_real_uncond_ca

            if cfg.ca_band_weight_enabled:
                # Item 2 measurement fix (see item2_diagnosis.md): one extra
                # no-grad teacher forward at the student's (x_t, t, c) so the
                # FEI gap is computed at the same sampler time as x_pred.
                # No uncond view needed — we only need x0 at t for FEI, not
                # the CFG direction.
                v_teacher_at_t = _forward(
                    "teacher", x_t.detach(), t, crossattn_emb, no_grad=True
                ).squeeze(2)
                delta_cfg, band_diag = apply_ca_band_deficit(
                    delta_cfg,
                    x_t=x_t.detach(),
                    v_teacher_at_t=v_teacher_at_t,
                    t=t,
                    x_pred=x_pred,
                    tau_ca=tau_ca,
                    beta=cfg.ca_band_beta,
                    divisor=cfg.ca_band_divisor,
                    window_lo=cfg.ca_band_window_lo,
                    window_hi=cfg.ca_band_window_hi,
                )
        else:
            delta_cfg = torch.zeros_like(x_pred)

        # --- 3. DM BRANCH (no grad teacher + no grad fake) ---
        tau_dm = torch.rand(B, device=device, dtype=dtype)
        eps_dm = torch.randn_like(x_pred)
        x_renoised_dm = renoise(x_pred.detach(), tau_dm, eps_dm)
        v_real_cond_dm = _forward(
            "teacher", x_renoised_dm, tau_dm, crossattn_emb, no_grad=True
        ).squeeze(2)
        v_fake_cond_dm = _forward(
            "fake", x_renoised_dm, tau_dm, crossattn_emb, no_grad=True
        ).squeeze(2)
        delta_dm = v_real_cond_dm - v_fake_cond_dm

        # --- 4. ASSEMBLE + BACKWARD into student ---
        warmup_frac = min(1.0, (step + 1) / max(1, cfg.alpha_warmup_steps))
        alpha_eff = cfg.teacher_cfg * warmup_frac + 1.0 * (1.0 - warmup_frac)

        # DMD2 gradient in x0 space. The DiT predicts velocity (v = ε − x0), so
        # the teacher/fake x0-prediction gap converts to velocity with a +τ
        # factor: x0_real − x0_fake = −τ·Δ_dm. We want x_pred to move toward
        # x0_real (and the CFG-baked endpoint), so the surrogate-loss gradient
        # on x_pred is +τ·grad_signal — descent then steps x_pred by −τ·grad,
        # the desired direction. Each branch carries its OWN renoise level τ.
        tau_dm_e = tau_dm.view(B, 1, 1, 1).float()
        grad_dm = tau_dm_e * delta_dm.float()
        if cfg.dm_x0_norm:
            # Policy (b): DMD per-sample x0-space magnitude normalization. The DM
            # x0-gap is x0_real − x_renoised = −τ·v_real_cond_dm, so its
            # magnitude is denom = τ·mean|v_real|. Dividing by it cancels τ
            # across the bulk (only the clamp_min bites, for τ < norm_floor /
            # mean|v_real|) → ≈ Δ_dm / mean|v_real|. This REPLACES the τ-damping
            # in grad_dm; stacking the two is policy (c) and is NOT what this
            # does. DM term only — the CA engine below keeps its own τ_ca
            # weighting.
            denom = (
                (tau_dm_e * v_real_cond_dm.float())
                .abs()
                .mean(dim=(1, 2, 3), keepdim=True)
                .clamp_min(cfg.norm_floor)
            )
            grad_dm = grad_dm / denom
        grad_signal = grad_dm
        tau_ca_e = None
        if do_ca:
            tau_ca_e = tau_ca.view(B, 1, 1, 1).float()
            grad_signal = (
                grad_signal + tau_ca_e * (alpha_eff - 1.0) * delta_cfg.float()
            )
        grad_signal = grad_signal.detach()

        # DMD2 grad trick: a dummy scalar whose ∂/∂x_pred equals grad_signal.
        # Backward walks x_pred -> v_student -> student params; the optimizer's
        # descent step then moves x_pred along −τ·grad_signal toward x0_real.
        # Masked loss (student-only): zeroing the surrogate in background latents
        # zeroes the student push there, focusing distribution-matching on the
        # foreground. Normalization stays /numel (no renorm by mask area),
        # matching apply_masked_loss — so a masked run sees a lower effective
        # gradient.
        if mask is not None:
            loss_student = (grad_signal * x_pred.float() * mask).mean()
        else:
            loss_student = (grad_signal * x_pred.float()).mean()

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
            delta_cfg=delta_cfg,
            x_pred=x_pred,
            v_student=v_student,
            tau_dm_e=tau_dm_e,
            v_real_cond_dm=v_real_cond_dm,
            v_fake_cond_dm=v_fake_cond_dm,
        )
        if do_ca:
            metrics.accumulate_dm_to_ca(
                tau_ca_e=tau_ca_e,
                alpha_eff=alpha_eff,
                delta_cfg=delta_cfg,
                delta_dm=delta_dm,
                tau_dm_e=tau_dm_e,
            )
        if band_diag is not None:
            metrics.accumulate_band(band_diag)
        metrics.add_alpha(alpha_eff)

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
        if (step + 1) % cfg.save_every == 0 or (step + 1) == cfg.iterations:
            save_path = str(Path(cfg.output_dir) / f"{cfg.output_name}.safetensors")
            turbo.save_student(
                save_path,
                dtype=torch.bfloat16,
                metadata={
                    "ss_turbo_student_rank": str(cfg.student_rank),
                    "ss_turbo_student_steps": str(cfg.student_steps),
                    "ss_turbo_teacher_cfg": str(cfg.teacher_cfg),
                    "ss_turbo_step": str(step + 1),
                },
            )

    if writer is not None:
        writer.close()
    logger.info("turbo distillation complete.")


if __name__ == "__main__":
    main()
