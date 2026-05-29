"""Fake (critic) head-start loop.

DMD2 generator-outruns-critic transient: the student LR warmup completes at
~0.02·iterations, so the student takes its first full-strength steps while the
zero-init fake LoRA is still ≈ the teacher (delta_dm large + misaligned) → the
early grad_signal_rms / v_student_rms spike (~step 50). Pre-train the fake net
for ``warmup_steps`` fake-only updates against the student's (init ≈ teacher)
x_pred distribution so the critic is calibrated before the student moves.

The student is left untouched (no-grad forward, no student optimizer step). The
fake scheduler IS stepped through this phase (caller sizes it over ``iterations
+ warmup_steps``), so the fake's 0.02 LR warmup overlaps the head-start and the
fake enters the main loop at full LR with a calibrated critic.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterator

import torch
import torch.nn as nn
from tqdm import tqdm

from .primitives import renoise, sample_t

logger = logging.getLogger(__name__)


def run_fake_warmup(
    *,
    warmup_steps: int,
    turbo,  # TurboDMDNetwork
    forward_fn: Callable[..., torch.Tensor],
    data_iter: Iterator,
    dataloader: torch.utils.data.DataLoader,
    fake_opt: torch.optim.Optimizer,
    fake_sched: torch.optim.lr_scheduler.LRScheduler,
    fake_steps_per_student_step: int,
    grad_clip: float,
    t_distribution: str,
    sigmoid_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    log_interval: int,
    writer,
) -> Iterator:
    """Run the critic head-start; returns the (possibly advanced) data_iter."""
    if warmup_steps <= 0:
        return data_iter

    logger.info(f"fake (critic) head-start: {warmup_steps} fake-only updates")
    for cw in tqdm(range(warmup_steps), desc="fake-warmup"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        latents, crossattn_emb = batch[1], batch[2]
        latents = latents.to(device, dtype=dtype, non_blocking=True)
        crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
        B = latents.shape[0]
        torch.compiler.cudagraph_mark_step_begin()

        t = sample_t(
            B, distribution=t_distribution, sigmoid_scale=sigmoid_scale,
            device=device, dtype=dtype,
        )
        eps = torch.randn_like(latents)
        t_e = t.view(B, 1, 1, 1)
        x_t = (1.0 - t_e) * latents + t_e * eps
        # Student (init) x_pred — no grad: the student is not trained here.
        with torch.no_grad():
            v_student = forward_fn(
                "student", x_t, t, crossattn_emb, no_grad=True
            ).squeeze(2)
            x_pred_d = (x_t.squeeze(2) - t_e * v_student).detach()

        cw_loss = torch.zeros((), device=device)
        for _ in range(fake_steps_per_student_step):
            tau_fake = sample_t(
                B, distribution=t_distribution, sigmoid_scale=sigmoid_scale,
                device=device, dtype=dtype,
            )
            eps_fake = torch.randn_like(x_pred_d)
            x_t_fake = renoise(x_pred_d, tau_fake, eps_fake).requires_grad_()
            v_fake = forward_fn(
                "fake", x_t_fake, tau_fake, crossattn_emb, no_grad=False
            ).squeeze(2)
            target_v_fake = eps_fake - x_pred_d
            fake_loss = nn.functional.mse_loss(
                v_fake.float(), target_v_fake.float()
            )
            fake_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    turbo.fake_params(), max_norm=grad_clip
                )
            fake_opt.step()
            fake_opt.zero_grad(set_to_none=True)
            fake_sched.step()
            cw_loss = cw_loss + fake_loss.detach()
        if writer is not None and (cw + 1) % log_interval == 0:
            writer.add_scalar(
                "warmup/fake_loss",
                (cw_loss / fake_steps_per_student_step).item(),
                cw + 1,
            )

    return data_iter
