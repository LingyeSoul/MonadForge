import logging

import torch
import bitsandbytes as bnb

logger = logging.getLogger(__name__)


class AdamW8bitKahan(bnb.optim.AdamW8bit):
    """AdamW 8-bit with Kahan compensated summation.

    The 8-bit quantized Adam states (exp_avg, exp_avg_sq) introduce per-step
    rounding noise in the parameter update. Over long runs the noise accumulates
    and degrades convergence. Kahan summation tracks this rounding error in a
    per-parameter ``shift`` buffer and folds it back into the next update,
    preserving ~11-12 effective bits from 8-bit storage.

    Compensated update per step (classic Kahan pattern):
        y = delta - shift            # subtract prior rounding error
        t = p_old + y                # compensated addition
        shift = (t - p_old) - y      # capture new rounding error
        p.data = t

    Extra kwargs (passed via ``--optimizer_args``):
        stabilize:             cap lr when gradient diverges from Adam's
                               running second-moment estimate
        kahan_buffer_offload:  keep shift buffers in CPU RAM
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, optim_bits=32, args=None,
                 min_8bit_size=4096, percentile_clipping=100, block_wise=True,
                 is_paged=False, stabilize=False, kahan_buffer_offload=False):
        super().__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            amsgrad=amsgrad, optim_bits=optim_bits, args=args,
            min_8bit_size=min_8bit_size, percentile_clipping=percentile_clipping,
            block_wise=block_wise, is_paged=is_paged,
        )
        self.stabilize = stabilize
        self.kahan_buffer_offload = kahan_buffer_offload

    def _cap_lr(self, group):
        lr = group["lr"]
        beta2 = group["betas"][1]
        eps = group["eps"]
        max_ratio = 0.0

        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "state2" not in state or state["state2"].dtype == torch.uint8:
                continue
            step = state.get("step", 1)
            bias_corr = 1 - beta2 ** step
            exp_avg_sq = state["state2"].float() / bias_corr
            grad = p.grad.float()
            ratio = (grad.pow(2) / (exp_avg_sq + eps)).mean().sqrt().item()
            max_ratio = max(max_ratio, ratio)

        if max_ratio > 1.0:
            return lr / max_ratio
        return lr

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        original_lrs = {}
        if self.stabilize:
            for i, group in enumerate(self.param_groups):
                original_lrs[i] = group["lr"]
                group["lr"] = self._cap_lr(group)

        # save p_old for every parameter that has a gradient
        saved_p = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    saved_p[id(p)] = p.data.clone()

        # base AdamW8bit step (modifies p.data in-place via 8-bit CUDA kernels)
        super().step()

        # Kahan compensated summation on the result
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "kahan_shift" not in state:
                    state["kahan_shift"] = torch.zeros_like(p.data)
                shift = state["kahan_shift"]
                if shift.device != p.device:
                    shift = shift.to(p.device, non_blocking=True)

                p_old = saved_p[id(p)]
                delta = p.data - p_old
                y = delta - shift
                p.data.copy_(p_old)
                p.data.add_(y)
                new_shift = (p.data - p_old) - y

                if self.kahan_buffer_offload:
                    state["kahan_shift"] = new_shift.cpu()
                else:
                    state["kahan_shift"] = new_shift

        if self.stabilize:
            for i, group in enumerate(self.param_groups):
                group["lr"] = original_lrs[i]

        return loss
