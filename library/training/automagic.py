"""Automagic optimizer — Adafactor-style per-parameter adaptive LR.

Adapted from Ostris AI Toolkit's Automagic optimizer (MIT License,
Copyright 2024 Ostris, LLC). Local version keeps core per-parameter LR
adaptation while avoiding AI Toolkit-specific dependencies.
Source: https://github.com/scvxzf1/anima_lora_webui
"""

from __future__ import annotations

import logging
import random
from typing import Iterable

import torch

logger = logging.getLogger(__name__)


class Automagic(torch.optim.Optimizer):
    """Adafactor-style optimizer with per-parameter adaptive LR masks.

    Starts from a small LR and adjusts an element-wise LR mask based on
    update polarity agreement: consecutive same-direction updates bump LR
    up, direction flips bump LR down. External schedulers are no-ops.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 1e-6,
        eps: float | tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
        do_paramiter_swapping: bool = False,
        paramiter_swapping_factor: float = 0.1,
        do_parameter_swapping: bool | None = None,
        parameter_swapping_factor: float | None = None,
    ) -> None:
        if min_lr <= 0:
            raise ValueError("min_lr must be positive")
        if max_lr < min_lr:
            raise ValueError("max_lr must be >= min_lr")
        if lr_bump <= 0:
            raise ValueError("lr_bump must be positive")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if clip_threshold <= 0:
            raise ValueError("clip_threshold must be positive")
        if lr > max_lr:
            logger.warning("Automagic lr=%s > max_lr=%s; clamping.", lr, max_lr)
            lr = max_lr
        if lr < min_lr:
            logger.warning("Automagic lr=%s < min_lr=%s; clamping.", lr, min_lr)
            lr = min_lr
        if do_parameter_swapping is not None:
            do_paramiter_swapping = do_parameter_swapping
        if parameter_swapping_factor is not None:
            paramiter_swapping_factor = parameter_swapping_factor

        defaults = {
            "lr": float(lr),
            "min_lr": float(min_lr),
            "max_lr": float(max_lr),
            "lr_bump": float(lr_bump),
            "eps": eps,
            "clip_threshold": float(clip_threshold),
            "beta2": float(beta2),
            "weight_decay": float(weight_decay),
        }
        super().__init__(params, defaults)
        self.do_paramiter_swapping = bool(do_paramiter_swapping)
        self.paramiter_swapping_factor = float(paramiter_swapping_factor)
        self._total_parameter_size = sum(
            p.numel() for group in self.param_groups for p in group["params"]
        )
        logger.info(
            "Automagic trainable parameters: %s", f"{self._total_parameter_size:,}"
        )
        if self.do_paramiter_swapping:
            self.enable_paramiter_swapping(self.paramiter_swapping_factor)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _first_eps(eps: float | tuple[float, float] | list[float]) -> float:
        if isinstance(eps, tuple | list):
            return float(eps[0])
        return float(eps)

    @staticmethod
    def _approx_sq_grad(
        exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor
    ) -> torch.Tensor:
        row_mean = exp_avg_sq_row.mean(dim=-1, keepdim=True).clamp_min_(1e-30)
        r_factor = (exp_avg_sq_row / row_mean).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.clamp_min(1e-30).unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def enable_paramiter_swapping(self, factor: float = 0.1) -> None:
        self.do_paramiter_swapping = True
        self.paramiter_swapping_factor = float(factor)
        self.swap_paramiters()

    def enable_parameter_swapping(self, factor: float = 0.1) -> None:
        self.enable_paramiter_swapping(factor)

    def swap_paramiters(self) -> None:
        all_params = [p for group in self.param_groups for p in group["params"]]
        for param in all_params:
            param.requires_grad_(False)
            param.grad = None
        random.shuffle(all_params)
        target = int(self._total_parameter_size * self.paramiter_swapping_factor)
        active = 0
        for param in all_params:
            if active >= target:
                break
            param.requires_grad_(True)
            active += param.numel()

    def swap_parameters(self) -> None:
        self.swap_paramiters()

    def get_learning_rates(self) -> list[float]:
        lrs: list[float] = []
        for group in self.param_groups:
            group_lrs = [
                float(self.state[p]["avg_lr"])
                for p in group["params"]
                if p in self.state and "avg_lr" in self.state[p]
            ]
            lrs.append(
                sum(group_lrs) / len(group_lrs) if group_lrs else float(group["lr"])
            )
        return lrs

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else 0.0

    def _init_state(self, p: torch.Tensor, grad: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        state["lr_mask"] = torch.full_like(
            p.detach(), float(group["lr"]), dtype=torch.float32, device=p.device
        )
        state["avg_lr"] = float(group["lr"])
        state["last_polarity"] = torch.zeros_like(
            p.detach(), dtype=torch.bool, device=p.device
        )
        state["RMS"] = 0.0
        if grad.ndim >= 2:
            state["exp_avg_sq_row"] = torch.zeros(
                grad.shape[:-1], dtype=torch.float32, device=grad.device
            )
            state["exp_avg_sq_col"] = torch.zeros(
                grad.shape[:-2] + grad.shape[-1:],
                dtype=torch.float32,
                device=grad.device,
            )
        else:
            state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)

    def _ensure_state(self, p: torch.Tensor, grad: torch.Tensor, group: dict) -> dict:
        state = self.state[p]
        if not state or "lr_mask" not in state or "last_polarity" not in state:
            self._init_state(p, grad, group)
            state = self.state[p]
        if grad.ndim >= 2:
            if (
                "exp_avg_sq_row" not in state
                or state["exp_avg_sq_row"].shape != grad.shape[:-1]
            ):
                state["exp_avg_sq_row"] = torch.zeros(
                    grad.shape[:-1], dtype=torch.float32, device=grad.device
                )
            else:
                state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(
                    device=grad.device, dtype=torch.float32
                )
            col_shape = grad.shape[:-2] + grad.shape[-1:]
            if (
                "exp_avg_sq_col" not in state
                or state["exp_avg_sq_col"].shape != col_shape
            ):
                state["exp_avg_sq_col"] = torch.zeros(
                    col_shape, dtype=torch.float32, device=grad.device
                )
            else:
                state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(
                    device=grad.device, dtype=torch.float32
                )
        elif "exp_avg_sq" not in state or state["exp_avg_sq"].shape != grad.shape:
            state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)
        else:
            state["exp_avg_sq"] = state["exp_avg_sq"].to(
                device=grad.device, dtype=torch.float32
            )
        state["lr_mask"] = state["lr_mask"].to(device=grad.device, dtype=torch.float32)
        state["last_polarity"] = state["last_polarity"].to(
            device=grad.device, dtype=torch.bool
        )
        return state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta2 = float(group["beta2"])
            eps = self._first_eps(group["eps"])
            clip_threshold = float(group["clip_threshold"])
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Automagic does not support sparse gradients")
                grad = p.grad.detach().to(dtype=torch.float32)
                state = self._ensure_state(p, grad, group)
                state["step"] += 1
                p_data_fp32 = p.detach().to(dtype=torch.float32)
                state["RMS"] = float(self._rms(p_data_fp32))
                update = grad.square().add(eps)
                if grad.ndim >= 2:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2
                    )
                    exp_avg_sq_col.mul_(beta2).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2
                    )
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
                    update = exp_avg_sq.rsqrt().mul_(grad)
                update.div_((self._rms(update) / clip_threshold).clamp_(min=1.0))
                current_polarity = update > 0
                sign_agreement = torch.where(
                    state["last_polarity"] == current_polarity, 1.0, -1.0
                )
                state["last_polarity"] = current_polarity
                lr_mask = state["lr_mask"]
                new_lr = torch.where(
                    sign_agreement > 0,
                    lr_mask + float(group["lr_bump"]),
                    lr_mask - float(group["lr_bump"]),
                ).clamp_(min=float(group["min_lr"]), max=float(group["max_lr"]))
                state["lr_mask"] = new_lr
                state["avg_lr"] = float(new_lr.mean().item())
                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32 * (-float(group["weight_decay"])) * new_lr
                    )
                p_data_fp32.add_(-update * new_lr)
                p.copy_(p_data_fp32.to(dtype=p.dtype))
        return loss
