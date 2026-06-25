import math
import random

import torch

from networks.lora_modules.base import BaseLoRAModule


class DyLoRAModule(BaseLoRAModule):
    """DyLoRA — trains multiple LoRA ranks simultaneously.

    At each training forward pass a random rank r is sampled from
    {unit, 2*unit, ..., lora_dim} and only the [:r] / [:, :r] slices of
    lora_A / lora_B are computed.  After training any rank <= lora_dim can
    be extracted.

    Ref: https://arxiv.org/abs/2202.05955
    """

    supports_conv2d = True

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
        unit=1,
        algo="",
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
        self.unit = max(1, int(unit))
        self.algo = algo

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = torch.nn.Conv2d(
                self.lora_dim, out_dim, (1, 1), (1, 1), bias=False
            )
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.org_module_ref = [org_module]
        self._fused = False

    def _random_rank(self):
        n = self.lora_dim // self.unit
        return random.randint(1, n) * self.unit

    def forward(self, x):
        if not self.enabled or getattr(self, "_fused", False):
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if not self.training:
            return org_forwarded + self._eval_delta(x, org_forwarded)

        if self._skip_module():
            return org_forwarded

        r = self._random_rank()
        scale = self.multiplier * self.alpha / r

        work = self._rank_compute_dtype(org_forwarded)
        with self._rank_autocast_context(x, work):
            x_lora = self._rebalance(x.to(work))
            if isinstance(self.lora_down, torch.nn.Linear):
                lx = torch.nn.functional.linear(
                    x_lora, self.lora_down.weight.to(work)[:r]
                )
                lx = torch.nn.functional.linear(
                    lx, self.lora_up.weight.to(work)[:, :r]
                )
            else:
                lx = torch.nn.functional.conv2d(
                    x_lora, self.lora_down.weight.to(work)[:r],
                    stride=self.lora_down.stride, padding=self.lora_down.padding,
                )
                lx = torch.nn.functional.conv2d(lx, self.lora_up.weight.to(work)[:, :r])

        return org_forwarded + (lx * scale).to(org_forwarded.dtype)

    def _eval_delta(self, x, org_forwarded):
        x_lora = self._rebalance(x)
        lx = self.lora_up(self.lora_down(x_lora))
        return lx * self.multiplier * self.scale

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        if self._has_channel_scale and down_weight.dim() == 2:
            down_weight = down_weight * self.inv_scale.to(down_weight).unsqueeze(0)

        if len(down_weight.size()) == 2:
            weight = multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            weight = (
                multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                .unsqueeze(2)
                .unsqueeze(3)
                * self.scale
            )
        else:
            conved = torch.nn.functional.conv2d(
                down_weight.permute(1, 0, 2, 3), up_weight
            ).permute(1, 0, 2, 3)
            weight = multiplier * conved * self.scale

        return weight

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w = weight.data.float()

            down_weight = sd["lora_down.weight"].to(torch.float).to(device)
            up_weight = sd["lora_up.weight"].to(torch.float).to(device)

            if "inv_scale" in sd:
                inv_scale = sd["inv_scale"].to(torch.float).to(device)
                if down_weight.dim() == 2:
                    down_weight = down_weight * inv_scale.unsqueeze(0)

            if len(w.size()) == 2:
                w += self.multiplier * (up_weight @ down_weight) * self.scale
            elif down_weight.size()[2:4] == (1, 1):
                w += (
                    self.multiplier
                    * (
                        up_weight.squeeze(3).squeeze(2)
                        @ down_weight.squeeze(3).squeeze(2)
                    )
                    .unsqueeze(2)
                    .unsqueeze(3)
                    * self.scale
                )
            else:
                conved = torch.nn.functional.conv2d(
                    down_weight.permute(1, 0, 2, 3), up_weight
                ).permute(1, 0, 2, 3)
                w += self.multiplier * conved * self.scale

            weight.data.copy_(w.to(dtype))

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data -= delta
        self._fused = False
