# VeRA (Vector-based Random Matrix Adaptation) — ΔW = diag(d) @ A @ B @ diag(b).
# Shared frozen random matrices A, B across layers; per-layer learnable
# diagonal scaling vectors vera_d and vera_b. ~10-100x fewer trainable
# parameters than standard LoRA.

import math
from typing import Dict

import torch

from networks.lora_modules.base import BaseLoRAModule


class VeRAModule(BaseLoRAModule):
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

        if isinstance(org_module, torch.nn.Conv2d):
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self._is_conv = True
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self._is_conv = False

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.vera_d = torch.nn.Parameter(torch.ones(lora_dim))
        self.vera_b = torch.nn.Parameter(torch.ones(out_dim))

        self._vera_seed: int | None = None

        self.org_module_ref = [org_module]
        self._fused = False

    def set_shared_matrices(self, seed: int) -> None:
        """Seed frozen random A, B matrices deterministically."""
        self._vera_seed = seed
        gen = torch.Generator()
        gen.manual_seed(seed)
        A = torch.empty(self.lora_dim, self.in_dim)
        B = torch.zeros(self.out_dim, self.lora_dim)
        torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=gen)
        self.register_buffer("A", A, persistent=False)
        self.register_buffer("B", B, persistent=False)

    def forward(self, x):
        if not self.enabled or getattr(self, "_fused", False):
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if not self.training:
            return org_forwarded + self._eval_delta(x, org_forwarded)

        if self._skip_module():
            return org_forwarded

        work = self._rank_compute_dtype(org_forwarded)
        with self._rank_autocast_context(x, work):
            x_r = self._rebalance(x.to(work))

            if self._is_conv:
                B_s, C, H, W = x_r.shape
                # (B, C, H, W) -> (B*H*W, C) @ A.T -> (B*H*W, r) -> scale -> @ B.T -> (B*H*W, out) -> (B, out, H, W)
                x_flat = x_r.permute(0, 2, 3, 1).reshape(-1, self.in_dim)
                h = torch.mm(x_flat, self.A.T.to(work)) * self.vera_d.to(work)
                lx = torch.mm(h, self.B.T.to(work)) * self.vera_b.to(work)
                lx = lx.reshape(B_s, H, W, self.out_dim).permute(0, 3, 1, 2)
            else:
                h = torch.mm(x_r, self.A.T.to(work)) * self.vera_d.to(work)
                lx = torch.mm(h, self.B.T.to(work)) * self.vera_b.to(work)

            if self.dropout is not None:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

        return org_forwarded + (lx * self.multiplier * self.scale).to(org_forwarded.dtype)

    def _eval_delta(self, x, org_forwarded):
        x_r = self._rebalance(x)

        if self._is_conv:
            B_s, C, H, W = x_r.shape
            x_flat = x_r.permute(0, 2, 3, 1).reshape(-1, self.in_dim)
            h = torch.mm(x_flat, self.A.T) * self.vera_d
            lx = torch.mm(h, self.B.T) * self.vera_b
            lx = lx.reshape(B_s, H, W, self.out_dim).permute(0, 3, 1, 2)
        else:
            h = torch.mm(x_r, self.A.T) * self.vera_d
            lx = torch.mm(h, self.B.T) * self.vera_b

        return lx * self.multiplier * self.scale

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        sd = {
            "vera_d": self.vera_d.data,
            "vera_b": self.vera_b.data,
            "vera_seed": torch.tensor(self._vera_seed),
        }
        delta = self._reconstruct_delta(sd, self.vera_d.device)
        return delta * multiplier * self.scale

    def _reconstruct_delta(self, sd, device) -> torch.Tensor:
        """Rebuild the full VeRA delta from scaling vectors and seed."""
        vera_d = sd["vera_d"].to(torch.float).to(device)
        vera_b = sd["vera_b"].to(torch.float).to(device)
        seed = int(sd["vera_seed"].item())

        gen = torch.Generator()
        gen.manual_seed(seed)
        A = torch.empty(self.lora_dim, self.in_dim)
        torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=gen)
        A = A.to(torch.float).to(device)
        B = torch.zeros(self.out_dim, self.lora_dim, device=device)

        # Forward: delta = (x @ A.T * d) @ B.T * b => ΔW = diag(b) @ B @ diag(d) @ A
        delta = vera_b.unsqueeze(1) * B * vera_d.unsqueeze(0) @ A
        return delta

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            vera_d = sd["vera_d"].to(torch.float).to(device)
            vera_b = sd["vera_b"].to(torch.float).to(device)
            seed = int(sd["vera_seed"].item())

            gen = torch.Generator()
            gen.manual_seed(seed)
            A = torch.empty(self.lora_dim, self.in_dim)
            torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=gen)
            A = A.to(torch.float).to(device)
            B = torch.zeros(self.out_dim, self.lora_dim, device=device)

            # ΔW = diag(b) @ B @ diag(d) @ A
            delta = vera_b.unsqueeze(1) * B * vera_d.unsqueeze(0) @ A

            w = weight.data.float()
            w += self.multiplier * delta * self.scale
            weight.data.copy_(w.to(dtype))

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        # Reconstruct full delta and bake in
        sd = {
            "vera_d": self.vera_d.data,
            "vera_b": self.vera_b.data,
            "vera_seed": torch.tensor(self._vera_seed),
        }
        delta = self._reconstruct_delta(sd, org_module.weight.device)
        org_module.weight.data += (self.multiplier * delta * self.scale).to(org_module.weight.dtype)
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        sd = {
            "vera_d": self.vera_d.data,
            "vera_b": self.vera_b.data,
            "vera_seed": torch.tensor(self._vera_seed),
        }
        delta = self._reconstruct_delta(sd, org_module.weight.device)
        org_module.weight.data -= (self.multiplier * delta * self.scale).to(org_module.weight.dtype)
        self._fused = False

    def distill_save_state_dict(self, prefix: str, state_dict: Dict[str, torch.Tensor]) -> None:
        state_dict[f"{prefix}.vera_d"] = self.vera_d.data.cpu()
        state_dict[f"{prefix}.vera_b"] = self.vera_b.data.cpu()
        state_dict[f"{prefix}.alpha"] = self.alpha.cpu()
        state_dict[f"{prefix}.vera_seed"] = torch.tensor(self._vera_seed)
