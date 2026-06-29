"""Tier-0 cross-attn-drive probe: log the CFG guidance-delta per denoising step.

The guidance delta ``v_cond - v_uncond`` is the text-driven component of the
flow-matching velocity — exactly what conditioning adds on top of the base
prior. ``||delta|| / ||v_cond||`` as a function of sigma is a cheap, zero-hook
proxy for *how much the cross-attention pathway drives the update* at each noise
level (the cross-attn-vs-self/latent question). The cosine between ``v_cond`` and
``v_uncond`` is the complementary directional read: cos→1 means text only rescales
the base velocity, cos<1 means text actively rotates it.

This is the cheapest tier of the cross-attn-drive measurement. It is whole-network
and only captures the guidance-visible effect (the part that differs cond vs
uncond) — for per-block / per-depth resolution use the cross-attn knockout probe.

Opt-in and zero-cost when off: set ``ANIMA_CFG_DELTA_LOG=<path.json>`` to enable.
Norms are reduced per-sample (all dims except batch) then mean-pooled, so the
curve is batch-size invariant.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch


class CfgDeltaProbe:
    """Accumulates per-step CFG guidance-delta diagnostics and dumps them to JSON.

    Created via :meth:`maybe_create` (returns ``None`` unless the env var is set),
    fed one row per step via :meth:`record`, and flushed once via :meth:`flush`.

    A run may invoke the denoise loop many times in one process (e.g. ``--from_file``
    runs one loop per distinct prompt). Each loop builds its own probe and flushes;
    flushes **accumulate** into the JSON as separate generations rather than
    overwriting. The first flush of a process truncates any stale file (keyed on the
    process-global ``_gen_counter`` starting at 0), so re-running is clean.
    """

    # process-global: how many generations have flushed so far this process
    _gen_counter = 0

    def __init__(self, path: str) -> None:
        self.path = path
        self.rows: list[dict] = []

    @classmethod
    def maybe_create(cls) -> Optional["CfgDeltaProbe"]:
        path = os.environ.get("ANIMA_CFG_DELTA_LOG")
        return cls(path) if path else None

    @staticmethod
    def _per_sample_norm(x: torch.Tensor) -> torch.Tensor:
        # reduce every dim except batch (dim 0) -> one scalar per sample
        return x.float().flatten(1).norm(dim=1)

    def record(
        self,
        step: int,
        sigma: float,
        v_cond: torch.Tensor,
        v_uncond: torch.Tensor,
    ) -> None:
        vc = self._per_sample_norm(v_cond)
        vu = self._per_sample_norm(v_uncond)
        delta = self._per_sample_norm(v_cond - v_uncond)
        cos = torch.nn.functional.cosine_similarity(
            v_cond.float().flatten(1), v_uncond.float().flatten(1), dim=1
        )
        ratio = delta / vc.clamp_min(1e-12)
        self.rows.append(
            {
                "step": int(step),
                "sigma": float(sigma),
                "v_cond_norm": float(vc.mean()),
                "v_uncond_norm": float(vu.mean()),
                "delta_norm": float(delta.mean()),
                # text-drive proxy: fraction of the conditional velocity that the
                # prompt is responsible for at this sigma.
                "ratio": float(ratio.mean()),
                # 1.0 => text only rescales the base velocity; <1 => it rotates it.
                "cos_cond_uncond": float(cos.mean()),
            }
        )

    def flush(self) -> None:
        if not self.rows:
            return
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        # First generation of the process starts fresh (drops any stale file);
        # later generations append so multi-prompt runs accumulate.
        payload = {"generations": []}
        if CfgDeltaProbe._gen_counter > 0 and os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                payload = {"generations": []}
        payload["generations"].append(
            {"gen": CfgDeltaProbe._gen_counter, "steps": self.rows}
        )
        CfgDeltaProbe._gen_counter += 1
        with open(self.path, "w") as f:
            json.dump(payload, f, indent=2)
