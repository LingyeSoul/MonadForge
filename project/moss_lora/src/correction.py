"""Correction training (arXiv 2610.14136) on MOSS-Image as a second LoRA.

The denoiser D is the DiT with a frozen LoRA; the corrector C is the same DiT
with D's LoRA plus a corrector LoRA chained after it, so C starts as a copy of D
(the paper's init) at the cost of two adapters rather than two models. MOSS is
v-predicting, so C follows the paper's gated native-space form (App. B.2,
eq. 8): it reads D's clean estimate ``x̂ = z_t − σ·v₀`` at D's timestep and
caption, and

    v₁ = v₀ + γ ⊙ (C(x̂, t) − v₀),    loss = ‖v₁ − (ε − z₀)‖²

with ``γ`` per packed channel (128), zero at init. Sampling repeats the update
``K`` times per step (``x̂ₖ = z_t − σ·vₖ``).

That form stalls here: C ≈ D at init reads the clean ``x̂`` as a noisy input at
``t``, so ``C − v₀`` is far off (γ init 0.5: L_corr +95 % over D at step 50,
the gate closing ~0.07 per 50 steps). ``corr_sigma`` > 0 switches to the
clean-sample form (``correct_once``), where C at init is the base model
cleaning ``x̂`` at a low noise level.

``γ = gate · gate_scale``: Adam's step on ``gate`` is ``lr`` whatever its
gradient, so ``gate_scale`` sets ``γ``'s own step size (the LoRA lr would move
it ~0.03 over a 600-step run). ``gate_init`` > 0 starts the gate open: at 0
the LoRA's gradient is scaled by ``γ`` ≈ 0 and stays noise (caststation, 600
steps: γ 0.006 → 0.012 over steps 50–200, L_corr −0.1 %). The file is a plain kohya LoRA; ``γ`` goes to a
``<stem>.gate.safetensors`` sidecar.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from library.moss.lora import LoRANetwork
from library.moss.train import NUM_TRAIN_TIMESTEPS

LATENT_CHANNELS = 128


def gate_path(path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}.gate.safetensors")


def _chained_forward(adapter, prev):
    def forward(x: torch.Tensor) -> torch.Tensor:
        return adapter(x, prev(x))

    return forward


class CorrectorLoRA(LoRANetwork):
    """A LoRA applied on top of whatever already patches each target (D's
    LoRA), plus the output gate ``γ``."""

    gate_scale = 1.0
    gate_init = 0.0  # initial γ (the paper's 0 = C off until the gate opens)
    corr_sigma = 0.0  # 0 = native v-space at t; > 0 = clean-sample form at σ_c

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = nn.Parameter(
            torch.full((LATENT_CHANNELS,), self.gate_init / self.gate_scale)
        )

    def gamma(self) -> torch.Tensor:
        return self.gate * self.gate_scale

    def apply_to(self) -> int:
        if self._applied:
            raise RuntimeError("already applied")
        for name, module in self._targets:
            module.forward = _chained_forward(
                self.adapters[name.replace(".", "_")], module.forward
            )
        self._applied = True
        return len(self._targets)

    def load_state_dict(self, state, strict: bool = True, assign: bool = False):
        state = dict(state)
        state.setdefault("gate", self.gate.detach().clone())
        return super().load_state_dict(state, strict=strict, assign=assign)

    def save(self, path, dtype: str = "bf16", **extra_meta) -> None:
        from safetensors.torch import save_file

        gate = self._parameters.pop("gate")
        try:
            super().save(path, dtype, corrector="gated_v", **extra_meta)
        finally:
            self._parameters["gate"] = gate
        gamma = (gate * self.gate_scale).detach().float().cpu().contiguous()
        corr_sigma = torch.tensor([float(self.corr_sigma)])
        save_file({"gamma": gamma, "corr_sigma": corr_sigma}, str(gate_path(path)))

    @classmethod
    def load(cls, transformer, path) -> CorrectorLoRA:
        """Hooked on ``transformer`` after D, ``γ`` from the sidecar, multiplier 0
        (the denoiser pass); compile the blocks after."""
        from safetensors.torch import load_file

        net = cls.for_inference(transformer, path, multiplier=0.0)
        side = load_file(str(gate_path(path)))
        net.gate_scale = 1.0
        net.gate.data = side["gamma"].to(net.gate.device)
        net.corr_sigma = float(side["corr_sigma"]) if "corr_sigma" in side else 0.0
        return net


def correct_once(transformer, net, z, sigma, t, embeds, v, mask=None, ref=None):
    """One gated correction of ``v`` at state ``z``, the corrector already on.

    ``net.corr_sigma`` 0: the paper's native-space form, C reads ``x̂`` at ``t``
    and ``v ← v + γ(C − v)``. > 0: clean-sample form (App. B.2), C denoises
    ``x̂`` as an image at noise ``σ_c = min(corr_sigma, σ)``,
    ``x̂ ← x̂ + γ(x̂ − σ_c·C − x̂)``, and ``v = (z − x̂)/σ``. Capping ``σ_c`` at
    ``σ`` keeps the v-space change ≤ γ·|C| (``σ_c/σ`` ≤ 1) on the low-σ steps."""
    g = net.gamma().view(1, -1, 1, 1)
    x = z - sigma * v
    if net.corr_sigma <= 0:
        vc = transformer(x, t, embeds, mask, return_dict=False, ref=ref)[0].float()
        return v + g * (vc - v)
    sc = min(float(net.corr_sigma), float(sigma))
    tc = torch.full_like(t, NUM_TRAIN_TIMESTEPS * sc)
    vc = transformer(x, tc, embeds, mask, return_dict=False, ref=ref)[0].float()
    x = x - g * (sc * vc)
    return (z - x) / sigma


@torch.no_grad()
def corrected_velocity(transformer, net, z, sigma, t, embeds, mask, v0, k: int):
    """``K`` corrections of ``v0`` at state ``z`` (fp32 in, fp32 out)."""
    v = v0
    if k <= 0:
        return v
    net.set_multiplier(1.0)
    try:
        for _ in range(k):
            v = correct_once(transformer, net, z, sigma, t, embeds, v, mask)
    finally:
        net.set_multiplier(0.0)
    return v
