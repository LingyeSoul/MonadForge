"""Foresight Guidance (FSG) — pre-step latent calibration toward the golden path.

FSG reframes CFG as a *fixed-point calibration*: at scheduled timesteps it runs
``K`` forward(conditional)–backward(unconditional) iterations over a long
interval ``Δσ`` to pull ``x_t → x̂_t`` onto the path where the conditional and
unconditional velocities agree, then the denoise step proceeds from ``x̂_t``.
Training-free, checkpoint-agnostic, deterministic. See
``docs/inference/fsg.md`` and ``docs/proposal/foresight_guidance.md``; the
premise/eyeball probes live in ``bench/fsg/``.

Paper: "Towards a Golden Classifier-Free Guidance Path via Foresight Fixed
Point Iterations" (NeurIPS 2025, arXiv 23177). The paper is ε-prediction + DDIM;
Anima is velocity-prediction flow-matching, so the forward-backward operator maps
onto the reversible Euler ODE (no DDIM machinery):

    v^γ  = v^u + γ·(v^c − v^u)              # CFG-guided velocity
    x'   = x  − Δσ · v^γ(x,   σ)            # denoise σ → σ−Δσ (guided)
    x''  = x' + Δσ · v^u(x',  σ−Δσ)         # re-noise back (unconditional)
    F(x) = x'' ;  iterate x ← F(x), K times

**Anima-specific band.** The paper concentrates iterations in the noisiest
stages; on Anima that is the dead zone (σ≈0.94 diverges, ρ>1). The operator
contracts only in **mid-σ**, and the contracting band **moves down with step
count**: at 20-step Euler it was [0.75, 0.85], but at the 28-step er_sde
production schedule σ≈0.84 stops contracting and the sweet spot is σ≈0.75, so
the default is the **[0.59, 0.75]** band (Plan-B calibration, bench/fsg). This is
a ``pre-step latent calibration`` seam: it mutates ``latents`` *before* the real
per-step forward and is otherwise invisible to the rest of the loop. Re-probe the
band with bench/fsg if you change ``infer_steps``.

This is a faithful port of ``bench/fsg/render_compare.py::_fsg_calibrate``; the
bench remains the calibration instrument (Phase-0 probe → band/K/Δσ/γ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from library.inference.adapters import (
    compute_and_set_hydra_fei,
    set_hydra_content,
    set_hydra_crossattn,
    set_hydra_sigma,
    set_step_expert_index,
)


@dataclass
class FSGCalibrator:
    """Forward-backward fixed-point calibrator, scheduled over a σ-band.

    Args:
        band: ``(σ_lo, σ_hi)`` — calibrate only when the step's σ falls inside.
            Default [0.59, 0.75] — the 28-step er_sde production band (Plan-B);
            the band shifts down with step count (was [0.75, 0.85] at 20-step).
        k: fixed-point iterations per scheduled step (error ~ρ^K, ρ≈0.93 ⇒
            K=3–4 captures ~all the gain). ``k=0`` makes the calibrator inert.
        d_sigma: calibration interval Δσ (the forward-backward stride; *not* the
            sampler's own per-step Δσ). Too-large Δσ is what makes σ≈0.94 diverge.
        gamma: calibration guidance γ. ``None`` → use the outer ``guidance_scale``
            passed at call time (the paper's operator uses plain γ-combine).
    """

    band: Tuple[float, float] = (0.59, 0.75)
    k: int = 3
    d_sigma: float = 0.1
    gamma: Optional[float] = None

    def __post_init__(self) -> None:
        self.k = int(self.k)
        self.d_sigma = float(self.d_sigma)
        lo, hi = float(self.band[0]), float(self.band[1])
        self.band = (lo, hi)

    def scheduled(self, sigma_i: float) -> bool:
        """True iff this step's σ is in-band and the calibrator is active (K>0)."""
        lo, hi = self.band
        return self.k > 0 and lo <= float(sigma_i) <= hi

    @staticmethod
    @torch.no_grad()
    def _velocity(anima, x, sigma, step_i, embed, padding_mask, pooled):
        """Full DiT velocity forward, mirroring ``generate_body``'s per-step call.

        Sets the hydra/step/FEI/content/crossattn context exactly as the sampler
        does (all no-ops on a base DiT), so adapter-routed checkpoints calibrate
        with the same routing the real step will use. ``embed`` selects
        conditional vs unconditional.
        """
        t_b = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=x.dtype)
        set_hydra_sigma(anima, t_b)
        set_step_expert_index(anima, step_i)
        compute_and_set_hydra_fei(anima, x)
        set_hydra_content(anima, embed)
        set_hydra_crossattn(anima, embed)
        kw = {"pooled_text_override": pooled} if pooled is not None else {}
        return anima(x, t_b, embed, padding_mask=padding_mask, **kw)

    @torch.no_grad()
    def calibrate(
        self,
        anima,
        latents: torch.Tensor,
        sigma_i: float,
        step_i: int,
        embed: torch.Tensor,
        negative_embed: torch.Tensor,
        padding_mask: torch.Tensor,
        guidance_scale: float,
        *,
        pooled_pos: Optional[torch.Tensor] = None,
        pooled_neg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``x̂`` after K forward-backward iterations, or ``latents``
        unchanged (same object, bit-exact) when this step is not scheduled.

        Costs ``3·K`` extra DiT forwards per scheduled step (v^c + v^u at σ,
        v^u at σ−Δσ). Deterministic — composes with mod-guidance / DAVE / DCW /
        CNS for free (they patch the model or the post-step x-space, both of
        which FSG's forwards inherit / precede).
        """
        if not self.scheduled(sigma_i):
            return latents

        gamma = float(guidance_scale) if self.gamma is None else float(self.gamma)
        ds = self.d_sigma
        s_lo = max(float(sigma_i) - ds, 1e-3)

        x = latents
        for _ in range(self.k):
            vc = self._velocity(
                anima, x, sigma_i, step_i, embed, padding_mask, pooled_pos
            )
            vu = self._velocity(
                anima, x, sigma_i, step_i, negative_embed, padding_mask, pooled_neg
            )
            vg = vu + gamma * (vc - vu)
            x_fwd = x - ds * vg  # denoise σ → σ−Δσ (guided)
            vu_lo = self._velocity(
                anima, x_fwd, s_lo, step_i, negative_embed, padding_mask, pooled_neg
            )
            x = x_fwd + ds * vu_lo  # invert back (uncond)
        return x.to(latents.dtype)
