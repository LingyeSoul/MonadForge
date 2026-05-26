"""Spectral Progressive Diffusion (SPD) — training-free inference acceleration.

Xiao et al., arXiv:2605.18736. Grow spatial resolution along the denoising
trajectory: run early (noise-dominated) steps at low resolution, then inject
high-frequency detail via *spectral noise expansion* only when finer
frequencies emerge from noise. The latent power spectrum decays as a power law
(`P_ω ∝ |ω|^{-β}`, β=2.26 on Anima — `bench/spd/`), so HF carries far less
signal and is cheap to defer.

This is the "Case A" / training-free path (`bench/spd/plan.md` Phase 3): the
bare DiT (or any existing LoRA checkpoint) runs the multi-resolution trajectory
through the standard inference path — no training. The math here is promoted
verbatim from the Phase-2 probe (`bench/spd/probe_lowres_denoise.py`), which
validated that the bare Anima DiT denoises low-res latents and accepts the
spectral-expansion handoff coherently (std ×0.95, no NaN, no smear).

Architecturally this mirrors ``networks/spectrum.py``: a sampler-level runner
that *replaces* the denoise loop and self-registers with
``library.inference.generation`` at import time, so ``library.inference`` keeps
no hard edge into ``networks/``. Dispatched from ``generate_body`` on
``--spd``.

v0 scope (runner + CLI, see `docs/proposal/spd_finetune_lora.md` for the
follow-on fine-tune):
  * **Euler only.** Spectral expansion re-spaces the remaining σ schedule
    mid-loop (Sec 4.3); ``ERSDESampler``/``LCMSampler`` precompute their
    coefficients from the *full* schedule at construction, so they are
    incompatible with re-spacing. The probe used plain Euler for exactly this
    reason. If a stochastic sampler is requested we fall back to Euler with a
    one-time warning.
  * **No DCW / SMC-CFG composition.** Those operate at the sampler boundary on
    the (re-spaced) σ and are unvalidated against the mid-loop reshape; passing
    them with ``--spd`` warns and ignores. (DCW for the SPD trajectory is its
    own calibration run — see the proposal "Out of scope".)
  * **Composes with LoRA / Hydra / soft-tokens / P-GRAFT** — the per-step
    adapter setters are mirrored from the standard loop, and the per-Linear
    LoRA delta is token-count-agnostic so it runs at any stage resolution.
"""

from __future__ import annotations

import logging
import math
from typing import List

import torch
from tqdm import tqdm

from library.inference.adapters import (
    compute_and_set_hydra_fei,
    set_hydra_content,
    set_hydra_crossattn,
    set_hydra_sigma,
)
from library.inference.sampler_context import SamplerSideChannels

log = logging.getLogger(__name__)


# ── DCT helpers (2D separable, type-II, pure PyTorch — matches comfyui-speed) ──
# Promoted verbatim from bench/spd/probe_lowres_denoise.py.


# The type-II DCT basis is constant for a given (n, device, dtype), and both the
# SPD sampler and the fine-tune target construction only ever see a handful of
# bucket sizes — so build each matrix once and reuse it read-only instead of
# rebuilding dense (n×n) matrices on every dct2/idct2 call. Callers must NOT
# mutate the returned tensor (dct2/idct2 only matmul against it). Math is
# bit-identical to the un-cached construction.
_DCT_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _dct_matrix(n: int, device, dtype) -> torch.Tensor:
    key = (n, device, dtype)
    m = _DCT_CACHE.get(key)
    if m is None:
        nr = torch.arange(n, device=device, dtype=dtype)
        k = nr.unsqueeze(1)
        m = torch.cos(torch.pi * k * (2 * nr + 1) / (2 * n))
        m[0] *= 1.0 / math.sqrt(n)
        m[1:] *= math.sqrt(2.0 / n)
        _DCT_CACHE[key] = m
    return m


def dct2(x: torch.Tensor) -> torch.Tensor:
    """2D type-II DCT over the last two dims of a (B, C, H, W) tensor."""
    B, C, H, W = x.shape
    Dh = _dct_matrix(H, x.device, x.dtype)
    Dw = _dct_matrix(W, x.device, x.dtype)
    y = x.reshape(B * C, H, W)
    y = Dh @ y
    y = y @ Dw.T
    return y.reshape(B, C, H, W)


def idct2(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    Dh = _dct_matrix(H, x.device, x.dtype)
    Dw = _dct_matrix(W, x.device, x.dtype)
    y = x.reshape(B * C, H, W)
    y = Dh.T @ y
    y = y @ Dw
    return y.reshape(B, C, H, W)


def _snap(v: float, mult: int) -> int:
    """Round to nearest positive multiple of ``mult`` (DiT patch_spatial)."""
    return max(mult, int(round(v / mult)) * mult)


# ── SPD spectral primitives (paper T_Φ + Eq. i–iii + Eq. 5–6) ──────────────────


def dct_lowpass_init(x5: torch.Tensor, scale: float, patch: int) -> torch.Tensor:
    """DCT low-pass of a (B,C,1,H,W) latent down to a (B,C,1,h,w) grid (paper T_Φ)."""
    B, C, T, H, W = x5.shape
    x4 = x5.squeeze(2).float()
    xi = dct2(x4)
    h = min(_snap(H * scale, patch), H)
    w = min(_snap(W * scale, patch), W)
    x_low = idct2(xi[:, :, :h, :w])
    return x_low.unsqueeze(2).to(x5.dtype)


def spectral_expand(
    x5: torch.Tensor,
    sigma_val: float,
    scale_lo: float,
    scale_hi: float,
    H_full: int,
    W_full: int,
    patch: int,
    gen: torch.Generator,
    hf_scale: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Embed the current low-res DCT block into a larger grid, fill HF slots with
    σ-scaled noise, iDCT, scale by κ (Eq. iii) and align the timestep (Eq. 5–6).

    ``hf_scale`` attenuates the fresh HF noise fill (paper prescription = 1.0).
    It is the continuity↔detail knob the SPD∘Spectrum frontier probe sweeps:
    γ→0 injects no fresh HF (max LL-feature continuity across the seam, but an
    off-manifold under-detailed state); γ=1 is the on-manifold paper default.
    Default 1.0 ⇒ bit-for-bit identical to the sampler/training path.

    Returns (expanded (B,C,1,h_hi,w_hi) latent, sigma_aligned).
    """
    B, C, T, h_lo, w_lo = x5.shape
    x4 = x5.squeeze(2).float()
    xi = dct2(x4)

    h_hi = max(_snap(H_full * scale_hi, patch), h_lo)
    w_hi = max(_snap(W_full * scale_hi, patch), w_lo)

    r = scale_hi / scale_lo
    sigma_aligned = (r * sigma_val) / (1.0 + (r - 1.0) * sigma_val)
    kappa = r / (1.0 + (r - 1.0) * sigma_val)

    xi_new = torch.zeros(B, C, h_hi, w_hi, device=x5.device, dtype=torch.float32)
    xi_new[:, :, :h_lo, :w_lo] = xi
    noise = torch.randn(
        xi_new.shape, generator=gen, device=x5.device, dtype=torch.float32
    )
    mask = torch.zeros_like(xi_new)
    mask[:, :, h_lo:, :] = 1.0
    mask[:, :, :h_lo, w_lo:] = 1.0
    xi_new = xi_new + mask * sigma_val * noise * hf_scale

    x4_new = idct2(xi_new) * kappa
    return x4_new.unsqueeze(2).to(x5.dtype), float(sigma_aligned)


# ── SPD fine-tune target construction (paper §4.3, Eq. 11–14) ──────────────────
# Shared with the training loop (``scripts/distill_spd.py``) so the train-time
# stage-entry state is built by the *same* primitives the sampler runs — the
# Phase-0 contract in ``docs/proposal/spd_finetune_lora.md`` ("diff the
# train-time x̃ against the sampler's expanded state … bit-for-bit at t_{i-1}").


def _aligned_sigma(scale_lo: float, scale_hi: float, sigma_val: float) -> float:
    """Timestep after spectral expansion (Eq. 5–6 / spectral_expand's t̃)."""
    r = scale_hi / scale_lo
    return (r * sigma_val) / (1.0 + (r - 1.0) * sigma_val)


def spd_schedule_bands(
    stages: List[float], transition_sigmas: List[float]
) -> List[tuple[float, float]]:
    """Per-stage query-σ band ``(t_lo, t_hi)`` for an SPD schedule.

    Stage ``i`` (ascending resolution) is queried over ``t ∈ (t_lo, t_hi)``:

      * ``t_lo`` = the transition to the *next* stage (0 for the final
        full-res stage).
      * ``t_hi`` = 1.0 for stage 0 (pure-noise entry); for later stages the
        **aligned** σ̃ the expansion lands at (``> t_{i-1}`` for r>1), because
        that is the σ the model is actually queried at post-expansion at
        inference (``spd_denoise`` re-spaces the schedule to σ̃).

    Bands are data-independent (functions of the schedule only), so the loop
    precomputes them once and weights stage sampling by band width to keep the
    marginal over ``t`` uniform (matches the paper's ``U(0,1)``).
    """
    S = len(stages)
    assert len(transition_sigmas) == S - 1, "transition_sigmas must be len(stages)-1"
    bands: list[tuple[float, float]] = []
    for i in range(S):
        t_lo = transition_sigmas[i] if i < S - 1 else 0.0
        if i == 0:
            t_hi = 1.0
        else:
            t_hi = _aligned_sigma(stages[i - 1], stages[i], transition_sigmas[i - 1])
        bands.append((float(t_lo), float(t_hi)))
    return bands


@torch.no_grad()
def spd_stage_target(
    x0_full: torch.Tensor,  # (B, C, 1, H, W) clean full-res VAE latent
    stage_idx: int,
    stages: List[float],
    transition_sigmas: List[float],
    patch: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct ``(x0^{s_i}, ε^{s_i})`` for an SPD stage (paper Eq. 11–12).

    Returns the clean latent low-passed to the stage scale and the *effective*
    noise field for that stage. Ordinary flow-matching at scale ``s_i`` with
    this noise field reproduces the §4.3 trajectory segment exactly: the
    straight-line velocity target is ``ε^{s_i} − x0^{s_i}`` and the training
    sample at any σ is ``(1−σ)·x0^{s_i} + σ·ε^{s_i}`` (Eq. 13 collapses to this
    because the segment velocity is constant — see the proposal derivation).

    For stage 0 the noise is plain ``N(0,I)`` at the low-res grid (the entry is
    pure noise at σ=1). For stage ``i>0`` the entry state ``x̃`` is built by
    spectrally expanding the previous stage's FM state at the transition σ —
    *via the sampler's own ``spectral_expand``* — and the effective noise is
    recovered from it: ``ε^{s_i} = (x̃ − (1−σ̃)·x0^{s_i}) / σ̃``. Its low-freq
    block carries the coherent (carried) noise; its high-freq block holds the
    κ-scaled fresh fill minus the true HF detail, so the velocity target there
    points the model toward revealing the deferred high frequencies — which is
    the train–inference gap the LoRA exists to close.
    """
    s_hi = stages[stage_idx]
    H_full, W_full = int(x0_full.shape[-2]), int(x0_full.shape[-1])
    x0_si = dct_lowpass_init(x0_full, s_hi, patch) if s_hi < 1.0 else x0_full

    if stage_idx == 0:
        eps_si = torch.randn(
            x0_si.shape, generator=gen, device=x0_si.device, dtype=torch.float32
        ).to(x0_si.dtype)
        return x0_si, eps_si

    s_lo = stages[stage_idx - 1]
    t_trans = float(transition_sigmas[stage_idx - 1])
    x0_lo = dct_lowpass_init(x0_full, s_lo, patch)
    eps_lo = torch.randn(
        x0_lo.shape, generator=gen, device=x0_lo.device, dtype=torch.float32
    ).to(x0_lo.dtype)
    x_entry_lo = (1.0 - t_trans) * x0_lo + t_trans * eps_lo
    x_tilde, t_tilde = spectral_expand(
        x_entry_lo, t_trans, s_lo, s_hi, H_full, W_full, patch, gen
    )
    # Recover the FM-consistent effective noise at scale s_i. The §4.3 κ +
    # alignment make x̃ a valid FM state at σ̃, so this inversion is exact.
    eps_si = (x_tilde.float() - (1.0 - t_tilde) * x0_si.float()) / t_tilde
    return x0_si, eps_si.to(x0_si.dtype)


@torch.no_grad()
def spd_rollout_to_stage(
    velocity_fn,
    init_noise_full: torch.Tensor,  # (B, C, 1, H, W) σ=1 full-res white noise
    stages: List[float],
    transition_sigmas: List[float],
    *,
    infer_steps: int,
    flow_shift: float,
    patch: int,
    gen: torch.Generator,
    stop_stage: int,
) -> tuple[torch.Tensor, float, float]:
    """SPD Euler prefix rollout from σ=1 down to the entry of ``stop_stage``.

    Mirrors ``spd_denoise``'s prefix *exactly* — DCT-lowpass init, Euler
    velocity steps, mid-loop ``spectral_expand`` + schedule re-spacing — but
    conditional-only (no CFG, no side-channels) and **stops just before
    expanding into ``stop_stage``**, returning the prefix's own low-res state at
    that transition. This is the on-policy counterpart of the *analytic*
    stage-entry ``spd_stage_target`` builds: instead of
    ``(1−t_trans)·x0_lo + t_trans·ε`` from the true clean LL, it returns the
    state the trained (or bare) prefix actually rolls to from pure noise.

    ``velocity_fn(x5, sigma_scalar) -> v`` is the per-step denoiser at the
    current resolution (e.g. ``forward_mini_train_dit`` with the adapter
    enabled). The σ schedule is built identically to inference
    (``sampling.get_timesteps_sigmas``) so the rollout sees the same grid the
    deployed sampler does.

    Returns ``(x_entry_lo, sigma_cross, scale_lo)``: the (B,C,1,h,w) pre-expansion
    low-res state, the live σ at the crossing into ``stop_stage``, and the
    current resolution scale. The caller applies
    ``spectral_expand(x_entry_lo, sigma_cross, scale_lo, stages[stop_stage], …)``
    to get the on-policy stage entry x̃ — the same primitive both the sampler
    and ``spd_stage_target`` use, keeping train/infer geometry bit-for-bit.
    """
    assert 1 <= stop_stage < len(stages), "stop_stage must index a non-first stage"
    H_full, W_full = int(init_noise_full.shape[-2]), int(init_noise_full.shape[-1])

    # σ schedule identical to inference (sampling.get_timesteps_sigmas).
    sigmas = torch.linspace(
        1.0, 0.0, infer_steps + 1, device=init_noise_full.device, dtype=torch.float32
    )
    sigmas = (flow_shift * sigmas) / (1.0 + (flow_shift - 1.0) * sigmas)

    cur_scale = stages[0]
    x5 = (
        dct_lowpass_init(init_noise_full, cur_scale, patch)
        if cur_scale < 1.0
        else init_noise_full
    )
    stage_idx = 0
    n = len(sigmas) - 1
    for i in range(n):
        sigma = float(sigmas[i])
        # Expand through any intermediate stage *strictly before* stop_stage.
        while (
            stage_idx < len(transition_sigmas) and sigma <= transition_sigmas[stage_idx]
        ):
            if stage_idx + 1 >= stop_stage:
                # Reached the transition into stop_stage — return pre-expansion.
                return x5, sigma, cur_scale
            nxt = stages[stage_idx + 1]
            if nxt > cur_scale:
                orig = float(sigmas[i])
                x5, sigma_new = spectral_expand(
                    x5, sigma, cur_scale, nxt, H_full, W_full, patch, gen
                )
                cur_scale = nxt
                if orig > 0 and sigma_new != orig:  # re-space remaining σ (Sec 4.3)
                    sigmas[i + 1 :] = sigma_new * (sigmas[i + 1 :] / orig)
                sigma = sigma_new
            stage_idx += 1
        v = velocity_fn(x5, sigma).float()
        dt = float(sigmas[i + 1]) - sigma
        x5 = (x5.float() + v * dt).to(x5.dtype)

    # Schedule ended before crossing into stop_stage (transition σ below σ_min):
    # return the final low-res state at the last σ.
    return x5, float(sigmas[-1]), cur_scale


# ── SPD denoise loop (Euler, velocity form, CFG, multi-resolution) ─────────────


@torch.no_grad()
def spd_denoise(
    anima,
    latents: torch.Tensor,
    timesteps: torch.Tensor,  # unused (SPD builds its own t from the live σ); kept for runner-signature parity
    sigmas: torch.Tensor,
    embed: torch.Tensor,
    negative_embed: torch.Tensor,
    padding_mask: torch.Tensor,
    guidance_scale: float,
    sampler,  # ERSDESampler / LCMSampler / None — SPD forces Euler (see module docstring)
    device: torch.device,
    ctx: SamplerSideChannels,
    *,
    stages: List[float],
    transition_sigmas: List[float],
    seed: int = 0,
) -> torch.Tensor:
    """Multi-resolution SPD denoising loop.

    ``stages`` is ascending resolution scales (e.g. ``[0.5, 1.0]``);
    ``transition_sigmas`` (len = len(stages)-1) are the σ thresholds at which to
    spectral-expand to the next stage. ``stages=[1.0]`` + ``[]`` is the plain
    full-res baseline.

    The first stage starts from a DCT low-pass of the full-res init latent; each
    transition fills the newly representable HF slots with σ-scaled noise and
    re-spaces the remaining σ schedule (Sec 4.3). ``padding_mask`` is rebuilt at
    each stage to match the new token grid.

    ``ctx`` carries the shared conditioning side-channels (see
    ``library.inference.sampler_context``). SPD v0 honors soft-tokens / P-GRAFT /
    pooled-text but ignores DCW / SMC-CFG (they act on the re-spaced σ boundary,
    unvalidated against the mid-loop reshape).
    """
    # Side-channels SPD v0 honors.
    pgraft_network = ctx.pgraft_network
    lora_cutoff_step = ctx.lora_cutoff_step
    pooled_text_pos = ctx.pooled_text_pos
    pooled_text_neg = ctx.pooled_text_neg
    soft_tokens_net = ctx.soft_tokens_net
    soft_tokens_embed_seqlens = ctx.soft_tokens_embed_seqlens
    soft_tokens_neg_seqlens = ctx.soft_tokens_neg_seqlens

    if sampler is not None:
        log.warning(
            "--spd forces Euler; the requested stochastic sampler is ignored "
            "(spectral expansion re-spaces σ mid-loop, which precomputed "
            "ER-SDE/LCM coefficients cannot follow)."
        )
    if ctx.dcw or ctx.dcw_calibrator is not None or ctx.smc_cfg is not None:
        log.warning(
            "--spd v0 does not compose with DCW / SMC-CFG (they act on the "
            "re-spaced σ boundary and are unvalidated against the mid-loop "
            "reshape); ignoring. See docs/proposal/spd_finetune_lora.md."
        )

    do_cfg = guidance_scale != 1.0
    patch = anima.patch_spatial
    H_full, W_full = latents.shape[-2], latents.shape[-1]
    sigmas = sigmas.clone().float()
    gen = torch.Generator(device=device).manual_seed(int(seed) + 10_000)

    cur_scale = stages[0]
    x5 = latents
    if cur_scale < 1.0:
        x5 = dct_lowpass_init(x5, cur_scale, patch)
    stage_idx = 0

    def _padding_mask_for(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x.shape[0], 1, x.shape[-2], x.shape[-1], dtype=torch.bfloat16, device=device
        )

    pad = _padding_mask_for(x5)

    def velocity(
        x: torch.Tensor, sigma_scalar: float, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        # timestep == σ in [0,1] for Anima flow-matching (matches generation.py
        # after its `timesteps /= 1000`).
        t = x.new_full((x.shape[0],), float(sigma_scalar))
        set_hydra_sigma(anima, t)
        compute_and_set_hydra_fei(anima, x)
        set_hydra_content(anima, embed)
        set_hydra_crossattn(anima, embed)
        if soft_tokens_net is not None:
            soft_tokens_net.append_postfix(
                embed, soft_tokens_embed_seqlens, timesteps=t
            )
        _pos_kw = (
            {"pooled_text_override": pooled_text_pos}
            if pooled_text_pos is not None
            else {}
        )
        v_c = anima(x, t, embed, padding_mask=pad_mask, **_pos_kw)
        if not do_cfg:
            return v_c
        set_hydra_content(anima, negative_embed)
        set_hydra_crossattn(anima, negative_embed)
        if soft_tokens_net is not None:
            soft_tokens_net.append_postfix(
                negative_embed, soft_tokens_neg_seqlens, timesteps=t
            )
        _neg_kw = (
            {"pooled_text_override": pooled_text_neg}
            if pooled_text_neg is not None
            else {}
        )
        v_u = anima(x, t, negative_embed, padding_mask=pad_mask, **_neg_kw)
        return v_u + guidance_scale * (v_c - v_u)

    n = len(sigmas) - 1
    with tqdm(total=n, desc=f"SPD denoising ({x5.shape[0]}x)") as pbar:
        for i in range(n):
            # P-GRAFT: disable LoRA at cutoff step (reference model takes over).
            if (
                pgraft_network is not None
                and lora_cutoff_step is not None
                and i == lora_cutoff_step
            ):
                pgraft_network.set_enabled(False)
                log.info("P-GRAFT: Disabled LoRA at step %d/%d", i, n)

            sigma = float(sigmas[i])
            # Expand through any stage whose transition σ we've crossed.
            while (
                stage_idx < len(transition_sigmas)
                and sigma <= transition_sigmas[stage_idx]
            ):
                nxt = stages[stage_idx + 1]
                if nxt > cur_scale:
                    orig = float(sigmas[i])
                    x5, sigma_new = spectral_expand(
                        x5, sigma, cur_scale, nxt, H_full, W_full, patch, gen
                    )
                    pad = _padding_mask_for(x5)
                    cur_scale = nxt
                    if orig > 0 and sigma_new != orig:  # re-space remaining σ (Sec 4.3)
                        sigmas[i + 1 :] = sigma_new * (sigmas[i + 1 :] / orig)
                    sigma = sigma_new
                stage_idx += 1

            v = velocity(x5, sigma, pad).float()
            dt = float(sigmas[i + 1]) - sigma
            x5 = (x5.float() + v * dt).to(torch.bfloat16)
            pbar.update(1)

    if cur_scale < 1.0:  # never handed off to full res — bicubic rescue so decode works
        import torch.nn.functional as F

        x5 = (
            F.interpolate(
                x5.squeeze(2).float(),
                size=(H_full, W_full),
                mode="bicubic",
                align_corners=False,
            )
            .unsqueeze(2)
            .to(torch.bfloat16)
        )
    return x5


# Side-effect registration (mirrors networks/spectrum.py:495).
from library.inference.generation import register_spd_runner  # noqa: E402

register_spd_runner(spd_denoise)
