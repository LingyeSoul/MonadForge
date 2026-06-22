"""Spectrum: Adaptive Spectral Feature Forecasting for Anima inference acceleration.

Implements the Spectrum method (Han et al., CVPR 2026) for training-free diffusion
sampling acceleration via Chebyshev polynomial feature forecasting.

Instead of running all transformer blocks at every denoising step, Spectrum:
1. Observes block outputs at a subset of steps (actual forwards)
2. Fits Chebyshev polynomial coefficients via ridge regression
3. Forecasts block outputs at skipped steps (cached)
4. Runs only t_embedder + final_layer + unpatchify on cached steps

Core forecasting algorithm adapted from:
  Spectrum (Han et al., CVPR 2026) — https://github.com/yangheng95/Spectrum
  Original source: src/utils/basis_utils.py
"""

import math
import logging
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from library.inference.adapters import clear_hydra_sigma, set_hydra_sigma
from library.inference import sampling as inference_utils
from library.inference.sampler_context import SamplerSideChannels
from networks.spectrum_sea import (
    l1rel,
    sea_filter,
    solve_delta_for_refresh_ratio,
)

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16

# Auto-δ calibration cache for the SEA schedule. Keyed by the schedule geometry
# (num_steps, warmup, stop_at, refresh_ratio); the first generate with
# ``--spectrum_delta auto`` runs the legacy window schedule while recording the
# SEA distance trace, derives δ to match the target refresh fraction, and caches
# it here so subsequent generates use the SEA trigger at matched compute.
# Mirrored to disk (``output/spectrum_sea_delta.json``) so the calibration
# survives across separate CLI processes — a one-process many-prompt run (the
# bench harness) only ever calibrates on the first prompt via the in-memory
# dict. See docs/inference/spectrum.md §"SEA schedule" ("The δ knob").
_AUTO_DELTA_CACHE: dict = {}


def _auto_delta_store_path():
    from library.env import anima_home

    return anima_home() / "output" / "spectrum_sea_delta.json"


def _auto_delta_lookup(key: tuple) -> Optional[float]:
    """In-memory then on-disk lookup of a calibrated δ for ``key``."""
    if key in _AUTO_DELTA_CACHE:
        return _AUTO_DELTA_CACHE[key]
    import json

    path = _auto_delta_store_path()
    try:
        with open(path) as f:
            disk = json.load(f)
    except (OSError, ValueError):
        return None
    val = disk.get("_".join(str(k) for k in key))
    if val is not None:
        val = float(val)
        _AUTO_DELTA_CACHE[key] = val  # promote to in-memory for this process
    return val


def _auto_delta_save(key: tuple, value: float) -> None:
    """Persist a calibrated δ to both the in-memory and on-disk caches."""
    _AUTO_DELTA_CACHE[key] = value
    import json

    path = _auto_delta_store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path) as f:
                disk = json.load(f)
        except (OSError, ValueError):
            disk = {}
        disk["_".join(str(k) for k in key)] = value
        with open(path, "w") as f:
            json.dump(disk, f, indent=2)
    except OSError as e:
        logger.warning("Spectrum SEA: could not persist auto-delta to %s (%s)", path, e)


def _window_decision_fraction(
    num_steps: int,
    warmup_steps: int,
    stop_at: int,
    window_size: float,
    flex_window: float,
) -> float:
    """Refresh fraction the growing-window schedule spends in the decision region.

    Replays the exact window rule (the ``else`` branch below + its curr_ws
    advance) and returns ``actual_decision_steps / decision_steps``. The SEA
    auto-δ target defaults to *this* so the SEA arm is a like-for-like swap at
    matched compute for any step count — the hard-coded 0.62 in the proposal was
    only the 24-step value and over-computes elsewhere (_archive/bench/
    spectrum_sea/prompt_generalization.py: 0.62 → +22% forwards at 28 steps).
    """
    curr_ws = window_size
    consec = 0
    actual_dec = 0
    n_dec = 0
    for i in range(num_steps):
        if i < warmup_steps or i >= stop_at:
            actual = True
        else:
            actual = (consec + 1) % max(1, math.floor(curr_ws)) == 0
            n_dec += 1
            actual_dec += int(actual)
        if actual:
            if i >= warmup_steps:
                curr_ws = round(curr_ws + flex_window, 3)
            consec = 0
        else:
            consec += 1
    return actual_dec / max(1, n_dec)


def _flatten(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    shape = x.shape
    return x.reshape(1, -1), shape


def _unflatten(x_flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return x_flat.reshape(shape)


class ChebyshevForecaster:
    """Chebyshev T-polynomial ridge regression forecaster.

    Maintains a sliding window of (t, feature) observations, fits Chebyshev
    polynomial coefficients via ridge regression (Cholesky solve), and predicts
    features at arbitrary timesteps.

    Args:
        M: Number of Chebyshev basis functions (degree).
        K: Maximum window size (number of observations to keep).
        lam: Ridge regression regularization strength.
        device: Torch device for buffers.
    """

    def __init__(
        self,
        M: int = 4,
        K: int = 10,
        lam: float = 1e-3,
        device: Optional[torch.device] = None,
        total_steps: int = 30,
    ):
        assert K >= M + 2, "K should exceed basis size for stability"
        self.M = M
        self.K = K
        self.lam = lam
        self.device = device
        self.total_steps = total_steps

        self.t_buf = torch.empty(0)  # (<=K,)
        self._H_buf: Optional[torch.Tensor] = None  # (<=K, F)
        self._shape: Optional[torch.Size] = None
        self._coef: Optional[torch.Tensor] = None  # (P, F)

    @property
    def P(self) -> int:
        return self.M + 1

    def _taus(self, t: torch.Tensor) -> torch.Tensor:
        """Map step index t ∈ [0, total_steps) to τ ∈ [-1, 1]."""
        return 2.0 * (t / self.total_steps) - 1.0

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        """Build Chebyshev design matrix [T0, T1, ..., TM] via recurrence."""
        taus = taus.reshape(-1, 1)
        K = taus.shape[0]
        T0 = torch.ones((K, 1), device=taus.device, dtype=taus.dtype)
        if self.M == 0:
            return T0
        T1 = taus
        cols = [T0, T1]
        for _ in range(2, self.M + 1):
            Tm = 2 * taus * cols[-1] - cols[-2]
            cols.append(Tm)
        return torch.cat(cols[: self.M + 1], dim=1)

    def update(self, t: float, h: torch.Tensor) -> None:
        """Append observation (t, h) to the sliding window."""
        device = self.device or h.device
        t_tensor = torch.as_tensor(t, dtype=DTYPE, device=device)
        h_flat, shape = _flatten(h)
        h_flat = h_flat.to(device)

        if self._shape is None:
            self._shape = shape
        else:
            assert shape == self._shape, "Feature shape must remain constant"

        if self.t_buf.numel() == 0:
            self.t_buf = t_tensor[None]
            self._H_buf = h_flat
        else:
            self.t_buf = torch.cat([self.t_buf, t_tensor[None]], dim=0)
            self._H_buf = torch.cat([self._H_buf, h_flat], dim=0)
            if self.t_buf.numel() > self.K:
                self.t_buf = self.t_buf[-self.K :]
                self._H_buf = self._H_buf[-self.K :]

        self._coef = None

    def _fit_if_needed(self) -> None:
        if self._coef is not None:
            return
        taus = self._taus(self.t_buf)
        X = self._build_design(taus).to(torch.float32)
        H = self._H_buf.to(torch.float32)
        P = X.shape[1]

        lamI = self.lam * torch.eye(P, device=X.device, dtype=X.dtype)
        Xt = X.T
        XtX = Xt @ X + lamI
        try:
            L = torch.linalg.cholesky(XtX)
        except torch.linalg.LinAlgError:
            jitter = 1e-6 * XtX.diag().mean()
            L = torch.linalg.cholesky(
                XtX + jitter * torch.eye(P, device=X.device, dtype=X.dtype)
            )
        XtH = Xt @ H
        self._coef = torch.cholesky_solve(XtH, L).to(DTYPE)

    @torch.no_grad()
    def predict(self, t_star: torch.Tensor) -> torch.Tensor:
        """Predict feature at timestep t_star via Chebyshev regression."""
        assert self._shape is not None
        self._fit_if_needed()
        tau_star = self._taus(t_star)
        x_star = self._build_design(tau_star[None])  # (1, P)
        h_flat = x_star @ self._coef  # (1, F)
        return _unflatten(h_flat, self._shape)


class SpectrumPredictor:
    """Chebyshev polynomial forecaster with optional first-order Taylor blending.

    Wraps ChebyshevForecaster and blends with a discrete Newton forward-difference
    extrapolation for improved stability on the most recent observations.
    """

    def __init__(
        self,
        m: int,
        lam: float,
        w: float,
        device: torch.device,
        feature_shape,
        total_steps: int = 30,
    ):
        self.cheb = ChebyshevForecaster(
            M=m, K=100, lam=lam, device=device, total_steps=total_steps
        )
        self.w = w

    def update(self, t: float, h: torch.Tensor):
        self.cheb.update(t, h)

    @torch.no_grad()
    def predict(self, t_star: float) -> torch.Tensor:
        device = self.cheb.t_buf.device
        t_star_t = torch.as_tensor(t_star, dtype=DTYPE, device=device)

        h_cheb = self.cheb.predict(t_star_t)

        if self.w >= 1.0 or self.cheb.t_buf.numel() < 2:
            return h_cheb

        # First-order discrete Taylor (Newton forward difference)
        H = self.cheb._H_buf  # (K, F) flattened
        t = self.cheb.t_buf
        dt = (t[-1] - t[-2]).clamp_min(1e-8)
        k = ((t_star_t - t[-1]) / dt).to(H.dtype)
        h_taylor = (H[-1] + k * (H[-1] - H[-2])).reshape(h_cheb.shape)

        return (1 - self.w) * h_taylor + self.w * h_cheb


def _spectrum_fast_forward(
    model, timesteps_B_T: torch.Tensor, predicted_feature: torch.Tensor
) -> torch.Tensor:
    """Fast path: t_embedder -> final_layer -> unpatchify (skips all blocks)."""
    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)
    t_emb, adaln = model.t_embedder(timesteps_B_T)
    t_emb = model.t_embedding_norm(t_emb)
    # Unconditional: buffer is zeros when mod guidance is disabled (see
    # Anima.__init__), so this collapses to identity.
    t_emb = t_emb + model._mod_guidance_delta.unsqueeze(1)
    x = model.final_layer(predicted_feature, t_emb, adaln_lora_B_T_3D=adaln)
    return model.unpatchify(x)


def spectrum_denoise(
    anima,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    embed: torch.Tensor,
    negative_embed: torch.Tensor,
    padding_mask: torch.Tensor,
    guidance_scale: float,
    sampler,  # ERSDESampler / LCMSampler / None — anything with .step(latents, denoised, i)
    device: torch.device,
    ctx: SamplerSideChannels,
    *,
    window_size: float = 2.0,
    flex_window: float = 0.25,
    warmup_steps: int = 6,
    w: float = 0.3,
    m: int = 3,
    lam: float = 0.1,
    stop_caching_step: int = -1,
    calibration_strength: float = 0.0,
    schedule: str = "window",
    delta: Optional[float] = None,
    refresh_ratio: float = -1.0,
    sea_beta: float = 2.0,
) -> torch.Tensor:
    """Spectrum-accelerated denoising loop.

    Replaces the standard step-by-step denoising with adaptive scheduling:
    early steps (high noise) get more actual forwards; later steps (refinement)
    are increasingly predicted via Chebyshev polynomial fitting.

    Args:
        window_size: Initial window N — actual forward every floor(N) steps.
        flex_window: Growth rate alpha — N += alpha after each actual forward.
        warmup_steps: Number of initial steps that always run actual forwards.
        w: Chebyshev/Taylor blend weight (1.0 = pure Chebyshev).
        m: Number of Chebyshev basis functions.
        lam: Ridge regression regularization strength.
        stop_caching_step: Force actual forwards from this step onward (-1 = auto: total_steps - 3).
        calibration_strength: Residual calibration strength (0.0 = disabled). On actual forwards,
            computes residual = actual - predicted; on cached steps, adds residual * strength.
        schedule: When-to-skip rule. ``"window"`` (default) = the content-blind
            growing window. ``"sea"`` = accumulate the SEA-filtered relative-L1
            distance of the input latent across steps and refresh when it crosses
            ``delta`` (SeaCache Eq. 4/8; only the *decision* changes — the
            forecast+head reuse path is untouched, so SMC-CFG / mod-guidance / DCW
            composition is unaffected). ``window_size``/``flex_window`` are unused
            in SEA mode.
        delta: SEA threshold. ``None`` or ``<=0`` = auto-calibrate to
            ``refresh_ratio`` (the first generate runs the window schedule while
            recording the distance trace, derives δ, and caches it; subsequent
            generates use the SEA trigger). A positive float pins δ explicitly
            (for sweeps).
        refresh_ratio: Target post-warmup refresh fraction for auto-δ. ``<=0``
            (default) matches the growing-window schedule's own refresh fraction
            at this exact (num_steps, warmup, stop, window_size, flex_window) —
            a true like-for-like swap at matched compute. A positive float pins
            the target explicitly (for sweeps). Do not hard-code: the window
            fraction varies with step count (~0.45 at 28 steps, ~0.62 at 24).
        sea_beta: Natural-image power-law exponent for the SEA filter (default 2).
        ctx: Shared conditioning side-channels (DCW / SMC-CFG / soft-tokens /
            P-GRAFT / pooled-text) — see ``library.inference.sampler_context``.
    """
    # Unpack the shared side-channels into the locals the loop body uses.
    pgraft_network = ctx.pgraft_network
    lora_cutoff_step = ctx.lora_cutoff_step
    pooled_text_pos = ctx.pooled_text_pos
    pooled_text_neg = ctx.pooled_text_neg
    dcw = ctx.dcw
    dcw_lambda = ctx.dcw_lambda
    dcw_schedule = ctx.dcw_schedule
    dcw_band_mask = ctx.dcw_band_mask
    dcw_calibrator = ctx.dcw_calibrator
    smc_cfg = ctx.smc_cfg
    soft_tokens_net = ctx.soft_tokens_net
    soft_tokens_embed_seqlens = ctx.soft_tokens_embed_seqlens
    soft_tokens_neg_seqlens = ctx.soft_tokens_neg_seqlens

    do_cfg = guidance_scale != 1.0
    num_steps = len(timesteps)

    curr_ws = window_size
    consec_cached = 0
    fwd_count = 0
    stop_at = num_steps - 3 if stop_caching_step < 0 else stop_caching_step

    # SEA schedule (SeaCache decision metric). delta_val is None while the
    # accumulator is uncalibrated (auto mode, first generate) — the loop then
    # falls back to the window rule and records the distance trace for δ tuning.
    use_sea = schedule == "sea"
    auto_delta = use_sea and (delta is None or float(delta) <= 0.0)
    # Default the auto-δ target to the window schedule's own refresh fraction at
    # this geometry (like-for-like at matched compute); a positive override is
    # honored for sweeps.
    if use_sea and float(refresh_ratio) <= 0.0:
        refresh_ratio = _window_decision_fraction(
            num_steps, warmup_steps, stop_at, window_size, flex_window
        )
    # δ rides the input-latent trajectory, so it must be re-calibrated whenever
    # the trajectory changes: step count, CFG scale, sampler rule, and latent
    # resolution all move it (the L1rel-normalized SEA gain is only *roughly*
    # scale-stable). refresh_ratio is the target itself. Prompt is deliberately
    # excluded — fixed δ + per-prompt-varying refresh pattern is the whole point
    # of content-adaptivity. A new config just triggers one window-scheduled
    # calibration pass, then the cached δ kicks in.
    sampler_label = type(sampler).__name__ if sampler is not None else "euler"
    sea_cache_key = (
        num_steps,
        warmup_steps,
        stop_at,
        round(float(refresh_ratio), 4),
        round(float(guidance_scale), 3),
        sampler_label,
        int(latents.shape[-2]),
        int(latents.shape[-1]),
    )
    if not use_sea:
        delta_val: Optional[float] = None
    elif auto_delta:
        delta_val = _auto_delta_lookup(sea_cache_key)
    else:
        delta_val = float(delta)
    sea_prev: Optional[torch.Tensor] = None
    sea_accum = 0.0
    sea_dists: list = []  # per-step distances over decision steps, for auto-δ

    # Forecasters (created lazily on first actual forward)
    cond_fc: Optional[SpectrumPredictor] = None
    uncond_fc: Optional[SpectrumPredictor] = None

    cond_residual: Optional[torch.Tensor] = None
    uncond_residual: Optional[torch.Tensor] = None

    # Register hook on final_layer to capture block output (its input)
    captured = {}

    def _capture_pre_hook(module, args):
        # args[0] = x_B_T_H_W_D (block output, after static unpadding)
        captured["feat"] = args[0].detach().clone()

    hook = anima.final_layer.register_forward_pre_hook(_capture_pre_hook)

    try:
        with tqdm(total=num_steps, desc="Spectrum") as pbar:
            for i, t in enumerate(timesteps):
                # P-GRAFT cutoff
                if (
                    pgraft_network is not None
                    and lora_cutoff_step is not None
                    and i == lora_cutoff_step
                ):
                    pgraft_network.set_enabled(False)
                    logger.info(f"P-GRAFT: Disabled LoRA at step {i}/{num_steps}")

                # SEA: accumulate the SEA-filtered relative-L1 distance of the
                # input latent (x_t == `latents`, shared across cond/uncond so
                # one accumulator drives both branches). One FFT/iFFT per step —
                # negligible vs a block forward, zero extra DiT forwards.
                if use_sea:
                    sea_now = sea_filter(latents[:, :, 0], float(sigmas[i]), sea_beta)
                    if sea_prev is not None:
                        d = l1rel(sea_now, sea_prev)
                        sea_accum += d
                        if warmup_steps <= i < stop_at:
                            sea_dists.append(d)
                    sea_prev = sea_now

                # Decide: actual forward or cached prediction?
                if i < warmup_steps or i >= stop_at:
                    actual = True
                elif use_sea and delta_val is not None:
                    actual = sea_accum >= delta_val
                else:
                    actual = (consec_cached + 1) % max(1, math.floor(curr_ws)) == 0

                t_exp = t.expand(latents.shape[0])
                set_hydra_sigma(anima, t_exp)
                if dcw_calibrator is not None:
                    # Capture FEI on the pre-forward latent at warmup steps
                    # for v6 fei_obs={replace,concat} artifacts. Spectrum forces
                    # actual forwards while i < warmup_steps, so this lines up
                    # with the v4 g_obs capture at line 409 below.
                    dcw_calibrator.record_latent_pre_forward(i, latents)

                if actual:
                    if soft_tokens_net is not None:
                        soft_tokens_net.append_postfix(
                            embed, soft_tokens_embed_seqlens, timesteps=t_exp
                        )
                    with torch.no_grad():
                        _pos_kw = (
                            {"pooled_text_override": pooled_text_pos}
                            if pooled_text_pos is not None
                            else {}
                        )
                        noise_pred = anima(
                            latents, t_exp, embed, padding_mask=padding_mask, **_pos_kw
                        )
                    feat = captured["feat"]
                    if cond_fc is None:
                        cond_fc = SpectrumPredictor(
                            m, lam, w, device, feat.shape[1:], num_steps
                        )
                    # Residual calibration: measure prediction error before updating
                    if calibration_strength > 0 and cond_fc.cheb.t_buf.numel() >= 2:
                        cond_residual = feat - cond_fc.predict(float(i))
                    cond_fc.update(float(i), feat)

                    if do_cfg:
                        if soft_tokens_net is not None:
                            soft_tokens_net.append_postfix(
                                negative_embed,
                                soft_tokens_neg_seqlens,
                                timesteps=t_exp,
                            )
                        with torch.no_grad():
                            _neg_kw = (
                                {"pooled_text_override": pooled_text_neg}
                                if pooled_text_neg is not None
                                else {}
                            )
                            uncond_noise_pred = anima(
                                latents,
                                t_exp,
                                negative_embed,
                                padding_mask=padding_mask,
                                **_neg_kw,
                            )
                        ufeat = captured["feat"]
                        if uncond_fc is None:
                            uncond_fc = SpectrumPredictor(
                                m, lam, w, device, ufeat.shape[1:], num_steps
                            )
                        if (
                            calibration_strength > 0
                            and uncond_fc.cheb.t_buf.numel() >= 2
                        ):
                            uncond_residual = ufeat - uncond_fc.predict(float(i))
                        uncond_fc.update(float(i), ufeat)
                        if smc_cfg is not None:
                            noise_pred = smc_cfg.combine(
                                noise_pred, uncond_noise_pred, guidance_scale
                            )
                        else:
                            noise_pred = uncond_noise_pred + guidance_scale * (
                                noise_pred - uncond_noise_pred
                            )

                    # Advance schedule (only post-warmup to avoid inflating window)
                    if i >= warmup_steps:
                        curr_ws = round(curr_ws + flex_window, 3)
                    consec_cached = 0
                    sea_accum = 0.0  # refresh resets the SEA accumulator (Eq. 8)
                    fwd_count += 1
                    pbar.set_postfix(mode="fwd", ws=f"{curr_ws:.1f}", n=fwd_count)

                else:
                    # Cached step: predict features, skip all blocks.
                    with torch.no_grad():
                        pred_feat = cond_fc.predict(float(i))
                        if cond_residual is not None:
                            pred_feat = pred_feat + calibration_strength * cond_residual
                        noise_pred = _spectrum_fast_forward(anima, t_exp, pred_feat)

                        if do_cfg:
                            upred_feat = uncond_fc.predict(float(i))
                            if uncond_residual is not None:
                                upred_feat = (
                                    upred_feat + calibration_strength * uncond_residual
                                )
                            uncond_noise_pred = _spectrum_fast_forward(
                                anima, t_exp, upred_feat
                            )
                            if smc_cfg is not None:
                                noise_pred = smc_cfg.combine(
                                    noise_pred, uncond_noise_pred, guidance_scale
                                )
                            else:
                                noise_pred = uncond_noise_pred + guidance_scale * (
                                    noise_pred - uncond_noise_pred
                                )

                    consec_cached += 1
                    pbar.set_postfix(mode="cached", n=fwd_count)

                denoised = latents.float() - sigmas[i] * noise_pred.float()
                if sampler is not None:
                    new_latents = sampler.step(latents, denoised, i)
                else:
                    new_latents = inference_utils.step(latents, noise_pred, sigmas, i)

                # DCW v4: observe post-CFG noise_pred + maybe fire the head.
                # Warmup observations all land within Spectrum's warmup window
                # (Spectrum forces actual forwards while i < warmup_steps), so
                # v4 sees real-DiT velocities even when caching kicks in later.
                if dcw_calibrator is not None:
                    dcw_calibrator.record(i, noise_pred)
                    dcw_calibrator.fire_head_if_due(i)

                # DCW: bias-correct against denoised x0_pred (carries Spectrum's
                # cached-step prediction error, but correction is bias-agnostic).
                if float(sigmas[i + 1]) > 0.0 and (dcw_calibrator is not None or dcw):
                    from networks.dcw import apply_dcw, parse_band_mask

                    if dcw_calibrator is not None:
                        lam_i_calib = dcw_calibrator.lambda_for_step(
                            i, float(sigmas[i])
                        )
                        new_latents = apply_dcw(
                            new_latents.float(),
                            denoised,
                            float(sigmas[i]),
                            lam=lam_i_calib,
                            schedule="const",
                            bands=frozenset({"LL"}),
                        )
                    else:
                        new_latents = apply_dcw(
                            new_latents.float(),
                            denoised,
                            float(sigmas[i]),
                            lam=dcw_lambda,
                            schedule=dcw_schedule,
                            bands=parse_band_mask(dcw_band_mask),
                        )

                latents = new_latents.to(latents.dtype)

                pbar.update()

        # Auto-δ: this generate ran the window schedule while recording the SEA
        # distance trace; derive the δ that matches the target refresh fraction
        # and cache it so subsequent generates use the SEA trigger.
        if auto_delta and delta_val is None and sea_dists:
            new_delta = solve_delta_for_refresh_ratio(sea_dists, refresh_ratio)
            _auto_delta_save(sea_cache_key, new_delta)
            logger.info(
                "Spectrum SEA: auto-calibrated delta=%.4g (target refresh_ratio="
                "%.2f over %d decision steps); this generate used the window "
                "schedule — subsequent generates (incl. new processes, via "
                "%s) use the SEA trigger.",
                new_delta,
                refresh_ratio,
                len(sea_dists),
                _auto_delta_store_path().name,
            )

        speedup = num_steps / max(1, fwd_count)
        cfg_label = " (x2 for CFG)" if do_cfg else ""
        sched_label = (
            f", schedule=sea (delta={delta_val:.4g})"
            if use_sea and delta_val is not None
            else (", schedule=sea (calibrating)" if use_sea else "")
        )
        logger.info(
            f"Spectrum: {fwd_count}/{num_steps} actual forwards "
            f"({speedup:.2f}x theoretical speedup{cfg_label}){sched_label}"
        )

    finally:
        clear_hydra_sigma(anima)
        if pgraft_network is not None and lora_cutoff_step is not None:
            pgraft_network.set_enabled(True)
        hook.remove()

    return latents


# Register with library.inference.generation so generate() can dispatch to us
# without holding a hard import edge from generation.py back into this file.
from library.inference.generation import register_spectrum_runner  # noqa: E402

register_spectrum_runner(spectrum_denoise)
