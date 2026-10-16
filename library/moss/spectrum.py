"""Spectrum for MOSS-Image — Chebyshev forecasting of the image-token hidden state.

The forecast target is the image-token hidden state after the last block (before
``final_norm_video``); a forecast step runs only ``time_conditioning`` + ``head``,
skipping all 42 blocks and the text refiner (Spectrum: Han et al., CVPR 2026).
The when-to-skip rule is Anima's growing window (``networks/spectrum.py``), precomputed in ``begin``
because it depends on nothing but the step count.

The estimator is ``networks/spectrum_forecast.py``'s (``SpectrumPredictor``: Chebyshev ridge fit over the
step index blended with a first-order Taylor step), written in its linear form. A
ridge prediction is ``x*ᵀ (XᵀX + λI)⁻¹ Xᵀ H``, i.e. a weighted sum ``Σ c_k H_k`` of
the stored observations, and the Taylor term is too. ``_Forecaster`` solves the
``K``-vector of weights in float64 and sums the bf16 history into an fp32
accumulator. The core instead ``torch.cat``-grows the history and casts all of it
to fp32 for every fit: at 1024² with CFG (2 × 4096 × 3072 features per step) that
cost +3 GiB of peak and allocator OOM retries. Numerically it is the core's fit
without its bf16 rounding of τ and of the coefficients.

Rows are stacked ``[cond…, uncond…]`` in one tensor. The weights are shared by
every element, so each element — and therefore each row — is forecast from its
own history only: cond and the skip-22 uncond stay separate trajectories.

Hook protocol, driven by ``library.moss.generate.denoise``::

    spectrum.begin(sigmas)
    for i in steps:
        if spectrum.is_full_step(i):
            feats = <42 blocks>; spectrum.observe(i, feats)
        else:
            feats = spectrum.forecast(i)
        v = transformer.head(feats, …)

The fit coordinate is the step index ``i`` (as in Anima), not σ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch



def window_schedule(
    num_steps: int,
    window_size: float = 2.0,
    flex_window: float = 0.25,
    warmup: int = 6,
    stop_caching_step: int = -1,
) -> list[bool]:
    """Per-step ``True`` = full forward. Warmup steps and the tail from
    ``stop_caching_step`` (−1 = ``num_steps − 3``) are always full; in between a
    full step lands every ``floor(N)`` steps, ``N`` starting at ``window_size``
    and growing by ``flex_window`` after each post-warmup full step."""
    stop_at = num_steps - 3 if stop_caching_step < 0 else stop_caching_step
    curr_ws, consec, full = window_size, 0, []
    for i in range(num_steps):
        if i < warmup or i >= stop_at:
            actual = True
        else:
            actual = (consec + 1) % max(1, math.floor(curr_ws)) == 0
        if actual:
            if i >= warmup:
                curr_ws = round(curr_ws + flex_window, 3)
            consec = 0
        else:
            consec += 1
        full.append(actual)
    return full


class _Forecaster:
    """Chebyshev (degree ``m``) ridge regression over ``t``, blended with a Taylor
    step by ``w``; the last ``k`` observations kept."""

    def __init__(self, m: int, lam: float, w: float, total_steps: int, k: int):
        if k < m + 2:
            raise ValueError("history must exceed the basis size by 2")
        self.m, self.lam, self.w, self.total, self.k = m, lam, w, total_steps, k
        self.t: list[float] = []
        self.h: list[torch.Tensor] = []

    def update(self, t: float, h: torch.Tensor) -> None:
        self.t.append(float(t))
        self.h.append(h.detach())
        if len(self.t) > self.k:
            del self.t[0], self.h[0]

    def _design(self, t: torch.Tensor) -> torch.Tensor:
        tau = (2.0 * t / self.total - 1.0).reshape(-1, 1)
        cols = [torch.ones_like(tau), tau]
        for _ in range(2, self.m + 1):
            cols.append(2 * tau * cols[-1] - cols[-2])
        return torch.cat(cols[: self.m + 1], 1)

    def weights(self, t_star: float) -> torch.Tensor:
        """``c`` with prediction ``Σ c_k h_k``, float64 ``(n,)``."""
        t = torch.tensor(self.t, dtype=torch.float64)
        x = self._design(t)
        a = x.T @ x + self.lam * torch.eye(x.shape[1], dtype=torch.float64)
        x_star = self._design(torch.tensor([t_star], dtype=torch.float64))[0]
        c = x @ torch.linalg.solve(a, x_star)
        if self.w < 1.0 and len(t) >= 2:
            dt = max(self.t[-1] - self.t[-2], 1e-8)
            kk = (t_star - self.t[-1]) / dt
            taylor = torch.zeros_like(c)
            taylor[-1], taylor[-2] = 1 + kk, -kk
            c = (1 - self.w) * taylor + self.w * c
        return c

    @torch.no_grad()
    def predict(self, t_star: float) -> torch.Tensor:
        c = self.weights(t_star).tolist()
        out = torch.zeros(self.h[0].shape, dtype=torch.float32, device=self.h[0].device)
        for ck, hk in zip(c, self.h):
            out.add_(hk, alpha=ck)
        return out.to(self.h[0].dtype)


@dataclass
class MossSpectrum:
    """Growing-window Spectrum. Defaults are Anima's shipped ones."""

    window_size: float = 2.0
    flex_window: float = 0.25
    warmup: int = 6
    stop_caching_step: int = -1
    w: float = 0.3  # Chebyshev / Taylor blend, 1.0 = pure Chebyshev
    m: int = 3  # Chebyshev degree
    lam: float = 0.1  # ridge λ
    history: int = 100  # observation window K
    calibration: float = 0.0  # residual bias correction on forecast steps

    def begin(self, sigmas) -> None:
        sigmas = torch.as_tensor(sigmas)
        n = len(sigmas) - int(bool(sigmas[-1] == 0))
        self.num_steps = n
        self.schedule = window_schedule(
            n, self.window_size, self.flex_window, self.warmup, self.stop_caching_step
        )
        self._pred: _Forecaster | None = None
        self._residual: torch.Tensor | None = None
        self.full_steps = 0
        self.forecast_steps = 0

    def is_full_step(self, i: int) -> bool:
        return self.schedule[i] or self._pred is None

    @torch.no_grad()
    def observe(self, i: int, features: torch.Tensor) -> None:
        if self._pred is None:
            self._pred = _Forecaster(self.m, self.lam, self.w, self.num_steps, self.history)
        elif self.calibration > 0 and len(self._pred.t) >= 2:
            self._residual = features - self._pred.predict(float(i))
        self._pred.update(float(i), features)
        self.full_steps += 1

    @torch.no_grad()
    def forecast(self, i: int) -> torch.Tensor:
        feats = self._pred.predict(float(i))
        if self._residual is not None:
            feats = feats + self.calibration * self._residual
        self.forecast_steps += 1
        return feats

    @property
    def planned_full_steps(self) -> int:
        return sum(self.schedule)
