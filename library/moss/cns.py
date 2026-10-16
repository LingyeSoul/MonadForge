"""CNS (colored-noise SDE, ``docs/inference/cns.md``) on MOSS-Image's packed latent.

The recolor numerics are Anima's (``library.inference.corrections.cns_core``). MOSS
adds one thing: the FFT runs on the **un-patchified** grid ``(B, 32, 2h, 2w)``, not
on the packed ``(B, 128, h, w)`` the DiT sees. On the packed grid the top octave of
the FLUX.2 latent lives in the 2×2 phase channels, out of reach of a spatial
filter, and that octave is the one CNS feeds late. The reshape is the pure
un-patchify of ``model.unpack_latents`` without its BatchNorm affine: white noise
stays white under it, and γ is per-channel scale-invariant.

γ comes from the deterministic Euler trajectory (``gamma_for_trajectory``, the
same estimator as ``scripts/calibration/gamma_probe.py``); the recolor applies on
``sampler="er_sde"`` only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from library.inference.corrections.cns_core import CNSRecolorer, radial_bins


def unpatchify(z: torch.Tensor) -> torch.Tensor:
    """``(B,128,h,w)`` → ``(B,32,2h,2w)``: ``unpack_latents`` without the BN affine."""
    b, c, h, w = z.shape
    return (
        z.reshape(b, c // 4, 2, 2, h, w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(b, c // 4, 2 * h, 2 * w)
    )


def patchify(z: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`unpatchify`."""
    b, c, h, w = z.shape
    return (
        z.view(b, c, h // 2, 2, w // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(b, c * 4, h // 2, w // 2)
    )


class MossCNS:
    """``recolor(white, σ)`` over the packed latent, for ``ERSDESampler(cns=…)``."""

    def __init__(self, core: CNSRecolorer):
        self.core = core

    @classmethod
    def from_npz(cls, path: str | Path, strength: float = 1.0) -> "MossCNS":
        return cls(CNSRecolorer.from_npz(str(path), strength=strength))

    def recolor(self, white: torch.Tensor, sigma_s: float) -> torch.Tensor:
        return patchify(self.core.recolor(unpatchify(white), sigma_s))


@torch.no_grad()
def gamma_for_trajectory(
    x0_preds: list[torch.Tensor], x0_final: torch.Tensor, n_bins: int
) -> np.ndarray:
    """γ ``(T, F)`` for one trajectory of packed ``(1,128,h,w)`` latents:
    ``1 − |X̂₀ − X₀|² / |X₀|²`` per FFT cell, channel-mean, clipped, radially binned."""
    xf = torch.fft.fft2(unpatchify(x0_final.float())[0])
    pf = xf.abs() ** 2
    eps = 1e-8 * pf.mean(dim=(-2, -1), keepdim=True) + 1e-12
    idx, _ = radial_bins(xf.shape[-2], xf.shape[-1], n_bins)
    idx = torch.from_numpy(idx).to(xf.device).reshape(-1)
    counts = torch.bincount(idx, minlength=n_bins).double().clamp_min(1.0)
    out = np.zeros((len(x0_preds), n_bins), dtype=np.float64)
    for t, xp in enumerate(x0_preds):
        xpf = torch.fft.fft2(unpatchify(xp.float())[0])
        g = (1.0 - (xpf - xf).abs() ** 2 / (pf + eps)).mean(0).clamp(0.0, 1.0)
        out[t] = (
            (
                torch.bincount(idx, weights=g.reshape(-1).double(), minlength=n_bins)
                / counts
            )
            .cpu()
            .numpy()
        )
    return out
