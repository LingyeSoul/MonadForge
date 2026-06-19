"""Forward operators ``H`` for closed-loop / feedback-aware methods.

A *forward operator* maps a generated image back into the conditioning domain —
the same transform that synthesized a task's condition from its target. For
EasyControl **colorize** that operator is ``mangafy`` (XDoG + halftone screentone),
the exact op ``easycontrol_adapters/colorization`` used to stage the training
conditions.

FlowBender (arXiv:2606.20404) consults ``H(x̂₁)`` — the condition re-extracted
from the model's *own* current clean estimate — at each step and feeds the
deviation from the given condition back as a refinement input. This module is the
thin, **non-differentiable (zero-order)** ``H`` wrapper shared by the trainer's
two-pass loop and the Phase-0 probe (``bench/flowbender/probe_alignment_drift.py``).

The operator round-trips through the VAE: ``latent → decode → H(pixels) → encode``.
Pixels live in ``[-1, 1]`` (Qwen VAE convention — ``decode_to_pixels`` returns and
``encode_pixels_to_latents`` consumes that range; the latter's ``[0,1]`` docstring
is stale, see ``bench/flowbender/probe_alignment_drift.py`` which round-trips
``[-1,1]``). ``mangafy`` itself is numpy/uint8 and per-image, so the whole op runs
under ``no_grad`` — exactly the regime the paper's zero-order branch targets
(JPEG, black-box ops): the manga halftone quantization is non-differentiable.

NB: this file lives under ``library/inference/`` but is import-clean of
``library.training`` (the inference↛training boundary test), so the trainer may
import it freely.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

# mangafy modules are flat (not a package); put their dir on sys.path the same
# way easycontrol_adapters/colorization/prep.py and the Phase-0 probe do.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANGAFY_DIR = _REPO_ROOT / "easycontrol_adapters" / "colorization"


def _ensure_mangafy_on_path() -> None:
    p = str(_MANGAFY_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


class MangafyOperator:
    """Zero-order forward operator ``H = mangafy`` over a VAE.

    ``__call__`` takes a clean-estimate latent (model latent space, 4D
    ``[B,C,H,W]`` or 5D ``[B,C,1,H,W]``) and returns the re-extracted manga
    condition encoded back into the **same** latent space the EasyControl cond
    stream consumes — directly feedable to ``network.set_feedback``.

    Args:
        vae: a loaded Anima VAE exposing ``decode_to_pixels`` /
            ``encode_pixels_to_latents`` (2D-fold recommended — caching/decode
            win, see ``project_qwen_vae_2d_fold``).
        mangafy_seed: fixed RNG seed for the halftone field so screentone phase
            is deterministic step-to-step (matches the probe's ``--mangafy_seed``).
    """

    def __init__(self, vae, *, mangafy_seed: int = 0):
        _ensure_mangafy_on_path()
        from mangafy_gpu import mangafy_tensor_gpu, set_screen_field_cache  # noqa: PLC0415

        self._mangafy_t = mangafy_tensor_gpu
        self.vae = vae
        self.mangafy_seed = int(mangafy_seed)
        # FlowBender re-mangafies the same handful of tier resolutions every step,
        # so memoize the per-resolution screen fields (meshgrid + sin + argsort) —
        # pure recompute otherwise. OFF outside this operator (one-shot staging
        # would just burn VRAM with no reuse).
        set_screen_field_cache(True)

    @torch.no_grad()
    def __call__(self, clean_latent: torch.Tensor) -> torch.Tensor:
        """``encode(mangafy(decode(x̂₁)))`` — feedback latent in cond space.

        The whole op is stop-grad (zero-order); the returned latent matches
        ``clean_latent``'s device/dtype and is 4D ``[B,C,H,W]`` (cond-stream
        layout — ``encode_cond_latent`` unsqueezes dim 2 itself). Stays fully
        on-device: ``mangafy_tensor_gpu`` takes/returns GPU tensors, so the only
        host hop left is the optional shadow-detail lift inside mangafy.
        """
        # 5D loop latent → 4D for decode. Squeeze dim 2 explicitly (the singleton
        # temporal axis), never squeeze() — B=1 would corrupt the batch axis.
        x = clean_latent
        if x.dim() == 5:
            x = x.squeeze(2)
        device, dtype = clean_latent.device, clean_latent.dtype

        pixels = self.vae.decode_to_pixels(x)  # [B,3,H,W] in [-1,1]
        if pixels.dim() == 5:  # defensive: 5D latent fed → 5D pixels
            pixels = pixels.squeeze(2)

        manga_chw = []
        for b in range(pixels.shape[0]):
            # [-1,1] → [0,1] CHW, mangafy on-device, back to [-1,1]
            rgb01 = (pixels[b].clamp(-1.0, 1.0) + 1.0) * 0.5
            manga01 = self._mangafy_t(rgb01, seed=self.mangafy_seed)
            manga_chw.append(manga01 * 2.0 - 1.0)
        manga = torch.stack(manga_chw, dim=0)  # [B,3,H,W] in [-1,1]

        feedback_latent = self.vae.encode_pixels_to_latents(manga)
        if feedback_latent.dim() == 5:
            feedback_latent = feedback_latent.squeeze(2)
        return feedback_latent.to(device=device, dtype=dtype)


def build_forward_operator(
    name: str, vae, *, mangafy_seed: int = 0
) -> Optional[MangafyOperator]:
    """Resolve a forward operator by name. ``"mangafy"`` is the only operator
    today (colorize); returns None for unknown/disabled names so callers can
    treat absence as "open-loop"."""
    if name in ("mangafy", "colorize"):
        return MangafyOperator(vae, mangafy_seed=mangafy_seed)
    return None
