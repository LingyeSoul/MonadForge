"""Screening engine: color illustration → on-manifold B&W screentoned manga.

Phase B replacement for the v0 ``mangafy.py`` (XDoG lineart + algorithmic sin-grid
halftone), which produces an *off-manifold* condition — uniform moiré tone carpeting
skin and every mid-tone. This module uses the **sketch2manga** weights (Lin Liu et al.,
arXiv 2403.08266; ``dmMaze/sketch2manga``) via ``diffusers`` so it runs locally and
batchable instead of against a Stable-Diffusion-WebUI server.

Three-stage mix (chosen after A/B'ing alternatives — see ``README.md`` Phase B):

1. **sd img2img tone.** init from the **grayscale intensity map** of the source and
   restyle at a low ``strength`` (~0.6) with the ``mangatone`` SD1.5 + no-LPIPS VAE
   and the lineart ControlNet pinning edges. This yields a *clean, aligned,
   manga-flattened* grayscale — white skin/background, simplified tonal regions —
   without redrawing (txt2img redrew the image + hallucinated backgrounds; ScreenVAE
   blurred fine detail).
2. **Screen the sd tone.** apply an algorithmic screen (:func:`mangafy._render_tone`)
   to the sd tone — the toned value range is split into a few luminance bands, each
   screened with its own pattern (dot / parallel-line / cross-hatch), so the texture
   changes where the value (명암) changes instead of one flat screen. Because the sd
   stage already flattened the tone, ink lands *selectively* — only in genuinely
   shaded regions, not carpeting bright skin the way raw-luminance cv2 mangafy did.
3. **Composite the ink line.** overlay the learned ctrlnet lineart
   (``LineartDetector``, fine) as crisp black strokes.

Net: aligned to the color target (sd), clean manga-like regions (sd flattening),
selective real screentone (per-value dot/line/cross screen over the flattened tone), crisp ink lines.

Public contract mirrors :func:`mangafy.mangafy_array` exactly — color RGB uint8
``(H, W, 3)`` → grayscale RGB uint8 ``(H, W, 3)`` of the **same** size — so it's a
drop-in engine swap in ``prep.py`` and the cond-latent shape keeps matching the
target-latent shape per stem. Stages 1–2 run at ``long_side`` and the result is
resized back to native resolution before returning.

SD weights live under ``models/sketch2manga/``
(``make easycontrol-download EASYADAPTER=colorize``). The lineart annotator
(``lllyasviel/Annotators``) is fetched lazily by ``controlnet_aux`` on first use.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from library.env import resolve_under_home
import mangafy as _mangafy  # shared _TONE_SS + band-plan constants
from mangafy_gpu import _render_tone_t  # region-wise screentone, rendered on the GPU.
# The numpy `mangafy._render_tone` dominated per-image cost (~15s on a large page: ss=4
# supersample + an argsort-based uniformize over ~14M pixels) and starved the SD GPU
# between images. The torch twin runs the same band plan (same rng draws) on-device,
# so it's structurally identical — only the sub-pixel trig/argsort math differs.

_MANGATONE = "models/sketch2manga/mangatone.ckpt"
_VAE = "models/sketch2manga/vae/mangatone_default.ckpt"
_CONTROLNET = "models/sketch2manga/control_v11p_sd15_lineart.pth"

_PROMPT = "greyscale, monochrome, screentone"
_NEGATIVE = ""
_STRENGTH = 0.6  # img2img denoise — lower = more faithful to the source structure
_TONE_PERIOD = 2.6  # fine halftone dot period (px) at the SD long-side
_TONE_WHITE_CUT = 0.90  # sd-tone ≥ this → stays white (highlights/skin, no dots)
_TONE_BLACK_CUT = 0.10  # sd-tone ≤ this → solid black

# Lazily-built singletons (the SD pipeline is ~3.5GB VRAM — build once, reuse).
_PIPE = None
_LINEART = None


def _round32(n: int) -> int:
    return int(round(n / 32)) * 32


def _sd_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """Native (H, W) → SD (H, W), long side = ``long_side``, both multiples of 32.

    Matches the upstream ``long_side_to`` helper so the screening resolution is the
    same one mangatone was conditioned at."""
    if h >= w:
        h2 = _round32(long_side)
        w2 = _round32(round(long_side * w / h))
    else:
        w2 = _round32(long_side)
        h2 = _round32(round(long_side * h / w))
    return max(h2, 32), max(w2, 32)


def _build_pipe():
    """Build (and cache) the SD1.5 + lineart-ControlNet + mangatone-VAE pipeline."""
    global _PIPE, _LINEART
    if _PIPE is not None:
        return _PIPE, _LINEART

    # Autotune conv algorithms per input shape. Worth it only because prep groups
    # images by SD-size (see ``stage_mangafy``): each distinct resolution then tunes
    # once and every later same-size page reuses it. On an unsorted, all-distinct
    # stream this would re-tune every image and lose — the sort is what makes it pay.
    torch.backends.cudnn.benchmark = True

    from controlnet_aux import LineartDetector
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        DPMSolverMultistepScheduler,
        StableDiffusionControlNetImg2ImgPipeline,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_single_file(
        str(resolve_under_home(_CONTROLNET)), torch_dtype=dtype
    )
    # mangatone.ckpt is UNet+TE only (no VAE — it ships separately), so the VAE
    # must be loaded first and passed in. This no-LPIPS decoder is the reason
    # tones come out clean, not muddy.
    vae = AutoencoderKL.from_single_file(
        str(resolve_under_home(_VAE)), torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
        str(resolve_under_home(_MANGATONE)),
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    # "DPM++ 2M SDE" (non-Karras) — the upstream sampler.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="sde-dpmsolver++"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)

    # The annotator matched to control_v11p_sd15_lineart (auto-downloads on first use).
    lineart = LineartDetector.from_pretrained("lllyasviel/Annotators").to(device)

    _PIPE, _LINEART = pipe, lineart
    return _PIPE, _LINEART


def _line01(line_pil: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Lineart PIL → float32 ``(H, W)`` in [0,1], white paper = 1, black ink = 0.

    Normalizes polarity (some detectors emit white-on-black) so the composite step
    can ``min`` it against the tone to drop ink lines in."""
    a = np.asarray(
        line_pil.convert("L").resize(size, Image.Resampling.LANCZOS), np.float32
    )
    a /= 255.0
    if a.mean() < 0.5:  # mostly black → invert to white-bg / black-line
        a = 1.0 - a
    return a


def screentone_array(
    img_rgb: np.ndarray,
    *,
    seed: int | None = None,
    steps: int = 40,
    cfg: float = 9.0,
    long_side: int = 1024,
    strength: float = _STRENGTH,
    tone_period: float = _TONE_PERIOD,
    controlnet_scale: float = 1.0,
) -> np.ndarray:
    """Color RGB uint8 ``(H, W, 3)`` → screentoned B&W RGB uint8 ``(H, W, 3)``.

    Three-stage mix (see module docstring): sd img2img tone → halftone over that
    tone → composite ctrlnet lineart. Stays pixel-aligned to the color target.
    Same size in/out (stages 1–2 run at ``long_side`` and are resized back), and a
    3-channel grayscale result for the RGB-only Qwen VAE — drop-in for
    :func:`mangafy.mangafy_array`. ``strength`` is the img2img denoise; ``seed``
    (e.g. a stable hash of the stem) makes the result reproducible and drives the
    per-value-band screen split so the model keys on structure, not one fixed dot operator."""
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb] * 3, axis=-1)
    img_rgb = img_rgb[:, :, :3]
    h0, w0 = img_rgb.shape[:2]

    pipe, lineart = _build_pipe()
    device = pipe.device

    src = Image.fromarray(img_rgb.astype(np.uint8), "RGB")
    h, w = _sd_size(h0, w0, long_side)
    src_sd = src.resize((w, h), Image.Resampling.LANCZOS)

    # Learned lineart — reused for both the ControlNet conditioning and the final
    # ink composite (compute once).
    line_pil = lineart(src_sd, detect_resolution=max(h, w), image_resolution=max(h, w))

    # Stage 1 — sd img2img tone. init = grayscale intensity map of the source so the
    # composition/background is preserved; the lineart ControlNet pins edges.
    init = src_sd.convert("L").convert("RGB")
    control = line_pil.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    sd_tone = pipe(
        prompt=_PROMPT,
        negative_prompt=_NEGATIVE,
        image=init,
        control_image=control,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=cfg,
        controlnet_conditioning_scale=controlnet_scale,
        generator=generator,
    ).images[0]

    # Stage 2 — screen the (flattened) sd tone. Per-stem region split + pattern/angle
    # jitter so the cond isn't one flat screen (several dot/line/cross zones a page).
    tone_gray = np.asarray(sd_tone.convert("L"), np.float32) / 255.0
    rng = np.random.default_rng(seed)
    short = float(min(h, w))
    period = tone_period * max(0.5, short / 900.0)
    tone = (
        _render_tone_t(
            torch.from_numpy(tone_gray).to(device=device, dtype=torch.float32),
            rng=rng,
            period=period,
            hatch_period=period * 1.25,  # line/cross only slightly coarser than dots
            ss=_mangafy._TONE_SS,
            white_cut=_TONE_WHITE_CUT,
            black_cut=_TONE_BLACK_CUT,
            kind=None,
            angle=None,
            n_bands=None,
        )
        .cpu()
        .numpy()
    )

    # Stage 3 — composite the ink line (ink wherever tone OR line are dark).
    line01 = _line01(line_pil, (w, h))
    manga = np.minimum(tone, line01)

    out = Image.fromarray((np.clip(manga, 0.0, 1.0) * 255.0).astype(np.uint8), "L")
    out = out.convert("RGB").resize((w0, h0), Image.Resampling.LANCZOS)
    return np.array(out, dtype=np.uint8)


def unload() -> None:
    """Drop the cached pipeline and free VRAM (mirrors ``prep.py`` VAE teardown)."""
    global _PIPE, _LINEART
    _PIPE = _LINEART = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
