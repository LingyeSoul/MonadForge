"""Channel-attribution bench for the mod-guidance pooled-text path.

WHY THIS EXISTS
---------------
`docs/findings/mod_guidance_quality_tag_axis.md` measured everything in
`pooled_text_proj` GEOMETRY (cosines between projected pooled vectors) and never
sampled an image. Two leaps were never tested:

  1. "quality axis" — the axis it maps is really a *content-magnitude* axis
     (an arbitrary artist tag drives it 3-4x harder than `score_9`, which makes
     no sense for a quality lever). The doc half-admits this ("isn't pure
     quality ... correlates with strong, specific content").
  2. "double-drive degrades quality / DC-blowout" — inferred from cosines,
     never observed in image space.

A tag edit does not act through one channel. It enters the DiT through TWO
separable inputs:

    pooled / mod channel :  crossattn_emb.max(1) -> pooled_text_proj -> AdaLN
    cross-attn channel   :  the full crossattn_emb sequence -> cross-attention

The pooled channel is permutation-invariant (max over the sequence) and collapses
to a single vector; the cross-attn channel is order- and shape-sensitive. They
are *separable at the forward* via `pooled_text_override` (models.py:1643): we can
run cross-attn from prompt A while feeding the pooled vector of prompt B.

This bench answers the live, mechanistic questions in IMAGE / LATENT space:

    swap       Causal channel decomposition of a tag edit. For (base, base+tag)
               render the 2x2 of {cross=base|tag} x {pool=base|tag} and split the
               image movement into a cross-attn delta and a pooled delta; measure
               whether the two channels reinforce (preload), conflict (cancel), or
               are orthogonal. THE KEY EXPERIMENT.
    order      Pure cross-attn isolator. Permute the tag order: the pooled vector
               is provably identical (max is order-invariant), so 100% of any image
               movement is cross-attn. Compares against the same-prompt seed floor.
    intensity  Pooled-channel response curve. Sweep the mod-guidance steering
               weight w (the doc's actual double-drive mechanism) and measure
               off-baseline movement + the DC-blowout proxy (pixel spatial std /
               tone shift). Directly tests "does a hard push drift quality worse".
    origin     Base-vs-distill split of the pooled channel's tag response. The
               pooled->AdaLN path is 100% distillation-induced (zero-init proj +
               enable_pooled_text_modulation=False in the base DiT -> identity), and
               distillation never trained on quality tags. So there is NO native
               pooled path to swap in; the meaningful split is the BASE-MODEL
               upstream pooled movement (max(crossattn_emb), pure Qwen3 + LLM-adapter,
               no trained proj) vs the DISTILLED proj's gain on it. Tells us whether
               quality-selectivity originates upstream (proj = tag-agnostic
               pass-through -> teacher-quality-conditioning has little headroom) or
               in the distilled proj (it amplifies quality -> headroom exists).
               Render-free -- pure pooled-vector geometry, but in the mechanistically
               exact spot the demoted geometry-only finding never isolated.

    sigma_window  PHASE 0 of docs/findings/mod_guidance_quality_tag_axis.md (§ Schedule axis).
               [CLOSED 2026-05-31: C1 DEAD -- the grade can't live in the σ<0.45 tail;
               it saturates ~4x short of uniform even dose-matched. Tool kept, axis dead.]
               The shipped steering schedule is per-block but σ-BLIND (applied at
               every denoising step). This gates that same schedule to an σ-band and
               renders the swap-prompt set under: off (unguided), uniform (σ-blind,
               shipped), a SWEEP of low windows (σ<0.55 / <0.45 / <0.35) each in two
               dose flavours -- equal_w (fixed w, isolates per-step leverage) and
               equal_dose (w boosted by uniform_steps/window_steps, matches integrated
               steering) -- and a σ>=0.45 high complement. The dose split matters: at
               flow_shift=3 the σ<0.45 tail is only ~4/20 steps, so an equal-w tail
               arm under-doses ~5x vs uniform; without the dose-matched arm "tail ≈
               off" could falsely kill the σ axis. Readout vs unguided: pixel SSIM
               (structure preserved), LF/HF latent-energy split of the steering delta
               (where the effect landed), structural J, grids. Tests C1 -- can the
               grade live in the tail (low ≈ uniform effect at HIGHER SSIM, esp.
               dose-matched) or does the tail lack leverage. No new architecture.

    layer_window  PHASE 0b of docs/findings/mod_guidance_quality_tag_axis.md (§ Schedule axis).
               [CLOSED 2026-05-31: C2 FALSIFIED -- it's pure DOSE, not placement. Partial
               arms interpolate off<->full (weaker-fulls, not different-fulls); full [8-27]
               wins, its movement is correction not drift. Shipped 8-26 full-dose validated.
               Tool kept for future schedule probes, axis dead.]
               Phase 0 killed the σ axis; the LAYER axis (the source paper's actual
               contribution) is the live thread. Anima ships only a hand-set 8-26 block
               range, never image-validated per-block. This steers ONE block at a time
               (single mode: [l, l+1)) -- or a cumulative [8..L] range (marginal-per-
               block view) -- at the shipped w, and reads delta_norm + pixel SSIM-to-
               unguided per block. Tests C2: does layer placement separate GRADE (effect
               at HIGH SSIM) from DRIFT (effect at LOW SSIM)? Flat map -> no layer lever,
               proposal closes; differentiated -> the grade set is the "proper layers"
               (a static / aspect-LUT / training-free-online schedule becomes real).
               Same readout as sigma_window; only the steered block range varies.

All experiments save image grids -- READ THE GRIDS, the scalar metrics are a guide,
not the verdict (cf. the pose-blind PE-cosine lesson elsewhere in this repo).

Outputs land in bench/mod_guidance/results/<ts>[-label]/ via bench/_common.py.

Run
---
    uv run python bench/mod_guidance/channel_attribution.py \
        --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
        --experiment all --label probe

    # smoke (1 prompt, 1 tag, fewer steps)
    uv run python bench/mod_guidance/channel_attribution.py \
        --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
        --experiment swap --prompts "1girl, solo, outdoors" --tags "score_9" \
        --infer_steps 12 --label smoke

    # Phase 0: σ-window ablation of the shipped steering schedule (C1 test).
    # Sweeps low windows x {equal_w, equal_dose} so a narrow tail window isn't
    # falsely killed by under-dosing. Read the grids at --grid_thumb 768.
    uv run python bench/mod_guidance/channel_attribution.py \
        --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
        --experiment sigma_window --dataset_samples 6 --seeds 0,1 \
        --sigwin_dose both --compile --label phase0

    # Phase 0b: per-block grade-vs-drift map of the shipped steering schedule.
    # Steers one block at a time over the shipped 8-26 band; read SSIM (structure
    # preserved = grade) + delta_norm per block, and the grids at --grid_thumb 768.
    uv run python bench/mod_guidance/channel_attribution.py \
        --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
        --experiment layer_window --dataset_samples 6 --seeds 0,1 \
        --layerwin_mode single --compile --label phase0b
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# bench/ is not an installed package -- bootstrap the repo root onto sys.path so
# `library` / `bench._common` import the same way the sibling benches do.
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("mod_guidance.channel_attribution")

# A near-square on-distribution training bucket (W, H) from CONSTANT_TOKEN_BUCKETS.
DEFAULT_W, DEFAULT_H = 1024, 1008

DEFAULT_PROMPTS = [
    "1girl, solo, looking at viewer, outdoors, standing",
    "a red fox sitting in a snowy forest, soft morning light",
]
# A quality tag, a second quality word, and a content tag. Pass artist/character
# tags via --tags to probe the finding's "named entities dominate the axis" claim.
DEFAULT_TAGS = ["score_9", "holding a sword"]
DEFAULT_NEG = ""

EPS = 1e-8

# Anima caption SLOT_ORDER is (rating, count, character, copyright, artist,
# general) -- see library/captioning/anima_tagger.py:90. Booru quality/meta tags
# (score_N, masterpiece, res/recency) are NOT a slot; the training captions place
# them right after the leading rating literal. So a quality tag spliced at the
# TAIL (general band) is off-distribution, and since cross-attn is position-
# sensitive (the `order` experiment), that skews the swap measurement. We insert
# quality tags after the rating band; everything else appends to the general band.
_QUALITY_RE = re.compile(
    r"^(score_\d(_up)?|masterpiece|(best|high|normal|low|worst) quality"
    r"|absurdres|highres|lowres|newest|oldest|recent|old|year \d{4})$",
    re.IGNORECASE,
)


def _is_quality_tag(tag: str) -> bool:
    return bool(_QUALITY_RE.match(tag.strip()))


def _splice_tag(base: str, tag: str, slot: str) -> str:
    """Insert `tag` into `base` at its schema-correct position.

    slot: auto    -> after_rating for quality/meta tags, append otherwise
          after_rating -> right after a leading rating literal (else prepend)
          append  -> general band (tail)
          prepend -> very front
    """
    from library.captioning.taxonomy import CAPTION_RATINGS

    toks = [t.strip() for t in base.split(",") if t.strip()]
    if slot == "auto":
        slot = "after_rating" if _is_quality_tag(tag) else "append"
    if slot == "prepend":
        return ", ".join([tag] + toks)
    if slot == "append":
        return ", ".join(toks + [tag])
    # after_rating: place between the rating literal and the count band.
    if toks and toks[0].lower() in CAPTION_RATINGS:
        return ", ".join([toks[0], tag] + toks[1:])
    return ", ".join([tag] + toks)  # no rating band -> front of prompt


# --------------------------------------------------------------------------- #
# Render job plumbing
# --------------------------------------------------------------------------- #
@dataclass
class RenderJob:
    """One image to render. `cross_prompt` feeds cross-attention; `pool_prompt`'s
    pooled vector feeds the AdaLN mod channel (None -> use cross_prompt's). When
    `mod_w` > 0 the steering buffers are armed (intensity experiment)."""

    key: str
    cross_prompt: str
    pool_prompt: Optional[str]
    seed: int
    mod_w: float = 0.0
    mod_pos: Optional[str] = None
    mod_neg: Optional[str] = None
    # layer_window (Phase 0b): the block range the steering schedule covers. Defaults
    # to the shipped step_i8_skip27 band (8..26). The layer_window experiment overrides
    # these per render to steer a single block [l, l+1) or a cumulative range [8, L+1).
    mod_start_layer: int = 8
    mod_end_layer: int = 27
    # sigma_window (Phase 0): per-step gating of the steering schedule by an σ-band.
    # Steering is active at step i iff  mod_sigma_lo <= σ_i < mod_sigma_hi. Both None
    # -> no per-step gate (schedule applied at every step = the shipped σ-blind path,
    # also what the intensity experiment wants). The band lets us sweep soft windows
    # (<0.55 / <0.45 / <0.35) and, by boosting mod_w on a narrow band, dose-match a
    # tail-only arm to uniform's integrated steering (w x active_steps).
    mod_sigma_lo: Optional[float] = None
    mod_sigma_hi: Optional[float] = None


@dataclass
class Rendered:
    latent: torch.Tensor = None  # (16, H_lat, W_lat) fp32 cpu
    pixel: torch.Tensor = None  # (3, H, W) [-1,1] fp32 cpu -- for SSIM (sigma_window)
    pe: torch.Tensor = None  # (D,) unit fp32 cpu
    pixel_std: float = 0.0  # mean over channels of per-image spatial std
    tone: float = 0.0  # mean abs pixel value in [0,1] (DC / pink proxy)


# --------------------------------------------------------------------------- #
# Stage 1: text encoding (TE loaded, then freed)
# --------------------------------------------------------------------------- #
def encode_prompts(model, prompts: list[str], args, device) -> dict[str, torch.Tensor]:
    """Encode each unique prompt to a (1, 512, 1024) crossattn_emb (post-LLMAdapter,
    padded) -- the *exact* tensor the live mod-guidance path pools, via the real
    `_encode_prompt_for_mod` helper against the loaded DiT (TE + DiT briefly
    coexist, same as the shipped mod-guidance setup). Frees the TE after."""
    from library.inference.text import ensure_text_strategies
    from library.inference.models import load_text_encoder
    from library.inference.corrections.mod_guidance import _encode_prompt_for_mod
    from library.runtime.device import clean_memory_on_device

    ensure_text_strategies(args.text_encoder)
    te = load_text_encoder(text_encoder=args.text_encoder, device=device)
    te.eval()

    out: dict[str, torch.Tensor] = {}
    for p in prompts:
        if p not in out:
            out[p] = _encode_prompt_for_mod(p, model, te, device).to("cpu", dtype=torch.bfloat16)
            logger.info(f"  encoded: {p!r} -> {tuple(out[p].shape)}")

    del te
    gc.collect()
    clean_memory_on_device(device)
    return out


# --------------------------------------------------------------------------- #
# Stage 2: denoising (DiT loaded, then freed)
# --------------------------------------------------------------------------- #
def build_dit(args, device):
    from library.anima import weights as anima_utils

    model = anima_utils.load_anima_model(
        device,
        args.dit,
        attn_mode=args.attn_mode,
        loading_device=device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima_utils.load_pooled_text_proj(model, args.pooled_text_proj, device)
    model.to(device)
    model.eval()
    # compile_blocks turns on native-shape flattening + traces one block graph per
    # distinct token count. The bench renders at a single bucket, so this is ~1-2
    # graphs that amortise across every render in the sweep. MUST run after
    # load_pooled_text_proj so the traced forward sees the live modulation path.
    if args.compile:
        logger.info(f"  torch.compile blocks (mode={args.compile_mode})")
        model.compile_blocks(mode=args.compile_mode)
    return model


def _pool(crossattn: torch.Tensor) -> torch.Tensor:
    """pooled = max over the (padded) sequence -> (1, 1024). Matches the live path."""
    return crossattn.max(dim=1).values


def _set_mod_buffers(model, delta_unit, schedule):
    model._mod_guidance_delta.copy_(
        delta_unit.to(model._mod_guidance_delta.device, dtype=model._mod_guidance_delta.dtype)
    )
    model._mod_guidance_schedule.copy_(
        torch.tensor(schedule, device=model._mod_guidance_schedule.device, dtype=model._mod_guidance_schedule.dtype)
    )
    model._mod_guidance_final_w.fill_(0.0)


def _zero_mod_buffers(model):
    model._mod_guidance_delta.zero_()
    model._mod_guidance_schedule.zero_()
    model._mod_guidance_final_w.fill_(0.0)


def render_jobs(model, jobs, cross_cache, args, device) -> dict[str, torch.Tensor]:
    """Render every job to a clean latent (16, H_lat, W_lat) fp32 cpu."""
    from library.inference import sampling as inference_utils

    H_lat, W_lat = args.height // 8, args.width // 8
    dtype = torch.bfloat16
    proj_dtype = model.pooled_text_proj[0].weight.dtype

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    timesteps = timesteps.to(device, dtype=dtype)
    do_cfg = abs(args.guidance_scale - 1.0) > 1e-6

    neg_cross = cross_cache[args.negative].to(device, dtype=dtype)
    neg_pool = _pool(neg_cross).to(proj_dtype)
    padding_mask = torch.zeros(1, 1, H_lat, W_lat, dtype=dtype, device=device)

    out: dict[str, torch.Tensor] = {}
    for job in jobs:
        # Steering buffers (intensity / sigma_window experiments) or off.
        armed = job.mod_w > 0.0
        full_sched_t = zero_sched_t = None
        if armed:
            from library.inference.corrections.mod_guidance import build_mod_schedule

            pos_c = cross_cache[job.mod_pos].to(device, dtype=dtype)
            negc = cross_cache[job.mod_neg].to(device, dtype=dtype)
            with torch.no_grad():
                d = model.pooled_text_proj(_pool(pos_c).to(proj_dtype)) - model.pooled_text_proj(
                    _pool(negc).to(proj_dtype)
                )
            sched_args = argparse.Namespace(
                mod_w=job.mod_w, mod_start_layer=job.mod_start_layer,
                mod_end_layer=job.mod_end_layer, mod_taper=0
            )
            _set_mod_buffers(model, d, build_mod_schedule(sched_args, len(model.blocks)))
            # Keep the armed schedule + an all-zero schedule so we can toggle the
            # steering on/off per step (σ-gating) by copy_ into the live buffer --
            # a buffer write, not a recompile trigger.
            full_sched_t = model._mod_guidance_schedule.clone()
            zero_sched_t = torch.zeros_like(full_sched_t)
        else:
            _zero_mod_buffers(model)

        cross_pos = cross_cache[job.cross_prompt].to(device, dtype=dtype)
        pool_src = job.pool_prompt if job.pool_prompt is not None else job.cross_prompt
        pool_override = _pool(cross_cache[pool_src].to(device, dtype=dtype)).to(proj_dtype)

        sampler = inference_utils.ERSDESampler(sigmas, seed=job.seed, device=device)
        gen = torch.Generator(device=device).manual_seed(job.seed)
        latents = torch.randn(
            (1, 16, 1, H_lat, W_lat), dtype=dtype, device=device, generator=gen
        )

        for i, t in enumerate(timesteps):
            # sigma_window: gate the steering schedule by this step's σ-band. No band
            # (both None) leaves the once-set schedule alone (the shipped σ-blind path).
            if armed and (job.mod_sigma_lo is not None or job.mod_sigma_hi is not None):
                sig = float(sigmas[i])
                lo = job.mod_sigma_lo if job.mod_sigma_lo is not None else -1.0
                hi = job.mod_sigma_hi if job.mod_sigma_hi is not None else float("inf")
                on = lo <= sig < hi
                model._mod_guidance_schedule.copy_(full_sched_t if on else zero_sched_t)
            t_exp = t.expand(latents.shape[0])
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                noise_pred = model.forward_mini_train_dit(
                    latents, t_exp, cross_pos,
                    padding_mask=padding_mask, pooled_text_override=pool_override,
                )
                if do_cfg:
                    uncond = model.forward_mini_train_dit(
                        latents, t_exp, neg_cross,
                        padding_mask=padding_mask, pooled_text_override=neg_pool,
                    )
                    noise_pred = uncond + args.guidance_scale * (noise_pred - uncond)
            denoised = latents.float() - sigmas[i] * noise_pred.float()
            latents = sampler.step(latents, denoised, i).to(latents.dtype)

        out[job.key] = latents.float().squeeze(2).squeeze(0).cpu()  # (16,H,W)
        logger.info(f"  rendered: {job.key}")

    _zero_mod_buffers(model)
    return out


# --------------------------------------------------------------------------- #
# Stage 3+4: decode (VAE) + perceptual features (PE)
# --------------------------------------------------------------------------- #
def decode_and_featurize(latents: dict, args, device) -> dict[str, Rendered]:
    from library.models.qwen_vae import load_vae
    from library.runtime.device import clean_memory_on_device

    vae = load_vae(args.vae, device=device, dtype=torch.bfloat16, eval=True)
    vae.to(device)

    pixels: dict[str, torch.Tensor] = {}  # [-1,1] CHW fp32 cpu
    rendered: dict[str, Rendered] = {}
    with torch.no_grad():
        for key, lat in latents.items():
            z = lat.unsqueeze(0).unsqueeze(2).to(device, dtype=vae.dtype)  # (1,16,1,H,W)
            px = vae.decode_to_pixels(z)
            if px.ndim == 5:
                px = px.squeeze(2)
            px = px.to("cpu", dtype=torch.float32)[0].clamp(-1, 1)  # (3,H,W)
            pixels[key] = px
            r = Rendered()
            r.latent = lat  # (16, H_lat, W_lat) fp32 cpu -- primary decomposition space
            r.pixel = px  # (3, H, W) [-1,1] fp32 cpu -- SSIM space (sigma_window)
            # DC-blowout proxies: spatial std collapse + tone toward a flat fill.
            r.pixel_std = float(px.std(dim=(-2, -1)).mean())
            r.tone = float(((px + 1) / 2).mean())
            rendered[key] = r
    vae.to("cpu")
    del vae
    gc.collect()
    clean_memory_on_device(device)

    # PE-Core features (the repo's CMMD feature space; pose-blind -- a guide).
    from library.vision.encoder import encode_pe_from_imageminus1to1, load_pe_encoder
    from library.training.cmmd import pool_and_normalize

    bundle = load_pe_encoder(device)
    with torch.no_grad():
        for key, px in pixels.items():
            feats = encode_pe_from_imageminus1to1(bundle, px.unsqueeze(0).to(device))[0]
            rendered[key].pe = pool_and_normalize(feats).cpu()
    del bundle
    gc.collect()
    clean_memory_on_device(device)

    return rendered, pixels


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _flat(lat: torch.Tensor) -> torch.Tensor:
    return lat.reshape(-1).float()


def _norm(v: torch.Tensor) -> float:
    return float(v.norm())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na < EPS or nb < EPS:
        return 0.0
    return float((a @ b) / (na * nb))


# --- sigma_window structural readouts -------------------------------------- #
# SSIM measures structure PRESERVED vs the unguided render (high = composition
# untouched); the LF/HF split says WHERE the steering's effect landed (HF tail =
# the detail band the grade is supposed to move; LF = composition disturbance).
_SSIM_WIN = 11


def _gaussian_window(ws: int = _SSIM_WIN, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(ws, dtype=torch.float32) - (ws - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]  # (ws, ws)


def _ssim(a: torch.Tensor, b: torch.Tensor, ws: int = _SSIM_WIN) -> float:
    """Mean SSIM between two (3, H, W) images in [-1, 1] (channel-averaged,
    Gaussian-windowed). Self-contained -- no skimage dependency."""
    a = ((a.clamp(-1, 1) + 1) / 2).unsqueeze(0).float()  # (1, 3, H, W) in [0,1]
    b = ((b.clamp(-1, 1) + 1) / 2).unsqueeze(0).float()
    c = a.shape[1]
    win = _gaussian_window(ws).to(a.dtype).expand(c, 1, ws, ws)
    pad = ws // 2
    conv = lambda x: F.conv2d(x, win, padding=pad, groups=c)  # noqa: E731
    mu_a, mu_b = conv(a), conv(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = conv(a * a) - mu_a2
    sb = conv(b * b) - mu_b2
    sab = conv(a * b) - mu_ab
    c1, c2 = 0.01**2, 0.03**2
    smap = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return float(smap.mean())


# Radial-frequency cutoff (fraction of Nyquist) splitting "composition" (LF) from
# "detail/texture" (HF) in latent space. 0.25 keeps the coarse layout below the line.
_LF_CUTOFF = 0.25


def _lf_hf_energy(delta_chw: torch.Tensor, cutoff: float = _LF_CUTOFF) -> tuple[float, float]:
    """Split a (C, H, W) latent delta's spectral energy into low/high frequency
    bands by normalized radial frequency. Returns (lf_energy, hf_energy)."""
    d = delta_chw.float()
    spec = torch.fft.rfft2(d, dim=(-2, -1))  # (C, H, Wf)
    power = (spec.real**2 + spec.imag**2).sum(0)  # (H, Wf)
    h, w = d.shape[-2], d.shape[-1]
    fy = torch.fft.fftfreq(h).abs().reshape(-1, 1)  # (H, 1) in [0, 0.5]
    fx = torch.fft.rfftfreq(w).reshape(1, -1)  # (1, Wf) in [0, 0.5]
    radial = torch.sqrt(fy**2 + fx**2) / (0.5 * math.sqrt(2.0))  # ~[0, 1]
    lf_mask = radial <= cutoff
    lf = float(power[lf_mask].sum())
    hf = float(power[~lf_mask].sum())
    return lf, hf


# --------------------------------------------------------------------------- #
# Experiments -- each builds jobs, then consumes the rendered map.
# --------------------------------------------------------------------------- #
def exp_swap(prompts, tags, seeds, neg, tag_slot):
    """2x2 channel decomposition. Returns (jobs, lambda(rendered)->rows)."""
    jobs, specs = [], []
    for pi, base in enumerate(prompts):
        for ti, tag in enumerate(tags):
            tagged = _splice_tag(base, tag, tag_slot)
            for s in seeds:
                kb = f"swap/p{pi}t{ti}s{s}"
                conf = {
                    "BB": (base, base), "TT": (tagged, tagged),
                    "TB": (tagged, base), "BT": (base, tagged),
                }
                for name, (cp, pp) in conf.items():
                    jobs.append(RenderJob(f"{kb}/{name}", cp, pp, s))
                specs.append((base, tag, s, kb))
    return jobs, lambda R: _swap_rows(R, specs)


def _swap_rows(R, specs):
    rows = []
    for base, tag, s, kb in specs:
        for space, get in (("latent", lambda r: _flat(r.latent)), ("pe", lambda r: r.pe.float())):
            bb, tt, tb, bt = (R[f"{kb}/{n}"] for n in ("BB", "TT", "TB", "BT"))
            d_full = get(tt) - get(bb)
            d_cross = get(tb) - get(bb)
            d_pool = get(bt) - get(bb)
            nf = _norm(d_full)
            rows.append({
                "experiment": "swap", "space": space, "base": base, "tag": tag, "seed": s,
                "norm_full": nf, "norm_cross": _norm(d_cross), "norm_pool": _norm(d_pool),
                "pool_share": _norm(d_pool) / (nf + EPS),
                "cross_share": _norm(d_cross) / (nf + EPS),
                "cos_cross_pool": _cos(d_cross, d_pool),
                "additivity_resid": _norm(d_full - (d_cross + d_pool)) / (nf + EPS),
            })
    return rows


def exp_order(prompts, seeds, neg, n_perm, rng):
    """Pure cross-attn isolator. Permute the comma-tags in the cross-attn sequence
    while PINNING pooled = pool(canonical) on every render (via pool_prompt=canon).

    Pooling happens *post-encoder* (contextualised Qwen3 + LLM-adapter), so a raw
    reorder is NOT pooled-invariant -- the encoder is order-sensitive. Pinning the
    pooled override forces the mod channel constant by construction, so 100% of the
    canon-vs-perm image movement is cross-attn. _order_pooled_drift logs how much
    pooled *would* have moved unpinned (the encoder's order-sensitivity), for context."""
    jobs, specs = [], []
    for pi, base in enumerate(prompts):
        toks = [t.strip() for t in base.split(",") if t.strip()]
        perms = []
        for _ in range(n_perm):
            q = toks[:]
            rng.shuffle(q)
            perms.append(", ".join(q))
        canon = ", ".join(toks)
        for s in seeds:
            # pool_prompt=canon everywhere -> pooled channel pinned -> isolator.
            jobs.append(RenderJob(f"order/p{pi}s{s}/canon", canon, canon, s))
            jobs.append(RenderJob(f"order/p{pi}s{s}/canon2", canon, canon, s + 100000))
            for j, pm in enumerate(perms):
                jobs.append(RenderJob(f"order/p{pi}s{s}/perm{j}", pm, canon, s))
            specs.append((base, canon, perms, s, pi))
    return jobs, lambda R: _order_rows(R, specs)


def _order_rows(R, specs):
    rows = []
    for base, canon, perms, s, pi in specs:
        for space, get in (("latent", lambda r: _flat(r.latent)), ("pe", lambda r: r.pe.float())):
            c = get(R[f"order/p{pi}s{s}/canon"])
            seed_floor = _norm(get(R[f"order/p{pi}s{s}/canon2"]) - c)
            d_orders = [
                _norm(get(R[f"order/p{pi}s{s}/perm{j}"]) - c) for j in range(len(perms))
            ]
            rows.append({
                "experiment": "order", "space": space, "base": base, "seed": s,
                "order_dist_mean": float(np.mean(d_orders)),
                "order_dist_max": float(np.max(d_orders)),
                "seed_floor": seed_floor,
                "order_vs_seed": float(np.mean(d_orders)) / (seed_floor + EPS),
            })
    return rows


def exp_intensity(prompts, seeds, neg, w_points, steer_pos, steer_neg):
    """Sweep steering w; measure off-baseline movement + DC-blowout proxies."""
    jobs, specs = [], []
    for pi, base in enumerate(prompts):
        for s in seeds:
            for w in w_points:
                jobs.append(RenderJob(
                    f"intensity/p{pi}s{s}/w{w}", base, base, s,
                    mod_w=float(w), mod_pos=steer_pos, mod_neg=steer_neg,
                ))
            specs.append((base, s, pi))
    return jobs, lambda R: _intensity_rows(R, specs, w_points)


def _intensity_rows(R, specs, w_points):
    rows = []
    for base, s, pi in specs:
        r0 = R[f"intensity/p{pi}s{s}/w{w_points[0]}"]
        base_pe = r0.pe.float()
        for w in w_points:
            r = R[f"intensity/p{pi}s{s}/w{w}"]
            rows.append({
                "experiment": "intensity", "space": "image", "base": base, "seed": s, "w": float(w),
                "pe_move_from_w0": _norm(r.pe.float() - base_pe),
                "pixel_std": r.pixel_std, "tone": r.tone,
                "pixel_std_drop": (r0.pixel_std - r.pixel_std) / (r0.pixel_std + EPS),
            })
    return rows


# --------------------------------------------------------------------------- #
# sigma_window (Phase 0): σ-window ablation of the shipped steering schedule.
# --------------------------------------------------------------------------- #
def _sigwin_arms(w, thresholds, dose_mode, sigmas_active):
    """Build the (name, lo, hi, w) arm table for the σ-window ablation.

    Two confounds the bench must separate (the reason the single-threshold v1 was
    inconclusive, see proposal kill-condition note):
      1. WHERE in σ the steering acts (per-step leverage)  -- the question of interest.
      2. HOW MUCH total steering it applies (integrated dose = w x active_steps). At
         flow_shift=3 a σ<0.45 window is only ~4/20 steps, so an equal-w tail arm
         carries ~5x LESS dose than uniform -- "tail ≈ off" could just be under-dosing.

    So for each soft threshold t we emit a low arm gated to σ<t, in up to two dose
    flavours: equal_w (same w as uniform -> isolates marginal/leverage) and equal_dose
    (w boosted by uniform_steps/window_steps -> matches integrated dose, asking "can
    the tail deliver the grade at all if you spend the same budget there"). Plus a
    σ>=0.45 high complement (the structure-forming bulk) and the σ-blind uniform.
    """
    n_total = len(sigmas_active)
    arms = [("uniform", None, None, w)]  # σ-blind shipped path (gate disabled)
    for t in thresholds:
        n = sum(1 for x in sigmas_active if x < t)
        ratio = n_total / max(n, 1)
        lbl = f"{int(round(t * 100)):03d}"  # 0.45 -> "045"
        if dose_mode in ("equal_w", "both"):
            arms.append((f"low{lbl}", 0.0, t, w))  # leverage at fixed w
        if dose_mode in ("equal_dose", "both") and n < n_total:
            arms.append((f"low{lbl}d", 0.0, t, w * ratio))  # dose-matched to uniform
    arms.append(("high045", 0.45, float("inf"), w))  # structure-forming complement
    return arms


def exp_sigma_window(prompts, seeds, steer_pos, steer_neg, w, thresholds, dose_mode, sigmas_active):
    """Render each prompt under unguided + a σ-band sweep of the shipped steering.

    Tests proposal claim C1 (docs/findings/mod_guidance_quality_tag_axis.md (§ Schedule axis)):
    does restricting the *existing* steering schedule to the σ<t refinement tail
    preserve the grade effect while better preserving resolved structure, vs the
    σ-blind ("uniform") application? Each arm reuses the shipped per-block schedule
    (build_mod_schedule step_i8_skip27); only the σ-gate (and, for dose-matched arms,
    the scalar w) differ. See _sigwin_arms for the arm table + dose rationale.
    """
    arms = _sigwin_arms(w, thresholds, dose_mode, sigmas_active)
    arm_names = [a[0] for a in arms]
    jobs, specs = [], []
    for pi, base in enumerate(prompts):
        for s in seeds:
            kb = f"sigma_window/p{pi}s{s}"
            jobs.append(RenderJob(f"{kb}/off", base, base, s))
            for name, lo, hi, wv in arms:
                jobs.append(RenderJob(
                    f"{kb}/{name}", base, base, s,
                    mod_w=float(wv), mod_pos=steer_pos, mod_neg=steer_neg,
                    mod_sigma_lo=lo, mod_sigma_hi=hi,
                ))
            specs.append((base, s, pi, arm_names))
    return jobs, lambda R: _sigwin_rows(R, specs)


def _sigwin_rows(R, specs, lam: float = 1.0):
    rows = []
    for base, s, pi, modes in specs:
        off = R[f"sigma_window/p{pi}s{s}/off"]
        for m in modes:
            r = R[f"sigma_window/p{pi}s{s}/{m}"]
            delta = r.latent - off.latent  # (16, H, W) steering effect in latent space
            lf, hf = _lf_hf_energy(delta)
            tot = lf + hf + EPS
            rows.append({
                "experiment": "sigma_window", "space": "image", "base": base, "seed": s,
                "mode": m,
                "ssim_to_off": _ssim(r.pixel, off.pixel),  # structure PRESERVED (high=good)
                "delta_norm": _norm(_flat(delta)),  # total steering effect magnitude
                "lf_energy": lf, "hf_energy": hf,
                "hf_frac": hf / tot,  # fraction of effect in the detail band
                "J": hf - lam * lf,  # structural objective: HF on-target - λ·LF disturbance
            })
    return rows


# --------------------------------------------------------------------------- #
# layer_window (Phase 0b): per-block grade-vs-drift map of the steering schedule.
# --------------------------------------------------------------------------- #
def _parse_blocks(spec: str) -> list[int]:
    """Parse a block spec into a sorted unique list. Accepts ranges ('8-26'),
    explicit lists ('8,10,12'), or a mix ('8-12,20,24-26')."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _split_contiguous(blocks: list[int], n: int) -> list[list[int]]:
    """Split a sorted block list into n nearly-equal contiguous groups."""
    n = max(1, min(n, len(blocks)))
    L = len(blocks)
    out = []
    for i in range(n):
        a, b = i * L // n, (i + 1) * L // n
        if b > a:
            out.append(blocks[a:b])
    return out


def _layerwin_arms(blocks: list[int], mode: str, n_bands: int = 3) -> list[tuple[str, int, int]]:
    """Build the (name, start_layer, end_layer) arm table for the per-block map.

    single      one arm per block l, steering only [l, l+1) -> the direct per-block
                effect. THE grade-vs-drift map (does steering block l grade or drift).
    cumulative  one arm per L, steering [blocks[0], L+1) -> the marginal-per-block view
                (block l's marginal contribution = cum..L minus cum..L-1). Disentangles
                blocks whose single-block effect is masked by interactions.
    band        n_bands contiguous thirds (early/mid/late) of the probed range -- a dose
                between a single block (≈off, safe) and the full stack (high effect but,
                per the Phase-0b grids, breaks hands). Localizes the full-stack damage to
                a third before a finer leave-one-out, when scalar SSIM is blind to the
                hand grade/drift the effect actually lives in.

    Every mode also emits a `full` arm covering the whole probed range -- with the
    default blocks (8..26) that IS the shipped step_i8_skip27 path, the reference the
    sub-range arms decompose. end_layer is exclusive (matches build_mod_schedule).
    """
    arms: list[tuple[str, int, int]] = []
    if mode in ("single", "both"):
        for lyr in blocks:
            arms.append((f"L{lyr:02d}", lyr, lyr + 1))
    if mode in ("cumulative", "both"):
        b0 = blocks[0]
        for lyr in blocks:
            arms.append((f"cum{b0:02d}-{lyr:02d}", b0, lyr + 1))
    if mode == "band":
        for g in _split_contiguous(blocks, n_bands):
            arms.append((f"b{g[0]:02d}-{g[-1]:02d}", g[0], g[-1] + 1))
    arms.append((f"full{blocks[0]:02d}-{blocks[-1]:02d}", blocks[0], blocks[-1] + 1))
    return arms


def exp_layer_window(prompts, seeds, steer_pos, steer_neg, w, blocks, mode, n_bands=3):
    """Render each prompt under unguided + a per-block sweep of the shipped steering.

    Tests proposal claim C2 (docs/findings/mod_guidance_quality_tag_axis.md (§ Schedule axis)):
    does layer placement separate GRADE (effect at high structure-SSIM) from DRIFT
    (effect at low SSIM)? Each arm reuses the shipped pooled delta + scalar w; only
    the steered block range differs (steered σ-blind, every step, like the shipped
    path). The readout per arm vs unguided is delta_norm (did it move the image) +
    pixel SSIM (was structure preserved). See _layerwin_arms for the arm table.

    Flat map (every block ≈ same delta & SSIM) -> no layer lever, proposal closes.
    Differentiated (a grade set + a drift set) -> the "proper layers" are the grade
    set; a static / aspect-LUT / training-free-online schedule becomes real.
    """
    arms = _layerwin_arms(blocks, mode, n_bands)
    arm_names = [a[0] for a in arms]
    jobs, specs = [], []
    for pi, base in enumerate(prompts):
        for s in seeds:
            kb = f"layer_window/p{pi}s{s}"
            jobs.append(RenderJob(f"{kb}/off", base, base, s))
            for name, start, end in arms:
                jobs.append(RenderJob(
                    f"{kb}/{name}", base, base, s,
                    mod_w=float(w), mod_pos=steer_pos, mod_neg=steer_neg,
                    mod_start_layer=start, mod_end_layer=end,
                ))
            specs.append((base, s, pi, arm_names))
    return jobs, lambda R: _layerwin_rows(R, specs)


def _layerwin_rows(R, specs, lam: float = 1.0):
    rows = []
    for base, s, pi, modes in specs:
        off = R[f"layer_window/p{pi}s{s}/off"]
        for m in modes:
            r = R[f"layer_window/p{pi}s{s}/{m}"]
            delta = r.latent - off.latent  # (16, H, W) steering effect in latent space
            lf, hf = _lf_hf_energy(delta)
            tot = lf + hf + EPS
            rows.append({
                "experiment": "layer_window", "space": "image", "base": base, "seed": s,
                "mode": m,
                "ssim_to_off": _ssim(r.pixel, off.pixel),  # structure PRESERVED (high=grade)
                "delta_norm": _norm(_flat(delta)),  # total steering effect magnitude
                "lf_energy": lf, "hf_energy": hf,
                "hf_frac": hf / tot,  # fraction of effect in the detail band
                "J": hf - lam * lf,  # guide only (HF-noise trap) -- read SSIM + grids
            })
    return rows


# --------------------------------------------------------------------------- #
# origin: base-model upstream vs distilled-proj split (render-free)
# --------------------------------------------------------------------------- #
def origin_specs(prompts, tags, tag_slot):
    """(base, tagged, tag, is_quality) per (prompt, tag). Seed-independent: text
    encoding is deterministic, so the pooled vector doesn't depend on the seed."""
    specs = []
    for base in prompts:
        for tag in tags:
            tagged = _splice_tag(base, tag, tag_slot)
            specs.append((base, tagged, tag, _is_quality_tag(tag)))
    return specs


def compute_origin_rows(specs, cross_cache, model, device):
    """Split a tag's pooled-channel response into its base-model and distilled parts.

    Upstream (rel_dpool): ‖pool(base+tag) − pool(base)‖ / ‖pool(base)‖, computed on
    the raw max-pooled crossattn vector -- 100% Qwen3 TE + LLM-adapter (ships with the
    base DiT), NO trained proj. This is the only piece a quality tag's routing could
    inherit from the base model.

    Downstream (rel_dproj): the same movement after the trained pooled_text_proj --
    100% distilled, and distilled *without* quality tags. proj_gain = rel_dproj /
    rel_dpool is how much the distilled proj amplifies (or attenuates) this tag's
    upstream pooled movement on its way into AdaLN.

    Read: if quality's rel_dpool already exceeds content's AND proj_gain is ~tag-
    agnostic, the quality-selectivity is upstream (base encoder) and the proj is a
    pass-through -> conditioning the distill teacher on quality buys little. If
    proj_gain is larger for quality than content, the distilled proj is preferentially
    amplifying quality directions -> headroom for a quality-conditioned teacher."""
    proj = model.pooled_text_proj
    w_dtype = proj[0].weight.dtype
    rows = []
    with torch.no_grad():
        for base, tagged, tag, is_q in specs:
            pb = _pool(cross_cache[base].to(device, dtype=w_dtype))  # (1, 1024) base
            pt = _pool(cross_cache[tagged].to(device, dtype=w_dtype))
            yb = proj(pb)  # (1, model_channels) distilled modulation
            yt = proj(pt)
            d_pool = (pt - pb).reshape(-1).float()
            d_proj = (yt - yb).reshape(-1).float()
            rel_dpool = float(d_pool.norm() / (pb.reshape(-1).float().norm() + EPS))
            rel_dproj = float(d_proj.norm() / (yb.reshape(-1).float().norm() + EPS))
            rows.append({
                "experiment": "origin", "space": "pooled", "base": base, "tag": tag,
                "is_quality": bool(is_q),
                "rel_dpool": rel_dpool,  # upstream base-model encoder movement
                "rel_dproj": rel_dproj,  # after distilled proj
                "proj_gain": rel_dproj / (rel_dpool + EPS),
                "norm_dproj": float(d_proj.norm()),  # absolute AdaLN-space movement
            })
    return rows


# --------------------------------------------------------------------------- #
# Grid saving (read the grids!)
# --------------------------------------------------------------------------- #
def save_grids(pixels, run_dir, experiment, thumb=512, cols=0):
    """One labelled grid PNG per render group, laid out as a 2D grid (per-cell label
    above each thumbnail). Thumbnails preserve aspect ratio (longer side = `thumb`)
    so non-square buckets aren't squashed; all renders in a group share a resolution
    so the cells tile cleanly.

    `cols`: fixed column count (0 = auto). Auto keeps a small group on a single row
    (≤8 cells, e.g. swap's BB/TT/TB/BT) but wraps a large sweep (layer_window's ~21
    per-block arms) to a near-square grid that's actually eyeball-able."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        logger.warning("PIL unavailable -- skipping grids")
        return []

    def to_img(px):
        x = ((px.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).numpy().transpose(1, 2, 0)
        img = Image.fromarray(x)
        scale = thumb / max(img.width, img.height)
        return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))

    header = max(16, thumb // 24)
    groups: dict[str, list[str]] = {}
    for k in pixels:
        if not k.startswith(experiment):
            continue
        groups.setdefault(k.rsplit("/", 1)[0], []).append(k)
    artifacts = []
    for grp, keys in sorted(groups.items()):
        keys = sorted(keys)
        thumbs = [to_img(pixels[k]) for k in keys]
        labels = [k.rsplit("/", 1)[1] for k in keys]
        n = len(thumbs)
        ncol = cols if cols and cols > 0 else (n if n <= 8 else math.ceil(math.sqrt(n)))
        ncol = max(1, min(ncol, n))
        nrow = math.ceil(n / ncol)
        cw = max(t.width for t in thumbs)  # uniform cell size (one resolution per group)
        ch = max(t.height for t in thumbs)
        cell_h = ch + header
        canvas = Image.new("RGB", (cw * ncol, cell_h * nrow), (16, 16, 16))
        d = ImageDraw.Draw(canvas)
        for idx, (t, lbl) in enumerate(zip(thumbs, labels)):
            r, c = divmod(idx, ncol)
            x, y = c * cw, r * cell_h
            canvas.paste(t, (x, y + header))
            d.text((x + 3, y + 3), lbl, fill=(230, 230, 230))
        name = grp.replace("/", "_") + ".png"
        canvas.save(run_dir / name)
        artifacts.append(name)
    return artifacts


# --------------------------------------------------------------------------- #
def sample_dataset_prompts(dataset_dir: str, n: int, max_chars: int, min_chars: int = 0) -> list[str]:
    """Sample N real captions from `.txt` sidecars under dataset_dir.

    image_dataset/ is a symlink to nested artist dirs, so rglob (which follows the
    start symlink) is used rather than a plain walk. Captions are the dense, real
    prompts where the geometry-only finding's numbers collapse 5-25x -- the regime
    we actually care about. `min_chars` keeps only LENGTHY (dense, busy-scene) captions
    -- the regime where the steering's hand grade/drift fires (Phase 0b: short/canonical
    poses barely move). Deterministic via a dedicated rng so permutation draws elsewhere
    stay stable."""
    root = Path(dataset_dir)
    txts = sorted(str(p) for p in root.rglob("*.txt"))
    if not txts:
        raise SystemExit(f"No .txt captions found under {root} (symlink? try --dataset_dir)")
    rng = np.random.default_rng(987)
    rng.shuffle(txts)
    out: list[str] = []
    for t in txts:
        try:
            cap = Path(t).read_text(encoding="utf-8").strip().replace("\n", ", ")
        except Exception:
            continue
        if not cap or len(cap) < min_chars:  # min_chars -> lengthy/dense only
            continue
        if len(cap) > max_chars:  # truncate at a tag (comma) boundary, not mid-tag
            cut = cap[:max_chars].rfind(",")
            cap = cap[:cut] if cut > 0 else cap[:max_chars]
        out.append(cap.strip())
        if len(out) >= n:
            break
    if len(out) < n:
        logger.warning(f"Only {len(out)}/{n} captions >= {min_chars} chars under {root}")
    logger.info(f"Sampled {len(out)} real captions from {root} (min_chars={min_chars})")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit", default="models/diffusion_models/anima-base-v1.0.safetensors")
    p.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    p.add_argument("--text_encoder", default="models/text_encoders/qwen_3_06b_base.safetensors")
    p.add_argument("--pooled_text_proj", required=True, help="trained pooled_text_proj checkpoint")
    p.add_argument(
        "--experiment",
        choices=["swap", "order", "intensity", "origin", "sigma_window", "layer_window", "all"],
        default="all",
        help="'all' runs swap/order/intensity/origin; 'sigma_window' (Phase-0 σ-schedule "
        "ablation) and 'layer_window' (Phase-0b per-block grade-vs-drift map) run on their own.",
    )
    p.add_argument("--prompts", type=str, default=None, help="';'-separated base prompts (wins over --dataset_samples)")
    p.add_argument("--dataset_samples", type=int, default=0, help="sample N real captions from --dataset_dir instead of the built-in defaults")
    p.add_argument("--dataset_dir", type=str, default="image_dataset", help="caption (.txt) source for --dataset_samples")
    p.add_argument("--max_prompt_chars", type=int, default=400, help="truncate sampled captions to this length")
    p.add_argument("--prompt_min_chars", type=int, default=0, help="--dataset_samples: keep only captions at least this long (lengthy/dense scenes where the hand grade/drift fires)")
    p.add_argument("--tags", type=str, default=None, help="','-separated tags to splice (swap)")
    p.add_argument(
        "--tag_slot", choices=["auto", "after_rating", "append", "prepend"], default="auto",
        help="where to insert a spliced tag (auto: quality->after rating, else append). "
        "Anima SLOT_ORDER puts quality/meta after the rating literal; appending is off-distribution.",
    )
    p.add_argument("--negative", type=str, default=DEFAULT_NEG)
    p.add_argument("--seeds", type=str, default="0", help="','-separated seeds")
    p.add_argument("--n_perm", type=int, default=3, help="order: permutations per prompt")
    p.add_argument("--w_points", type=str, default="0,2,3,5,8", help="intensity: steering w sweep")
    p.add_argument("--steer_pos", type=str, default="score_9, absurdres", help="intensity steering p+")
    p.add_argument("--steer_neg", type=str, default="", help="intensity steering p-")
    # sigma_window (Phase 0): faithful shipped steering pair + the σ split point.
    p.add_argument("--sigwin_pos", type=str, default="absurdres, masterpiece, score_9", help="sigma_window steering p+ (shipped default)")
    p.add_argument("--sigwin_neg", type=str, default="worst quality, low quality, score_1", help="sigma_window steering p- (shipped default)")
    p.add_argument("--sigwin_w", type=float, default=3.0, help="sigma_window steering weight (shipped mod_w default)")
    p.add_argument("--sigwin_thresholds", type=str, default="0.45", help="sigma_window: σ-window cutoff(s); each emits a σ<t 'low' arm. Default 0.45 (the structure-resolution boundary); comma-sep to sweep soft windows.")
    p.add_argument(
        "--sigwin_dose", choices=["equal_w", "equal_dose", "both"], default="both",
        help="sigma_window dose control: equal_w (fixed w -> per-step leverage), equal_dose "
        "(w boosted by uniform_steps/window_steps -> matched integrated dose), or both. "
        "Avoids falsely killing the σ axis by under-dosing a narrow tail window.",
    )
    # layer_window (Phase 0b): same shipped steering pair; the probe is WHICH blocks.
    p.add_argument("--layerwin_pos", type=str, default="absurdres, masterpiece, score_9", help="layer_window steering p+ (shipped default)")
    p.add_argument("--layerwin_neg", type=str, default="worst quality, low quality, score_1", help="layer_window steering p- (shipped default)")
    p.add_argument("--layerwin_w", type=float, default=3.0, help="layer_window steering weight (shipped mod_w default)")
    p.add_argument("--layerwin_blocks", type=str, default="8-26", help="layer_window: blocks to probe (range '8-26' / list '8,12,20' / mix). Default = the shipped step_i8_skip27 band.")
    p.add_argument(
        "--layerwin_mode", choices=["single", "cumulative", "band", "both"], default="single",
        help="layer_window: 'single' steers one block at a time (the direct per-block "
        "grade-vs-drift map); 'cumulative' steers [b0..L] (marginal-per-block view); "
        "'band' steers contiguous thirds (early/mid/late -- localizes the full-stack "
        "hand damage before a finer probe); 'both' = single+cumulative.",
    )
    p.add_argument("--layerwin_bands", type=int, default=3, help="layer_window band mode: number of contiguous bands to split the probed range into")
    p.add_argument("--height", type=int, default=DEFAULT_H)
    p.add_argument("--width", type=int, default=DEFAULT_W)
    p.add_argument("--infer_steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--attn_mode", type=str, default="torch")
    p.add_argument("--compile", action="store_true", help="torch.compile DiT blocks (amortises across the sweep)")
    p.add_argument("--compile_mode", type=str, default="default")
    p.add_argument("--grid_thumb", type=int, default=768, help="per-image grid thumbnail longer-side px")
    p.add_argument("--grid_cols", type=int, default=0, help="grid columns (0=auto: single row if <=8 cells, else near-square 2D wrap)")
    p.add_argument("--label", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.prompts:
        prompts = [s.strip() for s in args.prompts.split(";") if s.strip()]
    elif args.dataset_samples > 0:
        prompts = sample_dataset_prompts(
            args.dataset_dir, args.dataset_samples, args.max_prompt_chars, args.prompt_min_chars
        )
    else:
        prompts = DEFAULT_PROMPTS
    tags = (
        [s.strip() for s in args.tags.split(",") if s.strip()]
        if args.tags else DEFAULT_TAGS
    )
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    w_points = [float(w) for w in args.w_points.split(",") if w.strip()]
    rng = np.random.default_rng(1234)  # fixed seed -> reproducible permutations

    exps = ["swap", "order", "intensity", "origin"] if args.experiment == "all" else [args.experiment]

    # Build all jobs + their row-extractors.
    jobs, extractors = [], []
    if "swap" in exps:
        j, f = exp_swap(prompts, tags, seeds, args.negative, args.tag_slot)
        jobs += j
        extractors.append(f)
    if "order" in exps:
        j, f = exp_order(prompts, seeds, args.negative, args.n_perm, rng)
        jobs += j
        extractors.append(f)
    if "intensity" in exps:
        j, f = exp_intensity(prompts, seeds, args.negative, w_points, args.steer_pos, args.steer_neg)
        jobs += j
        extractors.append(f)
    if "sigma_window" in exps:
        from library.inference import sampling as _inf
        thresholds = [float(t) for t in args.sigwin_thresholds.split(",") if t.strip()]
        # σ actually used at each model call (flow_shift-warped) -> dose accounting.
        _, _sig = _inf.get_timesteps_sigmas(args.infer_steps, args.flow_shift, torch.device("cpu"))
        sigmas_active = _sig.tolist()[: args.infer_steps]
        j, f = exp_sigma_window(
            prompts, seeds, args.sigwin_pos, args.sigwin_neg, args.sigwin_w,
            thresholds, args.sigwin_dose, sigmas_active,
        )
        jobs += j
        extractors.append(f)
    if "layer_window" in exps:
        blocks = _parse_blocks(args.layerwin_blocks)
        if not blocks:
            raise SystemExit("--layerwin_blocks parsed to an empty block list")
        j, f = exp_layer_window(
            prompts, seeds, args.layerwin_pos, args.layerwin_neg, args.layerwin_w,
            blocks, args.layerwin_mode, args.layerwin_bands,
        )
        jobs += j
        extractors.append(f)
    # origin is render-free (pooled-vector geometry) -- it needs encodings, not jobs.
    orig_specs = origin_specs(prompts, tags, args.tag_slot) if "origin" in exps else []

    # Collect every prompt that any job needs to encode.
    needed = {args.negative}
    for j in jobs:
        needed.add(j.cross_prompt)
        if j.pool_prompt:
            needed.add(j.pool_prompt)
        if j.mod_pos:
            needed.add(j.mod_pos)
        if j.mod_neg:
            needed.add(j.mod_neg)
    for base, tagged, _tag, _q in orig_specs:
        needed.add(base)
        needed.add(tagged)

    logger.info(f"Channel-attribution bench: {len(jobs)} renders, {len(needed)} prompts, exps={exps}")

    # ---- staged pipeline. DiT carries the LLM adapter + pooled_text_proj, so it
    # must be resident to encode faithfully (TE coexists briefly, like the live
    # mod-guidance setup). Then: encode (free TE) -> render -> free DiT -> VAE -> PE.
    logger.info("[1/4] building DiT + encoding prompts (TE)")
    model = build_dit(args, device)
    cross_cache = encode_prompts(model, sorted(needed), args, device)

    # Informational: how much pooled would drift under reorder (we pin it).
    if "order" in exps:
        _order_pooled_drift(jobs, cross_cache)

    # origin: render-free, but needs the live proj -> compute before freeing the DiT.
    origin_rows = compute_origin_rows(orig_specs, cross_cache, model, device) if orig_specs else []

    logger.info("[2/4] denoising (DiT)")
    latents = render_jobs(model, jobs, cross_cache, args, device)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if latents:
        logger.info("[3/4] decode (VAE) + [4/4] features (PE)")
        rendered, pixels = decode_and_featurize(latents, args, device)
    else:
        logger.info("[3/4] no renders (origin-only) -- skipping VAE/PE")
        rendered, pixels = {}, {}

    # ---- metrics
    rows = list(origin_rows)
    for f in extractors:
        rows += f(rendered)

    run_dir = make_run_dir("mod_guidance", label=args.label)
    artifacts = ["rows.csv"]
    _write_csv(run_dir / "rows.csv", rows)
    for e in exps:
        artifacts += save_grids(pixels, run_dir, e, thumb=args.grid_thumb, cols=args.grid_cols)

    metrics = _summarize(rows)
    _log_summary(metrics)

    write_result(
        run_dir, script=__file__, args=args,
        metrics=metrics, artifacts=artifacts,
    )
    logger.info(f"\nDone. Results -> {run_dir}")


def _order_pooled_drift(jobs, cross_cache):
    """Report the encoder's pooled order-sensitivity (informational).

    The order experiment PINS pooled=pool(canon) via the override, so pooled does
    not contaminate the cross-attn attribution. But it's worth knowing how much the
    pooled vector *would* drift unpinned -- that drift is pure Qwen3/LLM-adapter
    order-sensitivity (post-encoder contextualisation), not the DiT. Compares each
    group's raw cross-attn prompts (canon vs perms) against the canon pooled."""
    by_group: dict[str, list[str]] = {}
    for j in jobs:
        if j.key.startswith("order"):
            by_group.setdefault(j.key.rsplit("/", 1)[0], []).append(j.cross_prompt)
    worst, rel = 0.0, 0.0
    for prompts in by_group.values():
        pooled = [_pool(cross_cache[p].float()) for p in prompts]
        ref = pooled[0]
        for v in pooled[1:]:
            worst = max(worst, float((v - ref).abs().max()))
            rel = max(rel, float((v - ref).norm() / (ref.norm() + EPS)))
    logger.info(
        f"  encoder pooled order-drift (suppressed via override): "
        f"max|Δpool|={worst:.2e}, rel‖Δ‖={rel:.3f}"
    )


def _write_csv(path, rows):
    import csv

    if not rows:
        path.write_text("")
        return
    keys = list({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _summarize(rows):
    def agg(pred, field):
        vals = [r[field] for r in rows if pred(r) and field in r]
        return float(np.mean(vals)) if vals else None

    def is_exp(experiment, space):
        return lambda r: r["experiment"] == experiment and r["space"] == space

    out = {}
    for space in ("latent", "pe"):
        sw = is_exp("swap", space)
        out[f"swap.{space}.pool_share_mean"] = agg(sw, "pool_share")
        out[f"swap.{space}.cross_share_mean"] = agg(sw, "cross_share")
        out[f"swap.{space}.cos_cross_pool_mean"] = agg(sw, "cos_cross_pool")
        out[f"swap.{space}.additivity_resid_mean"] = agg(sw, "additivity_resid")
        out[f"order.{space}.order_vs_seed_mean"] = agg(is_exp("order", space), "order_vs_seed")
    intensity_rows = [r for r in rows if r["experiment"] == "intensity"]
    if intensity_rows:
        out["intensity.max_pixel_std_drop"] = max(r["pixel_std_drop"] for r in intensity_rows)
        out["intensity.n_points"] = len({r["w"] for r in intensity_rows})
    sigwin_rows = [r for r in rows if r["experiment"] == "sigma_window"]
    if sigwin_rows:
        modes = sorted({r["mode"] for r in sigwin_rows})

        def sw_mode(m):
            return lambda r: r["experiment"] == "sigma_window" and r["mode"] == m

        for m in modes:
            pred = sw_mode(m)
            out[f"sigma_window.{m}.ssim_to_off_mean"] = agg(pred, "ssim_to_off")
            out[f"sigma_window.{m}.hf_frac_mean"] = agg(pred, "hf_frac")
            out[f"sigma_window.{m}.delta_norm_mean"] = agg(pred, "delta_norm")
            out[f"sigma_window.{m}.J_mean"] = agg(pred, "J")
        out["sigma_window.modes"] = ",".join(modes)
        out["sigma_window.n"] = len(sigwin_rows)
    layerwin_rows = [r for r in rows if r["experiment"] == "layer_window"]
    if layerwin_rows:
        modes = sorted({r["mode"] for r in layerwin_rows})

        def lw_mode(m):
            return lambda r: r["experiment"] == "layer_window" and r["mode"] == m

        for m in modes:
            pred = lw_mode(m)
            out[f"layer_window.{m}.ssim_to_off_mean"] = agg(pred, "ssim_to_off")
            out[f"layer_window.{m}.delta_norm_mean"] = agg(pred, "delta_norm")
            out[f"layer_window.{m}.hf_frac_mean"] = agg(pred, "hf_frac")
            out[f"layer_window.{m}.J_mean"] = agg(pred, "J")
        # Differentiation signal across the single-block arms (L##): spread of SSIM /
        # delta_norm. Flat (low spread) -> no layer lever (proposal closes); wide
        # spread with an effect set splitting on SSIM -> a grade set exists.
        single = [m for m in modes if m.startswith("L")]
        if single:
            ssims = [out[f"layer_window.{m}.ssim_to_off_mean"] for m in single]
            deltas = [out[f"layer_window.{m}.delta_norm_mean"] for m in single]
            out["layer_window.single.ssim_spread"] = float(max(ssims) - min(ssims))
            out["layer_window.single.delta_spread"] = float(max(deltas) - min(deltas))
        out["layer_window.modes"] = ",".join(modes)
        out["layer_window.n"] = len(layerwin_rows)
    origin_rows = [r for r in rows if r["experiment"] == "origin"]
    if origin_rows:
        def omean(field, q):
            vals = [r[field] for r in origin_rows if r["is_quality"] == q]
            return float(np.mean(vals)) if vals else None

        for q, lbl in ((True, "quality"), (False, "content")):
            out[f"origin.{lbl}.rel_dpool_mean"] = omean("rel_dpool", q)
            out[f"origin.{lbl}.rel_dproj_mean"] = omean("rel_dproj", q)
            out[f"origin.{lbl}.proj_gain_mean"] = omean("proj_gain", q)
        out["origin.n"] = len(origin_rows)
    out["n_rows"] = len(rows)
    return out


def _log_summary(m):
    logger.info("\n=== SUMMARY ===")
    for sp in ("latent", "pe"):
        ps = m.get(f"swap.{sp}.pool_share_mean")
        cs = m.get(f"swap.{sp}.cross_share_mean")
        cos = m.get(f"swap.{sp}.cos_cross_pool_mean")
        res = m.get(f"swap.{sp}.additivity_resid_mean")
        if ps is not None:
            logger.info(
                f"swap [{sp}]  pool_share={ps:.3f}  cross_share={cs:.3f}  "
                f"cos(cross,pool)={cos:+.3f}  additivity_resid={res:.3f}"
            )
        ovs = m.get(f"order.{sp}.order_vs_seed_mean")
        if ovs is not None:
            logger.info(f"order[{sp}]  order_dist / seed_floor = {ovs:.3f}")
    if "intensity.max_pixel_std_drop" in m:
        logger.info(f"intensity   max pixel-std drop vs w0 = {m['intensity.max_pixel_std_drop']:+.3f}")
    if "sigma_window.n" in m:
        logger.info("sigma_window (vs unguided; SSIM high=structure preserved, hf_frac high=effect in detail band):")
        for mode in m["sigma_window.modes"].split(","):
            ss = m.get(f"sigma_window.{mode}.ssim_to_off_mean")
            hf = m.get(f"sigma_window.{mode}.hf_frac_mean")
            dn = m.get(f"sigma_window.{mode}.delta_norm_mean")
            jj = m.get(f"sigma_window.{mode}.J_mean")
            if ss is not None:
                logger.info(
                    f"  {mode:8s}  ssim_to_off={ss:.3f}  hf_frac={hf:.3f}  "
                    f"delta_norm={dn:.3f}  J={jj:+.2e}"
                )
        logger.info(
            "  arms: uniform(σ-blind) | low{T}=σ<0.T equal-w | low{T}d=dose-matched (w boosted) | high045=σ>=0.45.\n"
            "  C1 true  -> a dose-matched tail arm (low045d / low055d) reaches ~uniform's grade at HIGHER ssim.\n"
            "  C1 dead  -> tail arms ≈ off even dose-matched (or they saturate/blow up) while uniform≈high045\n"
            "             -> leverage is high-σ only, tail can't carry the grade; don't schedule into the tail.\n"
            "  Watch the confound: low{T} (equal-w) under-doses ~5x vs uniform -- judge leverage from low{T}d.\n"
            "  READ THE GRIDS (sigma_window_*.png, 768px) -- scalar J is a guide (HF-noise inflates it), not the verdict."
        )
    if "layer_window.n" in m:
        logger.info("layer_window (Phase 0b; vs unguided -- per arm: ssim high=structure preserved=GRADE, low=DRIFT):")
        for mode in m["layer_window.modes"].split(","):
            ss = m.get(f"layer_window.{mode}.ssim_to_off_mean")
            dn = m.get(f"layer_window.{mode}.delta_norm_mean")
            hf = m.get(f"layer_window.{mode}.hf_frac_mean")
            jj = m.get(f"layer_window.{mode}.J_mean")
            if ss is not None:
                logger.info(
                    f"  {mode:10s}  ssim_to_off={ss:.3f}  delta_norm={dn:.3f}  "
                    f"hf_frac={hf:.3f}  J={jj:+.2e}"
                )
        sp_s = m.get("layer_window.single.ssim_spread")
        sp_d = m.get("layer_window.single.delta_spread")
        if sp_s is not None:
            logger.info(f"  single-block spread: ssim={sp_s:.3f}  delta_norm={sp_d:.3f}")
        logger.info(
            "  arms: L## = steer block ## only (per-block map) | cum## = cumulative [b0..##] |"
            " full## = whole probed band (= shipped path at default blocks).\n"
            "  C2 true  -> differentiated: a GRADE set (delta_norm up, ssim HIGH) vs a DRIFT set"
            " (delta_norm up, ssim LOW) -> the 'proper layers' are the grade set.\n"
            "  C2 dead  -> FLAT: every steered block ≈ same delta_norm & ssim (small spread)"
            " -> no layer lever, hand-set 8-26 is as good as any -> proposal closes.\n"
            "  READ THE GRIDS (layer_window_*.png, 768px) -- scalar J is a guide (HF-noise trap), not the verdict."
        )
    if "origin.n" in m:
        for lbl in ("quality", "content"):
            dp = m.get(f"origin.{lbl}.rel_dpool_mean")
            pj = m.get(f"origin.{lbl}.rel_dproj_mean")
            g = m.get(f"origin.{lbl}.proj_gain_mean")
            if dp is not None:
                logger.info(
                    f"origin[{lbl:7s}]  rel_dpool(base)={dp:.3f}  "
                    f"rel_dproj(distilled)={pj:.3f}  proj_gain={g:.3f}"
                )
    logger.info(
        "\nReads: pool_share≈0 -> the mod channel barely carries the edit (topic low-value). "
        "cos(cross,pool)>0 -> channels reinforce (preload/double-drive); <0 -> conflict/cancel. "
        "order/seed≪1 -> cross-attn weakly order-sensitive. "
        "Large pixel-std drop at high w -> DC-blowout is real in image space.\n"
        "origin: quality rel_dpool > content AND proj_gain tag-agnostic -> quality-selectivity is "
        "UPSTREAM (base encoder); distilled proj is a pass-through, teacher-quality-conditioning has "
        "little headroom. proj_gain(quality) > proj_gain(content) -> the distilled proj amplifies "
        "quality directions -> headroom for a quality-conditioned distill teacher."
    )


if __name__ == "__main__":
    main()
