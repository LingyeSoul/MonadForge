"""Core generation logic for Anima inference: denoising loops and tiled diffusion."""

import argparse
import logging
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List, Any, Dict, Tuple, Union

import torch
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor

from library.anima import models as anima_models
from library.inference.adapters import (
    clear_hydra_fei,
    clear_hydra_sigma,
    compute_and_set_hydra_fei,
    set_hydra_content,
    set_hydra_crossattn,
    set_hydra_sigma,
    set_step_expert_index,
)
from library.inference import sampling as inference_utils
from library.inference.output import check_inputs
from library.inference.text import prepare_text_inputs
from library.inference.models import load_dit_model
from library.inference.corrections.mod_guidance import setup_mod_guidance
from library.inference.corrections.smc_cfg import SMCCFGState
from library.inference.sampler_context import SamplerSideChannels

logger = logging.getLogger(__name__)


def _build_cns_recolorer(args):
    """Build a CNS noise recolorer from ``--cns`` (path or "auto"), or None.

    Only meaningful on the stochastic ``er_sde`` path — on euler/lcm there is no
    injected noise to recolor, so the recolorer (even if loaded) is a no-op. We
    still gate on the flag here and let the sampler ignore it otherwise.
    """
    spec = getattr(args, "cns", None)
    if not spec:
        return None
    if getattr(args, "sampler", "euler") != "er_sde":
        logger.warning(
            "--cns is set but --sampler=%s injects no noise; CNS is a no-op "
            "(use --sampler er_sde).",
            getattr(args, "sampler", "euler"),
        )
        return None
    from library.inference.corrections.cns import CNSRecolorer

    recolorer = CNSRecolorer.from_path(
        spec, strength=getattr(args, "cns_strength", 1.0)
    )
    logger.info("CNS noise recoloring enabled (strength=%.2f).", recolorer.strength)
    return recolorer


# Spectrum runner registry. The spectrum implementation lives in
# networks/spectrum.py (or a downstream package); it self-registers on import.
# generation.py never imports it directly so the dep edge can point inward
# from a downstream inference package without inverting.
_SPECTRUM_RUNNER = None

# SPD (Spectral Progressive Diffusion) runner registry — same pattern as
# Spectrum. networks/spd.py self-registers on import; --spd dispatches to it.
_SPD_RUNNER = None


def _setup_soft_tokens(args, anima, device):
    """Build + apply the soft_tokens network from ``--soft_tokens_weight``.

    Returns ``None`` when the flag isn't set. Otherwise returns the network with
    ``apply_to(unet=anima)`` already called — the per-block ``Block.forward``
    monkey-patches are live, but ``_step_layer_tokens`` is empty until the
    caller fires ``network.append_postfix(..., timesteps=t)`` each step.
    """
    soft_weight = getattr(args, "soft_tokens_weight", None)
    if soft_weight is None:
        return None
    from networks.methods.soft_tokens import create_network_from_weights

    net, _ = create_network_from_weights(
        multiplier=1.0,
        file=soft_weight,
        ae=None,
        text_encoders=None,
        unet=anima,
        for_inference=True,
    )
    net.load_weights(soft_weight)
    net.to(device, dtype=torch.bfloat16)
    net.apply_to(
        text_encoders=None, unet=anima, apply_text_encoder=False, apply_unet=True
    )
    logger.info(
        f"soft_tokens: loaded {soft_weight} "
        f"(n_layers={net.n_layers}, K={net.num_tokens}, "
        f"n_t_buckets={net.n_t_buckets}, splice={net.splice_position})"
    )
    return net


def _seqlens_from_context(context_dict, device):
    """Extract per-sample text seqlens from the context's attention mask.

    ``context['embed'][3]`` is the cached attention mask (1 inside text, 0 in
    padding) — sum along the sequence axis gives the real token count per
    sample, which the ``front_of_padding`` splice needs.
    """
    embed_mask = context_dict["embed"][3].to(device)
    return embed_mask.sum(dim=-1).to(torch.int32)


def register_spectrum_runner(fn):
    """Plug in a spectrum_denoise implementation.

    The runner must match networks.spectrum.spectrum_denoise's signature: the
    core positional args, a ``SamplerSideChannels`` (see
    ``library.inference.sampler_context``) carrying the shared DCW / SMC-CFG /
    soft-tokens / P-GRAFT / pooled-text channels, then the spectrum-specific
    keyword knobs. Called by networks/spectrum.py at import time, or by a
    downstream inference package that ships its own spectrum module.
    """
    global _SPECTRUM_RUNNER
    _SPECTRUM_RUNNER = fn


def register_spd_runner(fn):
    """Plug in an spd_denoise implementation (Spectral Progressive Diffusion).

    The runner must match networks.spd.spd_denoise's signature: the core
    positional args, a ``SamplerSideChannels`` (shared side-channels), then the
    SPD-specific keyword knobs. Called by networks/spd.py at import time,
    mirroring register_spectrum_runner.
    """
    global _SPD_RUNNER
    _SPD_RUNNER = fn


def _resolve_spd_schedule(args) -> Tuple[List[float], List[float]]:
    """Resolve (stages, transition_sigmas) for --spd from CLI args.

    Default is the bench-recommended single-late knee: one handoff
    ``0.5 → 1.0`` at σ≈0.7 (conservative; gentler trajectory than σ0.5). A
    final ``1.0`` stage is appended automatically. ``--spd_transition_sigmas``
    must have ``len(stages) - 1`` entries.
    """
    stages = list(getattr(args, "spd_stages", None) or [0.5])
    if not stages or stages[-1] != 1.0:
        stages.append(1.0)
    transition_sigmas = list(getattr(args, "spd_transition_sigmas", None) or [])
    if not transition_sigmas:
        # one default σ per handoff; single-late knee for the common 2-stage case
        transition_sigmas = [0.7] * (len(stages) - 1)
    if len(transition_sigmas) != len(stages) - 1:
        raise ValueError(
            f"--spd_transition_sigmas needs {len(stages) - 1} value(s) for "
            f"stages {stages}, got {transition_sigmas}"
        )
    return stages, transition_sigmas


class GenerationSettings:
    # ``dit_weight_dtype`` was dropped 2026-05-24: it was vestigial — the model
    # is forced to bf16 in ``load_dit_model`` regardless, so the field never
    # influenced anything. The DiT runs in bf16 for inference.
    def __init__(self, device: torch.device):
        self.device = device


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    # ``inference.parse_args`` defaults ``--device`` to None and resolves it in
    # main()'s body, but programmatic callers (GenerationRequest.to_args(), bench
    # probes) skip that block. Resolve the cuda-else-cpu default here so the one
    # chokepoint every caller funnels through can't see ``args.device is None``.
    dev = getattr(args, "device", None) or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(dev)
    logger.info(f"Using device: {device}, DiT weight precision: bfloat16")
    return GenerationSettings(device=device)


def resolve_seed(args: argparse.Namespace) -> int:
    """Return the seed to use: ``args.seed`` if set, else a fresh random one.

    Pure — does **not** mutate ``args``. Callers that need ``args.seed`` set for
    downstream saving (filename / metadata) should assign the return value
    themselves. ``generate()`` resolves a seed this way per call without writing
    back to the namespace, so one namespace is safe to reuse across calls.
    """
    return args.seed if args.seed is not None else random.randint(0, 2**32 - 1)


def compute_tile_positions(
    h_latent: int, w_latent: int, tile_size: int, overlap: int
) -> List[Tuple[int, int]]:
    """Compute (y, x) start positions for overlapping tiles covering the full latent grid."""
    stride = tile_size - overlap
    positions = []
    y = 0
    while y < h_latent:
        if y + tile_size > h_latent:
            y = h_latent - tile_size  # clamp last row
        x = 0
        while x < w_latent:
            if x + tile_size > w_latent:
                x = w_latent - tile_size  # clamp last column
            positions.append((y, x))
            if x + tile_size >= w_latent:
                break
            x += stride
        if y + tile_size >= h_latent:
            break
        y += stride
    return positions


def create_tile_blend_weight(
    tile_h: int,
    tile_w: int,
    overlap: int,
    y: int,
    x: int,
    h_latent: int,
    w_latent: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a (1, 1, 1, tile_h, tile_w) blend weight with cosine ramps on overlapping edges."""
    weight = torch.ones(1, 1, 1, tile_h, tile_w, device=device, dtype=dtype)
    if overlap <= 0:
        return weight

    # Precompute cosine ramp: (1 - cos(pi * t)) / 2 for t in [0, 1]
    ramp = torch.linspace(0.0, 1.0, overlap, device=device, dtype=dtype)
    ramp = (1.0 - torch.cos(math.pi * ramp)) / 2.0

    if y > 0:
        weight[:, :, :, :overlap, :] *= ramp[None, None, None, :, None]
    if y + tile_h < h_latent:
        weight[:, :, :, -overlap:, :] *= ramp.flip(0)[None, None, None, :, None]
    if x > 0:
        weight[:, :, :, :, :overlap] *= ramp[None, None, None, None, :]
    if x + tile_w < w_latent:
        weight[:, :, :, :, -overlap:] *= ramp.flip(0)[None, None, None, None, :]

    return weight


def generate_body_tiled(
    args: Union[argparse.Namespace, SimpleNamespace],
    anima: anima_models.Anima,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """MultiDiffusion-style tiled denoising for high-resolution generation."""
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)

    height, width = check_inputs(args)
    logger.info(
        f"Tiled diffusion: image size {height}x{width} (HxW), infer_steps: {args.infer_steps}"
    )

    tile_size = args.tile_size
    overlap = args.tile_overlap
    patch_spatial = anima.patch_spatial

    embed = context["embed"][0].to(device, dtype=torch.bfloat16)
    if context_null is None:
        context_null = context
    negative_embed = context_null["embed"][0].to(device, dtype=torch.bfloat16)

    # Soft tokens — see generate_body() for the long-form comment.
    soft_tokens_net = _setup_soft_tokens(args, anima, device)
    soft_tokens_embed_seqlens = (
        _seqlens_from_context(context, device) if soft_tokens_net is not None else None
    )
    soft_tokens_neg_seqlens = (
        _seqlens_from_context(context_null, device)
        if soft_tokens_net is not None
        else None
    )

    num_channels_latents = anima_models.Anima.LATENT_CHANNELS
    h_latent = height // 8
    w_latent = width // 8
    shape = (1, num_channels_latents, 1, h_latent, w_latent)
    latents = randn_tensor(shape, generator=seed_g, device=device, dtype=torch.bfloat16)

    positions = compute_tile_positions(h_latent, w_latent, tile_size, overlap)
    logger.info(
        f"Tiled diffusion: {len(positions)} tiles, tile_size={tile_size}, overlap={overlap}"
    )

    blend_weights = {}
    for y, x in positions:
        tile_h = min(tile_size, h_latent - y)
        tile_w = min(tile_size, w_latent - x)
        blend_weights[(y, x)] = create_tile_blend_weight(
            tile_h, tile_w, overlap, y, x, h_latent, w_latent, device, torch.bfloat16
        )

    embed = embed.to(torch.bfloat16)
    negative_embed = negative_embed.to(torch.bfloat16)

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps,
        args.flow_shift,
        device,
        tail_power=getattr(args, "sigma_tail_power", 1.0),
    )
    timesteps = timesteps.to(device, dtype=torch.bfloat16)  # σ∈[0,1] — DiT time arg

    # Create sampler. Variable kept named `er_sde` for historic minimum-diff
    # reasons; both ERSDESampler and LCMSampler share the same .step interface.
    cns = _build_cns_recolorer(args)
    er_sde = None
    if args.sampler == "er_sde":
        er_sde = inference_utils.ERSDESampler(
            sigmas, seed=args.seed, device=device, cns=cns
        )
    elif args.sampler == "lcm":
        er_sde = inference_utils.LCMSampler(sigmas, seed=args.seed, device=device)

    do_cfg = args.guidance_scale != 1.0
    smc_cfg = (
        SMCCFGState(
            lam=args.smc_cfg_lambda,
            alpha=args.smc_cfg_alpha,
        )
        if do_cfg and getattr(args, "smc_cfg", False)
        else None
    )

    pgraft_network = getattr(anima, "_pgraft_network", None)
    lora_cutoff_step = getattr(args, "lora_cutoff_step", None)

    try:
        with tqdm(total=len(timesteps), desc="Denoising steps (tiled)") as pbar:
            for i, t in enumerate(timesteps):
                # P-GRAFT: disable LoRA at cutoff step
                if (
                    pgraft_network is not None
                    and lora_cutoff_step is not None
                    and i == lora_cutoff_step
                ):
                    pgraft_network.set_enabled(False)
                    logger.info(f"P-GRAFT: Disabled LoRA at step {i}/{len(timesteps)}")

                t_expand = t.expand(latents.shape[0])
                set_hydra_sigma(anima, t_expand)
                set_step_expert_index(anima, i)
                # FEI router input — computed on the full latent (pre-tile)
                # so every tile in this step sees the same per-sample FEI.
                # Drives both the per-Linear FEI router (FEI-on-Hydra) and
                # the network-level GlobalRouter (FeRA / stacked_experts);
                # no-op when no FEI router is attached.
                compute_and_set_hydra_fei(anima, latents)

                noise_acc = torch.zeros_like(latents)
                weight_acc = torch.zeros(
                    1, 1, 1, h_latent, w_latent, device=device, dtype=torch.bfloat16
                )

                if do_cfg:
                    uncond_noise_acc = torch.zeros_like(latents)
                    uncond_weight_acc = torch.zeros(
                        1, 1, 1, h_latent, w_latent, device=device, dtype=torch.bfloat16
                    )

                for y, x in positions:
                    tile_h = min(tile_size, h_latent - y)
                    tile_w = min(tile_size, w_latent - x)
                    tile_latent = latents[:, :, :, y : y + tile_h, x : x + tile_w]
                    tile_padding_mask = torch.zeros(
                        1, 1, tile_h, tile_w, dtype=torch.bfloat16, device=device
                    )

                    h_off = y // patch_spatial
                    w_off = x // patch_spatial

                    bw = blend_weights[(y, x)]

                    if anima.blocks_to_swap:
                        anima.prepare_block_swap_before_forward()
                    # Caption-dependent routers (chimera ContentRouter and the
                    # crossattn-emb GlobalRouter) — gates depend on the caption,
                    # so fire separately for cond vs uncond. No-op otherwise.
                    set_hydra_content(anima, embed)
                    set_hydra_crossattn(anima, embed)
                    if soft_tokens_net is not None:
                        soft_tokens_net.append_postfix(
                            embed, soft_tokens_embed_seqlens, timesteps=t_expand
                        )
                    with torch.no_grad():
                        tile_pred = anima(
                            tile_latent,
                            t_expand,
                            embed,
                            padding_mask=tile_padding_mask,
                            h_offset=h_off,
                            w_offset=w_off,
                        )
                    noise_acc[:, :, :, y : y + tile_h, x : x + tile_w] += tile_pred * bw
                    weight_acc[:, :, :, y : y + tile_h, x : x + tile_w] += bw

                    if do_cfg:
                        if anima.blocks_to_swap:
                            anima.prepare_block_swap_before_forward()
                        set_hydra_content(anima, negative_embed)
                        set_hydra_crossattn(anima, negative_embed)
                        if soft_tokens_net is not None:
                            soft_tokens_net.append_postfix(
                                negative_embed,
                                soft_tokens_neg_seqlens,
                                timesteps=t_expand,
                            )
                        with torch.no_grad():
                            uncond_tile_pred = anima(
                                tile_latent,
                                t_expand,
                                negative_embed,
                                padding_mask=tile_padding_mask,
                                h_offset=h_off,
                                w_offset=w_off,
                            )
                        uncond_noise_acc[:, :, :, y : y + tile_h, x : x + tile_w] += (
                            uncond_tile_pred * bw
                        )
                        uncond_weight_acc[:, :, :, y : y + tile_h, x : x + tile_w] += bw

                noise_pred = noise_acc / weight_acc
                if do_cfg:
                    uncond_noise_pred = uncond_noise_acc / uncond_weight_acc
                    if smc_cfg is not None:
                        noise_pred = smc_cfg.combine(
                            noise_pred, uncond_noise_pred, args.guidance_scale
                        )
                    else:
                        noise_pred = uncond_noise_pred + args.guidance_scale * (
                            noise_pred - uncond_noise_pred
                        )

                denoised = latents.float() - sigmas[i] * noise_pred.float()
                if er_sde is not None:
                    new_latents = er_sde.step(latents, denoised, i)
                else:
                    new_latents = inference_utils.step(latents, noise_pred, sigmas, i)

                if getattr(args, "dcw", False) and float(sigmas[i + 1]) > 0.0:
                    from networks.dcw import apply_dcw, parse_band_mask

                    new_latents = apply_dcw(
                        new_latents.float(),
                        denoised,
                        float(sigmas[i]),
                        lam=args.dcw_lambda,
                        schedule=args.dcw_schedule,
                        bands=parse_band_mask(getattr(args, "dcw_band_mask", "LL")),
                    )

                latents = new_latents.to(latents.dtype)
                pbar.update()
    finally:
        clear_hydra_sigma(anima)
        clear_hydra_fei(anima)
        # P-GRAFT: restore LoRA for next generation
        if pgraft_network is not None and lora_cutoff_step is not None:
            pgraft_network.set_enabled(True)

    return latents


def generate_body(
    args: Union[argparse.Namespace, SimpleNamespace],
    anima: anima_models.Anima,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: Union[int, List[int]],
    latents: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Core denoising loop for Anima generation.

    Args:
        args: Generation arguments (image_size, infer_steps, guidance_scale, etc.)
        anima: Loaded DiT model.
        context: Dict with "embed" key containing text encoder outputs.
        context_null: Dict with negative prompt embeddings (or None for unconditional).
        device: Target device.
        seed: Single seed or list of seeds (for batch generation).
        latents: Optional pre-created latent noise tensor.  When provided, the
            batch dimension is taken from this tensor and seed is ignored for
            noise creation.  This enables callers (e.g. batch mode) to construct
            multi-seed batched latents externally.

    Returns:
        Denoised latent tensor (batch dimension preserved).
    """

    height, width = check_inputs(args)

    h_latent = height // 8
    w_latent = width // 8
    if (
        latents is None
        and getattr(args, "tiled_diffusion", False)
        and (h_latent > args.tile_size or w_latent > args.tile_size)
    ):
        return generate_body_tiled(args, anima, context, context_null, device, seed)

    if latents is None:
        seed_g = torch.Generator(device="cpu")
        seed_g.manual_seed(seed if isinstance(seed, int) else seed[0])

        logger.info(
            f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}"
        )

        num_channels_latents = anima_models.Anima.LATENT_CHANNELS
        shape = (1, num_channels_latents, 1, height // 8, width // 8)
        latents = randn_tensor(
            shape, generator=seed_g, device=device, dtype=torch.bfloat16
        )

    bs = latents.shape[0]
    h_latent = latents.shape[-2]
    w_latent = latents.shape[-1]

    logger.info(
        f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}, batch: {bs}"
    )

    logger.info(f"Prompt: {context['prompt']}")

    embed = context["embed"][0].to(device, dtype=torch.bfloat16)
    if context_null is None:
        context_null = context  # dummy for unconditional
    negative_embed = context_null["embed"][0].to(device, dtype=torch.bfloat16)

    # Optional pooled-text override for modulation guidance (left unset here;
    # downstream guards on ``is not None``).
    _pooled_text_pos = None
    _pooled_text_neg = None

    # Soft tokens: build + apply the monkey-patches once. The per-step
    # append_postfix(..., timesteps=t) call fires inside the loop below — and
    # is mirrored in the Spectrum runner for the --spectrum path.
    soft_tokens_net = _setup_soft_tokens(args, anima, device)
    soft_tokens_embed_seqlens = (
        _seqlens_from_context(context, device) if soft_tokens_net is not None else None
    )
    soft_tokens_neg_seqlens = (
        _seqlens_from_context(context_null, device)
        if soft_tokens_net is not None
        else None
    )

    padding_mask = torch.zeros(
        bs, 1, h_latent, w_latent, dtype=torch.bfloat16, device=device
    )

    logger.info(
        f"Embed: {embed.shape}, negative_embed: {negative_embed.shape}, latents: {latents.shape}"
    )

    if embed.shape[0] < bs:
        embed = embed.expand(bs, -1, -1)
    if negative_embed.shape[0] < bs:
        negative_embed = negative_embed.expand(bs, -1, -1)

    embed = embed.to(torch.bfloat16)
    negative_embed = negative_embed.to(torch.bfloat16)

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps,
        args.flow_shift,
        device,
        tail_power=getattr(args, "sigma_tail_power", 1.0),
    )
    timesteps = timesteps.to(device, dtype=torch.bfloat16)  # σ∈[0,1] — DiT time arg

    # DCW: load + setup the learnable calibrator if requested.
    dcw_calibrator = None
    calibrator_path = getattr(args, "dcw_calibrator", None) or getattr(
        args, "dcw_v4", None
    )
    if calibrator_path:
        from library.inference.corrections.dcw_calibrator import OnlineDCWCalibrator

        artifact_path = Path(calibrator_path)
        if artifact_path.is_dir():
            artifact_path = artifact_path / "fusion_head.safetensors"
        try:
            dcw_calibrator = OnlineDCWCalibrator.from_safetensors(
                artifact_path, device=device
            )
        except Exception as e:
            logger.warning(
                "DCW calibrator: failed to load %s: %s — disabling", artifact_path, e
            )
            dcw_calibrator = None
        if dcw_calibrator is not None:
            calib_embed_mask = (
                context["embed"][3].to(device)
                if len(context["embed"]) > 3 and context["embed"][3] is not None
                else None
            )
            dcw_calibrator.setup(
                embed=embed,
                embed_mask=calib_embed_mask,
                gain=getattr(args, "dcw_calibrator_gain", None)
                or getattr(args, "dcw_v4_alpha_gain", 1.0),
            )
            if not dcw_calibrator.is_active:
                dcw_calibrator = None  # graceful degrade to scalar/none

    # Create sampler. Variable kept named `er_sde` for historic minimum-diff
    # reasons; both ERSDESampler and LCMSampler share the same .step interface.
    cns = _build_cns_recolorer(args)
    er_sde = None
    if args.sampler == "er_sde":
        er_sde = inference_utils.ERSDESampler(
            sigmas, seed=args.seed, device=device, cns=cns
        )
    elif args.sampler == "lcm":
        er_sde = inference_utils.LCMSampler(sigmas, seed=args.seed, device=device)

    do_cfg = args.guidance_scale != 1.0
    smc_cfg = (
        SMCCFGState(
            lam=args.smc_cfg_lambda,
            alpha=args.smc_cfg_alpha,
        )
        if do_cfg and getattr(args, "smc_cfg", False)
        else None
    )

    pgraft_network = getattr(anima, "_pgraft_network", None)
    lora_cutoff_step = getattr(args, "lora_cutoff_step", None)

    # Shared conditioning side-channels handed to whichever loop runner is active
    # (spectrum / spd). The standard inline loop below reads the locals directly.
    _side_channels = SamplerSideChannels.from_args(
        args,
        pgraft_network=pgraft_network,
        lora_cutoff_step=lora_cutoff_step,
        pooled_text_pos=_pooled_text_pos,
        pooled_text_neg=_pooled_text_neg,
        dcw_calibrator=dcw_calibrator,
        smc_cfg=smc_cfg,
        soft_tokens_net=soft_tokens_net,
        soft_tokens_embed_seqlens=soft_tokens_embed_seqlens,
        soft_tokens_neg_seqlens=soft_tokens_neg_seqlens,
    )

    if getattr(args, "spd", False):
        if getattr(args, "spectrum", False):
            raise ValueError(
                "--spd and --spectrum are mutually exclusive (both replace the "
                "denoise loop). Compose them via a future SPD∘Spectrum runner, "
                "not by passing both."
            )
        if _SPD_RUNNER is None:
            raise RuntimeError(
                "--spd was passed but no SPD runner is registered. "
                "Import networks.spd before calling generate()."
            )

        stages, transition_sigmas = _resolve_spd_schedule(args)
        latents = _SPD_RUNNER(
            anima,
            latents,
            timesteps,
            sigmas,
            embed,
            negative_embed,
            padding_mask,
            args.guidance_scale,
            er_sde,
            device,
            _side_channels,
            stages=stages,
            transition_sigmas=transition_sigmas,
            seed=seed if isinstance(seed, int) else seed[0],
        )
    elif getattr(args, "spectrum", False):
        if _SPECTRUM_RUNNER is None:
            raise RuntimeError(
                "--spectrum was passed but no spectrum runner is registered. "
                "Import networks.spectrum (or the downstream inference package "
                "that registers it) before calling generate()."
            )

        latents = _SPECTRUM_RUNNER(
            anima,
            latents,
            timesteps,
            sigmas,
            embed,
            negative_embed,
            padding_mask,
            args.guidance_scale,
            er_sde,
            device,
            _side_channels,
            window_size=getattr(args, "spectrum_window_size", 2.0),
            flex_window=getattr(args, "spectrum_flex_window", 0.25),
            warmup_steps=getattr(args, "spectrum_warmup", 6),
            w=getattr(args, "spectrum_w", 0.3),
            m=getattr(args, "spectrum_m", 3),
            lam=getattr(args, "spectrum_lam", 0.1),
            stop_caching_step=getattr(args, "spectrum_stop_caching_step", -1),
            calibration_strength=getattr(args, "spectrum_calibration", 0.0),
        )
    else:
        try:
            with tqdm(total=len(timesteps), desc=f"Denoising steps ({bs}x)") as pbar:
                for i, t in enumerate(timesteps):
                    # P-GRAFT: disable LoRA at cutoff step (reference model takes over)
                    if (
                        pgraft_network is not None
                        and lora_cutoff_step is not None
                        and i == lora_cutoff_step
                    ):
                        pgraft_network.set_enabled(False)
                        logger.info(
                            f"P-GRAFT: Disabled LoRA at step {i}/{len(timesteps)}"
                        )

                    t_expand = t.expand(latents.shape[0])
                    set_hydra_sigma(anima, t_expand)
                    set_step_expert_index(anima, i)
                    compute_and_set_hydra_fei(anima, latents)
                    if dcw_calibrator is not None:
                        # Capture FEI on the pre-forward latent at warmup steps
                        # for v6 fei_obs={replace,concat} artifacts. No-op for v5.
                        dcw_calibrator.record_latent_pre_forward(i, latents)

                    set_hydra_content(anima, embed)
                    set_hydra_crossattn(anima, embed)
                    if soft_tokens_net is not None:
                        soft_tokens_net.append_postfix(
                            embed, soft_tokens_embed_seqlens, timesteps=t_expand
                        )
                    with torch.no_grad():
                        _pos_kw = (
                            {"pooled_text_override": _pooled_text_pos}
                            if _pooled_text_pos is not None
                            else {}
                        )
                        noise_pred = anima(
                            latents,
                            t_expand,
                            embed,
                            padding_mask=padding_mask,
                            **_pos_kw,
                        )

                    if do_cfg:
                        set_hydra_content(anima, negative_embed)
                        set_hydra_crossattn(anima, negative_embed)
                        if soft_tokens_net is not None:
                            soft_tokens_net.append_postfix(
                                negative_embed,
                                soft_tokens_neg_seqlens,
                                timesteps=t_expand,
                            )
                        with torch.no_grad():
                            _neg_kw = (
                                {"pooled_text_override": _pooled_text_neg}
                                if _pooled_text_neg is not None
                                else {}
                            )
                            uncond_noise_pred = anima(
                                latents,
                                t_expand,
                                negative_embed,
                                padding_mask=padding_mask,
                                **_neg_kw,
                            )
                        if smc_cfg is not None:
                            noise_pred = smc_cfg.combine(
                                noise_pred, uncond_noise_pred, args.guidance_scale
                            )
                        else:
                            noise_pred = uncond_noise_pred + args.guidance_scale * (
                                noise_pred - uncond_noise_pred
                            )

                    denoised = latents.float() - sigmas[i] * noise_pred.float()
                    if er_sde is not None:
                        new_latents = er_sde.step(latents, denoised, i)
                    else:
                        new_latents = inference_utils.step(
                            latents, noise_pred, sigmas, i
                        )

                    if dcw_calibrator is not None:
                        dcw_calibrator.record(i, noise_pred)
                        dcw_calibrator.fire_head_if_due(i)

                    lam_i_calib: Optional[float] = None
                    if float(sigmas[i + 1]) > 0.0 and (
                        dcw_calibrator is not None or getattr(args, "dcw", False)
                    ):
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
                                lam=args.dcw_lambda,
                                schedule=args.dcw_schedule,
                                bands=parse_band_mask(
                                    getattr(args, "dcw_band_mask", "LL")
                                ),
                            )

                    latents = new_latents.to(latents.dtype)

                    if dcw_calibrator is not None and dcw_calibrator.is_active:
                        # λ_scalar = α̂·gain — the prompt-adaptive equivalent of
                        # the bench's fixed `lam=0.01, schedule="one_minus_sigma"` scalar.
                        # λ_i is the post-envelope per-step value actually applied.
                        lam_scalar = dcw_calibrator.gain * dcw_calibrator.alpha_eff
                        if i < dcw_calibrator.k_warmup:
                            pbar.set_postfix_str(
                                f"λ_i={lam_i_calib:+.4f} (warmup {i + 1}/{dcw_calibrator.k_warmup})"
                                if lam_i_calib is not None
                                else f"warmup {i + 1}/{dcw_calibrator.k_warmup}"
                            )
                        else:
                            pbar.set_postfix_str(
                                f"λ_scalar={lam_scalar:+.4f} λ_i={lam_i_calib:+.4f} "
                                f"α={dcw_calibrator.alpha_eff:+.4g}"
                                if lam_i_calib is not None
                                else f"λ_scalar={lam_scalar:+.4f} α={dcw_calibrator.alpha_eff:+.4g}"
                            )

                    pbar.update()
        finally:
            clear_hydra_sigma(anima)
            clear_hydra_fei(anima)
            # P-GRAFT: restore LoRA for next generation
            if pgraft_network is not None and lora_cutoff_step is not None:
                pgraft_network.set_enabled(True)

    return latents


def generate(
    args: argparse.Namespace,
    gen_settings: GenerationSettings,
    shared_models: Optional[Dict] = None,
    precomputed_text_data: Optional[Dict] = None,
) -> torch.Tensor:
    """Main function for generation.

    Returns:
        torch.Tensor: generated latent
    """
    device = gen_settings.device

    # Resolve the seed for this call without mutating ``args`` (callers that
    # save by ``args.seed`` resolve it themselves — see ``resolve_seed`` /
    # ``GenerationRequest`` docs).
    seed = resolve_seed(args)

    if shared_models is None or "model" not in shared_models:
        # load DiT model (bf16 — see GenerationSettings note)
        anima = load_dit_model(args, device, torch.bfloat16)

        if shared_models is not None:
            shared_models["model"] = anima
    else:
        logger.info("Using shared DiT model.")
        anima: anima_models.Anima = shared_models["model"]

    if precomputed_text_data is not None:
        logger.info("Using precomputed text data.")
        context = precomputed_text_data["context"]
        context_null = precomputed_text_data["context_null"]

    else:
        logger.info("No precomputed data. Preparing image and text inputs.")
        context, context_null = prepare_text_inputs(args, device, anima, shared_models)

    # Phase 2 modulation guidance: compute guidance delta once
    if (
        getattr(args, "pooled_text_proj", None) is not None
        and getattr(args, "mod_w", 0.0) != 0.0
    ):
        setup_mod_guidance(args, anima, device, shared_models)
    else:
        anima.reset_mod_guidance()

    # EasyControl: load + apply network, VAE-encode reference image, run cond
    # pre-pass to prime per-block (K_c, V_c). Phase 1 — recomputed every step
    # at training; at inference we run it once here (no KV cache yet).
    _setup_easycontrol(args, anima, device, shared_models)

    # DAVE: arm the per-block DC-attenuation hooks (training-free diversity edit).
    # Hooks must be removed after generation — the model is shared across seeds,
    # so a stacked hook set would compound the attenuation. See dave.py.
    dave_hooks = None
    if getattr(args, "dave", None):
        from library.inference.corrections.dave import setup_dave

        dave_hooks = setup_dave(args, anima, device)
    else:
        anima.reset_dave()

    try:
        return generate_body(args, anima, context, context_null, device, seed)
    finally:
        if dave_hooks is not None:
            dave_hooks.remove()
            anima.reset_dave()


def _setup_easycontrol(args, anima, device, shared_models):
    """Load EasyControl weights, VAE-encode the reference image, prime cond KV cache.

    The cond stream is deterministic across denoising steps (cond_temb at t=0,
    no dependence on noisy target, frozen DiT + frozen LoRA), so we run it
    once via ``network.precompute_cond_kv()`` and reuse the per-block
    (K_c, V_c) tensors for every step and every CFG branch — the patched
    Block.forward then bypasses the cond stream entirely.
    """
    ec_weight = getattr(args, "easycontrol_weight", None)
    ec_image = getattr(args, "easycontrol_image", None)
    if ec_weight is None and ec_image is None:
        return None
    if ec_weight is None or ec_image is None:
        raise ValueError(
            "--easycontrol_weight and --easycontrol_image must be passed together "
            f"(got easycontrol_weight={ec_weight!r}, easycontrol_image={ec_image!r})"
        )

    from PIL import Image
    from torchvision import transforms

    from networks.methods.easycontrol import create_network_from_weights
    from library.models import qwen_vae as qwen_image_autoencoder_kl

    if getattr(args, "easycontrol_image_match_size", False):
        from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS

        with Image.open(ec_image) as _ref_for_size:
            _rw, _rh = _ref_for_size.size
        _target = _rw / _rh
        _best_wh = min(
            CONSTANT_TOKEN_BUCKETS, key=lambda wh: abs((wh[0] / wh[1]) - _target)
        )
        args.image_size = [_best_wh[1], _best_wh[0]]
        logger.info(
            f"EasyControl: image_size auto-picked from ref (aspect w/h={_target:.3f}) "
            f"-> {tuple(args.image_size)} (HxW)"
        )

    create_kwargs = {}
    if getattr(args, "easycontrol_scale", None) is not None:
        create_kwargs["cond_scale"] = float(args.easycontrol_scale)

    network, _sd = create_network_from_weights(
        multiplier=1.0,
        file=ec_weight,
        ae=None,
        text_encoders=None,
        unet=anima,
        **create_kwargs,
    )
    network.load_weights(ec_weight)
    network.to(device, dtype=torch.bfloat16)
    network.apply_to(text_encoders=None, unet=anima)

    # VAE-encode the reference image -> 4D latent.
    # Resize to args.image_size first so the cond bucket matches the target.
    h_pix, w_pix = args.image_size
    img = Image.open(ec_image).convert("RGB").resize((w_pix, h_pix), Image.LANCZOS)
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    img_t = (
        tfm(img).unsqueeze(0).to(device, dtype=torch.bfloat16)
    )  # [1,3,H,W] in [-1,1]

    vae = (shared_models or {}).get("vae")
    vae_was_shared = vae is not None
    if vae is None:
        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=getattr(args, "vae_chunk_size", None),
            disable_cache=getattr(args, "vae_disable_cache", False),
        )
        vae.to(torch.bfloat16)
        vae.eval()
        vae.to(device)

    with torch.no_grad():
        cond_latent_5d = vae.encode_pixels_to_latents(img_t)  # [1, C, 1, H', W']
        cond_latent = cond_latent_5d.squeeze(2)  # [1, C, H', W']

    if not vae_was_shared:
        vae.to("cpu")
        del vae
        torch.cuda.empty_cache()

    network.set_cond(cond_latent.to(device, dtype=torch.bfloat16))
    # KV cache: walk the cond stream once and pin per-block (K_c, V_c). Every
    # subsequent denoising step (and CFG branch) feeds these into target's
    # extended self-attention without re-running the cond stream.
    network.precompute_cond_kv()
    logger.info(
        f"EasyControl: loaded {ec_weight} "
        f"(r={network.cond_lora_dim}, scale={network.get_effective_scale():.3f}, kv-cached)"
    )
    anima._easycontrol_network = network
    return network
