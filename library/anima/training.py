import argparse
import gc
import math
import os
import random
import time
from typing import List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm
from PIL import Image

from library.runtime.device import clean_memory_on_device, synchronize_device
from library import train_util
from library.datasets.buckets import snap_sample_size
from library.training.checkpoints import (
    save_sd_model_on_epoch_end_or_stepwise_common,
    save_sd_model_on_train_end_common,
)
from library.anima import models as anima_models, weights as anima_utils
from library.models import qwen_vae as qwen_image_autoencoder_kl

from library.log import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


# Subdir under ``--output_dir`` (or the value of ``--sample_dir``) where the
# training preview gallery + staged latents land. Single source of truth for
# both ``sample_images`` and ``decode_pending_samples`` so they never drift
# apart (one writing, one reading the staged latents).
SAMPLE_SUBDIR = "sample"


def resolve_sample_dir(args: argparse.Namespace) -> str:
    """Where training samples/latents are written for this run.

    Explicit ``--sample_dir`` wins (the daemon injects a per-job path so a new
    task's gallery never replays the previous task's); otherwise it's
    ``<output_dir>/sample``. Used by both ``sample_images`` (writes PNGs +
    staged latents) and ``decode_pending_samples`` (reads the staged latents),
    so the two must share this single resolution.
    """
    return getattr(args, "sample_dir", None) or os.path.join(
        args.output_dir, SAMPLE_SUBDIR
    )


# Sentinel users can drop into captions that lack a real artist tag, so the
# shuffle/drop boundary keeps working. Stripped from caption variants before
# they reach the tokenizer (see _generate_caption_variants in
# scripts/preprocess/cache_text_embeddings.py). Callers of anima_smart_shuffle_caption
# that feed the result to a tokenizer must strip it themselves — kept inside
# the shuffle so the boundary index stays consistent with the input.
NO_ARTIST_SENTINEL = "@no-artist"


def _is_artist_tag(tag: str) -> bool:
    """True for @-prefixed artist handles; False for booru emoticons like ``@ @``.

    Matches the predicate the Anima Tagger uses (see
    ``scripts/anima_tagger/vocab.py``): a single ``@`` followed by non-space
    characters. ``@ @`` (``@_@`` after the corpus-wide ``_``→`` `` normalization)
    is a general-category eye-shape tag and must not trigger the shuffle
    boundary.
    """
    return len(tag) >= 2 and tag[0] == "@" and not tag[1].isspace()


def find_anima_prefix_end(tags: List[str]) -> int:
    """Index one past the trailing artist-handle in the leading run.

    Walks ``tags`` accepting any non-artist tags up front, then any consecutive
    artist tags, and stops at the first non-artist tag that follows an artist.
    Returns 0 if no artist tag is present anywhere (the no-artist case the
    ``@no-artist`` sentinel exists to fix). Multi-artist captions
    (e.g. ``@artist1, @artist2, …``) protect the full handle run, not just the
    first handle.
    """
    split_idx = 0
    saw_artist = False
    for idx, tag in enumerate(tags):
        if _is_artist_tag(tag):
            split_idx = idx + 1
            saw_artist = True
        elif saw_artist:
            break
    return split_idx


def strip_no_artist_sentinel(tags: List[str]) -> List[str]:
    """Drop every occurrence of :data:`NO_ARTIST_SENTINEL` from ``tags``."""
    return [t for t in tags if t != NO_ARTIST_SENTINEL]


def anima_smart_shuffle_caption(flex_tokens: List[str]) -> List[str]:
    """Shuffle caption tags with awareness of @artist prefix and 'on the ...' sections.

    - Tags up to and including the trailing artist tag of the leading run are
      kept in order (see :func:`find_anima_prefix_end`). Multi-artist captions
      and the ``@no-artist`` sentinel both preserve the full handle run.
    - Remaining tags are split into sections by 'on the ...' / 'in the ...'
      delimiters; tags within each section are shuffled independently.
    - The ``@no-artist`` sentinel is preserved in the output so the boundary
      index stays usable; callers that feed the result to a tokenizer must
      call :func:`strip_no_artist_sentinel` before joining.
    """
    split_idx = find_anima_prefix_end(flex_tokens)

    prefix = flex_tokens[:split_idx]
    suffix = flex_tokens[split_idx:]

    # Split suffix into sections delimited by "on the ..." tags
    sections: list[list[str]] = [[]]
    for tag in suffix:
        if tag.startswith("On the ") or tag.startswith("In the "):
            sections.append([tag])
        else:
            sections[-1].append(tag)

    result = list(prefix)
    for section in sections:
        if not section:
            continue
        if section[0].startswith("On the ") or section[0].startswith("In the "):
            header, body = [section[0]], section[1:]
        else:
            header, body = [], section
        shuffled = body.copy()
        random.shuffle(shuffled)
        result.extend(header + shuffled)
    return result


def add_anima_training_arguments(parser: argparse.ArgumentParser):
    """Add Anima-specific training arguments to the parser."""
    parser.add_argument(
        "--qwen3",
        type=str,
        default=None,
        help="Path to Qwen3-0.6B model (safetensors file or directory)",
    )
    parser.add_argument(
        "--llm_adapter_path",
        type=str,
        default=None,
        help="Path to separate LLM adapter weights. If None, adapter is loaded from DiT file if present",
    )
    parser.add_argument(
        "--llm_adapter_lr",
        type=float,
        default=None,
        help="Learning rate for LLM adapter. None=same as base LR, 0=freeze adapter",
    )
    parser.add_argument(
        "--self_attn_lr",
        type=float,
        default=None,
        help="Learning rate for self-attention layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--cross_attn_lr",
        type=float,
        default=None,
        help="Learning rate for cross-attention layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=None,
        help="Learning rate for MLP layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--mod_lr",
        type=float,
        default=None,
        help="Learning rate for AdaLN modulation layers. None=same as base LR, 0=freeze. Note: mod layers are not included in LoRA by default.",
    )
    parser.add_argument(
        "--t5_tokenizer_path",
        type=str,
        default=None,
        help="Path to T5 tokenizer directory. If None, uses bundled library/anima/configs/t5_old/",
    )
    parser.add_argument(
        "--qwen3_max_token_length",
        type=int,
        default=512,
        help="Maximum token length for Qwen3 tokenizer (default: 512)",
    )
    parser.add_argument(
        "--t5_max_token_length",
        type=int,
        default=512,
        help="Maximum token length for T5 tokenizer (default: 512)",
    )
    parser.add_argument(
        "--cache_llm_adapter_outputs",
        action="store_true",
        help="Cache LLM adapter outputs (cross-attention embeddings) in the text encoder cache and skip running the adapter during training. "
        "Requires --cache_text_encoder_outputs. Incompatible with LoRA training for the LLM adapter.",
    )
    parser.add_argument(
        "--use_easycontrol",
        action="store_true",
        help="Enable EasyControl image conditioning (extended self-attn KV with "
        "VAE-encoded reference). Requires the network module to expose set_cond_tokens "
        "(e.g. networks.methods.easycontrol). The cond input is the clean VAE latent of "
        "the reference image; for ref==target training (Phase 1) this reuses the "
        "existing VAE-cache output — no new sidecar required.",
    )
    parser.add_argument(
        "--easycontrol_drop_p",
        type=float,
        default=0.1,
        help="EasyControl image-conditioning dropout probability per batch (CFG dropout "
        "for image branch). Independent of text-side caption_dropout_rate; default 0.1.",
    )
    parser.add_argument(
        "--easycontrol_cond_noise_max",
        type=float,
        default=0.0,
        help="Per-step training-only additive Gaussian noise on the cond latent before "
        "set_cond. Sampled per-sample as sigma ~ U(0, this) and applied as "
        "cond + sigma * eps. 0.0 disables (default = current ref==target behavior). "
        "Use a small positive value (e.g. 0.3-0.7) to weaken cond's blueprint "
        "dominance and force text to carry the high-frequency residual; sigma=0 "
        "stays in the training distribution so clean-cond inference still works.",
    )
    parser.add_argument(
        "--use_controlnet",
        action="store_true",
        help="Enable ControlNet-LLLite image conditioning (per-block zero-conv "
        "lateral injection of conditioning features). Requires conditioning_data_dir "
        "in the dataset config pointing to paired conditioning images (depth maps, "
        "edge maps, reference images). The conditioning latent is loaded from "
        "pre-cached VAE latents with _anima_cond suffix.",
    )
    parser.add_argument(
        "--controlnet_drop_p",
        type=float,
        default=0.1,
        help="ControlNet image-conditioning dropout probability per batch (CFG dropout "
        "for image branch). Independent of text-side caption_dropout_rate; default 0.1.",
    )
    parser.add_argument(
        "--cond_diff_loss",
        action="store_true",
        help="Weight the per-pixel FM loss by the cond↔target latent difference "
        "(library/training/losses.py::compute_cond_diff_weight). For paired "
        "cond≠target tasks (cond_cache_dir subsets: sanitize / near-twin pose / "
        "hair-color twins) where the pair differs only in a narrow edit region — "
        "concentrates gradient there instead of on the copy-through behavior the "
        "extended attention already provides. Per-image mean-normalized (pure "
        "gradient reallocation, loss scale unchanged). No-op on batches without "
        "cond_latents. NOT for colorize (gray→color differs everywhere; the map "
        "degenerates to ~uniform).",
    )
    parser.add_argument(
        "--cond_diff_loss_floor",
        type=float,
        default=0.2,
        help="Minimum (pre-normalization) weight outside the edit region. Keeps "
        "the copy-everything-else anchor; 0 would license drift outside the "
        "bubble. Default 0.2.",
    )
    parser.add_argument(
        "--cond_diff_loss_blur",
        type=float,
        default=1.5,
        help="Gaussian sigma (latent px) applied to the diff map so the weight "
        "covers bubble interiors plus a halo, not just strokes/outlines. "
        "Default 1.5.",
    )
    parser.add_argument(
        "--cond_diff_loss_quantile",
        type=float,
        default=0.9,
        help="Per-image robust scale for the diff map: values at/above this "
        "quantile saturate to full weight. Default 0.9.",
    )
    # BYG (Bootstrap Your Generator): owning-step rank-64 LoRA, multi-forward
    # objective (bootstrap rollout + DDS prior + cycle + identity), conditioned on
    # a token-concat source latent; no edited target exists. See
    # docs/proposal/byg_unpaired_editing.md and networks/methods/byg.py.
    parser.add_argument(
        "--use_byg",
        action="store_true",
        help="Enable BYG unpaired-editing training (BYGMethodAdapter owns the "
        "whole step). Requires a BYG dataset providing the source latent "
        "(cond_cache_dir) plus four cached text conditionings per image "
        "(src_caption / tgt_caption / instruction / reverse_instruction).",
    )
    parser.add_argument(
        "--byg_lambda_prior",
        type=float,
        default=1.0,
        help="Weight of the DDS-style prior loss L_prior (paper Eq. 5, default 1.0).",
    )
    parser.add_argument(
        "--byg_lambda_id",
        type=float,
        default=0.2,
        help="Weight of the anti-collapse identity loss L_id (paper Eq. 5, default 0.2). "
        "Backwarded on an independent graph to keep peak VRAM down.",
    )
    parser.add_argument(
        "--byg_lambda_cycle",
        type=float,
        default=1.0,
        help="Weight of the cycle-consistency loss L_cycle (paper Eq. 5, default 1.0).",
    )
    parser.add_argument(
        "--byg_alpha",
        type=float,
        default=0.1,
        help="MSE magnitude-anchor weight inside the prior: L_prior = L_dir + alpha*L_MSE "
        "(paper §4.2). Anchors edit velocity magnitude against the frozen base.",
    )
    parser.add_argument(
        "--byg_rollout_steps",
        type=int,
        default=10,
        help="Number of Euler integration steps n for the bootstrap rollout that "
        "fabricates the pseudo-noisy-target y~_t and clean y~_0 (paper App. B.1 = 10). "
        "Shorter = cheaper, lower-quality pseudo-targets. Ignored when "
        "byg_rollout_sigmas is set (the explicit grid defines n).",
    )
    parser.add_argument(
        "--byg_rollout_sigmas",
        type=str,
        default=None,
        help="Explicit non-uniform rollout sigma grid, overriding the uniform "
        "byg_rollout_steps schedule. A strictly-descending list from 1.0 to 0.0 "
        "inclusive (n+1 nodes -> n Euler steps); the variable step size dsigma_j = "
        "sigma_j - sigma_{j+1} is used and the training timestep t is sampled from the "
        "interior nodes. Anima resolves x0 by sigma~=0.45 (project_sigma_signal_resolves"
        "_by_045), so concentrating nodes in the forming band [0.4, 0.9] and taking one "
        "coarse step through the resolved tail keeps y~_0 sharp at low NFE. TOML: a "
        "float array (e.g. [1.0, 0.85, 0.70, 0.55, 0.40, 0.0]); CLI: comma-separated. "
        "Default None = uniform (bit-identical to the byg_rollout_steps path).",
    )
    parser.add_argument(
        "--byg_snapshot_every",
        type=int,
        default=200,
        help="Hard-refresh the frozen weight snapshot used for the bootstrap rollout "
        "every N optimizer steps (v1 EMA-free path). Ignored when byg_ema_decay > 0.",
    )
    parser.add_argument(
        "--byg_ema_decay",
        type=float,
        default=0.0,
        help="EMA decay for the bootstrap copy of the trainable LoRA weights. 0.0 (default) "
        "uses periodic hard snapshots (byg_snapshot_every) instead; >0 enables the "
        "paper-faithful per-step EMA moving average (paper Alg. 1 line 36).",
    )
    parser.add_argument(
        "--byg_identity_warmup_steps",
        type=int,
        default=200,
        help="Train identity-only for the first N steps so the model learns to use the "
        "condition before editing (paper App. B.1 = 200). Anti-collapse, load-bearing.",
    )
    parser.add_argument(
        "--byg_identity_prob",
        type=float,
        default=0.15,
        help="Probability that a post-warmup step is a random identity step (paper "
        "App. B.1 = 0.15 for images). Regularizer against identity collapse.",
    )
    parser.add_argument(
        "--byg_text_dir",
        type=str,
        default=None,
        help="Directory of per-image BYG edit-tuple sidecars "
        "(<stem>_byg.safetensors). Default: post_image_dataset/byg.",
    )
    parser.add_argument(
        "--byg_prior_symmetric",
        action="store_true",
        help="Apply the prior loss to the reverse pass as well (L_prior^fwd + L_prior^rev, "
        "paper Eq. 5). On by default (v2); the byg.toml sets it. Disable for the v1 "
        "fwd-only prior to save two frozen-base forwards per step.",
    )
    parser.add_argument(
        "--use_shuffled_caption_variants",
        action="store_true",
        help="Consume preprocessed caption-shuffle variants from the text-encoder cache "
        "(written by `cache_text_embeddings.py --caption_shuffle_variants N`). When the "
        "cache file has a `num_variants` tensor, a random variant is drawn per sample. "
        "Falls back to single-variant silently if the cache has no variants. Inline "
        "multi-variant generation at training time is no longer supported — preprocess "
        "the variants first.",
    )
    parser.add_argument(
        "--use_shuffled_caption_variants_only",
        action="store_true",
        help="Like --use_shuffled_caption_variants but never draw the pristine v0 "
        "variant — sample uniformly over the shuffled+tag-dropped v1..v{N-1} only. "
        "Use when v0 (the full, unshuffled caption) should be excluded from training, "
        "e.g. colorize wants every step to see a partial color spec. Implies "
        "--use_shuffled_caption_variants; no-op on single-variant caches.",
    )
    parser.add_argument(
        "--use_randomized_caption_variants",
        action="store_true",
        help="Lexinvariant tag regularization (arXiv:2305.16349): consume the "
        "identity-randomized r-family from the TE cache (written by "
        "`cache_text_embeddings.py --caption_tag_randomize_rate p`). Samples v0 "
        "(pristine) 20%% / r1..r{N-1} (tags' identities erased) 80%%, instead of the "
        "shuffled v-family. One cache holds both families, so this is an A/B toggle "
        "against --use_shuffled_caption_variants. Falls back to the v-family (warns "
        "once) if the cache has no r-family. Implies --use_shuffled_caption_variants.",
    )
    parser.add_argument(
        "--use_randomized_caption_variants_only",
        action="store_true",
        help="Like --use_randomized_caption_variants but never draw the pristine v0 "
        "— sample uniformly over the identity-randomized r1..r{N-1} only. Implies "
        "--use_randomized_caption_variants; no-op on caches without an r-family.",
    )
    parser.add_argument(
        "--artist_filter",
        type=str,
        default=None,
        help="If set, only train on images whose caption contains the `@<artist>` tag. "
        "Pass with or without the leading `@` (e.g. `--artist_filter sincos`). "
        "Output is redirected to output/ckpt-artist/<output_name>_<artist>.safetensors.",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=1.0,
        help="Timestep distribution shift for rectified flow training (default: 1.0)",
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default="sigmoid",
        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift"],
        help="Timestep sampling method (default: sigmoid (logit normal))",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale factor for sigmoid (logit_normal) timestep sampling (default: 1.0)",
    )
    parser.add_argument(
        "--sigmoid_bias",
        type=float,
        default=0.0,
        help="Logit-space mean shift for sigmoid/shift/flux_shift sampling: "
        "sigmoid(sigmoid_scale*randn + sigmoid_bias). >0 skews mass toward high σ "
        "(structure regime), <0 toward low σ. 0.0 (default) = unbiased logit-normal.",
    )
    parser.add_argument(
        "--lora_fp32_accumulation",
        action="store_true",
        help="[DEPRECATED, no-op] fp32 accumulation is now unconditional in "
        "LoRA/Hydra bottleneck matmuls. Flag accepted for one release "
        "cycle; will be removed.",
    )
    parser.add_argument(
        "--attn_mode",
        choices=[
            "torch",
            "flash",
            # "flash4",  # not supported yet (flash-attention-sm120 disabled)
            "sageattn",
            "flex",
            "sdpa",
        ],  # "sdpa" is for backward compatibility
        default=None,
        help="Attention implementation to use. Default is None (torch). sageattn does not support training (inference only). This option overrides --sdpa."
        "",
    )
    parser.add_argument(
        "--attn_softmax_scale",
        type=float,
        default=None,
        help="Custom softmax scale for attention (default: 1/sqrt(head_dim)). Larger values sharpen attention and improve bf16 precision."
        " For head_dim=128, default is ~0.088. Try 0.1-0.15 for better low-precision stability.",
    )
    parser.add_argument(
        "--v100_flash_stability",
        choices=["off", "hybrid", "safe"],
        default=None,
        help="Experimental guard for flash-attention-v100 on Volta/V100 fp16 training. "
        "off keeps normal flash behavior; hybrid keeps self-attn on flash and routes "
        "cross-attn through torch SDPA; safe keeps flash but adds finite checks around "
        "attention. If unset, ANIMA_V100_FLASH_STABILITY may provide off|hybrid|safe.",
    )
    parser.add_argument(
        "--debug_finite_checks",
        action="store_true",
        help="Fail fast when q/k/v, attention outputs, block residuals, loss, or grads "
        "contain NaN/Inf. Also enabled by ANIMA_DEBUG_FINITE=1.",
    )
    parser.add_argument(
        "--vae_chunk_size",
        type=int,
        default=None,
        help="Spatial chunk size for VAE encoding/decoding to reduce memory usage. Must be even number. If not specified, chunking is disabled (official behavior)."
        + "",
    )
    parser.add_argument(
        "--vae_disable_cache",
        action="store_true",
        help="Disable internal VAE caching mechanism to reduce memory usage. Encoding / decoding will also be faster, but this differs from official behavior."
        + "",
    )

    parser.add_argument(
        "--ema",
        action="store_true",
        help="Enable Exponential Moving Average. Requires --ema_decay to be set.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate. Typical values: 0.999, 0.9999. (default: 0.9999)",
    )
    parser.add_argument(
        "--ema_device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device for EMA shadow parameters. 'cpu' saves GPU VRAM but slower update. (default: gpu)",
    )
    parser.add_argument(
        "--ema_use_num_updates",
        action="store_true",
        help="Use warmup schedule for EMA decay: min(decay, (1+num_updates)/(10+num_updates))",
    )
    parser.add_argument(
        "--ema_use_feedback",
        action="store_true",
        help="Feed back EMA decay to training parameters (experimental)",
    )
    parser.add_argument(
        "--ema_param_multiplier",
        type=float,
        default=1.0,
        help="Multiply parameters each EMA update step (default: 1.0, no effect)",
    )
    parser.add_argument(
        "--ema_resume_path",
        type=str,
        default=None,
        help="Path to EMA model safetensors file to resume EMA state from a previous run",
    )

    # Variance-reduced flow-matching loss (AsymFlow §5.2, arXiv:2605.12964;
    # bench/fm_vr_headroom/proposal.md). Gated off by default.
    parser.add_argument(
        "--vr_loss_weight",
        type=float,
        default=0.0,
        help="Weight of the VR control-variate correction on the FM loss. "
        "0 = standard FM (default). 1.0 = paper recipe. When > 0, the trainer "
        "runs one extra no-grad forward per step on the FEI-low-passed latent "
        "through the *same* trainable DiT with the adapter zeroed "
        "(network.set_multiplier(0)) — equivalent to a frozen base DiT for "
        "LoRA-family runs, without holding a second model copy in VRAM.",
    )
    parser.add_argument(
        "--vr_fei_sigma_low_div",
        type=float,
        default=4.0,
        help="Divisor for the FEI low-pass kernel used to build x_0^L "
        "(σ_low = min(H_lat, W_lat) / div). Matches live FEI default (4.0).",
    )
    parser.add_argument(
        "--vr_sigma_min",
        type=float,
        default=1e-3,
        help="Floor on σ_t in the VR loss denominator (AsymFlow §6.1). "
        "Defensive against low-σ instability in the 1/σ_t factor. 0 disables.",
    )
    parser.add_argument(
        "--vr_lambda_beta",
        type=float,
        default=0.01,
        help="EMA rate for the online λ estimator: "
        "λ_ema ← (1−β)·λ_ema + β·λ_batch. Default 0.01 over typically B=4.",
    )

    parser.add_argument(
        "--inversion_dir",
        type=str,
        default=None,
        help="Directory containing <stem>_inverted_run{i}.safetensors files. If set, "
        "inversions are loaded alongside each sample for functional-loss supervision.",
    )
    parser.add_argument(
        "--functional_loss_weight",
        type=float,
        default=0.0,
        help="Weight for functional MSE loss: MSE between cross_attn output_proj captures "
        "from (t5+postfix) forward and (inversion) forward, summed over functional_loss_blocks. "
        "0 disables (default).",
    )
    parser.add_argument(
        "--functional_loss_blocks",
        type=str,
        default="8,12,16,20",
        help="Comma-separated DiT block indices at which to capture cross_attn.output_proj "
        "outputs for functional MSE loss.",
    )

    # Liveness audit is always on: a configured-ON aux loss that never consumes
    # its aux input ERROR-logs with a greppable `LIVENESS:` prefix at step 25 and
    # run end. This flag escalates that to a hard abort.
    parser.add_argument(
        "--liveness_strict",
        action="store_true",
        help="Escalate the LIVENESS audit from ERROR log to hard abort: if a "
        "configured-ON aux loss has consumed its aux input on 0 train batches "
        "by the step-25 early check (or by run end), raise instead of training "
        "on as a silent baseline. Partial coverage only warns either way.",
    )
    parser.add_argument(
        "--functional_loss_num_runs",
        type=int,
        default=3,
        help="Number of inversion runs expected per image (aggregate_by of make invert). "
        "Each training step samples one run stochastically. Default: 3.",
    )


def compute_loss_weighting_for_anima(
    weighting_scheme: str, sigmas: torch.Tensor
) -> torch.Tensor:
    """Compute loss weighting for Anima training.

    Same schemes as SD3 but can add Anima-specific ones if needed in future.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    elif weighting_scheme == "none" or weighting_scheme is None:
        weighting = torch.ones_like(sigmas)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def get_anima_param_groups(
    dit,
    base_lr: float,
    self_attn_lr: Optional[float] = None,
    cross_attn_lr: Optional[float] = None,
    mlp_lr: Optional[float] = None,
    mod_lr: Optional[float] = None,
    llm_adapter_lr: Optional[float] = None,
):
    """Create parameter groups for Anima training with separate learning rates.

    Args:
        dit: Anima model
        base_lr: Base learning rate
        self_attn_lr: LR for self-attention layers (None = base_lr, 0 = freeze)
        cross_attn_lr: LR for cross-attention layers
        mlp_lr: LR for MLP layers
        mod_lr: LR for AdaLN modulation layers
        llm_adapter_lr: LR for LLM adapter

    Returns:
        List of parameter group dicts for optimizer
    """
    if self_attn_lr is None:
        self_attn_lr = base_lr
    if cross_attn_lr is None:
        cross_attn_lr = base_lr
    if mlp_lr is None:
        mlp_lr = base_lr
    if mod_lr is None:
        mod_lr = base_lr
    if llm_adapter_lr is None:
        llm_adapter_lr = base_lr

    base_params = []
    self_attn_params = []
    cross_attn_params = []
    mlp_params = []
    mod_params = []
    llm_adapter_params = []

    for name, p in dit.named_parameters():
        p.original_name = name

        if "llm_adapter" in name:
            llm_adapter_params.append(p)
        elif ".self_attn" in name:
            self_attn_params.append(p)
        elif ".cross_attn" in name:
            cross_attn_params.append(p)
        elif ".mlp" in name:
            mlp_params.append(p)
        elif ".adaln_modulation" in name:
            mod_params.append(p)
        else:
            base_params.append(p)

    logger.info("Parameter groups:")
    logger.info(f"  base_params: {len(base_params)} (lr={base_lr})")
    logger.info(f"  self_attn_params: {len(self_attn_params)} (lr={self_attn_lr})")
    logger.info(f"  cross_attn_params: {len(cross_attn_params)} (lr={cross_attn_lr})")
    logger.info(f"  mlp_params: {len(mlp_params)} (lr={mlp_lr})")
    logger.info(f"  mod_params: {len(mod_params)} (lr={mod_lr})")
    logger.info(
        f"  llm_adapter_params: {len(llm_adapter_params)} (lr={llm_adapter_lr})"
    )

    param_groups = []
    for lr, params, name in [
        (base_lr, base_params, "base"),
        (self_attn_lr, self_attn_params, "self_attn"),
        (cross_attn_lr, cross_attn_params, "cross_attn"),
        (mlp_lr, mlp_params, "mlp"),
        (mod_lr, mod_params, "mod"),
        (llm_adapter_lr, llm_adapter_params, "llm_adapter"),
    ]:
        if lr == 0:
            for p in params:
                p.requires_grad_(False)
            logger.info(f"  Frozen {name} params ({len(params)} parameters)")
        elif len(params) > 0:
            param_groups.append({"params": params, "lr": lr})

    total_trainable = sum(
        p.numel() for group in param_groups for p in group["params"] if p.requires_grad
    )
    logger.info(f"Total trainable parameters: {total_trainable:,}")

    return param_groups


def _get_ema_filename(ckpt_file):
    """Get EMA filename by adding ema_ prefix to the basename of a checkpoint file."""
    dirpath = os.path.dirname(ckpt_file)
    basename = os.path.basename(ckpt_file)
    return os.path.join(dirpath, f"ema_{basename}")


def _save_ema_model(ema, dit, ckpt_file, sai_metadata, save_dtype):
    """Save EMA model as standard format with ema_ prefix.

    Builds the EMA state dict directly from shadow params without mutating model
    parameters in place — keeps stepwise save side-effect-free on the live model.
    """
    if ema.shadow_params is None:
        logger.warning("EMA shadow_params is None (worker process?), skipping EMA save")
        return

    ema_file = _get_ema_filename(ckpt_file)

    dit_sd = dit.state_dict()
    trainable_idx = 0
    for name, p in dit.named_parameters():
        if p.requires_grad:
            dit_sd[name] = ema.shadow_params[trainable_idx].data.to(p.device)
            trainable_idx += 1

    anima_utils.save_anima_model(ema_file, dit_sd, sai_metadata, save_dtype)
    logger.info(f"EMA model saved: {ema_file}")
    del dit_sd


def save_anima_model_on_train_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    dit: anima_models.Anima,
    ema=None,
):
    """Save Anima model at the end of training."""

    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec_dataclass(
            args, lora=False
        ).to_metadata_dict()
        dit_sd = dit.state_dict()
        # Save with 'net.' prefix for ComfyUI compatibility
        anima_utils.save_anima_model(ckpt_file, dit_sd, sai_metadata, save_dtype)
        if ema is not None:
            _save_ema_model(ema, dit, ckpt_file, sai_metadata, save_dtype)

    save_sd_model_on_train_end_common(
        args, True, True, epoch, global_step, sd_saver, None
    )


def save_anima_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator: Accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    dit: anima_models.Anima,
    ema=None,
):
    """Save Anima model at epoch end or specific steps."""

    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec_dataclass(
            args, lora=False
        ).to_metadata_dict()
        dit_sd = dit.state_dict()
        anima_utils.save_anima_model(ckpt_file, dit_sd, sai_metadata, save_dtype)

        if ema is not None:
            _save_ema_model(ema, dit, ckpt_file, sai_metadata, save_dtype)

    save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        True,
        True,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        None,
    )


# Sampling (Euler discrete / ER-SDE for rectified flow)
def do_sample(
    height: int,
    width: int,
    seed: Optional[int],
    dit: anima_models.Anima,
    crossattn_emb: torch.Tensor,
    steps: int,
    dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float = 1.0,
    flow_shift: float = 3.0,
    neg_crossattn_emb: Optional[torch.Tensor] = None,
    show_progress: bool = True,
    sampler: str = "euler",
) -> torch.Tensor:
    """Generate a sample using configurable sampling for rectified flow.

    Args:
        height, width: Output image dimensions
        seed: Random seed (None for random)
        dit: Anima model
        crossattn_emb: Cross-attention embeddings (B, N, D)
        steps: Number of sampling steps
        dtype: Compute dtype
        device: Compute device
        guidance_scale: CFG scale (1.0 = no guidance)
        flow_shift: Flow shift parameter for rectified flow
        neg_crossattn_emb: Negative cross-attention embeddings for CFG
        sampler: Sampling algorithm - "euler" (deterministic ODE),
                 "er_sde" (Extended Reverse-Time SDE)

    Returns:
        Denoised latents
    """
    from library.inference import sampling as inference_utils

    # 5D latent (1, 16, T=1, H/8, W/8) — singleton temporal axis at dim 2.
    latent_h = height // 8
    latent_w = width // 8

    if seed is not None:
        # Device-matched generator: torch.manual_seed() returns the *CPU* default
        # generator, which raises when passed to a CUDA torch.randn. Build the
        # generator on the actual device so seeded previews are reproducible on
        # both CPU and CUDA.
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    # Noise in fp32 — the whole Euler trajectory accumulates in fp32 (x stays
    # fp32 below), so fp32 noise avoids the initial bf16 quantization error
    # compounding across steps. NOTE: the inference path (library/inference/
    # generation.py:296) generates noise in bf16; this preview path is
    # intentionally fp32 for preview quality, not parity.
    noise = torch.randn(
        (1, 16, 1, latent_h, latent_w),
        dtype=torch.float32,
        generator=generator,
        device=device,
    )

    # Reuse the inference-side sigma schedule (library/inference/sampling.py::
    # get_timesteps_sigmas) — bit-exact to the inline linspace + flow-shift map.
    # It returns sigmas on CPU; move to device for the per-step math below.
    _, sigmas = inference_utils.get_timesteps_sigmas(steps, float(flow_shift), device)
    sigmas = sigmas.to(device)

    x = noise.clone()

    # Padding mask (zeros = no padding) — fp32; the DiT's internal ops handle
    # dtype casting via autocast/weight dtype.
    padding_mask = torch.zeros(
        1, 1, latent_h, latent_w, dtype=torch.float32, device=device
    )

    use_cfg = guidance_scale > 1.0 and neg_crossattn_emb is not None

    # Create sampler object for er_sde; euler uses inline steps.
    # Pass fp32 sigmas — the samplers' .step() calls .float() internally.
    sampler_obj = None
    if sampler == "er_sde":
        sampler_obj = inference_utils.ERSDESampler(sigmas, seed=seed, device=device)

    for i in tqdm(range(steps), desc="Sampling", disable=not show_progress):
        # Cast x to model dtype for the DiT forward pass; keep sigmas/t in fp32
        # (the t_embedder handles dtype internally).
        t = sigmas[i].unsqueeze(0)

        if use_cfg:
            # CFG: two separate passes to reduce memory usage
            pos_out = dit(x.to(dtype), t, crossattn_emb, padding_mask=padding_mask)
            pos_out = pos_out.float()
            neg_out = dit(x.to(dtype), t, neg_crossattn_emb, padding_mask=padding_mask)
            neg_out = neg_out.float()

            model_output = neg_out + guidance_scale * (pos_out - neg_out)
        else:
            model_output = dit(x.to(dtype), t, crossattn_emb, padding_mask=padding_mask)
            model_output = model_output.float()

        # Compute denoised prediction in fp32
        denoised = x - sigmas[i] * model_output

        if sampler_obj is not None:
            # ER-SDE: delegate to sampler (expects fp32)
            x = sampler_obj.step(x, denoised, i)
        else:
            # Euler ODE step — reuse the inference-side step() (algebraically
            # identical to the inline x + model_output*(σ_{i+1}-σ_i)).
            x = inference_utils.step(x, model_output, sigmas, i)

    # x accumulated in fp32 throughout; normalize to the model dtype at the API
    # boundary (callers stash latents to disk — bf16 halves the footprint).
    return x.to(dtype)


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    dit: anima_models.Anima,
    vae,
    text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    sample_prompts_te_outputs=None,
    prompt_replacement=None,
    network=None,
    progress_sink=None,
):
    """Generate sample images during training.

    This is a simplified sampler for Anima - it generates images using the current model state.

    Mirrors the validation phase's model handling (see
    ``library/training/validation.py::run_validation``): the adapter ``network``
    is put in ``eval()`` for the duration (so dropout / rank-dropout match how
    samples would be generated at inference) and restored to ``train()``
    afterward, reusing the live in-GPU model rather than reloading anything. The
    optimizer eval/train swap is handled by the caller in ``loop.py``.
    """
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:
                return

    logger.info(f"Generating sample images at step {steps}")
    if not os.path.isfile(args.sample_prompts) and sample_prompts_te_outputs is None:
        logger.error(f"No prompt file: {args.sample_prompts}")
        return

    dit = accelerator.unwrap_model(dit)
    if text_encoder is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)
    # Adapter to eval for the duration — same as the validation phase, so
    # dropout / rank-dropout don't perturb the previews. Restored in finally.
    net = accelerator.unwrap_model(network) if network is not None else None
    if net is not None:
        net.eval()

    dit.switch_block_swap_for_inference()

    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = resolve_sample_dir(args)
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        )
    except Exception:
        pass

    try:
        with torch.no_grad(), accelerator.autocast():
            for prompt_dict in prompts:
                dit.prepare_block_swap_before_forward()
                _sample_image_inference(
                    accelerator,
                    args,
                    dit,
                    text_encoder,
                    vae,
                    tokenize_strategy,
                    text_encoding_strategy,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    sample_prompts_te_outputs,
                    prompt_replacement,
                )
    finally:
        # Restore RNG + model state even if a prompt errors, so training
        # resumes exactly where it left off (mirrors run_validation's finally).
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        dit.switch_block_swap_for_training()
        if net is not None:
            net.train()
        # No clean_memory_on_device() here on purpose: emptying the CUDA cache
        # at the sample<->train boundary is what made VRAM visibly fluctuate.
        # Letting the caching allocator hold its blocks keeps usage flat (peak
        # settles at max(training, sampling) and stays there).

    # Decode this round's latents now for per-epoch visibility; block-swap runs
    # defer to end-of-training decode_pending_samples (see _should_decode_inline).
    # Main process only.
    if accelerator.is_main_process and _should_decode_inline(args):
        # Evict the DiT to CPU before the VAE decode so the two are never
        # co-resident, then restore block-swap placement. Mirrors train.py teardown.
        dit.to("cpu")
        clean_memory_on_device(accelerator.device)
        try:
            decode_pending_samples(accelerator, args, vae, progress_sink=progress_sink)
        finally:
            dit.move_to_device_except_swap_blocks(accelerator.device)
            dit.prepare_block_swap_before_forward()


def _should_decode_inline(args) -> bool:
    """Whether to decode sample latents to PNG right after each sampling event
    (per-epoch visibility) vs. deferring the whole batch to end of training.

    Explicit ``--sample_decode_inline`` wins; otherwise auto — inline when the
    run isn't block-swapping (``blocks_to_swap == 0``), deferred when it is. The
    inline path parks the DiT on CPU before the VAE decode (so the two are never
    co-resident), so it's OOM-safe either way; the block-swap default still
    defers because on a tight card the repeated full-DiT CPU↔GPU transfer per
    sample event isn't worth paying when the end-of-training decode is free."""
    explicit = getattr(args, "sample_decode_inline", None)
    if isinstance(explicit, str):
        s = explicit.strip().lower()
        explicit = (
            None
            if s in ("", "auto", "none")
            else s
            in (
                "1",
                "true",
                "yes",
                "on",
            )
        )
    if explicit is not None:
        return bool(explicit)
    return not getattr(args, "blocks_to_swap", 0)


def _sample_image_inference(
    accelerator,
    args,
    dit,
    text_encoder,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    tokenize_strategy,
    text_encoding_strategy,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    """Generate a single sample image."""
    prompt = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt", "")
    sample_steps = prompt_dict.get("sample_steps", 30)
    _img_size = getattr(args, "sample_image_size", [1024, 1024])
    width = prompt_dict.get("width", _img_size[1])
    height = prompt_dict.get("height", _img_size[0])
    scale = prompt_dict.get("scale", getattr(args, "sample_guidance_scale", 3.5))
    seed = prompt_dict.get("seed")
    if seed is None:
        # No explicit --d on this prompt: pin to a stable per-prompt seed so
        # the same prompt renders from identical noise at every epoch — that's
        # what makes the per-epoch gallery comparable for spotting overfitting.
        # Offset by the prompt's index so different prompts still get distinct
        # noise. Falls back to 0 if the run somehow has no seed.
        base_seed = getattr(args, "seed", None)
        seed = (base_seed if base_seed is not None else 0) + int(
            prompt_dict.get("enum", 0)
        )
    flow_shift = prompt_dict.get("flow_shift", getattr(args, "sample_flow_shift", 5.0))
    sampler = prompt_dict.get(
        "sample_sampler", getattr(args, "sample_sampler", "euler")
    )

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt:
            negative_prompt = negative_prompt.replace(
                prompt_replacement[0], prompt_replacement[1]
            )

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # seed all CUDA devices for multi-GPU

    width, height = snap_sample_size(width, height)

    # Train-time sampling runs through the same compiled blocks as training. The
    # compile token budget already covers the prompts present at startup
    # (train.py::_sample_prompt_token_counts), but prompts are re-read from disk
    # at every sample event — a resolution added mid-run can fall outside the
    # dynamic-seq mark_dynamic range and would crash the run with a
    # ConstraintViolationError (#42). Skip it instead.
    seq_len = (width // 16) * (height // 16)
    seq_range = getattr(dit, "_dynamic_seq_range", None)
    if (
        getattr(dit, "_dynamic_seq", False)
        and seq_range is not None
        and not (seq_range[0] <= seq_len <= seq_range[1])
    ):
        logger.warning(
            f"Skipping sample prompt at {width}x{height} ({seq_len} tokens): outside "
            f"the compiled dynamic-seq token range {seq_range}. The compile budget "
            "covers the training buckets plus the sample prompts present at startup; "
            "to sample at this resolution, restart training with it in the prompt "
            "file, lower --w/--h, or disable torch_compile."
        )
        return

    logger.info(
        f"  prompt: {prompt}, size: {width}x{height}, steps: {sample_steps}, scale: {scale}, flow_shift: {flow_shift}, seed: {seed}, sampler: {sampler}"
    )

    def encode_prompt(prpt):
        if sample_prompts_te_outputs and prpt in sample_prompts_te_outputs:
            return sample_prompts_te_outputs[prpt]
        if text_encoder is not None:
            tokens = tokenize_strategy.tokenize(prpt)
            encoded = text_encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens
            )
            return encoded
        return None

    encoded = encode_prompt(prompt)
    if encoded is None:
        logger.warning("Cannot encode prompt, skipping sample")
        return

    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = encoded

    if isinstance(prompt_embeds, np.ndarray):
        prompt_embeds = torch.from_numpy(prompt_embeds).unsqueeze(0)
        attn_mask = torch.from_numpy(attn_mask).unsqueeze(0)
        t5_input_ids = torch.from_numpy(t5_input_ids).unsqueeze(0)
        t5_attn_mask = torch.from_numpy(t5_attn_mask).unsqueeze(0)

    prompt_embeds = prompt_embeds.to(accelerator.device, dtype=dit.dtype)
    attn_mask = attn_mask.to(accelerator.device)
    t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
    t5_attn_mask = t5_attn_mask.to(accelerator.device)

    if dit.use_llm_adapter:
        crossattn_emb = dit.llm_adapter(
            source_hidden_states=prompt_embeds,
            target_input_ids=t5_input_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=attn_mask,
        )
        crossattn_emb[~t5_attn_mask.bool()] = 0
        # Pad to 512 tokens (model expects fixed-length context)
        if crossattn_emb.shape[1] < 512:
            crossattn_emb = torch.nn.functional.pad(
                crossattn_emb, (0, 0, 0, 512 - crossattn_emb.shape[1])
            )
    else:
        crossattn_emb = prompt_embeds

    neg_crossattn_emb = None
    if scale > 1.0 and negative_prompt is not None:
        neg_encoded = encode_prompt(negative_prompt)
        if neg_encoded is not None:
            neg_pe, neg_am, neg_t5_ids, neg_t5_am = neg_encoded
            if isinstance(neg_pe, np.ndarray):
                neg_pe = torch.from_numpy(neg_pe).unsqueeze(0)
                neg_am = torch.from_numpy(neg_am).unsqueeze(0)
                neg_t5_ids = torch.from_numpy(neg_t5_ids).unsqueeze(0)
                neg_t5_am = torch.from_numpy(neg_t5_am).unsqueeze(0)

            neg_pe = neg_pe.to(accelerator.device, dtype=dit.dtype)
            neg_am = neg_am.to(accelerator.device)
            neg_t5_ids = neg_t5_ids.to(accelerator.device, dtype=torch.long)
            neg_t5_am = neg_t5_am.to(accelerator.device)

            if dit.use_llm_adapter:
                neg_crossattn_emb = dit.llm_adapter(
                    source_hidden_states=neg_pe,
                    target_input_ids=neg_t5_ids,
                    target_attention_mask=neg_t5_am,
                    source_attention_mask=neg_am,
                )
                neg_crossattn_emb[~neg_t5_am.bool()] = 0
                # Pad to 512 tokens (model expects fixed-length context)
                if neg_crossattn_emb.shape[1] < 512:
                    neg_crossattn_emb = torch.nn.functional.pad(
                        neg_crossattn_emb, (0, 0, 0, 512 - neg_crossattn_emb.shape[1])
                    )
            else:
                neg_crossattn_emb = neg_pe

    # Generate sample.
    # ControlNet-LLLite: if "controlnet_image" was specified in the prompt, load and encode
    # the conditioning image, then prime the network before do_sample().
    controlnet_net = (
        getattr(dit.blocks[0], "_controlnet_net", None)
        if hasattr(dit, "blocks")
        else None
    )
    controlnet_image_path = prompt_dict.get("controlnet_image")
    if controlnet_net is not None and controlnet_image_path:
        logger.info(f"  loading controlnet image: {controlnet_image_path}")
        try:
            from library.datasets.image_utils import load_image

            cn_img = load_image(controlnet_image_path)
            if cn_img is not None:
                cn_img = cn_img.convert("RGB")  # handle grayscale depth/edge maps
                cn_arr = np.array(cn_img).astype(np.float32) / 255.0
                cn_tensor = torch.from_numpy(cn_arr.transpose(2, 0, 1)).unsqueeze(0)
                cn_tensor = cn_tensor.to(accelerator.device, dtype=dit.dtype)
                # Resize to match the target latent dimensions
                cn_tensor = torch.nn.functional.interpolate(
                    cn_tensor,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                # VAE encode
                gc.collect()
                synchronize_device(accelerator.device)
                org_vae_device = vae.device
                vae.to(accelerator.device)
                try:
                    with torch.no_grad():
                        cn_latent = vae.encode_pixels_to_latents(cn_tensor)
                finally:
                    vae.to(org_vae_device)
                controlnet_net.set_cond_latent(cn_latent)
        except Exception as e:
            logger.warning(f"  failed to load controlnet image: {e}")
            controlnet_net.clear_cond_latent()

    # Deliberately no empty_cache here: freed tensors return to the caching
    # allocator and get reused on the next prompt / when training resumes, so
    # VRAM stays flat across the train<->sample transition instead of the
    # shrink-then-regrow churn that torch.cuda.empty_cache() causes.
    latents = do_sample(
        height,
        width,
        seed,
        dit,
        crossattn_emb,
        sample_steps,
        dit.dtype,
        accelerator.device,
        scale,
        flow_shift,
        neg_crossattn_emb,
        sampler=sampler,
    )

    # Clear ControlNet conditioning after generation
    if controlnet_net is not None:
        controlnet_net.clear_cond_latent()

    # Stash the latent instead of decoding now. Loading the VAE to GPU mid-run
    # (on top of the resident DiT + block-swap buffers + the heavy VAE decode
    # activations) is a real OOM risk on tight cards, so the actual decode is
    # deferred to the end of training — see decode_pending_samples(), called
    # from train.py once the training loop has torn down and freed the GPU.
    # (No empty_cache here either — keep VRAM flat; see do_sample call above.)

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i = prompt_dict.get("enum", 0)
    stem = (
        f"{'' if args.output_name is None else args.output_name + '_'}"
        f"{num_suffix}_{i:02d}_{ts_str}{seed_suffix}"
    )
    latents_dir = os.path.join(save_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)
    # ``epoch`` is None for step-triggered samples, int for epoch-triggered.
    # Persist with the latent so ``decode_pending_samples`` can re-emit a
    # ``sample`` progress event with the correct step/epoch metadata even
    # when the actual PNG decode is deferred to end-of-training.
    torch.save(
        {
            "latents": latents.detach().to("cpu"),
            "prompt": prompt,
            "enum": i,
            "global_step": steps,
            "epoch": epoch,
        },
        os.path.join(latents_dir, stem + ".pt"),
    )


def decode_pending_samples(
    accelerator: Accelerator,
    args,
    vae,
    *,
    progress_sink=None,
) -> None:
    """Decode the sample latents stashed during training into PNGs.

    Called once from train.py after the training loop tears down (optimizer /
    gradient / block-swap buffers freed → max GPU headroom). Loads the VAE to
    GPU a single time, decodes every ``output_dir/sample/latents/*.pt`` into a
    PNG in ``output_dir/sample/``, then parks the VAE back. Each latent file is
    removed after a successful decode; a failed one is left on disk so it can be
    recovered. No-op when sampling was disabled or the VAE isn't available.

    When ``progress_sink`` is provided, every successfully decoded PNG emits a
    ``sample`` event (with the originating ``global_step``/``epoch``/``prompt``
    metadata saved alongside the latent), so the GUI dashboard can show the
    preview as soon as it's on disk.
    """
    if vae is None:
        return
    save_dir = resolve_sample_dir(args)
    latents_dir = os.path.join(save_dir, "latents")
    if not os.path.isdir(latents_dir):
        return
    files = sorted(f for f in os.listdir(latents_dir) if f.endswith(".pt"))
    if not files:
        return

    logger.info(f"Decoding {len(files)} deferred sample image(s) to {save_dir}")
    wandb_tracker = None
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")
        import wandb

    org_vae_device = vae.device
    vae.to(accelerator.device)
    try:
        for fn in files:
            path = os.path.join(latents_dir, fn)
            try:
                rec = torch.load(path, map_location="cpu")
                latents = rec["latents"].to(accelerator.device)
                with torch.no_grad():
                    decoded = vae.decode_to_pixels(latents)
                image = decoded.float()
                image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
                if image.ndim == 4:  # drop temporal dim if present
                    image = image[:, 0, :, :]
                decoded_np = (255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).astype(
                    np.uint8
                )
                pil = Image.fromarray(decoded_np)
                stem = os.path.splitext(fn)[0]
                png_path = os.path.join(save_dir, stem + ".png")
                pil.save(png_path)
                if wandb_tracker is not None:
                    wandb_tracker.log(
                        {
                            f"sample_{rec.get('enum', 0)}": wandb.Image(
                                pil, caption=rec.get("prompt", "")
                            )
                        },
                        commit=False,
                    )
                if progress_sink is not None:
                    progress_sink.sample(
                        global_step=rec.get("global_step", 0) or 0,
                        epoch=rec.get("epoch"),
                        path=png_path,
                        prompt=rec.get("prompt"),
                    )
                os.remove(path)
            except Exception as exc:  # never let one bad latent abort the rest
                logger.error(f"Failed to decode sample latent {fn}: {exc}")
            clean_memory_on_device(accelerator.device)
    finally:
        vae.to(org_vae_device)
        clean_memory_on_device(accelerator.device)

    # Drop the staging dir if everything decoded cleanly.
    try:
        os.rmdir(latents_dir)
    except OSError:
        pass


def sample_image_to_tensor(
    *,
    accelerator: Accelerator,
    dit: anima_models.Anima,
    vae,
    height: int,
    width: int,
    crossattn_emb: torch.Tensor,
    neg_crossattn_emb: Optional[torch.Tensor] = None,
    sample_steps: int = 20,
    guidance_scale: float = 4.0,
    flow_shift: float = 3.0,
    seed: Optional[int] = None,
    show_progress: bool = True,
) -> torch.Tensor:
    """Sample one image and return the decoded pixel tensor in ``[-1, 1]``.

    Sibling of :func:`_sample_image_inference` that skips disk I/O and the
    PIL conversion. Returned tensor is ``[3, H, W]`` float (on
    ``accelerator.device``), suitable for direct PE-Core encoding.

    ``crossattn_emb`` is the prepared cross-attention embedding (already
    LLM-adapter'd and padded to the model's expected length); CMMD val
    builds it from cached TE outputs so we don't re-run the text encoder.
    """
    height = max(64, height - height % 16)
    width = max(64, width - width % 16)
    crossattn_emb = crossattn_emb.to(accelerator.device, dtype=dit.dtype)
    if neg_crossattn_emb is not None:
        neg_crossattn_emb = neg_crossattn_emb.to(accelerator.device, dtype=dit.dtype)

    clean_memory_on_device(accelerator.device)
    latents = do_sample(
        height,
        width,
        seed,
        dit,
        crossattn_emb,
        sample_steps,
        dit.dtype,
        accelerator.device,
        guidance_scale,
        flow_shift,
        neg_crossattn_emb,
        show_progress=show_progress,
    )

    gc.collect()
    synchronize_device(accelerator.device)
    clean_memory_on_device(accelerator.device)
    org_vae_device = vae.device
    vae.to(accelerator.device)
    decoded = vae.decode_to_pixels(latents)
    vae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)

    image = decoded.float()[0]
    if image.ndim == 4:
        # Drop temporal dim if the VAE returned [C, T, H, W].
        image = image[:, 0, :, :]
    return image.clamp(-1.0, 1.0)
