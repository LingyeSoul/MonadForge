"""Typed requests for the MOSS-Image jobs — caching, generation, LoRA training.

Torch-free on purpose (a GUI form can be built from these without importing
torch). Each field carries its CLI help in ``metadata``, so the sidecar CLIs
(``scripts/moss/``) read one definition:

    req = GenerateRequest(prompt="…", steps=30)
    argv = req.to_argv()                  # → daemon job argv
    req = GenerateRequest.from_argv(argv) # ← sidecar CLI

``to_argv`` writes only fields that differ from their defaults.

Model directory: ``--model_dir`` if given, else ``ANIMA_MOSS_MODEL_DIR`` (env or
``.env``), else ``models/moss_image`` under the repo home. It holds one diffusers
folder per variant, ``MOSS-Image-Base/`` and ``MOSS-Image-SFT/`` (``transformer/``,
``text_encoder/``, ``vae/``, ``tokenizer/``, ``scheduler/``); SFT differs from Base
in ``transformer/`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from library.downloads import MOSS_DIR as MODEL_DIR_NAME
from library.downloads import MOSS_VARIANT_DIRS as VARIANT_DIRS
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.lines.requests import Request, env_model_dir
from library.lines.requests import request_field as _f

MODEL_DIR_ENV = "ANIMA_MOSS_MODEL_DIR"
VARIANTS = tuple(VARIANT_DIRS)
# LoRA alpha when the request leaves it unset (lokr / loha fall back to rank).
LORA_ALPHA = 128.0


def default_model_dir() -> Path:
    """``ANIMA_MOSS_MODEL_DIR`` if set, else ``<repo home>/models/moss_image``."""
    return env_model_dir(MODEL_DIR_ENV, default_moss_dir())


def resolve_model_dir(model_dir: str | Path | None) -> Path:
    """An explicit ``model_dir`` wins; empty/None falls back to the default."""
    if model_dir:
        return resolve_under_home(model_dir)
    return default_model_dir()


def variant_dir(model_dir: str | Path | None, variant: str) -> Path:
    """The diffusers folder of one variant (``base`` / ``sft``)."""
    return resolve_model_dir(model_dir) / VARIANT_DIRS[variant]


_MODEL_DIR_HELP = f"folder holding MOSS-Image-Base/ and MOSS-Image-SFT/ (default: ${MODEL_DIR_ENV} or models/{MODEL_DIR_NAME})"


@dataclass
class CacheRequest(Request):
    """Precache Qwen3.5-9B text embeddings and FLUX.2 VAE latents for a training folder."""

    SCRIPT: ClassVar[str] = "scripts/moss/cache.py"

    src: str = _f(
        "post_image_dataset/resized",
        "folder of images + .txt captions, subfolders included (default: the "
        "Anima resized tree, whose captions are the revised ones; a caption there "
        "with no resized image uses its image_dataset/ original)",
    )
    out: str = _f("post_image_dataset/moss", "cache folder (two files per image, flat)")
    resolution: int = _f(
        1024,
        "pixel-area cap as a square edge — 1024 means at most ~1024^2 pixels at "
        "the image's own aspect, both sides multiples of 16; never upscaled",
    )
    caption_format: str = _f(
        "layout_json",
        "what the TE encodes: the Anima caption mapped onto MOSS's layout-JSON "
        "prompt (anime_tools.captions.layout_json; needs the danbooru tag CSV), or "
        "the .txt verbatim. Less leakage of the set's attributes onto other "
        "characters than plain tags (caststation A/B, 2026-10-17); "
        "generate with the same form",
        choices=("text", "layout_json"),
    )
    overwrite: bool = _f(False, "re-encode even when the cache is current")
    skip_text: bool = _f(False, "skip the text-encoder pass")
    skip_latents: bool = _f(False, "skip the VAE pass")
    save_crops: bool = _f(
        False, "also write the exact pixels each latent was encoded from", advanced=True
    )
    vae_dtype: str = _f(
        "fp32", "VAE encode dtype", choices=("fp32", "bf16"), advanced=True
    )
    max_text_tokens: int = _f(
        4096, "TE truncation length (the model's own limit)", advanced=True
    )
    model_dir: str = _f("", _MODEL_DIR_HELP, advanced=True)


@dataclass
class GenerateRequest(Request):
    """Render prompts with MOSS-Image (Base or SFT), optionally with a LoRA."""

    SCRIPT: ClassVar[str] = "scripts/moss/generate.py"

    prompt: str = _f(
        "An anime illustration of a girl with long silver hair and red eyes, "
        "wearing a black school uniform, sitting on a classroom windowsill at "
        "sunset, cherry blossom petals drifting outside the window, detailed cel "
        "shading, soft rim light.",
        "prompt (ignored when prompts_file or example is set)",
        multiline=True,
    )
    negative_prompt: str = _f("", "negative prompt (the pipeline default is empty)")
    prompt_format: str = _f(
        "layout_json",
        "layout_json = an Anima-format prompt mapped onto MOSS's layout JSON, as "
        "the cache's caption_format does; text = as typed, for a prompt already "
        "in JSON or a LoRA cached as text (not applied to example)",
        choices=("text", "layout_json"),
    )
    model: str = _f("base", "which DiT", choices=VARIANTS)
    lora: str = _f(
        "",
        "LoRA from scripts/moss/train.py (empty = none); a Base LoRA also loads on SFT",
    )
    lora_multiplier: float = _f(1.0, "LoRA strength; 0.0 is the base model exactly")
    width: int = _f(1024, "multiple of 16")
    height: int = _f(1024, "multiple of 16")
    steps: int = _f(50, "denoising steps (the snapped grid can return fewer)")
    cfg: float = _f(3.0, "CFG scale; 1.0 = no uncond forward")
    sampler: str = _f(
        "er_sde",
        "er_sde = ER-SDE-Solver-3 on the same σ grid, noise seeded by the prompt's "
        "seed (sharper line work, no extra forwards); euler = the pipeline's ODE",
        choices=("euler", "er_sde"),
    )
    uncond_skip: str = _f(
        "22",
        "comma-separated DiT blocks the uncond branch skips ('' = plain CFG); "
        "the pipeline default replaces the negative with a block-22-skipped "
        "prediction, still two forwards per step",
    )
    seed: int = _f(
        42, "seed of the first prompt (+1 per prompt); fp32 noise from a CPU generator"
    )
    out_dir: str = _f("output/tests", "images + manifest.json land here")
    prompts_file: str = _f("", "one prompt per line; overrides prompt", advanced=True)
    example: str = _f(
        "",
        "comma-separated Space gallery parameters.json files to replay; each sets "
        "its own prompt / negative / size / cfg / seed / uncond skip",
        advanced=True,
    )
    batched: bool = _f(
        False,
        "cond + uncond in one masked forward instead of two. Measured ~25%% "
        "slower at 1024² (the key mask takes SDPA off flash) and not bit-exact "
        "with the reference",
        advanced=True,
    )
    shift: float = _f(5.0, "scheduler shift (pipeline default)", advanced=True)
    decode_only: bool = _f(
        False,
        "decode every latent already in <out_dir>/latents/ to PNG and exit — no "
        "text encoder, no DiT",
        advanced=True,
    )
    compile: bool = _f(
        True,
        "torch.compile each block with dynamic shapes, after the LoRA is applied: "
        "1.37x at 1024² with or without Spectrum, a few seconds of compile once per "
        "job (2026-10-17). --no-compile for eager",
        advanced=True,
    )
    spectrum: bool = _f(
        False,
        "Spectrum: forecast the last block's image tokens on most steps instead of "
        "running the 42 blocks (library/moss/spectrum.py)",
    )
    spectrum_window_size: float = _f(
        2.0, "initial window N: a full forward every floor(N) steps", advanced=True
    )
    spectrum_flex_window: float = _f(
        0.5,
        "growth of N after each full forward: 0.25 / 0.5 / 1.0 = 22 / 19 / 16 full "
        "forwards of 50 steps (bench/moss_spectrum)",
        advanced=True,
    )
    spectrum_warmup: int = _f(6, "leading steps that always run full", advanced=True)
    spectrum_stop_caching_step: int = _f(
        -1, "full forwards from this step on (-1 = steps - 3)", advanced=True
    )
    spectrum_w: float = _f(
        0.3, "Chebyshev / Taylor blend (1.0 = pure Chebyshev)", advanced=True
    )
    spectrum_m: int = _f(3, "Chebyshev degree", advanced=True)
    spectrum_lam: float = _f(0.1, "ridge regularisation", advanced=True)
    dave: str = _f(
        "",
        "DAVE same-prompt diversity: attenuate the image tokens' per-channel mean on "
        "a block pool during the first steps. '' = off, 'auto' = the shipped MOSS "
        "mask, 'lo-hi' = blocks lo..hi, or a dave_alpha.npz path (library/moss/dave.py)",
    )
    dave_strength: float = _f(
        0.3, "fraction of the mean removed on a pooled block (0 = off, 1 = all)"
    )
    dave_tau: float = _f(
        0.10, "attenuate the first tau of the steps (at least one)", advanced=True
    )
    model_dir: str = _f("", _MODEL_DIR_HELP, advanced=True)


@dataclass
class TrainRequest(Request):
    """Train a LoRA on a precached folder — flow matching."""

    SCRIPT: ClassVar[str] = "scripts/moss/train.py"
    # The block's six bias-free linears, as a fullmatch over qualified names.
    TARGETS: ClassVar[str] = r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)"

    cache: str = _f(
        "post_image_dataset/moss", "cache folder written by scripts/moss/cache.py"
    )
    output: str = _f(
        "output/ckpt_moss/moss_lora.safetensors",
        "LoRA output path (a .json run report lands beside it)",
    )
    model: str = _f("base", "which DiT to train on", choices=VARIANTS)
    epochs: int = _f(8, "passes over the cache")
    max_steps: int = _f(0, "stop after this many steps (0 = epochs decide)")
    algo: str = _f(
        "lora",
        "adapter: lora, or LyCORIS lokr (Kronecker product) / loha (Hadamard "
        "product of two rank-r pairs)",
        choices=("lora", "lokr", "loha"),
    )
    rank: int = _f(32, "LoRA rank (lokr: rank of the larger Kronecker factor)")
    alpha: float | None = _f(
        None,
        f"LoRA alpha; scale = alpha / rank. Default {LORA_ALPHA:g} for lora (scale 4 "
        "at rank 32), = rank for lokr / loha. lokr ignores it "
        "when the rank reaches half the larger factor's side and both factors are full",
    )
    factor: int = _f(
        -1,
        "lokr only: cap on the small Kronecker factor's side (-1 = near-square "
        "split; LyCORIS's usual 4 / 8 / 16 give a larger full-rank w2)",
    )
    down_init: str = _f(
        "weight_svd",
        "lora only, down-projection init: weight_svd (the base weight's top-r right "
        "singular vectors / sqrt 3, Anima's SVD-Down; kept the caststation style "
        "better by eye, 2026-10-17) or kaiming",
        choices=("kaiming", "weight_svd"),
    )
    svd_slice: int = _f(
        0,
        "lora + weight_svd only: seed down from the base weight's right singular "
        "vectors [k·r, (k+1)·r) instead of the top-r. Slices of one basis are "
        "mutually orthogonal, so LoRAs trained on different slices own disjoint "
        "input subspaces (Anima's svd_slice, docs/methods/svd-down-lora.md); under "
        "dc_isolate every slice shares row 0, the DC direction. Stamped as "
        "ss_svd_slice. 0 = the top-r",
        advanced=True,
    )
    svd_slice_count: int = _f(
        0,
        "> 0 interleaves the slices: slice k takes singular vectors {k, k+N, "
        "k+2N, …}, so every slice gets one of the top-N directions instead of "
        "slice 0 holding the whole top of the spectrum. Needs svd_slice < N. "
        "0 = contiguous",
        advanced=True,
    )
    channel_scaling_alpha: float = _f(
        0.5,
        "lora only (lokr / loha ignore it): Anima's channel scaling — rescale each "
        "target's input channel c by mean|x[c]|^alpha inside the down projection, so "
        "Adam's step no longer favours the few high-magnitude channels (on MOSS: ch 42 "
        "into ffn_up_gate). Same output, plain LoRA on disk. At lr 1e-4 it kept prompt "
        "details plain LoRA lost (2026-10-18). 0 = off",
    )
    channel_stats: str = _f(
        "library/moss/calibration/channel_stats.safetensors",
        "per-target mean|x| calibration for channel_scaling_alpha (from "
        "bench/moss_channel_stats/probe_lora_inputs.py on Base)",
        advanced=True,
    )
    dc_isolate: bool = _f(
        True,
        "lora only: put the calibrated input DC direction (token-mean x̄ per target) "
        "into one rank row of down and keep the other rows orthogonal to it, so only "
        "one up column learns the token-independent bias and the rest of the up "
        "gradient is centered (the GraLoRA up-side entanglement, localised). Same "
        "output at init, plain LoRA on disk, composes with channel scaling. On by "
        "default with lr 5e-5 (2026-10-18): clean and on-prompt at lr 1e-4-like style "
        "strength; at lr 1e-4 it over-fits (text artifacts)",
    )
    channel_mean: str = _f(
        "library/moss/calibration/channel_mean.safetensors",
        "per-target signed token-mean x̄ for dc_isolate (from "
        "bench/moss_channel_stats/probe_lora_inputs.py on Base)",
        advanced=True,
    )
    lr: float = _f(
        5e-5,
        "AdamW learning rate (rank 32 / alpha 128 with channel scaling + dc_isolate: "
        "5e-5 is clean and on-prompt at lr 1e-4-like style strength; 1e-4 over-fits "
        "with dc_isolate; without dc_isolate 1e-4 was the pick, 2026-10-18)",
    )
    warmup_ratio: float = _f(0.05, "linear warmup, as a fraction of all steps")
    save_every_epochs: int = _f(0, "also save every N epochs (0 = final only)")
    caption_dropout: float = _f(
        0.0,
        "probability of training a step on the empty-prompt embedding (CFG's "
        "uncond); > 0 encodes it once into <cache>/_empty.moss_te.safetensors",
    )
    plain_caption_rate: float = _f(
        0.2,
        "probability of training a step on the plain Anima caption instead of its "
        "layout JSON; only a caption_format=layout_json cache carries the plain one",
    )
    lora_dtype: str = _f(
        "bf16",
        "dtype the LoRA is saved in; training keeps fp32 masters and runs the "
        "rank GEMMs in bf16",
        choices=("bf16", "fp16", "fp32"),
    )
    targets: str = _f(TARGETS, "fullmatch regex over linear names", advanced=True)
    sigma_mu: float = _f(
        -1.09,
        "noise sampling: Z ~ N(mu, s^2), sigma = 1 - sigmoid(Z) (report App. A.2, "
        "final pretraining interval)",
        advanced=True,
    )
    sigma_s: float = _f(0.95, "noise sampling std of Z", advanced=True)
    sigma_cutoff: float = _f(
        0.95,
        "truncate sigma <= cutoff, then rescale onto the 1/1000 grid (1.0 = off)",
        advanced=True,
    )
    sigma_uniform: float = _f(
        0.01, "mixture weight of discrete-uniform sigma", advanced=True
    )
    max_grad_norm: float = _f(1.0, "gradient clip", advanced=True)
    grad_checkpointing: bool = _f(
        False,
        "per-block activation checkpointing. Off by default: the activation "
        "budget plus block swap below run 1.78 vs 2.09 s/step at ~4.4k joint "
        "tokens on 16 GB (2026-10-17). Turn it on with --no-compile, or with "
        "--blocks_to_swap 0 to keep the whole DiT on the card",
    )
    activation_memory_budget: float = _f(
        0.1,
        "cap on the compiled blocks' saved-for-backward set (Inductor's min-cut "
        "partitioner recomputes the rest); applies with --compile and "
        "--no-grad_checkpointing, ignored otherwise. 0.1 at rank 32: ~1.7 GB free "
        "and 1.85 s/step on 16 GB at ~4.5k joint tokens (0.15 left ~0.7 GB at rank 16)",
        advanced=True,
    )
    blocks_to_swap: int = _f(
        12,
        "DiT blocks to swap to CPU (-1 = size from free VRAM). 12 fits budget "
        "0.1 at rank 32 and ~4.5k joint tokens on 16 GB with ~1.7 GB to spare (24 "
        "for more room); more swap costs PCIe time the step cannot hide. scripts/moss/tune_swap.py measures "
        "this machine's pair",
        advanced=True,
    )
    compile: bool = _f(
        True,
        "torch.compile each block with dynamic shapes (the joint image+text "
        "length changes with every caption): 2.0 vs 3.1 s/step at ~4.5k joint "
        "tokens and 0.7 GB lower peak, for ~30 s of compile and one recompile "
        "(2026-10-16). --no-compile for a quick smoke",
        advanced=True,
    )
    log_every: int = _f(10, "full log line every N steps", advanced=True)
    seed: int = _f(0, "RNG seed", advanced=True)
    model_dir: str = _f("", _MODEL_DIR_HELP, advanced=True)


@dataclass
class TuneSwapRequest(Request):
    """Find the --blocks_to_swap / --activation_memory_budget pair to train with on
    this machine: one arm per swap count at the largest budget that fits, timed
    on the largest sample."""

    SCRIPT: ClassVar[str] = "scripts/moss/tune_swap.py"

    cache: str = _f(
        "post_image_dataset/moss",
        "cache folder; its largest sample (image + text tokens) is the one timed",
    )
    sizes: str = _f(
        "",
        "synthetic WxH:text_tokens instead of the cache (e.g. 1024x1024:300)",
    )
    model: str = _f("base", "which DiT to time", choices=VARIANTS)
    rank: int = _f(32, "LoRA rank (the adapters and AdamW state take VRAM too)")
    swap_grid: str = _f(
        "8,12,16,20,24,28,32,36,40",
        "swap counts to try, ascending (at most num_layers - 2)",
    )
    patience: int = _f(
        2, "stop after this many arms in a row slower than the best so far"
    )
    margin_gb: float = _f(
        1.0,
        "free VRAM an arm must leave at its peak to be recommended (real runs "
        "vary in length and grow the allocator)",
    )
    tolerance: float = _f(
        0.02,
        "arms this close to the fastest count as tied; the least swap among "
        "them is recommended (the curve is flat below the PCIe knee)",
    )
    reference: bool = _f(
        True, "also time the checkpointed plan (--grad_checkpointing, swap 0)"
    )
    budget_step: float = _f(0.05, "budget grid", advanced=True)
    warmup: int = _f(2, "untimed steps per arm (compile lands here)", advanced=True)
    steps: int = _f(5, "timed steps per arm", advanced=True)
    output: str = _f(
        "output/ckpt_moss/tune/tune_swap.json", "JSON report path", advanced=True
    )
    seed: int = _f(0, "RNG seed", advanced=True)
    model_dir: str = _f("", _MODEL_DIR_HELP, advanced=True)
