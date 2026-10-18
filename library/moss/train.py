"""Train a LoRA on a precached folder — flow matching, batch 1.

Reads only what ``cache.py`` wrote (latents + text embeddings), so neither the
Qwen3.5-9B TE nor the VAE is loaded; the one exception is ``--caption_dropout``,
which encodes the empty prompt once into ``<cache>/_empty.moss_te.safetensors``
(key ``embeds``) if it is not there yet.

Objective (report App. A.2): ``z_σ = (1−σ)·z0 + σ·ε``, the DiT predicts
``ε − z0`` at timestep ``1000·σ``, MSE averaged over image tokens and channels —
text tokens never reach the head, and there is no σ/SNR weighting.

σ follows the final Stage-3 interval exactly: ``Z ~ N(μ, s²)``, ``S = 1 − sigmoid(Z)``,
truncated to ``S ≤ cutoff`` and rescaled onto the grid ``{1/1000 … 1}``
(``σ = ⌈1000·S/cutoff⌉ / 1000``), mixed with 1 % discrete uniform. Shift 1: that
interval trained at area ≤ 1024², which is the size the cache targets, so no
resolution shift is added. Median σ ≈ 0.75 — noise-heavy, matching an inference
grid that spends most of its steps at high σ under shift 5.

Default memory plan: no checkpointing, 12 blocks swapped, and the compiled
blocks' saved set capped by ``activation_memory_budget`` (0.15), which is faster
than checkpointing on 16 GB (``project/moss_lora/lora.md`` § Swap + budget).
``--grad_checkpointing`` wraps each block's ``forward`` (not ``__call__``), so
the block swapper's forward hooks fire once per step and never again during
the recompute; ``MOSSImageTransformer.gradient_checkpointing`` is left off.

    make daemon-run ARGS="scripts/moss/train.py --cache output/moss/cache/<set> --epochs 8"
"""

from __future__ import annotations

import functools
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from library.env import resolve_under_home
from library.io.output_layout import atomic_write_json
from library.lines.progress import TrainingProgress, training_progress
from library.moss.loader import (
    compile_blocks,
    empty_cache,
    free_vram_gb,
    load_transformer,
)
from library.moss.lora import LoRANetwork
from library.moss.requests import LORA_ALPHA, TrainRequest
from library.moss.validation import training_output_paths, validate_request

EMPTY_TE_NAME = "_empty.moss_te.safetensors"
NUM_TRAIN_TIMESTEPS = 1000
PROGRESS_EVERY = 2  # steps per "  progress i/n" line (the GUI bar reads them)


@dataclass
class Sample:
    """One training pair, lazily loadable. ``n_img`` / ``n_text`` order the run."""

    stem: str
    n_img: int
    n_text: int
    load: Callable[
        [], tuple[torch.Tensor, torch.Tensor]
    ]  # latent (128,h,w), embeds (S,4096)


def samples_from_cache(cache_dir: Path) -> list[Sample]:
    from library.moss import cache as moss_cache

    out = []
    for item in moss_cache.list_items(cache_dir):

        def load(item=item):
            # Embeds are stored at their true length (never padded).
            latent, embeds = moss_cache.load_item(item)
            return latent, embeds

        out.append(
            Sample(
                item.stem, (item.width // 16) * (item.height // 16), item.n_text, load
            )
        )
    return out


def sample_sigma(
    gen: torch.Generator,
    mu: float,
    s: float,
    cutoff: float = 0.95,
    uniform: float = 0.01,
    total: int = NUM_TRAIN_TIMESTEPS,
) -> float:
    """One σ on the ``{1/T … 1}`` grid, from report eq. (4)."""
    if torch.rand((), generator=gen).item() < uniform:
        j = int(torch.randint(1, total + 1, (), generator=gen))
        return j / total
    while True:
        z = torch.randn((), generator=gen).item() * s + mu
        noise = 1.0 - 1.0 / (1.0 + math.exp(-z))
        if noise <= cutoff:
            break
    j = min(total, max(1, math.ceil(noise * total / cutoff)))
    return j / total


def training_loss(transformer, latent, embeds, sigma: float, gen=None):
    """``latent (1,128,h,w)`` fp32, ``embeds (1,S,4096)`` at true length."""
    noise = torch.randn(
        latent.shape, generator=gen, device=latent.device, dtype=latent.dtype
    )
    noisy = (1.0 - sigma) * latent + sigma * noise
    target = noise - latent
    timestep = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=latent.device)
    pred = transformer(noisy, timestep, embeds, return_dict=False)[0]
    return F.mse_loss(pred.float(), target.float())


def checkpoint_blocks(blocks) -> None:
    """Per-block activation checkpointing around ``block.forward``."""
    for block in blocks:
        inner = block.forward

        @functools.wraps(inner)
        def forward(*args, _inner=inner, **kwargs):
            if not torch.is_grad_enabled():
                return _inner(*args, **kwargs)
            return checkpoint(_inner, *args, use_reentrant=False, **kwargs)

        block.forward = forward


# Uncheckpointed, the compiled saved set is ~5.25 MB per joint token at budget
# 1.0 (128 KiB per block x 42); the constant covers AdamW state and allocator
# growth (fit to budgets 0.15-0.25 at 4.4k tokens, ~1 GiB above measured).
_BUDGET_BASE_GB = 3.5
_BUDGET_MB_PER_TOKEN = 6.0


def step_reserve_gb(
    joint: int, *, grad_checkpointing: bool, budget: float = 1.0
) -> float:
    """VRAM one training step needs beyond the placed weights, at ``joint``
    (image + text) tokens."""
    if grad_checkpointing:
        # ~0.6 MB per joint token (qwen21's measured slope, same hidden width
        # class) plus one recomputed block.
        return 1.5 + 0.6 * joint / 1024
    return _BUDGET_BASE_GB + budget * _BUDGET_MB_PER_TOKEN * joint / 1024


def budget_for_free(free_gb: float, joint: int, step: float = 0.05) -> float:
    """The largest budget on a ``step`` grid whose :func:`step_reserve_gb` fits
    ``free_gb``; 0.0 when not even one ``step`` fits."""
    room = (free_gb - _BUDGET_BASE_GB) / (_BUDGET_MB_PER_TOKEN * joint / 1024)
    return min(1.0, math.floor(max(0.0, room) / step + 1e-9) * step)


def set_activation_memory_budget(budget: float) -> None:
    """Inductor's partitioner reads this at compile time: set it before
    ``compile_blocks`` (and reset ``torch._dynamo`` to change it later)."""
    import torch._functorch.config as functorch_config

    functorch_config.activation_memory_budget = budget


def place(transformer, device, blocks_to_swap: int, reserve_gb: float):
    """Everything on ``device``, or the block swapper for ``blocks_to_swap`` > 0
    (``-1`` sizes it from ``mem_get_info`` free). Returns the attachment or None."""
    from library.runtime.blockswap import place as place_blocks

    return place_blocks(
        transformer,
        "layers",
        device,
        blocks_to_swap=None if blocks_to_swap < 0 else blocks_to_swap,
        supports_backward=True,
        activation_reserve_gb=reserve_gb,
        label="transformer",
    )


def report_fit(transformer, attached, n_img: int, n_text: int) -> dict:
    """After step 1 (the largest sample): peak against what is actually free."""
    import library.runtime.blockswap as blockswap

    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    free = free_vram_gb()
    per = blockswap.block_size_gb(transformer.layers)
    swapped = attached.offloader.blocks_to_swap if attached else 0
    fit = {
        "image_tokens": n_img,
        "text_tokens": n_text,
        "peak_allocated_gb": round(allocated, 3),
        "peak_reserved_gb": round(reserved, 3),
        "free_gb": round(free, 3),
        "blocks_to_swap": swapped,
    }
    print(
        f"fit @ step 1 ({n_img} image + {n_text} text tokens, the largest sample): "
        f"peak {allocated:.2f} GB allocated / {reserved:.2f} GB reserved, "
        f"{free:.2f} GB actually free, swapping {swapped} at {per:.3f} GB each",
        flush=True,
    )
    if swapped:
        spare = max(0, int((free - per) // per))
        if spare:
            print(
                f"  -> rerun with --blocks_to_swap {max(0, swapped - spare)}",
                flush=True,
            )
    return fit


def empty_prompt_embeds(req: TrainRequest, cache_dir: Path) -> torch.Tensor:
    """``(S, 4096)`` bf16 embedding of ``""``, encoded once and kept in the cache."""
    from safetensors.torch import load_file, save_file

    path = cache_dir / EMPTY_TE_NAME
    if not path.exists():
        from library.moss.loader import encode_prompts

        print(f"encoding the empty prompt -> {path}", flush=True)
        embeds = encode_prompts(req.model_dir, [""])[""]
        save_file({"embeds": embeds.contiguous()}, str(path))
    return load_file(str(path))["embeds"]


def run_train(req: TrainRequest) -> Path:
    with training_progress("moss", req.output) as progress:
        validate_request(req)
        cache_dir = resolve_under_home(req.cache)
        samples = samples_from_cache(cache_dir)
        if not samples:
            raise SystemExit(f"no cached pairs under {cache_dir}")
        empty = empty_prompt_embeds(req, cache_dir) if req.caption_dropout > 0 else None
        return train(req, samples, empty, progress=progress)


def train(
    req: TrainRequest,
    samples: list[Sample],
    empty: torch.Tensor | None = None,
    *,
    progress: TrainingProgress | None = None,
) -> Path:
    """The loop, over already-built ``samples`` (``run_train`` reads the cache;
    a fit smoke passes synthetic ones)."""
    validate_request(req)
    out_path, report = training_output_paths(req.output)
    progress = progress if progress is not None else TrainingProgress()
    total = req.epochs * len(samples)
    if req.max_steps:
        total = min(total, req.max_steps) if req.epochs else req.max_steps
    epochs = math.ceil(total / len(samples))
    progress.start(total_steps=total, total_epochs=epochs)
    torch.manual_seed(req.seed)
    random.seed(req.seed)
    gen = torch.Generator().manual_seed(req.seed)
    device = torch.device("cuda")

    img = sorted(s.n_img for s in samples)
    txt = sorted(s.n_text for s in samples)
    print(
        f"{len(samples)} samples, image {img[0]}-{img[-1]} tokens, text {txt[0]}-{txt[-1]} tokens",
        flush=True,
    )
    data = {s.stem: s.load() for s in samples}

    transformer = load_transformer(req.model_dir, req.model)
    transformer.requires_grad_(False)
    network = LoRANetwork(
        transformer,
        rank=req.rank,
        alpha=req.alpha if req.alpha is not None or req.algo != "lora" else LORA_ALPHA,
        targets=req.targets,
        dtype=torch.float32,
        down_init=req.down_init,
        algo=req.algo,
        factor=req.factor,
    )
    patched = network.apply_to()
    network.to(device)
    detail = {
        "lora": f"{req.down_init} down",
        "lokr": f"factor {req.factor}",
        "loha": "hadamard",
    }[req.algo]
    print(
        f"{req.algo}: rank {req.rank} alpha {network.alpha} ({detail}) on {patched} linears, "
        f"{network.num_parameters / 1e6:.1f}M params (fp32 masters, saved {req.lora_dtype})",
        flush=True,
    )

    budgeted = req.compile and not req.grad_checkpointing
    budget = req.activation_memory_budget if budgeted else 1.0
    reserve_gb = step_reserve_gb(
        img[-1] + txt[-1], grad_checkpointing=req.grad_checkpointing, budget=budget
    )
    attached = place(transformer, device, req.blocks_to_swap, reserve_gb)
    if budgeted and budget < 1.0:
        set_activation_memory_budget(budget)
        print(f"activation memory budget {budget}", flush=True)
    elif not req.compile and not req.grad_checkpointing:
        print(
            "warning: eager without checkpointing saves ~10 MB per joint token "
            "per step; add --grad_checkpointing unless the card has room",
            flush=True,
        )
    if req.compile:
        compile_blocks(transformer.layers)
    if req.grad_checkpointing:
        checkpoint_blocks(transformer.layers)
    empty_cache()
    print(f"free VRAM before training: {free_vram_gb():.2f} GB", flush=True)

    params = list(network.parameters())
    optimizer = torch.optim.AdamW(params, lr=req.lr)
    warmup = max(1, int(total * req.warmup_ratio))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )
    print(
        f"{total} steps ({epochs} epochs x {len(samples)}), warmup {warmup}", flush=True
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_variant": req.model,
        "lr": req.lr,
        "samples": len(samples),
        "steps": total,
        "image_tokens": f"{img[0]}-{img[-1]}",
        "sigma": f"mu={req.sigma_mu} s={req.sigma_s} cutoff={req.sigma_cutoff} "
        f"uniform={req.sigma_uniform}",
        "caption_dropout": req.caption_dropout,
    }

    def save(tag: str | None = None) -> Path:
        path = (
            out_path
            if tag is None
            else out_path.with_name(f"{out_path.stem}_{tag}{out_path.suffix}")
        )
        network.save(path, req.lora_dtype, **meta)
        progress.checkpoint(path)
        return path

    by_size = sorted(samples, key=lambda s: s.n_img + s.n_text, reverse=True)
    history, losses_all, fit = [], [], None
    step = 0
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for epoch in range(epochs):
        # The largest sample first in epoch 1, so step 1 is the worst case the
        # fit report bounds; shuffled after that.
        order = by_size if epoch == 0 else random.sample(samples, len(samples))
        losses, t_epoch = [], time.time()
        for s in order:
            if step >= total:
                break
            latent, embeds = data[s.stem]
            if empty is not None and random.random() < req.caption_dropout:
                embeds = empty
            latent = latent.unsqueeze(0).to(device, torch.float32)
            embeds = embeds.unsqueeze(0).to(device)
            sigma = sample_sigma(
                gen, req.sigma_mu, req.sigma_s, req.sigma_cutoff, req.sigma_uniform
            )
            loss = training_loss(transformer, latent, embeds, sigma)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, req.max_grad_norm)
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            value = loss.item()
            losses.append(value)
            progress.step(
                global_step=step,
                epoch=epoch + 1,
                loss=value,
                average_loss=sum(losses) / len(losses),
                lr=lr_sched.get_last_lr()[0],
            )
            losses_all.append({"step": step, "loss": value, "sigma": sigma})
            if step == 1:
                fit = report_fit(transformer, attached, s.n_img, embeds.shape[1])
                t_start = time.time()  # s/step excludes step 1's allocator warm-up
            per_step = (time.time() - t_start) / max(1, step - 1)
            eta = int(per_step * (total - step))
            if step % PROGRESS_EVERY == 0 or step == total:
                print(
                    f"  progress {step}/{total} epoch {epoch + 1}/{epochs} "
                    f"loss {sum(losses) / len(losses):.4f} "
                    f"eta {eta // 3600}:{eta % 3600 // 60:02d}:{eta % 60:02d}",
                    flush=True,
                )
            if step % req.log_every == 0:
                print(
                    f"  step {step}/{total} loss {value:.4f} |g| {grad_norm.item():.3f} "
                    f"sigma {sigma:.3f} lr {lr_sched.get_last_lr()[0]:.2e} "
                    f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB "
                    f"{per_step:.2f}s/step",
                    flush=True,
                )
        if not losses:
            break
        mean = sum(losses) / len(losses)
        history.append(
            {"epoch": epoch + 1, "loss": mean, "seconds": time.time() - t_epoch}
        )
        print(
            f"epoch {epoch + 1}/{epochs}: loss {mean:.4f} {time.time() - t_epoch:.0f}s "
            f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
            flush=True,
        )
        if req.save_every_epochs and (epoch + 1) % req.save_every_epochs == 0:
            print(f"  saved {save(f'e{epoch + 1}')}", flush=True)

    final = save()
    seconds = time.time() - t_start
    summary = {
        "output": str(final),
        "request": {k: getattr(req, k) for k in req.__dataclass_fields__},
        "fit": fit,
        "steps": step,
        "minutes": seconds / 60,
        "s_per_step": seconds / max(1, step - 1),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "history": history,
        "losses": losses_all,
    }
    # Replace the report entry instead of following an existing symlink or
    # hard link and overwriting the file it references.
    atomic_write_json(report, summary)
    print(f"wrote {final} and {report} — {seconds / 60:.1f} min", flush=True)
    if attached is not None:
        attached.detach()
    return final
