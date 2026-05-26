"""Cache text-encoder (Qwen3) outputs and the pooled-text sidecar.

Orchestration extracted from ``preprocess/cache_text_embeddings.py`` and
``preprocess/cache_pooled_text.py`` (see
``docs/proposal/tooling_architecture.md`` §A). The scripts keep argparse + model
load + uncond staging; the caption-variant generation, the batched
tokenize→encode→(LLM-adapter)→save loop, and the pooled reduction live here.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from PIL import Image

from library.io.cache import POOLED_CACHE_SUFFIX, TE_CACHE_SUFFIX, resolve_cache_path
from library.preprocess._dataset import PreprocessStats, walk_images
from library.preprocess._progress import ProgressFn

logger = logging.getLogger(__name__)


def generate_caption_variants(
    caption: str, num_variants: int, tag_dropout_rate: float
) -> list[str]:
    """Generate ``num_variants`` caption variants for stochastic train-time sampling.

    v0 = pristine original caption. v1..v{N-1} are smart-shuffled (preserving
    the @artist prefix and section anchors), then every tag *after* the prefix
    is independently dropped with probability ``tag_dropout_rate``. The
    ``@no-artist`` sentinel participates in the boundary but is stripped from
    every variant (including v0) before it is written.
    """
    from library.anima import training as anima_train_utils

    sentinel = anima_train_utils.NO_ARTIST_SENTINEL

    tags = [t.strip() for t in caption.split(",")]
    split_idx = anima_train_utils.find_anima_prefix_end(tags)

    # v0 stays byte-identical to the source caption unless the sentinel is
    # actually present — re-joining would normalize whitespace around commas
    # for every existing dataset otherwise.
    if sentinel in tags:
        variants = [", ".join(anima_train_utils.strip_no_artist_sentinel(tags))]
    else:
        variants = [caption]

    for _ in range(max(0, num_variants - 1)):
        shuffled = anima_train_utils.anima_smart_shuffle_caption(tags.copy())
        if tag_dropout_rate > 0.0 and len(shuffled) > split_idx:
            kept = list(shuffled[:split_idx])
            for tag in shuffled[split_idx:]:
                if random.random() >= tag_dropout_rate:
                    kept.append(tag)
            if not kept:
                kept = shuffled[:1]
            shuffled = kept
        shuffled = anima_train_utils.strip_no_artist_sentinel(shuffled)
        variants.append(", ".join(shuffled))
    return variants


def _encode_batch(
    captions: list[str],
    tokenize_strategy,
    encoding_strategy,
    text_encoder,
    llm_adapter,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Tokenize, encode through Qwen3, optionally run the LLM adapter. CPU tensors out."""
    tokens_and_masks = tokenize_strategy.tokenize(captions)
    with torch.no_grad():
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = (
            encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens_and_masks
            )
        )

        crossattn_emb = None
        if llm_adapter is not None:
            crossattn_emb = llm_adapter(
                source_hidden_states=prompt_embeds,
                target_input_ids=t5_input_ids.to(device, dtype=torch.long),
                target_attention_mask=t5_attn_mask.to(device),
                source_attention_mask=attn_mask,
            )
            crossattn_emb[~t5_attn_mask.to(device).bool()] = 0
            crossattn_emb = crossattn_emb.to(dtype=torch.bfloat16).cpu()

    return (
        prompt_embeds.to(dtype=torch.bfloat16).cpu(),
        attn_mask.to(dtype=torch.int32).cpu(),
        t5_input_ids.to(dtype=torch.long).cpu(),
        t5_attn_mask.to(dtype=torch.int32).cpu(),
        crossattn_emb,
    )


def _te_cache_path(image_path: Path, cache_dir: Path | None, image_dir: Path) -> Path:
    if cache_dir is None:
        return image_path.with_name(image_path.stem + TE_CACHE_SUFFIX)
    return Path(
        resolve_cache_path(
            str(image_path),
            TE_CACHE_SUFFIX,
            cache_dir=str(cache_dir),
            image_dir=str(image_dir),
        )
    )


def cache_text_embeddings(
    data_dir: Path,
    tokenize_strategy,
    encoding_strategy,
    text_encoder,
    *,
    llm_adapter=None,
    device: torch.device,
    cache_dir: Path | None = None,
    recursive: bool = False,
    batch_size: int = 16,
    caption_shuffle_variants: int = 0,
    caption_tag_dropout_rate: float = 0.0,
    min_pixels: int = 500_000,
    verbose: bool = True,
    progress: ProgressFn | None = None,
) -> PreprocessStats:
    """Encode ``.txt`` captions for every captioned image under ``data_dir``.

    Strategies + encoder + (optional) ``llm_adapter`` are supplied loaded + on
    ``device``. Images below ``min_pixels`` are skipped (mirrors the resize
    filter). With ``caption_shuffle_variants > 0`` each cache holds N variants
    (v0 pristine, v1..v{N-1} shuffled + optionally tag-dropped). Returns counts;
    pass ``progress`` for a per-image bar.
    """
    candidates = walk_images(data_dir, recursive=recursive)

    entries: list[tuple[Path, str]] = []
    skipped_small = 0
    for p in candidates:
        caption_path = p.with_suffix(".txt")
        if not caption_path.exists():
            continue
        if min_pixels > 0:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception as e:
                logger.warning("could not read %s: %s", p.name, e)
                continue
            if w * h < min_pixels:
                skipped_small += 1
                continue
        # An empty caption file is a valid explicit empty caption
        # (unconditional / style-LoRA training) — encode "" rather than
        # dropping the image, so the cached set matches the training dataset.
        caption = caption_path.read_text(encoding="utf-8").strip().split("\n")[0]
        entries.append((p, caption))

    if skipped_small and verbose:
        print(
            f"Skipping {skipped_small} images below {min_pixels:,} pixels "
            f"({min_pixels / 1e6:.2f}MP) -- same filter as resize_images.py."
        )

    stats = PreprocessStats(seen=len(entries))
    caption_dropout_rate = torch.tensor(0.0, dtype=torch.float32)
    n_variants = caption_shuffle_variants
    tag_dropout_rate = float(caption_tag_dropout_rate)

    from safetensors.torch import save_file

    if progress is not None:
        progress(0, total=len(entries))

    for batch_start in range(0, len(entries), batch_size):
        batch = entries[batch_start : batch_start + batch_size]

        # Skip already-cached entries.
        to_encode: list[tuple[Path, str, Path]] = []
        for img_path, caption in batch:
            cache_path = _te_cache_path(img_path, cache_dir, data_dir)
            if cache_path.exists():
                stats.skipped += 1
                if progress is not None:
                    progress(1, detail=f"skip {img_path.name}")
            else:
                to_encode.append((img_path, caption, cache_path))

        if not to_encode:
            continue

        if n_variants <= 0:
            captions = [c for _, c, _ in to_encode]
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb = (
                _encode_batch(
                    captions,
                    tokenize_strategy,
                    encoding_strategy,
                    text_encoder,
                    llm_adapter,
                    device,
                )
            )

            for i, (img_path, _, cache_path) in enumerate(to_encode):
                save_dict = {
                    "prompt_embeds": prompt_embeds[i],
                    "attn_mask": attn_mask[i],
                    "t5_input_ids": t5_input_ids[i],
                    "t5_attn_mask": t5_attn_mask[i],
                    "caption_dropout_rate": caption_dropout_rate,
                }
                if crossattn_emb is not None:
                    save_dict["crossattn_emb"] = crossattn_emb[i]
                save_file(save_dict, str(cache_path))
                stats.written += 1
                if progress is not None:
                    progress(1, detail=f"{img_path.name}")
        else:
            all_captions: list[str] = []
            for _, caption, _ in to_encode:
                all_captions.extend(
                    generate_caption_variants(caption, n_variants, tag_dropout_rate)
                )

            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb = (
                _encode_batch(
                    all_captions,
                    tokenize_strategy,
                    encoding_strategy,
                    text_encoder,
                    llm_adapter,
                    device,
                )
            )

            for i, (img_path, _, cache_path) in enumerate(to_encode):
                save_dict = {
                    "num_variants": torch.tensor(n_variants, dtype=torch.int64),
                    # Marker: v0 is the pristine original caption (no shuffle,
                    # no tag dropout). Loaders use this to switch on weighted
                    # 20%/80% sampling between v0 and v1..v{N-1}.
                    "v0_intact": torch.tensor(1, dtype=torch.int8),
                    "caption_dropout_rate": caption_dropout_rate,
                }
                for vi in range(n_variants):
                    flat_idx = i * n_variants + vi
                    save_dict[f"prompt_embeds_v{vi}"] = prompt_embeds[flat_idx]
                    save_dict[f"attn_mask_v{vi}"] = attn_mask[flat_idx]
                    save_dict[f"t5_input_ids_v{vi}"] = t5_input_ids[flat_idx]
                    save_dict[f"t5_attn_mask_v{vi}"] = t5_attn_mask[flat_idx]
                    if crossattn_emb is not None:
                        save_dict[f"crossattn_emb_v{vi}"] = crossattn_emb[flat_idx]
                save_file(save_dict, str(cache_path))
                stats.written += 1
                if progress is not None:
                    progress(1, detail=f"{img_path.name} ({n_variants}v)")

    return stats


def _emit_pooled(te_path: Path, pooled_path: Path) -> bool:
    from safetensors.torch import load_file, save_file

    sd = load_file(str(te_path))
    out: dict = {}

    if "num_variants" in sd:
        n = int(sd["num_variants"])
        out["num_variants"] = sd["num_variants"]
        for vi in range(n):
            key = f"crossattn_emb_v{vi}"
            if key in sd:
                out[f"pooled_v{vi}"] = sd[key].amax(dim=0).contiguous()
        if not any(k.startswith("pooled_v") for k in out):
            return False
    elif "crossattn_emb_v0" in sd:
        out["pooled_v0"] = sd["crossattn_emb_v0"].amax(dim=0).contiguous()
    elif "crossattn_emb" in sd:
        out["pooled"] = sd["crossattn_emb"].amax(dim=0).contiguous()
    else:
        return False

    save_file(out, str(pooled_path))
    return True


def cache_pooled_text(
    cache_dir: Path,
    *,
    overwrite: bool = False,
    progress: ProgressFn | None = None,
) -> PreprocessStats:
    """Write ``{stem}_anima_pooled.safetensors`` next to each TE cache.

    ``pooled_v{i} = crossattn_emb_v{i}.amax(dim=0)`` for every variant present.
    Walks ``cache_dir`` recursively (nested caches mirror the source tree). Pure
    tensor reduction — no GPU / text encoder. ``failed`` counts TE files that
    carry no ``crossattn`` key. Returns counts; pass ``progress`` for a bar.
    """
    te_files = sorted(cache_dir.rglob(f"*{TE_CACHE_SUFFIX}"))
    stats = PreprocessStats(seen=len(te_files))

    if progress is not None:
        progress(0, total=len(te_files))

    for te_path in te_files:
        stem = te_path.name.removesuffix(TE_CACHE_SUFFIX)
        pooled_path = te_path.parent / (stem + POOLED_CACHE_SUFFIX)
        if pooled_path.exists() and not overwrite:
            stats.skipped += 1
        elif _emit_pooled(te_path, pooled_path):
            stats.written += 1
        else:
            stats.failed += 1
        if progress is not None:
            progress(1, detail=stem)

    return stats
