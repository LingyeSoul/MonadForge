"""Default-dataset preprocessing: resize → VAE latents → text-embedding caches."""

from __future__ import annotations

import os

from ._common import PY, _path, run


# Subfolders under the source dir are walked by default — matches the
# `recursive = true` subset default in configs/base.toml. Stems must stay
# unique across the tree (cache filenames are stem-keyed and flat). Pass
# `--no_recursive` (or edit configs) to opt out.
def _min_pixels_args() -> list[str]:
    """``--min_pixels <N>`` derived from the variant TOML's
    ``drop_lowres_images`` + ``min_pixels`` keys (resolved through the same
    base → preset → method merge chain training uses, via ``_path_overrides``
    in scripts/tasks/_common.py).

    Returns ``[]`` when both keys are absent so plain CLI use keeps each
    script's own argparse default (500_000 = 0.5MP). ``drop_lowres_images
    = false`` forces ``--min_pixels 0`` even when ``min_pixels`` is set, so
    the user can flip a single boolean to disable the filter."""
    from ._common import _path_overrides  # local import: avoids unused circular

    overrides = _path_overrides()
    if "drop_lowres_images" not in overrides and "min_pixels" not in overrides:
        return []
    if overrides.get("drop_lowres_images") is False:
        return ["--min_pixels", "0"]
    raw = overrides.get("min_pixels", 500_000)
    try:
        n = max(0, int(raw))
    except (TypeError, ValueError):
        return []
    return ["--min_pixels", str(n)]


def cmd_preprocess_resize(extra):
    run(
        [
            PY,
            "preprocess/resize_images.py",
            "--src",
            _path("source_image_dir", "image_dataset"),
            "--dst",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--no_copy_captions",
            "--recursive",
            *_min_pixels_args(),
            *extra,
        ]
    )


def cmd_preprocess_vae(extra):
    run(
        [
            PY,
            "preprocess/cache_latents.py",
            "--dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--vae",
            _path("vae", "models/vae/qwen_image_vae.safetensors"),
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
            "--recursive",
            *extra,
        ]
    )


def cmd_preprocess_te(extra):
    # CAPTION_SHUFFLE_VARIANTS / CAPTION_TAG_DROPOUT_RATE let the GUI's
    # Preprocessing tab control these without editing this file. Defaults
    # match the historical hardcoded values so non-GUI invocations are
    # unchanged.
    shuffle_variants = os.environ.get("CAPTION_SHUFFLE_VARIANTS", "4")
    tag_dropout_rate = os.environ.get("CAPTION_TAG_DROPOUT_RATE", "0.1")
    run(
        [
            PY,
            "preprocess/cache_text_embeddings.py",
            "--dir",
            _path("source_image_dir", "image_dataset"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--qwen3",
            _path("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors"),
            "--dit",
            _path("pretrained_model_name_or_path", "models/diffusion_models/anima-base-v1.0.safetensors"),
            "--caption_shuffle_variants",
            shuffle_variants,
            "--caption_tag_dropout_rate",
            tag_dropout_rate,
            "--recursive",
            *_min_pixels_args(),
            *extra,
        ]
    )


def cmd_preprocess_pooled(extra):
    """Cache pooled text embeddings (max over seq dim) from existing TE caches.

    Reads ``{stem}_anima_te.safetensors`` from the LoRA cache dir and writes
    ``{stem}_anima_pooled.safetensors`` sidecars next to them. Consumed by
    ``make distill-mod`` to skip a redundant ``.max(dim=1)`` per training
    microstep / val sigma. No GPU needed.
    """
    run(
        [
            PY,
            "preprocess/cache_pooled_text.py",
            "--dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            *extra,
        ]
    )


def cmd_preprocess_pe(extra):
    """Cache PE-Core-L14-336 vision-encoder features.

    Reads pre-resized images from ``post_image_dataset/resized/`` (the
    standard LoRA pipeline source) and writes
    ``{stem}_anima_pe.safetensors`` sidecars into the LoRA cache dir so the
    dataset's existing ``cache_dir`` lookup finds them.

    Consumed by IP-Adapter when reading PE features off disk and by the
    DCW v4 fusion head's pooled-image-feature input channel.
    """
    run(
        [
            PY,
            "preprocess/cache_pe_encoder.py",
            "--dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--encoder",
            "pe",
            "--recursive",
            *extra,
        ]
    )


def cmd_preprocess(extra):
    # PE features are intentionally NOT cached here — only IP-Adapter / CMMD /
    # DCW v4 need them, and those paths chain `preprocess-pe` explicitly (see
    # `exp-ip-adapter-preprocess`). Leaving PE out keeps the default LoRA
    # preprocess fast on machines that won't ever use the vision tower.
    cmd_preprocess_resize(extra)
    cmd_preprocess_vae(extra)
    cmd_preprocess_te(extra)
