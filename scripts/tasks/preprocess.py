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


def _config_min_pixels() -> int:
    """The configured ``min_pixels`` threshold (merged chain), default 0.5MP."""
    from ._common import _path_overrides

    raw = _path_overrides().get("min_pixels", 500_000)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 500_000


def _target_res_args(extra) -> list[str]:
    """``--target_res E1 E2 …`` derived from the merged TOML's ``target_res`` key.

    Returns ``[]`` when an explicit ``--target_res`` is already in ``extra`` (CLI
    ARGS wins, no duplicate) or when the config value is absent / a bare
    ``[1024]`` (the legacy single-tier default — leave it off so the resize
    script's own default path runs). Invalid / unknown edges are dropped here so
    a typo in the TOML doesn't abort preprocessing.
    """
    if "--target_res" in extra:
        return []

    from library.datasets.buckets import ALLOWED_TARGET_RES

    from ._common import _path_overrides

    raw = _path_overrides().get("target_res")
    if not raw:
        return []
    edges = raw if isinstance(raw, (list, tuple)) else [raw]
    try:
        edges = [int(e) for e in edges]
    except (TypeError, ValueError):
        return []
    edges = [e for e in edges if e in ALLOWED_TARGET_RES]
    if not edges or edges == [1024]:
        return []
    return ["--target_res", *(str(e) for e in edges)]


def _preprocess_path_pattern_args(extra) -> list[str]:
    """``--path_pattern <glob>`` for GUI preprocess subset filtering.

    CLI ARGS wins when it already carries a path-pattern flag. GUI submits pass
    ``PREPROCESS_PATH_PATTERN`` so training can keep using the method's regular
    ``path_pattern`` independently.
    """
    if "--path_pattern" in extra or "--path-pattern" in extra:
        return []

    from ._common import _path_overrides

    raw = os.environ.get("PREPROCESS_PATH_PATTERN")
    if raw is None:
        raw = _path_overrides().get("preprocess_path_pattern")
    pattern = str(raw or "").strip()
    if not pattern or pattern == "*":
        return []
    return ["--path_pattern", pattern]


def _repa_pe_encoder() -> str | None:
    """The REPA vision encoder to cache, or ``None`` when REPA is off.

    Reads ``use_repa`` / ``repa_encoder`` from the merged config chain (the same
    ``_path_overrides`` the path knobs use — populated from ``METHOD`` /
    ``METHODS_SUBDIR`` or a GUI ``CONFIG_FILE`` snapshot). This lets the ConfigTab
    Train auto-chain — and any ``make preprocess METHOD=<repa-variant>`` — cache
    the ``{stem}_anima_pe_spatial.safetensors`` (or ``_anima_pe``) sidecars in the
    same pass, so a ``use_repa=true`` run doesn't bounce off train.py's
    "PE features absent" error. Plain ``make preprocess`` (no variant config in
    scope) sees no ``use_repa`` and returns ``None`` — the default stays fast.
    """
    from ._common import _path_overrides

    overrides = _path_overrides()
    raw = overrides.get("use_repa")
    # TOML/snapshot bools arrive as real bools; tolerate a stringified value too.
    enabled = raw is True or str(raw).strip().lower() in ("1", "true", "yes")
    if not enabled:
        return None
    encoder = str(overrides.get("repa_encoder") or "pe_spatial").strip()
    return encoder or "pe_spatial"


def _pop_target_res(extra) -> list[str]:
    """Strip ``--target_res E1 E2 …`` (a resize-only flag) from ``extra``.

    The VAE/TE/PE stages read whatever latent shapes are already on disk, so
    they must never see ``--target_res`` (their argparse doesn't define it).
    Removes the flag and its following ``nargs='+'`` integer values up to the
    next ``--option``.
    """
    cleaned: list[str] = []
    it = iter(extra)
    for tok in it:
        if tok == "--target_res":
            # drop the flag, then drop trailing int values until the next flag
            for nxt in it:
                if nxt.startswith("--"):
                    cleaned.append(nxt)
                    break
            continue
        cleaned.append(tok)
    return cleaned


def _resolve_lowres_filter(extra) -> tuple[list[str], list[str]]:
    """Reconcile the low-res input filter against CLI ``ARGS``.

    Returns ``(min_pixels_args, cleaned_extra)`` where ``cleaned_extra`` has
    our two convenience flags popped so the underlying scripts never see an
    arg their argparse doesn't define. Precedence (highest first):

      1. An explicit ``--min_pixels N`` in ``ARGS`` — left in ``extra`` and
         wins outright; we inject nothing (no duplicate ``--min_pixels``).
      2. ``--no_drop_lowres`` in ``ARGS`` → ``--min_pixels 0`` (keep every
         image), overriding ``drop_lowres_images = true`` in the TOML.
      3. ``--drop_lowres`` in ``ARGS`` → force the configured ``min_pixels``
         threshold, overriding ``drop_lowres_images = false`` in the TOML.
      4. Neither flag → fall back to the merged-config behavior
         (``_min_pixels_args``)."""
    cleaned = list(extra)
    no_drop = "--no_drop_lowres" in cleaned
    drop = "--drop_lowres" in cleaned
    cleaned = [a for a in cleaned if a not in ("--no_drop_lowres", "--drop_lowres")]

    # An explicit threshold in ARGS is authoritative; leave it in place.
    if "--min_pixels" in cleaned:
        return [], cleaned
    if no_drop:  # disable wins over enable when both are passed
        return ["--min_pixels", "0"], cleaned
    if drop:
        return ["--min_pixels", str(_config_min_pixels())], cleaned
    return _min_pixels_args(), cleaned


def cmd_preprocess_resize(extra):
    mp_args, extra = _resolve_lowres_filter(extra)
    tr_args = _target_res_args(extra)
    pp_args = _preprocess_path_pattern_args(extra)
    run(
        [
            PY,
            "scripts/preprocess/resize_images.py",
            "--src",
            _path("source_image_dir", "image_dataset"),
            "--dst",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--no_copy_captions",
            "--recursive",
            *mp_args,
            *tr_args,
            *pp_args,
            *extra,
        ]
    )


def cmd_preprocess_reconcile(extra):
    """Remove caches stale for the configured ``target_res`` (dry-run by default).

    Pass ``ARGS="--delete"`` to actually remove. ``target_res`` comes from the
    merged config (same as resize); an explicit ``--target_res`` in ``ARGS``
    wins. Useful after adding/dropping a tier so re-running preprocess + mask
    regenerates only the images whose bucket moved.
    """
    # _target_res_args returns [] for a bare [1024]/absent config AND when ARGS
    # already carries --target_res. Inject the 1024 default only in the former
    # case (no explicit flag in ARGS) so we never duplicate it.
    tr_args = _target_res_args(extra)
    if not tr_args and "--target_res" not in extra:
        tr_args = ["--target_res", "1024"]
    run(
        [
            PY,
            "scripts/preprocess/reconcile_caches.py",
            "--image-dir",
            _path("source_image_dir", "image_dataset"),
            "--resized-dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache-dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--mask-dir",
            _path("mask_dir", "post_image_dataset/masks"),
            *tr_args,
            *extra,
        ]
    )


def cmd_preprocess_vae(extra):
    pp_args = _preprocess_path_pattern_args(extra)
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
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


def cmd_preprocess_cond_resize(extra):
    """Resize conditioning images to match target bucket resolutions."""
    run(
        [
            PY,
            "scripts/preprocess/resize_images.py",
            "--src",
            _path("conditioning_data_dir", os.environ.get("CONDITIONING_DATA_DIR", "conditioning_data")),
            "--dst",
            _path("conditioning_resized_dir", os.environ.get("CONDITIONING_RESIZED_DIR", "post_image_dataset/cond_resized")),
            "--no_copy_captions",
            "--recursive",
            *extra,
        ]
    )


def cmd_preprocess_cond_vae(extra):
    """Cache VAE latents for conditioning images."""
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
            "--dir",
            _path("conditioning_resized_dir", os.environ.get("CONDITIONING_RESIZED_DIR", "post_image_dataset/cond_resized")),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--vae",
            _path("vae", "models/vae/qwen_image_vae.safetensors"),
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
            "--recursive",
            *pp_args,
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
    mp_args, extra = _resolve_lowres_filter(extra)
    pp_args = _preprocess_path_pattern_args(extra)
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
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
            *mp_args,
            *pp_args,
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
            "scripts/preprocess/cache_pooled_text.py",
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

    Also emits the dataset-mean PE centroid sidecar
    (``post_image_dataset/ip_adapter/anima_pe_centroid_pe.safetensors``) via
    ``--centroid`` so IP-Adapter mean-centering works without a separate pass.
    """
    run(
        [
            PY,
            "scripts/preprocess/cache_pe_encoder.py",
            "--dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--encoder",
            "pe",
            "--recursive",
            # Emit the dataset-mean PE centroid sidecar after the cache pass so
            # IP-Adapter's mean-centering (ip_centroid_path) and DCW v4 have it
            # without a separate --centroid_only run.
            "--centroid",
            *extra,
        ]
    )


def cmd_preprocess_pe_spatial(extra):
    """Cache PE-Spatial-B16-512 dense patch-token features for REPA v2.

    Reads pre-resized images from ``post_image_dataset/resized/`` and writes
    ``{stem}_anima_pe_spatial.safetensors`` sidecars into the LoRA cache dir
    (disjoint from the PE-Core ``_anima_pe`` caches CMMD reads). No centroid —
    REPA aligns per-patch, not against a dataset mean. Run before a
    ``use_repa=true`` training arm.
    """
    run(
        [
            PY,
            "scripts/preprocess/cache_pe_encoder.py",
            "--dir",
            _path("resized_image_dir", "post_image_dataset/resized"),
            "--cache_dir",
            _path("lora_cache_dir", "post_image_dataset/lora"),
            "--encoder",
            "pe_spatial",
            "--recursive",
            *extra,
        ]
    )


def cmd_caption_index(extra):
    """Build the method-agnostic typed-tag caption index.

    Walks caption sidecars under the source dir, classifies tags into
    character / copyright / artist / count via the Anima Tagger vocab, and
    writes ``post_image_dataset/captions/caption_index.json`` (per-image typed
    tags + group inversions). Pure data, no GPU. Consumed by the IP-Adapter
    distinct-pair sampler, artist balancing, and dataset analytics. Regenerate
    when the dataset or vocab changes.
    """
    pp_args = _preprocess_path_pattern_args(extra)
    run(
        [
            PY,
            "scripts/preprocess/build_caption_index.py",
            "--src",
            _path("source_image_dir", "image_dataset"),
            *pp_args,
            *extra,
        ]
    )


# Same default the build_caption_index.py CLI uses. `cmd_preprocess` auto-fetches
# this (~0.7 MB) vocab on demand when it's absent: GUI users reach preprocess
# without ever running `make download-models` (via `download-tagger`), and the
# caption index it gates is a hard requirement for soft-tokens contrastive
# training (train.py raises FileNotFoundError without it). The fetch is
# best-effort — a partial/offline setup that can't pull it falls through to the
# skip path rather than aborting the already-done GPU work.
_CAPTION_INDEX_VOCAB = "models/captioners/anima-tagger-v2/vocab.json"


def cmd_preprocess(extra):
    # PE features are NOT cached here by default — only CMMD validation / DCW v4
    # need them, and those paths chain `preprocess-pe` explicitly. Leaving PE out
    # keeps the default LoRA preprocess fast on machines that won't use the vision
    # tower. The one exception is a `use_repa=true` variant in scope: REPA aligns
    # against PE features every step, so they're chained in at the end (see the
    # `_repa_pe_encoder()` block below).
    cmd_preprocess_resize(extra)
    # The VAE/TE steps read on-disk shapes — strip the low-res convenience flags
    # AND the resize-only --target_res so their argparse never sees an arg it
    # doesn't define. (resize pops lowres itself via _resolve_lowres_filter.)
    downstream = _pop_target_res(extra)
    _, vae_extra = _resolve_lowres_filter(downstream)
    cmd_preprocess_vae(vae_extra)
    cmd_preprocess_te(downstream)
    # Build the method-agnostic caption index (pure data, no GPU, ~seconds) as a
    # free by-product — it's consumed by the IP-Adapter pair sampler, artist
    # balancing, dataset analytics, AND soft-tokens contrastive training (which
    # hard-errors without it). Best-effort: the lowres/resize flags in `extra`
    # aren't part of its argparse, so pass none.
    vocab = _path("caption_index_vocab", _CAPTION_INDEX_VOCAB)
    if not os.path.exists(vocab):
        # GUI users reach preprocess without running `make download-models`, so
        # fetch the tiny tagger vocab on demand. Catch broadly: run() exits via
        # SystemExit on a non-zero `hf`, and a missing `hf` binary raises OSError
        # — either way fall through to the skip below instead of aborting the
        # already-done GPU work.
        print("  [preprocess] tagger vocab missing; fetching it for caption-index")
        try:
            from .downloads import cmd_download_tagger

            cmd_download_tagger([])
        except (SystemExit, OSError) as e:
            print(f"  [preprocess] tagger vocab auto-download failed: {e}")
    if os.path.exists(vocab):
        cmd_caption_index([])
    else:
        print(
            f"  [preprocess] skipping caption-index: tagger vocab not found at "
            f"{_CAPTION_INDEX_VOCAB} and auto-download failed. Run "
            f"`make download-tagger`, then `make caption-index` "
            f"(soft-tokens contrastive training needs it)."
        )

    # REPA arm: when the variant config in scope has `use_repa=true`, cache the
    # vision-encoder PE sidecars REPA aligns against in this same pass. train.py
    # errors out asking for them otherwise; chaining here means the ConfigTab
    # "Train" auto-preprocess (and `make preprocess METHOD=<repa-variant>`)
    # builds them without a second manual step. Idempotent — cache_pe_encoder.py
    # skips images already cached. PE-less variants never pay for the tower.
    encoder = _repa_pe_encoder()
    if encoder is not None:
        print(f"  [preprocess] use_repa=true → caching REPA PE features ({encoder})")
        if encoder == "pe_spatial":
            cmd_preprocess_pe_spatial([])
        else:
            cmd_preprocess_pe([])


def cmd_preprocess_config(extra):
    """Preprocess the exact directories named in a ``--dataset_config`` TOML.

    Unlike ``cmd_preprocess`` (which resolves the repo's standard
    ``image_dataset/`` → ``post_image_dataset/`` layout from the merged
    config), this drives off the same dataset config the *training* job will
    consume, so one file fully describes an ad-hoc job — no reliance on the
    default layout. For each ``[[datasets.subsets]]`` it:

      1. bucket-resizes ``--src`` (the originals, with caption sidecars) into
         that subset's ``image_dir`` — the source dir is never modified;
      2. caches VAE latents from ``image_dir`` into the subset's ``cache_dir``;
      3. caches text embeddings (captions read from ``--src``) into ``cache_dir``.

    A config can't encode where the *un-resized* originals live (its
    ``image_dir`` is the post-resize dir training reads), so the source is the
    one explicit flag: ``--src <dir>``. The ComfyUI trainer node uses this to
    cache a single-image temp dir before its chained training job runs.

    The VAE / text-encoder / DiT used for caching default to the config-resolved
    ``models/`` paths (base → preset → method merge), but can be overridden with
    ``--vae`` / ``--qwen3`` / ``--dit`` so a caller can point the cache at models
    living elsewhere — e.g. the ComfyUI trainer node passes the paths ComfyUI's
    own ``folder_paths`` registers, so it never assumes a copy under
    ``anima_lora/models/``.

    Usage: ``preprocess-config --dataset_config <path> --src <dir>
    [--vae <path>] [--qwen3 <path>] [--dit <path>] [extra…]``
    (any remaining args are forwarded to the resize step).
    """
    import toml

    args = list(extra)
    cfg_path: str | None = None
    src_dir: str | None = None
    vae_path = _path("vae", "models/vae/qwen_image_vae.safetensors")
    qwen3_path = _path("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors")
    dit_path = _path(
        "pretrained_model_name_or_path",
        "models/diffusion_models/anima-base-v1.0.safetensors",
    )
    rest: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--dataset_config" and i + 1 < len(args):
            cfg_path = args[i + 1]
            i += 2
        elif args[i] == "--src" and i + 1 < len(args):
            src_dir = args[i + 1]
            i += 2
        elif args[i] == "--vae" and i + 1 < len(args):
            vae_path = args[i + 1]
            i += 2
        elif args[i] == "--qwen3" and i + 1 < len(args):
            qwen3_path = args[i + 1]
            i += 2
        elif args[i] == "--dit" and i + 1 < len(args):
            dit_path = args[i + 1]
            i += 2
        else:
            rest.append(args[i])
            i += 1
    if not cfg_path or not src_dir:
        raise SystemExit(
            "preprocess-config requires --dataset_config <path> and --src <dir>"
        )

    # A real-time scanner (e.g. Windows Defender) often holds a brief exclusive
    # lock on a *just-created* file, surfaced as PermissionError [Errno 13] on
    # Windows. The ComfyUI trainer node writes this config milliseconds before
    # the daemon's preprocess job opens it, so retry through that transient lock
    # rather than failing the whole chain. A genuinely unreadable file still
    # raises after the budget is spent.
    import time

    last_err: OSError | None = None
    for attempt in range(10):
        try:
            cfg = toml.load(cfg_path)
            break
        except PermissionError as e:
            last_err = e
            time.sleep(0.2 * (attempt + 1))
    else:
        raise SystemExit(
            f"could not read {cfg_path} after retrying (last error: {last_err}). "
            "If this persists, exclude the dataset/temp dir from your antivirus."
        )
    subsets = [
        sub
        for ds in (cfg.get("datasets") or [])
        for sub in (ds.get("subsets") or [])
        if sub.get("image_dir")
    ]
    if not subsets:
        raise SystemExit(f"no [[datasets.subsets]] with image_dir in {cfg_path}")

    for sub in subsets:
        image_dir = sub["image_dir"]
        cache_dir = sub.get("cache_dir") or image_dir
        # 1) bucket-resize originals → image_dir. cache_latents.py keys caches
        #    by the on-disk (native) size, so the resized size must already be
        #    the constant-token bucket the trainer will select. Captions stay
        #    in --src (TE caching reads them there).
        run(
            [
                PY,
                "scripts/preprocess/resize_images.py",
                "--src",
                src_dir,
                "--dst",
                image_dir,
                "--no_copy_captions",
                "--min_pixels",
                "0",
                "--bucket_reso_steps",
                "64",
                "--recursive",
                *rest,
            ]
        )
        # 2) VAE latents → cache_dir
        run(
            [
                PY,
                "scripts/preprocess/cache_latents.py",
                "--dir",
                image_dir,
                "--cache_dir",
                cache_dir,
                "--vae",
                vae_path,
                "--batch_size",
                "4",
                "--chunk_size",
                "64",
                "--recursive",
            ]
        )
        # 3) text embeddings (captions read from --src) → cache_dir
        run(
            [
                PY,
                "scripts/preprocess/cache_text_embeddings.py",
                "--dir",
                src_dir,
                "--cache_dir",
                cache_dir,
                "--qwen3",
                qwen3_path,
                "--dit",
                dit_path,
                "--recursive",
            ]
        )
