"""Training entry-points for shipped methods (lora family + lora-gui + EasyControl).

Each ``cmd_*`` is a thin shim that translates env vars + extra argv into the
right ``train.py`` (via ``accelerate launch``) call. Experimental methods
(postfix, ip-adapter) live in ``scripts/experimental_tasks/training.py`` and are
wired up under ``make exp-*`` in ``tasks.py``.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tomllib
from pathlib import Path

import toml

from ._common import (
    PY,
    ROOT,
    _preset,
    bespoke_preset_flags,
    queue_command,
    run,
    train,
)

# EasyControl control-task projects are descriptor-driven: a self-contained
# ``configs/easycontrol/<EASYADAPTER>.toml`` (``name`` slug + [staging]/[preprocess]/
# [training] knob tables + a [[datasets]] blueprint tail) selected via the
# EASYADAPTER env var. All train the base ``easycontrol`` method with the
# descriptor's [training] table folded in as CLI overrides. Per-adapter
# staging/preprocess bodies are registered in ``_EASY_ADAPTERS`` (below).


def _easyadapter() -> str:
    """Resolve the EASYADAPTER env var (validated). "" → default easycontrol."""
    adapter = (os.environ.get("EASYADAPTER") or "").strip()
    if adapter and adapter not in _EASY_ADAPTERS:
        raise SystemExit(
            f"Unknown EASYADAPTER={adapter!r}. Known: {sorted(_EASY_ADAPTERS)}."
        )
    return adapter


def cmd_lora(extra):
    train("lora", extra)


def cmd_turbo(extra):
    """Turbo Anima — DP-DMD distillation (docs: docs/methods/turbo.md).

    Bypasses train.py / accelerate (single-GPU bespoke loop, like distill-mod).
    Reads ``configs/methods/turbo.toml``; trailing args are forwarded so user
    CLI flags override TOML values, e.g.::

        make turbo                                  # defaults: rank=64, 4-step
        make turbo ARGS="--student_rank 64 --iterations 5000"
        make turbo ARGS="--single_prompt_idx 0"     # Phase 0 single-prompt overfit
        make turbo --queue                          # enqueue on the daemon

    The output is a normal LoRA — a distilled student ships at
    https://huggingface.co/sorryhyun/anima-turbo-4step (infer with
    ``make test-turbo`` / ``--infer_steps 4 --cfg 1.0``).

    Honors ``PRESET`` (default ``default``) — translates ``blocks_to_swap`` and
    ``gradient_checkpointing`` from ``configs/presets.toml`` into CLI flags so
    ``make turbo PRESET=low_vram`` enables grad ckpt + unsloth offload, and
    ``PRESET=half/quarter/tenth`` shrinks the dataset via ``--sample_ratio``.
    ``extra`` is appended last, so user CLI overrides win.

    ``--queue`` anywhere in ``extra`` enqueues the distillation as a daemon
    command-job (run serially behind any other queued work) and returns
    immediately, instead of running it inline — the bespoke-loop analogue of
    ``make lora --queue``. The job is labeled ``turbo`` so the GUI's Turbo
    tab can re-attach to it. Preset flags are baked into the queued argv since the
    daemon's command path does no config merging.
    """
    extra = list(extra or [])
    preset_flags = bespoke_preset_flags(_preset())
    argv = ["-m", "scripts.distill_turbo.distill", *preset_flags, *extra]
    if "--queue" in argv:
        argv.remove("--queue")
        queue_command("turbo", argv)
        return
    run([PY, *argv])


def cmd_lora_gui(extra):
    """Train from configs/gui-methods/<variant>.toml.

    Variant is taken from GUI_PRESETS env var, falling back to the first
    positional extra arg (``python tasks.py lora-gui tlora ...``), then to
    ``lora`` (plain). Extra args after the variant are forwarded as usual.
    """
    variant = os.environ.get("GUI_PRESETS")
    if not variant and extra and not extra[0].startswith("-"):
        variant = extra[0]
        extra = extra[1:]
    variant = variant or "lora"

    expected = ROOT / "configs" / "gui-methods" / f"{variant}.toml"
    if not expected.exists():
        available = sorted(
            p.stem for p in (ROOT / "configs" / "gui-methods").glob("*.toml")
        )
        print(
            f"Unknown gui-methods variant: {variant!r}\n"
            f"Available: {', '.join(available)}",
            file=sys.stderr,
        )
        sys.exit(1)

    train(variant, extra, methods_subdir="gui-methods")


def _toml_table_to_argv(table: dict) -> list[str]:
    """Flatten a flat TOML table into ``--key value`` train.py argv.

    Bools become bare ``--flag`` when true (omitted when false); lists spread
    into ``--key v1 v2``; scalars become ``--key str(value)``. Used to fold a
    near_twins.toml ``[training]`` table into CLI overrides (CLI wins the merge
    chain, so these override the easycontrol method config).
    """
    argv: list[str] = []
    for key, val in table.items():
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        elif isinstance(val, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in val)
        else:
            argv.append(flag)
            argv.append(str(val))
    return argv


def _easy_cfg_path(adapter: str) -> Path:
    """The descriptor file for an EasyControl adapter project."""
    return ROOT / "configs" / "easycontrol" / f"{adapter}.toml"


def _easy_load(adapter: str) -> tuple[dict, str, str]:
    """Load ``configs/easycontrol/<adapter>.toml`` → ``(cfg, name, base)``.

    The top-level ``name`` key (default = the file stem) is the single source of
    truth for the run: it picks the ``post_image_dataset/easycontrol/<name>/``
    base tree and the ``anima_easycontrol_<name>`` output_name default. Set
    ``name = "sanitize"`` and the whole pipeline reroutes under
    ``easycontrol/sanitize/`` with no other path edits — re-run staging →
    preprocess → train so the generated blueprint tail tracks the new slug.
    Explicit ``[preprocess]`` path keys / ``[training].output_name`` still win if
    present (back-compat), but are no longer needed.
    """
    path = _easy_cfg_path(adapter)
    if not path.is_file():
        raise SystemExit(
            f"{path} not found — run `make easycontrol-staging "
            f"EASYADAPTER={adapter}` first to materialize the staging tree."
        )
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    name = str(cfg.get("name") or path.stem).strip()
    base = f"post_image_dataset/easycontrol/{name}"
    return cfg, name, base


def _resolve_blueprint_path(path: str, name: str) -> str:
    """Resolve a blueprint subset path against the current ``name`` slug.

    Two complementary steps so ``name`` stays the single source of truth:
    1. Interpolate the ``{name}`` placeholder the miner now writes into the
       blueprint tail (``post_image_dataset/easycontrol/{name}/resized`` …).
    2. As a fallback for older blueprints that baked a resolved slug in, swap the
       ``<slug>`` component of a ``post_image_dataset/easycontrol/<slug>/...``
       path to ``name``. Non-matching paths (custom locations) are left untouched.
    """
    path = path.replace("{name}", name)
    parts = Path(path).parts
    if len(parts) >= 3 and parts[:2] == ("post_image_dataset", "easycontrol"):
        return str(Path(*parts[:2], name, *parts[3:]))
    return path


def _easy_train_extra(adapter: str, extra) -> list[str]:
    """Build train.py extra-argv for an ``EASYADAPTER=<adapter>`` descriptor run.

    ``configs/easycontrol/<adapter>.toml`` is a multi-purpose file (top-level
    ``name`` + ``[staging]`` / ``[preprocess]`` / ``[training]`` knob tables above
    a ``[general]`` + ``[[datasets]]`` blueprint), but train.py's dataset-config
    validator rejects any top-level key outside the blueprint. So we extract just
    the blueprint sections into a clean generated sidecar and point
    ``--dataset_config`` at that, then fold the optional ``[training]`` table
    (with ``output_name`` defaulting to ``anima_easycontrol_<name>``) into CLI
    overrides on top of the easycontrol method config. User-supplied ``extra``
    argv is appended last so it still wins.
    """
    cfg, name, base = _easy_load(adapter)
    blueprint = {k: cfg[k] for k in ("general", "datasets") if k in cfg}
    if not blueprint.get("datasets"):
        raise SystemExit(
            f"{_easy_cfg_path(adapter)} has no [[datasets]] blueprint — run "
            f"`make easycontrol-staging EASYADAPTER={adapter}` first."
        )

    # Resolve subset paths against the current `name` slug so a `name` change
    # reroutes training. ``text_cache_dir`` rides the slug too (colorize redirects
    # the TE cache; near_twins leaves it unset).
    for ds in blueprint.get("datasets", []):
        for s in ds.get("subsets", []):
            for key in ("image_dir", "cache_dir", "cond_cache_dir", "text_cache_dir"):
                if key in s:
                    s[key] = _resolve_blueprint_path(s[key], name)

    # Write the blueprint-only dataset config under the slug base dir. Stable-pathed
    # so the --queue daemon path can re-read it later, regenerated each invocation.
    base_dir = ROOT / base
    base_dir.mkdir(parents=True, exist_ok=True)
    ds_path = base_dir / "dataset_config.toml"
    ds_path.write_text(
        f"# AUTO-GENERATED from configs/easycontrol/{adapter}.toml — do not edit.\n"
        "# Blueprint-only copy (train.py's dataset-config validator rejects the\n"
        "# name + [staging]/[preprocess]/[training] knobs in the source file).\n\n"
        + toml.dumps(blueprint),
        encoding="utf-8",
    )

    # output_name defaults to the name-derived slug; an explicit [training].output_name wins.
    training = dict(cfg.get("training") or {})
    training.setdefault("output_name", f"anima_easycontrol_{name}")

    return [
        "--dataset_config",
        str(ds_path),
        *_toml_table_to_argv(training),
        *list(extra or []),
    ]


def cmd_easycontrol(extra):
    """EasyControl. ``EASYADAPTER=<name>`` selects a control-task project
    described by ``configs/easycontrol/<name>.toml`` (e.g. ``colorize`` or
    ``near_twins``); unset → the default ref==target easycontrol.toml.

    A descriptor run always trains the base ``easycontrol`` method, folding the
    descriptor's blueprint tail in via ``--dataset_config`` and its optional
    ``[training]`` table in as CLI overrides (see ``_easy_train_extra``)."""
    adapter = _easyadapter()
    if adapter in _EASY_ADAPTERS:
        train("easycontrol", _easy_train_extra(adapter, extra))
        return
    train("easycontrol", extra)


def _near_twins_preprocess(adapter: str, cfg: dict, base: str, extra) -> None:
    """Resize + VAE/TE caching for the mined near-twin pair tree.

    Every knob is read from the ``[preprocess]`` table of
    ``configs/easycontrol/near_twins.toml`` (written by the staging step), so this
    stays in lockstep with the dataset blueprint that step also rewrites. The
    staging/resized/cache/cond dirs default to ``post_image_dataset/easycontrol/
    <name>/{staging,resized,cache,cond}`` (the top-level ``name`` slug), so they
    need no manual ``[preprocess]`` entry; an explicit path key still overrides.

    The mined ``staging/`` tree holds native-resolution images (symlinks to the
    corpus), so the pass first resizes them into constant-token buckets under
    ``resized/`` (``target_res`` tiers) — that resized tree is the training
    ``image_dir`` — and only then VAE/TE-encodes it into ``cache/``.
    """
    pp = cfg.get("preprocess") or {}
    staging = pp.get("image_dir", f"{base}/staging")
    resized = pp.get("resized_dir", f"{base}/resized")
    cache = pp.get("cache_dir", f"{base}/cache")
    recursive = ["--recursive"] if pp.get("recursive", True) else []
    # Bucket tiers: descriptor's [preprocess].target_res wins, else base.toml's
    # target_res (tracks the shared tier contract); final fallback [1024].
    target_res = pp.get("target_res")
    if target_res is None:
        from ._common import _path_overrides

        target_res = _path_overrides().get("target_res", [1024])
    if not isinstance(target_res, (list, tuple)):
        target_res = [target_res]
    target_res_flag = (
        ["--target_res", *[str(e) for e in target_res]] if target_res else []
    )
    # Free-fit (native-aspect token-band resize): [preprocess].freefit wins, else
    # base.toml's freefit (shared with the LoRA pipeline). CAVEAT pair coherence —
    # the cond/target pairing in _near_twins_build_cond matches twins on an EXACT
    # same bucket string, and the loader (_load_cond_latent) resolves+validates the
    # cond latent at the TARGET's bucket, so a pair whose members free-fit to
    # different shapes is dropped at cond-build with a warning (more often than
    # under coarse constant buckets, since free-fit granularity is per-token-count).
    # Near-twins share an aspect so most pairs survive; pairs from differently-sized
    # source scans are the casualties.
    freefit = pp.get("freefit")
    if freefit is None:
        from ._common import _path_overrides

        freefit = bool(_path_overrides().get("freefit", False))
    freefit_flag: list[str] = []
    if freefit:
        freefit_flag = ["--freefit"]
        if pp.get("freefit_max_ratio") is not None:
            freefit_flag += ["--freefit_max_ratio", str(pp["freefit_max_ratio"])]

    # 1. Resize staging tree into buckets. min_pixels defaults to 0 (not 0.5MP) so a
    #    small member can't be dropped and orphan its pair partner.
    run(
        [
            PY,
            "scripts/preprocess/resize_images.py",
            "--src",
            staging,
            "--dst",
            resized,
            "--min_pixels",
            str(pp.get("min_pixels", 0)),
            *target_res_flag,
            *freefit_flag,
            *recursive,
        ]
    )
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
            "--dir",
            resized,
            "--cache_dir",
            cache,
            "--vae",
            pp.get("vae", "models/vae/qwen_image_vae.safetensors"),
            "--batch_size",
            str(pp.get("batch_size", 4)),
            "--chunk_size",
            str(pp.get("chunk_size", 64)),
            *recursive,
        ]
    )
    # Text-encoder outputs (captions copied during resize).
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
            "--dir",
            resized,
            "--cache_dir",
            cache,
            "--qwen3",
            pp.get("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors"),
            "--dit",
            pp.get("dit", "models/diffusion_models/anima-base-v1.0.safetensors"),
            "--caption_shuffle_variants",
            str(pp.get("caption_shuffle_variants", 4)),
            "--caption_tag_dropout_rate",
            str(pp.get("caption_tag_dropout_rate", 0.1)),
            *recursive,
        ]
    )
    # Optional REPA vision-encoder sidecars, gated on [preprocess] pe_encoder so
    # plain runs skip the encoder pass. Idempotent (pre-skips cached).
    pe_encoder = pp.get("pe_encoder")
    if pe_encoder:
        run(
            [
                PY,
                "scripts/preprocess/cache_pe_encoder.py",
                "--dir",
                resized,
                "--cache_dir",
                cache,
                "--encoder",
                str(pe_encoder),
                *recursive,
            ]
        )
    # Pair the cond/ tree (the _tags reference latent for each _no_tags target).
    _near_twins_build_cond(pp, base)


def _near_twins_build_cond(pp: dict, base: str) -> None:
    """Materialize the ``cond/`` latent tree for the near-twins *removal* task.

    Pairing convention (matches the generated blueprint): the denoising target is
    the clean ``_no_tags`` member; its ``_tags`` twin is the EasyControl condition
    reference. The loader resolves the cond latent by the *target* stem (bucket
    first, then a stem-glob fallback) under ``cond_cache_dir``, so for each
    ``{id}_no_tags_{WxH}_anima.npz`` target latent we symlink the sibling
    ``{id}_tags_{W'xH'}_anima.npz`` into ``cond/<artist>/{id}_no_tags_{W'xH'}_anima.npz``
    (cond content = the _tags latent, filed under the _no_tags stem at the *twin's*
    bucket). Same-bucket twins are the constant-bucket common case; under free-fit a
    twin may resize to a different shape (W'xH' ≠ WxH) and is paired anyway — cond≠
    target shapes are supported (the cond stream encodes at its native token count,
    cond_diff_loss self-skips on a shape mismatch). Only a truly unpaired target
    (no _tags twin at any bucket) is skipped with a warning.

    Pure symlinks over the existing cache; the tree is rebuilt from scratch each
    run so a dropped pair can't leave a stale link behind.
    """
    cache_dir = ROOT / pp.get("cache_dir", f"{base}/cache")
    cond_dir = ROOT / pp.get("cond_dir", f"{base}/cond")
    if not cache_dir.is_dir():
        raise SystemExit(
            f"{cache_dir} not found — run the VAE/TE caching pass first "
            "(`make easycontrol-preprocess EASYADAPTER=near_twin`)."
        )
    if cond_dir.exists():
        shutil.rmtree(cond_dir)

    pat = re.compile(r"^(?P<id>.+)_no_tags_(?P<bucket>\d{4}x\d{4})_anima\.npz$")
    twin_pat = re.compile(r"_tags_(?P<bucket>\d{4}x\d{4})_anima\.npz$")
    linked = skipped = crossfit = 0
    for npz in sorted(cache_dir.rglob("*_no_tags_*_anima.npz")):
        m = pat.match(npz.name)
        if not m:
            continue
        # Prefer the same-bucket twin (zero-cost, the constant-bucket common case).
        twin = npz.with_name(f"{m['id']}_tags_{m['bucket']}_anima.npz")
        twin_bucket = m["bucket"]
        if not twin.is_file():
            # Free-fit may land the _tags twin at a DIFFERENT shape than the clean
            # target — that's fine (cond≠target is supported: the cond stream
            # encodes at its native token count, cond_diff_loss self-skips on a
            # shape mismatch). Pair the twin at whatever bucket it has.
            cands = sorted(npz.parent.glob(f"{m['id']}_tags_*_anima.npz"))
            if not cands:
                print(
                    f"  [near_twin cond] no _tags twin for "
                    f"{npz.relative_to(cache_dir)} — skipping (truly unpaired).",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            twin = cands[0]
            twin_bucket = twin_pat.search(twin.name)["bucket"]
            crossfit += 1
        # File the symlink under the target stem at the TWIN's actual bucket so the
        # loader's stem-glob fallback resolves it and reads the right reso. For the
        # same-bucket case the name is unchanged (target bucket == twin bucket).
        link = (
            cond_dir
            / npz.relative_to(cache_dir).parent
            / f"{m['id']}_no_tags_{twin_bucket}_anima.npz"
        )
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(twin.resolve())
        linked += 1
    print(
        f"[near_twins cond] linked {linked} cond latents into {cond_dir}"
        + (f" ({crossfit} cross-shape under free-fit)" if crossfit else "")
        + (f" ({skipped} skipped)" if skipped else "")
    )


def _near_twins_stage(adapter: str, cfg: dict, base: str, extra) -> None:
    """Mine the in-artist near-twin pair tree (the near_twins/sanitize staging step).

    The miner self-reads its ``[staging]`` table + ``name`` slug from the
    descriptor and rewrites the blueprint tail back into the *same* file, so we
    point both ``--config`` (read) and ``--config-out`` (write) at
    ``configs/easycontrol/<adapter>.toml`` — otherwise the miner falls back to its
    near_twins.toml default and a ``sanitize`` run would mine into the wrong file.
    ``cfg``/``base`` are unused (the miner re-reads the file itself); the signature
    just matches the registry contract. User ``extra`` argv wins last (the miner's
    precedence is CLI > toml > default), so an explicit ``--config`` still overrides."""
    cfg_path = str(_easy_cfg_path(adapter))
    run(
        [
            PY,
            "-m",
            "easycontrol_adapters.tools.near_twins",
            "--config",
            cfg_path,
            "--config-out",
            cfg_path,
            *extra,
        ]
    )


def _colorize_prep_paths(base: str) -> list[str]:
    """Slug-derived prep.py path flags so ``name`` reroutes colorize's trees.

    ``--src`` stays the shared color corpus (``post_image_dataset/resized`` — the
    colorize *targets*); only the synthetic staging tree + cond/text caches ride
    the slug, matching the blueprint's ``cond_cache_dir`` / ``text_cache_dir``.
    Injected before the descriptor knob tables so a ``[staging]``/``[preprocess]``
    key (or user ``extra``) still wins via argparse last-flag precedence."""
    return [
        "--staging",
        f"{base}/staging",
        "--cond_cache_dir",
        f"{base}/cond",
        "--text_cache_dir",
        f"{base}/text",
    ]


def _colorize_stage(adapter: str, cfg: dict, base: str, extra) -> None:
    """Colorize staging: synthesize the synthetic B&W manga condition tree.

    Runs only prep.py's mangafy stage (``--skip_encode --skip_text``) over the
    shared color corpus into ``{base}/staging``. Knobs come from the descriptor's
    ``[staging]`` table; user ``extra`` argv wins last. The cond-latent +
    color-only-text caching is the separate preprocess pass."""
    knobs = _toml_table_to_argv(cfg.get("staging") or {})
    run(
        [
            PY,
            "easycontrol_adapters/colorization/prep.py",
            "--skip_encode",
            "--skip_text",
            *_colorize_prep_paths(base),
            *knobs,
            *list(extra or []),
        ]
    )


def _colorize_preprocess(adapter: str, cfg: dict, base: str, extra) -> None:
    """Colorize preprocess: cache cond latents + color-only text over the staged tree.

    Runs prep.py's encode + color-text stages (``--skip_mangafy`` — mangafy is the
    staging step). The color *target* latents + TE are reused from the shared LoRA
    cache, so no target re-encode. Knobs come from the descriptor's
    ``[preprocess]`` table; pass ``ARGS="--no-skip_mangafy"`` to re-stage inline."""
    knobs = _toml_table_to_argv(cfg.get("preprocess") or {})
    run(
        [
            PY,
            "easycontrol_adapters/colorization/prep.py",
            "--skip_mangafy",
            *_colorize_prep_paths(base),
            *knobs,
            *list(extra or []),
        ]
    )


def _inpaint_prep_paths(base: str) -> list[str]:
    """Slug-derived prep.py path flags so ``name`` reroutes inpaint's trees.

    ``--src`` stays the shared corpus (``post_image_dataset/resized`` — the inpaint
    *targets*); only the masked staging tree + cond/text caches ride the slug,
    matching the blueprint's ``cond_cache_dir`` / ``text_cache_dir``. The text cache
    holds full captions with shuffle + tag-dropout variants (the cond-latent target
    captions are otherwise reused from the shared LoRA cache). Injected before the
    descriptor knob tables so a ``[staging]``/``[preprocess]`` key (or user
    ``extra``) still wins via argparse last-flag precedence."""
    return [
        "--staging",
        f"{base}/staging",
        "--cond_cache_dir",
        f"{base}/cond",
        "--text_cache_dir",
        f"{base}/text",
    ]


def _inpaint_stage(adapter: str, cfg: dict, base: str, extra) -> None:
    """Inpaint staging: synthesize the gray-masked condition tree.

    Runs only prep.py's mask stage (``--skip_encode --skip_text``) over the shared
    corpus into ``{base}/staging``. Knobs come from the descriptor's ``[staging]``
    table; user ``extra`` argv wins last. The cond-latent + caption-text caching is
    the separate preprocess pass."""
    knobs = _toml_table_to_argv(cfg.get("staging") or {})
    run(
        [
            PY,
            "easycontrol_adapters/inpainting/prep.py",
            "--skip_encode",
            "--skip_text",
            *_inpaint_prep_paths(base),
            *knobs,
            *list(extra or []),
        ]
    )


def _inpaint_preprocess(adapter: str, cfg: dict, base: str, extra) -> None:
    """Inpaint preprocess: cache cond latents over the masked staging tree.

    Runs prep.py's encode + caption-text stages (``--skip_mask`` — masking is the
    staging step). The target latents are reused from the shared LoRA cache (no
    re-encode); the text stage re-caches full captions with shuffle + tag-dropout
    variants into ``{base}/text``. Knobs come from the descriptor's ``[preprocess]``
    table; pass ``ARGS="--no-skip_mask"`` to re-stage inline, or ``--skip_text`` to
    reuse the shared LoRA TE cache instead."""
    knobs = _toml_table_to_argv(cfg.get("preprocess") or {})
    run(
        [
            PY,
            "easycontrol_adapters/inpainting/prep.py",
            "--skip_mask",
            *_inpaint_prep_paths(base),
            *knobs,
            *list(extra or []),
        ]
    )


# Per-adapter materialization bodies (training is generic via _easy_train_extra);
# only `stage` (data gen) + `preprocess` (VAE/TE caching) differ per adapter. Both
# receive ``(adapter, cfg, base, extra)``. ``sanitize`` reuses the near-twin miner
# wholesale (same pipeline, different discriminator tags via its descriptor file).
_EASY_ADAPTERS = {
    "near_twins": {"stage": _near_twins_stage, "preprocess": _near_twins_preprocess},
    "sanitize": {"stage": _near_twins_stage, "preprocess": _near_twins_preprocess},
    "colorize": {"stage": _colorize_stage, "preprocess": _colorize_preprocess},
    "inpaint": {"stage": _inpaint_stage, "preprocess": _inpaint_preprocess},
}


def cmd_easycontrol_preprocess(extra):
    """Full EasyControl preprocess: VAE latents + text-encoder outputs.

    Source: ``easycontrol-dataset/``  Caches: ``post_image_dataset/easycontrol/``.

    ``EASYADAPTER=<adapter>`` instead runs the adapter's descriptor-driven
    preprocess (every knob from the ``[preprocess]`` table of
    ``configs/easycontrol/<adapter>.toml``):
      • ``colorize`` caches the synthetic-manga *condition* latents + color-only
        text over the already-staged tree (mangafy is the separate staging step);
        the color target latents + TE are reused from the LoRA cache.
      • ``near_twins`` resizes + VAE/TE-caches the mined pair tree and symlinks the
        ``cond/`` reference latents.
    """
    adapter = _easyadapter()
    if adapter in _EASY_ADAPTERS:
        cfg, _name, base = _easy_load(adapter)
        _EASY_ADAPTERS[adapter]["preprocess"](adapter, cfg, base, extra)
        return

    src = "easycontrol-dataset"
    dst = "post_image_dataset/easycontrol"
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
        ]
    )
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "--caption_shuffle_variants",
            "4",
            "--caption_tag_dropout_rate",
            "0.1",
        ]
    )


def cmd_easycontrol_staging(extra):
    """Generate an EasyControl adapter's *staging* dataset (no VAE/TE caching).

    The adapter-specific data-generation step that materializes the training/
    condition tree, kept separate from the later ``easycontrol-preprocess`` VAE/TE
    caching pass. Knobs come from the ``[staging]`` table of
    ``configs/easycontrol/<adapter>.toml``; extra CLI args override them.

    ``EASYADAPTER=near_twins`` mines the in-artist near-twin pair tree into
    ``post_image_dataset/easycontrol/near_twins/staging/`` and (re)writes the
    descriptor's blueprint tail, e.g.::

        make easycontrol-staging EASYADAPTER=near_twins \\
            ARGS="--region --artists ama_mitsuki"

    ``EASYADAPTER=colorize`` runs only the mangafy stage of
    ``easycontrol_adapters/colorization/prep.py`` (synthesize the synthetic B&W
    manga condition tree under ``post_image_dataset/easycontrol/colorize/staging/``)
    — cond-latent + color-only-text caching stays in the later
    ``easycontrol-preprocess EASYADAPTER=colorize`` pass (idempotent: it skips the
    already-staged PNGs), e.g.::

        make easycontrol-staging EASYADAPTER=colorize ARGS="--engine cv2 --limit 8"
    """
    adapter = _easyadapter()
    spec = _EASY_ADAPTERS.get(adapter)
    if spec is None or "stage" not in spec:
        raise SystemExit(
            "easycontrol-staging needs a staging-capable EASYADAPTER. "
            f"Known: {sorted(_EASY_ADAPTERS)}.\n"
            "(The default EasyControl reads easycontrol-dataset/ directly.)"
        )
    cfg, _name, base = _easy_load(adapter)
    spec["stage"](adapter, cfg, base, extra)
