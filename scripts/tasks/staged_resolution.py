"""Task-runner commands for true staged-resolution preprocessing and training."""

from __future__ import annotations

import toml

from library.training.staged_resolution_plan import (
    compile_runtime_config,
    load_profile,
    normalize_profile_name,
    profile_status,
    remove_profile_orphans,
    require_ready,
    reset_profile_cache_if_stale,
    stage_paths,
    write_profile_manifest,
)

from ._common import PY, accelerate_launch, run
from .preprocess import caption_variant_args


def _profile_arg(
    extra: list[str], *, allow_training_flags: bool
) -> tuple[str, list[str]]:
    args = list(extra or [])
    if not args or args[0].startswith("-"):
        raise SystemExit("staged-resolution command requires a profile name")
    name = normalize_profile_name(args.pop(0))
    if not allow_training_flags and args:
        raise SystemExit("staged-preprocess accepts only a profile name")

    forwarded: list[str] = []
    allowed = {"--progress_jsonl", "--sample_dir"}
    while args:
        flag = args.pop(0)
        if flag not in allowed or not args:
            raise SystemExit(f"unsupported staged-train argument: {flag}")
        forwarded.extend((flag, args.pop(0)))
    return name, forwarded


def cmd_staged_preprocess(extra):
    """Build one resized/cache tree per stage in a saved WebUI profile."""
    name, _ = _profile_arg(extra, allow_training_flags=False)
    plan = load_profile(name)
    runtime_path = compile_runtime_config(name, plan)
    runtime = toml.loads(runtime_path.read_text(encoding="utf-8"))
    status = profile_status(name, plan)
    if not status["source_exists"]:
        raise SystemExit(
            f"source image directory not found: {status['source_image_dir']}"
        )
    if status["source_images"] <= 0:
        raise SystemExit("source image directory contains no supported images")

    source = status["source_image_dir"]
    vae = runtime.get("vae", "models/vae/qwen_image_vae.safetensors")
    qwen3 = runtime.get("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors")
    dit = runtime.get(
        "pretrained_model_name_or_path",
        "models/diffusion_models/anima-base-v1.0.safetensors",
    )
    reset_cache = reset_profile_cache_if_stale(name, plan)
    overwrite_args = ["--overwrite"] if reset_cache else []

    for stage, paths in zip(plan["stages"], stage_paths(name, plan)):
        edge = str(stage["resolution"])
        resized = str(paths["resized_dir"])
        print(f"[staged-resolution] preparing {edge}px tier")
        run(
            [
                PY,
                "scripts/preprocess/resize_images.py",
                "--src",
                source,
                "--dst",
                resized,
                "--no_copy_captions",
                "--min_pixels",
                "0",
                "--bucket_reso_steps",
                "64",
                "--recursive",
                "--target_res",
                edge,
                *overwrite_args,
            ]
        )

    remove_profile_orphans(name, plan)

    for stage, paths in zip(plan["stages"], stage_paths(name, plan)):
        resized = str(paths["resized_dir"])
        cache = str(paths["cache_dir"])
        run(
            [
                PY,
                "scripts/preprocess/cache_latents.py",
                "--dir",
                resized,
                "--cache_dir",
                cache,
                "--vae",
                str(vae),
                "--batch_size",
                "4",
                "--chunk_size",
                "64",
                "--recursive",
                *overwrite_args,
            ]
        )
        run(
            [
                PY,
                "scripts/preprocess/cache_text_embeddings.py",
                "--dir",
                source,
                "--cache_dir",
                cache,
                "--match_images_from",
                resized,
                "--qwen3",
                str(qwen3),
                "--dit",
                str(dit),
                *caption_variant_args(),
                "--recursive",
                "--min_pixels",
                "0",
                *overwrite_args,
            ]
        )

    write_profile_manifest(name, plan)
    final = profile_status(name, plan)
    if not final["all_ready"]:
        raise SystemExit(
            "staged-resolution preprocessing finished with incomplete caches"
        )
    print("[staged-resolution] all three tiers are ready")


def cmd_staged_train(extra):
    """Train from a validated, generated full config for a saved profile."""
    name, forwarded = _profile_arg(extra, allow_training_flags=True)
    plan = load_profile(name)
    require_ready(name, plan)
    config_path = compile_runtime_config(name, plan)
    print(f"[staged-resolution] training profile={name} config={config_path}")
    accelerate_launch("--config_file", str(config_path), *forwarded)
