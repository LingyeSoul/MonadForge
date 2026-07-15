"""User-owned plans for true staged-resolution training.

A plan describes intent (source, variant, preset, and three stages). This module
derives every writable path, validates the plan, reports cache readiness, and
compiles an immutable full training config. Browser and task callers therefore
only need to exchange a validated profile name instead of arbitrary file paths.
"""

from __future__ import annotations

import copy
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import toml
from PIL import Image

from library.config.io import load_dataset_config_from_base, load_method_preset
from library.datasets.buckets import ALLOWED_TARGET_RES
from library.env import anima_home
from library.training.staged_resolution import build_stage_schedule

PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
CONFIG_ID_RE = re.compile(r"^(?:custom/)?[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
DEFAULT_PROFILE = "default"
DEFAULT_STAGES = (
    {"resolution": 512, "ratio": 20.0, "batch_size": 4, "num_repeats": 1},
    {"resolution": 768, "ratio": 30.0, "batch_size": 2, "num_repeats": 1},
    {"resolution": 1024, "ratio": 50.0, "batch_size": 1, "num_repeats": 1},
)


def _root(root: Path | None = None) -> Path:
    return (root or anima_home()).resolve()


def normalize_profile_name(name: str) -> str:
    value = str(name or "").strip()
    if not PROFILE_NAME_RE.fullmatch(value):
        raise ValueError(
            "profile name must be 1-64 ASCII letters, digits, '_' or '-', "
            "and start with a letter or digit"
        )
    return value


def _validate_config_id(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not CONFIG_ID_RE.fullmatch(text):
        raise ValueError(f"{field} contains unsupported characters")
    return text


def profiles_dir(root: Path | None = None) -> Path:
    return _root(root) / "configs" / "custom" / "staged-resolution"


def profile_path(name: str, root: Path | None = None) -> Path:
    return profiles_dir(root) / f"{normalize_profile_name(name)}.toml"


def runtime_path(name: str, root: Path | None = None) -> Path:
    return profiles_dir(root) / "runtime" / f"{normalize_profile_name(name)}.toml"


def manifest_path(name: str, root: Path | None = None) -> Path:
    return profiles_dir(root) / "manifests" / f"{normalize_profile_name(name)}.toml"


def derived_profile_dir(name: str, root: Path | None = None) -> Path:
    return _root(root) / "post_image_dataset" / "staged" / normalize_profile_name(name)


def default_plan() -> dict[str, Any]:
    return {
        "version": 1,
        "method": "lora",
        "variant": "lora",
        "preset": "default",
        "source_image_dir": "image_dataset",
        "max_train_steps": 6000,
        "stages": [dict(stage) for stage in DEFAULT_STAGES],
    }


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("staged-resolution plan must be a table")

    normalized = {
        "version": 1,
        "method": _validate_config_id(plan.get("method", "lora"), "method"),
        "variant": _validate_config_id(plan.get("variant", "lora"), "variant"),
        "preset": _validate_config_id(plan.get("preset", "default"), "preset"),
        "source_image_dir": str(plan.get("source_image_dir") or "").strip(),
    }
    source = normalized["source_image_dir"]
    if not source or len(source) > 1024 or "\x00" in source:
        raise ValueError("source_image_dir must be a non-empty path")

    try:
        steps = int(plan.get("max_train_steps", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_train_steps must be an integer") from exc
    if steps < 3:
        raise ValueError("max_train_steps must be at least 3")
    normalized["max_train_steps"] = steps

    raw_stages = plan.get("stages")
    if not isinstance(raw_stages, list) or len(raw_stages) != 3:
        raise ValueError("a staged-resolution plan requires exactly three stages")

    stages: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, dict):
            raise ValueError(f"stage {index + 1} must be a table")
        try:
            resolution = int(raw.get("resolution"))
            ratio = float(raw.get("ratio"))
            batch_size = int(raw.get("batch_size"))
            num_repeats = int(raw.get("num_repeats", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"stage {index + 1} contains an invalid number") from exc
        if resolution not in ALLOWED_TARGET_RES:
            raise ValueError(
                f"stage {index + 1} resolution must be one of "
                f"{list(ALLOWED_TARGET_RES)}"
            )
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"stage {index + 1} ratio must be positive")
        if batch_size < 1 or batch_size > 128:
            raise ValueError(f"stage {index + 1} batch_size must be between 1 and 128")
        if num_repeats < 1 or num_repeats > 10000:
            raise ValueError(
                f"stage {index + 1} num_repeats must be between 1 and 10000"
            )
        stages.append(
            {
                "resolution": resolution,
                "ratio": ratio,
                "batch_size": batch_size,
                "num_repeats": num_repeats,
            }
        )

    resolutions = [stage["resolution"] for stage in stages]
    if resolutions != sorted(resolutions) or len(set(resolutions)) != 3:
        raise ValueError("stage resolutions must be unique and strictly increasing")
    ratio_total = sum(stage["ratio"] for stage in stages)
    if not math.isclose(ratio_total, 100.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"stage ratios must sum to 100, got {ratio_total:g}")

    previous_boundary = 0
    cumulative_ratio = 0.0
    for index, stage in enumerate(stages):
        cumulative_ratio += stage["ratio"]
        boundary = (
            steps
            if index == len(stages) - 1
            else min(
                steps,
                math.ceil(steps * cumulative_ratio / 100.0 - 1e-12),
            )
        )
        if boundary <= previous_boundary:
            raise ValueError(
                f"stage {index + 1} must receive at least one optimizer step; "
                "increase max_train_steps or its ratio"
            )
        previous_boundary = boundary

    normalized["stages"] = stages
    return normalized


def list_profiles(root: Path | None = None) -> list[str]:
    directory = profiles_dir(root)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.toml") if path.is_file())


def load_profile(
    name: str,
    root: Path | None = None,
    *,
    default_if_missing: bool = False,
) -> dict[str, Any]:
    path = profile_path(name, root)
    if not path.is_file():
        if default_if_missing:
            return default_plan()
        raise FileNotFoundError(path)
    return validate_plan(toml.loads(path.read_text(encoding="utf-8")))


def _atomic_write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(toml.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def save_profile(
    name: str, plan: dict[str, Any], root: Path | None = None
) -> dict[str, Any]:
    normalized = validate_plan(plan)
    _atomic_write_toml(profile_path(name, root), normalized)
    return normalized


def _resolve_source(plan: dict[str, Any], root: Path) -> Path:
    source = Path(plan["source_image_dir"]).expanduser()
    return source.resolve() if source.is_absolute() else (root / source).resolve()


def _source_inventory(source: Path) -> list[dict[str, Any]]:
    if not source.is_dir():
        return []
    inventory: list[dict[str, Any]] = []
    for path in sorted(
        (
            item
            for item in source.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTS
        ),
        key=lambda item: item.relative_to(source).as_posix(),
    ):
        stat = path.stat()
        inventory.append(
            {
                "path": path.relative_to(source).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return inventory


def _cache_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": plan["method"],
        "variant": plan["variant"],
        "preset": plan["preset"],
        "source_image_dir": plan["source_image_dir"],
        "resolutions": [stage["resolution"] for stage in plan["stages"]],
    }


def _profile_manifest(
    name: str, plan: dict[str, Any], root: Path | None = None
) -> dict[str, Any]:
    repo = _root(root)
    normalized = validate_plan(plan)
    source = _resolve_source(normalized, repo)
    return {
        "version": 1,
        "profile": normalize_profile_name(name),
        "source_root": str(source),
        "cache_plan": _cache_plan(normalized),
        "source_files": _source_inventory(source),
    }


def profile_manifest_matches(
    name: str, plan: dict[str, Any], root: Path | None = None
) -> bool:
    path = manifest_path(name, root)
    if not path.is_file():
        return False
    try:
        saved = toml.loads(path.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return False
    return saved == _profile_manifest(name, plan, root)


def write_profile_manifest(
    name: str, plan: dict[str, Any], root: Path | None = None
) -> Path:
    path = manifest_path(name, root)
    _atomic_write_toml(path, _profile_manifest(name, plan, root))
    return path


def reset_profile_cache_if_stale(
    name: str, plan: dict[str, Any], root: Path | None = None
) -> bool:
    if profile_manifest_matches(name, plan, root):
        return False
    owned = derived_profile_dir(name, root)
    if owned.is_dir():
        shutil.rmtree(owned)
    return True


def stage_paths(
    name: str, plan: dict[str, Any], root: Path | None = None
) -> list[dict[str, Any]]:
    repo = _root(root)
    slug = normalize_profile_name(name)
    base = repo / "post_image_dataset" / "staged" / slug
    paths: list[dict[str, Any]] = []
    for index, stage in enumerate(plan["stages"]):
        stage_root = base / str(stage["resolution"])
        paths.append(
            {
                "index": index,
                "resolution": stage["resolution"],
                "resized_dir": stage_root / "resized",
                "cache_dir": stage_root / "cache",
            }
        )
    return paths


def _expected_resized_paths(source_files: list[dict[str, Any]]) -> set[str]:
    return {
        Path(entry["path"]).with_suffix(".png").as_posix() for entry in source_files
    }


def _relative_files(path: Path, predicate) -> set[str]:
    if not path.is_dir():
        return set()
    return {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and predicate(item)
    }


def _expected_cache_paths(
    resized_dir: Path, cache_dir: Path, expected_resized: set[str]
) -> tuple[set[str], set[str], bool]:
    latents: set[str] = set()
    text_embeddings: set[str] = set()
    dimensions_valid = True
    for relative in expected_resized:
        image_path = resized_dir / Path(relative)
        if not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except (OSError, ValueError):
            dimensions_valid = False
            continue
        relative_path = Path(relative)
        cache_parent = relative_path.parent
        latents.add(
            (
                cache_parent
                / f"{relative_path.stem}_{width:04d}x{height:04d}_anima.npz"
            ).as_posix()
        )
        text_embeddings.add(
            (cache_parent / f"{relative_path.stem}_anima_te.safetensors").as_posix()
        )
    return latents, text_embeddings, dimensions_valid


def remove_profile_orphans(
    name: str, plan: dict[str, Any], root: Path | None = None
) -> int:
    repo = _root(root)
    normalized = validate_plan(plan)
    source_files = _source_inventory(_resolve_source(normalized, repo))
    expected_resized = _expected_resized_paths(source_files)
    removed = 0
    for paths in stage_paths(name, normalized, repo):
        resized_dir = paths["resized_dir"]
        cache_dir = paths["cache_dir"]
        actual_resized = _relative_files(
            resized_dir, lambda path: path.suffix.lower() in IMAGE_EXTS
        )
        for relative in actual_resized - expected_resized:
            (resized_dir / Path(relative)).unlink()
            removed += 1

        expected_latents, expected_te, _ = _expected_cache_paths(
            resized_dir, cache_dir, expected_resized
        )
        actual_latents = _relative_files(
            cache_dir, lambda path: path.name.endswith("_anima.npz")
        )
        actual_te = _relative_files(
            cache_dir,
            lambda path: path.name.endswith("_anima_te.safetensors"),
        )
        for relative in (actual_latents - expected_latents) | (actual_te - expected_te):
            (cache_dir / Path(relative)).unlink()
            removed += 1
    return removed


def _count_files(path: Path, predicate) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and predicate(item))


def profile_status(
    name: str, plan: dict[str, Any], root: Path | None = None
) -> dict[str, Any]:
    repo = _root(root)
    normalized = validate_plan(plan)
    source = _resolve_source(normalized, repo)
    source_files = _source_inventory(source)
    source_count = len(source_files)
    caption_count = _count_files(source, lambda path: path.suffix.lower() == ".txt")
    expected_resized = _expected_resized_paths(source_files)
    manifest_current = profile_manifest_matches(name, normalized, repo)
    stages: list[dict[str, Any]] = []
    for stage, paths in zip(normalized["stages"], stage_paths(name, normalized, repo)):
        actual_resized = _relative_files(
            paths["resized_dir"], lambda path: path.suffix.lower() in IMAGE_EXTS
        )
        actual_latents = _relative_files(
            paths["cache_dir"], lambda path: path.name.endswith("_anima.npz")
        )
        actual_text_embeddings = _relative_files(
            paths["cache_dir"],
            lambda path: path.name.endswith("_anima_te.safetensors"),
        )
        expected_latents, expected_text_embeddings, dimensions_valid = (
            _expected_cache_paths(
                paths["resized_dir"], paths["cache_dir"], expected_resized
            )
        )
        captions_current = True
        for entry in source_files:
            source_relative = Path(entry["path"])
            caption_path = (source / source_relative).with_suffix(".txt")
            if not caption_path.is_file():
                continue
            te_relative = source_relative.with_suffix("").with_name(
                source_relative.stem + "_anima_te.safetensors"
            )
            te_path = paths["cache_dir"] / te_relative
            if (
                not te_path.is_file()
                or te_path.stat().st_mtime_ns < caption_path.stat().st_mtime_ns
            ):
                captions_current = False
                break
        ready = (
            source_count > 0
            and manifest_current
            and len(expected_resized) == source_count
            and actual_resized == expected_resized
            and dimensions_valid
            and actual_latents == expected_latents
            and actual_text_embeddings == expected_text_embeddings
            and captions_current
        )
        stages.append(
            {
                **stage,
                "resized_dir": paths["resized_dir"].relative_to(repo).as_posix(),
                "cache_dir": paths["cache_dir"].relative_to(repo).as_posix(),
                "resized": len(actual_resized),
                "latents": len(actual_latents),
                "text_embeddings": len(actual_text_embeddings),
                "ready": ready,
            }
        )
    return {
        "profile": normalize_profile_name(name),
        "source_image_dir": str(source),
        "source_exists": source.is_dir(),
        "source_images": source_count,
        "captions": caption_count,
        "stages": stages,
        "all_ready": bool(stages) and all(stage["ready"] for stage in stages),
    }


def _variant_stem(variant: str) -> str:
    return variant.split("/", 1)[-1]


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def compile_runtime_config(
    name: str, plan: dict[str, Any], root: Path | None = None
) -> Path:
    repo = _root(root)
    normalized = validate_plan(plan)
    stem = _variant_stem(normalized["variant"])
    configs_dir = str(repo / "configs")
    merged = load_method_preset(
        stem,
        normalized["preset"],
        configs_dir=configs_dir,
        methods_subdir="gui-methods",
        # Match the normal GUI training chain. The process-global schema is
        # populated lazily and intentionally does not enumerate every model
        # path key, so strict mode becomes order-dependent after ConfigEditor
        # has initialized it.
        strict=False,
    )
    dataset_config = merged.get("dataset_config")
    if dataset_config:
        config_file = Path(str(dataset_config)).expanduser()
        if not config_file.is_absolute():
            config_file = repo / config_file
        candidate = (
            config_file
            if config_file.suffix.lower() == ".toml"
            else config_file.with_suffix(".toml")
        )
        if not candidate.is_file():
            raise ValueError(f"dataset_config not found: {candidate}")
        blueprint = load_dataset_config_from_base(
            configs_dir,
            overrides=merged,
            config_file=str(candidate),
        )
    else:
        blueprint = load_dataset_config_from_base(
            configs_dir,
            overrides=merged,
            method=stem,
            methods_subdir="gui-methods",
        )
    if not blueprint or not blueprint.get("datasets"):
        raise ValueError("the selected variant has no dataset blueprint")
    if len(blueprint["datasets"]) != 1:
        raise ValueError(
            "staged-resolution requires a dataset blueprint with exactly one dataset"
        )
    template_dataset = blueprint["datasets"][0]
    template_subsets = template_dataset.get("subsets") or []
    if len(template_subsets) != 1:
        raise ValueError(
            "staged-resolution requires a dataset blueprint with exactly one subset"
        )
    if _contains_key(blueprint, "conditioning_data_dir"):
        raise ValueError(
            "staged-resolution does not support dataset blueprints with "
            "conditioning_data_dir"
        )

    runtime = {key: value for key, value in merged.items() if value is not None}
    for key in (
        "dataset_config",
        "max_train_epochs",
        "target_res",
        "staged_resolution",
        "staged_resolution_ratios",
        "staged_resolution_base_sides",
        "stage_schedule",
        "stage_schedule_enabled",
    ):
        runtime.pop(key, None)
    runtime["max_train_steps"] = normalized["max_train_steps"]
    runtime["stage_schedule_enabled"] = True
    runtime["stage_schedule"] = build_stage_schedule(
        [stage["ratio"] for stage in normalized["stages"]],
        [stage["resolution"] for stage in normalized["stages"]],
    )

    rows: list[dict[str, Any]] = []
    for stage, paths in zip(normalized["stages"], stage_paths(name, normalized, repo)):
        dataset = copy.deepcopy(template_dataset)
        subset = copy.deepcopy(template_subsets[0])
        dataset["batch_size"] = stage["batch_size"]
        subset["image_dir"] = paths["resized_dir"].relative_to(repo).as_posix()
        subset["cache_dir"] = paths["cache_dir"].relative_to(repo).as_posix()
        subset["num_repeats"] = stage["num_repeats"]
        dataset["subsets"] = [subset]
        rows.append(dataset)
    runtime["datasets"] = rows
    runtime["general"] = copy.deepcopy(blueprint.get("general") or {})

    output = runtime_path(name, repo)
    _atomic_write_toml(output, runtime)
    return output


def require_ready(name: str, plan: dict[str, Any], root: Path | None = None) -> None:
    status = profile_status(name, plan, root)
    if not status["source_exists"]:
        raise ValueError(
            f"source image directory not found: {status['source_image_dir']}"
        )
    if status["source_images"] <= 0:
        raise ValueError("source image directory contains no supported images")
    if not status["all_ready"]:
        missing = [
            str(stage["resolution"]) for stage in status["stages"] if not stage["ready"]
        ]
        raise ValueError(
            "staged-resolution caches are incomplete for tiers: " + ", ".join(missing)
        )
