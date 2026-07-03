"""Preprocessing settings and status service — no Qt dependencies.

Reads/writes ``configs/custom/sam_mask.yaml`` and ``configs/webui_settings.json``,
and counts preprocess caches for the status dashboard. The SAM yaml lives under
``configs/custom/`` so WebUI edits never dirty the git-tracked repo copy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import toml
import yaml

from webui.services.config_service import ROOT, get_path_overrides

# ── Paths ─────────────────────────────────────────────────────────

CONFIGS_DIR = ROOT / "configs"
CUSTOM_DIR = CONFIGS_DIR / "custom"
SAM_YAML = CUSTOM_DIR / "sam_mask.yaml"
SETTINGS_FILE = CONFIGS_DIR / "webui_settings.json"
# target_res lives here (owned by this file, preserved across `make update`)
# — see library/config/io.py::load_path_overrides for the ownership contract.
# Both preprocess.toml and sam_mask.yaml sit under custom/ so WebUI edits
# never dirty the git-tracked repo copies at configs/{preprocess.toml,
# sam_mask.yaml}. configs/custom/ is gitignored (see .gitignore).
PREPROCESS_TOML = CUSTOM_DIR / "preprocess.toml"

# Allowed free-fit tier edge sizes (must match library/datasets/buckets.py).
ALLOWED_TARGET_RES = (512, 768, 896, 1024, 1280, 1536)


def _resolve(p: str) -> Path:
    """Resolve a possibly-relative path against ROOT."""
    pp = Path(p)
    return pp if pp.is_absolute() else ROOT / pp


def _get_paths(
    variant: str | None = None, preset: str | None = None
) -> dict[str, Path]:
    """Return resolved dataset paths from the config chain."""
    paths = get_path_overrides(preset=preset or "default", variant=variant)
    return {
        "resized": _resolve(paths["resized_image_dir"]),
        "masks": _resolve(paths["resized_image_dir"]).parent / "masks",
        "cache": _resolve(paths["lora_cache_dir"]),
        "cond_resized": _resolve(paths["conditioning_resized_dir"]),
    }


def get_paths(variant: str | None = None, preset: str | None = None) -> dict[str, str]:
    """Return the raw resolved path strings for the frontend."""
    paths = get_path_overrides(preset=preset or "default", variant=variant)
    return {
        "source_image_dir": paths["source_image_dir"],
        "resized_image_dir": paths["resized_image_dir"],
        "lora_cache_dir": paths["lora_cache_dir"],
        "conditioning_data_dir": paths["conditioning_data_dir"],
        "conditioning_resized_dir": paths["conditioning_resized_dir"],
    }


def save_path_overrides(variant: str, data: dict[str, str]) -> dict[str, str]:
    """Persist path overrides to the variant TOML and return updated paths."""
    from webui.services.config_service import save_variant_config

    allowed = {"source_image_dir", "resized_image_dir", "lora_cache_dir", "conditioning_data_dir", "conditioning_resized_dir"}
    filtered = {k: v for k, v in data.items() if k in allowed and v}
    if filtered:
        save_variant_config(variant, filtered)
    return get_paths(variant=variant)


# ── Cache-file suffixes ─────────────────────────────────────────

_LATENT_SUFFIX = "_anima.npz"
_TE_SUFFIX = "_anima_te.safetensors"
_PE_SUFFIX = "_anima_pe.safetensors"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ── Defaults ─────────────────────────────────────────────────────

DEFAULTS = {
    "sam_prompts": ["speech bubble", "text bubble"],
    "sam_threshold": 0.5,
    "sam_dilate": 5,
    "run_sam_mask": True,
    "run_mit_mask": True,
    "caption_shuffle_variants": 4,
    "caption_tag_dropout_rate": 0.1,
    "mit_text_threshold": 0.8,
    "mit_dilate": 5,
    # free-fit tier edges that preprocess actually resizes into. The legacy
    # single-number `resize_resolution` was vestigial under free-fit (the
    # `--resolution` it fed is dropped in library/preprocess/images.py), so it
    # never matched the real output. This list is the value the resize step
    # consumes (saved to configs/preprocess.toml).
    "target_res": [1024],
}


# ── YAML dumper that indents list items ──────────────────────────


class _IndentedListDumper(yaml.SafeDumper):
    """SafeDumper that indents list items under mapping keys.

    Matches the canonical sam_mask.yaml formatting (2-space indent on the dash).
    """


def _increase_indent(self, flow=False, indentless=False):
    return super(_IndentedListDumper, self).increase_indent(flow, indentless=False)


_IndentedListDumper.increase_indent = _increase_indent  # type: ignore[assignment]


# ── Settings CRUD ────────────────────────────────────────────────


def _load_sam() -> dict:
    if not SAM_YAML.exists():
        return {}
    try:
        return yaml.safe_load(SAM_YAML.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _load_gui_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _normalize_target_res(raw) -> list[int]:
    """Coerce a config/cli value into a sorted list of valid tier edges.

    Accepts the forms the config chain can produce: an int list
    (``[1024, 896]``), a single int, or a space/comma string. Drops anything
    not in ALLOWED_TARGET_RES. Always returns at least ``[1024]`` so preprocess
    never gets an empty tier set (which would fall back to the same default).
    """
    if raw is None:
        return list(DEFAULTS["target_res"])
    if isinstance(raw, int):
        edges = [raw]
    elif isinstance(raw, str):
        edges = [p for p in raw.replace(",", " ").split() if p.strip()]
    elif isinstance(raw, (list, tuple)):
        edges = list(raw)
    else:
        return list(DEFAULTS["target_res"])
    try:
        edges = [int(e) for e in edges]
    except (TypeError, ValueError):
        return list(DEFAULTS["target_res"])
    edges = sorted({e for e in edges if e in ALLOWED_TARGET_RES})
    return edges or [1024]


def get_target_res() -> list[int]:
    """The active free-fit tier edges from the config chain (preprocess.toml
    → base → preset → method).

    This is the value the resize step actually consumes — what the user should
    see/edit, as opposed to the old (vestigial) ``resize_resolution`` scalar.

    Reads ``load_path_overrides`` directly (NOT ``config_service.get_path_overrides``):
    the latter only projects the five dataset-path keys, so it would drop
    ``target_res`` and silently regress to the default.
    """
    from library.config.io import load_path_overrides

    overrides = load_path_overrides(
        preset="default",
        method=os.environ.get("METHOD") or None,
        methods_subdir="gui-methods" if os.environ.get("METHOD") else "methods",
    )
    return _normalize_target_res(overrides.get("target_res"))


def save_target_res(edges: list[int]) -> list[int]:
    """Persist the free-fit tier set to ``configs/custom/preprocess.toml``.

    Round-trips the file so the other user-owned knobs (freefit_max_ratio,
    caption_*, min_pixels, …) are preserved. ``preprocess.toml`` is the
    canonical home for target_res — a stray copy in base.toml is deliberately
    ignored at read time (library/config/io.py), so writing here wins.

    A corrupt file is never silently clobbered: a ``TomlDecodeError`` /
    ``OSError`` propagates so the API layer reports the failure instead of
    wiping the other user-owned keys by re-writing from an empty dict. The
    write is atomic (temp file + ``os.replace``) so a crash mid-write can't
    leave the truncated file that would itself trigger that path on the next
    save.
    """
    normalized = _normalize_target_res(edges)
    if PREPROCESS_TOML.exists():
        # Corrupt file → raise; do not fall back to {} and clobber the other
        # user-owned keys the docstring promises to preserve.
        data = toml.loads(PREPROCESS_TOML.read_text(encoding="utf-8"))
    else:
        data = {}
    data["target_res"] = normalized
    PREPROCESS_TOML.parent.mkdir(parents=True, exist_ok=True)
    tmp = PREPROCESS_TOML.parent / (PREPROCESS_TOML.name + ".tmp")
    tmp.write_text(toml.dumps(data), encoding="utf-8")
    os.replace(tmp, PREPROCESS_TOML)
    return normalized


def get_settings() -> dict:
    """Read both config files and return a unified settings dict."""
    sam = _load_sam()
    gui = _load_gui_settings()
    return {
        "sam": {
            "prompts": sam.get("prompts", DEFAULTS["sam_prompts"]),
            "threshold": sam.get("threshold", DEFAULTS["sam_threshold"]),
            "dilate": sam.get("dilate", DEFAULTS["sam_dilate"]),
        },
        "run_sam_mask": gui.get("run_sam_mask", DEFAULTS["run_sam_mask"]),
        "run_mit_mask": gui.get("run_mit_mask", DEFAULTS["run_mit_mask"]),
        "caption_shuffle_variants": gui.get(
            "caption_shuffle_variants", DEFAULTS["caption_shuffle_variants"]
        ),
        "caption_tag_dropout_rate": gui.get(
            "caption_tag_dropout_rate", DEFAULTS["caption_tag_dropout_rate"]
        ),
        "mit_text_threshold": gui.get(
            "mit_text_threshold", DEFAULTS["mit_text_threshold"]
        ),
        "mit_dilate": gui.get("mit_dilate", DEFAULTS["mit_dilate"]),
        # From the config chain (configs/custom/preprocess.toml → preset → method),
        # not webui_settings.json — it's the value resize actually uses.
        "target_res": get_target_res(),
    }


def save_settings(data: dict) -> dict:
    """Write settings back to the config files.

    Returns the saved settings (round-tripped through get_settings).
    """
    # ── SAM yaml ──
    sam = data.get("sam", {})
    sam_yaml = {
        "prompts": sam.get("prompts", DEFAULTS["sam_prompts"]),
        "threshold": sam.get("threshold", DEFAULTS["sam_threshold"]),
        "dilate": sam.get("dilate", DEFAULTS["sam_dilate"]),
    }
    SAM_YAML.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(
        sam_yaml,
        Dumper=_IndentedListDumper,
        default_flow_style=False,
        sort_keys=False,
    )
    # Match canonical layout: blank line between prompts list and threshold
    text = text.replace("\nthreshold:", "\n\nthreshold:", 1)
    SAM_YAML.write_text(text, encoding="utf-8")

    # ── GUI settings json ──
    gui = _load_gui_settings()
    for key in (
        "run_sam_mask",
        "run_mit_mask",
        "caption_shuffle_variants",
        "caption_tag_dropout_rate",
        "mit_text_threshold",
        "mit_dilate",
    ):
        if key in data:
            gui[key] = data[key]
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(gui, indent=2), encoding="utf-8")

    # ── target_res → configs/custom/preprocess.toml ──
    if "target_res" in data:
        save_target_res(data["target_res"])

    return get_settings()


# ── Status / cache counting ──────────────────────────────────────


def _count_cache_files(cache_dir: Path, fallback: Path | None = None) -> dict[str, int]:
    """Count latent / TE / PE cache sidecars under *cache_dir*."""
    d = fallback or cache_dir
    out = {"latents": 0, "te": 0, "pe": 0}
    if not d.is_dir():
        return out
    for p in d.rglob("*"):
        if not p.is_file():
            continue
        n = p.name
        if n.endswith(_TE_SUFFIX):
            out["te"] += 1
        elif n.endswith(_PE_SUFFIX):
            out["pe"] += 1
        elif n.endswith(_LATENT_SUFFIX):
            out["latents"] += 1
    return out


def count_caches(cache_dir: Path | None = None) -> dict[str, int]:
    """Count latent / TE / PE cache sidecars under *cache_dir*."""
    d = cache_dir or _get_paths()["cache"]
    return _count_cache_files(d)


def adapter_stats(source_dir: str) -> dict:
    """Return dataset statistics for an adapter's source directory.

    Counts source images, .txt captions, and cache coverage in the default
    lora cache directory matching the source stems.
    """
    src = _resolve(source_dir)
    cache_dir = _resolve(get_path_overrides().get("lora_cache_dir", "post_image_dataset/lora"))

    source_count = 0
    caption_count = 0
    stems: set[str] = set()
    if src.is_dir():
        for p in src.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() in IMAGE_EXTS:
                source_count += 1
                stems.add(p.stem)
            elif p.suffix == ".txt":
                caption_count += 1

    cache = {"latents": 0, "te": 0, "pe": 0}
    if cache_dir.is_dir() and stems:
        for p in cache_dir.iterdir():
            if not p.is_file():
                continue
            n = p.name
            # Check if this cache file belongs to any source stem
            matched = False
            for stem in stems:
                if n.startswith(stem + "_") or n.startswith(stem + "."):
                    matched = True
                    break
            if not matched:
                continue
            if n.endswith(_TE_SUFFIX):
                cache["te"] += 1
            elif n.endswith(_PE_SUFFIX):
                cache["pe"] += 1
            elif n.endswith(_LATENT_SUFFIX):
                cache["latents"] += 1

    return {
        "source_count": source_count,
        "caption_count": caption_count,
        "cache": cache,
    }


def _count_images(d: Path) -> int:
    """Count image files under *d*."""
    if not d.is_dir():
        return 0
    return sum(
        1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def count_resized() -> int:
    """Count resized images under the configured ``resized_image_dir``."""
    return _count_images(_get_paths()["resized"])


def _count_mask_files(d: Path) -> int:
    """Count merged mask files under *d*."""
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.rglob("*_mask.png"))


def count_masks() -> int:
    """Count merged mask files under the configured masks directory."""
    return _count_mask_files(_get_paths()["masks"])


def get_status(
    cache_dir: Path | None = None,
    variant: str | None = None,
    preset: str | None = None,
) -> dict:
    """Return a snapshot of preprocess pipeline counts."""
    p = _get_paths(variant=variant, preset=preset)
    return {
        "resized": _count_images(p["resized"]),
        "masks": _count_mask_files(p["masks"]),
        "cache": _count_cache_files(p["cache"], cache_dir),
        "cond_resized": _count_images(p["cond_resized"]),
    }
