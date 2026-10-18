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
from webui.services.config_storage import (
    CONFIG_WRITE_LOCK,
    atomic_write_many,
    atomic_write_text,
    read_json_object,
)
from library.preprocess.runs import run_from_manifest

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


def get_paths(
    variant: str | None = None,
    preset: str | None = None,
    manifest: str | None = None,
) -> dict[str, str]:
    """Return the raw resolved path strings for the frontend."""
    if manifest:
        run = run_from_manifest(manifest)
        return {
            "source_image_dir": str(run.source_dir),
            "resized_image_dir": str(run.resized_image_dir),
            "lora_cache_dir": str(run.lora_cache_dir),
            "conditioning_data_dir": str(run.conditioning_data_dir),
            "conditioning_resized_dir": str(run.conditioning_resized_dir),
        }
    paths = get_path_overrides(preset=preset or "default", variant=variant)
    return {
        "source_image_dir": paths["source_image_dir"],
        "resized_image_dir": paths["resized_image_dir"],
        "lora_cache_dir": paths["lora_cache_dir"],
        "conditioning_data_dir": paths["conditioning_data_dir"],
        "conditioning_resized_dir": paths["conditioning_resized_dir"],
    }


def save_path_overrides(variant: str, data: dict[str, str]) -> dict[str, str]:
    """Persist path overrides to the variant TOML and return updated paths.

    Conditioning keys (``conditioning_data_dir`` / ``conditioning_resized_dir``)
    are only persisted for conditioning-capable families (controlnet /
    easycontrol). For any other variant they're dropped: a stale default value
    landing on a plain LoRA variant would otherwise get merged into the subset's
    ``cond_cache_dir`` by the BlueprintGenerator and turn a normal run into a
    cond≠target task that crashes on the empty default dir.
    """
    from webui.services.config_service import (
        CONDITIONING_CAPABLE_FAMILIES,
        save_variant_config,
        variant_metadata,
    )

    allowed = {
        "source_image_dir",
        "resized_image_dir",
        "lora_cache_dir",
        "conditioning_data_dir",
        "conditioning_resized_dir",
    }
    filtered = {k: v for k, v in data.items() if k in allowed and v}
    # Gate conditioning keys by variant family — only conditioning-capable
    # families (controlnet / easycontrol) may carry them. Any other family
    # would have BlueprintGenerator map conditioning_data_dir → cond_cache_dir
    # onto the subset and crash the run. See CONDITIONING_CAPABLE_FAMILIES.
    meta = variant_metadata(variant)
    family = meta.get("family") if isinstance(meta, dict) else None
    if family not in CONDITIONING_CAPABLE_FAMILIES:
        filtered.pop("conditioning_data_dir", None)
        filtered.pop("conditioning_resized_dir", None)
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
    "mit_ctd_gate": True,
    # free-fit tier edges that preprocess actually resizes into. The legacy
    # single-number `resize_resolution` was vestigial under free-fit (the
    # `--resolution` it fed is dropped in library/preprocess/images.py), so it
    # never matched the real output. This list is the value the resize step
    # consumes (saved to configs/preprocess.toml).
    "target_res": [1024],
    "multires_per_image": False,
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


def get_multires_per_image() -> bool:
    from library.config.io import load_path_overrides

    overrides = load_path_overrides(
        preset="default",
        method=os.environ.get("METHOD") or None,
        methods_subdir="gui-methods" if os.environ.get("METHOD") else "methods",
    )
    raw = overrides.get("multires_per_image", DEFAULTS["multires_per_image"])
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


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
    with CONFIG_WRITE_LOCK:
        data = (
            toml.loads(PREPROCESS_TOML.read_text(encoding="utf-8"))
            if PREPROCESS_TOML.exists()
            else {}
        )
        data["target_res"] = normalized
        atomic_write_text(PREPROCESS_TOML, toml.dumps(data))
    return normalized


def save_multires_per_image(enabled: bool) -> bool:
    value = bool(enabled)
    with CONFIG_WRITE_LOCK:
        data = (
            toml.loads(PREPROCESS_TOML.read_text(encoding="utf-8"))
            if PREPROCESS_TOML.exists()
            else {}
        )
        data["multires_per_image"] = value
        atomic_write_text(PREPROCESS_TOML, toml.dumps(data))
    return value


def get_settings() -> dict:
    """Read both config files and return a unified settings dict."""
    sam = _load_sam()
    gui = _load_gui_settings()
    target_res = get_target_res()
    multires_per_image = get_multires_per_image() and len(target_res) >= 2
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
        "mit_ctd_gate": gui.get("mit_ctd_gate", DEFAULTS["mit_ctd_gate"]),
        # From the config chain (configs/custom/preprocess.toml → preset → method),
        # not webui_settings.json — it's the value resize actually uses.
        "target_res": target_res,
        "multires_per_image": multires_per_image,
    }


def save_settings(data: dict) -> dict:
    """Write settings back to the config files.

    Returns the saved settings (round-tripped through get_settings).
    """
    with CONFIG_WRITE_LOCK:
        return _save_settings(data)


def _save_settings(data: dict) -> dict:
    # Resolve and validate all settings before modifying any config file.
    current_target_res = get_target_res()
    current_multires = get_multires_per_image()
    target_res = (
        _normalize_target_res(data["target_res"])
        if "target_res" in data
        else current_target_res
    )
    multires_per_image = (
        bool(data["multires_per_image"])
        if "multires_per_image" in data
        else current_multires
    )
    if multires_per_image and len(target_res) < 2:
        raise ValueError("multires_per_image requires at least two target_res tiers")

    sam = data.get("sam", {})
    sam_yaml = (
        yaml.safe_load(SAM_YAML.read_text(encoding="utf-8"))
        if SAM_YAML.exists()
        else {}
    )
    if sam_yaml is None:
        sam_yaml = {}
    if not isinstance(sam_yaml, dict):
        raise ValueError("SAM settings must be a YAML object")
    for key in ("prompts", "threshold", "dilate"):
        sam_yaml[key] = sam.get(key, sam_yaml.get(key, DEFAULTS[f"sam_{key}"]))
    text = yaml.dump(
        sam_yaml,
        Dumper=_IndentedListDumper,
        default_flow_style=False,
        sort_keys=False,
    )
    # Match canonical layout: blank line between prompts list and threshold
    text = text.replace("\nthreshold:", "\n\nthreshold:", 1)
    # ── GUI settings json ──
    gui = read_json_object(SETTINGS_FILE)
    for key in (
        "run_sam_mask",
        "run_mit_mask",
        "caption_shuffle_variants",
        "caption_tag_dropout_rate",
        "mit_text_threshold",
        "mit_dilate",
        "mit_ctd_gate",
    ):
        if key in data:
            gui[key] = data[key]
    contents = {SAM_YAML: text, SETTINGS_FILE: json.dumps(gui, indent=2)}
    if "target_res" in data or "multires_per_image" in data:
        preprocess = (
            toml.loads(PREPROCESS_TOML.read_text(encoding="utf-8"))
            if PREPROCESS_TOML.exists()
            else {}
        )
        if "target_res" in data:
            preprocess["target_res"] = target_res
        if "multires_per_image" in data:
            preprocess["multires_per_image"] = multires_per_image
        contents[PREPROCESS_TOML] = toml.dumps(preprocess)
    atomic_write_many(contents)
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


def adapter_stats(source_dir: str) -> dict:
    """Return dataset statistics for an adapter's source directory.

    Counts source images, .txt captions, and cache coverage in the default
    lora cache directory matching the source stems.
    """
    src = _resolve(source_dir)
    cache_dir = _resolve(
        get_path_overrides().get("lora_cache_dir", "post_image_dataset/lora")
    )

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
                if n.startswith((stem + "_", stem + ".")):
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


def _count_mask_files(d: Path) -> int:
    """Count merged mask files under *d*."""
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.rglob("*_mask.png"))


def get_status(
    cache_dir: Path | None = None,
    variant: str | None = None,
    preset: str | None = None,
    manifest: str | None = None,
) -> dict:
    """Return a snapshot of preprocess pipeline counts."""
    if manifest:
        info = get_run_status(manifest)
        artifacts = info.get("artifacts") or {}
        return {
            "resized": int(artifacts.get("resized", 0)),
            "masks": int(artifacts.get("masks", 0)),
            "cache": {
                "latents": int(artifacts.get("latents", artifacts.get("lora", 0))),
                "te": int(artifacts.get("te", 0)),
                "pe": int(artifacts.get("pe", 0)),
            },
            "cond_resized": int(artifacts.get("conditioning", 0)),
            "preprocess_run": info,
        }
    p = _get_paths(variant=variant, preset=preset)
    return {
        "resized": _count_images(p["resized"]),
        "masks": _count_mask_files(p["masks"]),
        "cache": _count_cache_files(p["cache"], cache_dir),
        "cond_resized": _count_images(p["cond_resized"]),
    }


def _run_roots(variant: str | None = None, preset: str | None = None) -> list[Path]:
    """Resolve canonical run roots independently of training context.

    ``variant`` and ``preset`` remain accepted for API compatibility with older
    WebUI clients, but they must not affect discovery of a preprocessing run.
    A manifest is a source/config identity, not a training-method identity. The
    global path layer is still consulted so installations that moved the
    dataset root continue to be discoverable.
    """

    # Deliberately ignore the method/preset arguments here. They are only valid
    # for legacy fixed-directory fallback in ``get_status``/``get_paths``.
    paths = get_path_overrides(preset="default", variant=None)
    candidates: list[Path] = []
    configured_root = os.environ.get("POST_IMAGE_DATASET")
    if configured_root:
        candidates.append(_resolve(configured_root))
    # ``get_path_overrides`` intentionally exposes only dataset paths. Derive a
    # custom run root when a caller has already configured one of those paths
    # below ``.../runs/<source>/<config>/...``; otherwise retain the default.
    for key in ("resized_image_dir", "lora_cache_dir", "conditioning_resized_dir"):
        value = paths.get(key)
        if not value:
            continue
        path = _resolve(value)
        parts = path.parts
        if "runs" in parts:
            index = len(parts) - 1 - tuple(reversed(parts)).index("runs")
            candidates.append(Path(*parts[:index]))
        elif path.name in {"resized", "lora", "masks", "cond_resized"}:
            candidates.append(path.parent)
    candidates.append(_resolve("post_image_dataset"))
    roots: list[Path] = []
    for candidate in candidates:
        root = candidate if candidate.name == "runs" else candidate / "runs"
        root = root.resolve()
        if root not in roots:
            roots.append(root)
    return roots


def _manifest_summary(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "manifest": str(path),
            "status": "invalid",
            "complete": False,
            "error": str(exc),
        }
    if not isinstance(raw, dict):
        return {
            "manifest": str(path),
            "status": "invalid",
            "complete": False,
            "error": "manifest is not an object",
        }
    source = raw.get("source") or {}
    return {
        "manifest": str(path),
        "run_id": f"{raw.get('source_group', '')}/{raw.get('config_hash', '')}",
        "source": raw.get("source_dir") or source.get("path"),
        "source_group": raw.get("source_group") or source.get("safe_name"),
        "config_hash": raw.get("config_hash") or raw.get("config_fingerprint"),
        "status": raw.get("status") or raw.get("preprocess_status") or "unknown",
        "complete": bool(raw.get("complete", raw.get("status") == "ready")),
        "artifacts": raw.get("artifacts") or {},
        "directories": raw.get("directories") or {},
        "updated_at": raw.get("updated_at") or raw.get("completed_at"),
        "error": raw.get("error"),
    }


def list_runs(
    variant: str | None = None,
    preset: str | None = None,
    source: str | None = None,
) -> list[dict]:
    """List isolated preprocessing runs, newest manifest first."""

    source_abs = _resolve(source).resolve() if source else None
    found: list[dict] = []
    seen: set[Path] = set()
    for root in _run_roots():
        if not root.is_dir():
            continue
        for manifest in root.glob("*/*/manifest.json"):
            manifest = manifest.resolve()
            if manifest in seen:
                continue
            seen.add(manifest)
            item = _manifest_summary(manifest)
            if source_abs is not None:
                try:
                    if Path(str(item.get("source") or "")).resolve() != source_abs:
                        continue
                except OSError:
                    continue
            found.append(item)
    found.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return found


def _run_manifest_path(manifest: str | Path) -> Path:
    """Resolve a manifest path, restricted to this install's run roots.

    ``GET /api/preprocess/run-status?manifest=`` is browser-reachable, so the
    value must name a manifest the install actually owns — anything else would
    turn the endpoint into an arbitrary-file reader. A directory is expanded to
    its ``manifest.json``. Raises ``ValueError`` when the path escapes every
    run root (``POST_IMAGE_DATASET`` included, so a relocated dataset root
    keeps working).
    """

    raw = str(manifest or "").strip()
    if not raw:
        raise ValueError("manifest is required")
    candidate = Path(raw).expanduser()
    if ".." in candidate.parts:
        raise ValueError(f"manifest may not contain '..': {raw}")
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    )
    if resolved.is_dir():
        resolved = (resolved / "manifest.json").resolve()
    if not any(resolved.is_relative_to(root) for root in _run_roots()):
        raise ValueError(f"manifest is outside the preprocessing run roots: {raw}")
    return resolved


def get_run_status(manifest: str | Path) -> dict:
    """Return one manifest summary and validate complete runs when possible.

    ``manifest`` is restricted to the run roots (see :func:`_run_manifest_path`).
    """

    path = _run_manifest_path(manifest)
    summary = _manifest_summary(path)
    if summary.get("complete"):
        try:
            run = run_from_manifest(path)
            summary["paths"] = {
                key: str(value) for key, value in run.directories.items()
            }
        except Exception as exc:  # malformed identity should be visible in UI
            summary["status"] = "invalid"
            summary["complete"] = False
            summary["error"] = str(exc)
    return summary
