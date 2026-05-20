"""Preprocessing settings and status service — no Qt dependencies.

Reads/writes ``configs/sam_mask.yaml`` and ``gui/gui_settings.json``, and
counts preprocess caches for the status dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from webui.services.config_service import ROOT, get_path_overrides

# ── Paths ─────────────────────────────────────────────────────────

CONFIGS_DIR = ROOT / "configs"
SAM_YAML = CONFIGS_DIR / "sam_mask.yaml"
SETTINGS_FILE = ROOT / "gui" / "gui_settings.json"


def _resolve(p: str) -> Path:
    """Resolve a possibly-relative path against ROOT."""
    pp = Path(p)
    return pp if pp.is_absolute() else ROOT / pp


def _get_paths() -> dict[str, Path]:
    """Return resolved dataset paths from the config chain."""
    paths = get_path_overrides()
    return {
        "resized": _resolve(paths["resized_image_dir"]),
        "masks": _resolve(paths["resized_image_dir"]).parent / "masks",
        "cache": _resolve(paths["lora_cache_dir"]),
    }


# ── Cache-file suffixes (kept in sync with gui/__init__.py) ──────

_LATENT_SUFFIX = "_anima.npz"
_TE_SUFFIX = "_anima_te.safetensors"
_PE_SUFFIX = "_anima_pe.safetensors"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ── Defaults (match gui/tabs/preprocess_tab.py) ──────────────────

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
}


# ── YAML dumper that indents list items ──────────────────────────


class _IndentedListDumper(yaml.SafeDumper):
    """SafeDumper that indents list items under mapping keys.

    Matches the canonical sam_mask.yaml formatting (2-space indent on the dash).
    """


def _increase_indent(self, flow=False, indentless=False):
    return super().increase_indent(flow, indentless=False)


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
    }


def save_settings(data: dict) -> dict:
    """Write settings back to both config files.

    Returns the saved settings (round-tripped through get_settings).
    """
    # ── SAM yaml ──
    sam = data.get("sam", {})
    sam_yaml = {
        "prompts": sam.get("prompts", DEFAULTS["sam_prompts"]),
        "threshold": sam.get("threshold", DEFAULTS["sam_threshold"]),
        "dilate": sam.get("dilate", DEFAULTS["sam_dilate"]),
    }
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
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

    return get_settings()


# ── Status / cache counting ──────────────────────────────────────


def count_caches(cache_dir: Path | None = None) -> dict[str, int]:
    """Count latent / TE / PE cache sidecars under *cache_dir*."""
    d = cache_dir or _get_paths()["cache"]
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


def count_resized() -> int:
    """Count resized images under the configured ``resized_image_dir``."""
    d = _get_paths()["resized"]
    if not d.is_dir():
        return 0
    return sum(
        1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def count_masks() -> int:
    """Count merged mask files under the configured masks directory."""
    d = _get_paths()["masks"]
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.rglob("*_mask.png"))


def get_status(cache_dir: Path | None = None) -> dict:
    """Return a snapshot of preprocess pipeline counts."""
    return {
        "resized": count_resized(),
        "masks": count_masks(),
        "cache": count_caches(cache_dir),
    }
