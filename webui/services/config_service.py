"""Pure TOML config service — no Qt dependencies.

Extracts the merge/read/write logic from ``gui/__init__.py`` so both the
desktop GUI and WebUI can share it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import re

import toml

from gui.explanations import (
    FIELD_HELP as _FIELD_HELP,
    PREPROCESS_FIELD_HELP as _PRE_FIELD_HELP,
)

_ROOT = Path(__file__).resolve().parent.parent.parent
# Allow letters, digits, hyphens, underscores, and `custom/<name>` prefixes.
_SAFE_NAME_RE = re.compile(r"^(?:[a-zA-Z0-9_-]+/)?[a-zA-Z0-9_-]+$")


def _safe_variant(variant: str) -> str:
    """Reject path-traversal attempts in variant/preset names."""
    if not _SAFE_NAME_RE.match(variant):
        raise ValueError(f"Invalid variant name: {variant!r}")
    return variant


def _resolve_variant_path(variant: str) -> Path:
    """Resolve a variant name to its TOML file path.

    For custom/<name> variants, returns CUSTOM_VARIANTS_DIR/<name>.toml.
    For built-in variants, checks CUSTOM_VARIANTS_DIR/<variant>.toml first
    (overlay), then falls back to GUI_METHODS_DIR/<variant>.toml.
    """
    if variant.startswith("custom/"):
        stem = variant[len("custom/") :]
        return CUSTOM_VARIANTS_DIR / f"{stem}.toml"
    overlay = CUSTOM_VARIANTS_DIR / f"{variant}.toml"
    if overlay.exists():
        return overlay
    return GUI_METHODS_DIR / f"{variant}.toml"


ROOT = _ROOT
CONFIGS_DIR = ROOT / "configs"
METHODS_DIR = CONFIGS_DIR / "methods"
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
CUSTOM_DIR = CONFIGS_DIR / "custom"
CUSTOM_VARIANTS_DIR = CUSTOM_DIR / "variants"
CUSTOM_PRESETS_DIR = CUSTOM_DIR / "presets"

# ── Path overrides from the config chain ──────────────────────────

_DEFAULT_PATHS = {
    "source_image_dir": "image_dataset",
    "resized_image_dir": "post_image_dataset/resized",
    "lora_cache_dir": "post_image_dataset/lora",
}


def get_path_overrides(
    preset: str = "default", variant: str | None = None
) -> dict[str, str]:
    """Resolve dataset paths from the config chain (base → preset → method).

    Returns ``{"source_image_dir": ..., "resized_image_dir": ...,
    "lora_cache_dir": ...}`` with defaults filled in for missing keys.
    """
    from library.config.io import load_path_overrides

    method = variant
    overrides = load_path_overrides(
        preset=preset,
        method=method,
        methods_subdir="gui-methods" if method else "methods",
    )
    return {k: overrides.get(k, v) for k, v in _DEFAULT_PATHS.items()}


_METHOD_ORDER = (
    "lora",
    "tlora",
    "hydralora",
    "reft",
    "postfix",
    "fera",
    "chimera",
    "ip_adapter",
    "easycontrol",
)

_ATTN_MODES = ["flex", "flash"]
_SAMPLER_CHOICES = ["euler", "er_sde", "euler_a"]

_SELECT_OPTIONS: dict[str, list[str]] = {
    "attn_mode": _ATTN_MODES,
    "optimizer_type": [
        "AdamW",
        "AdamW8bit",
        "Lion",
        "Lion8bit",
        "SGDNesterov",
        "SGDNesterov8bit",
        "PagedAdamW",
        "PagedAdamW8bit",
        "PagedAdamW32bit",
        "PagedLion8bit",
        "DAdaptation",
        "DAdaptAdam",
        "DAdaptAdamPreprint",
        "DAdaptAdaGrad",
        "DAdaptAdan",
        "DAdaptAdanIP",
        "DAdaptLion",
        "DAdaptSGD",
        "Prodigy",
        "Adafactor",
        "RAdamScheduleFree",
        "AdamWScheduleFree",
        "SGDScheduleFree",
    ],
    "lr_scheduler": [
        "constant",
        "constant_with_warmup",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "inverse_sqrt",
        "cosine_with_min_lr",
        "piecewise_constant",
        "warmup_stable_decay",
    ],
    "timestep_sampling": [
        "sigmoid",
        "sigma",
        "uniform",
        "shift",
        "flux_shift",
    ],
    "sample_sampler": ["euler", "er_sde", "euler_a"],
}

_GROUPS = {
    "Architecture": {
        "network_dim",
        "network_alpha",
        "network_module",
        "network_args",
        "use_ortho",
        "use_timestep_mask",
        "use_moe_style",
        "route_per_layer",
        "router_source",
        "add_reft",
        "min_rank",
        "alpha_rank_scale",
        "num_experts",
        "balance_loss_weight",
        "balance_loss_warmup_ratio",
        "reft_dim",
        "reft_alpha",
        "reft_layers",
        "sigma_feature_dim",
        "router_targets",
        "per_bucket_balance_weight",
        "num_sigma_buckets",
        "specialize_experts_by_sigma_buckets",
        "sigma_bucket_boundaries",
        "network_train_unet_only",
    },
    "Training": {
        "learning_rate",
        "max_train_epochs",
        "save_every_n_epochs",
        "checkpointing_epochs",
        "batch_size",
        "num_repeats",
        "gradient_accumulation_steps",
        "use_shuffled_caption_variants",
        "caption_dropout_rate",
        "optimizer_type",
        "lr_scheduler",
        "timestep_sampling",
        "discrete_flow_shift",
        "use_valid",
        "validation_split_num",
    },
    "Performance": {
        "attn_mode",
        "gradient_checkpointing",
        "unsloth_offload_checkpointing",
        "blocks_to_swap",
        "torch_compile",
        "compile_mode",
        "trim_crossattn_kv",
        "cache_llm_adapter_outputs",
        "masked_loss",
        "mixed_precision",
        "static_token_count",
        "vae_chunk_size",
        "vae_disable_cache",
        "cache_latents",
        "cache_latents_to_disk",
        "cache_text_encoder_outputs",
        "cache_text_encoder_outputs_to_disk",
        "skip_cache_check",
        "layer_start",
        "use_cmmd",
    },
    "Paths": {
        "pretrained_model_name_or_path",
        "qwen3",
        "vae",
        "output_dir",
        "output_name",
        "save_model_as",
        "source_image_dir",
        "resized_image_dir",
        "lora_cache_dir",
        "path_pattern",
        "drop_lowres_images",
        "min_pixels",
    },
    "Preview Sampling": {
        "sample_every_n_epochs",
        "sample_every_n_steps",
        "sample_at_first",
        "sample_prompts",
        "sample_sampler",
        "sample_guidance_scale",
        "sample_flow_shift",
        "sample_image_size",
        "sample_seed",
    },
}
_K2G = {k: g for g, ks in _GROUPS.items() for k in ks}
_SKIP = {"base_config", "dataset_config", "general", "datasets", "variant"}
_VIRTUAL_KEYS = {"use_valid", "validation_split_num", "batch_size", "num_repeats"}

_BASIC = {
    "learning_rate",
    "max_train_epochs",
    "save_every_n_epochs",
    "network_dim",
    "network_alpha",
    "network_weights",
    "num_experts",
    "output_name",
    "batch_size",
    "num_repeats",
    "use_shuffled_caption_variants",
    "caption_dropout_rate",
    "gradient_checkpointing",
    "blocks_to_swap",
    "source_image_dir",
    "lora_cache_dir",
    "output_dir",
    "path_pattern",
    "drop_lowres_images",
    "min_pixels",
    "use_valid",
    "validation_split_num",
}


# ── Low-level helpers ──────────────────────────────────────────


def _load(p: Path) -> dict:
    return toml.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(toml.dumps(d), encoding="utf-8")


# ── Variant / preset discovery ─────────────────────────────────


def _read_variant_metadata(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = toml.loads(path.read_text(encoding="utf-8"))
    except (toml.TomlDecodeError, OSError):
        return {}
    meta = data.get("variant")
    return meta if isinstance(meta, dict) else {}


def _builtin_variants_by_family() -> dict[str, list[tuple[int, str, str]]]:
    by_family: dict[str, list[tuple[int, str, str]]] = {}
    if not GUI_METHODS_DIR.is_dir():
        return by_family
    for path in GUI_METHODS_DIR.glob("*.toml"):
        meta = _read_variant_metadata(path)
        family = meta.get("family")
        if not isinstance(family, str) or not family:
            continue
        order = meta.get("order")
        order_int = order if isinstance(order, int) else 100
        label = meta.get("label") if isinstance(meta.get("label"), str) else path.stem
        by_family.setdefault(family, []).append((order_int, path.stem, label))
    for entries in by_family.values():
        entries.sort(key=lambda e: (e[0], e[1]))
    return by_family


def list_methods() -> list[str]:
    return list(_METHOD_ORDER)


def list_gui_variants(method: str) -> list[str]:
    by_family = _builtin_variants_by_family()
    ordered = [stem for _order, stem, _label in by_family.get(method, [])]
    if CUSTOM_VARIANTS_DIR.exists():
        builtin_stems = set(ordered)
        for p in sorted(CUSTOM_VARIANTS_DIR.glob("*.toml")):
            if p.stem in builtin_stems:
                continue  # overlay for built-in; already listed
            ordered.append(f"custom/{p.stem}")
    return ordered


def variant_labels(method: str) -> dict[str, str]:
    by_family = _builtin_variants_by_family()
    out: dict[str, str] = {}
    for _order, stem, label in by_family.get(method, []):
        out[stem] = label
    if CUSTOM_VARIANTS_DIR.exists():
        builtin_stems = set(out)
        for p in sorted(CUSTOM_VARIANTS_DIR.glob("*.toml")):
            if p.stem in builtin_stems:
                continue
            out[f"custom/{p.stem}"] = p.stem
    return out


def get_field_groups() -> dict[str, set[str]]:
    """Return the field→group mapping (public accessor)."""
    return dict(_GROUPS)


def create_variant(name: str, seed_from: str | None = None) -> list[str]:
    """Create a new custom variant, optionally seeded from an existing one.

    Returns the updated variants list for the variant's family.
    """
    _safe_variant(name)
    CUSTOM_VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = CUSTOM_VARIANTS_DIR / f"{name}.toml"
    if dest.exists():
        raise ValueError(f"Variant already exists: {name}")

    seed_data: dict = {}
    if seed_from:
        _safe_variant(seed_from)
        seed_path = _resolve_variant_path(seed_from)
        if seed_path.is_file():
            seed_data = _load(seed_path)
            seed_data.pop("variant", None)  # strip [variant] metadata

    _save(dest, seed_data)
    # Return variants for the same family as seed_from (or all if no seed)
    if seed_from:
        meta = _read_variant_metadata(GUI_METHODS_DIR / f"{seed_from}.toml")
        family = meta.get("family", "lora")
    else:
        family = "lora"
    return list_gui_variants(family)


def variant_metadata(variant: str) -> dict:
    _safe_variant(variant)
    # Always read metadata from the built-in file, not the custom overlay
    if variant.startswith("custom/"):
        path = _resolve_variant_path(variant)
    else:
        path = GUI_METHODS_DIR / f"{variant}.toml"
    meta = _read_variant_metadata(path)
    return {
        "variant": variant,
        "family": meta.get("family"),
        "label": meta.get("label"),
        "description": meta.get("description"),
        "experimental": bool(meta.get("experimental")),
        "order": meta.get("order", 100),
    }


def variant_path(variant: str) -> Path:
    _safe_variant(variant)
    return _resolve_variant_path(variant)


def _load_all_presets() -> dict:
    presets: dict = {}
    if PRESETS_FILE.exists():
        data = toml.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        presets.update({k: v for k, v in data.items() if isinstance(v, dict)})
    if CUSTOM_PRESETS_DIR.exists():
        for p in sorted(CUSTOM_PRESETS_DIR.glob("*.toml")):
            try:
                presets[p.stem] = toml.loads(p.read_text(encoding="utf-8"))
            except (toml.TomlDecodeError, OSError):
                continue
    return presets


def list_presets() -> list[str]:
    return sorted(_load_all_presets())


def is_builtin_preset(name: str) -> bool:
    """Check if a preset is defined in the built-in presets.toml."""
    if not PRESETS_FILE.exists():
        return False
    data = toml.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    return name in data and isinstance(data[name], dict)


def create_custom_preset(name: str, data: dict) -> list[str]:
    """Create a custom preset in configs/custom/presets/<name>.toml."""
    _safe_variant(name)
    if is_builtin_preset(name):
        raise ValueError(f"Cannot overwrite built-in preset: {name}")
    CUSTOM_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    _save(CUSTOM_PRESETS_DIR / f"{name}.toml", data)
    return list_presets()


def delete_custom_preset(name: str) -> list[str]:
    """Delete a custom preset. Built-in presets cannot be deleted."""
    _safe_variant(name)
    if is_builtin_preset(name):
        raise ValueError(f"Cannot delete built-in preset: {name}")
    path = CUSTOM_PRESETS_DIR / f"{name}.toml"
    if not path.exists():
        raise ValueError(f"Custom preset not found: {name}")
    path.unlink()
    return list_presets()


# ── Merge chain ────────────────────────────────────────────────


def _validation_enabled_from_datasets(datasets: Any) -> Optional[bool]:
    if not isinstance(datasets, list) or not datasets:
        return None
    first = datasets[0]
    if not isinstance(first, dict):
        return None
    vsn = first.get("validation_split_num")
    vs = first.get("validation_split")
    if vsn is None and vs is None:
        return None
    return (vsn or 0) > 0 or (vs or 0.0) > 0.0


def _base_validation_enabled(base_data: dict) -> bool:
    return bool(_validation_enabled_from_datasets(base_data.get("datasets")))


def _base_validation_split_num(base_data: dict) -> int:
    datasets = base_data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return 0
    first = datasets[0]
    if not isinstance(first, dict):
        return 0
    vsn = first.get("validation_split_num")
    return int(vsn) if vsn is not None else 0


def _base_batch_size(base_data: dict) -> int:
    datasets = base_data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return 1
    first = datasets[0]
    if not isinstance(first, dict):
        return 1
    bs = first.get("batch_size")
    return int(bs) if bs is not None else 1


def _base_num_repeats(base_data: dict) -> int:
    """Read num_repeats from the first subset of the first dataset in *base_data*."""
    datasets = base_data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return 1
    first_ds = datasets[0]
    if not isinstance(first_ds, dict):
        return 1
    subsets = first_ds.get("subsets")
    if not isinstance(subsets, list) or not subsets:
        return 1
    first_sub = subsets[0]
    if not isinstance(first_sub, dict):
        return 1
    nr = first_sub.get("num_repeats")
    return int(nr) if nr is not None else 1


def merged_gui_variant_preset(variant: str, preset: str) -> tuple[dict, dict[str, str]]:
    """Merge base + preset + gui-methods/<variant>.toml.

    Returns (merged_dict, origin_map) where origin_map[key] is
    'base' | 'preset' | 'method'.
    """
    _safe_variant(variant)
    base = _load(CONFIGS_DIR / "base.toml")
    pset = _load_all_presets().get(preset, {})
    if variant.startswith("custom/"):
        meth = _load(_resolve_variant_path(variant))
    else:
        builtin = _load(GUI_METHODS_DIR / f"{variant}.toml")
        overlay = _load(CUSTOM_VARIANTS_DIR / f"{variant}.toml")
        meth = {**builtin, **overlay}
    merged: dict = {}
    origin: dict[str, str] = {}

    def _merge_layer(layer: dict, src: str) -> None:
        for k, v in layer.items():
            if isinstance(v, dict) and k not in _SKIP:
                # Flatten nested TOML sections (e.g. [preview]) into
                # top-level keys so they appear individually in the WebUI.
                for sub_k, sub_v in v.items():
                    merged[sub_k] = sub_v
                    origin[sub_k] = src
            else:
                merged[k] = v
                origin[k] = src

    _merge_layer(base, "base")
    _merge_layer(pset, "preset")
    _merge_layer(meth, "method")

    # Inject virtual keys from [[datasets]]
    variant_override = _validation_enabled_from_datasets(meth.get("datasets"))
    if variant_override is not None:
        merged["use_valid"] = variant_override
        origin["use_valid"] = "method"
    else:
        merged["use_valid"] = _base_validation_enabled(base)
        origin["use_valid"] = "base"

    datasets = meth.get("datasets")
    variant_vsn = None
    if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict):
        vsn = datasets[0].get("validation_split_num")
        if vsn is not None:
            try:
                variant_vsn = int(vsn)
            except (TypeError, ValueError):
                pass

    if variant_vsn is not None:
        merged["validation_split_num"] = variant_vsn
        origin["validation_split_num"] = "method"
    else:
        merged["validation_split_num"] = _base_validation_split_num(base)
        origin["validation_split_num"] = "base"

    # batch_size virtual key
    variant_bs = None
    if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict):
        bs = datasets[0].get("batch_size")
        if bs is not None:
            try:
                variant_bs = int(bs)
            except (TypeError, ValueError):
                pass

    if variant_bs is not None:
        merged["batch_size"] = variant_bs
        origin["batch_size"] = "method"
    else:
        merged["batch_size"] = _base_batch_size(base)
        origin["batch_size"] = "base"

    # num_repeats virtual key (subset-level: datasets[0].subsets[0].num_repeats)
    variant_nr = None
    if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict):
        subsets = datasets[0].get("subsets")
        if isinstance(subsets, list) and subsets and isinstance(subsets[0], dict):
            nr = subsets[0].get("num_repeats")
            if nr is not None:
                try:
                    variant_nr = int(nr)
                except (TypeError, ValueError):
                    pass

    if variant_nr is not None:
        merged["num_repeats"] = variant_nr
        origin["num_repeats"] = "method"
    else:
        merged["num_repeats"] = _base_num_repeats(base)
        origin["num_repeats"] = "base"

    return merged, origin


# ── Field metadata for the frontend ────────────────────────────


def _field_desc(key: str, lang: str) -> str | None:
    """Return the localized description for *key* from FIELD_HELP."""
    for src in (_PRE_FIELD_HELP, _FIELD_HELP):
        entry = src.get(key)
        if entry:
            return entry.get(lang) or entry.get("en")
    return None


def _field_desc_en(key: str) -> str | None:
    """Return the English description for *key* from FIELD_HELP."""
    for src in (_PRE_FIELD_HELP, _FIELD_HELP):
        entry = src.get(key)
        if entry:
            return entry.get("en")
    return None


def get_field_type(key: str, value: Any) -> str:
    """Map a Python value + key to a frontend widget type."""
    if key in _SELECT_OPTIONS:
        return "select"
    if key == "sample_sampler":
        return "select"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    return "str"


def build_merged_config(variant: str, preset: str, lang: str = "cn") -> dict:
    """Build the full merged config with field metadata for the frontend.

    Returns a dict with ``fields`` (list of FieldMeta-like dicts),
    ``variant``, and ``preset``.
    """
    merged, origin = merged_gui_variant_preset(variant, preset)
    custom_preset = not is_builtin_preset(preset)
    fields = []
    for key, value in sorted(merged.items()):
        if key in _SKIP:
            continue
        ftype = get_field_type(key, value)
        orig = origin.get(key, "base")
        # Custom presets: all fields are editable (user owns the preset)
        # Built-in presets: only method-layer fields are editable
        if custom_preset:
            read_only = False
        else:
            read_only = orig in ("base", "preset")
        fields.append(
            {
                "key": key,
                "value": value,
                "origin": orig,
                "field_type": ftype,
                "group": _K2G.get(key),
                "is_basic": key in _BASIC,
                "is_virtual": key in _VIRTUAL_KEYS,
                "read_only": read_only,
                "options": (
                    _SAMPLER_CHOICES
                    if key == "sample_sampler"
                    else _SELECT_OPTIONS.get(key)
                ),
                "description": _field_desc(key, lang),
                "description_en": _field_desc_en(key),
            }
        )
    return {"fields": fields, "variant": variant, "preset": preset}


# ── Read/write single layers ───────────────────────────────────


def read_layer(layer: str, variant: str = "lora", preset: str = "default") -> dict:
    """Read a single config layer as a dict."""
    if layer == "base":
        return _load(CONFIGS_DIR / "base.toml")
    if layer == "preset":
        return _load_all_presets().get(preset, {})
    if layer == "method":
        _safe_variant(variant)
        return _load(_resolve_variant_path(variant))
    raise ValueError(f"Unknown layer: {layer}")


def write_layer(
    layer: str, data: dict, variant: str = "lora", preset: str = "default"
) -> None:
    """Write a single config layer. Only method layer is user-editable."""
    if layer == "base":
        raise ValueError("base.toml is read-only; edit via custom/variants/ overlays")
    if layer == "preset":
        raise ValueError(
            "presets.toml is read-only; create custom presets in custom/presets/"
        )
    if layer == "method":
        _safe_variant(variant)
        if variant.startswith("custom/"):
            path = _resolve_variant_path(variant)
        else:
            path = CUSTOM_VARIANTS_DIR / f"{variant}.toml"
        _save(path, data)
    else:
        raise ValueError(f"Unknown layer: {layer}")


def apply_validation_choice(
    out: dict,
    enabled: bool,
    split_num: Optional[int] = None,
    base_split_num: Optional[int] = None,
) -> None:
    """Encode the use_valid checkbox into the variant TOML dict."""
    existing = out.get("datasets")
    if enabled:
        keep_override = (
            split_num is not None
            and split_num > 0
            and split_num != (base_split_num or 0)
        )
        if keep_override:
            if not isinstance(existing, list):
                existing = []
                out["datasets"] = existing
            if not existing:
                existing.append({})
            first = existing[0]
            if not isinstance(first, dict):
                first = {}
                existing[0] = first
            first["validation_split_num"] = int(split_num)
            first.pop("validation_split", None)
            return
        if not isinstance(existing, list) or not existing:
            return
        first = existing[0]
        if not isinstance(first, dict):
            return
        first.pop("validation_split_num", None)
        first.pop("validation_split", None)
        if not first and len(existing) == 1:
            del out["datasets"]
        return

    if not isinstance(existing, list):
        existing = []
        out["datasets"] = existing
    if not existing:
        existing.append({})
    first = existing[0]
    if not isinstance(first, dict):
        first = {}
        existing[0] = first
    first["validation_split_num"] = 0
    first["validation_split"] = 0.0


def _apply_batch_size(out: dict, value: int, base_value: int) -> None:
    """Write or strip the batch_size override in the variant TOML's [[datasets]]."""
    existing = out.get("datasets")
    if value != base_value:
        # Write override
        if not isinstance(existing, list):
            existing = []
            out["datasets"] = existing
        if not existing:
            existing.append({})
        first = existing[0]
        if not isinstance(first, dict):
            first = {}
            existing[0] = first
        first["batch_size"] = value
        return
    # Same as base — strip override to keep variant TOML clean
    if not isinstance(existing, list) or not existing:
        return
    first = existing[0]
    if not isinstance(first, dict):
        return
    first.pop("batch_size", None)
    if not first and len(existing) == 1:
        del out["datasets"]


def _apply_num_repeats(out: dict, value: int, base_value: int) -> None:
    """Write or strip the num_repeats override in the variant TOML's [[datasets.subsets]]."""
    if value == base_value:
        # Same as base — strip override to keep variant TOML clean
        datasets = out.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            return
        first_ds = datasets[0]
        if not isinstance(first_ds, dict):
            return
        subsets = first_ds.get("subsets")
        if not isinstance(subsets, list) or not subsets:
            return
        first_sub = subsets[0]
        if not isinstance(first_sub, dict):
            return
        first_sub.pop("num_repeats", None)
        # Clean up empty subset -> empty datasets chain
        if not first_sub and len(subsets) == 1:
            del first_ds["subsets"]
            if not first_ds and len(datasets) == 1:
                del out["datasets"]
        return

    # Write override — ensure datasets -> subsets chain exists
    datasets = out.get("datasets")
    if not isinstance(datasets, list):
        datasets = []
        out["datasets"] = datasets
    if not datasets:
        datasets.append({})
    first_ds = datasets[0]
    if not isinstance(first_ds, dict):
        first_ds = {}
        datasets[0] = first_ds
    subsets = first_ds.get("subsets")
    if not isinstance(subsets, list):
        subsets = []
        first_ds["subsets"] = subsets
    if not subsets:
        subsets.append({})
    first_sub = subsets[0]
    if not isinstance(first_sub, dict):
        first_sub = {}
        subsets[0] = first_sub
    first_sub["num_repeats"] = value


def validate_config(data: dict) -> list[str]:
    """Validate a config dict. Returns a list of error strings (empty = valid)."""
    errors: list[str] = []
    if "learning_rate" in data:
        lr = data["learning_rate"]
        if isinstance(lr, (int, float)) and lr <= 0:
            errors.append("learning_rate must be positive")
    if "network_dim" in data:
        dim = data["network_dim"]
        if isinstance(dim, int) and dim <= 0:
            errors.append("network_dim must be positive")
    if "max_train_epochs" in data:
        ep = data["max_train_epochs"]
        if isinstance(ep, int) and ep <= 0:
            errors.append("max_train_epochs must be positive")
    if "batch_size" in data:
        bs = data["batch_size"]
        if isinstance(bs, int) and bs <= 0:
            errors.append("batch_size must be positive")
    if "num_repeats" in data:
        nr = data["num_repeats"]
        if isinstance(nr, int) and nr < 1:
            errors.append("num_repeats must be at least 1")
    return errors


def save_variant_config(variant: str, data: dict) -> None:
    """Save user-edited config fields back to the variant TOML.

    Separates virtual keys (written into [[datasets]] blocks) from
    flat TOML keys. Skips ``_SKIP`` keys and keys that match the
    merged base+preset values (no-op saves).

    If ``data`` contains an ``extra_args`` key with a non-empty string,
    it is parsed as TOML and merged as overrides (extra_args win on
    conflicts with form values).
    """
    _safe_variant(variant)
    data = dict(data)  # copy to avoid mutating caller's dict
    if variant.startswith("custom/"):
        path = _resolve_variant_path(variant)
    else:
        # Built-in variant: always save to custom overlay, never to template
        path = CUSTOM_VARIANTS_DIR / f"{variant}.toml"
    current = _load(path)

    # Handle extra_args: parse as TOML and merge as overrides
    extra_text = data.pop("extra_args", None)
    extras: dict = {}
    if isinstance(extra_text, str) and extra_text.strip():
        text = extra_text.strip()
        try:
            parsed = toml.loads(text)
        except Exception:
            # Windows path backslash workaround: bare backslashes break TOML
            if "\\" in text:
                try:
                    parsed = toml.loads(text.replace("\\", "/"))
                except Exception:
                    parsed = {}
            else:
                parsed = {}
        extras = {k: v for k, v in parsed.items() if not isinstance(v, dict)}

    # Handle virtual keys
    use_valid = data.pop("use_valid", None)
    vsn = data.pop("validation_split_num", None)
    bs = data.pop("batch_size", None)
    nr = data.pop("num_repeats", None)
    if use_valid is not None:
        base = _load(CONFIGS_DIR / "base.toml")
        base_vsn = _base_validation_split_num(base)
        apply_validation_choice(current, use_valid, vsn, base_vsn)

    if bs is not None:
        base = _load(CONFIGS_DIR / "base.toml")
        base_bs = _base_batch_size(base)
        _apply_batch_size(current, int(bs), base_bs)

    if nr is not None:
        base = _load(CONFIGS_DIR / "base.toml")
        base_nr = _base_num_repeats(base)
        _apply_num_repeats(current, int(nr), base_nr)

    # Write flat keys
    for key, value in data.items():
        if key in _SKIP:
            continue
        current[key] = value

    # Extra args override form values
    if extras:
        current.update(extras)

    _save(path, current)


# ── Prelaunch checks (cache + checkpoint) ───────────────────────

# Cache-file suffixes (kept in sync with gui/__init__.py)
_LATENT_SUFFIX = "_anima.npz"
_TE_SUFFIX = "_anima_te.safetensors"
_PE_SUFFIX = "_anima_pe.safetensors"

# Variant families that require PE cache
_PE_REQUIRED_FAMILIES = {"ip_adapter", "easycontrol"}


def count_preprocess_caches(cache_dir: Path) -> dict[str, int]:
    """Count latent / TE / PE cache sidecars under *cache_dir*."""
    out = {"latents": 0, "te": 0, "pe": 0}
    if not cache_dir.is_dir():
        return out
    for p in cache_dir.rglob("*"):
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


def find_resumable_checkpoint(merged: dict) -> tuple[Path, int] | None:
    """Check for a resumable checkpoint state directory.

    Returns ``(state_dir, current_step)`` or ``None``.
    """
    if not merged.get("checkpointing_epochs"):
        return None
    output_dir = merged.get("output_dir")
    output_name = merged.get("output_name") or "last"
    if not output_dir:
        return None
    state_dir = ROOT / output_dir / f"{output_name}-checkpoint-state"
    train_state_file = state_dir / "train_state.json"
    if not train_state_file.is_file():
        return None
    try:
        data = json.loads(train_state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    step = int(data.get("current_step", 0))
    return state_dir, step


def prelaunch_check(variant: str, preset: str) -> dict:
    """Check cache counts and checkpoint state for a training launch.

    Returns a dict with cache_counts, has_cache, checkpoint info,
    and whether PE cache is required.
    """
    merged, _origin = merged_gui_variant_preset(variant, preset)

    # Resolve cache directory
    cache_dir_str = merged.get("lora_cache_dir", "post_image_dataset/lora")
    cache_dir = ROOT / cache_dir_str
    cache_counts = count_preprocess_caches(cache_dir)

    # Check if PE cache is required
    meta = _read_variant_metadata(GUI_METHODS_DIR / f"{variant}.toml")
    family = meta.get("family", "")
    requires_pe = family in _PE_REQUIRED_FAMILIES

    has_cache = (
        cache_counts["latents"] > 0
        or cache_counts["te"] > 0
        or (requires_pe and cache_counts["pe"] > 0)
    )

    # Check checkpoint
    ckpt = find_resumable_checkpoint(merged)
    checkpoint_info = None
    if ckpt is not None:
        state_dir, step = ckpt
        checkpoint_info = {
            "state_dir": str(state_dir),
            "step": step,
        }

    return {
        "cache_counts": cache_counts,
        "has_cache": has_cache,
        "checkpoint": checkpoint_info,
        "requires_pe": requires_pe,
    }


def wipe_checkpoint(output_dir: str, output_name: str) -> None:
    """Delete a checkpoint state directory and its sidecar adapter file.

    Raises ``OSError`` if deletion fails.
    """
    name = output_name or "last"
    state_dir = ROOT / output_dir / f"{name}-checkpoint-state"
    sidecar = state_dir.parent / f"{name}-checkpoint.safetensors"
    if state_dir.is_dir():
        shutil.rmtree(state_dir)
    if sidecar.is_file():
        sidecar.unlink()


# ── Sample prompts file I/O ────────────────────────────────────


_NEG_PROMPT_RE = re.compile(r"n (.+)", re.IGNORECASE)


def read_sample_prompts(path_str: str) -> list[dict]:
    """Parse a sample_prompts.txt file into structured prompt entries.

    Returns a list of ``{"prompt": str, "negative_prompt": str}`` dicts.
    Lines starting with ``#`` and blank lines are skipped.
    """
    path = ROOT / path_str
    if not path.is_file():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(" --")
        prompt = parts[0]
        neg = ""
        for part in parts[1:]:
            m = _NEG_PROMPT_RE.match(part.strip())
            if m:
                neg = m.group(1).strip()
                break
        entries.append({"prompt": prompt, "negative_prompt": neg})
    return entries


def write_sample_prompts(path_str: str, entries: list[dict]) -> None:
    """Write structured prompt entries back to a .txt file."""
    path = ROOT / path_str
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for entry in entries:
        prompt = entry.get("prompt", "").strip()
        if not prompt:
            continue
        neg = entry.get("negative_prompt", "").strip()
        if neg:
            lines.append(f"{prompt} --n {neg}")
        else:
            lines.append(prompt)
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


# ── Field help + method guide ──────────────────────────────────


def get_field_help_data(variant: str, lang: str = "cn") -> dict:
    """Return field help dict + method guide HTML for the config editor.

    Returns ``{"field_help": {key: text}, "guide_html": str}``.
    """
    _safe_variant(variant)
    meta = _read_variant_metadata(GUI_METHODS_DIR / f"{variant}.toml")
    family = meta.get("family", variant)

    # Collect all field help entries for the requested language
    field_help: dict[str, str] = {}
    for src in (_PRE_FIELD_HELP, _FIELD_HELP):
        for key, entry in src.items():
            if isinstance(entry, dict):
                text = entry.get(lang) or entry.get("en")
                if text:
                    field_help[key] = text

    # Load method guide HTML from gui/explanations/guides/
    guide_html = _load_method_guide(family, lang)

    return {"field_help": field_help, "guide_html": guide_html}


_GUIDES_DIR = ROOT / "gui" / "explanations" / "guides"
_NOT_MERGEABLE = frozenset({"postfix", "hydralora", "reft", "fera"})
_KNOWN_GUIDE_METHODS = frozenset({"lora", "tlora", "postfix", "hydralora", "reft", "fera"})


def _read_guide(name: str, lang: str) -> str:
    """Read a guide HTML file with fallback: lang → en."""
    path = _GUIDES_DIR / f"{name}.{lang}.html"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    path = _GUIDES_DIR / f"{name}.en.html"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def _load_method_guide(method: str, lang: str = "cn") -> str:
    """Assemble the full method guide HTML for the given method and language."""
    if method not in _KNOWN_GUIDE_METHODS:
        return ""
    parts = [_read_guide("_apply_note", lang)]
    if method in _NOT_MERGEABLE:
        parts.append(_read_guide("_not_mergeable", lang))
    parts.append(_read_guide(method, lang))
    return "".join(parts)
