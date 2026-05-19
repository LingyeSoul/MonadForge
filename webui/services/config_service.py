"""Pure TOML config service — no Qt dependencies.

Extracts the merge/read/write logic from ``gui/__init__.py`` so both the
desktop GUI and WebUI can share it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import re

import toml

_ROOT = Path(__file__).resolve().parent.parent.parent
# Allow letters, digits, hyphens, underscores, and `custom/<name>` prefixes.
_SAFE_NAME_RE = re.compile(r"^(?:[a-zA-Z0-9_-]+/)?[a-zA-Z0-9_-]+$")


def _safe_variant(variant: str) -> str:
    """Reject path-traversal attempts in variant/preset names."""
    if not _SAFE_NAME_RE.match(variant):
        raise ValueError(f"Invalid variant name: {variant!r}")
    return variant


ROOT = _ROOT
CONFIGS_DIR = ROOT / "configs"
METHODS_DIR = CONFIGS_DIR / "methods"
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
CUSTOM_DIR = CONFIGS_DIR / "custom"
CUSTOM_VARIANTS_DIR = GUI_METHODS_DIR / "custom"

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
}
_K2G = {k: g for g, ks in _GROUPS.items() for k in ks}
_SKIP = {"base_config", "dataset_config", "general", "datasets", "variant"}
_VIRTUAL_KEYS = {"use_valid", "validation_split_num"}

_BASIC = {
    "learning_rate",
    "max_train_epochs",
    "save_every_n_epochs",
    "network_dim",
    "network_alpha",
    "network_weights",
    "num_experts",
    "output_name",
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
        for p in sorted(CUSTOM_VARIANTS_DIR.glob("*.toml")):
            ordered.append(f"custom/{p.stem}")
    return ordered


def variant_labels(method: str) -> dict[str, str]:
    by_family = _builtin_variants_by_family()
    out: dict[str, str] = {}
    for _order, stem, label in by_family.get(method, []):
        out[stem] = label
    if CUSTOM_VARIANTS_DIR.exists():
        for p in sorted(CUSTOM_VARIANTS_DIR.glob("*.toml")):
            out[f"custom/{p.stem}"] = p.stem
    return out


def get_field_groups() -> dict[str, set[str]]:
    """Return the field→group mapping (public accessor)."""
    return dict(_GROUPS)


def variant_metadata(variant: str) -> dict:
    _safe_variant(variant)
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
    return GUI_METHODS_DIR / f"{variant}.toml"


def _load_all_presets() -> dict:
    presets: dict = {}
    if PRESETS_FILE.exists():
        data = toml.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        presets.update({k: v for k, v in data.items() if isinstance(v, dict)})
    if CUSTOM_DIR.exists():
        for p in sorted(CUSTOM_DIR.glob("*.toml")):
            try:
                presets[p.stem] = toml.loads(p.read_text(encoding="utf-8"))
            except (toml.TomlDecodeError, OSError):
                continue
    return presets


def list_presets() -> list[str]:
    return sorted(_load_all_presets())


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


def merged_gui_variant_preset(variant: str, preset: str) -> tuple[dict, dict[str, str]]:
    """Merge base + preset + gui-methods/<variant>.toml.

    Returns (merged_dict, origin_map) where origin_map[key] is
    'base' | 'preset' | 'method'.
    """
    _safe_variant(variant)
    base = _load(CONFIGS_DIR / "base.toml")
    pset = _load_all_presets().get(preset, {})
    meth = _load(GUI_METHODS_DIR / f"{variant}.toml")
    merged: dict = {}
    origin: dict[str, str] = {}
    for k, v in base.items():
        merged[k] = v
        origin[k] = "base"
    for k, v in pset.items():
        merged[k] = v
        origin[k] = "preset"
    for k, v in meth.items():
        merged[k] = v
        origin[k] = "method"

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

    return merged, origin


# ── Field metadata for the frontend ────────────────────────────


def get_field_type(key: str, value: Any) -> str:
    """Map a Python value + key to a frontend widget type."""
    if key == "attn_mode":
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


def build_merged_config(variant: str, preset: str) -> dict:
    """Build the full merged config with field metadata for the frontend.

    Returns a dict with ``fields`` (list of FieldMeta-like dicts),
    ``variant``, and ``preset``.
    """
    merged, origin = merged_gui_variant_preset(variant, preset)
    fields = []
    for key, value in sorted(merged.items()):
        if key in _SKIP:
            continue
        ftype = get_field_type(key, value)
        fields.append(
            {
                "key": key,
                "value": value,
                "origin": origin.get(key, "base"),
                "field_type": ftype,
                "group": _K2G.get(key),
                "is_basic": key in _BASIC,
                "is_virtual": key in _VIRTUAL_KEYS,
                "options": _ATTN_MODES if ftype == "select" else None,
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
        return _load(GUI_METHODS_DIR / f"{variant}.toml")
    raise ValueError(f"Unknown layer: {layer}")


def write_layer(
    layer: str, data: dict, variant: str = "lora", preset: str = "default"
) -> None:
    """Write a single config layer. Only method layer is user-editable."""
    if layer == "method":
        _safe_variant(variant)
        path = GUI_METHODS_DIR / f"{variant}.toml"
        _save(path, data)
    elif layer == "base":
        _save(CONFIGS_DIR / "base.toml", data)
    elif layer == "preset":
        # Presets are in one file — update the specific section
        if PRESETS_FILE.exists():
            all_presets = toml.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        else:
            all_presets = {}
        all_presets[preset] = data
        _save(PRESETS_FILE, all_presets)
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
    return errors


def save_variant_config(variant: str, data: dict) -> None:
    """Save user-edited config fields back to the variant TOML.

    Separates virtual keys (written into [[datasets]] blocks) from
    flat TOML keys. Skips ``_SKIP`` keys and keys that match the
    merged base+preset values (no-op saves).
    """
    _safe_variant(variant)
    data = dict(data)  # copy to avoid mutating caller's dict
    path = GUI_METHODS_DIR / f"{variant}.toml"
    current = _load(path)

    # Handle virtual keys
    use_valid = data.pop("use_valid", None)
    vsn = data.pop("validation_split_num", None)
    if use_valid is not None:
        base = _load(CONFIGS_DIR / "base.toml")
        base_vsn = _base_validation_split_num(base)
        apply_validation_choice(current, use_valid, vsn, base_vsn)

    # Write flat keys
    for key, value in data.items():
        if key in _SKIP:
            continue
        current[key] = value

    _save(path, current)
