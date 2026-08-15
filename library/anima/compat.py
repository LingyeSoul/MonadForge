"""Anima 28/40-block capability and adapter-identity validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

from library.anima.checkpoint import (
    AnimaCheckpointLayout,
    anima_checkpoint_sha256,
    inspect_anima_checkpoint,
)

ADAPTER_ARCH_KEYS = (
    "ss_anima_arch",
    "ss_anima_num_blocks",
    "ss_anima_model_channels",
)
ANIMA29_PREVIEW_V1_SHA256 = (
    "0b3020d1b906155f7eb30667622723e87160632c8c7a5f1c93bdce685f2a346d"
)
_CHANNEL_STATS_40_PATH = (
    Path(__file__).resolve().parents[2]
    / "networks"
    / "calibration"
    / "channel_stats_anima40.safetensors"
)


@dataclass(frozen=True)
class AnimaCompatibility:
    supported: bool
    profile: str
    blockers: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "profile": self.profile,
            "blockers": list(self.blockers),
        }


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _network_options(config: Any) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if isinstance(config, Mapping):
        options.update(config)
    else:
        options.update(vars(config))
    for item in _get(config, "network_args", None) or []:
        if isinstance(item, str) and "=" in item:
            key, value = item.split("=", 1)
            options[key] = value
    return options


def classify_anima_training(config: Any) -> AnimaCompatibility:
    """Classify the effective config against the certified 40-block profiles."""

    options = _network_options(config)
    blockers: list[str] = []
    module = str(_get(config, "network_module", "") or "").strip()
    method = str(_get(config, "method", "") or "").strip().lower()

    if module != "networks.lora_anima":
        blockers.append(f"network_module={module or '<unset>'}")

    unsupported_methods = {
        "byg",
        "controlnet",
        "easycontrol",
        "spd",
        "turbo",
        "distill_mod",
        "soft_tokens",
        "chimera",
        "chimera_hydra",
    }
    if method in unsupported_methods:
        blockers.append(f"method={method}")

    bool_features = {
        "use_repa": "REPA",
        "use_lokr": "LoKr",
        "use_glokr": "GLoKr",
        "use_loha": "LoHa",
        "use_dylora": "DyLoRA",
        "use_ve": "VeRA",
        "use_ortho_init": "OrthoInit",
        "use_chimera_hydra": "ChimeraHydra",
        "use_controlnet": "ControlNet",
        "use_easycontrol": "EasyControl",
        "use_byg": "BYG",
        "pgraft": "P-GRAFT",
        "train_llm_adapter": "train_llm_adapter",
        "network_train_text_encoder_only": "text-encoder-only training",
    }
    for key, label in bool_features.items():
        if _bool(options.get(key, _get(config, key, False))):
            blockers.append(label)

    if not _bool(
        options.get(
            "network_train_unet_only",
            _get(config, "network_train_unet_only", True),
        )
    ):
        blockers.append("text encoder training")

    try:
        if float(_get(config, "vr_loss_weight", 0.0) or 0.0) > 0:
            blockers.append("VR loss")
    except (TypeError, ValueError):
        blockers.append("invalid vr_loss_weight")

    base_compute = str(_get(config, "base_compute", "bf16") or "bf16").strip().lower()
    if base_compute not in {"bf16", "fp16", "none", "off", ""}:
        blockers.append(f"ConvRot/base_compute={base_compute}")

    moe_style = options.get("use_moe_style", False)
    if str(moe_style).strip().lower() not in {"", "0", "false", "none"}:
        blockers.append(f"Hydra/use_moe_style={moe_style}")
    if _bool(options.get("route_per_layer", False)):
        blockers.append("Hydra/route_per_layer")
    router_source = str(options.get("router_source", "none") or "none").lower()
    if router_source != "none":
        blockers.append(f"Hydra/router_source={router_source}")
    try:
        if int(options.get("step_expert_K", 0) or 0) > 1:
            blockers.append("Turbo step experts")
    except (TypeError, ValueError):
        blockers.append("invalid step_expert_K")

    timestep_mask = _bool(options.get("use_timestep_mask", False))
    use_ortho = _bool(options.get("use_ortho", False))
    down_init = str(options.get("down_init", "kaiming") or "kaiming").lower()
    if not timestep_mask and not use_ortho and down_init == "kaiming":
        profile = "plain_lora"
    elif timestep_mask and use_ortho and down_init == "kaiming":
        profile = "tlora_ortho"
    else:
        profile = "unsupported"
        blockers.append(
            "profile must be Plain LoRA or T-LoRA + OrthoLoRA "
            f"(use_timestep_mask={timestep_mask}, use_ortho={use_ortho}, "
            f"down_init={down_init})"
        )

    supported = not blockers and profile in {"plain_lora", "tlora_ortho"}
    return AnimaCompatibility(
        supported=supported,
        profile=profile if supported else "unsupported",
        blockers=tuple(dict.fromkeys(blockers)),
    )


def compatibility_for_layout(
    config: Any, layout: AnimaCheckpointLayout
) -> AnimaCompatibility:
    classified = classify_anima_training(config)
    if layout.num_blocks == 28:
        return AnimaCompatibility(True, classified.profile)
    return classified


def require_training_compatibility(config: Any, layout: AnimaCheckpointLayout) -> str:
    result = compatibility_for_layout(config, layout)
    if not result.supported:
        raise ValueError(
            f"{layout.variant} ({layout.num_blocks} blocks) does not support this training config: "
            + ", ".join(result.blockers)
        )
    return result.profile


def validate_channel_stats_compatibility(
    config: Any,
    layout: AnimaCheckpointLayout,
    base_sha256: str,
) -> None:
    """Validate the 40-block channel-scaling asset without reading tensors."""

    try:
        alpha = float(_get(config, "channel_scaling_alpha", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid channel_scaling_alpha") from exc
    if layout.num_blocks != 40 or alpha == 0.0:
        return
    if not _CHANNEL_STATS_40_PATH.is_file():
        raise ValueError(
            f"40-block channel stats are missing: {_CHANNEL_STATS_40_PATH}"
        )

    with safe_open(str(_CHANNEL_STATS_40_PATH), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})
        keys = list(handle.keys())
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
    expected = {
        "anima_stats_schema": "1",
        "anima_arch": layout.arch,
        "anima_num_blocks": str(layout.num_blocks),
        "anima_model_channels": str(layout.model_channels),
        "anima_base_sha256": base_sha256,
    }
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if str(metadata.get(key, "")).lower() != str(value).lower()
    ]
    block_indices: set[int] = set()
    block_suffixes: dict[int, set[str]] = {}
    for key, shape in shapes.items():
        marker = "lora_unet_blocks_"
        if not key.startswith(marker):
            continue
        rest = key[len(marker) :]
        index_text, separator, suffix = rest.partition("_")
        if not separator or not index_text.isdigit():
            continue
        index = int(index_text)
        block_indices.add(index)
        block_suffixes.setdefault(index, set()).add(suffix)
        if len(shape) != 1 or shape[0] <= 0:
            mismatches.append(f"{key} has invalid vector shape {shape}")
    if not mismatches:
        from safetensors.torch import load_file

        tensors = load_file(str(_CHANNEL_STATS_40_PATH), device="cpu")
        nonfinite = sorted(key for key, value in tensors.items() if not value.isfinite().all())
        if nonfinite:
            mismatches.append(f"non-finite vectors: {nonfinite[:5]}")
    if block_indices != set(range(40)):
        mismatches.append(
            f"block coverage is {sorted(block_indices)}, expected exactly 0..39"
        )
    if block_suffixes:
        reference = block_suffixes.get(0, set())
        for index in range(1, 40):
            if block_suffixes.get(index, set()) != reference:
                mismatches.append(f"block {index} stats keys differ from block 0")
                break
    if mismatches:
        raise ValueError("invalid 40-block channel stats: " + "; ".join(mismatches[:8]))


def read_adapter_metadata(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        return {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def adapter_identity_metadata(
    layout: AnimaCheckpointLayout, base_sha256: str
) -> dict[str, str]:
    return {
        "ss_anima_arch": layout.arch,
        "ss_anima_num_blocks": str(layout.num_blocks),
        "ss_anima_model_channels": str(layout.model_channels),
        "ss_new_sd_model_hash": base_sha256,
    }


def validate_adapter_compatibility(
    path: str | Path,
    layout: AnimaCheckpointLayout,
    base_sha256: str,
) -> dict[str, str]:
    """Reject cross-architecture/base adapters before loading their tensors."""

    metadata = read_adapter_metadata(path)
    present = [key in metadata for key in ADAPTER_ARCH_KEYS]
    if any(present) and not all(present):
        missing = [key for key, exists in zip(ADAPTER_ARCH_KEYS, present) if not exists]
        raise ValueError(
            f"Adapter has incomplete Anima architecture metadata: {missing}"
        )
    if not any(present):
        if layout.num_blocks == 40:
            raise ValueError(
                "40-block Anima requires adapter architecture metadata; "
                f"legacy/unstamped adapter is not allowed: {path}"
            )
        return metadata

    expected = {
        "ss_anima_arch": layout.arch,
        "ss_anima_num_blocks": str(layout.num_blocks),
        "ss_anima_model_channels": str(layout.model_channels),
    }
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    adapter_hash = metadata.get("ss_new_sd_model_hash")
    if adapter_hash is None:
        mismatches.append("ss_new_sd_model_hash is missing")
    elif adapter_hash.lower() != base_sha256.lower():
        mismatches.append(
            f"ss_new_sd_model_hash={adapter_hash!r} (expected {base_sha256!r})"
        )
    if mismatches:
        raise ValueError(
            f"Adapter/base Anima identity mismatch for {path}: " + "; ".join(mismatches)
        )
    return metadata


def validate_resume_model_signature(
    state: Mapping[str, Any],
    *,
    expected_signature: str | None,
    num_blocks: int | None,
) -> None:
    """Validate a resume sidecar against the selected Anima checkpoint."""

    if not expected_signature or num_blocks not in {28, 40}:
        return
    actual = state.get("anima_model_signature")
    if actual is None:
        if num_blocks == 40:
            raise ValueError(
                "40-block Anima resume state is missing anima_model_signature"
            )
        return
    if str(actual) != str(expected_signature):
        raise ValueError(
            "resume state Anima model signature mismatch: "
            f"state={actual}, expected={expected_signature}"
        )


def preflight_anima_training(
    config: Any,
    checkpoint_path: str | Path,
    *,
    adapter_paths: tuple[str | Path, ...] = (),
    raise_on_blockers: bool = True,
) -> tuple[AnimaCheckpointLayout, str, AnimaCompatibility]:
    """Run the shared model/capability/adapter preflight for all entry points."""

    layout = inspect_anima_checkpoint(checkpoint_path)
    base_sha256 = anima_checkpoint_sha256(checkpoint_path)
    classified = compatibility_for_layout(config, layout)
    blockers = list(classified.blockers)
    if layout.num_blocks == 40 and base_sha256.lower() != ANIMA29_PREVIEW_V1_SHA256:
        blockers.append(
            "uncertified 40-block checkpoint: "
            f"sha256={base_sha256}, expected={ANIMA29_PREVIEW_V1_SHA256}"
        )
    try:
        validate_channel_stats_compatibility(config, layout, base_sha256)
    except ValueError as exc:
        blockers.append(str(exc))
    for adapter_path in adapter_paths:
        try:
            validate_adapter_compatibility(adapter_path, layout, base_sha256)
        except (FileNotFoundError, OSError, ValueError) as exc:
            blockers.append(str(exc))

    result = AnimaCompatibility(
        supported=classified.supported and not blockers,
        profile=(
            classified.profile if classified.supported and not blockers else "unsupported"
        ),
        blockers=tuple(dict.fromkeys(blockers)),
    )
    if raise_on_blockers and not result.supported:
        raise ValueError(
            f"{layout.variant} ({layout.num_blocks} blocks) is incompatible with "
            "this training run: " + "; ".join(result.blockers)
        )
    return layout, base_sha256, result
