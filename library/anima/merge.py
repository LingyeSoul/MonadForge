"""Bake a LoRA adapter into the base DiT and save as a standalone safetensors.

The merged output is a standalone DiT checkpoint (ComfyUI-compatible, `net.`
prefixed) that reproduces LoRA+base inference without needing the adapter at
load time.

Supported: plain LoRA, OrthoLoRA, T-LoRA, LoKR, VeRA, DyLoRA, GLoKr.
(T-LoRA's timestep mask is training-only — inference already runs full rank,
so baking is bit-equivalent. GLoKr bakes with weight-REPLACEMENT semantics —
its per-module ``merge_to`` copies the BoRA-renormalized weight rather than
adding a delta.)

Not supported (refuse by default; ``allow_partial=True`` to drop and proceed):
  - HydraLoRA moe     (layer-local router can't be baked under static weights)
  - postfix / prefix  (cross-attn KV splice, not a weight delta)

Same merge path as train.py's --base_weights warm-start. The CLI shell over
this lives in ``scripts/merge_to_dit.py``.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import torch

from library.anima import weights as anima_weights
from library.anima.checkpoint import anima_checkpoint_sha256, inspect_anima_checkpoint
from library.anima.compat import validate_adapter_compatibility

logger = logging.getLogger(__name__)


# Non-bakeable marker (substring of a safetensors key) → human-readable kind.
_NON_BAKEABLE_MARKERS: dict[str, str] = {
    ".lora_up_weight": "HydraLoRA stacked (per-layer router)",
    ".lora_ups.": "HydraLoRA split (per-layer router) / step-expert turbo (per-step heads)",
    "postfix_": "postfix (cross-attn KV splice)",
    "prefix_": "prefix (cross-attn KV splice)",
}

DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class NonBakeableError(RuntimeError):
    """Raised when an adapter carries components that can't be folded into DiT
    Linear weights and ``allow_partial`` was not set."""


def pick_latest_adapter(adapter_dir: Path) -> Path:
    """Latest supported adapter weight in ``adapter_dir`` that is bakeable.

    Skips ``*_moe.safetensors`` (HydraLoRA router-live), ``*.bak.*`` (backups),
    and any file whose name contains ``postfix`` / ``prefix`` (those are
    separate non-weight-delta adapters).
    """
    from library.io.output_layout import discover_weights

    candidates = [
        f
        for f in discover_weights(adapter_dir)
        if "postfix" not in f.name.lower() and "prefix" not in f.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No bakeable adapter weights found in {adapter_dir} "
            "(excludes *_moe, *postfix*, *prefix*, and *.bak.*)"
        )
    return candidates[0]


def scan_non_bakeable_keys(weights_sd: dict) -> dict[str, int]:
    """Return ``{kind: count}`` for any key that matches a non-bakeable marker."""
    found: dict[str, int] = {}
    for key in weights_sd.keys():
        for marker, kind in _NON_BAKEABLE_MARKERS.items():
            if marker in key:
                found[kind] = found.get(kind, 0) + 1
                break
    return found


def read_adapter_metadata(adapter: Path) -> dict[str, str]:
    """Read safetensors metadata without loading adapter tensors."""
    if adapter.suffix.lower() != ".safetensors":
        return {}
    from safetensors import safe_open

    with safe_open(str(adapter), framework="pt") as handle:
        return dict(handle.metadata() or {})


def scan_non_bakeable_metadata(metadata: dict[str, str]) -> dict[str, int]:
    """Return runtime base modes that cannot be reproduced by a plain bake."""
    from library.runtime.convrot.metadata import metadata_indicates_convrot

    if metadata_indicates_convrot(metadata):
        return {
            "ConvRot base_compute (requires a dedicated dequantize-and-fold path)": 1
        }
    return {}


def _load_adapter_state_dict(adapter: Path) -> dict:
    """Load adapter tensors without executing pickled user code."""

    if adapter.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        weights_sd = load_file(str(adapter))
    else:
        weights_sd = torch.load(
            str(adapter),
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(weights_sd, dict):
        raise TypeError(f"Adapter checkpoint must contain a state dict: {adapter}")
    nested = weights_sd.get("state_dict")
    if isinstance(nested, dict):
        weights_sd = nested
    return weights_sd


def merge_adapter_into_dit(
    adapter: Path,
    dit: Path,
    out: Path | None = None,
    *,
    multiplier: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
    device: str | None = None,
    allow_partial: bool = False,
    network_module: str = "networks.lora_anima",
) -> Path:
    """Bake ``adapter`` into the base ``dit`` and write a standalone checkpoint.

    Returns the output path written. Raises :class:`NonBakeableError` if the
    adapter carries Hydra-moe / postfix / prefix components and ``allow_partial``
    is False.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"adapter: {adapter}")

    metadata = read_adapter_metadata(adapter)
    metadata_non_bakeable = scan_non_bakeable_metadata(metadata)
    if metadata_non_bakeable and not allow_partial:
        parts = [
            f"{count} {kind}" for kind, count in metadata_non_bakeable.items()
        ]
        raise NonBakeableError(
            "Non-bakeable keys detected: "
            + ", ".join(parts)
            + ". Re-run with allow_partial to drop them and bake the LoRA portion, "
            "or retrain without these components. These cannot be folded into "
            "DiT Linear weights."
        )

    checkpoint_layout = inspect_anima_checkpoint(dit)
    base_sha256 = anima_checkpoint_sha256(dit)
    validate_adapter_compatibility(adapter, checkpoint_layout, base_sha256)
    weights_sd = _load_adapter_state_dict(adapter)
    non_bakeable = scan_non_bakeable_keys(weights_sd)
    non_bakeable.update(metadata_non_bakeable)
    if non_bakeable:
        parts = [f"{count} {kind}" for kind, count in non_bakeable.items()]
        msg = "Non-bakeable keys detected: " + ", ".join(parts) + "."
        if not allow_partial:
            raise NonBakeableError(
                msg
                + " Re-run with allow_partial to drop them and bake the LoRA portion, "
                "or retrain without these components. These cannot be folded into DiT Linear weights."
            )
        logger.warning(
            msg + " allow_partial set; these components will be absent from the merged DiT."
        )

    logger.info(f"loading base DiT: {dit}")
    unet = anima_weights.load_anima_model(
        device=device,
        dit_path=str(dit),
        attn_mode="torch",  # merge never runs a forward pass
        loading_device=device,
        dit_weight_dtype=dtype,
        checkpoint_layout=checkpoint_layout,
    )

    logger.info(f"building adapter network from weights (multiplier={multiplier})")
    network_mod = importlib.import_module(network_module)
    network, weights_sd = network_mod.create_network_from_weights(
        multiplier, str(adapter), None, None, unet, for_inference=True
    )

    logger.info("merging adapter into DiT")
    network.merge_to(None, unet, weights_sd, dtype, device)

    out = out or adapter.with_name(adapter.stem + "_merged.safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ss_merged_from": adapter.name,
        "ss_merge_multiplier": str(multiplier),
        "ss_base_dit": dit.name,
    }
    logger.info(f"saving merged DiT: {out}")
    anima_weights.save_anima_model(str(out), unet.state_dict(), metadata, dtype=dtype)
    logger.info("done")
    return out
