"""Prequantized ConvRot weight checkpoint load/save (P0-C).

Native layout (``anima_lora_convrot_prequant_v1``)
-------------------------------------------------
Per Linear named by DiT ``original_name`` (e.g. ``blocks.0.mlp.layer1``):

* ``{name}.weight`` — ``int8`` ``[out, in]`` already in **rotated** domain
* ``{name}.scale`` — ``float32`` ``[out]`` per-output-channel absmax scale

Safetensors metadata (string map):

* ``format`` = ``anima_lora_convrot_prequant_v1``
* ``group_size`` = power-of-two group used for RHT (e.g. ``256``)
* ``rht`` = ``sylvester`` (this repo's current online path)
* optional ``mode`` = ``w8a16`` / ``w8a8`` (informational)

Comfy-style aliases (best-effort, no full MixedPrecisionOps dependency)
-----------------------------------------------------------------------
* ``{name}.weight`` + ``{name}.weight_scale``
* optional ``{name}.comfy_quant`` UTF-8 JSON blob with ``group_size`` / ``format``

This path removes **online weight RHT+quant at apply time**. It does **not**
remove per-step activation RHT.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from library.runtime.convrot.quant import rotate_and_quantize_weight
from library.runtime.convrot.rht import assert_group_divides

logger = logging.getLogger(__name__)

FORMAT_V1 = "anima_lora_convrot_prequant_v1"
_META_FORMAT = "format"
_META_GROUP = "group_size"
_META_RHT = "rht"
_META_MODE = "mode"
_META_LAYER_COUNT = "layer_count"

_WEIGHT_SUFFIXES = (".weight",)
_SCALE_SUFFIXES = (".scale", ".weight_scale")


@dataclass(frozen=True)
class PrequantLayerPayload:
    name: str
    quantized_weight: torch.Tensor  # int8 [out, in]
    scale: torch.Tensor  # float32 [out]

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(self.quantized_weight.shape[0]),
            int(self.quantized_weight.shape[1]),
        )


@dataclass
class PrequantCheckpoint:
    """In-memory index of prequant layers + file metadata."""

    layers: dict[str, PrequantLayerPayload]
    group_size: int | None
    format: str | None
    rht: str | None
    mode: str | None
    path: str | None = None
    raw_metadata: dict[str, str] | None = None

    def get(self, original_name: str) -> PrequantLayerPayload | None:
        for key in _candidate_layer_names(original_name):
            hit = self.layers.get(key)
            if hit is not None:
                return hit
        return None

    @property
    def layer_count(self) -> int:
        return len(self.layers)


def _candidate_layer_names(name: str) -> list[str]:
    name = str(name).strip().replace("_orig_mod.", "")
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    add(name)
    prefixes = (
        "model.",
        "diffusion_model.",
        "model.diffusion_model.",
    )
    for p in prefixes:
        if name.startswith(p):
            add(name[len(p) :])
        else:
            add(p + name)
    return out


def _decode_meta(meta: Mapping[str, Any] | None) -> dict[str, str]:
    if not meta:
        return {}
    out: dict[str, str] = {}
    for k, v in meta.items():
        out[str(k)] = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
    return out


def _parse_group_size(meta: Mapping[str, str], tensors: Mapping[str, torch.Tensor]) -> int | None:
    raw = meta.get(_META_GROUP) or meta.get("convrot_group_size")
    if raw:
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"invalid group_size metadata={raw!r}") from exc
    # Best-effort: scan first comfy_quant blob.
    for key, tensor in tensors.items():
        if not key.endswith(".comfy_quant"):
            continue
        try:
            blob = bytes(tensor.detach().cpu().to(torch.uint8).tolist()).decode("utf-8")
            conf = json.loads(blob)
            if "group_size" in conf:
                return int(conf["group_size"])
            if "convrot_group_size" in conf:
                return int(conf["convrot_group_size"])
        except Exception:
            continue
    return None


def _pair_tensors(
    state: Mapping[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Map layer name -> (weight_int8, scale_f32)."""
    keys = list(state.keys())
    weight_keys = [k for k in keys if any(k.endswith(s) for s in _WEIGHT_SUFFIXES)]
    # Prefer longer / more specific scale suffix when both exist.
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for wk in weight_keys:
        if not wk.endswith(".weight"):
            continue
        base = wk[: -len(".weight")]
        scale = None
        for suf in _SCALE_SUFFIXES:
            sk = base + suf
            if sk in state:
                scale = state[sk]
                break
        if scale is None:
            continue
        w = state[wk]
        if w.dtype not in (torch.int8, torch.uint8):
            # Skip non-quant dense weights (e.g. leftover bf16 tensors).
            continue
        if w.dtype is torch.uint8:
            # Interpret as two's-complement int8 storage if needed.
            w = w.view(torch.int8)
        if scale.dim() == 2 and scale.shape[-1] == 1:
            scale = scale.reshape(-1)
        if scale.dim() != 1:
            raise ValueError(
                f"prequant scale for {base!r} must be 1D [out], got {tuple(scale.shape)}"
            )
        if w.dim() != 2:
            raise ValueError(
                f"prequant weight for {base!r} must be 2D [out,in], got {tuple(w.shape)}"
            )
        if scale.shape[0] != w.shape[0]:
            raise ValueError(
                f"prequant scale/out mismatch for {base!r}: "
                f"weight {tuple(w.shape)} scale {tuple(scale.shape)}"
            )
        pairs[base] = (w.contiguous(), scale.to(torch.float32).contiguous())
    return pairs


def load_prequant_checkpoint(path: str | Path) -> PrequantCheckpoint:
    """Load a prequant safetensors (or ``.pt`` dict) into an index."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"convrot prequant path not found: {path}")

    meta: dict[str, str] = {}
    if path.suffix.lower() in {".safetensors", ".sft"}:
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            meta = _decode_meta(handle.metadata())
        state = load_file(str(path), device="cpu")
    else:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(obj, dict):
            raise TypeError(f"unsupported prequant torch object type={type(obj)}")
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
            raw_meta = obj.get("metadata") or obj.get("meta") or {}
            meta = {str(k): str(v) for k, v in dict(raw_meta).items()}
        else:
            state = obj
            meta = {}

    pairs = _pair_tensors(state)
    if not pairs:
        raise ValueError(
            f"no int8 weight+scale pairs found in {path}; "
            f"expected '{{name}}.weight' (int8) + '{{name}}.scale|weight_scale'"
        )

    group_size = _parse_group_size(meta, state)
    layers = {
        name: PrequantLayerPayload(
            name=name,
            quantized_weight=w,
            scale=s,
        )
        for name, (w, s) in pairs.items()
    }
    ckpt = PrequantCheckpoint(
        layers=layers,
        group_size=group_size,
        format=meta.get(_META_FORMAT),
        rht=meta.get(_META_RHT),
        mode=meta.get(_META_MODE),
        path=str(path),
        raw_metadata=meta or None,
    )
    logger.info(
        "[convrot-prequant] loaded %s layers=%d group_size=%s format=%s path=%s",
        path.name,
        ckpt.layer_count,
        ckpt.group_size,
        ckpt.format,
        path,
    )
    return ckpt


def save_prequant_checkpoint(
    path: str | Path,
    layers: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | Iterable[PrequantLayerPayload],
    *,
    group_size: int,
    mode: str | None = None,
    rht: str = "sylvester",
    metadata_extra: Mapping[str, str] | None = None,
) -> Path:
    """Write native v1 prequant checkpoint.

    ``layers`` values are ``(quantized_weight int8, scale float32)`` or payloads.
    """
    assert_group_divides(group_size, group_size)  # power-of-two check
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, torch.Tensor] = {}
    if isinstance(layers, Mapping):
        items = [
            PrequantLayerPayload(name=str(k), quantized_weight=v[0], scale=v[1])
            for k, v in layers.items()
        ]
    else:
        items = list(layers)

    for item in items:
        w = item.quantized_weight
        s = item.scale
        if w.dtype is not torch.int8:
            raise TypeError(f"{item.name}: weight must be int8, got {w.dtype}")
        if s.dim() != 1 or s.shape[0] != w.shape[0]:
            raise ValueError(f"{item.name}: bad scale shape {tuple(s.shape)}")
        assert_group_divides(int(w.shape[1]), group_size)
        state[f"{item.name}.weight"] = w.detach().cpu().contiguous()
        state[f"{item.name}.scale"] = s.detach().cpu().to(torch.float32).contiguous()

    meta = {
        _META_FORMAT: FORMAT_V1,
        _META_GROUP: str(int(group_size)),
        _META_RHT: str(rht),
        _META_LAYER_COUNT: str(len(items)),
    }
    if mode:
        meta[_META_MODE] = str(mode)
    if metadata_extra:
        meta.update({str(k): str(v) for k, v in metadata_extra.items()})

    if path.suffix.lower() in {".safetensors", ".sft", ""}:
        if path.suffix == "":
            path = path.with_suffix(".safetensors")
        from safetensors.torch import save_file

        save_file(state, str(path), metadata=meta)
    else:
        torch.save({"state_dict": state, "metadata": meta}, str(path))
    return path


def build_prequant_layers_from_modules(
    modules: Mapping[str, nn.Linear],
    *,
    group_size: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Online RHT+quant each Linear → native prequant tensors (CPU)."""
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, linear in modules.items():
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"{name}: expected nn.Linear")
        if linear.bias is not None:
            raise ValueError(f"{name}: bias Linear not supported for ConvRot prequant")
        q, scale = rotate_and_quantize_weight(linear.weight.detach(), group_size)
        out[str(name)] = (q.cpu().contiguous(), scale.cpu().to(torch.float32).contiguous())
    return out


def resolve_effective_group_size(
    ckpt: PrequantCheckpoint,
    *,
    requested_group_size: int,
    strict: bool = True,
) -> int:
    """Prefer file metadata group_size when present."""
    if ckpt.group_size is None:
        return int(requested_group_size)
    file_gs = int(ckpt.group_size)
    if int(requested_group_size) != file_gs:
        msg = (
            f"convrot group_size CLI={requested_group_size} differs from "
            f"prequant file={file_gs} ({ckpt.path})"
        )
        if strict:
            raise ValueError(msg + "; pass matching --convrot_group_size or fix the file")
        logger.warning("%s; using file group_size", msg)
    return file_gs
