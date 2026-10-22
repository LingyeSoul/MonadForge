"""Qwen3.5-9B text encoder → one fp8 safetensors in ComfyUI's mixed-precision format.

Every decoder-layer Linear (``language_model.layers.*``, 6.92 B params) is stored as
``float8_e4m3fn`` with one fp32 ``weight_scale`` per tensor (amax / 448) and a
``comfy_quant`` JSON (``float8_e4m3fn``, ``full_precision_matrix_mult``: the weight
is dequantised for a bf16 matmul, ComfyUI does not quantise the activations). The
token embedding, norms, GDN ``conv1d`` / ``A_log`` / ``dt_bias`` and the vision
tower keep their checkpoint dtype. Key names are the HF checkpoint's (``language_model.*``,
``visual.*``), which ``MOSSImageCLIPLoader`` reads as-is.

bench/moss_te_fp8 (20261021-1113): per-tensor and per-channel scales give the same
embedding error (rel L2 ~0.10), w8a8 is worse and no faster. The published file is
GPTQ-rounded on the same grid and scale (``quantized=``, from
``project/moss_lora/src/te_gptq.py``): rel L2 ~0.05 on held-out captions.

``load_fp8_text_encoder`` reads that file back into the HF ``Qwen3_5Model``: the
quantized linears become :class:`FP8Linear` (fp8 resident, dequantised to bf16 per
forward, as ComfyUI runs it), so the 32 decoder layers take 6.5 GiB on the card.
The same writer and loader take the INT8 ConvRot format (``library/moss/int8.py``),
told apart by each linear's ``comfy_quant``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from library.moss import int8

FP8 = torch.float8_e4m3fn
LINEARS = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
N_LINEARS = 24 * 5 + 8 * 4 + 32 * 3  # GDN layers, full-attention layers, MLPs
QUANT_CONF = {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}


def is_quantized_linear(key: str) -> bool:
    if not (key.startswith("language_model.layers.") and key.endswith(".weight")):
        return False
    module = key.split(".", 3)[3][: -len(".weight")]
    return module in LINEARS


def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(fp8 weight, fp32 scalar scale)``, ``w ≈ q.float() * scale``."""
    w = w.float()
    scale = w.abs().amax().clamp(min=1e-12) / torch.finfo(FP8).max
    return (w / scale).to(FP8), scale


def quantize_te(
    src_dir: str | Path,
    out_path: str | Path,
    log=print,
    quantized: dict | None = None,
    fmt: str = "fp8",
) -> dict:
    """``text_encoder/`` (sharded HF safetensors) → ``out_path``. Returns counts.
    ``quantized`` maps a linear's weight key to an already quantized ``(q, scale)``
    (a calibrated rounding) in place of round to nearest. ``fmt``: ``fp8`` or
    ``int8_convrot``."""
    src_dir, out_path = Path(src_dir), Path(out_path)
    index = json.loads((src_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    quant_conf, rtn = {
        "fp8": (QUANT_CONF, quantize_weight),
        "int8_convrot": (int8.QUANT_CONF, int8.quantize_weight),
    }[fmt]
    conf = torch.tensor(list(json.dumps(quant_conf).encode()), dtype=torch.uint8)
    out, n_q = {}, 0
    for shard in shards:
        with safe_open(str(src_dir / shard), "pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if is_quantized_linear(key):
                    q, scale = (quantized or {}).get(key) or rtn(t)
                    prefix = key[: -len("weight")]
                    out[key] = q
                    out[f"{prefix}weight_scale"] = scale
                    out[f"{prefix}comfy_quant"] = conf.clone()
                    n_q += 1
                else:
                    out[key] = t
        log(f"[{fmt}] {shard}: {len(out)} tensors so far, {n_q} quantized")
    if n_q != N_LINEARS:
        raise ValueError(f"quantized {n_q} linears, expected {N_LINEARS}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        out,
        str(out_path),
        metadata={
            "format": "pt",
            "source": "OpenMOSS-Team/MOSS-Image-Base text_encoder (Qwen3.5-9B)",
            "quantization": json.dumps(quant_conf),
        },
    )
    size = out_path.stat().st_size / 2**30
    log(f"[{fmt}] {n_q} linears quantized -> {out_path} ({size:.2f} GiB)")
    return {"quantized": n_q, "tensors": len(out), "gib": round(size, 2)}


class FP8Linear(torch.nn.Module):
    """A bias-free Linear whose weight stays ``float8_e4m3fn``; each forward
    dequantises it (``q · weight_scale`` in the input dtype) for the matmul."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.register_buffer("weight_scale", scale.float().reshape(()))

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype)
        return F.linear(x, w)


def fp8_te_path(path: str | Path | None = None, log=print) -> Path:
    """The fp8 TE file: ``path``, else the catalog row's (``moss_te_fp8``),
    fetched on first use."""
    if path:
        return Path(path)
    from library import downloads

    asset = downloads.by_id()["moss_te_fp8"]
    target = asset.dest / downloads.MOSS_TE_FP8_FILE
    if not target.is_file() or not (asset.dest / "config.json").is_file():
        log(f"[te] fetching {asset.repo} -> {asset.dest}")
        downloads.fetch(asset, log=log)
    return target


def load_fp8_text_encoder(
    path: str | Path | None = None, log=print, int8_act: bool = True
):
    """The fp8 file (with ``config.json`` beside it) → ``Qwen3_5Model`` on the CPU,
    eval mode, quantized linears as :class:`FP8Linear`. An INT8 ConvRot file loads
    as :class:`int8.Int8ConvRotLinear`, W8A8 as ComfyUI runs it unless ``int8_act``
    is off."""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    path = fp8_te_path(path, log)
    config = AutoConfig.from_pretrained(path.parent)
    with init_empty_weights(include_buffers=False):
        te = AutoModel.from_config(config, dtype=torch.bfloat16)
    state = load_file(str(path))
    n = 0
    for name, module in list(te.named_modules()):
        for cname, child in list(module.named_children()):
            key = f"{name}.{cname}.weight" if name else f"{cname}.weight"
            if isinstance(child, torch.nn.Linear) and is_quantized_linear(key):
                prefix = key[: -len("weight")]
                conf = state.pop(f"{prefix}comfy_quant", None)
                conf = json.loads(conf.numpy().tobytes()) if conf is not None else {}
                w, scale = state.pop(key), state.pop(f"{prefix}weight_scale")
                if conf.get("format") == "int8_tensorwise":
                    group = (
                        conf.get("convrot_groupsize", 0) if conf.get("convrot") else 0
                    )
                    q = int8.Int8ConvRotLinear(w, scale, group, int8_act)
                else:
                    q = FP8Linear(w, scale)
                setattr(module, cname, q)
                n += 1
    if n != N_LINEARS:
        raise ValueError(f"{path}: {n} fp8 linears, expected {N_LINEARS}")
    _, unexpected = te.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise ValueError(f"{path}: unexpected keys {unexpected[:5]}")
    meta = [k for k, p in te.named_parameters() if p.is_meta]
    if meta:
        raise ValueError(f"{path}: weights missing for {meta[:5]}")
    return te.eval()
