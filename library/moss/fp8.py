"""Qwen3.5-9B text encoder → one fp8 safetensors in ComfyUI's mixed-precision format.

Every decoder-layer Linear (``language_model.layers.*``, 6.92 B params) is stored as
``float8_e4m3fn`` with one fp32 ``weight_scale`` per tensor (amax / 448) and a
``comfy_quant`` JSON (``float8_e4m3fn``, ``full_precision_matrix_mult``: the weight
is dequantised for a bf16 matmul, ComfyUI does not quantise the activations). The
token embedding, norms, GDN ``conv1d`` / ``A_log`` / ``dt_bias`` and the vision
tower keep their checkpoint dtype. Key names are the HF checkpoint's (``language_model.*``,
``visual.*``), which ``MOSSImageCLIPLoader`` reads as-is.

bench/moss_te_fp8 (20261021-1113): per-tensor and per-channel scales give the same
embedding error (rel L2 ~0.10), w8a8 is worse and no faster.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

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


def quantize_te(src_dir: str | Path, out_path: str | Path, log=print) -> dict:
    """``text_encoder/`` (sharded HF safetensors) → ``out_path``. Returns counts."""
    src_dir, out_path = Path(src_dir), Path(out_path)
    index = json.loads((src_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    conf = torch.tensor(list(json.dumps(QUANT_CONF).encode()), dtype=torch.uint8)
    out, n_q = {}, 0
    for shard in shards:
        with safe_open(str(src_dir / shard), "pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if is_quantized_linear(key):
                    q, scale = quantize_weight(t)
                    prefix = key[: -len("weight")]
                    out[key] = q
                    out[f"{prefix}weight_scale"] = scale
                    out[f"{prefix}comfy_quant"] = conf.clone()
                    n_q += 1
                else:
                    out[key] = t
        log(f"[fp8] {shard}: {len(out)} tensors so far, {n_q} quantized")
    if n_q != N_LINEARS:
        raise ValueError(f"quantized {n_q} linears, expected {N_LINEARS}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        out,
        str(out_path),
        metadata={
            "format": "pt",
            "source": "OpenMOSS-Team/MOSS-Image-Base text_encoder (Qwen3.5-9B)",
            "quantization": json.dumps(QUANT_CONF),
        },
    )
    size = out_path.stat().st_size / 2**30
    log(f"[fp8] {n_q} linears quantized -> {out_path} ({size:.2f} GiB)")
    return {"quantized": n_q, "tensors": len(out), "gib": round(size, 2)}
