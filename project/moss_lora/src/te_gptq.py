#!/usr/bin/env python3
"""Calibrated fp8 rounding (GPTQ on the fp8 grid) for the MOSS Qwen3.5-9B TE.

Same file as ``scripts/moss/quantize_te_fp8.py`` — the 248 decoder linears as
``float8_e4m3fn`` with the same per-tensor ``weight_scale`` (amax / 448) and
``comfy_quant`` — so it loads wherever the round-to-nearest file does. Only the
rounding changes: each linear is quantized column by column, and every column's
rounding error is pushed onto the columns not yet quantized through the inverse
Hessian ``H = Σ xᵀx`` of that linear's inputs (Frantar et al., GPTQ), so the
output error ``‖(W − Q) x‖`` is minimised on the calibration inputs, not the
weight error.

Calibration is MOSS's own TE traffic, kept apart from what the evals read:
cached captions from ``post_image_dataset/moss/*`` (``--seed``, minus the captions
``bench/moss_te_fp8/run_bench.py`` samples at its default seed) and colorize
training pages from ``--pages_cache`` (page through the vision tower + colour
caption, never a holdout page), plus the empty prompt — the uncond of every
sampling step — repeated ``--empty_repeat`` times so its few rows carry weight in
``H``. Layers run in order on the calibration hidden states, each quantized layer
feeding the next (sequential); within a layer every linear's ``H`` comes from one
pass at bf16. Per linear the log gives the output error
``tr(Δ H Δᵀ) / tr(W H Wᵀ)`` of round-to-nearest and of GPTQ.

    make daemon-run ARGS="--stall-timeout 1800 project/moss_lora/src/te_gptq.py"
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import random
import shutil
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.fp8 import FP8, LINEARS, N_LINEARS, quantize_te
from library.moss.loader import _embed_on_cpu, load_tokenizer
from library.moss.requests import variant_dir
from library.moss.scan import TE_SUFFIX, caption_sha
from vision_te import te_inputs

FP8_MAX = torch.finfo(FP8).max
OUT = "models/moss_image/fp8_gptq/qwen35_9b_fp8_e4m3fn_gptq.safetensors"


def calib_captions(per_dir: int, seed: int, exclude: set[str]) -> list[str]:
    """Cached captions (``run_bench.sample_cached``'s filter), ``per_dir`` per cache
    dir, none of ``exclude``'s files."""
    rng = random.Random(seed)
    out = []
    for d in sorted(
        glob.glob(str(resolve_under_home("post_image_dataset/moss")) + "/*/")
    ):
        files = sorted(glob.glob(f"{d}*{TE_SUFFIX}"))
        rng.shuffle(files)
        got = 0
        for f in files:
            if got >= per_dir:
                break
            if str(Path(f).resolve()) in exclude:
                continue
            with safe_open(f, "pt") as s:
                meta = s.metadata() or {}
            cap = meta.get("caption")
            if meta.get("max_text_tokens") != "4096" or cap is None:
                continue
            if caption_sha(cap) != meta.get("caption_sha256"):
                continue
            out.append(cap)
            got += 1
    return list(dict.fromkeys(out))


def bench_caption_files() -> set[str]:
    """The files ``bench/moss_te_fp8/run_bench.py`` reads at its defaults (per_dir 4,
    seed 42)."""
    spec = importlib.util.spec_from_file_location(
        "moss_te_fp8_bench", resolve_under_home("bench/moss_te_fp8/run_bench.py")
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return {str(Path(f).resolve()) for _, _, f in bench.sample_cached(4, 42)}


def calib_pages(cache: Path, n: int, seed: int) -> list[tuple[Image.Image, str, int]]:
    """``n`` training pairs of a colorize cache, half manga: ``(page, caption, w·h)``."""
    files = sorted(cache.glob(f"*{TE_SUFFIX}"))
    rng = random.Random(seed)
    rng.shuffle(files)
    manga = [f for f in files if f.name.startswith("manga__")][: n // 2]
    illus = [f for f in files if not f.name.startswith("manga__")][: n - len(manga)]
    out = []
    for f in manga + illus:
        with safe_open(str(f), "pt") as s:
            meta = s.metadata()
        with Image.open(meta["source"]) as im:
            w, h = moss_cache.target_size(*im.size, 1024)
        page = moss_cache.fit(moss_cache.load_rgb(meta["condition"]), w, h)
        out.append((page, meta["caption"], w * h))
    return out


def qdq(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (w / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8).float() * scale


def out_err(delta: torch.Tensor, H: torch.Tensor, W: torch.Tensor) -> float:
    """``tr(Δ H Δᵀ) / tr(W H Wᵀ)``: relative output energy of the error."""
    return ((delta @ H) * delta).sum().item() / ((W @ H) * W).sum().item()


def gptq(
    W: torch.Tensor, H: torch.Tensor, scale, block: int, damp: float, act_order: bool
):
    """``W`` (out, in) fp32 rounded onto the fp8 grid at ``scale``; returns the
    dequantised weight. Columns no calibration input reached keep plain rounding."""
    W, H = W.clone(), H.clone()
    n = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
    H.diagonal().add_(damp * H.diagonal().mean())
    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True
    )
    Q = torch.zeros_like(W)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1, Hi = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Q1, E1 = torch.zeros_like(W1), torch.zeros_like(W1)
        for j in range(i2 - i1):
            q = qdq(W1[:, j], scale)
            e = (W1[:, j] - q) / Hi[j, j]
            Q1[:, j], E1[:, j] = q, e
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        Q = Q[:, torch.argsort(perm)]
    return Q


@torch.no_grad()
def capture(te, inputs, device):
    """Per sample: the hidden states entering layer 0 and each layer type's kwargs
    (masks, rotary embeddings), from one forward with every decoder layer stubbed."""
    lm = te.language_model
    types = te.config.text_config.layer_types
    got = {}

    def stub(i):
        def forward(hidden_states, **kwargs):
            if i == 0:
                got["h"] = hidden_states
            got.setdefault(types[i], kwargs)
            return hidden_states

        return forward

    for i, layer in enumerate(lm.layers):
        layer.forward = stub(i)
    samples = []
    for inp in inputs:
        got.clear()
        te(**inp, use_cache=False, return_dict=True)
        samples.append((got["h"], {t: got[t] for t in set(types)}))
    for layer in lm.layers:
        del layer.forward
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_dir", type=int, default=12, help="captions per cache dir")
    ap.add_argument("--pages", type=int, default=12)
    ap.add_argument("--pages_cache", default="post_image_dataset/moss/moss_colorize_3k")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--empty_repeat", type=int, default=32)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--no_act_order", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    out = resolve_under_home(args.out)
    t0 = time.time()

    from transformers import AutoModel

    src = variant_dir("", "base") / "text_encoder"
    tok = load_tokenizer("")
    captions = calib_captions(args.per_dir, args.seed, bench_caption_files())
    pages = calib_pages(resolve_under_home(args.pages_cache), args.pages, args.seed)
    inputs = [te_inputs(tok, None, c, 0, device) for c in captions]
    inputs += [te_inputs(tok, None, "", 0, device)] * args.empty_repeat
    n_cap = len(inputs)
    inputs += [te_inputs(tok, p, c, px, device) for p, c, px in pages]
    n_txt = sum(int(i["input_ids"].shape[1]) for i in inputs[:n_cap])
    n_img = sum(int(i["input_ids"].shape[1]) for i in inputs[n_cap:])
    print(
        f"[calib] {len(captions)} captions + {args.empty_repeat}x '' ({n_txt} tokens), "
        f"{len(pages)} pages ({n_img} tokens)",
        flush=True,
    )

    te = AutoModel.from_pretrained(src, dtype=torch.bfloat16).eval()
    lm = te.language_model
    for name, child in lm.named_children():
        if name == "embed_tokens":
            _embed_on_cpu(child, device)
        elif name != "layers":
            child.to(device)
    te.visual.to(device)
    samples = capture(te, inputs, device)
    te.visual.to("cpu")
    del inputs
    torch.cuda.empty_cache()
    types = te.config.text_config.layer_types

    quantized, report = {}, {}
    for i, layer in enumerate(lm.layers):
        layer.to(device)
        lins = {
            n: m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)
        }
        assert set(lins) <= set(LINEARS), sorted(set(lins) - set(LINEARS))
        H = {
            n: torch.zeros(m.in_features, m.in_features, device=device)
            for n, m in lins.items()
        }

        def acc(h):
            def hook(_m, a):
                x = a[0].reshape(-1, a[0].shape[-1]).float()
                h.addmm_(x.T, x)

            return hook

        hooks = [m.register_forward_pre_hook(acc(H[n])) for n, m in lins.items()]
        for h, kw in samples:
            layer(h, **kw[types[i]])
        for hk in hooks:
            hk.remove()

        for n, m in lins.items():
            W = m.weight.data.float()
            scale = W.abs().amax().clamp(min=1e-12) / FP8_MAX
            Q = gptq(W, H[n], scale, args.block, args.damp, not args.no_act_order)
            rtn, err = out_err(qdq(W, scale) - W, H[n], W), out_err(Q - W, H[n], W)
            key = f"language_model.layers.{i}.{n}.weight"
            report[key] = {"rtn": rtn, "gptq": err}
            quantized[key] = (
                (Q / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8).cpu(),
                scale.cpu(),
            )
            m.weight.data = Q.to(m.weight.dtype)
        del H
        samples = [(layer(h, **kw[types[i]]), kw) for h, kw in samples]
        layer.to("cpu")
        torch.cuda.empty_cache()
        layer_keys = [k for k in report if k.startswith(f"language_model.layers.{i}.")]
        rtn = sum(report[k]["rtn"] for k in layer_keys) / len(layer_keys)
        gq = sum(report[k]["gptq"] for k in layer_keys) / len(layer_keys)
        worst = max(layer_keys, key=lambda k: report[k]["rtn"])
        print(
            f"[gptq] layer {i:2d} ({types[i][:4]}): out err rtn {rtn:.2e} -> gptq {gq:.2e} | "
            f"worst {worst.rsplit('.', 2)[-2]} {report[worst]['rtn']:.2e} -> "
            f"{report[worst]['gptq']:.2e} | {time.time() - t0:.0f}s",
            flush=True,
        )
    if len(quantized) != N_LINEARS:
        raise ValueError(f"{len(quantized)} linears quantized, expected {N_LINEARS}")
    del te, lm, layer, lins, m, samples
    torch.cuda.empty_cache()

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "config.json", out.parent / "config.json")
    quantize_te(src, out, quantized=quantized)
    (out.parent / "gptq_report.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "captions": len(captions),
                "pages": len(pages),
                "text_tokens": n_txt,
                "page_tokens": n_img,
                "linears": report,
            },
            indent=1,
        )
    )
    print(f"[gptq] wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
