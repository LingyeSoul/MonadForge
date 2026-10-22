#!/usr/bin/env python3
"""Calibrated fp8 rounding (GPTQ on the fp8 grid) for the MOSS Qwen3.5-9B TE.

Writes the same file layout as ``scripts/moss/quantize_te_fp8.py`` (the 248
decoder linears as ``float8_e4m3fn``, per-tensor ``weight_scale`` amax / 448,
``comfy_quant``), so it loads wherever the round-to-nearest file does; only the
rounding changes (GPTQ against ``H = Σ xᵀx`` of each linear's inputs).

Calibration is MOSS's own TE traffic, kept apart from what the evals read:
cached captions from ``post_image_dataset/moss/*`` (``--seed``, minus the TE-bench eval
captions, ``bench/_moss.py`` ``sample_cached(4, 42)``), colorize training
pages from ``--pages_cache`` (never a holdout page), and the empty prompt (the
uncond of every sampling step) repeated ``--empty_repeat`` times so its few rows
carry weight in ``H``. Layers are quantized sequentially, each quantized layer
feeding the next; within a layer every linear's ``H`` comes from one bf16 pass.
Per linear the log gives the output error ``tr(Δ H Δᵀ) / tr(W H Wᵀ)`` of
round-to-nearest and of GPTQ.

``--manifest`` (``te_calib.py``'s) calibrates on its ``calib`` split instead: every
category weighs the same in ``H`` (each sample's ``xᵀx`` scaled by the mean category
token count over its own category's), and the empty prompt enters once at
``--empty_mass`` of one category.

``--grid int8_convrot`` rounds onto ``library/moss/int8.py``'s grid instead: the
weight and ``H`` are taken into the group-Hadamard basis (``W·R``, ``R·H·R``), the
per-row int8 scale is fixed at the rotated weight's amax / 127, and GPTQ runs there;
the layer's own forward (for the next layer's inputs) uses the rotated-back weight.
W8A8's activation rounding is not modelled unless ``--act_aware``.

Two INT8 ConvRot options target that rounding (the activation rotated and rounded
per token):

- ``--smooth α`` (SmoothQuant): before a layer is quantized, every input channel
  ``j`` of a linear group is divided by ``s_j = amax|X_j|^α / amax|W_{·j}|^(1−α)``
  (geometric mean 1) and the weight columns multiplied by it. ``s`` folds into the
  RMSNorm feeding the group (``input_layernorm`` → the attention input projections,
  ``post_attention_layernorm`` → ``gate_proj`` / ``up_proj``; stored as fp32
  ``(1 + w)/s − 1``) and into ``up_proj``'s rows for ``down_proj``. ``out_proj`` /
  ``o_proj`` are left alone.
- ``--act_aware``: ``H`` is taken from the rotated, per-token int8-rounded inputs
  ``X̃``, and GPTQ rounds ``W* = W_r·Gᵀ·(H̃ + λI)⁻¹`` (``G = X̃ᵀX_r``), the least-squares
  weight from ``X̃`` to the exact output ``X_r W_rᵀ``, so the activation rounding is
  partly compensated. ``λ`` is ``--ls_damp`` of ``H̃``'s mean diagonal, not GPTQ's
  ``--damp``: at 1e-2 the ridge shrinks ``W*`` and its own W8A8 error is ~6x round to
  nearest's. The layer's output for the next layer runs W8A8
  (``int8.Int8ConvRotLinear``). The log's ``rtn`` / ``gptq`` errors are then the
  W8A8 output error ``‖X̃Qᵀ − X_r W_rᵀ‖² / ‖X_r W_rᵀ‖²``.

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
from library.moss import int8
from library.moss.fp8 import FP8, LINEARS, N_LINEARS, quantize_te
from library.moss.loader import _embed_on_cpu, load_tokenizer
from library.moss.requests import variant_dir
from library.moss.scan import TE_SUFFIX, caption_sha
from vision_te import te_inputs

FP8_MAX = torch.finfo(FP8).max
OUT = {
    "fp8": "models/moss_image/fp8_gptq/qwen35_9b_fp8_e4m3fn_gptq.safetensors",
    "int8_convrot": "models/moss_image/int8_convrot_gptq/qwen35_9b_int8_convrot_gptq.safetensors",
}


def calib_captions(per_dir: int, seed: int, exclude: set[str]) -> list[str]:
    """Cached captions (``bench/_moss.py`` ``sample_cached``'s filter), ``per_dir`` per cache
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
    """The TE-bench eval captions' files: ``bench/_moss.py`` ``sample_cached(4, 42)``."""
    spec = importlib.util.spec_from_file_location(
        "bench_moss", resolve_under_home("bench/_moss.py")
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


def manifest_inputs(tok, path: Path, empty_mass: float, device):
    """The manifest's calib entries (+ the empty prompt) as TE inputs, with the
    per-sample ``H`` weights that give every category the same token mass."""
    entries = [
        e for e in json.loads(path.read_text())["entries"] if e["split"] == "calib"
    ]
    per_cat = {}
    for e in entries:
        per_cat[e["category"]] = per_cat.get(e["category"], 0) + e["tokens"]
    mean = sum(per_cat.values()) / len(per_cat)
    inputs, weights = [], []
    for e in entries:
        if e["image"]:
            w, h = e["width"], e["height"]
            page = moss_cache.fit(moss_cache.load_rgb(e["image"]), w, h)
            inputs.append(te_inputs(tok, page, e["text"], w * h, device))
        else:
            inputs.append(te_inputs(tok, None, e["text"], 0, device))
        weights.append(mean / per_cat[e["category"]])
    empty = te_inputs(tok, None, "", 0, device)
    inputs.append(empty)
    weights.append(empty_mass * mean / int(empty["input_ids"].shape[1]))
    for cat, n in sorted(per_cat.items()):
        print(f"[calib] {cat:14s} {n:6d} tokens, weight {mean / n:.3f}", flush=True)
    print(
        f"[calib] {len(entries)} entries + '' at {empty_mass} of a category", flush=True
    )
    return (
        inputs,
        weights,
        {"manifest": str(path), "tokens": per_cat, "empty_mass": empty_mass},
    )


# norm (or up_proj's rows) → the linears reading its output, per layer type
SMOOTH = {
    "linear_attention": {
        "input_layernorm": (
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_a",
        )
    },
    "full_attention": {
        "input_layernorm": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
    },
}
SMOOTH_MLP = {
    "post_attention_layernorm": ("mlp.gate_proj", "mlp.up_proj"),
    "mlp.up_proj": ("mlp.down_proj",),
}


@torch.no_grad()
def smooth_layer(layer, kind, samples, alpha):
    """Fold SmoothQuant scales into ``layer`` (bf16 linears, fp32 norms). Returns
    ``(exact fp32 smoothed weight per linear, new norm weight per norm)``."""
    mods = dict(layer.named_modules())
    groups = {**SMOOTH[kind], **SMOOTH_MLP}
    amax = {}

    def hook(src):
        def h(_m, a):
            x = a[0].reshape(-1, a[0].shape[-1]).abs().amax(0).float()
            amax[src] = torch.maximum(amax[src], x) if src in amax else x

        return h

    hooks = [
        mods[names[0]].register_forward_pre_hook(hook(src))
        for src, names in groups.items()
    ]
    for h, kw in samples:
        layer(h, **kw[kind])
    for hk in hooks:
        hk.remove()
    exact = {
        n: mods[n].weight.data.float() for names in groups.values() for n in names
    }
    norms = {}
    for src, names in groups.items():
        wa = torch.stack([mods[n].weight.abs().amax(0).float() for n in names]).amax(0)
        s = amax[src].clamp(min=1e-5).pow(alpha) / wa.clamp(min=1e-5).pow(1 - alpha)
        s = s / s.log().mean().exp()
        for n in names:
            exact[n] = exact[n] * s
        if src == "mlp.up_proj":
            exact[src] = exact[src] / s[:, None]
        else:
            norm = mods[src]
            norm.weight.data = (1 + norm.weight.float()) / s - 1
            norms[src] = norm.weight.data
    for n, w in exact.items():
        mods[n].weight.data = w.to(mods[n].weight.dtype)
    return exact, norms


def act_qdq(x: torch.Tensor) -> torch.Tensor:
    """Per-row int8 round trip, ``comfy_kitchen``'s activation quantizer."""
    s = (x.abs().amax(-1, keepdim=True) / int8.INT8_MAX).clamp(min=1e-30)
    return (x / s).round().clamp(-int8.INT8_MAX - 1, int8.INT8_MAX) * s


def w8a8_err(Q, Wr, Hq, G, Hr) -> float:
    """``‖X̃Qᵀ − X_r W_rᵀ‖² / ‖X_r W_rᵀ‖²`` from ``H̃ = X̃ᵀX̃``, ``G = X̃ᵀX_r``,
    ``H_r = X_rᵀX_r``."""
    ref = ((Wr @ Hr) * Wr).sum()
    return (((Q @ Hq) * Q).sum() - 2 * ((Q @ G) * Wr).sum() + ref).item() / ref.item()


def qdq(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (w / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8).float() * scale


def out_err(delta: torch.Tensor, H: torch.Tensor, W: torch.Tensor) -> float:
    """``tr(Δ H Δᵀ) / tr(W H Wᵀ)``: relative output energy of the error."""
    return ((delta @ H) * delta).sum().item() / ((W @ H) * W).sum().item()


def gptq(W: torch.Tensor, H: torch.Tensor, q, block: int, damp: float, act_order: bool):
    """``W`` (out, in) fp32 rounded column by column with ``q`` (a column → its
    dequantised rounding); returns the dequantised weight. Columns no calibration
    input reached keep plain rounding."""
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
            qj = q(W1[:, j])
            e = (W1[:, j] - qj) / Hi[j, j]
            Q1[:, j], E1[:, j] = qj, e
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
    ap.add_argument("--pages_cache", default="post_image_dataset/moss/moss_colorize_3k_ocr")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--empty_repeat", type=int, default=32)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--no_act_order", action="store_true")
    ap.add_argument("--manifest", default="", help="te_calib.py manifest (calib split)")
    ap.add_argument("--empty_mass", type=float, default=0.25)
    ap.add_argument("--grid", default="fp8", choices=tuple(OUT))
    ap.add_argument("--smooth", type=float, default=0.0, help="SmoothQuant α; 0 off")
    ap.add_argument("--act_aware", action="store_true")
    ap.add_argument("--ls_damp", type=float, default=1e-4, help="--act_aware's ridge")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if (args.smooth or args.act_aware) and args.grid != "int8_convrot":
        ap.error("--smooth / --act_aware need --grid int8_convrot")
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    out = resolve_under_home(args.out or OUT[args.grid])
    t0 = time.time()

    from transformers import AutoModel

    src = variant_dir("", "base") / "text_encoder"
    tok = load_tokenizer("")
    if args.manifest:
        inputs, weights, calib_info = manifest_inputs(
            tok, resolve_under_home(args.manifest), args.empty_mass, device
        )
    else:
        captions = calib_captions(args.per_dir, args.seed, bench_caption_files())
        pages = calib_pages(resolve_under_home(args.pages_cache), args.pages, args.seed)
        inputs = [te_inputs(tok, None, c, 0, device) for c in captions]
        inputs += [te_inputs(tok, None, "", 0, device)] * args.empty_repeat
        n_cap = len(inputs)
        inputs += [te_inputs(tok, p, c, px, device) for p, c, px in pages]
        weights = [1.0] * len(inputs)
        n_txt = sum(int(i["input_ids"].shape[1]) for i in inputs[:n_cap])
        n_img = sum(int(i["input_ids"].shape[1]) for i in inputs[n_cap:])
        calib_info = {
            "captions": len(captions),
            "pages": len(pages),
            "text_tokens": n_txt,
            "page_tokens": n_img,
        }
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

    quantized, report, replace = {}, {}, {}
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

        exact = {}
        if args.smooth:
            exact, norms = smooth_layer(layer, types[i], samples, args.smooth)
            for norm, w in norms.items():
                replace[f"language_model.layers.{i}.{norm}.weight"] = w.cpu()
        if args.act_aware:
            G = {n: torch.zeros_like(h) for n, h in H.items()}
            Hr = {n: torch.zeros_like(h) for n, h in H.items()}

        weight = [1.0]

        def acc(h, g=None, hr=None):
            def hook(_m, a):
                x = a[0].reshape(-1, a[0].shape[-1]).float()
                if g is None:
                    h.addmm_(x.T, x, alpha=weight[0])
                    return
                xr = int8.rotate(x)
                xq = act_qdq(xr)
                h.addmm_(xq.T, xq, alpha=weight[0])
                g.addmm_(xq.T, xr, alpha=weight[0])
                hr.addmm_(xr.T, xr, alpha=weight[0])

            return hook

        hooks = [
            m.register_forward_pre_hook(
                acc(H[n], G[n], Hr[n]) if args.act_aware else acc(H[n])
            )
            for n, m in lins.items()
        ]
        for (h, kw), w in zip(samples, weights):
            weight[0] = w
            layer(h, **kw[types[i]])
        for hk in hooks:
            hk.remove()

        for n, m in lins.items():
            W, Hn = exact.get(n, m.weight.data.float()), H[n]
            if args.grid == "int8_convrot":
                if args.act_aware:
                    Wr = int8.rotate(W)
                    A = Hn + args.ls_damp * Hn.diagonal().mean() * torch.eye(
                        Hn.shape[0], device=device
                    )
                    W = torch.cholesky_solve(G[n] @ Wr.T, torch.linalg.cholesky(A)).T
                else:
                    W, Hn = int8.rotate(W), int8.rotate(int8.rotate(Hn).T)
                scale = int8.row_scale(W)
                Q = gptq(
                    W,
                    Hn,
                    lambda c, s=scale: int8.qdq(c, s),
                    args.block,
                    args.damp,
                    not args.no_act_order,
                )
                q = (Q / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
                quantized_w = (q.cpu(), scale[:, None].cpu())
                if args.act_aware:
                    rtn_q = int8.qdq(Wr, int8.row_scale(Wr))
                    rtn = w8a8_err(rtn_q, Wr, Hn, G[n], Hr[n])
                    err = w8a8_err(Q, Wr, Hn, G[n], Hr[n])
                    parent, _, child = n.rpartition(".")
                    setattr(
                        layer.get_submodule(parent),
                        child,
                        int8.Int8ConvRotLinear(
                            q.to(device), scale[:, None], int8.GROUP, act=True
                        ),
                    )
                else:
                    rtn = out_err(int8.qdq(W, scale) - W, Hn, W)
                    err = out_err(Q - W, Hn, W)
                    m.weight.data = int8.rotate(Q).to(m.weight.dtype)
            else:
                scale = W.abs().amax().clamp(min=1e-12) / FP8_MAX
                Q = gptq(
                    W,
                    Hn,
                    lambda c, s=scale: qdq(c, s),
                    args.block,
                    args.damp,
                    not args.no_act_order,
                )
                rtn = out_err(qdq(W, scale) - W, Hn, W)
                err = out_err(Q - W, Hn, W)
                quantized_w = (
                    (Q / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8).cpu(),
                    scale.cpu(),
                )
                m.weight.data = Q.to(m.weight.dtype)
            key = f"language_model.layers.{i}.{n}.weight"
            report[key] = {"rtn": rtn, "gptq": err}
            quantized[key] = quantized_w
        if args.act_aware:
            del G, Hr
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
    quantize_te(src, out, quantized=quantized, fmt=args.grid, replace=replace)
    (out.parent / "gptq_report.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "calib": calib_info,
                "linears": report,
            },
            indent=1,
        )
    )
    print(f"[gptq] wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
