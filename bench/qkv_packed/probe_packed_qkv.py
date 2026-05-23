"""Microbench: split (current) vs packed FA2 for Anima self-attention.

Question (native-token-no-pad discussion): the self-attn block has a fused
qkv_proj, but q_norm/k_norm (RMSNorm) + RoPE sit between projection and the flash
call and split QKV into independent tensors. Can preserving the packed
(B,S,3,H,D) layout into flash_attn_qkvpacked_func buy anything?

v2 — rewritten after review. Fixes:
  * all param grads (not just x.grad) zeroed before every backward, incl. warmup
  * fresh Params/input per measurement so peak memory is isolated
  * backward gradient parity (x + every param), not just forward
  * THREE separate questions benched independently:
      [A] FA2 kernel only      flash_attn_func vs flash_attn_qkvpacked_func
      [B] norm+RoPE only       split(2-stream) vs packed-mask(3-stream) vs packed-stack
      [C] full attention block proj+norm+rope+attn+out_proj
  * full block tests both strided-V and contiguous-V split paths, and both packed
    emulations (mask / stack)
  * speedup ratios reported vs the split baseline

The packed-mask path overpays on V (rsqrt+rotate on V then masks it); the
packed-stack path pays a repack copy. Both are the only pure-PyTorch options —
removing the V cost needs a custom fused Q/K-only kernel. Benching both bounds it.

Run:  python bench/qkv_packed/probe_packed_qkv.py
"""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch

from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

N_HEADS = 16
HEAD_DIM = 128
INNER = N_HEADS * HEAD_DIM  # 2048
EPS = 1e-6
ROPE_THETA = 10000.0
SCALE = HEAD_DIM**-0.5


class Params(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        std = 1.0 / math.sqrt(INNER)
        self.qkv_w = torch.nn.Parameter(torch.randn(3 * INNER, INNER, device=device, dtype=dtype) * std)
        self.out_w = torch.nn.Parameter(torch.randn(INNER, INNER, device=device, dtype=dtype) * std)
        # RMSNorm weights are fp32 params (matches library RMSNorm). Jitter off 1.0
        # so q_w != k_w != ones — otherwise the V-waste is invisible to parity.
        self.q_w = torch.nn.Parameter(1.0 + 0.02 * torch.randn(HEAD_DIM, device=device, dtype=torch.float32))
        self.k_w = torch.nn.Parameter(1.0 + 0.02 * torch.randn(HEAD_DIM, device=device, dtype=torch.float32))


def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope(seq_len, device, dtype):
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)  # (S, D) each


def rmsnorm(x, w):  # x: (...,D) bf16; w: (D,) fp32
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)
    return (out * w).to(x.dtype)


# ------- norm+RoPE producers (the part that differs) ------------------------
def normrope_split(qkv, p, cos, sin):
    """Current path: norm+RoPE on q,k only; V untouched (Identity + no RoPE)."""
    q, k, v = qkv.unbind(dim=-3)
    q = rmsnorm(q, p.q_w)
    k = rmsnorm(k, p.k_w)
    cq, sq = cos[None, :, None, :], sin[None, :, None, :]
    q = q * cq + _rotate_half(q) * sq
    k = k * cq + _rotate_half(k) * sq
    return q, k, v


def normrope_packed_mask(qkv, p, cos3, sin3, vmask):
    """Keep packed; process all 3 streams, mask V back to identity (overpays V).

    w3 is rebuilt from live params each call so grads reach q_w/k_w (reviewer #7).
    """
    w3 = torch.stack([p.q_w, p.k_w, torch.ones_like(p.q_w)], dim=0).view(1, 1, 3, 1, HEAD_DIM)
    f = qkv.float()
    inv = torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS)
    inv = torch.where(vmask, torch.ones_like(inv), inv)
    qkv = ((f * inv) * w3).to(qkv.dtype)
    return qkv * cos3 + _rotate_half(qkv) * sin3


def normrope_packed_stack(qkv, p, cos, sin):
    """norm+RoPE on q,k only, then stack+contiguous to feed qkvpacked (repack copy)."""
    q, k, v = qkv.unbind(dim=-3)
    q = rmsnorm(q, p.q_w)
    k = rmsnorm(k, p.k_w)
    cq, sq = cos[None, :, None, :], sin[None, :, None, :]
    q = q * cq + _rotate_half(q) * sq
    k = k * cq + _rotate_half(k) * sq
    return torch.stack([q, k, v], dim=2).contiguous()


def make_packed_consts(cos, sin, device):
    """cos3/sin3 with the V stream as RoPE-identity; vmask marks the V stream.

    These are param-free constants; the per-stream RMSNorm weight is rebuilt
    inside normrope_packed_mask so grads flow to q_w/k_w.
    """
    S = cos.shape[0]
    one, zero = torch.ones_like(cos), torch.zeros_like(sin)
    cos3 = torch.stack([cos, cos, one], dim=1).view(1, S, 3, 1, HEAD_DIM)
    sin3 = torch.stack([sin, sin, zero], dim=1).view(1, S, 3, 1, HEAD_DIM)
    vmask = torch.zeros(1, 1, 3, 1, 1, dtype=torch.bool, device=device)
    vmask[:, :, 2] = True
    return cos3, sin3, vmask


# ------- full-block forwards ------------------------------------------------
def full_split(x, p, cos, sin, v_contiguous):
    B, S, _ = x.shape
    qkv = (x @ p.qkv_w.t()).unflatten(-1, (3, N_HEADS, HEAD_DIM))
    q, k, v = normrope_split(qkv, p, cos, sin)
    if v_contiguous:
        v = v.contiguous()
    out = flash_attn_func(q, k, v, 0.0, softmax_scale=SCALE).reshape(B, S, INNER)
    return out @ p.out_w.t()


def full_packed_mask(x, p, cos3, sin3, vmask):
    B, S, _ = x.shape
    qkv = (x @ p.qkv_w.t()).unflatten(-1, (3, N_HEADS, HEAD_DIM))
    qkv = normrope_packed_mask(qkv, p, cos3, sin3, vmask)
    out = flash_attn_qkvpacked_func(qkv, 0.0, softmax_scale=SCALE).reshape(B, S, INNER)
    return out @ p.out_w.t()


def full_packed_stack(x, p, cos, sin):
    B, S, _ = x.shape
    qkv = (x @ p.qkv_w.t()).unflatten(-1, (3, N_HEADS, HEAD_DIM))
    qkv = normrope_packed_stack(qkv, p, cos, sin)
    out = flash_attn_qkvpacked_func(qkv, 0.0, softmax_scale=SCALE).reshape(B, S, INNER)
    return out @ p.out_w.t()


def zero_grads(*tensors_or_modules):
    for t in tensors_or_modules:
        if isinstance(t, torch.nn.Module):
            for prm in t.parameters():
                prm.grad = None
        elif t is not None:
            t.grad = None


def bench(make_fn_and_inputs, niter, do_backward):
    """make_fn_and_inputs() -> (fn, x, grad_owners). Fresh state for isolation."""
    fn, x, owners = make_fn_and_inputs()
    for _ in range(3):  # warmup
        if do_backward:
            zero_grads(x, *owners)
        out = fn(x)
        if do_backward:
            out.sum().backward()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(niter):
        if do_backward:
            zero_grads(x, *owners)
        out = fn(x)
        if do_backward:
            out.sum().backward()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / niter * 1e3
    peak = torch.cuda.max_memory_allocated() / 1e6
    del fn, x, owners, out
    gc.collect()
    torch.cuda.empty_cache()
    return dt, peak


def grad_parity(S, device, dtype):
    """Forward + backward parity of split vs both packed variants (same weights)."""
    torch.manual_seed(1)
    p = Params(dtype, device)
    cos, sin = build_rope(S, device, dtype)
    cos3, sin3, vmask = make_packed_consts(cos, sin, device)
    x0 = torch.randn(1, S, INNER, device=device, dtype=dtype)

    def run(fn):
        zero_grads(p)
        x = x0.clone().detach().requires_grad_(True)
        fn(x).sum().backward()
        return {"x": x.grad.clone(), "qkv": p.qkv_w.grad.clone(), "out": p.out_w.grad.clone(),
                "q": p.q_w.grad.clone(), "k": p.k_w.grad.clone()}

    ref = run(lambda x: full_split(x, p, cos, sin, False))
    out = {}
    for name, fn in (("packed_mask", lambda x: full_packed_mask(x, p, cos3, sin3, vmask)),
                     ("packed_stack", lambda x: full_packed_stack(x, p, cos, sin))):
        g = run(fn)
        diffs = {}
        for key in ref:
            d = (g[key].float() - ref[key].float()).abs().max().item()
            r = d / (ref[key].float().abs().max().item() + 1e-12)
            diffs[key] = {"max_abs": d, "rel": r}
        out[name] = diffs
    return out


# ------- norm+RoPE-only forwards (return a tensor to backward through) ------
def nr_split(qkv, p, cos, sin):
    q, k, v = normrope_split(qkv, p, cos, sin)
    return torch.stack([q, k, v], dim=2)


def nr_packed_mask(qkv, p, cos3, sin3, vmask):
    return normrope_packed_mask(qkv, p, cos3, sin3, vmask)


def nr_packed_stack(qkv, p, cos, sin):
    return normrope_packed_stack(qkv, p, cos, sin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niter", type=int, default=40)
    ap.add_argument("--seqlens", type=int, nargs="+", default=[4032, 4200])
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"
    device, dtype = "cuda", torch.bfloat16

    results = {"gpu": torch.cuda.get_device_name(0), "dtype": "bf16", "batch": 1,
               "niter": args.niter, "grad_parity": {}, "rows": []}

    for S in args.seqlens:
        # ---- gradient parity (correctness) ----
        results["grad_parity"][S] = grad_parity(S, device, dtype)
        print(f"\n[S={S}] gradient parity vs split baseline:")
        for variant, diffs in results["grad_parity"][S].items():
            worst = max(diffs.values(), key=lambda d: d["rel"])
            detail = ", ".join(f"{k}:{v['rel']:.1e}" for k, v in diffs.items())
            print(f"   {variant:14s} worst rel={worst['rel']:.3e}  ({detail})")

        compile_modes = [False] if args.no_compile else [False, True]

        def record(stage, variant, compiled, do_bwd, ms, peak, baseline_ms):
            ratio = ms / baseline_ms if baseline_ms else 1.0
            tag = "fwd+bwd" if do_bwd else "fwd"
            comp = "compiled" if compiled else "eager"
            print(f"[S={S}] {stage:5s} {variant:14s} {comp:8s} {tag:8s} "
                  f"{ms:7.3f} ms ({ratio:5.2f}x) peak {peak:8.1f} MB")
            results["rows"].append({"seqlen": S, "stage": stage, "variant": variant,
                                    "compiled": compiled, "mode": tag, "ms": round(ms, 4),
                                    "ratio_vs_split": round(ratio, 4), "peak_mb": round(peak, 1)})

        for compiled in compile_modes:
            def C(f):  # maybe-compile
                return torch.compile(f, fullgraph=False) if compiled else f

            # leaf-tensor factories (fresh state per measurement → isolated memory)
            def leaf_qkv():
                return torch.randn(1, S, 3, N_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)

            def state():
                p = Params(dtype, device)
                cos, sin = build_rope(S, device, dtype)
                c3, s3, vm = make_packed_consts(cos, sin, device)
                return p, cos, sin, c3, s3, vm

            # [A] FA2 kernel only — q/k/v leaves vs packed-qkv leaf
            def mk_fa2_split():
                q, k, v = (torch.randn(1, S, N_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True) for _ in range(3))
                return C(lambda _: flash_attn_func(q, k, v, 0.0, softmax_scale=SCALE)), q, (k, v)

            def mk_fa2_packed():
                qkv = leaf_qkv()
                return C(lambda _: flash_attn_qkvpacked_func(qkv, 0.0, softmax_scale=SCALE)), qkv, ()

            # [B] norm+RoPE only — qkv leaf in, transformed tensor out
            def mk_nr(kind):
                def factory():
                    p, cos, sin, c3, s3, vm = state()
                    qkv = leaf_qkv()
                    if kind == "split":
                        fn = C(lambda q: nr_split(q, p, cos, sin))
                    elif kind == "packed_mask":
                        fn = C(lambda q: nr_packed_mask(q, p, c3, s3, vm))
                    else:
                        fn = C(lambda q: nr_packed_stack(q, p, cos, sin))
                    return fn, qkv, (p,)
                return factory

            # [C] full block
            def mk_full(kind):
                def factory():
                    p, cos, sin, c3, s3, vm = state()
                    x = torch.randn(1, S, INNER, device=device, dtype=dtype, requires_grad=True)
                    if kind == "split":
                        fn = C(lambda x: full_split(x, p, cos, sin, False))
                    elif kind == "split_vcontig":
                        fn = C(lambda x: full_split(x, p, cos, sin, True))
                    elif kind == "packed_mask":
                        fn = C(lambda x: full_packed_mask(x, p, c3, s3, vm))
                    else:
                        fn = C(lambda x: full_packed_stack(x, p, cos, sin))
                    return fn, x, (p,)
                return factory

            for do_bwd in (False, True):
                groups = [
                    ("fa2", [("split", mk_fa2_split), ("packed", mk_fa2_packed)]),
                    ("nr", [("split", mk_nr("split")), ("packed_mask", mk_nr("packed_mask")),
                            ("packed_stack", mk_nr("packed_stack"))]),
                    ("full", [("split", mk_full("split")), ("split_vcontig", mk_full("split_vcontig")),
                              ("packed_mask", mk_full("packed_mask")), ("packed_stack", mk_full("packed_stack"))]),
                ]
                for stage, variants in groups:
                    base_ms = None
                    for name, factory in variants:
                        ms, pk = bench(factory, args.niter, do_bwd)
                        if base_ms is None:
                            base_ms = ms
                        record(stage, name, compiled, do_bwd, ms, pk, base_ms)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    out_path = out_dir / f"{stamp}-v2.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
