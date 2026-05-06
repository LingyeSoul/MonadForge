#!/usr/bin/env python
"""APEX attention-visibility probe — kill criterion before a real APEX run.

Question this script answers
----------------------------
On Anima, does ``ConditionShift(c) = a·c + b`` actually produce a
meaningfully different ``(K, V)`` at every cross-attention block, or is
the perturbation absorbed by ``kv_proj + k_norm + softmax`` and rendered
invisible to the model?

Why it exists
-------------
APEX (arXiv:2604.12322) replaces a discriminator with a query of the
same DiT under a shifted text condition. The "adversarial" gradient
relies entirely on ``F_θ(x_t, t, c) ≠ F_θ(x_t, t, c_fake)`` — i.e. on
the perturbation actually changing what cross-attention sees. Anima's
cross-attn is bias-free ``kv_proj(c)`` → split into ``(k, v)`` →
``RMSNorm`` on ``k`` (``v_norm = Identity``) → softmax. Two structural
absorptions kick in:

  - softmax kills any token-independent additive offset to logits, so
    ``c_fake = c + b`` (broadcast bias) is invisible;
  - RMSNorm on ``k`` kills global rescaling, so ``c_fake = a·c`` is
    largely absorbed.

Scalar ``ConditionShift`` was empirically dead at Anima scale (config
comment in ``configs/methods/apex.toml``: ``(a, b)`` drifted <3% in 7k
steps). Diag is the workaround. Before burning compute on a full
training run, this probe verifies the diag init *or* a partially-
trained APEX checkpoint produces non-trivial ``(K, V)`` deltas across
every block.

What it measures (per cross-attn block, averaged over N prompts)
----------------------------------------------------------------
  1. symmetric attention KL with synthetic unit q's — direct measure of
     whether cross-attn weight maps differ under c vs. c_fake.
     **Primary kill criterion** (``--gate-kl``).
  2. token-independent fraction of ``ΔK_pre`` — the part softmax kills
     (``||mean_over_tokens(ΔK_pre)||² / ||ΔK_pre||²``). High = invisible.
     **Second kill criterion** (``--gate-indep``).
  3. ``||ΔK_post|| / ||K_post||`` — relative norm of K-perturbation
     after ``k_norm``. Sanity floor (``--gate-k-post``); not load-bearing
     under uniform init since uniform diag/scalar can be huge here while
     attention maps barely move.
  4. ``||ΔV|| / ||V||`` — same for V (no norm, reported for symmetry).

Decision
--------
PASS requires all three gates (median over blocks):

  1. ``attn_sym_kl ≥ --gate-kl``                  (default 0.5 nats)
       — attention maps actually differ under c vs. c_fake.
  2. ``k_pre_token_indep_frac ≤ --gate-indep``    (default 0.5)
       — at most half the ΔK_pre energy is in the softmax-invisible
         token-independent subspace.
  3. ``k_post_rel ≥ --gate-k-post``               (default 0.05, sanity floor)
       — perturbation isn't bit-identical post k_norm.

The first two are the load-bearing gates; the third is a sanity floor
matching the 2D toy's ``mean_rel_shift`` bound in
``archive/bench/apex_phase0.py``. Earlier versions gated only on
``k_post_rel``, which gave false PASS under uniform diag/scalar inits
where ``c_fake = -0.5·c + 1.0`` produced a huge but ~97%
token-independent perturbation that softmax mostly ignored.

Usage
-----
::

    # Probe the SHIPPED diag init (no training compute spent yet)
    python bench/apex/probe_attention_visibility.py

    # Probe under the warm-start LoRA (matches the actual APEX base
    # after promote_warmstart_to_merge bakes it in)
    python bench/apex/probe_attention_visibility.py \\
        --warmstart output/ckpt/anima_lora.safetensors

    # Probe a partially-trained APEX checkpoint's (a, b) state
    python bench/apex/probe_attention_visibility.py \\
        --warmstart output/ckpt/anima_lora.safetensors \\
        --apex-ckpt output/ckpt/anima_apex.safetensors

    # Sanity-check that scalar mode IS dead (expected FAIL)
    python bench/apex/probe_attention_visibility.py --mode scalar

Outputs ``bench/apex/results/<YYYYMMDD-HHMM>[-<label>]/``
  - ``result.json`` (standard envelope)
  - ``per_layer.csv``
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_weights  # noqa: E402
from networks.methods.apex import ConditionShift  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_text_embeddings(cache_dir: Path, n: int, *, device, dtype, seed: int = 0):
    """Sample n cached crossattn_emb_v0 tensors → [n, S, D]."""
    files = sorted(cache_dir.glob("*_anima_te.safetensors"))
    if len(files) < n:
        raise SystemExit(
            f"need ≥{n} cached TE sidecars in {cache_dir}, found {len(files)} "
            f"(run `make preprocess-te` first)"
        )
    rng = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(len(files), generator=rng)[:n].tolist()
    embs = []
    for i in idx:
        with safe_open(str(files[i]), framework="pt") as f:
            embs.append(f.get_tensor("crossattn_emb_v0").to(device=device, dtype=dtype))
    return torch.stack(embs, dim=0)  # [n, S, D]


def build_condition_shift(
    apex_ckpt: str | None,
    *,
    dim: int,
    mode: str,
    init_a: float,
    init_b: float,
    full_orth_jitter: float = 0.0,
    seed: int = 0,
    device,
    dtype,
):
    cs = ConditionShift(dim=dim, mode=mode, init_a=init_a, init_b=init_b)
    # Optional off-diagonal init for `full` mode. Without this, ``A = init_a · I``
    # is bit-equivalent to scalar/diag at uniform init under RMSNorm cross-attn —
    # the affine family's only attention-altering direction is a sign flip.
    # Adding ε·Q with Q random orthogonal injects per-token mixing the diagonal
    # parametrizations cannot express, so the probe can measure whether the
    # parametrization itself (not just (a,b)) has the capacity to clear
    # ``--gate-kl``.
    if mode == "full" and full_orth_jitter > 0.0:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        # Random Gaussian → QR → orthogonal Q (det ±1). Float32 for numerics.
        rand = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
        Q, _ = torch.linalg.qr(rand)
        with torch.no_grad():
            cs.A.add_(full_orth_jitter * Q.to(cs.A.dtype))
    if apex_ckpt:
        sd = load_file(str(_resolve(apex_ckpt)))
        sub = {
            k.removeprefix("apex_condition_shift."): v
            for k, v in sd.items()
            if k.startswith("apex_condition_shift.")
        }
        if not sub:
            print(
                f"  ! no apex_condition_shift.* keys in {apex_ckpt}; falling back to init"
            )
        else:
            missing, unexpected = cs.load_state_dict(sub, strict=False)
            if missing or unexpected:
                print(
                    f"  ! ConditionShift load mismatch — missing={missing} unexpected={unexpected}"
                )
            else:
                print(f"  loaded apex_condition_shift.* from {apex_ckpt}")
    return cs.to(device=device, dtype=dtype)


@torch.no_grad()
def capture_real_q_per_block(
    model,
    c: torch.Tensor,
    *,
    n_heads: int,
    head_dim: int,
    Q: int,
    seed: int,
    latent_hw: int = 64,
    timestep: float = 0.5,
    device,
    dtype,
):
    """Run one DiT forward to capture real cross-attn q's per block.

    Real q's are produced by ``cross_attn.q_proj(x)`` then ``q_norm`` on the
    image side. With trained weights they preferentially align with K(c) for
    high-attention tokens, so attention-map KL under c vs. c_fake reflects
    the *training-time* signal — not the symmetric averaging that random
    Gaussian q's give. Captured via forward hooks on ``cross_attn.q_norm``;
    no monkey-patching, no architectural change.
    """
    n_prompts = c.shape[0]
    in_channels = model.in_channels  # Anima: 16
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x_latent = torch.randn(
        n_prompts,
        in_channels,
        1,
        latent_hw,
        latent_hw,
        generator=gen,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    timesteps = torch.full((n_prompts,), float(timestep), device=device, dtype=dtype)
    # concat_padding_mask=True is hard-coded in load_anima_model — DiT expects
    # this 17th channel even at probe time. Zero = "no padding."
    padding_mask = torch.zeros(
        n_prompts, 1, latent_hw, latent_hw, device=device, dtype=dtype
    )

    captured: list[torch.Tensor | None] = [None] * len(model.blocks)
    handles = []
    for i, block in enumerate(model.blocks):

        def make_hook(idx):
            def hook(_module, _input, output):
                # q_norm output: [B, S_img, n_heads, head_dim]
                captured[idx] = output.detach()

            return hook

        handles.append(block.cross_attn.q_norm.register_forward_hook(make_hook(i)))

    try:
        _ = model.forward_mini_train_dit(
            x_B_C_T_H_W=x_latent,
            timesteps_B_T=timesteps,
            crossattn_emb=c,
            padding_mask=padding_mask,
        )
    finally:
        for h in handles:
            h.remove()

    # Sample Q tokens per block (same indices for all prompts in a block so
    # paired comparison is stable). One q-token sampler shared across blocks
    # would lose per-block independence; sample per block instead.
    sampled = []
    sample_gen = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0xA5A5)
    for q_full in captured:
        if q_full is None:
            raise RuntimeError("hook did not fire on every block")
        S = q_full.shape[1]
        idx = torch.randperm(S, generator=sample_gen)[:Q].to(q_full.device)
        sampled.append(q_full.index_select(1, idx))  # [B, Q, n_heads, head_dim]
    return sampled


@torch.no_grad()
def probe_block(cross_attn, c, c_fake, *, q_slab):
    """Per-block diagnostic. cross_attn is library.anima.models.Attention with
    is_selfattn=False."""
    n_heads, head_dim = cross_attn.n_heads, cross_attn.head_dim

    def kv(ctx):
        kvr = cross_attn.kv_proj(ctx)  # [B, S, 2*inner]
        k_pre, v_pre = kvr.unflatten(-1, (2, n_heads, head_dim)).unbind(dim=-3)
        return k_pre, v_pre, cross_attn.k_norm(k_pre), cross_attn.v_norm(v_pre)

    Kp, Vp, K, V = kv(c)
    Kp_, Vp_, K_, V_ = kv(c_fake)

    def relnorm(a, b):
        return float((a - b).float().norm() / b.float().norm().clamp(min=1e-8))

    out = {
        "k_pre_rel": relnorm(Kp_, Kp),
        "v_pre_rel": relnorm(Vp_, Vp),
        "k_post_rel": relnorm(K_, K),
        "v_post_rel": relnorm(V_, V),
    }

    # token-independent component of ΔK_pre = the part softmax absorbs:
    # if every token's K is shifted by the same vector, q·K_i + q·K_b adds a
    # constant to every logit and softmax doesn't see it.
    dK_pre = (Kp_ - Kp).float()  # [B, S, H, D]
    mean_t = dK_pre.mean(dim=1, keepdim=True)  # [B, 1, H, D]
    num = mean_t.expand_as(dK_pre).pow(2).sum()
    den = dK_pre.pow(2).sum().clamp(min=1e-12)
    out["k_pre_token_indep_frac"] = float(num / den)

    # Attention-map symmetric KL with synthetic q's. q_slab: [B, Q, H, D].
    # q's are randn (RMS≈1 per element, matching post-q_norm magnitude).
    scale = head_dim**-0.5
    L = torch.einsum("bqhd,bshd->bhqs", q_slab.float(), K.float()) * scale
    L_ = torch.einsum("bqhd,bshd->bhqs", q_slab.float(), K_.float()) * scale
    P = L.softmax(dim=-1).clamp(min=1e-9)
    P_ = L_.softmax(dim=-1).clamp(min=1e-9)
    sym_kl = ((P * (P / P_).log()).sum(-1) + (P_ * (P_ / P).log()).sum(-1)) * 0.5
    out["attn_sym_kl"] = float(sym_kl.mean())
    return out


def summarize(rows, key):
    xs = sorted(r[key] for r in rows)
    n = len(xs)
    return {
        "median": float(xs[n // 2]),
        "mean": float(sum(xs) / n),
        "min": float(xs[0]),
        "max": float(xs[-1]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dit",
        default="models/diffusion_models/anima-preview3-base.safetensors",
        help="DiT checkpoint (relative to anima_lora/ unless absolute)",
    )
    p.add_argument(
        "--warmstart",
        default=None,
        help="optional LoRA to merge into the DiT before probing — pass the same "
        "file APEX uses as network_weights so the probe matches the actual "
        "post-promote_warmstart_to_merge base",
    )
    p.add_argument(
        "--apex-ckpt",
        default=None,
        help="optional anima_apex.safetensors with trained apex_condition_shift.* "
        "keys; init values are ignored if these load successfully",
    )
    p.add_argument(
        "--cache-dir",
        default="post_image_dataset/lora",
        help="directory containing *_anima_te.safetensors sidecars",
    )
    p.add_argument("--n-prompts", type=int, default=16)
    p.add_argument("--mode", default="diag", choices=["scalar", "diag", "full"])
    p.add_argument("--init-a", type=float, default=-0.5)
    p.add_argument("--init-b", type=float, default=1.0)
    p.add_argument(
        "--full-orth-jitter",
        type=float,
        default=0.0,
        help="(--mode full only) initialize A = init_a·I + (this)·Q with Q "
        "random orthogonal. Off-diagonal mixing is the only direction in "
        "the c_fake = A·c + b family that produces per-token, RMSNorm-"
        "robust K/V deltas; without this, full mode at init is bit-"
        "equivalent to scalar/diag uniform.",
    )
    p.add_argument(
        "--gate-kl",
        type=float,
        default=0.5,
        help="median attn_sym_kl below this → FAIL — attention maps don't "
        "actually differ under c vs. c_fake (primary adversarial-signal gate).",
    )
    p.add_argument(
        "--gate-indep",
        type=float,
        default=0.5,
        help="median k_pre_token_indep_frac above this → FAIL — perturbation "
        "is mostly token-independent and softmax-invisible.",
    )
    p.add_argument(
        "--gate-k-post",
        type=float,
        default=0.05,
        help="median ||ΔK_post||/||K_post|| below this → FAIL (sanity floor; "
        "matches 2D toy mean_rel_shift bound). Not load-bearing under uniform "
        "init — kept to catch bit-identical c_fake bugs.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for synthetic q_slab and prompt sampling — runs with the "
        "same args produce bit-identical metrics.",
    )
    p.add_argument(
        "--use-real-q",
        action="store_true",
        help="capture real q's from one DiT forward at --latent-hw + "
        "--latent-t instead of synthetic randn. Real q's are trained to "
        "align with K(c), so attention-map KL on a sign-flip ≈ true "
        "training-time signal — randn q's average symmetrically and "
        "underestimate the divergence.",
    )
    p.add_argument(
        "--latent-hw",
        type=int,
        default=64,
        help="(--use-real-q) DiT latent H=W; 64 → 32×32 patches = 1024 "
        "image tokens per block (cheap, plenty of q's to sample 64 from).",
    )
    p.add_argument(
        "--latent-t",
        type=float,
        default=0.5,
        help="(--use-real-q) timestep for the captured forward. 0.5 = "
        "midpoint of the flow-matching trajectory; trade-offs with t→0 "
        "(image-anchored q's) vs t→1 (noise-anchored).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--label", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # ----- DiT (with optional warm-start merge)
    lora_weights_list = None
    if args.warmstart:
        ws = load_file(str(_resolve(args.warmstart)))
        lora_weights_list = [
            {k: v for k, v in ws.items() if k.startswith("lora_unet_")}
        ]
        print(
            f"warm-start: merging {args.warmstart} ({len(lora_weights_list[0])} keys)"
        )
    print(f"loading DiT from {args.dit}")
    model = (
        anima_weights.load_anima_model(
            device,
            str(_resolve(args.dit)),
            "torch",
            True,
            "cpu",
            dtype,
            lora_weights_list=lora_weights_list,
            lora_multipliers=[1.0] if lora_weights_list else None,
        )
        .eval()
        .requires_grad_(False)
    )
    model.to(device)

    ctx_dim = model.blocks[0].cross_attn.context_dim
    n_heads = model.blocks[0].cross_attn.n_heads
    head_dim = model.blocks[0].cross_attn.head_dim
    n_blocks = len(model.blocks)
    print(
        f"DiT: {n_blocks} blocks, ctx_dim={ctx_dim}, heads={n_heads}, head_dim={head_dim}"
    )

    # ----- text embeddings
    cache_dir = _resolve(args.cache_dir)
    print(f"sampling {args.n_prompts} cached TE embeddings from {cache_dir}")
    c = load_text_embeddings(
        cache_dir, args.n_prompts, device=device, dtype=dtype, seed=args.seed
    )

    # ----- ConditionShift
    cs = build_condition_shift(
        args.apex_ckpt,
        dim=ctx_dim,
        mode=args.mode,
        init_a=args.init_a,
        init_b=args.init_b,
        full_orth_jitter=args.full_orth_jitter,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )
    c_fake = cs(c)
    print(
        f"ConditionShift: mode={args.mode}, "
        f"||c_fake - c|| / ||c|| = {((c_fake - c).float().norm() / c.float().norm()).item():.4f}"
    )

    # ----- q slab: synthetic randn or real per-block from a DiT forward
    Q = 64
    if args.use_real_q:
        print(
            f"capturing real q per block via forward at "
            f"latent_hw={args.latent_hw}, t={args.latent_t}..."
        )
        q_per_block = capture_real_q_per_block(
            model,
            c,
            n_heads=n_heads,
            head_dim=head_dim,
            Q=Q,
            seed=args.seed,
            latent_hw=args.latent_hw,
            timestep=args.latent_t,
            device=device,
            dtype=dtype,
        )
        # Sanity print: average q-norm vs. randn q-norm so the difference is
        # visible at a glance.
        q0 = q_per_block[0]
        print(
            f"  real q[0]: shape={tuple(q0.shape)}, "
            f"per-token-vector RMS median = "
            f"{q0.float().pow(2).mean(dim=-1).sqrt().median().item():.4f} "
            f"(randn would be ~1.0)"
        )
    else:
        # Seeded synthetic q_slab so two runs with the same args produce
        # bit-identical attn_sym_kl — without this, q_slab ~ randn() drifts
        # the KL by ~1e-3 between repeats.
        q_gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
        q_slab_syn = torch.randn(
            args.n_prompts, Q, n_heads, head_dim, generator=q_gen, dtype=torch.float32
        ).to(device=device, dtype=dtype)
        q_per_block = [q_slab_syn] * n_blocks

    # ----- per-block probe
    print(f"probing {n_blocks} blocks...")
    rows = []
    for i, block in enumerate(model.blocks):
        m = probe_block(block.cross_attn, c, c_fake, q_slab=q_per_block[i])
        rows.append({"block": i, **m})

    # ----- summary
    keys = (
        "k_pre_rel",
        "v_pre_rel",
        "k_post_rel",
        "v_post_rel",
        "k_pre_token_indep_frac",
        "attn_sym_kl",
    )
    summary = {k: summarize(rows, k) for k in keys}
    median_k_post_rel = summary["k_post_rel"]["median"]
    median_attn_kl = summary["attn_sym_kl"]["median"]
    median_indep = summary["k_pre_token_indep_frac"]["median"]

    gates = {
        "kl": {
            "metric": "attn_sym_kl",
            "median": median_attn_kl,
            "threshold": args.gate_kl,
            "rule": "median >= threshold",
            "pass": median_attn_kl >= args.gate_kl,
        },
        "indep": {
            "metric": "k_pre_token_indep_frac",
            "median": median_indep,
            "threshold": args.gate_indep,
            "rule": "median <= threshold",
            "pass": median_indep <= args.gate_indep,
        },
        "k_post": {
            "metric": "k_post_rel",
            "median": median_k_post_rel,
            "threshold": args.gate_k_post,
            "rule": "median >= threshold",
            "pass": median_k_post_rel >= args.gate_k_post,
        },
    }
    pass_ = all(g["pass"] for g in gates.values())

    # ----- write artifacts
    label = args.label or (
        f"{args.mode}-trained" if args.apex_ckpt else f"{args.mode}-init"
    )
    run_dir = make_run_dir("apex", label=label)
    csv_path = run_dir / "per_layer.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    metrics = {
        "summary": summary,
        "gates": gates,
        "pass": pass_,
        "n_blocks": n_blocks,
        "n_prompts": args.n_prompts,
        "ctx_dim": ctx_dim,
        "n_heads": n_heads,
        "head_dim": head_dim,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=label,
        artifacts=["per_layer.csv"],
        device=device,
    )

    # ----- human-readable
    bar = "=" * 72
    print()
    print(bar)
    print(f"APEX attention-visibility probe — mode={args.mode}")
    if args.warmstart:
        print(f"  base       : warm-start ⊕ {Path(args.warmstart).name}")
    if args.apex_ckpt:
        print(f"  shift state: trained from {Path(args.apex_ckpt).name}")
    else:
        print(f"  shift state: init (a={args.init_a}, b={args.init_b})")
    print(bar)
    fmt = "  {name:<28s}  median {med:>9.4g}   min {mn:>9.4g}   max {mx:>9.4g}"
    print(f"per-layer over {n_blocks} blocks × {args.n_prompts} prompts:")
    for k, label_ in [
        ("k_post_rel", "||ΔK_post||/||K_post||"),
        ("v_post_rel", "||ΔV||/||V||"),
        ("k_pre_rel", "||ΔK_pre||/||K_pre||"),
        ("k_pre_token_indep_frac", "ΔK_pre token-indep frac"),
        ("attn_sym_kl", "symmetric attn-map KL"),
    ]:
        s = summary[k]
        print(fmt.format(name=label_, med=s["median"], mn=s["min"], mx=s["max"]))
    print()
    print("gates (median over blocks):")
    print(
        f"  attn_sym_kl              "
        f"{median_attn_kl:>9.4g}   "
        f"{'>=':<2} {args.gate_kl:<8.4g}   "
        f"{'PASS' if gates['kl']['pass'] else 'FAIL'}"
    )
    print(
        f"  k_pre_token_indep_frac   "
        f"{median_indep:>9.4g}   "
        f"{'<=':<2} {args.gate_indep:<8.4g}   "
        f"{'PASS' if gates['indep']['pass'] else 'FAIL'}"
    )
    print(
        f"  k_post_rel (sanity)      "
        f"{median_k_post_rel:>9.4g}   "
        f"{'>=':<2} {args.gate_k_post:<8.4g}   "
        f"{'PASS' if gates['k_post']['pass'] else 'FAIL'}"
    )
    print()
    if pass_:
        verdict = (
            "PASS  perturbation is alive in K-space AND softmax-visible — "
            "APEX adversary has signal to work with. (Other failure modes "
            "still apply during training; this only rules out the structural "
            "visibility problem.)"
        )
    else:
        failed = [g["metric"] for g in gates.values() if not g["pass"]]
        diag = []
        if not gates["kl"]["pass"]:
            diag.append(
                "attention maps barely move under c_fake — adversarial branch "
                "will produce ≈ self-distillation"
            )
        if not gates["indep"]["pass"]:
            diag.append(
                f"{median_indep * 100:.1f}% of ΔK_pre is token-independent "
                f"(softmax-invisible) — perturbation budget is mostly wasted"
            )
        if not gates["k_post"]["pass"]:
            diag.append(
                "post-norm K is bit-identical — c_fake = c, check ConditionShift wiring"
            )
        verdict = (
            f"FAIL  gates failed: {', '.join(failed)}.\n"
            + "\n".join(f"      • {d}" for d in diag)
            + "\n      Don't burn compute on a full APEX run with these settings."
        )
    print(verdict)
    print()
    print(f"results: {run_dir}")
    print("  result.json   (standard envelope)")
    print("  per_layer.csv (per-block raw numbers)")


if __name__ == "__main__":
    main()
