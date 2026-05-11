#!/usr/bin/env python
"""APEX temporal-shift bench — does t_fake = t + Δt give a comparable target?

Question this script answers
----------------------------
APEX's adversarial signal comes from querying the same DiT under a *shifted
condition* (``c_fake = A·c + b``) and using the resulting ``v_fake`` as a
stop-gradient target for ``L_mix``. EMF (arXiv:2602.02571, *Trajectory
Consistency for One-Step Generation on Euler Mean Flows*) suggests an
alternative supervision lever: query the same DiT at a *shifted timestep*
(``t_fake = t ± Δt``) and use the trajectory-consistency relation as the
target. Both are "second forward of the same network at a perturbed input",
but the perturbation lives in different spaces — one in condition, one in
time.

Before forking the trainer, this probe measures: **at the level of v_fake
itself**, is a temporal shift a comparably informative perturbation to the
shipped condition shift?

What it measures (per strategy × per t-bucket × n_prompts)
----------------------------------------------------------
For each text embedding ``c`` and each base timestep ``t``:

  v_real = anima(x_t, t, c)
  v_fake = anima(x_fake_input, t_fake, c_fake)        # strategy-dependent

Reports:

  - ``mse``       — MSE(v_fake, v_real). Large = informative perturbation.
  - ``cos_sim``   — flat-vector cosine. Low = directionally distinct.
  - ``snr``       — ``E[v_real²] / E[(v_fake - v_real)²]``. Low = noisy
                    target; very high (>>1) = perturbation barely visible.

Strategies
----------
  - ``identity``       — ``c_fake = c, t_fake = t``.  Sanity floor (~0).
  - ``cond_diag``      — ``c_fake = ConditionShift(c)`` at the shipped diag
                          init (a=-0.5, b=0.0). Matches the live APEX config.
  - ``cond_signflip``  — ``c_fake = -1.0·c`` (scalar, b=0). True sign-flip
                          at 2× the shipped magnitude; brackets cond_diag from
                          above so we can see whether stronger condition shift
                          is in the dt_big regime or still affine-family-bounded.
  - ``dt_pos_small``   — ``t_fake = clamp(t + 0.05, 0, 1)``.
  - ``dt_neg_small``   — ``t_fake = clamp(t - 0.05, 0, 1)``.
  - ``dt_pos_big``     — ``t_fake = clamp(t + 0.20, 0, 1)``.
  - ``dt_neg_big``     — ``t_fake = clamp(t - 0.20, 0, 1)``.

Why no PASS/FAIL gate
---------------------
This is a *characterization* probe, not a kill criterion. The decision is:

  - If ``dt_*`` strategies' median ``mse`` ≥ 0.5× ``cond_diag`` ``mse``,
    temporal-shift is in the same supervision-strength ballpark as the
    shipped APEX adversary. Worth a training run in
    ``bench/apex/probe_temporal_train.py`` (not yet written).
  - If ``dt_*`` produces ``mse`` >> ``cond_diag``, temporal-shift's target
    may be too noisy to anchor L_mix — investigate before training.
  - If ``dt_*`` produces ``mse`` << ``cond_diag``, temporal shift is too
    weak to drive useful adversarial gradient — not worth training.

Per-strategy summaries (median / IQR over prompts × buckets) are emitted
to ``per_strategy.csv`` and ``per_strategy_per_t.csv``; the JSON envelope
has the same numbers in nested form.

Cross-strategy perturbation-delta cosine
----------------------------------------
For each interesting strategy pair (a, b), reports
``cos(Δv_a, Δv_b)`` where ``Δv = v_fake − v_real``. This is the
perturbation *direction* alignment in output space — distinct from each
strategy's own ``cos(v_fake, v_real)``.

  - cos ≈ +1   strategies push the model in the same direction → redundant.
  - cos ≈  0   strategies push in orthogonal directions → complementary;
               combining them in training would add real information.
  - cos ≈ −1   strategies push in opposing directions → would partly cancel.

The pre-registered question is whether c-shift (``cond_diag``) and
small temporal shift (``dt_pos_small``) are orthogonal: structurally
they perturb different input axes (condition vs time), but whether
that translates to orthogonal output deltas is empirical.

Usage
-----
::

    python bench/apex/probe_temporal_shift.py
    python bench/apex/probe_temporal_shift.py \\
        --warmstart output/ckpt/anima_lora.safetensors

Outputs ``bench/apex/results/<YYYYMMDD-HHMM>[-<label>]/``
  - ``result.json`` (standard envelope)
  - ``per_strategy.csv``
  - ``per_strategy_per_t.csv``
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
    """Sample n cached crossattn_emb_v0 tensors → [n, S, D]. Same selector
    as probe_attention_visibility.py so paired runs see the same prompts."""
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


@torch.no_grad()
def dit_forward(model, x_5d, t_b, c, padding_mask):
    """One DiT forward → [B, C, 1, H, W] velocity prediction."""
    return model.forward_mini_train_dit(
        x_B_C_T_H_W=x_5d,
        timesteps_B_T=t_b,
        crossattn_emb=c,
        padding_mask=padding_mask,
    )


def per_sample_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reduce all dims except batch. Returns [B] in float32."""
    diff = (a.float() - b.float())
    return diff.pow(2).mean(dim=list(range(1, diff.ndim)))


def per_sample_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Flat-vector cosine over all non-batch dims. Returns [B] in float32."""
    af = a.float().flatten(1)
    bf = b.float().flatten(1)
    num = (af * bf).sum(dim=1)
    den = af.norm(dim=1).clamp(min=1e-12) * bf.norm(dim=1).clamp(min=1e-12)
    return num / den


def per_sample_snr(v_real: torch.Tensor, v_fake: torch.Tensor) -> torch.Tensor:
    """E[v_real²] / E[(v_fake - v_real)²], per sample. Returns [B] in float32."""
    sig = v_real.float().pow(2).mean(dim=list(range(1, v_real.ndim)))
    noise = (v_fake.float() - v_real.float()).pow(2).mean(
        dim=list(range(1, v_real.ndim))
    ).clamp(min=1e-12)
    return sig / noise


def summarize(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"median": float("nan"), "mean": float("nan"),
                "p25": float("nan"), "p75": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    s = sorted(xs)
    n = len(s)
    def q(p): return s[max(0, min(n - 1, int(p * (n - 1))))]
    return {
        "median": float(q(0.5)),
        "mean": float(sum(s) / n),
        "p25": float(q(0.25)),
        "p75": float(q(0.75)),
        "min": float(s[0]),
        "max": float(s[-1]),
        "n": int(n),
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
        help="optional LoRA to merge into the DiT — pass the same file APEX "
        "uses as network_weights so the probe matches the actual base.",
    )
    p.add_argument(
        "--cache-dir",
        default="post_image_dataset/lora",
        help="directory containing *_anima_te.safetensors sidecars",
    )
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument(
        "--latent-hw",
        type=int,
        default=64,
        help="DiT latent H=W; 64 → 32×32 patches = 1024 image tokens.",
    )
    p.add_argument(
        "--t-buckets",
        type=str,
        default="0.1,0.3,0.5,0.7,0.9",
        help="comma-separated base timesteps to sweep. Each prompt is run "
        "at every bucket so per-t structure is visible.",
    )
    p.add_argument(
        "--dt-small",
        type=float,
        default=0.05,
        help="small temporal-shift magnitude (dt_pos_small / dt_neg_small).",
    )
    p.add_argument(
        "--dt-big",
        type=float,
        default=0.20,
        help="big temporal-shift magnitude (dt_pos_big / dt_neg_big).",
    )
    p.add_argument(
        "--cond-init-a",
        type=float,
        default=-0.5,
        help="ConditionShift init_a for the cond_diag baseline (matches shipped).",
    )
    p.add_argument(
        "--cond-init-b",
        type=float,
        default=0.0,
        help="ConditionShift init_b for the cond_diag baseline (matches shipped).",
    )
    p.add_argument("--seed", type=int, default=0)
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
    in_channels = model.in_channels  # Anima: 16
    print(f"DiT: ctx_dim={ctx_dim}, in_channels={in_channels}")

    # ----- text embeddings + ConditionShift baselines
    cache_dir = _resolve(args.cache_dir)
    print(f"sampling {args.n_prompts} cached TE embeddings from {cache_dir}")
    c = load_text_embeddings(
        cache_dir, args.n_prompts, device=device, dtype=dtype, seed=args.seed
    )

    cs_diag = ConditionShift(
        dim=ctx_dim, mode="diag",
        init_a=args.cond_init_a, init_b=args.cond_init_b,
    ).to(device=device, dtype=dtype)
    # init_a=-1.0 (true sign-flip) — distinct from cond_diag, which uses the
    # shipped init_a=-0.5. At uniform init, mode="diag" with init_a=-0.5 is
    # bit-equivalent to mode="scalar" with init_a=-0.5; pinning signflip to
    # -1.0 makes this strategy carry information.
    cs_signflip = ConditionShift(
        dim=ctx_dim, mode="scalar", init_a=-1.0, init_b=0.0,
    ).to(device=device, dtype=dtype)
    c_fake_diag = cs_diag(c)
    c_fake_signflip = cs_signflip(c)
    print(
        f"||c_fake_diag - c|| / ||c|| = "
        f"{((c_fake_diag - c).float().norm() / c.float().norm()).item():.4f}"
    )
    print(
        f"||c_fake_signflip - c|| / ||c|| = "
        f"{((c_fake_signflip - c).float().norm() / c.float().norm()).item():.4f}"
    )

    # ----- shared latents + t buckets
    t_buckets = [float(x.strip()) for x in args.t_buckets.split(",") if x.strip()]
    print(f"t buckets: {t_buckets}")

    gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
    # One x_t per prompt, *reused across buckets and strategies* — we want to
    # measure pure perturbation effect, not noise-pair variance.
    x_init = torch.randn(
        args.n_prompts, in_channels, 1, args.latent_hw, args.latent_hw,
        generator=gen, dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    padding_mask = torch.zeros(
        args.n_prompts, 1, args.latent_hw, args.latent_hw,
        device=device, dtype=dtype,
    )

    def t_clamp(x: float) -> float:
        # Same clamp as the EMF paper's (1-t) >= 0.02 — the DiT's t_embedder
        # has no pole at 0 or 1, but staying away from the boundary keeps
        # comparisons consistent across strategies.
        return float(min(0.98, max(0.02, x)))

    strategies = [
        ("identity",      "cond",  c,                None),
        ("cond_diag",     "cond",  c_fake_diag,      None),
        ("cond_signflip", "cond",  c_fake_signflip,  None),
        ("dt_pos_small",  "time",  c,                +args.dt_small),
        ("dt_neg_small",  "time",  c,                -args.dt_small),
        ("dt_pos_big",    "time",  c,                +args.dt_big),
        ("dt_neg_big",    "time",  c,                -args.dt_big),
    ]

    # Per-(strategy, t) records collect the per-prompt scalars.
    per_st: dict[tuple[str, float], dict[str, list[float]]] = {}
    for s_name, _, _, _ in strategies:
        for t in t_buckets:
            per_st[(s_name, t)] = {"mse": [], "cos": [], "snr": []}

    # Cross-strategy perturbation-delta cosines: cos(Δv_a, Δv_b) where
    # Δv = v_fake − v_real. Tells us whether two perturbations push the
    # model in similar (cos→1), orthogonal (cos→0), or opposing (cos→−1)
    # directions in output space — the "complementary information vs same
    # information" check before deciding whether to combine c-shift and
    # t-shift in training.
    INTERESTING_PAIRS: list[tuple[str, str]] = [
        ("cond_diag",     "dt_pos_small"),   # headline: shipped c-shift vs +Δt
        ("cond_diag",     "dt_neg_small"),   # shipped c-shift vs −Δt
        ("cond_signflip", "dt_pos_small"),   # strong c-shift vs +Δt
        ("cond_signflip", "dt_neg_small"),   # strong c-shift vs −Δt
        ("dt_pos_small",  "dt_neg_small"),   # sanity: opposite t-shifts → cos≈−1?
        ("cond_diag",     "cond_signflip"),  # sanity: both c-family → cos≈+1?
        ("cond_signflip", "dt_pos_big"),     # extra: extreme regimes
    ]
    per_pair: dict[tuple[str, str], list[float]] = {p: [] for p in INTERESTING_PAIRS}

    print(f"running {len(strategies)} strategies × {len(t_buckets)} t-buckets...")
    for t in t_buckets:
        t_b = torch.full(
            (args.n_prompts,), t_clamp(t), device=device, dtype=dtype
        )
        # Real branch: shared baseline for this t.
        v_real = dit_forward(model, x_init, t_b, c, padding_mask)
        # Cache perturbation deltas for cross-strategy cosine.
        # ``identity`` is skipped — its delta is zero (cosine undefined).
        deltas_at_t: dict[str, torch.Tensor] = {}
        for s_name, kind, c_used, dt in strategies:
            if kind == "cond":
                t_fake_b = t_b
            else:
                t_fake_b = torch.full(
                    (args.n_prompts,), t_clamp(t + (dt or 0.0)),
                    device=device, dtype=dtype,
                )
            v_fake = dit_forward(model, x_init, t_fake_b, c_used, padding_mask)

            mse = per_sample_mse(v_fake, v_real).cpu().tolist()
            cos = per_sample_cos(v_fake, v_real).cpu().tolist()
            snr = per_sample_snr(v_real, v_fake).cpu().tolist()
            per_st[(s_name, t)]["mse"].extend(mse)
            per_st[(s_name, t)]["cos"].extend(cos)
            per_st[(s_name, t)]["snr"].extend(snr)
            if s_name != "identity":
                deltas_at_t[s_name] = v_fake - v_real
            print(
                f"  t={t:.2f}  {s_name:14s}  "
                f"mse={sum(mse)/len(mse):.5f}  "
                f"cos={sum(cos)/len(cos):+.4f}  "
                f"snr={sum(snr)/len(snr):.2f}"
            )

        # Cross-strategy delta cosines for this t-bucket.
        for (a_name, b_name) in INTERESTING_PAIRS:
            if a_name not in deltas_at_t or b_name not in deltas_at_t:
                continue
            cos_pair = per_sample_cos(
                deltas_at_t[a_name], deltas_at_t[b_name]
            ).cpu().tolist()
            per_pair[(a_name, b_name)].extend(cos_pair)

    # ----- per-strategy aggregate (across all t-buckets)
    summary_per_strategy: dict[str, dict[str, dict[str, float]]] = {}
    rows_strategy: list[dict] = []
    for s_name, _, _, _ in strategies:
        all_mse: list[float] = []
        all_cos: list[float] = []
        all_snr: list[float] = []
        for t in t_buckets:
            all_mse.extend(per_st[(s_name, t)]["mse"])
            all_cos.extend(per_st[(s_name, t)]["cos"])
            all_snr.extend(per_st[(s_name, t)]["snr"])
        s_mse = summarize(all_mse)
        s_cos = summarize(all_cos)
        s_snr = summarize(all_snr)
        summary_per_strategy[s_name] = {"mse": s_mse, "cos": s_cos, "snr": s_snr}
        rows_strategy.append({
            "strategy": s_name,
            "mse_median": s_mse["median"],  "mse_p25": s_mse["p25"],  "mse_p75": s_mse["p75"],
            "cos_median": s_cos["median"],  "cos_p25": s_cos["p25"],  "cos_p75": s_cos["p75"],
            "snr_median": s_snr["median"],  "snr_p25": s_snr["p25"],  "snr_p75": s_snr["p75"],
        })

    rows_st: list[dict] = []
    for s_name, _, _, _ in strategies:
        for t in t_buckets:
            mse = per_st[(s_name, t)]["mse"]
            cos = per_st[(s_name, t)]["cos"]
            snr = per_st[(s_name, t)]["snr"]
            rows_st.append({
                "strategy": s_name,
                "t": t,
                "mse_median": summarize(mse)["median"],
                "cos_median": summarize(cos)["median"],
                "snr_median": summarize(snr)["median"],
            })

    # ----- ratio summary: temporal-shift vs cond_diag (the live baseline)
    cond_ref = summary_per_strategy.get("cond_diag", {}).get("mse", {}).get("median")
    ratios = {}
    if cond_ref and cond_ref > 0:
        for s_name in (
            "dt_pos_small", "dt_neg_small", "dt_pos_big", "dt_neg_big",
            "cond_signflip", "identity",
        ):
            m = summary_per_strategy.get(s_name, {}).get("mse", {}).get("median")
            if m is not None:
                ratios[s_name] = float(m / cond_ref)

    # ----- cross-strategy perturbation-delta cosines
    cross_summary: dict[str, dict[str, float]] = {}
    rows_cross: list[dict] = []
    for (a_name, b_name) in INTERESTING_PAIRS:
        s = summarize(per_pair.get((a_name, b_name), []))
        cross_summary[f"{a_name}__{b_name}"] = s
        rows_cross.append({
            "strategy_a": a_name,
            "strategy_b": b_name,
            "cos_median": s["median"],
            "cos_mean": s["mean"],
            "cos_p25": s["p25"],
            "cos_p75": s["p75"],
            "cos_min": s["min"],
            "cos_max": s["max"],
            "n": s["n"],
        })

    # ----- write artifacts
    label = args.label or ("warmstart" if args.warmstart else "base")
    label = f"temporal-{label}"
    run_dir = make_run_dir("apex", label=label)

    csv_strategy = run_dir / "per_strategy.csv"
    with csv_strategy.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_strategy[0].keys()))
        w.writeheader()
        w.writerows(rows_strategy)

    csv_st = run_dir / "per_strategy_per_t.csv"
    with csv_st.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_st[0].keys()))
        w.writeheader()
        w.writerows(rows_st)

    csv_cross = run_dir / "cross_strategy_cos.csv"
    with csv_cross.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_cross[0].keys()))
        w.writeheader()
        w.writerows(rows_cross)

    metrics = {
        "summary_per_strategy": summary_per_strategy,
        "ratio_to_cond_diag_mse": ratios,
        "cross_strategy_delta_cos": cross_summary,
        "n_prompts": args.n_prompts,
        "t_buckets": t_buckets,
        "ctx_dim": ctx_dim,
        "latent_hw": args.latent_hw,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=label,
        artifacts=["per_strategy.csv", "per_strategy_per_t.csv", "cross_strategy_cos.csv"],
        device=device,
    )

    # ----- human-readable
    bar = "=" * 72
    print()
    print(bar)
    print("APEX temporal-shift bench")
    if args.warmstart:
        print(f"  base       : warm-start ⊕ {Path(args.warmstart).name}")
    print(f"  prompts    : {args.n_prompts}, t-buckets: {t_buckets}")
    print()
    print(f"  {'strategy':14s} {'mse(med)':>10s} {'cos(med)':>10s} {'snr(med)':>10s}  ratio_vs_cond_diag")
    for s_name, _, _, _ in strategies:
        s = summary_per_strategy[s_name]
        r = ratios.get(s_name)
        r_str = f"{r:.2f}" if r is not None else "  -  "
        print(
            f"  {s_name:14s} {s['mse']['median']:>10.5f} "
            f"{s['cos']['median']:>+10.4f} {s['snr']['median']:>10.2f}  {r_str}"
        )
    print()
    print("  cross-strategy perturbation-delta cosine  cos(Δv_a, Δv_b)")
    print("  → +1 = redundant (same direction); 0 = orthogonal (complementary); −1 = opposing")
    print(f"  {'pair':40s} {'cos(med)':>10s} {'cos(p25)':>10s} {'cos(p75)':>10s}")
    for (a_name, b_name) in INTERESTING_PAIRS:
        s = cross_summary.get(f"{a_name}__{b_name}", {})
        if not s or s.get("n", 0) == 0:
            continue
        pair_str = f"{a_name} × {b_name}"
        print(
            f"  {pair_str:40s} {s['median']:>+10.4f} "
            f"{s['p25']:>+10.4f} {s['p75']:>+10.4f}"
        )
    print()
    print(f"  artifacts → {run_dir}")
    print(bar)


if __name__ == "__main__":
    main()
