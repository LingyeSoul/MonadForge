#!/usr/bin/env python
"""Forecastability probe: does er_sde forecast as well as euler under Spectrum?

The decision this answers
-------------------------
The mod-guidance distillation pool (`post_image_dataset/distill_mod_synth/`) is a
large er_sde@cfg2.5 corpus we already paid for. Reusing it as the Spectrum bench
baseline is high-ROI — *if* er_sde's block-feature trajectory is as Chebyshev-
forecastable as euler's. er_sde injects fresh stochastic noise into the latent
each step, which could make the per-step feature sequence jittery and raise the
forecast residual, shrinking Spectrum's safe block-skips. The proposal makes
euler the primary target precisely for this reason
(`proposal.md` → "Recommended Search Space"). This probe measures the gap
cheaply before any harness is built.

Method (paired)
---------------
For each pool prompt × seed, capture an **all-actual** feature trajectory under
each sampler (same prompt / seed / native bucket / steps / cfg), then replay the
*shipped* forecaster at the production-default combo (m3/lam0.1/w0.3,
ws2/fx0.25/wu7) and record the relative-L2 forecast residual (mean / peak /
late-third over cached steps). er_sde and euler are compared paired per prompt.

Verdict
-------
PASS  : er_sde mean rel-L2 ≤ `--tol` × euler mean rel-L2 (default 1.15).
        → er_sde is ~as forecastable; the pool-reuse bet holds, build the bench
          on er_sde and reuse the stored x0 as free Phase-1/4 endpoints.
FAIL  : er_sde residual materially worse → er_sde is the wrong Spectrum target;
        the x0 pool can't be repurposed, fall back to euler + fresh generation.

This is a *shortlist* signal, not a quality oracle (cf. project memory
`fm_val_loss_uninformative`). It gates effort, not production defaults.

Usage
-----
    uv run python -m bench.spectrum_pareto.compare_samplers \
        --n 16 --seeds 0 1 --buckets 128x128 150x112 --steps 28 --guidance_scale 2.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.spectrum_pareto.capture_features import (  # noqa: E402
    _slug,
    capture_one,
    pool_samples,
)
from bench.spectrum_pareto.replay_forecaster import (  # noqa: E402
    PROD_COMBO,
    replay_at_combo,
)


def _feats_from_payload(payload: dict, device: torch.device):
    cond = payload["cond"]  # (num_steps, T, H, W, D) fp16
    num_steps = int(payload["num_steps"])
    cond_t = torch.from_numpy(cond.astype(np.float32)).to(device)
    feats = [cond_t[i] for i in range(num_steps)]
    _t, h, w, _dd = (int(x) for x in payload["feat_shape"])
    return feats, (h, w), num_steps


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=16, help="Pool prompts to sample.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument(
        "--samplers",
        type=str,
        nargs="+",
        default=["er_sde", "euler"],
        help="Samplers to compare (first is the reuse candidate, last the baseline).",
    )
    ap.add_argument(
        "--buckets",
        type=str,
        nargs="+",
        default=None,
        help="Restrict pool to latent-dim buckets, e.g. 128x128 150x112.",
    )
    ap.add_argument(
        "--pool_dir", type=str, default="post_image_dataset/distill_mod_synth"
    )
    ap.add_argument("--caption_dir", type=str, default="image_dataset")
    ap.add_argument("--pool_seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance_scale", type=float, default=2.5)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--lora", type=str, default=None)
    ap.add_argument(
        "--tol",
        type=float,
        default=1.15,
        help="PASS if candidate mean rel-L2 ≤ tol × baseline mean rel-L2.",
    )
    ap.add_argument(
        "--save_captures",
        action="store_true",
        help="Persist per-capture npz under captures/<sampler>/ (for later sweeps).",
    )
    ap.add_argument(
        "--compile_blocks", action="store_true",
        help="torch.compile the DiT blocks (graphs keyed on token count → amortized "
        "across captures sharing a bucket; keep --buckets small so they reuse).",
    )
    ap.add_argument("--compile_inductor_mode", type=str, default=None)
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--dit", type=str, default=None)
    ap.add_argument("--vae", type=str, default=None)
    ap.add_argument("--text_encoder", type=str, default=None)
    ap.add_argument("--label", type=str, default="cmp-samplers")
    args = ap.parse_args()

    device = torch.device(args.device)
    samples = pool_samples(
        args.n,
        pool_dir=args.pool_dir,
        caption_dir=args.caption_dir,
        buckets=args.buckets,
        seed=args.pool_seed,
    )
    print(
        f"[{len(samples)} pool prompt(s); samplers={args.samplers}; "
        f"seeds={args.seeds}; steps={args.steps} cfg={args.guidance_scale}]"
    )

    out_dir = make_run_dir("spectrum_pareto", label=args.label)
    print(f"out_dir = {out_dir}")
    cap_root = out_dir / "captures"

    shared_models: dict = {}  # reuse DiT/text encoder across every capture
    # per_sampler[sampler] = list of per-(prompt,seed) summaries (paired by job_id)
    per_sampler: dict[str, list[dict]] = {s: [] for s in args.samplers}
    pairs: list[dict] = []  # one row per (prompt, seed): residual per sampler

    for s_i, sample in enumerate(samples):
        prompt, hw = sample["prompt"], sample["hw"]
        for seed in args.seeds:
            job_id = f"p{s_i:02d}_seed{seed}"
            pair: dict = {
                "job": job_id,
                "prompt": prompt[:80],
                "hw": list(hw),
                "bucket": sample.get("latent_bucket"),
                "seed": seed,
            }
            for sampler in args.samplers:
                print(f"\n=== {job_id} [{sampler}]: '{prompt[:55]}' {hw} ===")
                payload = capture_one(
                    prompt,
                    seed,
                    hw,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    flow_shift=args.flow_shift,
                    sampler=sampler,
                    negative_prompt=args.negative_prompt,
                    lora=args.lora,
                    device=args.device,
                    dit=args.dit,
                    vae=args.vae,
                    text_encoder=args.text_encoder,
                    capture_uncond=False,
                    shared_models=shared_models,
                    compile_blocks=args.compile_blocks,
                    compile_inductor_mode=args.compile_inductor_mode,
                )
                if args.save_captures:
                    d = cap_root / sampler
                    d.mkdir(parents=True, exist_ok=True)
                    np.savez(d / f"{job_id}_{_slug(prompt)}.npz", **payload)
                feats, fhw, num_steps = _feats_from_payload(payload, device)
                summ = replay_at_combo(feats, fhw, num_steps, device, PROD_COMBO)
                del feats
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if summ is None:
                    print("  (no cached steps — skipped)")
                    continue
                summ = {**summ, "job": job_id}
                per_sampler[sampler].append(summ)
                pair[f"{sampler}_mean"] = round(summ["mean_rel"], 5)
                pair[f"{sampler}_peak"] = round(summ["peak_rel"], 5)
                pair[f"{sampler}_late"] = round(summ["late_rel"], 5)
                pair["n_cached"] = summ["n_cached"]
                print(
                    f"  rel-L2 mean={summ['mean_rel']:.4f} peak={summ['peak_rel']:.4f} "
                    f"late={summ['late_rel']:.4f} cached={summ['n_cached']}"
                )
            pairs.append(pair)

    # Aggregate per sampler.
    def _agg(recs: list[dict]) -> dict:
        return {
            "n": len(recs),
            "mean_rel_l2": float(np.mean([r["mean_rel"] for r in recs]))
            if recs
            else None,
            "peak_rel_l2": float(np.mean([r["peak_rel"] for r in recs]))
            if recs
            else None,
            "late_rel_l2": float(np.mean([r["late_rel"] for r in recs]))
            if recs
            else None,
            "mean_rel_l2_std": float(np.std([r["mean_rel"] for r in recs]))
            if recs
            else None,
        }

    agg = {s: _agg(per_sampler[s]) for s in args.samplers}

    # Verdict: candidate (first sampler) vs baseline (last sampler).
    candidate, baseline = args.samplers[0], args.samplers[-1]
    verdict = {"candidate": candidate, "baseline": baseline, "tol": args.tol}
    if (
        candidate != baseline
        and agg[candidate]["mean_rel_l2"]
        and agg[baseline]["mean_rel_l2"]
    ):
        ratio = agg[candidate]["mean_rel_l2"] / agg[baseline]["mean_rel_l2"]
        # Paired ratios guard against prompt-mix imbalance between samplers.
        by_job = {}
        for s in (candidate, baseline):
            for r in per_sampler[s]:
                by_job.setdefault(r["job"], {})[s] = r["mean_rel"]
        paired = [v for v in by_job.values() if candidate in v and baseline in v]
        paired_ratio = (
            float(np.median([v[candidate] / v[baseline] for v in paired]))
            if paired
            else None
        )
        passed = ratio <= args.tol
        verdict.update(
            {
                "mean_ratio": round(ratio, 4),
                "paired_median_ratio": round(paired_ratio, 4) if paired_ratio else None,
                "n_paired": len(paired),
                "pass": bool(passed),
                "summary": (
                    f"{candidate} forecast residual is {ratio:.2f}× {baseline} "
                    f"(tol {args.tol}) → {'PASS' if passed else 'FAIL'}"
                ),
            }
        )
    else:
        verdict["summary"] = "single sampler — no comparison"

    # verdict.md
    lines = [
        "# Spectrum forecastability probe — sampler comparison\n",
        f"- prompts: {len(samples)} from `{args.pool_dir}`"
        f"{' buckets=' + ','.join(args.buckets) if args.buckets else ''}",
        f"- seeds: {args.seeds} · steps: {args.steps} · cfg: {args.guidance_scale}",
        f"- forecaster: production default {PROD_COMBO}",
        "",
        "| sampler | n | mean rel-L2 | peak | late-third | std |",
        "|---|--:|--:|--:|--:|--:|",
    ]

    def _f(x):
        return f"{x:.4f}" if isinstance(x, float) else "—"

    for s in args.samplers:
        a = agg[s]
        lines.append(
            f"| {s} | {a['n']} | {_f(a['mean_rel_l2'])} | {_f(a['peak_rel_l2'])} | "
            f"{_f(a['late_rel_l2'])} | {_f(a['mean_rel_l2_std'])} |"
        )
    lines += ["", f"**Verdict:** {verdict.get('summary')}"]
    if "pass" in verdict:
        lines.append(
            f"\n- mean ratio ({candidate}/{baseline}): {verdict['mean_ratio']}"
            f" · paired median ratio: {verdict['paired_median_ratio']}"
            f" (n_paired={verdict['n_paired']})"
        )
        lines.append(
            "\n→ "
            + (
                "Pool reuse holds: build the bench on er_sde, reuse stored x0 as "
                "free Phase-1/4 endpoints."
                if verdict["pass"]
                else "Pool reuse rejected: er_sde forecasts materially worse — target "
                "euler with fresh generation."
            )
        )
    (out_dir / "verdict.md").write_text("\n".join(lines) + "\n")

    # per-pair CSV
    import csv

    if pairs:
        keys = sorted({k for p in pairs for k in p})
        with open(out_dir / "pairs.csv", "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=keys)
            wri.writeheader()
            wri.writerows(pairs)

    metrics = {
        "samplers": args.samplers,
        "n_prompts": len(samples),
        "seeds": args.seeds,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "prod_combo": PROD_COMBO,
        "aggregate": agg,
        "verdict": verdict,
        "pairs": pairs,
    }
    artifacts = ["verdict.md", "pairs.csv"] + (
        ["captures"] if args.save_captures else []
    )
    result_path = write_result(
        out_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=artifacts,
        device=args.device,
    )
    print("\n" + (out_dir / "verdict.md").read_text())
    print(f"result → {result_path}")


if __name__ == "__main__":
    main()
