#!/usr/bin/env python3
"""Spectrum × SeaCache — is a spectral-evolution-aware *decision metric* better?

Phase-0 probe (no training). Spectrum (Chebyshev feature forecasting) already
ships in this repo and the ComfyUI node; its *scheduling* is a content-blind
growing-window heuristic. SeaCache (arXiv:2602.18993v2) proposes a smarter
*decision metric*: measure cache distance in a Spectral-Evolution-Aware (SEA)
filtered space that downweights high-freq noise. This bench asks whether that
metric would actually make better skip decisions **on Anima** — where several
spectral-evolution priors have inverted before (CTCal, σ-reshape; x̂₀ resolves
by σ≈0.45), so it must be measured, not assumed.

Three candidate decision metrics are compared against a ground-truth skip-cost:

  raw_input   ‖Δ x_t‖            relative-L1 consecutive latent distance (TeaCache-style)
  sea_input   ‖Δ SEA_σ(x_t)‖     SeaCache's metric (SEA-filtered input distance)
  resid_feat  ‖feat − Cheb(feat)‖ Spectrum-NATIVE: its own forecast residual on the
                                  block-stack feature (simulated offline). This is the
                                  control that matters — Spectrum can *measure* its reuse
                                  error; SeaCache can only *proxy* it from the input.

Ground truth (skip-cost at step i) = SEA-filtered change of the guided running
x̂₀ estimate, ‖Δ SEA_σ(x̂₀)‖ — how much perceptual content the step actually
moves (SeaCache's "SEA(Output)" reference, Fig. 5, taken on x̂₀ which carries the
content). The unfiltered ‖Δ x̂₀‖ is reported alongside as a robustness check.

Claims tested:
  A  corr(sea_input) > corr(raw_input)?   → SEA filtering helps on Anima (SeaCache's core)
  B  corr(resid_feat) ≳ corr(sea_input)?  → Spectrum's own residual beats the input proxy
  C  where does skip-cost concentrate in σ vs where the blind window spends compute?

Build the SEA-residual schedule only if A holds AND C shows the blind window is
mismatched. If raw≈SEA on Anima, the SEA idea dies cheaply.

Capture seams (one real euler generate per prompt, eager — no compile):
  * forward-pre-hook on ``model.final_layer`` → the block-stack feature (cond branch),
    the exact tensor Spectrum forecasts (networks/spectrum.py:287).
  * monkeypatch ``library.inference.sampling.step`` → guided x̂₀ = x_t − σ·v
    (generation.py:799), the post-CFG velocity, x_t and σ per step.

Usage::

    python bench/spectrum_sea/run_bench.py --steps 24 --cfg 4.0 --label baseline
    python bench/spectrum_sea/run_bench.py --prompts-file my_prompts.txt --beta 2.0
"""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from bench.spectrum_sea.sea import l1rel, sea_filter  # noqa: E402

import anima_lora  # noqa: E402
import library.inference.sampling as _sampling  # noqa: E402
from library.runtime.harness import build_inference_bundle  # noqa: E402
from networks.spectrum import SpectrumPredictor  # noqa: E402

# Real, moderately-specified captions (the regime SeaCache averages over). The
# probe is content-adaptive, so variety across prompts is the point.
DEFAULT_PROMPTS = [
    "1girl, long silver hair, red eyes, school uniform, classroom, afternoon light",
    "1girl, kimono, holding a paper umbrella, cherry blossoms, full body, detailed",
    "1boy, hoodie, city street at night, neon signs, rain, cinematic lighting",
    "2girls, sitting on a rooftop, summer, blue sky, clouds, wide shot",
    "1girl, witch hat, casting a spell, glowing particles, dark forest background",
    "1girl, swimsuit, beach, ocean waves, sunset, looking back at viewer",
    "1girl, maid outfit, serving tea, cozy cafe interior, warm lighting",
    "1boy, samurai armor, holding a katana, snowy mountain, dramatic sky",
    "1girl, mecha pilot suit, cockpit, holographic displays, sci-fi",
    "1girl, gothic lolita dress, sitting in a library, candlelight, books",
    "2boys, sparring, dojo, motion blur, dynamic pose, wide shot",
    "1girl, angel wings, floating in clouds, soft sunlight, ethereal",
]

TAIL_SPLIT = 0.45  # σ below this = resolve tail (memory: Anima x̂₀ done by ~0.45)

# Spectrum's shipped defaults (spectrum_denoise signature + node).
WARMUP, WINDOW, FLEX = 6, 2.0, 0.25
CHEB_M, CHEB_LAM, CHEB_W = 3, 0.1, 0.3


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #
class Capture:
    """Collects per-step (x_t, σ, v, x̂₀, block-feature) over one generate()."""

    def __init__(self) -> None:
        self.feat_buf: list[torch.Tensor] = []
        self.steps: list[dict] = []

    def final_layer_pre_hook(self, module, args):
        # args[0] = pre-final block-stack feature. Under CFG the engine fires
        # cond then uncond (two forwards / a batched pair); take batch row 0 =
        # cond, on CPU fp16 to bound memory across ~24 steps.
        feat = args[0]
        self.feat_buf.append(feat[0:1].detach().to("cpu", torch.float16))

    @contextmanager
    def patched_step(self):
        orig = _sampling.step

        def patched(latents, noise_pred, sigmas, step_i):
            feat = self.feat_buf[0] if self.feat_buf else None
            self.feat_buf = []
            sig = float(sigmas[step_i])
            # latents/noise_pred are 5D (B,C,1,H,W) at the sampler seam; B=1.
            x_t = latents[0, :, 0].detach().to("cpu", torch.float32)  # (C,H,W)
            v = noise_pred[0, :, 0].detach().to("cpu", torch.float32)
            x0 = x_t - sig * v  # guided x̂₀ (generation.py:799)
            self.steps.append(
                {"i": int(step_i), "sigma": sig, "x_t": x_t, "x0": x0, "feat": feat}
            )
            return orig(latents, noise_pred, sigmas, step_i)

        _sampling.step = patched
        try:
            yield
        finally:
            _sampling.step = orig


# --------------------------------------------------------------------------- #
# offline analysis
# --------------------------------------------------------------------------- #
def forecast_residuals(
    feats: list[torch.Tensor], warmup: int, m: int, lam: float, w: float, total: int
) -> list[float]:
    """Per-step Chebyshev forecast residual (relative L2), simulated offline.

    Mirrors Spectrum's calibration path (spectrum.py:340): at step i, predict
    the block feature from all prior steps, then update. resid[i] is the error
    injected if step i alone were cached given everything before it was computed
    — a clean per-step skip-cost in feature space. NaN where undefined.
    """
    n = len(feats)
    resid = [float("nan")] * n
    fc = SpectrumPredictor(m, lam, w, torch.device("cpu"), feats[0].shape, total)
    for i in range(n):
        fi = feats[i].to(torch.float32)
        if i >= warmup and fc.cheb.t_buf.numel() >= max(2, m + 2):
            pred = fc.predict(float(i)).to(torch.float32)
            resid[i] = float((fi - pred).norm() / (fi.norm() + 1e-8))
        fc.update(float(i), fi)
    return resid


def per_prompt_metrics(steps: list[dict], beta: float) -> dict:
    """Compute the three decision metrics + GT for one prompt's trajectory.

    All pairwise quantities are aligned to the *end* step i (pair i-1 → i), so
    resid_feat[i] lines up with the distance metrics for the same step.
    """
    steps = sorted(steps, key=lambda r: r["i"])
    sig = [s["sigma"] for s in steps]
    feats = [s["feat"] for s in steps if s["feat"] is not None]
    have_feat = len(feats) == len(steps)
    resid = (
        forecast_residuals(feats, WARMUP, CHEB_M, CHEB_LAM, CHEB_W, len(steps))
        if have_feat
        else [float("nan")] * len(steps)
    )

    rows = []
    for i in range(1, len(steps)):
        xt, xt_p = steps[i]["x_t"], steps[i - 1]["x_t"]
        x0, x0_p = steps[i]["x0"], steps[i - 1]["x0"]
        rows.append(
            {
                "sigma": sig[i],
                "raw_input": l1rel(xt, xt_p),
                "sea_input": l1rel(
                    sea_filter(xt, sig[i], beta), sea_filter(xt_p, sig[i - 1], beta)
                ),
                "resid_feat": resid[i],
                "gt_x0": l1rel(
                    sea_filter(x0, sig[i], beta), sea_filter(x0_p, sig[i - 1], beta)
                ),
                "gt_x0_raw": l1rel(x0, x0_p),
            }
        )
    return {"rows": rows, "have_feat": have_feat}


def spearman(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def stratified_spearman(m, g, sig, min_group: int = 4) -> float:
    """Step-stratified Spearman: rank metric vs GT *across prompts at each step*.

    The pooled (untrended) correlation is dominated by the monotone denoising
    trend that the blind growing-window schedule *already* exploits — so it can't
    tell whether the metric adds anything over a fixed schedule. Under the shared
    sampler schedule, σ at a given step is *identical* across prompts, so grouping
    rows by σ-value groups them by step, making σ exactly constant within each
    group. The trend is then fully removed and the within-group rank correlation
    isolates the only thing a content-adaptive metric can add: correctly ranking
    *which prompt* is costlier to skip at this step. This is the verdict number.
    """
    m = np.asarray(m, float)
    g = np.asarray(g, float)
    s = np.asarray(sig, float)
    ok = np.isfinite(m) & np.isfinite(g) & np.isfinite(s)
    m, g, s = m[ok], g[ok], s[ok]
    rm, rg = [], []
    for val in np.unique(np.round(s, 6)):
        idx = np.where(np.round(s, 6) == val)[0]
        if len(idx) < min_group:
            continue
        mm, gg = m[idx], g[idx]
        if mm.std() < 1e-12 or gg.std() < 1e-12:
            continue
        rm.append(mm.argsort().argsort() / (len(idx) - 1))
        rg.append(gg.argsort().argsort() / (len(idx) - 1))
    if not rm:
        return float("nan")
    rm, rg = np.concatenate(rm), np.concatenate(rg)
    if rm.std() < 1e-12 or rg.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rm, rg)[0, 1])


def blind_schedule(n: int, warmup: int, window: float, flex: float) -> list[str]:
    """Replicate Spectrum's content-blind schedule (spectrum.py:306-387)."""
    stop_at = n - 3
    curr_ws, consec = window, 0
    modes = []
    for i in range(n):
        actual = (i < warmup or i >= stop_at) or (
            (consec + 1) % max(1, math.floor(curr_ws)) == 0
        )
        modes.append("actual" if actual else "cached")
        if actual:
            if i >= warmup:
                curr_ws = round(curr_ws + flex, 3)
            consec = 0
        else:
            consec += 1
    return modes


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def make_figures(run_dir: Path, rows: list[dict], modes: list[str]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.array([r["gt_x0"] for r in rows])
    arts = []

    # Fig 1: metric vs GT scatter (raw vs SEA vs residual)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, title in [
        (axes[0], "raw_input", "raw ‖Δx_t‖"),
        (axes[1], "sea_input", "SEA ‖Δx_t‖ (SeaCache)"),
        (axes[2], "resid_feat", "Chebyshev residual (Spectrum)"),
    ]:
        m = np.array([r[key] for r in rows], float)
        ok = np.isfinite(m) & np.isfinite(gt)
        ax.scatter(m[ok], gt[ok], s=12, alpha=0.5)
        ax.set_xlabel(title)
        ax.set_ylabel("skip-cost  SEA ‖Δx̂₀‖")
        ax.set_title(f"ρ = {spearman(m, gt):+.3f}", fontsize=11)
    fig.suptitle("decision metric vs ground-truth skip-cost", fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / "metric_vs_gt.png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    arts.append("metric_vs_gt.png")

    # Fig 2: skip-cost vs σ + blind compute density
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sig = np.array([r["sigma"] for r in rows])
    order = sig.argsort()
    ax.plot(sig[order], gt[order], "o-", ms=4, label="skip-cost  SEA ‖Δx̂₀‖")
    ax.axvline(TAIL_SPLIT, color="k", ls="--", lw=1, label=f"σ={TAIL_SPLIT}")
    ax.set_xlabel("σ")
    ax.set_ylabel("skip-cost")
    ax.invert_xaxis()  # denoising runs high σ → low σ
    ax2 = ax.twinx()
    # blind compute density per σ-bin (modes is over all N steps; align to step σ)
    ax2.set_ylabel("blind: actual=1 / cached=0", color="tab:red")
    ax.legend(loc="upper right")
    ax.set_title("where skip-cost concentrates vs where the blind window computes")
    fig.tight_layout()
    fig.savefig(run_dir / "skipcost_vs_sigma.png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    arts.append("skipcost_vs_sigma.png")
    return arts


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def make_args(ckpt, prompt, seed, a):
    req = anima_lora.GenerationRequest(
        dit=ckpt.dit,
        vae=ckpt.vae,
        text_encoder=ckpt.text_encoder,
        prompt=prompt,
        image_size=(a.size, a.size),
        infer_steps=a.steps,
        guidance_scale=a.cfg,
        flow_shift=a.flow_shift,
        sampler="euler",  # force euler so sampling.step fires (no er_sde/spectrum)
        seed=seed,
        # Fixed-res square gen → one static token count, so per-block compile
        # (no dynamic_seq) is safe and bit-exact; first generate pays warmup,
        # the reused bundle amortizes it over the remaining prompts. final_layer
        # gets the 5D feature either way (_unflatten_native_shape runs before
        # the head), so capture is unaffected by compile.
        extra_argv=("--compile_blocks",) if not a.no_compile else (),
    )
    return req.to_args()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow-shift", dest="flow_shift", type=float, default=3.0)
    p.add_argument("--size", type=int, default=1024, help="pixel edge (square)")
    p.add_argument("--beta", type=float, default=2.0, help="natural-image power-law β")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--label", default=None)
    p.add_argument(
        "--no-compile",
        dest="no_compile",
        action="store_true",
        help="disable compile_blocks (compile is ON by default for speed)",
    )
    a = p.parse_args()

    if a.prompts_file:
        prompts = [
            ln.strip()
            for ln in Path(a.prompts_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        prompts = DEFAULT_PROMPTS

    ckpt = anima_lora.default_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir("spectrum_sea", label=a.label)
    mode = "eager" if a.no_compile else "compiled"
    print(f"run dir: {run_dir}")
    print(f"{len(prompts)} prompts, {a.steps} steps, cfg {a.cfg}, β {a.beta}, {mode}")

    print(f"loading bundle ({mode}, latent-space probe)...")
    bundle = build_inference_bundle(
        make_args(ckpt, prompts[0], a.seed, a), device=device, with_vae=False
    )

    all_rows: list[dict] = []
    per_prompt: list[dict] = []
    feat_ok = 0
    modes = blind_schedule(a.steps, WARMUP, WINDOW, FLEX)

    for pi, prompt in enumerate(prompts):
        print(f"\n[{pi + 1}/{len(prompts)}] {prompt[:60]}")
        cap = Capture()
        h = bundle.model.final_layer.register_forward_pre_hook(cap.final_layer_pre_hook)
        try:
            with cap.patched_step():
                bundle.generate(make_args(ckpt, prompt, a.seed, a))
        finally:
            h.remove()
        pm = per_prompt_metrics(cap.steps, a.beta)
        feat_ok += int(pm["have_feat"])
        for r in pm["rows"]:
            r["prompt_idx"] = pi
        all_rows.extend(pm["rows"])
        gt = [r["gt_x0"] for r in pm["rows"]]
        per_prompt.append(
            {
                "prompt": prompt,
                "n_steps_captured": len(cap.steps),
                "have_feat": pm["have_feat"],
                "rho_raw": spearman([r["raw_input"] for r in pm["rows"]], gt),
                "rho_sea": spearman([r["sea_input"] for r in pm["rows"]], gt),
                "rho_resid": spearman([r["resid_feat"] for r in pm["rows"]], gt),
            }
        )
        print(
            f"  captured {len(cap.steps)} steps (feat={'ok' if pm['have_feat'] else 'MISSING'}); "
            f"ρ raw={per_prompt[-1]['rho_raw']:+.3f} "
            f"sea={per_prompt[-1]['rho_sea']:+.3f} "
            f"resid={per_prompt[-1]['rho_resid']:+.3f}"
        )

    # pooled correlations (the headline numbers)
    gt_all = [r["gt_x0"] for r in all_rows]
    gt_raw_all = [r["gt_x0_raw"] for r in all_rows]

    def corr(key, target):
        return spearman([r[key] for r in all_rows], target)

    # regime split
    tail = [r for r in all_rows if r["sigma"] < TAIL_SPLIT]
    early = [r for r in all_rows if r["sigma"] >= TAIL_SPLIT]
    gt_tail = [r["gt_x0"] for r in tail]
    gt_early = [r["gt_x0"] for r in early]

    # σ-detrended (the number that actually matters — within-σ adaptive signal)
    sig_all = [r["sigma"] for r in all_rows]

    def dcorr(key, target):
        return stratified_spearman([r[key] for r in all_rows], target, sig_all)

    pooled = {
        "rho_raw_vs_gt": corr("raw_input", gt_all),
        "rho_sea_vs_gt": corr("sea_input", gt_all),
        "rho_resid_vs_gt": corr("resid_feat", gt_all),
        # robustness: also vs the unfiltered x̂₀ change
        "rho_raw_vs_gtraw": corr("raw_input", gt_raw_all),
        "rho_sea_vs_gtraw": corr("sea_input", gt_raw_all),
        "rho_resid_vs_gtraw": corr("resid_feat", gt_raw_all),
        # Claim A / B deltas (trend-included — diagnostic only, NOT the verdict)
        "A_sea_minus_raw": corr("sea_input", gt_all) - corr("raw_input", gt_all),
        "B_resid_minus_sea": corr("resid_feat", gt_all) - corr("sea_input", gt_all),
        # σ-DETRENDED (the verdict): within-σ adaptive signal vs envelope-free GT
        "d_rho_raw_vs_gtraw": dcorr("raw_input", gt_raw_all),
        "d_rho_sea_vs_gtraw": dcorr("sea_input", gt_raw_all),
        "d_rho_resid_vs_gtraw": dcorr("resid_feat", gt_raw_all),
        "dA_sea_minus_raw": dcorr("sea_input", gt_raw_all)
        - dcorr("raw_input", gt_raw_all),
        "dB_resid_minus_sea": dcorr("resid_feat", gt_raw_all)
        - dcorr("sea_input", gt_raw_all),
        # Claim C: skip-cost concentration vs blind compute
        "skipcost_frac_tail": float(np.sum(gt_tail) / (np.sum(gt_all) + 1e-9)),
        "skipcost_frac_early": float(np.sum(gt_early) / (np.sum(gt_all) + 1e-9)),
        "blind_actual_frac_overall": float(np.mean([m == "actual" for m in modes])),
    }

    # per-step CSV so re-analysis never needs another GPU run
    import csv

    csv_path = run_dir / "per_step.csv"
    cols = [
        "prompt_idx",
        "sigma",
        "raw_input",
        "sea_input",
        "resid_feat",
        "gt_x0",
        "gt_x0_raw",
    ]
    with csv_path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in all_rows:
            wr.writerow({k: r.get(k) for k in cols})

    arts = (make_figures(run_dir, all_rows, modes) if all_rows else []) + [
        "per_step.csv"
    ]

    metrics = {
        "pooled": pooled,
        "per_prompt": per_prompt,
        "config": {
            "warmup": WARMUP,
            "window": WINDOW,
            "flex": FLEX,
            "cheb_m": CHEB_M,
            "cheb_lam": CHEB_LAM,
            "cheb_w": CHEB_W,
            "beta": a.beta,
            "tail_split": TAIL_SPLIT,
            "blind_schedule": modes,
            "feat_capture_prompts": f"{feat_ok}/{len(prompts)}",
        },
    }
    write_result(
        run_dir,
        script=__file__,
        args=a,
        metrics=metrics,
        label=a.label,
        artifacts=arts,
        device=device,
    )

    print("\n=== trend-included (DIAGNOSTIC — dominated by the monotone σ-trend) ===")
    print(
        f"  raw ρ={pooled['rho_raw_vs_gt']:+.3f}  sea ρ={pooled['rho_sea_vs_gt']:+.3f}  "
        f"resid ρ={pooled['rho_resid_vs_gt']:+.3f}   (NOT the verdict)"
    )
    print(
        "\n=== σ-DETRENDED (THE VERDICT — within-σ adaptive signal, envelope-free GT) ==="
    )
    print(f"  raw_input   ρ = {pooled['d_rho_raw_vs_gtraw']:+.3f}")
    print(f"  sea_input   ρ = {pooled['d_rho_sea_vs_gtraw']:+.3f}   (SeaCache metric)")
    print(
        f"  resid_feat  ρ = {pooled['d_rho_resid_vs_gtraw']:+.3f}   (Spectrum native)"
    )
    print(
        f"  A: SEA − raw    = {pooled['dA_sea_minus_raw']:+.3f}  "
        f"({'SEA helps on Anima' if pooled['dA_sea_minus_raw'] > 0.05 else 'no real SEA win on Anima'})"
    )
    print(
        f"  B: resid − SEA  = {pooled['dB_resid_minus_sea']:+.3f}  "
        f"({'Spectrum residual wins' if pooled['dB_resid_minus_sea'] > 0.05 else 'input proxy ≥ residual'})"
    )
    print(
        f"  C: skip-cost in σ<{TAIL_SPLIT} = {pooled['skipcost_frac_tail']:.2%}  "
        f"(blind computes {pooled['blind_actual_frac_overall']:.0%} of steps)"
    )
    print(f"\nresult.json + figures + per_step.csv in: {run_dir}")


if __name__ == "__main__":
    main()
