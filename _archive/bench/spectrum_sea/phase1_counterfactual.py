#!/usr/bin/env python3
"""Phase 1 — decision-grade: do the metrics rank the TRUE counterfactual skip-cost?

Phase 0 (run_bench.py) confirmed SEA-filtering the cache-decision distance beats
raw L2 on Anima, but judged metrics against a *proxy* skip-cost (x̂₀ motion). This
replaces the proxy with the true counterfactual: at each step, actually CACHE it —
Chebyshev-forecast the cond+uncond block features, run only the head
(``_spectrum_fast_forward`` = t_embedder + final_layer + unpatchify), CFG-combine,
and measure how far the resulting x̂₀ lands from the real full-compute x̂₀. That
deviation IS the cost of skipping step i.

Non-circularity. A cache scheduler must decide skip-or-compute *before* computing
step i, so a metric is only fair if it uses information available then. The
SEA-filtered residual measured AT step i equals the counterfactual GT itself
(ρ=1, useless). So the metrics under test are the genuinely *available* ones:

  sea_input   ‖Δ SEA_σ(x_t)‖   leading — SeaCache, computable before the forward
  raw_input   ‖Δ x_t‖          leading — baseline
  lag_resid   last actual-step forecast residual under the blind schedule —
              lagging, the signal Spectrum actually has when it decides

GT_cf = l1rel(SEA_σ(x̂₀_true), SEA_σ(x̂₀_cached)) — the real injected content error.
We also re-score the Phase-0 proxy (x̂₀ motion) against GT_cf to see whether the
proxy misled the Phase-0 verdict.

All correlations are step-stratified (σ constant across prompts at a step → the
monotone trend is removed; see run_bench.stratified_spearman).

Usage::

    python bench/spectrum_sea/phase1_counterfactual.py --label phase1
"""

from __future__ import annotations

import argparse
import csv
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
from bench.spectrum_sea.run_bench import (  # noqa: E402
    CHEB_LAM,
    CHEB_M,
    CHEB_W,
    DEFAULT_PROMPTS,
    FLEX,
    WARMUP,
    WINDOW,
    blind_schedule,
    forecast_residuals,
    spearman,
    stratified_spearman,
)

import anima_lora  # noqa: E402
import library.inference.sampling as _sampling  # noqa: E402
from library.runtime.harness import build_inference_bundle  # noqa: E402
from networks.spectrum import SpectrumPredictor, _spectrum_fast_forward  # noqa: E402


# --------------------------------------------------------------------------- #
# capture — both CFG branches + the model timestep (for the head replay)
# --------------------------------------------------------------------------- #
class Capture:
    def __init__(self) -> None:
        self.feat_buf: list[torch.Tensor] = []
        self.t_buf: list[torch.Tensor] = []
        self.steps: list[dict] = []

    def final_layer_pre_hook(self, module, args):
        # keep every firing this step: [cond, uncond] under CFG (cond fires first,
        # generation.py:415-448). On CPU fp16 to bound memory across both branches.
        self.feat_buf.append(args[0].detach().to("cpu", torch.float16))

    def t_embedder_pre_hook(self, module, args):
        # raw timesteps_B_T entering the head; identical for cond/uncond, grab once.
        self.t_buf.append(args[0].detach().clone())

    @contextmanager
    def patched_step(self):
        orig = _sampling.step

        def patched(latents, noise_pred, sigmas, step_i):
            feats = list(self.feat_buf)
            t_model = self.t_buf[0] if self.t_buf else None
            self.feat_buf, self.t_buf = [], []
            sig = float(sigmas[step_i])
            x_t = latents[0, :, 0].detach().to("cpu", torch.float32)  # (C,H,W)
            v = noise_pred[0, :, 0].detach().to("cpu", torch.float32)
            self.steps.append(
                {
                    "i": int(step_i),
                    "sigma": sig,
                    "x_t": x_t,
                    "x0_true": x_t - sig * v,
                    "feats": feats,
                    "t_model": t_model,
                }
            )
            return orig(latents, noise_pred, sigmas, step_i)

        _sampling.step = patched
        try:
            yield
        finally:
            _sampling.step = orig


# --------------------------------------------------------------------------- #
# counterfactual: cache each step for real, measure injected x̂₀ error
# --------------------------------------------------------------------------- #
@torch.no_grad()
def counterfactual_skipcost(
    model, steps: list[dict], device, beta: float, guidance_scale: float
) -> list[dict]:
    """For each post-warmup step, forecast+head-replay both branches and measure
    the true injected x̂₀ error (raw + SEA-weighted). Returns per-step rows."""
    n = len(steps)
    have_cfg = all(len(s["feats"]) >= 2 for s in steps)
    cond = [s["feats"][0] for s in steps]
    unc = [s["feats"][1] for s in steps] if have_cfg else cond

    fc_c = SpectrumPredictor(CHEB_M, CHEB_LAM, CHEB_W, torch.device("cpu"), cond[0].shape, n)
    fc_u = SpectrumPredictor(CHEB_M, CHEB_LAM, CHEB_W, torch.device("cpu"), unc[0].shape, n)
    rows: list[dict] = []
    for i in range(n):
        s = steps[i]
        if (
            i >= WARMUP
            and fc_c.cheb.t_buf.numel() >= max(2, CHEB_M + 2)
            and s["t_model"] is not None
        ):
            t_m = s["t_model"].to(device)
            cf = fc_c.predict(float(i)).to(device, torch.bfloat16)
            nc = _spectrum_fast_forward(model, t_m, cf)[0, :, 0].float().cpu()
            if have_cfg:
                uf = fc_u.predict(float(i)).to(device, torch.bfloat16)
                nu = _spectrum_fast_forward(model, t_m, uf)[0, :, 0].float().cpu()
                noise_cached = nu + guidance_scale * (nc - nu)
            else:
                noise_cached = nc
            x0_cached = s["x_t"] - s["sigma"] * noise_cached
            gt_cf = l1rel(
                sea_filter(s["x0_true"], s["sigma"], beta),
                sea_filter(x0_cached, s["sigma"], beta),
            )
            gt_cf_raw = l1rel(s["x0_true"], x0_cached)
        else:
            gt_cf = gt_cf_raw = float("nan")
        rows.append({"i": i, "sigma": s["sigma"], "gt_cf": gt_cf, "gt_cf_raw": gt_cf_raw})
        fc_c.update(float(i), cond[i].to(torch.float32))
        fc_u.update(float(i), unc[i].to(torch.float32))
    return rows


def lagging_residual(resid: list[float], modes: list[str]) -> list[float]:
    """The forecast residual carried from the most recent *actual* step — the
    signal Spectrum has on hand when it decides whether to cache step i."""
    out, last = [], float("nan")
    for i, mode in enumerate(modes):
        out.append(last)  # value available BEFORE deciding step i
        if mode == "actual" and i < len(resid) and np.isfinite(resid[i]):
            last = resid[i]
    return out


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
        sampler="euler",
        seed=seed,
        extra_argv=("--compile_blocks",) if not a.no_compile else (),
    )
    return req.to_args()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow-shift", dest="flow_shift", type=float, default=3.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--no-compile", dest="no_compile", action="store_true")
    a = p.parse_args()

    if a.cfg == 1.0:
        sys.exit("Phase 1 needs CFG>1 to exercise the cond/uncond cache replay.")
    prompts = (
        [
            ln.strip()
            for ln in Path(a.prompts_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        if a.prompts_file
        else DEFAULT_PROMPTS
    )

    ckpt = anima_lora.default_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir("spectrum_sea", label=a.label)
    print(f"run dir: {run_dir}")
    print(f"{len(prompts)} prompts, {a.steps} steps, cfg {a.cfg}, β {a.beta}")

    bundle = build_inference_bundle(
        make_args(ckpt, prompts[0], a.seed, a), device=device, with_vae=False
    )
    modes = blind_schedule(a.steps, WARMUP, WINDOW, FLEX)

    all_rows: list[dict] = []
    for pi, prompt in enumerate(prompts):
        print(f"\n[{pi + 1}/{len(prompts)}] {prompt[:58]}")
        cap = Capture()
        hf = bundle.model.final_layer.register_forward_pre_hook(cap.final_layer_pre_hook)
        ht = bundle.model.t_embedder.register_forward_pre_hook(cap.t_embedder_pre_hook)
        try:
            with cap.patched_step():
                bundle.generate(make_args(ckpt, prompt, a.seed, a))
        finally:
            hf.remove()
            ht.remove()

        st = sorted(cap.steps, key=lambda r: r["i"])
        # true counterfactual skip-cost (head replay)
        cf = counterfactual_skipcost(bundle.model, st, device, a.beta, a.cfg)
        # leading decision metrics from x_t + the lagging residual
        feats_ok = all(s["feats"] for s in st)
        cond_feats = [s["feats"][0] for s in st] if feats_ok else []
        resid = (
            forecast_residuals(cond_feats, WARMUP, CHEB_M, CHEB_LAM, CHEB_W, len(st))
            if feats_ok
            else [float("nan")] * len(st)
        )
        lag = lagging_residual(resid, modes)
        for i in range(1, len(st)):
            all_rows.append(
                {
                    "prompt_idx": pi,
                    "sigma": st[i]["sigma"],
                    "raw_input": l1rel(st[i]["x_t"], st[i - 1]["x_t"]),
                    "sea_input": l1rel(
                        sea_filter(st[i]["x_t"], st[i]["sigma"], a.beta),
                        sea_filter(st[i - 1]["x_t"], st[i - 1]["sigma"], a.beta),
                    ),
                    "lag_resid": lag[i],
                    # Phase-0 proxy (x̂₀ motion), to check whether it misled
                    "proxy_x0": l1rel(
                        sea_filter(st[i]["x0_true"], st[i]["sigma"], a.beta),
                        sea_filter(st[i - 1]["x0_true"], st[i - 1]["sigma"], a.beta),
                    ),
                    "gt_cf": cf[i]["gt_cf"],
                    "gt_cf_raw": cf[i]["gt_cf_raw"],
                }
            )
        nvalid = sum(np.isfinite(cf[i]["gt_cf"]) for i in range(1, len(st)))
        print(f"  {len(st)} steps, {nvalid} counterfactual GT points")

    sig = [r["sigma"] for r in all_rows]
    gt = [r["gt_cf"] for r in all_rows]
    gt_raw = [r["gt_cf_raw"] for r in all_rows]

    def d(key, target):
        return stratified_spearman([r[key] for r in all_rows], target, sig)

    pooled = {
        # THE decision-grade verdict: leading metrics vs TRUE counterfactual cost
        "d_sea_vs_cf": d("sea_input", gt),
        "d_raw_vs_cf": d("raw_input", gt),
        "d_lag_resid_vs_cf": d("lag_resid", gt),
        "dA_sea_minus_raw": d("sea_input", gt) - d("raw_input", gt),
        "dB_sea_minus_lagresid": d("sea_input", gt) - d("lag_resid", gt),
        # did the Phase-0 proxy track the true counterfactual?
        "d_proxy_vs_cf": d("proxy_x0", gt),
        "proxy_vs_cf_pooled": spearman(
            [r["proxy_x0"] for r in all_rows], gt
        ),
        # robustness vs the unfiltered counterfactual error
        "d_sea_vs_cfraw": d("sea_input", gt_raw),
        "d_raw_vs_cfraw": d("raw_input", gt_raw),
    }

    # per-step CSV
    cols = ["prompt_idx", "sigma", "raw_input", "sea_input", "lag_resid",
            "proxy_x0", "gt_cf", "gt_cf_raw"]
    with (run_dir / "per_step.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in all_rows:
            wr.writerow({k: r.get(k) for k in cols})

    metrics = {
        "pooled": pooled,
        "config": {
            "warmup": WARMUP, "window": WINDOW, "flex": FLEX,
            "cheb_m": CHEB_M, "cheb_lam": CHEB_LAM, "cheb_w": CHEB_W,
            "beta": a.beta, "n_prompts": len(prompts), "n_rows": len(all_rows),
        },
    }
    write_result(
        run_dir, script=__file__, args=a, metrics=metrics,
        label=a.label, artifacts=["per_step.csv"], device=device,
    )

    print("\n=== Phase 1: step-detrended ρ vs TRUE counterfactual skip-cost ===")
    print(f"  sea_input   ρ = {pooled['d_sea_vs_cf']:+.3f}   (SeaCache, leading)")
    print(f"  raw_input   ρ = {pooled['d_raw_vs_cf']:+.3f}   (baseline)")
    print(f"  lag_resid   ρ = {pooled['d_lag_resid_vs_cf']:+.3f}   (Spectrum has, lagging)")
    print(
        f"  A: SEA − raw          = {pooled['dA_sea_minus_raw']:+.3f}  "
        f"({'SEA helps' if pooled['dA_sea_minus_raw'] > 0.05 else 'no SEA win'})"
    )
    print(
        f"  B: SEA − lag_resid    = {pooled['dB_sea_minus_lagresid']:+.3f}  "
        f"({'leading SEA beats lagging residual' if pooled['dB_sea_minus_lagresid'] > 0.05 else 'residual competitive'})"
    )
    print(
        f"  proxy(x̂₀) vs true cf  = {pooled['d_proxy_vs_cf']:+.3f} detrended  "
        f"({'Phase-0 proxy was faithful' if pooled['d_proxy_vs_cf'] > 0.3 else 'Phase-0 proxy was MISLEADING'})"
    )
    print(f"\nresult.json + per_step.csv in: {run_dir}")


if __name__ == "__main__":
    main()
