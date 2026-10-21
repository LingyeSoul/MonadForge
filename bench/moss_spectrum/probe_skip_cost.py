#!/usr/bin/env python3
"""Where does a Spectrum forecast cost the most on MOSS-Image? Open-loop skip-cost
of every candidate full-step schedule, scored on the plain 50-step trajectory.

One plain 50-step Euler run per prompt records the image-token features after the
last block at every step (cond and uncond rows, ``denoise``'s ``spectrum`` hook
with every step full). A schedule is then replayed on those features with no DiT:
its full steps feed ``_Forecaster`` the recorded feature, its skipped steps take
the forecast through ``head`` + CFG, and the velocity error ``dv_i = v̂_i − v_i``
is scored against the recorded velocity. Open loop: every forecast fits on the
true trajectory, so a schedule's own drift is not compounded (the closed-loop
check is ``run_bench.py`` with ``full=``).

Per schedule, pooled over prompts, normalised by ``‖x_final‖``:

  * ``local`` = ``Σ_i |Δσ_i|·‖dv_i‖`` — the summed per-step displacement error;
  * ``acc`` = ``‖Σ_i Δσ_i·dv_i‖`` — the final-latent error if every displacement
    carried through unchanged (cancellation included);
  * ``*_low`` = the same on a 4×4 average-pooled latent (the staircase's layout
    view, ``bench/moss_sigma_staircase``);
  * ``band`` = ``local`` split at the staircase knees σ 0.87 / 0.57.

Candidates, at three full-step budgets: the growing window (``window_schedule``
over warmup / ws / fw / tail) and anchored schedules that space the post-warmup
full steps evenly in a coordinate — step index, 1 − σ, log-SNR, or the
staircase's pooled / full ``norm95`` progress.

    make daemon-run ARGS="bench/moss_spectrum/probe_skip_cost.py --label base5"
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from bench.moss_spectrum.run_bench import PROMPTS  # noqa: E402

STAIRCASE = Path(__file__).resolve().parents[1] / (
    "moss_sigma_staircase/results/20261021-2204-base-sft/result.json"
)
POOL = 4
KNEES = (0.87, 0.57)
SHIPPED = dict(ws=2.0, fw=0.5, warm=6, tail=3)


class _Recorder:
    """``denoise``'s spectrum hook with every step full: keeps the features."""

    def begin(self, sigmas):
        self.feats = []

    def is_full_step(self, i):
        return True

    def observe(self, i, features):
        self.feats.append(features.to("cpu", torch.bfloat16))


def fmt(full: list[bool]) -> str:
    """``0-5+7+9+…`` — the ``full=`` knob of ``run_bench.py``."""
    idx = [i for i, f in enumerate(full) if f]
    runs, start = [], idx[0]
    for a, b in itertools.pairwise(idx + [None]):
        if b != a + 1:
            runs.append(f"{start}" if start == a else f"{start}-{a}")
            start = b
    return "+".join(runs)


def window(n, ws, fw, warm, tail):
    from library.moss.spectrum import window_schedule

    return window_schedule(n, ws, fw, warm, n - tail)


def anchored(u: list[float], budget: int, warm: int, tail: int) -> list[bool] | None:
    """Full at ``0…warm−1`` and the last ``tail`` steps; the rest of ``budget`` at
    the steps nearest even spacing of ``u`` (increasing) between the last warmup
    step and the first tail step."""
    n = len(u)
    k = budget - warm - tail
    lo, hi = warm - 1, n - tail
    if k < 0 or k > hi - lo - 1:
        return None
    full = [i < warm or i >= hi for i in range(n)]
    free = list(range(lo + 1, hi))
    for j in range(1, k + 1):
        target = u[lo] + j / (k + 1) * (u[hi] - u[lo])
        best = min(free, key=lambda i: abs(u[i] - target))
        full[best] = True
        free.remove(best)
    return full


def candidates(sigmas: list[float], stair: dict, budgets) -> dict[str, list[bool]]:
    n = len(sigmas)
    out: dict[str, list[bool]] = {}
    for ws, fw, warm, tail in itertools.product(
        (2.0, 3.0), (0.0, 0.25, 0.5, 0.75, 1.0, 1.5), (3, 4, 5, 6), (1, 2, 3)
    ):
        out[f"win ws{ws:g} fw{fw:g} warm{warm} tail{tail}"] = window(
            n, ws, fw, warm, tail
        )
    s = torch.tensor(sigmas, dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
    coords = {
        "step": list(range(n)),
        "sigma": (1 - s).tolist(),
        "logsnr": torch.log((1 - s) / s).tolist(),
    }
    for key in ("norm95_low", "norm95"):
        curve = [stair[f"{sig:.3f}"][key] for sig in sigmas]
        coords[f"stair_{key}"] = [-c for c in curve]
    for (name, u), b, warm, tail in itertools.product(
        coords.items(), budgets, (3, 4, 6), (1, 2, 3)
    ):
        full = anchored(u, b, warm, tail)
        if full is not None:
            out[f"anc {name} B{b} warm{warm} tail{tail}"] = full
    # One row per distinct schedule; the window family keeps its name.
    seen, uniq = {}, {}
    for name, full in out.items():
        key = tuple(full)
        if key not in seen:
            seen[key] = name
            uniq[name] = full
    return uniq


@torch.no_grad()
def replay(transformer, feats, sigmas, full, cfg, hw, x_norm, x_norm_low, spec):
    """Open-loop scores of one schedule on one recorded trajectory."""
    from library.moss.spectrum import _Forecaster

    n = len(sigmas)
    h, w = hw
    sig = sigmas + [0.0]
    pred = None
    acc = acc_low = None
    local = local_low = 0.0
    band = [0.0, 0.0, 0.0]
    per_step = [0.0] * n
    for i in range(n):
        if full[i] or pred is None:
            if pred is None:
                pred = _Forecaster(spec.m, spec.lam, spec.w, n, spec.history)
            pred.update(float(i), feats[i]["f"])
            continue
        branches = pred.predict(float(i)).chunk(2)
        v = [transformer.head(b, feats[i]["hm"], h, w).float() for b in branches]
        dv = v[1] + cfg * (v[0] - v[1]) - feats[i]["v"]
        step = sig[i + 1] - sig[i]
        d = step * dv
        d_low = F.avg_pool2d(d, POOL)
        acc = d if acc is None else acc + d
        acc_low = d_low if acc_low is None else acc_low + d_low
        e = d.norm().item() / x_norm
        local += e
        local_low += d_low.norm().item() / x_norm_low
        band[0 if sigmas[i] > KNEES[0] else 1 if sigmas[i] > KNEES[1] else 2] += e
        per_step[i] = e
    return dict(
        local=local,
        local_low=local_low,
        acc=0.0 if acc is None else acc.norm().item() / x_norm,
        acc_low=0.0 if acc_low is None else acc_low.norm().item() / x_norm_low,
        band=band,
        per_step=per_step,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="base")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--budgets", default="16,19,22")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--image_root", default="/media/sorryhyun/data/moss_image/bench")
    args = ap.parse_args()
    start_heartbeat(label="moss_skip_cost")

    from library.moss.generate import cached_embeds, denoise, initial_noise
    from library.moss.loader import empty_cache, load_transformer, pad_embeds
    from library.moss.model import snapped_sigmas
    from library.moss.requests import DEFAULT_UNCOND_SKIP, resolve_model_dir
    from library.moss.spectrum import MossSpectrum

    keys = [k for k in args.prompts.split(",") if k]
    budgets = [int(b) for b in args.budgets.split(",")]
    model_dir = resolve_model_dir(args.model_dir)
    run_dir = make_run_dir(
        "moss_spectrum", label=f"skipcost-{args.label or args.model}"
    )
    device = torch.device("cuda")
    h = w = args.size
    sigmas_t = snapped_sigmas(50)
    sigmas = sigmas_t.tolist()
    stair = json.loads(STAIRCASE.read_text())["metrics"]["base"]["by_sigma"]
    cands = candidates(sigmas, stair, budgets)
    print(f"{len(cands)} distinct schedules over {len(sigmas)} steps", flush=True)

    embeds = cached_embeds(
        model_dir,
        [PROMPTS[k] for k in keys] + [""],
        Path(args.image_root) / "_te_cache",
    )
    uncond, uncond_mask = pad_embeds([embeds[""]], device=device)
    transformer = load_transformer(model_dir, args.model, device=device)

    runs = []
    for key in keys:
        cond, cond_mask = pad_embeds([embeds[PROMPTS[key]]], device=device)
        rec = _Recorder()
        t0 = time.perf_counter()
        z = denoise(
            transformer,
            initial_noise(args.seed, h, w).to(device),
            cond,
            cond_mask,
            uncond,
            uncond_mask,
            sigmas_t,
            cfg=args.cfg,
            uncond_skip=DEFAULT_UNCOND_SKIP,
            spectrum=rec,
        )
        print(f"recorded {key} in {time.perf_counter() - t0:.1f}s", flush=True)
        runs.append((key, rec.feats, z.cpu()))

    # Replay needs only time_conditioning + head.
    transformer.layers.to("cpu")
    empty_cache()
    spec = MossSpectrum()
    lh, lw = h // 16, w // 16
    scores = {name: [] for name in cands}
    t0 = time.perf_counter()
    for key, cpu_feats, z in runs:
        z = z.to(device)
        x_norm = z.norm().item()
        x_norm_low = F.avg_pool2d(z, POOL).norm().item()
        feats = []
        for i, f in enumerate(cpu_feats):
            f = f.to(device)
            _mod, hm = transformer.time_conditioning(sigmas[i] * 1000, 1)
            v = [transformer.head(b, hm, lh, lw).float() for b in f.chunk(2)]
            feats.append(dict(f=f, hm=hm, v=v[1] + args.cfg * (v[0] - v[1])))
        for name, full in cands.items():
            scores[name].append(
                replay(
                    transformer,
                    feats,
                    sigmas,
                    full,
                    args.cfg,
                    (lh, lw),
                    x_norm,
                    x_norm_low,
                    spec,
                )
            )
        del feats
        empty_cache()
        print(f"replayed {key} ({time.perf_counter() - t0:.0f}s)", flush=True)

    table = []
    for name, full in cands.items():
        rows = scores[name]
        mean = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
        table.append(
            dict(
                name=name,
                nfe=sum(full),
                full=fmt(full),
                local=round(mean("local"), 5),
                local_low=round(mean("local_low"), 5),
                acc=round(mean("acc"), 5),
                acc_low=round(mean("acc_low"), 5),
                band=[
                    round(sum(r["band"][b] for r in rows) / len(rows), 5)
                    for b in range(3)
                ],
                per_step=[
                    round(sum(r["per_step"][i] for r in rows) / len(rows), 6)
                    for i in range(len(sigmas))
                ],
            )
        )
    table.sort(key=lambda r: (r["nfe"], r["local"]))
    shipped = window(
        len(sigmas), SHIPPED["ws"], SHIPPED["fw"], SHIPPED["warm"], SHIPPED["tail"]
    )
    shipped_name = next(r["name"] for r in table if r["full"] == fmt(shipped))
    by_nfe: dict[int, list] = {}
    for r in table:
        by_nfe.setdefault(r["nfe"], []).append(r)
    for nfe in sorted(set(budgets) | {sum(shipped)}):
        rows = by_nfe.get(nfe, [])
        print(f"\n== NFE {nfe}: {len(rows)} schedules (sorted by local)", flush=True)
        for r in rows[:8] + [r for r in rows if r["name"] == shipped_name]:
            print(
                f"  {r['name']:<40} local {r['local']:.4f} low {r['local_low']:.4f} "
                f"acc {r['acc']:.4f} acc_low {r['acc_low']:.4f} band {r['band']}  {r['full']}",
                flush=True,
            )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=dict(
            sigmas=[round(s, 4) for s in sigmas],
            shipped=shipped_name,
            schedules=table,
        ),
    )
    print(f"result -> {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
