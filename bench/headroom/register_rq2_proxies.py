"""RQ2 proxies on the trained register adapters — is the untuned visual change
register-mediated, or generic LoRA drift?

Follow-up to the Phase-0.5 RQ3-negative result (`bench/headroom/README.md`): the
relocation crossover failed at the LoRA budget, but arm B re-plans matched-seed
images heavily. Two metric-free-wall-dodging proxies adjudicate whether anything
register-specific survived training, before any hyperparameter tuning:

* **Proxy (a) — sink-patch detail recovery** (DSR's core prediction). The DSR
  claim is that the high-norm outlier is a *symptom of corrupted local patch
  semantics*: the sink's own patches lose local detail. Per generation we locate
  the sink tokens (mid-block top-0.2% ‖x‖, accumulated over the trajectory), map
  them onto the pixel grid, and compare per-patch Laplacian variance on sink
  patches vs the rest (``sink_detail_ratio``, <1 = sink patches locally
  degraded). If arm B's weak sink reduction (−10% patch_sink_ratio) is real
  relocation, its sink patches should recover detail: ratio closer to 1 than
  base. Caveat: sinks preferentially sit on low-information patches (the ViT
  story), so <1 is partly selection, not damage — the *cross-condition delta*
  at matched seed is the signal, not the absolute value.

* **Proxy (b) — load-bearing sensitivity drop**. `sink_intervention.py` showed
  clamping the sink re-plans the whole image (median pix L1 ~16% on the complex
  prompt). If arm B moved any load off-canvas, the SAME clamp on the trained
  model should perturb its output *less* per unit clamp. Arm A (fixed-zero
  registers + same QKV-LoRA) is the drift control: if arm A's sensitivity drops
  just as much, the change is generic LoRA drift, not registers.

Conditions run in order **base → armA → armB** on matched (prompt, seed); base's
sink locations are also scored in the adapter images ("ex-sink" detail — did the
places base sacrificed recover under the adapter?).

Run::

    uv run python bench/headroom/register_rq2_proxies.py \
        --adapter_a results/20260701-2046-rq3_armA/adapter.pt \
        --adapter_b results/20260701-2052-rq3_armB/adapter.pt \
        --seeds 3 --label rq2_proxies
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sink_intervention as si  # noqa: E402
import sink_probe as sp  # noqa: E402
from anima_lora import GenerationRequest, load_vae  # noqa: E402
from bench._anima import add_common_args, add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.inference.generation import generate_body  # noqa: E402
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.output import decode_latent, pixels_to_pil  # noqa: E402
from library.inference.text import ensure_text_strategies, prepare_text_inputs  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402
from register_eval import load_adapter  # noqa: E402


class SinkTracker:
    """Mid-block forward hook: accumulates, over a generation's trajectory, how
    often each PATCH token lands in the top-0.2% ‖x‖ set (the sink signature),
    plus the outlier ratio for cross-validation against register_eval."""

    def __init__(self, K: int):
        self.K = K
        self.reset()

    def reset(self):
        self.counts: torch.Tensor | None = None
        self.ratios: list[float] = []
        self.seq: int | None = None

    def __call__(self, module, inputs, output):
        # base: (B,1,H,W,D) grid; adapter: (B,1,seq+K,1,D) native-flatten.
        # norm over D then mean over batch → one scalar per token either way.
        n = output.detach().float().norm(dim=-1).mean(0).flatten()
        seq = n.numel() - self.K
        patch = n[:seq]
        med = patch.median().clamp_min(1e-6)
        k = max(1, int(0.002 * seq))
        top = patch.topk(k)
        self.ratios.append((top.values.mean() / med).item())
        if self.counts is None:
            self.counts = torch.zeros(seq)
            self.seq = seq
        self.counts[top.indices.cpu()] += 1

    def sink_set(self) -> list[int]:
        k = max(1, int(0.002 * self.seq))
        return self.counts.topk(k).indices.tolist()

    def mean_ratio(self) -> float:
        return float(np.mean(self.ratios)) if self.ratios else float("nan")


def _patch_lapvar(img_chw: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
    """Per-patch Laplacian variance (local-detail proxy) on luma, (hp, wp)."""
    luma = img_chw.float().mean(0)[None, None]
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    lap = F.conv2d(luma, k, padding=1)[0, 0]
    h, w = lap.shape
    bh, bw = h // hp, w // wp
    lap = lap[: bh * hp, : bw * wp].reshape(hp, bh, wp, bw).permute(0, 2, 1, 3)
    return lap.reshape(hp, wp, -1).var(dim=-1)


def _detail_ratio(lapvar_grid: torch.Tensor, sink_idx: list[int]) -> float:
    """mean lap-var over sink patches / mean over the rest (<1 = sink degraded)."""
    hp, wp = lapvar_grid.shape
    flat = lapvar_grid.flatten()
    mask = torch.zeros(hp * wp, dtype=torch.bool)
    mask[torch.tensor(sink_idx)] = True
    return float(flat[mask].mean() / flat[~mask].mean().clamp_min(1e-9))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    add_model_args(p)
    here = Path(__file__).resolve().parent
    p.add_argument("--adapter_a", default=str(here / "results/20260701-2046-rq3_armA/adapter.pt"))
    p.add_argument("--adapter_b", default=str(here / "results/20260701-2052-rq3_armB/adapter.pt"))
    p.add_argument("--adapter_l", default=None,
                   help="optional LoRA-only (K=0) adapter.pt — the register-free drift control")
    p.add_argument("--only", default=None,
                   help="comma list of conditions to run (e.g. 'armL'); generation is "
                        "seed-deterministic so skipped conditions can be compared from a "
                        "previous run's images. Without base, ex-sink detail falls back "
                        "to the condition's own sink locations.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--mid_block", type=int, default=-1, help="probe block; -1 = len//2")
    p.add_argument("--clamp_ratio", type=float, default=4.0)
    p.add_argument("--mid_lo", type=float, default=1 / 3)
    p.add_argument("--mid_hi", type=float, default=2 / 3)
    args = p.parse_args()

    device = torch.device(args.device)
    req = GenerationRequest(
        prompt="x", dit=args.dit, vae=args.vae, text_encoder=args.text_encoder,
        negative_prompt="", image_size=(args.size, args.size), infer_steps=args.infer_steps,
        guidance_scale=args.cfg, flow_shift=args.flow_shift, attn_mode=args.attn_mode,
        device=args.device, seed=0,
    )
    gen_args = req.to_args()
    gen_args.compile_blocks = False

    ensure_text_strategies(args.text_encoder)
    text_encoder = load_text_encoder(gen_args, dtype=torch.bfloat16, device=device)
    text_encoder.eval()
    shared = {"text_encoder": text_encoder, "conds_cache": {}}
    anima = load_dit_model(gen_args, device, torch.bfloat16)
    anima.requires_grad_(False)
    shared["model"] = anima
    vae = load_vae(args.vae, device="cpu", vae_2d=True)
    vae.to(torch.bfloat16).eval()

    clamp = si.Clamp(anima, args.clamp_ratio, args.mid_lo, args.mid_hi)
    si.install(anima, clamp)
    clamp.enabled = False

    mid = args.mid_block if args.mid_block >= 0 else len(anima.blocks) // 2

    out_dir = make_run_dir("headroom", args.label or "rq2_proxies")
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    conditions = [("base", None), ("armA", args.adapter_a), ("armB", args.adapter_b)]
    if args.adapter_l:
        conditions.append(("armL", args.adapter_l))
    if args.only:
        keep = {c.strip() for c in args.only.split(",")}
        conditions = [c for c in conditions if c[0] in keep]
    prompts = [("repro", sp._REPRO), ("control", sp._CONTROL)]
    base_sinks: dict[tuple[str, int], list[int]] = {}
    rows = []

    for cond, apath in conditions:
        adapter = load_adapter(anima, apath, device) if apath else None
        if adapter:
            adapter.apply_to()
        tracker = SinkTracker(adapter.K if adapter else 0)
        handle = anima.blocks[mid].register_forward_hook(tracker)
        try:
            for pname, ptext in prompts:
                ctx, ctx_null = prepare_text_inputs(gen_args, device, anima, shared, prompt=ptext)
                for s in range(args.seeds):
                    imgs = {}
                    for mode in ("noclamp", "clamp"):
                        clamp.enabled = mode == "clamp"
                        if clamp.enabled:
                            clamp.begin()
                        else:
                            tracker.reset()
                        with torch.no_grad():
                            latent = generate_body(gen_args, anima, ctx, ctx_null, device, seed=s)
                        img = decode_latent(vae, latent, device)
                        clean_memory_on_device(device)
                        pixels_to_pil(img).save(img_dir / f"{cond}_{pname}_s{s}_{mode}.png")
                        imgs[mode] = img
                    clamp.enabled = False

                    seq = tracker.seq
                    side = int(math.isqrt(seq))
                    assert side * side == seq, f"non-square patch grid, seq={seq}"
                    sink = tracker.sink_set()
                    if cond == "base":
                        base_sinks[(pname, s)] = sink

                    lapvar = _patch_lapvar(imgs["noclamp"], side, side)
                    own_detail = _detail_ratio(lapvar, sink)
                    exsink_detail = _detail_ratio(lapvar, base_sinks.get((pname, s), sink))

                    diff = ((imgs["noclamp"] - imgs["clamp"]) / 2).abs().float()
                    pix_l1 = float(diff.mean())
                    lb, lc = si._lap_var(imgs["noclamp"]), si._lap_var(imgs["clamp"])
                    rows.append({
                        "cond": cond, "prompt": pname, "seed": s,
                        "patch_sink_ratio": round(tracker.mean_ratio(), 3),
                        "sink_detail_ratio": round(own_detail, 4),
                        "exsink_detail_ratio": round(exsink_detail, 4),
                        "clamp_pix_l1": round(pix_l1, 5),
                        "clamp_sharp_ratio": round(lc / lb, 4) if lb else None,
                        "clamp_frac": round(clamp.frac(), 5) if clamp.frac() is not None else None,
                    })
                    print(f"{cond} {pname} s{s}: sink_ratio={rows[-1]['patch_sink_ratio']} "
                          f"detail(own/ex)={own_detail:.3f}/{exsink_detail:.3f} "
                          f"clamp_l1={pix_l1:.4f}", flush=True)
        finally:
            handle.remove()
            if adapter:
                adapter.remove()

    def agg(cond, key):
        vals = [r[key] for r in rows if r["cond"] == cond and r[key] is not None]
        return round(float(np.median(vals)), 4) if vals else float("nan")

    summary = {
        cond: {k: agg(cond, k) for k in (
            "patch_sink_ratio", "sink_detail_ratio", "exsink_detail_ratio",
            "clamp_pix_l1", "clamp_sharp_ratio", "clamp_frac")}
        for cond, _ in conditions
    }
    write_result(out_dir, script=__file__, args=args,
                 metrics={"summary": summary, "rows": rows, "mid_block": mid},
                 label=args.label)

    print("\n=== RQ2 PROXIES (median over prompts×seeds, mid block %d) ===" % mid)
    hdr = f"{'cond':6} {'sink_ratio':>10} {'detail_own':>10} {'detail_ex':>10} {'clamp_l1':>9} {'sharp':>6}"
    print(hdr)
    for cond, _ in conditions:
        s = summary[cond]
        print(f"{cond:6} {s['patch_sink_ratio']:>10} {s['sink_detail_ratio']:>10} "
              f"{s['exsink_detail_ratio']:>10} {s['clamp_pix_l1']:>9} {s['clamp_sharp_ratio']:>6}")
    print("\nproxy (a): register-real ⇒ armB detail ratios rise toward 1 vs base, armA flat")
    print("proxy (b): register-real ⇒ armB clamp_l1 falls vs base MORE than armA does")
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
