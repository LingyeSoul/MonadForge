"""Does the DSR self-attn sink hurt image quality? — the masking test, on Anima.

Follow-up to `sink_probe.py` / `border_toggle.py`, which established that the base DiT
carries a strong, ALWAYS-PRESENT mid-layer self-attn sink (~26× high-norm outlier
token) that is *decoupled from the border*. This probe asks DSR's actual question:
is that sink load-bearing, inert, or a pure quality-costing artifact?

Because Anima has no per-image quality reward (CMMD is distribution-level; FM-val
doesn't track quality) we can't correlate "sink strength → quality." And because the
sink is in *every* generation, we can't compare with-vs-without images. So we do the
causal thing DSR did (their Tab. 1): **tame the outlier at inference and watch the
image.** For matched (prompt, seed) we generate twice — baseline, and with a hook that
clamps the norm of the top mid-layer ‖x‖ tokens down toward the median (removing the
outlier magnitude while preserving direction/content). Then:

  - degrades  ⇒ sink is load-bearing (DSR: masking hurts; registers must ABSORB it,
                and whether that nets a quality win needs Phase-1 training + a metric
                we don't have)
  - unchanged ⇒ inert
  - improves  ⇒ pure artifact ⇒ registers genuinely promising for general quality

No quality reward is used or implied. The deliverable is a matched baseline-vs-clamped
image set for eyeballing, plus change magnitude (pixel L1), a detail proxy (Laplacian
variance ratio), and how concentrated the change is on the sink tokens' own patches.

Usage::

    uv run python bench/headroom/sink_intervention.py --seeds 3 --clamp_ratio 4 --label clamp4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sink_probe as sp  # noqa: E402
from anima_lora import GenerationRequest, load_vae  # noqa: E402
from bench._anima import add_common_args, add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima.models import Block  # noqa: E402
from library.inference.generation import generate_body  # noqa: E402
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.output import decode_latent, pixels_to_pil  # noqa: E402
from library.inference.text import ensure_text_strategies, prepare_text_inputs  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

_BLOCK_RE = sp._BLOCK_RE


class Clamp:
    """Shared mutable state for the intervention hook: when enabled, mid-depth
    blocks clamp any token whose ‖x‖ exceeds ``ratio × median`` back down to that
    cap (norm-clamp, direction preserved). Records which patches were clamped so we
    can check the pixel change concentrates there."""

    def __init__(self, anima, ratio: float, lo: float, hi: float):
        self.enabled = False
        self.ratio = ratio
        self.lo = lo
        self.hi = hi
        idxs = sorted(
            int(_BLOCK_RE.search(n).group(1))
            for n, m in anima.named_modules()
            if isinstance(m, Block) and _BLOCK_RE.search(n)
        )
        self.depth = {b: (i / max(1, len(idxs) - 1)) for i, b in enumerate(idxs)}
        self.union = None  # (H,W) bool: any token clamped anywhere in the run
        self.tok_clamped = 0
        self.tok_total = 0

    def begin(self):
        self.union = None
        self.tok_clamped = 0
        self.tok_total = 0

    def frac(self):
        return (self.tok_clamped / self.tok_total) if self.tok_total else None

    def in_range(self, block: int) -> bool:
        return self.lo <= self.depth.get(block, 0.0) < self.hi


def _patch_forward(block: Block, clamp: Clamp, bidx: int):
    """Mirror of Block._forward that norm-clamps outlier tokens at block output when
    the intervention is enabled and this block is mid-depth. Bit-identical otherwise."""

    def _forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, attn_params, rope_cos_sin=None, adaln_lora_B_T_3D=None):
        if block.use_adaln_lora:
            fused_down = block.adaln_fused_down(emb_B_T_D)
            down_self, down_cross, down_mlp = fused_down.chunk(3, dim=-1)
            sh_s, sc_s, g_s = (block.adaln_up_self_attn(down_self) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            sh_c, sc_c, g_c = (block.adaln_up_cross_attn(down_cross) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            sh_m, sc_m, g_m = (block.adaln_up_mlp(down_mlp) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        else:
            sh_s, sc_s, g_s = block.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            sh_c, sc_c, g_c = block.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            sh_m, sc_m, g_m = block.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        def _b(t):
            return t[:, :, None, None, :]

        sh_s, sc_s, g_s = _b(sh_s), _b(sc_s), _b(g_s)
        sh_c, sc_c, g_c = _b(sh_c), _b(sc_c), _b(g_c)
        sh_m, sc_m, g_m = _b(sh_m), _b(sc_m), _b(g_m)
        _, T, H, W, _ = x_B_T_H_W_D.shape

        def adaln(_x, _ln, _scale, _shift):
            return _ln(_x) * (1 + _scale) + _shift

        nx = adaln(x_B_T_H_W_D, block.layer_norm_self_attn, sc_s, sh_s)
        xf = nx.flatten(1, 3)
        x_B_T_H_W_D = x_B_T_H_W_D + g_s * block.self_attn(xf, attn_params, xf, rope_cos_sin=rope_cos_sin).unflatten(1, (T, H, W))
        nx = adaln(x_B_T_H_W_D, block.layer_norm_cross_attn, sc_c, sh_c)
        x_B_T_H_W_D = g_c * block.cross_attn(nx.flatten(1, 3), attn_params, crossattn_emb, rope_cos_sin=rope_cos_sin).unflatten(1, (T, H, W)) + x_B_T_H_W_D
        nx = adaln(x_B_T_H_W_D, block.layer_norm_mlp, sc_m, sh_m)
        x_B_T_H_W_D = x_B_T_H_W_D + g_m * block.mlp(nx)

        if clamp.enabled and clamp.in_range(bidx):
            n = x_B_T_H_W_D.float().norm(dim=-1, keepdim=True)  # (B,T,H,W,1)
            med = n.flatten(1).median(dim=1).values.view(-1, 1, 1, 1, 1)
            cap = med * clamp.ratio
            scale = torch.clamp(cap / n.clamp_min(1e-6), max=1.0)
            mask = (scale[0, 0, :, :, 0] < 1.0).detach().cpu().numpy()
            clamp.union = mask if clamp.union is None else (clamp.union | mask)
            clamp.tok_clamped += int(mask.sum())
            clamp.tok_total += int(mask.size)
            x_B_T_H_W_D = x_B_T_H_W_D * scale.to(x_B_T_H_W_D.dtype)
        return x_B_T_H_W_D

    return _forward


def install(anima, clamp: Clamp):
    for name, mod in anima.named_modules():
        m = _BLOCK_RE.search(name)
        if isinstance(mod, Block) and m:
            mod._forward = _patch_forward(mod, clamp, int(m.group(1)))


def _lap_var(img_chw: torch.Tensor) -> float:
    """Laplacian variance (detail/sharpness proxy) on luma in [-1,1]."""
    luma = img_chw.float().mean(0)[None, None]  # (1,1,H,W)
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(luma, k)
    return float(lap.var())


def build_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    add_common_args(p)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--clamp_ratio", type=float, default=4.0, help="Clamp tokens with ‖x‖ > ratio×median down to the cap.")
    p.add_argument("--mid_lo", type=float, default=1 / 3, help="Mid-depth band low (fraction).")
    p.add_argument("--mid_hi", type=float, default=2 / 3, help="Mid-depth band high (fraction).")
    return p.parse_args()


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    req = GenerationRequest(
        prompt="placeholder", dit=args.dit, vae=args.vae, text_encoder=args.text_encoder,
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
    shared["model"] = anima
    vae = load_vae(args.vae, device="cpu", vae_2d=True)
    vae.to(torch.bfloat16).eval()

    clamp = Clamp(anima, args.clamp_ratio, args.mid_lo, args.mid_hi)
    install(anima, clamp)

    out_dir = make_run_dir("headroom", args.label)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    prompts = [("repro", sp._REPRO), ("control", sp._CONTROL)]
    rows = []
    for name, text in prompts:
        ctx, ctx_null = prepare_text_inputs(gen_args, device, anima, shared, prompt=text)
        for s in range(args.seeds):
            imgs = {}
            for mode in ("base", "clamp"):
                clamp.enabled = mode == "clamp"
                if clamp.enabled:
                    clamp.begin()
                with torch.no_grad():
                    latent = generate_body(gen_args, anima, ctx, ctx_null, device, seed=s)
                img = decode_latent(vae, latent, device)
                clean_memory_on_device(device)
                pixels_to_pil(img).save(img_dir / f"{name}_s{s}_{mode}.png")
                imgs[mode] = img
            b01 = (imgs["base"] + 1) / 2
            c01 = (imgs["clamp"] + 1) / 2
            diff = (b01 - c01).abs().float()
            pix_l1 = float(diff.mean())
            lb, lc = _lap_var(imgs["base"]), _lap_var(imgs["clamp"])
            # is the change concentrated on the clamped sink patches (union over run)?
            conc = None
            if clamp.union is not None:
                hp, wp = clamp.union.shape
                dmap = diff.mean(0)  # (H,W) pixel
                Hh, Ww = dmap.shape
                bh, bw = Hh // hp, Ww // wp
                pd = dmap[: bh * hp, : bw * wp].reshape(hp, bh, wp, bw).mean(dim=(1, 3)).numpy()
                m = clamp.union
                if m.any() and (~m).any():
                    conc = round(float(pd[m].mean() / (pd[~m].mean() + 1e-9)), 3)
            rows.append({
                "prompt": name, "seed": s,
                "pix_l1": round(pix_l1, 5),
                "lap_var_base": round(lb, 2), "lap_var_clamp": round(lc, 2),
                "sharp_ratio": round(lc / lb, 4) if lb else None,
                "sink_frac_clamped": round(clamp.frac(), 5) if clamp.frac() is not None else None,
                "change_conc_on_sink": conc,
            })
            print(f"{name} s{s}: pix_l1={pix_l1:.4f} sharp {lb:.0f}->{lc:.0f} conc_on_sink={conc}", flush=True)

    med_l1 = float(np.median([r["pix_l1"] for r in rows]))
    med_sharp = float(np.median([r["sharp_ratio"] for r in rows if r["sharp_ratio"]]))
    med_conc = float(np.median([r["change_conc_on_sink"] for r in rows if r["change_conc_on_sink"]]))
    with (out_dir / "rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_result(
        out_dir, script="bench/headroom/sink_intervention.py", label=args.label, args=args,
        metrics={
            "median_pix_l1": round(med_l1, 5),
            "median_sharp_ratio": round(med_sharp, 4),
            "median_change_conc_on_sink": round(med_conc, 3),
            "clamp_ratio": args.clamp_ratio,
            "note": "read the images; no per-image quality reward exists for Anima",
        },
        artifacts=["rows.csv"],
    )
    print("\n" + "=" * 70)
    print(f"median pixel L1 (base vs clamp): {med_l1:.4f}  (0 = clamp did nothing)")
    print(f"median sharpness ratio (clamp/base): {med_sharp:.3f}  (<1 = clamp blurs, >1 = sharpens)")
    print(f"median change concentration on sink patches: {med_conc:.2f}× (>1 = change lands on the sink tokens)")
    print("VERDICT is by eye — compare <run>/images/*_base.png vs *_clamp.png. No quality metric on Anima.")
    print(f"\nresults → {out_dir}")


if __name__ == "__main__":
    main()
