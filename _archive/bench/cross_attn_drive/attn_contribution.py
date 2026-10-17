"""Self-attn vs cross-attn vs MLP *contribution* probe — which pathway drives the
residual update at each σ?

Companion to ``attn_evolution.py``. That probe measures WHERE patches look
(softmax q·k); it is blind to WHAT each pathway actually writes into the residual.
This probe measures the magnitude each block adds to the residual stream:

  residual after block = x
                       + gate_self  · self_attn(x)     ← self_contrib
                       + gate_cross · cross_attn(x, txt) ← cross_contrib
                       + gate_mlp   · mlp(x)            ← mlp_contrib

The gates are adaln-modulated by the timestep, so this captures the σ-scheduled
down-weighting of each pathway — the mechanism behind "self-attn dominates from
σ<0.9". We log per block per step the L2 norm of each *gated* contribution
(exactly the term added to ``x_B_T_H_W_D`` in ``Block._forward``), plus the share
``cross_frac = ‖cross‖ / (‖self‖ + ‖cross‖)``.

Why gated, not the raw attention output: the raw attn-output norm misses the gate,
which is precisely the σ-knob that fades cross-attn late. We mirror ``Block._forward``
(it computes the gates as locals) so the logged term is bit-identical to what the
real forward adds — generation is unchanged.

Conditional pass only (the text-driven forward); CFG uncond is skipped.

Usage:
    uv run python bench/cross_attn_drive/attn_contribution.py \
        --captions 12 --seeds 2 [--tags "@sincos"] \
        [--lora_weight output/ckpt/anima_sincos2.safetensors] --label sincos
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima_lora import GenerationRequest  # noqa: E402
from bench._anima import add_common_args, add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima.models import Block  # noqa: E402
from library.inference.generation import generate_body  # noqa: E402
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.text import (  # noqa: E402
    ensure_text_strategies,
    prepare_text_inputs,
)
from library.runtime.device import clean_memory_on_device  # noqa: E402

# reuse the caption loader + tag selection from the sibling probe
from importlib import util as _ilu  # noqa: E402

_ae_spec = _ilu.spec_from_file_location(
    "_attn_evolution", str(Path(__file__).parent / "attn_evolution.py")
)
_ae = _ilu.module_from_spec(_ae_spec)
_ae_spec.loader.exec_module(_ae)
load_captions = _ae.load_captions
caption_has_tag = _ae.caption_has_tag

# Block modules are named exactly "blocks.<N>" (no trailing dot).
_BLOCK_RE = re.compile(r"blocks\.(\d+)$")


def _per_sample_norm(x: torch.Tensor) -> float:
    return float(x.float().flatten(1).norm(dim=1).mean())


class ContribRecorder:
    def __init__(self, anima):
        self.rows: list[dict] = []
        self._sigma = None
        self._last_sigma = None
        self._step = -1
        self._pass = 0
        self._gen = -1
        idxs = sorted(
            int(_BLOCK_RE.search(n).group(1))
            for n, m in anima.named_modules()
            if isinstance(m, Block) and _BLOCK_RE.search(n)
        )
        self._depth = {b: (i / max(1, len(idxs) - 1)) for i, b in enumerate(idxs)}
        self._n_blocks = len(idxs)

    def begin_generation(self):
        self._gen += 1
        self._sigma = None
        self._last_sigma = None
        self._step = -1
        self._pass = 0

    def on_model_forward(self, timesteps: torch.Tensor):
        sigma = float(timesteps.flatten()[0])
        if self._last_sigma is None or sigma != self._last_sigma:
            self._step += 1
            self._pass = 0
            self._last_sigma = sigma
            self._sigma = sigma
        else:
            self._pass += 1

    def record(self, block, self_c, cross_c, mlp_c):
        if self._pass != 0:
            return
        self.rows.append(
            {
                "gen": self._gen,
                "block": block,
                "depth_frac": round(self._depth.get(block, 0.0), 4),
                "step": self._step,
                "sigma": round(self._sigma, 5),
                "self_norm": round(_per_sample_norm(self_c), 5),
                "cross_norm": round(_per_sample_norm(cross_c), 5),
                "mlp_norm": round(_per_sample_norm(mlp_c), 5),
            }
        )


def _patched_forward(block: Block, rec: ContribRecorder, block_idx: int):
    """Mirror of ``Block._forward`` that logs the three gated contribution norms.
    Bit-identical to the original computation (same submodule calls)."""

    def _forward(
        x_B_T_H_W_D,
        emb_B_T_D,
        crossattn_emb,
        attn_params,
        rope_cos_sin=None,
        adaln_lora_B_T_3D=None,
    ):
        if block.use_adaln_lora:
            fused_down = block.adaln_fused_down(emb_B_T_D)
            down_self, down_cross, down_mlp = fused_down.chunk(3, dim=-1)
            sh_s, sc_s, g_s = (
                block.adaln_up_self_attn(down_self) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            sh_c, sc_c, g_c = (
                block.adaln_up_cross_attn(down_cross) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            sh_m, sc_m, g_m = (block.adaln_up_mlp(down_mlp) + adaln_lora_B_T_3D).chunk(
                3, dim=-1
            )
        else:
            sh_s, sc_s, g_s = block.adaln_modulation_self_attn(emb_B_T_D).chunk(
                3, dim=-1
            )
            sh_c, sc_c, g_c = block.adaln_modulation_cross_attn(emb_B_T_D).chunk(
                3, dim=-1
            )
            sh_m, sc_m, g_m = block.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        def _b(t):
            return t[:, :, None, None, :]

        sh_s, sc_s, g_s = _b(sh_s), _b(sc_s), _b(g_s)
        sh_c, sc_c, g_c = _b(sh_c), _b(sc_c), _b(g_c)
        sh_m, sc_m, g_m = _b(sh_m), _b(sc_m), _b(g_m)

        B, T, H, W, D = x_B_T_H_W_D.shape

        def adaln(_x, _ln, _scale, _shift):
            return _ln(_x) * (1 + _scale) + _shift

        nx = adaln(x_B_T_H_W_D, block.layer_norm_self_attn, sc_s, sh_s)
        xf = nx.flatten(1, 3)
        r_self = block.self_attn(
            xf, attn_params, xf, rope_cos_sin=rope_cos_sin
        ).unflatten(1, (T, H, W))
        self_contrib = g_s * r_self
        x_B_T_H_W_D = x_B_T_H_W_D + self_contrib

        nx = adaln(x_B_T_H_W_D, block.layer_norm_cross_attn, sc_c, sh_c)
        r_cross = block.cross_attn(
            nx.flatten(1, 3), attn_params, crossattn_emb, rope_cos_sin=rope_cos_sin
        ).unflatten(1, (T, H, W))
        cross_contrib = r_cross * g_c
        x_B_T_H_W_D = cross_contrib + x_B_T_H_W_D

        nx = adaln(x_B_T_H_W_D, block.layer_norm_mlp, sc_m, sh_m)
        mlp_contrib = g_m * block.mlp(nx)
        x_B_T_H_W_D = x_B_T_H_W_D + mlp_contrib

        rec.record(block_idx, self_contrib, cross_contrib, mlp_contrib)
        return x_B_T_H_W_D

    return _forward


def install_hooks(anima, rec: ContribRecorder):
    orig_fwd = anima.forward

    def model_fwd(latents, timesteps, *a, **kw):
        rec.on_model_forward(timesteps)
        return orig_fwd(latents, timesteps, *a, **kw)

    anima.forward = model_fwd

    for name, mod in anima.named_modules():
        m = _BLOCK_RE.search(name)
        if isinstance(mod, Block) and m:
            mod._forward = _patched_forward(mod, rec, int(m.group(1)))


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------
_BANDS = [
    ("hi", 1.01, 0.9),
    ("upper_mid", 0.9, 0.7),
    ("mid", 0.7, 0.45),
    ("low", 0.45, -0.01),
]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _r(x):
    return round(x, 5) if x is not None else None


def _band_of(sig):
    for name, hi, lo in _BANDS:
        if lo < sig <= hi:
            return name
    return "low"


def aggregate(rows: list[dict]) -> dict:
    def frac(r):
        s, c = r["self_norm"], r["cross_norm"]
        return c / (s + c) if (s + c) else None

    banded = {}
    for bn, *_ in _BANDS:
        sel = [r for r in rows if _band_of(r["sigma"]) == bn]
        banded[bn] = {
            "self_norm": _r(_mean([r["self_norm"] for r in sel])),
            "cross_norm": _r(_mean([r["cross_norm"] for r in sel])),
            "mlp_norm": _r(_mean([r["mlp_norm"] for r in sel])),
            "cross_frac": _r(_mean([frac(r) for r in sel])),
            "n": len(sel),
        }

    curve = []
    for s in sorted({r["step"] for r in rows}):
        sel = [r for r in rows if r["step"] == s]
        curve.append(
            {
                "step": s,
                "sigma": round(_mean([r["sigma"] for r in sel]), 5),
                "self_norm": _r(_mean([r["self_norm"] for r in sel])),
                "cross_norm": _r(_mean([r["cross_norm"] for r in sel])),
                "mlp_norm": _r(_mean([r["mlp_norm"] for r in sel])),
                "cross_frac": _r(_mean([frac(r) for r in sel])),
            }
        )

    # crossover σ: highest σ at which self_norm first exceeds cross_norm
    crossover = None
    for c in curve:  # curve is σ high→low
        if c["self_norm"] is not None and c["self_norm"] > c["cross_norm"]:
            crossover = c["sigma"]
            break
    return {
        "banded": banded,
        "curve": curve,
        "self_dominant_below_sigma": crossover,
        "note": "gated contribution L2 norm added to the residual; cross_frac = "
        "cross/(self+cross). MLP shown separately (usually the largest term).",
    }


def maybe_plot(agg, out_dir: Path, label: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    c = agg["curve"]
    sig = [x["sigma"] for x in c]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(sig, [x["self_norm"] for x in c], "o-", label="self-attn")
    ax[0].plot(sig, [x["cross_norm"] for x in c], "o-", label="cross-attn")
    ax[0].plot(sig, [x["mlp_norm"] for x in c], "o-", color="0.6", label="mlp")
    ax[0].set_xlabel("σ (high→low)")
    ax[0].set_ylabel("gated contribution L2 norm")
    ax[0].set_title(f"residual contribution by pathway — {label}")
    ax[0].invert_xaxis()
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=9)
    ax[1].plot(sig, [x["cross_frac"] for x in c], "s-", color="C1")
    ax[1].axhline(0.5, color="0.5", lw=0.8)
    ax[1].set_xlabel("σ (high→low)")
    ax[1].set_ylabel("cross / (self + cross)")
    ax[1].set_title("cross-attn share of attention update")
    ax[1].invert_xaxis()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "attn_contribution.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


# ---------------------------------------------------------------------------
def build_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    add_model_args(p)
    p.add_argument("--prompts", default=None)
    p.add_argument(
        "--caption_index", default="post_image_dataset/captions/caption_index.json"
    )
    p.add_argument("--lora_weight", default=None)
    p.add_argument("--lora_multiplier", type=float, default=1.0)
    p.add_argument(
        "--tags", default="", help="If set, select captions containing these tags."
    )
    p.add_argument("--captions", type=int, default=12)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--negative_prompt", default="")
    return p.parse_args()


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    req = GenerationRequest(
        prompt="placeholder",
        dit=args.dit,
        vae=args.vae,
        text_encoder=args.text_encoder,
        negative_prompt=args.negative_prompt,
        image_size=(args.size, args.size),
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg,
        flow_shift=args.flow_shift,
        attn_mode=args.attn_mode,
        device=args.device,
        seed=0,
    )
    gen_args = req.to_args()
    if args.lora_weight:
        gen_args.lora_weight = [args.lora_weight]
        gen_args.lora_multiplier = [args.lora_multiplier]
    gen_args.compile_blocks = False  # patched _forward must stay eager

    ensure_text_strategies(args.text_encoder)
    text_encoder = load_text_encoder(gen_args, dtype=torch.bfloat16, device=device)
    text_encoder.eval()
    shared = {"text_encoder": text_encoder, "conds_cache": {}}
    anima = load_dit_model(gen_args, device, torch.bfloat16)
    shared["model"] = anima

    corpus = load_captions(args)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if tags:
        sel, seen = [], set()
        for tag in tags:
            hits = [c for c in corpus if caption_has_tag(c, tag)]
            stride = max(1, len(hits) // max(1, args.captions))
            for i in range(min(args.captions, len(hits))):
                c = hits[i * stride]
                if c not in seen:
                    seen.add(c)
                    sel.append(c)
        captions = sel
    else:
        n = min(args.captions, len(corpus))
        stride = max(1, len(corpus) // n)
        captions = [corpus[i * stride] for i in range(n)]

    rec = ContribRecorder(anima)
    install_hooks(anima, rec)

    total = len(captions) * args.seeds
    done = 0
    for cap in captions:
        ctx, ctx_null = prepare_text_inputs(gen_args, device, anima, shared, prompt=cap)
        for s in range(args.seeds):
            rec.begin_generation()
            with torch.no_grad():
                generate_body(gen_args, anima, ctx, ctx_null, device, seed=s)
            clean_memory_on_device(device)
            done += 1
            print(f"[{done}/{total}] cap={cap[:48]!r} seed={s}", flush=True)

    agg = aggregate(rec.rows)
    out_dir = make_run_dir("cross_attn_drive", args.label)
    plot = maybe_plot(agg, out_dir, args.label or "base")

    import csv

    with (out_dir / "rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "gen",
                "block",
                "depth_frac",
                "step",
                "sigma",
                "self_norm",
                "cross_norm",
                "mlp_norm",
            ],
        )
        w.writeheader()
        w.writerows(rec.rows)

    write_result(
        out_dir,
        script="bench/cross_attn_drive/attn_contribution.py",
        label=args.label,
        args=args,
        metrics={
            "n_captions": len(captions),
            "n_seeds": args.seeds,
            "n_blocks": rec._n_blocks,
            "n_rows": len(rec.rows),
            "lora_weight": args.lora_weight,
            "tags": tags,
            **agg,
        },
        artifacts=["rows.csv"] + ([plot] if plot else []),
    )
    b = agg["banded"]
    print(f"\nself-dominant below σ = {agg['self_dominant_below_sigma']}")
    for bn in ["hi", "upper_mid", "mid", "low"]:
        e = b[bn]
        print(
            f"  {bn:9s} self={e['self_norm']} cross={e['cross_norm']} "
            f"mlp={e['mlp_norm']} cross_frac={e['cross_frac']} n={e['n']}"
        )
    print(f"\nresults -> {out_dir}")


if __name__ == "__main__":
    main()
