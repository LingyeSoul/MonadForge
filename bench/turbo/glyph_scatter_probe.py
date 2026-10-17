"""Glyph-scatter probe — does the turbo student spatially scatter glyph-token
conditioning that the teacher confines?

Motivation
----------
Observed failure (turbo_S, 4-step cfg1): a caption carrying a quoted glyph string
('… speech bubble that reads "But it was fast"') renders letter-like texture
scattered across the whole image ("sand"), while the 28-step teacher confines the
(garbled) text to the bubble. The softrank A/B came back inert — consistent with
the deficit being a per-token *spatial routing* failure the whole-caption rank
objective cannot see. This probe gates the follow-on lever (cross-attn map
distillation to the base teacher): it must show the student's glyph-column maps
are measurably more scattered than the teacher's. If the maps match and only the
render differs, the leak is in the value path / self-attn elaboration and map
distillation is dead.

Two measurements per arm (one invocation = one arm at its operating point):

1. **Map-level (WHERE)** — per cross-attn block, conditional pass, the head-avg
   softmax map's *per-column* spatial statistics for the glyph tokens and the
   region tag: spatial concentration = 1 − H(M_c/ΣM_c)/log(Nq) (CTCal's ruler, 1
   = a delta, 0 = uniform), per-token mass, and σ-banded mean spatial heatmaps.
2. **Image-level (the failure)** — same-seed latent diff between the full caption
   and the clause-dropped baseline: channel-L2 diff map, its normalized spatial
   entropy (spread, 1 = uniform sand) and area50 (fraction of pixels holding 50%
   of diff mass; small = confined). Diff heatmaps + decoded renders saved.

Usage
-----
    # teacher arm (base, its operating point)
    uv run python bench/turbo/glyph_scatter_probe.py \
        --infer_steps 28 --cfg 4.0 --label glyph_teacher

    # student arm (turbo, its operating point)
    uv run python bench/turbo/glyph_scatter_probe.py \
        --lora_weight output/ckpt/anima_turbo_S.safetensors \
        --infer_steps 4 --cfg 1.0 --label glyph_turboS

Same seeds ⇒ identical init noise across arms, so renders/diffs pair up 1:1.
Compile is forced OFF (eager map hook must run every step).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima_lora import GenerationRequest  # noqa: E402
from bench._anima import add_common_args, add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima.models import Attention  # noqa: E402
from library.anima.text_strategies import TokenizeStrategy  # noqa: E402
from library.inference.generation import generate_body  # noqa: E402
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.output import decode_latent, pixels_to_pil  # noqa: E402
from library.inference.text import (  # noqa: E402
    ensure_text_strategies,
    prepare_text_inputs,
)
from library.models.qwen_vae import load_vae  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

from attn_evolution import _BLOCK_RE, tag_token_cols  # noqa: E402

_BANDS = [
    ("hi", 1.01, 0.9),
    ("upper_mid", 0.9, 0.7),
    ("mid", 0.7, 0.45),
    ("low", 0.45, -0.01),
]


def _band_of(sig: float) -> str:
    for name, hi, lo in _BANDS:
        if lo < sig <= hi:
            return name
    return "low"


def glyph_token_cols(tok, full_ids, glyph: str) -> list[int]:
    """The quoted-string variant of tag_token_cols: the glyph sits after ``reads "``
    so its first subtoken may fuse with the opening quote — try quoted variants too.
    Returns the columns of the *glyph words only* when a bare match exists, else the
    quoted span."""
    cols = tag_token_cols(tok, full_ids, glyph)
    if cols:
        return cols
    f = full_ids.tolist()
    for variant in (' "' + glyph + '"', '"' + glyph + '"', ' "' + glyph, '"' + glyph):
        vid = tok(variant, add_special_tokens=False).input_ids
        n = len(vid)
        if n == 0:
            continue
        for i in range(len(f) - n + 1):
            if f[i : i + n] == vid:
                return list(range(i, i + n))
    return []


# ---------------------------------------------------------------------------
# Recorder: per-column spatial stats + banded spatial heatmap accumulation.
# ---------------------------------------------------------------------------
class GlyphMapRecorder:
    def __init__(self, anima, grid: int):
        self.enabled = False
        self.grid = grid  # patch grid edge (Nq == grid*grid)
        self.rows: list[dict] = []
        # (tag_kind, band) -> [Nq] running sum of column-normalized spatial maps
        self.heat: dict[tuple[str, str], torch.Tensor] = {}
        self.heat_n: dict[tuple[str, str], int] = {}
        self._sigma = None
        self._last_sigma = None
        self._step = -1
        self._pass = 0
        self._gen = -1
        self._tag_cols: dict[str, list[int]] = {}
        self.n_blocks = sum(
            1
            for n, m in anima.named_modules()
            if isinstance(m, Attention) and not m.is_selfattn and _BLOCK_RE.search(n)
        )

    def begin_generation(self, tag_cols: dict[str, list[int]]):
        self._gen += 1
        self._sigma = None
        self._last_sigma = None
        self._step = -1
        self._pass = 0
        self._tag_cols = tag_cols

    def on_model_forward(self, timesteps: torch.Tensor):
        sigma = float(timesteps.flatten()[0])
        if self._last_sigma is None or sigma != self._last_sigma:
            self._step += 1
            self._pass = 0
            self._last_sigma = sigma
            self._sigma = sigma
        else:
            self._pass += 1

    @torch.no_grad()
    def on_cross_attn(self, block: int, q: torch.Tensor, k: torch.Tensor):
        if self._pass != 0:
            return
        B, Nq, H, D = q.shape
        scale = D**-0.5
        qf, kf = q.float(), k.float()
        amap = q.new_zeros((B, Nq, k.shape[1]), dtype=torch.float32)
        for h in range(H):
            scores = torch.einsum("bqd,bkd->bqk", qf[:, :, h], kf[:, :, h]) * scale
            amap += torch.softmax(scores, dim=-1)
        amap /= H
        cur = amap.mean(0)  # [Nq, Nk]

        band = _band_of(self._sigma)
        for tag_kind, cols in self._tag_cols.items():
            if not cols:
                continue
            sub = cur[:, cols]  # [Nq, n_cols]
            # per-column spatial concentration over patches
            p = sub / sub.sum(0, keepdim=True).clamp_min(1e-12)  # [Nq, n_cols]
            ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(0)  # [n_cols]
            conc = 1.0 - ent / math.log(sub.shape[0])
            self.rows.append(
                {
                    "gen": self._gen,
                    "block": block,
                    "step": self._step,
                    "sigma": round(self._sigma, 5),
                    "tag": tag_kind,
                    "n_cols": sub.shape[1],
                    "conc_mean": round(float(conc.mean()), 6),
                    "conc_max": round(float(conc.max()), 6),
                    "mass_per_tok": round(float(sub.sum(1).mean()) / sub.shape[1], 8),
                }
            )
            key = (tag_kind, band)
            hm = p.mean(1)  # [Nq] column-avg normalized spatial map
            if key in self.heat:
                self.heat[key] += hm.cpu()
            else:
                self.heat[key] = hm.cpu().clone()
            self.heat_n[key] = self.heat_n.get(key, 0) + 1


def install_hooks(anima, rec: GlyphMapRecorder):
    orig_fwd = anima.forward

    def model_fwd(latents, timesteps, *a, **kw):
        if rec.enabled:
            rec.on_model_forward(timesteps)
        return orig_fwd(latents, timesteps, *a, **kw)

    anima.forward = model_fwd

    for name, mod in anima.named_modules():
        m = _BLOCK_RE.search(name)
        if not (isinstance(mod, Attention) and not mod.is_selfattn and m):
            continue
        block = int(m.group(1))
        orig_attn = mod.forward

        def make(mod_ref, block_ref, orig):
            def attn_fwd(x, attn_params, context, rope_cos_sin=None):
                if rec.enabled and rec._pass == 0:
                    q, k, _ = mod_ref.compute_qkv(x, context, rope_cos_sin=None)
                    rec.on_cross_attn(block_ref, q, k)
                return orig(x, attn_params, context, rope_cos_sin=rope_cos_sin)

            return attn_fwd

        mod.forward = make(mod, block, orig_attn)


# ---------------------------------------------------------------------------
# Image-level diff metrics.
# ---------------------------------------------------------------------------
def diff_spread(lat_full: torch.Tensor, lat_base: torch.Tensor) -> dict:
    """Channel-L2 latent diff map -> spread metrics.

    spread   = normalized spatial entropy of the diff distribution (1 = uniform
               "sand", 0 = a delta).
    area50   = fraction of positions holding the top 50% of diff mass (small =
               confined).
    """
    a, b = lat_full.float(), lat_base.float()
    if a.ndim == 5:
        a = a.squeeze(2)
    if b.ndim == 5:
        b = b.squeeze(2)
    d = (a - b).pow(2).sum(1).sqrt()[0]  # [H, W]
    flat = d.flatten()
    p = flat / flat.sum().clamp_min(1e-12)
    ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum()
    spread = float(ent / math.log(p.numel()))
    sorted_p = p.sort(descending=True).values
    cum = sorted_p.cumsum(0)
    area50 = float((cum < 0.5).sum() + 1) / p.numel()
    return {"spread": round(spread, 5), "area50": round(area50, 5), "map": d}


# ---------------------------------------------------------------------------
# Aggregation & plotting.
# ---------------------------------------------------------------------------
def aggregate(rows: list[dict]) -> dict:
    out: dict = {}
    tags = sorted({r["tag"] for r in rows})
    for tag in tags:
        banded = {}
        for bname, *_ in _BANDS:
            sel = [r for r in rows if r["tag"] == tag and _band_of(r["sigma"]) == bname]
            if not sel:
                continue
            banded[bname] = {
                "conc_mean": round(sum(r["conc_mean"] for r in sel) / len(sel), 6),
                "conc_max": round(sum(r["conc_max"] for r in sel) / len(sel), 6),
                "mass_per_tok": round(
                    sum(r["mass_per_tok"] for r in sel) / len(sel), 8
                ),
                "n": len(sel),
            }
        sigmas = sorted({r["sigma"] for r in rows if r["tag"] == tag}, reverse=True)
        curve = []
        for s in sigmas:
            sel = [r for r in rows if r["tag"] == tag and r["sigma"] == s]
            curve.append(
                {
                    "sigma": s,
                    "conc_mean": round(sum(r["conc_mean"] for r in sel) / len(sel), 6),
                    "mass_per_tok": round(
                        sum(r["mass_per_tok"] for r in sel) / len(sel), 8
                    ),
                }
            )
        out[tag] = {"banded": banded, "curve": curve}
    return out


def save_heatmaps(rec: GlyphMapRecorder, out_dir: Path, label: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    tags = sorted({t for t, _ in rec.heat})
    bands = [b for b, *_ in _BANDS]
    fig, axes = plt.subplots(
        len(tags), len(bands), figsize=(3.2 * len(bands), 3.2 * len(tags)), squeeze=False
    )
    for i, tag in enumerate(tags):
        for j, band in enumerate(bands):
            ax = axes[i][j]
            key = (tag, band)
            if key in rec.heat:
                hm = (rec.heat[key] / rec.heat_n[key]).reshape(rec.grid, rec.grid)
                ax.imshow(hm.numpy(), cmap="magma")
            ax.set_title(f"{tag} @ {band}", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"mean glyph/region cross-attn spatial maps — {label}")
    fig.tight_layout()
    p = out_dir / "map_heatmaps.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


def save_diff_heatmap(dmap: torch.Tensor, path: Path, title: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(dmap.float().cpu().numpy(), cmap="magma")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def build_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    add_model_args(p)
    p.add_argument(
        "--pairs",
        default=str(Path(__file__).parent / "glyph_prompts.json"),
        help="JSON list of {full, base, glyph, region_tag} caption pairs.",
    )
    p.add_argument("--lora_weight", default=None)
    p.add_argument("--lora_multiplier", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--skip_decode", action="store_true")
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
    gen_args.compile_blocks = False  # eager map hook must run every step

    ensure_text_strategies(args.text_encoder)
    text_encoder = load_text_encoder(gen_args, dtype=torch.bfloat16, device=device)
    text_encoder.eval()
    shared = {"text_encoder": text_encoder, "conds_cache": {}}
    anima = load_dit_model(gen_args, device, torch.bfloat16)
    shared["model"] = anima

    pairs = json.loads(Path(args.pairs).read_text())
    tok = TokenizeStrategy.get_strategy().qwen3_tokenizer
    tokenize = TokenizeStrategy.get_strategy().tokenize

    grid = args.size // 16  # latent /8, patch 2
    rec = GlyphMapRecorder(anima, grid)
    install_hooks(anima, rec)

    def context_of(prompt):
        return prepare_text_inputs(gen_args, device, anima, shared, prompt=prompt)

    out_dir = make_run_dir("turbo", args.label)
    diff_rows: list[dict] = []
    latents_store: list[tuple[str, torch.Tensor]] = []  # (name, cpu latent)

    total = len(pairs) * args.seeds
    done = 0
    for pi, pair in enumerate(pairs):
        full_ids = tokenize([pair["full"]])[0][0]
        cols = {
            "glyph": glyph_token_cols(tok, full_ids, pair["glyph"]),
            "region": tag_token_cols(tok, full_ids, pair["region_tag"]),
        }
        if not cols["glyph"]:
            raise RuntimeError(
                f"glyph columns not found for pair {pi}: {pair['glyph']!r}"
            )
        print(f"pair {pi}: glyph cols={cols['glyph']} region cols={cols['region']}")

        ctx_full, ctx_null = context_of(pair["full"])
        ctx_base, _ = context_of(pair["base"])
        for s in range(args.seeds):
            rec.enabled = True
            rec.begin_generation(cols)
            with torch.no_grad():
                lat_full = generate_body(
                    gen_args, anima, ctx_full, ctx_null, device, seed=s
                )
            rec.enabled = False
            with torch.no_grad():
                lat_base = generate_body(
                    gen_args, anima, ctx_base, ctx_null, device, seed=s
                )
            d = diff_spread(lat_full, lat_base)
            dmap = d.pop("map")
            diff_rows.append({"pair": pi, "seed": s, **d})
            save_diff_heatmap(
                dmap,
                out_dir / f"diff_p{pi}_s{s}.png",
                f"pair {pi} seed {s} | spread={d['spread']} area50={d['area50']}",
            )
            latents_store.append((f"full_p{pi}_s{s}", lat_full.detach().cpu()))
            latents_store.append((f"base_p{pi}_s{s}", lat_base.detach().cpu()))
            clean_memory_on_device(device)
            done += 1
            print(
                f"[{done}/{total}] pair={pi} seed={s} "
                f"spread={d['spread']} area50={d['area50']}",
                flush=True,
            )

    agg = aggregate(rec.rows)
    heat_png = save_heatmaps(rec, out_dir, args.label or "arm")

    import csv

    with (out_dir / "rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "gen",
                "block",
                "step",
                "sigma",
                "tag",
                "n_cols",
                "conc_mean",
                "conc_max",
                "mass_per_tok",
            ],
        )
        w.writeheader()
        w.writerows(rec.rows)

    artifacts = ["rows.csv"] + ([heat_png] if heat_png else [])
    if not args.skip_decode:
        del text_encoder, shared["text_encoder"]
        clean_memory_on_device(device)
        vae = load_vae(args.vae, device="cpu")
        for name, lat in latents_store:
            img = pixels_to_pil(decode_latent(vae, lat, device))
            img.save(out_dir / f"{name}.png")
            artifacts.append(f"{name}.png")
        del vae
        clean_memory_on_device(device)

    mean_spread = round(sum(r["spread"] for r in diff_rows) / len(diff_rows), 5)
    mean_area50 = round(sum(r["area50"] for r in diff_rows) / len(diff_rows), 5)
    write_result(
        out_dir,
        script="bench/turbo/glyph_scatter_probe.py",
        label=args.label,
        args=args,
        metrics={
            "lora_weight": args.lora_weight,
            "infer_steps": args.infer_steps,
            "cfg": args.cfg,
            "n_pairs": len(pairs),
            "n_seeds": args.seeds,
            "n_blocks": rec.n_blocks,
            "diff": {"rows": diff_rows, "mean_spread": mean_spread,
                     "mean_area50": mean_area50},
            "maps": agg,
        },
        artifacts=artifacts,
    )

    print(f"\ndiff: mean_spread={mean_spread} mean_area50={mean_area50}")
    for tag, r in agg.items():
        for band, b in r["banded"].items():
            print(
                f"  map {tag} @ {band}: conc_mean={b['conc_mean']} "
                f"conc_max={b['conc_max']} mass/tok={b['mass_per_tok']}"
            )
    print(f"\nresults -> {out_dir}")


if __name__ == "__main__":
    main()
