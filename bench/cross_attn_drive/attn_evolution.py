"""Cross-attn *pattern* evolution probe — does the attention map keep re-routing
into mid-σ, or lock in the first few high-σ steps?

Motivation
----------
The shipped finding (``CfgDeltaProbe`` / ANIMA_CFG_DELTA_LOG, see
``project_crossattn_drive_frontloaded``) measures the *velocity-level* text drive:
``||v_cond − v_uncond|| / ||v_cond||`` peaks at σ=1 and ``cos(v_cond, v_uncond)``
→0.997 below σ≈0.85, i.e. below mid-σ text only *rescales* the base velocity, it
does not *redirect* it. That is a statement about the guidance delta, NOT about the
cross-attention map itself.

This probe asks the complementary question directly: **does the (token→patch)
cross-attention map keep moving as σ drops?** We log, per block per step, the eager
softmax attention map (image-patch queries × text-token keys, head-averaged) and
reduce it to two scalars vs the *previous* step of the same trajectory:

  * **drift** = 1 − cos(vec(A_t), vec(A_{t-1})) — how much the routing re-arranges
    between consecutive steps. High through mid-σ ⇒ "the pattern keeps moving";
    collapses early ⇒ "locked in the first steps, later steps only follow".
  * **entropy** = mean over patches of H(text-token distribution) — sharpening of
    the routing (does each patch commit to fewer text tokens over σ?).

Why this is the *right* probe for the question (and dodges a confound)
---------------------------------------------------------------------
The diff-based ``tag_influence`` signals (influence / region mask) subtract a
tag-dropped baseline, so they inherit any quality gap between the full-caption and
the tag-dropped image (e.g. on the sincos LoRA the dropped baseline rendered
*cleaner* — that asymmetry leaks straight into the diff). This probe reads the
attention map of a single generation, **no baseline subtraction**, so the
baseline-quality confound cannot touch it.

What it does NOT measure: the per-step *magnitude* the map contributes to the
residual (that rides the adaln ``gate_cross_attn`` and is the CfgDelta story). A
map can keep re-routing while contributing ~nothing — high drift + low CfgDelta
ratio is exactly the "moves but doesn't steer" regime. Read this probe alongside
the CfgDelta curve, not instead of it.

Usage
-----
    uv run python bench/cross_attn_drive/attn_evolution.py \
        --captions 6 --seeds 2 --infer_steps 28 --cfg 4 \
        [--lora_weight output/ckpt/anima_sincos2.safetensors] [--label sincos2]

Compile is forced OFF (the eager logging hook must run every step; torch.compile
would trace it once and freeze the side effect).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima_lora import GenerationRequest  # noqa: E402
from bench._anima import add_common_args, add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima.models import Attention  # noqa: E402
from library.anima.text_strategies import TokenizeStrategy  # noqa: E402
from library.env import anima_home  # noqa: E402
from library.inference.generation import generate_body  # noqa: E402
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.text import (  # noqa: E402
    ensure_text_strategies,
    prepare_text_inputs,
)
from library.runtime.device import clean_memory_on_device  # noqa: E402

_BLOCK_RE = re.compile(r"blocks\.(\d+)\.")


# ---------------------------------------------------------------------------
# Caption corpus (mirrors tag_influence.load_captions, trimmed).
# ---------------------------------------------------------------------------
def load_captions(args) -> list[str]:
    def _clean(line: str) -> str:
        line = line.strip()
        if " --" in line:
            line = line.split(" --", 1)[0].strip()
        return line

    if args.prompts:
        lines = [_clean(x) for x in Path(args.prompts).read_text().splitlines()]
        return [x for x in lines if x]

    index_path = Path(args.caption_index)
    if not index_path.is_absolute():
        index_path = anima_home() / index_path
    if index_path.exists():
        meta = json.loads(index_path.read_text())
        src_root = anima_home() / meta.get("meta", {}).get("src", "image_dataset")
        caps: list[str] = []
        for rec in meta.get("image_meta", {}).values():
            try:
                caps.append(_clean((src_root / rec["path"]).read_text()))
            except OSError:
                continue
        caps = [c for c in caps if c]
        if caps:
            return caps

    fallback = Path(__file__).parent / "real_prompts.txt"
    return [_clean(x) for x in fallback.read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# Tag → cross-attn key columns. The cross-attn keys are the Qwen3 encoder hidden
# states over ``qwen3_input_ids`` (512, max-padded), position i ↔ token i, so a
# tag's columns are exactly its token-id subsequence within that id tensor. We
# match ids (not chars) → alignment is exact regardless of special tokens; the
# leading-space BPE variant (" tag") is what appears mid-caption after ", ".
# ---------------------------------------------------------------------------
def tag_token_cols(tok, full_ids, tag: str) -> list[int]:
    f = full_ids.tolist()
    for variant in (" " + tag, tag, ", " + tag):
        vid = tok(variant, add_special_tokens=False).input_ids
        n = len(vid)
        if n == 0:
            continue
        for i in range(len(f) - n + 1):
            if f[i : i + n] == vid:
                return list(range(i, i + n))
    return []


def caption_has_tag(caption: str, tag: str) -> bool:
    return tag in {t.strip() for t in caption.split(",")}


# ---------------------------------------------------------------------------
# Attention-map recorder. Threaded into the live denoise loop via two monkey-
# patches: a wrapper on the whole-model forward (to latch σ + cond/uncond
# parity) and a wrapper on every cross_attn module (to read the map).
# ---------------------------------------------------------------------------
class AttnRecorder:
    def __init__(self, anima):
        self.anima = anima
        self.rows: list[
            dict
        ] = []  # {gen, block, depth_frac, step, sigma, drift, entropy}
        self._prev: dict[int, torch.Tensor] = {}  # block -> head-avg map [Nq,Nk]
        self._sigma = None
        self._last_sigma = None
        self._step = -1
        self._pass = 0  # 0 == conditional (first call at a σ), 1 == uncond
        self._gen = -1
        self._n_blocks = 0
        self._tag_cols: dict[str, list[int]] = {}
        # discover depth ordering
        idxs = sorted(
            int(_BLOCK_RE.search(n).group(1))
            for n, m in anima.named_modules()
            if isinstance(m, Attention) and not m.is_selfattn and _BLOCK_RE.search(n)
        )
        self._depth = {b: (i / max(1, len(idxs) - 1)) for i, b in enumerate(idxs)}
        self._n_blocks = len(idxs)

    # --- generation lifecycle (outer loop calls these) ---
    def begin_generation(self, tag_cols: dict[str, list[int]] | None = None):
        self._gen += 1
        self._prev.clear()
        self._sigma = None
        self._last_sigma = None
        self._step = -1
        self._pass = 0
        # {tag: column indices} present in THIS generation's caption
        self._tag_cols = tag_cols or {}

    # --- whole-model forward wrapper sets σ / pass parity ---
    def on_model_forward(self, timesteps: torch.Tensor):
        sigma = float(timesteps.flatten()[0])
        if self._last_sigma is None or sigma != self._last_sigma:
            self._step += 1
            self._pass = 0  # conditional pass leads each step
            self._last_sigma = sigma
            self._sigma = sigma
        else:
            self._pass += 1

    # --- per-block cross_attn map read (conditional pass only) ---
    @torch.no_grad()
    def on_cross_attn(self, block: int, q: torch.Tensor, k: torch.Tensor):
        if self._pass != 0:  # only the text-conditional pass carries the prompt
            return
        # q:[B,Nq,H,D] k:[B,Nk,H,D]; eager head-averaged softmax map, fp32.
        B, Nq, H, D = q.shape
        Nk = k.shape[1]
        scale = D**-0.5
        qf = q.float()
        kf = k.float()
        amap = q.new_zeros((B, Nq, Nk), dtype=torch.float32)
        ent = q.new_zeros((), dtype=torch.float32)
        for h in range(H):  # loop heads to bound transient memory
            scores = torch.einsum("bqd,bkd->bqk", qf[:, :, h], kf[:, :, h]) * scale
            p = torch.softmax(scores, dim=-1)  # [B,Nq,Nk]
            amap += p
            ent += -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1).mean()
        amap /= H
        entropy = float(ent / H)  # nats, averaged over heads & patches

        cur = amap.mean(0)  # [Nq,Nk], batch-averaged
        prev = self._prev.get(block)

        def _drift(c, p):
            if p is None or p.shape != c.shape:
                return None
            return float(
                1.0
                - torch.nn.functional.cosine_similarity(c.flatten(), p.flatten(), dim=0)
            )

        base_row = {
            "gen": self._gen,
            "block": block,
            "depth_frac": round(self._depth.get(block, 0.0), 4),
            "step": self._step,
            "sigma": round(self._sigma, 5),
        }
        # whole-map row (tag=None): diluted by the padding-sink block, kept for
        # reference / continuity with the un-scoped runs.
        self.rows.append(
            {
                **base_row,
                "tag": None,
                "drift": _drift(cur, prev),
                "entropy": round(entropy, 5),
                "mass": None,
            }
        )
        # per-tag rows: drift + attention MASS restricted to the tag's key columns
        # (the padding-free signal). mass = mean over patches of total attention
        # the patches place on the tag's tokens — "is the tag attended here at all".
        for tag, cols in self._tag_cols.items():
            if not cols:
                continue
            c_sub = cur[:, cols]
            p_sub = prev[:, cols] if prev is not None else None
            self.rows.append(
                {
                    **base_row,
                    "tag": tag,
                    "drift": _drift(c_sub, p_sub),
                    "entropy": None,
                    "mass": round(float(c_sub.sum(-1).mean()), 6),
                }
            )
        self._prev[block] = cur


def install_hooks(anima, rec: AttnRecorder):
    # whole-model forward (instance attribute shadows the bound method)
    orig_fwd = anima.forward

    def model_fwd(latents, timesteps, *a, **kw):
        rec.on_model_forward(timesteps)
        return orig_fwd(latents, timesteps, *a, **kw)

    anima.forward = model_fwd

    # every cross_attn module
    for name, mod in anima.named_modules():
        m = _BLOCK_RE.search(name)
        if not (isinstance(mod, Attention) and not mod.is_selfattn and m):
            continue
        block = int(m.group(1))
        orig_attn = mod.forward

        def make(mod_ref, block_ref, orig):
            def attn_fwd(x, attn_params, context, rope_cos_sin=None):
                if rec._pass == 0:
                    q, k, _ = mod_ref.compute_qkv(x, context, rope_cos_sin=None)
                    rec.on_cross_attn(block_ref, q, k)
                return orig(x, attn_params, context, rope_cos_sin=rope_cos_sin)

            return attn_fwd

        mod.forward = make(mod, block, orig_attn)


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------
_BANDS = [
    ("hi", 1.01, 0.9),
    ("upper_mid", 0.9, 0.7),
    ("mid", 0.7, 0.45),
    ("low", 0.45, -0.01),
]
_DEPTH_GROUPS = [("early", 0.0, 0.34), ("mid", 0.34, 0.67), ("late", 0.67, 1.01)]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _band_of(sig):
    for name, hi, lo in _BANDS:
        if lo < sig <= hi:
            return name
    return "low"


def _depth_group(df):
    for name, lo, hi in _DEPTH_GROUPS:
        if lo <= df < hi:
            return name
    return "late"


def _summarize(rows: list[dict]) -> dict:
    """Banded + per-step drift (and mass/entropy when present) for one homogeneous
    row set (all whole-map, or all one tag).

    Drift is reported BOTH raw (1−cos vs previous step) and Δσ-normalized
    (``drift_norm = drift / Δσ`` = re-routing rate per unit σ). The normalized
    form is the honest one for comparing across σ: late steps span a wider Δσ, so
    raw per-step drift over-states tail re-routing. The verdict keys off the
    normalized rate.
    """
    has_mass = any(r.get("mass") is not None for r in rows)
    has_ent = any(r.get("entropy") is not None for r in rows)

    # global step -> Δσ (schedule is shared across blocks/gens; derive from the
    # mean sigma per step). First step has no predecessor -> Δσ None.
    step_sigma = {}
    for r in rows:
        step_sigma.setdefault(r["step"], []).append(r["sigma"])
    step_sigma = {s: sum(v) / len(v) for s, v in step_sigma.items()}
    ordered = sorted(step_sigma)
    dsig = {}
    for i, s in enumerate(ordered):
        dsig[s] = (step_sigma[ordered[i - 1]] - step_sigma[s]) if i > 0 else None

    def _norm(r):
        d, ds = r["drift"], dsig.get(r["step"])
        return (d / ds) if (d is not None and ds) else None

    banded = {}
    for bname, *_ in _BANDS:
        sel = [r for r in rows if _band_of(r["sigma"]) == bname]
        entry = {
            "drift": _r(_mean([r["drift"] for r in sel])),
            "drift_norm": _r(_mean([_norm(r) for r in sel])),
            "n": len(sel),
            "by_depth": {
                g: _r(
                    _mean([_norm(r) for r in sel if _depth_group(r["depth_frac"]) == g])
                )
                for g, *_ in _DEPTH_GROUPS
            },
        }
        if has_mass:
            entry["mass"] = _r(_mean([r["mass"] for r in sel]))
        if has_ent:
            entry["entropy"] = _r(_mean([r["entropy"] for r in sel]))
        banded[bname] = entry

    curve = []
    for s in ordered:
        sel = [r for r in rows if r["step"] == s]
        c = {
            "step": s,
            "sigma": round(step_sigma[s], 5),
            "dsigma": _r(dsig[s]),
            "drift": _r(_mean([r["drift"] for r in sel])),
            "drift_norm": _r(_mean([_norm(r) for r in sel])),
        }
        if has_mass:
            c["mass"] = _r(_mean([r["mass"] for r in sel]))
        if has_ent:
            c["entropy"] = _r(_mean([r["entropy"] for r in sel]))
        curve.append(c)

    return {"banded": banded, "curve": curve, "verdict": _verdict(banded)}


def aggregate(rows: list[dict]) -> dict:
    whole = _summarize([r for r in rows if r.get("tag") is None])
    tags = sorted({r["tag"] for r in rows if r.get("tag") is not None})
    per_tag = {t: _summarize([r for r in rows if r.get("tag") == t]) for t in tags}
    return {"whole": whole, "per_tag": per_tag}


def _r(x):
    return round(x, 5) if x is not None else None


def _verdict(banded) -> dict:
    # Keys off the Δσ-normalized re-routing RATE (raw per-step drift mis-ranks the
    # wide-Δσ tail). "moves into mid-σ" if the mid rate is a meaningful fraction of
    # the high-σ rate; "tail churn floor" if the low-σ rate stays non-trivial.
    hi = banded.get("hi", {}).get("drift_norm")
    mid = banded.get("mid", {}).get("drift_norm")
    low = banded.get("low", {}).get("drift_norm")
    ratio = (mid / hi) if (hi and mid) else None
    low_ratio = (low / hi) if (hi and low) else None
    if ratio is None:
        tag = "inconclusive"
    elif ratio >= 0.5:
        tag = "PATTERN_MOVES_INTO_MID_SIGMA"
    elif ratio >= 0.2:
        tag = "PARTIAL_DECAY_BY_MID_SIGMA"
    else:
        tag = "LOCKS_EARLY"
    return {
        "tag": tag,
        "mid_over_hi_rate_ratio": _r(ratio),
        "low_over_hi_rate_ratio": _r(low_ratio),
        "hi_band_rate": _r(hi),
        "mid_band_rate": _r(mid),
        "low_band_rate": _r(low),
        "note": (
            "rate = Δσ-normalized drift (1-cos of consecutive head-avg cross-attn "
            "maps / Δσ); RE-ROUTING speed, not steering magnitude (mass / CfgDelta "
            "carry that). Tail rebound in RAW drift is a wide-Δσ artifact."
        ),
    }


def maybe_plot(agg: dict, out_dir: Path, label: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    # left: Δσ-normalized re-routing RATE vs σ — whole-map (diluted) + each tag.
    # Normalized (not raw drift): raw per-step drift fakes a low-σ rebound that is
    # just the wider tail Δσ. Rate = honest re-routing speed per unit σ.
    wc = agg["whole"]["curve"]
    ax[0].plot(
        [c["sigma"] for c in wc],
        [c["drift_norm"] for c in wc],
        "o-",
        color="0.6",
        label="whole-map (diluted)",
    )
    for t, r in agg["per_tag"].items():
        c = r["curve"]
        ax[0].plot([x["sigma"] for x in c], [x["drift_norm"] for x in c], "o-", label=t)
    ax[0].set_xlabel("σ (high→low)")
    ax[0].set_ylabel("re-routing rate  (drift / Δσ)")
    ax[0].set_title(f"cross-attn re-routing rate — {label}")
    ax[0].invert_xaxis()
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    # right: per-tag attention MASS vs σ (is the tag attended at all?)
    for t, r in agg["per_tag"].items():
        c = r["curve"]
        ax[1].plot([x["sigma"] for x in c], [x["mass"] for x in c], "s-", label=t)
    ax[1].set_xlabel("σ (high→low)")
    ax[1].set_ylabel("attention mass on tag tokens")
    ax[1].set_title("tag attention mass")
    ax[1].invert_xaxis()
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    p = out_dir / "attn_evolution.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p.name


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def build_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    add_model_args(p)
    p.add_argument("--prompts", default=None, help="Caption file (overrides index).")
    p.add_argument(
        "--caption_index",
        default="post_image_dataset/captions/caption_index.json",
    )
    p.add_argument("--lora_weight", default=None)
    p.add_argument("--lora_multiplier", type=float, default=1.0)
    p.add_argument(
        "--tags",
        default="speech bubble,japanese text,english text",
        help="Comma-list of tags to scope the map onto (empty => whole-map only).",
    )
    p.add_argument("--captions", type=int, default=6, help="Captions per tag.")
    p.add_argument("--seeds", type=int, default=2, help="Seeds per caption.")
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
    gen_args.compile_blocks = False  # hook must stay eager every step

    ensure_text_strategies(args.text_encoder)
    text_encoder = load_text_encoder(gen_args, dtype=torch.bfloat16, device=device)
    text_encoder.eval()
    shared = {"text_encoder": text_encoder, "conds_cache": {}}
    anima = load_dit_model(gen_args, device, torch.bfloat16)
    shared["model"] = anima

    corpus = load_captions(args)
    target_tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    tok = TokenizeStrategy.get_strategy().qwen3_tokenizer
    tokenize = TokenizeStrategy.get_strategy().tokenize

    # Caption selection: when tags are given, pick captions that actually CONTAIN
    # each tag (so its key columns exist). Deterministic spread, deduped, capped
    # per tag. With no tags, fall back to a plain spread across the corpus.
    if target_tags:
        selected: list[str] = []
        seen: set[str] = set()
        for tag in target_tags:
            hits = [c for c in corpus if caption_has_tag(c, tag)]
            stride = max(1, len(hits) // max(1, args.captions))
            picked = [hits[i * stride] for i in range(min(args.captions, len(hits)))]
            for c in picked:
                if c not in seen:
                    seen.add(c)
                    selected.append(c)
        captions = selected
    else:
        n = min(args.captions, len(corpus))
        stride = max(1, len(corpus) // n)
        captions = [corpus[i * stride] for i in range(n)]

    # Pre-resolve {tag: columns} per caption from the exact padded id tensor.
    cap_tag_cols: dict[str, dict[str, list[int]]] = {}
    for cap in captions:
        full_ids = tokenize([cap])[0][0]  # qwen3_input_ids [512]
        cols = {t: tag_token_cols(tok, full_ids, t) for t in target_tags}
        cap_tag_cols[cap] = {t: c for t, c in cols.items() if c}

    rec = AttnRecorder(anima)
    install_hooks(anima, rec)

    def context_of(prompt):
        return prepare_text_inputs(gen_args, device, anima, shared, prompt=prompt)

    total = len(captions) * args.seeds
    done = 0
    for cap in captions:
        ctx, ctx_null = context_of(cap)
        tcols = cap_tag_cols.get(cap, {})
        for s in range(args.seeds):
            rec.begin_generation(tag_cols=tcols)
            with torch.no_grad():
                generate_body(gen_args, anima, ctx, ctx_null, device, seed=s)
            clean_memory_on_device(device)
            done += 1
            print(
                f"[{done}/{total}] tags={list(tcols)} cap={cap[:42]!r} seed={s}",
                flush=True,
            )

    agg = aggregate(rec.rows)

    out_dir = make_run_dir("cross_attn_drive", args.label)
    plot = maybe_plot(agg, out_dir, args.label or "base")

    import csv

    with (out_dir / "rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "gen",
                "tag",
                "block",
                "depth_frac",
                "step",
                "sigma",
                "drift",
                "mass",
                "entropy",
            ],
        )
        w.writeheader()
        w.writerows(rec.rows)

    write_result(
        out_dir,
        script="bench/cross_attn_drive/attn_evolution.py",
        label=args.label,
        args=args,
        metrics={
            "n_captions": len(captions),
            "n_seeds": args.seeds,
            "n_blocks": rec._n_blocks,
            "n_rows": len(rec.rows),
            "lora_weight": args.lora_weight,
            "target_tags": target_tags,
            **agg,
        },
        artifacts=["rows.csv"] + ([plot] if plot else []),
    )
    # headline: whole-map + per-tag verdicts (Δσ-normalized re-routing rate)
    print("whole-map:", json.dumps(agg["whole"]["verdict"]))
    for t, r in agg["per_tag"].items():
        b, v = r["banded"], r["verdict"]
        print(
            f"  tag {t!r}: {v['tag']} "
            f"(rate hi={b['hi']['drift_norm']} mid={b['mid']['drift_norm']} "
            f"low={b['low']['drift_norm']} | mass hi={b['hi'].get('mass')}"
            f"→low={b['low'].get('mass')})"
        )
    print(f"\nresults -> {out_dir}")


if __name__ == "__main__":
    main()
