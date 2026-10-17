#!/usr/bin/env python3
"""Front-loaded text-drive boost — Phase-0/1 render grid (proposal
_archive/proposals/frontload_text_boost.md).

Inference-only arms against a matched-seed/NFE baseline, all gated to the
high-σ window where cross-attn text drive actually exists (peaks at σ=1, ~0.02
floor below σ≈0.85 — docs/findings/crossattn_self_attn_dominance.md):

  * **baseline**   — plain CFG at --guidance.
  * **cfg_hi g**   — arm (a): guidance scale g for σ ≥ --band, --guidance below.
                     Boosts the whole cond−uncond delta (blunt). CLOSED by
                     Phase 0 (style collapse fails G2 at every useful strength)
                     — kept runnable for reference; default strengths empty.
  * **xattn λ**    — arm (b): cross-attn residual gain λ for σ ≥ --band,
                     **conditional forward only** (uncond runs at identity, so
                     the boost lands entirely in the guidance delta and isn't
                     spent amplifying the negative prompt). Uses the per-block
                     ``_xattn_gain`` buffer (compile-safe fill_, no recompile).
                     Phase-0 G1+G2 PASS at λ ∈ {1.5, 2.0}.
  * **token α**    — arm (c), Phase 1: token-selective embedding boost. Scales
                     ONLY the annotated weak-tag token rows of the conditional
                     embedding by α for σ ≥ --band (uncond + all other tokens
                     untouched). Spans come from each prompt's ``boost`` list;
                     rows are located with the **T5** tokenizer's offset
                     mapping — the crossattn embedding is LLM-adapter output
                     aligned to T5 target positions (library/inference/text.py),
                     NOT Qwen3 positions. NB the Phase-1 framing "(c) works
                     through K/V and can shift allocation" was WRONG: k_norm
                     (RMSNorm per token×head) is scale-invariant, so a positive
                     row scale is annihilated on the K path — arm (c) is
                     architecturally a pure token-selective **V (loudness)**
                     gain. Allocation was never tested; that's arm (d).
  * **kbias β**    — arm (d), Phase 1': allocation-only probe. Additive
                     per-key logit bias on every cross-attn QK^T row
                     (``_ctx_k_bias`` buffer → SDPA additive mask), cond
                     forward in-band only. Sink-preserving shape: −β on
                     non-span *content* tokens, 0 on span tokens AND on
                     padding/special positions — mass reallocates from
                     competing content toward the span (and slightly toward
                     the padding sinks, the safe direction) instead of
                     draining the sinks (the known burn direction; the model
                     leans on zero-padded positions as attention sinks).
                     Odds vs demoted tokens multiply by e^β.
  * **tokα+kbβ**   — arm (e): (c) + (d) composed — loudness × allocation.
                     If (d) is a genuine second axis this should beat both.
  * **fade λ**     — arm (f): arm (b)'s uniform gain with the hard σ cutoff
                     replaced by a cosine fade (full λ at σ ≥ --band, cosine
                     ramp to identity over --fade_width below it). Checks the
                     smooth window keeps (b)'s wins while removing the
                     vector-field discontinuity at the band edge.
  * **renorm λ**   — arm (g): arm (b) with norm matching — after the
                     λ·residual add, the hidden state is rescaled back
                     toward its gain-1.0 norm, so the boost rotates the
                     state toward the cross-attn direction WITHOUT leaving
                     the norm shell downstream self-attn was trained on
                     (the OOD-mitigation read on arm (b)'s desat/framing
                     side effects). Spec 'λ[:tok|img][:ρ]': full per-token
                     matching ('tok', the default) flattens the token-norm
                     distribution and greys the tone (Phase-1'' finding —
                     the tokens whose energy legitimately spikes under the
                     boost are exactly the highlight/neon peaks, and they
                     get clamped hardest); 'img' matches the per-image
                     MEAN norm instead (energy budget bounded, relative
                     peaks preserved); ρ applies scale**ρ (partial
                     correction).
  * **pdg α**      — arm (h): prompt-delta guidance — the fully
                     in-distribution upper bound. No activation is touched;
                     in-band the CFG velocity gains α·(v(c) − v(c\\S)) where
                     c\\S is the prompt with the boost spans textually
                     removed (re-encoded). The span's semantic direction as
                     the model itself measures it, at +1 cond forward per
                     in-band step.

Prompts come from bench/frontload_text_boost/prompts.py: complicated multi-tag
prompts with per-prompt watch tags (G1), the border reproducer + ordinary
regressions (G2). Judgment is eyeball on the contact sheet; sat/RMS-contrast
per arm is logged as the burn detector. Boosted token spans are logged per
prompt into result.json (audit against silent tokenizer misalignment).

NB the σ≥0.85 band covers ~10 of 28 shifted-schedule steps (flow_shift 3.0),
not "2-3 steps" — the in-band step count and σ values are logged per run; use
--band 0.95 for a tighter window if the default reads as too aggressive.

Run from anima_lora/::

    # Phase-1 (b)-vs-(c) grid:
    uv run python bench/frontload_text_boost/render_compare.py --compile
    # Phase-0 reproduction (arm (a) included):
    uv run python bench/frontload_text_boost/render_compare.py \
        --cfg_hi_values 7 10 --token_values --compile
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from anima_lora import decode_to_pil, load_vae  # noqa: E402
from bench._anima import (  # noqa: E402
    DEFAULT_NEG,
    add_common_args,
    add_model_args,
    build_anima,
)
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.frontload_text_boost.prompts import PROMPTS  # noqa: E402
from bench.fsg.probe_golden_path import _velocity  # noqa: E402
from library.anima import models as anima_models  # noqa: E402
from library.anima import text_strategies  # noqa: E402
from library.inference.adapters import (  # noqa: E402
    set_xattn_gain,
    set_xattn_kbias,
    set_xattn_renorm,
)
from library.inference.models import load_text_encoder  # noqa: E402
from library.inference.sampling import (  # noqa: E402
    ERSDESampler,
    get_timesteps_sigmas,
)
from library.inference.text import prepare_text_inputs, process_escape  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

log = logging.getLogger("bench.frontload_text_boost.render")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _t5_span_indices(
    t5_tokenizer, prompt: str, spans: list[str]
) -> tuple[list[int], list[int], list[str]]:
    """(span row indices, ALL content row indices, span tokens for audit).

    The conditional crossattn embedding is LLM-adapter output aligned to the
    **T5** target token positions (library/inference/text.py feeds
    ``target_input_ids`` = T5 ids), so spans are located with the T5 fast
    tokenizer's offset mapping on the escape-processed prompt — every
    occurrence of every span. Special/pad positions have empty offsets and
    never match (so the kbias arm's −β never lands on EOS/padding — those
    positions are the attention sinks and must stay untouched); a span
    missing from the prompt is a hard error (a silent no-op here would fake
    a null result).
    """
    prompt = process_escape(prompt)
    enc = t5_tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    ranges: list[tuple[int, int]] = []
    for span in spans:
        start = prompt.find(span)
        if start < 0:
            raise ValueError(f"boost span {span!r} not found in prompt {prompt!r}")
        while start >= 0:
            ranges.append((start, start + len(span)))
            start = prompt.find(span, start + 1)
    content = [i for i, (a, b) in enumerate(offsets) if b > a]
    idxs = [
        i
        for i, (a, b) in enumerate(offsets)
        if b > a and any(a < e and b > s for s, e in ranges)
    ]
    toks = t5_tokenizer.convert_ids_to_tokens([enc["input_ids"][i] for i in idxs])
    return idxs, content, toks


def _ablate_prompt(prompt: str, spans: list[str]) -> str:
    """Prompt with every occurrence of every boost span removed (arm (h)).

    Textual removal, then comma/space/period tidy-up so c\\S stays a
    well-formed caption. Ablating must change the prompt — a span list that
    removes nothing would silently turn pdg into a zero direction.
    """
    out = prompt
    for span in spans:
        out = out.replace(span, "")
    for a, b in (
        (" ,", ","),
        (",,", ","),
        ("  ", " "),
        (" .", "."),
        (".,", "."),
        ("..", "."),
    ):
        while a in out:
            out = out.replace(a, b)
    out = out.replace(", .", ".").strip(" ,")
    if out == prompt:
        raise ValueError(f"span ablation left prompt unchanged: {prompt!r}")
    return out


def _sat_contrast(im: Image.Image) -> tuple[float, float]:
    """(mean HSV saturation, RMS contrast) in [0,1] — the arm-(a) burn detector."""
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    lum = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return float(sat.mean()), float(lum.std())


@torch.no_grad()
def _trajectory(
    anima,
    x0,
    sig,
    sigmas_t,
    embed,
    nembed,
    pad,
    guidance,
    *,
    kind,
    value,
    band,
    sampler,
    er_seed,
    device,
    dtype,
    embed_hi=None,
    kbias=None,
    fade_width=0.25,
    embed_ab=None,
):
    """One full denoise. ``kind`` is 'base' | 'cfg' | 'xattn' | 'token' |
    'kbias' | 'combo' | 'fade' | 'renorm' | 'pdg'.

    The er_sde solver is rebuilt per call seeded by ``er_seed`` so the injected
    noise sequence is identical across arms — differences are the intervention.
    ``embed_hi`` (kinds 'token'/'combo') is the span-boosted conditional
    embedding, used in place of ``embed`` on the cond forward for in-band
    steps; ``kbias`` (kinds 'kbias'/'combo') is the (L,) per-key logit bias
    set on the cond forward for in-band steps. Uncond and below-band steps
    always see the plain embeddings / identity model. 'fade' replaces the
    hard in-band gate with a cosine ramp: gain λ at σ ≥ band, cosine to 1.0
    at σ = band − fade_width. 'renorm' is arm (b) with per-token norm
    matching (set_xattn_renorm). 'pdg' leaves every forward untouched and
    adds value·(v(c) − v(c\\S)) to the CFG velocity in-band, where
    ``embed_ab`` is the span-ablated conditional embedding (one extra cond
    forward per in-band step).
    """
    x = x0.clone()
    er = (
        ERSDESampler(sigmas_t, seed=er_seed, device=device)
        if sampler == "er_sde"
        else None
    )
    for i in range(len(sig) - 1):
        s_i = sig[i]
        in_band = s_i >= band
        if kind == "xattn" and in_band:
            set_xattn_gain(anima, value)
        if kind == "renorm" and in_band:
            lam_rn, per_tok_rn, frac_rn = value
            set_xattn_gain(anima, lam_rn)
            set_xattn_renorm(anima, True, per_token=per_tok_rn, frac=frac_rn)
        fade_gain = 1.0
        if kind == "fade":
            if in_band:
                w = 1.0
            elif s_i > band - fade_width:
                w = 0.5 * (
                    1.0 - math.cos(math.pi * (s_i - (band - fade_width)) / fade_width)
                )
            else:
                w = 0.0
            fade_gain = 1.0 + (value - 1.0) * w
            if fade_gain != 1.0:
                set_xattn_gain(anima, fade_gain)
        boosted_kb = kind in ("kbias", "combo") and in_band
        if boosted_kb:
            set_xattn_kbias(anima, kbias)
        cond_embed = embed_hi if (kind in ("token", "combo") and in_band) else embed
        vc = _velocity(anima, x, s_i, cond_embed, pad, i)
        if (kind in ("xattn", "renorm") and in_band) or fade_gain != 1.0:
            set_xattn_gain(anima, 1.0)  # cond-only: uncond runs at identity
        if kind == "renorm" and in_band:
            set_xattn_renorm(anima, False)
        if boosted_kb:
            set_xattn_kbias(anima, None)
        vu = _velocity(anima, x, s_i, nembed, pad, i)
        g = value if (kind == "cfg" and in_band) else guidance
        noise_pred = vu + g * (vc - vu)
        if kind == "pdg" and in_band:
            va = _velocity(anima, x, s_i, embed_ab, pad, i)
            noise_pred = noise_pred + value * (vc - va)
        if er is not None:
            denoised = x.float() - s_i * noise_pred.float()
            x = er.step(x, denoised, i).to(dtype)
        else:
            x = x - (sig[i] - sig[i + 1]) * noise_pred
    return x


def _panel(
    images: list[Image.Image], labels: list[str], caption: str, thumb: int
) -> Image.Image:
    """Horizontal strip of thumbnails with column labels + caption header."""
    imgs = [im.resize((thumb, thumb), Image.LANCZOS) for im in images]
    w, h = imgs[0].size
    head = 72
    panel = Image.new("RGB", (w * len(imgs), h + head), "white")
    d = ImageDraw.Draw(panel)
    lines = [caption[j : j + 110] for j in range(0, len(caption), 110)][:2]
    d.text((6, 4), "\n".join(lines), fill="black")
    for k, (im, lab) in enumerate(zip(imgs, labels)):
        panel.paste(im, (k * w, head))
        d.text((k * w + 6, head - 20), lab, fill="black")
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap)  # --dit/--vae/--text_encoder
    ap.add_argument("--adapter", default=None, help="Optional LoRA checkpoint.")
    ap.add_argument(
        "--kinds",
        nargs="+",
        default=None,
        help="Filter prompt kinds (complex/border/regression). Default: all.",
    )
    ap.add_argument(
        "--ids", nargs="+", default=None, help="Filter prompt ids. Default: all."
    )
    ap.add_argument(
        "--neg_prompt",
        default=DEFAULT_NEG,
        help="Default = the production make-test negative (CFG arms should be "
        "judged against the worded negative actually used in production).",
    )
    ap.add_argument("--num_seeds", type=int, default=1)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--infer_steps", type=int, default=28)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument(
        "--band",
        type=float,
        default=0.85,
        help="σ cutoff: arms act at σ ≥ band. Default 0.85 (the drive-floor σ "
        "from the front-loading finding; ~10/28 shifted steps).",
    )
    ap.add_argument(
        "--cfg_hi_values",
        type=float,
        nargs="*",
        default=[],
        help="Arm (a) strengths — one cfg_hi arm per value. Default empty: "
        "arm (a) was CLOSED by Phase 0 (style collapse); pass values only to "
        "reproduce the Phase-0 grid.",
    )
    ap.add_argument(
        "--xattn_values",
        type=float,
        nargs="*",
        default=[2.0],
        help="Arm (b) strengths — one xattn arm per value. Default 2.0 (the "
        "Phase-0 winner; 2.0 strictly dominated 1.5 on adherence). Empty to "
        "drop arm (b).",
    )
    ap.add_argument(
        "--token_values",
        type=float,
        nargs="*",
        default=[1.5, 2.0],
        help="Arm (c) strengths — token-selective embedding boost α, one arm "
        "per value. Requires every selected prompt to carry a ``boost`` span "
        "list. Empty to drop arm (c).",
    )
    ap.add_argument(
        "--kbias_values",
        type=float,
        nargs="*",
        default=[0.7, 1.4],
        help="Arm (d) strengths — per-key logit bias β (−β on non-span "
        "content tokens; span/padding/special untouched). e^β = odds "
        "multiplier of span vs demoted tokens (0.7≈2×, 1.4≈4×). Requires "
        "boost spans. Empty to drop arm (d).",
    )
    ap.add_argument(
        "--combo_values",
        nargs="*",
        default=["1.5:0.7"],
        help="Arm (e) strengths — 'α:β' pairs composing token loudness α "
        "with kbias allocation β. Empty to drop arm (e).",
    )
    ap.add_argument(
        "--fade_values",
        type=float,
        nargs="*",
        default=[2.0],
        help="Arm (f) strengths — arm (b) gain λ with a cosine fade below "
        "the band edge instead of the hard cutoff. Empty to drop arm (f).",
    )
    ap.add_argument(
        "--fade_width",
        type=float,
        default=0.25,
        help="Arm (f) σ-width of the cosine ramp below --band (λ full at "
        "σ ≥ band, identity at σ ≤ band − fade_width).",
    )
    ap.add_argument(
        "--renorm_values",
        nargs="*",
        default=["2"],
        help="Arm (g) specs 'λ[:tok|img][:ρ]' — arm (b) gain λ with norm "
        "matching. 'tok' (default) matches each token's gain-1 norm (full "
        "shell — flattens norm peaks, greys the tone); 'img' matches the "
        "per-image MEAN norm (energy budget bounded, relative peaks kept); "
        "ρ applies scale**ρ (partial correction, default 1). E.g. '2', "
        "'2:img', '2:tok:0.5'. Empty to drop arm (g).",
    )
    ap.add_argument(
        "--pdg_values",
        type=float,
        nargs="*",
        default=[2.0, 4.0],
        help="Arm (h) strengths — prompt-delta guidance α on v(c) − v(c\\S) "
        "in-band (+1 cond forward per in-band step). Requires boost spans. "
        "Empty to drop arm (h).",
    )
    ap.add_argument(
        "--sampler",
        choices=["euler", "er_sde"],
        default="er_sde",
        help="Default er_sde = production 28-step protocol (seed shared across "
        "arms). 'euler' for a deterministic isolation read.",
    )
    ap.add_argument(
        "--thumb", type=int, default=448, help="Contact-sheet thumbnail edge px."
    )
    add_common_args(ap)
    args = ap.parse_args()

    prompts = [
        p
        for p in PROMPTS
        if (args.kinds is None or p["kind"] in args.kinds)
        and (args.ids is None or p["id"] in args.ids)
    ]
    if not prompts:
        raise SystemExit("prompt filter matched nothing")

    # (label, kind, value)
    arms: list[tuple[str, str, float | None]] = [("baseline", "base", None)]
    arms += [(f"cfg_hi {g:g}", "cfg", g) for g in args.cfg_hi_values]
    arms += [(f"xattn {lam:g}", "xattn", lam) for lam in args.xattn_values]
    arms += [(f"token {a:g}", "token", a) for a in args.token_values]
    combos = [tuple(float(x) for x in c.split(":")) for c in args.combo_values]
    arms += [(f"kbias {b:g}", "kbias", b) for b in args.kbias_values]
    arms += [(f"tok{a:g}+kb{b:g}", "combo", (a, b)) for a, b in combos]
    arms += [(f"fade {lam:g}", "fade", lam) for lam in args.fade_values]
    for spec in args.renorm_values:
        parts = spec.split(":")
        lam = float(parts[0])
        per_tok = (parts[1] if len(parts) > 1 else "tok") == "tok"
        frac = float(parts[2]) if len(parts) > 2 else 1.0
        lab = f"renorm{'' if per_tok else 'I'} {lam:g}"
        if frac != 1.0:
            lab += f"@{frac:g}"
        arms.append((lab, "renorm", (lam, per_tok, frac)))
    arms += [(f"pdg {a:g}", "pdg", a) for a in args.pdg_values]

    need_spans = bool(
        args.token_values or args.kbias_values or combos or args.pdg_values
    )
    if need_spans:
        missing = [p["id"] for p in prompts if not p.get("boost")]
        if missing:
            raise SystemExit(
                f"span-selective arms requested but prompts lack boost spans: {missing}"
            )

    bundle = build_anima(args, adapter=args.adapter, train_mode=False)
    anima, device, dtype = bundle.anima, bundle.device, bundle.dtype
    vae = load_vae(args.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True)

    _, sigmas = get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)
    sig = [float(s) for s in sigmas]
    in_band = [round(s, 3) for s in sig[:-1] if s >= args.band]
    log.info(f"arms: {[a[0] for a in arms]}")
    log.info(
        f"band σ≥{args.band}: {len(in_band)}/{args.infer_steps} steps in band "
        f"(σ={in_band})"
    )

    # Encode every prompt up front with one shared text encoder, then free it
    # before the denoise loops (lazy-loading discipline; the DiT is already
    # resident, so avoid 17× TE load/free churn mid-run).
    text_encoder = load_text_encoder(
        text_encoder=args.text_encoder, dtype=torch.bfloat16, device=device
    )
    text_encoder.eval()
    shared = {"text_encoder": text_encoder}
    encoded: list[tuple[torch.Tensor, torch.Tensor]] = []
    for p in prompts:
        ctx, ctx_null = prepare_text_inputs(
            device=device,
            anima=anima,
            prompt=p["prompt"],
            negative_prompt=args.neg_prompt,
            text_encoder_path=args.text_encoder,
            shared_models=shared,
        )
        encoded.append(
            (
                ctx["embed"][0].to(device, dtype).expand(1, -1, -1),
                ctx_null["embed"][0].to(device, dtype).expand(1, -1, -1),
            )
        )
    # Arm (h): encode the span-ablated prompt c\S with the same shared TE.
    encoded_ab: list[torch.Tensor | None] = [None] * len(prompts)
    if args.pdg_values:
        for pi, p in enumerate(prompts):
            ctx_ab, _ = prepare_text_inputs(
                device=device,
                anima=anima,
                prompt=_ablate_prompt(p["prompt"], p["boost"]),
                negative_prompt=args.neg_prompt,
                text_encoder_path=args.text_encoder,
                shared_models=shared,
            )
            encoded_ab[pi] = ctx_ab["embed"][0].to(device, dtype).expand(1, -1, -1)
    # Arm (c) span → embedding-row resolution. The tokenize strategy singleton
    # was installed by prepare_text_inputs above; its t5_tokenizer defines the
    # row alignment of the crossattn embedding.
    span_idxs: list[list[int]] = []
    content_idxs: list[list[int]] = []
    boost_audit: list[dict] = []
    if need_spans:
        t5_tok = text_strategies.TokenizeStrategy.get_strategy().t5_tokenizer
        for p in prompts:
            idxs, content, toks = _t5_span_indices(t5_tok, p["prompt"], p["boost"])
            if not idxs:
                raise SystemExit(f"{p['id']}: boost spans matched zero T5 tokens")
            span_idxs.append(idxs)
            content_idxs.append(content)
            boost_audit.append(
                {
                    "id": p["id"],
                    "spans": p["boost"],
                    "n_tokens": len(idxs),
                    "n_content_tokens": len(content),
                    "token_indices": idxs,
                    "tokens": toks,
                    "ablated_prompt": (
                        _ablate_prompt(p["prompt"], p["boost"])
                        if args.pdg_values
                        else None
                    ),
                }
            )
            log.info(
                f"  [{p['id']}] boosting {len(idxs)}/{len(content)} content "
                f"T5 tokens: {toks}"
            )
    del text_encoder, shared
    gc.collect()
    clean_memory_on_device(device)

    C = anima_models.Anima.LATENT_CHANNELS
    hl, wl = args.height // 8, args.width // 8
    pad = torch.zeros(1, 1, hl, wl, dtype=dtype, device=device)

    run_dir = make_run_dir("frontload_text_boost", label=args.label or "phase0")
    sheet_files: list[str] = []
    records: list[dict] = []
    sat_con: dict[str, list[tuple[float, float]]] = {lab: [] for lab, _, _ in arms}

    for pi, (p, (embed, nembed)) in enumerate(zip(prompts, encoded)):
        for sj in range(max(1, args.num_seeds)):
            g = torch.Generator(device=device).manual_seed(args.seed + pi * 1000 + sj)
            x0 = torch.randn((1, C, 1, hl, wl), generator=g, device=device, dtype=dtype)
            er_seed = args.seed + pi * 1000 + sj  # shared across arms

            imgs, labels = [], []
            for lab, kind, value in arms:
                embed_hi = None
                kbias_vec = None
                if kind in ("token", "combo"):
                    # α on the span rows only; 1.5/2.0 are dyadic-exact in bf16.
                    alpha = value[0] if kind == "combo" else value
                    embed_hi = embed.clone()
                    embed_hi[:, span_idxs[pi], :] *= alpha
                if kind in ("kbias", "combo"):
                    # Sink-preserving allocation bias: −β on non-span content
                    # rows, 0 on span AND padding/special (the sinks).
                    beta = value[1] if kind == "combo" else value
                    kbias_vec = torch.zeros(
                        embed.shape[1], device=device, dtype=torch.float32
                    )
                    kbias_vec[content_idxs[pi]] = -beta
                    kbias_vec[span_idxs[pi]] = 0.0
                lat = _trajectory(
                    anima,
                    x0,
                    sig,
                    sigmas,
                    embed,
                    nembed,
                    pad,
                    args.guidance,
                    kind=kind,
                    value=value,
                    band=args.band,
                    sampler=args.sampler,
                    er_seed=er_seed,
                    device=device,
                    dtype=dtype,
                    embed_hi=embed_hi,
                    kbias=kbias_vec,
                    fade_width=args.fade_width,
                    embed_ab=encoded_ab[pi],
                )
                im = decode_to_pil(vae, lat, device)
                sat, con = _sat_contrast(im)
                sat_con[lab].append((sat, con))
                imgs.append(im)
                labels.append(f"{lab}  sat{sat:.3f} c{con:.3f}")
                safe = lab.replace(" ", "_").replace(".", "p")
                im.save(run_dir / f"{p['id']}_s{sj}_{safe}.png")
                records.append(
                    {
                        "id": p["id"],
                        "kind": p["kind"],
                        "seed": sj,
                        "arm": lab,
                        "saturation": sat,
                        "rms_contrast": con,
                    }
                )
            caption = f"[{p['kind']}:{p['id']}] watch: {p['watch']}"
            # One sheet per prompt+seed (all arms side by side) — saved as it
            # renders, so a killed run still leaves complete rows behind.
            sheet_name = f"sheet_{p['id']}_s{sj}.png"
            _panel(imgs, labels, caption, args.thumb).save(run_dir / sheet_name)
            sheet_files.append(sheet_name)
            log.info(f"  [{pi + 1}/{len(prompts)} seed {sj}] {p['id']} rendered")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Per-arm burn read: Δsat/Δcontrast vs baseline (arm (a) G2 gate helper).
    base_sat = float(np.mean([s for s, _ in sat_con["baseline"]]))
    base_con = float(np.mean([c for _, c in sat_con["baseline"]]))
    agg: dict[str, dict] = {}
    log.info("\narm            sat     Δsat%    contrast  Δcon%")
    for lab, _, _ in arms:
        vals = sat_con[lab]
        msat = float(np.mean([s for s, _ in vals]))
        mcon = float(np.mean([c for _, c in vals]))
        agg[lab] = {
            "saturation": msat,
            "rms_contrast": mcon,
            "d_saturation_pct": 100.0 * (msat - base_sat) / max(base_sat, 1e-8),
            "d_rms_contrast_pct": 100.0 * (mcon - base_con) / max(base_con, 1e-8),
            "n": len(vals),
        }
        log.info(
            f"{lab:14s} {msat:.4f}  {agg[lab]['d_saturation_pct']:+6.1f}   "
            f"{mcon:.4f}   {agg[lab]['d_rms_contrast_pct']:+6.1f}"
        )

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "n_prompts": len(prompts),
            "num_seeds": args.num_seeds,
            "sampler": args.sampler,
            "infer_steps": args.infer_steps,
            "guidance": args.guidance,
            "band": args.band,
            "band_steps": in_band,
            "arms": [a[0] for a in arms],
            "sat_contrast": agg,
            "boost_spans": boost_audit,
            "per_image": records,
        },
        label=args.label,
        artifacts=sheet_files,
        device=device,
    )
    log.info(f"\n→ {run_dir}/  ({len(sheet_files)} sheet_<id>_s<seed>.png)")


if __name__ == "__main__":
    main()
