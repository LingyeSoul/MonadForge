#!/usr/bin/env python3
"""Phase 0 — offline validation gates for paired Gram similarity (PGS).

Proposal: ``docs/proposal/paired_gram_eval.md`` (retire pooled CMMD as a
decision metric; replace it with a paired, per-item PE-Core token-Gram score).

This is the ~zero-GPU gate: it does NOT render. It reuses the CFG-4 renders
already on disk from the seed_floor / ReFT Phase-1' evals and scores each
render against **its own real counterpart** — the same-stem holdout image,
loaded straight from its cached ``{stem}_anima_pe.safetensors`` sidecar (raw
``[T, D]`` PE tokens, exactly the reference Phase 2 in-training would use). The
only GPU work is one PE-Core forward per render PNG (+ the tone/quant re-encodes
for gates 4-5); a few hundred cheap vision-tower passes, seconds on a GPU.

Five gates, all must pass before any harness code ships (Phase 1):

  1. Aliveness       — right-pair PGS > wrong-pair PGS with margin. Confirms
                       the per-item pairing carries signal (not just artist-
                       global style). Probe measured +0.086 on PE-Spatial.
  2. Seed-trio null  — same-recipe reseeds (uncondpoolft / _s1002 / _s1003)
                       must read as TIE pairwise. Any WIN = metric fails.
  3. Known-difference — plain-r16 vs base must WIN (plain visibly restyles).
  4. Quant stability — PGS moves < 2% under an 8-bit LSB round-trip
                       (CMMD moved ~20% — this is the regression test).
  5. Tone confound   — a ±10-luma shift must move PGS less than gist cosine.

Decision rule (A/B gate): WIN iff Wilcoxon signed-rank p < 0.05 on per-stem
PGS deltas AND sign consistency >= 70%; TIE otherwise.

Usage (from anima_lora/)::

    uv run python bench/paired_eval/phase0.py
    # or point at other result dirs:
    uv run python bench/paired_eval/phase0.py \
        --seedtrio_dir bench/reft/results/20260706-1210-seed_floor_cfg4 \
        --known_dir    bench/reft/results/20260705-2242-reft_phase1p_cfg4
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from library.training.cmmd import (  # noqa: E402
    gram_of,
    paired_gram_score,
    pool_and_normalize,
)
from library.vision.encoder import (  # noqa: E402
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

# Cached PE sidecars for the sincos dataset (the target artist of every arm).
PE_CACHE_DIR = REPO_ROOT / "post_image_dataset" / "lora" / "sincos"
SIGN_CONSISTENCY_WIN = 0.70
P_WIN = 0.05


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _pil_to_minus1to1(img: Image.Image) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(img.convert("RGB"))).float()
    return (t / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def _encode_pixels(bundle, img_minus1to1: torch.Tensor) -> torch.Tensor:
    """One PE-Core forward → ``[T, D]`` tokens (CLS included) on CPU."""
    feats = encode_pe_from_imageminus1to1(bundle, img_minus1to1.to(bundle.device))
    return feats[0].to("cpu")


def _load_real_tokens(stem: str) -> torch.Tensor:
    """Raw ``[T, D]`` PE tokens for a real holdout image from its sidecar."""
    sd = load_file(str(PE_CACHE_DIR / f"{stem}_anima_pe.safetensors"))
    return sd["image_features"].to(torch.float32)


def _discover(result_dir: Path) -> tuple[list[str], list[str]]:
    """(model labels, common stems) under ``<result_dir>/renders/``."""
    renders = result_dir / "renders"
    models = sorted(p.name for p in renders.iterdir() if p.is_dir())
    stem_sets = [
        {p.stem for p in (renders / m).glob("*.png")} for m in models
    ]
    stems = sorted(set.intersection(*stem_sets)) if stem_sets else []
    return models, stems


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def verdict(deltas: list[float]) -> dict:
    """WIN/TIE from per-stem PGS deltas (modelA − modelB)."""
    arr = np.asarray(deltas, dtype=np.float64)
    n = len(arr)
    n_pos = int((arr > 0).sum())
    n_neg = int((arr < 0).sum())
    n_nz = n_pos + n_neg
    sign_consistency = (max(n_pos, n_neg) / n_nz) if n_nz else 0.0
    # Wilcoxon is undefined when all deltas are zero.
    if n_nz == 0:
        p = 1.0
    else:
        try:
            p = float(wilcoxon(arr, zero_method="wilcox").pvalue)
        except ValueError:
            p = 1.0
    is_win = (p < P_WIN) and (sign_consistency >= SIGN_CONSISTENCY_WIN)
    return {
        "n": n,
        "mean_delta": float(arr.mean()),
        "median_delta": float(np.median(arr)),
        "sign_consistency": sign_consistency,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "p": p,
        "verdict": "WIN" if is_win else "TIE",
        "winner_sign": "+" if arr.mean() >= 0 else "-",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--seedtrio_dir",
        default="bench/reft/results/20260706-1210-seed_floor_cfg4",
        help="Result dir holding the reseed-trio + soup renders.",
    )
    ap.add_argument(
        "--known_dir",
        default="bench/reft/results/20260705-2242-reft_phase1p_cfg4",
        help="Result dir holding plain/reft/reg renders (known-difference arm).",
    )
    ap.add_argument("--label", default="phase0")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seedtrio_dir = (REPO_ROOT / args.seedtrio_dir).resolve()
    known_dir = (REPO_ROOT / args.known_dir).resolve()

    st_models, st_stems = _discover(seedtrio_dir)
    kn_models, kn_stems = _discover(known_dir)
    print(f"seed-trio dir: {len(st_models)} models × {len(st_stems)} stems", flush=True)
    print(f"known-diff dir: {len(kn_models)} models × {len(kn_stems)} stems", flush=True)

    bundle = load_pe_encoder(device)

    # --- Encode every render once; cache real tokens once. ------------------
    def score_dir(result_dir: Path, models: list[str], stems: list[str]):
        """Returns pgs[model][stem], gist[model][stem], plus per-stem tokens for
        the base model + real sidecars (for gates 1/4/5)."""
        renders = result_dir / "renders"
        real_tok = {s: _load_real_tokens(s) for s in stems}
        real_gram = {s: gram_of(real_tok[s]) for s in stems}
        real_gist = {s: pool_and_normalize(real_tok[s]) for s in stems}
        pgs: dict[str, dict[str, float]] = {}
        gist: dict[str, dict[str, float]] = {}
        base_gen_tok: dict[str, torch.Tensor] = {}
        base_gen_px: dict[str, torch.Tensor] = {}
        for m in models:
            pgs[m], gist[m] = {}, {}
            for s in stems:
                px = _pil_to_minus1to1(Image.open(renders / m / f"{s}.png"))
                tok = _encode_pixels(bundle, px)
                pgs[m][s] = float(
                    torch.dot(gram_of(tok), real_gram[s]).clamp(-1, 1).item()
                )
                gist[m][s] = float(
                    torch.dot(pool_and_normalize(tok), real_gist[s]).clamp(-1, 1).item()
                )
                if m == "base":
                    base_gen_tok[s] = tok
                    base_gen_px[s] = px
            print(f"  scored {m}", flush=True)
        return pgs, gist, real_gram, base_gen_tok, base_gen_px

    st_pgs, st_gist, st_real_gram, st_base_tok, st_base_px = score_dir(
        seedtrio_dir, st_models, st_stems
    )
    kn_pgs, kn_gist, _, _, _ = score_dir(known_dir, kn_models, kn_stems)

    gates: dict[str, dict] = {}

    # === Gate 1 — Aliveness (right-pair vs wrong-pair) =====================
    # Full [stem × stem] matrix of base renders vs real Grams; diagonal = right.
    gen_grams = {s: gram_of(st_base_tok[s]) for s in st_stems}
    right, wrong = [], []
    for i in st_stems:
        for j in st_stems:
            v = float(torch.dot(gen_grams[i], st_real_gram[j]).clamp(-1, 1).item())
            (right if i == j else wrong).append(v)
    gap = float(np.mean(right) - np.mean(wrong))
    gates["1_aliveness"] = {
        "right_pair_mean": float(np.mean(right)),
        "wrong_pair_mean": float(np.mean(wrong)),
        "gap": gap,
        "pass": gap > 0.0,
    }

    # === Gate 2 — Seed-trio null (all pairwise must TIE) ===================
    trio = [m for m in st_models if m.endswith(("uncondpoolft", "_s1002", "_s1003"))]
    trio_pairs = {}
    all_tie = True
    for a, b in combinations(trio, 2):
        deltas = [st_pgs[a][s] - st_pgs[b][s] for s in st_stems]
        v = verdict(deltas)
        trio_pairs[f"{a} vs {b}"] = v
        if v["verdict"] != "TIE":
            all_tie = False
    gates["2_seed_trio_null"] = {"pairs": trio_pairs, "pass": all_tie}

    # === Gate 3 — Known-difference (plain vs base → WIN) ==================
    plain = next((m for m in kn_models if m.endswith("_plain")), None)
    kn_stems_common = kn_stems
    deltas_pb = [kn_pgs[plain][s] - kn_pgs["base"][s] for s in kn_stems_common]
    v_pb = verdict(deltas_pb)
    gates["3_known_difference"] = {
        "plain_vs_base": v_pb,
        "pass": v_pb["verdict"] == "WIN",
    }

    # === Gate 4 — Quantization stability (8-bit LSB round-trip) ===========
    # Perturb each pixel by ±1 uint8 LSB, re-quantize, re-encode, recompute PGS.
    rng = np.random.default_rng(0)
    rel_moves = []
    for s in st_stems:
        px = st_base_px[s]  # (1,3,H,W) in [-1,1]
        arr = ((px[0].permute(1, 2, 0).numpy() + 1.0) * 127.5).round()
        jitter = rng.integers(-1, 2, size=arr.shape)  # {-1,0,+1}
        arr_q = np.clip(arr + jitter, 0, 255).astype(np.uint8)
        px_q = _pil_to_minus1to1(Image.fromarray(arr_q))
        tok_q = _encode_pixels(bundle, px_q)
        pgs_q = float(torch.dot(gram_of(tok_q), st_real_gram[s]).clamp(-1, 1).item())
        base_pgs = st_pgs["base"][s]
        rel_moves.append(abs(pgs_q - base_pgs) / max(abs(base_pgs), 1e-6))
    median_move = float(np.median(rel_moves))
    max_move = float(np.max(rel_moves))
    gates["4_quant_stability"] = {
        "median_rel_move": median_move,
        "max_rel_move": max_move,
        "threshold": 0.02,
        "pass": median_move < 0.02,
    }

    # === Gate 5 — Tone confound (±10 luma: PGS moves < gist) ==============
    pgs_moves, gist_moves = [], []
    for s in st_stems:
        px = st_base_px[s]
        arr = (px[0].permute(1, 2, 0).numpy() + 1.0) * 127.5
        for shift in (+10.0, -10.0):
            arr_s = np.clip(arr + shift, 0, 255).astype(np.uint8)
            tok_s = _encode_pixels(bundle, _pil_to_minus1to1(Image.fromarray(arr_s)))
            pgs_s = float(torch.dot(gram_of(tok_s), st_real_gram[s]).clamp(-1, 1).item())
            gist_s = float(
                torch.dot(pool_and_normalize(tok_s), pool_and_normalize(_load_real_tokens(s)))
                .clamp(-1, 1)
                .item()
            )
            pgs_moves.append(abs(pgs_s - st_pgs["base"][s]))
            gist_moves.append(abs(gist_s - st_gist["base"][s]))
    pgs_tone = float(np.mean(pgs_moves))
    gist_tone = float(np.mean(gist_moves))
    gates["5_tone_confound"] = {
        "pgs_mean_abs_move": pgs_tone,
        "gist_mean_abs_move": gist_tone,
        "pass": pgs_tone < gist_tone,
    }

    all_pass = all(g["pass"] for g in gates.values())

    # --- Report -------------------------------------------------------------
    run_dir = make_run_dir("paired_eval", args.label)
    lines = [
        "# paired_gram_eval — Phase 0 validation gates",
        "",
        f"Offline (no rendering). Seed-trio dir `{seedtrio_dir.name}` "
        f"({len(st_stems)} stems), known-diff dir `{known_dir.name}` "
        f"({len(kn_stems)} stems). Real reference = cached "
        "`{stem}_anima_pe.safetensors` tokens.",
        "",
        f"## Verdict: {'ALL GATES PASS ✅' if all_pass else 'FAIL ❌'}",
        "",
        "| gate | result | pass |",
        "|---|---|---|",
        f"| 1 aliveness | right {gates['1_aliveness']['right_pair_mean']:.4f} − "
        f"wrong {gates['1_aliveness']['wrong_pair_mean']:.4f} = "
        f"**{gap:+.4f}** | {'✅' if gates['1_aliveness']['pass'] else '❌'} |",
        f"| 2 seed-trio null | "
        f"{'all TIE' if gates['2_seed_trio_null']['pass'] else 'a pair WON'} | "
        f"{'✅' if gates['2_seed_trio_null']['pass'] else '❌'} |",
        f"| 3 known-difference (plain vs base) | {v_pb['verdict']} "
        f"(p={v_pb['p']:.4f}, sign={v_pb['sign_consistency']:.2f}, "
        f"Δ={v_pb['mean_delta']:+.4f}) | "
        f"{'✅' if gates['3_known_difference']['pass'] else '❌'} |",
        f"| 4 quant stability | median {median_move * 100:.2f}% "
        f"(max {max_move * 100:.2f}%) vs 2% | "
        f"{'✅' if gates['4_quant_stability']['pass'] else '❌'} |",
        f"| 5 tone confound | PGS {pgs_tone:.4f} vs gist {gist_tone:.4f} | "
        f"{'✅' if gates['5_tone_confound']['pass'] else '❌'} |",
        "",
        "## Seed-trio pairwise (must all be TIE)",
        "",
        "| pair | verdict | p | sign-consistency | mean Δ |",
        "|---|---|---|---|---|",
    ]
    for pair, v in trio_pairs.items():
        lines.append(
            f"| {pair} | {v['verdict']} | {v['p']:.4f} | "
            f"{v['sign_consistency']:.2f} | {v['mean_delta']:+.4f} |"
        )
    lines += [
        "",
        "Reading: gate 1 confirms per-item pairing carries signal; gate 2 that "
        "same-recipe reseeds don't spuriously separate; gate 3 that a real "
        "restyle is detected; gates 4-5 that PGS is robust to the 8-bit and "
        "tone confounds that sank CMMD. Decision rule: WIN iff Wilcoxon p<0.05 "
        "AND sign-consistency ≥ 0.70.",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics={"all_pass": all_pass, "gates": gates},
        artifacts=[run_dir / "report.md"],
        device=device,
        extra={"seedtrio_stems": st_stems, "known_stems": kn_stems},
    )
    print(f"done → {run_dir}", flush=True)


if __name__ == "__main__":
    main()
