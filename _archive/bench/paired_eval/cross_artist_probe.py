#!/usr/bin/env python3
"""Disambiguator for Phase-0 gate 3: does PGS discriminate the ARTIST axis?

Gate 3 (plain-r16 vs base) failed, but the montages show that restyle is
subtle at CFG-4/paired-seed, so a TIE is defensible — it may be a weak positive
control rather than a metric failure. Gate-1 aliveness is within-artist, so it
only proves PGS is content/palette-sensitive, not artist-style-sensitive.

This test isolates the artist axis: score base's sincos renders' Grams against
(a) their PAIRED sincos real and (b) reals from OTHER artists. If
PGS(same-artist) >> PGS(cross-artist), PGS encodes artist style and gate 3 is a
weak-control artifact. If PGS(same) ≈ PGS(cross), the Gram is content-only and
we escalate to CSD.
"""

from __future__ import annotations

import sys
from pathlib import Path
from random import Random

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from library.training.cmmd import gram_of  # noqa: E402
from library.vision.encoder import (  # noqa: E402
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

CACHE_ROOT = REPO_ROOT / "post_image_dataset" / "lora"
BASE_RENDERS = (
    REPO_ROOT
    / "bench/reft/results/20260705-2242-reft_phase1p_cfg4/renders/base"
)
OTHER_ARTISTS = ["suujiniku", "hews", "ama_mitsuki", "kat_(bu-kunn)", "mignon"]


def _pil_to_minus1to1(img: Image.Image) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(img.convert("RGB"))).float()
    return (t / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)


def _real_gram(sidecar: Path) -> torch.Tensor:
    return gram_of(load_file(str(sidecar))["image_features"].to(torch.float32))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = Random(0)
    bundle = load_pe_encoder(device)

    # Base renders of sincos captions → gen Grams.
    render_paths = sorted(BASE_RENDERS.glob("*.png"))
    gen_gram, same_real_gram = {}, {}
    for p in render_paths:
        stem = p.stem
        sc = CACHE_ROOT / "sincos" / f"{stem}_anima_pe.safetensors"
        if not sc.exists():
            continue
        feats = encode_pe_from_imageminus1to1(bundle, _pil_to_minus1to1(Image.open(p)).to(device))
        gen_gram[stem] = gram_of(feats[0].to("cpu"))
        same_real_gram[stem] = _real_gram(sc)
    stems = sorted(gen_gram)
    print(f"{len(stems)} base sincos renders", flush=True)

    # Cross-artist real Gram pool.
    cross_grams = []
    for art in OTHER_ARTISTS:
        cands = list((CACHE_ROOT / art).glob("*_anima_pe.safetensors"))
        for sc in rng.sample(cands, min(8, len(cands))):
            cross_grams.append(_real_gram(sc))
    print(f"{len(cross_grams)} cross-artist reals from {len(OTHER_ARTISTS)} artists", flush=True)

    same_pgs, cross_pgs = [], []
    for s in stems:
        g = gen_gram[s]
        same_pgs.append(float(torch.dot(g, same_real_gram[s]).clamp(-1, 1)))
        cross_pgs.append(float(np.mean([float(torch.dot(g, cg).clamp(-1, 1)) for cg in cross_grams])))

    same = np.asarray(same_pgs)
    cross = np.asarray(cross_pgs)
    deltas = same - cross
    n_pos = int((deltas > 0).sum())
    p = float(wilcoxon(deltas).pvalue) if len(deltas) else 1.0
    print("\n=== Cross-artist discrimination (base sincos renders) ===", flush=True)
    print(f"  PGS same-artist (paired sincos real):  mean {same.mean():.4f}", flush=True)
    print(f"  PGS cross-artist (other artists' pool): mean {cross.mean():.4f}", flush=True)
    print(f"  Δ (same − cross): mean {deltas.mean():+.4f}  "
          f"sign {n_pos}/{len(deltas)}  Wilcoxon p={p:.5f}", flush=True)
    verdict = "DISCRIMINATES artist" if (p < 0.05 and n_pos / len(deltas) >= 0.7) else "does NOT discriminate"
    print(f"  → PGS {verdict}", flush=True)


if __name__ == "__main__":
    main()
