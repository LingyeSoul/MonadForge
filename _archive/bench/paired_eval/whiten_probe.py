#!/usr/bin/env python3
"""Phase-0 remedy test: does a WHITENED Gram sharpen the artist-style channel?

Plain PGS failed gate 3 and the cross-artist probe showed its artist-style
signal (~0.025 at matched content) is ~6x weaker than its content/palette
signal (~0.14). The proposal's escalation note (gate 5) says: "center per-
channel AND whiten before the Gram, re-gate; if it still fails, escalate to
CSD." This tests the whitened variant on the two discriminating axes:

  * gate-3 axis: plain-r16 vs base, paired per-stem PGS deltas (want plain WIN)
  * artist axis: base sincos renders, same-artist vs cross-artist PGS

"correlation Gram" = standardize each channel to unit std over tokens before
the second moment, so the dominant-variance channels (tone/palette) stop
dominating the Frobenius norm.
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

from library.vision.encoder import (  # noqa: E402
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

CACHE_ROOT = REPO_ROOT / "post_image_dataset" / "lora"
KN = REPO_ROOT / "bench/reft/results/20260705-2242-reft_phase1p_cfg4/renders"
OTHER_ARTISTS = ["suujiniku", "hews", "ama_mitsuki", "kat_(bu-kunn)", "mignon"]


def gram_variants(feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(plain Gram, whitened/correlation Gram) as flat unit-Frobenius vectors."""
    feats = feats.to(torch.float32)
    if feats.shape[0] > 1:
        feats = feats[1:]
    feats = feats - feats.mean(dim=0, keepdim=True)
    t = feats.shape[0]
    plain = (feats.t() @ feats) / t
    plain = plain / plain.norm().clamp_min(1e-12)
    # Correlation Gram: standardize channels to unit variance first.
    std = feats.std(dim=0, keepdim=True).clamp_min(1e-6)
    fw = feats / std
    whit = (fw.t() @ fw) / t
    whit = whit / whit.norm().clamp_min(1e-12)
    return plain.reshape(-1), whit.reshape(-1)


def _pil_to_minus1to1(img: Image.Image) -> torch.Tensor:
    a = torch.from_numpy(np.asarray(img.convert("RGB"))).float()
    return (a / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)


def cos(a, b):
    return float(torch.dot(a, b).clamp(-1, 1))


def verdict(deltas):
    d = np.asarray(deltas)
    nz = int((d != 0).sum())
    sign = max((d > 0).sum(), (d < 0).sum()) / nz if nz else 0.0
    p = float(wilcoxon(d).pvalue) if nz else 1.0
    return d.mean(), sign, p, ("WIN" if p < 0.05 and sign >= 0.7 else "TIE")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_pe_encoder(device)
    rng = Random(0)

    def enc(p):
        f = encode_pe_from_imageminus1to1(bundle, _pil_to_minus1to1(Image.open(p)).to(device))
        return gram_variants(f[0].to("cpu"))

    stems = sorted(p.stem for p in (KN / "base").glob("*.png"))
    real, base, plain = {}, {}, {}
    for s in stems:
        sc = CACHE_ROOT / "sincos" / f"{s}_anima_pe.safetensors"
        real[s] = gram_variants(load_file(str(sc))["image_features"])
        base[s] = enc(KN / "base" / f"{s}.png")
        plain[s] = enc(KN / "bench_sincos_half_plain" / f"{s}.png")
    print(f"{len(stems)} stems encoded (base+plain+real)", flush=True)

    cross = []
    for art in OTHER_ARTISTS:
        cands = list((CACHE_ROOT / art).glob("*_anima_pe.safetensors"))
        for sc in rng.sample(cands, min(8, len(cands))):
            cross.append(gram_variants(load_file(str(sc))["image_features"]))

    for gi, name in ((0, "plain-Gram"), (1, "whitened-Gram")):
        # gate 3: plain vs base
        d3 = [cos(plain[s][gi], real[s][gi]) - cos(base[s][gi], real[s][gi]) for s in stems]
        m3, sg3, p3, v3 = verdict(d3)
        # artist axis: same (base non-paired) vs cross
        same, crs = [], []
        for i, s in enumerate(stems):
            others = [stems[j] for j in range(len(stems)) if j != i]
            same.append(np.mean([cos(base[s][gi], real[o][gi]) for o in others]))
            crs.append(np.mean([cos(base[s][gi], c[gi]) for c in cross]))
        same, crs = np.asarray(same), np.asarray(crs)
        art_gap = float(same.mean() - crs.mean())
        print(f"\n=== {name} ===")
        print(f"  gate3 plain vs base: Δ={m3:+.4f} sign={sg3:.2f} p={p3:.4f} → {v3}")
        print(f"  artist gap (same-artist non-paired {same.mean():.4f} − "
              f"cross {crs.mean():.4f}) = {art_gap:+.4f}")


if __name__ == "__main__":
    main()
