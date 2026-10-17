#!/usr/bin/env python3
"""Is it the Gram feature or the PAIRING that fails at CFG-4?

Gate 3 tested PGS = per-item *paired* Gram cosine and failed on CFG-4 renders,
while pooled CMMD (unpaired gist MMD²) ranked plain<base correctly on the SAME
renders. This isolates the two axes by testing the Gram feature in POOLED form:

  per-model mean-Gram → cosine to the real-holdout mean-Gram (unpaired,
  distribution-level), for both plain and whitened Grams, alongside the
  mean-gist baseline. If pooled Gram ranks plain>base (closer to artist),
  the Gram feature carries the style signal and the per-item PAIRING is what
  killed gate 3 — pooling (or larger n) could rescue it. If pooled Gram still
  ranks base≥plain, the Gram feature itself is style-blind → CSD.

Lower CMMD = closer; here HIGHER mean-Gram cosine = closer, so plain should
score ABOVE base if the feature works.
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

from library.training.cmmd import pool_and_normalize  # noqa: E402
from library.vision.encoder import (  # noqa: E402
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

CACHE = REPO_ROOT / "post_image_dataset" / "lora" / "sincos"
KN = REPO_ROOT / "bench/reft/results/20260705-2242-reft_phase1p_cfg4/renders"
MODELS = ["base", "bench_sincos_half_plain", "bench_sincos_half_reft64",
          "bench_sincos_half_reg36", "bench_sincos_half_reg36_reft64"]


def grams(feats: torch.Tensor):
    """(plain, whitened) flat unit-Frobenius Gram vectors from [T,D] tokens."""
    f = feats.to(torch.float32)
    if f.shape[0] > 1:
        f = f[1:]
    f = f - f.mean(0, keepdim=True)
    t = f.shape[0]
    pl = (f.t() @ f) / t
    pl = (pl / pl.norm().clamp_min(1e-12)).reshape(-1)
    fw = f / f.std(0, keepdim=True).clamp_min(1e-6)
    wh = (fw.t() @ fw) / t
    wh = (wh / wh.norm().clamp_min(1e-12)).reshape(-1)
    return pl, wh


def _pil(img):
    a = torch.from_numpy(np.asarray(Image.open(img).convert("RGB"))).float()
    return (a / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_pe_encoder(device)
    rng = Random(0)

    render_stems = sorted(p.stem for p in (KN / "base").glob("*.png"))

    def enc_render(model):
        pls, whs, gis = [], [], []
        for s in render_stems:
            f = encode_pe_from_imageminus1to1(bundle, _pil(KN / model / f"{s}.png").to(device))[0].cpu()
            pl, wh = grams(f)
            pls.append(pl); whs.append(wh); gis.append(pool_and_normalize(f))
        return torch.stack(pls), torch.stack(whs), torch.stack(gis)

    # Real holdout pool: sincos sidecars NOT among the render stems.
    excl = set(render_stems)
    real_scs = [sc for sc in CACHE.glob("*_anima_pe.safetensors")
                if sc.name.split("_anima_pe")[0] not in excl]
    real_scs = rng.sample(real_scs, min(96, len(real_scs)))
    rpl, rwh, rgi = [], [], []
    for sc in real_scs:
        f = load_file(str(sc))["image_features"]
        pl, wh = grams(f)
        rpl.append(pl); rwh.append(wh); rgi.append(pool_and_normalize(f))
    real_pl, real_wh, real_gi = torch.stack(rpl).mean(0), torch.stack(rwh).mean(0), torch.stack(rgi).mean(0)
    print(f"{len(render_stems)} render stems/model · {len(real_scs)} real-holdout refs", flush=True)

    def mc(pool_mean, ref_mean):
        a = pool_mean / pool_mean.norm(); b = ref_mean / ref_mean.norm()
        return float(torch.dot(a, b))

    print("\nmodel                              gist↑    gram↑    whitened-gram↑", flush=True)
    rows = {}
    for m in MODELS:
        pls, whs, gis = enc_render(m)
        rows[m] = (mc(gis.mean(0), real_gi), mc(pls.mean(0), real_pl), mc(whs.mean(0), real_wh))
        print(f"  {m:32s} {rows[m][0]:.4f}  {rows[m][1]:.4f}  {rows[m][2]:.4f}", flush=True)

    print("\n=== plain vs base (higher = closer to artist; want plain > base) ===", flush=True)
    for i, name in ((0, "gist"), (1, "gram"), (2, "whitened-gram")):
        pv, bv = rows["bench_sincos_half_plain"][i], rows["base"][i]
        print(f"  {name:14s} plain {pv:.4f}  base {bv:.4f}  Δ={pv - bv:+.4f}  "
              f"{'plain closer ✅' if pv > bv else 'base closer ❌'}", flush=True)


if __name__ == "__main__":
    main()
