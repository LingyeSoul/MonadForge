#!/usr/bin/env python3
"""Build the REPA global-anchor calibration (patch-mean μ/σ).

Companion to ``discriminability.py`` (Phase 0). That probe established the live
target for the global-anchor arm: the **pooled patch-mean** of the PE-Spatial
features (drop CLS), normalized by a **fixed per-dim z-score** affine fit on the
dataset (``project_pe_cls_collapse_patchmean`` — patch-mean+zscore won the AUC
table, CLS is collapsed). This script computes that affine once and ships it as
``networks/calibration/pe_patchmean_stats.safetensors`` (mirrors the
channel-scaling / cond-stream calib pattern).

What it computes, over the cached ``{stem}_anima_{encoder}.safetensors``
sidecars: for each image the patch-mean vector ``feats[1:].mean(0)`` (CLS at
index 0 dropped), then the **per-dim mean and std across images** — exactly the
``zscore`` fit in ``discriminability.py::_normalize`` (``mu = x.mean(0)``,
``sd = x.std(0)``). Both ``REPAMethodAdapter``'s DiT-side head output and the PE
target are standardized by this same affine before L2-norm + cosine, so the two
sides live in one comparable space.

No model forward, no checkpoint — purely a property of the cached features.

Run from anima_lora/::

    uv run python bench/pe_cls_probe/build_calib.py \
        --data_dir post_image_dataset/lora \
        --out networks/calibration/pe_patchmean_stats.safetensors
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("pe_patchmean_calib")


def _patch_mean(path: Path) -> np.ndarray | None:
    """Return the CLS-dropped patch-mean vector (D,) fp32, or None if unreadable.

    PE sidecars store ``image_features`` of shape ``[T, D]`` with CLS at index 0
    (pe_spatial always uses a CLS); confirmed against grid metadata when present
    (``T == gh*gw + 1``).
    """
    try:
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            feats_t = f.get_tensor("image_features")
    except Exception:  # noqa: BLE001 — a corrupt sidecar shouldn't kill the run
        return None
    feats = feats_t.float().cpu().numpy().astype(np.float32)
    if feats.ndim != 2 or feats.shape[0] < 2:
        return None
    cls_present = True
    gh, gw = meta.get("grid_h"), meta.get("grid_w")
    if gh is not None and gw is not None:
        cls_present = feats.shape[0] - int(gh) * int(gw) == 1
    return feats[1:].mean(axis=0) if cls_present else feats.mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data_dir", default="post_image_dataset/lora")
    ap.add_argument("--encoder", default="pe_spatial")
    ap.add_argument(
        "--out", default="networks/calibration/pe_patchmean_stats.safetensors"
    )
    ap.add_argument("--num_samples", type=int, default=0, help="0 = all images")
    ap.add_argument("--eps", type=float, default=1e-8)
    args = ap.parse_args()

    data_dir = (REPO_ROOT / args.data_dir).resolve()
    suffix = f"_anima_{args.encoder}.safetensors"
    files = sorted(p for p in data_dir.rglob(f"*{suffix}") if p.is_file())
    if args.num_samples and len(files) > args.num_samples:
        files = files[: args.num_samples]
    if not files:
        log.error("No %s sidecars under %s", args.encoder, data_dir)
        sys.exit(1)

    rows: list[np.ndarray] = []
    for p in files:
        pm = _patch_mean(p)
        if pm is not None:
            rows.append(pm)
    if len(rows) < 50:
        log.error("Only %d readable sidecars — too few to calibrate.", len(rows))
        sys.exit(1)

    x = np.stack(rows).astype(np.float64)  # (N, D)
    mu = x.mean(0).astype(np.float32)
    sd = (x.std(0) + args.eps).astype(np.float32)
    log.info(
        "Patch-mean calib over %d images (D=%d): ‖μ‖=%.4f, σ∈[%.4f, %.4f] (mean %.4f)",
        len(rows),
        x.shape[1],
        float(np.linalg.norm(mu)),
        float(sd.min()),
        float(sd.max()),
        float(sd.mean()),
    )

    out = (REPO_ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"mean": torch.from_numpy(mu), "std": torch.from_numpy(sd)},
        str(out),
        metadata={
            "encoder": args.encoder,
            "n_images": str(len(rows)),
            "source": "patch-mean (CLS dropped), per-dim z-score affine",
            "scheme": "zscore",
        },
    )
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
