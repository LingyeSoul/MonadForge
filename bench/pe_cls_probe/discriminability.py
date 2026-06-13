#!/usr/bin/env python3
"""PE global-token discriminability probe (Phase 0 of the CLS-alignment arm).

THE QUESTION. The validated REPA arm aligns block-8 features to PE-Spatial
*patch* tokens and deliberately drops the CLS (and ``spatial_norm`` further
standardizes away the global component). The REG paper (NeurIPS 2025,
arXiv-companion in repo root) argues the *high-level class/global* token is
where the gain lives. Before spending a training run on a CLS-alignment arm we
need to know: is the PE CLS token actually a *discriminative* target on this
dataset, or is it collapsed? We already observed near-twin pairs at CLS cosine
~0.99 — which is exactly the saturation that would make a raw-CLS target a
near-constant (and therefore useless) alignment objective.

So this probe is purely a property of the cached PE features — **no model
forward, no checkpoint**. It measures, over cached ``{stem}_anima_{encoder}``
sidecars, whether the global token separates images that *should* be different
(different character / copyright / artist per ``caption_index.json``) from
images that share those tags, and whether a normalization recovers
discriminability that the raw cosine hides.

WHAT IT COMPUTES.

* **Collapse check (raw).** Nearest-neighbour cosine distribution + the
  fraction of total variance carried by the top principal direction. A single
  dominant shared direction is the mechanism behind the ~0.99 floor — and the
  thing mean-centering / whitening removes.

* **Discriminability** under four normalizations fit on the sampled set —
  ``raw`` / ``center`` (subtract dataset mean) / ``zscore`` (per-dim
  standardize) / ``whiten`` (ZCA) — scored on labeled pairs drawn from
  ``caption_index.json`` groups, per axis (character / copyright / artist):

    - d-prime ``(mean_in − mean_out) / sqrt(½(var_in+var_out))``
    - AUC (Mann-Whitney): P(in-group cosine > out-group cosine)

  AUC≈0.5 / d-prime≈0 ⇒ the target can't tell same-from-different at all
  (dead target). Bigger ⇒ usable signal; the scheme that maximizes it is the
  normalization the arm should apply to its target.

* **CLS vs pooled-patch.** The same pipeline on the mean of the patch tokens,
  so we can see whether the *global anchor* should even be the CLS token or
  just a pooled-patch vector (which the relational arm already has on disk).

PRE-REGISTERED READOUT (the gate for implementing the arm):

* best-scheme AUC ≥ --gate_auc (default 0.65) on at least the character axis
  ⇒ a normalized global target carries real signal → build the arm using the
  winning normalization;
* every scheme ≈ 0.5 ⇒ CLOSE the line without training: the PE global token is
  collapsed beyond recovery on this data and the spatial arm already owns the
  usable structure.

Run from anima_lora/::

    uv run python bench/pe_cls_probe/discriminability.py \
        --data_dir post_image_dataset/lora \
        --index post_image_dataset/captions/caption_index.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from safetensors import safe_open  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("pe_cls_probe")

AXES = ("character", "copyright", "artist")
SCHEMES = ("raw", "center", "zscore", "whiten")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _load_global_and_patchmean(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (cls_vec, patch_mean_vec) in fp32, or None if unreadable.

    PE sidecars store one key ``image_features`` of shape ``[T, D]`` with the
    CLS at index 0 when the encoder uses it (pe_spatial always does). CLS
    presence is confirmed from grid metadata when available: ``T == gh*gw + 1``.
    """
    try:
        # framework="pt": the sidecars are bf16 on disk, which the numpy
        # backend can't read; torch loads it, then we upcast to fp32 numpy.
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            feats_t = f.get_tensor("image_features")
    except Exception:  # noqa: BLE001 — a corrupt sidecar shouldn't kill the run
        return None
    feats = feats_t.float().cpu().numpy().astype(np.float32)
    if feats.ndim != 2 or feats.shape[0] < 2:
        return None
    t = feats.shape[0]
    cls_present = True
    gh, gw = meta.get("grid_h"), meta.get("grid_w")
    if gh is not None and gw is not None:
        cls_present = t - int(gh) * int(gw) == 1
    if cls_present:
        return feats[0].copy(), feats[1:].mean(axis=0)
    # No CLS row — fall back to pooled patches for both (CLS test degenerates).
    pm = feats.mean(axis=0)
    return pm, pm


def _discover(data_dir: Path, encoder: str, cap: int) -> list[tuple[str, Path]]:
    suffix = f"_anima_{encoder}.safetensors"
    files = sorted(p for p in data_dir.rglob(f"*{suffix}") if p.is_file())
    out = [(p.name[: -len(suffix)], p) for p in files]
    return out[:cap] if cap and len(out) > cap else out


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #
def _normalize(x: np.ndarray, scheme: str) -> np.ndarray:
    """Fit the scheme's stats on ``x`` (N,D), return L2-normalized rows."""
    x = x.astype(np.float64)
    if scheme == "raw":
        y = x
    elif scheme == "center":
        y = x - x.mean(0, keepdims=True)
    elif scheme == "zscore":
        mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
        y = (x - mu) / (sd + 1e-8)
    elif scheme == "whiten":  # ZCA: decorrelate + unit per-component variance
        mu = x.mean(0, keepdims=True)
        xc = x - mu
        cov = (xc.T @ xc) / max(1, xc.shape[0] - 1)
        evals, evecs = np.linalg.eigh(cov)
        w = evecs @ np.diag(1.0 / np.sqrt(np.clip(evals, 1e-8, None))) @ evecs.T
        y = xc @ w
    else:
        raise ValueError(scheme)
    n = np.linalg.norm(y, axis=1, keepdims=True)
    return (y / np.clip(n, 1e-12, None)).astype(np.float64)


def _top_pc_explained(x: np.ndarray, k: int = 5) -> list[float]:
    xc = x.astype(np.float64) - x.mean(0, keepdims=True)
    cov = (xc.T @ xc) / max(1, xc.shape[0] - 1)
    evals = np.linalg.eigvalsh(cov)[::-1]
    tot = float(evals.sum()) or 1.0
    return [float(v / tot) for v in evals[:k]]


# --------------------------------------------------------------------------- #
# labeled pairs from caption_index
# --------------------------------------------------------------------------- #
def _pair_indices(
    stems: list[str],
    image_meta: dict,
    axis: str,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (in_group, out_group) index-pair arrays for one axis.

    in-group = share ≥1 tag on this axis; out-group = share none (both must
    have at least one tag on the axis, so "untagged" images aren't scored).
    """
    idx_of = {s: i for i, s in enumerate(stems)}
    tagged = [
        s for s in stems if image_meta.get(s, {}).get(axis)
    ]  # only images with a tag on this axis
    tagsets = {s: frozenset(image_meta[s][axis]) for s in tagged}
    by_tag: dict[str, list[str]] = {}
    for s in tagged:
        for t in tagsets[s]:
            by_tag.setdefault(t, []).append(s)

    in_pairs: list[tuple[int, int]] = []
    groups = [g for g in by_tag.values() if len(g) >= 2]
    if groups:
        for _ in range(n_pairs * 4):
            if len(in_pairs) >= n_pairs:
                break
            g = groups[rng.integers(len(groups))]
            a, b = rng.integers(len(g)), rng.integers(len(g))
            if a != b:
                in_pairs.append((idx_of[g[a]], idx_of[g[b]]))

    out_pairs: list[tuple[int, int]] = []
    if len(tagged) >= 2:
        for _ in range(n_pairs * 8):
            if len(out_pairs) >= n_pairs:
                break
            a, b = tagged[rng.integers(len(tagged))], tagged[rng.integers(len(tagged))]
            if a != b and tagsets[a].isdisjoint(tagsets[b]):
                out_pairs.append((idx_of[a], idx_of[b]))

    return np.array(in_pairs, dtype=np.int64), np.array(out_pairs, dtype=np.int64)


def _cos_pairs(unit: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.empty(0)
    return np.sum(unit[pairs[:, 0]] * unit[pairs[:, 1]], axis=1)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) via Mann-Whitney U on the rank transform."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    # tie-averaged ranks (Mann-Whitney with ties)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    avg = np.empty(counts.size)
    start = 0
    for i, c in enumerate(counts):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = avg[inv]
    r_pos = ranks[: pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _score(unit: np.ndarray, in_p: np.ndarray, out_p: np.ndarray) -> dict:
    si, so = _cos_pairs(unit, in_p), _cos_pairs(unit, out_p)
    if si.size == 0 or so.size == 0:
        return {
            "dprime": float("nan"),
            "auc": float("nan"),
            "n_in": int(si.size),
            "n_out": int(so.size),
        }
    pooled = np.sqrt(0.5 * (si.var() + so.var())) or 1.0
    return {
        "dprime": float((si.mean() - so.mean()) / pooled),
        "auc": _auc(si, so),
        "mean_in": float(si.mean()),
        "mean_out": float(so.mean()),
        "std_in": float(si.std()),
        "std_out": float(so.std()),
        "n_in": int(si.size),
        "n_out": int(so.size),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data_dir", default="post_image_dataset/lora")
    ap.add_argument("--index", default="post_image_dataset/captions/caption_index.json")
    ap.add_argument("--encoder", default="pe_spatial")
    ap.add_argument(
        "--num_samples", type=int, default=4000, help="cap on images loaded"
    )
    ap.add_argument(
        "--num_pairs", type=int, default=20000, help="pairs per axis per class"
    )
    ap.add_argument("--gate_auc", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    import json

    rng = np.random.default_rng(args.seed)
    data_dir = (REPO_ROOT / args.data_dir).resolve()
    found = _discover(data_dir, args.encoder, args.num_samples)
    if not found:
        log.error("No %s sidecars under %s", args.encoder, data_dir)
        sys.exit(1)

    index = json.loads((REPO_ROOT / args.index).read_text(encoding="utf-8"))
    image_meta = index.get("image_meta", {})

    stems: list[str] = []
    cls_rows: list[np.ndarray] = []
    pm_rows: list[np.ndarray] = []
    for stem, path in found:
        if stem not in image_meta:  # need labels to score it
            continue
        loaded = _load_global_and_patchmean(path)
        if loaded is None:
            continue
        cls_v, pm_v = loaded
        stems.append(stem)
        cls_rows.append(cls_v)
        pm_rows.append(pm_v)

    if len(stems) < 50:
        log.error("Only %d labeled+cached images — too few to score.", len(stems))
        sys.exit(1)

    cls = np.stack(cls_rows)
    pm = np.stack(pm_rows)
    log.info(
        "Loaded %d labeled images (encoder=%s, D=%d)",
        len(stems),
        args.encoder,
        cls.shape[1],
    )

    # collapse diagnostics on raw CLS
    cls_unit_raw = cls / np.clip(
        np.linalg.norm(cls, axis=1, keepdims=True), 1e-12, None
    )
    sim_full = cls_unit_raw @ cls_unit_raw.T
    np.fill_diagonal(sim_full, -np.inf)
    nn_sim = sim_full.max(axis=1)
    collapse = {
        "nn_sim_raw_mean": float(nn_sim.mean()),
        "nn_sim_raw_p95": float(np.percentile(nn_sim, 95)),
        "top5_pc_explained_var": _top_pc_explained(cls),
    }
    log.info(
        "RAW CLS collapse: nearest-neighbour cosine mean=%.4f p95=%.4f | top-PC var=%.3f",
        collapse["nn_sim_raw_mean"],
        collapse["nn_sim_raw_p95"],
        collapse["top5_pc_explained_var"][0],
    )

    # labeled pairs per axis (shared across schemes/targets for a fair compare)
    pairs = {
        ax: _pair_indices(stems, image_meta, ax, args.num_pairs, rng) for ax in AXES
    }

    targets = {"cls": cls, "patchmean": pm}
    schemes_out: dict = {}
    rows: list[str] = ["target,scheme,axis,dprime,auc,mean_in,mean_out,n_in,n_out"]
    best_auc_char = 0.0
    best_combo = ("", "")
    for tname, mat in targets.items():
        schemes_out[tname] = {}
        for scheme in SCHEMES:
            unit = _normalize(mat, scheme)
            schemes_out[tname][scheme] = {}
            for ax in AXES:
                in_p, out_p = pairs[ax]
                s = _score(unit, in_p, out_p)
                schemes_out[tname][scheme][ax] = s
                rows.append(
                    f"{tname},{scheme},{ax},{s.get('dprime', float('nan')):.4f},"
                    f"{s.get('auc', float('nan')):.4f},{s.get('mean_in', float('nan')):.4f},"
                    f"{s.get('mean_out', float('nan')):.4f},{s['n_in']},{s['n_out']}"
                )
                if (
                    tname == "cls"
                    and ax == "character"
                    and np.isfinite(s.get("auc", np.nan))
                ):
                    if s["auc"] > best_auc_char:
                        best_auc_char = s["auc"]
                        best_combo = (tname, scheme)

    passed = best_auc_char >= args.gate_auc
    verdict = (
        f"PASS — best CLS character AUC={best_auc_char:.3f} "
        f"(scheme={best_combo[1]}) ≥ gate {args.gate_auc}: build the arm with this normalization."
        if passed
        else f"FAIL — best CLS character AUC={best_auc_char:.3f} < gate {args.gate_auc}: "
        f"global token collapsed beyond recovery; close the line."
    )
    log.info(verdict)

    run_dir = make_run_dir("pe_cls_probe", label=args.label or "discriminability")
    (run_dir / "scores.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # readable summary
    md = [
        "# PE global-token discriminability (Phase 0)",
        "",
        f"- images: {len(stems)}",
        f"- encoder: {args.encoder} (D={cls.shape[1]})",
        f"- raw NN cosine mean: {collapse['nn_sim_raw_mean']:.4f}",
        f"- top-PC explained var: {collapse['top5_pc_explained_var'][0]:.3f}",
        "",
        "## AUC (P[in-group cosine > out-group]) — 0.5 = dead target",
        "",
    ]
    for tname in targets:
        md.append(f"### target = {tname}")
        md.append("| scheme | " + " | ".join(AXES) + " |")
        md.append("|" + "---|" * (len(AXES) + 1))
        for scheme in SCHEMES:
            cells = [
                f"{schemes_out[tname][scheme][ax].get('auc', float('nan')):.3f}"
                for ax in AXES
            ]
            md.append(f"| {scheme} | " + " | ".join(cells) + " |")
        md.append("")
    md.append(f"**Verdict:** {verdict}")
    (run_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "n_images": len(stems),
            "d_enc": int(cls.shape[1]),
            "collapse": collapse,
            "schemes": schemes_out,
            "best_cls_character_auc": best_auc_char,
            "best_combo": {"target": best_combo[0], "scheme": best_combo[1]},
            "gate_auc": args.gate_auc,
            "passed": bool(passed),
        },
        label=args.label,
        artifacts=["scores.csv", "summary.md"],
    )
    log.info("Wrote %s", run_dir)


if __name__ == "__main__":
    main()
