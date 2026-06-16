#!/usr/bin/env python
"""REPA-DoG Phase 0 — data-only σ-axis probe on the PE-Spatial target.

THE QUESTION (``docs/proposal/repa_dog_target.md``). The shipped relational
REPA arm matches a *Gram affinity* between block-8 DiT tokens and PE-Spatial
patch tokens, with ``repa_spatial_norm`` standardizing the target tokens
``(pe − mean_tok)/(std_tok + ε)`` first — i.e. **DC removal**. Two threads
say the low-frequency / global band of the target is a distractor: our own
refuted global-anchor A/B (re-adding the DC band *hurt* CMMD) and the
Spectrum-Matching paper (arXiv:2603.14645v1 §3.5), which generalizes iREPA's
DC removal to a **Difference-of-Gaussians band-pass** and reports
REPA-DoG > iREPA > REPA on ImageNet/SiT.

So before spending any training compute, this probe sweeps the *low-freq-strip
axis* on the **cached PE features alone** (no DiT, no text encoder) and asks:
does stripping a **broader low band than DC** make the target *more
discriminative* of content identity?

THE AXIS (four points, each a spatial filter applied to the per-channel PE
feature map ``(d, gh, gw)`` before the standardize + per-token L2-norm + Gram):

    point  filter H(Z)                         role
    -----  ---------------------------------   --------------------------------
     −1    Z                 (keep DC)         refuted endpoint — should be worst
      0    Z − mean_spatial  (DC removal)      shipped ``spatial_norm`` baseline
    +1a    Z − LP(Z, σ_lp)   (high-pass)       strip a broader low band than DC
    +1b    LP(Z, σ_in)−LP(Z, σ_lp) (band-pass) +1a + high-freq rolloff (σ₂ risk)

After the filter we standardize ``H / (std_spatial(H) + ε)`` (held identical
across points so only the low-freq strip varies — for point 0 this reduces
*exactly* to ``relational_gram_loss``'s ``spatial_norm``), per-token L2-norm
over ``d`` (the relational arm's normalization), then summarize each image by
the **second-order Gram kernel** of its token-direction cloud: a fixed-size,
``N``-invariant random-projected feature-gram whose cosine is the
``<M_i, M_j>_F`` overlap of the two images' per-token L2-normalized fields.
That descriptor *is* the Gram affinity the relational arm matches, made
comparable across aspect buckets (which have different token counts ``N``).

READOUT = **target discriminability**, reusing the global-anchor probe's metric
(``bench/pe_cls_probe/discriminability.py``): AUC of
``P(in-group cosine > out-group cosine)`` over character / copyright / artist
pairs from ``caption_index.json``. AUC≈0.5 ⇒ dead target.

GATE (pre-registered, ``docs/proposal/repa_dog_target.md`` §"Phase 0"):

* AUC rises monotonically −1 → 0 → +1a ⇒ low-side DoG is worth training; leave
  σ₂ off (train ``repa_target_dog`` with the winning ``σ_lp``).
* AUC flattens after point 0 (``+1a ≈ 0``) ⇒ DoG is **irrelevant, not harmful**
  — the DC removal we ship already owns the usable contrast; close the line
  cheaply.
* +1b < +1a ⇒ the high-freq rolloff hurts; pin σ₂→∞ permanently.

Operator + sweep recipe borrowed from ``bench/fera_artist/probe_fei_artist.py``
(``σ = min(gh, gw) / div``, bucket-invariant); metric machinery imported from
``bench/pe_cls_probe/discriminability.py`` so it can't drift from the shipped
probe. NB σ here is calibrated on the **coarse PE grid** (~28–46 patches/side),
not the ~64-patch latent grid FEI tunes — its own sweep.

Run from anima_lora/::

    uv run python bench/repa/probe_dog_target.py \\
        --lp_divs 4,8,16,32 --num_samples 3000 --label dog-axis
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from bench.pe_cls_probe.discriminability import (  # noqa: E402
    AXES,
    _auc,
    _discover,
    _pair_indices,
)
from library.runtime.fei import gaussian_blur_2d  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("repa-dog-probe")


# --------------------------------------------------------------------------- #
# loading — PE patch-token grid (CLS dropped) from a cached sidecar
# --------------------------------------------------------------------------- #
def _load_patch_grid(path: Path) -> tuple[np.ndarray, int, int] | None:
    """Return ``(patches[N, d] fp32, gh, gw)`` or ``None`` if unreadable.

    PE sidecars store ``image_features`` ``[T, d]`` (bf16) with the CLS at
    index 0 and ``grid_h``/``grid_w`` in metadata. We need the *patch* field
    on its ``(gh, gw)`` grid, so drop the CLS row when ``T == gh*gw + 1``.
    """
    try:
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
            feats_t = f.get_tensor("image_features")
    except Exception:  # noqa: BLE001 — a corrupt sidecar shouldn't kill the run
        return None
    gh, gw = meta.get("grid_h"), meta.get("grid_w")
    if gh is None or gw is None:
        return None
    gh, gw = int(gh), int(gw)
    feats = feats_t.float().cpu().numpy().astype(np.float32)
    if feats.ndim != 2:
        return None
    t = feats.shape[0]
    if t == gh * gw + 1:  # CLS present
        feats = feats[1:]
    elif t != gh * gw:  # neither CLS+grid nor bare grid — bad sidecar
        return None
    return feats, gh, gw


# --------------------------------------------------------------------------- #
# the four axis points — spatial filter H(Z), then standardize + L2-norm + Gram
# --------------------------------------------------------------------------- #
def _sigma_for(div: float, gh: int, gw: int) -> float:
    """``σ = min(gh, gw) / div`` — bucket-invariant, mirrors FEI."""
    return float(min(gh, gw)) / float(div)


def _safe_blur(grid: torch.Tensor, sigma: float, gh: int, gw: int) -> torch.Tensor:
    """``gaussian_blur_2d`` guarded against a kernel wider than the grid.

    ``gaussian_blur_2d`` reflect-pads by ``ceil(3σ)``; reflect padding needs
    ``pad ≤ min_dim − 1``. For our default divisors (≥4) on PE grids this never
    triggers (``3σ = 0.75·min`` at div=4), but we clamp σ defensively so an
    aggressive ``--lp_divs 2`` degrades to the widest valid blur instead of
    crashing.
    """
    if sigma <= 0:
        return grid
    max_pad = min(gh, gw) - 1
    if math.ceil(3.0 * sigma) > max_pad:
        sigma = max(1e-3, (max_pad - 0.01) / 3.0)
    return gaussian_blur_2d(grid, sigma)


def _filtered(grid: torch.Tensor, point: dict, gh: int, gw: int) -> torch.Tensor:
    """Apply the point's spatial filter ``H(Z)`` to ``grid`` ``(1, d, gh, gw)``."""
    kind = point["kind"]
    if kind == "minus1":  # keep DC
        return grid
    if kind == "spatial_norm":  # DC removal (per-channel spatial mean)
        return grid - grid.mean(dim=(2, 3), keepdim=True)
    if kind == "dog_lowsub":  # high-pass: strip the low band up to σ_lp's cutoff
        return grid - _safe_blur(grid, _sigma_for(point["lp_div"], gh, gw), gh, gw)
    if kind == "dog_bandpass":  # band-pass: +1a with the very-high tail rolled off
        s_lp = _sigma_for(point["lp_div"], gh, gw)
        s_in = s_lp / point["inner_mult"]  # tighter inner kernel (higher cutoff)
        return _safe_blur(grid, s_in, gh, gw) - _safe_blur(grid, s_lp, gh, gw)
    raise ValueError(kind)


def _descriptor(
    patches: np.ndarray,
    gh: int,
    gw: int,
    point: dict,
    proj: torch.Tensor,
    triu: tuple[torch.Tensor, torch.Tensor],
    offdiag_w: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """One image → one L2-normalized Gram-kernel descriptor for ``point``.

    Pipeline (matching the relational arm, then summarizing for cross-image
    comparison): reshape to ``(1, d, gh, gw)`` → spatial filter ``H`` →
    ``H / (std_spatial + ε)`` → per-token L2-norm over ``d`` → random-project
    ``d → r`` (JL-preserves token cosines) → feature-gram ``M = YᵀY`` ``(r, r)``
    → vectorize the upper triangle with off-diagonals scaled by ``√2`` so the
    descriptor's dot product equals ``<M_i, M_j>_F`` → L2-normalize.
    """
    d = patches.shape[1]
    grid = (
        torch.from_numpy(patches)
        .to(device)
        .reshape(gh, gw, d)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )  # (1, d, gh, gw)
    h = _filtered(grid, point, gh, gw)
    h = h / (h.std(dim=(2, 3), keepdim=True) + 1e-6)
    tokens = h.squeeze(0).permute(1, 2, 0).reshape(gh * gw, d)  # (N, d)
    tokens = torch.nn.functional.normalize(tokens, dim=-1)  # per-token L2-norm
    y = tokens @ proj  # (N, r) — JL projection of the unit tokens
    m = y.transpose(0, 1) @ y  # (r, r) feature-gram (token-gram's dual spectrum)
    vec = m[triu] * offdiag_w  # upper-tri, off-diag ×√2 ⇒ dot == <M_i,M_j>_F
    vec = torch.nn.functional.normalize(vec, dim=0)
    return vec.cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# scoring — cosine over the precomputed image×image affinity, reusing _auc
# --------------------------------------------------------------------------- #
def _score_from_sim(sim: np.ndarray, in_p: np.ndarray, out_p: np.ndarray) -> dict:
    """d-prime + AUC of in-group vs out-group descriptor cosines."""
    if in_p.size == 0 or out_p.size == 0:
        return {"dprime": float("nan"), "auc": float("nan"), "n_in": 0, "n_out": 0}
    si = sim[in_p[:, 0], in_p[:, 1]]
    so = sim[out_p[:, 0], out_p[:, 1]]
    pooled = math.sqrt(0.5 * (float(si.var()) + float(so.var()))) or 1.0
    return {
        "dprime": float((si.mean() - so.mean()) / pooled),
        "auc": _auc(si, so),
        "mean_in": float(si.mean()),
        "mean_out": float(so.mean()),
        "n_in": int(si.size),
        "n_out": int(so.size),
    }


def _build_points(
    lp_divs: list[float], with_bandpass: bool, inner_mult: float
) -> list[dict]:
    """The ordered axis: −1, 0, then +1a per σ_lp, then optional +1b per σ_lp."""
    points: list[dict] = [
        {
            "key": "minus1",
            "kind": "minus1",
            "axis_pos": -1.0,
            "role": "+global (refuted)",
        },
        {
            "key": "spatial_norm",
            "kind": "spatial_norm",
            "axis_pos": 0.0,
            "role": "DC removal (shipped)",
        },
    ]
    for div in lp_divs:
        points.append(
            {
                "key": f"dog_lowsub_div{div:g}",
                "kind": "dog_lowsub",
                "lp_div": div,
                "axis_pos": 1.0,
                "role": f"+1a low-sub σ_lp=min/{div:g}",
            }
        )
    if with_bandpass:
        for div in lp_divs:
            points.append(
                {
                    "key": f"dog_bandpass_div{div:g}",
                    "kind": "dog_bandpass",
                    "lp_div": div,
                    "inner_mult": inner_mult,
                    "axis_pos": 2.0,
                    "role": f"+1b band-pass σ_lp=min/{div:g}, σ_in=σ_lp/{inner_mult:g}",
                }
            )
    return points


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data_dir", default="post_image_dataset/lora")
    ap.add_argument("--index", default="post_image_dataset/captions/caption_index.json")
    ap.add_argument("--encoder", default="pe_spatial")
    ap.add_argument(
        "--num_samples", type=int, default=3000, help="cap on images loaded"
    )
    ap.add_argument(
        "--num_pairs", type=int, default=20000, help="pairs per axis per class"
    )
    ap.add_argument(
        "--lp_divs",
        default="4,8,16,32",
        help="σ_lp divisor sweep for +1a/+1b (σ_lp = min(gh,gw)/div). Bigger "
        "div = smaller σ = broader low band stripped.",
    )
    ap.add_argument(
        "--with_bandpass",
        action="store_true",
        help="Also run +1b (full DoG band-pass). Default off — σ₂ is the only "
        "harmful corner (high-freq rolloff); enable to test it.",
    )
    ap.add_argument(
        "--bandpass_inner_mult",
        type=float,
        default=4.0,
        help="+1b inner kernel tightness: σ_in = σ_lp / mult (>1 ⇒ rolls off "
        "only the very-high tail above +1a).",
    )
    ap.add_argument(
        "--proj_dim",
        type=int,
        default=256,
        help="random-projection dim r for the Gram-kernel descriptor (JL). "
        "Descriptor length = r(r+1)/2.",
    )
    ap.add_argument(
        "--gate_margin",
        type=float,
        default=0.01,
        help="AUC rise needed to call a step monotone",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    import json

    lp_divs = [float(x) for x in args.lp_divs.split(",") if x.strip()]
    if not lp_divs:
        raise SystemExit("need at least one --lp_divs value")
    if any(div < 4 for div in lp_divs):
        log.warning("lp_div < 4 may exceed the PE grid; σ is clamped defensively.")

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    rng = np.random.default_rng(args.seed)

    data_dir = (REPO_ROOT / args.data_dir).resolve()
    found = _discover(data_dir, args.encoder, args.num_samples)
    if not found:
        raise SystemExit(f"no {args.encoder} sidecars under {data_dir}")

    index = json.loads((REPO_ROOT / args.index).read_text(encoding="utf-8"))
    image_meta = index.get("image_meta", {})
    if not image_meta:
        raise SystemExit("caption_index has no image_meta — run `make caption-index`")

    points = _build_points(lp_divs, args.with_bandpass, args.bandpass_inner_mult)

    # Fixed JL projection (d→r) + upper-tri index / off-diag √2 weights, shared
    # across all images and points so descriptors live in one comparable space.
    r = args.proj_dim
    iu = torch.triu_indices(r, r, offset=0, device=device)
    triu = (iu[0], iu[1])
    offdiag_w = torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0)).to(device)
    proj: torch.Tensor | None = None  # built lazily once d is known

    # Single streaming pass: load each image once, compute every point's
    # descriptor, append to per-point lists. Raw tokens never accumulate.
    stems: list[str] = []
    descriptors: dict[str, list[np.ndarray]] = {p["key"]: [] for p in points}
    n_bad = 0
    for stem, path in found:
        if stem not in image_meta:  # need labels to score it
            continue
        loaded = _load_patch_grid(path)
        if loaded is None:
            n_bad += 1
            continue
        patches, gh, gw = loaded
        if proj is None:
            d = patches.shape[1]
            g = torch.Generator(device=device).manual_seed(args.seed)
            proj = torch.randn(d, r, generator=g, device=device) / math.sqrt(r)
        stems.append(stem)
        for p in points:
            descriptors[p["key"]].append(
                _descriptor(patches, gh, gw, p, proj, triu, offdiag_w, device)
            )
        if len(stems) % 250 == 0:
            log.info(f"  processed {len(stems)} images")

    if len(stems) < 50:
        raise SystemExit(f"only {len(stems)} labeled+cached images — too few to score")
    log.info(f"loaded {len(stems)} labeled images ({n_bad} unreadable/bad-grid), r={r}")

    # Labeled pairs per axis — shared across all points for a fair compare.
    pairs = {
        ax: _pair_indices(stems, image_meta, ax, args.num_pairs, rng) for ax in AXES
    }

    # Score each point: descriptor cosine matrix → in/out AUC per axis.
    results: dict[str, dict] = {}
    for p in points:
        u = np.stack(descriptors[p["key"]])  # (Nimg, P), rows already L2-normed
        sim = (u @ u.T).astype(np.float32)  # cosine affinity
        per_axis = {}
        for ax in AXES:
            in_p, out_p = pairs[ax]
            per_axis[ax] = _score_from_sim(sim, in_p, out_p)
        results[p["key"]] = {"meta": p, "scores": per_axis}

    # --------------------------------------------------------------------- #
    # gate verdict
    #
    # The proposal's load-bearing comparison is **+1a (broad low-band strip) vs
    # the shipped baseline (point 0, spatial_norm/DC-removal)**: PASS if +1a
    # beats it, CLOSE if flat ("spatial_norm already owns the usable contrast").
    # The −1 (keep-DC) endpoint is the *sanity check* ("should be worst"), not
    # the gate — so we report whether it holds rather than letting it veto.
    # Gate on the primary character axis; report the all-axis trend alongside.
    # --------------------------------------------------------------------- #
    def auc(key: str, ax: str = "character") -> float:
        return float(results[key]["scores"][ax]["auc"])

    lowsub_keys = [p["key"] for p in points if p["kind"] == "dog_lowsub"]

    m = args.gate_margin
    # Per-axis: shipped baseline (point 0), keep-DC sanity endpoint, best +1a.
    axis_summ: dict[str, dict] = {}
    for ax in AXES:
        a_minus1 = auc("minus1", ax)
        a_dc = auc("spatial_norm", ax)
        best_key = max(lowsub_keys, key=lambda k: auc(k, ax)) if lowsub_keys else None
        a_lowsub = auc(best_key, ax) if best_key else float("nan")
        axis_summ[ax] = {
            "auc_minus1": a_minus1,
            "auc_spatial_norm": a_dc,
            "best_lowsub_key": best_key,
            "auc_best_lowsub": a_lowsub,
            "lowsub_beats_shipped": a_lowsub > a_dc + m,
            "dc_removal_beats_keepdc": a_dc > a_minus1 + m,  # the sanity expectation
        }

    # Headline decision on the character axis.
    ch = axis_summ["character"]
    auc_dc = ch["auc_spatial_norm"]
    best_lowsub_key, auc_lowsub = ch["best_lowsub_key"], ch["auc_best_lowsub"]
    n_axes_pass = sum(1 for ax in AXES if axis_summ[ax]["lowsub_beats_shipped"])
    role = results[best_lowsub_key]["meta"]["role"] if best_lowsub_key else "n/a"
    # Did the −1 "keep-DC should be worst" sanity hold anywhere?
    sanity_violated = [ax for ax in AXES if not axis_summ[ax]["dc_removal_beats_keepdc"]]

    if ch["lowsub_beats_shipped"]:
        verdict = (
            f"PASS (DoG worth training) — best +1a beats the shipped spatial_norm "
            f"baseline on the character axis ({auc_lowsub:.3f} vs {auc_dc:.3f}, "
            f"{role}) and on {n_axes_pass}/3 axes. Train repa_target_dog with this "
            f"σ_lp; leave σ₂ off (see +1b note). HOLD repa_dog_norm_std == the "
            f"shipped spatial_norm std in the A/B (the paper's std confound)."
        )
        decision = "pass"
    elif auc_lowsub < auc_dc - m:
        verdict = (
            f"CLOSE (DoG harmful) — broad low-band strip UNDERPERFORMS shipped "
            f"spatial_norm on character ({auc_lowsub:.3f} < {auc_dc:.3f}); the "
            f"mid-low band carries usable contrast. Keep DC-only removal."
        )
        decision = "close_harmful"
    else:
        verdict = (
            f"CLOSE (DoG irrelevant) — best +1a ≈ shipped spatial_norm on character "
            f"({auc_lowsub:.3f} vs {auc_dc:.3f}); the DC removal we ship already "
            f"owns the usable contrast. Cheap close, not harmful."
        )
        decision = "close_irrelevant"

    # The −1 sanity is itself a finding when violated: spatial_norm (DC-only
    # removal) dipping below keep-DC means the shipped baseline is suboptimal.
    if sanity_violated:
        verdict += (
            f"  NB sanity-violation on {sanity_violated}: DC-only removal "
            f"(shipped spatial_norm) scores BELOW keep-DC there — the shipped "
            f"baseline is a local dip, not the floor."
        )

    # +1b commentary (only if it was run)
    bandpass_note = None
    if args.with_bandpass and best_lowsub_key:
        bp_keys = [p["key"] for p in points if p["kind"] == "dog_bandpass"]
        best_bp_key = max(bp_keys, key=auc) if bp_keys else None
        if best_bp_key:
            auc_bp = auc(best_bp_key)
            if auc_bp + m < auc_lowsub:
                bandpass_note = (
                    f"+1b high-freq rolloff HURTS ({auc_bp:.3f} < +1a {auc_lowsub:.3f}); "
                    f"pin σ₂→∞."
                )
            else:
                bandpass_note = (
                    f"+1b ≈/≥ +1a ({auc_bp:.3f} vs {auc_lowsub:.3f}); the rolloff is "
                    f"not harmful on PE."
                )
    log.info(verdict)
    if bandpass_note:
        log.info(bandpass_note)

    # --------------------------------------------------------------------- #
    # artifacts
    # --------------------------------------------------------------------- #
    run_dir = make_run_dir("repa", label=args.label or "dog-axis")

    csv_rows = [
        "point,kind,axis_pos,role,"
        + ",".join(f"auc_{ax}" for ax in AXES)
        + ","
        + ",".join(f"dprime_{ax}" for ax in AXES)
    ]
    for p in points:
        sc = results[p["key"]]["scores"]
        aucs = ",".join(f"{sc[ax]['auc']:.4f}" for ax in AXES)
        dps = ",".join(f"{sc[ax].get('dprime', float('nan')):.4f}" for ax in AXES)
        csv_rows.append(
            f'{p["key"]},{p["kind"]},{p["axis_pos"]:g},"{p["role"]}",{aucs},{dps}'
        )
    (run_dir / "auc_by_point.csv").write_text(
        "\n".join(csv_rows) + "\n", encoding="utf-8"
    )

    md = [
        "# REPA-DoG Phase 0 — target discriminability along the low-freq-strip axis",
        "",
        f"- images: {len(stems)} (encoder={args.encoder}, proj_dim r={r})",
        f"- σ_lp divisor sweep: {lp_divs}",
        f"- +1b band-pass: {'on' if args.with_bandpass else 'off'}",
        "",
        "## AUC P(in-group cosine > out-group) — 0.5 = dead target",
        "",
        "| point | role | " + " | ".join(AXES) + " |",
        "|---|---|" + "---|" * len(AXES),
    ]
    for p in points:
        sc = results[p["key"]]["scores"]
        cells = " | ".join(f"{sc[ax]['auc']:.3f}" for ax in AXES)
        md.append(f"| {p['axis_pos']:g} `{p['key']}` | {p['role']} | {cells} |")
    md += [
        "",
        "## Load-bearing comparison: best +1a vs shipped spatial_norm (per axis)",
        "",
        "| axis | keep-DC (−1) | spatial_norm (0) | best +1a | +1a − shipped |",
        "|---|---|---|---|---|",
    ]
    for ax in AXES:
        s = axis_summ[ax]
        delta = s["auc_best_lowsub"] - s["auc_spatial_norm"]
        flag = " ✓" if s["lowsub_beats_shipped"] else ""
        md.append(
            f"| {ax} | {s['auc_minus1']:.3f} | {s['auc_spatial_norm']:.3f} | "
            f"{s['auc_best_lowsub']:.3f} ({s['best_lowsub_key']}) | "
            f"{delta:+.3f}{flag} |"
        )
    md += ["", f"**Verdict:** {verdict}"]
    if bandpass_note:
        md += ["", f"**+1b:** {bandpass_note}"]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "n_images": len(stems),
            "n_bad": n_bad,
            "proj_dim": r,
            "lp_divs": lp_divs,
            "with_bandpass": args.with_bandpass,
            "points": {p["key"]: results[p["key"]] for p in points},
            "axis_summary": axis_summ,
            "n_axes_lowsub_beats_shipped": n_axes_pass,
            "sanity_violated_axes": sanity_violated,
            "decision": decision,
            "verdict": verdict,
            "bandpass_note": bandpass_note,
        },
        label=args.label,
        artifacts=["auc_by_point.csv", "summary.md"],
        device=device,
    )
    log.info(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
