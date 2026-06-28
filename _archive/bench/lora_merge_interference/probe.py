"""Phase-0 probe: do independently-trained artist LoRAs interfere when merged/stacked?

Treats each artist LoRA as a task vector ΔW = (alpha/r)·up@down and measures, per
pair, how much two adapters geometrically collide. Three metrics per module,
aggregated:

  1. flattened-ΔW |cos|  — Fig-5 (OrthoReg paper) metric: cosine between the
     vectorised task vectors. The headline "do they point the same way" number.
  2. input-subspace overlap  — mean squared cosine of principal angles between
     rowspace(down_A) and rowspace(down_B). "Do they read the same input dirs."
  3. output-subspace overlap — same for colspace(up_A)/colspace(up_B).
     "Do they write the same output dirs."

Each subspace overlap is compared to the analytic random-subspace null
(r/ambient_dim): a measured overlap ≈ null means already-orthogonal (OrthoReg
buys little); overlap ≫ null means co-located (interference is real, OrthoReg /
disjoint-slice has a target).

down_init="weight_svd" seeds the down/A side from W₀ SVD, so the *input* side is
expected to be more co-located than the output side — the breakdown will show it.

Usage:
    uv run python bench/lora_merge_interference/probe.py \
        output/ckpt/anima_ama2.safetensors \
        output/ckpt/anima_sincos2.safetensors \
        output/ckpt/anima_channel2.safetensors
"""

from __future__ import annotations

import argparse
import csv
import re
from itertools import combinations
from pathlib import Path

import torch
from safetensors import safe_open

from bench._common import make_run_dir, write_result


def _family(stem: str) -> str:
    if "cross_attn" in stem:
        return "cross_attn"
    if "self_attn" in stem:
        return "self_attn"
    if "_ffn" in stem or "mlp" in stem or "feed_forward" in stem:
        return "ffn"
    return "other"


def load_lora(path: str, device: str) -> dict[str, dict]:
    """stem -> {down:(r,in), up:(out,r), scale: alpha/r} as fp32 on device."""
    downs, ups, alphas = {}, {}, {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.endswith(".lora_down.weight"):
                downs[k[: -len(".lora_down.weight")]] = f.get_tensor(k)
            elif k.endswith(".lora_up.weight"):
                ups[k[: -len(".lora_up.weight")]] = f.get_tensor(k)
            elif k.endswith(".alpha"):
                alphas[k[: -len(".alpha")]] = f.get_tensor(k)
    out = {}
    for stem, down in downs.items():
        up = ups[stem]
        r = down.shape[0]
        alpha = float(alphas[stem].item()) if stem in alphas else float(r)
        out[stem] = {
            "down": down.float().to(device),
            "up": up.float().to(device),
            "scale": alpha / r,
        }
    return out


def orth(mat: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis for the column space of `mat` (cols ≤ rank)."""
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q


def subspace_overlap(qa: torch.Tensor, qb: torch.Tensor) -> float:
    """mean squared cosine of principal angles ∈ [0,1] (0=orth, 1=identical)."""
    r = qa.shape[1]
    return (torch.linalg.matrix_norm(qa.T @ qb) ** 2 / r).item()


def pair_metrics(la: dict, lb: dict, device: str) -> list[dict]:
    rows = []
    for stem in sorted(set(la) & set(lb)):
        a, b = la[stem], lb[stem]
        # task vectors ΔW = scale * up @ down  (out x in)
        dwa = a["scale"] * (a["up"] @ a["down"])
        dwb = b["scale"] * (b["up"] @ b["down"])
        cos = torch.nn.functional.cosine_similarity(
            dwa.flatten(), dwb.flatten(), dim=0
        ).item()
        # input side: rowspace(down) = colspace(down^T) in R^in
        qin_a, qin_b = orth(a["down"].T), orth(b["down"].T)
        # output side: colspace(up) in R^out
        qout_a, qout_b = orth(a["up"]), orth(b["up"])
        in_dim, out_dim, r = a["down"].shape[1], a["up"].shape[0], a["down"].shape[0]
        rows.append(
            {
                "stem": stem,
                "family": _family(stem),
                "abs_cos": abs(cos),
                "cos": cos,
                "in_overlap": subspace_overlap(qin_a, qin_b),
                "out_overlap": subspace_overlap(qout_a, qout_b),
                "in_null": r / in_dim,
                "out_null": r / out_dim,
                "fro_a": torch.linalg.matrix_norm(dwa).item(),
                "fro_b": torch.linalg.matrix_norm(dwb).item(),
            }
        )
    return rows


def _wmean(rows, key, wa="fro_a", wb="fro_b"):
    w = [(r[wa] * r[wb]) for r in rows]
    tot = sum(w) or 1.0
    return sum(r[key] * wi for r, wi in zip(rows, w)) / tot


def _mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def summarize(rows: list[dict]) -> dict:
    fams = sorted({r["family"] for r in rows})
    by_family = {
        fam: {
            "n": sum(1 for r in rows if r["family"] == fam),
            "abs_cos_mean": _mean([r for r in rows if r["family"] == fam], "abs_cos"),
            "in_overlap_mean": _mean(
                [r for r in rows if r["family"] == fam], "in_overlap"
            ),
            "out_overlap_mean": _mean(
                [r for r in rows if r["family"] == fam], "out_overlap"
            ),
        }
        for fam in fams
    }
    return {
        "n_modules": len(rows),
        # Frobenius-weighted (big-norm modules dominate the merged result)
        "abs_cos_wmean": _wmean(rows, "abs_cos"),
        "abs_cos_mean": _mean(rows, "abs_cos"),
        "abs_cos_p90": sorted(r["abs_cos"] for r in rows)[int(0.9 * len(rows))],
        "in_overlap_mean": _mean(rows, "in_overlap"),
        "out_overlap_mean": _mean(rows, "out_overlap"),
        "in_null_mean": _mean(rows, "in_null"),
        "out_null_mean": _mean(rows, "out_null"),
        "in_overlap_vs_null": _mean(rows, "in_overlap") / (_mean(rows, "in_null") or 1),
        "out_overlap_vs_null": _mean(rows, "out_overlap")
        / (_mean(rows, "out_null") or 1),
        "by_family": by_family,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loras", nargs="+", help="paths to LoRA safetensors")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    names = [Path(p).stem for p in args.loras]
    print(f"loading {len(args.loras)} LoRAs on {args.device}: {', '.join(names)}")
    loras = {n: load_lora(p, args.device) for n, p in zip(names, args.loras)}

    run_dir = make_run_dir("lora_merge_interference", label="artist-trio")
    all_rows = []
    pairs = {}
    for (na, la), (nb, lb) in combinations(loras.items(), 2):
        rows = pair_metrics(la, lb, args.device)
        for r in rows:
            r["pair"] = f"{na}__{nb}"
        all_rows += rows
        pairs[f"{na}__{nb}"] = summarize(rows)
        s = pairs[f"{na}__{nb}"]
        print(
            f"\n[{na} vs {nb}]  n={s['n_modules']}"
            f"\n  |cos| Fro-wmean={s['abs_cos_wmean']:.3f}  mean={s['abs_cos_mean']:.3f}"
            f"  p90={s['abs_cos_p90']:.3f}"
            f"\n  in-overlap ={s['in_overlap_mean']:.3f} (null {s['in_null_mean']:.3f},"
            f" {s['in_overlap_vs_null']:.1f}x)"
            f"\n  out-overlap={s['out_overlap_mean']:.3f} (null {s['out_null_mean']:.3f},"
            f" {s['out_overlap_vs_null']:.1f}x)"
        )

    # per-module CSV artifact
    csv_path = run_dir / "per_module.csv"
    with open(csv_path, "w", newline="") as fh:
        cols = [
            "pair", "stem", "family", "abs_cos", "cos",
            "in_overlap", "out_overlap", "in_null", "out_null", "fro_a", "fro_b",
        ]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r[c] for c in cols})

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"loras": names, "pairs": pairs},
        artifacts=["per_module.csv"],
    )
    print(f"\nwrote {run_dir}/result.json")


if __name__ == "__main__":
    main()
