#!/usr/bin/env python
"""Bench harness for ortho-postfix structural-orthogonality validation.

Per `docs/proposal/orthogonal_postfix.md` validation criteria — runs the
weights-only checks from `archive/bench/postfix/analyze_ortho_postfix.py`
end-to-end on a checkpoint and writes the standard `result.json` envelope.

Tier 1.5 per CONTRIBUTING.md: numerics revision to an existing method
(postfix). The bench is deliberately weights-only — it doesn't run
inference. The qualitative A/B comparison (DiT vs DiT + ortho-postfix vs
DiT + legacy collapsed postfix) is left to manual eyeballing as the
proposal calls out, since it needs human judgment.

Usage::

    uv run python bench/postfix_ortho/run_bench.py \\
        --postfix_weight output/ckpt/anima_postfix_ortho.safetensors \\
        --label first-run

Writes::

    bench/postfix_ortho/results/<YYYYMMDD-HHMM>[-<label>]/
        result.json
        analyze.json   (full analyzer payload)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402

# Reuse the analyzer instead of re-implementing the checks — single source of
# truth for the orthogonality / lambda / NN-probe metrics.
sys.path.insert(0, str(REPO_ROOT / "archive" / "bench" / "postfix"))
import analyze_ortho_postfix as analyzer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--postfix_weight",
        default="output/ckpt/anima_postfix_ortho_v2.safetensors",
        help="Ortho-postfix safetensors checkpoint to validate",
    )
    p.add_argument(
        "--dataset_dir",
        default="post_image_dataset/lora",
        help="Cached TE corpus for the T5 NN probe",
    )
    p.add_argument("--num_captions", type=int, default=256)
    p.add_argument("--min_count", type=int, default=3)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--skip_t5", action="store_true",
        help="Skip the T5 NN probe (use when no cached TE corpus is available)",
    )
    p.add_argument(
        "--label", default=None,
        help="Optional label appended to the bench run dir name",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = make_run_dir("postfix_ortho", label=args.label)
    analyze_json = run_dir / "analyze.json"

    # Forward to the analyzer with the same argv it expects. Mutating sys.argv
    # is the simplest way to drive it without splitting its arg parsing — the
    # analyzer is also kept usable as a standalone script.
    analyzer_argv = [
        "analyze_ortho_postfix.py",
        "--postfix_weight", str(args.postfix_weight),
        "--dataset_dir", str(args.dataset_dir),
        "--num_captions", str(args.num_captions),
        "--min_count", str(args.min_count),
        "--top_k", str(args.top_k),
        "--seed", str(args.seed),
        "--out_json", str(analyze_json),
    ]
    if args.skip_t5:
        analyzer_argv.append("--skip_t5")

    saved_argv = sys.argv
    sys.argv = analyzer_argv
    try:
        analyzer.main()
    finally:
        sys.argv = saved_argv

    with open(analyze_json) as f:
        report = json.load(f)

    # Headline metrics for the result envelope. Two schemas:
    #   - postfix-mode ortho (v2): structural_orthogonality + lambda_global
    #   - cond-mode ortho (cond_v2): per_caption_orthogonality + lambda_distribution
    #                                 + cross_caption_diversity
    metrics = {
        "K": report["K"],
        "D": report["D"],
        "ortho_basis_kind": report["ortho_basis_kind"],
        "mode": report.get("mode", "postfix"),
    }
    if "structural_orthogonality" in report:
        # postfix-mode ortho v2
        so = report["structural_orthogonality"]
        lam = report["lambda_global"]
        metrics.update({
            "ortho_residual_frobenius": so["frobenius_residual"],
            "ortho_diag_mismatch_max": so["diag_mismatch_max"],
            "ortho_off_diag_max": so["off_diag_max"],
            "ortho_pass": so["pass"],
            "lambda_global": lam["value"],
            "lambda_global_abs": lam["abs_value"],
            "lambda_alive_pass": lam["alive_pass"],
        })
        if report.get("t5_nn_probe"):
            per_slot = report["t5_nn_probe"].get("per_slot_topk") or []
            if per_slot:
                top1_set = {entry["ids"][0] for entry in per_slot if entry["ids"]}
                metrics["t5_nn_distinct_top1_per_slot"] = len(top1_set)
                metrics["t5_nn_lexicon_size"] = report["t5_nn_probe"]["lexicon_size"]
    else:
        # cond-mode ortho cond_v2
        pco = report["per_caption_orthogonality"]
        lam = report["lambda_distribution"]
        div = report["cross_caption_diversity"]
        metrics.update({
            "n_captions": report["n_captions"],
            "per_caption_ortho_max": pco["residual_max"],
            "per_caption_ortho_mean": pco["residual_mean"],
            "per_caption_ortho_pass_fraction": pco["fraction_pass_1e_4"],
            "per_caption_ortho_pass": pco["pass"],
            "lambda_abs_min": lam["abs_min"],
            "lambda_abs_max": lam["abs_max"],
            "lambda_abs_mean": lam["abs_mean"],
            "lambda_alive_fraction": lam["alive_fraction"],
            "lambda_alive_pass": lam["alive_pass"],
            "cross_caption_cos_mean": div["cos_mean"],
            "cross_caption_cos_max": div["cos_max"],
            "cross_caption_diversity_pass": div["pass"],
        })

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["analyze.json"],
    )
    print(f"\n  > bench result: {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
