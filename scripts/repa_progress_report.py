"""Post-hoc REPA read-out of a training ``*.progress.jsonl``.

Companion to the ``repa/align_loss`` / ``repa/active`` step metrics
(``library/training/repa.py``); supports the lever-1 (anneal) decision
per ``docs/experimental/repa.md`` §"Annealing plan":

- **Effective weight**: per-decile unweighted align loss, total loss, FM-only
  loss (``loss/current`` is the *composed* loss, so FM-only ≈
  ``loss/current − repa_weight · align_loss`` while the term is active), and
  the weighted REPA share of the total.
- **Plateau**: where the smoothed align curve reaches within 10/5/2% of its
  final mean — the curve-justified anneal-cutoff candidate.
- **Anneal release** (annealed runs only): detects the sustained
  ``repa/active`` 1→0 transition and compares mean FM-only loss in a window
  before vs after the cutoff. FM-only *dropping* after release ⇒ REPA was
  fighting the FM objective late (supports annealing).

Usage:
    python scripts/repa_progress_report.py output/logs/<run>.progress.jsonl \
        [--repa-weight 0.05] [--window 150]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_steps(path: Path) -> list[dict]:
    steps = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("ev") == "step":
                steps.append(rec)
    return steps


def _smooth(xs: list[float], k: int = 21) -> list[float]:
    out = []
    for i in range(len(xs)):
        w = xs[max(0, i - k // 2) : i + k // 2 + 1]
        out.append(sum(w) / len(w))
    return out


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl", type=Path, help="output/logs/<run>.progress.jsonl")
    ap.add_argument(
        "--repa-weight",
        type=float,
        default=0.05,
        help="repa_weight the run trained with (default 0.05; see its "
        "snapshot.toml) — used to back out the FM-only loss",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=150,
        help="steps on each side of the anneal cutoff for the release A/B",
    )
    args = ap.parse_args(argv)

    steps = _load_steps(args.jsonl)
    if not steps:
        print(f"no step events in {args.jsonl}", file=sys.stderr)
        return 1

    w = args.repa_weight
    rows = []
    for r in steps:
        total = r.get("loss/current")
        if total is None:
            continue
        align = r.get("repa/align_loss")
        active = r.get("repa/active")
        # While active, loss/current includes the weighted term; back it out.
        # align_loss is a last-active-step snapshot, so only trust it when
        # active == 1 this step.
        if active == 1.0 and align is not None:
            fm_only = total - w * align
        else:
            fm_only = total
        rows.append(
            {
                "gstep": r["global_step"],
                "total": total,
                "align": align if active == 1.0 else None,
                "active": active,
                "fm_only": fm_only,
            }
        )

    n = len(rows)
    print(
        f"{args.jsonl.name}: {n} steps, gstep {rows[0]['gstep']}..{rows[-1]['gstep']}"
    )
    n_active = sum(1 for r in rows if r["active"] == 1.0)
    print(f"repa active on {n_active}/{n} steps ({n_active / n:.0%}); weight={w:g}\n")

    # ------------------------------------------------------------- deciles
    print(
        f"{'decile':>6} {'gstep':>12} {'align':>8} {'total':>8} "
        f"{'fm_only':>8} {'repa share':>10}"
    )
    for d in range(10):
        lo, hi = d * n // 10, (d + 1) * n // 10
        chunk = rows[lo:hi]
        aligns = [r["align"] for r in chunk if r["align"] is not None]
        a = _mean(aligns)
        t = _mean(r["total"] for r in chunk)
        f = _mean(r["fm_only"] for r in chunk)
        share = (w * a / t) if aligns else float("nan")
        span = f"{chunk[0]['gstep']}-{chunk[-1]['gstep']}"
        print(f"{d:>6} {span:>12} {a:>8.4f} {t:>8.4f} {f:>8.4f} {share:>9.1%}")

    # ------------------------------------------------------------- plateau
    act_rows = [r for r in rows if r["align"] is not None]
    if len(act_rows) >= 50:
        aligns = _smooth([r["align"] for r in act_rows])
        tail = aligns[int(len(aligns) * 0.7) :]
        final = _mean(tail)
        last_g = act_rows[-1]["gstep"]
        print(f"\nlate-run align mean: {final:.4f}")
        for thr in (1.10, 1.05, 1.02):
            for r, v in zip(act_rows, aligns):
                if v <= final * thr:
                    print(
                        f"smoothed align within {round((thr - 1) * 100)}% of final "
                        f"at gstep {r['gstep']} ({r['gstep'] / last_g:.0%} of the "
                        f"active span)"
                    )
                    break

    # ------------------------------------------------------- anneal release
    # Sustained 1→0: last active step followed only by inactive ones.
    last_active_i = max(
        (i for i, r in enumerate(rows) if r["active"] == 1.0), default=None
    )
    if last_active_i is not None and last_active_i < n - 1:
        cutoff = rows[last_active_i + 1]["gstep"]
        before = [
            r["fm_only"]
            for r in rows
            if cutoff - args.window <= r["gstep"] < cutoff and r["active"] == 1.0
        ]
        after = [
            r["fm_only"]
            for r in rows
            if cutoff <= r["gstep"] < cutoff + args.window and r["active"] == 0.0
        ]
        if before and after:
            mb, ma = _mean(before), _mean(after)
            print(
                f"\nanneal cutoff at gstep {cutoff}: FM-only "
                f"{mb:.4f} (last {len(before)} active) → {ma:.4f} "
                f"(first {len(after)} released), delta {ma - mb:+.4f}"
            )
            print(
                "  negative delta ⇒ FM improved once REPA released "
                "(supports annealing); positive ⇒ no late conflict."
            )
        else:
            print(f"\nanneal cutoff at gstep {cutoff}: window too thin to compare")
    else:
        print("\nno anneal cutoff in this run (repa active to the end)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
