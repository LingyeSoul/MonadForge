#!/usr/bin/env python
"""Phase-0 tagger-groups probe — is there headroom in the group taxonomy?

THE QUESTIONS (all answered from the cached label matrix alone — no training):

  A. NEGATIVE-RELIABILITY HEADROOM (the multilabel thesis).
     Booru labels are partial: an absent tag is often present-but-unlabelled.
     The grouping gives a prior on *which* negatives to trust — if any tag in
     group G fired, the annotator was attending to G, so other absences in G
     are reliable negatives; if NO tag in G fired, G may simply be un-annotated.
     A group-conditional negative weighting is only worth building if a large
     share of negative (image, tag) cells actually fall in *inactive*-group
     cells. This probe sizes that mass per group and globally.

  B. SOFTMAX-GROUP COST AUDIT (the `softmax_when_solo` fragility worry).
     softmax converts a group to 1-of-K via argmax. That silently DISCARDS a
     real label whenever a solo image has >=2 in-group tags. We measure the
     real solo-multi-positive rate per softmax group (the YAML descriptions
     *claim* "X% solo multi-rate" — verify it) and flag groups where argmax
     throws away too many true labels (-> candidates to demote to multilabel).
     We also score every multilabel group's exclusivity to surface any group
     that is *secretly* near-exclusive (-> candidate to PROMOTE to softmax).

  C. SOLO-GATE RELIABILITY.
     The solo gate is inferred from count tags ({solo,1girl,1boy,1other} AND
     no multi-count tag). If those are noisy/contradictory the gate misfires.
     We bucket every train image into single-only / multi-only / both
     (contradiction) / neither (undefined) to size that fragility directly.

READOUT: bench/tagger_groups/results/<ts>[-label]/{group_stats.csv,
         solo_gate.json, sample_cells.csv, summary.md, result.json}

GATE (heuristics, not hard pass/fail — Phase 0 is for sizing):
  * negative-reliability lever REAL if inactive-group negative mass is a
    non-trivial fraction of all in-group negatives (default flag > 0.15).
  * softmax group FRAGILE if solo-multi-positive rate > --softmax_discard_flag
    (default 0.05) — argmax discards a true label that often.
  * multilabel group PROMOTABLE if, when active, it is exactly-one >= 0.90.

Run from anima_lora/::

    uv run python bench/tagger_groups/probe_group_structure.py --label v2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Mirror of scripts/anima_tagger/train_common.py solo-gate definition.
SINGLE_COUNT_NAMES = {"solo", "1girl", "1boy", "1other"}
# Mirror of library/captioning/taxonomy.py::_COUNT_RE (multi-subject counts).
_COUNT_RE = re.compile(
    r"^(?:\d+\+?(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$"
)


def load_data(ckpt_dir: Path, split: str):
    """Return (multi_hot [N,T] bool, vocab_dict, tag_names list)."""
    ds = json.load(open(ckpt_dir / "dataset.json", encoding="utf-8"))
    vocab = json.load(open(ckpt_dir / "vocab.json", encoding="utf-8"))
    n_tags = int(ds["n_tags"])

    stems = ds["stems"]
    tag_idx_rows = ds["tag_indices"]
    if split == "all":
        rows = range(len(stems))
    else:
        keep = set(ds["split"][split])
        rows = [i for i, s in enumerate(stems) if s in keep]

    mh = np.zeros((len(rows), n_tags), dtype=bool)
    for r, i in enumerate(rows):
        idxs = tag_idx_rows[i]
        if idxs:
            mh[r, idxs] = True

    tag_names = [None] * n_tags
    for t in vocab["tags"]:
        tag_names[int(t["index"])] = t["name"]
    return mh, vocab, tag_names, [stems[i] for i in rows]


def solo_gate_buckets(mh: np.ndarray, tag_names):
    """Replicate solo_mask + audit its degenerate buckets.

    Returns (solo_mask [N] bool, bucket_counts dict).
    """
    name_to_idx = {n: i for i, n in enumerate(tag_names) if n is not None}
    single_idx = [name_to_idx[n] for n in SINGLE_COUNT_NAMES if n in name_to_idx]
    multi_idx = [
        i
        for i, n in enumerate(tag_names)
        if n is not None and n not in SINGLE_COUNT_NAMES and _COUNT_RE.match(n)
    ]
    has_single = (
        mh[:, single_idx].any(axis=1) if single_idx else np.zeros(len(mh), bool)
    )
    has_multi = mh[:, multi_idx].any(axis=1) if multi_idx else np.zeros(len(mh), bool)

    buckets = {
        "single_only": int((has_single & ~has_multi).sum()),
        "multi_only": int((~has_single & has_multi).sum()),
        "both_contradiction": int((has_single & has_multi).sum()),
        "neither_undefined": int((~has_single & ~has_multi).sum()),
    }
    solo_mask = has_single & ~has_multi  # exactly what the trainer routes on
    return solo_mask, buckets, single_idx, multi_idx


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt_dir", default="models/captioners/anima-tagger-v2")
    ap.add_argument("--split", default="train", choices=["train", "val", "all"])
    ap.add_argument("--label", default=None)
    ap.add_argument(
        "--softmax_discard_flag",
        type=float,
        default=0.05,
        help="flag a softmax group whose solo-multi-positive rate exceeds this",
    )
    ap.add_argument(
        "--promote_excl_flag",
        type=float,
        default=0.90,
        help="flag a multilabel group as promotable if exactly-one rate (when active) exceeds this",
    )
    ap.add_argument(
        "--inactive_mass_flag",
        type=float,
        default=0.15,
        help="negative-reliability lever flagged REAL above this inactive-negative mass fraction",
    )
    ap.add_argument(
        "--sample_cells",
        type=int,
        default=200,
        help="dump this many random (image,tag) negative cells per regime for gold spot-check",
    )
    args = ap.parse_args()

    ckpt_dir = (
        (REPO_ROOT / args.ckpt_dir)
        if not Path(args.ckpt_dir).is_absolute()
        else Path(args.ckpt_dir)
    )
    mh, vocab, tag_names, stems = load_data(ckpt_dir, args.split)
    N, T = mh.shape
    log.info(f"loaded {N} images x {T} tags  (split={args.split})")

    solo_mask, gate_buckets, single_idx, multi_idx = solo_gate_buckets(mh, tag_names)
    n_solo = int(solo_mask.sum())
    log.info(f"solo gate: {gate_buckets}  (solo={n_solo}, {n_solo / N:.1%})")

    groups = vocab.get("groups") or []
    rows = []  # per-group dict
    # Global negative-mass accumulators (over tags that belong to ANY group).
    tot_ingroup_neg = 0  # total negative cells over in-group tags
    tot_inactive_neg = (
        0  # of those, ones where the cell's group is inactive for that image
    )

    for g in groups:
        name = g["name"]
        mode = g["mode"]
        idx = np.array(g["tag_indices"], dtype=np.int64)
        esc = np.array(g.get("escape_indices") or [], dtype=np.int64)
        K = len(idx)
        if K == 0:
            continue

        sub = mh[:, idx]  # [N, K]
        card = sub.sum(axis=1)  # [N] tags present in group per image
        active = card > 0  # [N]
        n_active = int(active.sum())
        activation_rate = n_active / N

        # Exclusivity / cardinality structure when active.
        mean_card_active = float(card[active].mean()) if n_active else 0.0
        frac_exactly_one = float((card[active] == 1).mean()) if n_active else 0.0

        # --- B: softmax-cost audit on SOLO images (where the gate would fire) ---
        has_escape = mh[:, esc].any(axis=1) if esc.size else np.zeros(N, bool)
        # samples the trainer would route through CE for a softmax_when_solo group:
        ce_applicable = solo_mask & ~has_escape & active
        n_ce = int(ce_applicable.sum())
        # of those, how many carry >=2 in-group labels -> argmax discards a true label
        discard = int((card[ce_applicable] >= 2).sum()) if n_ce else 0
        solo_multi_rate = (discard / n_ce) if n_ce else 0.0

        # --- A: inactive-group negative mass over this group's tags ---
        # negative cells over group tags = (~sub). Of those, the ones in images
        # where the group is INACTIVE are the "unreliable / un-annotated" negatives.
        neg = ~sub  # [N, K]
        ingroup_neg = int(neg.sum())
        inactive_neg = int(neg[~active].sum())  # all-K negatives for inactive imgs
        tot_ingroup_neg += ingroup_neg
        tot_inactive_neg += inactive_neg
        inactive_neg_frac = inactive_neg / ingroup_neg if ingroup_neg else 0.0

        # --- flags ---
        flags = []
        if (
            mode in ("softmax", "softmax_when_solo")
            and solo_multi_rate > args.softmax_discard_flag
        ):
            flags.append("SOFTMAX_FRAGILE")  # argmax discards real labels too often
        if (
            mode == "multilabel"
            and n_active >= 200
            and frac_exactly_one >= args.promote_excl_flag
        ):
            flags.append("PROMOTABLE")  # secretly near-exclusive

        rows.append(
            {
                "group": name,
                "mode": mode,
                "K": K,
                "activation_rate": round(activation_rate, 4),
                "mean_card_when_active": round(mean_card_active, 3),
                "frac_exactly_one": round(frac_exactly_one, 4),
                "n_solo_ce": n_ce,
                "solo_multi_pos_rate": round(solo_multi_rate, 4),
                "argmax_discards": discard,
                "inactive_neg_frac": round(inactive_neg_frac, 4),
                "flags": "|".join(flags),
            }
        )

    global_inactive_mass = (
        tot_inactive_neg / tot_ingroup_neg if tot_ingroup_neg else 0.0
    )

    # ---- verdicts ----
    fragile = [r["group"] for r in rows if "SOFTMAX_FRAGILE" in r["flags"]]
    promotable = [r["group"] for r in rows if "PROMOTABLE" in r["flags"]]
    lever_real = global_inactive_mass > args.inactive_mass_flag

    # ---- write artifacts ----
    run_dir = make_run_dir("tagger_groups", label=args.label or args.split)

    # group_stats.csv
    cols = [
        "group",
        "mode",
        "K",
        "activation_rate",
        "mean_card_when_active",
        "frac_exactly_one",
        "n_solo_ce",
        "solo_multi_pos_rate",
        "argmax_discards",
        "inactive_neg_frac",
        "flags",
    ]
    lines = [",".join(cols)]
    for r in sorted(
        rows, key=lambda x: (x["mode"] != "softmax_when_solo", -x["activation_rate"])
    ):
        lines.append(",".join(str(r[c]) for c in cols))
    (run_dir / "group_stats.csv").write_text("\n".join(lines) + "\n")

    # solo_gate.json
    (run_dir / "solo_gate.json").write_text(
        json.dumps(
            {
                "buckets": gate_buckets,
                "n_solo_routed": n_solo,
                "solo_frac": n_solo / N,
                "single_count_tags": [tag_names[i] for i in single_idx],
                "n_multi_count_tags": len(multi_idx),
            },
            indent=2,
        )
    )

    # sample_cells.csv — random in-group negative cells split by regime, for a
    # MANUAL gold spot-check (active-group vs inactive-group negatives). The
    # whole headroom thesis stands on inactive-group negatives being noisier;
    # this can't be confirmed from labels alone, so emit cells to eyeball.
    rng = np.random.default_rng(0)
    sc = ["stem_row,tag,group,regime"]
    for g in groups:
        idx = np.array(g["tag_indices"], dtype=np.int64)
        if not len(idx):
            continue
        sub = mh[:, idx]
        active = sub.sum(axis=1) > 0
        for regime, mask in (("active", active), ("inactive", ~active)):
            ii = np.where(mask)[0]
            if not len(ii):
                continue
            pick = rng.choice(
                ii,
                size=min(args.sample_cells // max(len(groups), 1) + 1, len(ii)),
                replace=False,
            )
            for r in pick:
                negk = np.where(~sub[r])[0]
                if not len(negk):
                    continue
                k = idx[rng.choice(negk)]
                sc.append(f"{stems[r]},{tag_names[k]},{g['name']},{regime}")
    (run_dir / "sample_cells.csv").write_text("\n".join(sc) + "\n")

    # summary.md
    md = [
        f"# tagger-groups Phase-0 ({args.split}, N={N}, T={T})\n",
        "## A. Negative-reliability lever (multilabel headroom)",
        f"- in-group negative cells: {tot_ingroup_neg:,}",
        f"- of those, in INACTIVE groups (unreliable): {tot_inactive_neg:,} "
        f"= **{global_inactive_mass:.1%}**",
        f"- lever REAL (> {args.inactive_mass_flag:.0%}): **{lever_real}**\n",
        "## B. Softmax-cost audit",
        f"- FRAGILE softmax groups (solo-multi-pos > {args.softmax_discard_flag:.0%}): "
        f"{fragile or 'none'}",
        f"- PROMOTABLE multilabel groups (exactly-one >= {args.promote_excl_flag:.0%}): "
        f"{promotable or 'none'}\n",
        "## C. Solo-gate buckets",
        *[f"- {k}: {v:,} ({v / N:.1%})" for k, v in gate_buckets.items()],
        "\n## softmax_when_solo groups (verify the YAML 'solo multi-rate' claims)",
        "| group | K | n_solo_ce | solo_multi_pos_rate | discards |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r["mode"] == "softmax_when_solo":
            md.append(
                f"| {r['group']} | {r['K']} | {r['n_solo_ce']} | "
                f"{r['solo_multi_pos_rate']:.1%} | {r['argmax_discards']} |"
            )
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")

    log.info("\n".join(md))

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "n_images": N,
            "n_tags": T,
            "n_groups": len(rows),
            "solo_gate_buckets": gate_buckets,
            "global_inactive_neg_mass": global_inactive_mass,
            "lever_real": lever_real,
            "fragile_softmax_groups": fragile,
            "promotable_multilabel_groups": promotable,
            "per_group": rows,
        },
        label=args.label,
        artifacts=[
            "group_stats.csv",
            "solo_gate.json",
            "sample_cells.csv",
            "summary.md",
        ],
    )
    log.info(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
