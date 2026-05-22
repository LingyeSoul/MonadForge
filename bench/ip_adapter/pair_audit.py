#!/usr/bin/env python3
"""IP-Adapter distinct-pair audit (Phase 0).

Consumes the method-agnostic typed-tag index
(``post_image_dataset/captions/caption_index.json``, built by
``make caption-index``) and reports whether the dataset can supply enough
*distinct* same-identity references for the identity-pair training contract in
``docs/proposal/ip-adapter-identity-pairs.md``.

This script owns the IP-Adapter *sampling policy* — the character → copyright →
artist tiered backoff — which is deliberately NOT baked into the shared index.
It computes:

  * per-tier coverage / groups≥2 / Σ nC2 positive pairs (the proposal table),
  * top cross-artist characters (the tightest, most valuable signal),
  * tiered reachability: at which tier each image first finds a distinct
    partner, or whether it is self-only,
  * group imbalance (largest groups → the inverse-sqrt cap risk).

Writes ``bench/ip_adapter/pair_audit.md`` and prints a summary. Pure data.
"""

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "post_image_dataset/captions/caption_index.json"
OUT_MD = Path(__file__).resolve().parent / "pair_audit.md"

# IP-Adapter sampling policy: tightest → loosest identity tier.
LEVEL_PRIORITY = ["character", "copyright", "artist"]


def _n_pairs(n: int) -> int:
    return n * (n - 1) // 2


def main():
    if not INDEX_PATH.exists():
        sys.exit(f"missing {INDEX_PATH} — run `make caption-index` first")
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    image_meta = index["image_meta"]
    groups = index["groups"]
    n_images = len(image_meta)

    # ── per-tier coverage table ───────────────────────────────────────────
    tier_rows = []
    for level in LEVEL_PRIORITY:
        g = groups[level]
        cov = sum(1 for v in image_meta.values() if v[level])
        ge2 = {tag: stems for tag, stems in g.items() if len(stems) >= 2}
        pairs = sum(_n_pairs(len(s)) for s in ge2.values())
        tier_rows.append(
            {
                "level": level,
                "coverage": cov,
                "pct": 100 * cov / n_images,
                "groups_ge2": len(ge2),
                "positive_pairs": pairs,
            }
        )

    # ── top cross-artist characters ───────────────────────────────────────
    char_artist = []
    for tag, stems in groups["character"].items():
        artists = set()
        for s in stems:
            artists.update(image_meta[s]["artist"])
        char_artist.append((tag, len(artists), len(stems)))
    char_artist.sort(key=lambda r: (-r[1], -r[2]))
    n_cross = sum(1 for _t, a, _n in char_artist if a >= 2)
    top_cross = char_artist[:8]

    # ── tiered reachability ───────────────────────────────────────────────
    # For each image: tightest tier with a *distinct* same-identity partner.
    reach = Counter()
    cross_artist_at_char = 0  # char-tier resolves that also have a diff-artist partner
    for stem, meta in image_meta.items():
        resolved = None
        for level in LEVEL_PRIORITY:
            has_distinct = any(
                len(groups[level][tag]) >= 2
                for tag in meta[level]
                if tag in groups[level]
            )
            # >=2 in a group the image belongs to guarantees a distinct partner
            if has_distinct:
                resolved = level
                break
        reach[resolved or "self"] += 1
        if resolved == "character":
            my_artists = set(meta["artist"])
            for tag in meta["character"]:
                if any(
                    set(image_meta[o]["artist"]) - my_artists
                    for o in groups["character"].get(tag, [])
                    if o != stem
                ):
                    cross_artist_at_char += 1
                    break

    # ── group imbalance (largest groups per tier) ─────────────────────────
    largest = {
        level: sorted(
            ((tag, len(s)) for tag, s in groups[level].items()), key=lambda r: -r[1]
        )[:5]
        for level in LEVEL_PRIORITY
    }

    # ── render markdown ───────────────────────────────────────────────────
    L = []
    L.append("# IP-Adapter distinct-pair audit (Phase 0)\n")
    L.append(
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"from `{INDEX_PATH.relative_to(REPO_ROOT)}`\n"
    )
    L.append(
        f"(index built {index['meta']['generated']} · vocab "
        f"`{index['meta']['vocab_path']}`).\n"
    )
    L.append(f"\n**Dataset: {n_images} captioned images.**\n")

    L.append("\n## Tiered positive-pair coverage\n")
    L.append("| Tier | Coverage | Groups ≥2 imgs | Same-group positive pairs (Σ nC2) |")
    L.append("|---|---|---|---|")
    for r in tier_rows:
        L.append(
            f"| **{r['level'].capitalize()}** | {r['coverage']} imgs / "
            f"{r['pct']:.0f}% | {r['groups_ge2']} | {r['positive_pairs']:,} |"
        )

    L.append("\n## Tiered reachability (deployment sampler)\n")
    L.append(
        "Each image routed through the character → copyright → artist backoff; "
        "tier = tightest level with a *distinct* same-identity partner.\n"
    )
    L.append("| Resolves at | Images | % |")
    L.append("|---|---|---|")
    for level in [*LEVEL_PRIORITY, "self"]:
        c = reach.get(level, 0)
        label = (
            "self-only (no distinct positive)"
            if level == "self"
            else level.capitalize()
        )
        L.append(f"| {label} | {c} | {100 * c / n_images:.1f}% |")
    reachable = n_images - reach.get("self", 0)
    L.append(
        f"\n**{reachable} / {n_images} ({100 * reachable / n_images:.1f}%) images have a "
        f"distinct positive at some tier.**"
    )
    L.append(
        f" Of the {reach.get('character', 0)} character-tier images, "
        f"{cross_artist_at_char} also have a *different-artist* partner "
        f"(usable for `identity_cross_artist`)."
    )

    L.append("\n## Top cross-artist characters\n")
    L.append(
        f"{n_cross} characters appear across ≥2 artists — the tightest, "
        "most valuable identity signal.\n"
    )
    L.append("| Character | Artists | Images |")
    L.append("|---|---|---|")
    for tag, a, n in top_cross:
        L.append(f"| `{tag}` | {a} | {n} |")

    L.append("\n## Group imbalance (largest groups → inverse-sqrt cap)\n")
    for level in LEVEL_PRIORITY:
        rows = ", ".join(f"`{t}` ({n})" for t, n in largest[level])
        L.append(f"- **{level.capitalize()}**: {rows}")

    L.append("\n## Caveats\n")
    L.append(
        "- **Vocab-limited character classification.** Characters are tagged via the "
        "Anima Tagger vocab (frequency-cutoff subset); rarer character tags fall "
        "through to the franchise/artist tiers. The 32% character coverage is a "
        "floor — tagging more of the set lifts it (proposal risk: *thin character "
        "coverage*)."
    )
    L.append(
        "- **Artist is exact.** Detected by the `@` prefix, not vocab membership, so "
        "artist coverage is complete (100%)."
    )
    L.append(
        "- **Sampling policy is not in the index.** This audit applies the tiered "
        "backoff; the shared `caption_index.json` only stores typed tags + groups."
    )

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ── console summary ───────────────────────────────────────────────────
    print(f"pair audit → {OUT_MD.relative_to(REPO_ROOT)}")
    print(f"  images: {n_images}")
    for r in tier_rows:
        print(
            f"  {r['level']:9s}: {r['coverage']:5d} imgs ({r['pct']:4.1f}%), "
            f"{r['groups_ge2']:4d} groups≥2, {r['positive_pairs']:,} pairs"
        )
    print(
        f"  reachable (distinct positive at some tier): {reachable}/{n_images} "
        f"({100 * reachable / n_images:.1f}%); self-only: {reach.get('self', 0)}"
    )
    print(f"  cross-artist characters (≥2 artists): {n_cross}")


if __name__ == "__main__":
    main()
