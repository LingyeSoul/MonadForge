"""``python -m anime_tools.downloads`` — pre-fetch the weights, or list them.

A repair tool and a preflight, nothing more: every loader still auto-fetches on
first use, so this only moves the wait somewhere the user chose. One gated repo
failing must not abort the rest, which is why every failure is collected and
reported at the end rather than raised.
"""

from __future__ import annotations

import argparse
import sys

from library.curation_compat.downloads._assets import PACK_BY_ID, PACKS
from library.curation_compat.downloads._catalog import by_id, by_pack, catalog, expand

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m anime_tools.downloads",
        description="Pre-fetch the model weights the stages need. With no ID, "
        "downloads everything that is missing; with IDs, re-fetches exactly "
        "those (a repair). An ID is a row (`sam3`) or a pack (`ocr` — every row "
        "that installs under it). Every loader still auto-fetches on first use — "
        "this only moves the wait somewhere you chose.",
    )
    p.add_argument(
        "ids",
        nargs="*",
        metavar="ID",
        help="Row or pack ids to fetch (default: every missing row); "
        f"packs: {', '.join(p.id for p in PACKS)}",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Show the catalog, grouped by pack, and exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # One catalog for the whole run: building it resolves the curation home and
    # reads the installed checkpoint's config to name its backbone repo.
    rows = catalog()
    assets = by_id(rows)

    if args.list:
        for pack_id, pack_rows in by_pack(rows).items():
            pack = PACK_BY_ID[pack_id]
            print(f"[{pack.id}] {pack.title} — {pack.description}")
            for a in pack_rows:
                mark = "installed" if a.installed else "MISSING  "
                print(f"  {mark}  {a.id:<16} {a.repo:<48} → {a.location}")
        return 0

    try:
        ids = expand(args.ids, rows)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2

    picked = [assets[i] for i in ids] or [a for a in rows if not a.installed]
    if not picked:
        print("every model is already installed.")
        return 0

    failed: list[tuple[str, Exception]] = []
    for a in picked:
        print(f"\n{a.title}  [{a.repo}] → {a.location}", flush=True)
        try:
            a.fetch()
        except Exception as e:  # noqa: BLE001 — one gated repo must not
            # abort the rest; every failure is reported at the end.
            failed.append((a.title, e))
            print(f"  FAILED: {e}", flush=True)

    print(flush=True)
    for title, e in failed:
        print(f"{title}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    print(f"{len(picked) - len(failed)}/{len(picked)} model(s) ready.", flush=True)
    return 1 if failed else 0
