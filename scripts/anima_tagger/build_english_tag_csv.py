"""Build an English sibling of ``models/danbooru_tags_classified.csv``.

The shipped tag KB (``danbooru_tags_classified.csv``) comes from
``Localsmile/danbooru_KR_wiki_tag_search`` — Danbooru's English tag taxonomy
with **Korean** wiki descriptions. That keeps the GUI tag-explanation tooltip
gated to the Korean UI (see ``gui/tabs/image_tab.py::_on_tag_clicked`` and
``CONTRIBUTING.md`` §5).

This script regenerates the ``description`` column in English by joining the
existing CSV's tag names against the upstream the Korean repo itself translated
from — the Danbooru wiki, mirrored as ``isek-ai/danbooru-wiki-2024`` on the Hub.
It keeps ``name`` / ``category`` / ``post_count`` byte-for-byte from the base CSV
(so tag classification stays identical) and only swaps the description, writing
``models/danbooru_tags_classified.en.csv``.

Usage::

    python -m scripts.anima_tagger.build_english_tag_csv          # default paths
    python -m scripts.anima_tagger.build_english_tag_csv --revision 202408-at20240906

Requires the ``datasets`` library (``uv sync`` / ``uv pip install datasets``).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SRC = _REPO_ROOT / "models" / "danbooru_tags_classified.csv"
_DEFAULT_DST = _REPO_ROOT / "models" / "danbooru_tags_classified.en.csv"
_HF_REPO = "isek-ai/danbooru-wiki-2024"

# ── DText → plain-text cleanup ────────────────────────────────────────────────
# Danbooru wiki bodies are DText, not MediaWiki. Strip the markup down to the
# bare prose so it renders cleanly in the Qt tooltip.
_WIKI_LINK_LABELLED = re.compile(r"\[\[[^\]|]+\|([^\]]+)\]\]")  # [[tag|label]] -> label
_WIKI_LINK = re.compile(r"\[\[([^\]]+)\]\]")  # [[tag]] -> tag
_SEARCH_LINK = re.compile(r"\{\{([^}]+)\}\}")  # {{search}} -> search
_NAMED_URL_BRACKET = re.compile(r'"([^"]+)":\[[^\]]+\]')  # "label":[url] -> label
_NAMED_URL = re.compile(r'"([^"]+)":https?://\S+')  # "label":url -> label
_BBCODE = re.compile(
    r"\[/?(?:b|i|u|s|tn|quote|code|spoiler|expand)(?:=[^\]]*)?\]", re.IGNORECASE
)
_HEADING = re.compile(r"(?m)^h[1-6]\.\s*")
_LIST_MARKER = re.compile(r"(?m)^\s*[*]+\s*")
_WS = re.compile(r"\s+")


def clean_dtext(body: str, *, max_len: int = 600) -> str:
    """Reduce a DText wiki body to a concise one-paragraph English blurb."""
    if not body:
        return ""
    # The first paragraph is the definition; trailing paragraphs are usage notes
    # / related-tag dumps that bloat the tooltip. Keep the definition.
    paragraph = body.strip().split("\n\n", 1)[0]
    text = _WIKI_LINK_LABELLED.sub(r"\1", paragraph)
    text = _WIKI_LINK.sub(r"\1", text)
    text = _SEARCH_LINK.sub(r"\1", text)
    text = _NAMED_URL_BRACKET.sub(r"\1", text)
    text = _NAMED_URL.sub(r"\1", text)
    text = _BBCODE.sub("", text)
    text = _HEADING.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _WS.sub(" ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0].rstrip() + "…"
    return text


def _norm_key(tag: str) -> str:
    return tag.strip().replace(" ", "_").lower()


def load_wiki_descriptions(revision: str | None) -> dict[str, str]:
    """tag-key -> cleaned English description from the Danbooru wiki mirror."""
    try:
        from datasets import load_dataset
    except ImportError:  # pragma: no cover - dependency guard
        raise SystemExit(
            "The 'datasets' library is required. Install it with "
            "`uv pip install datasets` (or `uv sync`)."
        )

    print(f"  loading {_HF_REPO}" + (f"@{revision}" if revision else "") + " …")
    ds = load_dataset(_HF_REPO, split="train", revision=revision)
    out: dict[str, str] = {}
    for row in ds:
        if row.get("is_deleted"):
            continue
        body = clean_dtext(row.get("body") or "")
        if not body:
            continue
        # `tag` is the canonical tag; `title` mirrors it. Index both for join.
        for field in ("tag", "title"):
            name = row.get(field)
            if name:
                out.setdefault(_norm_key(name), body)
    print(f"  {len(out):,} tags carry an English description")
    return out


def build(src: Path, dst: Path, revision: str | None) -> None:
    if not src.exists():
        raise SystemExit(
            f"source CSV not found: {src}\n"
            "Run `python tasks.py download-danbooru-tags` first."
        )
    descriptions = load_wiki_descriptions(revision)

    total = matched = 0
    with (
        src.open("r", encoding="utf-8-sig", newline="") as fin,
        dst.open("w", encoding="utf-8", newline="") as fout,
    ):
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(
            fout, fieldnames=["name", "category", "post_count", "description"]
        )
        writer.writeheader()
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            total += 1
            desc = descriptions.get(_norm_key(name), "")
            if desc:
                matched += 1
            writer.writerow(
                {
                    "name": name,
                    "category": row.get("category", ""),
                    "post_count": row.get("post_count", ""),
                    "description": desc,
                }
            )

    pct = (100.0 * matched / total) if total else 0.0
    print(f"  ✓ wrote {dst}")
    print(f"  {matched:,}/{total:,} rows have an English description ({pct:.1f}%)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=_DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=_DEFAULT_DST)
    ap.add_argument(
        "--revision",
        default=None,
        help="git tag/revision of isek-ai/danbooru-wiki-2024 (default: latest)",
    )
    args = ap.parse_args(argv)
    build(args.src, args.dst, args.revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
