#!/usr/bin/env python3
"""Build a method-agnostic typed-tag index from caption sidecars.

Walks caption ``.txt`` sidecars under a source dir, classifies each
comma-separated tag into character / copyright / artist / count via the Anima
Tagger vocab (artist additionally by the ``@`` prefix, which is exact and not
limited by the vocab's frequency cutoff), and writes a single JSON index to
``post_image_dataset/captions/caption_index.json``::

    {
      "meta":  {... provenance: vocab path+mtime, src, n_images, generated ...},
      "image_meta": {
        "<stem>": {"path": "<rel>", "character": [...], "copyright": [...],
                   "artist": [...], "count": [...]},
        ...
      },
      "groups": {
        "character": {"<tag>": ["<stem>", ...], ...},
        "copyright": {...},
        "artist":    {...}
      }
    }

This is a *dataset artifact* — it lives beside the VAE/PE caches under
``post_image_dataset/`` (not in any checkpoint) and is regenerated when the
dataset or vocab changes. It encodes **no sampling policy**: how a method backs
off across the character → copyright → artist tiers is the method's own concern
(e.g. the IP-Adapter distinct-pair sampler). Consumers: the IP-Adapter
identity-pair sampler, artist balancing, dataset analytics.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_VOCAB = "models/captioners/anima-tagger-v2/vocab.json"
DEFAULT_OUT = "post_image_dataset/captions/caption_index.json"
# Axes we group on. Artist is detected by the `@` prefix (superset of the
# vocab artist list); character/copyright/count are classified by vocab
# membership, matching the Anima Tagger's category labels.
VOCAB_AXES = ("character", "copyright", "count")


def _load_vocab_sets(vocab_path: str) -> dict[str, set[str]]:
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    sets: dict[str, set[str]] = {axis: set() for axis in VOCAB_AXES}
    for entry in vocab["tags"]:
        cat = entry.get("category")
        if cat in sets:
            sets[cat].add(entry["name"].strip().lower())
    return sets


def _iter_captions(src: Path):
    """Yield ``(stem, rel_path, text)`` for every ``.txt`` under ``src``.

    ``image_dataset`` is a symlink to a tree of (possibly symlinked) artist
    dirs, so resolve the root and walk with ``followlinks=True`` — a plain walk
    descends into neither. Stems are assumed unique across the tree (the same
    invariant the stem-keyed VAE/PE caches rely on)."""
    root = Path(os.path.realpath(src))
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if not name.endswith(".txt"):
                continue
            abs_path = Path(dirpath) / name
            try:
                text = abs_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            rel = os.path.relpath(abs_path, root)
            yield name[:-4], rel, text


def _classify(text: str, vsets: dict[str, set[str]]) -> dict[str, list[str]]:
    tags = [t.strip().lower() for t in text.split(",")]
    tags = [t for t in tags if t]
    out: dict[str, list[str]] = {axis: [] for axis in (*VOCAB_AXES, "artist")}
    seen = {axis: set() for axis in out}

    def _add(axis: str, tag: str):
        if tag not in seen[axis]:
            seen[axis].add(tag)
            out[axis].append(tag)

    for tag in tags:
        if tag.startswith("@"):
            _add("artist", tag)
        for axis in VOCAB_AXES:
            if tag in vsets[axis]:
                _add(axis, tag)
    return out


def build_index(src: str, vocab_path: str) -> dict:
    vsets = _load_vocab_sets(vocab_path)
    image_meta: dict[str, dict] = OrderedDict()
    groups: dict[str, dict[str, list[str]]] = {
        axis: {} for axis in ("character", "copyright", "artist")
    }

    n_seen = 0
    for stem, rel, text in sorted(_iter_captions(Path(src))):
        n_seen += 1
        typed = _classify(text, vsets)
        if stem in image_meta:
            # Stems must be unique across the tree; surface a collision rather
            # than silently dropping one image's tags.
            raise SystemExit(
                f"duplicate caption stem {stem!r} (paths: "
                f"{image_meta[stem]['path']} vs {rel}); stems must be unique"
            )
        image_meta[stem] = {
            "path": rel,
            "character": typed["character"],
            "copyright": typed["copyright"],
            "artist": typed["artist"],
            "count": typed["count"],
        }
        for axis in ("character", "copyright", "artist"):
            for tag in typed[axis]:
                groups[axis].setdefault(tag, []).append(stem)

    # Stable ordering: stems sorted within each group, groups sorted by key.
    for axis in groups:
        groups[axis] = {
            tag: sorted(stems) for tag, stems in sorted(groups[axis].items())
        }

    vstat = os.stat(vocab_path)
    return {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "src": str(src),
            "vocab_path": vocab_path,
            "vocab_mtime": datetime.fromtimestamp(
                vstat.st_mtime, timezone.utc
            ).isoformat(timespec="seconds"),
            "n_images": n_seen,
            "axes": ["character", "copyright", "artist", "count"],
            "note": "method-agnostic typed-tag parse; sampling policy lives in method config",
        },
        "image_meta": image_meta,
        "groups": groups,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--src",
        default="image_dataset",
        help="Caption sidecar root (default: image_dataset)",
    )
    ap.add_argument(
        "--vocab",
        default=DEFAULT_VOCAB,
        help=f"Tagger vocab (default: {DEFAULT_VOCAB})",
    )
    ap.add_argument(
        "--out", default=DEFAULT_OUT, help=f"Output JSON (default: {DEFAULT_OUT})"
    )
    args = ap.parse_args()

    index = build_index(args.src, args.vocab)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    m = index["meta"]
    cov = {
        axis: sum(1 for v in index["image_meta"].values() if v[axis])
        for axis in ("character", "copyright", "artist")
    }
    n = m["n_images"] or 1
    print(f"caption index → {out}")
    print(f"  images: {m['n_images']}")
    for axis in ("character", "copyright", "artist"):
        print(
            f"  {axis:9s}: {cov[axis]:5d} imgs ({100 * cov[axis] / n:4.1f}%), "
            f"{len(index['groups'][axis]):4d} groups"
        )


if __name__ == "__main__":
    main()
