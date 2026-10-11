#!/usr/bin/env python
"""The KO dialogue lines: songys Chatbot_data (MIT) → the line file, the
ruler's held strings and the run's rows. CPU, seconds::

    .venv/bin/python project/cjk_ko/corpus.py

Reads ``~/ko_corpus/songys_chatbot/ChatbotData.csv`` (``Q,A,label``; the repo
``songys/Chatbot_data`` at ``4cf20d13``, its LICENSE beside it). A line is
normalised — spaces collapsed, a ``..`` / ``...`` run → ``…``, one trailing
``.`` dropped — and kept when every char is a Hangul syllable, a space, ``?``,
``!`` or ``…`` (no digit, Latin, jamo, ``~`` or other mark; no ``.`` left)
and it holds 2+ syllables; deduplicated, Q before A. Writes under
``~/ko_corpus/derived/``:

- ``chatbot_held.tsv``: the ruler's strings, ``HELD`` per syllable bin
  (``BINS``: short / mid / long), drawn at ``SEED``; ``text\\tbin\\tbook``.
- ``chatbot_lines.tsv``: every other kept line, ``line\\tbook\\tn_pieces``
  (the JA files' format; ``book`` = ``chatbot_q`` / ``chatbot_a``,
  ``n_pieces`` = Qwen pieces). A line equal to a held string, or sharing a
  5-gram with one, is left out (the builder's ``held`` rule repeats the
  5-gram cut).
- ``chatbot_rows.txt``: every syllable the line file holds, by frequency
  (``syllable\\tcount``) — a run's ``rows``.

``--rows N`` (the micro arms, user 10-11), from those files: the lines whose
every syllable is among the ``N`` most frequent → ``chatbot_lines_top<N>.tsv``,
and the read sets → ``assets/micro<N>.json``: ``trained`` (the ``N``),
``next`` (ranks N+1 … N+``NEXT``: in no line of the file, every arm's data
alike) and ``held`` (the held strings whose every syllable is in the ``N``).

``--jamo N`` (user 10-11, the J line grown in this line): the reseed line's
J256 (``assets/jamo_sets.json``) + the next ``N`` − 256 syllables of
``chatbot_rows.txt`` by frequency, KS X 1001 only, J256 and H out →
``assets/jamo<N>.json`` (``rows``: J256's order, then the added ones).
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("~/ko_corpus").expanduser()
SRC = ROOT / "songys_chatbot" / "ChatbotData.csv"
DST = ROOT / "derived"
SEED = 0
HELD = 32
BINS = {"short": (2, 4), "mid": (5, 9), "long": (10, 20)}  # syllables, the JA ruler's
NEXT = 64
HOME = Path(__file__).resolve().parent
RESEED = HOME.parent / "cjk_anima_reseed"
MARKS = set(" ?!…")
_DOTS = re.compile(r"\.{2,}|…+")


def is_syllable(c: str) -> bool:
    return 0xAC00 <= ord(c) <= 0xD7A3


def syllables(s: str) -> int:
    return sum(map(is_syllable, s))


def normalise(t: str) -> str | None:
    """``t`` as a line, or ``None`` when it is not one."""
    s = " ".join(t.split())
    s = _DOTS.sub("…", s)
    if s.endswith(".") and not s.endswith(".."):
        s = s[:-1].rstrip()
    if any(not (is_syllable(c) or c in MARKS) for c in s):
        return None
    return s if syllables(s) >= 2 else None


def grams(s: str, n: int = 5) -> set:
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from transformers import AutoTokenizer

    from library.env import resolve_under_home

    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    lines: dict = {}  # line → book
    drop = Counter()
    for side in ("Q", "A"):
        for r in rows:
            s = normalise(r[side])
            if s is None:
                drop[side] += 1
            else:
                lines.setdefault(s, f"chatbot_{side.lower()}")
    rng = random.Random(SEED)
    held = []
    for name, (lo, hi) in BINS.items():
        cands = sorted(s for s in lines if lo <= syllables(s) <= hi)
        held += [(s, name) for s in rng.sample(cands, HELD)]
    hs = {s for s, _ in held}
    h5 = set().union(*(grams(s) for s in hs))
    kept = [s for s in lines if s not in hs and not grams(s) & h5]
    tok = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/qwen3_06b")
    )
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "chatbot_held.tsv").write_text(
        "".join(f"{s}\t{b}\t{lines[s]}\n" for s, b in held), encoding="utf-8"
    )
    (DST / "chatbot_lines.tsv").write_text(
        "".join(
            f"{s}\t{lines[s]}\t{len(tok.encode(s, add_special_tokens=False))}\n"
            for s in kept
        ),
        encoding="utf-8",
    )
    freq = Counter(c for s in kept for c in s if is_syllable(c))
    (DST / "chatbot_rows.txt").write_text(
        "".join(f"{c}\t{n}\n" for c, n in freq.most_common()), encoding="utf-8"
    )
    print(
        f"{SRC}: {2 * len(rows)} texts → {len(lines)} lines (dropped Q {drop['Q']}, "
        f"A {drop['A']}); held {len(held)} "
        f"({', '.join(f'{k} {HELD}' for k in BINS)}); {len(kept)} lines kept "
        f"({len(lines) - len(hs) - len(kept)} share a held 5-gram); "
        f"{len(freq)} syllables → {DST}",
        flush=True,
    )


def micro(n: int) -> None:
    rows = [
        ln.split("\t")[0]
        for ln in (DST / "chatbot_rows.txt").read_text(encoding="utf-8").splitlines()
    ]
    top = set(rows[:n])
    src = (DST / "chatbot_lines.tsv").read_text(encoding="utf-8").splitlines()
    kept = [
        ln for ln in src if all(c in top for c in ln.split("\t")[0] if is_syllable(c))
    ]
    dst = DST / f"chatbot_lines_top{n}.tsv"
    dst.write_text("".join(f"{ln}\n" for ln in kept), encoding="utf-8")
    held = [
        ln.split("\t")[:2]
        for ln in (DST / "chatbot_held.tsv").read_text(encoding="utf-8").splitlines()
    ]
    sets = {
        "trained": "".join(rows[:n]),
        "next": "".join(rows[n : n + NEXT]),
        "held": [[s, b] for s, b in held if all(c in top for c in s if is_syllable(c))],
    }
    out = HOME / "assets" / f"micro{n}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(sets, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"top {n}: {len(kept)} of {len(src)} lines → {dst}; reads → {out}: "
        f"{n} trained, {len(sets['next'])} next, {len(sets['held'])} held "
        f"({dict(Counter(b for _, b in sets['held']))})",
        flush=True,
    )


def jamo(n: int) -> None:
    sys.path.insert(0, str(RESEED))
    from reseed.jamo import in_ksx1001

    sets = json.loads(
        (RESEED / "assets" / "jamo_sets.json").read_text(encoding="utf-8")
    )
    base, out_ = list(sets["J256"]), set(sets["J256"]) | set(sets["H"])
    rows = [
        ln.split("\t")[0]
        for ln in (DST / "chatbot_rows.txt").read_text(encoding="utf-8").splitlines()
    ]
    added = [c for c in rows if c not in out_ and in_ksx1001(c)][: n - len(base)]
    assert len(added) == n - len(base), len(added)
    out = HOME / "assets" / f"jamo{n}.json"
    out.write_text(
        json.dumps(
            {"rows": "".join(base + added), "added": "".join(added)},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"J{n}: J256 + {len(added)} by chatbot frequency → {out}", flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=0, help="the micro arms' line file")
    p.add_argument("--jamo", type=int, default=0, help="J256 + chatbot rows to N")
    a = p.parse_args()
    jamo(a.jamo) if a.jamo else micro(a.rows) if a.rows else main()
