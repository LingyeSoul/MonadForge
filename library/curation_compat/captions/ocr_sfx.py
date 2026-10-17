"""Speech vs SFX for an OCR line — torch-free, text-only.

Manga text is either *spoken* (a balloon, a caption box, a whisper) or a
*sound effect* drawn onto the artwork (``ぱんぱん``, ``ばるん``, ``びくっ``).
The published caption keeps them apart (``Japanese text reads as "…".
Japanese SFX reads as "…".`` — :func:`~anime_tools.captions.ocr_sidecar.with_ocr_clause`,
what Export's ``--combine_ocr`` attaches) so the address a line gets is the
kind of text it is. The split reads the string alone — no geometry — so it
applies to any reader's records. Promoted from the trainer's research tree
(``project/cjk_aware_anima/datasets/ocr_sfx.py``, the C10 ``sentence``
format) on 2026-09-07; that file now re-exports this one.

Rules, in order, on the kana core of the line (kanji / ASCII / punctuation
stripped):

1. any kanji, an empty core, or more than ``MAX_SFX_KANA`` kana → speech
   (onomatopoeia is short and kana-only);
2. a vowel / h-row / ``ん`` initial → speech: ``あっ…うっ…``, ``はぁ``,
   ``おおおん`` are a mouth, not an object;
3. a repeated unit (``ぱんぱん``, ``カリカリ``), a lexicon onset
   (``ちら``, ``きゅ``), a voiced / semi-voiced initial (``じゃぽ``,
   ``ブルン``, ``でくv``) or a sokuon initial → SFX;
4. katakana-only up to 4 kana (``ウズ``), or a sokuon-final up to 4 kana
   (``きゅっ``) → SFX; everything else → speech.

Both clauses are **deduplicated**, by a key each kind deserves.

The SFX clause collapses by sound (:func:`dedupe_sfx`, 2026-09-06 — "쥬포
쥬포쥬포 는 빼도 될듯"): a page's SFX lines become one per :func:`sfx_key` —
the kana core minus sokuon / long-vowel marks, folded to its minimal
repeating unit — keeping the first in reading order, so ``じゅぽ, じゅぽ,
じゅぽじゅぽ`` and ``ぱん♡, ぱん♡, ぱんッ`` each become one line.

The speech clause collapses by **exact text** (:func:`dedupe_speech`,
2026-09-07 — the user's call on the merge sheet): a page of panting is read
as ``はあ`` seven times and the caption said it seven times (12971620; 6.5%
of the speech lines on the sincos corpus repeat a neighbour verbatim), which
teaches a count nobody meant. Speech does *not* take the SFX key — that key
folds ``はっ`` and ``はー`` together and ``んっ♡`` into ``ん``, and two
different words of dialogue are two lines however alike they sound. Only the
same string twice is one line.

Known misses (sincos, 2026-09-05): a reader that turns ``ぱ`` into ``は``
(``はんぱん``) lands on rule 2; a voiced-initial 6-kana garble
(``がんばんがば``) lands on rule 3. Both are the reader's, not the rule's.
"""

from __future__ import annotations

import re

MAX_SFX_KANA = 6

_KANJI_RE = re.compile("[一-鿿]")
_KANA_RE = re.compile("[ぁ-んァ-ヶー]")
_KATAKANA_ONLY_RE = re.compile("[ァ-ヶー]+")

VOCAL_INITIAL = frozenset(
    "あいうえおんはひふへほぁぃぅぇぉアイウエオンハヒフヘホァィゥェォ"
)
VOICED_INITIAL = frozenset(
    "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
    "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポヴ"
)
SOKUON = frozenset("っッ")

# Onsets of common manga onomatopoeia whose initial is unvoiced (the voiced
# ones are caught by ``VOICED_INITIAL``). Matched as a prefix of the core.
SFX_LEXICON: tuple[str, ...] = (
    "ちら",
    "ちゅ",
    "ちゃぷ",
    "きゅ",
    "くちゅ",
    "くい",
    "くる",
    "こし",
    "ころ",
    "とく",
    "とろ",
    "たゆ",
    "ぷる",
    "ぴく",
    "ぴちゃ",
    "ぺろ",
    "ぺた",
    "ぱく",
    "ふに",
    "ふる",
    "ふわ",
    "むぎゅ",
    "むに",
    "もみ",
    "すり",
    "ずり",
    "しこ",
    "つん",
    "つぷ",
    "かり",
    "こく",
    "きらきら",
    "はむ",
    "れろ",
    "にゅ",
    "チラ",
    "チュ",
    "キュ",
    "クチュ",
    "トク",
    "プル",
    "ピク",
    "ペロ",
    "フワ",
    "スリ",
    "ツン",
    "カリ",
    "コク",
    "ハム",
    "レロ",
    "ニュ",
    "ムニ",
    "モミ",
)


def kana_core(text: str) -> str:
    """The kana of ``text`` in order, everything else dropped."""
    return "".join(_KANA_RE.findall(text))


_KEY_DROP = frozenset("っッー")


def sfx_key(text: str) -> str:
    """The identity of an SFX line for :func:`dedupe_sfx`: kana core, sokuon
    and long-vowel marks dropped, folded to its minimal repeating unit
    (``ぱん♡ぱん♡`` → ``ぱん``, ``びくッ`` → ``びく``). Empty when the line has
    no kana — such a line is then only ever its own duplicate."""
    core = "".join(ch for ch in kana_core(text) if ch not in _KEY_DROP)
    for n in range(1, len(core) // 2 + 1):
        if len(core) % n == 0 and core[:n] * (len(core) // n) == core:
            return core[:n]
    return core


def sfx_groups(lines: list[str]) -> list[int]:
    """For each line, the index of the first line sharing its :func:`sfx_key`
    (itself when it is the first). Keyless lines group by exact text."""
    first: dict[str, int] = {}
    out = []
    for i, ln in enumerate(lines):
        key = sfx_key(ln) or f"\0{ln}"
        out.append(first.setdefault(key, i))
    return out


def dedupe_sfx(lines: list[str]) -> list[str]:
    """``lines`` minus the later members of each :func:`sfx_groups` group,
    order kept."""
    return [
        ln
        for i, (ln, g) in enumerate(zip(lines, sfx_groups(lines), strict=True))
        if g == i
    ]


def speech_groups(lines: list[str]) -> list[int]:
    """For each line, the index of the first line with the same text once
    surrounding whitespace is off (itself when it is the first) — the speech
    counterpart of :func:`sfx_groups`, on a key that folds nothing."""
    first: dict[str, int] = {}
    return [first.setdefault(ln.strip(), i) for i, ln in enumerate(lines)]


def dedupe_speech(lines: list[str]) -> list[str]:
    """``lines`` minus every later repeat of a line already said verbatim,
    order kept (``はあ, はあ, いいわよ, はあ`` → ``はあ, いいわよ``)."""
    return [
        ln
        for i, (ln, g) in enumerate(zip(lines, speech_groups(lines), strict=True))
        if g == i
    ]


def _repeated(core: str) -> bool:
    return any(
        core[:k] * 2 == core[: 2 * k]
        for k in (1, 2, 3)
        if core[:k] and 2 * k <= len(core)
    )


def line_kind(text: str, in_bubble: bool | None = None) -> str:
    """``"sfx"`` or ``"speech"`` for one OCR line (rules in the module doc).

    ``in_bubble`` is the geometric veto when the caller has a balloon mask
    (SAM3 ``speech bubble``, ``bubble_kind.py``): a line inside a balloon is
    speech whatever it says (``カリカリ``, ``バスト91``). ``False`` / ``None``
    fall through to the text rules — a line outside every balloon is *not*
    thereby SFX (narration, floating dialogue, UI chrome), and SAM3 misses
    balloons (34 of 97 sincos pages got one, 2026-09-05).
    """
    if in_bubble:
        return "speech"
    core = kana_core(text)
    if not core or _KANJI_RE.search(text) or len(core) > MAX_SFX_KANA:
        return "speech"
    if core[0] in VOCAL_INITIAL:
        return "speech"
    if (
        _repeated(core)
        or core.startswith(SFX_LEXICON)
        or core[0] in VOICED_INITIAL
        or core[0] in SOKUON
    ):
        return "sfx"
    if len(core) <= 4 and (_KATAKANA_ONLY_RE.fullmatch(core) or core[-1] in SOKUON):
        return "sfx"
    return "speech"


def split_lines(
    lines: list[str], in_bubble: list[bool | None] | None = None
) -> tuple[list[str], list[str]]:
    """``(speech, sfx)`` in the input (reading) order; ``in_bubble`` is the
    per-line balloon veto, when known."""
    flags = in_bubble if in_bubble is not None else [None] * len(lines)
    kinds = [line_kind(ln, b) for ln, b in zip(lines, flags, strict=True)]
    speech = [ln for ln, k in zip(lines, kinds, strict=True) if k == "speech"]
    sfx = [ln for ln, k in zip(lines, kinds, strict=True) if k == "sfx"]
    return speech, sfx


__all__ = [
    "MAX_SFX_KANA",
    "SFX_LEXICON",
    "SOKUON",
    "VOCAL_INITIAL",
    "VOICED_INITIAL",
    "dedupe_sfx",
    "dedupe_speech",
    "kana_core",
    "line_kind",
    "sfx_groups",
    "sfx_key",
    "speech_groups",
    "split_lines",
]
