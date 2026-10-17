"""Caption-variant generation + on-disk variant sidecars (torch-free).

* :func:`generate_caption_variants` — the stochastic shuffle / tag-dropout /
  identity-randomize expansion for train-time caption sampling, materialized as
  text **before** the TE encoder is loaded.
* :func:`build_erasure_token_pool` — the dual-single erasure pool the
  identity-randomize axis draws from.
* the ``{stem}.variants.txt`` sidecar read/write pair: the caption step writes
  it, the TE step encodes exactly its lines, the GUI previews it. The file shape
  is :mod:`anime_tools.captions._sidecar`'s; the vocabulary is local — a record
  is a ``(label, text)`` pair whose ``v*``/``r*`` labels key the TE cache layout.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Collection
from pathlib import Path

from library.curation_compat.captions._sidecar import (
    read_rows,
    sidecar_header,
    sidecar_path,
    write_rows,
)
from library.curation_compat.captions.position_clauses import is_clause_header

VARIANTS_SIDECAR_SUFFIX = ".variants.txt"
VARIANTS_FIELDS = 2
"""``label ⇥ text`` — and the text may hold tabs, being last."""


def build_erasure_token_pool(
    qwen3_tokenizer,
    t5_tokenizer,
    *,
    exclude: Collection[str] | None = None,
    min_len: int = 4,
    max_len: int = 9,
) -> list[str]:
    """Pool of tokens used to *erase a tag's identity* while keeping its slot
    (lexinvariant tag regularization, arXiv:2305.16349).

    A randomized slot is filled by a *different* pool token each draw, so the
    literal symbol stays unreliable. The filler is a *real* vocab token, not
    gibberish. Distinct from tag-dropout, which removes the slot.

    **Dual-single** is the load-bearing constraint: Anima tokenizes captions
    twice (Qwen3 for the text encoder, T5 for the LLM adapter's target ids), so
    a filler must be exactly one token in *both* vocabularies or it shreds into
    subword pieces. Keeps lowercase ascii alphabetic words single in both;
    ``exclude`` (the dataset's real tags) drops words that are genuine tags.
    Returns bare words (no leading space).

    Selection keys off the ``Ġ`` (leading-space) Qwen3 vocab form — exact for
    this token class, so only the T5 single-token property is checked at
    runtime.
    """
    excl = {t.lower() for t in exclude} if exclude else set()
    try:
        vocab = qwen3_tokenizer.get_vocab()
    except (AttributeError, TypeError):
        # Duck-typed tokenizer without the expected API — no pool, no variants.
        return []
    # Qwen3-single candidates: leading-space lowercase ascii.
    candidates = sorted(
        sym[1:]
        for sym in vocab
        if sym.startswith("Ġ")
        and (core := sym[1:]).isascii()
        and core.isalpha()
        and core.islower()
        and min_len <= len(core) <= max_len
        and core not in excl
    )
    # Keep only those that are also a single T5 token in a ``", "``-joined caption.
    try:
        t5_base = len(t5_tokenizer(", ", add_special_tokens=False)["input_ids"])
    except (AttributeError, KeyError, TypeError):
        return []
    pool: list[str] = []
    for word in candidates:
        ids = t5_tokenizer(", " + word, add_special_tokens=False)["input_ids"]
        if len(ids) - t5_base == 1:
            pool.append(word)
    return pool


def _perturb_tags(
    tags: list[str],
    split_idx: int,
    *,
    tag_dropout_rate: float,
    tag_randomize_rate: float,
    protect_fn: Callable[[str], bool] | None,
    pool: list[str] | None,
    sentinel: str,
) -> list[str]:
    """Apply the presence (dropout) then identity (randomize) axes to ``tags``.

    ``split_idx`` is the @artist-prefix boundary: both axes leave ``tags`` up to
    it untouched. ``split_idx=0`` is how a clause body is perturbed.
    """
    if tag_dropout_rate > 0.0 and len(tags) > split_idx:
        kept = list(tags[:split_idx])
        for tag in tags[split_idx:]:
            if (protect_fn is not None and protect_fn(tag)) or (
                random.random() >= tag_dropout_rate
            ):
                kept.append(tag)
        if not kept:
            kept = tags[:1]
        tags = kept
    if tag_randomize_rate > 0.0:
        tags = [
            random.choice(pool)
            if (
                i >= split_idx
                and tag != sentinel
                and not is_clause_header(tag)
                and not (protect_fn is not None and protect_fn(tag))
                and random.random() < tag_randomize_rate
            )
            else tag
            for i, tag in enumerate(tags)
        ]
    return tags


def generate_caption_variants(
    caption: str,
    num_variants: int,
    tag_dropout_rate: float,
    protect_fn: Callable[[str], bool] | None = None,
    tag_randomize_rate: float = 0.0,
    erasure_pool: Collection[str] | None = None,
    clause_dropout_rate: float | None = None,
) -> list[str]:
    """Generate ``num_variants`` caption variants for stochastic train-time sampling.

    v0 = pristine original caption. v1..v{N-1} are smart-shuffled (preserving
    the @artist prefix and section anchors), then every tag *after* the prefix
    is independently dropped at ``tag_dropout_rate``, then every survivor has
    its identity erased (a fresh token from ``erasure_pool``) at
    ``tag_randomize_rate``. The ``@no-artist`` sentinel participates in the
    boundary but is stripped from every variant, v0 included.

    Both axes are **prefix-protected**: neither touches the @artist prefix, so
    the trigger tag stays intact. Clause headers and the sentinel are never
    randomized. ``protect_fn`` marks tags that must survive both axes verbatim
    even past the prefix (still shuffled).

    ``erasure_pool`` is **required** whenever ``tag_randomize_rate > 0``.

    **Position clauses are atomic**: each clause is kept or dropped *whole* at
    ``clause_dropout_rate`` (default: ``tag_dropout_rate``), with its tags
    shuffled inside — per-tag dropout would leave a half-described position.
    **Text clauses** (``Japanese text reads as "…", "…"``) ride every variant
    verbatim: never dropped, shuffled or randomized — the lines' reading
    order is content, and the sentence is the page's text address.
    """
    from library.curation_compat.captions import shuffle as anima_train_utils
    from library.curation_compat.captions.position_clauses import (
        PositionClause,
        compose_caption,
        parse_caption,
    )

    sentinel = anima_train_utils.NO_ARTIST_SENTINEL
    if tag_randomize_rate > 0.0 and not erasure_pool:
        raise ValueError(
            "tag_randomize_rate > 0 requires a non-empty erasure_pool "
            "(build_erasure_token_pool); there is no random-ASCII fallback."
        )
    pool = list(erasure_pool) if erasure_pool else None
    clause_rate = (
        tag_dropout_rate if clause_dropout_rate is None else float(clause_dropout_rate)
    )

    parsed = parse_caption(caption)
    if parsed.has_clauses:
        tags = list(parsed.flat_tags)
    else:
        # No clauses: keep the historical raw split so v0 stays byte-identical
        # (parse_caption normalizes whitespace around commas).
        tags = [t.strip() for t in caption.split(",")]
    split_idx = anima_train_utils.find_anima_prefix_end(tags)

    # v0 stays byte-identical to the source caption unless the sentinel is present
    # — re-joining would otherwise normalize whitespace around commas.
    if sentinel in tags:
        stripped = anima_train_utils.strip_no_artist_sentinel(tags)
        variants = [compose_caption(stripped, parsed.clauses)]
    else:
        variants = [caption]

    for _ in range(max(0, num_variants - 1)):
        shuffled = anima_train_utils.anima_smart_shuffle_caption(tags.copy())
        shuffled = _perturb_tags(
            shuffled,
            split_idx,
            tag_dropout_rate=tag_dropout_rate,
            tag_randomize_rate=tag_randomize_rate,
            protect_fn=protect_fn,
            pool=pool,
            sentinel=sentinel,
        )
        shuffled = anima_train_utils.strip_no_artist_sentinel(shuffled)

        clauses: list[PositionClause] = []
        for clause in parsed.clauses:
            if clause.is_text:
                clauses.append(clause)
                continue
            protected = protect_fn is not None and any(
                protect_fn(t) for t in clause.tags
            )
            if not protected and clause_rate > 0.0 and random.random() < clause_rate:
                continue  # atomic: the whole clause goes, header included
            body = list(clause.tags)
            random.shuffle(body)
            body = _perturb_tags(
                body,
                0,
                tag_dropout_rate=0.0,  # clauses are all-or-nothing
                tag_randomize_rate=tag_randomize_rate,
                protect_fn=protect_fn,
                pool=pool,
                sentinel=sentinel,
            )
            clauses.append(
                PositionClause(
                    position=clause.position, tags=tuple(body), prefix=clause.prefix
                )
            )
        variants.append(compose_caption(shuffled, clauses))
    return variants


def variants_sidecar_path(image_or_caption_path: Path) -> Path:
    """``{stem}.variants.txt`` next to a resized image (or its ``.txt`` caption).

    A multi-dot stem is preserved (``a.b.png`` → ``a.b.variants.txt``).
    """
    return sidecar_path(image_or_caption_path, VARIANTS_SIDECAR_SUFFIX)


def write_variants_sidecar(path: Path, variants: list[tuple[str, str]]) -> None:
    """Write a combined variant sidecar — one ``label\\ttext`` line per variant.

    ``variants`` is an ordered ``(label, text)`` list (``v0``, ``v1`` …, then
    ``r1`` …); v0 is the pristine/corrected caption that also lives in
    ``{stem}.txt``. An empty list still writes the file — the TE step encodes
    exactly these lines, so "no variants" is a statement, not an absence.
    """
    write_rows(path, sidecar_header("variants"), variants)


def read_variants_sidecar(path: Path) -> list[tuple[str, str]]:
    """Parse a ``{stem}.variants.txt`` into an ordered ``(label, text)`` list.

    ``path`` must exist — a missing sidecar is a bug in the caller, not an empty
    variant set. Records arrive with the format's own hand-edit tolerance. Order
    is preserved so the TE writer can map ``v*``/``r*`` labels straight onto its
    flat cache layout.
    """
    return [(label.strip(), text) for label, text in read_rows(path, VARIANTS_FIELDS)]
