"""Caption-variant generation + on-disk variant sidecars (torch-free).

Two concerns live here, both deliberately free of any torch / model import so the
caption-correction step and the GUI can reuse them without dragging the heavy
deps into their process:

* :func:`generate_caption_variants` — the stochastic shuffle / tag-dropout /
  identity-randomize expansion used for train-time caption sampling. Extracted
  from ``text.py`` (which still re-exports it for backward-compat) so the
  preprocess *caption* step can materialize variants as text **before** the TE
  encoder is ever loaded.
* :func:`build_erasure_token_pool` — the dual-single erasure pool the
  identity-randomize axis draws from. Takes the two tokenizers as plain
  arguments (no torch), so the caption step can build it from a tokenizer-only
  load.
* the ``{stem}.variants.txt`` sidecar read/write pair — the combined,
  human-readable file that makes the generated variants the single source of
  truth: the caption step writes it, the TE step encodes exactly its lines, and
  the GUI previews it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Collection
from pathlib import Path

VARIANTS_SIDECAR_SUFFIX = ".variants.txt"
_SIDECAR_HEADER = "# anima caption variants — auto-generated, do not hand-edit"


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
    literal symbol stays unreliable — pushing the adapter to ground the concept
    in surrounding structure (image content + co-occurring tags) rather than the
    exact trigger vector (role-based, not vector-based, binding). Per the cited
    method the filler is a *real* vocab token, not gibberish: unreliability comes
    from the per-draw variation, not from the symbol being intrinsically
    meaningless. Distinct from tag-dropout (which removes the slot).

    **Dual-single** is the load-bearing constraint. Anima tokenizes captions
    twice — Qwen3 for the text encoder (``prompt_embeds``) *and* T5 for the LLM
    adapter's target ids (``crossattn_emb``). A filler must therefore be exactly
    one token in *both* vocabularies; otherwise it shreds into junk subword
    pieces on whichever side fragments it (random ASCII fragments in Qwen3; rare
    foreign tokens are clean in Qwen3 but explode in T5's English-centric
    sentencepiece). We pick lowercase ascii alphabetic words and keep only those
    that survive as a single token in both. ``exclude`` (the dataset's real tags)
    drops any word that is itself a genuine tag, so a filler is never mistakable
    for a true concept. Returns bare words (no leading space).

    Selection keys off the ``Ġ`` (leading-space) Qwen3 vocab form — exact for
    this token class (verified: heuristic == round-trip), so no Qwen3 re-encode
    is needed; only the T5 single-token property is checked at runtime.
    """
    excl = {t.lower() for t in exclude} if exclude else set()
    try:
        vocab = qwen3_tokenizer.get_vocab()
    except Exception:
        return []
    # Qwen3-single candidates (cheap, pure vocab): leading-space lowercase ascii.
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
    except Exception:
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
    it untouched. Shared by the flat tag bag and — with ``split_idx=0`` — the
    body of each surviving position clause.
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
                and not tag.startswith(("On the ", "In the "))
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
    is independently dropped with probability ``tag_dropout_rate``, then every
    surviving tag *after the prefix* has its identity erased (replaced by a fresh
    vocab token drawn from ``erasure_pool``) with probability
    ``tag_randomize_rate``. The ``@no-artist`` sentinel participates in the
    boundary but is stripped from every variant (including v0) before it is
    written.

    ``protect_fn`` (when given) marks tags that must survive tag-dropout *and*
    tag-randomization: a tag for which it returns True is always kept verbatim
    even past the @artist prefix. It is still subject to shuffling. Used by the
    colorize prep to keep copyright tags present/intact in every variant.

    Both axes are **prefix-protected**: tag-dropout is the *presence* axis and
    tag-randomize is the *identity* axis, and neither touches the @artist prefix
    (the trigger tag stays intact, only tags *after* ``split_idx`` are
    randomized). Section headers (``On the …`` / ``In the …``) and the sentinel
    are never randomized either.

    ``erasure_pool`` (see :func:`build_erasure_token_pool`) is the source of
    erasure symbols: each randomized slot draws a fresh dual-single vocab token
    (clean one-token in both Qwen3 and T5). It is **required** whenever
    ``tag_randomize_rate > 0`` (no random-ASCII fallback); ignored otherwise.

    **Position clauses are atomic.** A caption carrying the ``…, white socks.
    On the left, blonde hair.`` convention is parsed into its flat bag plus its
    clauses (:mod:`library.captioning.position_clauses`) and each clause is kept
    or dropped *whole* at ``clause_dropout_rate`` (default: ``tag_dropout_rate``),
    with its tags shuffled inside. Per-tag dropout inside a clause would leave a
    half-described position, and — worse — the naive comma split glues the header
    onto the preceding tag (``"white socks. On the left"``), which is what used
    to scatter clause attributes across the whole caption and reassign them to
    the wrong subject.
    """
    from library.anima import training as anima_train_utils
    from library.captioning.position_clauses import (
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

    Uses ``with_name`` (not ``with_suffix``) so a multi-dot stem is preserved and
    the marker double-suffix lands cleanly: ``a.b.png`` → ``a.b.variants.txt``.
    """
    p = image_or_caption_path
    stem = p.name[: -len(p.suffix)] if p.suffix else p.name
    return p.with_name(stem + VARIANTS_SIDECAR_SUFFIX)


def write_variants_sidecar(path: Path, variants: list[tuple[str, str]]) -> None:
    """Write a combined variant sidecar — one ``label\\ttext`` line per variant.

    ``variants`` is an ordered ``(label, text)`` list (``v0``, ``v1`` …, then
    ``r1`` …). v0 is the pristine/corrected caption that also lives in
    ``{stem}.txt``; the rest are the shuffled / dropped / randomized draws the TE
    step encodes verbatim. Tab-delimited because captions are comma-separated and
    never contain tabs, so the split is unambiguous.
    """
    lines = [_SIDECAR_HEADER]
    for label, text in variants:
        lines.append(f"{label}\t{text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_variants_sidecar(path: Path) -> list[tuple[str, str]]:
    """Parse a ``{stem}.variants.txt`` into an ordered ``(label, text)`` list.

    Comment (``#``) and blank lines are skipped; a line without a tab is ignored
    (defensive against hand-edits). Order is preserved so the TE writer can map
    ``v*``/``r*`` labels straight onto its flat cache layout.
    """
    out: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\r")
        if not line or line.lstrip().startswith("#"):
            continue
        label, sep, text = line.partition("\t")
        if not sep:
            continue
        out.append((label.strip(), text))
    return out
