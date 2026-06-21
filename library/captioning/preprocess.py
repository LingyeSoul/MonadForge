"""Caption correction helpers for preprocessing outputs."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

from library.captioning.correction import (
    CaptionCorrectionOptions,
    TagKnowledgeBase,
    correct_caption,
)
from library.preprocess._dataset import walk_images
from library.preprocess.caption_variants import (
    build_erasure_token_pool,
    generate_caption_variants,
    read_variants_sidecar,
    variants_sidecar_path,
    write_variants_sidecar,
)


@dataclass
class PreprocessCaptionStats:
    seen: int = 0
    written: int = 0
    unchanged: int = 0
    removed_stale: int = 0
    missing_source: int = 0
    variants_written: int = 0
    variants_removed: int = 0


def _resolve_n_rand(num_variants: int, tag_randomize_rate: float) -> int:
    """Size of the identity-randomized r-family that rides alongside v0..v{N-1}.

    Mirrors the TE writer: the r-family shares v0 as its anchor, so it carries
    ``N-1`` entries (r1..r{N-1}) and only exists with >=2 variants. Keeping this
    identical here is what lets the TE step encode the sidecar lines verbatim and
    land them on the same ``prompt_embeds_r*`` keys the loader expects.
    """
    return (num_variants - 1) if (tag_randomize_rate > 0.0 and num_variants >= 2) else 0


def _build_variant_rows(
    corrected: str,
    *,
    num_variants: int,
    tag_dropout_rate: float,
    tag_randomize_rate: float,
    erasure_pool: Collection[str] | None,
    protect_fn: Callable[[str], bool] | None,
) -> list[tuple[str, str]]:
    """``(label, text)`` rows for one image: v0..v{N-1} then r1..r{n_rand}.

    v0 is the *corrected* caption (the pristine anchor that also lives in
    ``{stem}.txt``). The r-family drops its own v0 since it equals the shared v0.
    """
    rows: list[tuple[str, str]] = []
    v_variants = generate_caption_variants(
        corrected,
        num_variants,
        tag_dropout_rate,
        protect_fn,
    )
    for i, text in enumerate(v_variants):
        rows.append((f"v{i}", text))

    n_rand = _resolve_n_rand(num_variants, tag_randomize_rate)
    if n_rand:
        r_variants = generate_caption_variants(
            corrected,
            num_variants,
            tag_dropout_rate,
            protect_fn,
            tag_randomize_rate=tag_randomize_rate,
            erasure_pool=erasure_pool,
        )
        for j, text in enumerate(r_variants[1:], start=1):  # skip shared v0 anchor
            rows.append((f"r{j}", text))
    return rows


def _sidecar_is_current(
    path: Path, corrected: str, num_variants: int, n_rand: int
) -> bool:
    """True iff an on-disk variant sidecar already matches what we'd generate.

    The variant draws are stochastic, so rewriting them every run would bump the
    sidecar mtime and force a needless TE re-encode. We treat a sidecar as
    current when its pristine v0 equals the current corrected caption *and* it
    carries the expected v/r counts — so an unchanged caption with unchanged
    variant settings is left byte-stable across reruns.
    """
    if not path.exists():
        return False
    try:
        rows = read_variants_sidecar(path)
    except OSError:
        return False
    labels = [label for label, _ in rows]
    n_v = sum(1 for label in labels if label.startswith("v"))
    n_r = sum(1 for label in labels if label.startswith("r"))
    if n_v != num_variants or n_r != n_rand:
        return False
    v0 = next((text for label, text in rows if label == "v0"), None)
    return v0 == corrected


def write_corrected_preprocess_captions(
    source_dir: Path,
    resized_dir: Path,
    kb: TagKnowledgeBase | None,
    *,
    options: CaptionCorrectionOptions,
    recursive: bool = True,
    path_pattern: str | None = None,
    correct: bool = True,
    num_variants: int = 0,
    tag_dropout_rate: float = 0.0,
    tag_randomize_rate: float = 0.0,
    qwen3_tokenizer=None,
    t5_tokenizer=None,
    protect_fn: Callable[[str], bool] | None = None,
) -> PreprocessCaptionStats:
    """Write ``.txt`` captions next to already-resized images.

    The source captions are never modified. The resized image tree is the
    authority because it already reflects low-res filtering, curation decisions,
    path_scope, and path_pattern from the resize stage.

    ``correct`` (default True) bucket-reorders each caption via
    :func:`correct_caption`; pass ``correct=False`` to mirror the raw source
    caption verbatim as v0 (the variant-only path, where the user wants shuffle
    sidecars without reordering). Either way v0 is what lands in ``{stem}.txt``
    and anchors the variant sidecar. ``kb`` is only consulted when
    ``correct=True`` — the passthrough path may pass ``None`` and skip the
    Danbooru tag-CSV dependency entirely.

    When ``num_variants > 0`` each image also gets a combined ``{stem}.variants.txt``
    sidecar holding the shuffle / tag-dropout / identity-randomize draws the TE
    step encodes verbatim — making the visible text the single source of truth
    for what trains. v0 is the corrected caption itself; v1..v{N-1} are shuffled
    (+ tag-dropped at ``tag_dropout_rate``); when ``tag_randomize_rate > 0`` an
    r-family (r1..r{N-1}) rides alongside with per-tag identity erasure.

    Identity-randomize needs the dual-single erasure pool, which is built from the
    two tokenizers (``qwen3_tokenizer`` + ``t5_tokenizer``, loaded tokenizer-only
    by the caller) excluding this dataset's real tags. Passing
    ``tag_randomize_rate > 0`` without both tokenizers is an error.
    """

    stats = PreprocessCaptionStats()
    images = walk_images(resized_dir, recursive=recursive, pattern=path_pattern)
    stats.seen = len(images)

    # First pass: resolve each image's source/destination + corrected caption.
    # Collected up front (captions are tiny) so the erasure pool can exclude the
    # full real-tag set before any variant is drawn.
    @dataclass
    class _Entry:
        src: Path
        dst: Path
        corrected: str

    entries: list[_Entry] = []
    for image_path in images:
        rel_caption = image_path.relative_to(resized_dir).with_suffix(".txt")
        src_caption = source_dir / rel_caption
        dst_caption = resized_dir / rel_caption

        if not src_caption.exists():
            stats.missing_source += 1
            if dst_caption.exists():
                dst_caption.unlink()
                stats.removed_stale += 1
            # A vanished source also orphans any variant sidecar.
            sidecar = variants_sidecar_path(dst_caption)
            if sidecar.exists():
                sidecar.unlink()
                stats.variants_removed += 1
            continue

        raw = src_caption.read_text(encoding="utf-8").strip()
        if correct:
            if kb is None:
                raise ValueError(
                    "correct=True requires a tag knowledge base "
                    "(danbooru_tags_classified.csv)."
                )
            corrected = correct_caption(raw, kb, options=options).text
        else:
            corrected = raw
        entries.append(_Entry(src_caption, dst_caption, corrected))

    n_rand = _resolve_n_rand(num_variants, tag_randomize_rate)
    erasure_pool: list[str] | None = None
    if n_rand:
        if qwen3_tokenizer is None or t5_tokenizer is None:
            raise ValueError(
                "tag_randomize_rate > 0 requires qwen3_tokenizer and t5_tokenizer "
                "(load them tokenizer-only) to build the dual-single erasure pool."
            )
        real_tags = {
            t.strip().lower()
            for e in entries
            for t in e.corrected.split(",")
            if t.strip()
        }
        erasure_pool = build_erasure_token_pool(
            qwen3_tokenizer, t5_tokenizer, exclude=real_tags
        )
        if not erasure_pool:
            raise ValueError(
                "Identity-randomize requested but the erasure-token pool is empty "
                "(tokenizers lack the expected API or no qualifying tokens)."
            )

    # Second pass: write the corrected caption + (optionally) the variant sidecar.
    for e in entries:
        if e.dst.exists() and e.dst.read_text(encoding="utf-8") == e.corrected:
            stats.unchanged += 1
        else:
            e.dst.parent.mkdir(parents=True, exist_ok=True)
            e.dst.write_text(e.corrected, encoding="utf-8")
            stats.written += 1

        sidecar = variants_sidecar_path(e.dst)
        if num_variants > 0:
            if not _sidecar_is_current(sidecar, e.corrected, num_variants, n_rand):
                rows = _build_variant_rows(
                    e.corrected,
                    num_variants=num_variants,
                    tag_dropout_rate=tag_dropout_rate,
                    tag_randomize_rate=tag_randomize_rate,
                    erasure_pool=erasure_pool,
                    protect_fn=protect_fn,
                )
                write_variants_sidecar(sidecar, rows)
                stats.variants_written += 1
        elif sidecar.exists():
            # Variants turned off → drop the now-stale sidecar.
            sidecar.unlink()
            stats.variants_removed += 1

    return stats
