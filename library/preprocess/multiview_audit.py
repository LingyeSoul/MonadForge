"""Find images that draw one character several times but are captioned ``1girl``.

The blind spot this closes: :func:`library.preprocess.position_captions.is_candidate`
skips any caption whose girls-count is 1 and which carries no ``_LAYOUT_TAGS``
member, so a sheet that *is* ``multiple views`` but was never tagged as one never
reaches detection at all. Worse, if it did, the missing layout tag would flip
:func:`is_repeated_subject_layout` to ``False`` and the clause writer would bind
the character's name and view-invariant traits per view — asserting the same girl
as two girls.

So this is an audit over the complement set: every image ``caption-position``
throws away as ``single-subject``. Two boxes where the caption claims one girl is
the finding; three signals then argue about what it means, and a verdict needs
two of them to agree before ``--apply`` will write it:

1. **Identity agreement** across the per-instance crops (hair / eye / hairstyle,
   plus the character name). Agreeing crops are one girl drawn twice; disagreeing
   crops are a character the caption never counted. This is the only signal that
   separates those two answers — but it goes silent on a headless close-up, which
   is why it can't be the only one.
2. **The whole-image ``multiple views`` head**, which needs no crop to be legible
   and is the fallback when (1) has nothing to say.
3. **The people-count head** insisting on ``1girl`` while the geometry sees
   several bodies — the view-sheet signature stated from the other direction.

Detection differs from the clause pipeline in one deliberate way: the escalation
target is forced to ``min_instances`` (2) rather than the caption's count, so the
low-threshold retry and the body-part fallback actually fire. Gating them on the
caption's ``expected=1`` would stop the search the moment one box was found,
which is precisely the image we are trying to catch. Signal (2) also runs on
every audited image, so a sheet whose views SAM merged into a single box still
surfaces — flagged ``tagger-only`` and never better than ``weak``.

Read-only apart from :func:`apply_findings`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from PIL import Image

from library.captioning.position_clauses import (
    assign_positions,
    compose_caption,
    ordered_indices,
    parse_caption,
)

from .position_captions import (
    _GIRLS_COUNT_RE,
    _IDENTITY_GROUPS,
    ClauseVocabulary,
    Detection,
    PositionCaptionOptions,
    caption_boy_count,
    crop_instance,
    detect_subjects,
    is_candidate,
)

# The audit population, named by `is_candidate`'s own reason string. Keeping the
# link explicit means the two can't drift apart: whatever that function starts
# skipping for this reason is what we start auditing.
AUDIT_SKIP_REASON = "single-subject"

# Verdicts.
MULTIPLE_VIEWS = "multiple views"
EXTRA_CHARACTER = "extra-character"
UNSURE = "unsure"
# Not a finding: the caption's own girls+boys counts already cover every box.
COUNT_EXPLAINED = "count-explained"

# How much of the identity evidence has to agree before two boxes are called the
# same character. Set at "all comparable groups but one" so a single misread eye
# colour on a mask-blanked crop doesn't split a genuine view pair.
_SAME_CHARACTER_AGREEMENT = 0.66
# ...and how little agreement forces the opposite call. Between the two lies
# `unsure`, which the report keeps but never proposes an edit for.
_DIFFERENT_CHARACTER_AGREEMENT = 0.34
# Identity groups needed before the agreement ratio is trusted at all. One group
# is a coin flip: two crops of one girl agreeing only on `brown hair` says
# nothing, and half the corpus is brown-haired.
_MIN_COMPARABLE_GROUPS = 2
# P(multiple views) from the whole-image tagger at which it counts as a witness —
# both as the fallback when the crops carry no identity evidence, and on its own
# for an image where detection found only one box.
DEFAULT_MULTIVIEW_PROB = 0.5
# Probability an identity-group winner needs before it is believed. High on
# purpose, and measured rather than picked: on the crops of this audit's founding
# example a legible face scored 0.978-1.000 on hair/eye colour, while the
# headless backside panel's invented `blonde hair` / `blue eyes` scored 0.54 /
# 0.63. Anything in that gap is the head guessing from an empty crop.
DEFAULT_IDENTITY_CONFIDENCE = 0.9

# Which detector produced the finding.
SOURCE_DETECTION = "detection"
SOURCE_TAGGER = "tagger-only"


@dataclass
class CropIdentity:
    """What the tagger read off one crop, reduced to the identity trio + name.

    GOTCHA: ``groups`` holds only the values that cleared
    ``DEFAULT_IDENTITY_CONFIDENCE`` **on the raw probability**. The group heads
    are an argmax over a softmax, so they name a hair colour for a headless
    backside crop as confidently as for a portrait — and, being softmax winners,
    they are emitted into ``kept`` by argmax rather than by their own sigmoid
    threshold, so filtering on ``kept`` membership does nothing at all. Reading
    them ungated made the very image this audit exists for (a bent-over close-up
    beside its own full-body view) come out as two girls with different hair.
    """

    box: tuple[float, float, float, float]
    score: float
    position: str = ""
    source: str = "subject"
    # Did this box clear the detector's own confidence floor, or was it recovered
    # by the retry escalation? Only a reliable box's identity gets a vote.
    reliable: bool = True
    groups: dict[str, str] = field(default_factory=dict)
    # What the ungated argmax said, and how sure it was — so a reviewer can see
    # whether a missing identity is a shy head or a genuinely faceless crop.
    raw_groups: dict[str, str] = field(default_factory=dict)
    group_scores: dict[str, float] = field(default_factory=dict)
    name: str | None = None


@dataclass
class MultiviewFinding:
    """One audited image that looks under-counted."""

    image: str
    caption_path: str
    instances: int
    girls: int | None
    boys: int | None
    # Whether the geometry or the whole-image tagger raised this one.
    source: str = SOURCE_DETECTION
    # Whole-image corroboration, independent of the geometry: the tagger's own
    # probability for the tag we suspect is missing, and its people-count head.
    tagger_multiple_views: float | None = None
    people_count: str | None = None
    identity_agreement: float | None = None
    comparable_groups: int = 0
    crops: list[CropIdentity] = field(default_factory=list)
    verdict: str = UNSURE
    witnesses: list[str] = field(default_factory=list)
    confidence: str = "weak"
    suggested_tag: str | None = None
    caption: str = ""
    proposed: str = ""
    # Contact sheet for this row, relative to the sheets dir; "" when off.
    sheet: str = ""


@dataclass
class MultiviewAuditStats:
    seen: int = 0
    audited: int = 0
    findings: int = 0
    skipped: Counter = field(default_factory=Counter)
    verdicts: Counter = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1


def is_audit_target(caption: str) -> tuple[bool, str]:
    """Is this one of the captions ``caption-position`` never looks at?

    True exactly when :func:`is_candidate` rejects it as ``single-subject`` —
    i.e. no clauses, no layout tag, and a girls-count of 0 or 1. A count of 0
    (scenery, ``1boy``, an uncounted caption) stays in: the ``girl`` prompt
    finding two subjects there is just as much a caption bug.
    """
    ok, reason = is_candidate(caption)
    if ok:
        return False, f"handled-by-position-captions:{reason}"
    if reason != AUDIT_SKIP_REASON:
        return False, reason
    return True, AUDIT_SKIP_REASON


def _girls_count(caption: str) -> int | None:
    tags = {t.strip().lower() for t in parse_caption(caption).flat_tags}
    counts = [int(m.group(1)) for t in tags if (m := _GIRLS_COUNT_RE.match(t))]
    return max(counts) if counts else None


def identity_agreement(
    crops: Sequence[CropIdentity],
) -> tuple[float | None, int]:
    """Fraction of identity groups on which every crop agrees, and how many voted.

    Two things are excluded from the vote:

    * A crop the detector was **not confident about** (``reliable=False``, i.e.
      recovered only by the retry escalation below ``score_threshold``). Those are
      where the instance mask comes out shredded, and mask-blanking then hands the
      tagger a near-white canvas that it reads a hair colour off *at 0.99* — a
      confident answer to a question it cannot see, which the probability gate
      therefore cannot catch. Detection score is the signal that does separate
      these; mask fill measurably does not (see the settled negative in
      ``docs/experimental/position_captions.md``).
    * Any group not *every* surviving crop resolved — a headless panel reports no
      eye colour, and scoring that as disagreement would call every close-up a
      second character.
    """
    usable = [c for c in crops if c.reliable]
    if len(usable) < 2:
        return None, 0
    agree = 0
    comparable = 0
    for group in sorted(_IDENTITY_GROUPS):
        values = [c.groups.get(group) for c in usable]
        if any(v is None for v in values):
            continue
        comparable += 1
        if len(set(values)) == 1:
            agree += 1
    if not comparable:
        return None, 0
    return agree / comparable, comparable


def _verdict(
    crops: Sequence[CropIdentity],
    multiview_prob: float | None,
    multiview_threshold: float,
) -> tuple[str, float | None, int]:
    """Same character in every box, or a character the caption never counted?

    Identity first, because it is the only signal that distinguishes the two
    answers. When the crops can't supply it — a headless close-up panel resolves
    no gated identity group at all — the whole-image ``multiple views`` head is
    the fallback: it is weaker evidence, but it is evidence *for* the same-girl
    reading specifically, and its absence leaves ``unsure`` rather than a guess.
    """
    agreement, comparable = identity_agreement(crops)
    names = {c.name for c in crops if c.name and c.reliable}
    # A name disagreement is decisive on its own — two *named* characters is the
    # one signal the identity trio can't fake (two girls can share brown hair).
    if len(names) > 1:
        return EXTRA_CHARACTER, agreement, comparable
    if agreement is not None and comparable >= _MIN_COMPARABLE_GROUPS:
        if agreement >= _SAME_CHARACTER_AGREEMENT:
            return MULTIPLE_VIEWS, agreement, comparable
        if agreement <= _DIFFERENT_CHARACTER_AGREEMENT:
            return EXTRA_CHARACTER, agreement, comparable
        return UNSURE, agreement, comparable
    if (multiview_prob or 0.0) >= multiview_threshold:
        return MULTIPLE_VIEWS, agreement, comparable
    return UNSURE, agreement, comparable


def suggest_tag(
    verdict: str,
    instances: int,
    girls: int | None,
    *,
    suggest_counts: bool = False,
) -> str | None:
    """The tag the caption is missing, or ``None`` when we won't guess.

    ``multiple views`` is count-agnostic — it says "these boxes are one girl" —
    so it needs no arithmetic and is always safe to propose.

    The extra-character case is **report-only by default**. Detection counts
    *subjects*, not girls: the ``girl`` prompt does not reliably exclude males
    (which is why the clause pipeline's count gate accepts a range rather than an
    equality), so an ``Ngirls`` proposal is wrong whenever the extra body is a
    boy. Opt in with ``suggest_counts`` and read the people-count head in the
    report before applying any of it.
    """
    if verdict == MULTIPLE_VIEWS:
        return MULTIPLE_VIEWS
    if (
        suggest_counts
        and verdict == EXTRA_CHARACTER
        and girls is not None
        and instances > girls
    ):
        return f"{instances}girls"
    return None


def propose_caption(caption: str, tag: str) -> str:
    """Append ``tag`` to the end of the flat bag.

    "End of the bag", not end of the *text*: position clauses have to stay
    trailing or the grammar stops parsing them, so the splice goes through
    :func:`compose_caption` rather than a string concat. On a clause-free caption
    the two are the same thing.

    Deliberately a minimal splice rather than a re-correction — the ordering pass
    (``make preprocess-captions``) is what buckets tags properly. An ``Ngirls``
    suggestion is the exception: it *replaces* the stale count in place, since
    two girls-counts in one bag contradict each other.
    """
    parsed = parse_caption(caption)
    flat = [t for t in parsed.flat_tags if t.strip()]
    lowered = [t.strip().lower() for t in flat]
    if tag.lower() in lowered:
        return caption.strip()
    count_at = next(
        (i for i, t in enumerate(lowered) if _GIRLS_COUNT_RE.match(t)),
        None,
    )
    if _GIRLS_COUNT_RE.match(tag.lower()) and count_at is not None:
        flat[count_at] = tag
    else:
        flat.append(tag)
    return compose_caption(flat, parsed.clauses)


def audit_image(
    image: Image.Image,
    caption: str,
    *,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    tag_fn: Callable[[Image.Image], Mapping[str, object]],
    vocabulary: ClauseVocabulary,
    options: PositionCaptionOptions,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
    crop_sink: Callable[[int, str, Image.Image], str] | None = None,
    multiview_threshold: float = DEFAULT_MULTIVIEW_PROB,
    identity_confidence: float = DEFAULT_IDENTITY_CONFIDENCE,
    suggest_counts: bool = False,
) -> MultiviewFinding:
    """Detect, and when more than one subject lands, ask the tagger who they are.

    Returns a finding whose ``verdict`` is ``UNSURE`` and ``instances < 2`` for an
    image with nothing to report — the caller filters.
    """
    girls = _girls_count(caption)
    finding = MultiviewFinding(
        image="",
        caption_path="",
        instances=0,
        girls=girls,
        boys=caption_boy_count(caption),
        caption=caption.strip(),
    )

    # The whole-image pass runs unconditionally, before the geometry decides
    # anything: it is one extra forward, and it is the only thing that can flag a
    # sheet whose views SAM merged into a single box.
    whole = tag_fn(image)
    scores = whole.get("scores") or {}
    finding.tagger_multiple_views = float(scores.get(MULTIPLE_VIEWS, 0.0))
    people = whole.get("people_count")
    finding.people_count = str(people) if people is not None else None

    # expected=None, NOT caption_subject_count(): a caption claiming one girl
    # would satisfy the target on the first box and suppress both escalations.
    dets = detect_subjects(image, detect_fn, options, None, part_detect_fn)
    order = ordered_indices([d.box for d in dets], image.size, row_tol=options.row_tol)
    dets = [dets[i] for i in order]
    finding.instances = len(dets)

    # A caption whose counts already cover every detected body has nothing wrong
    # with it. `1girl, 1boy` lands in this audit only because the *girls*-count is
    # one, and the `girl` prompt does not exclude males — so the second box is the
    # boy the caption already named, not a missing tag. Same range the clause
    # pipeline's count gate uses; an unknown boy count (None) is unbounded.
    counted = None if finding.boys is None else (girls or 0) + finding.boys
    explained = counted is None or counted >= len(dets)

    if explained or len(dets) < max(2, options.min_instances):
        # Nothing for the geometry to report. The tagger alone can still raise the
        # image — a sheet whose views SAM merged into one box, or one whose second
        # view the caption's boy-count coincidentally covers — but it is a single
        # witness by construction, so it never rises above `weak`.
        if finding.tagger_multiple_views >= multiview_threshold:
            finding.source = SOURCE_TAGGER
            finding.verdict = MULTIPLE_VIEWS
            finding.witnesses = ["tagger-multiple-views"]
            finding.suggested_tag = MULTIPLE_VIEWS
            finding.proposed = propose_caption(caption, MULTIPLE_VIEWS)
        elif explained:
            finding.verdict = COUNT_EXPLAINED
        return finding

    positions = assign_positions(
        [d.box for d in dets], image.size, row_tol=options.row_tol
    )
    for index, det in enumerate(dets[: options.max_instances]):
        # Part boxes take the plain padded bbox: on a part detection the mask IS
        # the part, so blanking would delete the very pixels it recovered.
        crop = crop_instance(
            image,
            det,
            pad=options.pad,
            blank=options.blank_crops and det.source == "subject",
        )
        if crop_sink is not None:
            crop_sink(index, positions[index] or "crop", crop)
        out = tag_fn(crop)
        raw = {
            g: v
            for g, v in (out.get("groups") or {}).items()
            if g in _IDENTITY_GROUPS and v
        }
        kept = out.get("kept") or {}
        crop_scores = out.get("scores") or {}
        name = max(
            (t for t in kept if t in vocabulary.characters),
            key=lambda t: kept[t],
            default=None,
        )
        if name is not None and kept[name] < options.name_confidence:
            name = None
        finding.crops.append(
            CropIdentity(
                box=det.box,
                score=det.score,
                position=positions[index],
                source=det.source,
                reliable=det.score >= options.score_threshold,
                # Confidence-gated on the raw probability — see the CropIdentity
                # docstring for why `kept` membership is not a gate here.
                groups={
                    g: v
                    for g, v in raw.items()
                    if float(crop_scores.get(v, 0.0)) >= identity_confidence
                },
                raw_groups=raw,
                group_scores={
                    g: round(float(crop_scores.get(v, 0.0)), 4) for g, v in raw.items()
                },
                name=name,
            )
        )

    verdict, agreement, comparable = _verdict(
        finding.crops, finding.tagger_multiple_views, multiview_threshold
    )
    finding.verdict = verdict
    finding.identity_agreement = agreement
    finding.comparable_groups = comparable
    finding.suggested_tag = suggest_tag(
        verdict, finding.instances, girls, suggest_counts=suggest_counts
    )
    if finding.suggested_tag:
        finding.proposed = propose_caption(caption, finding.suggested_tag)

    # Independent witnesses to "this is one girl drawn twice". Detection alone is
    # never enough — it cannot tell two views from two girls — so `strong` means
    # at least two of the three agreed, and that is the tier `--apply` writes.
    witnesses: list[str] = []
    if comparable >= _MIN_COMPARABLE_GROUPS and (agreement or 0.0) >= (
        _SAME_CHARACTER_AGREEMENT
    ):
        witnesses.append("identity-agreement")
    if (finding.tagger_multiple_views or 0.0) >= multiview_threshold:
        witnesses.append("tagger-multiple-views")
    # The people-count head insisting on one girl while the geometry sees several
    # bodies is exactly the signature of a view sheet.
    if finding.people_count == "1girl":
        witnesses.append("people-count-1girl")
    if verdict == EXTRA_CHARACTER:
        # Different question, different witnesses: what makes an extra character
        # credible is the identity evidence splitting, not the view head firing.
        witnesses = []
        if len({c.name for c in finding.crops if c.name and c.reliable}) > 1:
            witnesses.append("distinct-names")
        if comparable >= _MIN_COMPARABLE_GROUPS and (agreement or 0.0) <= (
            _DIFFERENT_CHARACTER_AGREEMENT
        ):
            witnesses.append("identity-disagreement")
        if finding.people_count not in (None, "1girl", "no_people"):
            witnesses.append(f"people-count-{finding.people_count}")
    finding.witnesses = witnesses
    finding.confidence = "strong" if len(witnesses) >= 2 else "weak"
    return finding


def run_multiview_audit(
    *,
    resized_dir: Path,
    source_dir: Path,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    tag_fn: Callable[[Image.Image], Mapping[str, object]],
    vocabulary: ClauseVocabulary,
    options: PositionCaptionOptions | None = None,
    path_pattern: str | None = None,
    crops_dir: Path | None = None,
    sheets_dir: Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
    multiview_threshold: float = DEFAULT_MULTIVIEW_PROB,
    identity_confidence: float = DEFAULT_IDENTITY_CONFIDENCE,
    suggest_counts: bool = False,
) -> tuple[list[MultiviewFinding], MultiviewAuditStats]:
    """Walk the resized tree and report every under-counted caption.

    Same caption resolution as ``run_position_captions`` — derived caption first,
    master as the fallback for an unmirrored image — but ``caption_path`` is
    reported relative so the caller can decide which tree to edit. Nothing here
    writes.
    """
    from library.preprocess._dataset import walk_images

    options = options or PositionCaptionOptions()
    stats = MultiviewAuditStats()
    rows: list[MultiviewFinding] = []

    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)

    for index, image_path in enumerate(images, 1):
        rel = image_path.relative_to(resized_dir).with_suffix(".txt")
        dst_caption = resized_dir / rel
        caption_path = dst_caption if dst_caption.exists() else source_dir / rel
        if progress is not None:
            progress(index, len(images), str(rel))
        if not caption_path.exists():
            stats.skip("no-caption")
            continue
        caption = caption_path.read_text(encoding="utf-8").strip()
        ok, reason = is_audit_target(caption)
        if not ok:
            stats.skip(reason)
            continue
        stats.audited += 1

        # The sheet needs the exact crops the tagger read, so when sheets are on
        # we tee them into memory as they are produced (one image's worth at a
        # time) rather than re-cropping from the boxes afterwards — a plain bbox
        # re-crop would show different pixels than the mask-blanked ones scored.
        held: list[Image.Image] = []
        save_crop = None
        if crops_dir is not None:
            from .position_captions import _crop_sink

            save_crop = _crop_sink(crops_dir, rel)

        def crop_sink(i: int, pos: str, crop: Image.Image) -> str:
            if sheets_dir is not None:
                held.append(crop.copy())
            return save_crop(i, pos, crop) if save_crop is not None else ""

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        finding = audit_image(
            image,
            caption,
            detect_fn=detect_fn,
            tag_fn=tag_fn,
            vocabulary=vocabulary,
            options=options,
            part_detect_fn=part_detect_fn,
            crop_sink=crop_sink,
            multiview_threshold=multiview_threshold,
            identity_confidence=identity_confidence,
            suggest_counts=suggest_counts,
        )
        # Only the tagger can raise an image the geometry had nothing to say
        # about — too few boxes, or a caption that already counts them all.
        if finding.source != SOURCE_TAGGER:
            if finding.verdict == COUNT_EXPLAINED:
                stats.skip(COUNT_EXPLAINED)
                continue
            if finding.instances < max(2, options.min_instances):
                stats.skip("single-instance")
                continue
        finding.image = str(image_path.relative_to(resized_dir))
        finding.caption_path = str(rel)
        stats.findings += 1
        stats.verdicts[finding.verdict] += 1
        rows.append(finding)

        if sheets_dir is not None:
            from .multiview_sheet import render_contact_sheet, sheet_path

            target = sheet_path(sheets_dir, finding)
            target.parent.mkdir(parents=True, exist_ok=True)
            render_contact_sheet(image, finding, held).save(target)
            finding.sheet = str(target.relative_to(sheets_dir))

    return rows, stats


def apply_findings(
    findings: Iterable[MultiviewFinding],
    *,
    source_dir: Path,
    verdicts: Sequence[str] = (MULTIPLE_VIEWS,),
    confidences: Sequence[str] = ("strong",),
) -> list[tuple[str, str, str]]:
    """Write proposed captions into the **master** tree. Returns (rel, before, after).

    The master is the right target here, unlike the clause rewrite: a missing
    ``multiple views`` is a fact about the picture, not a derived re-phrasing, and
    every later stage (mirror, correction, TE) reads down from it.

    GOTCHA: ``image_dataset/`` is gitignored, so this is not git-recoverable —
    the caller is expected to have kept the report, whose ``caption`` field is
    the verbatim before-text.
    """
    written: list[tuple[str, str, str]] = []
    for finding in findings:
        if finding.verdict not in verdicts or finding.confidence not in confidences:
            continue
        if not finding.proposed or finding.proposed == finding.caption:
            continue
        target = source_dir / finding.caption_path
        if not target.exists():
            continue
        before = target.read_text(encoding="utf-8").strip()
        if before != finding.caption:
            continue  # caption moved under us since the audit — leave it alone
        target.write_text(finding.proposed + "\n", encoding="utf-8")
        written.append((finding.caption_path, before, finding.proposed))
    return written


def curated_proposal(row: dict) -> tuple[str, str] | None:
    """(tag, proposed caption) for a hand-accepted report row.

    Derives what the audit run didn't propose: an accepted ``unsure`` means the
    reviewer judged the sheet a view layout (same splice as ``multiple views``),
    and an accepted ``extra-character`` means the body count is real (the
    ``suggest_counts`` semantics — ``{instances}girls`` replacing the stale
    count). Returns ``None`` when no edit applies (count not derivable, or the
    tag is already present).
    """
    caption = row["caption"]
    if row["verdict"] == EXTRA_CHARACTER:
        girls, instances = row.get("girls"), row.get("instances") or 0
        if girls is None or instances <= girls:
            return None
        tag = f"{instances}girls"
    else:
        tag = MULTIPLE_VIEWS
    proposed = propose_caption(caption, tag)
    return None if proposed == caption.strip() else (tag, proposed)


def apply_curated(
    rows: Iterable[dict],
    accepted: set[str],
    *,
    source_dir: Path,
    apply: bool,
) -> tuple[list[dict], list[str]]:
    """Apply a reviewer-curated accept list of report rows to the caption master.

    Returns ``(manifest, unmatched)``: one manifest entry per accepted row with
    the verbatim before/after (the revert record — ``image_dataset/`` is
    gitignored, so this manifest is the only undo), and any accepted image the
    report has no finding for. Same drift guard as :func:`apply_findings`: a
    master caption that moved since the audit is skipped, never overwritten.
    """
    by_image = {r["image"]: r for r in rows if r.get("verdict")}
    unmatched = sorted(accepted - set(by_image))
    manifest: list[dict] = []
    for image in sorted(accepted & set(by_image)):
        row = by_image[image]
        entry = {
            "image": image,
            "caption_path": row["caption_path"],
            "verdict": row["verdict"],
            "confidence": row["confidence"],
            "tag": None,
            "before": row["caption"],
            "after": None,
            "status": "pending",
        }
        manifest.append(entry)
        derived = curated_proposal(row)
        if derived is None:
            entry["status"] = "no-edit"
            continue
        entry["tag"], entry["after"] = derived
        target = source_dir / row["caption_path"]
        if not target.exists():
            entry["status"] = "missing-caption"
            continue
        current = target.read_text(encoding="utf-8").strip()
        if current == entry["after"]:
            entry["status"] = "already-applied"
            continue
        if current != row["caption"]:
            entry["status"] = "drifted"
            continue
        if apply:
            target.write_text(entry["after"] + "\n", encoding="utf-8")
            entry["status"] = "written"
        else:
            entry["status"] = "would-write"
    return manifest, unmatched


def revert_curated(
    manifest: Iterable[dict],
    *,
    source_dir: Path,
    apply: bool,
) -> list[dict]:
    """Undo :func:`apply_curated` from its manifest. Restores ``before`` only
    where the master still holds exactly ``after`` — a caption edited since the
    apply is reported as drifted and left alone."""
    results: list[dict] = []
    for entry in manifest:
        outcome = {"image": entry["image"], "status": "skipped"}
        results.append(outcome)
        # Content-based, not status-based: the current==after check below is
        # the real guard, and it stays valid however many times apply ran.
        if not entry.get("after"):
            outcome["status"] = f"no-edit ({entry.get('status')})"
            continue
        target = source_dir / entry["caption_path"]
        if not target.exists():
            outcome["status"] = "missing-caption"
            continue
        current = target.read_text(encoding="utf-8").strip()
        if current == entry["before"]:
            outcome["status"] = "already-reverted"
        elif current != entry["after"]:
            outcome["status"] = "drifted"
        elif apply:
            target.write_text(entry["before"] + "\n", encoding="utf-8")
            outcome["status"] = "reverted"
        else:
            outcome["status"] = "would-revert"
    return results
