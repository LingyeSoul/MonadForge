"""Subject instances: box geometry, duplicate suppression, mask-blanked crops.

The detector-side primitives of the position-clause pipeline
(:mod:`library.preprocess.position_captions`) and the multiview audit — a
:class:`Detection` record plus the pure geometry around it (IoU / containment /
area), the NMS pass that turns raw detector output into one box per subject, the
body-part fallback merge, and the crop the tagger actually sees.

Detector-agnostic: nothing here imports SAM3. The caller supplies detections;
these functions only reason about boxes, masks, and pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Detection:
    """One detected subject: box in pixels, score, and optional instance mask.

    ``source`` records which detector pass produced the box — ``"subject"`` for
    the ``girl`` prompt, the part prompt itself for a body-part fallback box.
    Carried into the report so a reviewer can tell the two apart.
    """

    box: tuple[float, float, float, float]
    score: float
    mask: np.ndarray | None = None
    source: str = "subject"


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_containment(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over the *smaller* box — how nested the pair is.

    IoU is blind to nesting: a box wholly inside another scores tiny (`area_small
    / area_large`), so an inset icon and a group box spanning every subject both
    hide from it while scoring ~1.0 here.

    GOTCHA: suppressing on this is off by default — a *real* second subject
    (one girl in front of another) is just as nested as a group box, and
    ablation showed far more of the former in this corpus than the latter.
    Kept as an opt-in knob; :func:`drop_small_boxes` handles the inset case
    instead, and a surviving group box only costs one `count-mismatch` skip.
    """
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smallest = min(box_area(a), box_area(b))
    return (ix * iy) / smallest if smallest > 0 else 0.0


def _binary_mask(det: Detection) -> np.ndarray | None:
    """The detection's mask as a 2-D boolean array, or ``None`` if it has none."""
    if det.mask is None:
        return None
    mask = np.asarray(det.mask)
    while mask.ndim > 2:
        mask = mask[0]
    return mask > 0.5


def mask_containment(a: Detection, b: Detection) -> float | None:
    """Intersection over the *smaller* mask — how nested the pair really is.

    The mask analogue of :func:`box_containment`, and the discriminator that one
    cannot provide: two boxes nest identically whether the inner detection is a
    fragment of the outer figure or a second girl standing in front of her, but
    their *masks* do not. A fragment's mask is a subset of the whole figure's;
    an occluding subject's mask is disjoint from the figure behind it, because
    SAM3 segments the two separately.

    Every box-nested pair in the pair-level probe landed either above 0.98
    (genuinely one object) or below 0.02 (two subjects) — no middle ground, so
    the 0.8 default is not a tuned edge.

    KNOWN FAILURE: when SAM3 emits a *group* mask spanning both girls, an
    individual's mask is a subset of it and gets suppressed. That is the mask
    analogue of the group-box problem and it cost 2 of 480 candidate rows; see
    ``docs/experimental/position_captions.md``.

    ``None`` when either detection carries no mask, or when the two masks are
    differently shaped, so callers fall back to box-only behaviour — the same
    convention :func:`mask_box_fill` uses for the fill tie-break.
    """
    ma, mb = _binary_mask(a), _binary_mask(b)
    if ma is None or mb is None or ma.shape != mb.shape:
        return None
    smallest = min(float(ma.sum()), float(mb.sum()))
    if smallest <= 0:
        return None
    return float(np.logical_and(ma, mb).sum()) / smallest


def mask_box_fill(det: Detection) -> float | None:
    """Fraction of the detection's own box that its mask actually claims.

    ``None`` when the detection carries no mask (stub tests, part boxes) so
    callers can fall back to score-only behaviour.
    """
    if det.mask is None:
        return None
    mask = np.asarray(det.mask)
    if mask.ndim == 3:
        mask = mask[0]
    height, width = mask.shape
    x1, y1, x2, y2 = det.box
    window = (
        mask[
            max(0, int(y1)) : min(height, int(y2)),
            max(0, int(x1)) : min(width, int(x2)),
        ]
        > 0.5
    )
    return float(window.mean()) if window.size else 0.0


def dedupe_detections(
    detections: Iterable[Detection],
    iou_threshold: float,
    containment_threshold: float = 1.01,
    fill_ratio_threshold: float = 0.0,
    mask_containment_threshold: float = 1.01,
) -> list[Detection]:
    """Greedy IoU + containment suppression, highest score first.

    A threshold above 1.0 disables either containment rule — nothing can be more
    than fully inside something else — leaving plain-IoU behaviour.

    ``mask_containment_threshold`` suppresses on :func:`mask_containment`
    instead of box geometry, and unlike the box rule it ships **on** (0.8). The
    box rule is a settled negative because it cannot tell a fragment from a real
    second subject; the mask rule mostly can, and the full candidate ledger
    bears that out — 480 candidates, **7 rows recovered, 2 broken** (the box
    rule's ledger was 2 recovered, 34 broken). Every recovery landed on the
    caption's own girls-count, which is an independent corroboration that the
    merge produced the *right* number and not merely a smaller one. A pair with
    no usable mask falls back to the box rules, so stub detections and part
    boxes are unaffected.

    ``fill_ratio_threshold`` > 0 enables the mask-quality tie-break
    (``docs/experimental/multiview_audit.md`` §5.4 fixed the default at
    2.0): when a candidate collides with a kept box — the pair already judged
    to be the same object — and the candidate's :func:`mask_box_fill` beats the
    kept box's by at least this ratio, the candidate *replaces* the kept box
    instead of being dropped. SAM3's score is box-level confidence and says
    nothing about mask coherence, so a near-empty duplicate can outscore the
    clean mask by a hair and hand every downstream consumer a blank crop.
    Instance count is invariant by construction — the swap only changes which
    of two matched duplicates represents the object. This is deliberately NOT
    an absolute fill gate (settled negative: clean figures live at fill ~0.27);
    the ratio only ever compares the two halves of one matched pair. When
    either mask is missing, the pair falls back to score-only suppression.
    Single-pass: the swapped-in geometry is not re-checked against other kept
    boxes (bounded, order-stable; no cascade observed over the full corpus).
    """

    def matches(det: Detection, kept: Detection) -> bool:
        if box_iou(det.box, kept.box) >= iou_threshold:
            return True
        if box_containment(det.box, kept.box) >= containment_threshold:
            return True
        if mask_containment_threshold > 1.0:
            return False
        overlap = mask_containment(det, kept)
        return overlap is not None and overlap >= mask_containment_threshold

    ranked = sorted(detections, key=lambda d: -d.score)
    keep: list[Detection] = []
    for det in ranked:
        hit = next(
            (i for i, k in enumerate(keep) if matches(det, k)),
            None,
        )
        if hit is None:
            keep.append(det)
            continue
        if fill_ratio_threshold > 0:
            kept_fill = mask_box_fill(keep[hit])
            det_fill = mask_box_fill(det)
            if (
                kept_fill is not None
                and det_fill is not None
                and det_fill / max(kept_fill, 1e-9) >= fill_ratio_threshold
            ):
                keep[hit] = det
    return keep


def merge_part_detections(
    subjects: Sequence[Detection],
    parts: Iterable[Detection],
    *,
    iou_threshold: float,
    containment_threshold: float,
) -> list[Detection]:
    """Add body-part boxes that the subject prompt missed, never displacing one.

    Recovers a sheet whose panels are headless close-ups (hip/crotch/backside)
    that the `girl` prompt can't see.

    Containment is applied here even though :func:`dedupe_detections` leaves it
    off by default — the asymmetry is deliberate. A *part* nested in a subject
    is never a real second subject (unlike two subjects nested in each other),
    it's that subject's own body, so typing the rule to the part pass gets
    duplicate suppression without the false positives the global rule costs.

    Subjects are kept unconditionally; parts are considered highest-score first
    against everything kept so far, so duplicate part boxes on one panel
    collapse to one.
    """
    keep = list(subjects)
    for det in sorted(parts, key=lambda d: -d.score):
        if any(
            box_iou(det.box, k.box) >= iou_threshold
            or box_containment(det.box, k.box) >= containment_threshold
            for k in keep
        ):
            continue
        keep.append(det)
    return keep


def drop_small_boxes(
    detections: Iterable[Detection],
    image_size: tuple[int, int],
    min_area_frac: float,
) -> list[Detection]:
    """Discard boxes too small to be a bindable subject.

    A detection covering 0.3% of the canvas is an inset — a character drawn on a
    phone screen, a poster, a chibi in a corner — not a subject a position clause
    can meaningfully describe.
    """
    if min_area_frac <= 0:
        return list(detections)
    floor = min_area_frac * image_size[0] * image_size[1]
    return [d for d in detections if box_area(d.box) >= floor]


def crop_instance(
    image: Image.Image,
    det: Detection,
    *,
    pad: float = 0.06,
    blank: bool = True,
    blank_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Padded bbox crop with every non-instance pixel blanked out.

    GOTCHA if skipped: a neighbor standing inside the padded box contributes
    their hair/outfit to this subject's tags. Falls back to a plain crop when
    the detector supplied no mask.
    """
    width, height = image.size
    x1, y1, x2, y2 = det.box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    box = (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(width, int(x2 + px)),
        min(height, int(y2 + py)),
    )
    if det.mask is None or not blank:
        return image.crop(box)
    mask = np.asarray(det.mask)
    if mask.ndim == 3:
        mask = mask[0]
    keep = mask[box[1] : box[3], box[0] : box[2]] > 0.5
    pixels = np.asarray(image.crop(box).convert("RGB")).copy()
    pixels[~keep] = blank_color
    return Image.fromarray(pixels)
