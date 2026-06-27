import math
import random
from typing import NamedTuple, Tuple

# NB: numpy is imported lazily inside BucketManager (its only consumer) so that
# the free-fit helpers below stay numpy-free. The GUI imports this module for
# those helpers via library.preprocess.resize_preview; pulling in numpy at module
# top added ~110ms to GUI startup for a class the GUI never instantiates.

# ---------------------------------------------------------------------------
# Per-tier token-count bands — the free-fit search range for each tier edge.
#
# Free-fit (the only resize mode; see freefit_bucket / freefit_band_for_edge
# below and docs/proposal/free_aspect_token_band_resize.md) preserves an image's
# native aspect ratio and lands its patch-grid token count anywhere inside its
# tier's band. The per-tier discrete (W, H) "constant token bucket" pool that used
# to define these tiers (and snapped every image to one of its entries) is gone —
# only the numeric band survives, and every forward runs at its true token count
# with zero intra-bucket padding under compile_dynamic_seq (the whole band is one
# block graph). Token count = (W//16)*(H//16); every count here is within the rope
# per-axis cap (256 patches). ``--target_res`` (preprocess-only) selects which
# tiers are active; each image is assigned to the tier that resizes it the *least*
# (``choose_edge``), reproducing v1.0's diverse 512–1536 spread.
#
# The bands are the natural (min, max) token count each tier historically carried:
# single-family tiers (768/1280/1536) have lo == hi; 512/896/1024 carry two
# families. ``freefit_band_for_edge`` widens the non-frozen tiers slightly so the
# solver has aspect freedom; 1024 stays frozen at (4032, 4200) for DCW.
EDGE_TOKEN_BANDS: dict = {
    512: (1008, 1024),
    768: (2160, 2160),
    896: (3000, 3024),
    1024: (4032, 4200),
    1280: (6300, 6300),
    1536: (8640, 8640),
}
ALLOWED_TARGET_RES = tuple(sorted(EDGE_TOKEN_BANDS))
DEFAULT_TARGET_RES = (1024,)


def _band(edge: int) -> tuple[int, int]:
    """The (lo, hi) token-count band for a tier edge, or a clear error."""
    try:
        return EDGE_TOKEN_BANDS[edge]
    except KeyError:
        raise ValueError(
            f"target_res {edge} not in allowed tiers {ALLOWED_TARGET_RES}"
        ) from None


def token_count_families(target_res) -> int:
    """Number of distinct token counts (== compiled block graphs) for the tiers.

    Each tier contributes its band endpoints (one count if the band is a single
    family, two if it carries two — e.g. 1024 → {4032, 4200}), deduped across the
    active tiers. Drives the dynamo cache-size budget in ``compile_blocks``. Under
    ``compile_dynamic_seq`` each tier's whole band collapses to one graph, so this
    is a safe upper bound. 1024 alone → 2.
    """
    counts: set = set()
    for edge in target_res:
        lo, hi = _band(edge)
        counts.add(lo)
        counts.add(hi)
    return len(counts)


def token_count_range(target_res) -> tuple[int, int]:
    """(min, max) token count across the active tiers.

    Bounds the ``mark_dynamic`` seq-length hint in ``compile_blocks`` (so inductor
    guards against a real range, not ``[2, ∞)``). 1024 alone → (4032, 4200).
    """
    los = [_band(edge)[0] for edge in target_res]
    his = [_band(edge)[1] for edge in target_res]
    if not los:
        raise ValueError("token_count_range requires at least one tier")
    return min(los), max(his)


def token_counts_for_resos(resos) -> set:
    """Distinct token counts ``(W//16)*(H//16)`` over a set of (W, H) resolutions."""
    return {(w // 16) * (h // 16) for w, h in resos}


def snap_sample_size(width: int, height: int) -> Tuple[int, int]:
    """Snap a requested sample (W, H) to the DiT's 16px pixel grid.

    The single definition of the snap ``_sample_image_inference`` applies before
    sampling — shared with the compile token budget so both sides agree on the
    seq len a sample prompt will actually run at.
    """
    return max(64, width - width % 16), max(64, height - height % 16)


def token_counts_for_sample_prompts(prompts) -> set:
    """Distinct DiT token counts the training sample prompts will request.

    ``prompts`` are ``train_util.load_prompts`` dicts; width/height default to
    512, matching ``_sample_image_inference``. Folded into the torch.compile
    token budget so a sample resolution outside the training buckets (e.g.
    ``--w 1024 --h 1536`` over 1024-tier data → 6144 tokens vs a (4032, 4200)
    range) widens the compiled range instead of crashing mid-training with a
    dynamic-seq ConstraintViolationError (issue #42).
    """
    counts: set = set()
    for prompt_dict in prompts:
        w, h = snap_sample_size(
            int(prompt_dict.get("width", 512)),
            int(prompt_dict.get("height", 512)),
        )
        counts.add((w // 16) * (h // 16))
    return counts


def edge_for_token_count(n_tokens: int, edges=ALLOWED_TARGET_RES) -> int:
    """Map a patch-token count back to the tier edge it belongs to.

    The inverse of the preprocess tier assignment: given a cached latent's token
    count ``(W//16)*(H//16)``, return which tier (512…1536) emitted it. Used by
    the resolution-curriculum schedule (``autoscale_mode``) to rank a dataset's
    populated buckets into a low→high ladder. A count inside a tier's free-fit
    band maps to that tier; anything off-band (snap-era caches) falls back to the
    nearest tier by ``|log(nominal / n)|`` (same scale-symmetric metric as
    ``choose_edge``), so legacy data still ranks sensibly.
    """
    for edge in edges:
        lo, hi = freefit_band_for_edge(edge)
        if lo <= n_tokens <= hi:
            return edge
    n = max(int(n_tokens), 1)
    return min(
        edges,
        key=lambda e: abs(math.log(((_band(e)[0] + _band(e)[1]) / 2.0) / n)),
    )


def choose_edge(width: int, height: int, target_res) -> int:
    """Assign an image to the tier that resizes it the *least*.

    Free-fit preserves the native aspect ratio, so the only thing that varies
    across tiers is the total patch-token budget (area). Each tier's nominal token
    count is the midpoint of its band (``EDGE_TOKEN_BANDS``); the chosen tier
    minimizes ``|log(nominal / native_tokens)|`` — the tier whose budget is closest
    to the image's native patch area, up or down. So a 0.95MP image stays at 1024
    (a tiny upscale) instead of being shoved down to 768 (a big downscale), while a
    0.6MP image still picks 768. Scale-symmetric. Single-element ``target_res`` is
    a no-op. (Equivalent to the old nearest-aspect cover-scale rule: with aspect
    preserved, cover-scale ≈ sqrt(nominal / native_tokens).)
    """
    if len(target_res) == 1:
        return target_res[0]
    native_tokens = (width / 16.0) * (height / 16.0)
    best_edge: int | None = None
    best_cost = float("inf")
    for edge in target_res:
        lo, hi = _band(edge)
        nominal = (lo + hi) / 2.0
        cost = abs(math.log(nominal / native_tokens))
        if cost < best_cost:
            best_cost, best_edge = cost, edge
    return best_edge


# ---------------------------------------------------------------------------
# Free-fit ("free-aspect token-band") solver — see
# docs/proposal/free_aspect_token_band_resize.md.
#
# Free-fit preserves the native aspect ratio and lands the patch-grid token count
# *anywhere* inside the tier's band (EDGE_TOKEN_BANDS, e.g. [4032, 4200] for 1024).
# Each forward runs at its true token count with zero padding; under
# compile_dynamic_seq the whole band is one block graph, so the finer shape
# granularity is free at compile time. Pure, deterministic functions — no I/O.

DEFAULT_FREEFIT_MAX_RATIO = 4.0

# Free-fit's aspect freedom comes entirely from the token-count band width: the
# solver can only match a native aspect to sub-patch if the band admits an integer
# grid near it. The single-family tiers (768 → 2160, 1280 → 6300, 1536 → 8640) and
# the near-degenerate 512 (16-wide) have a *natural* band (lo == hi or nearly so)
# that leaves free-fit no room — it falls back to the coarse divisor grids of that
# one count and crops just like the old snap (the bug in #53's Phase 0: a 0.866-
# aspect image on the 768 tier landed at 0.9375, a ~7.7% crop). Widening the band
# symmetrically by this tolerance restores the "preserve aspect, crop → 0" promise.
# It is free at compile time: under ``compile_dynamic_seq`` the whole [lo, hi] is
# one graph regardless of width (the band only changes *which* counts appear, never
# the graph count), and the train-side seq_range auto-derives from the on-disk
# caches.
FREEFIT_BAND_TOLERANCE = 0.025  # ±2.5% → ~5% interval around the tier's nominal

# The 1024 tier stays frozen at its natural (4032, 4200): DCW calibration keys off
# the exact 1024-tier aspect set (``DCW_ASPECT_BUCKETS``), and its 2-family band is
# already the reference width. Bump this set only with the DCW story in mind.
FREEFIT_FROZEN_EDGES: Tuple[int, ...] = (1024,)

# Bumped whenever the band derivation changes, so free-fit resized PNGs cached
# under an older band re-resize (folded into the resize metadata signature).
FREEFIT_BAND_VERSION = 2


def freefit_band_for_edge(
    edge: int, tol: float = FREEFIT_BAND_TOLERANCE
) -> tuple[int, int]:
    """Token-count band ``(lo, hi)`` for a single tier — the free-fit search range.

    Free-fit lands the patch-grid token count anywhere in this closed interval and
    picks the grid closest to the image's native aspect, so a *wider* band lets it
    preserve aspect more exactly (less crop). The whole band collapses to one
    ``compile_dynamic_seq`` graph at train time, so width is free.

    Starts from the tier's natural ``(min, max)`` token band (``EDGE_TOKEN_BANDS``),
    then widens it symmetrically by ``tol`` for every tier **except** the frozen
    ones (``FREEFIT_FROZEN_EDGES`` — currently 1024, kept at ``(4032, 4200)`` for
    DCW). Without the widening the single-family tiers (768 → 2160, 1280 → 6300,
    1536 → 8640) and the near-degenerate 512 leave the solver no aspect freedom and
    free-fit crops like the old snap.
    """
    lo, hi = token_count_range((edge,))
    if edge in FREEFIT_FROZEN_EDGES:
        return lo, hi
    return round(lo * (1.0 - tol)), round(hi * (1.0 + tol))


def freefit_bucket(
    width: int,
    height: int,
    band: tuple[int, int],
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
    patch: int = 16,
    rope_cap: int = 256,
) -> tuple[int, int]:
    """Native-aspect resize target whose patch grid fills the token ``band``.

    Returns pixel ``(W, H)`` (both multiples of ``patch``) whose patch grid
    ``(W//patch)*(H//patch)`` lies in ``[lo, hi]`` and whose aspect ratio is as
    close as possible to the image's — clamped to ``[1/max_ratio, max_ratio]`` —
    subject to ``max(W//patch, H//patch) <= rope_cap``. Deterministic in its
    inputs.

    Aspect distortion is sub-patch by construction (the cropped residual on the
    covering axis is < ``patch`` px). Crop is zero unless the ratio clamp fired
    (a degenerate input the caller explicitly allowed), in which case the caller
    cover-crops to the clamped aspect just as the snap path does. The search is
    exhaustive over the band — small (~10³ pairs) because the band is narrow and
    bounded by ``rope_cap`` — so the result is the global aspect-error minimum,
    tie-broken toward the grid that resizes the image the least.
    """
    lo, hi = int(band[0]), int(band[1])
    if lo <= 0 or hi < lo:
        raise ValueError(f"invalid free-fit band {band}")
    a = width / height
    a_clamped = min(max(a, 1.0 / max_ratio), float(max_ratio))

    best: tuple | None = None
    hp_max = min(rope_cap, hi)
    for hp in range(1, hp_max + 1):
        wp_lo = max(1, -(-lo // hp))  # ceil(lo / hp)
        wp_hi = min(rope_cap, hi // hp)  # floor(hi / hp)
        for wp in range(wp_lo, wp_hi + 1):
            aspect_err = abs(wp / hp - a_clamped)
            cover_scale = max(wp * patch / width, hp * patch / height)
            # (aspect first, then least rescale, then a deterministic shape key).
            key = (aspect_err, abs(math.log(cover_scale)), hp, wp)
            if best is None or key < best:
                best = key
    if best is None:
        raise ValueError(
            f"free-fit band {band} admits no grid under rope_cap={rope_cap}"
        )
    _, _, hp, wp = best
    return wp * patch, hp * patch


# DCW v4 calibration aspect-bucket set — a frozen standalone literal.
#
# These were the top-5 (H, W) resolutions by frequency in post_image_dataset/lora/
# back when training snapped to the discrete 1024-tier bucket pool (recounted
# 2026-05-23). That pool is gone (free-fit is the only resize mode now), but this
# set stays frozen exactly as-is: list order *is* the canonical aspect_id index —
# DCW v4's per-aspect statistics (fusion_head.safetensors per-bucket μ_g, σ²_prior,
# λ_scalar) key off this order, so a reorder invalidates every shipped fusion-head
# checkpoint.
#
# Read by both the calibration data-gen path (scripts/tasks/dcw.py drives
# `make dcw` over these buckets) and the fusion-head trainer
# (scripts/dcw/fusion_data.py uses the dict for the (H, W) → aspect_id lookup that
# decides which run rows feed the trainer). Inference itself is bucket-agnostic
# post-cleanup — see project_dcw_bucket_prior_cosmetic.
DCW_ASPECT_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (1200, 896),  # 0 — 896x1200 portrait (most common, 4200-tok)
    (1344, 800),  # 1 — 800x1344 tall portrait (4200-tok)
    (896, 1200),  # 2 — 1200x896 landscape (4200-tok)
    (1344, 768),  # 3 — 768x1344 tall portrait (4032-tok)
    (1152, 896),  # 4 — 896x1152 portrait (4032-tok)
)
DCW_ASPECT_NAMES: Tuple[str, ...] = tuple(f"{h}x{w}" for h, w in DCW_ASPECT_BUCKETS)
DCW_ASPECT_TABLE: dict = {hw: i for i, hw in enumerate(DCW_ASPECT_BUCKETS)}
N_DCW_ASPECTS: int = len(DCW_ASPECT_BUCKETS)


def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
    """Generate bucket resolutions for multi-aspect-ratio training.
    Moved from model_util.py to avoid dependency."""
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))

        width += divisible

    resos = list(resos)
    resos.sort()
    return resos


class BucketManager:
    def __init__(
        self, max_reso=None, min_size=None, max_size=None, reso_steps=None
    ) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], (
                    "the max_size should be larger than the width of max_reso"
                )
                assert max_size >= max_reso[1], (
                    "the max_size should be larger than the height of max_reso"
                )
            if min_size is not None:
                assert max_size >= min_size, (
                    "the max_size should be larger than the min_size"
                )

        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(
        self,
        freefit_resos=None,
        target_res=None,
    ):
        if freefit_resos is not None:
            # Free-fit (the only native-shape mode): the predefined set IS the
            # distinct on-disk cached (W, H). Every cached latent then exact-matches
            # in select_bucket and keeps its true (W, H) — nothing AR-snaps at load.
            # The caches are literally the source of truth for which shapes/tiers
            # are present; ``target_res`` is preprocess-only and inert here, and the
            # compile budget is derived from the buckets actually populated
            # (train.py::_derive_token_budget), not from any tier list.
            resos = sorted(set(tuple(r) for r in freefit_resos))
        else:
            resos = make_bucket_resolutions(
                self.max_reso, self.min_size, self.max_size, self.reso_steps
            )
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        import numpy as np

        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        reso = (image_width, image_height)
        if reso in self.predefined_resos_set:
            pass
        else:
            import numpy as np

            ar_errors = self.predefined_aspect_ratios - aspect_ratio
            predefined_bucket_id = np.abs(ar_errors).argmin()
            reso = self.predefined_resos[predefined_bucket_id]

        ar_reso = reso[0] / reso[1]
        if aspect_ratio > ar_reso:
            scale = reso[1] / image_height
        else:
            scale = reso[0] / image_width

        resized_size = (
            int(image_width * scale + 0.5),
            int(image_height * scale + 0.5),
        )

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int
