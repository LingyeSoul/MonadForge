import math
import random
from typing import NamedTuple, Tuple

import numpy as np

# Bucket resolutions as (W, H), grouped into two token-count families: 4032
# (= 63*64) and 4200 (= 60*70). Both are highly composite, so each factors into
# many near-square→elongated patch grids — and crucially every bucket *exactly*
# fills its token count, so there is zero intra-bucket padding by construction.
#
# This table is designed for native shapes (the only mode): it collapses to
# just TWO distinct token counts → two compiled block graphs (via
# compile_blocks' flatten), with no padding and therefore no flash pad leak.
# The rope per-axis cap is 256 patches (max_img/patch_spatial); the largest dim
# here is 2016px → 126.
#
# Two families instead of one because a single token count's divisors near √N
# are sparse (4032 alone jumps aspect 1.29→1.75); interleaving 4032 and 4200
# densely covers aspect space at the cost of one extra graph. Landscape mirrors
# (swap W, H) are included explicitly. Token count = (W//16)*(H//16).
#
# NOTE: DCW_ASPECT_BUCKETS below now draws its top-5 from this table (every
# entry is a real training bucket), so `make dcw` recalibration produces rows
# for every aspect_id. Do not reorder the DCW table (shipped fusion-head
# checkpoints key off it).
CONSTANT_TOKEN_BUCKETS = [
    # ---- 4032-token family (63*64) ----
    (1008, 1024),  # 63 x 64, ar 0.98 (nearest to square)
    (1024, 1008),  #          ar 1.02
    (896, 1152),  # 56 x 72, ar 0.78
    (1152, 896),  #          ar 1.29
    (768, 1344),  # 48 x 84, ar 0.57
    (1344, 768),  #          ar 1.75
    (672, 1536),  # 42 x 96, ar 0.44
    (1536, 672),  #          ar 2.29
    (576, 1792),  # 36 x 112, ar 0.32
    (1792, 576),  #           ar 3.11
    (512, 2016),  # 32 x 126, ar 0.25
    (2016, 512),  #           ar 3.94
    # ---- 4200-token family (60*70) ----
    (960, 1120),  # 60 x 70, ar 0.86
    (1120, 960),  #          ar 1.17
    (896, 1200),  # 56 x 75, ar 0.75
    (1200, 896),  #          ar 1.34
    (800, 1344),  # 50 x 84, ar 0.60
    (1344, 800),  #          ar 1.68
    (672, 1600),  # 42 x 100, ar 0.42
    (1600, 672),  #           ar 2.38
    (640, 1680),  # 40 x 105, ar 0.38
    (1680, 640),  #           ar 2.62
    (560, 1920),  # 35 x 120, ar 0.29
    (1920, 560),  #           ar 3.43
]

# ---------------------------------------------------------------------------
# Multi-scale tiers (opt-in via --target_res at preprocess time).
#
# CONSTANT_TOKEN_BUCKETS above is the canonical *1024* tier (two families,
# 4032/4200) and stays frozen — the DCW fusion-head keys off it. Each tier
# below is one or two highly-composite token-count families (one block graph
# each) of (W, H) factor-pair buckets + landscape mirrors, exactly filling its
# count (zero intra-bucket padding). 768/1280/1536 carry one family; 512 carries
# two (the 1024-tok square + a 1008-tok diversity family). target_res selects
# which tiers are active; each image is assigned to the tier that resizes it the
# *least* (nearest bucket by cover-scale — see ``choose_edge``), reproducing
# v1.0's diverse 512–1536 spread from whatever native resolutions the dataset
# happens to have.
#
# Token count = (W//16)*(H//16). All within the rope per-axis cap (256 patches
# / 4096px): the largest dim here is 2880px (180 patches).

# 512 tier — two counts {1024, 1008}: the exact 512×512 square (1024 tok = 32×32,
# the only square near this scale since 1024 = 2¹⁰) plus a 1008-tok family for the
# mid aspects (1024 alone factors only into ar 1.0 / 0.25). Two graphs, like the
# 1024-edge tier.
CONSTANT_TOKEN_BUCKETS_512 = [
    (512, 512),                # 32 x 32 = 1024, exact square
    (448, 576), (576, 448),    # 28 x 36 = 1008, ar 0.78 / 1.29
    (384, 672), (672, 384),    # 24 x 42 = 1008, ar 0.57
    (336, 768), (768, 336),    # 21 x 48 = 1008, ar 0.44
    (288, 896), (896, 288),    # 18 x 56 = 1008, ar 0.32
    (256, 1008), (1008, 256),  # 16 x 63 = 1008, ar 0.25
]  # fmt: skip

# 768 tier — 2160 tok (< 768²/256 = 2304). Near-square pair at ar 0.94/1.07.
CONSTANT_TOKEN_BUCKETS_768 = [
    (720, 768), (768, 720),    # 45 x 48, ar 0.94 (nearest to square)
    (640, 864), (864, 640),    # 40 x 54, ar 0.74
    (576, 960), (960, 576),    # 36 x 60, ar 0.60
    (480, 1152), (1152, 480),  # 30 x 72, ar 0.42
    (432, 1280), (1280, 432),  # 27 x 80, ar 0.34
    (384, 1440), (1440, 384),  # 24 x 90, ar 0.27
]  # fmt: skip

# 896 tier — two counts {3024, 3000}, both just under the exact square
# (896²/256 = 3136). Like the 1024 tier's 4032/4200, the two families'
# square-most pairs are deliberately *offset* — 3024 anchors at 54×56 (ar 0.96)
# and 3000 at 50×60 (ar 0.83) — so together they cover the common near-square
# portrait band (4:5=0.80 / 5:6=0.83 / 6:7=0.857) that two square-hugging
# families (e.g. 3024+3360, both ~0.95) would leave as a hole. Two graphs, like
# the 1024 and 512 tiers. Trade-off: 3024 carries the spread while 3000 only
# fills the 0.83 anchor (it's divisor-sparse), so the 2:3 band (ar ~0.58–0.76)
# is thin here — 896 is tuned for near-square portrait data; 2:3 / elongated
# images map cleaner at 1024.
CONSTANT_TOKEN_BUCKETS_896 = [
    # ---- 3024-token family (54*56) ----
    (864, 896), (896, 864),    # 54 x 56, ar 0.96 (nearest to square)
    (768, 1008), (1008, 768),  # 48 x 63, ar 0.76
    (672, 1152), (1152, 672),  # 42 x 72, ar 0.58
    (576, 1344), (1344, 576),  # 36 x 84, ar 0.43
    (448, 1728), (1728, 448),  # 28 x 108, ar 0.26
    # ---- 3000-token family (50*60) ----
    (800, 960), (960, 800),    # 50 x 60, ar 0.83
    (640, 1200), (1200, 640),  # 40 x 75, ar 0.53
    (480, 1600), (1600, 480),  # 30 x 100, ar 0.30
]  # fmt: skip

# 1280 tier — 6300 tok (< 1280²/256 = 6400). Near-square pair at ar 0.89/1.12.
CONSTANT_TOKEN_BUCKETS_1280 = [
    (1200, 1344), (1344, 1200),  # 75 x 84, ar 0.89 (nearest to square)
    (1120, 1440), (1440, 1120),  # 70 x 90, ar 0.78
    (1008, 1600), (1600, 1008),  # 63 x 100, ar 0.63
    (960, 1680), (1680, 960),    # 60 x 105, ar 0.57 (~16:9)
    (800, 2016), (2016, 800),    # 50 x 126, ar 0.40
    (720, 2240), (2240, 720),    # 45 x 140, ar 0.32
    (672, 2400), (2400, 672),    # 42 x 150, ar 0.28
]  # fmt: skip

# 1536 tier — 8640 tok (< 1536²/256 = 9216); aspect set mirrors the 768 tier
# (8640 = 4*2160). Near-square pair at ar 0.94/1.07.
CONSTANT_TOKEN_BUCKETS_1536 = [
    (1440, 1536), (1536, 1440),  # 90 x 96, ar 0.94 (nearest to square)
    (1280, 1728), (1728, 1280),  # 80 x 108, ar 0.74
    (1152, 1920), (1920, 1152),  # 72 x 120, ar 0.60
    (1024, 2160), (2160, 1024),  # 64 x 135, ar 0.47
    (960, 2304), (2304, 960),    # 60 x 144, ar 0.42
    (864, 2560), (2560, 864),    # 54 x 160, ar 0.34
    (768, 2880), (2880, 768),    # 48 x 180, ar 0.27
]  # fmt: skip

# Edge (square-equivalent target px) → that tier's bucket table. 1024 reuses the
# canonical frozen list. ``--target_res`` picks a subset of these keys.
CONSTANT_TOKEN_BUCKETS_BY_EDGE = {
    512: CONSTANT_TOKEN_BUCKETS_512,
    768: CONSTANT_TOKEN_BUCKETS_768,
    896: CONSTANT_TOKEN_BUCKETS_896,
    1024: CONSTANT_TOKEN_BUCKETS,
    1280: CONSTANT_TOKEN_BUCKETS_1280,
    1536: CONSTANT_TOKEN_BUCKETS_1536,
}
ALLOWED_TARGET_RES = tuple(sorted(CONSTANT_TOKEN_BUCKETS_BY_EDGE))
DEFAULT_TARGET_RES = (1024,)

# Each tier's *actual* smallest bucket area in pixels (min token count * 16²).
# Informational: the area below which the tier's near-square bucket would have to
# upscale the image. The 1536 tier costs 2.21MP (not 1536²=2.36MP) because the
# tables are "diversity-first, below exact square cost".
TIER_COST_PX = {
    edge: min((w // 16) * (h // 16) for w, h in bk) * 256
    for edge, bk in CONSTANT_TOKEN_BUCKETS_BY_EDGE.items()
}


def buckets_for_edges(target_res):
    """Concatenate the bucket tables for each requested tier edge.

    ``buckets_for_edges([1024])`` reproduces the canonical single-scale list, so
    the default path is byte-identical to pre-multiscale behavior.
    """
    out: list = []
    for edge in target_res:
        if edge not in CONSTANT_TOKEN_BUCKETS_BY_EDGE:
            raise ValueError(
                f"target_res {edge} not in allowed tiers {ALLOWED_TARGET_RES}"
            )
        out.extend(CONSTANT_TOKEN_BUCKETS_BY_EDGE[edge])
    return out


def token_count_families(target_res) -> int:
    """Number of distinct token counts (== compiled block graphs) for the tiers.

    1024 alone → 2 (4032/4200); each extra tier adds 1. Drives the dynamo
    cache-size budget in ``compile_blocks``.
    """
    return len({(w // 16) * (h // 16) for w, h in buckets_for_edges(target_res)})


def token_count_range(target_res) -> tuple[int, int]:
    """(min, max) token count across the active tiers.

    Bounds the ``mark_dynamic`` seq-length hint in ``compile_blocks`` (so inductor
    guards against a real range, not ``[2, ∞)``). 1024 alone → (4032, 4200).
    """
    counts = {(w // 16) * (h // 16) for w, h in buckets_for_edges(target_res)}
    return min(counts), max(counts)


def all_constant_token_buckets() -> list:
    """Every preprocessed tier's buckets, deduped — the full native-shape catalog.

    The train-time predefined bucket set. Because every cached latent sits, by
    construction, at one of these resolutions, ``select_bucket`` always hits the
    exact-match branch and keeps each latent at its true (W, H) — nothing ever
    AR-snaps. So the *on-disk caches are the source of truth* for which tiers are
    present; ``target_res`` is a preprocess-only knob and is inert at train time
    (you can no longer silently drop a tier's caches by omitting it). The compile
    token-family budget is derived from the buckets actually populated by the
    selected (path_pattern-filtered) images — see ``token_counts_for_resos``.
    """
    seen: set = set()
    out: list = []
    for edge in ALLOWED_TARGET_RES:
        for reso in CONSTANT_TOKEN_BUCKETS_BY_EDGE[edge]:
            if reso not in seen:
                seen.add(reso)
                out.append(reso)
    return out


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


def _nearest_aspect_bucket(width: int, height: int, table) -> tuple[int, int]:
    """The bucket in ``table`` whose aspect ratio is closest to the image's —
    same selection rule as ``BucketManager.select_bucket`` (argmin |Δ aspect|)."""
    ar = width / height
    return min(table, key=lambda r: abs(r[0] / r[1] - ar))


def _cover_scale(width: int, height: int, bw: int, bh: int) -> float:
    """Resize factor applied to fit ``(width,height)`` onto bucket ``(bw,bh)``.

    Mirrors ``process_image``'s aspect-preserving cover-then-crop: the image is
    scaled so it covers the bucket, i.e. by ``max(bw/width, bh/height)``. >1
    upscales, <1 downscales.
    """
    return max(bw / width, bh / height)


def choose_edge(width: int, height: int, target_res) -> int:
    """Assign an image to the tier that resizes it the *least*.

    For each tier we find the bucket it would actually map into (nearest aspect)
    and the cover-scale that mapping needs; the chosen tier minimizes
    ``|log(scale)|`` — i.e. the bucket closest to the image's native size, up or
    down. So a 0.95MP image stays at 1024 (a tiny upscale) instead of being
    shoved down to 768 (a big downscale), while a 0.6MP image still picks 768.
    Single-element ``target_res`` is a no-op.
    """
    if len(target_res) == 1:
        return target_res[0]
    best_edge: int | None = None
    best_cost = float("inf")
    for edge in target_res:
        bw, bh = _nearest_aspect_bucket(
            width, height, CONSTANT_TOKEN_BUCKETS_BY_EDGE[edge]
        )
        cost = abs(math.log(_cover_scale(width, height, bw, bh)))
        if cost < best_cost:
            best_cost, best_edge = cost, edge
    return best_edge


# DCW v4 calibration aspect-bucket set.
#
# Top 5 (H, W) resolutions by frequency in post_image_dataset/lora/ (recounted
# 2026-05-23; every entry is a CONSTANT_TOKEN_BUCKETS training bucket). List
# order *is* the canonical aspect_id index — DCW v4's per-aspect statistics
# (fusion_head.safetensors per-bucket μ_g, σ²_prior, λ_scalar) key off this
# order, so a reorder invalidates every shipped fusion-head checkpoint.
#
# Read by both the calibration data-gen path (scripts/tasks/dcw.py drives
# `make dcw` over these buckets) and the fusion-head trainer
# (scripts/dcw/fusion_data.py uses the dict for the (H, W) → aspect_id
# lookup that decides which run rows feed the trainer). Inference itself
# is bucket-agnostic post-cleanup — see project_dcw_bucket_prior_cosmetic.
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

    def make_buckets(self, constant_token_buckets: bool = False, target_res=None):
        if constant_token_buckets:
            # The full native-shape catalog (every tier), so select_bucket hits the
            # exact-match branch for any cached reso and keeps each latent at its
            # true (W, H) — a multi-tier dataset never AR-snaps non-1024 caches into
            # a 1024 bucket. target_res is preprocess-only and inert here: the
            # on-disk caches are the source of truth for which tiers are present,
            # and the compile budget is derived from the buckets actually populated
            # (train.py), not from this list. So omitting a tier at train time can
            # no longer silently drop its caches.
            resos = all_constant_token_buckets()
        else:
            resos = make_bucket_resolutions(
                self.max_reso, self.min_size, self.max_size, self.reso_steps
            )
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
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
