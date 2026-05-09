"""AnimaTagger — multi-label tagger trained on the Anima caption distribution.

Drop-in replacement for :class:`WDTagger` when feeding ψ_src to DirectEdit.
Mirrors the same public surface (``predict``, ``predict_caption``) so existing
callers swap by changing only the import.

Checkpoint layout (produced by ``python -m scripts.anima_tagger.cli``):

::

    ckpt_dir/
      config.json              # model config + training metadata
      model.safetensors        # AnimaTaggerHead state dict
      pe_lora.safetensors      # PE-LoRA delta on PE-Core trailing blocks (optional)
      thresholds.safetensors   # per-tag F1-optimal thresholds
      vocab.json               # tag list with category + median_pos + group info
      rules.yaml               # caption-normalization rules snapshot
      groups.yaml              # tag-group taxonomy (optional)

If ``config.json`` has ``pe_lora: true`` and ``pe_lora.safetensors`` exists,
the wrapper injects PE-LoRA on the encoder's trailing blocks and loads the
delta weights — same code path as ``python -m scripts.anima_tagger.cli``.

When ``groups.yaml`` is present, prediction is group-aware: ``softmax`` and
``softmax_when_solo`` (the latter gated on solo + no-escape) groups emit
exactly one tag per group (argmax over group logits), even when the
sigmoid threshold would have admitted several. Multi-label groups and
ungrouped tags fall back to the standard threshold path.

The vision encoder (PE-Core-L14-336 by default) is loaded lazily on first
``predict`` call. Captions are emitted in Anima's canonical slot order:
``rating, count_tags, characters, copyrights, @artists, generals``, with
underscores replaced by spaces (matching how Anima's training-time T5 saw
the data).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as st_load

from library.captioning import tag_groups as tg
from library.captioning import tag_rules as tr
from library.captioning.anima_tagger_data import pil_resize_to_bucket
from library.captioning.anima_tagger_model import AnimaTaggerConfig, AnimaTaggerHead
from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.vision.encoder import (
    VisionEncoderBundle,
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

logger = logging.getLogger(__name__)

# Matches "1girl", "2girls", …, "6+girls" (digit-prefixed girls counts).
# "multiple girls" is intentionally not matched — it carries no exact count
# so we leave the character head's output untouched in that case.
_GIRLS_COUNT_RE = re.compile(r"^(\d+)\+?girls?$")


# Canonical caption-format slot order (matches Anima training captions).
SLOT_ORDER: Tuple[str, ...] = (
    "rating",
    "count",
    "character",
    "copyright",
    "artist",
    "general",
)


@dataclass
class _TagEntry:
    name: str
    index: int
    category: str
    median_pos: float


def _underscore_to_space(s: str) -> str:
    """Anima caption format: tags with spaces, not underscores.

    The cache key uses underscores; the canonical caption uses spaces.
    Apply at emit time (not vocab-build) so tag indexing stays stable.
    """
    return s.replace("_", " ")


def _load_thresholds(path: Path, n_tags: int, default: float = 0.5) -> torch.Tensor:
    """Load per-tag thresholds; missing → uniform default."""
    if not path.exists():
        logger.warning("no thresholds.safetensors at %s — using default=%.2f", path, default)
        return torch.full((n_tags,), default)
    d = st_load(str(path))
    t = d["thresholds"]
    if t.shape != (n_tags,):
        raise ValueError(f"thresholds shape {tuple(t.shape)} != ({n_tags},)")
    return t


class AnimaTagger:
    """Drop-in replacement for ``WDTagger`` with Anima-distribution vocabulary."""

    def __init__(
        self,
        ckpt_dir: str | Path = "models/captioners/anima-tagger-v1",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        pe_ckpt: str | Path | None = None,
        character_floor: float = 0.5,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.pe_ckpt = Path(pe_ckpt) if pe_ckpt else None
        # Absolute confidence floor for character predictions. Sits *above*
        # the per-tag F1-optimal threshold for the low-confidence end of the
        # character vocab (some F1 thresholds are as low as 0.05 — chasing
        # F1 there produces visible false positives on gender-ambiguous /
        # stylized art). Below the floor → suppress and fall back to the
        # `original` copyright tag.
        self._character_floor = float(character_floor)

        with open(self.ckpt_dir / "config.json") as f:
            cfg_d = json.load(f)
        self.encoder_name: str = cfg_d.get("encoder", "pe")
        self.cfg = AnimaTaggerConfig.from_dict(cfg_d["model"])
        self._cfg_d = cfg_d

        self.model = AnimaTaggerHead(self.cfg)
        self.model.load_state_dict(st_load(str(self.ckpt_dir / "model.safetensors")))
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        with open(self.ckpt_dir / "vocab.json") as f:
            vocab = json.load(f)
        self.tag_entries: List[_TagEntry] = [
            _TagEntry(
                name=t["name"],
                index=int(t["index"]),
                category=str(t["category"]),
                median_pos=float(t.get("median_pos", 0.0)),
            )
            for t in vocab["tags"]
        ]
        self.ratings: List[str] = list(vocab["ratings"])
        # Vocab index of the canonical "original" copyright tag, or None
        # when absent. Used by predict() as the uncertainty-fallback when
        # a character was guessed but didn't clear `_character_floor`.
        self._original_idx: Optional[int] = next(
            (
                e.index
                for e in self.tag_entries
                if e.name == "original" and e.category == "copyright"
            ),
            None,
        )
        # Map category → list of (index, median_pos, name) sorted by median_pos.
        self._by_cat: Dict[str, List[Tuple[int, float, str]]] = {}
        for e in self.tag_entries:
            cat = e.category if e.category in SLOT_ORDER else "general"
            self._by_cat.setdefault(cat, []).append((e.index, e.median_pos, e.name))
        for cat in self._by_cat:
            self._by_cat[cat].sort(key=lambda triple: (triple[1], triple[2]))

        self.thresholds = _load_thresholds(
            self.ckpt_dir / "thresholds.safetensors", n_tags=self.cfg.n_tags
        )
        self.thresholds_dev = self.thresholds.to(self.device)

        self.rules = tr.load_rules(self.ckpt_dir / "rules.yaml")

        # Optional groups snapshot. Built per-group caches so predict()
        # doesn't reparse names every call. When the snapshot is missing
        # (older checkpoints / flat-vocab builds) self._groups is None.
        groups_path = self.ckpt_dir / "groups.yaml"
        self._groups: Optional[tg.TagGroups] = None
        # Per-group: name → {mode, tag_idx_tensor[K], escape_idx_tensor[E],
        #                    tag_names_set, escape_names_set}
        self._group_lookup: Dict[str, Dict] = {}
        # Vocab indices used to detect "single-subject" at inference. We
        # mirror the trainer's GroupRouter logic — `solo`/`1girl`/`1boy`/
        # `1other` are single-count, anything else matching the count
        # regex is multi-count.
        self._single_count_names = {"solo", "1girl", "1boy", "1other"}
        self._multi_count_names: set = set()
        if groups_path.exists():
            self._groups = tg.load_groups(groups_path)
            tag_to_idx = {e.name: e.index for e in self.tag_entries}
            for g in self._groups.groups:
                if g.mode not in ("softmax", "softmax_when_solo"):
                    continue
                tag_idx = [tag_to_idx[t] for t in g.tags if t in tag_to_idx]
                if not tag_idx:
                    continue
                self._group_lookup[g.name] = {
                    "mode": g.mode,
                    "tag_idx": torch.tensor(tag_idx, dtype=torch.long, device=self.device),
                    "tag_names": tuple(g.tags),
                    "escape_names": tuple(g.escape),
                }
            # Detect multi-count tags by regex over the vocab.
            from re import compile as _re_compile
            count_re = _re_compile(
                r"^(?:\d+(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$"
            )
            for e in self.tag_entries:
                if e.name in self._single_count_names:
                    continue
                if count_re.match(e.name):
                    self._multi_count_names.add(e.name)
        self._encoder: Optional[VisionEncoderBundle] = None

    # ── Encoder lazy-load ──────────────────────────────────────────────

    def _bundle(self) -> VisionEncoderBundle:
        if self._encoder is None:
            self._encoder = load_pe_encoder(
                self.device,
                name=self.encoder_name,
                model_id=str(self.pe_ckpt) if self.pe_ckpt else None,
                dtype=self.dtype,
            )
            self._maybe_apply_pe_lora(self._encoder)
        return self._encoder

    def _maybe_apply_pe_lora(self, bundle: VisionEncoderBundle) -> None:
        """Inject PE-LoRA on the encoder's trailing blocks and load delta weights.

        Idempotent on a fresh bundle. Skips when the checkpoint was trained
        without PE-LoRA (``config.pe_lora`` False / missing) or when the
        ``pe_lora.safetensors`` sidecar is absent. The injected LoRA params
        are switched to ``eval()`` and ``requires_grad_(False)`` since this
        is the inference path.
        """
        cfg_d = self._cfg_d
        if not cfg_d.get("pe_lora", False):
            return
        pe_lora_path = self.ckpt_dir / "pe_lora.safetensors"
        if not pe_lora_path.exists():
            logger.warning(
                "config.pe_lora=true but %s is missing — encoder will run frozen "
                "without the trained delta",
                pe_lora_path,
            )
            return
        from networks.methods.ip_adapter_pe_lora import inject_pe_lora

        pe_inner = bundle.encoder.inner
        pe_lora = inject_pe_lora(
            pe_inner,
            rank=int(cfg_d.get("pe_lora_rank", 16)),
            alpha=float(cfg_d.get("pe_lora_alpha", 16.0)),
            target_qkv=bool(cfg_d.get("pe_lora_qkv", True)),
            target_attn_out=bool(cfg_d.get("pe_lora_attn_out", True)),
            target_mlp=bool(cfg_d.get("pe_lora_mlp", True)),
            layer_from=int(cfg_d.get("pe_lora_layers", 8)),
        )
        state = st_load(str(pe_lora_path))
        missing, unexpected = pe_lora.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(
                "PE-LoRA load: missing=%d unexpected=%d (e.g. missing=%s unexpected=%s)",
                len(missing),
                len(unexpected),
                missing[:3],
                unexpected[:3],
            )
        pe_lora.to(device=self.device, dtype=torch.float32)
        pe_lora.eval()
        for p in pe_lora.parameters():
            p.requires_grad_(False)
        # Stash a reference so it isn't GC'd. The patched forward closures
        # in inject_pe_lora hold strong refs already, but keeping this on
        # the wrapper makes the LoRA params introspectable post-init.
        self._pe_lora = pe_lora
        logger.info(
            "applied PE-LoRA (rank=%s, last %s blocks) from %s",
            cfg_d.get("pe_lora_rank"),
            cfg_d.get("pe_lora_layers"),
            pe_lora_path.name,
        )

    @torch.no_grad()
    def _encode_image(self, pil_img: Image.Image) -> torch.Tensor:
        """Image → pooled feature ``[d_in]`` on ``self.device``."""
        bundle = self._bundle()
        pil_img = pil_resize_to_bucket(pil_img.convert("RGB"), bundle.bucket_spec)
        tensor = IMAGE_TRANSFORMS(np.array(pil_img)).unsqueeze(0)
        feats_list = encode_pe_from_imageminus1to1(bundle, tensor, same_bucket=True)
        feats = feats_list[0]                # [T, d_enc]
        return feats.mean(dim=0).to(torch.float32)

    # ── Public API (mirrors WDTagger) ──────────────────────────────────

    @torch.no_grad()
    def predict(self, pil_img: Image.Image) -> Dict[str, object]:
        """Run one image through the head; return raw + thresholded outputs.

        Returns a dict with:

        * ``rating``: predicted rating string (one of ``self.ratings``)
        * ``rating_scores``: dict ``{rating: prob}``
        * ``scores``: dict ``{tag: prob}`` for *all* in-vocab tags
        * ``kept``: dict ``{tag: prob}`` for tags emitted as positives.
          When typed groups are loaded, softmax-group winners are picked
          by argmax (one per group) instead of by sigmoid threshold.
        * ``groups``: dict ``{group_name: predicted_tag_or_None}`` — only
          present when typed groups are loaded.
        """
        feat = self._encode_image(pil_img).unsqueeze(0).to(self.device)
        tag_logits, rating_logits = self.model(feat)
        tag_logits_row = tag_logits[0]                       # [n_tags]
        tag_probs = tag_logits_row.sigmoid()                 # [n_tags]
        rating_probs = rating_logits.softmax(dim=-1)[0]      # [n_ratings]
        kept_mask = (tag_probs >= self.thresholds_dev).cpu()
        tag_probs_cpu = tag_probs.cpu()
        scores = {
            self.tag_entries[i].name: float(tag_probs_cpu[i])
            for i in range(self.cfg.n_tags)
        }
        kept = {
            self.tag_entries[i].name: float(tag_probs_cpu[i])
            for i in range(self.cfg.n_tags)
            if kept_mask[i]
        }
        rating_idx = int(rating_probs.argmax().item())
        out: Dict[str, object] = {
            "rating": self.ratings[rating_idx],
            "rating_scores": {
                r: float(rating_probs[i].cpu()) for i, r in enumerate(self.ratings)
            },
            "scores": scores,
            "kept": kept,
        }

        # Group-aware refinement of `kept`. Replaces softmax-group sigmoid
        # threshold output with one argmax winner per applicable group.
        if self._group_lookup:
            kept_names = set(kept.keys())
            is_solo = (
                bool(kept_names & self._single_count_names)
                and not (kept_names & self._multi_count_names)
            )
            group_preds: Dict[str, Optional[str]] = {}
            for name, info in self._group_lookup.items():
                mode = info["mode"]
                escape_fired = bool(kept_names & set(info["escape_names"]))
                if mode == "softmax_when_solo":
                    applicable = is_solo and not escape_fired
                else:                                            # "softmax"
                    applicable = not escape_fired
                if not applicable:
                    # Leave the group's tags exactly as the per-tag
                    # threshold decided. predict_caption can pull whatever
                    # set the sigmoid path admitted.
                    group_preds[name] = None
                    continue
                idx_t = info["tag_idx"]
                group_logits = tag_logits_row.index_select(0, idx_t)
                winner_local = int(group_logits.argmax().item())
                winner_idx = int(idx_t[winner_local].item())
                winner_name = self.tag_entries[winner_idx].name
                # Drop any sigmoid-admitted tags from this group, then add
                # the argmax winner back with its sigmoid probability so
                # downstream callers can still inspect a confidence.
                for t in info["tag_names"]:
                    kept.pop(t, None)
                kept[winner_name] = float(tag_probs_cpu[winner_idx])
                group_preds[name] = winner_name
            out["kept"] = kept
            out["groups"] = group_preds

        # Cap character-tag predictions by the largest digit-prefixed
        # girls-count tag in `kept`. The character head emits an independent
        # sigmoid per tag, so on gender-ambiguous art where both `1boy` and
        # `1girl` fire it can still admit several borderline character
        # matches for what is actually a single subject. Trim to top-N by
        # score where N is parsed from `1girl`/`2girls`/`6+girls`/...; if no
        # digit-prefixed girls tag is in `kept` (e.g. only `1boy`, or only
        # `multiple girls`) leave the character set alone.
        girl_caps = [
            int(m.group(1))
            for name in kept
            if (m := _GIRLS_COUNT_RE.match(name))
        ]
        if girl_caps:
            cap = max(girl_caps)
            char_scored = sorted(
                (
                    (kept[e.name], e.name)
                    for e in self.tag_entries
                    if e.category == "character" and e.name in kept
                ),
                reverse=True,
            )
            for _, name in char_scored[cap:]:
                kept.pop(name, None)
            out["kept"] = kept

        # Suppress uncertain character predictions and fall back to the
        # `original` copyright tag. Any kept character whose score is below
        # `_character_floor` is treated as a guess and dropped. When that
        # empties the character slot AND no copyright tag is in `kept`,
        # add "original" (using its raw sigmoid score) so the caption has
        # a slot-filling copyright — booru convention for non-IP work.
        # Confident characters are preserved; the floor only fires on
        # borderline admits where the F1 threshold lets a noisy guess
        # through.
        dropped_any = False
        for e in self.tag_entries:
            if e.category != "character" or e.name not in kept:
                continue
            if kept[e.name] < self._character_floor:
                kept.pop(e.name, None)
                dropped_any = True
        if dropped_any and self._original_idx is not None:
            has_char = any(
                e.category == "character" and e.name in kept
                for e in self.tag_entries
            )
            has_copy = any(
                e.category == "copyright" and e.name in kept
                for e in self.tag_entries
            )
            if not has_char and not has_copy:
                kept["original"] = float(tag_probs_cpu[self._original_idx])

        # Cap artist and copyright slots to top-1 by score. Both heads emit
        # independent sigmoids, so multiple borderline tags can clear their
        # F1 thresholds on a single image — booru convention is one artist
        # / one copyright per work, and downstream callers expect that.
        for cat in ("artist", "copyright"):
            cat_scored = sorted(
                (
                    (kept[e.name], e.name)
                    for e in self.tag_entries
                    if e.category == cat and e.name in kept
                ),
                reverse=True,
            )
            for _, name in cat_scored[1:]:
                kept.pop(name, None)
        out["kept"] = kept
        return out

    def predict_caption(self, pil_img: Image.Image) -> str:
        """Image → canonical Anima caption string (rating + slotted tags)."""
        out = self.predict(pil_img)
        kept_idxs = {
            self.tag_entries[i].index
            for i, name in enumerate([e.name for e in self.tag_entries])
            if name in out["kept"]
        }
        # Slot tags by canonical category order, within-slot by median_pos.
        slotted: Dict[str, List[str]] = {cat: [] for cat in SLOT_ORDER}
        slotted["rating"].append(out["rating"])
        for cat, entries in self._by_cat.items():
            for idx, _, name in entries:
                if idx in kept_idxs:
                    slotted.setdefault(cat, []).append(name)
        # Re-apply tag rules at emit time as a safety net (the dedup map
        # already fired during training-data normalization, but the model
        # could in principle predict both ``bra`` and ``black bra``;
        # apply_rules drops ``bra`` in that case).
        flat: List[str] = []
        for cat in SLOT_ORDER:
            flat.extend(slotted.get(cat, []))
        rating_held = flat[:1]
        rest = tr.apply_rules(flat[1:], self.rules)
        out_tags = rating_held + rest
        return ", ".join(_underscore_to_space(t) for t in out_tags)
