"""Identity-pair reference sampler for IP-Adapter distinct-pair training.

Consumes the *method-agnostic* caption index built by
``preprocess/build_caption_index.py`` (``post_image_dataset/captions/
caption_index.json``) and owns the IP-Adapter **policy** on top of it: a
tiered character → copyright → artist back-off that, given a target image,
returns a *different* image of the same identity to feed the IP path.

The index encodes no policy (see that script's docstring); everything
opinionated — level priority, cross-artist constraint, the candidate-pool
restriction — lives here so the disk artifact stays reusable.

Why distinct pairs: under self-pairing (reference == VAE target) the IP path
can lower the loss by copying the target's own pixels, which never forces it to
learn *identity*. With a reference that is a different image of the same
identity, the only signal that consistently helps is what is invariant across
the pair — identity — because pose/crop/background/lighting differ and are
useless to copy. See ``docs/proposal/ip-adapter-identity-pairs.md``.
"""

from __future__ import annotations

import json
import os
import random
from typing import Iterable, Optional

# Tightest → loosest. ``resolve`` walks this in order and returns the first
# tier that has a distinct same-group image available.
LEVELS = ("character", "copyright", "artist")


class IdentityPairSampler:
    """Picks a distinct same-identity (or unrelated) reference stem.

    Parameters
    ----------
    index_path:
        Path to ``caption_index.json``.
    min_level:
        Loosest tier allowed before falling back to self. One of
        ``character`` / ``copyright`` / ``artist``. ``artist`` (default) allows
        the full back-off; ``character`` restricts pairing to same-character
        only (no franchise/artist fallback).
    cross_artist:
        When True, character/copyright-tier matches must come from a *different*
        artist — forces the IP path to carry identity without the source
        artist's style (the ``identity_cross_artist`` mode). Artist-tier
        matches are unaffected (they share the artist by definition).
    restrict_stems:
        Optional set of stems the *reference* may be drawn from. Used to keep
        training references inside the training split (no validation-image
        leakage). ``None`` ⇒ any stem in the index is eligible.
    """

    def __init__(
        self,
        index_path: str,
        *,
        min_level: str = "artist",
        cross_artist: bool = False,
        restrict_stems: Optional[Iterable[str]] = None,
    ) -> None:
        if min_level not in LEVELS:
            raise ValueError(f"min_level must be one of {LEVELS}, got {min_level!r}")
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        self.index_path = index_path
        self.image_meta: dict[str, dict] = index["image_meta"]
        self.groups: dict[str, dict[str, list[str]]] = index["groups"]
        self.cross_artist = bool(cross_artist)
        # Active tiers = prefix of LEVELS up to and including min_level.
        self.levels = LEVELS[: LEVELS.index(min_level) + 1]

        self._restrict: Optional[set[str]] = (
            set(restrict_stems) if restrict_stems is not None else None
        )
        # Candidate pool for shuffled() — every reference-eligible stem.
        self._all_stems: list[str] = sorted(
            s for s in self.image_meta if self._eligible(s)
        )

    # -- helpers ----------------------------------------------------------

    def _eligible(self, stem: str) -> bool:
        return self._restrict is None or stem in self._restrict

    def has(self, stem: str) -> bool:
        """True when ``stem`` is present in the index (so we can pair it)."""
        return stem in self.image_meta

    def rel_dir(self, stem: str) -> str:
        """Directory of the index's recorded path for ``stem`` (e.g. the
        artist subdir), used to resolve the nested PE-feature cache."""
        meta = self.image_meta.get(stem)
        if meta is None:
            return ""
        return os.path.dirname(meta.get("path", ""))

    def _identity_tags(self, stem: str) -> set[str]:
        """character + copyright tags, used to test 'unrelated' for shuffled."""
        meta = self.image_meta.get(stem, {})
        return set(meta.get("character", [])) | set(meta.get("copyright", []))

    # -- policy -----------------------------------------------------------

    def resolve(self, target_stem: str, rng: random.Random) -> tuple[str, str]:
        """Return ``(reference_stem, level)`` for a *distinct* same-identity
        reference, walking character → copyright → artist. Falls back to
        ``(target_stem, "self")`` when no distinct positive is reachable."""
        meta = self.image_meta.get(target_stem)
        if meta is None:
            return target_stem, "self"
        target_artists = set(meta.get("artist", []))

        for level in self.levels:
            candidates: set[str] = set()
            for tag in meta.get(level, []):
                for s in self.groups.get(level, {}).get(tag, []):
                    if s == target_stem or not self._eligible(s):
                        continue
                    if (
                        self.cross_artist
                        and level in ("character", "copyright")
                        and target_artists
                        and target_artists
                        & set(self.image_meta.get(s, {}).get("artist", []))
                    ):
                        # Same artist — rejected in cross-artist mode so the
                        # match can't preserve the source artist's style.
                        continue
                    candidates.add(s)
            if candidates:
                return rng.choice(sorted(candidates)), level
        return target_stem, "self"

    def shuffled(self, target_stem: str, rng: random.Random) -> tuple[str, str]:
        """Return ``(reference_stem, "shuffled")`` for an *unrelated* image —
        a different identity, used as the validation negative baseline. Tries
        to avoid sharing any character/copyright tag; accepts any distinct stem
        after a few rejections (small datasets may be densely connected)."""
        if not self._all_stems:
            return target_stem, "self"
        target_identity = self._identity_tags(target_stem)
        for _ in range(8):
            cand = rng.choice(self._all_stems)
            if cand == target_stem:
                continue
            if target_identity and (self._identity_tags(cand) & target_identity):
                continue
            return cand, "shuffled"
        # Fall back to any distinct stem.
        for _ in range(8):
            cand = rng.choice(self._all_stems)
            if cand != target_stem:
                return cand, "shuffled"
        return target_stem, "self"
