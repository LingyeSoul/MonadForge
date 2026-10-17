"""User-facing tag-group names for batch caption filtering.

Maps a small English vocabulary (``artist``, ``clothing``, ``pose``,
``lighting`` …) onto the ``[대분류 > 소분류]`` taxonomy path every row of
``danbooru_tags_classified.csv`` carries, so ``--caption_drop_groups
artist,lighting,pose`` can strip a *kind* of tag from every caption at
mirror time without touching the caption master.

Resolution for one tag, in order:

1. Tag *shape* (``taxonomy.py``) — ``@`` prefix → ``artist``, count regex →
   ``count``, rating literal → ``rating``. Shape wins over the KB, so a
   dataset-only artist is still an artist.
2. KB kind (``TagInfo.kind``) — ``character`` / ``copyright`` / ``meta`` from
   danbooru's numeric category.
3. KB taxonomy path — the coarse ``대분류`` (or ``대분류 > 소분류`` for finer
   entries like ``lighting``).
4. Unknown to the KB → ``None``, and **never dropped**.

A selector that isn't a known slug is treated as a **literal path prefix**
against the KB's ``category_path`` (``"효과/연출 > 조명"``), so any subgroup in
the CSV is reachable without editing this table. Pure stdlib, torch-free.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from library.curation_compat.captions.taxonomy import is_artist_tag, is_count_tag, is_rating_tag

if TYPE_CHECKING:  # pragma: no cover
    from library.curation_compat.captions.correction import TagKnowledgeBase


# slug -> taxonomy path prefix(es) in the KB. The coarse 대분류 slugs cover the
# whole subtree; the finer ones (``lighting`` …) pick one 소분류. Paths are
# matched as prefixes on the normalized ``category_path``.
DROP_GROUP_PATHS: dict[str, tuple[str, ...]] = {
    # Structural kinds — resolved by tag shape / danbooru numeric category
    # first; the path is only the fallback for Korean-tree-typed rows.
    "artist": ("아티스트", "작가"),
    "character": ("캐릭터",),
    "copyright": ("작품/출처", "작품"),
    "meta": ("메타",),
    "count": ("인물 > 인원수",),
    # Coarse content groups (one 대분류 each).
    "clothing": ("의상",),
    "accessory": ("액세서리",),
    "action": ("행동",),
    "pose": ("포즈/구도",),
    "expression": ("표정/감정",),
    "face": ("얼굴/눈",),
    "hair": ("머리카락",),
    "body": ("신체",),
    "background": ("배경/장소",),
    "effect": ("효과/연출",),
    "object": ("사물",),
    "animal": ("동물/생물",),
    "color": ("색상/패턴",),
    "person": ("인물",),
    # Finer picks the issue asked for by name.
    "lighting": ("효과/연출 > 조명",),
    "weather": ("효과/연출 > 날씨",),
    "text": ("효과/연출 > 텍스트/말풍선",),
    "style_parody": ("효과/연출 > 그림체_패러디",),
    "framing": ("포즈/구도 > 프레이밍", "포즈/구도 > 시점/앵글"),
    "sexual": ("행동 > 성적행위",),
    "weapon": ("사물 > 무기",),
    "food": ("사물 > 음식/음료",),
    "species": ("인물 > 종족/비인간",),
    "fashion_style": ("인물 > 패션_스타일",),
}

# Structural slugs whose resolution is shape/kind-based, not path-based.
_KIND_SLUGS = frozenset({"artist", "character", "copyright", "meta", "count"})


def drop_group_names() -> tuple[str, ...]:
    """Every user-facing slug, in table order (for ``--help`` / GUI lists)."""
    return tuple(DROP_GROUP_PATHS)


def parse_drop_groups(spec: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize a comma-separated (or iterable) selector list.

    Slugs are lower-cased, anything else kept verbatim as a literal
    taxonomy-path prefix. Order is stable; duplicates and blanks dropped.
    """
    if not spec:
        return ()
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        low = s.lower()
        s = low if low in DROP_GROUP_PATHS else s
        if s not in out:
            out.append(s)
    return tuple(out)


def _selector_paths(selector: str) -> tuple[str, ...]:
    return DROP_GROUP_PATHS.get(selector, (selector,))


def _path_matches(category_path: str, prefix: str) -> bool:
    """Prefix match on the normalized taxonomy path.

    A bare 대분류 (``"인물"``) covers its whole subtree, and CSV editions spell
    some roots differently (``작품`` vs ``작품/출처``).
    """
    return bool(category_path and prefix) and category_path.startswith(prefix)


def tag_drop_group(tag: str, kb: TagKnowledgeBase) -> str | None:
    """The coarse slug ``tag`` belongs to, or ``None`` when unclassifiable.

    Only the *coarse* structural/대분류 slug (``effect``, not ``lighting``);
    :func:`should_drop_tag` handles finer selectors by path.
    """
    if tag == "@no-artist":
        return None
    if is_artist_tag(tag):
        return "artist"
    if is_count_tag(tag):
        return "count"
    if is_rating_tag(tag):
        return None
    info = kb.describe(tag)
    if info is None:
        return None
    if info.kind in ("artist", "character", "copyright", "meta"):
        return info.kind
    path = info.category_path
    if not path:
        return None
    for slug, prefixes in DROP_GROUP_PATHS.items():
        if slug in _KIND_SLUGS or " > " in prefixes[0]:
            continue
        if any(_path_matches(path, p) for p in prefixes):
            return slug
    return None


def should_drop_tag(tag: str, kb: TagKnowledgeBase, selectors: Iterable[str]) -> bool:
    """True when ``tag`` falls under any of ``selectors``.

    Structural slugs resolve by shape/kind (so an unknown ``@name`` still drops
    under ``artist``); everything else resolves against the KB taxonomy path.
    Unknown tags and rating literals never drop.
    """
    sels = tuple(selectors)
    if not sels or tag == "@no-artist" or is_rating_tag(tag):
        return False
    coarse = tag_drop_group(tag, kb)
    if coarse is not None and coarse in sels:
        return True
    if coarse in _KIND_SLUGS:
        # A structural tag (``1girl`` sits under ``인물 > 인원수``) must not
        # fall to a content selector like ``person`` via its path.
        return False
    info = kb.describe(tag)
    path = info.category_path if info is not None else ""
    if not path:
        return False
    return any(
        _path_matches(path, p)
        for sel in sels
        if sel not in _KIND_SLUGS
        for p in _selector_paths(sel)
    )
