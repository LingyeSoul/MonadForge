"""Derive ``tag_groups.yaml`` candidates from the danbooru taxonomy KB.

Reads the kept ``vocab.json`` (general tags only), looks each up in
``danbooru_tags_classified.csv`` to recover its ``[대분류 > 소분류]`` taxonomy
path, and buckets the general vocab into candidate groups — one per distinct
taxonomy ``소분류``. Each candidate's **mode** (``softmax_when_solo`` vs
``multilabel``) is decided by measuring how often two members of the group
co-occur on a *single-subject* image: a family whose members almost never
share a solo image (eye color, hair length) is mutually exclusive →
``softmax_when_solo``; a family whose members routinely stack (accessories,
clothing details) → ``multilabel``.

The output is a **candidate** ``groups.yaml`` for human review, not a drop-in
file: group keys are slugged from the (Korean) taxonomy and want renaming,
``softmax`` groups want an ``escape:`` list curated, and low-support groups are
flagged. Nothing in ``--out_dir`` is mutated.

Name matching honors the custom ``tag_rules.yaml``: vocab names are already
post-rule, and any tag the rules *renamed* away from its danbooru form is
recovered via the inverse replacement map before it's logged as unmatched.
The co-occurrence scan reads captions through :func:`tag_rules.parse_caption`
(same normalization the trainer saw), or the prebuilt ``dataset.json`` manifest
when present.

Read-only. Usage::

    python -m scripts.anima_tagger.cli --mode derive_groups \\
        --out_dir models/captioners/anima-tagger-v2 \\
        --tag_cache models/danbooru_tags_classified.csv \\
        --out_yaml /tmp/groups.candidate.yaml --report
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import yaml

from library.captioning.correction import load_tag_knowledge_base

logger = logging.getLogger(__name__)


# Single-subject predicate — mirrors the trainer's GroupRouter / role_markers
# solo check, but on space-form tag *names* (captions, not vocab indices).
_SINGLE_COUNT_NAMES = {"solo", "1girl", "1boy", "1other"}
_MULTI_COUNT_RE = re.compile(
    r"^(?:\d+(?:girl|boy|other)s?|multiple[ _](?:girls|boys|others))$"
)


def _is_solo(names: Set[str]) -> bool:
    """True when the caption is single-subject (one count tag, no multi tag).

    The single-count names (``1girl``…) also satisfy ``_MULTI_COUNT_RE``
    (``\\d+girls?``), so they must be excluded from the multi test first —
    exactly the precedence role_markers gets from its ``elif``.
    """
    has_single = bool(names & _SINGLE_COUNT_NAMES)
    has_multi = any(
        n not in _SINGLE_COUNT_NAMES and _MULTI_COUNT_RE.match(n) for n in names
    )
    return has_single and not has_multi


def _slug(path: str) -> str:
    """Slug a ``대분류 > 소분류`` path into a YAML-key-safe token.

    Uses the ``소분류`` (last segment) for readability, keeps unicode word
    characters, and collapses separators to ``_``. Collisions are resolved by
    the caller (it prefixes the ``대분류`` segment).
    """
    leaf = path.split(">")[-1].strip()
    leaf = re.sub(r"[\\s/]+", "_", leaf)
    return re.sub(r"_+", "_", leaf).strip("_") or "group"


# --------------------------------------------------------------------------- #
# Co-occurrence sources — manifest (fast, index-based) or caption scan.
# --------------------------------------------------------------------------- #


def _solo_sets_from_manifest(vocab: dict, manifest: dict) -> List[Set[str]]:
    """Per-sample tag-name sets restricted to single-subject samples."""
    idx2name = {int(t["index"]): t["name"] for t in vocab["tags"]}
    out: List[Set[str]] = []
    for idxs in manifest.get("tag_indices", []):
        names = {idx2name[i] for i in idxs if i in idx2name}
        if _is_solo(names):
            out.append(names)
    return out


def solo_sets_from_index(index) -> List[Set[str]]:
    """Solo tag-name sets from a ``build_caption_index`` result.

    ``index`` maps ``stem → (caption_path, image_path, parsed_tags)``. Lets
    ``build_vocab`` derive groups from the scan it already did — no second pass.
    """
    out: List[Set[str]] = []
    for _stem, (_cap, _img, tags) in index.items():
        names = set(tags)
        if _is_solo(names):
            out.append(names)
    return out


def _solo_sets_from_captions(caption_roots: Sequence[Path], rules) -> List[Set[str]]:
    """Scan caption ``.txt`` files (rules applied) → solo tag-name sets."""
    from .vocab import build_caption_index, find_caption_files

    paths = find_caption_files([Path(r) for r in caption_roots])
    index = build_caption_index(paths, rules)
    return solo_sets_from_index(index)


# --------------------------------------------------------------------------- #
# Group derivation.
# --------------------------------------------------------------------------- #


def build_groups(
    vocab: dict,
    kb,
    rules,
    *,
    min_group_size: int,
    min_member_freq: int,
    rename_recovery: Dict[str, str],
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Bucket general vocab tags by taxonomy ``소분류``.

    Returns ``(path → [member names], unmatched names)``. ``rename_recovery``
    maps our (post-rule) name → original danbooru name, consulted when a vocab
    tag isn't found in the KB under its current name.
    """
    name2freq = {t["name"]: int(t["freq"]) for t in vocab["tags"]}
    by_path: Dict[str, List[str]] = {}
    unmatched: List[str] = []
    for t in vocab["tags"]:
        if t["category"] != "general":
            continue
        name = t["name"]
        if name2freq.get(name, 0) < min_member_freq:
            continue
        info = kb.describe(name)
        if info is None and name in rename_recovery:
            info = kb.describe(rename_recovery[name])
        if info is None or not info.category_path:
            unmatched.append(name)
            continue
        by_path.setdefault(info.category_path, []).append(name)
    groups = {
        p: sorted(m, key=lambda n: -name2freq.get(n, 0)) for p, m in by_path.items()
    }
    groups = {p: m for p, m in groups.items() if len(m) >= min_group_size}
    return groups, sorted(unmatched)


def score_group(
    members: List[str], solo_sets: List[Set[str]]
) -> Tuple[int, int, float]:
    """Return ``(n_solo_any, n_solo_multi, multi_rate)`` for a group.

    ``n_solo_any`` = solo images with ≥1 member; ``n_solo_multi`` = those with
    ≥2; rate = multi / any (0.0 when no support).
    """
    mset = set(members)
    n_any = n_multi = 0
    for s in solo_sets:
        k = len(s & mset)
        if k >= 1:
            n_any += 1
            if k >= 2:
                n_multi += 1
    rate = (n_multi / n_any) if n_any else 0.0
    return n_any, n_multi, rate


def decide_mode(
    n_any: int,
    multi_rate: float,
    *,
    min_support: int,
    softmax_cooc_max: float,
    borderline_cooc_max: float,
) -> Tuple[str, str, str]:
    """Pick ``(yaml_mode, tier, rationale)`` for a group.

    Three tiers, because danbooru's hierarchical double-tagging (``long hair`` +
    ``very long hair``) and 'mixed' members (``two-tone hair``) inflate the
    multi-rate of genuinely attribute-like families above a strict exclusivity
    cut. The YAML mode stays conservative (only the confident tier becomes
    ``softmax_when_solo``); the borderline tier is emitted ``multilabel`` but
    loudly flagged so the curator can promote it with an ``escape:`` list.

    * ``confident`` (rate ≤ softmax_cooc_max)         → ``softmax_when_solo``
    * ``borderline`` (≤ borderline_cooc_max)          → ``multilabel`` (PROMOTE?)
    * ``multilabel`` (otherwise)                      → ``multilabel``
    """
    if n_any < min_support:
        return (
            "multilabel",
            "low_support",
            (f"low-support (n_solo={n_any}<{min_support}); verify by eye"),
        )
    if multi_rate <= softmax_cooc_max:
        return (
            "softmax_when_solo",
            "confident",
            (f"mutually exclusive (solo multi-rate {multi_rate:.1%})"),
        )
    if multi_rate <= borderline_cooc_max:
        return (
            "multilabel",
            "borderline",
            (
                f"borderline exclusive ({multi_rate:.1%}) — likely a 'primary' "
                "family inflated by hierarchical/mixed tags; consider "
                "softmax_when_solo + an escape: list"
            ),
        )
    return (
        "multilabel",
        "multilabel",
        f"members stack (solo multi-rate {multi_rate:.1%})",
    )


# --------------------------------------------------------------------------- #
# Emit.
# --------------------------------------------------------------------------- #


def _unique_key(path: str, used: Set[str]) -> str:
    key = _slug(path)
    if key in used:
        # Prefix the 대분류 to disambiguate two leaves sharing a name.
        head = _slug(path.split(">")[0])
        key = f"{head}_{key}"
    n = key
    i = 2
    while n in used:
        n = f"{key}_{i}"
        i += 1
    used.add(n)
    return n


def emit_yaml(rows: List[dict]) -> str:
    """Build a valid ``groups.yaml`` candidate with per-group review comments.

    Each group block is dumped via :func:`yaml.safe_dump` (guaranteed-valid,
    unicode-preserving) and preceded by a ``# REVIEW`` comment carrying the
    stats and the rationale for the chosen mode.
    """
    out: List[str] = [
        "# Auto-derived tag-group candidates — REVIEW before use.",
        "# Keys are slugged from the danbooru 소분류 taxonomy; rename to English.",
        "# softmax_when_solo groups: curate an `escape:` list for 'mixed' members",
        "# (e.g. heterochromia, multicolored hair). Drop groups you don't want.",
        "version: 1",
        "",
    ]
    used: Set[str] = set()
    for r in sorted(rows, key=lambda r: -len(r["tags"])):
        key = _unique_key(r["path"], used)
        block = {
            key: {
                "mode": r["mode"],
                "tags": r["tags"],
                "description": f"[{r['path']}] — {r['rationale']}",
            }
        }
        tag = "PROMOTE? " if r["tier"] == "borderline" else ""
        out.append(
            f"# REVIEW {tag}{key}: {len(r['tags'])} members, "
            f"n_solo_any={r['n_any']}, multi_rate={r['multi_rate']:.1%} → {r['mode']}"
        )
        out.append(f"#   {r['rationale']}")
        if r["escape_hints"] and r["tier"] in ("confident", "borderline"):
            out.append(f"#   escape candidates: {', '.join(r['escape_hints'])}")
        out.append(yaml.safe_dump(block, allow_unicode=True, sort_keys=False).rstrip())
        out.append("")
    return "\n".join(out)


def _escape_hints(members: List[str]) -> List[str]:
    """Heuristic 'mixed' member names worth considering as escape tags."""
    pat = re.compile(
        r"\b(multicolored|two-tone|gradient|streaked|hetero|multiple|mixed)\b"
    )
    return [m for m in members if pat.search(m)]


def derive_rows(
    vocab: dict,
    kb,
    rules,
    solo_sets: List[Set[str]],
    *,
    min_group_size: int,
    min_member_freq: int,
    min_group_support: int,
    softmax_cooc_max: float,
    borderline_cooc_max: float,
) -> Tuple[List[dict], List[str]]:
    """Bucket general vocab by taxonomy, score co-occurrence, decide mode.

    The shared core of ``--mode derive_groups`` and ``build_vocab
    --derive_groups``. Returns ``(rows, unmatched)`` where each row carries the
    members, support stats, mode, tier and escape hints for one taxonomy bucket.
    """
    rename_recovery = {tgt: src for src, tgt in rules.replacements}
    groups, unmatched = build_groups(
        vocab,
        kb,
        rules,
        min_group_size=min_group_size,
        min_member_freq=min_member_freq,
        rename_recovery=rename_recovery,
    )
    rows: List[dict] = []
    for path, members in groups.items():
        n_any, n_multi, rate = score_group(members, solo_sets)
        mode, tier, rationale = decide_mode(
            n_any,
            rate,
            min_support=min_group_support,
            softmax_cooc_max=softmax_cooc_max,
            borderline_cooc_max=borderline_cooc_max,
        )
        rows.append(
            {
                "path": path,
                "tags": members,
                "n_any": n_any,
                "n_multi": n_multi,
                "multi_rate": rate,
                "mode": mode,
                "tier": tier,
                "rationale": rationale,
                "escape_hints": _escape_hints(members),
            }
        )
    return rows, unmatched


# --------------------------------------------------------------------------- #
# --apply: merge derived groups into a curated, English-keyed groups.yaml.
#
# Curation table — Korean taxonomy path → English group key. Where a derived
# key collides with a group the preserved groups.yaml already defines (eye_color
# / hair_color / hair_length), the preserved hand-curated group wins and the
# derived one is dropped — see merge_apply. With nothing to preserve, those
# attribute families are emitted as derived softmax groups instead.
# --------------------------------------------------------------------------- #

_EN_KEY: Dict[str, str] = {
    "얼굴/눈 > 눈 색상": "eye_color",
    "머리카락 > 머리 색상": "hair_color",
    "머리카락 > 머리 길이": "hair_length",
    "행동 > 성적행위": "sexual_acts",
    "신체 > 신체 부위": "body_parts",
    "포즈/구도 > 포즈": "pose",
    "의상 > 의상 디테일": "clothing_details",
    "머리카락 > 헤어스타일": "hairstyle",
    "액세서리 > 장식 요소": "decorative_elements",
    "표정/감정 > 표정": "expression",
    "의상 > 속옷": "underwear",
    "행동 > 상호작용": "interaction",
    "액세서리 > 머리장식": "hair_accessory",
    "의상 > 상의": "top_clothing",
    "동물/생물 > 동물 부위": "animal_parts",
    "사물 > 기타 사물": "misc_objects",
    "의상 > 수영복": "swimwear",
    "얼굴/눈 > 얼굴 특징": "face_features",
    "액세서리 > 양말/스타킹": "socks_stockings",
    "의상 > 하의": "bottom_clothing",
    "신체 > 피부": "skin",
    "표정/감정 > 제스처": "gesture",
    "효과/연출 > 텍스트/말풍선": "text_speech_bubble",
    "인물 > 패션 스타일": "fashion_style",
    "의상 > 코스튬": "costume",
    "신체 > 체형": "body_shape",
    "효과/연출 > 배경효과": "background_effect",
    "행동 > 일상행동": "daily_action",
    "포즈/구도 > 시점/앵글": "viewpoint_angle",
    "포즈/구도 > 프레이밍": "framing",
    "사물 > 가구": "furniture",
    "인물 > 종족/비인간": "species_nonhuman",
    "인물 > 인물 관계": "character_relationship",
    "액세서리 > 귀걸이/목걸이": "earrings_necklace",
    "표정/감정 > 감정": "emotion",
    "액세서리 > 신발": "footwear",
    "배경/장소 > 배경 디테일": "background_detail",
    "색상/패턴 > 패턴/무늬": "pattern",
    "액세서리 > 모자": "hat",
    "의상 > 원피스/드레스": "dress_onepiece",
    "배경/장소 > 실내": "indoor",
    "액세서리 > 장갑": "gloves_acc",
    "얼굴/눈 > 눈 형태": "eye_shape",
    "배경/장소 > 자연": "nature",
    "메타 > 구도기법": "composition_technique",
    "메타 > 장르": "genre",
    "동물/생물 > 식물": "plants",
    "사물 > 음식/음료": "food_drink",
    "효과/연출 > 날씨": "weather",
    "액세서리 > 안경/아이웨어": "eyewear",
    "메타 > 매체": "medium",
    "의상 > 교복": "school_uniform",
    "액세서리 > 가방": "bag",
    "의상 > 전통의상": "traditional_clothing",
    "인물 > 연령 변화": "age",
    "사물 > 전자기기": "electronics",
    "효과/연출 > 조명": "lighting",
    "인물 > 성별": "gender",
}

# Groups to emit as softmax_when_solo: the confident, conceptually exclusive
# attribute families. eye_shape is deliberately NOT here — the data shows shape
# tags co-occur (closed eyes + tsurime), so forcing a single pick would be wrong
# supervision; it stays multilabel. The eye/hair-color/length entries only fire
# when there is no preserved groups.yaml to defer to.
_PROMOTE_SOFTMAX = {
    "얼굴/눈 > 눈 색상",
    "머리카락 > 머리 색상",
    "머리카락 > 머리 길이",
    "인물 > 연령 변화",
    "효과/연출 > 조명",
    "인물 > 성별",
}


def merge_apply(
    rows: List[dict],
    existing_yaml: Optional[Path],
    *,
    min_group_size: int,
) -> Tuple[str, List[str]]:
    """Merge derived ``rows`` onto an existing groups.yaml.

    Existing groups are preserved verbatim and claim their tags first; derived
    groups then add only their *unclaimed* members (tags are disjoint across
    groups — a loader invariant). Returns ``(yaml_text, notes)``.
    """
    notes: List[str] = []
    claimed: Set[str] = set()
    preserved_blocks: List[str] = []
    preserved_keys: Set[str] = set()

    if existing_yaml and existing_yaml.exists():
        existing = yaml.safe_load(existing_yaml.read_text()) or {}
        existing.pop("version", None)
        for key, body in existing.items():
            preserved_keys.add(key)
            for t in body.get("tags", []) or []:
                claimed.add(t)
            preserved_blocks.append(
                yaml.safe_dump(
                    {key: body}, allow_unicode=True, sort_keys=False
                ).rstrip()
            )
        notes.append(f"preserved {len(existing)} existing group(s) verbatim")

    new_blocks: List[str] = []
    n_new = n_promoted = 0
    for r in sorted(rows, key=lambda r: -len(r["tags"])):
        path = r["path"]
        key = _EN_KEY.get(path) or _slug(path)
        # A derived group whose key the preserved file already defines defers to
        # the hand-curated one (e.g. eye_color with its tuned escape list).
        if key in preserved_keys:
            continue
        members = [t for t in r["tags"] if t not in claimed]
        if len(members) < min_group_size:
            if r["tags"]:
                notes.append(
                    f"dropped [{path}] — only {len(members)} unclaimed member(s)"
                )
            continue
        mode = "softmax_when_solo" if path in _PROMOTE_SOFTMAX else "multilabel"
        body = {"mode": mode, "tags": members}
        escape = _escape_hints(members)
        if mode == "softmax_when_solo" and escape:
            body["escape"] = escape
        body["description"] = (
            f"[{path}] {r['multi_rate']:.0%} solo multi-rate, n={r['n_any']}"
        )
        for t in members:
            claimed.add(t)
        if mode == "softmax_when_solo":
            n_promoted += 1
        n_new += 1
        new_blocks.append(
            yaml.safe_dump({key: body}, allow_unicode=True, sort_keys=False).rstrip()
        )

    notes.append(f"added {n_new} new group(s) ({n_promoted} softmax_when_solo)")
    header = [
        "# Anima tagger tag-groups — curated from the danbooru taxonomy KB.",
        "# Existing curated groups preserved verbatim; new groups derived via",
        "# scripts.anima_tagger.cli --mode derive_groups --apply.",
        "version: 1",
        "",
    ]
    text = "\n".join(header + preserved_blocks + [""] + new_blocks) + "\n"
    return text, notes


def cmd_derive_groups(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    vocab_path = out_dir / "vocab.json"
    if not vocab_path.exists():
        raise SystemExit(f"need {vocab_path} — run --mode build_vocab first")
    vocab = json.loads(vocab_path.read_text())

    kb = load_tag_knowledge_base(args.tag_cache)

    # Rules: drive caption co-occurrence + build the inverse rename map.
    from library.captioning import tag_rules as tr

    rules = (
        tr.load_rules(Path(args.rules))
        if args.rules and Path(args.rules).exists()
        else tr.TagRules((), frozenset(), {}, {}, ())
    )
    # Co-occurrence source: prefer the manifest (trainer's exact view), else
    # scan captions through the same rules.
    manifest_path = out_dir / "dataset.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        solo_sets = _solo_sets_from_manifest(vocab, manifest)
        source = f"manifest ({manifest_path.name})"
    else:
        solo_sets = _solo_sets_from_captions(args.caption_roots, rules)
        source = "caption scan"
    logger.info(
        "co-occurrence from %s — %d single-subject samples", source, len(solo_sets)
    )

    rows, unmatched = derive_rows(
        vocab,
        kb,
        rules,
        solo_sets,
        min_group_size=args.min_group_size,
        min_member_freq=args.min_member_freq,
        min_group_support=args.min_group_support,
        softmax_cooc_max=args.softmax_cooc_max,
        borderline_cooc_max=args.borderline_cooc_max,
    )

    n_general = sum(1 for t in vocab["tags"] if t["category"] == "general")
    n_softmax = sum(1 for r in rows if r["mode"] == "softmax_when_solo")
    logger.info(
        "%d general tags → %d candidate groups (%d softmax_when_solo, %d multilabel); "
        "%d general tags unmatched in KB",
        n_general,
        len(rows),
        n_softmax,
        len(rows) - n_softmax,
        len(unmatched),
    )

    if args.report:
        _print_report(rows, unmatched, n_general)

    if args.apply:
        preserve = Path(args.preserve_groups) if args.preserve_groups else None
        text, notes = merge_apply(rows, preserve, min_group_size=args.min_group_size)
        dest = Path(args.out_yaml) if args.out_yaml else out_dir / "groups.yaml"
        # Back up an existing destination before overwriting (merge_apply has
        # already read --preserve_groups into memory, so this is safe even when
        # dest == preserve).
        if dest.exists():
            backup = dest.with_suffix(dest.suffix + ".bak")
            backup.write_text(dest.read_text(), encoding="utf-8")
            logger.info("backed up %s → %s", dest, backup)
        dest.write_text(text, encoding="utf-8")
        # Validate the merged file loads through the real consumer.
        from library.captioning import tag_groups as tg

        merged = tg.load_groups(dest)
        for n in notes:
            logger.info("  %s", n)
        logger.info(
            "wrote merged groups.yaml (%d groups) → %s", len(merged.groups), dest
        )
    elif args.out_yaml:
        Path(args.out_yaml).write_text(emit_yaml(rows), encoding="utf-8")
        logger.info("wrote %d candidate groups → %s", len(rows), args.out_yaml)
    else:
        logger.info("no --out_yaml; pass one to write the candidate file")


def _print_report(rows: List[dict], unmatched: List[str], n_general: int) -> None:
    matched = n_general - len(unmatched)
    print(
        f"\nKB taxonomy coverage: {matched}/{n_general} general tags matched "
        f"({matched / n_general:.1%}); {len(unmatched)} unmatched\n"
    )
    head = f"{'tier':<11}  {'members':>7}  {'solo_any':>8}  {'multi%':>6}  group"
    for tier in ("confident", "borderline", "multilabel", "low_support"):
        sub = [r for r in rows if r["tier"] == tier]
        if not sub:
            continue
        print(f"\n=== {tier} ({len(sub)}) ===")
        print(head)
        print("-" * len(head))
        for r in sorted(sub, key=lambda r: -len(r["tags"])):
            print(
                f"{r['tier']:<11}  {len(r['tags']):>7d}  {r['n_any']:>8d}  "
                f"{r['multi_rate'] * 100:>5.1f}%  [{r['path']}]"
            )
    if unmatched:
        print(f"\nUnmatched general tags ({len(unmatched)}) — no taxonomy in KB:")
        print("  " + ", ".join(unmatched))
