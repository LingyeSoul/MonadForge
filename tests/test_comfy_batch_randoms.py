"""__random__ pool substitution in scripts/toolkits/comfy_batch.py."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    """comfy_batch lives in scripts/toolkits/, which is not an installed package."""
    spec = importlib.util.spec_from_file_location(
        "comfy_batch", ROOT / "scripts" / "toolkits" / "comfy_batch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfy_batch"] = mod
    spec.loader.exec_module(mod)
    return mod


cb = _load_module()


def _wf(text):
    return {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": text}}}


def _text(wf):
    return wf["1"]["inputs"]["text"]


def _tags(entries):
    return [", ".join(e.tags) for e in entries]


@pytest.fixture
def pool(tmp_path):
    (tmp_path / "r.yaml").write_text(
        "default: [a]\na: [alpha, beta]\nb: [gamma, delta]\nunder_score: [eps]\n"
    )
    return cb.load_randoms(str(tmp_path / "r.yaml"))


def test_default_key_selects_groups(pool):
    assert _tags(pool.default) == ["alpha", "beta"]
    assert _tags(pool.pool_for("b")) == ["gamma", "delta"]


def test_group_union_and_dedup(pool):
    assert _tags(pool.pool_for("a|b")) == ["alpha", "beta", "gamma", "delta"]
    assert _tags(pool.pool_for("a|a")) == ["alpha", "beta"]


def test_group_name_may_contain_underscore(pool):
    wf, draws = cb.substitute_random(_wf("__random:under_score__"), pool)
    assert _text(wf) == "eps" and draws == ["eps"]


def test_bare_random_draws_from_default_only(pool):
    for _ in range(20):
        wf, _d = cb.substitute_random(_wf("__random__"), pool)
        assert _text(wf) in ("alpha", "beta")


def test_draws_without_replacement_within_a_job(pool):
    for _ in range(20):
        wf, draws = cb.substitute_random(_wf("__random__ / __random__"), pool)
        assert sorted(draws) == ["alpha", "beta"], _text(wf)


def test_exhausted_pool_repeats_rather_than_starving(pool):
    wf, draws = cb.substitute_random(_wf("__random__ __random__ __random__"), pool)
    assert len(draws) == 3 and "__random__" not in _text(wf)


def test_unknown_group_leaves_placeholder(pool, capsys):
    wf, draws = cb.substitute_random(_wf("__random:nope__"), pool)
    assert _text(wf) == "__random:nope__" and draws == []
    assert "unknown __random__ group" in capsys.readouterr().out


def test_empty_pool_leaves_placeholder(tmp_path, capsys):
    empty = cb.load_randoms(str(tmp_path / "missing.yaml"))
    wf, draws = cb.substitute_random(_wf("__random__"), empty)
    assert _text(wf) == "__random__" and draws == []
    assert "randoms file is empty/missing" in capsys.readouterr().out


def test_flat_txt_pool_still_works(tmp_path):
    (tmp_path / "r.txt").write_text("# comment\non bed\nclassroom\n")
    txt = cb.load_randoms(str(tmp_path / "r.txt"))
    assert _tags(txt.default) == ["on bed", "classroom"]
    assert _tags(txt.pool_for("all")) == ["on bed", "classroom"]


def test_draw_is_json_escaped(tmp_path):
    (tmp_path / "r.yaml").write_text("a: ['say \"hi\"']\n")
    p = cb.load_randoms(str(tmp_path / "r.yaml"))
    wf, _d = cb.substitute_random(_wf("__random__"), p)
    assert _text(wf) == 'say "hi"'


def test_bad_yaml_shapes_are_rejected(tmp_path):
    (tmp_path / "list.yaml").write_text("- a\n- b\n")
    with pytest.raises(SystemExit):
        cb.load_randoms(str(tmp_path / "list.yaml"))
    (tmp_path / "scalar.yaml").write_text("a: nope\n")
    with pytest.raises(SystemExit):
        cb.load_randoms(str(tmp_path / "scalar.yaml"))
    (tmp_path / "dangling.yaml").write_text("default: [missing]\na: [x]\n")
    with pytest.raises(SystemExit):
        cb.load_randoms(str(tmp_path / "dangling.yaml"))


def test_apply_prompt_targets_a_random_only_node():
    wf = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, __random__"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "worst quality"}},
    }
    cb.apply_prompt(wf, "replaced")
    assert wf["1"]["inputs"]["text"] == "replaced"
    assert wf["2"]["inputs"]["text"] == "worst quality"


# --- shipped-pool invariants -------------------------------------------------
# workflows/ is gitignored: both the generated pool (randoms.yaml) and the
# build spec (randoms_build.yaml) exist only on the working machine. The spec
# also carries every corpus-specific expectation these tests assert, so the
# committed test file stays content-free. Without the files, skip.

SPEC_PATH = ROOT / "workflows" / "randoms_build.yaml"
SHIPPED_PATH = ROOT / "workflows" / "randoms.yaml"
HAVE_SHIPPED = SPEC_PATH.exists() and SHIPPED_PATH.exists()
shipped_only = pytest.mark.skipif(
    not HAVE_SHIPPED, reason="gitignored workflows/ pool + build spec not present"
)
if HAVE_SHIPPED:
    import yaml

    SPEC = yaml.safe_load(SPEC_PATH.read_text())
    SHIPPED = cb.load_randoms(str(SHIPPED_PATH))
else:
    SPEC, SHIPPED = {}, None


@shipped_only
def test_shipped_pool_groups_resolve():
    """workflows/randoms.yaml: every slot the spec defines exists."""
    assert SHIPPED.default, "default draw pool is empty"
    for name in SPEC["slots"]:
        assert SHIPPED.pool_for(name), f"{name} slot is empty"
    for name in set(SPEC["slots"]) - set(SPEC["default_slots"]):
        # the exclusive axes must stay out of the bare-__random__ draw
        assert not set(SHIPPED.pool_for(name)) & set(SHIPPED.default)


@shipped_only
def test_shipped_position_always_names_an_act():
    """The bug this pool was rebuilt for: a position draw that isn't an act.

    Every position cluster must carry a tag that says what is happening —
    otherwise the prompt renders a pose with no scene.
    """
    acts = set(SPEC["position_acts"]) | set(SPEC["core_companions"])
    for entry in SHIPPED.pool_for(SPEC["position_slot"]):
        assert acts & set(entry.tags), f"no act in position cluster {entry.tags}"


@shipped_only
def test_shipped_clusters_carry_their_anatomy():
    """An act that needs a partner must name one (spec: requires_companion)."""
    for needed, acts in SPEC["test_invariants"]["requires_companion"].items():
        for entry in SHIPPED.pool_for(SPEC["position_slot"]):
            if set(acts) & set(entry.tags):
                assert needed in entry.tags, f"{entry.tags} lacks {needed}"


@shipped_only
def test_shipped_pool_excludes_identity_and_negatives():
    """Nothing from the block lists survives into any cluster."""
    banned = set(SPEC["block_tags"]) | set(SPEC["block_tags_deliberate"])
    # spot-checks for block_groups (identity axes owned by __chara__)
    banned |= {"long hair", "blonde hair", "blue eyes", "large breasts"}
    for slot, entries in SHIPPED.groups.items():
        for entry in entries:
            assert not banned & set(entry.tags), f"{slot}: {entry.tags}"


@shipped_only
def test_avoid_blocks_incompatible_draws():
    """An entry's avoid-list can never land in the same prompt as its anchor.

    The probe anchor must appear in exactly one entry across the whole pool,
    so seeing it drawn identifies the entry whose avoid-list applies.
    """
    from collections import Counter

    tag_count = Counter(
        t for entries in SHIPPED.groups.values() for e in entries for t in e.tags
    )
    probes = [
        e
        for e in SHIPPED.pool_for(SPEC["position_slot"])
        if e.avoid and tag_count[e.tags[0]] == 1
    ]
    assert probes, "no position entry with an avoid-list and a unique anchor"
    wf = _wf(", ".join(f"__random:{slot}__" for slot in SPEC["slots"]))
    for _ in range(120):
        out, _d = cb.substitute_random(wf, SHIPPED)
        tags = set(_text(out).split(", "))
        for probe in probes:
            if probe.tags[0] in tags:
                assert not tags & probe.avoid, (probe.tags[0], tags & probe.avoid)


@shipped_only
def test_pinned_prompt_tags_are_not_redrawn():
    """A tag the template already states is not repeated by a draw."""
    pinned = list(dict.fromkeys(SPEC["pinned"] + SPEC["context"]))
    wf = _wf(", ".join(pinned) + f", __random:{SPEC['position_slot']}__")
    for _ in range(40):
        out, draws = cb.substitute_random(wf, SHIPPED)
        rendered = _text(out).split(", ")
        assert len(rendered) == len(set(rendered)), rendered


def test_substitution_is_deterministic_under_seed(pool):
    random.seed(11)
    first = json.dumps(cb.substitute_random(_wf("__random__ __random:b__"), pool)[0])
    random.seed(11)
    second = json.dumps(cb.substitute_random(_wf("__random__ __random:b__"), pool)[0])
    assert first == second


def test_weight_parsing_and_validation(tmp_path):
    (tmp_path / "r.yaml").write_text(
        "a:\n  - tags: hot\n    weight: 2.0\n  - tags: cold\n    weight: 0.5\n"
    )
    p = cb.load_randoms(str(tmp_path / "r.yaml"))
    assert [e.weight for e in p.pool_for("a")] == [2.0, 0.5]
    (tmp_path / "bad.yaml").write_text("a:\n  - tags: x\n    weight: -1\n")
    with pytest.raises(SystemExit):
        cb.load_randoms(str(tmp_path / "bad.yaml"))


def test_weighted_draw_biases_toward_heavy_entries(tmp_path):
    (tmp_path / "r.yaml").write_text(
        "a:\n  - tags: hot\n    weight: 9.0\n  - tags: cold\n    weight: 1.0\n"
    )
    p = cb.load_randoms(str(tmp_path / "r.yaml"))
    random.seed(3)
    hot = sum(
        cb.substitute_random(_wf("__random:a__"), p)[1] == ["hot"] for _ in range(400)
    )
    assert 320 <= hot <= 400, hot  # E[hot] = 360


@shipped_only
def test_shipped_pool_carries_taste_weights():
    """Regenerating must keep selection-lift weights (needs the crawl volume)."""
    weighted = [
        e for entries in SHIPPED.groups.values() for e in entries if e.weight != 1.0
    ]
    assert len(weighted) > 50, "randoms.yaml lost its taste weights"
    by_anchor = {
        e.tags[0]: e.weight for entries in SHIPPED.groups.values() for e in entries
    }
    # the strongest measured preference and the strongest measured aversion
    probes = SPEC["test_invariants"]["taste_probes"]
    assert by_anchor.get(probes["high"]["tag"], 0) >= probes["high"]["min"]
    assert by_anchor.get(probes["low"]["tag"], 1.0) <= probes["low"]["max"]
