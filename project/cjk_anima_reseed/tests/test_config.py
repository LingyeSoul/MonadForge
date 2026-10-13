"""Every kept run config loads (cjk_ko's too), and every ``KEYS`` entry is
used by one of them (a key only archived configs used is dropped with its
code)."""

import sys
import tomllib
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))


def _configs() -> list:
    """This line's configs and cjk_ko's (it runs on this package)."""
    return sorted((HOME / "configs").glob("*.toml")) + sorted(
        (HOME.parent / "cjk_ko" / "configs").glob("*.toml")
    )


def test_configs_load():
    from reseed.config import load

    assert _configs()
    for f in _configs():
        run = load(str(f))
        assert run.seed_rows().name == "trained.pt", f.stem


def test_keys_used():
    from reseed.config import KEYS

    used = set().union(
        *(tomllib.loads(f.read_text(encoding="utf-8")) for f in _configs())
    )
    assert set(KEYS) == used, f"unused {sorted(set(KEYS) - used)}"


def test_cf_bubble_tiers_twin_bubble1():
    # cf_bubble1_<px> draws as bubble1_<px> does, with the counterfactual hosts
    from reseed.table import TABLE

    by = {t.name: t for t in TABLE}
    for px in ("52", "32"):
        cf, twin = by[f"cf_bubble1_{px}"], by[f"bubble1_{px}"]
        assert cf.share == 0 and cf.recipe == twin.recipe == "bubble1"
        assert (cf.band, cf.px_keep) == (twin.band, twin.px_keep)
        assert cf.params == {**twin.params, "cf": True}
