"""The line's configs load onto the reseed package, write under
``output/cjk_ko``, and the corpus / KO lettering rules hold."""

import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
RESEED = HOME.parent / "cjk_anima_reseed"
sys.path.insert(0, str(HOME))
sys.path.insert(0, str(RESEED))

from reseed import OUT, bootstrap  # noqa: E402

bootstrap()


def test_configs_load():
    from reseed.config import load

    fs = sorted((HOME / "configs").glob("*.toml"))
    assert fs
    for f in fs:
        run = load(str(f))
        assert run.dir == OUT.parent / "cjk_ko" / f.stem, run.dir
        if not f.stem.startswith("jamo_"):  # a jamo_* run draws no lines (phase 1)
            assert run.line_lang() == "korean", f.stem
        assert set(run.lang) == {g for r in run.rows for g in r.split(":", 1)[1]}


def test_warm_from_reseed():
    from reseed.config import load

    run = load(str(HOME / "configs" / "ko_chat_warm.toml"))
    assert run.seed_rows() == OUT / "jamo_j256_ko" / "trained.pt"


def test_normalise():
    from corpus import normalise

    assert normalise("하루가 또 가네요.") == "하루가 또 가네요"
    assert normalise("좋아...  진짜") == "좋아… 진짜"
    assert normalise("뭐해?") == "뭐해?"
    assert normalise("12시 땡!") is None  # a digit
    assert normalise("ㅋㅋ 웃겨") is None  # a standalone jamo
    assert normalise("또 왔어~") is None
    assert normalise("아.") is None  # one syllable


def test_sent_plan_h():
    from reseed import table as T
    from reseed.recipes import _sent_plan_h, _word_cuts

    from common.render.scene import split_lines

    text = "오늘 날씨 정말 좋다"
    cuts = _word_cuts(text)
    assert cuts == [3, 6, 9]
    got = _sent_plan_h((0, 0, 180, 220), text, cuts, 30.0, 25)
    assert got is not None
    k, f = got
    assert 1 <= k <= T.SENT_H_LINES and T.SENT_FILL[0] <= f <= T.SENT_FILL[1]
    lines = split_lines(text, k, cuts)
    assert all(ln[0] != " " for ln in lines)  # every line opens on a word


def test_word_pool():
    from reseed.pools import word_pool

    glyphs = set("오늘날씨정말좋다하하")
    got = word_pool(glyphs, ["오늘 날씨 정말 좋다?", "하하 좋다…"], held=("정말로",))
    # whole words only; 정말 holds no held trigram; 하하 repeats a glyph
    assert got == ["날씨", "오늘", "정말", "좋다"]
