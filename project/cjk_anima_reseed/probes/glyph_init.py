#!/usr/bin/env python
"""glyph_init — can a glyph's bitmap predict its row's glyph component? (user
10-15, after ``hira_cold_sball``: a better init space than a random ball)

``status.md`` § 7 splits a trained row's offset as ``s + (1 − α)·q_i … + e_i``:
``e_i`` is the glyph (bitmap Spearman 0.28 / 0.18, kana NN rank 1), ``s`` the
family's stick, ``q_i`` the pack row's spike. ``hira_cold_sball`` started the
81 hiragana at pack + 0930's hiragana stick + a random ball; this asks whether
``e_i`` can be predicted instead, from rows the hiragana never see.

CPU. Per family (0930's hiragana / katakana / kanji rows): ``s`` = the mean
offset, ``q_i`` = pack row − the family's mean pack row, ``e_i`` = (offset −
``s``) with ``q̂_i`` projected out. Features: each glyph drawn on the em box
(size and position kept, so small kana stay small) in every top-level face
(``assets/fonts``) whose cmap covers it, blurred, averaged over faces; PCA on
every glyph's bitmap. A ridge (``RidgeCV``) from those features to ``e_i`` is
fitted on katakana / kanji / both, and read against 0930's own hiragana
``e_i`` — the answer the cold arms never see: cos, and the rank of the true
row among the 81 (chance mean 41). Controls: the same ridge on shuffled
targets, and the mean ``e`` of the 5 bitmap-nearest training glyphs.

The predictions land in ``OUT/glyph_init/<train set>.pt`` (``ext_ids``,
``e_hat`` in effective units, ``beta`` — the scale cross-validated on the
training rows), for ``init = "stick_glyph"``.

    .venv/bin/python project/cjk_anima_reseed/probes/glyph_init.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reseed import HOME, OUT, SEED_ROWS, bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402

FONT_DIR = HOME / "assets" / "fonts"
CANVAS, EM, BLUR = 48, 40, 1.0
N_PC = 128
SHUFFLE_SEED = 0


def families(ids: list, text: dict) -> dict:
    """``{family: [ext id]}`` over 0930's single-glyph rows."""
    out = {"hira": [], "kata": [], "kanji": []}
    for e in ids:
        t = text.get(e, "")
        if len(t) != 1:
            continue
        o = ord(t)
        if 0x3041 <= o <= 0x309F:
            out["hira"].append(e)
        elif 0x30A1 <= o <= 0x30FA:
            out["kata"].append(e)
        elif 0x4E00 <= o <= 0x9FFF:
            out["kanji"].append(e)
    return out


def bitmaps(chars: list) -> np.ndarray:
    """``(n, CANVAS²)``: each glyph centred on the em box, per-face, blurred,
    averaged over the faces whose cmap has it."""
    from fontTools.ttLib import TTFont
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    faces = []
    for p in sorted(FONT_DIR.glob("*.[ot]tf")):
        cmap = TTFont(str(p), lazy=True).getBestCmap() or {}
        faces.append((ImageFont.truetype(str(p), EM), set(cmap)))
    out = np.zeros((len(chars), CANVAS * CANVAS), np.float32)
    for i, c in enumerate(chars):
        acc, n = np.zeros((CANVAS, CANVAS), np.float32), 0
        for font, cov in faces:
            if ord(c) not in cov:
                continue
            im = Image.new("L", (CANVAS, CANVAS), 0)
            ImageDraw.Draw(im).text((CANVAS / 2, CANVAS / 2), c, 255, font, anchor="mm")
            acc += (
                np.asarray(im.filter(ImageFilter.GaussianBlur(BLUR)), np.float32) / 255
            )
            n += 1
        assert n, f"no face covers {c!r}"
        out[i] = (acc / n).ravel()
    return out


def _covered() -> set:
    from fontTools.ttLib import TTFont

    out = set()
    for p in FONT_DIR.glob("*.[ot]tf"):
        out |= set(TTFont(str(p), lazy=True).getBestCmap() or {})
    return out


def decompose(off: torch.Tensor, pack: torch.Tensor) -> tuple:
    """``(e, s, R)``: the family's glyph components, stick and mean spike."""
    s = off.mean(0)
    q = pack - pack.mean(0)
    r = off - s
    qh = torch.nn.functional.normalize(q, dim=1)
    e = r - (r * qh).sum(1, keepdim=True) * qh
    return e, s, float((off - s).norm(dim=1).mean())


def read(pred: np.ndarray, true: np.ndarray) -> dict:
    """cos to the true row, and the true row's rank among the 81."""
    p = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    t = true / np.linalg.norm(true, axis=1, keepdims=True)
    sim = p @ t.T
    rank = (sim > np.diag(sim)[:, None]).sum(1) + 1
    pc = p - p.mean(0)
    pc /= np.linalg.norm(pc, axis=1, keepdims=True)
    return {
        "cos": round(float(np.diag(sim).mean()), 4),
        "cos_centred": round(float((pc * t).sum(1).mean()), 4),
        "rank_mean": round(float(rank.mean()), 1),
        "top1": int((rank == 1).sum()),
        "top5": int((rank <= 5).sum()),
        "pred_pairwise_cos": round(
            float((p @ p.T)[np.triu_indices(len(p), 1)].mean()), 3
        ),
    }


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict

    from library.anima.vocab_pack import load_vocab_pack
    from library.env import default_checkpoints
    from library.inference.text import ensure_text_strategies
    from reseed.config import load

    t0 = time.time()
    ck = default_checkpoints()
    tok, _ = ensure_text_strategies(ck.text_encoder, vocab_pack=ck.vocab_pack)
    pack = load_vocab_pack(ck.vocab_pack)
    src = torch.load(SEED_ROWS, map_location="cpu", weights_only=False)["delta"]
    ids = [int(e) for e in src["ext_ids"]]
    at = {e: i for i, e in enumerate(ids)}
    off_all = src["raw"].float() * float(src["row_scale"])
    text = _texts(tok, pack, ids)
    fam = families(ids, text)
    hira_run = [c for spec in load("hira_cold_hp").rows for c in spec.split(":", 1)[1]]
    fam["hira"] = [e for e in fam["hira"] if text[e] in set(hira_run)]
    print({k: len(v) for k, v in fam.items()}, f"(run: {len(hira_run)})", flush=True)
    assert len(fam["hira"]) == len(hira_run), "0930 must hold the run's 81 rows"

    covered = _covered()
    for f in ("kata", "kanji"):  # a training glyph no top-level face draws is left out
        fam[f] = [e for e in fam[f] if ord(text[e]) in covered]
    assert all(ord(text[e]) in covered for e in fam["hira"])
    E, S, R = {}, {}, {}
    for f, rows in fam.items():
        idx = [at[e] for e in rows]
        o = off_all[idx]
        keep = o.norm(dim=1) > 1.0
        rows = [e for e, k in zip(rows, keep.tolist()) if k]
        fam[f] = rows
        E[f], S[f], R[f] = decompose(
            off_all[[at[e] for e in rows]], pack.table[rows].float()
        )
    geo = {
        f: {
            "rows": len(fam[f]),
            "stick": round(float(S[f].norm()), 1),
            "spike": round(R[f], 1),
            "e_norm": round(float(E[f].norm(dim=1).mean()), 1),
        }
        for f in fam
    }
    geo["stick_cos"] = {
        a + "·" + b: round(
            float(torch.nn.functional.cosine_similarity(S[a], S[b], dim=0)), 3
        )
        for a, b in (("hira", "kata"), ("hira", "kanji"), ("kata", "kanji"))
    }
    print("geometry:", json.dumps(geo, ensure_ascii=False), flush=True)

    chars = {f: [text[e] for e in fam[f]] for f in fam}
    print({f: len(v) for f, v in chars.items()}, "drawn", flush=True)
    allc = chars["hira"] + chars["kata"] + chars["kanji"]
    B = bitmaps(allc)
    pca = PCA(N_PC, random_state=0).fit(B)
    Z = pca.transform(B)
    Z /= Z[:, :1].std()
    n_h, n_k = len(chars["hira"]), len(chars["kata"])
    X = {"hira": Z[:n_h], "kata": Z[n_h : n_h + n_k], "kanji": Z[n_h + n_k :]}
    Y = {f: E[f].numpy() for f in E}
    print(
        f"bitmaps {B.shape}, PCA {N_PC} ({pca.explained_variance_ratio_.sum():.3f} var), "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )

    alphas = np.logspace(-1, 4, 16)
    res = {"geometry": geo, "sets": {}}
    for name, fs in (
        ("kata", ("kata",)),
        ("kanji", ("kanji",)),
        ("kata_kanji", ("kata", "kanji")),
    ):
        Xt = np.concatenate([X[f] for f in fs])
        Yt = np.concatenate([Y[f] for f in fs])
        m = RidgeCV(alphas=alphas).fit(Xt, Yt)
        cvp = cross_val_predict(RidgeCV(alphas=alphas), Xt, Yt, cv=5)
        beta = float((cvp * Yt).sum() / (cvp * cvp).sum())
        cv_cos = float(
            (
                (cvp / np.linalg.norm(cvp, axis=1, keepdims=True))
                * (Yt / np.linalg.norm(Yt, axis=1, keepdims=True))
            )
            .sum(1)
            .mean()
        )
        ph = m.predict(X["hira"])
        rng = np.random.default_rng(SHUFFLE_SEED)
        shuf = (
            RidgeCV(alphas=alphas)
            .fit(Xt, Yt[rng.permutation(len(Yt))])
            .predict(X["hira"])
        )
        d = ((X["hira"][:, None] - Xt[None]) ** 2).sum(-1)
        knn = Yt[np.argsort(d, 1)[:, :5]].mean(1)
        res["sets"][name] = {
            "n_train": len(Yt),
            "alpha": float(m.alpha_),
            "train_cv_cos": round(cv_cos, 4),
            "beta": round(beta, 3),
            "pred_norm": round(float(np.linalg.norm(beta * ph, axis=1).mean()), 1),
            "hira": read(ph, Y["hira"]),
            "hira_shuffled": read(shuf, Y["hira"]),
            "hira_knn5": read(knn, Y["hira"]),
        }
        print(name, json.dumps(res["sets"][name], ensure_ascii=False), flush=True)
        dst = OUT / "glyph_init"
        dst.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ext_ids": fam["hira"],
                "chars": chars["hira"],
                "e_hat": torch.from_numpy(beta * ph).float(),
                "beta": beta,
                "train": list(fs),
                "seed_rows": str(SEED_ROWS),
            },
            dst / f"{name}.pt",
        )

    rd = HOME / "results" / f"{time.strftime('%Y%m%d-%H%M')}-glyph_init"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"→ {rd} ({time.time() - t0:.0f}s)", flush=True)


def _texts(tok, pack, ids) -> dict:
    """ext row → its piece text (``cjk_anima_scale``'s ``row_texts``, verbatim
    logic: a Qwen piece, a char row or a symbol row)."""
    inv_q = {int(v): int(k) for k, v in pack.mapping["qwen"].items()}
    inv_c = {int(v): k for k, v in pack.mapping.get("char", {}).items()}
    inv_s = {int(v): k for k, v in pack.mapping.get("sym_char", {}).items()}
    out = {}
    for r in ids:
        if r in inv_q:
            t = tok.qwen3_tokenizer.decode([inv_q[r]]).strip()
        elif r in inv_c:
            t = inv_c[r]
        elif r in inv_s:
            t = inv_s[r]
        else:
            continue
        if t:
            out[r] = t
    return out


if __name__ == "__main__":
    main()
