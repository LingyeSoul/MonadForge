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

``--features`` (user 10-15, after ``hira_cold_sglyph``: the misses keep the
base glyph and lose the mark) swaps the bitmap for a PE tower's features on
the same glyphs, drawn black on white at the tower's native size (em box
scaled with it), averaged over the faces, row-L2'd, then the same PCA /
ridge / read: ``pe_core`` (PE-Core's attn-pooled embedding), ``pe_core_grid`` /
``pe_spatial`` / ``pe_spatial_l`` (the patch tokens average-pooled to a
``GRID``² grid). PE kinds need the GPU (run through the daemon); their
features are cached under ``OUT/glyph_init/feats/``. Every set is also read
**composed**: a (han)dakuten hiragana's prediction + the mark's mean ``e``
step over 0930's katakana pairs (バ − ハ), so the mark comes from another
script; ``oracle_base`` puts the step on the true base row instead (leaks
hiragana, diagnostic only). The rank is read on predictions centred over the
81, as the init starts them.

The predictions land in ``OUT/glyph_init/<train set>.pt`` (``ext_ids``,
``e_hat`` in effective units, ``beta`` — the scale cross-validated on the
training rows), for ``init = "stick_glyph"``; a non-bitmap kind writes
``<kind>-<set>.pt`` and the composed one ``…+comp.pt``.

    .venv/bin/python project/cjk_anima_reseed/probes/glyph_init.py
    make daemon-run ARGS="project/cjk_anima_reseed/probes/glyph_init.py \
        --features bitmap,pe_core,pe_core_grid,pe_spatial,pe_spatial_l"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
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
GRID = 4  # the PE patch tokens' pooled grid
# kind → (registry name in ``library.vision.encoders``, native px, pooled / grid)
PE_KINDS = {
    "pe_core": ("pe", 336, "pooled"),
    "pe_core_grid": ("pe", 336, "grid"),
    "pe_spatial": ("pe_spatial", 512, "grid"),
    "pe_spatial_l": ("pe_spatial_l", 448, "grid"),
}
MARKS = ("゙", "゚")  # combining (han)dakuten, NFD's second code point


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


@torch.no_grad()
def pe_feats(chars: list, kind: str, face_paths=None) -> np.ndarray:
    """``(n, d)``: each glyph black on white at the tower's native px (the em
    box scaled from ``EM / CANVAS``), one forward per glyph over the faces that
    have it (``face_paths``; default the top-level faces), the features
    averaged over those faces."""
    from fontTools.ttLib import TTFont
    from PIL import Image, ImageDraw, ImageFont

    from library.vision.encoder import load_pe_encoder

    name, px, mode = PE_KINDS[kind]
    em = round(px * EM / CANVAS)
    faces = []
    for p in face_paths or sorted(FONT_DIR.glob("*.[ot]tf")):
        cmap = TTFont(str(p), lazy=True).getBestCmap() or {}
        faces.append((ImageFont.truetype(str(p), em), set(cmap)))
    dev = torch.device("cuda")
    enc = load_pe_encoder(dev, name=name, dtype=torch.bfloat16).encoder
    out = None
    for i, c in enumerate(chars):
        ims = []
        for font, cov in faces:
            if ord(c) not in cov:
                continue
            im = Image.new("L", (px, px), 255)
            ImageDraw.Draw(im).text((px / 2, px / 2), c, 0, font, anchor="mm")
            ims.append(np.asarray(im, np.float32))
        x = torch.from_numpy(np.stack(ims)).to(dev) / 127.5 - 1
        x = x[:, None].expand(-1, 3, -1, -1).to(torch.bfloat16)
        o = enc(x)
        if mode == "pooled":
            f = o.pooler_output
        else:
            t = o.last_hidden_state[:, 1:]  # CLS first
            g = int(round(t.shape[1] ** 0.5))
            assert g * g == t.shape[1], t.shape
            t = t.transpose(1, 2).reshape(len(ims), -1, g, g).float()
            f = torch.nn.functional.adaptive_avg_pool2d(t, GRID).flatten(1)
        f = f.float().mean(0).cpu().numpy()
        if out is None:
            out = np.zeros((len(chars), f.size), np.float32)
        out[i] = f
        if i % 200 == 0:
            print(f"  {kind}: {i} / {len(chars)}", flush=True)
    return out


def features(kind: str, chars: list) -> np.ndarray:
    """``(n, N_PC)``: the kind's per-glyph features, row-L2'd for a PE kind,
    PCA'd and scaled by the first PC's std (cached under ``feats/``)."""
    from sklearn.decomposition import PCA

    if kind == "bitmap":
        F = bitmaps(chars)
    else:
        cache = OUT / "glyph_init" / "feats" / f"{kind}.npz"
        hit = cache.exists() and list(np.load(cache)["chars"]) == chars
        if hit:
            F = np.load(cache)["F"]
        else:
            F = pe_feats(chars, kind)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, F=F, chars=np.array(chars))
        F = F / np.linalg.norm(F, axis=1, keepdims=True)
    pca = PCA(N_PC, random_state=0).fit(F)
    Z = pca.transform(F)
    print(
        f"{kind}: features {F.shape}, PCA {N_PC} "
        f"({pca.explained_variance_ratio_.sum():.3f} var)",
        flush=True,
    )
    return Z / Z[:, :1].std()


def mark_steps(chars: list, e: np.ndarray) -> dict:
    """``{mark: mean e(marked) − e(base)}`` over the pairs inside ``chars``
    (0930's katakana: バ − ハ, パ − ハ, ヴ − ウ …)."""
    at = {c: i for i, c in enumerate(chars)}
    steps = {m: [] for m in MARKS}
    for c in chars:
        d = unicodedata.normalize("NFD", c)
        if len(d) == 2 and d[1] in steps and d[0] in at:
            steps[d[1]].append(e[at[c]] - e[at[d[0]]])
    return {m: np.mean(v, 0) for m, v in steps.items() if v}


def compose(chars: list, pred: np.ndarray, steps: dict, base=None) -> np.ndarray:
    """A marked glyph's row + the mark's step: on its own prediction, or with
    ``base`` (a table of the same glyphs: the oracle read) on its base's row;
    every other row ``pred``'s. 0930's marked rows are not their base's
    (cos(e(は), e(ば)) 0.25), but the mean step is shared across the scripts
    (dakuten: cos 0.67 katakana ↔ hiragana)."""
    at = {c: i for i, c in enumerate(chars)}
    out = pred.copy()
    for i, c in enumerate(chars):
        d = unicodedata.normalize("NFD", c)
        if len(d) == 2 and d[1] in steps and d[0] in at:
            out[i] = (pred[i] if base is None else base[at[d[0]]]) + steps[d[1]]
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
    """cos to the true row, and the true row's rank among the 81; the rank on
    the prediction centred over the 81 (as ``Rows._rand_start`` starts it)."""
    p = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    t = true / np.linalg.norm(true, axis=1, keepdims=True)
    pc = pred - pred.mean(0)
    pc /= np.linalg.norm(pc, axis=1, keepdims=True)
    sim = pc @ t.T
    rank = (sim > np.diag(sim)[:, None]).sum(1) + 1
    return {
        "cos": round(float((p * t).sum(1).mean()), 4),
        "cos_centred": round(float((pc * t).sum(1).mean()), 4),
        "rank_mean": round(float(rank.mean()), 1),
        "top1": int((rank == 1).sum()),
        "top5": int((rank <= 5).sum()),
        "pred_pairwise_cos": round(
            float((p @ p.T)[np.triu_indices(len(p), 1)].mean()), 3
        ),
    }


def main():
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict

    from library.anima.vocab_pack import load_vocab_pack
    from library.env import default_checkpoints
    from library.inference.text import ensure_text_strategies
    from reseed.config import load

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features",
        default="bitmap",
        help="comma list: bitmap, " + ", ".join(PE_KINDS),
    )
    kinds = ap.parse_args().features.split(",")
    assert all(k == "bitmap" or k in PE_KINDS for k in kinds), kinds
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
    n_h, n_k = len(chars["hira"]), len(chars["kata"])
    Y = {f: E[f].numpy() for f in E}
    steps = mark_steps(chars["kata"], Y["kata"])
    n_marked = int((compose(chars["hira"], Y["hira"], steps) != Y["hira"]).any(1).sum())
    print(
        f"mark steps from {len(chars['kata'])} katakana: {sorted(steps)}; "
        f"{n_marked} hiragana composed",
        flush=True,
    )

    alphas = np.logspace(-1, 4, 16)
    res = {"geometry": geo, "marked_hira": n_marked, "sets": {}}
    for kind in kinds:
        Z = features(kind, allc)
        X = {"hira": Z[:n_h], "kata": Z[n_h : n_h + n_k], "kanji": Z[n_h + n_k :]}
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
            # composed in effective units: the step is a true-e step
            pc = compose(chars["hira"], beta * ph, steps)
            key = name if kind == "bitmap" else f"{kind}-{name}"
            res["sets"][key] = {
                "n_train": len(Yt),
                "alpha": float(m.alpha_),
                "train_cv_cos": round(cv_cos, 4),
                "beta": round(beta, 3),
                "pred_norm": round(float(np.linalg.norm(beta * ph, axis=1).mean()), 1),
                "hira": read(ph, Y["hira"]),
                "hira_comp": read(pc, Y["hira"]),
                "hira_oracle_base": read(
                    compose(chars["hira"], beta * ph, steps, base=Y["hira"]), Y["hira"]
                ),
                "hira_shuffled": read(shuf, Y["hira"]),
                "hira_knn5": read(knn, Y["hira"]),
            }
            print(key, json.dumps(res["sets"][key], ensure_ascii=False), flush=True)
            dst = OUT / "glyph_init"
            dst.mkdir(parents=True, exist_ok=True)
            for suffix, e_hat in (("", beta * ph), ("+comp", pc)):
                torch.save(
                    {
                        "ext_ids": fam["hira"],
                        "chars": chars["hira"],
                        "e_hat": torch.from_numpy(e_hat).float(),
                        "beta": beta,
                        "train": list(fs),
                        "features": kind,
                        "composed": bool(suffix),
                        "seed_rows": str(SEED_ROWS),
                    },
                    dst / f"{key}{suffix}.pt",
                )

    print(
        "\nset (centred rank)            top1 / rank / cos   composed: top1 / rank / cos"
    )
    for k, v in res["sets"].items():
        a, c = v["hira"], v["hira_comp"]
        print(
            f"  {k:<26} {a['top1']:>3} / {a['rank_mean']:>4} / {a['cos']:.3f}"
            f"        {c['top1']:>3} / {c['rank_mean']:>4} / {c['cos']:.3f}",
            flush=True,
        )

    rd = HOME / "results" / f"{time.strftime('%Y%m%d-%H%M')}-glyph_init"
    if kinds != ["bitmap"]:
        rd = rd.with_name(rd.name + "_" + "+".join(kinds))
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
