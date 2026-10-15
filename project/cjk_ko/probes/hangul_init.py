#!/usr/bin/env python
"""hangul_init — can a syllable's glyph predict its row, and through it the
jamo factors? (user 10-15, after ``hira_cold_spe``: the PE-Spatial glyph
predicts a hiragana row at rank 2.3 / 81 from katakana + kanji rows)

Stage 1 of a factor init: offline, no training. The jamo curriculum's lone
init (``reports/jamo_curriculum_2026_10_11.md``) put a jamo's standalone row
into its factor and cost H. The trained factors were orthogonal to the lone
directions (cos ≈ 0). This instead predicts each *syllable* from its block's
image and projects the predictions onto the factor span, so it can fail the
same way, and the read checks that.

Targets: each factor run's trained syllables (``jamo_j256_ko``,
``jamo_j256_ko_cf``, ``jamo_j1024_ko``, cold on seed_1008). ``e`` is
``glyph_init.decompose`` over a run's trained syllables: offset − the
syllables' stick, ``q̂`` projected out.

- **ceiling**: the runs against each other on the syllables they share
  (cos, rank of the true row among the shared, chance n / 2).
- **direct**: a ridge from PE-Spatial-B16 4×4 features (``glyph_init``'s
  ``pe_spatial``) to ``e``, fitted on seed_1008's katakana + kanji rows, read
  on each run's syllables. The syllables are drawn in the KO faces
  (``kozh/`` + ``ko_extra/``, NanumPenScript out, as the runs' data). Rank
  among all of them, and among those of the same layout (V / H / C, ± final)
  only, so a prediction that knows only the block's layout cannot score.
  Control: shuffled targets.
- **projected**: the predictions over every syllable drawn, least-squares
  onto ``b + C[cho, cls] + V[jung] + F[jong]`` (the 106 factors), then read
  again. The fitted ``C`` / ``V`` / ``F`` are compared with each run's
  trained factors by mean cos (the curriculum's lone init: ≈ 0). The ceiling
  is j256_ko against j1024_ko, and the floor the same fit on shuffled
  predictions.

Writes ``output/cjk_ko/hangul_init/pe_spatial.pt`` (syllables, ``e_hat`` in
effective units, the fitted factors) and
``results/<ts>-hangul_init/result.json``.

    make daemon-run ARGS="project/cjk_ko/probes/hangul_init.py"
"""

from __future__ import annotations

import json
import sys
import time
import tomllib
from pathlib import Path

KO = Path(__file__).resolve().parents[1]
HOME = KO.parent / "cjk_anima_reseed"
sys.path.insert(0, str(HOME))
sys.path.insert(0, str(HOME / "probes"))
from reseed import OUT as ROUT  # noqa: E402
from reseed import REPO, bootstrap  # noqa: E402

bootstrap()

import glyph_init as G  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from reseed.jamo import JUNG_CLS, N_JONG, N_JUNG, decompose, is_syllable  # noqa: E402

KIND = "pe_spatial"
SEED_1008 = ROUT / "seed_1008" / "trained.pt"
RUNS = {
    "j256": (ROUT / "jamo_j256_ko", KO / "configs" / "jamo_j256_ko.toml"),
    "j256_cf": (
        REPO / "output/cjk_ko/jamo_j256_ko_cf",
        KO / "configs" / "jamo_j256_ko_cf.toml",
    ),
    "j1024": (
        REPO / "output/cjk_ko/jamo_j1024_ko",
        KO / "configs" / "jamo_j1024_ko.toml",
    ),
}
FONTS = HOME / "assets" / "fonts"
FACES_OUT = {"NanumPenScript-Regular.ttf"}
DST = REPO / "output" / "cjk_ko" / "hangul_init"
N_CELLS = 1 + 19 * 3 + N_JUNG + (N_JONG - 1)


def ko_faces() -> list:
    return [
        p
        for d in (FONTS / "kozh", FONTS / "ko_extra")
        for p in sorted(d.glob("*.[ot]tf"))
        if p.name not in FACES_OUT
    ]


def layout(s: str) -> str:
    _, jung, jong = decompose(s)
    return "VHC"[JUNG_CLS[jung]] + ("F" if jong else "")


def design(syl: list) -> np.ndarray:
    """``(n, 106)`` one-hot: b, C[cho, cls], V[jung], F[jong > 0]."""
    X = np.zeros((len(syl), N_CELLS), np.float32)
    for i, s in enumerate(syl):
        cho, jung, jong = decompose(s)
        X[i, 0] = 1
        X[i, 1 + cho * 3 + JUNG_CLS[jung]] = 1
        X[i, 58 + jung] = 1
        if jong:
            X[i, 58 + N_JUNG + jong - 1] = 1
    return X


def split(Z: np.ndarray) -> dict:
    """The fitted rows of ``design``'s columns as ``Jamo``'s b / C / V / F."""
    return {
        "b": Z[:1],
        "C": Z[1:58].reshape(19, 3, -1),
        "V": Z[58 : 58 + N_JUNG],
        "F": Z[58 + N_JUNG :],
    }


def factor_cos(a: dict, b: dict, used: set) -> dict:
    """Mean cos per factor over the cells ``used`` (C cells a syllable set
    touches; every V / F a syllable carries). Every syllable carries one C and
    one V, so a shift of every C by u and every V by −u composes the same:
    C and V are centred over their used cells first (F[0] = 0 pins F)."""
    out = {}
    for k in ("C", "V", "F"):
        x = torch.as_tensor(a[k]).reshape(-1, a[k].shape[-1]).float()
        y = torch.as_tensor(b[k]).reshape(-1, b[k].shape[-1]).float()
        m = [i for i in range(len(x)) if (k, i) in used]
        x, y = x[m], y[m]
        if k != "F":
            x, y = x - x.mean(0), y - y.mean(0)
        out[k] = round(float(F.cosine_similarity(x, y, dim=1).mean()), 3)
    return out


def used_cells(syl: list) -> set:
    u = set()
    for s in syl:
        cho, jung, jong = decompose(s)
        u |= {("C", cho * 3 + JUNG_CLS[jung]), ("V", jung)}
        if jong:
            u.add(("F", jong - 1))
    return u


def read_layout(pred: np.ndarray, true: np.ndarray, syl: list) -> dict:
    """The true row's rank among the syllables of its own layout (the
    prediction centred over all of them, as ``G.read``)."""
    pc = pred - pred.mean(0)
    pc /= np.linalg.norm(pc, axis=1, keepdims=True)
    t = true / np.linalg.norm(true, axis=1, keepdims=True)
    lay = np.array([layout(s) for s in syl])
    rank, chance = [], []
    for i in range(len(syl)):
        m = lay == lay[i]
        sim = t[m] @ pc[i]
        rank.append(int((sim > float(t[i] @ pc[i])).sum()) + 1)
        chance.append((m.sum() + 1) / 2)
    rank = np.array(rank)
    return {
        "rank_mean": round(float(rank.mean()), 1),
        "chance_mean": round(float(np.mean(chance)), 1),
        "top1": int((rank == 1).sum()),
        "top5": int((rank <= 5).sum()),
    }


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict

    from library.anima.vocab_pack import load_vocab_pack
    from library.env import default_checkpoints
    from library.inference.text import ensure_text_strategies

    t0 = time.time()
    ck = default_checkpoints()
    tok, _ = ensure_text_strategies(ck.text_encoder, vocab_pack=ck.vocab_pack)
    pack = load_vocab_pack(ck.vocab_pack)

    def load(path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        ids = [int(e) for e in d["delta"]["ext_ids"]]
        off = d["delta"]["raw"].float() * float(d["delta"]["row_scale"])
        return d, ids, off

    # -- the JP training rows (seed_1008's katakana + kanji) ----------------
    _, ids, off = load(SEED_1008)
    at = {e: i for i, e in enumerate(ids)}
    text = G._texts(tok, pack, ids)
    fam = G.families(ids, text)
    # the features come from glyph_init's cache (0930's glyphs): 1008's rows
    # whose glyph it holds
    jp_cached = set(np.load(ROUT / "glyph_init" / "feats" / f"{KIND}.npz")["chars"])
    jp = {}
    for f in ("kata", "kanji"):
        rows = [e for e in fam[f] if text[e] in jp_cached and off[at[e]].norm() > 1]
        e, _, _ = G.decompose(off[[at[r] for r in rows]], pack.table[rows].float())
        jp[f] = ([text[r] for r in rows], e.numpy())
        print(
            f"seed_1008 {f}: {len(rows)} rows ({len(fam[f])} in the family)", flush=True
        )

    # -- the KO targets: each run's trained syllables ----------------------
    runs = {}
    for name, (out, cfg) in RUNS.items():
        d, ids, off = load(out / "trained.pt")
        text = G._texts(tok, pack, ids)
        want = set(tomllib.load(open(cfg, "rb"))["rows"][0].split(":", 1)[1])
        pos = {text[e]: i for i, e in enumerate(ids) if text.get(e, "") in want}
        syl = sorted(s for s in pos if is_syllable(s))
        rows = [ids[pos[s]] for s in syl]
        e, s_, _ = G.decompose(off[[pos[s] for s in syl]], pack.table[rows].float())
        rs = float(d["delta"]["row_scale"])
        fac = {k: d["jamo"][k].float().numpy() * rs for k in ("b", "C", "V", "F")}
        runs[name] = {"syl": syl, "e": e.numpy(), "stick": float(s_.norm()), "fac": fac}
        print(
            f"{name}: {len(syl)} trained syllables, |stick| {float(s_.norm()):.0f}",
            flush=True,
        )
    h = json.loads((HOME / "assets" / "jamo_sets.json").read_text(encoding="utf-8"))[
        "H"
    ]
    drawn = sorted(set(h).union(*(r["syl"] for r in runs.values())))
    print(
        f"{len(drawn)} syllables drawn ({len(h)} H), {time.time() - t0:.0f}s",
        flush=True,
    )

    res = {"ceiling": {}, "direct": {}, "projected": {}, "factor_cos": {}}

    # -- ceiling: the runs against each other ------------------------------
    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(set(runs[a]["syl"]) & set(runs[b]["syl"]))
            ea = runs[a]["e"][[runs[a]["syl"].index(s) for s in shared]]
            eb = runs[b]["e"][[runs[b]["syl"].index(s) for s in shared]]
            r = {
                "n": len(shared),
                **G.read(ea, eb),
                "layout": read_layout(ea, eb, shared),
            }
            u = used_cells(shared)
            r["factor_cos"] = factor_cos(runs[a]["fac"], runs[b]["fac"], u)
            res["ceiling"][f"{a}·{b}"] = r
            print(f"ceiling {a}·{b}", json.dumps(r, ensure_ascii=False), flush=True)

    # -- features: the JP rows from glyph_init's cache, the syllables drawn --
    cache = ROUT / "glyph_init" / "feats" / f"{KIND}.npz"
    z = np.load(cache)
    jp_at = {c: i for i, c in enumerate(z["chars"])}
    kc = DST / "feats" / f"{KIND}.npz"
    if kc.exists() and list(np.load(kc)["chars"]) == drawn:
        FK = np.load(kc)["F"]
    else:
        FK = G.pe_feats(drawn, KIND, face_paths=ko_faces())
        kc.parent.mkdir(parents=True, exist_ok=True)
        np.savez(kc, F=FK, chars=np.array(drawn))
    jp_chars = jp["kata"][0] + jp["kanji"][0]
    FJ = z["F"][[jp_at[c] for c in jp_chars]]
    Fall = np.concatenate([FJ, FK])
    Fall /= np.linalg.norm(Fall, axis=1, keepdims=True)
    Z = PCA(G.N_PC, random_state=0).fit_transform(Fall)
    Z /= Z[:, :1].std()
    XJ, XK = Z[: len(FJ)], Z[len(FJ) :]
    YJ = np.concatenate([jp["kata"][1], jp["kanji"][1]])
    alphas = np.logspace(-1, 4, 16)
    m = RidgeCV(alphas=alphas).fit(XJ, YJ)
    cvp = cross_val_predict(RidgeCV(alphas=alphas), XJ, YJ, cv=5)
    beta = float((cvp * YJ).sum() / (cvp * cvp).sum())
    pk = beta * m.predict(XK)
    rng = np.random.default_rng(G.SHUFFLE_SEED)
    shuf = beta * RidgeCV(alphas=alphas).fit(XJ, YJ[rng.permutation(len(YJ))]).predict(
        XK
    )
    res["fit"] = {"n_train": len(YJ), "alpha": float(m.alpha_), "beta": round(beta, 3)}
    print("fit", res["fit"], flush=True)

    # -- projected: least squares onto the factor span ---------------------
    D = design(drawn)
    ridge = 1e-3 * np.eye(D.shape[1], dtype=np.float32)

    def project(P):
        W = np.linalg.solve(D.T @ D + ridge, D.T @ P)
        return D @ W, split(W)

    proj, fac_hat = project(pk)
    proj_s, fac_s = project(shuf)
    at_d = {s: i for i, s in enumerate(drawn)}
    for name, r in runs.items():
        sel = [at_d[s] for s in r["syl"]]
        u = used_cells(r["syl"])
        res["direct"][name] = {
            "n": len(sel),
            **G.read(pk[sel], r["e"]),
            "layout": read_layout(pk[sel], r["e"], r["syl"]),
            "shuffled": G.read(shuf[sel], r["e"]),
        }
        res["projected"][name] = {
            **G.read(proj[sel], r["e"]),
            "layout": read_layout(proj[sel], r["e"], r["syl"]),
            "shuffled": G.read(proj_s[sel], r["e"]),
        }
        res["factor_cos"][name] = {
            "pred": factor_cos(fac_hat, r["fac"], u),
            "shuffled": factor_cos(fac_s, r["fac"], u),
        }
        for k in ("direct", "projected", "factor_cos"):
            print(
                f"{k} {name}", json.dumps(res[k][name], ensure_ascii=False), flush=True
            )

    print(
        "\nrun       n    direct top1 / rank (chance) / layout rank (chance)   projected top1 / rank / layout rank   factor cos C/V/F (shuffled)"
    )
    for name, r in runs.items():
        n = len(r["syl"])
        a, p, fc = res["direct"][name], res["projected"][name], res["factor_cos"][name]
        print(
            f"  {name:<7}{n:>5}   {a['top1']:>4} / {a['rank_mean']:>6} ({(n + 1) / 2:.0f})"
            f" / {a['layout']['rank_mean']:>5} ({a['layout']['chance_mean']})"
            f"      {p['top1']:>4} / {p['rank_mean']:>6} / {p['layout']['rank_mean']:>5}"
            f"      {fc['pred']['C']:.3f} / {fc['pred']['V']:.3f} / {fc['pred']['F']:.3f}"
            f" ({fc['shuffled']['C']:.3f} / {fc['shuffled']['V']:.3f} / {fc['shuffled']['F']:.3f})",
            flush=True,
        )
    for k, v in res["ceiling"].items():
        print(
            f"  ceiling {k}: n {v['n']} top1 {v['top1']} rank {v['rank_mean']} layout rank "
            f"{v['layout']['rank_mean']} factor cos {v['factor_cos']}",
            flush=True,
        )

    DST.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "syllables": drawn,
            "e_hat": torch.from_numpy(pk).float(),
            "factors": {
                k: torch.from_numpy(np.ascontiguousarray(v)).float()
                for k, v in fac_hat.items()
            },
            "beta": beta,
            "features": KIND,
            "train": ["kata", "kanji"],
            "seed_rows": str(SEED_1008),
        },
        DST / f"{KIND}.pt",
    )
    rd = KO / "results" / f"{time.strftime('%Y%m%d-%H%M')}-hangul_init"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"→ {rd} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
