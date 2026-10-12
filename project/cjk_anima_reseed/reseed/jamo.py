"""jamo — Hangul syllables as (initial, vowel, final) and the factor rows
over them (``proposal_jamo.md`` § 2).

A syllable is ``0xAC00 + (cho · 21 + jung) · 28 + jong`` (19 × 21 × 28).
``cls`` is the vowel's class — vertical (the initial on the left), horizontal
(on top), compound (both) — and with or without a final it sets the block's
layout (6 of them). ``KSX1001`` is the 2 350 syllables the KO faces in
``assets/fonts/kozh/`` cover (a two-byte EUC-KR code; Python's codec spells
every other syllable as an 8-byte jamo sequence).

``Jamo`` composes, in delta units over the pack row,

    Δ(s) = b + C[cho, cls(jung)] + V[jung] + F[jong]       (F[0] = 0)

into ``Rows.delta.raw`` before every forward, so the optimizer holds the
106 factor vectors and the gradient reaches them through the composition.

A partial form (cjk_ko, user 10-12) is a syllable with its initial left out:
the vowel alone (``PART_JUNG``, the conjoining medial ᅡ … ᅵ) or the final
alone (``PART_JONG``, ᆨ … ᇂ), drawn where it sits in a syllable (``PART_SPELL``:
behind the fillers, so the font places it as in 어 / 언). Its row is the
syllable's sum without the absent parts — ᅥ = b + V[ㅓ], ᆫ = b + F[ㄴ] — so V
and F each have to draw their letter alone, in place.
"""

from __future__ import annotations

import torch

BASE, N_CHO, N_JUNG, N_JONG = 0xAC00, 19, 21, 28
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
# the compatibility jamo (U+3131–U+3163): the standalone glyphs, one ext row
# each — every letter CHO / JUNG / JONG names
COMPAT = "".join(map(chr, range(0x3131, 0x3164)))
CLASSES = ("vertical", "horizontal", "compound")
_CLS = {**dict.fromkeys("ㅏㅐㅑㅒㅓㅔㅕㅖㅣ", 0), **dict.fromkeys("ㅗㅛㅜㅠㅡ", 1)}
JUNG_CLS = tuple(_CLS.get(v, 2) for v in JUNG)
LAYOUTS = ("V", "H", "C", "VF", "HF", "CF")
# the partial forms: one conjoining jamo each, its own ext row; drawn behind
# the choseong (and, for a final, jungseong) filler
CHO_FILL, JUNG_FILL = "\u115f", "\u1160"
PART_JUNG = "".join(map(chr, range(0x1161, 0x1161 + N_JUNG)))
PART_JONG = "".join(map(chr, range(0x11A8, 0x11A8 + N_JONG - 1)))
PART_SPELL = {
    **{v: CHO_FILL + v for v in PART_JUNG},
    **{f: CHO_FILL + JUNG_FILL + f for f in PART_JONG},
}


def is_syllable(ch: str) -> bool:
    return len(ch) == 1 and BASE <= ord(ch) < BASE + N_CHO * N_JUNG * N_JONG


def is_part(ch: str) -> bool:
    return len(ch) == 1 and ch in PART_SPELL


def decompose(ch: str) -> tuple[int, int, int]:
    """``(cho, jung, jong)`` indices; ``jong`` 0 = no final."""
    assert is_syllable(ch), ch
    k = ord(ch) - BASE
    return k // (N_JUNG * N_JONG), k // N_JONG % N_JUNG, k % N_JONG


def compose(cho: int, jung: int, jong: int) -> str:
    return chr(BASE + (cho * N_JUNG + jung) * N_JONG + jong)


def layout(ch: str) -> str:
    _, jung, jong = decompose(ch)
    return "VHC"[JUNG_CLS[jung]] + ("F" if jong else "")


def cells(ch: str) -> frozenset:
    """What a trained ``ch`` gives the factor model: its ``C`` cell (initial ×
    vowel class), its vowel, its final (0 included) and its layout. 57 + 21 +
    28 + 6 = 112 cells."""
    cho, jung, jong = decompose(ch)
    return frozenset(
        (("C", cho, JUNG_CLS[jung]), ("V", jung), ("F", jong), ("L", layout(ch)))
    )


def in_ksx1001(ch: str) -> bool:
    return is_syllable(ch) and len(ch.encode("euc_kr")) == 2


KSX1001 = tuple(
    c for c in map(chr, range(BASE, BASE + N_CHO * N_JUNG * N_JONG)) if in_ksx1001(c)
)
ALL = tuple(map(chr, range(BASE, BASE + N_CHO * N_JUNG * N_JONG)))


def syllable_rows() -> dict:
    """{syllable: its ext row} for all 11 172 on the run's pack: one row each
    (2 512 single Qwen tokens, the rest byte-split onto ``char`` rows)."""
    from data.inventory import pieces, qwen_pieces

    tok, q = qwen_pieces(char_rows=True)
    out = {}
    for s in ALL:
        got = pieces(tok, q, s)
        assert len(got) == 1 and got[0][1] is not None, (s, got)
        out[s] = int(got[0][1])
    assert len(set(out.values())) == len(out), "two syllables share a row"
    return out


def part_rows() -> dict:
    """{partial form: its ext row} on the run's pack."""
    from data.inventory import pieces, qwen_pieces

    tok, q = qwen_pieces(char_rows=True)
    out = {}
    for j in PART_SPELL:
        got = pieces(tok, q, j)
        assert len(got) == 1 and got[0][1] is not None, (j, got)
        out[j] = int(got[0][1])
    return out


def compat_rows() -> dict:
    """{compatibility jamo: its ext row} on the run's pack."""
    from data.inventory import pieces, qwen_pieces

    tok, q = qwen_pieces(char_rows=True)
    out = {}
    for j in COMPAT:
        got = pieces(tok, q, j)
        assert len(got) == 1 and got[0][1] is not None, (j, got)
        out[j] = int(got[0][1])
    return out


def codes(syllables) -> torch.Tensor:
    """``(n, 4)`` long: cho, cls, jung, jong per syllable; a partial form's
    absent initial is ``N_CHO`` (cls 0), its absent vowel ``N_JUNG``."""
    out = []
    for s in syllables:
        if s in PART_JUNG:
            j = PART_JUNG.index(s)
            out.append((N_CHO, 0, j, 0))
            continue
        if s in PART_JONG:
            out.append((N_CHO, 0, N_JUNG, PART_JONG.index(s) + 1))
            continue
        cho, jung, jong = decompose(s)
        out.append((cho, JUNG_CLS[jung], jung, jong))
    return torch.tensor(out, dtype=torch.long).reshape(-1, 4)


class Jamo:
    """The factor rows: ``b`` (1, d), ``C`` (19, 3, d), ``V`` (21, d), ``F``
    (27, d) — the finals past 0, all zero at init (Δ 0: the cold start).

    ``rows`` is the run's ``Rows``; ``syl`` = {ext id: syllable} for its
    trained syllable rows (and partial forms, ``PART_SPELL``). ``apply`` writes the composition into
    ``rows.delta.raw`` (out of place: the free ``raw`` keeps the frozen
    context rows, ``ExtDelta`` reads ``raw`` at forward time); call it before
    each step's forward and before a save.

    A trained row that is no syllable (cjk_ko J1024, 10-11: the 51 compat
    jamo, ㅋㅋ / ㅠㅠ's own glyphs) stays a free row of ``raw``, in the
    optimizer beside the factors; the composition never writes it.

    ``residual`` (cjk_ko B′, 10-11): a per-syllable free row ``r`` on top of
    the composition, 0 at init, in the optimizer beside the factors;
    ``penalty`` = ``pull`` · mean ‖r‖² (the norm pull on ``r`` only, so the
    factors carry what they can)."""

    N_VECTORS = 1 + N_CHO * len(CLASSES) + N_JUNG + (N_JONG - 1)

    def __init__(
        self, rows, syl: dict, lr: float, residual: bool = False, pull: float = 0.0
    ):
        d = rows.delta.raw.shape[1]
        dev = rows.delta.raw.device
        ids = rows.delta.ext_ids
        trained = [i for i, m in enumerate(rows.frozen_mask.tolist()) if not m]
        live = [i for i in trained if ids[i] in syl]
        own = [i for i in trained if ids[i] not in syl]
        assert {ids[i] for i in live} == set(syl), "factor rows: every syllable trained"
        self.rows = rows
        self.n_own = len(own)
        # the frozen context rows: held; the own rows (no syllable) train in place
        self.free = rows.delta.raw.requires_grad_(bool(own))
        if own:
            keep = torch.zeros(len(ids), 1, device=dev)
            keep[own] = 1.0
            self.free.register_hook(lambda g: g * keep)
        self.loc = torch.tensor(live, dtype=torch.long, device=dev)
        self.code = codes([syl[ids[i]] for i in live]).to(dev)

        def p(*shape):
            return torch.nn.Parameter(torch.zeros(*shape, d, device=dev))

        self.b, self.C, self.V, self.F = (
            p(1),
            p(N_CHO, len(CLASSES)),
            p(N_JUNG),
            p(N_JONG - 1),
        )
        self.r = p(len(live)) if residual else None
        self.pull = float(pull)
        ps = [self.b, self.C, self.V, self.F] + ([self.r] if residual else [])
        ps += [self.free] if own else []
        rows.params = [{"params": ps, "lr": lr}]

    @torch.no_grad()
    def init_state(self, state: dict, row_scale: float) -> dict:
        """The factors at a factor run's (``trained.pt`` ``jamo``: b / C / V /
        F, raw units at that run's ``row_scale``), rescaled into this run's;
        ``r`` stays 0 (cjk_ko, 10-11). Returns the norms for the record."""
        k = float(row_scale) / float(self.rows.row_scale)
        for n in ("b", "C", "V", "F"):
            getattr(self, n).copy_(state[n].float().to(self.b.device) * k)
        return {
            "row_scale_ratio": round(k, 4),
            "init_delta_norm": round(
                float(self.compose(self.code).norm(dim=1).mean()) * self.rows.row_scale,
                1,
            ),
        }

    def penalty(self) -> torch.Tensor | float:
        """``pull`` · mean ‖r‖² (0 without a residual or a pull)."""
        if self.r is None or not self.pull:
            return 0.0
        return self.pull * (self.r**2).sum(1).mean()

    @torch.no_grad()
    def init_from(self, lone: dict, scale: float) -> dict:
        """The jamo curriculum's stage 2 (user 10-11): ``lone`` = {compat
        jamo: its full row (pack + Δ, absolute units)} from a run that
        trained them alone; each is centred over the 51 and ``C[cho, every
        cls]`` / ``V[jung]`` / ``F[jong]`` start at scale × its letter's,
        ``b`` at 0. Returns the norms for the record."""
        assert set(lone) == set(COMPAT), sorted(set(COMPAT) - set(lone))[:10]
        full = torch.stack([lone[j] for j in COMPAT]).float().to(self.b.device)
        e = dict(zip(COMPAT, (full - full.mean(0)) * scale / self.rows.row_scale))
        for i, j in enumerate(CHO):
            self.C[i] = e[j]
        for i, j in enumerate(JUNG):
            self.V[i] = e[j]
        for i, j in enumerate(JONG[1:]):
            self.F[i] = e[j]
        rs = self.rows.row_scale
        return {
            "lone_norm": round(float(full.norm(dim=1).mean()), 1),
            "centred_norm": round(float((full - full.mean(0)).norm(dim=1).mean()), 1),
            "init_delta_norm": round(
                float(self.compose(self.code).norm(dim=1).mean()) * rs, 1
            ),
        }

    def compose(self, code: torch.Tensor) -> torch.Tensor:
        """Δ (raw units) for ``code`` (``codes``' rows); an absent part adds 0."""
        F = torch.cat([torch.zeros_like(self.F[:1]), self.F])
        C = torch.cat([self.C, torch.zeros_like(self.C[:1])])
        V = torch.cat([self.V, torch.zeros_like(self.V[:1])])
        return self.b + C[code[:, 0], code[:, 1]] + V[code[:, 2]] + F[code[:, 3]]

    def apply(self) -> None:
        live = self.compose(self.code)
        if self.r is not None:
            live = live + self.r
        self.rows.delta.raw = self.free.index_put((self.loc,), live)

    def state(self) -> dict:
        out = {
            "model": "b + C[cho, cls] + V[jung] + F[jong]",
            "classes": list(CLASSES),
            **{k: getattr(self, k).detach().cpu() for k in ("b", "C", "V", "F")},
        }
        if self.r is not None:
            ids = self.rows.delta.ext_ids
            out["r"] = self.r.detach().cpu()
            out["r_ext_ids"] = [int(ids[i]) for i in self.loc.tolist()]
            out["r_pull"] = self.pull
        return out

    @torch.no_grad()
    def merge_all(self, sd: dict, ext_of: dict) -> int:
        """Every syllable's composed row into ``sd['delta']`` (``Rows
        .state_dict``'s), in place of whatever it holds for that id — the
        held-out syllables read zero-shot from the file as written.
        ``ext_of`` = {syllable: ext id}. Returns how many rows were added.
        With a residual, a trained row keeps its composition + ``r`` (what
        ``apply`` wrote)."""
        delta = sd["delta"]
        have = {int(e): i for i, e in enumerate(delta["ext_ids"])}
        ids0 = self.rows.delta.ext_ids
        keep = (
            {int(ids0[i]) for i in self.loc.tolist()} if self.r is not None else set()
        )
        syls = sorted(ext_of)
        comp = self.compose(codes(syls).to(self.b.device)).cpu().to(delta["raw"].dtype)
        raw = delta["raw"].clone()
        new_ids, new_rows = [], []
        for s, r in zip(syls, comp):
            e = int(ext_of[s])
            if e in keep:
                continue
            if e in have:
                raw[have[e]] = r
            else:
                new_ids.append(e)
                new_rows.append(r)
        ids = [int(e) for e in delta["ext_ids"]] + new_ids
        raw = torch.cat([raw, torch.stack(new_rows)]) if new_rows else raw
        order = sorted(range(len(ids)), key=ids.__getitem__)
        sd["delta"] = {
            **delta,
            "ext_ids": [ids[j] for j in order],
            "raw": raw[order].clone(),
        }
        return len(new_ids)
