"""Anima tagger head — multi-label tags + 3-class rating + 8-class people-count, off frozen PE.

Each encoder side independently picks ``"mean"`` (consume a pre-pooled
``[B, D]`` feature) or ``"map"`` (consume ``[B, T, D]`` tokens via a learned
``MAPHead`` + optional CLS / mean concat). The two sides are configured
separately via ``pool_kind`` (main) and ``pool_kind_aux`` (aux), so e.g.
PE-Core can ride a cheap mean pool while PE-Spatial gets the full MAP
treatment for spatial detail.

Architecture (dual-encoder, ``pool_kind="map"`` + ``pool_kind_aux="map"``):

::

    main tokens [T_m, d_in]                         # PE-Core patch tokens, CLS at [0]
        ├─ MAPHead(K queries, H heads)  → [K, d_in]
        ├─ CLS  = tokens[:, 0]          → [1, d_in]
        └─ mean = tokens.mean(dim=1)    → [1, d_in]
              concat → [(K+use_cls+use_mean) * d_in]

    aux tokens  [T_a, d_in_aux]   (optional)        # PE-Spatial patch tokens
        ├─ MAPHead(K_a queries, H_a heads) → [K_a, d_in_aux]
        ├─ CLS  = tokens[:, 0]          → [1, d_in_aux]
        └─ mean = tokens.mean(dim=1)    → [1, d_in_aux]
              concat → [(K_a+use_cls_aux+use_mean_aux) * d_in_aux]

    [main_pool ‖ aux_pool] → LayerNorm + Linear(trunk_in_dim, d_hidden) + GELU + Dropout
    trunk_h [d_hidden]
        ├─→ Linear(d_hidden, n_tags)          → tag_logits
        ├─→ Linear(d_hidden, n_ratings)       → rating_logits
        └─→ Linear(d_hidden, n_people_counts) → people_logits  (omitted when n_people_counts == 0)

Mixed example (``pool_kind="mean"`` + ``pool_kind_aux="map"``): main side
contributes a single ``[d_in]`` channel (no MAPHead, no CLS / mean concat —
the cached feature *is* the mean pool), aux side gets the full MAP +
CLS / mean concat. ``trunk_in_dim`` becomes ``d_in + d_in_aux *
(K_a + use_cls_aux + use_mean_aux)``.

The aux encoder is **opt-in** via ``d_in_aux`` — when None the head is
single-encoder (PE-Core only) and the second forward arg must be omitted.
This preserves backward-compat with anima-tagger-v1 checkpoints whose
``config.json`` lacks the aux fields entirely. ``pool_kind_aux`` defaults
to ``None`` which inherits ``pool_kind`` — the dual-MAP path therefore
loads with no extra config keys.

The trunk is shared between heads so the auxiliary signals (rating /
people-count) nudge the same representation that's predicting tags.
``n_tags``/``n_ratings``/``n_people_counts``/``d_in``/``d_in_aux`` all come
from ``vocab.json`` + the cached PE token dimension (always ``d_enc``, not
the pooled feature dim — pool-output dim is derived from ``d_in`` × the
active pool channels).

Inference receives all heads in one forward; training computes per-head
losses and combines with ``λ_rating`` / ``λ_people``. ``n_people_counts=0``
in the config means "no people head was trained" — used to load legacy
checkpoints; ``forward`` returns ``None`` in that slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AnimaTaggerConfig:
    d_in: int                        # PE token dim (d_enc), not the pool-output dim.
    n_tags: int
    n_ratings: int = 3
    # 0 = no people head (legacy checkpoint). Trainer always sets this from
    # the manifest (currently len(PEOPLE_COUNT_LABELS) == 8) when in use.
    n_people_counts: int = 0
    d_hidden: int = 1024
    dropout: float = 0.1
    # Main encoder pool. ``"mean"`` (legacy) consumes a pre-pooled [B, d_in]
    # feature; ``"map"`` consumes a [B, T, d_in] token sequence and runs
    # MAPHead + (optional) CLS + (optional) mean inside the head. Default
    # is "mean" so legacy config.json files load unchanged.
    pool_kind: str = "mean"
    pool_n_queries: int = 4
    pool_n_heads: int = 8
    pool_use_cls: bool = True
    pool_use_mean: bool = True
    # Optional auxiliary encoder (e.g. PE-Spatial-B16-512). When d_in_aux
    # is None the head is single-encoder.
    d_in_aux: Optional[int] = None
    # Aux pool kind. ``None`` inherits ``pool_kind`` (so dual-MAP and dual-mean
    # configs don't need an extra key). Setting it explicitly lets the two
    # sides differ — e.g. main="mean" + aux="map" pays the MAP cost only on
    # the encoder where spatial detail matters.
    pool_kind_aux: Optional[str] = None
    pool_n_queries_aux: int = 4
    pool_n_heads_aux: int = 8
    pool_use_cls_aux: bool = True
    pool_use_mean_aux: bool = True

    @property
    def effective_pool_kind_aux(self) -> str:
        """Resolved aux pool kind. Mirrors ``pool_kind`` when unset."""
        return self.pool_kind_aux or self.pool_kind

    def _trunk_chans(
        self, d_in: int, kind: str, n_q: int, use_cls: bool, use_mean: bool,
    ) -> int:
        """One side's contribution to ``trunk_in_dim``."""
        if kind == "mean":
            return d_in
        if kind == "map":
            return d_in * (n_q + int(use_cls) + int(use_mean))
        raise ValueError(f"unknown pool_kind={kind!r}")

    @property
    def trunk_in_dim(self) -> int:
        """Width of the trunk's first Linear — main + (optional) aux contributions."""
        # Single-encoder mean: legacy single-vector trunk. (Stays at d_in
        # so anima-tagger-v1 checkpoints keep loading bit-identically.)
        if not self.has_aux and self.pool_kind == "mean":
            return self.d_in
        total = self._trunk_chans(
            self.d_in, self.pool_kind,
            self.pool_n_queries, self.pool_use_cls, self.pool_use_mean,
        )
        if self.has_aux:
            total += self._trunk_chans(
                self.d_in_aux, self.effective_pool_kind_aux,
                self.pool_n_queries_aux, self.pool_use_cls_aux, self.pool_use_mean_aux,
            )
        return total

    @property
    def has_aux(self) -> bool:
        return self.d_in_aux is not None

    def to_dict(self) -> dict:
        d = {
            "d_in": self.d_in,
            "n_tags": self.n_tags,
            "n_ratings": self.n_ratings,
            "n_people_counts": self.n_people_counts,
            "d_hidden": self.d_hidden,
            "dropout": self.dropout,
            "pool_kind": self.pool_kind,
            "pool_n_queries": self.pool_n_queries,
            "pool_n_heads": self.pool_n_heads,
            "pool_use_cls": self.pool_use_cls,
            "pool_use_mean": self.pool_use_mean,
        }
        # Only emit the aux block when configured. Keeps single-encoder
        # config.json files visually identical to the v1 layout.
        if self.d_in_aux is not None:
            d.update({
                "d_in_aux": self.d_in_aux,
                "pool_n_queries_aux": self.pool_n_queries_aux,
                "pool_n_heads_aux": self.pool_n_heads_aux,
                "pool_use_cls_aux": self.pool_use_cls_aux,
                "pool_use_mean_aux": self.pool_use_mean_aux,
            })
            # pool_kind_aux is only emitted when it differs from main —
            # absent value means "inherit", which keeps the dual-MAP-from-
            # default-pool_kind config.json identical to the prior version.
            if self.pool_kind_aux is not None and self.pool_kind_aux != self.pool_kind:
                d["pool_kind_aux"] = self.pool_kind_aux
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AnimaTaggerConfig":
        d_in_aux_raw = d.get("d_in_aux")
        pool_kind_aux_raw = d.get("pool_kind_aux")
        return cls(
            d_in=int(d["d_in"]),
            n_tags=int(d["n_tags"]),
            n_ratings=int(d.get("n_ratings", 3)),
            n_people_counts=int(d.get("n_people_counts", 0)),
            d_hidden=int(d.get("d_hidden", 1024)),
            dropout=float(d.get("dropout", 0.1)),
            pool_kind=str(d.get("pool_kind", "mean")),
            pool_n_queries=int(d.get("pool_n_queries", 4)),
            pool_n_heads=int(d.get("pool_n_heads", 8)),
            pool_use_cls=bool(d.get("pool_use_cls", True)),
            pool_use_mean=bool(d.get("pool_use_mean", True)),
            d_in_aux=int(d_in_aux_raw) if d_in_aux_raw is not None else None,
            pool_kind_aux=str(pool_kind_aux_raw) if pool_kind_aux_raw is not None else None,
            pool_n_queries_aux=int(d.get("pool_n_queries_aux", 4)),
            pool_n_heads_aux=int(d.get("pool_n_heads_aux", 8)),
            pool_use_cls_aux=bool(d.get("pool_use_cls_aux", True)),
            pool_use_mean_aux=bool(d.get("pool_use_mean_aux", True)),
        )


class MAPHead(nn.Module):
    """Multi-query attention pool — K learnable queries attend over the token grid.

    Shape: ``[B, T, D] → [B, K, D]``. Pre-norm on K/V (the queries are
    learnable parameters and don't need it). Uses :class:`nn.MultiheadAttention`
    with ``batch_first=True``; PyTorch routes through SDPA so this is a
    single fused kernel on CUDA.

    Initialization: queries drawn from N(0, 1/√D) so the dot-product scale
    matches the post-LayerNorm key/value scale and the initial attention
    map is roughly uniform (no early collapse onto a single token).
    """

    def __init__(self, d: int, n_queries: int = 4, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"MAPHead: d={d} must be divisible by n_heads={n_heads}")
        self.q = nn.Parameter(torch.randn(n_queries, d) * (d ** -0.5))
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, D]
        B = tokens.shape[0]
        q = self.q.unsqueeze(0).expand(B, -1, -1)        # [B, K, D]
        kv = self.norm_kv(tokens)                        # [B, T, D]
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out                                       # [B, K, D]


class AnimaTaggerHead(nn.Module):
    def __init__(self, cfg: AnimaTaggerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.pool_kind not in ("mean", "map"):
            raise ValueError(f"unknown pool_kind={cfg.pool_kind!r}")
        if cfg.has_aux and cfg.effective_pool_kind_aux not in ("mean", "map"):
            raise ValueError(
                f"unknown pool_kind_aux={cfg.effective_pool_kind_aux!r}"
            )
        # Per-side MAPHead — only instantiated when that side uses MAP pool.
        # Kept as None on mean-pool sides so the state_dict stays minimal
        # (no phantom buffers on legacy mean-pool checkpoints).
        self.pool: Optional[MAPHead] = (
            MAPHead(
                d=cfg.d_in,
                n_queries=cfg.pool_n_queries,
                n_heads=cfg.pool_n_heads,
                dropout=0.0,
            )
            if cfg.pool_kind == "map" else None
        )
        self.pool_aux: Optional[MAPHead] = (
            MAPHead(
                d=cfg.d_in_aux,
                n_queries=cfg.pool_n_queries_aux,
                n_heads=cfg.pool_n_heads_aux,
                dropout=0.0,
            )
            if cfg.has_aux and cfg.effective_pool_kind_aux == "map" else None
        )

        self.trunk = nn.Sequential(
            nn.LayerNorm(cfg.trunk_in_dim),
            nn.Linear(cfg.trunk_in_dim, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.tag_head = nn.Linear(cfg.d_hidden, cfg.n_tags)
        self.rating_head = nn.Linear(cfg.d_hidden, cfg.n_ratings)
        # Optional — older checkpoints have n_people_counts=0 and no people
        # head in the state_dict. Keeping the attribute as None lets `forward`
        # return a stable 3-tuple shape in both cases.
        self.people_head: Optional[nn.Linear] = (
            nn.Linear(cfg.d_hidden, cfg.n_people_counts)
            if cfg.n_people_counts > 0 else None
        )

    @staticmethod
    def _pool_one(
        tokens: torch.Tensor,
        pool: MAPHead,
        use_cls: bool,
        use_mean: bool,
    ) -> torch.Tensor:
        """[B, T, D] → [B, (K + use_cls + use_mean) * D] via MAP + (optional) CLS / mean concat."""
        chans = [pool(tokens).flatten(1)]                       # [B, K*D]
        if use_cls:
            chans.append(tokens[:, 0])                          # [B, D]
        if use_mean:
            chans.append(tokens.mean(dim=1))                    # [B, D]
        return torch.cat(chans, dim=-1)

    def _pool_side(
        self,
        feat: torch.Tensor,
        kind: str,
        pool: Optional[MAPHead],
        use_cls: bool,
        use_mean: bool,
        side_name: str,
    ) -> torch.Tensor:
        """Apply the right pooling for one side, returning [B, channels].

        Validates the input tensor rank against ``kind``: ``mean`` expects
        ``[B, D]`` (the cached feature is already the pool); ``map`` expects
        ``[B, T, D]`` (head's MAPHead pools internally).
        """
        if kind == "mean":
            if feat.dim() != 2:
                raise ValueError(
                    f"{side_name} side: pool_kind='mean' expects pre-pooled "
                    f"[B, D] but got rank {feat.dim()}"
                )
            return feat
        if kind == "map":
            if feat.dim() != 3:
                raise ValueError(
                    f"{side_name} side: pool_kind='map' expects [B, T, D] "
                    f"tokens but got rank {feat.dim()}"
                )
            assert pool is not None, f"{side_name} MAP path called without configured pool"
            return self._pool_one(feat, pool, use_cls, use_mean)
        raise ValueError(f"{side_name} side: unknown pool_kind={kind!r}")

    def forward(
        self,
        feat: torch.Tensor,
        feat_aux: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.cfg
        # Aux presence must match config.
        if cfg.has_aux and feat_aux is None:
            raise ValueError(
                "config has aux encoder (d_in_aux is set) but feat_aux was not "
                "provided to forward()"
            )
        if feat_aux is not None and not cfg.has_aux:
            raise ValueError(
                "feat_aux provided but config has no aux encoder (d_in_aux is None)"
            )

        main = self._pool_side(
            feat, cfg.pool_kind, self.pool,
            cfg.pool_use_cls, cfg.pool_use_mean, "main",
        )
        if cfg.has_aux:
            assert feat_aux is not None
            aux = self._pool_side(
                feat_aux, cfg.effective_pool_kind_aux, self.pool_aux,
                cfg.pool_use_cls_aux, cfg.pool_use_mean_aux, "aux",
            )
            x = torch.cat([main, aux], dim=-1)
        else:
            x = main

        h = self.trunk(x)
        people_logits = self.people_head(h) if self.people_head is not None else None
        return self.tag_head(h), self.rating_head(h), people_logits
