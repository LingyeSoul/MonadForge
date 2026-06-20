"""Anima tagger head — multi-label tags + 3-class rating + 8-class people-count, off frozen PE.

Single architecture: **dual-encoder, hard-routed**. Two vision encoders feed two
parallel projection trunks, and each output head reads exactly one trunk — no
learned gating, no concat-trunk, no single-encoder fallback:

* **PE-Core** (``d_in``) → ``trunk_core`` → rating head, people-count head, and
  the *identity / global* tag sub-head (``tag_head_core`` over
  ``tag_indices_core`` = character / copyright / artist / count).
* **PE-Spatial** (``d_in_aux``) → ``trunk_spatial`` → the *localized* tag sub-head
  (``tag_head_spatial`` over ``tag_indices_spatial`` = everything else).

The two tag sub-heads are scattered back into a full ``[B, n_tags]`` tensor in
vocab order so the downstream loss / threshold paths see one flat tag logit
vector. ``tag_indices_core`` and ``tag_indices_spatial`` MUST partition
``[0, n_tags)``.

Each encoder side independently picks ``"mean"`` (consume a pre-pooled ``[B, D]``
feature) or ``"map"`` (consume ``[B, T, D]`` tokens via a learned ``MAPHead`` +
optional CLS / mean concat), via ``pool_kind`` (core) and ``pool_kind_aux``
(spatial). The common production setup is PE-Core ``mean`` + PE-Spatial ``map``
— a cheap global pool for identity signals and the full spatial pool where
localized detail matters.

Architecture (``pool_kind_aux="map"``)::

    core tokens   [T_c, d_in]                      # PE-Core (or pre-pooled [d_in])
        → pool_core   → [core_trunk_in_dim]
        → trunk_core  → h_core [d_hidden]
            ├─→ Linear(d_hidden, n_ratings)        → rating_logits
            ├─→ Linear(d_hidden, n_people_counts)  → people_logits  (None when 0)
            └─→ Linear(d_hidden, |core|)           → scatter → tag_logits[core idx]

    spatial tokens [T_s, d_in_aux]                 # PE-Spatial patch tokens
        → pool_spatial → [spatial_trunk_in_dim]
        → trunk_spatial→ h_spatial [d_hidden]
            └─→ Linear(d_hidden, |spatial|)        → scatter → tag_logits[spatial idx]

``n_tags`` / ``n_ratings`` / ``n_people_counts`` / ``d_in`` / ``d_in_aux`` come
from ``vocab.json`` + the cached PE token dims (always ``d_enc``, not the pooled
feature dim). ``n_people_counts=0`` means "no people head" and ``forward``
returns ``None`` in that slot. Inference receives all heads in one forward;
training computes per-head losses and combines with ``λ_rating`` / ``λ_people``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AnimaTaggerConfig:
    d_in: int  # PE-Core token dim (d_enc), not the pool-output dim.
    n_tags: int
    d_in_aux: int  # PE-Spatial token dim — always required.
    n_ratings: int = 3
    # 0 = no people head; trainer sets this from the manifest when in use.
    n_people_counts: int = 0
    d_hidden: int = 1024
    dropout: float = 0.1
    # "map" consumes [B, T, d_in] tokens (MAPHead + optional CLS/mean concat); "mean" consumes a pre-pooled [B, d_in] feature.
    pool_kind: str = "map"
    pool_n_queries: int = 4
    pool_n_heads: int = 8
    pool_use_cls: bool = True
    pool_use_mean: bool = True
    pool_kind_aux: str = "map"
    pool_n_queries_aux: int = 4
    pool_n_heads_aux: int = 8
    pool_use_cls_aux: bool = True
    pool_use_mean_aux: bool = True
    # Hard routing partition (must partition [0, n_tags)): PE-Core reads identity/global (character / copyright / artist / count), PE-Spatial reads the localized complement.
    tag_indices_core: List[int] = field(default_factory=list)
    tag_indices_spatial: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        combined = sorted(list(self.tag_indices_core) + list(self.tag_indices_spatial))
        if combined != list(range(self.n_tags)):
            raise ValueError(
                f"tag_indices_core ∪ tag_indices_spatial must partition "
                f"[0, n_tags={self.n_tags}); got "
                f"{len(self.tag_indices_core)} core + "
                f"{len(self.tag_indices_spatial)} spatial = {len(combined)} "
                f"total (with duplicates or gaps)."
            )

    @staticmethod
    def _trunk_chans(
        d_in: int,
        kind: str,
        n_q: int,
        use_cls: bool,
        use_mean: bool,
    ) -> int:
        """One side's contribution to its trunk's input width."""
        if kind == "mean":
            return d_in
        if kind == "map":
            return d_in * (n_q + int(use_cls) + int(use_mean))
        raise ValueError(f"unknown pool_kind={kind!r}")

    @property
    def core_trunk_in_dim(self) -> int:
        """Width of the PE-Core trunk's first Linear."""
        return self._trunk_chans(
            self.d_in,
            self.pool_kind,
            self.pool_n_queries,
            self.pool_use_cls,
            self.pool_use_mean,
        )

    @property
    def spatial_trunk_in_dim(self) -> int:
        """Width of the PE-Spatial trunk's first Linear."""
        return self._trunk_chans(
            self.d_in_aux,
            self.pool_kind_aux,
            self.pool_n_queries_aux,
            self.pool_use_cls_aux,
            self.pool_use_mean_aux,
        )

    def to_dict(self) -> dict:
        return {
            "d_in": self.d_in,
            "n_tags": self.n_tags,
            "d_in_aux": self.d_in_aux,
            "n_ratings": self.n_ratings,
            "n_people_counts": self.n_people_counts,
            "d_hidden": self.d_hidden,
            "dropout": self.dropout,
            "pool_kind": self.pool_kind,
            "pool_n_queries": self.pool_n_queries,
            "pool_n_heads": self.pool_n_heads,
            "pool_use_cls": self.pool_use_cls,
            "pool_use_mean": self.pool_use_mean,
            "pool_kind_aux": self.pool_kind_aux,
            "pool_n_queries_aux": self.pool_n_queries_aux,
            "pool_n_heads_aux": self.pool_n_heads_aux,
            "pool_use_cls_aux": self.pool_use_cls_aux,
            "pool_use_mean_aux": self.pool_use_mean_aux,
            "tag_indices_core": list(self.tag_indices_core),
            "tag_indices_spatial": list(self.tag_indices_spatial),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnimaTaggerConfig":
        if "d_in_aux" not in d or d["d_in_aux"] is None:
            raise ValueError(
                "AnimaTaggerConfig.from_dict requires 'd_in_aux' (dual encoder "
                "is mandatory). Pre-dual / v1 single-encoder checkpoints no "
                "longer load."
            )
        if "tag_indices_core" not in d or "tag_indices_spatial" not in d:
            raise ValueError(
                "AnimaTaggerConfig.from_dict requires 'tag_indices_core' and "
                "'tag_indices_spatial' (the hard-routing partition)."
            )
        return cls(
            d_in=int(d["d_in"]),
            n_tags=int(d["n_tags"]),
            d_in_aux=int(d["d_in_aux"]),
            n_ratings=int(d.get("n_ratings", 3)),
            n_people_counts=int(d.get("n_people_counts", 0)),
            d_hidden=int(d.get("d_hidden", 1024)),
            dropout=float(d.get("dropout", 0.1)),
            pool_kind=str(d.get("pool_kind", "map")),
            pool_n_queries=int(d.get("pool_n_queries", 4)),
            pool_n_heads=int(d.get("pool_n_heads", 8)),
            pool_use_cls=bool(d.get("pool_use_cls", True)),
            pool_use_mean=bool(d.get("pool_use_mean", True)),
            pool_kind_aux=str(d.get("pool_kind_aux", "map")),
            pool_n_queries_aux=int(d.get("pool_n_queries_aux", 4)),
            pool_n_heads_aux=int(d.get("pool_n_heads_aux", 8)),
            pool_use_cls_aux=bool(d.get("pool_use_cls_aux", True)),
            pool_use_mean_aux=bool(d.get("pool_use_mean_aux", True)),
            tag_indices_core=[int(i) for i in d["tag_indices_core"]],
            tag_indices_spatial=[int(i) for i in d["tag_indices_spatial"]],
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

    def __init__(
        self, d: int, n_queries: int = 4, n_heads: int = 8, dropout: float = 0.0
    ):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"MAPHead: d={d} must be divisible by n_heads={n_heads}")
        self.q = nn.Parameter(torch.randn(n_queries, d) * (d**-0.5))
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        kv = self.norm_kv(tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out


class AnimaTaggerHead(nn.Module):
    """Dual-encoder, hard-routed tagger head.

    Two parallel trunks (``trunk_core`` off PE-Core, ``trunk_spatial`` off
    PE-Spatial). Rating / people-count / identity-tag heads read ``h_core``;
    the localized-tag head reads ``h_spatial``. The two tag sub-heads scatter
    into one ``[B, n_tags]`` logit tensor via the routing index buffers.
    """

    def __init__(self, cfg: AnimaTaggerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.pool_kind not in ("mean", "map"):
            raise ValueError(f"unknown pool_kind={cfg.pool_kind!r}")
        if cfg.pool_kind_aux not in ("mean", "map"):
            raise ValueError(f"unknown pool_kind_aux={cfg.pool_kind_aux!r}")

        # Per-side MAPHead, None on mean-pool sides so the state_dict stays minimal.
        self.pool_core: Optional[MAPHead] = (
            MAPHead(
                d=cfg.d_in,
                n_queries=cfg.pool_n_queries,
                n_heads=cfg.pool_n_heads,
                dropout=0.0,
            )
            if cfg.pool_kind == "map"
            else None
        )
        self.pool_spatial: Optional[MAPHead] = (
            MAPHead(
                d=cfg.d_in_aux,
                n_queries=cfg.pool_n_queries_aux,
                n_heads=cfg.pool_n_heads_aux,
                dropout=0.1,
            )
            if cfg.pool_kind_aux == "map"
            else None
        )

        # Two parallel projection trunks — no shared concat trunk, no gating.
        self.trunk_core = nn.Sequential(
            nn.LayerNorm(cfg.core_trunk_in_dim),
            nn.Linear(cfg.core_trunk_in_dim, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.trunk_spatial = nn.Sequential(
            nn.LayerNorm(cfg.spatial_trunk_in_dim),
            nn.Linear(cfg.spatial_trunk_in_dim, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # Tag sub-heads over the disjoint vocab partition, guarded so a degenerate all-one-side partition still builds.
        n_core = len(cfg.tag_indices_core)
        n_spatial = len(cfg.tag_indices_spatial)
        self.tag_head_core = nn.Linear(cfg.d_hidden, n_core) if n_core > 0 else None
        self.tag_head_spatial = (
            nn.Linear(cfg.d_hidden, n_spatial) if n_spatial > 0 else None
        )
        self.rating_head = nn.Linear(cfg.d_hidden, cfg.n_ratings)
        self.people_head: Optional[nn.Linear] = (
            nn.Linear(cfg.d_hidden, cfg.n_people_counts)
            if cfg.n_people_counts > 0
            else None
        )

        # Routing index buffers — ride device/dtype moves and round-trip in state_dict.
        self.register_buffer(
            "tag_idx_core",
            torch.tensor(cfg.tag_indices_core, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "tag_idx_spatial",
            torch.tensor(cfg.tag_indices_spatial, dtype=torch.long),
            persistent=True,
        )

    @staticmethod
    def _pool_one(
        tokens: torch.Tensor,
        pool: MAPHead,
        use_cls: bool,
        use_mean: bool,
    ) -> torch.Tensor:
        """[B, T, D] → [B, (K + use_cls + use_mean) * D] via MAP + (optional) CLS / mean concat."""
        chans = [pool(tokens).flatten(1)]
        if use_cls:
            chans.append(tokens[:, 0])
        if use_mean:
            chans.append(tokens.mean(dim=1))
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

        ``mean`` expects ``[B, D]`` (the cached feature is already the pool);
        ``map`` expects ``[B, T, D]`` (head's MAPHead pools internally).
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
            assert pool is not None, (
                f"{side_name} MAP path called without configured pool"
            )
            return self._pool_one(feat, pool, use_cls, use_mean)
        raise ValueError(f"{side_name} side: unknown pool_kind={kind!r}")

    def forward(
        self,
        feat_core: torch.Tensor,
        feat_spatial: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.cfg
        core = self._pool_side(
            feat_core,
            cfg.pool_kind,
            self.pool_core,
            cfg.pool_use_cls,
            cfg.pool_use_mean,
            "core",
        )
        spatial = self._pool_side(
            feat_spatial,
            cfg.pool_kind_aux,
            self.pool_spatial,
            cfg.pool_use_cls_aux,
            cfg.pool_use_mean_aux,
            "spatial",
        )
        h_core = self.trunk_core(core)
        h_spatial = self.trunk_spatial(spatial)

        B = h_core.shape[0]
        tag_logits = h_core.new_zeros((B, cfg.n_tags))
        if self.tag_head_core is not None:
            tag_logits.index_copy_(1, self.tag_idx_core, self.tag_head_core(h_core))
        if self.tag_head_spatial is not None:
            tag_logits.index_copy_(
                1, self.tag_idx_spatial, self.tag_head_spatial(h_spatial)
            )

        rating_logits = self.rating_head(h_core)
        people_logits = (
            self.people_head(h_core) if self.people_head is not None else None
        )
        return tag_logits, rating_logits, people_logits
