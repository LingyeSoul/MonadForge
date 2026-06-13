"""Smoke tests for the Anima Tagger dual-encoder, hard-routed head.

Doesn't touch real PE checkpoints — exercises only the config / head /
encoder-registry / bucket-spec wiring with synthetic tensors. The
encoder-loader path (auto-fetch + checkpoint load) is left to manual
verification since it costs disk + network.
"""

from __future__ import annotations

import torch

from library.captioning.anima_tagger_model import (
    AnimaTaggerConfig,
    AnimaTaggerHead,
)


def _routing(n_tags: int, n_core: int | None = None):
    """A valid (core, spatial) partition of ``[0, n_tags)`` for tests."""
    n_core = n_tags // 2 if n_core is None else n_core
    return list(range(n_core)), list(range(n_core, n_tags))


def _cfg(n_tags: int = 50, n_core: int | None = None, **kw) -> AnimaTaggerConfig:
    core, spatial = _routing(n_tags, n_core)
    base = dict(
        d_in=1024,
        n_tags=n_tags,
        d_in_aux=768,
        tag_indices_core=core,
        tag_indices_spatial=spatial,
    )
    base.update(kw)
    return AnimaTaggerConfig(**base)


# ── Config validation ─────────────────────────────────────────────────────


def test_config_requires_d_in_aux():
    """Dual encoder is mandatory — from_dict rejects a missing d_in_aux."""
    try:
        AnimaTaggerConfig.from_dict(
            {
                "d_in": 1024,
                "n_tags": 4,
                "tag_indices_core": [0, 1],
                "tag_indices_spatial": [2, 3],
            }
        )
    except ValueError as e:
        assert "d_in_aux" in str(e)
    else:
        raise AssertionError("expected ValueError for missing d_in_aux")


def test_config_requires_routing_partition():
    """tag_indices_core ∪ tag_indices_spatial must partition [0, n_tags)."""
    # Gap: index 2 missing.
    try:
        AnimaTaggerConfig(
            d_in=1024,
            n_tags=4,
            d_in_aux=768,
            tag_indices_core=[0, 1],
            tag_indices_spatial=[3],
        )
    except ValueError as e:
        assert "partition" in str(e)
    else:
        raise AssertionError("expected ValueError for malformed partition")
    # Duplicate: index 1 in both.
    try:
        AnimaTaggerConfig(
            d_in=1024,
            n_tags=4,
            d_in_aux=768,
            tag_indices_core=[0, 1],
            tag_indices_spatial=[1, 2, 3],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for duplicate index")


def test_config_roundtrip_flat():
    """to_dict / from_dict round-trip emits every field flat and preserves it."""
    cfg = _cfg(n_tags=100, n_people_counts=8, pool_kind="mean", pool_kind_aux="map")
    d = cfg.to_dict()
    # Flat — both pool kinds + the routing partition are always present.
    assert d["pool_kind"] == "mean" and d["pool_kind_aux"] == "map"
    assert d["tag_indices_core"] == list(range(50))
    assert d["tag_indices_spatial"] == list(range(50, 100))
    cfg2 = AnimaTaggerConfig.from_dict(d)
    assert cfg2.d_in_aux == 768
    assert cfg2.core_trunk_in_dim == cfg.core_trunk_in_dim
    assert cfg2.spatial_trunk_in_dim == cfg.spatial_trunk_in_dim


def test_trunk_widths():
    """Per-side trunk widths follow each side's pool channels independently."""
    cfg = _cfg(pool_kind="mean", pool_kind_aux="map")
    # core mean → d_in; spatial map → 768 * (4 queries + cls + mean) = 4608.
    assert cfg.core_trunk_in_dim == 1024
    assert cfg.spatial_trunk_in_dim == 768 * (4 + 1 + 1)


# ── Forward ───────────────────────────────────────────────────────────────


def test_hard_routed_forward_shapes():
    cfg = _cfg(n_tags=50, n_people_counts=8, pool_kind="map", pool_kind_aux="map")
    head = AnimaTaggerHead(cfg)
    tag, rate, people = head(
        torch.randn(2, 577, 1024),  # PE-Core tokens
        torch.randn(2, 1025, 768),  # PE-Spatial tokens
    )
    assert tag.shape == (2, 50)
    assert rate.shape == (2, 3)
    assert people.shape == (2, 8)


def test_core_mean_spatial_map_forward_shapes():
    """Production target: PE-Core mean + PE-Spatial map."""
    cfg = _cfg(n_tags=50, n_people_counts=8, pool_kind="mean", pool_kind_aux="map")
    head = AnimaTaggerHead(cfg)
    tag, rate, people = head(
        torch.randn(2, 1024),  # core: pre-pooled [B, D]
        torch.randn(2, 1025, 768),  # spatial: [B, T_a, D_a]
    )
    assert tag.shape == (2, 50)
    assert people.shape == (2, 8)


def test_routing_is_structural():
    """Spatial tag logits come ONLY from the spatial head; zeroing it zeros
    exactly the spatial-routed slice and leaves the core slice intact."""
    cfg = _cfg(n_tags=20, n_core=12, pool_kind="map", pool_kind_aux="map")
    head = AnimaTaggerHead(cfg).eval()
    with torch.no_grad():
        head.tag_head_spatial.weight.zero_()
        head.tag_head_spatial.bias.zero_()
        tag, _, _ = head(torch.randn(2, 577, 1024), torch.randn(2, 1025, 768))
    spatial_idx = cfg.tag_indices_spatial
    core_idx = cfg.tag_indices_core
    assert torch.all(tag[:, spatial_idx] == 0), "spatial slice must be all zero"
    assert not torch.all(tag[:, core_idx] == 0), (
        "core slice must be driven by core head"
    )


def test_forward_requires_both_inputs():
    cfg = _cfg(n_tags=10, pool_kind="map", pool_kind_aux="map")
    head = AnimaTaggerHead(cfg)
    try:
        head(torch.randn(1, 577, 1024))  # missing feat_spatial
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for missing feat_spatial")


def test_core_mean_rejects_map_input():
    """Helpful error when caller passes [B, T, D] to a mean-pool core side."""
    cfg = _cfg(n_tags=10, pool_kind="mean", pool_kind_aux="map")
    head = AnimaTaggerHead(cfg)
    try:
        head(torch.randn(2, 577, 1024), torch.randn(2, 1025, 768))
    except ValueError as e:
        assert "core side" in str(e) and "pre-pooled" in str(e)
    else:
        raise AssertionError("expected ValueError on rank-3 core input with mean pool")


# ── State dict layout ─────────────────────────────────────────────────────


def test_state_dict_layout():
    """Dual hard-routed head has the new keys and none of the legacy ones."""
    cfg = _cfg(n_tags=10, pool_kind="map", pool_kind_aux="map")
    sd = AnimaTaggerHead(cfg).state_dict()
    keys = list(sd.keys())
    for prefix in (
        "trunk_core.",
        "trunk_spatial.",
        "tag_head_core.",
        "tag_head_spatial.",
        "pool_core.",
        "pool_spatial.",
    ):
        assert any(k.startswith(prefix) for k in keys), f"missing {prefix}* keys"
    assert "tag_idx_core" in keys and "tag_idx_spatial" in keys
    # Legacy artifacts must be gone.
    assert not any(k.startswith("trunk.") for k in keys), "legacy concat trunk"
    assert not any(k.startswith("tag_head.") for k in keys), "legacy single tag_head"
    assert not any("gate" in k for k in keys), "legacy soft-gate params"
    assert not any(k.startswith("pool.") or k.startswith("pool_aux.") for k in keys)


def test_state_dict_core_mean_has_no_core_pool():
    """When core is mean-pool, no pool_core MAPHead is built."""
    cfg = _cfg(n_tags=10, pool_kind="mean", pool_kind_aux="map")
    sd = AnimaTaggerHead(cfg).state_dict()
    assert not any(k.startswith("pool_core.") for k in sd.keys()), (
        "core side is mean — no core MAPHead expected"
    )
    assert any(k.startswith("pool_spatial.") for k in sd.keys()), (
        "spatial side is map — pool_spatial MAPHead expected"
    )


# ── Encoder registry / bucket spec (unchanged invariants) ─────────────────


def test_pe_spatial_config_present_in_registry():
    """PE-Spatial-B16-512 must be in PE_CONFIGS and produce 1025 tokens at 512px."""
    from library.models.pe import PE_CONFIGS, build_pe_vision

    assert "PE-Spatial-B16-512" in PE_CONFIGS
    cfg = PE_CONFIGS["PE-Spatial-B16-512"]
    assert (cfg.image_size, cfg.patch_size, cfg.width) == (512, 16, 768)
    assert cfg.layers == 12 and cfg.heads == 12
    assert cfg.use_cls_token is True
    assert cfg.pool_type == "none"
    assert cfg.use_ln_post is False
    assert cfg.output_dim is None

    m = build_pe_vision("PE-Spatial-B16-512").eval()
    with torch.no_grad():
        feats, _pooled = m.encode(torch.randn(1, 3, 512, 512))
    assert feats.shape == (1, 1025, 768)


def test_pe_spatial_bucket_spec_aspect_aligned():
    """PE-Spatial buckets mirror PE-Core aspects so dual-cache batching is
    1:1 across encoders."""
    import math

    from library.vision.buckets import get_bucket_spec, pick_bucket

    spec_core = get_bucket_spec("pe")
    spec_spatial = get_bucket_spec("pe_spatial")
    assert len(spec_core.buckets) == len(spec_spatial.buckets)

    aspects_core = sorted(h / w for h, w in spec_core.buckets)
    aspects_spatial = sorted(h / w for h, w in spec_spatial.buckets)
    for ac, asp in zip(aspects_core, aspects_spatial):
        assert abs(math.log(ac) - math.log(asp)) < 0.05, (ac, asp)

    for src_h, src_w in [
        (1024, 1024),
        (1024, 768),
        (768, 1024),
        (1024, 512),
        (512, 1024),
    ]:
        b_core = pick_bucket(src_h, src_w, spec_core)
        b_spat = pick_bucket(src_h, src_w, spec_spatial)
        rank_core = sorted(spec_core.buckets, key=lambda hw: hw[0] / hw[1]).index(
            b_core
        )
        rank_spat = sorted(spec_spatial.buckets, key=lambda hw: hw[0] / hw[1]).index(
            b_spat
        )
        assert rank_core == rank_spat, (src_h, src_w, b_core, b_spat)


def test_pe_spatial_encoder_registry_entry():
    from library.vision.encoders import get_encoder_info

    info = get_encoder_info("pe_spatial")
    assert info.d_enc == 768
    assert info.bucket_spec.patch == 16
    assert info.t_max_tokens() >= 1024


def test_dual_dataset_class_supports_per_side_pool_kind():
    """CachedDualDataset rejects a bad pool_kind before any disk access."""
    from pathlib import Path

    from library.captioning.anima_tagger_data import (
        CachedDualDataset,
        TaggerManifest,
    )

    fake_manifest = TaggerManifest(
        stems=[],
        image_paths=[],
        tag_indices=[],
        rating_indices=[],
        people_count_indices=[],
        train_stems=[],
        val_stems=[],
        n_tags=0,
        n_ratings=0,
        n_people_counts=0,
    )
    try:
        CachedDualDataset(
            fake_manifest,
            Path("/nonexistent/main"),
            "weird",
            None,
            Path("/nonexistent/aux"),
            "map",
            None,
        )
    except ValueError as e:
        assert "pool_kind" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown pool_kind")
