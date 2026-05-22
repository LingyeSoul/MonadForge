"""Round-trip test for the plain-LoRA per_channel_scaling save/load path.

Regression for the silent-drop bug where ``defuse_standard_qkv`` (save side)
and ``_refuse_unfused_attn_lora_keys`` (load side) ignored ``inv_scale`` on
fused qkv/kv modules. With channel scaling on, the absorbed ``lora_down``
columns ship to disk while ``inv_scale`` is dropped — reload then treats
``lora_down`` as un-absorbed and silently produces wrong output.

Coverage:

* ``inv_scale`` survives the defuse → refuse round trip on a fused-qkv-shaped
  state_dict (``defuse_standard_qkv`` / ``_refuse_unfused_attn_lora_keys`` in
  isolation — the bake runs *after* defuse, so these helpers are unchanged).
* End-to-end: build a real ``LoRAModule`` with ``channel_scale``, run defuse,
  refuse, and verify the rebuilt fused module produces identical output to the
  in-memory original.
* Bake: the production ``defuse_and_bake_standard`` path now folds
  ``inv_scale`` into ``lora_down`` and drops the key, so a plain (no
  channel-scale) consumer reproduces the channel-scaled forward bitwise.
"""

from __future__ import annotations

import torch

from networks.lora_anima.loading import _refuse_unfused_attn_lora_keys
from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.lora import (
    LoRAModule,
    bake_inv_scale,
    defuse_and_bake_standard,
    defuse_standard_qkv,
)


def _make_calibration(in_dim: int, seed: int = 0) -> torch.Tensor:
    """Synthetic mean_abs vector with a single dominant channel.

    Matches the real bench: most channels small, one outlier that dominates.
    """
    torch.manual_seed(seed)
    stats = torch.rand(in_dim, dtype=torch.float32) * 0.3 + 0.05
    stats[7] = 50.0  # the dominant channel
    return stats


def test_defuse_carries_inv_scale_into_split_keys():
    """Save side: defuse_standard_qkv must clone inv_scale into each split key.

    Key names use the underscore-separated lora_name format
    (`lora_unet_blocks_0_self_attn_qkv_proj`) that match_fused_spec expects —
    the same shape LoRANetwork.create_modules produces in network.py.
    """
    in_dim = 32
    rank = 4
    out_per = 8
    n = 3  # q, k, v

    fused = "lora_unet_blocks_0_self_attn_qkv_proj"
    sd = {
        f"{fused}.lora_down.weight": torch.randn(rank, in_dim),
        f"{fused}.lora_up.weight": torch.randn(out_per * n, rank),
        f"{fused}.alpha": torch.tensor(float(rank)),
        f"{fused}.inv_scale": torch.rand(in_dim),
    }
    original_inv = sd[f"{fused}.inv_scale"].clone()

    defuse_standard_qkv(sd)

    for letter in ("q", "k", "v"):
        key = f"lora_unet_blocks_0_self_attn_{letter}_proj.inv_scale"
        assert key in sd, f"defuse dropped inv_scale at {key}"
        assert torch.equal(sd[key], original_inv), (
            f"inv_scale at {key} diverged from the fused source"
        )
    # And the fused-prefix inv_scale was popped — it doesn't survive on-disk.
    assert f"{fused}.inv_scale" not in sd


def test_refuse_picks_first_inv_scale_for_fused():
    """Load side: _refuse_unfused_attn_lora_keys must lift inv_scale to fused."""
    in_dim = 32
    rank = 4
    out_per = 8
    inv = torch.rand(in_dim)
    down = torch.randn(rank, in_dim)

    sd = {}
    for letter in ("q", "k", "v"):
        prefix = f"lora_unet_blocks_0_self_attn_{letter}_proj"
        sd[f"{prefix}.lora_down.weight"] = down.clone()
        sd[f"{prefix}.lora_up.weight"] = torch.randn(out_per, rank)
        sd[f"{prefix}.alpha"] = torch.tensor(float(rank))
        sd[f"{prefix}.inv_scale"] = inv.clone()

    _refuse_unfused_attn_lora_keys(sd)

    fused = "lora_unet_blocks_0_self_attn_qkv_proj"
    assert f"{fused}.inv_scale" in sd, "refuse failed to lift inv_scale to fused prefix"
    assert torch.equal(sd[f"{fused}.inv_scale"], inv)
    for letter in ("q", "k", "v"):
        assert (
            f"lora_unet_blocks_0_self_attn_{letter}_proj.inv_scale" not in sd
        ), "refuse left an orphan per-component inv_scale"


def test_refuse_warns_on_partial_inv_scale(caplog):
    """Mixing channel-scaled + non-channel-scaled per-component LoRAs is
    indeterminate on the fused runtime — log + drop, don't crash."""
    in_dim = 32
    rank = 4
    out_per = 8
    down = torch.randn(rank, in_dim)

    sd = {}
    for letter in ("q", "k", "v"):
        prefix = f"lora_unet_blocks_0_self_attn_{letter}_proj"
        sd[f"{prefix}.lora_down.weight"] = down.clone()
        sd[f"{prefix}.lora_up.weight"] = torch.randn(out_per, rank)
        sd[f"{prefix}.alpha"] = torch.tensor(float(rank))
    # Only q has inv_scale — partial set.
    sd["lora_unet_blocks_0_self_attn_q_proj.inv_scale"] = torch.rand(in_dim)

    with caplog.at_level("WARNING"):
        _refuse_unfused_attn_lora_keys(sd)

    fused = "lora_unet_blocks_0_self_attn_qkv_proj"
    assert f"{fused}.inv_scale" not in sd
    assert any("partial inv_scale" in r.message for r in caplog.records)


def test_end_to_end_roundtrip_preserves_forward():
    """Build a real LoRAModule with channel_scale, run save defuse + load refuse,
    rebuild a fused LoRAModule, confirm forward output matches the original."""
    in_dim = 32
    out_dim_per_comp = 16
    n = 3  # qkv
    out_dim = out_dim_per_comp * n
    rank = 4

    torch.manual_seed(42)
    base = torch.nn.Linear(in_dim, out_dim, bias=False)
    calibration = _make_calibration(in_dim)

    lora = LoRAModule(
        "lora_unet_blocks_0_self_attn_qkv_proj",
        base,
        multiplier=1.0,
        lora_dim=rank,
        alpha=rank,
        channel_scale=calibration,
    )
    # Mimic the post-apply training state: nonzero lora_up so the adapter
    # actually contributes (zero-init would make the test trivially pass).
    torch.nn.init.normal_(lora.lora_up.weight, std=0.05)
    lora.eval()

    # Capture the original adapter contribution on a fresh input.
    x = torch.randn(2, 7, in_dim) * 4.0
    x[..., 7] += 30.0  # exercise the dominant channel
    with torch.no_grad():
        lora_only_orig = lora.lora_up(lora.lora_down(lora._rebalance(x))) * lora.scale

    sd = {f"{lora.lora_name}.{k}": v for k, v in lora.state_dict().items()}
    assert f"{lora.lora_name}.inv_scale" in sd, (
        "channel-scaled LoRAModule must register inv_scale as a persistent buffer"
    )
    on_disk_inv = sd[f"{lora.lora_name}.inv_scale"].clone()

    # 1. Save defuse → on-disk split layout.
    defuse_standard_qkv(sd)
    for letter in ("q", "k", "v"):
        assert (
            f"lora_unet_blocks_0_self_attn_{letter}_proj.inv_scale" in sd
        ), "round-trip lost inv_scale on save"

    # 2. Load refuse → back to fused layout.
    _refuse_unfused_attn_lora_keys(sd)
    fused_prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    assert f"{fused_prefix}.inv_scale" in sd
    assert torch.equal(sd[f"{fused_prefix}.inv_scale"], on_disk_inv)

    # 3. Rebuild a fresh LoRAModule and load the round-tripped state_dict.
    base_rebuilt = torch.nn.Linear(in_dim, out_dim, bias=False)
    # Use the same base weight so the org_forward contribution matches.
    base_rebuilt.load_state_dict(base.state_dict())
    lora_rebuilt = LoRAModule(
        "lora_unet_blocks_0_self_attn_qkv_proj",
        base_rebuilt,
        multiplier=1.0,
        lora_dim=rank,
        alpha=rank,
        channel_scale=calibration,
    )
    rebuilt_sd = {
        k[len(fused_prefix) + 1 :]: v for k, v in sd.items() if k.startswith(fused_prefix + ".")
    }
    missing, unexpected = lora_rebuilt.load_state_dict(rebuilt_sd, strict=False)
    assert not unexpected, f"unexpected keys after round-trip: {unexpected}"
    lora_rebuilt.eval()

    with torch.no_grad():
        lora_only_new = (
            lora_rebuilt.lora_up(
                lora_rebuilt.lora_down(lora_rebuilt._rebalance(x))
            )
            * lora_rebuilt.scale
        )

    # The pre-fused detection in _refuse_unfused_attn_lora_keys collapses
    # identical-down splits back into the original fused layout, so output
    # must match bitwise after the rebuild (fp32 throughout).
    assert torch.allclose(lora_only_new, lora_only_orig, atol=1e-6), (
        "round-tripped LoRA adapter output diverged from the original"
    )


def test_inv_scale_round_trip_is_idempotent_under_repeated_defuse():
    """Reapplying defuse on an already-split state_dict must not duplicate
    or strip inv_scale (regression for "double defuse" workflows)."""
    in_dim = 32
    rank = 4
    out_per = 8
    inv = torch.rand(in_dim)

    sd = {}
    for letter in ("q", "k", "v"):
        prefix = f"blocks.0.self_attn.{letter}_proj"
        sd[f"{prefix}.lora_down.weight"] = torch.randn(rank, in_dim)
        sd[f"{prefix}.lora_up.weight"] = torch.randn(out_per, rank)
        sd[f"{prefix}.alpha"] = torch.tensor(float(rank))
        sd[f"{prefix}.inv_scale"] = inv.clone()

    snapshot = {k: v.clone() for k, v in sd.items()}
    # Per-component prefixes don't end in `_qkv_proj`/`_kv_proj` so
    # match_fused_spec returns None → no-op.
    defuse_standard_qkv(sd)
    for k, v in snapshot.items():
        assert torch.equal(sd[k], v), f"defuse mutated already-split key {k}"


def test_bake_inv_scale_drops_key_and_folds_down():
    """bake_inv_scale folds inv_scale into the sibling down and removes the key."""
    in_dim, rank, out_dim = 32, 4, 16
    down = torch.randn(rank, in_dim)
    inv = torch.rand(in_dim) + 0.1
    prefix = "lora_unet_blocks_0_self_attn_q_proj"
    sd = {
        f"{prefix}.lora_down.weight": down.clone(),
        f"{prefix}.lora_up.weight": torch.randn(out_dim, rank),
        f"{prefix}.alpha": torch.tensor(float(rank)),
        f"{prefix}.inv_scale": inv.clone(),
    }
    bake_inv_scale(sd)
    assert f"{prefix}.inv_scale" not in sd, "bake must drop the inv_scale key"
    expected = down * inv.unsqueeze(0)
    assert torch.allclose(sd[f"{prefix}.lora_down.weight"], expected, atol=1e-6)


def test_baked_save_reproduces_channel_scaled_forward():
    """The production save path (defuse_and_bake_standard) bakes
    inv_scale, so a *plain* LoRAModule (no channel_scale) loading the baked
    weights reproduces the original channel-scaled forward bitwise — this is
    what stock ComfyUI / merge_to_dit now apply correctly."""
    in_dim = 32
    out_dim_per_comp = 16
    n = 3
    out_dim = out_dim_per_comp * n
    rank = 4

    torch.manual_seed(7)
    base = torch.nn.Linear(in_dim, out_dim, bias=False)
    calibration = _make_calibration(in_dim)

    lora = LoRAModule(
        "lora_unet_blocks_0_self_attn_qkv_proj",
        base,
        multiplier=1.0,
        lora_dim=rank,
        alpha=rank,
        channel_scale=calibration,
    )
    torch.nn.init.normal_(lora.lora_up.weight, std=0.05)
    lora.eval()

    x = torch.randn(2, 7, in_dim) * 4.0
    x[..., 7] += 30.0
    with torch.no_grad():
        orig = lora.lora_up(lora.lora_down(lora._rebalance(x))) * lora.scale

    # Production save: defuse → bake. After this there are NO inv_scale keys.
    sd = {f"{lora.lora_name}.{k}": v for k, v in lora.state_dict().items()}
    defuse_and_bake_standard(sd)
    assert not any(k.endswith(".inv_scale") for k in sd), (
        "baked save must not ship inv_scale keys"
    )

    # Load refuse (re-fuse split q/k/v) and rebuild a PLAIN module (channel
    # scaling off — the absence of inv_scale keys is the inference signal).
    _refuse_unfused_attn_lora_keys(sd)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"
    base_rebuilt = torch.nn.Linear(in_dim, out_dim, bias=False)
    base_rebuilt.load_state_dict(base.state_dict())
    plain = LoRAModule(fused, base_rebuilt, multiplier=1.0, lora_dim=rank, alpha=rank)
    rebuilt_sd = {
        k[len(fused) + 1 :]: v for k, v in sd.items() if k.startswith(fused + ".")
    }
    _, unexpected = plain.load_state_dict(rebuilt_sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    plain.eval()
    assert not plain._has_channel_scale

    with torch.no_grad():
        new = plain.lora_up(plain.lora_down(plain._rebalance(x))) * plain.scale

    assert torch.allclose(new, orig, atol=1e-6), (
        "baked plain forward diverged from the channel-scaled original"
    )


# Sanity: BaseLoRAModule.inv_scale buffer is persistent so state_dict() carries it.
def test_inv_scale_buffer_is_persistent():
    in_dim = 8
    base = torch.nn.Linear(in_dim, 16, bias=False)
    calibration = torch.rand(in_dim)

    lora = LoRAModule(
        "test_lora", base, multiplier=1.0, lora_dim=4, alpha=4,
        channel_scale=calibration,
    )
    assert "inv_scale" in lora.state_dict(), (
        "inv_scale must be a persistent buffer so save_weights picks it up"
    )
    assert isinstance(lora, BaseLoRAModule)
