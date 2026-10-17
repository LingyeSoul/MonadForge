"""Invariant tests for the --traj_stats passive recorder (Phase 0).

Pins the proposal's Tier-1.5 contract (_archive/proposals/traj_latent_stats.md):
pure observation (inputs bit-identical after record), explicit dim-2 squeeze
discipline, commit/trace semantics, npz round-trip, and the Qwen VAE
normalization constants staying in lockstep with the model defaults.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from library.inference.traj_stats import (
    QWEN_LATENTS_MEAN,
    QWEN_LATENTS_STD,
    TrajStatsRecorder,
    derive_summary,
)


def _mk_recorder(tmp_path, **kw) -> TrajStatsRecorder:
    return TrajStatsRecorder(str(tmp_path), **kw)


def test_constants_match_qwen_vae_defaults():
    from library.models.qwen_vae import AutoencoderKLQwenImage

    sig = inspect.signature(AutoencoderKLQwenImage.__init__)
    assert sig.parameters["latents_mean"].default == QWEN_LATENTS_MEAN
    assert sig.parameters["latents_std"].default == QWEN_LATENTS_STD


def test_from_args_off_by_default():
    assert TrajStatsRecorder.from_args(SimpleNamespace()) is None
    assert TrajStatsRecorder.from_args(SimpleNamespace(traj_stats=False)) is None


def test_from_args_on(tmp_path):
    args = SimpleNamespace(
        traj_stats=True,
        traj_stats_dir=str(tmp_path),
        traj_stats_k=5,
        prompt="a girl",
        sampler="er_sde",
        infer_steps=4,
        guidance_scale=4.0,
        image_size=(512, 512),
    )
    rec = TrajStatsRecorder.from_args(args, seed=42)
    assert rec is not None
    assert rec.k == 5 and rec.levels == 32
    assert rec.seed == 42


def test_record_is_pure_observation(tmp_path):
    torch.manual_seed(0)
    rec = _mk_recorder(tmp_path)
    latents = torch.randn(2, 16, 1, 8, 8, dtype=torch.bfloat16)
    noise = torch.randn(2, 16, 1, 8, 8, dtype=torch.bfloat16)
    uncond = torch.randn(2, 16, 1, 8, 8, dtype=torch.bfloat16)
    refs = [t.clone() for t in (latents, noise, uncond)]
    for i, sigma in enumerate([1.0, 0.6, 0.2]):
        rec.record(i, sigma, latents, noise, uncond)
    for t, ref in zip((latents, noise, uncond), refs):
        assert torch.equal(t, ref)


def test_dim2_squeeze_discipline_b1(tmp_path):
    # B=1 is the trap a bare squeeze() would corrupt (batch dim dropped).
    rec = _mk_recorder(tmp_path)
    x = torch.randn(1, 16, 1, 8, 8)
    rec.record(0, 1.0, x, torch.zeros_like(x))
    step = rec._steps[0]
    assert step["activity"].shape == (1, 4, 4)
    assert step["codes"].shape == (1, 16, 4, 4)
    assert step["cbits"].shape == (16,)
    # a wrong T dim must be rejected, not silently squeezed
    with pytest.raises(AssertionError):
        rec.record(1, 0.5, torch.randn(1, 16, 2, 8, 8), torch.zeros(1, 16, 2, 8, 8))


def test_commit_and_activity_semantics(tmp_path):
    # C=3 -> identity normalization (non-Qwen channel count), so values are
    # controlled exactly. noise_pred=0, sigma anything => x̂₀ == latents.
    rec = _mk_recorder(tmp_path)
    frames = []
    for i in range(4):
        x = torch.full((1, 3, 1, 4, 4), -3.9)
        if i == 0:
            # token cell (0,0) (= latent pixels [0:2, 0:2]) starts far across
            # the quantization support, flips at step 1, then locks.
            x[:, :, :, 0:2, 0:2] = 3.9
        frames.append(x)
        rec.record(i, 0.5, x, torch.zeros_like(x))

    commit = rec._commit  # (1, 2, 2)
    assert commit.shape == (1, 2, 2)
    # token (0,0): last code change at step 1 (3.9 -> -3.9 between steps 0->1)
    assert int(commit[0, 0, 0]) == 1
    # all other tokens never changed after step 0
    assert int(commit[0, 0, 1]) == 0
    assert int(commit[0, 1, 0]) == 0
    assert int(commit[0, 1, 1]) == 0

    # activity: NaN at step 0, positive at flip steps, zero once locked
    a0 = rec._steps[0]["activity"]
    assert torch.isnan(a0).all()
    a1 = rec._steps[1]["activity"].float()
    assert a1[0, 0, 0] > 0
    assert a1[0, 1, 1] == 0
    a3 = rec._steps[3]["activity"].float()
    assert a3[0, 0, 0] == 0


def test_guide_nan_without_cfg(tmp_path):
    rec = _mk_recorder(tmp_path)
    x = torch.randn(1, 16, 1, 8, 8)
    rec.record(0, 1.0, x, torch.randn_like(x))  # no uncond
    assert torch.isnan(rec._steps[0]["guide"]).all()
    rec.record(1, 0.5, x, torch.randn_like(x), torch.randn_like(x))
    assert not torch.isnan(rec._steps[1]["guide"]).any()


def test_invert_recorder_is_pure_observation(tmp_path):
    # Phase 1 real-image arm: the recorder threaded through DirectEdit
    # inversion must leave z_inv / delta_z bit-identical.
    from library.inference.editing.directedit import invert

    def fake_anima(latents, t_expand, embed, padding_mask=None):
        # deterministic, latent- and σ-dependent pseudo-velocity
        return (latents * 0.25 + t_expand.view(-1, 1, 1, 1, 1).to(latents.dtype)).to(
            latents.dtype
        )

    torch.manual_seed(3)
    z_clean = torch.randn(1, 16, 1, 8, 8, dtype=torch.bfloat16)
    embed = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    sigmas = torch.linspace(1.0, 0.0, 5)

    z_ref, d_ref = invert(fake_anima, z_clean.clone(), embed, None, sigmas)

    rec = _mk_recorder(tmp_path)
    z_rec, d_rec = invert(
        fake_anima, z_clean.clone(), embed, None, sigmas, traj_recorder=rec
    )
    assert all(torch.equal(a, b) for a, b in zip(z_ref, z_rec))
    assert all(torch.equal(a, b) for a, b in zip(d_ref, d_rec))

    # one record per step, natural inversion order: σ ascending (clean → noise)
    path = rec.flush()
    d = np.load(path)
    assert d["codes"].shape[0] == sigmas.shape[0] - 1
    rec_sigmas = d["sigmas"]
    assert all(a < b for a, b in zip(rec_sigmas, rec_sigmas[1:]))
    assert rec_sigmas[-1] == pytest.approx(1.0)


def test_flush_roundtrip_and_idempotent(tmp_path):
    torch.manual_seed(1)
    rec = _mk_recorder(tmp_path, seed=7)
    steps = 3
    for i, sigma in enumerate(np.linspace(1.0, 0.1, steps)):
        x = torch.randn(1, 16, 1, 8, 8)
        rec.record(i, float(sigma), x, torch.randn_like(x), torch.randn_like(x))
    path = rec.flush()
    assert path is not None and path.endswith("_seed7.npz")

    d = np.load(path)
    assert d["tok"].shape == (steps, 1, 16, 4, 4)
    assert d["tok"].dtype == np.float16
    assert d["activity"].shape == (steps, 1, 4, 4)
    assert d["hf"].shape == (steps, 1, 4, 4)
    assert d["guide"].shape == (steps, 1, 4, 4)
    assert d["cbits"].shape == (steps, 16)
    assert d["codes"].shape == (steps, 1, 16, 4, 4)
    assert d["codes"].dtype == np.uint8
    assert d["commit"].shape == (1, 4, 4)
    assert d["sigmas"].shape == (steps,)
    header = json.loads(str(d["header_json"]))
    assert header["k"] == 4 and header["seed"] == 7 and header["num_steps"] == steps

    # idempotent: state cleared, second flush is a no-op
    assert rec.flush() is None

    summary = derive_summary(path)
    assert len(summary["E"]) == steps
    assert len(summary["commit_cdf"]) == steps
    assert summary["commit_cdf"][-1] == 1.0
    assert all(a <= b for a, b in zip(summary["commit_cdf"], summary["commit_cdf"][1:]))
