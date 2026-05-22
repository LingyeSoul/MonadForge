"""Round-trip the soft-token on-disk format through the ComfyUI custom node's
``soft_tokens.py`` module.

``networks/methods/soft_tokens.py::SoftTokensNetwork`` saves exactly two
tensors (``tokens`` + ``t_offsets.weight``) plus ``ss_*`` metadata. This test
hand-rolls a tiny checkpoint matching that layout and verifies:

  1. ``load_soft_tokens`` parses the bank, infers (n_layers, K, D, n_t_buckets)
     from the tensor shapes, and reads ``ss_splice_position`` from metadata.
  2. The diffusion_model pre-hook divides comfy's ``sigma * 1000`` timesteps
     back to [0, 1], bucketizes per-sample, and precomputes a
     ``(n_layers, B, K, D)`` token bank.
  3. The per-block pre-hook splices its layer's tokens into the ``crossattn_emb``
     positional arg for both splice positions, and that swapping the t-bucket
     (via different sigma) changes the spliced tokens.

We avoid exercising ``apply_soft_tokens`` end-to-end because it needs ComfyUI's
ModelPatcher (not on the unit-test sys.path) — this is a loader/runtime smoke
test, not a full ComfyUI integration test.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


def _load_soft_tokens_module():
    """Import the node's ``soft_tokens.py`` without depending on ComfyUI.

    ``soft_tokens.py`` has no ComfyUI imports at module scope (only
    ``apply_soft_tokens`` touches the ModelPatcher), so we can load it as a
    standalone submodule of a stub package — skipping the node's real
    ``__init__.py`` (which imports ComfyUI via ``folder_paths``).
    """
    here = Path(__file__).resolve().parent.parent
    node_dir = here / "custom_nodes" / "comfyui-hydralora"

    pkg_name = "_anima_node_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(node_dir)]
        sys.modules[pkg_name] = pkg

    mod_name = f"{pkg_name}.soft_tokens"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(mod_name, node_dir / "soft_tokens.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_soft_tokens_checkpoint(
    path: Path,
    *,
    n_layers: int,
    K: int,
    D: int,
    n_t_buckets: int,
    splice_position: str,
) -> None:
    """Synth a minimal soft-token safetensors file."""
    torch.manual_seed(7)
    sd = {
        "tokens": torch.randn(n_layers, K, D) * 0.02,
        # Distinct, large per-bucket offsets so the bucketize result is
        # observable downstream (bucket k gets value ~k).
        "t_offsets.weight": (
            torch.arange(n_t_buckets, dtype=torch.float32)
            .view(n_t_buckets, 1)
            .expand(n_t_buckets, n_layers * D)
            .contiguous()
        ),
    }
    metadata = {
        "ss_num_tokens": str(K),
        "ss_embed_dim": str(D),
        "ss_n_layers": str(n_layers),
        "ss_n_t_buckets": str(n_t_buckets),
        "ss_splice_position": splice_position,
    }
    save_file(sd, str(path), metadata=metadata)


def test_load_soft_tokens_infers_shapes_and_splice(tmp_path):
    st = _load_soft_tokens_module()
    st._bank_cache.clear()

    path = tmp_path / "anima_soft_tokens.safetensors"
    _write_soft_tokens_checkpoint(
        path, n_layers=4, K=3, D=8, n_t_buckets=10, splice_position="end_of_sequence"
    )

    bank = st.load_soft_tokens(str(path))
    assert bank.n_layers == 4
    assert bank.num_tokens == 3
    assert bank.embed_dim == 8
    assert bank.n_t_buckets == 10
    assert bank.splice_position == "end_of_sequence"


def test_load_soft_tokens_rejects_missing_keys(tmp_path):
    st = _load_soft_tokens_module()
    st._bank_cache.clear()
    path = tmp_path / "bad.safetensors"
    save_file({"tokens": torch.randn(2, 2, 4)}, str(path))
    with pytest.raises(ValueError, match="t_offsets.weight"):
        st.load_soft_tokens(str(path))


def test_step_pre_hook_divides_sigma_and_bucketizes(tmp_path):
    """The diffusion_model pre-hook recovers [0,1] sigma from comfy's
    ``sigma * 1000`` timesteps, then picks the matching t-bucket offset."""
    st = _load_soft_tokens_module()
    st._bank_cache.clear()
    path = tmp_path / "anima_soft_tokens.safetensors"
    n_layers, K, D, n_t_buckets = 4, 3, 8, 10
    _write_soft_tokens_checkpoint(
        path,
        n_layers=n_layers,
        K=K,
        D=D,
        n_t_buckets=n_t_buckets,
        splice_position="end_of_sequence",
    )
    bank = st.load_soft_tokens(str(path))

    state: dict = {}
    hook = st._make_step_pre_hook(state, bank, strength=1.0)

    # comfy timesteps = sigma * 1000. sigma=0.05 -> bucket 0; sigma=0.95 -> 9.
    timesteps = torch.tensor([50.0, 950.0])
    latent = torch.randn(2, 4, 1, 16, 16)
    hook(None, (latent, timesteps))

    step_tokens = state.get("step_tokens")
    assert step_tokens is not None
    assert step_tokens.shape == (n_layers, 2, K, D)
    # Offset for bucket b is the constant b (see synth checkpoint). The base
    # token is ~0.02, so the per-step token mean should track the bucket index.
    mean_sample0 = step_tokens[:, 0].mean().item()
    mean_sample1 = step_tokens[:, 1].mean().item()
    assert abs(mean_sample0 - 0.0) < 0.5, mean_sample0  # bucket 0
    assert abs(mean_sample1 - 9.0) < 0.5, mean_sample1  # bucket 9


def test_block_pre_hook_splices_end_of_sequence(tmp_path):
    """Block pre-hook overwrites the K tail slots of crossattn_emb (arg index
    2) and leaves the shape unchanged."""
    st = _load_soft_tokens_module()
    st._bank_cache.clear()
    path = tmp_path / "anima_soft_tokens.safetensors"
    n_layers, K, D, n_t_buckets = 4, 3, 8, 10
    _write_soft_tokens_checkpoint(
        path,
        n_layers=n_layers,
        K=K,
        D=D,
        n_t_buckets=n_t_buckets,
        splice_position="end_of_sequence",
    )
    bank = st.load_soft_tokens(str(path))

    state: dict = {}
    step_hook = st._make_step_pre_hook(state, bank, strength=1.0)
    block_hook = st._make_block_pre_hook(
        1, state, bank.splice_position, bank.num_tokens
    )

    B, S = 2, 16
    timesteps = torch.tensor([950.0, 950.0])  # bucket 9 for both
    step_hook(None, (torch.randn(B, 4, 1, 16, 16), timesteps))

    x = torch.randn(B, 1, 16, 16, 32)
    emb = torch.randn(B, 1, 64)
    ctx = torch.randn(B, S, D)
    new_args = block_hook(None, (x, emb, ctx))
    assert new_args is not None
    new_ctx = new_args[2]
    assert new_ctx.shape == ctx.shape
    # Head (real text) is untouched; the K tail is overwritten with the bank.
    assert torch.allclose(new_ctx[:, : S - K, :], ctx[:, : S - K, :])
    assert not torch.allclose(new_ctx[:, S - K :, :], ctx[:, S - K :, :])
    # x / emb args pass through unchanged.
    assert new_args[0] is x and new_args[1] is emb


def test_block_pre_hook_front_of_padding_scatters_after_text(tmp_path):
    """front_of_padding places the tokens right after the real (non-zero)
    text tokens, derived from the crossattn_emb non-zero mask."""
    st = _load_soft_tokens_module()
    st._bank_cache.clear()
    path = tmp_path / "anima_soft_tokens.safetensors"
    n_layers, K, D, n_t_buckets = 2, 2, 8, 10
    _write_soft_tokens_checkpoint(
        path,
        n_layers=n_layers,
        K=K,
        D=D,
        n_t_buckets=n_t_buckets,
        splice_position="front_of_padding",
    )
    bank = st.load_soft_tokens(str(path))

    state: dict = {}
    step_hook = st._make_step_pre_hook(state, bank, strength=1.0)
    block_hook = st._make_block_pre_hook(
        0, state, bank.splice_position, bank.num_tokens
    )

    B, S = 1, 16
    real_len = 5
    step_hook(None, (torch.randn(B, 4, 1, 16, 16), torch.tensor([950.0])))

    ctx = torch.zeros(B, S, D)
    ctx[:, :real_len, :] = torch.randn(B, real_len, D)  # real text, rest padding
    new_ctx = block_hook(
        None, (torch.randn(B, 1, 16, 16, 32), torch.randn(B, 1, 64), ctx)
    )[2]
    # The K slots starting at real_len now hold the (non-zero) bank tokens.
    assert new_ctx[:, real_len : real_len + K, :].abs().sum() > 0
    # Real text head is preserved.
    assert torch.allclose(new_ctx[:, :real_len, :], ctx[:, :real_len, :])


def test_block_pre_hook_noops_without_step_tokens(tmp_path):
    """If the step pre-hook hasn't run (no step_tokens in state), the block
    pre-hook is a pass-through (returns None)."""
    st = _load_soft_tokens_module()
    st._bank_cache.clear()
    block_hook = st._make_block_pre_hook(0, {}, "end_of_sequence", 3)
    out = block_hook(
        None, (torch.randn(1, 1, 1), torch.randn(1, 1), torch.randn(1, 16, 8))
    )
    assert out is None


def test_strength_scales_spliced_tokens(tmp_path):
    """strength multiplies the precomputed bank, so strength=0 leaves the
    padding-slot splice at zero (i.e. no effective conditioning change)."""
    st = _load_soft_tokens_module()
    st._bank_cache.clear()
    path = tmp_path / "anima_soft_tokens.safetensors"
    _write_soft_tokens_checkpoint(
        path,
        n_layers=2,
        K=2,
        D=8,
        n_t_buckets=10,
        splice_position="end_of_sequence",
    )
    bank = st.load_soft_tokens(str(path))

    state_half: dict = {}
    st._make_step_pre_hook(state_half, bank, strength=0.5)(
        None, (torch.randn(1, 4, 1, 16, 16), torch.tensor([950.0]))
    )
    state_full: dict = {}
    st._make_step_pre_hook(state_full, bank, strength=1.0)(
        None, (torch.randn(1, 4, 1, 16, 16), torch.tensor([950.0]))
    )
    assert torch.allclose(
        state_half["step_tokens"], state_full["step_tokens"] * 0.5, atol=1e-5
    )
