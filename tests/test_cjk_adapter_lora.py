"""plan3 — ext-gated adapter LoRA (scripts/distill_cjk/adapter_lora.py).

(a) EN bit-exactness by construction: a pure-EN batch through the LoRA-hooked
    adapter with a random NON-zero up matrix is ``torch.equal`` to the base.
(b) The gate flips on a single ext id — and only for that sequence.
(c) State-dict keys round-trip through the kohya lora_name convention the
    Adapter Loader parses, and a save→load rebuilds the same delta.
"""

from __future__ import annotations

import torch

from library.anima.models import LLMAdapter
from scripts.distill_cjk import adapter_lora as al
from scripts.distill_cjk.ext_table import T5_TABLE_SIZE

torch.manual_seed(0)


def _adapter(n_layers=2):
    m = LLMAdapter(1024, 1024, 1024, num_layers=n_layers, self_attn=True)
    m.embed = torch.nn.Embedding(T5_TABLE_SIZE + 64, 1024)  # room for ext ids
    return m.eval()


def _run(adapter, ids, qwen):
    mask = torch.ones_like(ids, dtype=torch.bool)
    return adapter(
        source_hidden_states=qwen,
        target_input_ids=ids,
        target_attention_mask=mask,
        source_attention_mask=torch.ones(qwen.shape[:2], dtype=torch.bool),
    )


def _randomize_up(lora):
    for m in lora.loras.values():
        torch.nn.init.normal_(m.lora_up.weight, std=0.05)


def test_pure_en_batch_is_bit_exact_with_nonzero_lora():
    adapter = _adapter()
    ids = torch.randint(0, T5_TABLE_SIZE, (3, 12))
    qwen = torch.randn(3, 20, 1024)
    with torch.no_grad():
        ref = _run(adapter, ids, qwen)
        lora = al.AdapterLoRA(adapter, rank=4).attach(adapter)
        _randomize_up(lora)
        out = _run(adapter, ids, qwen)
    assert torch.equal(out, ref)
    assert lora._gate is None  # post-hook cleared it


def test_gate_flips_on_one_ext_id_and_only_that_sequence():
    adapter = _adapter()
    ids = torch.randint(0, T5_TABLE_SIZE, (3, 12))
    ids[1, 5] = T5_TABLE_SIZE + 3  # one ext id in sequence 1 only
    qwen = torch.randn(3, 20, 1024)
    g = al.AdapterLoRA.gate_from_ids(ids)
    assert g.shape == (3, 1, 1) and g.flatten().tolist() == [0.0, 1.0, 0.0]
    with torch.no_grad():
        ref = _run(adapter, ids, qwen)
        lora = al.AdapterLoRA(adapter, rank=4).attach(adapter)
        _randomize_up(lora)
        out = _run(adapter, ids, qwen)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[2], ref[2])
    assert not torch.allclose(out[1], ref[1])
    # zero-init up ⇒ identity even for the ext sequence
    lora2 = al.AdapterLoRA(adapter, rank=4)
    lora.detach()
    lora2.attach(adapter)
    with torch.no_grad():
        assert torch.equal(_run(adapter, ids, qwen), ref)


def test_detach_restores_the_base_adapter():
    adapter = _adapter()
    ids = torch.full((1, 6), T5_TABLE_SIZE + 1)
    qwen = torch.randn(1, 8, 1024)
    with torch.no_grad():
        ref = _run(adapter, ids, qwen)
        lora = al.AdapterLoRA(adapter, rank=4).attach(adapter)
        _randomize_up(lora)
        assert not torch.allclose(_run(adapter, ids, qwen), ref)
        lora.detach()
        assert torch.equal(_run(adapter, ids, qwen), ref)


def test_only_lora_leaves_are_parameters_and_targets_resolve():
    adapter = _adapter()
    lora = al.AdapterLoRA(adapter, rank=16, targets="self_qkvo,cross_q")
    assert lora.targets == (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "cross_attn.q_proj",
    )
    assert len(lora.loras) == 2 * 5
    assert lora.n_params() == 2 * 5 * 2 * 1024 * 16
    assert all(p.requires_grad for p in lora.parameters())


def test_keys_round_trip_the_kohya_lora_name_convention():
    adapter = _adapter()
    lora = al.AdapterLoRA(adapter, rank=4, targets="self_qkvo,cross_q,cross_kv,mlp")
    linears = {n for n, m in adapter.named_modules() if isinstance(m, torch.nn.Linear)}
    for key, path in lora._paths.items():
        assert key.startswith("lora_unet_llm_adapter_")
        assert al.lora_name(path) == key
        assert al.module_path(key) == path
        assert path in linears
    sd = lora.export_state_dict()
    for key in lora._paths:
        assert f"{key}.lora_down.weight" in sd
        assert f"{key}.lora_up.weight" in sd
        assert float(sd[f"{key}.alpha"]) == 4.0


def test_save_load_reproduces_the_delta(tmp_path):
    adapter = _adapter()
    ids = torch.randint(0, T5_TABLE_SIZE, (2, 10))
    ids[:, 0] = T5_TABLE_SIZE + 7
    qwen = torch.randn(2, 8, 1024)
    lora = al.AdapterLoRA(adapter, rank=4, targets="self_qkvo,cross_q").attach(adapter)
    _randomize_up(lora)
    with torch.no_grad():
        want = _run(adapter, ids, qwen)
    path = tmp_path / "x.adapter_lora.safetensors"
    lora.save(path)
    lora.detach()

    fresh = _adapter()
    fresh.load_state_dict(adapter.state_dict())
    loaded = al.AdapterLoRA.load(fresh, path)
    assert loaded.rank == 4 and loaded.targets == lora.targets
    with torch.no_grad():
        got = _run(fresh, ids, qwen)
    assert torch.equal(got, want)

    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata()
    assert meta["ss_network_dim"] == "4"
    assert meta["ss_adapter_lora_targets"] == ",".join(lora.targets)


def test_attach_does_not_absorb_the_adapter_into_the_lora_parameters():
    adapter = _adapter()
    lora = al.AdapterLoRA(adapter, rank=4, targets="self_qkvo,cross_q").attach(adapter)
    assert lora.n_params() == 2 * 5 * 2 * 1024 * 4
    lora_ids = {id(p) for p in lora.parameters()}
    assert not any(id(p) in lora_ids for p in adapter.parameters())
