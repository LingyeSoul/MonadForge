"""Small CPU-only upstream/fork adapter contracts; no backward or optimizer."""
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from networks import NETWORK_KWARGS, NETWORK_REGISTRY, resolve_network_spec
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_modules import LoRAModule
from networks.lora_save import save_network_weights
from networks.lora_utils import relayout_adaln_comfy_to_runtime


def test_retained_families_and_upstream_kwargs_are_registered():
    families = {"lokr", "glokr", "loha", "ortho", "ortho_init", "ortho_hydra",
                "chimera_hydra", "stacked_experts_global_fei", "vera", "dylora"}
    assert families <= NETWORK_REGISTRY.keys()
    assert {"svd_slice", "svd_slice_count", "network_reg_alphas", "train_adaln",
            "adaln_rank", "adaln_alpha", "lora_fp32_compute"} <= NETWORK_KWARGS
    assert resolve_network_spec({"use_moe_style": "independent_A"}).name == "stacked_experts_global_fei"


def test_fork_routing_and_new_adaln_overrides_compose():
    cfg = LoRANetworkCfg.from_kwargs(
        {"use_moe_style": "independent_A", "router_source": "fei",
         "train_adaln": True, "adaln_rank": 2, "adaln_alpha": 1.5},
        network_dim=4, network_alpha=4, neuron_dropout=None,
        module_class=NETWORK_REGISTRY["stacked_experts_global_fei"].module_class,
    )
    assert cfg.use_moe_style == "independent_A"
    assert cfg.reg_dims[".*adaln_up_.*"] == 2
    assert cfg.reg_alphas[".*adaln_up_.*"] == 1.5


def test_from_weights_accepts_both_hydra_interface_names():
    args = dict(modules_dim={"m": 2}, modules_alpha={"m": 2},
                module_class=LoRAModule, train_llm_adapter=False,
                hydra_num_experts=2, sigma_feature_dim_detected=None,
                sigma_router_names=None, hydra_router_names=None,
                channel_scales_dict=None, new_use_moe_style="shared_A",
                new_route_per_layer=True, new_router_source="input")
    upstream = LoRANetworkCfg.from_weights(**args, is_hydra=True)
    retained = LoRANetworkCfg.from_weights(**args, is_hydra_or_ortho_hydra=True)
    assert upstream == retained


@pytest.mark.parametrize("variant", ["lokr", "glokr", "loha"])
def test_native_adapter_adaln_layout_roundtrip(tmp_path: Path, variant: str):
    prefix = "lora_unet_blocks_0_adaln_up_mlp"
    keys = {"lokr": {"lokr_w1": torch.ones(2, 2), "lokr_w2": torch.ones(4, 4)},
            "glokr": {"glokr_w1": torch.ones(2, 2), "glokr_w2": torch.ones(4, 4)},
            "loha": {"hada_w1_a": torch.ones(8, 2), "hada_w1_b": torch.ones(2, 8),
                     "hada_w2_a": torch.ones(8, 2), "hada_w2_b": torch.ones(2, 8)}}
    state = {f"{prefix}.{k}": v for k, v in keys[variant].items()}
    state[f"{prefix}.alpha"] = torch.tensor(2.)
    path = tmp_path / f"{variant}.safetensors"
    save_network_weights(state, file=str(path), dtype=torch.float32,
                         metadata={"ss_network_dim": "2"}, save_variant=variant)
    loaded = load_file(str(path))
    assert loaded
    assert all("adaln_up_" not in key for key in loaded)
    assert all("adaln_modulation_mlp_2" in key for key in loaded)
    restored = relayout_adaln_comfy_to_runtime(loaded)
    assert all(key.startswith(prefix + ".") for key in restored)
    with safe_open(path, framework="pt") as handle:
        assert handle.metadata()["ss_adaln_layout"] == "comfy"
