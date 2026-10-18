"""Opt-in real tiny-DiT regressions for adapter training and checkpoint reload."""

import pytest
import torch

from bench.anima_minimal_training.run_bench import run, tiny_anima
from networks.methods.controlnet_lllite import create_network


@pytest.mark.training_step
@pytest.mark.parametrize("variant", ["vera", "controlnet", "hydralora-8gb"])
def test_real_dit_adapter_training_and_reload(variant, tmp_path):
    result = run(variant, tmp_path)
    assert result["status"] == "passed", result.get("error")
    assert result["changed_parameters"]
    assert result["checkpoint_output_max_abs_error"] < 1e-4
    if variant == "controlnet":
        changed = result["changed_parameters"]
        assert "zero_conv_in.weight" in changed
        assert "zero_conv_blocks.0.weight" in changed
        assert result["condition_output_delta"] > 0
    elif variant == "vera":
        assert result["adapter_classes"]["VeRAModule"] > 0
    else:
        assert result["adapter_classes"]["HydraLoRAModule"] > 0


def test_controlnet_initial_injection_preserves_frozen_dit():
    torch.manual_seed(51)
    dit = tiny_anima().eval().requires_grad_(False)
    latent = torch.randn(1, 16, 1, 4, 4)
    timestep = torch.tensor([[0.5]])
    context = torch.randn(1, 4, 32)
    with torch.no_grad():
        baseline = dit(latent, timestep, context)
        network = create_network(1.0, 1, 1.0, None, [], dit)
        network.apply_to([], dit, False, True)
        try:
            network.set_cond_latent(torch.randn(1, 16, 4, 4))
            actual = dit(latent, timestep, context)
        finally:
            network.remove_from()
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
