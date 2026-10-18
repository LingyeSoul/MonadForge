"""CPU tiny real-Anima adapter training and safetensors reload audit.

No production data/config changes. Uses actual model/adapter/conditioning code;
only architecture dimensions and workload are reduced. No model downloads.
"""

# CPU visibility and the repository import path must be set before Torch loads.
# ruff: noqa: E402

import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import copy
import importlib
import json
import time
import traceback
from collections import Counter
from types import SimpleNamespace
import torch
from library.anima.models import Anima
from library.config.io import load_method_preset
from library.training.forward.router_conditioning import apply_router_conditioning
from networks import NETWORK_KWARGS
from scripts.tasks.training import _resolve_gui_variant


def tiny_anima(context_dim=32, blocks=1):
    return Anima(
        max_img_h=4,
        max_img_w=4,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=32,
        num_blocks=blocks,
        num_heads=2,
        mlp_ratio=4,
        crossattn_emb_channels=context_dim,
        adaln_lora_dim=4,
        use_llm_adapter=False,
        pos_emb_learnable=False,
    )


def config(variant):
    c = load_method_preset(
        _resolve_gui_variant(variant), "default", methods_subdir="gui-methods"
    )
    kw = {k: str(v) for k, v in c.items() if k in NETWORK_KWARGS}
    kw.update(dict(a.split("=", 1) for a in c.get("network_args", []) or []))
    # Numeric smoke uses no external calibration/teacher weights, compile,
    # activation swapping or optional visual teacher losses.
    kw.update(channel_scaling_alpha="0", use_repa="false", adaln_rank="1")
    rank = (
        2
        if any(
            str(c.get(k, False)).lower() == "true"
            for k in ("use_timestep_mask", "use_dylora", "use_chimera_hydra")
        )
        else 1
    )
    kw.update(min_rank="1", dylora_unit="1")
    if str(c.get("use_chimera_hydra", False)).lower() == "true":
        kw.update(num_experts_content="2", num_experts_freq="2")
    if "soft_tokens" in c.get("network_module", ""):
        kw.update(
            embed_dim="32",
            n_layers="1",
            n_t_buckets="2",
            contrastive_every_n="1",
            contrastive_warmup_ratio="0",
        )
    return c, kw, rank


def condition(net, x, t, ctx, cond, step, steps, branch=0):
    apply_router_conditioning(
        network=net,
        noisy_model_input=x,
        timesteps=t.flatten(),
        is_train=net.training,
        warmup_step=step,
        max_train_steps=steps,
    )
    if hasattr(net, "set_content"):
        net.set_content(ctx)
    if hasattr(net, "set_crossattn_routing"):
        net.set_crossattn_routing(ctx)
    if hasattr(net, "set_cond"):
        net.set_cond(cond)
    if hasattr(net, "set_cond_latent"):
        net.set_cond_latent(cond)
    if hasattr(net, "_set_step_tokens"):
        net._set_step_tokens(t, torch.tensor([2]), branch=branch)


def run(variant, output_dir):
    torch.manual_seed(314)
    c, kw, rank = config(variant)
    context_dim = 1024 if kw.get("use_chimera_hydra", "").lower() == "true" else 32
    blocks = 2 if "easycontrol" in c.get("network_module", "") else 1
    dit = tiny_anima(context_dim, blocks)
    base_state = copy.deepcopy(dit.state_dict())
    dit.requires_grad_(False)
    module = importlib.import_module(c.get("network_module", "networks.lora_anima"))
    net = module.create_network(1.0, rank, float(rank), None, [], dit, **kw)
    net.apply_to([], dit, False, True)
    net.prepare_grad_etc(None, dit)
    net.train()
    convrot = None
    if c.get("base_compute") == "w8a16_convrot":
        from library.training.convrot import maybe_apply_convrot_base

        args = SimpleNamespace(
            **{
                **c,
                "convrot_group_size": 64,
                "convrot_min_in_features": 0,
                "convrot_largest_in_features_only": False,
                "blocks_to_swap": 0,
                "block_swap_transfer_dtype": "bf16",
            }
        )
        assert maybe_apply_convrot_base(args, net, unet=dit, network_kwargs=kw)
        convrot = {
            "patched_count": args._convrot_apply_result.patched_count,
            "group_size": 64,
        }
    named = {n: p for n, p in net.named_parameters() if p.requires_grad}
    before = {n: p.detach().clone() for n, p in named.items()}
    assert named, "no trainable parameters"
    opt = torch.optim.AdamW(list(named.values()), lr=1e-3, weight_decay=0)
    x = torch.randn(1, 16, 1, 4, 4)
    target = torch.randn_like(x)
    ctx = torch.randn(1, 4, context_dim)
    ctx[:, 2:] = 0  # short padded text; no padding mask passed to cross attention
    cond = torch.randn(1, 16, 4, 4)
    routed = bool(kw.get("use_moe_style") not in (None, "False", "false"))
    steps = 6 if routed else 3
    history = []
    grad_names = set()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        sigma = (
            ([0.25, 0.65, 0.9] * 2)[step] if routed else (1.0 if step == 0 else 0.25)
        )
        t = torch.tensor([[sigma]])
        condition(net, x, t, ctx, cond, step, steps)
        pred = dit(x, t, ctx)
        loss = (pred - target).square().mean()
        extra = {}
        if (
            hasattr(net, "_contrastive_target_weight")
            and net._contrastive_target_weight > 0
        ):
            negatives = []
            for i in range(net.contrastive_k):
                neg_ctx = ctx.clone()
                neg_ctx[:, :2] = torch.randn_like(neg_ctx[:, :2])
                condition(net, x, t, neg_ctx, cond, step, steps, branch=1)
                negatives.append(dit(x, t, neg_ctx).squeeze(2))
            con, diag = net.contrastive_loss(
                pred.squeeze(2), torch.stack(negatives, dim=1), target.squeeze(2)
            )
            loss = loss + net._contrastive_target_weight * con
            extra = {
                "contrastive_loss": float(con.detach()),
                "negative_forwards": len(negatives),
            }
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        grads = {
            n: float(p.grad.detach().norm())
            for n, p in named.items()
            if p.grad is not None
        }
        assert grads and all(
            torch.isfinite(p.grad).all() for p in named.values() if p.grad is not None
        )
        grad_names.update(n for n, v in grads.items() if v > 0)
        total = sum(v * v for v in grads.values()) ** 0.5
        assert total > 0, "all-zero gradients"
        opt.step()
        history.append(
            {
                "step": step + 1,
                "sigma": sigma,
                "loss": float(loss.detach()),
                "grad_norm": total,
                "nonzero_grad_tensors": sum(v > 0 for v in grads.values()),
                **extra,
            }
        )
        if hasattr(net, "clear_step_caches"):
            net.clear_step_caches()
    changed = {
        n: float((p.detach() - before[n]).abs().max())
        for n, p in named.items()
        if not torch.equal(before[n], p.detach())
    }
    assert changed, "optimizer did not change parameters"
    case_dir = Path(output_dir) / variant.replace("/", "__")
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "adapter.safetensors"
    net.save_weights(
        str(path), torch.float32, metadata={"smoke": "tiny-real-anima-cpu"}
    )
    # Also preserve exact training state (including optimizer) for real resume.
    torch.save(
        {"adapter": net.state_dict(), "optimizer": opt.state_dict(), "step": steps},
        case_dir / "training-state.pt",
    )
    net.eval()
    if hasattr(net, "clear_timestep_mask"):
        net.clear_timestep_mask()
    t = torch.tensor([[0.4]])
    with torch.no_grad():
        condition(net, x, t, ctx, cond, steps, steps)
        expected = dit(x, t, ctx)
        if "controlnet" in c.get("network_module", ""):
            condition(net, x, t, ctx, cond + 1, steps, steps)
            cond_delta = float((dit(x, t, ctx) - expected).abs().max())
        else:
            cond_delta = None
    dit2 = tiny_anima(context_dim, blocks)
    dit2.load_state_dict(base_state)
    dit2.requires_grad_(False)
    sibling = list(case_dir.glob("*_moe.safetensors")) + list(
        case_dir.glob("*_chimera.safetensors")
    )
    reload_path = sibling[0] if sibling else path
    fresh, sd = module.create_network_from_weights(
        1.0, str(reload_path), None, [], dit2, for_inference=False
    )
    fresh.apply_to([], dit2, False, True)
    # Factory normalizes fused projection and per-expert on-disk layouts.
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    fresh.eval()
    if convrot:
        assert maybe_apply_convrot_base(args, fresh, unet=dit2, network_kwargs=kw)
    if hasattr(fresh, "clear_timestep_mask"):
        fresh.clear_timestep_mask()
    with torch.no_grad():
        condition(fresh, x, t, ctx, cond, steps, steps)
        actual = dit2(x, t, ctx)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    state = torch.load(case_dir / "training-state.pt", weights_only=True)
    assert state["step"] == steps and state["optimizer"]["state"]
    restored_optimizer = torch.optim.AdamW(list(named.values()), lr=1e-3)
    restored_optimizer.load_state_dict(state["optimizer"])
    assert len(restored_optimizer.state) == len(state["optimizer"]["state"])
    result = {
        "variant": variant,
        "resolved_variant": _resolve_gui_variant(variant),
        "status": "passed",
        "module": c.get("network_module", "networks.lora_anima"),
        "spec": getattr(
            getattr(net, "_network_spec", None),
            "name",
            getattr(net, "network_spec", None),
        ),
        "adapter_classes": dict(
            Counter(type(m).__name__ for m in getattr(net, "unet_loras", []))
        ),
        "model_parameters": sum(v.numel() for v in base_state.values()),
        "trainable_parameters": sum(p.numel() for p in named.values()),
        "minimal": {
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "blocks": blocks,
            "width": 32,
            "heads": 2,
            "latent_shape": list(x.shape),
            "rank_or_tokens": rank,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "steps": steps,
            "network_kwargs": kw,
        },
        "history": history,
        "nonzero_grad_parameter_names": sorted(grad_names),
        "changed_parameters": changed,
        "checkpoint": str(reload_path),
        "checkpoint_output_max_abs_error": float((actual - expected).abs().max()),
        "optimizer_state_restored": len(restored_optimizer.state),
        "convrot": convrot,
        "condition_output_delta": cond_delta,
    }
    if cond_delta is not None and cond_delta == 0:
        result["status"] = "failed"
        result["error"] = (
            "ControlNet output remains independent of condition after optimization; zero input and zero output projections form dead weight chain."
        )
    return result


def main():
    import argparse
    from bench._common import make_run_dir, write_result

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = make_run_dir("anima_minimal_training", root=args.output_root)
    variants = args.variants or [
        p.stem for p in sorted((REPO_ROOT / "configs/gui-methods").glob("*.toml"))
    ]
    results = []
    for variant in variants:
        start = time.monotonic()
        try:
            result = run(variant, output)
        except Exception as exc:
            result = {
                "variant": variant,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        result["seconds"] = round(time.monotonic() - start, 3)
        results.append(result)
        print(
            "RESULT",
            json.dumps(
                {
                    k: result[k]
                    for k in ("variant", "status", "seconds", "error")
                    if k in result
                }
            ),
            flush=True,
        )
    (output / "cases.json").write_text(json.dumps(results, indent=2))
    write_result(
        output,
        script=__file__,
        args=args,
        metrics={"cases": results},
        artifacts=["cases.json"],
        device="cpu",
    )
    failed = sum(result["status"] != "passed" for result in results)
    print(f"{len(results) - failed}/{len(results)} passed; artifacts: {output}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
