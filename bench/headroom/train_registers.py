"""Phase-0.5 RQ3-first training: does a frozen Anima DiT *adopt* register tokens
on a small adapter budget, and does the self-attn sink relocate off the image
patches onto the registers?

Self-contained FM train loop (user chose bench-loop over train.py integration):
frozen DiT + `RegisterAdapter`, eager native-flatten (no compile), Adam on the
adapter's trainable surface only. Faithful to the repo FM recipe
(`library/runtime/noise.py`): shift-logit-normal σ, noisy=(1-σ)·x+σ·ε,
timesteps=σ, rectified-flow target ε-x, unweighted MSE.

The training-time signal is the **relocation crossover** (bench README §RQ3):
register-token residual ‖x‖ RISING while image-patch ‖x‖ falls. Soft-tokens
already proved a frozen DiT adopts trained non-decoded tokens, so the crossover
should appear on a few-hundred-step budget — that's the thing this run checks.
The proper sink-token analysis + the RQ2 proxy gates live in the eval script.

Run:
  uv run python bench/headroom/train_registers.py --arm B --max_steps 400 \
      --max_images 400 --label rq3_armB
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima_lora import GenerationRequest  # noqa: E402
from bench._anima import add_common_args, add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.headroom.register_adapter import RegisterAdapter  # noqa: E402
from library.io.cache import load_cached_latents, load_cached_text_features  # noqa: E402
from library.runtime.noise import fm_training_batch  # noqa: E402
from library.inference.models import load_dit_model  # noqa: E402

_LORA_CACHE = "post_image_dataset/lora"


# --------------------------------------------------------------------------
# minimal cached dataset (pair latent npz + te safetensors by stem)
# --------------------------------------------------------------------------
def _latent_stem(p: str) -> str:  # {stem}_{WxH}_anima.npz
    return re.sub(r"_\d+x\d+_anima\.npz$", "", os.path.basename(p))


def _te_stem(p: str) -> str:
    return os.path.basename(p).replace("_anima_te.safetensors", "")


def _npz_tokens(npz_path: str) -> int:
    """Patch-token count from the ``{stem}_{W}x{H}_anima.npz`` filename
    (VAE /8, patch /2 → tokens = (W/16)·(H/16)). Returns a big number if
    unparseable so the caller's cap drops it."""
    m = re.search(r"_(\d+)x(\d+)_anima\.npz$", os.path.basename(npz_path))
    if not m:
        return 1 << 30
    w, h = int(m.group(1)), int(m.group(2))
    return (w // 16) * (h // 16)


def build_pairs(
    root: str, max_images: int, seed: int, max_tokens: int
) -> list[tuple[str, str]]:
    npzs = glob.glob(os.path.join(root, "**", "*_anima.npz"), recursive=True)
    tes = {
        _te_stem(p): p
        for p in glob.glob(os.path.join(root, "**", "*_anima_te.safetensors"), recursive=True)
    }
    pairs = []
    for np_path in npzs:
        te = tes.get(_latent_stem(np_path))
        if te is not None and _npz_tokens(np_path) <= max_tokens:
            pairs.append((np_path, te))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:max_images] if max_images > 0 else pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    add_model_args(p)
    p.add_argument("--arm", choices=["A", "B"], default="B")
    p.add_argument("--num_registers", type=int, default=16)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument(
        "--qkv_mode",
        choices=["lora", "unfrozen"],
        default="lora",
        help="'lora' = low-rank QKV adapter (RQ3 budget); 'unfrozen' = full-rank "
        "trainable ΔW on qkv_proj (proposal K36-b8-QKV, max reachability).",
    )
    p.add_argument(
        "--register_lr",
        type=float,
        default=None,
        help="Separate (usually higher, ~1e-2) lr for the register embeddings; "
        "defaults to --lr. The registers must grow ~14-24x the median patch norm "
        "to become sinks, which a shared LoRA-scale lr rarely reaches.",
    )
    p.add_argument(
        "--target_blocks",
        default="all",
        help="'all' or comma list of block indices for the QKV-LoRA (DSR sweet spot ~mid).",
    )
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--max_images", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--data_root", default=_LORA_CACHE)
    p.add_argument(
        "--max_latent_tokens", type=int, default=4400,
        help="Skip cached latents above this patch-token count (keeps the ~1024 "
        "tier and below so grad-ckpt backward fits a 16GB card).",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # DiT (frozen) — reuse the inference loader; a bench never trains the base.
    req = GenerationRequest(
        prompt="x", dit=args.dit, vae=args.vae, text_encoder=args.text_encoder,
        image_size=(1024, 1024), attn_mode=args.attn_mode, device=args.device, seed=0,
    )
    gen_args = req.to_args()
    gen_args.compile_blocks = False
    anima = load_dit_model(gen_args, device, torch.bfloat16)
    anima.requires_grad_(False)
    # Grad checkpointing: backward through 28 blocks OOMs a 16GB card otherwise.
    # Default path = torch_checkpoint(use_reentrant=False) — the safe variant for
    # our closed-over QKV-LoRA params (avoids the unsloth reentrant grad-drop bug).
    # The block gates on `self.training`, so put the frozen DiT in train mode
    # (dropout is Identity, no BN — mode flip is numerically inert here).
    anima.enable_gradient_checkpointing(unsloth_offload=False)
    anima.train()

    tb = (
        list(range(len(anima.blocks)))
        if args.target_blocks == "all"
        else [int(x) for x in args.target_blocks.split(",")]
    )
    adapter = RegisterAdapter(
        anima,
        num_registers=args.num_registers,
        arm=args.arm,
        lora_rank=args.lora_rank,
        target_blocks=tb,
        qkv_mode=args.qkv_mode,
    ).to(device=device, dtype=torch.bfloat16)
    adapter.apply_to()
    # keep adapter params in fp32 for stable Adam
    for prm in adapter.trainable_parameters():
        prm.data = prm.data.float()
    reg_lr = args.register_lr if args.register_lr is not None else args.lr
    if args.register_lr is not None and adapter.register_parameters():
        groups = [
            {"params": adapter.qkv_parameters(), "lr": args.lr},
            {"params": adapter.register_parameters(), "lr": reg_lr},
        ]
        opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(
            adapter.trainable_parameters(), lr=args.lr, weight_decay=0.0
        )

    pairs = build_pairs(
        args.data_root, args.max_images, args.seed, args.max_latent_tokens
    )
    if not pairs:
        raise SystemExit(f"no cached latent+te pairs under {args.data_root}")
    print(
        f"arm={args.arm} qkv_mode={args.qkv_mode} K={args.num_registers} "
        f"rank={args.lora_rank} blocks={len(tb)} lr={args.lr} reg_lr={reg_lr} "
        f"trainable={adapter.num_trainable():,} pairs={len(pairs)}"
    )

    out_dir = make_run_dir("headroom", args.label or f"train_arm{args.arm}")
    history = []
    step = 0
    idx = 0
    while step < args.max_steps:
        npz_path, te_path = pairs[idx % len(pairs)]
        idx += 1
        latents = load_cached_latents(npz_path)[0].unsqueeze(0).to(device)  # (1,16,H,W)
        crossattn, _ = load_cached_text_features(te_path, variant=0)
        if crossattn is None:
            continue
        crossattn = crossattn.unsqueeze(0).to(device=device, dtype=torch.bfloat16)  # (1,512,1024)

        noise = torch.randn_like(latents)
        # Default rectified-flow step (timestep_sampling="shift"); issues.md DX2.
        noisy, sigma, target = fm_training_batch(
            latents, noise, timestep_sampling="shift", discrete_flow_shift=args.flow_shift
        )

        noisy_5d = noisy.unsqueeze(2).to(torch.bfloat16)  # (1,16,1,H,W)
        padding_mask = torch.zeros(
            latents.shape[0], 1, latents.shape[-2], latents.shape[-1],
            dtype=torch.bfloat16, device=device,
        )
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # else-branch: cached post-adapter context used as-is (no target_input_ids)
                pred = anima(noisy_5d, sigma, crossattn, padding_mask=padding_mask)
            pred = pred.squeeze(2).float()
            loss = torch.nn.functional.mse_loss(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.trainable_parameters(), 1.0)
            opt.step()
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            del latents, crossattn, noise, noisy, noisy_5d
            torch.cuda.empty_cache()
            print(f"  OOM on {os.path.basename(npz_path)} — skipped")
            continue

        if step % args.log_every == 0 or step == args.max_steps - 1:
            rn, pn = adapter.last_reg_norm, adapter.last_patch_norm
            ratio = rn / pn if pn else float("nan")
            psr = adapter.last_patch_sink_ratio  # topk-patch / median (the sink)
            rr = adapter.last_reg_ratio  # max register / median patch (adoption)
            row = {
                "step": step,
                "loss": round(loss.item(), 5),
                "reg_norm": round(rn, 2),
                "patch_norm": round(pn, 2),
                "reg_over_patch": round(ratio, 4),
                "patch_sink_ratio": round(psr, 3) if psr is not None else None,
                "reg_ratio": round(rr, 3) if rr is not None else None,
            }
            history.append(row)
            print(
                f"[{step:4d}] loss={row['loss']:.5f} reg_norm={rn:.1f} "
                f"patch_norm={pn:.1f} reg/patch={ratio:.4f} "
                f"sink={row['patch_sink_ratio']} reg_ratio={row['reg_ratio']}"
            )
        step += 1

    # save adapter (fp32 trainable tensors) + config. safetensors is the
    # canonical/ComfyUI-loadable artifact (config JSON-encoded into the header
    # metadata, which only takes str values); .pt kept for back-compat with the
    # bench eval consumers (register_eval.py).
    ckpt = {"config": adapter.config(), "arm": args.arm}
    sd = {
        k: v.detach().cpu().to(torch.float32).contiguous()
        for k, v in adapter.state_dict().items()
    }
    torch.save({"state_dict": sd, "meta": ckpt}, out_dir / "adapter.pt")
    from safetensors.torch import save_file

    save_file(
        sd,
        str(out_dir / "adapter.safetensors"),
        metadata={
            "format": "anima-register-v1",
            "anima_register_config": json.dumps(adapter.config()),
        },
    )

    crossover = None
    if len(history) >= 2:
        crossover = {
            "reg_over_patch_start": history[0]["reg_over_patch"],
            "reg_over_patch_end": history[-1]["reg_over_patch"],
            "delta": round(history[-1]["reg_over_patch"] - history[0]["reg_over_patch"], 4),
            "loss_start": history[0]["loss"],
            "loss_end": history[-1]["loss"],
        }
    write_result(
        out_dir,
        script=__file__,
        args=args,
        metrics={
            "arm": args.arm,
            "config": adapter.config(),
            "max_steps": args.max_steps,
            "n_pairs": len(pairs),
            "lr": args.lr,
            "history": history,
            "crossover": crossover,
        },
        label=args.label,
    )
    print("saved:", out_dir)
    if crossover:
        print(
            f"CROSSOVER reg/patch {crossover['reg_over_patch_start']:.4f} "
            f"-> {crossover['reg_over_patch_end']:.4f} (Δ {crossover['delta']:+.4f}); "
            f"loss {crossover['loss_start']:.4f} -> {crossover['loss_end']:.4f}"
        )


if __name__ == "__main__":
    main()
