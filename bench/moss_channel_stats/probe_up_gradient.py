#!/usr/bin/env python3
"""Where does the LoRA ``up`` gradient come from, per input channel? — the GraLoRA question.

Measures GraLoRA's (arXiv:2505.20355 § 2.2) decomposition
``∂L/∂up = Σ_c (∂L/∂W)[:, c] ⊗ down[:, c]`` on MOSS, with the trainer's own ``down``
init (``weight_svd``, rank 32) and loss, one backward per (sample, σ):

  * ``share[c] = ⟨(∂L/∂W)[:, c] ⊗ down[:, c], ∂L/∂up⟩ / ‖∂L/∂up‖²``, channel c's
    projection share of the up gradient (sums to 1 over c)
  * the same share in the FFT gradient, ``‖(∂L/∂W)[:, c]‖² / ‖∂L/∂W‖²``
  * rank-1 fraction of ``∂L/∂up`` (``σ₁² / Σσ²``)
  * DC share: the projection share of ``(Σ_t g_t) ⊗ (down · mean_t x)`` and the same
    share of ``(Σ_t g_t) ⊗ mean_t x`` in ``∂L/∂W`` (the FFT reference); plus the
    rank-1 fraction of ``∂L/∂W`` (power iteration) next to ``∂L/∂up``'s

Per step, per (block, target); reported as medians over steps. ``ffn_up_gate`` and
``q`` are residual-indexed, so ch 42 / 1356 can be read off directly. Eager +
per-block checkpointing (the trainer's ``--grad_checkpointing`` path,
early-stop off so every Linear's forward hook fires again in the recompute); the
Linear inputs are stashed only during the recompute, so one block lives at a time
(see ``Probe`` for the hook ordering).

    make daemon-run ARGS="bench/moss_channel_stats/probe_up_gradient.py --label base"
    make daemon-run ARGS="bench/moss_channel_stats/probe_up_gradient.py --dc_isolate --label dc"

``--dc_isolate`` builds the network with the DC-isolated down init (row 0 = the
calibrated ``x̄`` direction, rows 1.. ⊥ it); the ``*_rest`` metrics (DC share and rank-1
fraction of ``∂L/∂up[:, 1:]``) measure it. Writes ``stats.npz``,
``up_gradient.png`` and ``result.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.checkpoint import set_checkpoint_early_stop  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.moss import cache as moss_cache  # noqa: E402
from library.moss.dave import DEFAULT_CHANNELS_PATH  # noqa: E402
from library.moss.loader import empty_cache, load_transformer  # noqa: E402
from library.moss.lora import LoRANetwork, load_channel_stats  # noqa: E402
from library.moss.requests import LORA_ALPHA  # noqa: E402
from library.moss.train import (  # noqa: E402
    checkpoint_blocks,
    place,
    step_reserve_gb,
    training_loss,
)

HOOKED = ("q", "o", "ffn_up_gate", "ffn_down")
RESIDUAL_INPUTS = ("q", "ffn_up_gate")
TOPK = 8


class Probe:
    """Hooks on one Linear. Non-reentrant checkpointing drives backward through
    the pass-1 graph and only swaps in recomputed saved tensors, so the output
    gradient hook goes on the pass-1 output and ``x`` is taken from the recompute
    (pass 2). Either can arrive first inside a block (the last Linear's output
    grad lands before the block's first unpack triggers the recompute), so the
    decomposition runs when the second of the two shows up."""

    def __init__(self, key, lin, down):
        self.key = key
        self.down = down  # (r, N) fp32, the adapter's down weight (a view)
        self.calls = 0
        self.x = None
        self.g = None
        self.out = None  # filled once per step
        self.last_U = None
        lin.register_forward_pre_hook(self.pre)
        lin.register_forward_hook(self.post)

    def reset(self):
        self.calls = 0
        self.x = self.g = self.out = None

    def pre(self, _m, inp):
        self.calls += 1
        if self.calls == 2:  # the recompute pass
            self.x = inp[0].detach()
            if self.g is not None:
                self.compute()

    def post(self, _m, _inp, out):
        if self.calls == 1 and out.requires_grad:
            out.register_hook(self.grad)

    def grad(self, g):
        self.g = g.detach()
        if self.x is not None:
            self.compute()

    @torch.no_grad()
    def compute(self):
        if self.out is not None:
            raise RuntimeError(f"{self.key}: decomposed twice in one step")
        x = self.x[0].float()  # (T, N)
        g = self.g[0].float()  # (T, M)
        self.x = self.g = None
        down = self.down.float()  # (r, N)
        w = g.T @ x  # (M, N)  the full-fine-tune gradient ∂L/∂W
        U = w @ down.T  # (M, r)  ∂L/∂up up to the α/r scale
        u2 = (U * U).sum()
        share = (down * (U.T @ w)).sum(0) / u2  # (N,)
        w2 = (w * w).sum(0)
        fft = w2 / w2.sum()
        g_sum, x_mean = g.sum(0), x.mean(0)
        dc = torch.outer(g_sum, down @ x_mean)
        dc_share = (dc * U).sum() / u2
        w_dc = torch.outer(g_sum, x_mean)
        fft_dc_share = (w_dc * w).sum() / w2.sum()
        sv = torch.linalg.svdvals(U)
        U_rest, dc_rest = U[:, 1:], dc[:, 1:]
        u2_rest = (U_rest * U_rest).sum()
        sv_rest = torch.linalg.svdvals(U_rest)
        self.last_U = U
        self.out = {
            "share": share,
            "fft": fft,
            "dc_share": float(dc_share),
            "fft_dc_share": float(fft_dc_share),
            "rank1": float(sv[0] ** 2 / (sv**2).sum()),
            "fft_rank1": float(_top_sv2(w) / w2.sum()),
            "u_norm": float(u2.sqrt()),
            "dc_share_rest": float((dc_rest * U_rest).sum() / u2_rest),
            "rank1_rest": float(sv_rest[0] ** 2 / (sv_rest**2).sum()),
            "col0_frac": float((U[:, 0] ** 2).sum() / u2),
        }


def _top_sv2(w: torch.Tensor, iters: int = 12) -> torch.Tensor:
    """σ₁² of ``w`` by power iteration on ``wᵀw`` (an svd of 3072×8192 per Linear
    per step is too slow; 12 iterations resolve a dominant direction well)."""
    v = w.sum(0)
    v = v / v.norm().clamp_min(1e-12)
    for _ in range(iters):
        v = w.T @ (w @ v)
        v = v / v.norm().clamp_min(1e-12)
    return (w @ v).square().sum()


def plot(path, arrays, massive_idx, num_blocks):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    xs = np.arange(num_blocks)
    i42 = int(np.where(massive_idx == 42)[0][0])
    i1356 = int(np.where(massive_idx == 1356)[0][0])
    for name in HOOKED:
        med = lambda k: np.median(arrays[f"{name}_{k}"], axis=0)  # noqa: E731
        axes[0, 0].plot(xs, med("top1_share"), label=f"{name} up-grad")
        axes[0, 0].plot(xs, med("fft_top1_share"), ":", label=f"{name} FFT")
        axes[0, 1].plot(xs, med("rank1"), label=f"{name} up-grad")
        axes[0, 1].plot(xs, med("rank1_rest"), "--", label=f"{name} cols 1..")
        axes[0, 1].plot(xs, med("fft_rank1"), ":", label=f"{name} FFT")
        axes[1, 1].plot(xs, med("dc_share"), label=f"{name} up-grad")
        axes[1, 1].plot(xs, med("dc_share_rest"), "--", label=f"{name} cols 1..")
        axes[1, 1].plot(xs, med("fft_dc_share"), ":", label=f"{name} FFT")
        if name in RESIDUAL_INPUTS:
            axes[1, 0].plot(xs, med("massive_share")[:, i42], label=f"{name} ch42")
            axes[1, 0].plot(
                xs, med("massive_share")[:, i1356], "--", label=f"{name} ch1356"
            )
            axes[1, 0].plot(
                xs, med("fft_massive_share")[:, i42], ":", label=f"{name} ch42 FFT"
            )
    axes[0, 0].set(title="top-1 channel share of the gradient", yscale="log")
    axes[0, 1].set(title="rank-1 fraction (σ₁²/Σσ²): ∂L/∂up vs ∂L/∂W", ylim=(0, 1))
    axes[1, 0].set(title="massive-channel share of ∂L/∂up", yscale="log")
    axes[1, 1].set(title="DC (bias-learner) share: ∂L/∂up vs ∂L/∂W")
    for ax in axes.flat:
        ax.set_xlabel("block")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default="post_image_dataset/moss/_calib_mixed")
    ap.add_argument("--n", type=int, default=16, help="samples, evenly spaced by stem")
    ap.add_argument("--sigmas", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--model", default="base", help="base | sft")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--down_init", default="weight_svd", help="weight_svd | kaiming")
    ap.add_argument("--dc_isolate", action="store_true")
    ap.add_argument(
        "--channel_mean", default="library/moss/calibration/channel_mean.safetensors"
    )
    ap.add_argument(
        "--blocks_to_swap", type=int, default=-1, help="-1 = size from free VRAM"
    )
    ap.add_argument("--massive", default=DEFAULT_CHANNELS_PATH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    sigmas = [float(s) for s in args.sigmas.split(",")]
    items = moss_cache.list_items(resolve_under_home(args.cache))
    if not items:
        raise SystemExit(f"no cached pairs under {args.cache}")
    pick = np.linspace(0, len(items) - 1, min(args.n, len(items))).round().astype(int)
    items = [items[i] for i in sorted(set(pick))]
    massive = np.load(resolve_under_home(args.massive))["channels"].astype(np.int64)
    print(
        f"{len(items)} samples × {len(sigmas)} σ from {args.cache}, model {args.model}, "
        f"rank {args.rank} {args.down_init}{' dc_isolate' if args.dc_isolate else ''}",
        flush=True,
    )

    transformer = load_transformer(args.model_dir, args.model)
    transformer.requires_grad_(False)
    network = LoRANetwork(
        transformer,
        rank=args.rank,
        alpha=LORA_ALPHA,
        dtype=torch.float32,
        down_init=args.down_init,
        channel_mean=load_channel_stats(resolve_under_home(args.channel_mean))
        if args.dc_isolate
        else None,
        dc_isolate=args.dc_isolate,
    )
    network.apply_to()
    network.to(device)
    num_blocks = len(transformer.layers)
    probes: dict[tuple[int, str], Probe] = {}
    for i, layer in enumerate(transformer.layers):
        for name in HOOKED:
            adapter = network.adapters[f"layers_{i}_proj_{name}"]
            probes[(i, name)] = Probe((i, name), layer.proj[name], adapter.down.weight)

    joint = max((it.width // 16) * (it.height // 16) + it.n_text for it in items)
    reserve = step_reserve_gb(joint, grad_checkpointing=True) + 1.5  # + probe scratch
    attached = place(transformer, device, args.blocks_to_swap, reserve)
    checkpoint_blocks(transformer.layers)
    empty_cache()
    start_heartbeat()

    gen = torch.Generator(device=device)
    steps = len(items) * len(sigmas)
    dims = {name: transformer.layers[0].proj[name].in_features for name in HOOKED}
    arrays = {}
    for name in HOOKED:
        z = lambda *s: np.zeros((steps, num_blocks, *s), dtype=np.float32)  # noqa: E731
        arrays[f"{name}_top1_share"] = z()
        arrays[f"{name}_top1_idx"] = z()
        arrays[f"{name}_topk_share"] = z()
        arrays[f"{name}_eff_channels"] = z()  # 1 / Σ share²
        arrays[f"{name}_fft_top1_share"] = z()
        arrays[f"{name}_fft_top1_idx"] = z()
        arrays[f"{name}_fft_eff_channels"] = z()
        arrays[f"{name}_rank1"] = z()
        arrays[f"{name}_dc_share"] = z()
        arrays[f"{name}_fft_dc_share"] = z()
        arrays[f"{name}_fft_rank1"] = z()
        arrays[f"{name}_dc_share_rest"] = z()
        arrays[f"{name}_rank1_rest"] = z()
        arrays[f"{name}_col0_frac"] = z()
        arrays[f"{name}_u_norm"] = z()
        arrays[f"{name}_massive_share"] = z(massive.size)
        arrays[f"{name}_fft_massive_share"] = z(massive.size)
        arrays[f"{name}_mean_share"] = np.zeros((num_blocks, dims[name]), np.float32)
    massive_t = torch.from_numpy(massive).to(device)

    t0 = time.time()
    step = 0
    validated = None
    for item in items:
        latent, embeds = moss_cache.load_item(item)
        latent = latent.unsqueeze(0).to(device, torch.float32)
        embeds = embeds.unsqueeze(0).to(device)
        for sigma in sigmas:
            for p in probes.values():
                p.reset()
            gen.manual_seed(args.seed * 100003 + step)
            with set_checkpoint_early_stop(False):
                loss = training_loss(transformer, latent, embeds, sigma, gen=gen)
                loss.backward()
            for (i, name), p in probes.items():
                if p.out is None:
                    raise RuntimeError(
                        f"no gradient captured for layers.{i}.proj.{name}"
                    )
                o = p.out
                share, fft = o["share"], o["fft"]
                top = torch.topk(share, TOPK)
                arrays[f"{name}_top1_share"][step, i] = top.values[0]
                arrays[f"{name}_top1_idx"][step, i] = top.indices[0]
                arrays[f"{name}_topk_share"][step, i] = top.values.sum()
                arrays[f"{name}_eff_channels"][step, i] = 1.0 / (share * share).sum()
                ftop = torch.topk(fft, 1)
                arrays[f"{name}_fft_top1_share"][step, i] = ftop.values[0]
                arrays[f"{name}_fft_top1_idx"][step, i] = ftop.indices[0]
                arrays[f"{name}_fft_eff_channels"][step, i] = 1.0 / (fft * fft).sum()
                arrays[f"{name}_rank1"][step, i] = o["rank1"]
                arrays[f"{name}_dc_share"][step, i] = o["dc_share"]
                arrays[f"{name}_fft_dc_share"][step, i] = o["fft_dc_share"]
                arrays[f"{name}_fft_rank1"][step, i] = o["fft_rank1"]
                arrays[f"{name}_dc_share_rest"][step, i] = o["dc_share_rest"]
                arrays[f"{name}_rank1_rest"][step, i] = o["rank1_rest"]
                arrays[f"{name}_col0_frac"][step, i] = o["col0_frac"]
                arrays[f"{name}_u_norm"][step, i] = o["u_norm"]
                if dims[name] > int(massive.max()):
                    arrays[f"{name}_massive_share"][step, i] = share[massive_t].cpu()
                    arrays[f"{name}_fft_massive_share"][step, i] = fft[massive_t].cpu()
                arrays[f"{name}_mean_share"][i] += share.cpu().numpy() / steps
            if validated is None:
                # autograd's ∂L/∂up on one adapter == scale · U from the hook
                key = (num_blocks // 2, "ffn_up_gate")
                adapter = network.adapters[f"layers_{key[0]}_proj_ffn_up_gate"]
                ref = adapter.up.weight.grad.float()
                mine = probes[key].last_U * adapter.scale
                validated = float(
                    (ref - mine).abs().max() / ref.abs().max().clamp_min(1e-12)
                )
                print(
                    f"  check: hook U vs autograd up.grad, rel max err {validated:.2e}"
                )
            network.zero_grad(set_to_none=True)
            step += 1
            print(
                f"  progress {step}/{steps}  {item.stem} σ={sigma} loss {loss.item():.4f}  "
                f"{(time.time() - t0) / step:.1f}s/step  "
                f"peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB",
                flush=True,
            )
    if attached is not None:
        attached.detach()

    run_dir = make_run_dir("moss_channel_stats", args.label)
    np.savez(run_dir / "stats.npz", massive=massive, **arrays)
    plot(run_dir / "up_gradient.png", arrays, massive, num_blocks)

    i42 = int(np.where(massive == 42)[0][0])
    i1356 = int(np.where(massive == 1356)[0][0])
    metrics = {"hook_vs_autograd_rel_err": validated}
    for name in HOOKED:
        med = lambda k: np.median(arrays[f"{name}_{k}"], axis=0)  # noqa: E731
        n = dims[name]
        row = {
            "channels": n,
            "top1_share_median_over_blocks": round(
                float(np.median(med("top1_share"))), 4
            ),
            "top1_share_max_block": round(float(med("top1_share").max()), 4),
            "fft_top1_share_median_over_blocks": round(
                float(np.median(med("fft_top1_share"))), 4
            ),
            "topk_share_median": round(float(np.median(med("topk_share"))), 4),
            "eff_channels_median": round(float(np.median(med("eff_channels"))), 1),
            "fft_eff_channels_median": round(
                float(np.median(med("fft_eff_channels"))), 1
            ),
            "rank1_median": round(float(np.median(med("rank1"))), 3),
            "rank1_max_block": round(float(med("rank1").max()), 3),
            "dc_share_median": round(float(np.median(med("dc_share"))), 4),
            "dc_share_max_block": round(float(med("dc_share").max()), 4),
            "fft_dc_share_median": round(float(np.median(med("fft_dc_share"))), 4),
            "fft_dc_share_max_block": round(float(med("fft_dc_share").max()), 4),
            "fft_rank1_median": round(float(np.median(med("fft_rank1"))), 3),
            "dc_share_rest_median": round(float(np.median(med("dc_share_rest"))), 4),
            "dc_share_rest_max_block": round(float(med("dc_share_rest").max()), 4),
            "rank1_rest_median": round(float(np.median(med("rank1_rest"))), 3),
            "col0_frac_median": round(float(np.median(med("col0_frac"))), 3),
            "uniform_share": round(1.0 / n, 6),
        }
        if name in RESIDUAL_INPUTS:
            ms = med("massive_share")
            fs = med("fft_massive_share")
            row["ch42_share_range"] = [
                round(float(ms[:, i42].min()), 4),
                round(float(ms[:, i42].max()), 4),
            ]
            row["ch42_fft_share_range"] = [
                round(float(fs[:, i42].min()), 4),
                round(float(fs[:, i42].max()), 4),
            ]
            row["ch1356_share_range"] = [
                round(float(ms[:, i1356].min()), 4),
                round(float(ms[:, i1356].max()), 4),
            ]
            top_idx = np.median(arrays[f"{name}_top1_idx"], axis=0).astype(int)
            row["blocks_top1_is_massive"] = int(np.isin(top_idx, massive).sum())
            row["blocks_top1_is_42"] = int((top_idx == 42).sum())
        metrics[name] = row
        print(name, row, flush=True)
    for name in RESIDUAL_INPUTS:
        ms = np.median(arrays[f"{name}_massive_share"], axis=0)[:, i42]
        r1 = np.median(arrays[f"{name}_rank1"], axis=0)
        dc = np.median(arrays[f"{name}_dc_share"], axis=0)
        fdc = np.median(arrays[f"{name}_fft_dc_share"], axis=0)
        dcr = np.median(arrays[f"{name}_dc_share_rest"], axis=0)
        print(
            f"{name} per block — ch42 share {np.round(ms, 3).tolist()}\n"
            f"    rank1 {np.round(r1, 2).tolist()}\n    dc share {np.round(dc, 3).tolist()}\n"
            f"    dc share cols 1.. {np.round(dcr, 3).tolist()}\n"
            f"    FFT dc share {np.round(fdc, 3).tolist()}",
            flush=True,
        )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["stats.npz", "up_gradient.png"],
        device=device,
        extra={"samples": [it.stem for it in items], "steps": steps},
    )
    print(f"wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
