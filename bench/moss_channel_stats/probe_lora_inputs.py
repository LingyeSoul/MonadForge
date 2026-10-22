#!/usr/bin/env python3
"""Per-channel input magnitude at MOSS-Image's LoRA targets — the channel-scaling question.

Measures what arrives at the Linear inputs after RMSNorm + adaLN (findings in
``project/moss_lora/done/channel_scaling.md``). Training-path forward (no grad):
cached latents noised at each σ in ``--sigmas``,
cond embeds at true length, the reference per-sample forward. A ``forward_pre_hook``
on ``layers.N.proj.{q,o,ffn_up_gate,ffn_down}`` (k / v share q's input) accumulates
per input channel, over all tokens (what the LoRA gradient sees) and over the image
tokens alone: ``sum|x|``, ``sum x`` and ``max|x|``.

Per (block, target):

  * dominance ``max / median`` of ``mean|x|`` (Anima's E-analysis number)
  * leverage dominance ``max / mean`` of ``mean|x|^(1−α)`` at α = 0 and 0.5
  * peak-to-mean ``max|x| / mean|x|`` on the top channel (≈1–3 = DC bias)
  * for the residual-indexed inputs (q, ffn_up_gate), the rank, ``mean|x|/median``
    and signed image mean of every massive channel

    make daemon-run ARGS="bench/moss_channel_stats/probe_lora_inputs.py"
    make daemon-run ARGS="bench/moss_channel_stats/probe_lora_inputs.py --model sft --label sft"

Writes ``channel_stats.safetensors`` (``mean|x|`` per target, keyed
``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}``), ``channel_mean.safetensors`` (the
signed token mean ``x̄`` per target, same keys: the DC direction ``dc_isolate``
reads), ``stats.npz``, ``dominance.png`` and
``result.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.moss import cache as moss_cache  # noqa: E402
from library.moss.dave import DEFAULT_CHANNELS_PATH  # noqa: E402
from library.moss.loader import empty_cache, load_transformer  # noqa: E402
from library.runtime.blockswap import place  # noqa: E402

HOOKED = ("q", "o", "ffn_up_gate", "ffn_down")
SHARES_Q = ("k", "v")
RESIDUAL_INPUTS = ("q", "ffn_up_gate")  # channel index == residual channel index
ALPHAS = (0.0, 0.5)


class Stats:
    """Per-channel accumulators for one Linear's input, split all / image tokens."""

    def __init__(self, dim, device):
        z = lambda: torch.zeros(dim, device=device, dtype=torch.float64)  # noqa: E731
        self.abs_all, self.abs_img, self.sum_img, self.sum_all = z(), z(), z(), z()
        self.max_all = torch.zeros(dim, device=device)
        self.n_all = self.n_img = 0

    def add(self, x, n_img):
        x = x[0].float()  # (T, D)
        a = x.abs()
        self.abs_all += a.sum(0, dtype=torch.float64)
        self.abs_img += a[:n_img].sum(0, dtype=torch.float64)
        self.sum_all += x.sum(0, dtype=torch.float64)
        self.sum_img += x[:n_img].sum(0, dtype=torch.float64)
        self.max_all = torch.maximum(self.max_all, a.amax(0))
        self.n_all += x.shape[0]
        self.n_img += n_img


def leverage_dominance(m: np.ndarray, alpha: float) -> float:
    lev = np.clip(m, 1e-6, None) ** (1.0 - alpha)
    return float(lev.max() / lev.mean())


def plot(path, dom, lev, num_blocks):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.arange(num_blocks)
    for name in HOOKED:
        axes[0].plot(xs, dom[name], label=name)
        axes[1].plot(xs, lev[name][0.0], label=f"{name} α=0")
        axes[1].plot(xs, lev[name][0.5], "--", label=f"{name} α=0.5")
    axes[0].set(title="max / median mean|x|", xlabel="block", yscale="log")
    axes[1].set(title="leverage max / mean", xlabel="block", yscale="log")
    for ax in axes:
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
        f"{massive.size} massive channels",
        flush=True,
    )

    transformer = load_transformer(args.model_dir, args.model)
    transformer.requires_grad_(False).eval()
    num_blocks = len(transformer.layers)
    stats: dict[tuple[int, str], Stats] = {}
    current = {"n_img": 0}
    handles = []
    for i, layer in enumerate(transformer.layers):
        for name in HOOKED:
            lin = layer.proj[name]
            st = stats[(i, name)] = Stats(lin.in_features, device)

            def hook(_m, inp, st=st):
                st.add(inp[0], current["n_img"])

            handles.append(lin.register_forward_pre_hook(hook))

    attached = place(
        transformer,
        "layers",
        device,
        blocks_to_swap=None if args.blocks_to_swap < 0 else args.blocks_to_swap,
        activation_reserve_gb=3.0,
        label="transformer",
    )
    empty_cache()
    start_heartbeat()

    gen = torch.Generator(device=device)
    t0 = time.time()
    done, total = 0, len(items) * len(sigmas)
    with torch.no_grad():
        for item in items:
            latent, embeds = moss_cache.load_item(item)
            latent = latent.unsqueeze(0).to(device, torch.float32)
            embeds = embeds.unsqueeze(0).to(device)
            current["n_img"] = (item.width // 16) * (item.height // 16)
            for sigma in sigmas:
                gen.manual_seed(args.seed * 100003 + done)
                noise = torch.randn(latent.shape, generator=gen, device=device)
                noisy = (1.0 - sigma) * latent + sigma * noise
                timestep = torch.tensor([1000.0 * sigma], device=device)
                transformer(noisy, timestep, embeds, return_dict=False)
                done += 1
                print(
                    f"  progress {done}/{total}  {item.stem} σ={sigma}  "
                    f"{(time.time() - t0) / done:.1f}s/fwd",
                    flush=True,
                )
    for h in handles:
        h.remove()
    if attached is not None:
        attached.detach()

    run_dir = make_run_dir("moss_channel_stats", args.label)
    dump, dump_mean, dom, lev, p2m = {}, {}, {}, {}, {}
    arrays = {}
    massive_rows = {}
    for name in HOOKED:
        dom[name] = np.zeros(num_blocks)
        lev[name] = {a: np.zeros(num_blocks) for a in ALPHAS}
        p2m[name] = np.zeros(num_blocks)
        means = []
        for i in range(num_blocks):
            st = stats[(i, name)]
            m = (st.abs_all / st.n_all).float().cpu().numpy()
            m_img = (st.abs_img / st.n_img).float().cpu().numpy()
            mu_img = (st.sum_img / st.n_img).float().cpu().numpy()
            mx = st.max_all.cpu().numpy()
            means.append(m)
            med = float(np.median(m))
            top = int(m.argmax())
            dom[name][i] = m.max() / med
            for a in ALPHAS:
                lev[name][a][i] = leverage_dominance(m, a)
            p2m[name][i] = mx[top] / m[top]
            t = torch.from_numpy(m.copy())
            t_mean = (st.sum_all / st.n_all).float().cpu()
            dump[f"layers.{i}.proj.{name}"] = t
            dump_mean[f"layers.{i}.proj.{name}"] = t_mean
            if name == "q":
                for share in SHARES_Q:
                    dump[f"layers.{i}.proj.{share}"] = t.clone()
                    dump_mean[f"layers.{i}.proj.{share}"] = t_mean.clone()
            if name in RESIDUAL_INPUTS:
                rank = (-m).argsort().argsort()
                med_img = float(np.median(m_img))
                massive_rows.setdefault(name, []).append(
                    {
                        int(c): {
                            "rank": int(rank[c]),
                            "x_median": round(float(m[c] / med), 2),
                            "img_signed_x_median": round(float(mu_img[c] / med_img), 2),
                        }
                        for c in massive
                    }
                )
        arrays[f"{name}_mean_abs"] = np.stack(means)
        arrays[f"{name}_dominance"] = dom[name]
        arrays[f"{name}_p2m"] = p2m[name]
        for a in ALPHAS:
            arrays[f"{name}_leverage_a{a}"] = lev[name][a]

    from safetensors.torch import save_file

    save_file(dump, str(run_dir / "channel_stats.safetensors"))
    save_file(dump_mean, str(run_dir / "channel_mean.safetensors"))
    np.savez(run_dir / "stats.npz", massive=massive, **arrays)
    plot(run_dir / "dominance.png", dom, lev, num_blocks)

    metrics = {}
    for name in HOOKED:
        row = {
            "dominance_median": round(float(np.median(dom[name])), 1),
            "dominance_max": round(float(dom[name].max()), 1),
            "leverage_a0_range": [
                round(float(lev[name][0.0].min()), 1),
                round(float(lev[name][0.0].max()), 1),
            ],
            "leverage_a0.5_range": [
                round(float(lev[name][0.5].min()), 1),
                round(float(lev[name][0.5].max()), 1),
            ],
            "top_p2m_median": round(float(np.median(p2m[name])), 2),
        }
        if name in RESIDUAL_INPUTS:
            tops = arrays[f"{name}_mean_abs"].argmax(1)
            row["blocks_top1_massive"] = int(np.isin(tops, massive).sum())
            top4 = (-arrays[f"{name}_mean_abs"]).argsort(1)[:, :4]
            row["blocks_any_top4_massive"] = int(np.isin(top4, massive).any(1).sum())
        metrics[name] = row
        print(name, row, flush=True)
    for name in RESIDUAL_INPUTS:
        for c in (1356, 42):
            if c in massive:
                ranks = [r[c]["rank"] for r in massive_rows[name]]
                xm = [r[c]["x_median"] for r in massive_rows[name]]
                print(
                    f"{name} ch{c}: rank per block {ranks}\n    ×median {xm}",
                    flush=True,
                )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=[
            "channel_stats.safetensors",
            "channel_mean.safetensors",
            "stats.npz",
            "dominance.png",
        ],
        device=device,
        extra={"massive_channels": massive_rows, "samples": [it.stem for it in items]},
    )
    print(f"wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
