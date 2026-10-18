#!/usr/bin/env python3
"""2:4 activation-sparsity error on MOSS-Image's block linears — the RT-Lynx premise.

Tier 1 §2 of ``docs/proposal/moss_anima_negatives_revisit.md``. RT-Lynx
(arXiv:2605.26632) keeps the top-2-by-magnitude of every 4 contiguous input
channels of a GEMM (the contraction dim K), rescales each token row back to its
dense L2 norm (norm compensation, NC) and runs the matmul on Sparse Tensor Cores.
On Anima the plain-GELU MLP kept ``mlp_down`` dense: output RFE 0.25 against the
paper's 0.01 (``_archive/rt_lynx/``). MOSS's FFN is gated (SiTU-GLU), the setting the
paper's low error comes from.

Training-path forward (no grad), as ``bench/moss_channel_stats/probe_lora_inputs.py``:
cached latents noised at each σ in ``--sigmas``, cond embeds at true length, the
reference per-sample forward. A ``forward_pre_hook`` on every
``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`` recomputes the GEMM on the 2:4
input and leaves the live activation dense. Per (σ, block, target):

  * ``rfe`` — ``‖y − ŷ‖_F / ‖y‖_F`` per call (NC), averaged over calls. This is the
    archived Anima number (RT-Lynx Eq. 10)
  * the same, pooled over image rows and over text rows, with and without NC
  * input-side ``‖x − x̂‖ / ‖x‖`` (no NC): how much of the activation the mask drops

**Gate**: ``ffn_down`` RFE ≤ 0.05 on most blocks. Then a kernel-level timing before
any quality arm.

    make daemon-run ARGS="bench/moss_rfe/probe_rfe.py --label base"
    make daemon-run ARGS="bench/moss_rfe/probe_rfe.py --model sft --label sft"

Writes ``rfe.npz``, ``rfe.png`` and ``result.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.moss import cache as moss_cache  # noqa: E402
from library.moss.loader import empty_cache, load_transformer  # noqa: E402
from library.runtime.blockswap import place  # noqa: E402

TARGETS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
GATE_TARGET = "ffn_down"
GATE_RFE = 0.05
# accumulator slots, each a sum over calls
# per-call rfe (nc), then squared norms pooled over image / text rows
SLOTS = (
    "rfe_call",
    "y_img",
    "y_txt",
    "e_nc_img",
    "e_nc_txt",
    "e_raw_img",
    "e_raw_txt",
    "x_img",
    "x_txt",
    "dx_img",
    "dx_txt",
)


def sparsify_24(x: torch.Tensor) -> torch.Tensor:
    *lead, k = x.shape
    g = x.reshape(-1, k // 4, 4)
    idx = g.abs().topk(2, dim=-1).indices
    keep = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, True)
    return (g * keep).reshape(*lead, k)


def norm_comp(x: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    num = x.float().square().sum(-1, keepdim=True)
    den = xs.float().square().sum(-1, keepdim=True) + 1e-8
    return xs * (num / den).sqrt().to(xs.dtype)


def plot(path, rfe, sigmas, num_blocks):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.arange(num_blocks)
    for t, name in enumerate(TARGETS):
        axes[0].plot(xs, rfe[:, :, t].mean(0), label=name)
    for s, sigma in enumerate(sigmas):
        axes[1].plot(xs, rfe[s, :, TARGETS.index(GATE_TARGET)], label=f"σ={sigma}")
    for ax in axes:
        ax.axhline(GATE_RFE, color="k", ls=":", lw=1)
        ax.set(xlabel="block", ylabel="output RFE (NC)", ylim=(0, None))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_title("2:4 RFE by target (mean over σ)")
    axes[1].set_title(f"{GATE_TARGET} by σ")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default="post_image_dataset/moss/_calib_mixed")
    ap.add_argument("--n", type=int, default=8, help="samples, evenly spaced by stem")
    ap.add_argument("--sigmas", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--model", default="base", help="base | sft")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument(
        "--blocks_to_swap", type=int, default=-1, help="-1 = size from free VRAM"
    )
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
    print(
        f"{len(items)} samples × {len(sigmas)} σ from {args.cache}, model {args.model}",
        flush=True,
    )

    transformer = load_transformer(args.model_dir, args.model)
    transformer.requires_grad_(False).eval()
    num_blocks = len(transformer.layers)
    acc = torch.zeros(
        len(SLOTS),
        len(sigmas),
        num_blocks,
        len(TARGETS),
        device=device,
        dtype=torch.float64,
    )
    calls = torch.zeros(len(sigmas), num_blocks, len(TARGETS), device=device)
    current = {"n_img": 0, "s": 0}
    slot = {name: i for i, name in enumerate(SLOTS)}

    def measure(lin, x, b, t):
        x = x[0].to(lin.weight.dtype)  # (T, K)
        n = current["n_img"]
        xs = sparsify_24(x)
        xs_nc = norm_comp(x, xs)
        y = F.linear(x, lin.weight).float()
        e_nc = (F.linear(xs_nc, lin.weight).float() - y).square().sum(-1)
        e_raw = (F.linear(xs, lin.weight).float() - y).square().sum(-1)
        y2 = y.square().sum(-1)
        x2 = x.float().square().sum(-1)
        dx2 = (x.float() - xs.float()).square().sum(-1)
        s = current["s"]
        a = acc[:, s, b, t]
        a[slot["rfe_call"]] += (e_nc.sum() / (y2.sum() + 1e-12)).sqrt()
        for key, v in (
            ("y", y2),
            ("e_nc", e_nc),
            ("e_raw", e_raw),
            ("x", x2),
            ("dx", dx2),
        ):
            a[slot[f"{key}_img"]] += v[:n].sum()
            a[slot[f"{key}_txt"]] += v[n:].sum()
        calls[s, b, t] += 1

    handles = []
    for b, layer in enumerate(transformer.layers):
        for t, name in enumerate(TARGETS):
            lin = layer.proj[name]

            def hook(m, inp, b=b, t=t):
                measure(m, inp[0], b, t)

            handles.append(lin.register_forward_pre_hook(hook))

    attached = place(
        transformer,
        "layers",
        device,
        blocks_to_swap=None if args.blocks_to_swap < 0 else args.blocks_to_swap,
        activation_reserve_gb=4.0,
        label="transformer",
    )
    empty_cache()
    start_heartbeat()

    gen = torch.Generator(device=device)
    t0 = time.time()
    done, total = 0, len(items) * len(sigmas)
    n_text = []
    with torch.no_grad():
        for item in items:
            latent, embeds = moss_cache.load_item(item)
            latent = latent.unsqueeze(0).to(device, torch.float32)
            embeds = embeds.unsqueeze(0).to(device)
            current["n_img"] = (item.width // 16) * (item.height // 16)
            n_text.append(int(embeds.shape[1]))
            for s, sigma in enumerate(sigmas):
                current["s"] = s
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

    a = acc.cpu().numpy()
    c = calls.cpu().numpy()

    def ratio(num, den):
        return np.sqrt(a[slot[num]] / np.maximum(a[slot[den]], 1e-30))

    rfe = a[slot["rfe_call"]] / np.maximum(c, 1)  # (σ, block, target)
    arrays = {
        "rfe": rfe,
        "rfe_nc_img": ratio("e_nc_img", "y_img"),
        "rfe_nc_txt": ratio("e_nc_txt", "y_txt"),
        "rfe_raw_img": ratio("e_raw_img", "y_img"),
        "rfe_raw_txt": ratio("e_raw_txt", "y_txt"),
        "input_drop_img": ratio("dx_img", "x_img"),
        "input_drop_txt": ratio("dx_txt", "x_txt"),
    }
    run_dir = make_run_dir("moss_rfe", args.label)
    np.savez(
        run_dir / "rfe.npz",
        sigmas=np.array(sigmas),
        targets=np.array(TARGETS),
        **arrays,
    )
    plot(run_dir / "rfe.png", rfe, sigmas, num_blocks)

    def r3(v):
        return round(float(v), 4)

    by_target = {}
    for t, name in enumerate(TARGETS):
        per_block = rfe[:, :, t].mean(0)
        by_target[name] = {
            "rfe_mean": r3(per_block.mean()),
            "rfe_median_block": r3(np.median(per_block)),
            "rfe_block_range": [r3(per_block.min()), r3(per_block.max())],
            "blocks_le_gate": int((per_block <= GATE_RFE).sum()),
            "rfe_by_sigma": {
                str(sg): r3(rfe[s, :, t].mean()) for s, sg in enumerate(sigmas)
            },
            **{k: r3(v[:, :, t].mean()) for k, v in arrays.items() if k != "rfe"},
        }
        print(name, by_target[name], flush=True)
    gate_row = rfe[:, :, TARGETS.index(GATE_TARGET)].mean(0)
    passed = int((gate_row <= GATE_RFE).sum())
    gate = {
        "target": GATE_TARGET,
        "threshold": GATE_RFE,
        "blocks_pass": passed,
        "num_blocks": num_blocks,
        "pass": passed > num_blocks // 2,
    }
    print(f"gate {gate}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={"gate": gate, "by_target": by_target},
        label=args.label,
        artifacts=["rfe.npz", "rfe.png"],
        device=device,
        extra={
            "per_block_rfe": {
                name: [r3(v) for v in rfe[:, :, t].mean(0)]
                for t, name in enumerate(TARGETS)
            },
            "samples": [it.stem for it in items],
            "text_tokens": n_text,
            "anima_reference": {"mlp_down": 0.2533, "mlp_up": 0.1241},
        },
    )
    print(f"wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
