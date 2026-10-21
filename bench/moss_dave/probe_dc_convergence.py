#!/usr/bin/env python3
"""DAVE premise probe on MOSS-Image: is the image tokens' DC the seed-shared part?

The MOSS port of ``bench/dave/probe_dc_convergence.py`` (read its docstring for
the premise). Same prompt over N seeds, the shipped sampler (Euler, CFG, uncond
skip 22, reference per-sample forward); per (step, block) of the **cond**
forward, on the image tokens ``x[:, :H·W]`` only (the text tokens that follow
them are left out):

  * DC  μ = mean over image tokens, ``(D,)``
  * AC  (h − μ) avg-pooled to G×G, ``(D·G·G,)``  — disjoint from the DC
  * power ratio ``‖μ‖²·H·W / ‖h‖²``

and the mean pairwise cross-seed cosine of DC and of AC. The premise holds where
DC sim ≫ AC sim in the early steps. Each prompt is its own seed set; the
``(S, L)`` matrices are averaged over prompts.

On MOSS the first run (2026-10-18) found the DC holding ~99 % of the energy and
seed-identical from step 0 (``project/moss_lora/dave.md``), so every statistic is
also reported with the top-k DC-energy channels per block left out
(``--exclude_k``), alongside the top-k energy share, the cross-prompt cosine of
the early DC and how far block 0's top-16 set persists through the stack. Per-seed
DC and pooled AC are kept on the host (fp16 AC, ~125 MB per seed at 30 steps).

``--dave SPEC`` turns on DAVE (``library.moss.dave``) for the run; captures stay
the block's own output before the edit, so the probe reads the emergent unlock,
as the Anima probe does.

    make daemon-run ARGS="bench/moss_dave/probe_dc_convergence.py --compile"
    make daemon-run ARGS="bench/moss_dave/probe_dc_convergence.py --compile --dave 8-18 --label dave"

Writes ``per_block.npz`` (``dc_sim`` / ``ac_sim`` / ``power_ratio`` /
``heavy_blocks``, the schema ``bench/dave/derive_alpha_mask.py --probe`` reads) and
``per_block_k{k}.npz`` per excluded k, ``channels.npz`` (per-block channel ranking,
early DC energy, cross-prompt cosine), ``probe_k{k}.png`` and ``result.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moss_spectrum"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from run_bench import PROMPTS  # noqa: E402


class Probe:
    """``layer_hook`` that keeps each seed's cond-forward DC ``(S, L, D)`` and pooled
    AC ``(S, L, D, G²)`` (fp16, on the host) plus the per-channel energy sum."""

    def __init__(self, num_blocks, num_steps, dim, grid, hw, do_cfg, device, dave):
        self.L, self.S, self.D, self.G, self.hw = num_blocks, num_steps, dim, grid, hw
        self.do_cfg, self.dave, self.device = do_cfg, dave, device
        self.energy = torch.zeros(num_steps, num_blocks, dim, device=device)
        self.dcs, self.acs = [], []
        self.forward_idx = -1

    def start_seed(self):
        self.forward_idx = -1
        self._dc = torch.zeros(self.S, self.L, self.D, device=self.device)
        self._ac = torch.zeros(
            self.S,
            self.L,
            self.D,
            self.G * self.G,
            device=self.device,
            dtype=torch.float16,
        )

    def end_seed(self):
        self.dcs.append(self._dc.cpu())
        self.acs.append(self._ac.cpu())

    def __call__(self, index, x, n_img):
        if index == 0:
            self.forward_idx += 1
        fi = self.forward_idx
        step = fi // 2 if self.do_cfg else fi
        if (not self.do_cfg or fi % 2 == 0) and step < self.S:
            h = x[0, :n_img].float()  # (H·W, D)
            dc = h.mean(0)
            ac = (h - dc).T.reshape(1, self.D, *self.hw)
            ac = F.adaptive_avg_pool2d(ac, (self.G, self.G)).reshape(self.D, -1)
            self._dc[step, index] = dc
            self._ac[step, index] = ac.half()
            self.energy[step, index] += (h * h).sum(0)
        if self.dave is not None and self.dave.is_active(step):
            return self.dave(index, x, n_img)
        return x


def pairwise_cos(x: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine over dim 0, the vector on the last dim."""
    n = x.shape[0]
    total = F.normalize(x.float(), dim=-1, eps=1e-8).sum(0)
    return (total.pow(2).sum(-1) - n) / (n * (n - 1))


def analyze(dcs, acs, energy, n_img, early, ks, device):
    """Per prompt ``dcs[p] (N,S,L,D)``, ``acs[p] (N,S,L,D,G²)``, ``energy[p] (S,L,D)``.

    Channels are ranked per block by early-window DC energy pooled over every prompt
    and seed. For each k, the top-k channels are dropped from DC, AC and the power
    ratio; returns ``{k: (dc_sim, ac_sim, power_ratio)}`` as ``(S, L)`` prompt means,
    the ranking, and the cross-prompt DC cosine per k ``(L,)``."""
    e_dc = torch.stack([d[:, :early].pow(2).mean((0, 1)) for d in dcs]).mean(0)  # (L,D)
    order = e_dc.argsort(-1, descending=True)
    num_blocks, dim = e_dc.shape
    out, cross = {}, {}
    for k in ks:
        keep = torch.ones(num_blocks, dim, dtype=torch.bool)
        keep.scatter_(1, order[:, :k], False)
        sims = []
        for dc, ac, en in zip(dcs, acs, energy):
            dcm = dc * keep  # zeroed channels drop out of every cosine
            dsim = pairwise_cos(dcm.to(device)).cpu()
            asim = torch.stack(
                [
                    pairwise_cos(
                        (ac[:, :, b] * keep[b, :, None]).flatten(-2).to(device)
                    ).cpu()
                    for b in range(num_blocks)
                ],
                -1,
            )
            pr = (dcm.pow(2).mean(0) * n_img * keep).sum(-1) / (
                (en * keep).sum(-1).clamp_min(1e-12)
            )
            sims.append((dsim, asim, pr))
        out[k] = tuple(torch.stack(m).mean(0).numpy() for m in zip(*sims))
        if len(dcs) > 1:
            means = torch.stack([(d[:, :early] * keep).mean((0, 1)) for d in dcs])
            cross[k] = pairwise_cos(means).numpy()
    return out, order.numpy(), e_dc.numpy(), cross


def plot(path, dc, ac, ratio, heavy):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    panels = (
        (dc, "DC cross-seed sim", "viridis", 0.5, 1.0),
        (ac, "AC cross-seed sim", "viridis", 0.0, 1.0),
        (dc - ac, "DC − AC (red = DC more locked)", "coolwarm", -0.5, 0.5),
        (ratio, "DC power ratio ‖μ‖²/‖h‖²", "magma", None, None),
    )
    for ax, (m, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(
            m.T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("step")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("block")
    for bl in heavy:
        axes[3].axhline(bl, c="cyan", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="base")
    ap.add_argument(
        "--prompts",
        default="anime_sunset,ja_manga,danbooru_tags",
        help="keys of moss_spectrum PROMPTS",
    )
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--ac_grid", type=int, default=4)
    ap.add_argument(
        "--heavy_ratio", type=float, default=0.30, help="DC-heavy block threshold"
    )
    ap.add_argument(
        "--exclude_k",
        default="4,16,64",
        help="also report with the top-k DC-energy channels per block left out",
    )
    ap.add_argument("--dave", default="", help="DAVE arm: 'lo-hi', npz path or auto")
    ap.add_argument("--dave_strength", type=float, default=0.3)
    ap.add_argument("--dave_tau", type=float, default=0.10)
    ap.add_argument("--model_dir", default="")
    ap.add_argument(
        "--te_cache", default="/media/sorryhyun/data/moss_image/bench/_te_cache"
    )
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    if args.seeds < 2:
        raise SystemExit("--seeds must be at least 2")
    start_heartbeat(label="moss_dave_probe")

    from library.moss.dave import MossDAVE
    from library.moss.generate import cached_embeds, denoise, initial_noise
    from library.moss.loader import compile_blocks, load_transformer, pad_embeds
    from library.moss.model import snapped_sigmas
    from library.moss.requests import DEFAULT_UNCOND_SKIP, resolve_model_dir

    keys = [k for k in args.prompts.split(",") if k]
    model_dir = resolve_model_dir(args.model_dir)
    device = torch.device("cuda")
    do_cfg = args.cfg != 1.0

    embeds = cached_embeds(
        model_dir, [PROMPTS[k] for k in keys] + [""], Path(args.te_cache)
    )
    uncond, uncond_mask = pad_embeds([embeds[""]], device=device)
    transformer = load_transformer(model_dir, args.model, device=device)
    if args.compile:
        compile_blocks(transformer.layers)
    num_blocks = len(transformer.layers)
    dim = transformer.config.dim
    sigmas = snapped_sigmas(args.steps)
    n_steps = len(sigmas)
    dave = None
    if args.dave:
        dave = MossDAVE.from_spec(
            args.dave, num_blocks, args.dave_strength, args.dave_tau
        )
        dave.begin(n_steps)
        print(f"[probe] {dave.describe()}", flush=True)

    ks = sorted({0, *(int(k) for k in args.exclude_k.split(",") if k)})
    dcs, acs, energy = [], [], []
    for key in keys:
        cond, cond_mask = pad_embeds([embeds[PROMPTS[key]]], device=device)
        probe = Probe(
            num_blocks,
            n_steps,
            dim,
            args.ac_grid,
            (args.size // 16,) * 2,
            do_cfg,
            device,
            dave,
        )
        transformer.layer_hook = probe
        try:
            for k in range(args.seeds):
                probe.start_seed()
                denoise(
                    transformer,
                    initial_noise(args.seed0 + k, args.size, args.size).to(device),
                    cond,
                    cond_mask,
                    uncond,
                    uncond_mask,
                    sigmas,
                    cfg=args.cfg,
                    uncond_skip=DEFAULT_UNCOND_SKIP,
                )
                probe.end_seed()
                print(f"[probe] {key} seed {k + 1}/{args.seeds}", flush=True)
        finally:
            transformer.layer_hook = None
        dcs.append(torch.stack(probe.dcs))
        acs.append(torch.stack(probe.acs))
        energy.append((probe.energy / args.seeds).cpu())
        del probe
        torch.cuda.empty_cache()

    early = max(1, n_steps // 3)
    n_img = (args.size // 16) ** 2
    curves, order, e_dc, cross = analyze(dcs, acs, energy, n_img, early, ks, device)
    share = np.cumsum(np.take_along_axis(e_dc, order, 1), 1) / e_dc.sum(
        1, keepdims=True
    )

    run_dir = make_run_dir("moss_dave", label=args.label)
    artifacts = ["channels.npz"]
    np.savez(
        run_dir / "channels.npz",
        order=order,
        dc_energy_early=e_dc,
        **{f"cross_prompt_dc_k{k}": v for k, v in cross.items()},
    )
    per_k = {}
    for k in ks:
        dc_sim, ac_sim, pratio = curves[k]
        early_ratio = pratio[:early].mean(0)
        heavy = [int(b) for b in np.flatnonzero(early_ratio >= args.heavy_ratio)]
        name = "per_block.npz" if k == 0 else f"per_block_k{k}.npz"
        np.savez(
            run_dir / name,
            dc_sim=dc_sim,
            ac_sim=ac_sim,
            power_ratio=pratio,
            heavy_blocks=np.array(heavy, dtype=np.int64),
        )
        artifacts.append(name)
        try:
            plot(run_dir / f"probe_k{k}.png", dc_sim, ac_sim, pratio, heavy)
            artifacts.append(f"probe_k{k}.png")
        except ImportError:
            pass
        dc_e, ac_e = dc_sim[:early].mean(0), ac_sim[:early].mean(0)
        hd = float(dc_e[heavy].mean()) if heavy else float("nan")
        ha = float(ac_e[heavy].mean()) if heavy else float("nan")
        per_k[k] = dict(
            dc_sim_all_early=float(dc_e.mean()),
            ac_sim_all_early=float(ac_e.mean()),
            heavy_blocks=heavy,
            dc_sim_heavy_early=hd,
            ac_sim_heavy_early=ha,
            premise_holds=bool(heavy) and hd > 0.85 and hd - ha > 0.05,
            dc_sim_early_per_block=dc_e.tolist(),
            gap_early_per_block=(dc_e - ac_e).tolist(),
            power_ratio_early_per_block=early_ratio.tolist(),
            cross_prompt_dc_per_block=cross[k].tolist() if k in cross else None,
        )

    def top(b, n=16):
        return set(order[b, :n].tolist())

    metrics = dict(
        arm=f"dave {args.dave}" if args.dave else "vanilla",
        prompts=keys,
        num_blocks=num_blocks,
        num_steps=n_steps,
        early_steps=early,
        dc_energy_top_share={
            n: share[:, n - 1].tolist() for n in (1, 4, 16, 64) if n <= dim
        },
        top8_channels={
            b: order[b, :8].tolist() for b in (0, 10, 20, 30, num_blocks - 1)
        },
        top16_jaccard_vs_block0=[
            len(top(0) & top(b)) / len(top(0) | top(b)) for b in range(num_blocks)
        ],
        by_excluded_k=per_k,
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=artifacts,
        device=device,
    )

    print("\n" + "=" * 78)
    print(
        f"  arm {metrics['arm']}, {len(keys)} prompts × {args.seeds} seeds, "
        f"early = first {early}/{n_steps} steps"
    )
    print(
        "  excluded top-k channels:  DC sim | AC sim | gap  (all blocks, early)"
        "  cross-prompt DC"
    )
    for k in ks:
        m = per_k[k]
        xp = f"{np.mean(cross[k]):.3f}" if k in cross else "-"
        print(
            f"    k={k:3d}   {m['dc_sim_all_early']:.3f} | {m['ac_sim_all_early']:.3f} | "
            f"{m['dc_sim_all_early'] - m['ac_sim_all_early']:+.3f}    {xp}"
            f"   premise {m['premise_holds']}"
        )
    print("  top-8 DC channels:", metrics["top8_channels"])
    hdr = "".join(f"  k{k}: DC   gap  pow  xP " for k in ks)
    print(f"  block  top1  top16 {hdr}")
    for b in range(num_blocks):
        row = "".join(
            f"  {per_k[k]['dc_sim_early_per_block'][b]:.3f} "
            f"{per_k[k]['gap_early_per_block'][b]:+.2f} "
            f"{per_k[k]['power_ratio_early_per_block'][b]:.2f} "
            f"{cross[k][b] if k in cross else float('nan'):.2f}"
            for k in ks
        )
        print(f"  {b:5d}  {share[b, 0]:.2f}  {share[b, 15]:.2f} {row}")
    print("=" * 78)
    print(f"→ {run_dir}", flush=True)


if __name__ == "__main__":
    main()
