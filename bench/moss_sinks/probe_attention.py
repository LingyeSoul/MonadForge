#!/usr/bin/env python3
"""Attention-sink premise probe on MOSS-Image: does a few-key set take the mass?

Tier 1 §1 of ``docs/proposal/moss_anima_negatives_revisit.md``. Same prompt over N
seeds, the shipped sampler (Euler, CFG, uncond skip 22, reference per-sample
forward). On the probe steps, every joint block's ``_attention`` is recomputed
with an explicit softmax (the block output still comes from SDPA), cond and
uncond branches apart. Keys are split into classes

  image | tok0 (``<|im_start|>``) | prefix (rest of ``<|im_start|>user\\n``) |
  caption | suffix (``<|im_end|>\\n<|im_start|>assistant\\n<think>\\n``)

(the uncond prompt is ``''``: its text is template only), and per (block, head):

  * mass each query class (image / text) puts on each key class
  * top-1 % key share: the head's mean attention per key over the image (or text)
    queries, the sum over its top ``ceil(0.01·N)`` keys; uniform = 0.01

and, from ``layer_hook`` on the block output, a per-token norm census with the
massive channels of ``library/moss/calibration/massive_channels.npz`` left out
(raw norms are also kept; with the channels in, the 0.99 image DC caps image
max/median near 1 on blocks 0–18): image max/median, image tokens ≥ 5× / 10×
median, and each text class's mean norm over the image median.

Gate (proposal): on most blocks, ≥ 25 % of heads put ≥ 30 % of their image-query
mass on ≤ 1 % of the keys; or excluded-channel image max/median ≥ 10 on most
blocks.

    make daemon-run ARGS="bench/moss_sinks/probe_attention.py"

Writes ``per_block.npz`` (all per-sample arrays), ``top_keys.json`` (per block,
the head-mean top keys by received image-query mass, with class labels),
``probe.png`` and ``result.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moss_spectrum"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from run_bench import PROMPTS  # noqa: E402

CLASSES = ("image", "tok0", "prefix", "caption", "suffix")
BRANCHES = ("cond", "uncond")
NORM_FIELDS = (
    "img_ratio_ex",
    "img_ratio_raw",
    "img_ge5_ex",
    "img_ge10_ex",
    *(f"{c}_over_img_ex" for c in CLASSES[1:]),
)


def template_split(tok, text):
    """Text-token class bounds ``(prefix_len, caption_len, suffix_len)`` of
    ``text`` under the chat template; the template's own prefix / suffix are the
    common ends of two one-token prompts."""
    from library.moss.loader import prompt_ids

    def ids(t):
        return prompt_ids(tok, t).input_ids[0].tolist()

    a, b = ids("a"), ids("b")
    pre = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    suf = next(i for i, (x, y) in enumerate(zip(a[::-1], b[::-1])) if x != y)
    return pre, len(ids(text)) - pre - suf, suf


def class_bounds(n_img, split):
    """Half-open ``[(lo, hi)]`` per class over the joint sequence."""
    pre, cap, suf = split
    t = n_img
    return [
        (0, n_img),
        (t, t + 1),
        (t + 1, t + pre),
        (t + pre, t + pre + cap),
        (t + pre + cap, t + pre + cap + suf),
    ]


class Probe:
    """Explicit-softmax ``_attention`` wrapper + norm ``layer_hook`` for one
    (prompt, branch-pair); per-seed arrays are filled in place."""

    def __init__(
        self, num_blocks, heads, probe_steps, n_img, bounds, massive, device, chunk
    ):
        self.L, self.H, self.n_img = num_blocks, heads, n_img
        self.slot = {s: j for j, s in enumerate(probe_steps)}
        self.bounds, self.chunk, self.device = bounds, chunk, device
        self.N = bounds[-1][1]
        self.keep = torch.ones(1, device=device)  # set per D on first use
        self.massive = massive
        self.block = None
        self.fi = -1
        ns = len(probe_steps)
        nan = dict(dtype=np.float32)
        self.shape = (2, ns, num_blocks)
        self.mass_img = np.full((*self.shape, heads, len(CLASSES)), np.nan, **nan)
        self.mass_txt = np.full_like(self.mass_img, np.nan)
        self.top_img = np.full((*self.shape, heads), np.nan, **nan)
        self.top_txt = np.full_like(self.top_img, np.nan)
        self.norms = np.full((*self.shape, len(NORM_FIELDS)), np.nan, **nan)
        # head-wise mean image-query mass per key, summed over seeds and probe steps
        self.col = torch.zeros(2, num_blocks, heads, self.N, device=device)

    def new_seed(self):
        self.fi = -1
        for a in (self.mass_img, self.mass_txt, self.top_img, self.top_txt, self.norms):
            a.fill(np.nan)

    @property
    def where(self):
        """``(branch, slot)`` of the running forward, or None off the probe steps."""
        step, branch = divmod(self.fi, 2)
        j = self.slot.get(step)
        return None if j is None else (branch, j)

    # --- hooks -----------------------------------------------------------------

    def attention(self, q, k, rope, heads):
        """Stats of the explicit softmax over one sample's joint sequence."""
        from library.moss.model import _rotate

        w = self.where
        if w is None or self.block is None or q.shape[1] != self.N:
            return
        branch, j, b = *w, self.block
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        dtype = q.dtype
        q = _rotate(q.reshape(shape), rope).transpose(1, 2)[0].to(dtype)
        k = _rotate(k.reshape(shape), rope).transpose(1, 2)[0].to(dtype)
        scale = 1.0 / math.sqrt(shape[-1])
        n_img, top_n = self.n_img, math.ceil(0.01 * self.N)
        for h0 in range(0, heads, self.chunk):
            hs = slice(h0, min(h0 + self.chunk, heads))
            p = torch.softmax(
                (q[hs].float() @ k[hs].float().transpose(-1, -2)) * scale, -1
            )
            for qs, mass, top in (
                (slice(0, n_img), self.mass_img, self.top_img),
                (slice(n_img, self.N), self.mass_txt, self.top_txt),
            ):
                col = p[:, qs].mean(1)  # (h, N)
                mass[branch, j, b, hs] = (
                    torch.stack([col[:, lo:hi].sum(-1) for lo, hi in self.bounds], -1)
                    .cpu()
                    .numpy()
                )
                top[branch, j, b, hs] = col.topk(top_n, -1).values.sum(-1).cpu().numpy()
                if qs.start == 0:
                    self.col[branch, b, hs] += col
            del p

    def __call__(self, index, x, n_img):
        w = self.where
        if w is None:
            return x
        branch, j = w
        h = x[0].float()
        if self.keep.numel() != h.shape[-1]:
            self.keep = torch.ones(h.shape[-1], device=h.device)
            self.keep[self.massive] = 0
        raw = h.norm(dim=-1)
        ex = (h * self.keep).norm(dim=-1)
        img_ex, img_raw = ex[:n_img], raw[:n_img]
        med_ex, med_raw = img_ex.median(), img_raw.median()
        row = [
            img_ex.max() / med_ex,
            img_raw.max() / med_raw,
            (img_ex >= 5 * med_ex).sum(),
            (img_ex >= 10 * med_ex).sum(),
        ]
        for lo, hi in self.bounds[1:]:
            row.append(
                ex[lo:hi].mean() / med_ex if hi > lo else torch.tensor(float("nan"))
            )
        self.norms[branch, j, index] = torch.stack(
            [r.float().cpu() for r in row]
        ).numpy()
        return x


def plot(path, top_cond, top_uncond, mass_cond, norm_ex):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, m, title in (
        (axes[0], top_cond, "cond: top-1 % key share, image queries"),
        (axes[1], top_uncond, "uncond: top-1 % key share, image queries"),
    ):
        im = ax.imshow(m.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_xlabel("block")
        ax.set_ylabel("head")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    blocks = np.arange(mass_cond.shape[0])
    bottom = np.zeros_like(blocks, dtype=np.float64)
    for c, name in enumerate(CLASSES):
        axes[2].bar(blocks, mass_cond[:, c], bottom=bottom, label=name)
        bottom += mass_cond[:, c]
    axes[2].set_title("cond: image-query mass by key class (head mean)")
    axes[2].set_xlabel("block")
    axes[2].legend(fontsize=8)
    axes[3].plot(blocks, norm_ex, marker=".")
    axes[3].axhline(10, c="red", lw=0.6)
    axes[3].set_yscale("log")
    axes[3].set_title("image norm max/median, massive channels out (cond)")
    axes[3].set_xlabel("block")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="base")
    ap.add_argument(
        "--prompts",
        default="anime_sunset,ja_manga",
        help="keys of moss_spectrum PROMPTS",
    )
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--probe_steps", default="0,3,7,11,15,19,23,29")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--head_chunk", type=int, default=6)
    ap.add_argument("--sink_share", type=float, default=0.30)
    ap.add_argument("--head_frac", type=float, default=0.25)
    ap.add_argument("--norm_ratio", type=float, default=10.0)
    ap.add_argument(
        "--massive", default="library/moss/calibration/massive_channels.npz"
    )
    ap.add_argument("--model_dir", default="")
    ap.add_argument(
        "--te_cache", default="/media/sorryhyun/data/moss_image/bench/_te_cache"
    )
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    if args.cfg == 1.0:
        raise SystemExit("--cfg 1 has no uncond branch; the probe reads both")
    start_heartbeat(label="moss_sinks_probe")

    import library.moss.model as moss_model
    from library.env import resolve_under_home
    from library.moss.generate import _cached_embeds, denoise, initial_noise
    from library.moss.loader import load_tokenizer, load_transformer, pad_embeds
    from library.moss.model import DEFAULT_UNCOND_SKIP, snapped_sigmas
    from library.moss.requests import resolve_model_dir

    keys = [k for k in args.prompts.split(",") if k]
    probe_steps = [int(s) for s in args.probe_steps.split(",") if s]
    model_dir = resolve_model_dir(args.model_dir)
    device = torch.device("cuda")
    massive = torch.from_numpy(
        np.load(resolve_under_home(args.massive))["channels"]
    ).to(device)

    tok = load_tokenizer(model_dir)
    splits = {k: template_split(tok, PROMPTS[k]) for k in keys}
    splits[""] = template_split(tok, "")
    del tok
    embeds = _cached_embeds(
        model_dir, [PROMPTS[k] for k in keys] + [""], Path(args.te_cache)
    )
    uncond, uncond_mask = pad_embeds([embeds[""]], device=device)
    transformer = load_transformer(model_dir, args.model, device=device)
    num_blocks, heads = len(transformer.layers), transformer.config.num_heads
    sigmas = snapped_sigmas(args.steps)
    n_steps = len(sigmas)
    probe_steps = [s for s in probe_steps if s < n_steps]
    n_img = (args.size // 16) ** 2
    print(f"[probe] template splits {splits}", flush=True)

    original = moss_model._attention
    results = {}
    for key in keys:
        cond, cond_mask = pad_embeds([embeds[PROMPTS[key]]], device=device)
        bounds = {
            0: class_bounds(n_img, splits[key]),
            1: class_bounds(n_img, splits[""]),
        }
        probes = [
            Probe(
                num_blocks,
                heads,
                probe_steps,
                n_img,
                bounds[br],
                massive,
                device,
                args.head_chunk,
            )
            for br in (0, 1)
        ]
        state = {"fi": -1}

        def active():
            return probes[state["fi"] % 2]

        def on_forward(_m, _a):
            state["fi"] += 1
            for p in probes:
                p.fi = state["fi"]

        def wrapped(q, k, v, rope, heads_, key_mask=None):
            active().attention(q, k, rope, heads_)
            return original(q, k, v, rope, heads_, key_mask)

        def layer_hook(index, x, n):
            return active()(index, x, n)

        def set_block(index):
            def hook(_m, _a):
                for p in probes:
                    p.block = index

            return hook

        handles = [transformer.register_forward_pre_hook(on_forward)]
        # the text refiner's blocks also call _attention; None keeps them out
        handles.append(
            transformer.text_embedder.register_forward_pre_hook(set_block(None))
        )
        for i, layer in enumerate(transformer.layers):
            handles.append(layer.register_forward_pre_hook(set_block(i)))
        moss_model._attention = wrapped
        transformer.layer_hook = layer_hook
        per_seed = {
            n: [] for n in ("mass_img", "mass_txt", "top_img", "top_txt", "norms")
        }
        try:
            for s in range(args.seeds):
                state["fi"] = -1
                for p in probes:
                    p.new_seed()
                denoise(
                    transformer,
                    initial_noise(args.seed0 + s, args.size, args.size).to(device),
                    cond,
                    cond_mask,
                    uncond,
                    uncond_mask,
                    sigmas,
                    cfg=args.cfg,
                    uncond_skip=DEFAULT_UNCOND_SKIP,
                )
                for name in per_seed:
                    # branch b's probe only ever fills its own branch row
                    per_seed[name].append(
                        np.stack([getattr(probes[b], name)[b] for b in (0, 1)])
                    )
                print(f"[probe] {key} seed {s + 1}/{args.seeds}", flush=True)
        finally:
            moss_model._attention = original
            transformer.layer_hook = None
            for h in handles:
                h.remove()
        results[key] = {n: np.stack(v) for n, v in per_seed.items()}  # (seeds, 2, …)
        # cond and uncond have different joint lengths: one array per branch
        results[key]["col"] = [
            probes[b].col[b].cpu().numpy() / (args.seeds * len(probe_steps))
            for b in (0, 1)
        ]
        results[key]["bounds"] = bounds
        torch.cuda.empty_cache()

    # --- analysis ----------------------------------------------------------------

    def pooled(name):
        """Mean over prompts, seeds and probe steps → ``(2, L, …)``."""
        return np.nanmean(
            np.stack([np.nanmean(results[k][name], axis=(0, 2)) for k in keys]), 0
        )

    top_img, top_txt = pooled("top_img"), pooled("top_txt")  # (2, L, H)
    mass_img, mass_txt = pooled("mass_img"), pooled("mass_txt")  # (2, L, H, C)
    norms = pooled("norms")  # (2, L, F)
    # head fraction per sample, then pooled: a head counts per (seed, step)
    frac = np.nanmean(
        np.stack(
            [
                np.nanmean(
                    np.where(
                        np.isnan(results[k]["top_img"]),
                        np.nan,
                        results[k]["top_img"] >= args.sink_share,
                    ).mean(-1),
                    axis=(0, 2),
                )
                for k in keys
            ]
        ),
        0,
    )  # (2, L)
    ratio_ex = norms[..., NORM_FIELDS.index("img_ratio_ex")]
    blocks_attn = frac[0] >= args.head_frac
    blocks_norm = ratio_ex[0] >= args.norm_ratio
    gate_attn = bool(blocks_attn.mean() > 0.5)
    gate_norm = bool(blocks_norm.mean() > 0.5)

    top_keys = {}
    for key in keys:
        col = [c.mean(1) for c in results[key]["col"]]  # per branch (L, N) head mean
        bounds = results[key]["bounds"]
        top_keys[key] = {}
        for br in (0, 1):
            rows = []
            for b in range(num_blocks):
                vals, idx = torch.from_numpy(col[br][b]).topk(8)
                rows.append(
                    [
                        [
                            int(i),
                            next(
                                CLASSES[c]
                                for c, (lo, hi) in enumerate(bounds[br])
                                if lo <= i < hi
                            ),
                            round(float(v), 4),
                        ]
                        for v, i in zip(vals.tolist(), idx.tolist())
                    ]
                )
            top_keys[key][BRANCHES[br]] = rows

    run_dir = make_run_dir("moss_sinks", label=args.label)
    arrays = {}
    for key in keys:
        for name in ("mass_img", "mass_txt", "top_img", "top_txt", "norms"):
            arrays[f"{key}__{name}"] = results[key][name]
        for br in (0, 1):
            arrays[f"{key}__col_{BRANCHES[br]}"] = results[key]["col"][br]
    np.savez(
        run_dir / "per_block.npz",
        classes=np.array(CLASSES),
        norm_fields=np.array(NORM_FIELDS),
        probe_steps=np.array(probe_steps),
        **arrays,
    )
    (run_dir / "top_keys.json").write_text(json.dumps(top_keys))
    artifacts = ["per_block.npz", "top_keys.json"]
    try:
        plot(
            run_dir / "probe.png",
            top_img[0],
            top_img[1],
            np.nanmean(mass_img[0], 1),
            ratio_ex[0],
        )
        artifacts.append("probe.png")
    except ImportError:
        pass

    def per_class(m):
        return {c: np.nanmean(m[..., i], -1).tolist() for i, c in enumerate(CLASSES)}

    metrics = dict(
        prompts=keys,
        template_splits={k: list(v) for k, v in splits.items()},
        num_blocks=num_blocks,
        num_heads=heads,
        probe_steps=probe_steps,
        gate_attention=gate_attn,
        gate_norm=gate_norm,
        premise_holds=gate_attn or gate_norm,
        blocks_attention_pass=[int(b) for b in np.flatnonzero(blocks_attn)],
        blocks_norm_pass=[int(b) for b in np.flatnonzero(blocks_norm)],
        **{
            f"{br}_{name}": val
            for i, br in enumerate(BRANCHES)
            for name, val in (
                ("head_frac_sink_per_block", frac[i].tolist()),
                (
                    "top1pct_img_q_head_mean_per_block",
                    np.nanmean(top_img[i], -1).tolist(),
                ),
                (
                    "top1pct_img_q_head_max_per_block",
                    np.nanmax(top_img[i], -1).tolist(),
                ),
                (
                    "top1pct_txt_q_head_mean_per_block",
                    np.nanmean(top_txt[i], -1).tolist(),
                ),
                ("img_q_mass_by_class", per_class(mass_img[i])),
                ("txt_q_mass_by_class", per_class(mass_txt[i])),
                (
                    "norms",
                    {f: norms[i, :, n].tolist() for n, f in enumerate(NORM_FIELDS)},
                ),
            )
        },
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

    print("\n" + "=" * 96)
    print(
        f"  {len(keys)} prompts × {args.seeds} seeds, probe steps {probe_steps}; "
        f"gate attention {gate_attn} ({int(blocks_attn.sum())}/{num_blocks} blocks), "
        f"norm {gate_norm} ({int(blocks_norm.sum())}/{num_blocks})"
    )
    print(
        "  block | cond: hfrac top1%(mean/max) | img-q mass image tok0 pre cap suf "
        "| norm ex/raw | uncond: hfrac top1% tok0+pre+suf"
    )
    for b in range(num_blocks):
        mc = np.nanmean(mass_img[0, b], 0)
        mu = np.nanmean(mass_img[1, b], 0)
        print(
            f"  {b:5d} | {frac[0, b]:.2f} {np.nanmean(top_img[0, b]):.2f}/"
            f"{np.nanmax(top_img[0, b]):.2f} | "
            + " ".join(f"{v:.3f}" for v in mc)
            + f" | {ratio_ex[0, b]:6.1f}/{norms[0, b, 1]:5.2f} | "
            f"{frac[1, b]:.2f} {np.nanmean(top_img[1, b]):.2f} "
            f"{mu[1] + mu[2] + mu[4]:.3f}"
        )
    print("=" * 96)
    print(f"→ {run_dir}", flush=True)


if __name__ == "__main__":
    main()
