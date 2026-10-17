#!/usr/bin/env python3
"""SCFM Phase 0 (precondition) — is the teacher's noise→image transport straight?

SCFM — like any consistency / reflow few-step method — can only reach the
*many-step* teacher's quality at 4 steps if the teacher's noise→image
trajectories are nearly **straight** (the velocity points at its own endpoint the
whole way) and **non-crossing** (nearby mid-trajectory states head to nearby
images). That is *why* SCFM works on Flux (reflow-trained → straight) and the
open question for Anima (neither reflow- nor guidance-distilled). If Anima's
cfg-guided teacher transport is curved/crossing, the straightened few-step field
is multivalued → blur → SCFM's ceiling is provably below the 28-step teacher
(Arm 1), a NO-GO with no training. If it is straight, building the SCFM training
branch is justified.

This is the cheaper gate that should precede the training port: pure teacher
rollouts + dot products, no adapter, no VAE decode.

Three latent-space reads on the `turbo2` prompt (1 prompt × R seeds, cfg=4 Euler,
flow_shift=3, 28 steps):

  A. straightness-to-endpoint   cos(v_i, x_i − x_0)   (1 ⇒ v points at the final
                                image at every step ⇒ few-step Euler lands there)
  B. step-to-step turning       cos(v_i, v_{i+1})     (1 ⇒ field never bends)
  C. crossing / neighbor-preservation: at mid-σ snapshots, does proximity in x_t
     predict proximity in x_0 across seeds? (corr→1 and NN-preserved ⇒ no
     crossing ⇒ a single-valued straight field can reproduce all endpoints)

Read per σ-band; the **low-σ band (<0.5)** is where Anima text/detail resolves
and where a curved/crossing transport would sink SCFM.

    python bench/turbo/probe_teacher_straightness.py --n_seeds 16 --size 1344 768
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from anima_lora import GenerationRequest  # noqa: E402
from bench._anima import add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.runtime.harness import build_anima  # noqa: E402
from library.inference.models import load_shared_models  # noqa: E402
from library.inference.text import prepare_text_inputs  # noqa: E402
from library.inference.sampling import get_timesteps_sigmas  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

TURBO2_PROMPT = (
    'sensitive, 1girl, takagi-san, karakai jouzu no takagi-san, @sweetonedollar, '
    'upper body, blush, solo, medium breasts, swimsuit, bikini, looking at viewer, '
    'outdoors, photo background, v. She is walking on the beach. She is saying '
    'something to the viewer. There is a speech bubble that reads "But it was fast"'
)
TURBO2_NEG = "blurry, jpeg artifacts, sepia, multiple views"


def _cos_per_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine over flattened (R, ...) tensors → (R,)."""
    af = a.reshape(a.shape[0], -1).float()
    bf = b.reshape(b.shape[0], -1).float()
    return torch.nn.functional.cosine_similarity(af, bf, dim=1)


def _pair_dists(x: torch.Tensor) -> torch.Tensor:
    """Flattened pairwise euclidean distances among the R rows → (R*(R-1)/2,)."""
    f = x.reshape(x.shape[0], -1).float()
    d = torch.cdist(f[None], f[None])[0]
    iu = torch.triu_indices(d.shape[0], d.shape[0], offset=1)
    return d[iu[0], iu[1]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--n_seeds", type=int, default=16)
    p.add_argument("--teacher_steps", type=int, default=28)
    p.add_argument("--teacher_cfg", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--size", type=int, nargs=2, default=[1344, 768], metavar=("H", "W"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default=TURBO2_PROMPT)
    p.add_argument("--negative_prompt", default=TURBO2_NEG)
    p.add_argument("--label", default="scfm-teacher-straightness")
    opts = p.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R, (H, W) = opts.n_seeds, opts.size

    req = GenerationRequest(
        dit=opts.dit, vae=opts.vae, text_encoder=opts.text_encoder,
        prompt=opts.prompt, save_path="output/tests/_scfm_straight.png",
        infer_steps=opts.teacher_steps, guidance_scale=opts.teacher_cfg,
        image_size=(H, W), seed=opts.seed,
    )
    args = req.to_args()
    args.device = device
    args.dtype = "bf16"
    args.attn_mode = getattr(args, "attn_mode", "flash") or "flash"
    args.gradient_checkpointing = False
    args.compile = False
    args.compile_mode = None

    sigmas = get_timesteps_sigmas(opts.teacher_steps, opts.flow_shift, "cpu")[1]

    print("[probe] loading base DiT …")
    bundle = build_anima(args, dit_path=opts.dit, adapter=None, train_mode=False)
    anima = bundle.anima

    print("[probe] encoding turbo2 prompt …")
    shared = load_shared_models(args)
    ctx, ctx_null = prepare_text_inputs(
        args, device=device, anima=anima, shared_models=shared,
        prompt=opts.prompt, negative_prompt=opts.negative_prompt,
        text_encoder_path=args.text_encoder,
    )
    e_pos = ctx["embed"][0].to(device, torch.bfloat16).expand(R, -1, -1)
    e_neg = ctx_null["embed"][0].to(device, torch.bfloat16).expand(R, -1, -1)
    del shared
    clean_memory_on_device(device)

    pad = torch.zeros(R, 1, H // 8, W // 8, dtype=torch.bfloat16, device=device)
    gens = [torch.Generator(device="cpu").manual_seed(opts.seed + j) for j in range(R)]
    x = torch.stack([
        torch.randn(16, 1, H // 8, W // 8, generator=g, dtype=torch.float32)
        for g in gens
    ]).to(device, torch.bfloat16)

    # snapshot σ levels for the crossing read (incoming σ of the step)
    snap_targets = [0.9, 0.75, 0.5, 0.25]
    xs, vs, sig_list = [], [], []        # CPU stores: x_i, v_i, σ_i per step
    snaps = {}                            # σ -> x_t (R,16,1,h,w) cpu

    print(f"[probe] teacher rollout cfg={opts.teacher_cfg}, R={R}, {opts.teacher_steps} steps …")
    for i in range(opts.teacher_steps):
        s_i, s_next = sigmas[i], sigmas[i + 1]
        t_b = s_i.to(device=device, dtype=torch.bfloat16).expand(R)
        v_c = anima(x, t_b, e_pos, padding_mask=pad).float()
        v_u = anima(x, t_b, e_neg, padding_mask=pad).float()
        v = v_u + opts.teacher_cfg * (v_c - v_u)

        xs.append(x.float().cpu())
        vs.append(v.cpu())
        sig_list.append(float(s_i))
        for st in snap_targets:
            if abs(float(s_i) - st) < 0.03 and st not in snaps:
                snaps[st] = x.float().cpu()

        dt = (s_i - s_next).to(device=device, dtype=torch.float32)
        x = (x.float() - dt * v).to(x.dtype)
    x0 = x.float().cpu()  # final image latent (σ=0)
    print("[probe] rollout done; computing metrics …")

    del anima, bundle, pad, e_pos, e_neg
    clean_memory_on_device(device)

    # ---- A & B per step -----------------------------------------------------
    per_step = []
    for i in range(opts.teacher_steps):
        a_cos = _cos_per_sample(vs[i], xs[i] - x0).mean().item()
        if i + 1 < opts.teacher_steps:
            b_cos = _cos_per_sample(vs[i], vs[i + 1]).mean().item()
        else:
            b_cos = float("nan")
        per_step.append({"sigma": sig_list[i], "straight_endpoint_cos": a_cos, "turn_cos": b_cos})

    def band(lo, hi, key):
        vals = [s[key] for s in per_step if lo <= s["sigma"] < hi and s[key] == s[key]]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    bands = {
        "high (σ≥0.85)": {"straight": band(0.85, 1.01, "straight_endpoint_cos"),
                          "turn": band(0.85, 1.01, "turn_cos")},
        "mid (0.5–0.85)": {"straight": band(0.5, 0.85, "straight_endpoint_cos"),
                           "turn": band(0.5, 0.85, "turn_cos")},
        "low (<0.5)": {"straight": band(0.0, 0.5, "straight_endpoint_cos"),
                       "turn": band(0.0, 0.5, "turn_cos")},
    }

    # ---- C crossing at snapshots --------------------------------------------
    d0 = _pair_dists(x0)
    crossing = []
    for st in sorted(snaps, reverse=True):
        dt_ = _pair_dists(snaps[st])
        # Pearson corr between mid-σ pairwise dist and endpoint pairwise dist
        dtm, d0m = dt_ - dt_.mean(), d0 - d0.mean()
        corr = float((dtm @ d0m) / (dtm.norm() * d0m.norm() + 1e-9))
        # NN preservation: nearest neighbour in x_t-space also nearest in x0-space
        ft = snaps[st].reshape(R, -1).float()
        fz = x0.reshape(R, -1).float()
        Dt = torch.cdist(ft[None], ft[None])[0] + torch.eye(R) * 1e9
        Dz = torch.cdist(fz[None], fz[None])[0] + torch.eye(R) * 1e9
        nn_pres = float((Dt.argmin(1) == Dz.argmin(1)).float().mean())
        crossing.append({"sigma": st, "dist_corr_xt_x0": corr, "nn_preserved": nn_pres})

    # ---- report -------------------------------------------------------------
    print(f"\n{'σ':>6} {'straight(v,x-x0)':>18} {'turn(v_i,v_i+1)':>16}")
    for s in per_step:
        print(f"{s['sigma']:6.3f} {s['straight_endpoint_cos']:18.3f} {s['turn_cos']:16.3f}")
    print("\nBands:")
    for k, v in bands.items():
        print(f"  {k:16s} straight={v['straight']:.3f}  turn={v['turn']:.3f}")
    print("\nCrossing (neighbor preservation across seeds):")
    for c in crossing:
        print(f"  σ≈{c['sigma']:.2f}  dist_corr(x_t,x0)={c['dist_corr_xt_x0']:.3f}  NN_preserved={c['nn_preserved']:.3f}")

    run_dir = make_run_dir("turbo", label=opts.label)
    write_result(
        run_dir, script=__file__, args=vars(opts),
        metrics={
            "per_step": per_step,
            "bands": bands,
            "crossing": crossing,
            "n_seeds": R,
            "size": [H, W],
        },
        device=device,
    )
    print(f"\nresult.json → {run_dir}")


if __name__ == "__main__":
    main()
