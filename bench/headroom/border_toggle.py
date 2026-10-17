"""RQ1, sharpened: matched-seed border TOGGLE contrast.

The sweep probe (``sink_probe.py``) compared the self-attn sink on border-positive
vs border-negative generations, but those were *different noise draws* (the border is
a ~17% stochastic event), so a null Δ is suggestive, not airtight. This probe holds
the **noise fixed** and toggles the border via the prompt/negative — the strongest
causal test: same seed, flip only the border, does the sink move with it?

Two levers (per the reviewer's note that `cropped` is a muddy proxy — in tag
semantics it means "subject cut off by frame," not "rendered bar"):
  - **suppress** on the known border-making seeds (negative = "border, letterbox, …")
    → does removing the rendered border also remove/relocate the sink? (ON→OFF)
  - **induce** on known clean seeds (prompt += "black border, letterboxed")
    → does forcing a border summon the sink? (OFF→ON)

If the mid-layer self-attn sink is the border's *cause*, its border-enrichment must
rise and fall WITH the rendered border across matched seeds. If it sits at the same
level whether the border is on or off, the sink is a constant background feature and
the border is a data-prior artifact — RQ1 kill (redirect to crop-conditioning /
border augmentation, exactly the proposal's own kill branch).

Usage::

    uv run python bench/headroom/border_toggle.py \
        --border_seeds 4,10,23,25,29 --clean_seeds 0,1,2,3,5 --label toggle
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sink_probe as sp  # noqa: E402  (sibling; reuses recorder + hooks + helpers)
from anima_lora import GenerationRequest, load_vae  # noqa: E402
from bench._anima import add_common_args, add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.inference.generation import generate_body  # noqa: E402
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.output import decode_latent, pixels_to_pil  # noqa: E402
from library.inference.text import ensure_text_strategies, prepare_text_inputs  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

_BORDER_NEG = "border, black border, white border, black bars, letterbox, letterboxed, pillarboxed, framed, frame, bordered"
_BORDER_POS_SUFFIX = ", black border, letterboxed, pillarboxed, framed"


def _parse_seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def build_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    add_common_args(p)
    p.add_argument("--border_seeds", default="4,10,23,25,29", help="Seeds that render a border at base (ON→OFF suppression test).")
    p.add_argument("--clean_seeds", default="0,1,2,3,5", help="Seeds that don't border at base (OFF→ON induction test).")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--border_ring", type=int, default=2)
    p.add_argument("--border_px_frac", type=float, default=0.04)
    p.add_argument("--flat_thr", type=float, default=0.05)
    p.add_argument("--extreme_thr", type=float, default=0.80)
    p.add_argument("--topk_frac", type=float, default=0.02)
    p.add_argument("--delta_min", type=float, default=0.50, help="Min ON-vs-OFF enrichment gap (×chance) to call the sink border-tracking.")
    return p.parse_args()


def _mid_hi(rows):
    return [r for r in rows if sp._depth_bucket(r["depth_frac"]) == "mid" and sp._sigma_band(r["sigma"]) == "hi"]


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    req = GenerationRequest(
        prompt="placeholder",
        dit=args.dit,
        vae=args.vae,
        text_encoder=args.text_encoder,
        negative_prompt="",
        image_size=(args.size, args.size),
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg,
        flow_shift=args.flow_shift,
        attn_mode=args.attn_mode,
        device=args.device,
        seed=0,
    )
    gen_args = req.to_args()
    gen_args.compile_blocks = False

    ensure_text_strategies(args.text_encoder)
    text_encoder = load_text_encoder(gen_args, dtype=torch.bfloat16, device=device)
    text_encoder.eval()
    shared = {"text_encoder": text_encoder, "conds_cache": {}}
    anima = load_dit_model(gen_args, device, torch.bfloat16)
    shared["model"] = anima
    vae = load_vae(args.vae, device="cpu", vae_2d=True)
    vae.to(torch.bfloat16).eval()

    rec = sp.SinkRecorder(anima, ring=args.border_ring, topk_frac=args.topk_frac)
    sp.install_hooks(anima, rec)

    out_dir = make_run_dir("headroom", args.label)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # chance = fraction of tokens on the border ring (enrichment baseline).
    hp = wp = args.size // 8 // 2  # latent = size/8, patch grid = latent/2 (square)
    chance = float((sp._edge_distance(hp, wp) < args.border_ring).mean())

    border_seeds = _parse_seeds(args.border_seeds)
    clean_seeds = _parse_seeds(args.clean_seeds)

    # (condition, prompt_suffix, negative, seed_set) — matched-seed toggles.
    plans = [
        ("base_border", "", "", border_seeds),  # ON baseline
        ("neg_border", "", _BORDER_NEG, border_seeds),  # suppression: hope OFF
        ("base_clean", "", "", clean_seeds),  # OFF baseline
        ("pos_clean", _BORDER_POS_SUFFIX, "", clean_seeds),  # induction: hope ON
    ]

    per_gen = []
    total = sum(len(pl[3]) for pl in plans)
    done = 0
    for cond, suffix, neg, seeds in plans:
        prompt = sp._REPRO + suffix
        ctx, ctx_null = prepare_text_inputs(
            gen_args, device, anima, shared, prompt=prompt, negative_prompt=neg
        )
        for s in seeds:
            rec.begin_generation(prompt_name=cond, variant=f"seed{s}")
            gid = rec.gen
            with torch.no_grad():
                latent = generate_body(gen_args, anima, ctx, ctx_null, device, seed=s)
            img = decode_latent(vae, latent, device)
            clean_memory_on_device(device)
            bscore = sp._border_artifact_score(img, args.border_px_frac, args.flat_thr, args.extreme_thr)
            pixels_to_pil(img).save(img_dir / f"{cond}_s{s}_b{bscore:.2f}.png")
            per_gen.append({"gen": gid, "cond": cond, "seed": s, "border_score": round(bscore, 4)})
            done += 1
            print(f"[{done}/{total}] {cond} seed={s} border={bscore:.2f}", flush=True)

    # ---- analysis ---------------------------------------------------------
    gid_border = {p["gen"]: p["border_score"] > 0 for p in per_gen}
    gid_cond = {p["gen"]: p["cond"] for p in per_gen}
    gid_seed = {p["gen"]: p["seed"] for p in per_gen}

    def enr(rows):
        mh = _mid_hi(rows)
        return (sp._median([r["act_topk_border_frac"] for r in mh]) / chance) if mh else None

    # per-generation enrichment at mid/hi.
    gen_enr = {}
    for gid in gid_cond:
        rows = [r for r in rec.rows if r["gen"] == gid]
        gen_enr[gid] = enr(rows)

    # condition-level: border rate + mean enrichment.
    cond_summary = {}
    for cond, _, _, _ in plans:
        gids = [g for g, c in gid_cond.items() if c == cond]
        rate = sum(gid_border[g] for g in gids) / len(gids) if gids else 0.0
        es = [gen_enr[g] for g in gids if gen_enr[g] is not None]
        cond_summary[cond] = {
            "n": len(gids),
            "border_rate": round(rate, 3),
            "mean_enrichment_midhi": round(sum(es) / len(es), 3) if es else None,
        }

    # the causal contrast: pool ALL gens, bucket by whether THIS image has a
    # border, compare mid/hi enrichment ON vs OFF (spans natural+induced+suppressed).
    on_e = [gen_enr[g] for g, on in gid_border.items() if on and gen_enr[g] is not None]
    off_e = [gen_enr[g] for g, on in gid_border.items() if not on and gen_enr[g] is not None]
    on_mean = round(sum(on_e) / len(on_e), 3) if on_e else None
    off_mean = round(sum(off_e) / len(off_e), 3) if off_e else None
    delta = round(on_mean - off_mean, 3) if (on_mean is not None and off_mean is not None) else None

    # matched within-seed pairs: base_border(ON?) vs neg_border(OFF?) at same seed.
    pairs = []
    for s in border_seeds:
        b = next((g for g in gid_cond if gid_cond[g] == "base_border" and gid_seed[g] == s), None)
        n = next((g for g in gid_cond if gid_cond[g] == "neg_border" and gid_seed[g] == s), None)
        if b is not None and n is not None:
            pairs.append(
                {
                    "seed": s,
                    "base_border": gid_border[b],
                    "neg_border": gid_border[n],
                    "base_enrichment": gen_enr[b],
                    "neg_enrichment": gen_enr[n],
                    "toggled_off": gid_border[b] and not gid_border[n],
                }
            )
    # among seeds where the negative actually removed the border, did enrichment drop?
    removed = [p for p in pairs if p["toggled_off"] and p["base_enrichment"] and p["neg_enrichment"]]
    paired_drop = (
        round(sum(p["base_enrichment"] - p["neg_enrichment"] for p in removed) / len(removed), 3)
        if removed
        else None
    )

    # ---- verdict ----------------------------------------------------------
    reasons = []
    neg_rate = cond_summary["neg_border"]["border_rate"]
    pos_rate = cond_summary["pos_clean"]["border_rate"]
    base_rate = cond_summary["base_border"]["border_rate"]
    lever_worked = (neg_rate < base_rate - 0.2) or (pos_rate > cond_summary["base_clean"]["border_rate"] + 0.2)
    if not lever_worked:
        reasons.append(
            f"CAUTION: border lever weak (base {base_rate}→neg {neg_rate}; clean {cond_summary['base_clean']['border_rate']}→pos {pos_rate}) — toggle didn't move the border much; contrast underpowered"
        )
    if delta is None:
        verdict = "INCONCLUSIVE"
        reasons.append("no ON or OFF gens to compare")
    elif delta >= args.delta_min:
        verdict = "PASS"
        reasons.append(f"sink TRACKS the border: mid/hi enrichment ON {on_mean}× vs OFF {off_mean}× (Δ={delta}× ≥ {args.delta_min}×)")
    else:
        verdict = "KILL"
        reasons.append(f"sink does NOT track the border: mid/hi enrichment ON {on_mean}× vs OFF {off_mean}× (Δ={delta}× < {args.delta_min}×) — constant background feature, border is a data prior")

    agg = {
        "verdict": verdict,
        "verdict_reasons": reasons,
        "chance_border_frac": round(chance, 4),
        "cond_summary": cond_summary,
        "on_off_contrast": {"on_mean_enrichment": on_mean, "off_mean_enrichment": off_mean, "delta": delta, "n_on": len(on_e), "n_off": len(off_e)},
        "matched_pairs": pairs,
        "paired_enrichment_drop_when_border_removed": paired_drop,
    }

    with (out_dir / "per_gen.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_gen[0].keys()))
        w.writeheader()
        w.writerows(per_gen)
    with (out_dir / "rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.rows[0].keys()))
        w.writeheader()
        w.writerows(rec.rows)
    write_result(
        out_dir,
        script="bench/headroom/border_toggle.py",
        label=args.label,
        args=args,
        metrics=agg,
        artifacts=["per_gen.csv", "rows.csv"],
    )

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    for r in reasons:
        print(f"  - {r}")
    print("\ncondition summary (border rate + mid/hi enrichment):")
    for c, v in cond_summary.items():
        print(f"  {c:12s} n={v['n']} border_rate={v['border_rate']} enrichment={v['mean_enrichment_midhi']}")
    print(f"\nON vs OFF enrichment: {on_mean}× vs {off_mean}×  Δ={delta}×  (n_on={len(on_e)}, n_off={len(off_e)})")
    print(f"paired drop when border removed by negative: {paired_drop}")
    print("\nmatched pairs (base vs neg on border seeds):")
    for p in pairs:
        print(f"  seed {p['seed']}: base_border={p['base_border']} neg_border={p['neg_border']} enr {p['base_enrichment']}→{p['neg_enrichment']} toggled_off={p['toggled_off']}")
    print(f"\nresults → {out_dir}")


if __name__ == "__main__":
    main()
