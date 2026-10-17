#!/usr/bin/env python3
"""Does the LoRA's extra early-σ wander cause REAL pixel breakage? (SAM3 detector)

The discriminator probe (wander_is_it_real.py) came back null: the LoRA's extra
*backtrack* is diffuse, coinciding with neither added detail nor seed-lottery
regions. That rules out the strong benign/harmful spatial signatures but can't
answer "does it degrade" — there's no quality reward for Anima, and crossvar is
only a breakage-PRONE proxy, not breakage. report.md names the real follow-up:
"a hand/anatomy detector ... measure mode correctness, not path geometry."

OpenPose is useless here (0 poses on anime — trained on photos). SAM3 is
text-promptable and anime-robust: "hand" -> hand instance masks+scores, "finger"
-> finger instances. Finger-count is THE canonical t2i breakage (the 6-finger
hand). So:

  breakage(image) = excess fingers = Σ_hand max(0, n_fingers_in_hand - 5)
  (excess only — a fist showing <5 is occlusion, not breakage; >5 is unambiguous)

Paired base-vs-LoRA on the SAME prompt+seed, so any systematic SAM3 bias cancels.

Two questions, one harness:
  A. DEGRADATION (does it degrade?) — does the LoRA produce more excess fingers
     than base on the same prompt+seed?  Direct, pixel-space, detector-based.
  B. MECHANISM (is wander the cause?) — early-σ wander map vs the hand region:
       * locality   : enrichment = mean(wander|hand) / mean(wander|¬hand)
       * link       : corr(per-image hand-region wander, excess fingers)
       * paired link: corr(Δ hand-region wander, Δ excess)  [lora-base]
     Positive link ⇒ wander predicts the breakage; null ⇒ wander ≠ breakage axis
     (would corroborate report.md's n=1 eyes dissection: breakage = overfit, not
     path).

Two phases (OOM-safe): generate+decode+capture all images with the DiT (base
then LoRA, each bundle freed), THEN load SAM3 and analyze the saved PNGs + maps.

Usage::

    python bench/x0_contradiction/wander_vs_breakage.py \
        --prompts-file bench/x0_contradiction/prompts/hands.txt \
        --seeds 0,1,2,3,4 --compile --label hands
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# numpy<2 shim sam3 needs (np.bool removed); matches generate_masks.py.
if not hasattr(np, "bool"):
    np.bool = np.bool_

from bench._common import make_run_dir, write_result  # noqa: E402

import anima_lora  # noqa: E402
from bench.x0_contradiction.run_bench import capture_x0  # noqa: E402
from bench.x0_contradiction.wander_is_it_real import early_maps, pearson  # noqa: E402
from library.runtime.harness import build_inference_bundle  # noqa: E402


# --------------------------------------------------------------------------- #
# SAM3 detector
# --------------------------------------------------------------------------- #
class HandFingerDetector:
    """SAM3 'hand'/'finger' concept segmentation -> finger-count breakage."""

    def __init__(self, device="cuda", score_thr=0.5):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = build_sam3_image_model(device=device, eval_mode=True)
        self.proc = Sam3Processor(self.model)
        self.thr = score_thr

    def _instances(self, state, concept):
        out = self.proc.set_text_prompt(state=state, prompt=concept)
        masks, scores = [], []
        for m, s in zip(out["masks"], out["scores"]):
            s = float(s)
            if s < self.thr:
                continue
            mn = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
            if mn.ndim == 3:
                mn = mn[0]
            masks.append(mn > 0.5)
            scores.append(s)
        return masks, scores

    def detect(self, pil_img) -> dict:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = self.proc.set_image(pil_img)
            hand_masks, hand_scores = self._instances(state, "hand")
            finger_masks, finger_scores = self._instances(state, "finger")

        # finger -> hand association by centroid-in-dilated-hand-mask
        def centroid(m):
            ys, xs = np.nonzero(m)
            return (ys.mean(), xs.mean()) if len(ys) else (np.nan, np.nan)

        fcent = [centroid(m) for m in finger_masks]
        per_hand = []
        union = np.zeros(pil_img.size[::-1], dtype=bool)  # (H,W)
        for hm in hand_masks:
            union |= hm
            # dilate hand mask by ~10% of its bbox diagonal so fingers just
            # outside the palm mask still count
            ys, xs = np.nonzero(hm)
            if not len(ys):
                per_hand.append(0)
                continue
            pad = int(0.12 * np.hypot(np.ptp(ys) + 1, np.ptp(xs) + 1))
            y0, y1 = max(0, ys.min() - pad), ys.max() + pad
            x0, x1 = max(0, xs.min() - pad), xs.max() + pad
            n = sum(
                1
                for (cy, cx) in fcent
                if not np.isnan(cy) and y0 <= cy <= y1 and x0 <= cx <= x1
            )
            per_hand.append(n)

        excess = sum(max(0, n - 5) for n in per_hand)
        return {
            "n_hands": len(hand_masks),
            "total_fingers": len(finger_masks),
            "per_hand_fingers": per_hand,
            "excess_fingers": excess,
            "mean_hand_score": float(np.mean(hand_scores)) if hand_scores else np.nan,
            "hand_union": union,  # (H,W) bool
        }


def downsample_mask(mask_hw: np.ndarray, out_hw) -> np.ndarray:
    """Area-pool a pixel (H,W) bool mask to the latent grid -> (h,w) float in [0,1]."""
    from PIL import Image

    h, w = out_hw
    return (
        np.asarray(
            Image.fromarray((mask_hw * 255).astype(np.uint8)).resize(
                (w, h), Image.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="output/ckpt/anima_sincos.safetensors")
    p.add_argument("--multiplier", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow-shift", dest="flow_shift", type=float, default=3.0)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--split", type=float, default=0.45)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--prompts-file", default="bench/x0_contradiction/prompts/hands.txt")
    p.add_argument("--label", default="breakage")
    p.add_argument("--compile", action="store_true")
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    prompts = [
        ln.strip()
        for ln in Path(a.prompts_file).read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if not Path(a.adapter).is_absolute():
        a.adapter = str(REPO_ROOT / a.adapter)
    if not Path(a.adapter).exists():
        sys.exit(f"adapter not found: {a.adapter}")

    ck = anima_lora.default_checkpoints()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir("x0_contradiction", label=a.label)
    extra = ("--compile_blocks",) if a.compile else ()
    (run_dir / "img").mkdir(exist_ok=True)
    print(f"run dir: {run_dir}")
    print(f"{len(prompts)} prompts x {len(seeds)} seeds")

    def mkargs(prompt, seed, with_lora):
        return anima_lora.GenerationRequest(
            dit=ck.dit,
            vae=ck.vae,
            text_encoder=ck.text_encoder,
            prompt=prompt,
            image_size=(a.size, a.size),
            infer_steps=a.steps,
            guidance_scale=a.cfg,
            flow_shift=a.flow_shift,
            sampler="euler",
            seed=seed,
            lora_weight=[a.adapter] if with_lora else None,
            lora_multiplier=[a.multiplier] if with_lora else None,
            extra_argv=extra,
        ).to_args()

    # ---- Phase 1: generate + capture wander + decode, both models -----------
    # records[(model, pi, seed)] = {"img": path, "wander": (h,w) early-σ path map}
    # Resumable: image PNG + wander .npy persisted, so a phase-2 crash never
    # re-runs the ~10min generation. Existing files (same run --label) are reused.
    records: dict = {}

    def paths(model, pi, seed):
        return (
            run_dir / "img" / f"{model}_p{pi}_s{seed}.png",
            run_dir / "img" / f"{model}_p{pi}_s{seed}_wander.npy",
        )

    need_gen = any(
        not (lambda ip, wp: ip.exists() and wp.exists())(*paths(m, pi, seed))
        for m in ("base", "lora")
        for pi in range(len(prompts))
        for seed in seeds
    )
    for with_lora, model in ((False, "base"), (True, "lora")):
        bundle = None
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                ipath, wpath = paths(model, pi, seed)
                if ipath.exists() and wpath.exists():
                    records[(model, pi, seed)] = {
                        "img": ipath,
                        "wander": np.load(wpath),
                    }
                    continue
                if bundle is None:
                    print(f"\n[phase1] loading {model} bundle (compile={a.compile})...")
                    bundle = build_inference_bundle(
                        mkargs(prompts[0], seeds[0], with_lora),
                        device=device,
                        with_vae=True,
                    )
                with capture_x0() as cap:
                    latent = bundle.generate(mkargs(prompt, seed, with_lora))
                wander = early_maps(cap, a.split)["path"]  # (h,w) early-σ path
                img = anima_lora.decode_to_pil(bundle.vae, latent, bundle.device)
                img.save(ipath)
                np.save(wpath, wander)
                records[(model, pi, seed)] = {"img": ipath, "wander": wander}
            print(f"  {model} p{pi} done ({prompt[:40]})")
        if bundle is not None:
            del bundle
            torch.cuda.empty_cache()
    if not need_gen:
        print("[phase1] all images + wander maps cached — skipped generation")

    # ---- Phase 2: SAM3 finger-count breakage on every saved image -----------
    print("\n[phase2] loading SAM3 detector...")
    from PIL import Image

    det = HandFingerDetector(device=device)
    for key, rec in records.items():
        try:
            d = det.detect(Image.open(rec["img"]).convert("RGB"))
        except Exception as exc:
            print(f"  [detect failed {key}: {exc}]")
            rec.update(
                excess=0,
                total_fingers=0,
                n_hands=0,
                mean_hand_score=np.nan,
                hand_wander=np.nan,
                enrich=np.nan,
            )
            continue
        hw = rec["wander"].shape
        hand_lat = downsample_mask(d["hand_union"], hw) > 0.25  # latent-grid hand mask
        w = rec["wander"]
        inside = float(w[hand_lat].mean()) if hand_lat.any() else np.nan
        outside = float(w[~hand_lat].mean()) if (~hand_lat).any() else np.nan
        rec.update(
            excess=d["excess_fingers"],
            total_fingers=d["total_fingers"],
            n_hands=d["n_hands"],
            mean_hand_score=d["mean_hand_score"],
            hand_wander=inside,
            enrich=inside / (outside + 1e-9) if not np.isnan(inside) else np.nan,
        )
    del det
    torch.cuda.empty_cache()

    # ---- Phase 3: pair base vs lora, compute the two answers ----------------
    pairs = []
    for pi in range(len(prompts)):
        for seed in seeds:
            b = records[("base", pi, seed)]
            lo = records[("lora", pi, seed)]
            pairs.append(
                {
                    "pi": pi,
                    "seed": seed,
                    "base_excess": b["excess"],
                    "lora_excess": lo["excess"],
                    "d_excess": lo["excess"] - b["excess"],
                    "base_fingers": b["total_fingers"],
                    "lora_fingers": lo["total_fingers"],
                    "base_hand_wander": b["hand_wander"],
                    "lora_hand_wander": lo["hand_wander"],
                    "d_hand_wander": (lo["hand_wander"] - b["hand_wander"])
                    if not (np.isnan(lo["hand_wander"]) or np.isnan(b["hand_wander"]))
                    else np.nan,
                    "lora_enrich": lo["enrich"],
                    "base_enrich": b["enrich"],
                }
            )

    def m(key, sub=None):
        vals = [pr[key] for pr in pairs]
        if sub:
            vals = [v for v, pr in zip(vals, pairs) if sub(pr)]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    # A. degradation
    base_excess = m("base_excess")
    lora_excess = m("lora_excess")
    n_worse = sum(1 for pr in pairs if pr["d_excess"] > 0)
    n_better = sum(1 for pr in pairs if pr["d_excess"] < 0)
    # B. locality + link
    lora_enrich = m("lora_enrich")
    base_enrich = m("base_enrich")
    link = pearson(
        np.array([pr["lora_hand_wander"] for pr in pairs], float),
        np.array([pr["lora_excess"] for pr in pairs], float),
    )
    paired_link = pearson(
        np.array([pr["d_hand_wander"] for pr in pairs], float),
        np.array([pr["d_excess"] for pr in pairs], float),
    )

    overall = {
        "n_pairs": len(pairs),
        "base_excess_fingers_mean": base_excess,
        "lora_excess_fingers_mean": lora_excess,
        "lora_minus_base_excess": lora_excess - base_excess,
        "pairs_lora_worse": n_worse,
        "pairs_lora_better": n_better,
        "base_total_fingers_mean": m("base_fingers"),
        "lora_total_fingers_mean": m("lora_fingers"),
        "lora_hand_wander_enrichment": lora_enrich,
        "base_hand_wander_enrichment": base_enrich,
        "corr_handwander_excess_lora": link,
        "corr_d_handwander_d_excess": paired_link,
    }
    write_result(
        run_dir,
        script=__file__,
        args=a,
        metrics={"overall": overall, "pairs": pairs},
        label=a.label,
        device=device,
    )

    print("\n=== A. DEGRADATION (does the LoRA break more hands?) ===")
    print(
        f"  excess fingers/img:  base {base_excess:.3f}  ->  lora {lora_excess:.3f}"
        f"   (Δ {lora_excess - base_excess:+.3f})"
    )
    print(f"  pairs LoRA worse than base: {n_worse}/{len(pairs)}  (better: {n_better})")
    print(
        f"  total fingers/img:   base {m('base_fingers'):.2f}  ->  "
        f"lora {m('lora_fingers'):.2f}"
    )
    print("\n=== B. MECHANISM (does early-wander locate/predict breakage?) ===")
    print(
        f"  hand-region wander enrichment (mean in-hand / out):  "
        f"base {base_enrich:.2f}  lora {lora_enrich:.2f}   (>1 ⇒ wander on hands)"
    )
    print(f"  corr(hand-region wander, excess fingers)  [lora]: {link:+.3f}")
    print(f"  corr(Δ hand-region wander, Δ excess)  [paired]:   {paired_link:+.3f}")
    print("\n  positive link ⇒ wander predicts breakage; ~0 ⇒ wander ≠ breakage axis")
    print(f"\nresult.json + img/ in: {run_dir}")


if __name__ == "__main__":
    main()
