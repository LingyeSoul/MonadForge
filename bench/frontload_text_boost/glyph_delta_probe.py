#!/usr/bin/env python3
"""Does a glyph clause change the CFG-delta (cond - uncond) curve? — measurement probe.

Feeds this bench's open gate. The Phase-0/1 verdict left the *localized
feature-commit-band* boost demand-gated on "a regime that demonstrably DROPS
small features" (`report.md`, and `docs/findings/crossattn_self_attn_dominance.md`
Result 4 — rendered glyphs are the canonical such regime). This probe asks the
prior question: **when a caption carries a glyph to render, does the model put
any extra text drive into the feature-commit band at all?**

It answers with the production CfgDelta probe (`ANIMA_CFG_DELTA_LOG`,
`library/inference/cfg_delta_probe.py`) — no hooks, no bench-local sampler.

Design — five arms appended to each real caption, paired by (caption, seed):

    A_bare      <caption>                                                   0 tok
    B_category  + ", there is a speech bubble"                              6 tok
    C_glyph     + ', ... that reads "say hello from anima_lora"'           20 tok
    D_glyph_id  + ', ... that reads "But it was fast"'                     15 tok
    E_nonglyph  + ", ... that is round and white and slightly tilted"      16 tok

Why five. A vs C alone confounds three things: a clause was appended, the clause
is longer, and the clause carries a glyph. The controls split them:

  * **D vs E is the load-bearing contrast** — 15 vs 16 T5 tokens, both extend B by
    a relative clause, only D carries a glyph. Length-matched glyph-vs-not.
  * E vs B isolates pure token count (10 extra tokens, no glyph).
  * C vs D isolates the OOD token axis: `anima_lora` shreds into 6 subtokens
    (`▁/anim/a/_/lor/a`) where `But it was fast` is 4 clean words.

E is a *conservative* control: "round and white and slightly tilted" specifies
visual attributes, so it carries some rendering demand of its own. A positive
D − E therefore understates the glyph effect.

Read the PAIRED deltas, not the per-arm means: cross-prompt std of `ratio` is
~0.078 at sigma=1 (`_archive/bench/cross_attn_drive/report.md`) and swamps a
~0.004 effect; the paired std is ~0.0025.

Read `rel_dDelta` (the numerator, ||v_cond - v_uncond||), not `ratio`, when
attributing: a ratio move could come from the ||v_cond|| denominator instead of
the guidance delta itself. Both are reported.

Run (base DiT, ~28 min for 140 gens on a 16 GB card — one GPU job at a time):

    python bench/frontload_text_boost/glyph_delta_probe.py --label glyph-delta

Analyze an existing log without regenerating:

    python bench/frontload_text_boost/glyph_delta_probe.py --log <cfg_delta.json>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from bench._common import make_run_dir, write_result

REPO_ROOT = Path(__file__).resolve().parents[2]
NAME = "glyph_delta_probe"

GLYPH_OOD = "say hello from anima_lora"
GLYPH_ID = (
    "But it was fast"  # the bench/turbo glyph string; plain in-distribution words
)

ARMS: dict[str, object] = {
    "A_bare": lambda c: c,
    "B_category": lambda c: f"{c}, there is a speech bubble",
    "C_glyph": lambda c: f'{c}, there is a speech bubble that reads "{GLYPH_OOD}"',
    "D_glyph_id": lambda c: f'{c}, there is a speech bubble that reads "{GLYPH_ID}"',
    "E_nonglyph": lambda c: (
        f"{c}, there is a speech bubble that is round and white and slightly tilted"
    ),
}

# T5 tokens of the appended clause (the cond embedding rows are T5 target
# positions, not Qwen3 — see render_compare.py::_t5_span_indices). Recomputed
# and asserted by --check-tokens; hardcoded so the read is legible here.
CLAUSE_TOKENS = {
    "A_bare": 0,
    "B_category": 6,
    "C_glyph": 20,
    "D_glyph_id": 15,
    "E_nonglyph": 16,
}

PAIRS = [
    ("B_category", "A_bare"),  # does appending any clause change drive?
    ("E_nonglyph", "B_category"),  # pure token-count effect (no glyph)
    ("D_glyph_id", "E_nonglyph"),  # LENGTH-MATCHED glyph vs non-glyph  <-- the key one
    ("D_glyph_id", "B_category"),  # plain-word glyph vs category
    ("C_glyph", "B_category"),  # OOD glyph vs category
    ("C_glyph", "D_glyph_id"),  # OOD token axis (both glyphs)
]

# The band the bump lives in = the feature-commit band from the knockout diff-map
# (localized prompt-specified features commit through sigma 0.6-0.8).
BAND = (0.5, 0.8)

INFER_BASE = [
    "inference.py",
    "--dit", "models/diffusion_models/anima-base-v1.0.safetensors",
    "--text_encoder", "models/text_encoders/qwen_3_06b_base.safetensors",
    "--vae", "models/vae/qwen_image_vae.safetensors",
    "--vae_chunk_size", "64",
    "--vae_disable_cache",
    "--attn_mode", "flash",
    "--negative_prompt",
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    "--image_size", "1024", "1024",
    "--infer_steps", "28",
    "--flow_shift", "3.0",
    "--sampler", "euler",
    "--guidance_scale", "4.0",
]  # fmt: skip


def build_prompts(
    captions_path: Path, seeds: list[int]
) -> tuple[list[str], list[dict]]:
    """Return (prompt lines, manifest). Drops captions that already tag a bubble."""
    src = [c.strip() for c in captions_path.read_text().splitlines() if c.strip()]
    caps = [c for c in src if "speech bubble" not in c.lower()]
    if not caps:
        raise SystemExit(f"no usable captions in {captions_path}")

    lines: list[str] = []
    manifest: list[dict] = []
    for ci, cap in enumerate(caps):
        for arm, build in ARMS.items():
            for seed in seeds:
                prompt = build(cap)  # type: ignore[operator]
                # parse_prompt_line splits on " --"; a prompt containing it would
                # silently become a flag.
                assert " --" not in prompt, (
                    f"prompt breaks parse_prompt_line: {prompt!r}"
                )
                lines.append(f"{prompt} --d {seed}")
                manifest.append(
                    {"gen": len(manifest), "caption_idx": ci, "arm": arm, "seed": seed}
                )
    print(
        f"captions: {len(caps)} kept / {len(src) - len(caps)} dropped (already bubble-tagged)"
    )
    return lines, manifest


def check_tokens() -> None:
    """Assert the hardcoded T5 clause lengths — D/E length-matching is the whole point."""
    from library.anima import weights as anima_utils

    tok = anima_utils.load_t5_tokenizer(None)
    bare = "x"
    print("T5 tokens of the appended clause:")
    for arm, build in ARMS.items():
        clause = build(bare)[len(bare) :].lstrip(", ")  # type: ignore[operator]
        n = len(tok(clause, add_special_tokens=False)["input_ids"]) if clause else 0
        flag = (
            ""
            if n == CLAUSE_TOKENS[arm]
            else f"  <-- MISMATCH (table says {CLAUSE_TOKENS[arm]})"
        )
        print(f"  {n:>3}  {arm}{flag}")
        assert n == CLAUSE_TOKENS[arm], (
            f"{arm}: CLAUSE_TOKENS stale ({n} != {CLAUSE_TOKENS[arm]})"
        )
    d, e = CLAUSE_TOKENS["D_glyph_id"], CLAUSE_TOKENS["E_nonglyph"]
    assert abs(d - e) <= 2, (
        f"D/E no longer length-matched ({d} vs {e}) — D-E contrast is void"
    )
    print(f"D/E length-matched: {d} vs {e} tokens  OK")


def run_generation(prompt_file: Path, log: Path, save_path: Path) -> None:
    argv = [
        sys.executable,
        *INFER_BASE,
        "--from_file",
        str(prompt_file),
        "--save_path",
        str(save_path),
    ]
    env = {**os.environ, "ANIMA_CFG_DELTA_LOG": str(log)}
    print(
        f"$ ANIMA_CFG_DELTA_LOG={log} {' '.join(argv[:3])} ... --from_file {prompt_file}"
    )
    subprocess.run(argv, cwd=REPO_ROOT, env=env, check=True)


def analyze(log: Path, manifest: list[dict]) -> dict:
    blob = json.loads(log.read_text())
    gens = {g["gen"]: g["steps"] for g in blob["generations"]}
    cell: dict[tuple, dict[str, list]] = {}
    for m in manifest:
        if m["gen"] in gens:
            cell.setdefault((m["caption_idx"], m["seed"]), {})[m["arm"]] = gens[
                m["gen"]
            ]
    complete = {k: v for k, v in cell.items() if all(a in v for a in ARMS)}
    if not complete:
        raise SystemExit("no complete paired cells — was the run cut short?")
    n_steps = min(len(s) for s in gens.values())
    sigma = [
        next(iter(complete.values()))["A_bare"][i]["sigma"] for i in range(n_steps)
    ]
    band = [i for i in range(n_steps) if BAND[0] <= sigma[i] <= BAND[1]]
    print(f"\ngenerations {len(gens)} / paired cells {len(complete)} / steps {n_steps}")

    def mean_over(arm: str, key: str, i: int) -> float:
        return statistics.fmean([v[arm][i][key] for v in complete.values()])

    # ---- per-arm mean ratio curve ----
    print("\n=== per-arm mean ratio (||v_cond - v_uncond|| / ||v_cond||) ===")
    print(f"{'sigma':>7} |" + "".join(f"{a:>12}" for a in ARMS))
    print(f"{'clausetok':>7} |" + "".join(f"{CLAUSE_TOKENS[a]:>12}" for a in ARMS))
    for i in range(n_steps):
        print(
            f"{sigma[i]:>7.4f} |"
            + "".join(f"{mean_over(a, 'ratio', i):>12.4f}" for a in ARMS)
        )

    # ---- paired deltas + band summary ----
    print(
        f"\n=== PAIRED band summary (sigma in [{BAND[0]}, {BAND[1]}], n={len(complete)}) ==="
    )
    print(f"{'pair':>30} {'rel_dDelta':>11} {'rel_dVcond':>11} {'n_pos':>8}")
    metrics: dict[str, dict] = {}
    for hi, lo in PAIRS:
        rel_d, rel_v, npos = [], [], 0
        for v in complete.values():
            hd = statistics.fmean([v[hi][i]["delta_norm"] for i in band])
            ld = statistics.fmean([v[lo][i]["delta_norm"] for i in band])
            hv = statistics.fmean([v[hi][i]["v_cond_norm"] for i in band])
            lv = statistics.fmean([v[lo][i]["v_cond_norm"] for i in band])
            rel_d.append((hd - ld) / ld * 100)
            rel_v.append((hv - lv) / lv * 100)
            npos += hd > ld
        key = f"{hi}-{lo}"
        metrics[key] = {
            "rel_delta_pct": statistics.fmean(rel_d),
            "rel_vcond_pct": statistics.fmean(rel_v),
            "n_pos": npos,
            "n": len(rel_d),
        }
        print(
            f"{key:>30} {statistics.fmean(rel_d):>10.2f}% {statistics.fmean(rel_v):>10.2f}% "
            f"{str(npos) + '/' + str(len(rel_d)):>8}"
        )

    per_arm = {
        a: {
            "peak_ratio": max(mean_over(a, "ratio", i) for i in range(n_steps)),
            "band_ratio": statistics.fmean([mean_over(a, "ratio", i) for i in band]),
            "min_cos": min(mean_over(a, "cos_cond_uncond", i) for i in range(n_steps)),
            "band_cos": statistics.fmean(
                [mean_over(a, "cos_cond_uncond", i) for i in band]
            ),
        }
        for a in ARMS
    }
    return {
        "pairs": metrics,
        "per_arm": per_arm,
        "n_cells": len(complete),
        "band": list(BAND),
        "curve": {
            "sigma": sigma,
            "ratio": {
                a: [mean_over(a, "ratio", i) for i in range(n_steps)] for a in ARMS
            },
            "cos": {
                a: [mean_over(a, "cos_cond_uncond", i) for i in range(n_steps)]
                for a in ARMS
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--captions",
        type=Path,
        default=REPO_ROOT / "bench/turbo/real_prompts.txt",
        help="one real caption per line (synthetic prompts under-drive this probe)",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1234])
    ap.add_argument("--label", default="glyph-delta")
    ap.add_argument(
        "--log",
        type=Path,
        default=None,
        help="analyze this existing CfgDelta log; skip generation",
    )
    ap.add_argument(
        "--check-tokens",
        action="store_true",
        help="verify the T5 clause lengths and exit",
    )
    args = ap.parse_args()

    if args.check_tokens:
        check_tokens()
        return

    run_dir = make_run_dir("frontload_text_boost", args.label)
    lines, manifest = build_prompts(args.captions, args.seeds)
    prompt_file = run_dir / "prompts.txt"
    prompt_file.write_text("\n".join(lines) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"arms {list(ARMS)} x seeds {args.seeds} -> {len(lines)} generations")

    log = run_dir / "cfg_delta.json"
    if args.log is None:
        run_generation(prompt_file, log, run_dir / "renders")
    else:
        # keep the run dir self-contained even when re-analyzing an old log
        log.write_text(args.log.read_text())

    metrics = analyze(log, manifest)
    curve = metrics.pop("curve")
    (run_dir / "curve.json").write_text(json.dumps(curve, indent=2))
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["prompts.txt", "manifest.json", "curve.json", "cfg_delta.json"],
        extra={
            "clause_tokens": CLAUSE_TOKENS,
            "glyph_ood": GLYPH_OOD,
            "glyph_id": GLYPH_ID,
        },
    )
    print(f"\nwrote {run_dir}/result.json")


if __name__ == "__main__":
    main()
