#!/usr/bin/env python3
"""Region adapter — realistic use: real *safe* images, real SAM paints, prompt cases.

``run_bench.py`` scores placement on synthetic plates + ellipse blobs. This
sibling asks the question a user asks: take a **real safe-rated image**, paint
the character the way the training paints were made (SAM3 girl mask →
tight / slack / face augment, gray 128 over the real background), and run
several **prompt cases** through the adapter — does the result look like a
usable edit?

Per image × paint level the prompt cases are

  * ``own``        — the image's own caption (reconstruct the girl in place)
  * ``positioned`` — ``own`` + ``On the <pos>, <character>.`` clause (named girls only)
  * ``swap``       — identity swap: character/copyright + hair/eye tags replaced
                     by ``--persona`` (default silver hair / red eyes / short hair)
  * ``minimal``    — ``safe, 1girl, solo`` (+``1boy`` on pairs): scene-only cue
  * ``action``     — ``own`` with pose tags stripped + ``--action`` tags appended

plus one ``control`` sample per image (own caption, **unpainted** image as cond)
to show what the adapter does with nothing to regenerate.

Metrics (per sample, SAM3 ``girl`` on the output):

  * girl_in_paint / area_ratio / center_dist   — as in run_bench (paint = the
    real augmented mask, so ``iou`` vs the *source girl mask* is also scored:
    on tight paints a high ``iou_src`` means the silhouette was kept)
  * bg_psnr            — vs the REAL source image outside dilate(paint ∪ girl)
                          (a real reference now, unlike the synthetic plates)
  * inpaint_bg_psnr    — vs the source inside paint − dilate(girl) (slack/face)
  * tagger             — Anima Tagger on the output: ``rating`` (stays safe?),
                          ``persona_hit`` (swap: fraction of persona tags
                          predicted), ``char_hit`` (own/positioned: the
                          character tag predicted), ``tag_recall`` (fraction
                          of prompt tags predicted — a coarse adherence read)

Images come from the region staging pool (``post_image_dataset/easycontrol/
region``) so the SAM masks + report records exist — i.e. they are TRAINING
images (the whole solo-1girl corpus was staged; there is no held-out safe
slice). Pass ``--keys artist/stem …`` to pick specific ones.

Run (GPU — through the daemon)::

    make daemon-run ARGS="--label region-real bench/region/run_real.py --label v5-real"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))  # run_bench sibling

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from bench._common import REPO_ROOT, make_run_dir, write_result  # noqa: E402
from run_bench import (  # noqa: E402
    PAINT_COLOR,
    _centroid,
    _decode_pending,
    _latest_region_adapter,
    _load_mask,
)

REGION_BASE = REPO_ROOT / "post_image_dataset" / "easycontrol" / "region"
CAPTION_INDEX = REPO_ROOT / "post_image_dataset" / "captions" / "caption_index.json"

CASES = ("own", "positioned", "swap", "minimal", "action")
PAINTS = ("tight", "slack", "face")
FACE_CASES = ("own", "swap")  # face paint = body visible; position/pose cases are moot

# Pose/action tags stripped for the ``action`` case (the paint carries the
# silhouette on tight paints; on slack the prompt is free to change it).
POSE_TAGS = {
    "standing",
    "sitting",
    "lying",
    "kneeling",
    "squatting",
    "walking",
    "running",
    "on back",
    "on side",
    "on stomach",
    "all fours",
    "arms up",
    "arm up",
    "hand up",
    "hands up",
    "arms behind back",
    "hands on hips",
    "hand on hip",
    "crossed arms",
    "leaning forward",
    "looking back",
    "from behind",
    "from side",
    "from above",
    "from below",
    "waving",
    "v",
    "double v",
    "peace sign",
    "arms at sides",
    "hand on own chest",
    "hands on own chest",
    "outstretched arms",
    "outstretched arm",
    "reaching",
    "spread legs",
    "crossed legs",
    "legs up",
    "wariza",
    "seiza",
    "indian style",
    "hugging own legs",
    "knees up",
}


def _prep():
    """Lazy import — ``easycontrol_adapters`` is not a package (sys.path bootstrap)."""
    sys.path.insert(0, str(REPO_ROOT / "easycontrol_adapters" / "region"))
    import prep  # type: ignore

    return prep


# ── selection ───────────────────────────────────────────────────────────────


def _first_tag(caption: str) -> str:
    return caption.split(",", 1)[0].strip().lower()


def select_images(args) -> list[dict]:
    select = json.loads((REGION_BASE / "select.json").read_text(encoding="utf-8"))
    report = json.loads((REGION_BASE / "report.json").read_text(encoding="utf-8"))
    records = {r["image"]: r for r in report["records"]}
    meta = json.loads(CAPTION_INDEX.read_text(encoding="utf-8"))["image_meta"]

    pool = []
    for key, sel in sorted(select.items()):
        cap = REGION_BASE / "staging" / f"{key}.txt"
        if key not in records or not cap.is_file():
            continue
        caption = cap.read_text(encoding="utf-8").strip()
        if _first_tag(caption) != "safe":
            continue
        pool.append(
            {
                "key": key,
                "slice": sel["slice"],
                "caption": caption,
                "character": list(meta.get(key, {}).get("character") or []),
                "copyright": list(meta.get(key, {}).get("copyright") or []),
                "record": records[key],
            }
        )
    if args.keys:
        by_key = {p["key"]: p for p in pool}
        missing = [k for k in args.keys if k not in by_key]
        if missing:
            raise SystemExit(f"--keys not in the safe staged pool: {missing}")
        return [by_key[k] for k in args.keys]

    # Deterministic, representative: named girls first (they exercise the
    # positioned + swap cases), pairs kept if any, then fill.
    rng = random.Random(args.select_seed)
    rng.shuffle(pool)
    named = [p for p in pool if len(p["character"]) == 1]
    pairs = [p for p in pool if p["slice"] == "pair"]
    rest = [p for p in pool if p not in named and p not in pairs]
    picked: list[dict] = []
    for bucket in (pairs, named, rest):
        for p in bucket:
            if p not in picked and len(picked) < args.n_images:
                picked.append(p)
    return picked


# ── paints ──────────────────────────────────────────────────────────────────


def build_paints(args, info: dict, size_wh: tuple[int, int]) -> dict[str, np.ndarray]:
    """{paint_level: binary mask} from the staged SAM masks, training-style."""
    prep = _prep()
    import cv2

    key = info["key"]
    girl = _load_mask(REGION_BASE / "masks" / f"{key}_mask.png", size_wh)
    if girl is None or not girl.any():
        return {}
    girl = prep._largest_component(girl)
    rng = random.Random(prep._stable_seed(key))
    out: dict[str, np.ndarray] = {}
    tight = prep._augment_mask(girl, rng, 1)  # smooth — the most common level
    out["tight"] = tight
    slack = prep._augment_mask(
        girl,
        rng,
        4,
        slack_grow=tuple(args.slack_grow),
        max_slack_coverage=args.max_slack_coverage,
    )
    out["slack"] = slack if slack is not None else tight
    head = prep._load_mask(REGION_BASE / "masks_head" / f"{key}_mask.png", size_wh)
    if head is not None and head.any():
        edge = max(size_wh)
        k = prep._odd(int(edge * 0.01))
        face = prep._largest_component(
            head & cv2.dilate(girl, np.ones((k, k), np.uint8))
        )
        frac = face.mean()
        if frac >= args.min_face_frac and face.sum() <= 0.7 * girl.sum():
            out["face"] = prep._augment_mask(face, rng, 1)
    out["_girl"] = girl
    return out


# ── prompts ─────────────────────────────────────────────────────────────────


def _pos_word(info: dict) -> str:
    return info["record"].get("position") or "middle"


def build_prompts(args, info: dict) -> dict[str, str]:
    from anime_tools.captions.position_clauses import (
        PositionClause,
        compose_caption,
        parse_caption,
    )

    parsed = parse_caption(info["caption"])
    flat = list(parsed.flat_tags)
    names = info["character"]
    prompts = {"own": compose_caption(flat, parsed.clauses)}

    if len(names) == 1 and not parsed.has_clauses:
        clause = PositionClause(position=_pos_word(info), tags=(names[0],))
        prompts["positioned"] = compose_caption(flat, (clause,))

    drop = {t.lower() for t in names + info["copyright"]}
    swap_flat = [
        t
        for t in flat
        if t.lower() not in drop
        and not t.lower().endswith(" hair")
        and not t.lower().endswith(" eyes")
        and t.lower()
        not in {"twintails", "ponytail", "ahoge", "braid", "side ponytail"}
    ]
    prompts["swap"] = compose_caption(swap_flat + list(args.persona), ())

    minimal = ["safe", "1girl", "solo"]
    if info["slice"] == "pair":
        minimal = ["safe", "1girl", "1boy"]
    prompts["minimal"] = compose_caption(minimal, ())

    action_flat = [t for t in flat if t.lower() not in POSE_TAGS]
    prompts["action"] = compose_caption(action_flat + list(args.action), ())
    return prompts


# ── plan ────────────────────────────────────────────────────────────────────


def build_plan(args, infos: list[dict], run_dir: Path) -> list[dict]:
    conds = run_dir / "conds"
    conds.mkdir(exist_ok=True)
    plan = []
    for ii, info in enumerate(infos):
        key = info["key"]
        tag = key.replace("/", "__")
        src_path = REGION_BASE / "staging" / f"{key}.png"
        src = Image.open(src_path).convert("RGB")
        size_wh = src.size
        src_np = np.array(src)
        src.save(conds / f"{tag}_src.png")
        paints = build_paints(args, info, size_wh)
        if not paints:
            print(f"[select] {key}: no girl mask — skipped")
            continue
        Image.fromarray(paints.pop("_girl") * 255).save(conds / f"{tag}_girl.png")
        prompts = build_prompts(args, info)
        (conds / f"{tag}_prompts.json").write_text(
            json.dumps(prompts, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        base = dict(
            key=key,
            tag=tag,
            slice=info["slice"],
            size=[size_wh[1], size_wh[0]],  # (H, W)
            seed=1000 + 17 * ii,
        )
        plan.append(
            dict(
                base,
                paint="none",
                case="control",
                prompt=prompts["own"],
                cond=f"{tag}_src.png",
                name=f"{tag}__control",
            )
        )
        for level, mask in paints.items():
            Image.fromarray(mask * 255).save(conds / f"{tag}_{level}_mask.png")
            canvas = src_np.copy()
            canvas[mask > 0] = PAINT_COLOR
            cond_name = f"{tag}_{level}_cond.png"
            Image.fromarray(canvas).save(conds / cond_name)
            cases = FACE_CASES if level == "face" else CASES
            for case in cases:
                if case not in prompts:
                    continue
                plan.append(
                    dict(
                        base,
                        paint=level,
                        case=case,
                        prompt=prompts[case],
                        cond=cond_name,
                        name=f"{tag}__{level}__{case}",
                    )
                )
    (run_dir / "plan.json").write_text(
        json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    return plan


# ── generate ────────────────────────────────────────────────────────────────


def phase_generate(args, run_dir: Path, plan: list[dict]) -> None:
    from anima_lora import (
        GenerationRequest,
        default_checkpoints,
        generate,
        get_generation_settings,
    )
    from library.runtime.device import clean_memory_on_device

    ckpt = default_checkpoints()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = run_dir / "images"
    images_dir.mkdir(exist_ok=True)

    shared: dict = {}
    pending: list[tuple] = []
    for i, s in enumerate(plan):
        out = images_dir / f"{s['name']}.png"
        if out.is_file():
            print(f"[{i + 1}/{len(plan)}] skip {s['name']} (exists)")
            continue
        request = GenerationRequest(
            dit=ckpt.dit,
            vae=ckpt.vae,
            text_encoder=ckpt.text_encoder,
            prompt=s["prompt"],
            save_path=str(out),
            infer_steps=args.steps,
            guidance_scale=args.cfg,
            image_size=tuple(s["size"]),
            seed=s["seed"],
            attn_mode=args.attn_mode,
            easycontrol_weight=str(args.adapter),
            easycontrol_image=str(run_dir / "conds" / s["cond"]),
            extra_argv=(
                ["--easycontrol_b_offset", str(args.b_offset)]
                if args.b_offset is not None
                else []
            ),
        )
        gen_args = request.to_args()
        gen_args.device = device
        print(f"[{i + 1}/{len(plan)}] {s['name']}  «{s['prompt'][:70]}»")
        latent = generate(
            gen_args, get_generation_settings(gen_args), shared_models=shared
        )
        pending.append((gen_args, latent.to("cpu"), s["name"]))
        anima = shared.get("model")
        network = getattr(anima, "_easycontrol_network", None)
        if network is not None:  # unpatch — see run_bench.phase_generate
            network.remove_from()
            anima._easycontrol_network = None
        clean_memory_on_device(device)
    shared.clear()
    clean_memory_on_device(device)
    _decode_pending(pending, images_dir, device)


def phase_segment(run_dir: Path) -> None:
    cfg = {
        "prompts": [],
        "focus_prompts": ["girl"],
        "threshold": 0.4,
        "dilate": 0,
        "path_pattern": "*",
    }
    cfg_path = run_dir / "sam_bench.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "anime_tools.masking.cli.generate_masks",
            "--config",
            str(cfg_path),
            "--image-dir",
            str(run_dir / "images"),
            "--mask-dir",
            str(run_dir / "masks"),
            "--checkpoint",
            "models/sam3/sam3.pt",
            "--batch-size",
            "4",
            "--recursive",
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def phase_tag(run_dir: Path, plan: list[dict]) -> dict[str, dict]:
    """Anima Tagger over every output → {name: {rating, kept:[tags]}} (cached)."""
    cache = run_dir / "tags.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    from anime_tools.tagger.tagger import (
        DEFAULT_TAGGER_DIR,
        AnimaTagger,
        ensure_tagger_checkpoint,
    )

    tagger = AnimaTagger(ensure_tagger_checkpoint(DEFAULT_TAGGER_DIR))
    out: dict[str, dict] = {}
    for s in plan:
        p = run_dir / "images" / f"{s['name']}.png"
        if not p.is_file():
            continue
        pred = tagger.predict(Image.open(p))
        out[s["name"]] = {
            "rating": pred.get("rating"),
            "kept": sorted(pred["kept"]),
        }
    cache.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    del tagger
    torch.cuda.empty_cache()
    return out


# ── metrics ─────────────────────────────────────────────────────────────────


def _psnr(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float | None:
    if region.sum() < 64:
        return None
    d = (a[region > 0] - b[region > 0]) ** 2
    return float(10 * np.log10(255.0**2 / max(d.mean(), 1e-6)))


def phase_metrics(
    args, run_dir: Path, plan: list[dict], tags: dict
) -> tuple[dict, list[dict]]:
    import cv2

    from anime_tools.captions.position_clauses import parse_caption

    persona = {t.lower() for t in args.persona}
    rows: list[dict] = []
    for s in plan:
        size_wh = (s["size"][1], s["size"][0])
        diag = float(np.hypot(*size_wh))
        row = {k: v for k, v in s.items() if k != "prompt"}
        girl = _load_mask(run_dir / "masks" / f"{s['name']}_mask.png", size_wh)
        row["found"] = girl is not None and bool(girl.any())
        src = np.array(
            Image.open(run_dir / "conds" / f"{s['tag']}_src.png").convert("RGB"),
            dtype=np.float32,
        )
        out_p = run_dir / "images" / f"{s['name']}.png"
        out = np.array(
            Image.open(out_p).convert("RGB").resize(size_wh), dtype=np.float32
        )
        src_girl = _load_mask(run_dir / "conds" / f"{s['tag']}_girl.png", size_wh)
        k = max(3, int(max(size_wh) * 0.01) | 1)
        kern = np.ones((k, k), np.uint8)
        if s["paint"] != "none":
            paint = _load_mask(
                run_dir / "conds" / f"{s['tag']}_{s['paint']}_mask.png", size_wh
            )
            row["paint_area_frac"] = float(paint.mean())
            if row["found"]:
                inter = int((girl & paint).sum())
                pcx, pcy = _centroid(paint)
                gcx, gcy = _centroid(girl)
                row.update(
                    girl_in_paint=inter / int(girl.sum()),
                    area_ratio=int(girl.sum()) / int(paint.sum()),
                    center_dist=float(np.hypot(pcx - gcx, pcy - gcy)) / diag,
                    iou_paint=inter / int((girl | paint).sum()),
                    iou_src=int((girl & src_girl).sum()) / int((girl | src_girl).sum()),
                )
                dg = cv2.dilate(girl, kern)
            else:
                dg = np.zeros_like(paint)
            outside = 1 - cv2.dilate(paint | dg, kern)
            row["bg_psnr"] = _psnr(out, src, outside)
            inside_bg = (paint > 0) & (dg == 0)
            if inside_bg.sum() >= 0.02 * paint.sum():
                row["inpaint_bg_psnr"] = _psnr(out, src, inside_bg.astype(np.uint8))
        else:
            row["recon_psnr"] = _psnr(out, src, np.ones(src.shape[:2], np.uint8))
            if row["found"]:
                row["iou_src"] = int((girl & src_girl).sum()) / int(
                    (girl | src_girl).sum()
                )

        t = tags.get(s["name"])
        if t:
            kept = {x.lower() for x in t["kept"]}
            row["rating"] = t["rating"]
            prompt_tags = {x.lower() for x in parse_caption(s["prompt"]).flat_tags}
            prompt_tags -= {
                "safe",
                "sensitive",
                "nsfw",
                "explicit",
                "1girl",
                "solo",
                "1boy",
            }
            prompt_tags = {x for x in prompt_tags if not x.startswith("@")}
            if prompt_tags:
                row["tag_recall"] = len(prompt_tags & kept) / len(prompt_tags)
            if s["case"] == "swap":
                row["persona_hit"] = len(persona & kept) / len(persona)
            info_chars = {c.lower() for c in _plan_chars(s, run_dir)}
            if info_chars and s["case"] in ("own", "positioned", "control", "action"):
                row["char_hit"] = float(bool(info_chars & kept))
            elif info_chars and s["case"] == "swap":
                row["char_leak"] = float(bool(info_chars & kept))
        rows.append(row)

    def _mean(key, pred=lambda r: True):
        vals = [r[key] for r in rows if pred(r) and r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    metrics: dict = {"n_samples": len(rows), "n_images": len({r["key"] for r in rows})}
    metrics["found_rate"] = _mean("found", lambda r: r["paint"] != "none")
    rated = [r["rating"] == "safe" for r in rows if "rating" in r]
    metrics["safe_rate"] = float(np.mean(rated)) if rated else None
    per: dict = {}
    for paint in ("none",) + PAINTS:
        for case in ("control",) + CASES:
            sub = [r for r in rows if r["paint"] == paint and r["case"] == case]
            if not sub:
                continue
            pred = lambda r, p=paint, c=case: r["paint"] == p and r["case"] == c  # noqa: E731
            per[f"{paint}/{case}"] = {
                "n": len(sub),
                "found": _mean("found", pred),
                "girl_in_paint": _mean("girl_in_paint", pred),
                "area_ratio": _mean("area_ratio", pred),
                "iou_src": _mean("iou_src", pred),
                "bg_psnr": _mean("bg_psnr", pred),
                "inpaint_bg_psnr": _mean("inpaint_bg_psnr", pred),
                "recon_psnr": _mean("recon_psnr", pred),
                "tag_recall": _mean("tag_recall", pred),
                "char_hit": _mean("char_hit", pred),
                "persona_hit": _mean("persona_hit", pred),
                "char_leak": _mean("char_leak", pred),
                "safe": float(
                    np.mean([r.get("rating") == "safe" for r in sub if "rating" in r])
                )
                if any("rating" in r for r in sub)
                else None,
            }
    metrics["per_cell"] = per
    for key in (
        "girl_in_paint",
        "area_ratio",
        "iou_src",
        "bg_psnr",
        "inpaint_bg_psnr",
        "tag_recall",
        "persona_hit",
        "char_hit",
        "char_leak",
    ):
        for paint in PAINTS:
            v = _mean(key, lambda r, p=paint: r["paint"] == p)
            if v is not None:
                metrics[f"{key}_{paint}"] = v
    return metrics, rows


_CHARS_CACHE: dict[str, list[str]] = {}


def _plan_chars(s: dict, run_dir: Path) -> list[str]:
    if not _CHARS_CACHE:
        meta = json.loads(CAPTION_INDEX.read_text(encoding="utf-8"))["image_meta"]
        for r in json.loads((run_dir / "plan.json").read_text(encoding="utf-8")):
            _CHARS_CACHE[r["key"]] = list(meta.get(r["key"], {}).get("character") or [])
    return _CHARS_CACHE.get(s["key"], [])


# ── contact sheets ──────────────────────────────────────────────────────────


def contact_sheets(run_dir: Path, plan: list[dict]) -> None:
    """One sheet per image: rows = paint level, cols = source · cond · cases."""
    import cv2

    cell = 256
    sheets = run_dir / "contact"
    sheets.mkdir(exist_ok=True)
    by_tag: dict[str, list[dict]] = {}
    for s in plan:
        by_tag.setdefault(s["tag"], []).append(s)

    def load(p: Path, contour: np.ndarray | None = None) -> np.ndarray:
        img = np.array(Image.open(p).convert("RGB").resize((cell, cell)))
        if contour is not None:
            m = cv2.resize(contour, (cell, cell), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, (255, 0, 0), 2)
        return img

    def label(img: np.ndarray, text: str) -> np.ndarray:
        img = img.copy()
        cv2.rectangle(img, (0, 0), (cell, 18), (0, 0, 0), -1)
        cv2.putText(
            img, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
        )
        return img

    index = ["<html><body style='background:#111;color:#ddd;font-family:sans-serif'>"]
    blank = np.zeros((cell, cell, 3), np.uint8)
    for tag, samples in by_tag.items():
        src = run_dir / "conds" / f"{tag}_src.png"
        rows_img = []
        control = next((s for s in samples if s["case"] == "control"), None)
        for level in PAINTS:
            lvl = [s for s in samples if s["paint"] == level]
            if not lvl:
                continue
            mask = (
                np.array(
                    Image.open(run_dir / "conds" / f"{tag}_{level}_mask.png").convert(
                        "L"
                    )
                )
                > 127
            ).astype(np.uint8)
            cells = [
                label(load(src), "source"),
                label(
                    load(run_dir / "conds" / f"{tag}_{level}_cond.png"), f"cond:{level}"
                ),
            ]
            if control is not None:
                p = run_dir / "images" / f"{control['name']}.png"
                cells.append(
                    label(load(p) if p.exists() else blank, "control (no paint)")
                )
            for case in CASES:
                s = next((x for x in lvl if x["case"] == case), None)
                if s is None:
                    cells.append(label(blank, f"{case}: n/a"))
                    continue
                p = run_dir / "images" / f"{s['name']}.png"
                cells.append(label(load(p, mask) if p.exists() else blank, case))
            rows_img.append(np.concatenate(cells, axis=1))
        if not rows_img:
            continue
        width = max(r.shape[1] for r in rows_img)
        rows_img = [
            np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows_img
        ]
        Image.fromarray(np.concatenate(rows_img, axis=0)).save(sheets / f"{tag}.png")
        prompts = json.loads(
            (run_dir / "conds" / f"{tag}_prompts.json").read_text(encoding="utf-8")
        )
        index.append(f"<h3>{tag}</h3><img src='{tag}.png' style='max-width:100%'><pre>")
        index.extend(f"{k:11s} {v}" for k, v in prompts.items())
        index.append("</pre>")
    index.append("</body></html>")
    (sheets / "index.html").write_text("\n".join(index), encoding="utf-8")


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="default: newest anima_easycontrol_region*",
    )
    ap.add_argument("--n_images", type=int, default=8)
    ap.add_argument(
        "--keys", nargs="*", default=None, help="explicit staged keys (artist/stem)"
    )
    ap.add_argument("--select_seed", type=int, default=0)
    ap.add_argument(
        "--persona",
        nargs="+",
        default=["silver hair", "red eyes", "short hair", "hair ornament"],
        help="identity tags for the swap case",
    )
    ap.add_argument(
        "--action",
        nargs="+",
        default=["waving", "hand up", "open mouth", "smile", "looking at viewer"],
        help="tags appended for the action case (pose tags stripped first)",
    )
    ap.add_argument("--slack_grow", type=float, nargs=2, default=[1.5, 3.0])
    ap.add_argument("--max_slack_coverage", type=float, default=0.7)
    ap.add_argument("--min_face_frac", type=float, default=0.004)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--attn_mode", type=str, default="flash")
    ap.add_argument("--b_offset", type=float, default=None)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--run_dir", type=Path, default=None)
    ap.add_argument("--skip_generate", action="store_true")
    ap.add_argument("--skip_segment", action="store_true")
    ap.add_argument("--skip_tag", action="store_true")
    args = ap.parse_args()

    if args.adapter is None:
        args.adapter = _latest_region_adapter()
    args.adapter = args.adapter.resolve()
    run_dir = args.run_dir or make_run_dir("region", args.label)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}\nadapter: {args.adapter}")

    plan_path = run_dir / "plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        plan = build_plan(args, select_images(args), run_dir)
    print(f"{len(plan)} samples over {len({s['key'] for s in plan})} images")

    if not args.skip_generate:
        phase_generate(args, run_dir, plan)
    if not args.skip_segment:
        phase_segment(run_dir)
    tags = {} if args.skip_tag else phase_tag(run_dir, plan)
    metrics, rows = phase_metrics(args, run_dir, plan, tags)
    contact_sheets(run_dir, plan)

    with (run_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_result(run_dir, script=__file__, args=args, metrics=metrics, label=args.label)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_cell"}, indent=1))
    for cell, m in metrics["per_cell"].items():
        print(
            f"  {cell:18s} "
            + "  ".join(f"{k}={v:.2f}" for k, v in m.items() if isinstance(v, float))
        )
    print(f"contact sheets: {run_dir / 'contact' / 'index.html'}")


if __name__ == "__main__":
    main()
