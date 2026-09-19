"""Phase 0b — learn a SAM3 soft prompt (textual inversion) for anime subjects.

Only the post-tower prompt tensor ``language_features`` ``(32, 1, 256)`` moves;
SAM3 stays frozen end to end (trunk under `no_grad`, fusion encoder + decoder
traversed by the backward pass but never updated). Init from a real phrase
(``--init``), supervise against the pseudo-targets from `build_targets.py`
with the usual DETR recipe — Hungarian matching, sigmoid focal objectness with
presence, L1 + GIoU boxes, BCE + dice masks — and hold out ``--val_frac`` of
the clean pool for a loss/recall readout.

Output: ``results/<ts>-<label>/soft_prompt.safetensors`` (+ checkpoints), a
`result.json` envelope. Evaluate with `probe_nms_pairs.py --prompt_embed …`
and `ab_sam3_prompt.py`.

    make daemon-run ARGS="bench/sam3_soft_prompt/train_soft_prompt.py --label animegirl-init"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from torchvision.ops import generalized_box_iou  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from bench.sam3_soft_prompt.common import (  # noqa: E402
    PROMPT_KEYS,
    box_cxcywh_to_xyxy,
    encode_image,
    encode_text,
    ground,
    install_prompt,
    load_sam3,
    nms,
    proposals,
    save_soft_prompt,
)
from library.env import resolve_under_home  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--targets", default="post_image_dataset/captions/sam3_soft_prompt/targets"
    )
    p.add_argument("--dst", default="post_image_dataset/resized")
    p.add_argument(
        "--extra_manifest",
        action="append",
        default=[],
        help="extra target manifests (e.g. pair_negatives.py) whose train rows "
        "join the train pool only — never the val split",
    )
    p.add_argument("--init", default="anime girl", help="phrase the prompt starts from")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--accum", type=int, default=4, help="images per optimizer step")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument(
        "--max_train", type=int, default=0, help="cap the train pool (0 = all)"
    )
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--w_ce", type=float, default=2.0)
    p.add_argument("--w_presence", type=float, default=1.0)
    p.add_argument("--w_l1", type=float, default=5.0)
    p.add_argument("--w_giou", type=float, default=2.0)
    p.add_argument("--w_mask", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--l2", type=float, default=0.0, help="pull toward the init prompt")
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--label", default=None)
    return p.parse_args()


def focal(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float
) -> torch.Tensor:
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * loss).sum()


def dice(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = pred_logits.sigmoid().flatten(1)
    t = target.flatten(1)
    num = 2 * (p * t).sum(1)
    den = p.sum(1) + t.sum(1)
    return (1 - (num + 1) / (den + 1)).sum()


def detr_loss(out: dict, tgt_boxes: torch.Tensor, tgt_masks: torch.Tensor, a) -> dict:
    """One image. ``tgt_boxes`` normalized xyxy ``(N,4)``, ``tgt_masks`` ``(N,288,288)`` float."""
    logits = out["pred_logits"][0].float().squeeze(-1)  # (Q,)
    boxes_c = out["pred_boxes"][0].float()  # (Q,4) cxcywh
    boxes = box_cxcywh_to_xyxy(boxes_c)
    masks = out["pred_masks"][0].float()  # (Q,288,288)
    presence = out["presence_logit_dec"].float().flatten()  # (1,)
    n = tgt_boxes.shape[0]

    with torch.no_grad():
        prob = logits.sigmoid()
        cost = (
            -prob[:, None]
            + a.w_l1 * torch.cdist(boxes, tgt_boxes, p=1)
            - a.w_giou * generalized_box_iou(boxes, tgt_boxes)
        )
        qi, ti = linear_sum_assignment(cost.cpu().numpy())
    qi = torch.as_tensor(qi, device=logits.device)
    ti = torch.as_tensor(ti, device=logits.device)

    cls_t = torch.zeros_like(logits)
    cls_t[qi] = 1.0
    loss_ce = focal(logits, cls_t, a.focal_alpha, a.focal_gamma) / n
    loss_presence = F.binary_cross_entropy_with_logits(
        presence, torch.ones_like(presence)
    )
    loss_l1 = F.l1_loss(boxes[qi], tgt_boxes[ti], reduction="sum") / n
    loss_giou = (
        1 - torch.diag(generalized_box_iou(boxes[qi], tgt_boxes[ti]))
    ).sum() / n
    pm, tm = masks[qi], tgt_masks[ti]
    loss_mask = F.binary_cross_entropy_with_logits(pm, tm) + dice(pm, tm) / n
    total = (
        a.w_ce * loss_ce
        + a.w_presence * loss_presence
        + a.w_l1 * loss_l1
        + a.w_giou * loss_giou
        + a.w_mask * loss_mask
    )
    return {
        "total": total,
        "ce": loss_ce.detach(),
        "presence": loss_presence.detach(),
        "l1": loss_l1.detach(),
        "giou": loss_giou.detach(),
        "mask": loss_mask.detach(),
    }


def load_target(path: Path, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    z = np.load(path)
    return (
        torch.as_tensor(z["boxes"], device=device),
        torch.as_tensor(z["masks"], device=device).float(),
    )


@torch.no_grad()
def evaluate(model, processor, prompt, items, dst, a) -> dict:
    """Val loss + a recall proxy: survivors@0.5 == n with every fill ≥ 0.2."""
    tot: dict[str, float] = {}
    hits = 0
    for rel, tpath in items:
        with Image.open(dst / rel) as h:
            image = h.convert("RGB")
        bo = install_prompt(encode_image(model, processor, image), prompt)
        out = ground(model, processor, bo)
        tb, tm = load_target(tpath, a.device)
        losses = detr_loss(out, tb, tm, a)
        for k, v in losses.items():
            tot[k] = tot.get(k, 0.0) + float(v.detach())
        surv = nms([r for r in proposals(out, 0.5) if r["area_frac"] >= 0.005])
        hits += int(len(surv) == tb.shape[0] and all(r["fill"] >= 0.2 for r in surv))
    m = {k: v / max(1, len(items)) for k, v in tot.items()}
    m["clean_recall"] = hits / max(1, len(items))
    return m


def main() -> None:
    a = parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dst = resolve_under_home(a.dst)
    tdir = resolve_under_home(a.targets)
    manifest = json.loads((tdir / "manifest.json").read_text(encoding="utf-8"))
    items = [
        (r["image"], tdir / r["target"]) for r in manifest["rows"] if r.get("train")
    ]
    random.shuffle(items)
    n_val = min(max(1, int(len(items) * a.val_frac)), len(items) // 2)
    val, train = items[:n_val], items[n_val:]
    if a.max_train:
        train = train[: a.max_train]
    for extra in a.extra_manifest:
        edir = resolve_under_home(extra).parent
        em = json.loads(resolve_under_home(extra).read_text(encoding="utf-8"))
        added = [(r["image"], edir / r["target"]) for r in em["rows"] if r.get("train")]
        print(f"extra {extra}: +{len(added)} train rows", flush=True)
        train += added
    print(f"train {len(train)}  val {len(val)}", flush=True)

    run_dir = make_run_dir("sam3_soft_prompt", a.label)
    model, processor = load_sam3(resolve_under_home(a.checkpoint), a.device)
    init = encode_text(model, a.init, a.device)
    feats0 = init["language_features"].float().clone()
    delta = torch.zeros_like(feats0, requires_grad=True)
    prompt = {k: init[k] for k in PROMPT_KEYS}
    opt = torch.optim.Adam([delta], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    print(
        f"init |feats| = {feats0.norm():.3f}  valid tokens = {int((~init['language_mask']).sum())}"
    )

    def current_prompt():
        return {**prompt, "language_features": feats0 + delta}

    log: list[dict] = []
    metrics_val = evaluate(model, processor, current_prompt(), val, dst, a)
    print(
        f"[0] val {json.dumps({k: round(v, 4) for k, v in metrics_val.items()})}",
        flush=True,
    )
    log.append({"step": 0, "val": metrics_val})

    order: list = []
    t0 = time.time()
    run_loss: dict[str, float] = {}
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            if not order:
                order = train[:]
                random.shuffle(order)
            rel, tpath = order.pop()
            with Image.open(dst / rel) as h:
                image = h.convert("RGB")
            bo = install_prompt(encode_image(model, processor, image), current_prompt())
            out = ground(model, processor, bo)
            tb, tm = load_target(tpath, a.device)
            losses = detr_loss(out, tb, tm, a)
            loss = losses["total"] / a.accum
            if a.l2:
                loss = loss + a.l2 * delta.pow(2).sum() / a.accum
            loss.backward()
            for k, v in losses.items():
                run_loss[k] = run_loss.get(k, 0.0) + float(v.detach()) / a.accum
        opt.step()
        sched.step()
        if step % 25 == 0:
            avg = {k: round(v / 25, 4) for k, v in run_loss.items()}
            avg["|delta|"] = round(float(delta.norm()), 3)
            avg["it/s"] = round(step / (time.time() - t0), 2)
            print(f"[{step}] {json.dumps(avg)}", flush=True)
            log.append({"step": step, "train": avg})
            run_loss = {}
        if step % a.eval_every == 0 or step == a.steps:
            metrics_val = evaluate(model, processor, current_prompt(), val, dst, a)
            print(
                f"[{step}] val {json.dumps({k: round(v, 4) for k, v in metrics_val.items()})}",
                flush=True,
            )
            log.append({"step": step, "val": metrics_val})
            save_soft_prompt(
                run_dir / f"soft_prompt_{step:05d}.safetensors",
                current_prompt(),
                {"init": a.init, "step": step, "val": metrics_val, "args": vars(a)},
            )

    final = run_dir / "soft_prompt.safetensors"
    save_soft_prompt(
        final,
        current_prompt(),
        {"init": a.init, "step": a.steps, "val": metrics_val, "args": vars(a)},
    )
    (run_dir / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    write_result(
        run_dir,
        script=__file__,
        args=a,
        metrics={
            "val_final": metrics_val,
            "val_init": log[0]["val"],
            "train": len(train),
            "val": len(val),
        },
        label=a.label,
        artifacts=[final, run_dir / "log.json"],
    )
    print(f"wrote: {final}")


if __name__ == "__main__":
    main()
