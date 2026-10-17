#!/usr/bin/env python3
"""Phase 0 of the seed-lottery noise floor (docs/proposal/seed_lottery_noise_floor.md).

The gate question: **does CMMD separate training seeds at all?** i.e. is the
between-seed spread (`sigma_between`) larger than the sampling/eval noise floor
(`sigma_within`)? If not, seed-fishing is pointless and every CMMD-ranked
single-run A/B we've banked is unverified.

What it does
------------
Given N checkpoints trained from the same recipe with different ``--seed`` (see
the proposal for the queued ``make lora`` commands), this scores each checkpoint
with CMMD over K *validation* seeds against the **same held-out reference set**
the training runs carved out — reusing the trainer's own validation split. The
result is an ``N x K`` CMMD matrix reduced to:

    per_seed_cmmd  = M.mean(axis=1)               # one mean per training seed
    sigma_between  = std(per_seed_cmmd)           # the floor for A/Bs
    sigma_within   = median_n( std_k(M[n]) )      # can CMMD even see a seed diff?
    cov_between    = sigma_between / mean(M)       # headline floor number
    cov_ratio      = sigma_between / sigma_within  # GATE: > ~1.5 → build Phase 1

This is the inner loop of ``library/training/validation.py::_try_cmmd_validation``
lifted out of the trainer and wrapped in two loops (checkpoints x val-seeds). No
new training infra — it reuses ``build_anima``, ``sample_image_to_tensor``,
``library.training.cmmd``, and the PE encoder.

Three correctness points baked in (see proposal "gotchas"):
  1. **Disjoint per-item seeds across validation seeds.** Validation samples item
     ``i`` with ``seed_base + i``; naively stepping seed_base by 1 overlaps
     adjacent val-seeds and deflates ``sigma_within``. We use
     ``seed = vseed * SEED_STRIDE + i`` with a large stride so the per-item draws
     are disjoint — this is what makes ``sigma_within`` honest.
  2. **Identical held-out set across all checkpoints.** Built once from the
     trainer split and reused for every checkpoint.
  3. **Compile with dynamic-seq (default on, ``--no-compile`` to opt out).**
     Free-fit gives held-outs many shapes, so we compile via the training path
     WITH ``dynamic_seq`` (token budget derived from the held-out buckets) — each
     tier collapses to one block graph and the ~1.2x sampling speedup amortizes
     across the N*K passes. The DiT↔PE CPU round-trip is the same pattern the
     trainer's compiled CMMD validation already uses.

Usage
-----
    python bench/seed_lottery/probe.py \
        --method lora --preset default \
        --checkpoints output/ckpt/seedfloor_lora_s0.safetensors \
                      output/ckpt/seedfloor_lora_s1.safetensors \
                      output/ckpt/seedfloor_lora_s2.safetensors \
                      output/ckpt/seedfloor_lora_s3.safetensors \
                      output/ckpt/seedfloor_lora_s4.safetensors \
        --k_val 5 --validation_split_num 16 --label artist_200

``--validation_split_num`` / ``--validation_seed`` MUST match the training runs so
the held-out set is the same one the checkpoints never saw.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import train as train_mod  # noqa: E402  (the trainer module — args + split builders)
from anima_lora import load_vae  # noqa: E402
from bench._anima import build_anima  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import strategy as strategy_anima  # noqa: E402
from library.anima import training as anima_train_utils  # noqa: E402
from library.runtime.accelerator import prepare_accelerator  # noqa: E402
from library.runtime.device import clean_memory_on_device, str_to_dtype  # noqa: E402
from library.training.cmmd import (  # noqa: E402
    cmmd_from_pools,
    load_reference_features,
    pool_and_normalize,
    resolve_pe_sidecar,
)
from library.training.validation import (  # noqa: E402
    _build_val_crossattn_emb,
    _load_safetensors,
)
from library.vision.encoder import (  # noqa: E402
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

# Stride between validation seeds so per-item draws (vseed*STRIDE + i) never
# overlap across vseeds — keeps sigma_within an honest sampling-noise estimate.
SEED_STRIDE = 1_000_000


# ---------------------------------------------------------------------------
# Args: reuse the trainer's own config chain so the dataset split is identical.
# ---------------------------------------------------------------------------


def build_training_args(
    method: str, preset: str, overrides: dict
) -> argparse.Namespace:
    """Build the *exact* args namespace a real training run would, then apply
    probe overrides. Mirrors ``train.py`` ``__main__``: setup_parser →
    populate_schema → parse_args(--method/--preset) → read_config_from_file."""
    parser = train_mod.setup_parser()
    train_mod._config_schema.populate_schema(
        parser, extras=train_mod.build_network_extras()
    )
    args = parser.parse_args(["--method", method, "--preset", preset])
    args = train_mod.read_config_from_file(args, parser)
    args.method = method  # read_config_from_file clears it after the merge
    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"
    # Expand the semantic cache knobs (use_text_cache → cache_text_encoder_outputs
    # etc.) the dataset/strategy code reads — train.py does this via
    # verify_training_args before touching the dataset.
    from library.config.cli_args import verify_training_args  # noqa: PLC0415

    verify_training_args(args)
    for k, v in overrides.items():
        if v is not None:
            setattr(args, k, v)
    # Probe never samples training prompts or trains; force a clean eval config.
    args.sample_prompts = None
    args.use_cmmd = True
    return args


def build_val_items(args) -> list:
    """Reuse the trainer split: build the val DatasetGroup via the same
    BlueprintGenerator / generate_dataset_group_by_blueprint path
    ``AnimaTrainer._prepare_dataset`` uses, then resolve each held-out item's
    cached TE/PE paths (no encoding — caches must already exist on disk).

    The base blueprint pins ``validation_split_num`` inside its ``[[datasets]]``
    block, and that literal wins over the argparse value (search_value takes the
    first non-None). So we inject the requested split directly onto the dataset
    dict before ``generate`` — the only place it actually takes effect.

    Returns a list of (image_info, te_npz_path, pe_sidecar_path)."""
    from library.config import loader as config_util  # noqa: PLC0415
    from library.config.io import load_dataset_config_from_base  # noqa: PLC0415

    strategy_anima.setup_training_strategies(args)  # dataset init reads these
    te_strategy = strategy_anima.setup_text_encoder_outputs_caching_strategy(args)
    if te_strategy is None:
        raise RuntimeError(
            "cache_text_encoder_outputs is off — the probe reads cached TE/PE "
            "outputs and cannot run in live-encoding mode."
        )

    user_config = load_dataset_config_from_base(
        overrides=vars(args),
        method=getattr(args, "method", None),
        methods_subdir=getattr(args, "methods_subdir", None) or "methods",
    )
    if user_config is None:
        raise RuntimeError("No dataset blueprint found in configs/base.toml.")
    if not args.validation_split_num or args.validation_split_num <= 0:
        raise RuntimeError(
            "Set --validation_split_num (>0) to match the training runs so the "
            "held-out set exists."
        )
    for ds in user_config.get("datasets", []):
        ds["validation_split_num"] = int(args.validation_split_num)
        if args.validation_seed is not None:
            ds["validation_seed"] = int(args.validation_seed)

    blueprint_generator = config_util.BlueprintGenerator(
        config_util.ConfigSanitizer(support_dropout=True)
    )
    blueprint = blueprint_generator.generate(user_config, args)
    _train_group, val_group = config_util.generate_dataset_group_by_blueprint(
        blueprint.dataset_group, target_res=getattr(args, "target_res", None)
    )
    if val_group is None:
        raise RuntimeError(
            "Validation split is empty — the pool may be below the "
            "min-train-images-for-validation floor, or the blueprint carved "
            "nothing. Check --validation_split_num."
        )

    items: list = []
    for ds in val_group.datasets:
        for info in ds.image_data.values():
            subset = ds.image_to_subset.get(info.image_key)
            # Path resolution copied from base.py::new_cache_text_encoder_outputs
            # (the path-setting lines, minus the encode pass).
            te_npz = te_strategy.get_outputs_npz_path(
                info.absolute_path,
                cache_dir=(
                    getattr(subset, "text_cache_dir", None)
                    or getattr(subset, "cache_dir", None)
                ),
                image_dir=getattr(subset, "image_dir", None),
            )
            info.text_encoder_outputs_npz = te_npz
            if not Path(te_npz).exists():
                continue
            pe_sidecar = resolve_pe_sidecar(
                info.absolute_path, encoder="pe", cache_dir=str(Path(te_npz).parent)
            )
            if not Path(pe_sidecar).exists():
                continue
            items.append((info, te_npz, pe_sidecar))
    return items


# ---------------------------------------------------------------------------
# CMMD over K validation seeds for one fixed checkpoint.
# ---------------------------------------------------------------------------


class _Acc:
    """Minimal stand-in for the bits of ``Accelerator`` the sampler reads
    (``device`` + ``autocast``). build_anima already loaded the model to its
    own device; we just need a thin shim for ``sample_image_to_tensor``."""

    def __init__(self, device, dtype):
        self.device = device
        self._dtype = dtype

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self._dtype)


def _token_budget(items: list):
    """(n_token_families, seq_range) from the held-out buckets — mirrors
    ``train.py::_derive_token_budget`` so dynamic-seq compile collapses each
    tier to one block graph instead of a per-shape cascade."""
    from library.datasets.buckets import token_counts_for_resos  # noqa: PLC0415

    resos = {tuple(info.bucket_reso) for (info, _t, _p) in items}
    if not resos:
        return None, None
    counts = token_counts_for_resos(resos)
    if not counts:
        return None, None
    return len(counts), (min(counts), max(counts))


@torch.no_grad()
def cmmd_for_checkpoint(
    *,
    ckpt: str,
    args,
    acc,
    vae,
    pe_bundle,
    items: list,
    ref_pool: torch.Tensor,
    vseeds: list,
    sample_steps: int,
    cfg_scale: float,
    flow_shift: float,
    compile_budget=None,
):
    """Load one checkpoint, return (list_of_K_cmmd, first_vseed_pixel_grid).

    Two-phase per vseed (like _try_cmmd_validation): generate every sample with
    the DiT resident, park pixels on CPU, then swap DiT→CPU / PE→GPU and encode.
    """
    bundle = build_anima(args, adapter=ckpt, train_mode=False)
    dit = bundle.anima
    device = bundle.device

    # Compile last (post apply_to/load_weights — the build_anima invariant). Use
    # the training compile path WITH dynamic_seq so free-fit's many held-out
    # shapes collapse to one block graph per tier (see CLAUDE.md free-fit notes).
    if compile_budget is not None:
        from library.runtime.harness import compile_blocks_for_training  # noqa: PLC0415

        n_families, seq_range = compile_budget
        compile_blocks_for_training(
            dit,
            bundle.network,
            backend=getattr(args, "dynamo_backend", "inductor"),
            mode=getattr(args, "compile_inductor_mode", None),
            n_token_families=n_families,
            seq_range=seq_range,
            dynamic_seq=bool(getattr(args, "compile_dynamic_seq", True)),
            grad_ckpt=False,
        )

    cmmd_values: list[float] = []
    contact_row: list[torch.Tensor] = []  # decoded pixels from the first vseed

    try:
        for vi, vseed in enumerate(vseeds):
            pixel_images: list[torch.Tensor] = []
            with acc.autocast():
                if hasattr(dit, "prepare_block_swap_before_forward"):
                    dit.prepare_block_swap_before_forward()
                for i, (info, te_npz, _pe) in enumerate(items):
                    sd = _load_safetensors(te_npz)
                    crossattn_emb = _build_val_crossattn_emb(dit, sd, acc)
                    bucket_w, bucket_h = info.bucket_reso
                    image = anima_train_utils.sample_image_to_tensor(
                        accelerator=acc,
                        dit=dit,
                        vae=vae,
                        height=int(bucket_h),
                        width=int(bucket_w),
                        crossattn_emb=crossattn_emb,
                        sample_steps=sample_steps,
                        guidance_scale=cfg_scale,
                        flow_shift=flow_shift,
                        seed=vseed * SEED_STRIDE + i,  # disjoint per-item draws
                        show_progress=False,
                    )
                    pixel_images.append(image.detach().cpu())
                    del image, crossattn_emb
                    clean_memory_on_device(device)

            # Hand the GPU to PE: DiT → CPU, PE → GPU.
            dit.to("cpu")
            clean_memory_on_device(device)
            pe_bundle.encoder.inner.to(device)
            try:
                bucket_groups: dict[tuple[int, int], list[int]] = {}
                for idx, img in enumerate(pixel_images):
                    key = (int(img.shape[-2]), int(img.shape[-1]))
                    bucket_groups.setdefault(key, []).append(idx)
                pooled: list[torch.Tensor | None] = [None] * len(pixel_images)
                for indices in bucket_groups.values():
                    batch = torch.stack(
                        [pixel_images[idx] for idx in indices], dim=0
                    ).to(device)
                    feats_list = encode_pe_from_imageminus1to1(
                        pe_bundle, batch, same_bucket=True
                    )
                    for idx, feats in zip(indices, feats_list):
                        pooled[idx] = pool_and_normalize(feats).cpu()
                    del batch, feats_list
                gen_pool = torch.stack([t for t in pooled if t is not None], dim=0)
            finally:
                pe_bundle.encoder.inner.to("cpu")
                clean_memory_on_device(device)
                dit.to(device)

            cmmd_value = cmmd_from_pools(ref_pool, gen_pool.to(device))
            cmmd_values.append(float(cmmd_value))
            if vi == 0:
                contact_row = pixel_images[: min(4, len(pixel_images))]
            print(
                f"  {Path(ckpt).name}  vseed={vseed}  CMMD={cmmd_value:.3f}",
                flush=True,
            )
    finally:
        del bundle, dit
        clean_memory_on_device(acc.device)

    return cmmd_values, contact_row


# ---------------------------------------------------------------------------
# Contact sheet + main.
# ---------------------------------------------------------------------------


def save_contact_sheet(rows: list[list[torch.Tensor]], path: Path) -> bool:
    """Grid of decoded samples: one row per checkpoint, up to 4 columns, all at
    the first validation seed. The gate also requires the rendered ranking not be
    visibly random — this is the artifact for that human check."""
    try:
        import torchvision.utils as tvu  # noqa: PLC0415

        # Resize every tile to a common size so heterogeneous free-fit shapes
        # tile cleanly.
        tile = 256
        cells: list[torch.Tensor] = []
        ncol = max((len(r) for r in rows), default=0)
        if ncol == 0:
            return False
        for row in rows:
            for c in range(ncol):
                if c < len(row):
                    img = row[c].float().clamp(-1, 1)  # [3,H,W] in [-1,1]
                    img = (img + 1) / 2
                    img = torch.nn.functional.interpolate(
                        img.unsqueeze(0),
                        size=(tile, tile),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                else:
                    img = torch.zeros(3, tile, tile)
                cells.append(img)
        grid = tvu.make_grid(torch.stack(cells), nrow=ncol, padding=2)
        tvu.save_image(grid, str(path))
        return True
    except Exception as exc:  # contact sheet is best-effort, never fatal
        print(f"  (contact sheet skipped: {exc})", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="lora")
    ap.add_argument("--preset", default="default")
    ap.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="The N per-seed checkpoints (one per training --seed).",
    )
    ap.add_argument(
        "--k_val", type=int, default=5, help="Number of validation seeds (K>=2)."
    )
    ap.add_argument(
        "--val_seed_base",
        type=int,
        default=1,
        help="First validation seed; the K seeds are base..base+K-1.",
    )
    ap.add_argument(
        "--validation_split_num",
        type=int,
        default=None,
        help="MUST match the training runs so the held-out set is identical.",
    )
    ap.add_argument("--validation_seed", type=int, default=None)
    ap.add_argument(
        "--max_val_items",
        type=int,
        default=None,
        help="Cap held-out items (cost control); default uses the whole split.",
    )
    ap.add_argument("--sample_steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=True,
        help="torch.compile DiT blocks with dynamic-seq (default on — ~1.2x "
        "faster sampling, amortized across the N*K passes).",
    )
    ap.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Disable compile and run eager.",
    )
    ap.add_argument("--label", default=None)
    ap.add_argument(
        "--gate_ratio",
        type=float,
        default=1.5,
        help="cov_ratio threshold above which Phase 1 is worth building.",
    )
    args_cli = ap.parse_args()

    if args_cli.k_val < 2:
        ap.error("--k_val must be >= 2 (sigma_within needs >=2 validation seeds).")
    if len(args_cli.checkpoints) < 2:
        ap.error("Need >= 2 checkpoints to measure sigma_between.")

    dtype = str_to_dtype(args_cli.dtype)

    # 1. Trainer-faithful args + held-out split.
    targs = build_training_args(
        args_cli.method,
        args_cli.preset,
        {
            "validation_split_num": args_cli.validation_split_num,
            "validation_seed": args_cli.validation_seed,
            "validation_sample_steps": args_cli.sample_steps,
            "validation_cfg_scale": args_cli.cfg,
        },
    )
    acc = prepare_accelerator(targs)
    items = build_val_items(targs)
    if args_cli.max_val_items is not None:
        items = items[: args_cli.max_val_items]
    if not items:
        raise RuntimeError(
            "No held-out items with cached TE + PE-Core sidecars. CMMD needs "
            "'{stem}_anima_pe.safetensors' (PE-Core global features), distinct "
            "from the '_anima_pe_spatial' REPA sidecars — run `make preprocess-pe` "
            "on this dataset. Also confirm --validation_split_num matches the runs."
        )
    print(f"Held-out items: {len(items)}", flush=True)

    # 2. Reference PE pool (real held-out images) — fixed across all checkpoints.
    ref_pool = load_reference_features([pe for (_i, _t, pe) in items]).to(acc.device)

    # 3. PE encoder (parked on CPU between encodes, like validation.py).
    pe_bundle = load_pe_encoder(acc.device)
    pe_bundle.encoder.inner.to("cpu")

    flow_shift = float(getattr(targs, "discrete_flow_shift", 3.0))
    vseeds = list(
        range(args_cli.val_seed_base, args_cli.val_seed_base + args_cli.k_val)
    )

    # VAE for decoding samples only — loaded once, parked on CPU between decodes
    # (sample_image_to_tensor shuttles it to GPU per decode and back).
    vae = load_vae(targs.vae, dtype=dtype, eval=True, vae_2d=True, device="cpu")

    compile_budget = _token_budget(items) if args_cli.compile else None
    if compile_budget is not None:
        print(
            f"compile ON — token families={compile_budget[0]} "
            f"seq_range={compile_budget[1]} (dynamic_seq)",
            flush=True,
        )

    # 4. N x K CMMD matrix.
    matrix: list[list[float]] = []
    contact_rows: list[list[torch.Tensor]] = []
    for ckpt in args_cli.checkpoints:
        row, contact = cmmd_for_checkpoint(
            ckpt=ckpt,
            args=targs,
            acc=acc,
            vae=vae,
            pe_bundle=pe_bundle,
            items=items,
            ref_pool=ref_pool,
            vseeds=vseeds,
            sample_steps=args_cli.sample_steps,
            cfg_scale=args_cli.cfg,
            flow_shift=flow_shift,
            compile_budget=compile_budget,
        )
        matrix.append(row)
        contact_rows.append(contact)

    # 5. Reduce to the gate numbers.
    M = torch.tensor(matrix)  # [N, K]
    per_seed_cmmd = M.mean(dim=1)
    grand_mean = float(M.mean())
    sigma_between = float(per_seed_cmmd.std(unbiased=True))
    per_ckpt_within = M.std(dim=1, unbiased=True)
    sigma_within = float(median([float(x) for x in per_ckpt_within]))
    cov_between = sigma_between / grand_mean if grand_mean else float("nan")
    cov_ratio = sigma_between / sigma_within if sigma_within > 0 else float("inf")
    gate_fires = bool(cov_ratio > args_cli.gate_ratio)
    verdict = "GATE_FIRES (build Phase 1)" if gate_fires else "GATE_FAILS"

    print("\n=== Seed-lottery Phase 0 ===")
    print(f"per_seed_cmmd  = {[round(float(x), 3) for x in per_seed_cmmd]}")
    print(f"sigma_between  = {sigma_between:.4f}")
    print(f"sigma_within   = {sigma_within:.4f}  (median over checkpoints)")
    print(f"cov_between    = {cov_between:.4%}   (headline floor)")
    print(f"cov_ratio      = {cov_ratio:.2f}     (gate at {args_cli.gate_ratio})")
    print(f"VERDICT        = {verdict}")
    print(
        "NB: confirm the rendered-sample ranking isn't visibly random "
        "(contact_sheet.png) before trusting GATE_FIRES."
    )

    # 6. Envelope + artifact.
    run_dir = make_run_dir("seed_lottery", label=args_cli.label)
    has_sheet = save_contact_sheet(contact_rows, run_dir / "contact_sheet.png")
    write_result(
        run_dir,
        script=__file__,
        args=args_cli,
        metrics={
            "sigma_between": sigma_between,
            "sigma_within": sigma_within,
            "cov_between": cov_between,
            "cov_ratio": cov_ratio,
            "n_train": len(args_cli.checkpoints),
            "k_val": args_cli.k_val,
            "n_val_items": len(items),
            "per_seed_cmmd": [float(x) for x in per_seed_cmmd],
            "cmmd_matrix": matrix,
            "vseeds": vseeds,
            "checkpoints": [Path(c).name for c in args_cli.checkpoints],
            "gate_ratio": args_cli.gate_ratio,
            "gate_fires": gate_fires,
            "verdict": "REAL" if gate_fires else "INCONCLUSIVE",
        },
        label=args_cli.label,
        artifacts=["contact_sheet.png"] if has_sheet else [],
        device=acc.device,
    )
    print(f"\nWrote {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
