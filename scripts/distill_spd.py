"""SPD fine-tuning LoRA — trajectory adapter for progressive-resolution inference.

Trains a *plain* LoRA on one frozen Anima DiT to follow the stage-specific
straight-line velocity targets of the Spectral Progressive Diffusion (SPD)
multi-resolution trajectory (Xiao et al., arXiv:2605.18736, §4.3, Eq. 11–14).
This is "Case B" of the SPD investigation — see
``_archive/proposals/spd_finetune_lora.md``. Output ``output/ckpt/anima_spd.safetensors``
is a normal LoRA: load it through the standard inference path and run it with
the SPD sampler (``--spd``) at the *same* schedule it was trained on.

Models the structure on ``scripts/distill_mod/distill.py`` /
``scripts/distill_turbo/distill.py`` (frozen-DiT + adapter-only + single MSE backward),
but strictly simpler: one adapter, one optimizer.

Usage::

    make exp-spd                                  # defaults from spd.toml
    make exp-spd ARGS="--iterations 2000 --single_prompt_idx 0"   # Phase 0
    make exp-spd PRESET=low_vram                  # block swap + grad ckpt
    make exp-spd ARGS="--torch_compile"           # per-stage static-shape compile

"""

from __future__ import annotations

import contextlib
import json
import logging
import random
from pathlib import Path


import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from tqdm import tqdm  # noqa: E402

from library.anima import weights as anima_utils  # noqa: E402
from library.anima.models import Anima  # noqa: E402
from library.datasets.cache import make_cached_collate  # noqa: E402
from library.datasets.cache import CachedDataset  # noqa: E402
from library.runtime.harness import (  # noqa: E402
    compile_dit_blocks_for_pool,
    enable_training_grad_ckpt,
    place_dit_for_training,
)
from library.training.accumulator import ScalarAccumulator  # noqa: E402
from library.training.distill_runtime import (  # noqa: E402
    apply_single_prompt_slice,
    create_tb_writer,
    ensure_dynamic_seq_for_freefit,
    resolve_device_dtype,
)
from library.training.forward import PadCache, renoise, to_dit_5d  # noqa: E402
from library.training.schedulers import make_warmup_cosine_scheduler  # noqa: E402
from networks.lora_anima.factory import create_network  # noqa: E402
from networks.lora_save import save_network_weights  # noqa: E402
from networks.spd import (  # noqa: E402
    SpdSnrGate,
    _snap,
    dct_lowpass_init,
    measure_dct_power_profile,
    spd_rollout_to_stage,
    spd_schedule_bands,
    spd_stage_target,
    spectral_expand,
)
from library.io.cache import get_latent_resolution, load_cached_latents  # noqa: E402
from scripts.distill_spd_config import (  # noqa: E402
    build_argparser,
    load_toml,
    resolve_config,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    args = build_argparser().parse_args()
    c = resolve_config(args, load_toml(args.config))

    # Unpack the resolved frozen config to locals (the training loop below is
    # unchanged). The argparser + CLI/TOML precedence + schedule sanity now live
    # in scripts/distill_spd_config.py; the few raw flags the loop still reads
    # (single_prompt_idx / dry_run / torch_compile / blocks_to_swap / grad_ckpt /
    # sample_ratio / dynamo_backend / no_log) stay on ``args``.
    dit_path = c.dit_path
    data_dir = c.data_dir
    output_dir = c.output_dir
    output_name = c.output_name
    iterations = c.iterations
    batch_size = c.batch_size
    seed = c.seed
    rank = c.rank
    alpha = c.alpha
    attn_mode = c.attn_mode
    channel_scaling_alpha = c.channel_scaling_alpha
    compile_inductor_mode = c.compile_inductor_mode
    compile_dynamic_seq = c.compile_dynamic_seq
    activation_memory_budget = c.activation_memory_budget
    stages = c.stages
    transition_sigmas = c.transition_sigmas
    schedule_label = c.schedule_label
    sigma_jitter = c.sigma_jitter
    stage_weights = c.stage_weights
    lr = c.lr
    weight_decay = c.weight_decay
    grad_clip = c.grad_clip
    grad_accum = c.grad_accum
    warmup = c.warmup
    save_every = c.save_every
    log_interval = c.log_interval
    log_dir = c.log_dir
    val_split = c.val_split
    val_interval = c.val_interval
    n_val_sigmas = c.n_val_sigmas
    ema_decay = c.ema_decay
    onpolicy = c.onpolicy
    flow_shift = c.flow_shift
    dagger_warmup = c.dagger_warmup
    onpolicy_ratio = c.onpolicy_ratio
    rollout_steps = c.rollout_steps
    snr_gate_enabled = c.snr_gate_enabled
    snr_gate_mode = c.snr_gate_mode
    snr_gate_delta = c.snr_gate_delta
    snr_gate_profile_n = c.snr_gate_profile_n
    snr_gate_n_bins = c.snr_gate_n_bins

    torch.manual_seed(seed)

    # Schedule bands (data-independent; weights keep marginal-over-t uniform).
    bands = spd_schedule_bands(stages, transition_sigmas)
    band_widths = torch.tensor([hi - lo for (lo, hi) in bands], dtype=torch.float64)
    # Stage sampling weight = band width × per-stage multiplier. The band-width
    # factor keeps σ marginally-uniform within each sampled stage (paper U(0,1));
    # stage_weights tilt mass across stages without touching in-band σ density.
    stage_w = torch.tensor(stage_weights, dtype=torch.float64)
    stage_sample_w = band_widths * stage_w
    stage_sample_w_f = stage_sample_w.float()  # hoisted for the per-step multinomial
    stage_probs = (stage_sample_w / stage_sample_w.sum()).tolist()
    logger.info(
        "SPD schedule '%s': stages=%s transition_sigmas=%s stage_weights=%s",
        schedule_label,
        stages,
        transition_sigmas,
        stage_weights,
    )
    for i, ((lo, hi), p) in enumerate(zip(bands, stage_probs)):
        logger.info(
            "  stage %d  scale=%.3f  query σ∈(%.4f, %.4f)  w=%.3g  p=%.3f",
            i,
            stages[i],
            lo,
            hi,
            stage_weights[i],
            p,
        )

    device, dtype = resolve_device_dtype()

    # Dataset (bucket-grouped, one resolution per batch). CachedDataset carves a
    # deterministic per-bucket val slice (seeded by validation_seed) that never
    # overlaps train, mirroring the LoRA pipeline.
    dataset = CachedDataset(
        data_dir,
        batch_size=batch_size,
        sample_ratio=args.sample_ratio,
        split="train",
        validation_split=val_split,
        validation_seed=seed,
    )
    if args.single_prompt_idx is not None:
        apply_single_prompt_slice(dataset, args.single_prompt_idx, logger=logger)

    # Stacking collate (pooled-text slot returned but unused by SPD). Shared with
    # the val loader; pickle-safe under the Windows/spawn DataLoader start method.
    collate_fn = make_cached_collate()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # bucket-grouped: shuffling would mix resolutions
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Held-out val loader — same batch_size as train so the compiled (stage ×
    # bucket × B) block graphs are reused (a different B would force recompiles
    # and could blow the dynamo cache budget sized below).
    val_loader = None
    if val_split > 0.0:
        val_dataset = CachedDataset(
            data_dir,
            batch_size=batch_size,
            sample_ratio=args.sample_ratio,
            split="val",
            validation_split=val_split,
            validation_seed=seed,
        )
        if len(val_dataset) == 0:
            logger.warning(
                "val_split=%.3g produced 0 held-out samples (dataset too small "
                "for per-bucket carving at this batch_size); validation disabled.",
                val_split,
            )
        else:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_fn,
            )

    # Generator for stage construction (fresh HF noise per step; seed offset so
    # it's independent of the torch global stream used for stage selection).
    gen = torch.Generator(device=device).manual_seed(seed + 7919)

    if args.dry_run:
        for i, (_idx, lat, te, _pooled) in enumerate(tqdm(dataloader, desc="dry-run")):
            lat = lat.to(device, dtype=dtype)
            x0_full = to_dit_5d(lat)
            for s in range(len(stages)):
                x0_si, eps_si = spd_stage_target(
                    x0_full, s, stages, transition_sigmas, patch=1, gen=gen
                )
                assert x0_si.shape == eps_si.shape
            if i >= 20:
                break
        logger.info("Dry run OK: stage-target construction + collation clean.")
        return

    # SNR gate: orthonormal-DCT per-coefficient power, radially binned — measured
    # not power-law-fitted (anime latents carry line-art HF a 2-param fit smooths
    # over). Train split only; a dataset statistic, so a small sample suffices.
    snr_gate = None
    if snr_gate_enabled:
        prof_rng = random.Random(seed + 31)
        prof_paths = [npz for (npz, _te) in dataset.samples]
        prof_rng.shuffle(prof_paths)
        prof_paths = prof_paths[: max(1, snr_gate_profile_n)]
        profile = measure_dct_power_profile(
            (load_cached_latents(p)[0].to(device, torch.float32) for p in prof_paths),
            n_bins=snr_gate_n_bins,
        )
        snr_gate = SpdSnrGate(profile, mode=snr_gate_mode, delta=snr_gate_delta)
        logger.info(
            "SNR gate ON (mode=%s%s): P_w profile from %d latents, %d bins; "
            "P[lowest bin]=%.3g, P[median bin]=%.3g, P[last bin]=%.3g",
            snr_gate_mode,
            f", delta={snr_gate_delta}" if snr_gate_mode == "hard" else "",
            len(prof_paths),
            snr_gate_n_bins,
            float(profile[0]),
            float(profile[snr_gate_n_bins // 2]),
            float(profile[-1]),
        )

    logger.info("Loading DiT model...")
    model: Anima = anima_utils.load_anima_model(
        device,
        dit_path,
        attn_mode=attn_mode,
        loading_device="cpu" if args.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )
    patch = model.patch_spatial

    # Plain LoRA adapter (paper-faithful: no MoE / ortho / T-LoRA).
    if channel_scaling_alpha:
        logger.info(
            "channel_scaling enabled (alpha=%.3g); inv_scale baked at save",
            channel_scaling_alpha,
        )
    network = create_network(
        multiplier=1.0,
        network_dim=rank,
        network_alpha=alpha,
        vae=None,
        text_encoders=[],
        unet=model,
        channel_scaling_alpha=channel_scaling_alpha,
    )
    network.apply_to(
        text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True
    )

    place_dit_for_training(model, device, blocks_to_swap=args.blocks_to_swap)

    enable_training_grad_ckpt(model, enabled=args.grad_ckpt)
    model.train()

    # Freeze base DiT; only the LoRA params train. apply_to add_module'd the
    # LoRA submodules onto the unet, so a wholesale freeze then re-enabling the
    # network's own params leaves exactly the adapter trainable.
    for p in model.parameters():
        p.requires_grad_(False)
    network.to(device=device, dtype=dtype)
    network.prepare_grad_etc(None, model)  # network.requires_grad_(True)

    trainable = [p for p in network.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    logger.info(
        "trainable: %s LoRA params over %d modules",
        f"{n_train:,}",
        len(network.unet_loras),
    )

    # SPD runs each stage at a DOWNSAMPLED resolution not in CONSTANT_TOKEN_BUCKETS
    # (dct_lowpass_init snaps each latent dim to _snap(dim*scale, patch)), so real
    # forward token counts span more than the 2 full-res families and run *below*
    # them. Enumerate every (stage × on-disk bucket) token count so both compile
    # paths size to the true shape set.
    if args.torch_compile:
        # Distinct (stage × bucket) token counts in the cached pool — each image's
        # full-res latent downsampled per stage scale. This derivation is coupled
        # to SPD's multi-stage schedule, so it stays here; the budget → cache
        # isolation → block-compile glue is the shared library helper.
        stage_bucket_tokens: set[int] = set()
        full_res_tokens: set[int] = set()
        for npz, _te in dataset.samples:
            w_lat, h_lat = (int(v) for v in get_latent_resolution(npz).split("x"))
            full_res_tokens.add((w_lat // patch) * (h_lat // patch))
            for s in stages:
                h = min(_snap(h_lat * s, patch), h_lat) if s < 1.0 else h_lat
                w = min(_snap(w_lat * s, patch), w_lat) if s < 1.0 else w_lat
                stage_bucket_tokens.add((h // patch) * (w // patch))
        # Free-fit fail-safe (mirrors train.py's auto-enable, which never reaches
        # this bespoke loop — project_daemon_wiring_pattern). NB: detect on the
        # *full-res* on-disk latents, not stage_bucket_tokens — SPD's downsampling
        # already produces off-table sub-band counts on a snapped dataset, so the
        # native cached shapes are the only true free-fit signature here.
        compile_dynamic_seq = ensure_dynamic_seq_for_freefit(
            full_res_tokens, compile_dynamic_seq, logger=logger
        )
        pc = compile_dit_blocks_for_pool(
            model,
            stage_bucket_tokens,
            enabled=True,
            dynamic_seq=compile_dynamic_seq,
            backend=args.dynamo_backend,
            mode=compile_inductor_mode,
            activation_memory_budget=activation_memory_budget,
            grad_ckpt=args.grad_ckpt,
            logger=logger,
        )
        logger.info(
            "torch_compile: %d block._forward compiled (backend=%s, mode=%s, "
            "dynamic_seq=%s); %d distinct (stage x bucket) token counts in %s.",
            len(model.blocks),
            args.dynamo_backend,
            compile_inductor_mode,
            compile_dynamic_seq,
            pc.n_shapes,
            (
                f"seq_range={pc.seq_range} (one symbolic graph)"
                if compile_dynamic_seq
                else "static per-shape graphs"
            ),
        )

    optimizer = torch.optim.AdamW(
        trainable, lr=lr, weight_decay=weight_decay, fused=torch.cuda.is_available()
    )

    # EMA (optional): the shadow tracks a decaying average, so saved/validated
    # weights sit near the sweet spot without hand-picking an iteration. copy_
    # into live params (stable addresses) keeps it cudagraph-safe under reduce-overhead.
    ema_shadow = [p.detach().clone() for p in trainable] if ema_decay > 0.0 else None
    if ema_shadow is not None:
        logger.info(
            "EMA enabled (decay=%.5f); EMA weights are validated + saved.", ema_decay
        )

    @contextlib.contextmanager
    def _ema_weights():
        """Temporarily swap the EMA shadow into the live params (no-op if off)."""
        if ema_shadow is None:
            yield
            return
        backup = [p.detach().clone() for p in trainable]
        with torch.no_grad():
            for p, s in zip(trainable, ema_shadow):
                p.data.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(trainable, backup):
                    p.data.copy_(b)

    warmup_steps = int(warmup) if warmup >= 1 else int(warmup * iterations)
    scheduler = make_warmup_cosine_scheduler(
        optimizer, iterations, lr, warmup_steps=warmup_steps
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config_text = "  \n".join(
        f"{k}: {v}"
        for k, v in {
            "schedule_label": schedule_label,
            "stages": stages,
            "transition_sigmas": transition_sigmas,
            "stage_weights": stage_weights,
            "rank": rank,
            "alpha": alpha,
            "channel_scaling_alpha": channel_scaling_alpha,
            "lr": lr,
            "iterations": iterations,
            "sigma_jitter": sigma_jitter,
            "val_split": val_split,
            "val_interval": val_interval,
            "n_val_sigmas": n_val_sigmas,
            "ema_decay": ema_decay,
            "snr_gate": snr_gate_mode if snr_gate is not None else "off",
            "snr_gate_delta": snr_gate_delta,
        }.items()
    )
    writer, _run_log = create_tb_writer(
        log_dir, config_text, enabled=not args.no_log, logger=logger
    )

    def _save(step: int):
        save_path = str(Path(output_dir) / f"{output_name}.safetensors")
        with _ema_weights():
            sd = {k: v.detach().clone() for k, v in network.state_dict().items()}
        # Keep .inv_scale buffers (per_channel_scaling): the standard write path's
        # bake_inv_scale folds them into lora_down, so dropping them here would
        # silently emit a wrong delta whenever channel_scaling_alpha>0.
        sd = {
            k: v
            for k, v in sd.items()
            if ".lora_" in k or k.endswith(".alpha") or k.endswith(".inv_scale")
        }
        save_network_weights(
            sd,
            file=save_path,
            dtype=torch.bfloat16,
            metadata={
                # R2 / open-question #2: snapshot the schedule so inference can't
                # silently mismatch the geometry the LoRA learned.
                "ss_spd_stages": json.dumps(stages),
                "ss_spd_transition_sigmas": json.dumps(transition_sigmas),
                "ss_spd_stage_weights": json.dumps(stage_weights),
                "ss_spd_schedule_label": str(schedule_label),
                "ss_spd_rank": str(rank),
                "ss_channel_scaling_alpha": str(channel_scaling_alpha),
                "ss_spd_step": str(step),
                "ss_spd_onpolicy": str(onpolicy),
                "ss_spd_flow_shift": str(flow_shift),
                "ss_spd_snr_gate": snr_gate_mode if snr_gate is not None else "off",
                "ss_spd_snr_gate_delta": str(snr_gate_delta),
            },
            save_variant="standard",
        )
        logger.info("saved SPD LoRA → %s  (step %d, %d keys)", save_path, step, len(sd))

    stage_rng = torch.Generator().manual_seed(seed + 1)  # CPU: stage / mode selection

    # Pad mask recycled across forwards — a fresh allocation each call would hand
    # the compiled forward a new input address (hostile to reduce-overhead CUDA
    # graphs). cudagraph step-marking stays decoupled (once per optimizer step /
    # val pass), so _forward_dit is hand-rolled, not run_mini_train_forward
    # (which marks per forward).
    pad_cache = PadCache(dtype)

    def _forward_dit(x5, sig_vec, cattn):
        """Single conditional forward at x5's own resolution (adapter on)."""
        pad = pad_cache.get(x5)
        if model.blocks_to_swap:
            model.prepare_block_swap_before_forward()
        with torch.autocast("cuda", dtype=dtype):
            return model.forward_mini_train_dit(
                x5, sig_vec, cattn, padding_mask=pad, skip_pooled_text_proj=True
            )

    def _band_edges(stage_idx, trans):
        """(t_lo, t_hi) query band — precomputed unless σ-jitter built a fresh `trans`."""
        return (
            bands[stage_idx]
            if trans is transition_sigmas
            else spd_schedule_bands(stages, trans)[stage_idx]
        )

    def _stage_entry(x0_full, cattn, stage_idx, trans, gen_, use_onpolicy):
        """Build ``(x0_si, eps_si, t_lo, t_hi)`` for a stage.

        Analytic (default): ``spd_stage_target`` — straight line from the true
        clean LL. On-policy (stage>0): roll the adapter-on prefix from pure noise
        to the entry of ``stage_idx`` (``spd_rollout_to_stage``, no-grad), expand
        to this stage's grid, and recover the FM-consistent effective noise so the
        velocity target ``eps_si − x0_si`` still points at the *true* clean x0 —
        from the off-manifold state inference actually visits. The rollout is
        detached, so gradients flow only through the supervised forward later.
        """
        if not (use_onpolicy and stage_idx > 0):
            x0_si, eps_si = spd_stage_target(
                x0_full, stage_idx, stages, trans, patch=patch, gen=gen_
            )
            t_lo, t_hi = _band_edges(stage_idx, trans)
            return x0_si, eps_si, t_lo, t_hi

        s_hi = stages[stage_idx]
        H_full, W_full = int(x0_full.shape[-2]), int(x0_full.shape[-1])
        x0_si = dct_lowpass_init(x0_full, s_hi, patch) if s_hi < 1.0 else x0_full
        init_noise = torch.randn(
            x0_full.shape, generator=gen_, device=device, dtype=dtype
        )

        def _vfn(x5, sig):
            sig_vec = torch.full((x5.shape[0],), float(sig), device=device, dtype=dtype)
            return _forward_dit(x5, sig_vec, cattn)

        x_entry_lo, sigma_cross, scale_lo = spd_rollout_to_stage(
            _vfn,
            init_noise,
            stages,
            trans,
            infer_steps=rollout_steps,
            flow_shift=flow_shift,
            patch=patch,
            gen=gen_,
            stop_stage=stage_idx,
        )
        x_tilde, t_tilde = spectral_expand(
            x_entry_lo, sigma_cross, scale_lo, s_hi, H_full, W_full, patch, gen_
        )
        t_lo = trans[stage_idx] if stage_idx < len(stages) - 1 else 0.0
        # Degenerate crossing (rollout fell below the band) → analytic fallback.
        if t_tilde <= t_lo + 1e-6:
            x0_si, eps_si = spd_stage_target(
                x0_full, stage_idx, stages, trans, patch=patch, gen=gen_
            )
            lo, hi = _band_edges(stage_idx, trans)
            return x0_si, eps_si, lo, hi
        eps_si = (x_tilde.float() - (1.0 - t_tilde) * x0_si.float()) / t_tilde
        return x0_si, eps_si.to(dtype), t_lo, float(t_tilde)

    def _onpolicy_active(step):
        """Per-step probability a stage>0 micro-step uses the on-policy entry."""
        if not onpolicy or step < dagger_warmup:
            return 0.0
        ramp = (step - dagger_warmup) / max(1, iterations - dagger_warmup)
        return onpolicy_ratio * min(1.0, ramp)

    logger.info("Starting SPD distillation: %d iterations", iterations)
    # Under reduce-overhead (CUDA graphs), grad_accum keeps the previous step's
    # autograd outputs alive when the next step's forward begins, so inductor
    # skips the cudagraph fast path ("outputs from a previous step still require
    # backward"). Marking the step boundary lets the cudagraph tree recycle its
    # static pool each optimizer step. No-op when cudagraphs aren't active.
    cudagraph_step = bool(
        args.torch_compile and compile_inductor_mode == "reduce-overhead"
    )
    data_iter = [iter(dataloader)]  # boxed so _micro_step can refresh on exhaustion
    progress = tqdm(range(iterations), desc="spd")
    n_stages = len(stages)
    # Named GPU-side accumulators flushed in one CUDA sync per log boundary
    # (library.training.accumulator), replacing per-micro-step loss.item() syncs
    # and the per-parameter LoRA-norm .item() walk. Keys: "loss"; per-stage
    # "stage_loss"/"stage_cnt" (width n_stages); "gate_w"/"ungated" (SNR gate on);
    # "up_sq"/"down_sq" added just before flush so they ride the same sync.
    acc = ScalarAccumulator(device)

    # Fixed RNG for validation: reseeded each eval so the ε field (and hence the
    # analytic target) is identical across checkpoints → val/loss is a pure
    # function of the weights, directly comparable step-to-step. (Training's `gen`
    # advances freely instead, so each epoch sees a fresh ε per image — the target
    # is an expectation there, which is the regularizing behaviour we want.)
    val_gen = torch.Generator(device=device)

    @torch.no_grad()
    def _validate():
        """Deterministic held-out velocity-MSE over every stage × a fixed σ grid.

        Sweeps the *full* validation set, all stages, at ``n_val_sigmas`` fixed
        band-midpoints per stage — no sampling, no PE-Core, same memory footprint
        as one training micro-step (won't OOM where CMMD does). When ``onpolicy``
        is on the stage>0 entry is the rolled prefix state (same exposure-bias
        geometry as inference), so best-ckpt selection tracks the deployed sampler
        rather than the analytic line. Returns (overall_mse,
        per_stage_mse[n_stages], per_stage_count[n_stages]).
        """
        val_gen.manual_seed(seed + 104729)
        sums = torch.zeros(n_stages, device=device)
        cnts = torch.zeros(n_stages, device=device)
        if cudagraph_step:
            torch.compiler.cudagraph_mark_step_begin()
        with _ema_weights():
            for _idx, latents, crossattn_emb, _pooled in val_loader:
                latents = latents.to(device, dtype=dtype, non_blocking=True)
                crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
                B = latents.shape[0]
                x0_full = to_dit_5d(latents)
                for stage_idx in range(n_stages):
                    # Entry (and ε) drawn once per (batch, stage), reused across σ.
                    x0_si, eps_si, t_lo, t_hi = _stage_entry(
                        x0_full,
                        crossattn_emb,
                        stage_idx,
                        transition_sigmas,
                        val_gen,
                        onpolicy,
                    )
                    v_target = (eps_si - x0_si).float()
                    for k in range(n_val_sigmas):
                        frac = (k + 0.5) / n_val_sigmas
                        t = torch.full(
                            (B,),
                            t_lo + (t_hi - t_lo) * frac,
                            device=device,
                            dtype=dtype,
                        )
                        x_t = renoise(x0_si, t, eps_si)
                        pred = _forward_dit(x_t, t, crossattn_emb)
                        if snr_gate is not None:
                            v_loss, _ = snr_gate.gated_mse(
                                pred,
                                v_target,
                                t,
                                (int(x0_full.shape[-2]), int(x0_full.shape[-1])),
                            )
                            sums[stage_idx] += v_loss
                        else:
                            sums[stage_idx] += nn.functional.mse_loss(
                                pred.float(), v_target
                            )
                        cnts[stage_idx] += 1
        overall = sums.sum() / cnts.sum().clamp(min=1)
        return overall, sums / cnts.clamp(min=1), cnts

    def _micro_step(step):
        """One sample → scaled backward. Returns (unscaled_loss_tensor, stage_idx).

        The loss is returned as a *detached GPU tensor* (not ``.item()``) so the
        accumulation in the training loop stays sync-free; grad_accum micro-steps
        would otherwise force that many CUDA syncs per optimizer step.

        Stage is resampled here (not once per optimizer step), so when
        grad_accum > 1 each update averages gradients across the low-res and
        full-res regimes instead of swinging between them — the high CoV in the
        stage losses is regime-switching noise, which accumulation cancels.
        """
        try:
            _idx, latents, crossattn_emb, _pooled = next(data_iter[0])
        except StopIteration:
            data_iter[0] = iter(dataloader)
            _idx, latents, crossattn_emb, _pooled = next(data_iter[0])

        latents = latents.to(device, dtype=dtype, non_blocking=True)
        crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
        B = latents.shape[0]
        x0_full = to_dit_5d(latents)  # (B, 16, 1, H, W)

        # Optional R2 jitter: perturb the transition σ so the segment geometry is
        # learned as a band, not a point.
        trans = transition_sigmas
        if sigma_jitter > 0.0 and len(transition_sigmas) > 0:
            trans = [
                float(
                    min(
                        0.999,
                        max(0.001, s + (torch.rand(1).item() * 2 - 1) * sigma_jitter),
                    )
                )
                for s in transition_sigmas
            ]

        # Sample one stage for this micro-batch (single-resolution per forward),
        # weighted by band width.
        stage_idx = int(
            torch.multinomial(stage_sample_w_f, 1, generator=stage_rng).item()
        )
        # On-policy entry for stage>0 with annealed probability (DAgger): roll the
        # adapter-on prefix from pure noise instead of the analytic straight line,
        # so the LoRA trains on the off-manifold state inference visits. Decided
        # per micro-step (not per optimizer step) so grad_accum keeps mixing
        # analytic/on-policy and the low-/full-res regimes. _stage_entry returns
        # the matching query band (on-policy t_hi = the rollout's aligned σ̃).
        use_op = stage_idx > 0 and (
            float(torch.rand(1, generator=stage_rng).item()) < _onpolicy_active(step)
        )
        x0_si, eps_si, t_lo, t_hi = _stage_entry(
            x0_full, crossattn_emb, stage_idx, trans, gen, use_op
        )
        # FM training sample + analytic velocity target at scale s_i (Eq. 13–14).
        t = (t_lo + (t_hi - t_lo) * torch.rand(B, device=device)).to(dtype)
        x_t = renoise(x0_si, t, eps_si)
        if args.grad_ckpt:  # reentrant checkpoint needs a grad-requiring input
            x_t.requires_grad_()
        v_target = (eps_si - x0_si).float()
        # Native shapes: the forward runs at this stage's real token count (no
        # padding → no flash pad-leak). Flattening is enabled once by
        # compile_blocks above, which traces one graph per (stage × bucket) shape
        # keyed on the real seq_len — nothing per-step to set here.
        pred = _forward_dit(x_t, t, crossattn_emb)
        if snr_gate is not None:
            # Information-aware loss: DCT-domain error weighted by the per-band
            # recoverable fraction at this t. The plain MSE is kept (detached)
            # for the train/loss_ungated comparison curve.
            loss, gate_w = snr_gate.gated_mse(
                pred, v_target, t, (int(x0_full.shape[-2]), int(x0_full.shape[-1]))
            )
            acc.add("gate_w", gate_w)
            acc.add("ungated", nn.functional.mse_loss(pred.detach().float(), v_target))
        else:
            loss = nn.functional.mse_loss(pred.float(), v_target)
        # Scale so accumulated grads are the *mean* over micro-steps (matches a
        # true batch); LR/grad_clip semantics stay invariant to grad_accum.
        (loss / grad_accum).backward()
        return loss.detach(), stage_idx

    # When validation is on, save only on val/loss improvement (best-ckpt-only,
    # like distill-mod) instead of overwriting every save_every steps. With
    # val off, fall back to the step-cadence save below.
    best_val_loss = float("inf")
    for step in progress:
        if cudagraph_step:
            torch.compiler.cudagraph_mark_step_begin()
        step_loss = torch.zeros((), device=device)  # mean micro-loss, GPU-side
        for _ in range(grad_accum):
            micro_loss, stage_idx = _micro_step(step)
            step_loss = step_loss + micro_loss / grad_accum
            acc.add_at(
                "stage_loss", stage_idx, micro_loss, width=n_stages
            )  # python idx → no sync
            acc.add_at("stage_cnt", stage_idx, 1.0, width=n_stages)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        acc.add("loss", step_loss)
        if ema_shadow is not None:
            with torch.no_grad():
                for s, p in zip(ema_shadow, trainable):
                    s.mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)

        if (step + 1) % log_interval == 0:
            # LoRA L2 norms: accumulate squared sums on-device and fold them into
            # the single sync below (was one .item() per trainable parameter).
            with torch.no_grad():
                up_sq = torch.zeros((), device=device)
                down_sq = torch.zeros((), device=device)
                for name, p in network.named_parameters():
                    if not p.requires_grad:
                        continue
                    s = p.detach().float().pow(2).sum()
                    if "lora_up" in name:
                        up_sq = up_sq + s
                    elif "lora_down" in name:
                        down_sq = down_sq + s
            acc.add("up_sq", up_sq)
            acc.add("down_sq", down_sq)
            # One CUDA sync per log boundary: read every accumulator by name.
            # Per-key reductions (mean over the interval, sqrt of squared-norm
            # sums) run on the returned floats — no further sync.
            m = acc.flush_reset()
            n_micro = log_interval * grad_accum  # micro-steps per log interval
            avg = m["loss"] / log_interval
            up_norm = m["up_sq"] ** 0.5
            down_norm = m["down_sq"] ** 0.5
            gate_w_mean = m.get("gate_w", 0.0) / n_micro
            ungated_mean = m.get("ungated", 0.0) / n_micro
            stage_cnts = m["stage_cnt"]
            stage_vals = [
                ls / c if c > 0 else 0.0 for ls, c in zip(m["stage_loss"], stage_cnts)
            ]
            cur_lr = scheduler.get_last_lr()[0]  # CPU-side; no sync
            progress.set_postfix(
                loss=f"{avg:.5f}",
                stage=stage_idx,
                lr=f"{cur_lr:.2e}",
                up=f"{up_norm:.3f}",
            )
            if writer is not None:
                writer.add_scalar("train/loss", avg, step + 1)
                writer.add_scalar("train/lr", cur_lr, step + 1)
                writer.add_scalar("train/lora_up_norm", up_norm, step + 1)
                writer.add_scalar("train/lora_down_norm", down_norm, step + 1)
                if snr_gate is not None:
                    # Effective supervised fraction + the plain MSE the gated
                    # loss replaced (diverging curves = the gate is doing work).
                    writer.add_scalar("train/gate_w", gate_w_mean, step + 1)
                    writer.add_scalar("train/loss_ungated", ungated_mean, step + 1)
                # Per-stage mean loss over the interval (only stages touched).
                for si in range(n_stages):
                    if stage_cnts[si] > 0:
                        writer.add_scalar(
                            f"train/loss_stage{si}", stage_vals[si], step + 1
                        )

        # Held-out analytic-MSE validation (CMMD-free overfit signal).
        improved = False
        v_overall = None
        if val_loader is not None and (
            (step + 1) % val_interval == 0 or (step + 1) == iterations
        ):
            val_overall, val_stage, val_cnt = _validate()
            packed = torch.cat([val_overall.reshape(1), val_stage, val_cnt]).tolist()
            v_overall = packed[0]
            v_stage = packed[1 : 1 + n_stages]
            v_cnt = packed[1 + n_stages : 1 + 2 * n_stages]
            logger.info(
                "val @ step %d: loss=%.6f  %s",
                step + 1,
                v_overall,
                "  ".join(
                    f"stage{si}={v_stage[si]:.6f}(n={int(v_cnt[si])})"
                    for si in range(n_stages)
                ),
            )
            if writer is not None:
                writer.add_scalar("val/loss", v_overall, step + 1)
                for si in range(n_stages):
                    if v_cnt[si] > 0:
                        writer.add_scalar(f"val/loss_stage{si}", v_stage[si], step + 1)
            if v_overall < best_val_loss:
                best_val_loss = v_overall
                improved = True

        # Save: with validation on, only overwrite the checkpoint when val/loss
        # improves (keep the best, like distill-mod). With val off, fall back to
        # the step-cadence save.
        if val_loader is not None:
            should_save = improved
        else:
            should_save = (step + 1) % save_every == 0 or (step + 1) == iterations
        if should_save:
            _save(step + 1)
        elif v_overall is not None:
            logger.info(
                "skipped save at step %d: val=%.6f >= best=%.6f",
                step + 1,
                v_overall,
                best_val_loss,
            )

    if writer is not None:
        writer.close()
    logger.info("SPD distillation complete.")


if __name__ == "__main__":
    main()
