"""MOSS-Image text-to-image: the reference sampler, run in three phases.

Encode every distinct prompt with the layer-offloaded Qwen3.5-9B and drop it,
denoise every job with the DiT resident, then take the DiT off the card before
the fp32 VAE decode (at ~2 MP the decode needs more than is free beside the DiT).

``denoise`` is the sampler loop on its own — Euler on the snapped shift-5 grid,
CFG with the uncond branch skipping block 22, all combining in fp32 — and with
``spectrum=None`` it is ``MOSSImagePipeline.__call__`` op for op.

    make daemon-run ARGS="scripts/moss/generate.py --prompt '…' --steps 30"
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path

import torch

from library.env import resolve_under_home
from library.moss.loader import (
    MAX_SEQUENCE_LENGTH,
    compile_blocks,
    empty_cache,
    encode_prompts,
    load_transformer,
    load_vae,
    pad_embeds,
)
from library.moss.model import decode, snapped_sigmas
from library.moss.requests import GenerateRequest, resolve_model_dir


def _parse_skip(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(v) for v in str(value).split(",") if v.strip())


def _features(transformer, latents, t, embeds, mask, skip, batched):
    """Image-token features of the cond rows then the uncond rows, ``(R, H·W, D)``.
    ``embeds`` / ``mask`` / ``skip`` are per-branch lists (one entry = no CFG)."""
    if len(embeds) == 1:
        return [transformer(latents, t, embeds[0], mask[0], skip_layers=skip[0], return_features=True)]
    if not batched:
        return [
            transformer(latents, t, e, m, skip_layers=s, return_features=True)
            for e, m, s in zip(embeds, mask, skip)
        ]
    rows = []
    skips = []
    for e, m, s in zip(embeds, mask, skip):
        for b in range(e.shape[0]):
            rows.append(e[b, m[b]])
            skips.append(s)
    joint, joint_mask = pad_embeds(rows, device=latents.device)
    feats = transformer.forward_batched(
        latents.repeat(len(embeds), 1, 1, 1), t, joint, joint_mask,
        skip_layers=skips, return_features=True,
    )
    return list(feats.chunk(len(embeds)))


@torch.no_grad()
def denoise(
    transformer,
    latents,
    cond,
    cond_mask,
    uncond,
    uncond_mask,
    sigmas,
    *,
    cfg=3.0,
    uncond_skip=(22,),
    batched=False,
    spectrum=None,
    progress=None,
):
    """Euler-integrate ``latents (B,128,h,w)`` over ``sigmas`` (``snapped_sigmas``:
    descending, no terminal 0). Returns fp32 latents.

    ``cond`` / ``uncond`` are ``(B,S,4096)`` embeds with bool masks; ``uncond`` is
    unused when ``cfg == 1``. ``spectrum`` (optional) is called as ``begin(sigmas)``,
    then per step ``is_full_step(i)`` → ``observe(i, features)`` after a real
    forward or ``forecast(i)`` instead of one; ``features`` are the image tokens
    after the last block, ``(R, H·W, D)``, cond rows then uncond rows.
    """
    device = latents.device
    batch, _, height, width = latents.shape
    latents = latents.to(dtype=torch.float32)
    # FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=…): fp32, ×1000, append 0.
    sigma_t = sigmas.to(dtype=torch.float32, device=device)
    timesteps = sigma_t * 1000
    sigma_t = torch.cat([sigma_t, torch.zeros(1, device=device)])
    do_cfg = cfg != 1.0
    skip = tuple(uncond_skip or ())
    embeds, masks, skips = [cond], [cond_mask], [()]
    if do_cfg:
        embeds.append(uncond)
        masks.append(uncond_mask)
        skips.append(skip)
    if spectrum is not None:
        spectrum.begin(sigmas)
    for i, t in enumerate(timesteps):
        if spectrum is None or spectrum.is_full_step(i):
            branches = _features(transformer, latents, t, embeds, masks, skips, batched)
            if spectrum is not None:
                spectrum.observe(i, torch.cat(branches))
        else:
            branches = list(spectrum.forecast(i).chunk(len(embeds)))
        # The head per branch, as each reference forward applies it.
        _mod, head_mod = transformer.time_conditioning(t, batch)
        velocity = [transformer.head(f, head_mod, height, width).float() for f in branches]
        prediction = velocity[0]
        if do_cfg:
            prediction = velocity[1] + cfg * (velocity[0] - velocity[1])
        latents = latents + (sigma_t[i + 1] - sigma_t[i]) * prediction
        if progress is not None:
            progress(i, len(timesteps))
    return latents


def initial_noise(seed: int, height: int, width: int, channels: int = 128) -> torch.Tensor:
    """fp32 noise from a CPU generator, ``(1, 128, H/16, W/16)`` — what the Space
    and ``randn_tensor(generator=torch.Generator('cpu'))`` draw."""
    gen = torch.Generator("cpu").manual_seed(seed)
    return torch.randn((1, channels, height // 16, width // 16), generator=gen, dtype=torch.float32)


def _jobs(req: GenerateRequest) -> list[dict]:
    base = dict(
        negative=req.negative_prompt,
        width=req.width,
        height=req.height,
        cfg=req.cfg,
        skip=_parse_skip(req.uncond_skip),
    )
    if req.example:
        jobs = []
        for n, path in enumerate(filter(None, (p.strip() for p in req.example.split(",")))):
            path = resolve_under_home(path)
            p = json.loads(path.read_text(encoding="utf-8"))
            p = p.get("parameters", p)
            jobs.append(
                dict(
                    # Gallery entries are <id>/parameters.json: key by the folder.
                    key=f"{n:03d}_space{path.parent.name if path.stem == 'parameters' else path.stem}",
                    prompt=p["prompt"],
                    negative=p.get("negative_prompt", ""),
                    width=int(p["width"]),
                    height=int(p["height"]),
                    cfg=float(p.get("cfg", p.get("guidance_scale", req.cfg))),
                    seed=int(p["seed"]),
                    skip=_parse_skip(p.get("uncond_skip_layers", req.uncond_skip)),
                    steps=int(p.get("steps", p.get("num_inference_steps", req.steps))),
                    shift=float(p.get("shift", req.shift)),
                )
            )
        return jobs
    if req.prompts_file:
        source = resolve_under_home(req.prompts_file)
        prompts = [ln.strip() for ln in source.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        source, prompts = "prompt", [req.prompt.strip()] if req.prompt.strip() else []
    if not prompts:
        raise SystemExit(f"no prompts in {source}")
    return [
        dict(base, key=f"{i:03d}", prompt=p, seed=req.seed + i, steps=req.steps, shift=req.shift)
        for i, p in enumerate(prompts)
    ]


def _cached_embeds(model_dir, texts, cache_dir: Path) -> dict[str, torch.Tensor]:
    """TE outputs keyed by text, read from / written to ``cache_dir`` so a rerun of
    the same prompts skips the 17 GB text encoder."""
    from safetensors.torch import load_file, save_file

    def path(text):
        digest = hashlib.sha1(f"{MAX_SEQUENCE_LENGTH}\0{text}".encode()).hexdigest()[:16]
        return cache_dir / f"{digest}.safetensors"

    out, missing = {}, []
    for text in dict.fromkeys(texts):
        if path(text).is_file():
            out[text] = load_file(path(text))["embeds"]
        else:
            missing.append(text)
    if missing:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for text, emb in encode_prompts(model_dir, missing).items():
            save_file({"embeds": emb.contiguous()}, path(text))
            out[text] = emb
    print(f"[te] {len(out) - len(missing)} cached, {len(missing)} encoded", flush=True)
    return out


def _build_spectrum(req: GenerateRequest, job: dict):
    """Spectrum forecaster for one job, or None."""
    if not req.spectrum:
        return None
    from library.moss.spectrum import MossSpectrum

    return MossSpectrum(
        window_size=req.spectrum_window_size,
        flex_window=req.spectrum_flex_window,
        warmup=req.spectrum_warmup,
        stop_caching_step=req.spectrum_stop_caching_step,
        w=req.spectrum_w,
        m=req.spectrum_m,
        lam=req.spectrum_lam,
    )


def _fingerprint(req: GenerateRequest, job: dict) -> dict:
    """What a saved latent was made from; a resume reuses it only on a match."""
    spectrum = {k: v for k, v in dataclasses.asdict(req).items() if k.startswith("spectrum")}
    return dict(
        {k: v for k, v in job.items() if k != "skip"},
        uncond_skip=list(job["skip"]),
        model=req.model,
        lora=str(resolve_under_home(req.lora)) if req.lora else None,
        lora_multiplier=req.lora_multiplier if req.lora else None,
        batched=req.batched,
        spectrum=spectrum if req.spectrum else None,
    )


def _latent_path(out_dir: Path, rec: dict) -> Path:
    return out_dir / "latents" / f"{rec['key']}_{rec['model']}_s{rec['seed']}.safetensors"


def _image_path(out_dir: Path, rec: dict) -> Path:
    return out_dir / f"{rec['key']}_{rec['model']}_s{rec['seed']}.png"


def save_latent(path: Path, latent: torch.Tensor, record: dict) -> None:
    """Final latent + its record in the metadata, written atomically."""
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file(
        {"latent": latent.contiguous()}, tmp,
        metadata={"record": json.dumps(record, ensure_ascii=False)},
    )
    tmp.replace(path)


def read_latent_record(path: Path) -> dict | None:
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            return json.loads((f.metadata() or {})["record"])
    except Exception:
        return None


def load_latent(path: Path) -> torch.Tensor:
    from safetensors.torch import load_file

    return load_file(path)["latent"]


def _write_manifest(out_dir: Path, req: GenerateRequest, records: list[dict]) -> Path:
    manifest = out_dir / "manifest.json"
    manifest.write_text(
        json.dumps({"request": req.to_argv(), "images": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _denoise_pending(req, model_dir, out_dir, pending, device) -> None:
    """TE for the pending jobs only, then the DiT; each latent hits disk as soon
    as its denoise returns, so a killed job loses at most the one in flight."""
    texts = [j["prompt"] for j, _ in pending]
    texts += [j["negative"] for j, _ in pending if j["cfg"] != 1.0]
    embeds = _cached_embeds(model_dir, texts, out_dir / "_te_cache")

    t0 = time.time()
    transformer = load_transformer(model_dir, req.model, device=device)
    print(
        f"[dit] {req.model} loaded in {time.time() - t0:.1f}s, "
        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB",
        flush=True,
    )
    if req.lora:
        from library.moss.lora import load_for_inference

        load_for_inference(transformer, resolve_under_home(req.lora), req.lora_multiplier)
        print(f"[lora] {req.lora} x{req.lora_multiplier}", flush=True)
    if req.compile:
        compile_blocks(transformer.layers)
    for j, rec in pending:
        cond, cond_mask = pad_embeds([embeds[j["prompt"]]], device=device)
        uncond = uncond_mask = None
        if j["cfg"] != 1.0:
            uncond, uncond_mask = pad_embeds([embeds[j["negative"]]], device=device)
        sigmas = snapped_sigmas(j["steps"], j["shift"])
        noise = initial_noise(j["seed"], j["height"], j["width"]).to(device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        latent = denoise(
            transformer, noise, cond, cond_mask, uncond, uncond_mask, sigmas,
            cfg=j["cfg"], uncond_skip=j["skip"], batched=req.batched,
            spectrum=_build_spectrum(req, j),
        ).cpu()
        torch.cuda.synchronize()
        seconds = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        rec.update(
            steps_run=len(sigmas),
            text_tokens=int(cond.shape[1]),
            denoise_s=round(seconds, 2),
            denoise_peak_gib=round(peak, 2),
            compiled=req.compile,
        )
        save_latent(_latent_path(out_dir, rec), latent, rec)
        print(
            f"[{req.model}] {j['key']} {j['width']}x{j['height']} cfg {j['cfg']} "
            f"{len(sigmas)} steps: {seconds:.1f}s, peak {peak:.1f} GiB -> "
            f"{_latent_path(out_dir, rec).name}",
            flush=True,
        )
    # fp32 decode with the DiT off the card.
    transformer.to("cpu")
    del transformer
    empty_cache()


def _decode_all(model_dir, out_dir, records, device) -> None:
    """Decode each record's latent from disk; an existing PNG is kept."""
    todo = [r for r in records if not _image_path(out_dir, r).is_file()]
    for r in records:
        r["image"] = str(_image_path(out_dir, r))
    if not todo:
        return
    from diffusers.image_processor import VaeImageProcessor

    vae = load_vae(model_dir, device=device)
    processor = VaeImageProcessor(vae_scale_factor=16)
    for rec in todo:
        torch.cuda.reset_peak_memory_stats()
        image = decode(load_latent(_latent_path(out_dir, rec)).to(device), vae)
        processor.postprocess(image, output_type="pil")[0].save(rec["image"])
        rec["decode_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        print(f"[vae] {rec['key']} -> {rec['image']} (peak {rec['decode_peak_gib']} GiB)", flush=True)
    del vae
    empty_cache()


def run_generate(req: GenerateRequest) -> Path:
    """Render every job; returns the output folder.

    Latents land in ``<out_dir>/latents/`` (record in the metadata) before any
    decode, and a rerun skips every job whose saved latent matches it, then
    decodes only the PNGs that are missing. ``decode_only`` decodes whatever is
    in ``latents/`` and loads nothing else."""
    model_dir = resolve_model_dir(req.model_dir)
    out_dir = resolve_under_home(req.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    device = torch.device("cuda")

    if req.decode_only:
        records = [r for r in map(read_latent_record, sorted((out_dir / "latents").glob("*.safetensors"))) if r]
        if not records:
            raise SystemExit(f"no latents in {out_dir / 'latents'}")
    else:
        jobs = _jobs(req)
        for j in jobs:
            if j["width"] % 16 or j["height"] % 16:
                raise SystemExit(f"{j['key']}: {j['width']}x{j['height']} is not a multiple of 16")
        records, pending = [], []
        for j in jobs:
            rec = _fingerprint(req, j)
            saved = _latent_path(out_dir, rec)
            old = read_latent_record(saved) if saved.is_file() else None
            if old is not None and all(old.get(k) == v for k, v in rec.items()):
                print(f"[resume] {j['key']}: latent on disk, skipped", flush=True)
                records.append(old)
                continue
            if old is not None:
                print(f"[resume] {j['key']}: saved latent was made differently, redoing", flush=True)
                _image_path(out_dir, rec).unlink(missing_ok=True)
            records.append(rec)
            pending.append((j, rec))
        if pending:
            _denoise_pending(req, model_dir, out_dir, pending, device)

    _decode_all(model_dir, out_dir, records, device)
    manifest = _write_manifest(out_dir, req, records)
    print(f"manifest -> {manifest}", flush=True)
    return out_dir
