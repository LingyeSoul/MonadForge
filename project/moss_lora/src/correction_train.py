#!/usr/bin/env python3
"""Train a corrector LoRA (``correction.py``), behind a frozen denoiser LoRA
(``--denoiser``) or jointly with a new one (``--joint``).

``library.moss.train.train`` runs unchanged with names swapped in its module:
``LoRANetwork`` is the corrector network, ``training_loss`` the correction
loss, and ``place`` keeps the block-swap attachment. Every other
``TrainRequest`` flag means what it does for a plain LoRA (``--joint`` applies
them to both LoRAs).

``--denoiser``: ``load_transformer`` also hooks the frozen LoRA D. Per step: D's
forward without grad (corrector off), the swapped blocks put back where a
forward expects them (a no-grad forward under ``supports_backward`` leaves the
tail resident and no backward walks it back), then C's forward and backward on
``x̂ = z_t − σ·v₀``.

``--joint`` (the paper's Algorithm 1): D is a new LoRA from the base, trained
on ``L_denoise`` while C trains on ``L_corr``, both with stop-gradient into D —
``x̂`` is detached, and in C's pass D's adapters run on detached weights
(``_joint_forward``), so the C pass is the same function as at inference but
sends nothing to D. Two forward/backward pairs per step (D's backward runs
inside the loss, C's in the trainer), one optimizer step: each backward walks
the swapped blocks back, so the swapper sees two ordinary steps. D is saved as
``<output stem>_D.safetensors``, C at ``--output`` (+ its ``.gate`` sidecar).

``<output>.corr.json`` logs D's loss, the corrected loss and ``γ`` per step.

    make daemon-run ARGS="project/moss_lora/src/correction_train.py \\
        --joint --corr_sigma 0.15 \\
        --config configs/moss/lora.toml --cache 'post_image_dataset/moss/channel_(caststation)_layout' \\
        --epochs 20 --output output/ckpt_moss/<name>.safetensors"
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import library.moss.train as T
from correction import CorrectorLoRA, correct_once
from library.env import resolve_under_home
from library.moss.lora import LoRANetwork, load_for_inference
from library.moss.requests import TrainRequest

LOG_EVERY = 50
_RUN: dict = {"log": []}


def denoiser_path(path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_D{path.suffix}")


def _load_transformer(model_dir, variant="base", *args, **kwargs):
    tr = _orig_load_transformer(model_dir, variant, *args, **kwargs)
    d = load_for_inference(tr, _RUN["denoiser"])
    d.to(torch.device("cuda"))
    _RUN["d"] = d
    print(f"denoiser (frozen): {_RUN['denoiser']}", flush=True)
    return tr


class _Corrector(CorrectorLoRA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _RUN["net"] = self


def _lora_detached(adapter, x, base_out):
    """``LoRAAdapter.forward`` with its weights cut from the graph."""
    work = base_out.dtype
    lx = F.linear(x.to(work), adapter.down_weight().detach().to(work))
    delta = F.linear(lx, adapter.up.weight.detach().to(work))
    return base_out + delta * (adapter.scale * adapter.multiplier)


def _joint_forward(module, d, c):
    base = module.__class__.forward

    def forward(x):
        out = base(module, x)
        if c.multiplier == 0.0:
            return d(x, out)
        return c(x, _lora_detached(d, x, out))

    return forward


class _Joint(_Corrector):
    """C with a trainable D inside it (``self.denoiser``), so the trainer's one
    optimizer, clip and ``reproject_dc`` cover both."""

    def __init__(self, model, *args, **kwargs):
        if kwargs.get("algo", "lora") != "lora":
            raise ValueError("--joint needs algo lora")
        denoiser = LoRANetwork(model, *args, **kwargs)
        super().__init__(model, *args, **kwargs)
        self.denoiser = denoiser

    def apply_to(self) -> int:
        for name, module in self._targets:
            key = name.replace(".", "_")
            module.forward = _joint_forward(
                module, self.denoiser.adapters[key], self.adapters[key]
            )
        self._applied = self.denoiser._applied = True
        return len(self._targets)

    @torch.no_grad()
    def reproject_dc(self) -> None:
        super().reproject_dc()
        self.denoiser.reproject_dc()

    def save(self, path, dtype: str = "bf16", **extra_meta) -> None:
        d = self._modules.pop("denoiser")
        try:
            d.save(denoiser_path(path), dtype, **extra_meta)
            super().save(path, dtype, **extra_meta)
        finally:
            self._modules["denoiser"] = d


def _place(*args, **kwargs):
    attached = _orig_place(*args, **kwargs)
    _RUN["attached"] = attached
    return attached


def _noised(latent, sigma, gen):
    noise = torch.randn(
        latent.shape, generator=gen, device=latent.device, dtype=latent.dtype
    )
    noisy = (1.0 - sigma) * latent + sigma * noise
    timestep = torch.tensor([T.NUM_TRAIN_TIMESTEPS * sigma], device=latent.device)
    return noisy, (noise - latent).float(), timestep


def _log(net, sigma: float, l0: float, l1: float) -> None:
    g = net.gamma().detach().flatten()
    rec = {
        "sigma": sigma,
        "l0": l0,
        "l1": l1,
        "gamma_mean": g.mean().item(),
        "gamma_absmax": g.abs().max().item(),
    }
    _RUN["log"].append(rec)
    if len(_RUN["log"]) % LOG_EVERY == 0:
        window = _RUN["log"][-LOG_EVERY:]
        m0 = sum(r["l0"] for r in window) / len(window)
        m1 = sum(r["l1"] for r in window) / len(window)
        print(
            f"  corr {len(_RUN['log'])}: L_denoise {m0:.4f} L_corr {m1:.4f} "
            f"({(m1 - m0) / m0:+.2%}) γ mean {rec['gamma_mean']:.3f} "
            f"|γ|max {rec['gamma_absmax']:.3f}",
            flush=True,
        )


def _correction_loss(transformer, latent, embeds, sigma, gen=None, ref=None):
    net, attached = _RUN["net"], _RUN["attached"]
    noisy, target, timestep = _noised(latent, sigma, gen)
    net.set_multiplier(0.0)
    with torch.no_grad():
        v0 = transformer(noisy, timestep, embeds, return_dict=False, ref=ref)[0]
    v0 = v0.float()
    if attached is not None:
        attached.offloader.prepare_block_devices_before_forward(
            attached.blocks, free_cache=False
        )
    net.set_multiplier(1.0)
    v1 = correct_once(transformer, net, noisy, sigma, timestep, embeds, v0, ref=ref)
    loss = F.mse_loss(v1, target)
    _log(net, sigma, F.mse_loss(v0, target).item(), loss.item())
    return loss


def _joint_loss(transformer, latent, embeds, sigma, gen=None, ref=None):
    """Runs D's backward itself; returns ``λ·L_corr`` for the trainer's."""
    net = _RUN["net"]
    noisy, target, timestep = _noised(latent, sigma, gen)
    net.set_multiplier(0.0)
    v0 = transformer(noisy, timestep, embeds, return_dict=False, ref=ref)[0].float()
    l0 = F.mse_loss(v0, target)
    l0.backward()
    v0 = v0.detach()
    net.set_multiplier(1.0)
    v1 = correct_once(transformer, net, noisy, sigma, timestep, embeds, v0, ref=ref)
    loss = F.mse_loss(v1, target)
    _log(net, sigma, l0.item(), loss.item())
    return _RUN["corr_weight"] * loss


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    mode = pre.add_mutually_exclusive_group(required=True)
    mode.add_argument("--denoiser", help="frozen LoRA D")
    mode.add_argument(
        "--joint", action="store_true", help="train D from the base alongside C"
    )
    pre.add_argument(
        "--corr_weight", type=float, default=1.0, help="λ on L_corr (--joint)"
    )
    pre.add_argument(
        "--gate_scale",
        type=float,
        default=40.0,
        help="γ = gate · gate_scale; γ's Adam step is lr · gate_scale",
    )
    pre.add_argument("--gate_init", type=float, default=0.0, help="initial γ")
    pre.add_argument(
        "--corr_sigma",
        type=float,
        default=0.0,
        help="0 = native v-space at t; > 0 = clean-sample form at min(this, σ)",
    )
    ours, rest = pre.parse_known_args()
    req = TrainRequest.from_argv(rest)
    net_cls = _Joint if ours.joint else _Corrector
    net_cls.gate_scale = ours.gate_scale
    net_cls.gate_init = ours.gate_init
    net_cls.corr_sigma = ours.corr_sigma
    _RUN["corr_weight"] = ours.corr_weight

    global _orig_load_transformer, _orig_place
    _orig_load_transformer, _orig_place = T.load_transformer, T.place
    T.LoRANetwork = net_cls
    T.place = _place
    if ours.joint:
        T.training_loss = _joint_loss
    else:
        _RUN["denoiser"] = resolve_under_home(ours.denoiser)
        T.load_transformer = _load_transformer
        T.training_loss = _correction_loss

    out = T.run_train(req)
    log = out.with_suffix(".corr.json")
    log.write_text(
        json.dumps(
            {
                "denoiser": str(denoiser_path(out))
                if ours.joint
                else str(_RUN["denoiser"]),
                "joint": ours.joint,
                "corr_weight": ours.corr_weight,
                "gate_scale": ours.gate_scale,
                "gate_init": ours.gate_init,
                "corr_sigma": ours.corr_sigma,
                "steps": _RUN["log"],
            },
            indent=1,
        )
    )
    print(f"wrote {log}", flush=True)


if __name__ == "__main__":
    main()
