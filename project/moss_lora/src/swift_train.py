#!/usr/bin/env python3
"""SWIFT target extrapolation (arXiv 2610.25844) on a MOSS LoRA (``swift.md``).

Trains on ``y = v + α (v − v_weak)``, ``v = noise − latent`` and ``v_weak`` the
base plus a frozen guide LoRA (``--guide``, an early checkpoint of the same
recipe), instead of on ``v``. ``library.moss.train.train`` runs unchanged with
names swapped in its module, as in ``correction_train.py``: ``load_transformer``
hooks the guide, ``LoRANetwork`` chains the trained LoRA after it, and
``training_loss`` is the SWIFT loss. Per step: the guide's forward without grad
(trained LoRA off), the swapped blocks put back where a forward expects them,
then the trained LoRA's forward (guide off) against ``y``. One ``randn`` per
step as in ``training_loss``, so a run sees the same σ / noise / caption
sequence as the α 0 run with the same seed.

``<output>.swift.json`` logs per step σ, the loss against plain ``v`` and
``‖v − v_weak‖²`` (the extrapolation's size).

    make daemon-run ARGS="--stall-timeout 900 project/moss_lora/src/swift_train.py \\
        --guide output/ckpt_moss/<run>/<run>_e5.safetensors --alpha 0.4 \\
        --config configs/moss/lora.toml --cache 'post_image_dataset/moss/channel_(caststation)_layout' \\
        --epochs 20 --plain_caption_rate 0 --output output/ckpt_moss/<name>.safetensors"
"""

import argparse
import json

import torch
import torch.nn.functional as F

import library.moss.train as T
from library.env import resolve_under_home
from library.moss.lora import LoRANetwork, load_for_inference
from library.moss.requests import TrainRequest

LOG_EVERY = 50
_RUN: dict = {"log": []}


def _chained_forward(adapter, prev):
    def forward(x: torch.Tensor) -> torch.Tensor:
        return adapter(x, prev(x))

    return forward


class _Student(LoRANetwork):
    """The trained LoRA, applied on top of the guide's hook."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _RUN["net"] = self

    def apply_to(self) -> int:
        if self._applied:
            raise RuntimeError("already applied")
        for name, module in self._targets:
            module.forward = _chained_forward(
                self.adapters[name.replace(".", "_")], module.forward
            )
        self._applied = True
        return len(self._targets)


def _load_transformer(model_dir, variant="base", *args, **kwargs):
    tr = _orig_load_transformer(model_dir, variant, *args, **kwargs)
    guide = load_for_inference(tr, _RUN["guide_path"])
    guide.to(torch.device("cuda"))
    _RUN["guide"] = guide
    print(f"guide (frozen): {_RUN['guide_path']}, α {_RUN['alpha']:g}", flush=True)
    return tr


def _place(*args, **kwargs):
    attached = _orig_place(*args, **kwargs)
    _RUN["attached"] = attached
    return attached


def _swift_loss(transformer, latent, embeds, sigma, gen=None, ref=None):
    net, guide, attached = _RUN["net"], _RUN["guide"], _RUN["attached"]
    noise = torch.randn(
        latent.shape, generator=gen, device=latent.device, dtype=latent.dtype
    )
    noisy = (1.0 - sigma) * latent + sigma * noise
    v = (noise - latent).float()
    timestep = torch.tensor([T.NUM_TRAIN_TIMESTEPS * sigma], device=latent.device)

    def nograd_forward():
        with torch.no_grad():
            out = transformer(noisy, timestep, embeds, return_dict=False, ref=ref)[0]
        if attached is not None:
            attached.offloader.prepare_block_devices_before_forward(
                attached.blocks, free_cache=False
            )
        return out.float()

    net.set_multiplier(0.0)
    if not _RUN["log"]:
        # Once: the guide hook is live (base and guide outputs differ).
        guide.set_multiplier(0.0)
        v_base = nograd_forward()
    guide.set_multiplier(1.0)
    v_weak = nograd_forward()
    if not _RUN["log"]:
        print(
            f"  step 1: ‖v_weak−v_base‖² {F.mse_loss(v_weak, v_base).item():.5f}",
            flush=True,
        )
    guide.set_multiplier(0.0)
    net.set_multiplier(1.0)

    alpha = _RUN["alpha"]
    target = v + alpha * (v - v_weak)
    pred = transformer(noisy, timestep, embeds, return_dict=False, ref=ref)[0].float()
    loss = F.mse_loss(pred, target)

    rec = {
        "sigma": sigma,
        "loss": loss.item(),
        "loss_v": F.mse_loss(pred.detach(), v).item(),
        "gap": F.mse_loss(v_weak, v).item(),
    }
    _RUN["log"].append(rec)
    if len(_RUN["log"]) % LOG_EVERY == 0:
        window = _RUN["log"][-LOG_EVERY:]
        mean = {k: sum(r[k] for r in window) / len(window) for k in rec}
        print(
            f"  swift {len(_RUN['log'])}: loss {mean['loss']:.4f} "
            f"vs v {mean['loss_v']:.4f} ‖v−v_weak‖² {mean['gap']:.4f}",
            flush=True,
        )
    return loss


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--guide", required=True, help="frozen guide LoRA")
    pre.add_argument("--alpha", type=float, default=0.4, help="extrapolation α")
    ours, rest = pre.parse_known_args()
    req = TrainRequest.from_argv(rest)
    if req.algo != "lora":
        raise SystemExit("swift_train needs algo lora")
    _RUN["guide_path"] = resolve_under_home(ours.guide)
    _RUN["alpha"] = ours.alpha

    global _orig_load_transformer, _orig_place
    _orig_load_transformer, _orig_place = T.load_transformer, T.place
    T.load_transformer = _load_transformer
    T.place = _place
    T.LoRANetwork = _Student
    T.training_loss = _swift_loss

    out = T.run_train(req)
    log = out.with_suffix(".swift.json")
    log.write_text(
        json.dumps(
            {
                "guide": str(_RUN["guide_path"]),
                "alpha": ours.alpha,
                "steps": _RUN["log"],
            },
            indent=1,
        )
    )
    print(f"wrote {log}", flush=True)


if __name__ == "__main__":
    main()
