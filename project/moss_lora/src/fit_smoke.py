#!/usr/bin/env python3
"""LoRA fit smoke on synthetic data: does a checkpointed step fit, and how fast is it?

Random latents / embeddings at the given sizes, through ``library.moss.train.train``
(the real loop), for a few steps. Prints the step-1 fit report and s/step.

    make daemon-run ARGS="project/moss_lora/src/fit_smoke.py --sizes 1024x1024:300 --steps 8"
"""

import argparse

import torch

from library.moss.requests import TrainRequest
from library.moss.train import Sample, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes",
        default="1024x1024:300,1184x880:300",
        help="comma list of WxH:text_tokens",
    )
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--output", default="output/moss/lora/_fit_smoke.safetensors")
    args, rest = ap.parse_known_args()

    samples = []
    for i, spec in enumerate(args.sizes.split(",")):
        size, n_text = spec.split(":")
        w, h = map(int, size.split("x"))
        n_text = int(n_text)

        def load(w=w, h=h, n_text=n_text, i=i):
            g = torch.Generator().manual_seed(i)
            latent = torch.randn(128, h // 16, w // 16, generator=g)
            embeds = torch.randn(n_text, 4096, generator=g).to(torch.bfloat16)
            return latent, embeds

        samples.append(Sample(f"synthetic{i}_{w}x{h}", (w // 16) * (h // 16), n_text, load))

    req = TrainRequest.from_argv(
        ["--output", args.output, "--max_steps", str(args.steps), "--log_every", "1", *rest]
    )
    req.epochs = max(1, -(-args.steps // len(samples)))
    train(req, samples)


if __name__ == "__main__":
    main()
