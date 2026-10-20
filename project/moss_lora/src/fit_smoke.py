#!/usr/bin/env python3
"""LoRA fit smoke on synthetic data: does a checkpointed step fit, and how fast is it?

Random latents / embeddings at the given sizes, through ``library.moss.train.train``
(the real loop), for a few steps. Prints the step-1 fit report and s/step.

    make daemon-run ARGS="project/moss_lora/src/fit_smoke.py --sizes 1024x1024:300 --steps 8"
"""

import argparse

from library.moss.requests import TrainRequest
from library.moss.train import train
from library.moss.tune import synthetic_samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes",
        default="1024x1024:300,1184x880:300",
        help="comma list of WxH:text_tokens",
    )
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--output", default="output/ckpt_moss/_fit_smoke.safetensors")
    args, rest = ap.parse_known_args()

    samples = synthetic_samples(args.sizes)

    req = TrainRequest.from_argv(
        [
            "--output",
            args.output,
            "--max_steps",
            str(args.steps),
            "--log_every",
            "1",
            "--fit_report",
            *rest,
        ]
    )
    req.epochs = max(1, -(-args.steps // len(samples)))
    train(req, samples)


if __name__ == "__main__":
    main()
