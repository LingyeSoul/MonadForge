"""V100 FlashAttention build and validation orchestration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINNED_REPOSITORY = "https://github.com/ai-bond/flash-attention-v100.git"
PINNED_COMMIT = "c91cad40c0539805754819e6ea96c75184d816a6"
CAPTURE_SHA256 = "91f67505dd66914718a3de61d361c71a1d621fcda46f0fa0d43731d11f05fa0d"
DIT_SHA256 = "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e"
CROSSATTN_SHA256 = "baa504ab68bfbb92e8d632ebf47c1ee6244f66073be0b2ac56a07e4cf2be3b54"


__all__ = [
    "CAPTURE_SHA256",
    "CROSSATTN_SHA256",
    "DIT_SHA256",
    "PINNED_COMMIT",
    "PINNED_REPOSITORY",
    "ROOT",
]
