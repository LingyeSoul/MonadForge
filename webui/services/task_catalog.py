"""WebUI-safe task catalog backed by the canonical ``tasks.py`` registry."""

from __future__ import annotations

from tasks import COMMANDS as CLI_COMMANDS

# This remains an explicit security boundary: only commands listed here can be
# submitted by a browser. Descriptions come from tasks.py so command renames
# cannot silently leave the WebUI pointing at a removed command.
_WEBUI_COMMAND_NAMES = frozenset(
    {
        "lora",
        "lora-gui",
        "staged-train",
        "staged-preprocess",
        "turbo",
        "easycontrol",
        "exp-spd",
        "exp-chimera",
        "test",
        "test-hydra",
        "test-merge",
        "test-dcw",
        "test-dcw-v4",
        "test-smc-cfg",
        "test-spectrum-dcw",
        "test-dcw-v4-spectrum",
        "test-easycontrol",
        "test-turbo",
        "exp-test-directedit",
        "exp-test-directedit-dry",
        "preprocess",
        "preprocess-resize",
        "preprocess-vae",
        "preprocess-te",
        "preprocess-pe",
        "preprocess-cond-resize",
        "preprocess-cond-vae",
        "easycontrol-preprocess",
        "mask",
        "mask-clean",
        "merge",
        "merge-loras",
        "dcw",
        "dcw-train",
        "distill-prep",
        "distill-mod",
        # ResShift SR/RSD sidecar.  ``sr-setup`` is intentionally excluded:
        # it mutates the WebUI interpreter and must remain a terminal command.
        "sr-prep",
        "sr-phase0",
        "sr-test",
        "sr-build-hr-pool",
        "sr-detect-text",
        "sr-train",
        "sr-rsd-train",
        "sr-rsd-dryrun",
        "sr-rsd-infer",
        "download-models",
        "download-anima",
        "download-sam3",
        "download-mit",
        "download-pe",
        "download-pe-spatial",
        "test-unit",
        "print-config",
        "update",
        "vendor-sync",
        "export-logs",
    }
)

_missing = _WEBUI_COMMAND_NAMES.difference(CLI_COMMANDS)
if _missing:
    missing = ", ".join(sorted(_missing))
    raise RuntimeError(
        f"WebUI task catalog references unknown tasks.py commands: {missing}"
    )

COMMAND_CATALOG = {name: CLI_COMMANDS[name][1] for name in sorted(_WEBUI_COMMAND_NAMES)}

TRAINING_DASHBOARD_COMMANDS = frozenset(
    {
        "lora",
        "lora-gui",
        "staged-train",
        "turbo",
        "easycontrol",
        "exp-spd",
        "exp-chimera",
        "distill-mod",
        "dcw",
        "dcw-train",
    }
)


def task_category(command: str) -> str:
    return "training" if command in TRAINING_DASHBOARD_COMMANDS else "task"
