"""Config merge chain — ``anima_lora.config``.

Both exports live in ``library.config.io``: ``load_method_preset`` applies the
``base.toml → presets.toml[<preset>] → methods/<method>.toml`` chain;
``read_config_from_file`` layers CLI overrides on a parsed namespace.
"""

from __future__ import annotations

from anima_lora._lazy import attach

attach(
    globals(),
    {
        "load_method_preset": "library.config.io",
        "read_config_from_file": "library.config.io",
    },
)
