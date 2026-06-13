# ReFT — re-integration map

ReFT was **removed from the live codebase on 2026-06-08** and downgraded to this
bench probe (see `plan.md`). It was unproven on Anima (off-by-default, never
benched, not foldable). This file is the obituary-and-restore map: if the bench
in `plan.md` returns `REFT EARNS A NICHE`, re-applying the points below restores
the shipped wiring exactly as it was. If it returns `REFT IS REDUNDANT`, this
directory is the only record kept.

The removal commit's diff is the authoritative source; this file is the index so
re-integration doesn't require commit archaeology.

## Preserved source (`impl/`)

| File | Was at | Notes |
|---|---|---|
| `impl/reft.py` | `networks/lora_modules/reft.py` | The module — `forward` is the whole story. Copy back verbatim. |
| `impl/reft.toml` | `configs/gui-methods/reft.toml` | ReFT-only GUI variant (kept LoRA at rank 16). |
| `impl/tlora_ortho_reft.toml` | `configs/gui-methods/tlora_ortho_reft.toml` | The old "default recommended stack". On removal it was re-saved **without** ReFT as `configs/gui-methods/tlora_ortho.toml`; this is the pre-removal copy with ReFT still in it. |
| `impl/docs_methods_reft.md` | `docs/methods/reft.md` | Method deep-dive. |
| `impl/docs_structure_reft.md` | `docs/structure/reft.md` | Structure deep-dive. |
| `impl/guides/{en,ja,ko,cn}_reft.html` | `gui/explanations/guides/<lang>/reft.html` | GUI help HTML, 4 languages. |

## Live-code hook points removed (re-apply in this order)

The module composes as an additive side-channel — every hook is a small block,
not a rewrite. Restoring is mechanical.

### 1. Module registration
- `networks/lora_modules/reft.py` — restore file from `impl/reft.py`.
- `networks/lora_modules/__init__.py` — re-add `from networks.lora_modules.reft import ReFTModule` and `"ReFTModule"` to `__all__`.
- `networks/__init__.py` — re-add the 4 SHARED_KWARG_FLAGS: `add_reft`, `reft_dim`, `reft_alpha`, `reft_layers` (the "ReFT add-on" block). **Required** — a cfg+TOML net kwarg is inert and fails the config test until registered here.

### 2. Config (`networks/lora_anima/config.py`)
- `LoRANetworkCfg` dataclass: re-add the `# ReFT` field block — `add_reft: bool = False`, `reft_dim: int = 4`, `reft_alpha: Optional[float] = None`, `reft_layers: object = "all"`.
- `from_kwargs`: re-add the parse block (`add_reft = _as_bool(...)`, `reft_dim` defaulting to `network_dim`, `reft_alpha`, `reft_layers` default `"all"`) and the 4 fields in the returned `cls(...)`.
- `from_weights`: re-add the keyword-only params `has_reft: bool`, `reft_dim: Optional[int]`, `reft_block_indices`, and in the returned `cls(...)`: `add_reft=has_reft`, `reft_dim=reft_dim if reft_dim is not None else 4`, `reft_layers=sorted(reft_block_indices) if has_reft else "all"`.

### 3. Loading (`networks/lora_anima/loading.py`)
- Restore `_parse_reft_layers(spec, num_blocks)` — resolves a `reft_layers` spec (`"all"`, `"last_N"`, `"every_N"`, explicit list) to a sorted list of block indices.

### 4. Factory (`networks/lora_anima/factory.py`)
- `create_network`: re-add the `if cfg.add_reft:` logging block (`reft_dim`/`reft_alpha`/`layers`).
- `create_network_from_weights` key-sniff loop: re-add `has_reft`/`reft_dim`/`reft_block_indices` init, the `_reft_block_re = re.compile(r"^reft_unet_blocks_(\d+)$")`, and the `if lora_name.startswith("reft_"):` branch (block-index parse, `reft_dim` from `rotate_layer` weight, `continue`). Pass `has_reft=`, `reft_dim=`, `reft_block_indices=` into the `LoRANetworkCfg.from_weights(...)` call.

### 5. Network (`networks/lora_anima/network.py`) — the bulk
- Imports: `_parse_reft_layers` from `.loading`, `ReFTModule` from `..lora_modules`.
- `__init__` local aliases: `add_reft`, `reft_dim`, `reft_alpha`, `reft_layers` from `cfg`.
- Construction block (was ~L635): build `self.unet_refts` / `self.text_encoder_refts` lists; when `add_reft`, wrap each block index from `_parse_reft_layers(reft_layers, num_blocks)` in a `ReFTModule(embed_dim=block.x_dim, …)`, `original_name = f"blocks.{idx}"`.
- Duplicate-name assertion + the param-iteration lists must include `+ self.text_encoder_refts + self.unet_refts`.
- `set_multiplier`: propagate `self.multiplier` to `self.text_encoder_refts + self.unet_refts`.
- `set_reft_timestep_mask(timesteps, max_timestep=1.0)`: full method — builds `_shared_reft_mask`, rebinds each reft's `_timestep_mask`, power-law `r = frac.pow(alpha_rank_scale) * (reft_dim-1) + 1`.
- `clear_timestep_mask`: re-add the `_shared_reft_mask` fill-to-ones tail.
- `apply_to`: reset `self.text_encoder_refts = []` / `self.unet_refts = []` when text-encoder/unet not applied, and the `for reft in …: reft.apply_to(); self.add_module(reft.lora_name, reft)` loop.
- `prepare_args_for_optimizer` (the LR-group assembly, was ~L1899): re-add the `if self.text_encoder_refts:` and `if self.unet_refts:` param groups ("reft textencoder" / "reft unet").

### 6. Metrics (`networks/lora_anima/network_metrics.py`)
- `regularization()`: re-add the `for reft in self.text_encoder_refts + self.unet_refts: total_reg += reft.regularization()` loop (the `‖RRᵀ−I‖²` term).

### 7. Training hook (`library/training/forward/router_conditioning.py`)
- Re-add the `if hasattr(network, "set_reft_timestep_mask"): network.set_reft_timestep_mask(timesteps, max_timestep=1.0)` call (sequence: timestep_mask → reft_timestep_mask → sigma → fei → balance).

### 8. Merge refusal (`scripts/merge_to_dit.py`, `gui/tabs/merge_tab.py`)
- `merge_to_dit.py`: re-add `"reft_": "ReFT (block-level hook)"` to the unsupported-key dict + the doc/help mentions.
- `merge_tab.py`: re-add the `"reft"` counter, the `k.startswith("reft_")` count, and the verdict branches (`merge_verdict_reft_only` when reft-only, partial when reft+lora).

### 9. GUI surface (`gui/`)
- `gui/__init__.py`: `"reft"` in the methods list + form-field names `add_reft`, `reft_dim`, `reft_alpha`, `reft_layers`.
- `gui/app.py`: `"reft"` in the `methods=[...]` list.
- `gui/explanations/__init__.py`: `"reft"` in the two method sets/lists.
- Restore `gui/explanations/guides/<lang>/reft.html` from `impl/guides/`.

### 10. i18n (`gui/i18n/{en,ja,ko,cn}.py`)
- The 3 merge strings still reference ReFT generically (`merge_verdict_partial`, `merge_verdict_reft_only`, `merge_allow_partial`). On removal, `merge_verdict_reft_only` was dropped and the other two reworded to drop "ReFT". Re-add the ReFT wording across all 4 languages.

### 11. Config TOML (re-enable the variant)
- `configs/gui-methods/reft.toml` — restore from `impl/`.
- `configs/gui-methods/tlora_ortho_reft.toml` — restore from `impl/` (or re-add the ReFT block to `tlora_ortho.toml`).
- `configs/methods/lora.toml` — re-add the commented `# reft` toggle block + the header mention.

### 12. Docs
- Restore `docs/methods/reft.md` + `docs/structure/reft.md` from `impl/`.
- Re-thread the README / CLAUDE / networks/CLAUDE / docs index / guidebook (4-lang) mentions.

## Key facts the bench must honor (from removal-time state)

- **Block-level only.** Keys are `reft_unet_blocks_<idx>.*`. The per-Linear ReFT
  wiring was already retired — the loader hard-errors on non-block keys.
- **Not foldable.** `merge` refuses `reft_` keys; ComfyUI's weight-patcher
  silently drops them — needs the `AnimaAdapterLoader` custom node.
- **Default `reft_layers` in the shipped variants was `"last_8"`**, `reft_dim`
  16/32/64 (see `plan.md` param table).
- **T-LoRA mask hits ReFT too** via `set_reft_timestep_mask` (separate shared
  mask from the LoRA one) — training-only.
