# Config chain & placeholder substitution

How training configuration is merged from layered TOML files, and how the
`{placeholder}` mechanism wires top-level path keys into the dataset
blueprint. This is the usage guide for the mechanism; [`base-config.md`](base-config.md)
documents what each key *means*.

Implementation lives in `library/config/io.py` (merge + substitution) and
`library/config/loader.py` (blueprint → dataset objects).

---

## The merge chain

```
configs/model.toml                     # model checkpoints (repo defaults)
  → configs/base.toml                  # shared infrastructure + dataset blueprint
    → configs/custom/model.toml        # machine-local model paths (gitignored)
      → configs/presets.toml[<preset>] # hardware profile: [default] [low_vram] [graft] [half] [quarter] [tenth] [debug]
        → configs/methods/<method>.toml (or configs/gui-methods/<variant>.toml + custom overlay)
          → CLI args                    # always win
```

Later layers override earlier ones key-by-key; **method beats preset** on
overlap (a frozen-DiT method can force its own hardware constraints). Each
layer is a flat scalar table — sections are flattened before merging, so the
chain is one namespace, not a tree.

Select layers with `--method` / `--preset` (or `METHOD=` / `PRESET=` env when
going through `make` / `tasks.py`). `--methods_subdir gui-methods` switches the
method layer to the per-variant files used by the WebUI.

### Inspecting the result

```bash
make print-config METHOD=lora PRESET=default   # merged config, annotated with each key's source layer
```

Training writes the same annotated dump to
`<output_dir>/<output_name>.snapshot.toml` (and mirrors it into the run's
TensorBoard dir), so every checkpoint carries a record of the config that
produced it.

### gui-methods variants

For `--methods_subdir gui-methods`, the effective method layer is
`{**configs/gui-methods/<variant>.toml, **configs/custom/variants/<variant>.toml}`
— the user overlay inherits every builtin knob and wins on conflict. This
merge happens for *both* the flat config chain and the dataset blueprint, so
the WebUI and training always see the same effective config. A sparse overlay
must stay sparse: a subset table in a gui overlay is treated as an override of
base's subsets (see below), never as a fresh blueprint.

---

## Single-file mode (`--config_file`)

`--config_file path/to/file.toml` bypasses the chain: the file is loaded via
`_load_toml_with_base`, CLI args are re-parsed on top as usual. One extra
feature: a `base_config = "other.toml"` key inside the file pulls in a parent
config recursively — **resolved relative to the referencing file's directory**,
not the working directory.

Note the dataset blueprint behaves differently here: `--dataset_config` (raw
dataset TOML) skips `base.toml` entirely, while a `--config_file` that carries
its own `[general]`/`[[datasets]]` sections *replaces* base's blueprint — see
[Dataset blueprint overrides](#dataset-blueprint-overrides).

---

## Placeholder substitution

The dataset blueprint (`[general]` / `[[datasets]]`) may reference top-level
path keys with `{curly}` placeholders:

```toml
# configs/base.toml (top level)
resized_image_dir = "post_image_dataset/resized"
lora_cache_dir    = "post_image_dataset/lora"

[[datasets]]
  [[datasets.subsets]]
  image_dir = "{resized_image_dir}"   # ← substituted at load time
  cache_dir = "{lora_cache_dir}"
```

`load_dataset_config_from_base` substitutes every string in the blueprint tree
(recursively through `[[datasets]]` tables and their subsets) **after** the
merge chain has resolved the top-level scalars. That ordering is the whole
point: overriding `resized_image_dir` in a preset, method file, on the CLI, or
in the WebUI automatically flows into the subset's `image_dir` — one key to
change, blueprint stays in sync with the preprocess commands.

### What the placeholder resolves against

The context is built in two steps (`load_dataset_config_from_base`):

1. Every **top-level string scalar** of the file that supplied the blueprint
   (`configs/base.toml`, or the `--config_file`/method file that replaced it).
2. Updated with every **string value from the merged config namespace** —
   i.e. the full chain result *including CLI overrides*. Step 2 wins.

So all of these change where training reads images from:

```toml
# per-method or per-preset: add a top-level scalar to configs/methods/<method>.toml
# (or a custom preset / configs/custom/variants/<variant>.toml)
resized_image_dir = "post_image_dataset/runs/mydata-abc123/resized"
```

```bash
# WebUI flow: pick a completed preprocess run; its directories are pinned into
# these keys before the blueprint loads (_apply_preprocess_run in train.py)
python train.py --method lora --preprocess_run post_image_dataset/runs/<source>-<hash>/<config-hash>/manifest.json
```

> There is **no** `--resized_image_dir` CLI flag — these path keys live in the
> TOML layers and the preprocess-run injection, not argparse. The mechanism
> itself is generic, though: *any* string value present in the merged config
> namespace feeds the placeholder context.

### Rules and limits

- **Strings only.** The context collects `isinstance(v, str)` values. An int,
  bool, or list key referenced as `{key}` is *unknown* — see next rule.
- **Unknown keys stay literal.** `{no_such_key}` passes through untouched —
  no error. This lets captions and paths containing braces survive intact
  (`_SafeFormatDict`).
- **Malformed templates are left as-is** rather than raising (the substitution
  wraps `format_map` and swallows `ValueError`/`IndexError`).
- **Only the blueprint is substituted.** Placeholders inside preset/method
  scalar values or CLI args are not processed.
- **`--dataset_config` files get no substitution.** A raw dataset TOML passed
  with `--dataset_config` is loaded verbatim by `load_user_config`; write
  absolute paths (or cwd-relative ones) there, not `{placeholders}`.

### Adding your own placeholder

1. Declare the top-level scalar in `configs/base.toml`
   (e.g. `mask_dir = "post_image_dataset/masks"`).
2. Reference it in the blueprint: `mask_dir = "{mask_dir}"` — works for any
   subset key the dataset schema accepts.
3. Override from any higher layer (preset / method / gui-variant overlay /
   WebUI); the blueprint follows automatically.

Prefer reusing the existing keys (`resized_image_dir`, `lora_cache_dir`)
when they fit — preprocess and training agree on them by construction.

---

## Dataset blueprint overrides

`[general]` / `[[datasets]]` blocks in a **method** TOML interact with base's
blueprint in one of two ways (`_apply_dataset_overrides` /
`load_dataset_config_from_base`):

| Method file contains | Effect |
|---|---|
| A **full blueprint** (`[[datasets]]` with at least one `[[datasets.subsets]]`) | Base's blueprint is **replaced wholesale** — including base's `image_dir`/`cache_dir`. Template placeholders then resolve against the *method file's* own top-level scalars. This is the self-contained per-method layout (`configs/easycontrol/easycontrol.toml`). |
| Scalar-only `[general]` / `[[datasets]]` (no subsets) | **Shallow merge onto base**: dataset-level scalars by index; subset-level scalars matched **by index** (`[[datasets.subsets]]` #0 overrides base's subset #0). Extra subsets in the override are ignored with a warning. |

gui-methods overlays are always on the shallow path — a sparse
`configs/custom/variants/<variant>.toml` can never drop base's
`image_dir`/`cache_dir`.

Per-subset keys that don't map by index (a *new* subset, reordering) or
arbitrary one-off blueprints go through `--dataset_config <file>` instead —
remember that file gets **no placeholder substitution**.

The `[half]`-family presets set a global `sample_ratio` (0.5 / 0.25 / 0.1)
that shrinks the **train** pool only; validation counts stay exact.

---

## Path anchoring: what is relative to what

The chain resolves *files* under one root and *path values* under another —
knowing which is which prevents most "why is it reading that directory" bugs:

| Path | Anchor | Where enforced |
|---|---|---|
| Chain TOMLs (`configs_dir`) | repo home (`anima_home()`, override with `$ANIMA_HOME`) — NOT the cwd | `library/env.py` `resolve_under_home`, applied at every chain entry point |
| `base_config` reference in a config file | the referencing file's directory | `_load_toml_with_base` |
| Model paths (`pretrained_model_name_or_path`, `vae`, `qwen3`), `network_weights`, `base_weights`, resume checkpoints | repo home | `train.py` via `resolve_under_home` |
| `image_dir`, `cache_dir` (blueprint values after substitution) | **the training process's cwd** — never absolutized | dataset loader globs the value as-is |
| `output_dir`, `logging_dir` | the training process's cwd | `output_layout.py` keeps them relative when relative |
| Paths inside `run_manifest.json` | the run root (`output/ckpt/<name>/`) — so output trees survive being copied between machines | `output_layout.py` |

The cwd dependency is deliberate: the daemon launches every job with
`cwd = repo root` (`scripts/daemon/manager.py`), so repo-relative values in
configs behave consistently. If you run `train.py` by hand from another
directory, use absolute paths for dataset/output keys (model paths and config
*file locations* stay fine — they anchor to the repo home).

---

## The preprocess side

`make preprocess` reads path knobs through `load_path_overrides`, a sibling
chain with one twist:

```
configs/custom/preprocess.toml  →  base.toml  →  preset  →  method
        (gitignored, WebUI-owned)      ↑ legacy copies of preprocess keys in
                                         base.toml still win — except target_res,
                                         which preprocess.toml owns outright
```

`source_image_dir` (raw input) and `resized_image_dir` / `lora_cache_dir`
(outputs) land here; because training interpolates the same two keys into its
blueprint, re-pointing them once re-targets both pipelines. Training itself
seeds only the dataset-contract fields (`target_res`, `multires_per_image`)
from `custom/preprocess.toml`; the rest are preprocess-only.

---

## Gotchas checklist

- Method file with a full `[[datasets]]`+subsets blueprint **replaces** base's
  blueprint — don't expect base's `image_dir` to survive.
- `{placeholders}` in a `--dataset_config` file are **not** substituted.
- Non-string scalars can't be referenced by placeholders (they stay literal).
- Blank-string `network_weights` is dropped as "unset" at merge time
  (`_EMPTY_STRING_IS_UNSET`) — older UIs serialized unset paths as `""`.
- Subset overrides match **by index**; inserting a subset in the middle of a
  method's override list shifts what it overrides.
- Turbo (`configs/methods/turbo.toml`) is a bespoke sectioned schema read by
  `scripts/distill_turbo/`, not the flat chain — `print-config` doesn't apply.

---

## See also

- [`base-config.md`](base-config.md) — per-key reference for `base.toml` and the blueprint.
- [`training.md`](training.md) — method/variant selection.
- `library/config/io.py` — merge chain, substitution, provenance (implementation).
- `library/config/loader.py` — blueprint → dataset objects.
- `CLAUDE.md` → **Config flow** — the authoritative invariant list.
