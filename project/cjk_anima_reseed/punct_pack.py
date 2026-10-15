"""The punct base pack (user, 10-05, after the green leaf): the raw pack with

- the encode fold widened: ``〜`` → ``～`` (one wave row, ext 87), ``―`` → ``ー``,
  ``，`` → ``、``;
- dot runs (``mapping["dots"]``, ``ext_vocab.Dots``): one ``・`` → ``.``, 2–3
  dots → ``…``, 4+ → ``……``; ``...`` / ``…`` only beside a routed char;
- a ``…`` row appended (Qwen 1940, routed), at T5's ``...`` row (233) — what
  ``…`` encoded as before, so the row starts where the render was.

Every other row and id is the raw pack's, so the seed rows (``trained.pt``
deltas over ext ids < 69 558) ride on it unchanged. CPU:

    .venv/bin/python project/cjk_anima_reseed/punct_pack.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from library.anima.ext_vocab import T5_TABLE_SIZE, load_ext_assets, pack_digest  # noqa: E402
from library.env import default_checkpoints, resolve_under_home  # noqa: E402

BASE = resolve_under_home("models/vocab_packs/anima_cjk_vocab_pack")
OUT = resolve_under_home("models/vocab_packs/anima_cjk_vocab_pack_punct")
FOLD = {"〜": "～", "―": "ー", "，": "、"}
DOTS = {
    "run": "・･.．‥…",
    "count": {"‥": 2, "…": 3},
    "lone": {"・": ".", "･": "."},
    "anchored": ".…",
    "plain": {"…": "..."},
    "to": "…",
    "long": "……",
    "long_at": 4,
}
ELLIPSIS_QWEN = 1940  # Qwen's `…`
T5_DOTS = 233  # T5's `...`


def t5_row(i: int) -> torch.Tensor:
    with safe_open(default_checkpoints().dit, "pt") as f:
        return f.get_tensor("net.llm_adapter.embed.weight")[i].float()


def build() -> tuple[torch.Tensor, dict, dict]:
    table, mapping = load_ext_assets(BASE)
    assert "iso" not in mapping, "the raw pack has no iso block"
    n = int(table.shape[0])
    assert int(mapping["rows"]) == n == mapping["sym_rows"][1]
    m = json.loads(json.dumps(mapping, ensure_ascii=False))
    assert str(ELLIPSIS_QWEN) not in m["sym"] and "…" not in m["sym_char"]
    m["fold"] = {**m["fold"], **FOLD}
    m["dots"] = DOTS
    m["sym"][str(ELLIPSIS_QWEN)] = n
    m["sym_char"]["…"] = n
    m["route"]["chars"] += "…"
    m["sym_rows"] = [m["sym_rows"][0], n + 1]
    m["rows"] = n + 1
    m["provenance"].append("t5")
    table = torch.cat([table, t5_row(T5_DOTS)[None].to(table.dtype)])
    with safe_open(str(BASE.with_suffix(".safetensors")), "pt") as f:
        meta = dict(f.metadata() or {})
    blocks = json.loads(meta.get("anima_blocks", "{}"))
    blocks["sym+sym_char (symbols)"] = m["sym_rows"]
    meta.update(
        anima_rows=str(n + 1),
        anima_blocks=json.dumps(blocks, ensure_ascii=False),
        anima_pack_label=meta.get("anima_pack_label", "") + "+punct",
        anima_punct=json.dumps(
            {"fold": FOLD, "dots": DOTS, "ellipsis_row": n, "init": f"t5 {T5_DOTS}"},
            ensure_ascii=False,
        ),
    )
    return table, m, meta


def main():
    table, m, meta = build()
    save_file(
        {"ext_embed": table.contiguous()}, str(OUT.with_suffix(".safetensors")), meta
    )
    OUT.with_suffix(".json").write_text(
        json.dumps(m, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"{OUT.name}: {table.shape[0]} rows, `…` = ext {m['sym_char']['…']} "
        f"(id {T5_TABLE_SIZE + m['sym_char']['…']}), sha {pack_digest(table, m)[:12]}"
    )


if __name__ == "__main__":
    main()
