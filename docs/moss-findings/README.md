# MOSS-Image findings

Observations and analyses of MOSS-Image (`library/moss/`) that are useful but are not
a shipped feature. Each doc keeps two things apart: what was measured (with the run
it came from) and the hypotheses the measurements suggest, which are not yet tested.
Run logs, probe code and raw numbers stay with the bench (`bench/moss_*/`, the folder
named after the item); a doc here is the summary to read first.

| Doc | What it covers |
|---|---|
| [attention_sink_chat_template.md](attention_sink_chat_template.md) | The 10 chat-template tokens are MOSS's attention sink: an OFF-gate plus a per-head bias written into every image token. Trained LoRAs barely move it. |
| [vision_tower_reference.md](vision_tower_reference.md) | An image put through Qwen3.5's own vision tower conditions MOSS zero-shot: content and coarse layout come back, pixel alignment does not, and the DiT reads the image-token rows. |
