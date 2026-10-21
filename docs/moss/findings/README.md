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
| [sigma_staircase.md](sigma_staircase.md) | One forward's `x0_pred` resolves at Anima's σ knee (recognisable 0.74), but layout is set by σ 0.87–0.80 and shift 5 puts that at steps 22–28 of 50. Base and SFT resolve alike. |
| [spectrum_schedule.md](spectrum_schedule.md) | Spectrum's full-step schedule re-checked against the staircase: the shipped growing window beats log-SNR, step-even and layout-dense schedules at NFE 19 and an NFE-16 one. An open-loop skip-cost score ranks them backwards. |
