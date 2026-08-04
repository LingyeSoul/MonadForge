@CLAUDE.md

## 常用子项目

- `scripts/run_analyzer/` — 训练档案分析器（独立 Web 工具）。启动：
  `.venv/bin/python -m scripts.run_analyzer.server --port 8320`。
  读取 progress.jsonl / tensorboard / snapshot.toml / stdout.log / job.json。
  UI 设计遵循 `~/Projects/toolbox/终末地官网/analysis/design-system-report.md`。
- `webui/` — 主 WebUI（FastAPI + Vue3/Vuetify），实时训练看板与数据集预处理。
- `scripts/daemon/` — 训练守护进程（HTTP API + MCP + job 队列）。
- `library/training/progress.py` — progress.jsonl 事件流写入方（run_start/step/ckpt/sample/run_end）。

## 数据约定

- 每个 daemon 训练 job：`output/daemon/jobs/<job_id>/`（job.json + progress.jsonl + stdout.log + sample/）
- 日志与快照：`output/logs/<run>_<时间戳>/`（snapshot.toml + network_train/tfevents）
- 事件流 schema 见 `library/training/progress.py` 顶部注释
