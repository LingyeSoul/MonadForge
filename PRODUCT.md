# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- 主要用户：Anima 扩散模型的 LoRA 训练者社区 —— 在自己的 Windows/Linux 消费级 GPU（8–16GB 显存档位，见 `configs/presets.toml`）上训练角色/风格适配器的创作者与研究者。通过浏览器访问本地 WebUI（http://127.0.0.1:8000）完成数据预处理、训练配置、实时监控与推理验证。（2026-10-06 访谈确认："面向社区的产品"——上手路径与反馈渠道是一级需求。）
- 次要操作者：AI 代理 —— 经 daemon 的 stdio MCP 桥（`scripts/daemon/mcp.py`）提交/监控/停止训练任务。
- 维护者本人（LingyeSoul）同时是日常重度用户，但产品决策以社区用户为准。

## Product Purpose

MonadForge（太初玄鼎）是 Anima DiT（流匹配）扩散模型的一站式训练工作台，覆盖完整链路：

1. 数据预处理：自由比例原生分桶（free-fit token band）、Anima Tagger 自动打标、位置从句（position clauses）、VAE/TE/PE 缓存；
2. 多家族适配器训练：LoRA / OrthoLoRA / T-LoRA / HydraLoRA / FeRA / EasyControl / Soft Tokens / ChimeraHydra / Turbo（DP-DMD 蒸馏）；
3. 推理验证与 LoRA 合并。

存在的理由：将上游 anima_lora 的 PySide6 桌面 GUI 全面重构为跨平台浏览器 WebUI，并在此基础上扩展工程化与体验改进。成功的定义（访谈确认的首要标准）：**用户用它训练出质量更高、产出更快的 LoRA 模型**。

## Positioning

- 基于 FastAPI + Vue 3（Vuetify / Material Design 3）的浏览器 WebUI —— 上游为 PySide6 桌面 GUI；Windows/Linux 一致体验。
- 本地串行训练守护进程（`127.0.0.1:8765`，GPU 互斥守卫、状态持久化、WebUI 作为受管 sidecar）+ `progress.jsonl` 事件流，使长训练任务从 **WebUI / CLI / AI 代理（MCP 桥）三个面**都可观察、可控制 —— 相邻的训练脚本仓库难以照搬的运营模型。
- 训练管线深度工程：free-fit 原生比例分桶、深度自探测 DiT 加载（checkpoint 头部读 block 数）、方法/预设配置合并链。

访谈确认：**独立演进** —— 不受与上游可合并性约束，可自由重构目录结构与配置 schema；但 Anima 生态兼容性（checkpoint/缓存格式、ComfyUI 节点）是必须保留的产品事实。

## Operating Context

- 用户环境：本地单机 Windows/Linux，Python 3.13 + PyTorch CUDA（uv 管理），一键安装脚本（`setup-win.bat` / `setup-linux.sh`），浏览器访问 `http://127.0.0.1:8000`。
- 界面语言：内置中英文切换（WebUI i18n）；Windows 托盘应用（`scripts/tray/`）默认中文，语言选择独立持久化。
- 任务模型：daemon 一次只跑一个训练 job（GPU 守卫）；每个 job 落在 `output/daemon/jobs/<job_id>/`（job.json + progress.jsonl + stdout.log + sample/）；日志与快照在 `output/logs/<run>_<时间戳>/`（snapshot.toml + network_train/tfevents）。事件流 schema 见 `library/training/progress.py` 顶部注释。
- 配置体系：TOML 合并链 `model.toml → base.toml → custom → preset → method → CLI`；"方法 × 硬件预设"组合出全部训练变体。
- 数据约定：训练图 `image_dataset/` + `.txt` 打标 sidecar；缓存在 `post_image_dataset/lora/`。
- 周边生态：ComfyUI 自定义节点（多个独立仓库 + in-tree `custom_nodes/`）；训练档案分析器 `scripts/run_analyzer/`（独立 Web 工具，端口 8320）。

## Capabilities and Constraints

- 已确认能力：预处理链（resize → 自动打标 → 位置从句 → VAE/TE/PE 缓存；幂等、可协调）、多方法训练、WebSocket 实时训练监控（指标/损失/学习率/预览图）、数据集浏览与打标编辑、推理测试与 LoRA 合并、daemon 任务队列 + MCP、Windows 托盘控制。
- 约束：单机 localhost、无鉴权（127.0.0.1 是设计边界，不是缺陷）；GPU 单任务串行；预训练模型经 HuggingFace 下载；MIT 许可（上游 © Seunghyun Ji），`NOTICE` 管辖第三方模型权重。

## Brand Commitments

- 名称：**太初玄鼎 · MonadForge**；logo：`webui/frontend/public/logo.svg`。
- 界面多语言为产品级承诺：WebUI 内置英/简中/韩/日四语切换（`webui/frontend/src/App.vue` languageOptions；README 对外表述为"中英文切换"）。
- README 声明的视觉承诺（仅记录）：Material Design 3 主题、琥珀色品牌系统、JetBrains Mono 等宽字体。
- `scripts/run_analyzer/` 的 UI 设计遵循外部设计系统报告 `~/Projects/toolbox/终末地官网/analysis/design-system-report.md`（AGENTS.md 记载；仅该表面）。

## Evidence on Hand

- `README.md`（中文，产品主页文案）、`README.ko.md`；`docs/methods/`、`docs/structure/`、`docs/experimental/` 方法深度文档；`CONTRIBUTING.md`（分层 PR 制度）。
- 公开产物：`huggingface.co/sorryhyun/anima-turbo-4step`（上游发布的 4 步 turbo 学生模型）；GitHub `github.com/LingyeSoul/MonadForge`。
- 缺失（未来工作不得虚构）：无用户证言、案例研究、基准对比图表、社区采用数据。

## Product Principles

1. **模型产出质量优先** —— 成功以用户训练出的 LoRA/图像质量衡量；UI 与工程改进都服务于训练结果。
2. **社区可用是默认** —— 双语界面、一键安装、清晰上手路径是每个新功能的一级要求，不是事后补充。
3. **可运营而非仅可驱动** —— 长任务必须可观察、可恢复，且可从 WebUI/CLI/AI 代理三个面控制；`progress.jsonl` 事件流是契约。
4. **独立演进** —— 与上游 anima_lora 不保持可合并性；但 Anima 生态兼容性（checkpoint 格式、缓存格式、ComfyUI 节点）必须保留。
5. **配置链即契约** —— 方法/预设 TOML 合并链与磁盘缓存格式是用户依赖的稳定接缝，重构时保持其语义。
