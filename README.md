<h1 align="center">太初玄鼎 · MonadForge</h1>

<p align="center">
  <img src="webui/frontend/public/logo.svg" alt="MonadForge" width="120">
</p>

<p align="center">
  <strong>太初玄鼎 · 下一代 Anima 扩散模型训练工作台</strong>
</p>

<p align="center">
  <a href="https://github.com/LingyeSoul/MonadForge">GitHub</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#功能概览">功能概览</a> ·
  <a href="#与上游版本的区别">与上游版本的区别</a>
</p>

---

MonadForge 是 [anima_lora](https://github.com/sorryhyun/anima_lora) 的一个深度改进分支，将原有的 PySide6 桌面 GUI 全面重构为 **基于 FastAPI + Vue 3 的现代化 WebUI**，并在此基础上扩展了大量工程化改进与用户体验增强。

> 🌐 **双语界面** — 内置中英文切换，代码注释与界面文案全面国际化  
> ⚡ **实时训练监控** — WebSocket 流式传输训练指标、损失曲线、学习率变化与预览图  
> 🎨 **Material Design 3 主题** — 琥珀色品牌系统、JetBrains Mono 等宽字体、精致的交互动效  
> 🖥️ **跨平台一致体验** — Windows / Linux 统一的浏览器界面，告别 Qt 平台差异

---

## 快速开始

### Windows 一键安装

```powershell
# 1. 克隆仓库
git clone https://github.com/LingyeSoul/MonadForge.git
cd MonadForge

# 2. 运行安装脚本（自动安装 uv、依赖、构建前端）
setup-win.bat

# 3. 启动 WebUI
start-webui-win.bat
```

安装脚本会自动完成：
- 检测并安装 [uv](https://astral.sh/uv) 包管理器
- 同步 Python 3.13 + PyTorch 2.12 + CUDA 13.2 环境
- 自动下载便携版 Node.js（如系统未安装）
- 构建 Vue 3 前端并输出到 `webui/frontend/dist/`
- 自动打开浏览器访问 `http://127.0.0.1:8000`

### Linux 一键安装

```bash
# 1. 克隆仓库
git clone https://github.com/LingyeSoul/MonadForge.git
cd MonadForge

# 2. 运行安装脚本
chmod +x setup-linux.sh
./setup-linux.sh

# 3. 启动 WebUI
chmod +x start-webui-linux.sh
./start-webui-linux.sh
```

### 手动安装（开发者）

```bash
# 1. 环境准备
uv sync                                    # Python 3.13 + 所有依赖

# 2. 构建前端
cd webui/frontend
npm install
npm run build
cd ../..

# 3. 下载模型
python tasks.py download-models          # DiT + Qwen3 TE + QwenImage VAE

# 4. 启动服务
python tasks.py daemon                     # 启动 daemon（它会自动拉起 WebUI）
#    或直接：python -m webui（此时 WebUI 会自行 ensure_daemon）
#    Windows 双击 start-webui-win.bat = 启动 daemon + WebUI + 系统托盘（见下）
```

服务启动后访问 `http://127.0.0.1:8000`。

### 训练守护进程（Daemon）与系统托盘

MonadForge 由一个**本地训练守护进程**（`scripts/daemon/`，`127.0.0.1:8765`）统筹：
它既是串行作业队列，也是 WebUI 服务器的宿主。

- **串行队列 + GPU 守卫** —— 同时只跑一个训练任务，避免多任务抢占显存导致 OOM
- **持久化** —— 任务状态落盘到 `output/daemon/`，WebUI/托盘关闭重启后任务继续跑
- **chain_train** —— "预处理 → 训练"可串联，提交者关闭终端后链路仍继续
- **托管 WebUI**（sidecar）—— daemon 启动时拉起 uvicorn WebUI 子进程，崩溃自动重启、
  daemon 关闭时一并退出。WebUI 作为 sidecar **不计入串行队列**，不影响训练调度
- **MCP 桥** —— AI agent（如 Claude Code）可直接提交/监控/停止训练任务

```bash
python tasks.py daemon              # 启动 daemon（幂等、后台常驻；同时拉起 WebUI）
python tasks.py daemon-status       # 查看状态 + 作业列表（JSON）
python tasks.py daemon-attach       # 实时跟随当前作业输出
python tasks.py daemon-kill         # 停止当前作业（daemon 继续运行）
python tasks.py daemon-terminate    # 关闭整个 daemon（WebUI 一并退出）
# CLI 训练加 --queue 即提交到队列而非直接运行:
python tasks.py lora --queue --network_dim 32
# 仅 CLI/ComfyUI 场景不需要 WebUI 时禁用 sidecar:
set ANIMA_DAEMON_HOST_WEBUI=0 && python tasks.py daemon
```

> 若希望 WebUI 单独运行（不由 daemon 托管），仍可 `python -m webui`，它会在启动时
> 自行 `ensure_daemon`（daemon 已在跑则跳过）。

**Windows 系统托盘**（`scripts/tray/`）：状态指示器 + 控制器。
```bash
.venv\Scripts\pythonw.exe -m scripts.tray   # 或双击 start-tray-win.bat
```

托盘图标颜色反映 daemon 状态（🟢 空闲 / 🟡 运行中，含作业名+步数 / 🔴 出错 / ⚪ 未运行），
右键菜单可：打开 WebUI、暂停/恢复队列、停止当前作业、重启 daemon、**切换语言（中/英）**。
托盘语言独立于 WebUI，默认中文，选择持久化在 `output/daemon/tray-prefs.json`。

### 📖 详细训练教程

如果你在 **AutoDL 云端** 用 **RTX 5090** 训练 Anima 风格 LoRA，建议阅读完整教程：

📘 [`docs/guidelines/太初玄鼎训练炉使用指南.md`](docs/guidelines/太初玄鼎训练炉使用指南.md)

该文档涵盖：选机与镜像、首次部署、日常开机重启、数据集准备与打标、Preprocess 三步流水线、
Config 页面参数详解、训练启动与 resume 策略、Loss / LR / 预览图解读、风格 LoRA 调参配方，
以及常见错误排查表与命令速查。

---

## 功能概览

MonadForge WebUI 提供完整的训练工作流覆盖，从数据准备到模型合并一站式完成：

| 页面 | 功能描述 |
|------|---------|
| **⚙️ Config** | 可视化配置编辑器 — 方法/变体/预设选择、TOML 字段级编辑、实时帮助提示、自定义变体与预设创建 |
| **📁 Dataset** | 数据集浏览器 — 多目录管理、缩略图网格/列表视图、搜索筛选、排序、图片详情与标签编辑 |
| **🔧 Preprocess** | 预处理流水线 — 状态仪表盘、数据集路径配置、一键执行 resize/VAE cache/TE cache/PE cache/mask，并支持同一源图在单个 epoch 内遍历多个分辨率档位 |
| **🧩 Adapter** | 适配器管理 — 查看已训练 LoRA/HydraLoRA 检查点、版本信息、元数据浏览 |
| **🧪 Distill** | 蒸馏配置 — Modulation Guidance 蒸馏参数编辑、训练启动、状态监控 |
| **🔀 Merge** | 模型合并 — 文件树浏览、合并策略选择、多检查点融合为独立 DiT |
| **📊 Dashboard** | 训练仪表盘 — 实时进度环、损失曲线、学习率曲线、GPU/CPU/内存监控、训练预览图流 |
| **📋 Tasks** | 任务监控 — 所有后台任务状态、实时日志流（彩色分级）、取消/重试操作 |
| **🔧 System** | 系统管理 — 环境变量查看、硬件信息、模型路径配置、一键更新 |

### 核心特性详解

#### 🎯 实时训练监控（Training Dashboard）

- **WebSocket 流式数据** — 训练过程中通过 `/api/ws` 实时推送 step/epoch/loss/lr/speed
- **动态损失曲线** — 基于 Canvas 绘制的实时 Loss Curve，支持历史数据点回溯
- **学习率可视化** — 独立的 LR Curve 面板，清晰展示 warmup / decay 过程
- **GPU 监控** — 显存占用、利用率、温度实时条形图（基于 `nvidia-ml-py`）
- **训练预览图** — 每 N 步生成的 sample 图片自动流式展示，支持点击放大查看

#### 🌐 国际化（i18n）

- 内置 **English / 简体中文** 双语支持
- 界面语言一键切换，无需重启
- 所有训练概念、字段帮助、错误提示均已翻译
- 基于 Vue 组合式 API 的 `useI18n` 系统，易于扩展新语言

#### 🎨 Material Design 3 品牌主题

- **琥珀色（Amber）主色调** — 区别于上游的默认蓝紫，温暖而具有辨识度
- **JetBrains Mono** 等宽字体用于所有数值输入与代码展示
- **精致的交互动效** — 卡片悬浮抬升、焦点发光环、按钮按压反馈、骨架屏加载、抖动错误提示
- **深色模式优先** — 专为长时间训练场景优化的暗色界面

#### 📡 任务系统（Daemon + Task Service）

- WebUI 不再直接启动训练子进程，而是将训练、预处理、合并等操作提交给本地 **training daemon**
- daemon 负责串行作业队列、GPU 守卫、进程树管理、任务取消与状态落盘，避免多个训练任务争抢显存
- WebUI TaskService 作为适配层，尾随 daemon 的 `stdout.log` 与每个 job 独立的 `progress.jsonl`，再通过 WebSocket 推送日志、指标与预览图
- 训练进度使用每任务隔离的 `output/daemon/jobs/<job_id>/progress.jsonl`，避免跨 run 复用 `output/logs/<output_name>.progress.jsonl` 导致旧数据串入
---

## 与上游版本的区别

### 架构层面

| 特性 | MonadForge (本分支) | upstream (sorryhyun/anima_lora) |
|------|---------------------|--------------------------------|
| **GUI 框架** | FastAPI + Vue 3 + Vuetify 3 (WebUI) | PySide6 (Qt 桌面应用) |
| **前端构建** | Vite + TypeScript + Sass | 无（纯 Python Qt） |
| **通信方式** | HTTP REST + WebSocket | 本地进程间通信 / 文件系统轮询 |
| **任务调度** | 本地 training daemon 串行队列 + WebUI TaskService 适配层 | 外部 Python daemon 进程 |
| **更新机制** | `make update` 支持 + 内置更新器 | 仅 `make update` |
| **跨平台** | 浏览器统一体验，完全一致 | Qt 平台差异（Linux 字体渲染等） |
### 功能增强

| 功能 | MonadForge | upstream |
|------|-----------|----------|
| **实时训练仪表盘** | ✅ 完整 — 损失曲线、LR 曲线、GPU 监控、预览图流 | ❌ 仅基础日志输出 |
| **双语界面** | ✅ 内置 EN/CN 切换 | ❌ 仅英文 |
| **训练预览图流** | ✅ WebSocket 实时推送 sample 图片 | ❌ 需手动查看输出目录 |
| **ConvRot int8 底模训练** | ✅ 实验性 W8A16/W8A8 冻结底模路径，WebUI 配置、数值探针与 block-swap 支持（[说明](docs/experimental/convrot_int8_training.md)） | ❌ |
| **WD Tagger 集成** | ✅ 内置 timm-based WD 标签器，WebUI 一键打标 | ❌ 仅命令行 |
| **ControlNet 预处理** | ✅ 内置 canny/depth/pose 预处理与数据集生成 | ❌ 实验性 / 缺失 |
| **蒸馏 UI** | ✅ 完整的 Modulation Guidance 蒸馏配置界面 | ❌ 仅命令行 |
| **自定义数据集路径** | ✅ WebUI 配置多组 source/resized/cache 路径 | ❌ 手动编辑 TOML |
| **单 epoch 多分辨率** | ✅ 每张源图按所选 free-fit 档位分别缓存，并在每个 epoch 中全部参与训练（[使用说明](docs/guidelines/training.md#multiple-resolutions-per-image-in-one-epoch)） | ❌ 每张图只使用一个就近档位 |
| **自定义预设** | ✅ 图形化创建硬件预设（VRAM 档位） | ❌ 仅内置预设 |
| **文件浏览器** | ✅ 内置目录树浏览、模型选择器 | ❌ 系统文件对话框 |
| **WebSocket 日志** | ✅ 彩色分级实时日志流（INFO/WARN/ERROR） | ❌ 文件 tail |
| **主题系统** | ✅ MD3 琥珀色品牌主题 + 动效系统 | ❌ 原生 Qt 样式 |

### 工程改进

| 改进项 | 说明 |
|--------|------|
| **移除 PySide6** | 完全移除 Qt 依赖，减少 ~200MB 安装体积，消除跨平台字体/渲染问题 |
| **Daemon 托管 WebUI** | 训练守护进程成为任务调度中心并可托管 WebUI sidecar；WebUI 只做提交、订阅与展示，训练任务即使浏览器关闭也会继续运行 |
| **Windows 原生优化** | 专用 `setup-win.bat` / `start-webui-win.bat`，自动处理 Defender 排除、Node 便携版、无窗口训练进程 |
| **前端独立构建** | Vue SPA 独立构建输出到 `dist/`，支持 CDN 部署或内嵌到 FastAPI StaticFiles |
| **SPA 路由回退** | 支持直接刷新 `/dashboard`、`/config` 等子路由，不会 404 |
| **SVG 品牌标识** | 矢量 Logo 与 Favicon，任意缩放清晰 |
| **WandB 集成** | 训练指标自动上报，WebUI 内嵌 WandB 看板链接 |
| **JSONL 进度追踪** | 优化的训练日志格式，支持流式解析与历史回放 |

### 来自 lora-rescripts 的算法优化

以下训练优化移植自 [WhitecrowAurora/lora-rescripts](https://github.com/WhitecrowAurora/lora-rescripts)（AGPL-3.0），该项目是 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) 的维护分支，专注于 LoRA 训练效率与算法创新。

| 优化项 | 来源文件 | 说明 | 本项目对应文件 |
|--------|---------|------|---------------|
| **AdamW 8-bit Kahan 补偿求和** | `scripts/stable/library/adamw_8bit_kahan.py` | 在 8-bit 量化 AdamW 上叠加 Kahan 补偿求和，将舍入误差回注下一步，有效精度从 8-bit 提升至 ~11-12 bit，显著改善长训练 run 的收敛稳定性 | `library/training/adamw_8bit_kahan.py` |
| **分阶段混合分辨率训练** | `mikazuki/utils/mixed_resolution.py` | 单进程课程学习：按全局步切换预处理完成的数据行，模型、优化器和调度器状态连续；每个阶段需要独立的图片与缓存目录 | `library/training/stage_schedule.py` |
| **自适应噪声偏移** | `scripts/stable/library/custom_train_functions.py` | 基于 latent 通道统计量的自适应噪声偏移，替代固定噪声偏移，改善生成图像的动态范围 | `library/runtime/noise.py` |
| **对比流匹配损失** | `scripts/stable/train_network.py` | 在 Rectified Flow 目标上叠加对比损失项，通过正/负流方向对比提升 latent 表征质量 | `library/training/losses.py` |
| **金字塔多分辨率噪声** | `scripts/stable/library/custom_train_functions.py` | 多尺度噪声叠加（6 层，discount=0.4），让模型同时接触多频段噪声模式 | `library/runtime/noise.py` |
| **VeRA 适配器** | `scripts/stable/networks/vera.py` | 向量随机矩阵适配 — 跨层共享冻结随机投影矩阵，仅学习对角缩放向量，参数量比标准 LoRA 低 10-100× | `networks/lora_modules/vera.py` |
| **DyLoRA 适配器** | `scripts/stable/networks/dylora.py` | 动态 LoRA — 每次前向随机选择 rank，同时训练多个 rank 级别，训练后可提取任意 rank ≤ max_rank | `networks/lora_modules/dylora.py` |
| **优化器状态 CPU 卸载** | `scripts/stable/library/optimizer_offload_util.py` | 将 Adam 系优化器的 momentum/variance 缓冲区卸载到 CPU RAM，step 时按需搬回 GPU，可节省 ~50% 优化器显存 | `library/training/optimizer_offload.py` |
| **训练步计时分析器** | `scripts/stable/train_network.py` (ExperimentalAttentionStepProfiler) | 窗口化计时分析器，按 forward/backward/optimizer/save 分段统计每步耗时与占比 | `library/training/profiler.py` |

所有特性均为**可选激活**，不改变默认训练行为。通过 CLI 参数或配置文件启用。`--staged_resolution` 仅作为弃用的三行 shorthand 保留；新配置请使用 `stage_schedule_enabled` / `stage_schedule`，并按 [`docs/guidelines/staged-resolution-training.md`](docs/guidelines/staged-resolution-training.md) 预处理每个数据行。

> 感谢 [WhitecrowAurora](https://github.com/WhitecrowAurora) 的出色工作。lora-rescripts 在注意力后端（FlashAttention / SageAttention / SparseAttention）、多平台支持（NVIDIA / AMD ROCm / Intel XPU）、优化器生态（50+ 种优化器）等方面的工程实践为本项目提供了重要参考。

### 优化器生态

MonadForge 支持丰富的优化器选择，以下列出各优化器的来源与许可：

| 优化器 | `--optimizer_type` 值 | 来源 | 许可 | 说明 |
|--------|----------------------|------|------|------|
| **AdamW** | `AdamW` | PyTorch 内置 | BSD-3 | 默认优化器，fused CUDA 实现 |
| **AdamW 8-bit** | `AdamW8bit` | [bitsandbytes](https://github.com/bitsandbytesai/bitsandbytes) | MIT | 8-bit 量化 AdamW |
| **AdamW 8-bit Kahan** | `AdamW8bitKahan` | [WhitecrowAurora/lora-rescripts](https://github.com/WhitecrowAurora/lora-rescripts) | AGPL-3.0 | 8-bit AdamW + Kahan 补偿求和 |
| **PagedAdamW** | `PagedAdamW` / `PagedAdamW8bit` / `PagedAdamW32bit` | [bitsandbytes](https://github.com/bitsandbytesai/bitsandbytes) | MIT | 分页内存优化 |
| **Lion** | `Lion` / `Lion8bit` / `PagedLion8bit` | [lion-pytorch](https://github.com/lucidrains/lion-pytorch) | MIT | 符号动量优化器 |
| **SGD Nesterov** | `SGDNesterov` / `SGDNesterov8bit` | PyTorch / bitsandbytes | BSD-3 / MIT | 经典 SGD + Nesterov 动量 |
| **D-Adaptation** | `DAdaptAdam` / `DAdaptLion` / `DAdaptSGD` 等 | [dadaptation](https://github.com/facebookresearch/dadaptation) | MIT | 自动学习率调整 |
| **Prodigy** | `Prodigy` | [prodigyopt](https://github.com/konstmish/prodigy) | MIT | D-Adaptation 改进版 |
| **Prodigy Plus Schedule-Free** | `ProdigyPlusScheduleFree` | [LoganBooker/prodigy-plus-schedule-free](https://github.com/LoganBooker/prodigy-plus-schedule-free) | MIT | Prodigy + schedule-free |
| **Adafactor** | `Adafactor` | [transformers](https://github.com/huggingface/transformers) (Google) | Apache-2.0 | 内存高效二阶优化器 |
| **CAME** | `CAME` | [kozistr/pytorch_optimizer](https://github.com/kozistr/pytorch_optimizer) | Apache-2.0 | 置信度引导的内存高效优化器，兼容多分辨率批次 |
| **Automagic** | `Automagic` | [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit) → [scvxzf1/anima_lora_webui](https://github.com/scvxzf1/anima_lora_webui) | MIT | 逐参数自适应 LR |
| **Schedule-Free** | `RAdamScheduleFree` / `AdamWScheduleFree` / `SGDScheduleFree` | [schedulefree](https://github.com/facebookresearch/schedulefree) | MIT | 无需调度器 |
| **Rose** | *(注释中，需手动启用)* | [MatthewK78/Rose](https://github.com/MatthewK78/Rose) | — | 无状态优化器 |

使用示例：

```bash
# Adafactor — 大模型低显存训练
python tasks.py lora --optimizer_type Adafactor --lr_scheduler adafactor

# CAME — 置信度引导的内存高效优化器（支持多分辨率批次）
python tasks.py lora --optimizer_type CAME --learning_rate 1e-4

# Automagic — 逐参数自适应 LR（无需调度器）
python tasks.py lora --optimizer_type Automagic --lr_scheduler constant

# Prodigy Plus Schedule-Free
python tasks.py lora --optimizer_type ProdigyPlusScheduleFree --learning_rate 1.0 --lr_scheduler constant
```

### 保留的上游核心能力

MonadForge 完整继承并同步上游的所有训练与推理能力：

- ✅ **Fast LoRA Training** — 恒定 token 分桶 + 逐块 `torch.compile`，RTX 5060 Ti 上 13.4GB VRAM / 1.1s per step
- ✅ **LoRA / OrthoLoRA / T-LoRA** — 三种变体可叠加，无损合并为独立 DiT 检查点
- ✅ **GLoKr (BoRA)** — Kronecker 分解 ΔW + BoRA 双维度权重分解（行/列可训练幅值，[文档](docs/methods/glokr.md)）
- ✅ **Spectrum 推理加速** — 训练无关的 ~1.75× 加速，支持 ComfyUI 节点
- ✅ **DCW / SMC-CFG** — 采样器级 SNR 校正与滑模 CFG 修正
- ✅ **HydraLoRA / ChimeraHydra** — MoE 多专家路由与双池加法架构
- ✅ **EasyControl** — 扩展自注意力图像条件控制
- ✅ **Turbo (DP-DMD)** — 2 步蒸馏生成
- ✅ **DirectEdit** — 流反演图像编辑
- ✅ **Soft Tokens / SPD / Embedding Inversion** — 完整实验方法支持

---

## 项目结构

```
MonadForge/
├── webui/                          # WebUI 主目录
│   ├── server.py                   # FastAPI 应用工厂
│   ├── api/                        # REST API 路由层
│   │   ├── config.py               # 配置读写 API
│   │   ├── tasks.py                # 任务管理 API
│   │   ├── preview.py              # 训练预览图 API
│   │   ├── system.py               # 系统信息 API
│   │   ├── tagger.py               # WD 标签器 API
│   │   ├── distill.py              # 蒸馏 API
│   │   ├── files.py                # 文件浏览 API
│   │   ├── images.py               # 图片服务 API
│   │   ├── preprocess.py           # 预处理 API
│   │   ├── merge.py                # 合并 API
│   │   ├── ws.py                   # WebSocket 端点
│   │   └── i18n.py                 # 国际化文案 API
│   ├── services/                   # 业务逻辑层
│   │   ├── task_service.py         # 异步任务调度核心
│   │   ├── training_log_parser.py  # 训练日志流式解析
│   │   ├── config_service.py       # TOML 配置管理服务
│   │   ├── image_service.py        # 图片处理与缓存
│   │   ├── wd_tagger_service.py    # WD 标签器封装
│   │   ├── distill_service.py    # 蒸馏任务管理
│   │   ├── file_service.py         # 文件系统操作
│   │   ├── merge_service.py        # 模型合并逻辑
│   │   └── preprocess_service.py   # 预处理流水线
│   └── frontend/                   # Vue 3 前端
│       ├── src/
│       │   ├── App.vue              # 根组件（MD3 导航抽屉）
│       │   ├── router.ts           # 路由定义
│       │   ├── main.ts             # 入口（Pinia + Vuetify + i18n）
│       │   ├── views/              # 页面级组件
│       │   │   ├── ConfigEditor.vue
│       │   │   ├── TrainingDashboard.vue
│       │   │   ├── DatasetBrowser.vue
│       │   │   ├── PreprocessView.vue
│       │   │   ├── TaskMonitorView.vue
│       │   │   ├── AdapterView.vue
│       │   │   ├── DistillView.vue
│       │   │   ├── MergeView.vue
│       │   │   └── SystemView.vue
│       │   ├── stores/             # Pinia 状态管理
│       │   ├── composables/        # Vue 组合式函数
│       │   ├── i18n/               # 中英文文案
│       │   ├── plugins/            # Vuetify 主题配置
│       │   └── styles/             # SCSS 全局样式
│       └── public/
│           └── logo.svg              # 品牌 Logo
├── library/                        # 核心训练/推理库（与上游同步）
├── networks/                         # 适配器网络实现（与上游同步）
├── scripts/                          # 任务脚本（与上游同步）
├── tasks.py                          # 跨平台任务入口
├── setup-win.bat / setup-linux.sh    # 一键安装脚本
├── start-webui-win.bat / start-webui-linux.sh  # 启动脚本
├── build-webui-win.bat               # 前端构建脚本
└── pyproject.toml                    # 项目配置（uv 锁定）
```

---

## 常用命令

```bash
# 训练
python tasks.py lora                     # 标准 LoRA 训练
python tasks.py lora-gui                 # 从 GUI 预设训练
python tasks.py easycontrol              # EasyControl 训练

# 推理测试
python tasks.py test                     # 最新 LoRA 推理测试
python tasks.py test SPECTRUM=1          # 启用 Spectrum 加速
python tasks.py test MOD=1               # 启用 Modulation Guidance

# 预处理
python tasks.py preprocess               # 完整预处理流水线
python tasks.py mask                     # 生成 SAM3 掩码

# 工具
python tasks.py merge                    # 合并 LoRA 到 DiT
python tasks.py download-models          # 下载基础模型
python tasks.py --help                   # 查看所有命令
```

WebUI 启动后，上述所有操作均可在浏览器中完成，无需记忆命令行参数。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.13, FastAPI, Uvicorn, WebSocket |
| 前端 | Vue 3, Vuetify 3, Pinia, Vue Router, TypeScript |
| 构建 | Vite, Sass, vue-tsc |
| 包管理 | uv (Python), npm (Node.js) |
| ML 框架 | PyTorch 2.12（Ampere+）/ 2.10+cu129（SM 7.x），Transformers, Diffusers, Flash Attention 2（SM 8.0+） |
| 字体 | JetBrains Mono, Roboto |

> **老显卡兼容方案（SM 7.0 / SM 7.5）**：V100、T4、RTX 2060/2070/2080 等 Volta/Turing 架构显卡需要单独的兼容环境。核心差异：
> - **PyTorch 版本**：使用 `torch==2.10.0+cu129`（而非默认的 2.12），确保 SM 7.x 兼容
> - **不安装 flash-attn**：flash-attn 要求 SM 8.0+（Ampere 及以上），在 SM 7.x 上会产生 NaN
> - **注意力后端**：生产训练使用 `attn_mode="torch"`（SDPA）
> - **精度策略**：默认 bf16 会自动切换为 fp16（V100/T4 无 bf16 张量核心）；训练器自动启用 `lora_fp32_compute`，LoRA rank 分支保持 fp32 计算以降低偏色/语义退化风险（可用 `network_args = ["lora_fp32_compute=false"]` 做 A/B）
> - **torch_compile**：可正常启用，配合 `gradient_checkpointing=true`
>
> **V100 专用安装脚本（适配所有 SM 7.x 老显卡）**：
> - **Windows**: `setup-v100-win.bat` — 一键创建 `.venv` 并安装 torch==2.10.0+cu129
> - **Linux**: `setup-v100-linux.sh` — 自动配置兼容环境
> - **依赖文件**: `requirements-v100.txt` — 不含 flash-attn 的依赖清单
>
> 脚本以 V100 命名，但同样适用于 T4、RTX 20 系列等所有 SM 7.0/7.5 显卡。安装完成后，在 WebUI 或 TOML 配置中使用上述参数即可。
>
> 详见 `bench/v100_flash/README.md`。

---

## 系统要求

- **GPU**: NVIDIA Volta 或更新架构（V100 / T4 / RTX 20 系列及以上）；推荐 Ampere+（RTX 3000 / A100 / RTX 5060 Ti）
- **驱动**: NVIDIA Driver ≥ 595
- **CUDA**: 13.2 Toolkit（用于 `torch.compile` / Triton）
- **Python**: 3.13（由 uv 自动管理）
- **OS**: Windows 10/11, Linux
- **内存**: 16GB+ 系统内存
- **VRAM**: 8GB+（推荐 16GB+ 用于完整功能）

---

## 贡献

欢迎提交 Issue 与 PR！本分支特别关注的改进方向：

- 🎨 **UI/UX 改进** — 新主题、动画、交互细节
- 🌐 **更多语言** — 日语、韩语等社区翻译
- 📱 **移动端适配** — 响应式布局优化
- 🔌 **新功能页面** — 如模型对比、批量推理等

---

## 致谢与许可

MonadForge 基于 [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) 构建，感谢上游项目的卓越工作。

训练算法优化部分移植自 [WhitecrowAurora/lora-rescripts](https://github.com/WhitecrowAurora/lora-rescripts)，感谢该项目在 LoRA 训练效率与算法创新方面的贡献。

优化器部分参考了 [scvxzf1/anima_lora_webui](https://github.com/scvxzf1/anima_lora_webui)（MIT）的实现，感谢该项目在 Anima LoRA WebUI 工程化方面的贡献。Automagic 优化器源自 [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit)（MIT）。ProdigyPlusScheduleFree 来自 [LoganBooker/prodigy-plus-schedule-free](https://github.com/LoganBooker/prodigy-plus-schedule-free)（MIT）。CAME 优化器来自 [kozistr/pytorch_optimizer](https://github.com/kozistr/pytorch_optimizer)（Apache-2.0）。

- **Toolkit 代码**: [MIT](LICENSE)
- **上游衍生部分**: [Apache 2.0](LICENSE-APACHE)（源自 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)）
- **lora-rescripts 衍生部分**: [AGPL-3.0](https://github.com/WhitecrowAurora/lora-rescripts/blob/main/LICENSE)（源自 [WhitecrowAurora/lora-rescripts](https://github.com/WhitecrowAurora/lora-rescripts)）
- **Anima 基础模型权重**: [CircleStone Labs Non-Commercial License v1.0](NOTICE)

---

<p align="center">
  <sub>太初玄鼎 · 以简驭繁</sub>
</p>
