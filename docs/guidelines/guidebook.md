# Anima LoRA 指南手册

本文档是从零开始使用 **Anima LoRA** 训练/推理全流程的中文综合指南。涵盖从 CUDA 驱动安装到数据集准备、训练、推理、ComfyUI 部署的完整过程。本指南仅面向 Windows 初级用户，WSL、Linux 以及（本项目核心目标的）训练优化等内容请参考其他文档。

---

## 目录

1. [系统要求](#1-系统要求)
2. [CUDA 13.0.2 安装](#2-cuda-1302-安装)
3. [Python 环境与仓库准备](#3-python-环境与仓库准备)
4. [Hugging Face 认证与模型下载](#4-hugging-face-认证与模型下载)
5. [数据集准备](#5-数据集准备)
6. [预处理：缩放 · 潜变量 · 文本嵌入缓存](#6-预处理缩放--潜变量--文本嵌入缓存)
7. [GUI 使用方法](#7-gui-使用方法)
8. [训练执行](#8-训练执行)
9. [LoRA / 适配器变体选择指南](#9-lora--适配器变体选择指南)
10. [推理](#10-推理)
11. [部署到 ComfyUI](#11-部署到-comfyui)
12. [更新](#12-更新)

---

## 1. 系统要求

| 项目 | 最低配置 | 推荐配置 |
|---|---|---|
| GPU | **RTX 3060 起步，2xxx 系列及以下不可用** | VRAM 16 GB 以上 |
| 系统内存 | 16 GB | 32 GB 以上 |
| 磁盘 | 60 GB 可用空间 | 200 GB 以上（用于缓存 + 输出累积） |
| 操作系统 | Windows 11 / Ubuntu 22.04+ | Ubuntu 24.04（FA2/CUDA 13 构建稳定） |
| Python | **必须 3.13** | - |

---

## 2. CUDA 13.0.2 安装

需要最新的 CUDA 才能让 PyTorch 2.x + Flash Attention 2 稳定运行。请从 NVIDIA 官方存档下载并安装 13.0.2 版本。

下载页面：<https://developer.nvidia.com/cuda-13-0-2-download-archive>

### 2.1 Windows 安装

1. 在上述页面选择 **Operating System: Windows → Architecture: x86_64 → Version: 11/10 → Installer Type: exe (local)**。
2. 运行下载的 `cuda_13.0.2_windows.exe` → 选择"Express（推荐）"安装。
3. 安装完成后在 PowerShell 中验证：

   ```powershell
   nvidia-smi
   nvcc --version
   ```

4. 如果 `nvcc` 无法识别，请在系统环境变量 `Path` 中添加以下路径：

   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\libnvvp
   ```

5. 重启计算机后再次运行 `nvcc --version` 验证。

> **驱动注意事项**：CUDA 13.x 需要 NVIDIA 驱动 580 以上版本。如果驱动版本较旧，请先通过 GeForce Experience 或 NVIDIA 下载中心进行更新。

### 2.2 （可选）切换到 CUDA 13.2 + torch 2.12 nightly

默认安装的是 CUDA 13.0 + torch 2.11 stable。如果在 RTX 50 系列等显卡上希望获得约 **10% 的训练速度提升**，可以切换到 CUDA 13.2 + torch 2.12 nightly（基准测试参考 [docs/optimizations/cuda132.md](../optimizations/cuda132.md)）。不需要编译工具——`pyproject.toml` 的 URL 中已上传了预编译的精简版 FA2 轮子包，`uv sync` 可以直接下载安装。

操作步骤：

1. **下载并安装 CUDA 13.2。**
   在 <https://developer.nvidia.com/cuda-downloads?target_os=Windows> 页面选择 **Windows → x86_64 → 11/10 → exe (local)**。可以直接覆盖安装在 13.0 之上，两个版本也可以共存（会分别创建 `v13.2` 和 `v13.0` 目录）。

2. **修改 `pyproject.toml` 中的注释切换。** 需要修改两处：

   **(a) torch / torchvision** — 在 `dependencies` 中注释掉"Windows: stable"的两行，并取消注释"Windows: cuda132 opt-in"的两行：

   ```toml
   # Windows: stable (default).
   # "torch>=2.11.0,<2.12 ; sys_platform == 'win32'",
   # "torchvision>=0.26.0,<0.27 ; sys_platform == 'win32'",
   # Windows: cuda132 opt-in. ...
   "torch>=2.12.0.dev0,<2.13 ; sys_platform == 'win32'",
   "torchvision>=0.27.0.dev0,<0.28 ; sys_platform == 'win32'",
   ```

   **(b) flash-attn** — 在同一 `dependencies` 中注释掉"Windows: stable (default) — built against torch 2.11 + CUDA 13.0"行，并取消注释下方的"Windows: cuda132 opt-in — trimmed FA2"行：

   ```toml
   # Windows: stable (default) — built against torch 2.11 + CUDA 13.0.
   # "flash-attn @ https://github.com/mjun0812/.../flash_attn-2.8.3+cu130torch2.11-cp313-cp313-win_amd64.whl ; sys_platform == 'win32'",
   # Windows: cuda132 opt-in — trimmed FA2 ...
   "flash-attn @ https://github.com/sorryhyun/flash-attention-sm120-fix/releases/download/fa2cuda132/flash_attn-2.8.4-cp313-cp313-win_amd64.whl ; sys_platform == 'win32'",
   ```

3. **重新同步**：运行 `uv sync`。会从 cu132 索引下载 torch 2.12 nightly，从 release 下载精简版 FA2 轮子包并安装。

> 如需回退：将 `pyproject.toml` 中的注释恢复原样，然后重新运行 `uv sync`。
>
> 如果需要自行编译（例如不是 RTX 5060 Ti 的其他 GPU、Python 版本不同、或需要在自己的 fork 中管理轮子包）：请参考 [docs/optimizations/cuda132.md](../optimizations/cuda132.md)。

---

## 3. Python 环境与仓库准备

本项目使用 [`uv`](https://github.com/astral-sh/uv) 管理依赖。使用 Python 3.13。

### 3.1 安装 `uv`

  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  ```

安装完成后打开新的终端窗口，确认 `uv --version` 能正常输出。

### 3.2 克隆仓库

```bash
git clone https://github.com/sorryhyun/anima_lora.git
cd anima_lora
```

> 本指南中所有路径均以 `anima_lora/` 目录为基准。所有命令都在此文件夹内执行。

### 3.3 安装依赖

```bash
winget install ezwinports.make
uv sync
```

`uv sync` 会根据 `pyproject.toml`/`uv.lock` 创建虚拟环境并安装所有依赖。安装完成后通过以下任一方式激活环境：

- 每次会话手动激活（推荐）：`.venv\Scripts\activate`
- 使用 VSCode 可以更方便地自动激活环境。

---

## 4. Hugging Face 认证与模型下载

### 4.1 创建令牌并登录

1. 在 <https://huggingface.co/settings/tokens> 创建一个 **read** 权限的令牌。
2. 在终端中登录：

   ```bash
   hf auth login
   ```

   粘贴令牌并按回车。

### 4.2 下载模型

```bash
make download-models
```

此命令会自动下载以下三个模型到 `models/` 目录下并分类存放。

| 文件 | 路径 |
|---|---|
| Anima DiT（扩散模型主体） | `models/diffusion_models/anima-base-v1.0.safetensors` |
| Qwen3 0.6B 文本编码器 | `models/text_encoders/qwen_3_06b_base.safetensors` |
| QwenImage VAE | `models/vae/qwen_image_vae.safetensors` |

如果还需要使用遮罩功能，请一并下载 SAM3 和 MIT 模型（上述命令已包含）。

> **下载中断时**：可以使用 `make download-anima`、`make download-sam3`、`make download-mit` 等单独的命令分批重新下载。

---

## 5. 数据集准备

Anima LoRA 使用*图像 + 同名 `.txt` 标注旁文件*的结构。以下是 `image_dataset/` 文件夹的示例。

```
image_dataset/
├─ 00001.png
├─ 00001.txt
├─ 00002.jpg
├─ 00002.txt
├─ subfolder/
│  ├─ 00010.webp
│  └─ 00010.txt
└─ ...
```

### 5.1 标注编写建议

- 按照 anima 官方规范，标签顺序始终为 [meta] [character] [series] [artist] [general]。例如：

```
absurdres, safe, 1girl, chitanda eru, hyouka, @channel (caststation), full body, serafuku, She is saying hi.
```

- 经过个人实验，absurdres、highres、masterpiece 等质量标签建议不写或尽量少写。或者使用后续正式发布的 mod guidance 后可以完全省略。
- 原始图像放在 `image_dataset/` 中（命名不限，请使用此路径）。

### 5.2 `num_repeats` 是什么，什么时候需要调整？（总结：**不要动**）

在 `configs/base.toml` 的 `[[datasets.subsets]]` 中可以看到 `num_repeats = 1` 这一项。这是一个 kohya-ss 风格的选项，用于指定**每个 epoch 内每张图像重复使用的次数**，在跟随其他 LoRA 训练器指南时经常会遇到。

- **在本指南的标准工作流中请保持为 `1`。** 在将所有图像放在 `image_dataset/` 单一文件夹中训练的常见使用场景下，增大 `num_repeats` 只会*延长单个 epoch 的长度*，效果等同于增加 `max_train_epochs`。训练量始终通过 epoch 数来调节更直观，本项目的所有预设和方法配置均以 `num_repeats = 1` 为前提进行调优。
- **什么时候增大有意义？** 当一次训练中有*多个不同子集（文件夹）*且各文件夹图像数量差异很大时，`num_repeats` 可作为*平衡工具*使小文件夹的曝光频率匹配大文件夹（例如：角色 A 1000 张 + 角色 B 50 张，则仅对 B 子集设置 `num_repeats = 20`）。单一文件夹训练不适用。
- **在哪里修改？** `num_repeats` 属于*数据集配置*而非方法配置，因此不会暴露在 `configs/methods/`、`configs/gui-methods/` 或 GUI 训练标签页中。如果确实需要修改，请直接编辑 `configs/base.toml` 中的 `[[datasets.subsets]]`（或通过 `--dataset_config <path>` 指定的 TOML 文件）。*如果只是想让同一张图像训练更多次数*，正确做法是增大 `max_train_epochs`，而非修改 `num_repeats`。

---

## 6. 预处理：缩放 · 潜变量 · 文本嵌入缓存

为了优化训练速度和 VRAM 使用，需要预先执行**缩放 → VAE 潜变量缓存 → 文本嵌入缓存**三个步骤。

```bash
make preprocess              # 执行全部三个步骤（LoRA / 通用训练用）
# 或按步骤执行
make preprocess-resize       # 1) image_dataset/ → post_image_dataset/resized/
make preprocess-vae          # 2) VAE 潜变量缓存 → post_image_dataset/lora/
make preprocess-te           # 3) 文本编码器输出缓存 → post_image_dataset/lora/
make preprocess-pe           # （可选）PE-Core 视觉编码器特征缓存 — 仅用于 IP-Adapter / REPA
```

> **缓存会被重复利用——不会自动删除。**
> `make preprocess`（以及 GUI 中的 *Preprocess* 按钮）会**复用已有的缓存**。`post_image_dataset/lora/` 中的 `.npz` / `_te.safetensors` / `_pe.safetensors` 文件*不会被覆盖或删除*，仅处理缺失的项目。因此重新执行非常快速，中途断开也是安全的。
>
> 也就是说，已有缓存的状态下再次运行 `make preprocess` 不会丢失现有数据，请放心。反过来说，如果**修改了标注、分词器或缩放选项，想要*从头*重新生成缓存**，需要手动删除缓存文件夹（`post_image_dataset/lora/` 或 `post_image_dataset/easycontrol/`）后再重新执行。

### 6.1 缩放的作用

- 按照 VAE 要求的像素对齐方式缩放图像
- 自动归类到满足 (H/16) × (W/16) ≈ 4096 个 patch 的*固定 token 分辨率桶*
- 自动排除过小的图像（默认低于 0.5MP），并生成报告
- 结果以 PNG 格式保存到 `post_image_dataset/resized/`

### 6.2 潜变量缓存

- 对所有缩放后的图像只运行一次 VAE，将结果保存到磁盘
- 训练过程中不需要在 GPU 上加载 VAE，大幅节省 VRAM
- 缓存位置：`post_image_dataset/lora/{stem}_{WxH}_anima.npz`
- 脚本：`preprocess/cache_latents.py`

### 6.3 文本嵌入缓存

- 预先计算 Qwen3 0.6B + LLM 适配器的输出
- 当 `use_shuffled_caption_variants = true` 时，还会缓存标注逗号随机重排的变体（训练时随机选取）
- 缓存位置：`post_image_dataset/lora/{stem}_anima_te.safetensors`
- 标注始终从 `image_dataset/` 中的原始 `.txt` 读取（不会复制到缩放文件夹中）
- 脚本：`preprocess/cache_text_embeddings.py`

### 6.4 PE 视觉特征缓存（可选）

- 仅在启用 REPA 辅助损失（`use_repa = true`）时需要
- 预先计算 PE-Core-L14-336 视觉编码器的输出，以便训练时不需要加载视觉编码器
- 缓存位置：`post_image_dataset/lora/{stem}_anima_pe.safetensors`

> **什么时候需要重新生成缓存？**
> - 新增了图像 → 直接重新运行 `make preprocess` 即可（现有缓存保持不变，仅添加新项目）。
> - **修改了标注**或**更改了分词器/填充相关选项** → 请按照上述说明手动删除缓存文件夹（`post_image_dataset/lora/`）后重新执行。直接重新运行会*复用现有缓存*，不会反映更改。

---

## 7. GUI 使用方法

使用基于 WebUI 的浏览器界面，可以在一个界面中完成配置编辑、数据集浏览、预处理、训练执行/监控以及 LoRA 合并。

```bash
python -m webui              # 启动 WebUI 服务（http://127.0.0.1:8000）
```

GUI 主要标签页：

- **训练配置（Training Config）**：从下拉菜单选择 LoRA 系列变体（推荐：`tlora` — Ortho + T-LoRA / 其他如 `lora`、`tlora-8gb`、`tlora_ortho_reft`、`hydralora`、`reft` 等），并直接修改 `presets.toml` 预设（default / low_vram 等）和所有训练参数，然后开始训练
- **预处理（Preprocess）**：一键完成缩放 + VAE + 文本嵌入缓存
- **数据集（Dataset）**：预览图像/标注并直接编辑标注
- **合并（Merge）**：将训练好的 LoRA 烘焙到基础 DiT 中，保存为 ComfyUI 独立检查点（仅支持基础 LoRA / OrthoLoRA / T-LoRA）

GUI 训练在内部也是调用 `train.py`，因此相同参数完全可以在 CLI 中复现。GUI 读取的是 `configs/gui-methods/<variant>.toml`（无切换块的单文件变体），因此 GUI 中暴露的变体列表与 CLI 的 `make lora-gui GUI_PRESETS=<variant>` 产生的结果一致。变体列表的当前状态可通过 `ls configs/gui-methods/` 查看。

### 7.1 表单编辑与保存行为

训练/预处理子进程会重新读取磁盘上的变体 TOML 文件，因此如果只修改了表单而未保存，这些更改不会反映到训练中。GUI 通过以下两种方式处理：

- **变更检测**：当编辑了表单中的任何字段（或 `+ Extra args` 文本框）时，`Save` 按钮会变为橙色并显示 `Save *` 标记。这表示*磁盘上的变体文件与屏幕显示不一致*。点击 `Save` 或重新选择变体从磁盘重新加载即可消除标记。
- **自动保存**：即使忘记保存就点击了 `Train` / `Preprocess`，当前表单的值会先自动写入变体文件，然后再执行子进程。也就是说，屏幕上显示的值就是实际用于训练的值。（`Test` 是对最后一次训练结果的检查点进行推理，因此不触发自动保存。）

> 如果只想*试用*更改而不想保存，编辑表单后不要点击 `Train`，而是切换到其他变体再切回来——会从磁盘重新加载，编辑会被丢弃。

### 7.2 自动续训（checkpointing_epochs）

即使训练中途中断，再次点击 `Train` 就能**自动从最后保存的检查点继续训练**。这是应对断电、OOM、误关窗口等情况最有用的功能之一，默认已启用。

GUI 中的使用方法：

- 训练配置标签页的 **Training** 分组中有 `checkpointing_epochs` 字段（gui-methods 变体默认值 `2`，`methods/lora.toml` 默认值 `4`）。每 N 个 epoch 保存一次续训状态，覆盖写入同一文件，不会膨胀磁盘。
- 训练中断后用相同变体再次点击 `Train`，日志窗口中会显示 `auto-resuming from checkpoint at step N`，并从该点继续训练。无需手动修改任何标志。
- 训练正常结束后，续训临时文件会自动删除，最终结果保存为 `output/ckpt/<output_name>.safetensors`。
- **如果更改了数据集或核心配置（rank、LR、epoch 数等）**，想要重新训练，请手动删除 `output/ckpt/<output_name>-checkpoint-state/` 文件夹后再点击 `Train`。否则会在旧状态上继续训练。

详细行为请参见 [§8.6 自动续训](#86-自动续训checkpointing_epochs)——其中也说明了与 `save_every_n_epochs` 的区别。

---

## 8. 训练执行

所有训练均通过 TOML 配置文件和 HuggingFace Accelerate 执行。配置合并顺序为 `configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI 参数`，方法配置会覆盖预设配置。

### 8.1 快速开始

**最推荐的起步方案是 OrthoLoRA + T-LoRA（即 `tlora` 变体）**。这是稳定性、细节和风格保持之间平衡最好的组合，可直接用于常规角色/风格 LoRA 训练。

```bash
# 推荐：Ortho + T-LoRA (gui-methods/tlora.toml)
make lora-gui GUI_PRESETS=tlora                  # 一般环境
PRESET=low_vram make lora-gui GUI_PRESETS=tlora-8gb   # VRAM 8~12GB

# 其他变体 (configs/gui-methods/<variant>.toml — 无切换块的单文件)
make lora-gui GUI_PRESETS=lora                   # 仅基础 LoRA
make lora-gui GUI_PRESETS=tlora_ortho_reft       # Ortho + T-LoRA + ReFT 组合
make lora-gui GUI_PRESETS=hydralora              # MoE 多头路由
make lora-gui GUI_PRESETS=reft                   # 单独 ReFT

# 切换块方式 (在 configs/methods/lora.toml 中直接选择变体)
make lora                          # presets.toml[default]
PRESET=low_vram make lora          # presets.toml[low_vram] — VRAM 8~12GB
PRESET=half make lora              # 使用数据集的一半进行快速实验
```

> **在 CLI 中直接覆盖参数**：可以像 `make lora -- --network_dim 32 --max_train_epochs 24` 这样传递额外参数（`tasks.py` 也相同）。

### 8.2 遮罩损失（排除文字气泡）

在漫画/漫画风格数据中，将*文字气泡或文本区域*从训练损失中排除，可以让结果更加干净。

```bash
make mask          # SAM3 + MIT（在临时目录中运行）→ post_image_dataset/masks/
make mask-clean    # 删除 post_image_dataset/masks/
```

结果 PNG 为黑白图像：**白色（255）= 训练目标**，**黑色（0）= 排除区域**。数据集子集会自动优先使用 `post_image_dataset/masks/`（如果存在），否则依次回退到旧版 `masks/{merged,sam,mit}/`（使用旧布局的用户也能正常工作）。如果没有遮罩文件则直接忽略，不创建也无妨。

### 8.3 常用配置参数（LoRA 基准默认值）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `network_dim` | `32` | LoRA rank。越大表达能力越强，参数也越多 |
| `network_alpha` | `32` | LoRA 缩放系数（通常与 `network_dim` 相同） |
| `learning_rate` | `2e-5` | 学习率。Hydra 可以进一步降低 |
| `max_train_epochs` | `4` | 数据集越小，epoch 数应越大 |
| `save_every_n_epochs` | `2`（gui-methods）/ `4`（methods） | 适配器权重累积保存周期 |
| `checkpointing_epochs` | `2`（gui-methods）/ `4`（methods） | 续训状态保存周期（覆盖单文件） |
| `caption_dropout_rate` | `0.1` | 将部分标注替换为空字符串（有助于 CFG） |
| `use_shuffled_caption_variants` | `true` | 使用标注逗号重排变体 |

各变体的切换开关（`use_ortho`、`use_timestep_mask`、`add_reft`、`use_moe_style`、`router_source` 等）可通过取消 `configs/methods/lora.toml` 中的注释块来激活，或直接使用 `configs/gui-methods/<variant>.toml` 中的变体专用文件。**推荐起步方案 `tlora` 变体是已预设 `use_ortho = true` + `use_timestep_mask = true` 的 OrthoLoRA + T-LoRA 组合。**

### 8.4 训练过程中发生了什么

1. 加载文本编码器 → 创建/验证缓存 → 卸载
2. 加载 VAE → 创建/验证缓存 → 卸载
3. *延迟加载* DiT，避免缓存阶段的 VRAM 冲突
4. 将适配器网络注入 DiT 的 attention / FFN 模块（不同变体的注入目标不同）
5. 噪声采样 → DiT 前向传播 → flow-matching 损失 → 反向传播 → 优化器更新
6. （可选）通过 `validation_split` 测量验证损失并生成样本图像

### 8.5 输出物

- 训练后的权重：`output/ckpt/<output_name>.safetensors`（按变体自动分支为 `anima`、`anima_tlora_ortho`、`anima_tlora_reft`、`anima_hydra`、`anima_postfix` 等）
- 检查点：按 `save_every_n_epochs` 周期保存到 `output/ckpt/`（附带 `.snapshot.toml` 旁文件，Hydra 还有 `_moe` 伴随文件）
- 验证样本：`output/ckpt/sample/`
- 推理结果图像：`output/tests/`

### 8.6 自动续训（checkpointing_epochs）

即使训练中途中断，也可以**自动从最后保存的检查点继续训练**的功能。在断电、OOM、误关窗口、需要临时关机休息等情况下非常有用。默认方法文件中已启用，无需额外配置。

```toml
checkpointing_epochs = 2     # 每 2 个 epoch 保存一次续训状态（覆盖写入）
```

与 `save_every_n_epochs` 的作用不同。

| 配置键 | 保存内容 | 累积？ | 用途 |
|---|---|---|---|
| `save_every_n_epochs` | 适配器权重（如 `anima_lora-000004.safetensors`） | **累积**（或受 `save_last_n_epochs` 限制） | 用中间结果跑推理或比较过拟合时间点 |
| `checkpointing_epochs` | 训练续训完整状态（优化器 / 调度器 / RNG / 适配器权重） | **覆盖单文件** | 训练中断后自动续训 |

工作方式：

- **自动保存**：按 `checkpointing_epochs` 周期，将 `output/ckpt/<output_name>-checkpoint-state/`（状态目录）+ `<output_name>-checkpoint.safetensors`（权重）一起保存，覆盖上次内容。不会膨胀磁盘。
- **自动续训**：重新运行相同命令（如 `make lora`）时，如果存在已保存的检查点且尚未达到 `max_train_steps`，会**自动续训**。无需手动添加 `--resume` 等标志。日志中出现 `auto-resuming from checkpoint at step N` 即表示续训成功。
- **自动清理**：训练正常结束后，上述两个文件会被自动删除——续训状态是临时文件，最终产物是 `output/ckpt/<output_name>.safetensors`（参见 §8.5）。
- **手动续训**：如需回到其他时间点的状态，可以手动指定 `--resume <state_dir>`。

> **什么时候禁用？** 在短时间实验训练或磁盘空间不足时，注释掉 `checkpointing_epochs` 即可。对于数据集较大的正式训练，建议始终保持启用。
>
> **注意**：如果更改了数据集、标注或训练配置（rank、LR、epoch 数等），从旧检查点续训没有意义甚至可能有风险。这种情况下请手动删除 `output/ckpt/<output_name>-checkpoint-state/` 文件夹后重新开始。

---

## 9. LoRA / 适配器变体选择指南

> **推荐**：如果是初次使用或需要制作通用角色/风格 LoRA，请从 **`tlora`（OrthoLoRA + T-LoRA）** 开始。细节/风格保持和训练稳定性表现最佳。

| 变体 | 运行方式 | 适用场景 |
|---|---|---|
| **OrthoLoRA + T-LoRA** | `make lora-gui GUI_PRESETS=tlora` | **推荐**。基于 SVD 的正交旋转（OrthoLoRA）+ 按时间步的 rank 屏蔽（T-LoRA）组合。输出 `anima_tlora_ortho.safetensors` |
| **OrthoLoRA + T-LoRA（8GB）** | `make lora-gui GUI_PRESETS=tlora-8gb` 或 `PRESET=low_vram make lora-gui GUI_PRESETS=tlora` | 在 VRAM 8~12GB 环境下使用上述推荐组合 |
| **基础 LoRA** | `make lora-gui GUI_PRESETS=lora` 或 `make lora` | 最简基线，用于对比实验 |
| **基础 LoRA（8GB）** | `make lora-gui GUI_PRESETS=lora-8gb` 或 `PRESET=low_vram make lora` | VRAM 8~12GB |
| **T-LoRA + Ortho + ReFT** | `make lora-gui GUI_PRESETS=tlora_ortho_reft` | 在推荐组合基础上增加表达编辑（ReFT），用更少的额外参数进行微调 |
| **HydraLoRA** | `make lora-gui GUI_PRESETS=hydralora`（8GB 版本为 `hydralora-8gb`） | MoE 多头路由，将多种概念整合到一个适配器中 |
| **单独 ReFT** | `make lora-gui GUI_PRESETS=reft` 或在 `methods/lora.toml` 中设置 `add_reft = true` | 表达编辑（Representation Fine-Tuning），极少参数量 |
| **Postfix Tuning**（*实验性*） | `make exp-postfix` 或 `make lora-gui GUI_PRESETS=postfix_ortho_cond` | 在 cross-attention 末尾追加可训练的 N 个向量（标注条件 + 正交变体） |
| **ChimeraHydra**（*实验性*） | `make exp-chimera` 或 `make lora-gui GUI_PRESETS=chimera_hydra` | 内容/频率双池 MoE — 仅用于研究 |

各变体的详细选项请参阅 [`docs/guidelines/training.md`](training.md) 和 `docs/methods/` 中的各单独文档。

> **兼容性说明**
> - HydraLoRA / ReFT 等适配器变体需要默认启用 `cache_llm_adapter_outputs = true` 才能正常运行。
> - `tlora` 和 `tlora_ortho_reft` 中的 OrthoLoRA + T-LoRA 部分可以通过 `make merge` 烘焙到基础 DiT 中，制作独立的 ComfyUI 检查点（ReFT 部分无法烘焙——需要 `--allow-partial`）。

---

## 10. 推理

### 10.1 最快测试方式

如果想立即使用刚训练的适配器生成样本，请使用对应变体的 `make test-*`。所有命令都会自动从 `output/ckpt/` 中选取最近保存的相应适配器进行推理。

```bash
make test                        # 普通 LoRA / OrthoLoRA / T-LoRA / ReFT
make test SPECTRUM=1             # Spectrum 加速推理
make test MOD=1                  # Modulation guidance (pooled_text_proj) — 可与 SPECTRUM=1 组合
make test NOLORA=1               # 单独基础 DiT 推理（省略 --lora_weight）；与 MOD=1 组合即为 mod-only 路径
make test-hydra                  # HydraLoRA（路由器在线，anima_hydra*_moe.safetensors）
make test-merge                  # 用烘焙后的独立 DiT（`*_merged.safetensors`）推理
make test-dcw                    # LoRA + DCW 标量校正（采样器级 SNR-t 校正）
make test-dcw-v4                 # LoRA + DCW v4 学习型校正器
# 实验性推理
make exp-test-postfix            # Postfix tuning（标准版）
make exp-test-postfix-exp        # postfix_exp 变体
make exp-test-postfix-func       # postfix_func 变体
```

### 10.2 通用推理（手动）

```bash
python inference.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --lora_weight output/ckpt/anima_lora.safetensors \
    --lora_multiplier 1.0 \
    --prompt "masterpiece, best quality, an anime girl in a sunlit forest" \
    --negative_prompt "worst quality, low quality, blurry" \
    --image_size 1024 1024 \
    --infer_steps 30 \
    --guidance_scale 4.0 \
    --sampler er_sde \
    --flow_shift 1.0 \
    --seed 42 \
    --save_path output/tests
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--lora_weight` | 训练好的适配器路径。可以同时指定多个 |
| `--lora_multiplier` | 适配器强度（0.0~1.5） |
| `--image_size H W` | 输出分辨率（例如：`1024 1024`、`1024 1536`） |
| `--infer_steps` | 去噪步数（通常 20~50） |
| `--guidance_scale` | CFG 强度（推荐 3.0~5.0） |
| `--sampler` | `er_sde`、`euler`、`dpm++` 等 |
| `--seed` | 用于复现的随机种子 |
| `--spectrum` | 启用 Spectrum 加速 |
| `--pgraft` | P-GRAFT（去噪后期 LoRA 截断），让基础模型负责后期细节 |

推理选项的完整列表和 P-GRAFT 推理方法请参阅 [`docs/guidelines/inference.md`](inference.md)。

---

## 11. 部署到 ComfyUI

ComfyUI 核心已原生支持 Anima 基础 DiT（通过 `UNETLoader` / `CLIPLoader` 直接加载）。根据适配器种类不同，部署方式也不同。

### 11.1 经典 LoRA / OrthoLoRA / T-LoRA

将 `output/ckpt/` 中生成的 `.safetensors` 文件直接复制到 `ComfyUI/models/loras/`，即可通过 ComfyUI 默认的 LoraLoader 节点使用。如果想制作更简洁的独立检查点：

```bash
make merge ADAPTER_DIR=output/ckpt                 # 将最新权重烘焙到基础 DiT
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8  # 调整强度
```

烘焙后的 `*_merged.safetensors` 可直接作为独立模型在 ComfyUI 的 `UNETLoader` 中加载。

### 11.2 HydraLoRA / ReFT / Postfix

这些变体无法通过 ComfyUI 默认的 LoraLoader 加载（因为不仅仅是权重增量，还涉及路由和 token 插入），需要专用节点：

- **Anima Adapter Loader**（`custom_nodes/comfyui-hydralora/`）——统一处理 LoRA / Hydra / ReFT / postfix。详细用法请参阅该文件夹中的 `README.md`。
- **Spectrum KSampler / Mod Guidance / DCW 节点**——独立仓库 <https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler>

---

## 12. 更新

```bash
make update              # 从 GitHub release 下载最新版本并应用 + 自动运行 uv sync
make update -- --dry-run # 预览哪些文件会被更改
```

`update` 不会触及 `image_dataset/`、`post_image_dataset/`、`output/`、`models/` 目录，对于用户修改过的配置文件会在冲突时询问确认。

---

## 更多资料

- [`docs/guidelines/training.md`](training.md) — 适配器变体、标注随机重排、遮罩损失、数据集配置详情
- [`docs/guidelines/inference.md`](inference.md) — 推理参数、DCW、Spectrum、提示词文件格式
- [`docs/guidelines/difference_between_comfy.md`](difference_between_comfy.md) — anima_lora 与 ComfyUI 核心实现差异
- [`docs/methods/timestep_mask.md`](../methods/timestep_mask.md) — T-LoRA 时间步遮罩
- [`docs/methods/psoft-integrated-ortholora.md`](../methods/psoft-integrated-ortholora.md) — OrthoLoRA 细节（推荐 `tlora` 变体的正交旋转部分）
- [`docs/methods/spectrum.md`](../methods/spectrum.md) — Spectrum 加速的原理与选项
- [`docs/methods/dcw.md`](../methods/dcw.md) — DCW（标量 + v4 学习型校正器）
- [`docs/methods/mod-guidance.md`](../methods/mod-guidance.md) — Modulation guidance
- [`docs/methods/hydra-lora.md`](../methods/hydra-lora.md) — HydraLoRA 多头路由
- [`docs/methods/reft.md`](../methods/reft.md) — ReFT 表达编辑
- [`docs/experimental/postfix.md`](../experimental/postfix.md) — Postfix（cond+ortho）
- [`docs/optimizations/cuda132.md`](../optimizations/cuda132.md) — 如何升级到 CUDA 13.2
- [`docs/optimizations/full_model_cudagraph.md`](../optimizations/full_model_cudagraph.md) — `compile_mode=full` + CUDAGraph 不变量与调试

如有问题或 Bug 报告，欢迎在 GitHub Issues 中用中文提交。祝训练愉快！
