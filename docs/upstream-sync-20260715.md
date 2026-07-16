# 上游同步评估（2026-07-15）

## 结论

不要直接执行 `git merge upstream/main`，也不要再次 revert `70bfdc1f`。

原因有两个：

1. `71745620` 已经把 `cece8e10` 记入本 fork 的祖先历史，后来的 `70bfdc1f` 只回退了文件树，没有抹掉祖先关系。普通 merge 只会考虑 `cece8e10` 之后的 100 个提交，不会自动恢复此前被撤销的 128 个路径。
2. fork 此后已经重写 WebUI、daemon、配置和训练链路。当前全量合并模拟产生 116 条冲突记录，涉及约 113 个唯一路径，不是一次可审阅的同步。

正确策略是按功能簇选择性移植：先合并 3 个无冲突的独立修复，再手工适配 caption-index、LoRA 合并分析和 MIT CTD 加速；其余新功能按实际需求单独立项。

## 执行结果（2026-07-16）

已在 `codex/upstream-review-20260715` 按推荐方案完成第一、二批，共形成 8 个可独立审阅的提交：

| 批次 | 功能 | 上游来源 | fork 提交 | 结果 |
| --- | --- | --- | --- | --- |
| 第一批 | cosine restart 周期数 | `d5a1975f` | `95d5d693` + `8a25242a` | 直接移植，并补参数到 scheduler 的定向回归 |
| 第一批 | ComfyUI Tagger namespace 清理 | `7a192c52` | `7b9ec468` + `42df69ff` | 直接移植，并用真实子进程覆盖多 vendor import |
| 第一批 | free-fit VRAM/partitioner 开关 | `5289d461` | `47cc83c2` | 直接移植；Windows allocator 分支确认 no-op |
| 第二批 | subdir-aware caption index | `9e2a4457` | `d3ca9cbb` | 只提取 `subdir/stem` schema，未带入 memorization 的 `image_keys` 行为 |
| 第二批 | 符号感知 LoRA 子空间分析 | `9285a812` + `4359a7ca` | `0c260b72` | backend 与 Vue WebUI 同步适配；兼容旧四字段 trailer |
| 第二批 | MIT CTD ONNX Runtime CUDA | `ebe76820` + `344981a4` | `d7584865` | 抽取 CTD gate 前置与 CUDA 加速，保留 OpenCV CPU fallback |

MIT 项在执行时补正了一处依赖边界：`344981a4` 只加速已经存在的 CTD gate，而本 fork 回退后并没有 gate 本体；本体位于捆绑了 SR 改动的 `ebe76820`。本次只抽取 MIT 脚本与 masking task 所需部分，没有引入 SR 流水线。`uv.lock` 也没有采用上游版本，而是从本 fork 的 `pyproject.toml` 重新解析，仅新增 `onnxruntime-gpu 1.27.0` 与 `flatbuffers`。

第三批仍全部保持未合并：Turbo、xattn、Tagger 模型升级、daemon 能力，以及 Soup、Register、SR/RSD、EasyControl colorize、Atlas 等可选路线均不在本分支范围内。上游对 DCW/SPD/PID 的删除、Qt GUI/daemon 迁移和研究归档整理也继续排除。

## 调研锚点

| 项目 | 对象 |
| --- | --- |
| fork 基线 | `main@90706fce`（2026-07-15） |
| 调研分支 | `codex/upstream-review-20260715` |
| 上次合并 | `71745620`，父提交为 `13b39082` + `cece8e10` |
| 整包回退 | `70bfdc1f`（2026-07-02） |
| 当前 merge-base | `cece8e108d73072af48c0407413b0bfb76cfc49c` |
| 最新上游 | `upstream/main@4b1b30f7`（2026-07-13） |
| 最新上游版本位置 | `v1.13.2-22-g4b1b30f7` |
| 分叉提交数 | fork 侧 189，上游侧 100 |

本次先用 `git fetch upstream +refs/heads/main:refs/remotes/upstream/main` 刷新了上游主线。带 `--tags` 的抓取因 fork 与上游的旧同名标签指向不同对象而被 Git 拒绝；没有强制覆盖 fork 标签。

## 上次回退了什么

### 时间线

1. `71745620` 合并上游，解决了 `CLAUDE.md`、LoRA 配置、compile 文档、旧 GUI merge tab 和 network registry 等 6 处冲突。
2. 合并后增加了 4 个 fork 提交：
   - `9663a136`：修正 REPA 默认值测试。
   - `32a9d4d6`：给 Vue WebUI 增加 LoRA fusion。
   - `5f26ef12`：WebUI regularization subset 与校验。
   - `d75247d2`：validation best-effort，并因 checkpoint 不可用禁用 REPA。
3. `756d0106` 到 `9fe766d3` 逐个撤销上述 4 个提交。
4. `70bfdc1f` 再撤销 merge 本身。

`13b39082` 与 `70bfdc1f` 的树完全一致，证明这次是完整回滚，不是部分回滚。

### 回退规模

`71745620^1..71745620` 共改动 128 个路径：

| 类型 | 数量 |
| --- | ---: |
| 新增 | 55 |
| 修改 | 66 |
| 删除 | 5 |
| 重命名 | 2 |
| 行数 | `+12,900 / -1,771` |

主要内容包括：

- LoRA N-to-1 合并、干扰分析和测试。
- FSG、CNS、SMC-CFG、mod-guidance、Spectrum/SEA、SPD 等推理实验与拆分。
- Turbo soft-rank、配置与指标。
- artist shard、validation、preprocess、CLI/runtime 调整。
- 大量 bench、proposal、findings 和已归档研究材料。
- 删除 PID 和 Qwen VAE 2D 的旧 probe。

### 当前状态

128 个路径与当时三个树版本逐 blob 比较后的结果：

| 状态 | 数量 | 说明 |
| --- | ---: | --- |
| 已按当时版本恢复 | 3 | `library/anima/merge_analysis.py`、`scripts/merge_loras.py`、`tests/test_merge_interference.py` |
| 仍保持回退 | 110 | 包括 `_archive` 10、bench 25、docs 15、library 23、networks 8、scripts 12、tests 9 等 |
| fork 已独立改写 | 15 | 不能用 revert-revert 覆盖 |

15 个已独立改写路径为：

```text
.gitignore
configs/base.toml
library/anima/training.py
library/config/cli_args.py
library/datasets/base.py
library/datasets/dreambooth.py
library/inference/args.py
library/runtime/harness.py
networks/CLAUDE.md
networks/__init__.py
networks/lora_anima/config.py
networks/lora_anima/factory.py
scripts/tasks/preprocess.py
tests/test_network_registry.py
train.py
```

回退意图有直接证据：`862834d9` 的提交说明明确写道，恢复整个 revert 会带回约 128 个无关噪音文件，因此只恢复 LoRA merge 的 3 个相关文件。它的前一个提交 `92b28bb2` 又单独恢复了 Vue WebUI 的 fusion 界面。当前状态符合“选择性恢复”，不是遗漏。

## 上游最近更新概览

范围为 `cece8e10..4b1b30f7`：100 个提交，净改动 505 个文件，约 `+47,437 / -27,385`。其中包含大量先加入后删除、研究归档和独立产品线代码，提交数不能等同于可合并价值。

### 可以直接 cherry-pick

| 优先级 | 提交 | 内容 | 合并模拟 | 判断 |
| --- | --- | --- | --- | --- |
| P0 | `d5a1975f` | 恢复 `lr_scheduler_num_cycles`，修复 `cosine_with_restarts` 静默退化为普通 cosine | 0 冲突 | 正确性修复，范围仅 CLI + scheduler，应先合并 |
| P0 | `7a192c52` | ComfyUI 多 Anima node 同时加载时清理被兄弟 vendor 污染的 `library`/`networks` namespace | 0 冲突 | 当前 fork 仍是修复前实现，直接有用 |
| P1 | `5289d461` | free-fit 显存碎片默认值与 AOT partitioner 重计算开关 | 0 冲突 | Linux allocator 默认有用；Windows 上 allocator 部分无效，但 opt-in partitioner 开关仍可用 |

建议三者分别 cherry-pick，分别跑测试，不要 squash 后才定位问题。

### 有价值，但必须按 fork 手工适配

| 优先级 | 提交/提交簇 | 价值 | 为什么不能整提交拿入 |
| --- | --- | --- | --- |
| P0 | `9e2a4457` | caption index 改为 `subdir/stem`，修复跨目录同名 stem 冲突 | 1 个 `library/datasets/base.py` 冲突；提交还夹带 memorization 的 `image_keys` 行为。应只移植 `caption_key`，并同时覆盖 identity-pair 与 contrastive 两个消费者 |
| P1 | `9285a812` + `4359a7ca` | LoRA 干扰分析增加子空间 overlap，并修正“高 overlap 但同向增强”被误报为 collision | fork 正在使用旧版分析器和 Vue banner。Python backend 可移植，但必须同步扩展 `MergeView.vue` 的 `ANALYZE_RESULT` 解析和多语言文案 |
| P1 | `344981a4` | MIT comic text detector 改用 ONNX Runtime CUDA，CPU 保留 fallback | 2 个冲突（脚本、`uv.lock`）；需按 fork 依赖组处理，不能覆盖 lockfile |
| P1 | `48d8ec3a` + `4b1b30f7` 中 Turbo 部分 | 从已有 adapter warm-start Turbo，增加可恢复训练状态 | 配置/训练 loop 已分叉；`4b1b30f7` 同时捆绑 Atlas、proposal 和大量文档，应抽取 `resume.py`、状态保存与对应测试 |
| P2 | `8cd50b40` + `67b2726c` | xattn boost 与 norm-matched renorm，属于上游已有 bench 证据的推理质量开关 | 分别有 6/11 条冲突；依赖模型 block、generation、Spectrum/SPD side-channel，且上游同时引用 fork 已删除的 foveated 路线 |
| P2 | `b81a6e36`、`ce3c083e`、`b1f37906`、`49f6a8ad`、`8435cbda` | Tagger label embedding、sentinel/maxsup、spatial-head v3-refit | 是完整 Tagger 模型/权重升级，不是小 bugfix；需确认新 checkpoint 获取与 vendor-sync 后整体移植 |
| P2 | `7fb6dbb8`、`884c744e`、`7df1a852`、`05940198` | daemon 指纹/环境捕获、结果 envelope、暂停恢复、状态过滤 | 概念有用，但 fork daemon 已加入 WebUI sidecar、任务恢复与 Windows runtime compat。只移植能力，不迁包 |

### 可选新能力，不应混入同步修复

| 提交簇 | 内容 | 建议 |
| --- | --- | --- |
| `91b49a55`、`b6d74da2`、`05d1c5d7`、`705ec999` | LoRA soup 训练/构建流水线 | 有明确需求再立项；保留 backend，重新做 Vue 页面，不移植 Qt tab |
| `2131cdce`、`a0994dd0`、`0cd59cbf` | register-token 训练方法与 ComfyUI node | 作为实验方法单独评审，不能与基础同步绑定 |
| `527f945f` 起的一组 SR/RSD 提交 | ResShift/RSD 超分支线，含大量 vendored 源码 | 与 LoRA 训练主线耦合弱且体积巨大，默认不合并；需要超分产品线时独立引入 |
| `68b1d2cb` | EasyControl colorize 白平衡 target cache | 仅在继续维护 colorize adapter 时移植 |
| `4b1b30f7` 中 Atlas 部分 | artist atlas pack | 独立工具，可从 Turbo resume 中拆开评估 |

### 没有用或会破坏 fork 的更新

以下内容不应合并：

- `dac7b048` 删除 DCW、`2cf6f95b` 删除 SPD、`517e78dc` 删除 PID calibration。fork 的文档和任务面仍把 DCW/SPD 作为已发布能力，合入会直接回退功能。
- `0235f17e` + `c098384d` 将 `scripts/daemon/` 迁到 `anima_daemon/` 并删除兼容 shim。fork 的 daemon、WebUI sidecar、自愈和 Windows 进程环境都围绕现目录继续演进，迁包收益不足以覆盖冲突成本。
- `6bf12220` 的 Qt GUI preset 收敛，以及 `79a3f57b`、`8c70e891`、`14a9b861` 等 Qt GUI 重构。fork 已切到 FastAPI + Vue，代码路径不同。
- SCFM 的加入/删除序列（`12f372bc` 到 `7e2e93f4`）和 foveated 的加入/删除序列（`28dd9ca7` 到 `b0e67fe2`）。上游最终净结果已删除，不需要重演中间提交。
- `ec573acd`、`cdf67c84`、`63250da4`、`438c3b09`、`6a9bdd3b`、`220b133c`、`a29453ba`、`dc5982af`、`43e43195` 等 bench/archive 整理。它们是上游研究历史管理，不改善 fork 运行时。
- `336b92c7`、`b160a979`、`d14532de` 等 proposal 归档/删除。除非对应功能也完整移植，否则只会制造文档噪音。

## 为什么不能全量 merge

对 `main` 与 `upstream/main` 执行不改工作区的 `git merge-tree` 模拟：

| 冲突类型 | 数量 |
| --- | ---: |
| content | 45 |
| modify/delete | 33 |
| rename/delete | 22 |
| file location | 16 |
| 合计 | 116 |

冲突主要来自三条架构分叉：

1. 上游继续维护 `gui/` Qt 应用，fork 已迁移为 `webui/` FastAPI + Vue。
2. 上游把 daemon 迁到顶层 `anima_daemon/`，fork 继续在 `scripts/daemon/` 上加入 sidecar 和恢复机制。
3. 上游删除 SPD/DCW/foveated 等路线，fork 保留 SPD/DCW，并对推理/训练接口做了独立扩展。

端点差异已达到 869 个文件、约 `+77,199 / -74,659`。即使人工解决文本冲突，也无法证明语义正确。

## 推荐执行顺序

### 第一批：独立修复

1. cherry-pick `d5a1975f`，增加 scheduler 定向测试。
2. cherry-pick `7a192c52`，增加多 vendor namespace 的 import 回归测试或至少运行 ComfyUI node import smoke。
3. cherry-pick `5289d461`，运行 `tests/test_partitioner_tuning.py`，并在 Windows 明确验证 allocator 分支为 no-op。

### 第二批：核心手工适配

1. 从 `9e2a4457` 提取 `caption_key` 与 index schema 迁移；更新 identity-pair、contrastive consumer 和重建提示。
2. 移植 `9285a812` 后再移植 `4359a7ca` 的 backend；更新 Vue banner 的 overlap/collision/reinforcing/cancelling 展示。
3. 按当前依赖声明手工移植 `344981a4`，不要 cherry-pick 上游 `uv.lock`。

### 第三批：按使用场景选择

- 使用 Turbo：提取 warm-start + resume，不带 Atlas 和上游配置覆盖。
- 追求推理 prompt binding：单独移植完整 xattn boost 两阶段并跑图像 bench。
- 需要新训练方法：Soup、Register、SR/RSD 各自独立分支和验收标准。
- 需要 daemon 能力：在现有 `scripts/daemon/` 上重新实现指纹、环境捕获、结果 envelope 和过滤，不做目录迁移。

## 调研阶段验证（合并前）

- `git diff 13b39082 70bfdc1f` 无输出：整包回退后树与合并前完全一致。
- 逐 blob 比较：3 个路径恢复、110 个仍回退、15 个已分叉。
- `git merge-tree` 全量模拟：116 条冲突记录。
- 逐提交合并模拟：`d5a1975f`、`5289d461`、`7a192c52` 均为 0 冲突。
- 当前 fork 基线测试：

```text
51 passed, 14 warnings in 7.79s
```

测试命令：

```powershell
& '.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider `
  tests/test_merge_interference.py `
  tests/test_caption_index.py `
  tests/test_network_registry.py
```

## 合并后验证（2026-07-16）

- 最终组合定向测试：`79 passed, 3 skipped`。覆盖 scheduler、Tagger vendor namespace、partitioner、caption index/identity pairs、LoRA merge analysis、MIT CTD 与 masking task。
- 分功能扩展验证：scheduler `21 passed`；Tagger `16 passed`；partitioner/harness `12 passed, 3 skipped`；caption-index 全链路 `97 passed`；LoRA merge analysis `13 passed`；MIT/masking + preprocess `16 passed`。
- Windows allocator 实测返回 `False` 且未写 allocator 环境变量。
- MIT 路径验证显式 CPU 不导入 ONNX Runtime；CUDA provider 不可用时回退 OpenCV DNN；CLI `--ctd-gate` / `--no-ctd-gate` smoke 通过。
- Python Ruff（本次触及文件）通过；Vue `vue-tsc -b` 通过；Vite production build 通过。
- `uv lock` 成功解析 214 个包，只新增 `onnxruntime-gpu` 与 `flatbuffers`。

## 可复核命令

```powershell
git merge-base --all main upstream/main
git rev-list --left-right --count main...upstream/main
git diff --stat 71745620^1 71745620
git diff --name-status 13b39082 70bfdc1f
git diff --stat 71745620^2..upstream/main
git merge-tree --write-tree --name-only --messages main upstream/main
```
