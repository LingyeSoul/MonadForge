# 2026-10-18 上游同步交付记录

本次按 [批准计划](upstream-sync-20261018-plan.md) 完成真实上游合并、MonadForge 功能保留、三模型工作区及 Vue 原生整理接入。详细逐项核对见 [同步审计](upstream-sync-20261018-audit.md)。交付只在本地集成分支，不推送、不创建 PR、不合并回 main。

## 固定来源与保全

- 分支：`codex/sync-anima-upstream-20261018`。
- 上游：`sorryhyun/anima_lora`，`25eaf52718e2f8f3b3549a6cdad62353da5d8c78`。
- 同步前本地：`396f79b33bca283f45a05d67142656116d41e6ae`；原有未提交修复先保存为 `e981e6f0c852eafe2244adedebcaa9b806832b7a`。
- anime-tools：`0.7.15`，`7ea30612ebec736bbc3a243d153dca4d0d2e3853`；普通环境依赖由 `uv.lock` 固定，V100 纯工具副本保留来源清单和许可证。
- 已核对上游 1,952 条路径，全部存在。回退历史和本地旧入口另做反向核对，恢复遗漏的 DCW、mod 蒸馏、SPD、Chimera、tagger、DirectEdit 入口；vendor 使用同步脚本重建。
- 模型、源素材、缓存、训练输出和历史任务没有作为升级清理目标。`.snow/` 保留在工作区和外部备份，不进入提交。

外部备份位于 `/home/buxinzi/Projects/AI/MonadForge-sync-backups/`：

| 目录 | 内容 |
| --- | --- |
| `20261018-035816/` | 开始同步时的 HEAD、refs、index、差异、未跟踪内容、基线与初始冲突清单 |
| `20261018-continuation/` | 接续时合并中工作树/index 快照、所有验证日志、白名单、浏览器脚本与截图 |
| `20261018-continuation/venv-before.tar.zst` | 更新前完整 `.venv`，已通过 `zstd -t`；包版本清单为 `venv-before.json`，校验和为 `venv-before.sha256` |

## 功能及迁移

`anima_daemon` 是唯一实现，`scripts.daemon` 兼容旧导入。历史任务启动清理默认关闭；手动清理入口保留。`/unpause` 恢复暂停，`/continue` 创建合法断点 attempt，旧 `/resume` 按状态分派。保存中、排队和已完成任务不会被当作断点恢复；paused attempt 仍计入活跃 lineage。Job 双读 `rc/returncode`，保留未知字段和恢复信息；同 job 重启以持久化流偏移隔离终态，旧 paused 事件不能掩盖新进程启动失败，历史 JSONL 保持追加。

模型上下文为 `anima | qwen21 | moss`。配置、预处理、任务、看板和结果入口共用现有 Vue 工作区；Qwen 的目录/分片/Comfy 单文件、空值/零和量化拒绝边界保留。MOSS 使用自己的 schema、profile、缓存约定、preflight、task manifest 与图片入口。两模型输出标准 JSONL 指标，但不获得 Anima 专用 continuation 协议。MOSS 自动调参入口属于训练命令，本次没有执行。

数据集页新增原生整理区域，覆盖 resize、autotag、position、correct、drop_groups、audit、OCR、groups、SAM masks、mask merge、export。包含版本梯级/差异、标签及官方语法子句编辑、OCR 框、遮罩、分组、排除恢复、设置及历史报告。普通保存写 workspace overlay。每项作业固定 roots、参数、scope、步骤和报告；前置/主阶段/后续处理在一个 daemon job 中执行，首错/取消停止后续步骤。历史应用/撤销始终使用该任务快照。

导出必须经过预览与审阅。源 master 标注回写、输出覆盖和过期 sidecar 清理明确显示；应用前检查输入与目标漂移，发布前保存备份和恢复 journal，失败或取消后的已发布内容仍可撤销。

本地适配器、量化、V100 attention 分派、分阶段分辨率、预处理 run、下载镜像、采样和续训调用链继续保留。上游汤化训练、Spectrum SEA、FSG/CFG++、vocab-pack cache stamp、Turbo 拆分流程与三阶段 validation 已补齐；未以清除冲突标记替代调用链复核。

## 运行环境

本机普通 `.venv` 已更新到 Torch `2.14.1+cu132`、TorchVision `0.29.1+cu132`、Transformers `5.16.1`、anime-tools `0.7.15`、FlashAttention `2.8.3+cu132torch2.14`。先在临时环境完成安装及 106 项定向验证，再更新项目环境。

更新项目环境第一次因磁盘满中断，造成部分包仅剩元数据；普通补装未完全修复，随后使用固定锁文件 `uv sync --frozen --reinstall` 重装，实际导入与项目环境回归已验证。原始失败日志保留，不能把前两次操作记录视为成功。

登记文件为 `configs/custom/runtime.toml`，路径相对仓库解释：

```toml
[runtimes.default]
python = ".venv/bin/python"

# 需要分离整理环境时登记；未登记时 curation 继承 default。
[runtimes.curation]
python = "/absolute/path/to/curation/bin/python"

# 只指向已经安装好的受保护 V100 环境。
[runtimes.v100]
python = "/absolute/path/to/v100/bin/python"
```

上例独立路径必须替换为真实解释器。浏览器从 `/api/system/runtimes` 选择可用名称，不能提交解释器路径或环境变量覆盖；缺失登记/解释器、恢复时登记变化均明确拒绝，不静默换环境。V100 保持 Torch 2.10/CUDA 12.9；更新器跳过普通 `uv sync`，PE 保留独立本地实现。整理后端的固定 JSON RPC 可使用单独解释器，不要求 V100 安装完整 anime-tools。

## 实际验证

最终 CPU 非训练白名单保存在外部 `final-whitelist.json`，明确列出 65 个文件或 node ID；不运行未经审查的全套 pytest，不执行 backward、optimizer step、真实缓存或模型生成。所有模型相关测试及其子进程受 `MemoryMax=4G`、`MemorySwapMax=0` 限制。

| 检查 | 结果及证据 |
| --- | --- |
| 项目环境统一白名单 | `final-nontraining.txt`：443 passed，1 failed；失败为上游断言 independent-A 已退役，与本次保留策略冲突。已改为验证合法 FEI 路由和非法输入拒绝，`final-recheck.txt`：1 passed。合计 444 项检查得到通过结果；没有重复全套回归 |
| 聚合资源边界 | 主白名单进程树峰值 2,296,406,016 字节；swap 上限 0；OOM/oom_kill 均为 0 |
| 前端测试 | `frontend-final-tests.txt`：52 passed、1 skipped、0 failed。跳过项是要求 `CHROMIUM_BIN` 的原 toolbar 浏览器测试；对应页面已通过本次实际浏览器 QA |
| 生产构建 | `ui-qa/final-build.log`：通过；保留大于 500 kB chunk 的构建体积提示 |
| 实际浏览器 | 系统 Chromium + 已有 Playwright；1440×1000 与 390×844；`ui-qa/qa-final.json` 的 errors/failed/blocked 均为空 |
| 更新后的实际 HTTP | `http-updated-environment.json`：项目 `.venv` 启动成功，首页、runtime 目录、Qwen/MOSS schema 和 11 阶段整理 schema 均为 200；临时服务验证后已停止 |
| 整理真实 API | 临时素材的 overlay 保存、引用逗号子句、版本、11 阶段、CPU resize、排除恢复、合成 OCR/遮罩显示通过；外层标签切换保留未保存草稿 |
| 导出文件闭环 | `ui-qa/export-roundtrip.json`：预览→应用→撤销完成，实际临时源标注改变并精确恢复，重复 Undo 入口消失 |
| 模型结果归属 | `ui-qa/simulated-results.json`：明确模拟的 Qwen/MOSS job/manifest/image 响应验证隔离，没有实际模型生成 |
| 静态与差异 | 活跃 Python 树 Ruff 未定义名/重复定义/重复字典键检查、本轮手工增量差异检查通过；注册 CLI 与文件树双向核对完成。完整合并的空白告警来自保留的上游/本地历史文本和原编码许可证，未做无关格式重写 |

## 已知边界

- `uv pip check` **未全部通过**：SAM3 元数据要求 `numpy>=1.26,<2`，项目与 anime-tools 要求 NumPy ≥2。本次保留上游 `override-dependencies`，实际安装 NumPy 2.4.4，仍有 1 项声明冲突；还会提示 Torch 特定版本没有对应的额外 dependency-metadata 条目。没有用无依赖安装或改写声明宣称兼容。
- 未执行 Windows、ROCm、V100 实机安装/运行，以及真实 Anima/Qwen/MOSS/Turbo 模型数值、GPU性能、完整数据缓存、打标/OCR/SAM推理或自动调参。真实训练验收由用户负责。
- 原本地 BYG 推理入口本就明确报告未接线，此边界保留。FECL 的 loss handler/注册恢复且小型数学检查通过，但完整训练中的辅助特征生产链没有在本次运行验证。
- 原会话曾误执行包含 CPU backward 的适配器测试，已在当时披露；该结果不计入本报告验收。本次接续的所有模型检查遵守非训练白名单。

## 恢复方式

代码的原始工作区快照和检查点保留。需要撤销同步时先保全新的工作，再切回本地检查点 `e981e6f0` 或原 main；不要对带未保存工作区直接执行 reset/clean。

恢复旧环境时先停止使用项目环境的进程，将当前 `.venv` 移到备份位置，再在仓库根目录解包 `venv-before.tar.zst`。备份含 `.venv/` 根和原始符号链接；恢复后核对 `venv-before.json`。本报告不自动执行回滚，也不删除新环境、用户数据或任何备份。
