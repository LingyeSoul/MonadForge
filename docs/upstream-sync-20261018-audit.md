# 2026-10-18 全面上游同步审计

本文件对照已批准的 [同步计划](upstream-sync-20261018-plan.md)，记录工作树审查事实、已修复的断链、已保存的验收证据和剩余边界。2026-10-18 收尾复核已读取仓库外 `MonadForge-sync-backups/20261018-continuation/` 的日志和浏览器报告；下文明确区分源码审查、实际 API/临时文件验证与模拟结果。项目环境重装与最终白名单结果已写入 [交付记录](upstream-sync-20261018.md)，其中明确记录定向复验及依赖声明冲突。本审计未运行训练、backward、optimizer step、模型生成或真实数据集处理。

## 固定来源与历史回退

| 对象 | 本次核对值 |
| --- | --- |
| 固定上游、`anima-upstream/main`、`MERGE_HEAD` | `25eaf52718e2f8f3b3549a6cdad62353da5d8c78` |
| 审计时本地 `HEAD` | `e981e6f0c852eafe2244adedebcaa9b806832b7a` |
| 集成分支 | `codex/sync-anima-upstream-20261018` |
| 原批准计划记录的同步前本地提交 | `396f79b3`；不是本次工作树快照的 `HEAD` |
| 纯工具来源 | anime-tools `v0.7.15` / `7ea30612ebec736bbc3a243d153dca4d0d2e3853` |

[2026-07-15 报告](upstream-sync-20260715.md) 记载 `71745620` 已将 `cece8e10` 放入祖先历史，`70bfdc1f` 回退的是文件树，普通 merge 不会重新引入祖先提交中的全部文件。[2026-09 报告](upstream-sync-2026-09.md) 又明确拒收了当时的 anime_tools、daemon 迁包和多个研究支线。本次计划已经扩大为完整上游同步，因此两份旧报告用于定位历史缺口，不能继续把其中的旧排除清单当作此次授权边界。

按固定上游的 `git ls-tree -r` 做正向路径覆盖审计：**1,952 个上游路径全部存在于当前工作树**。这只证明路径覆盖，不证明这些文件的所有调用关系、运行行为或数值一致性。

反向比较审计时 `HEAD` 中的本地独有路径：共 774 个。初次检查发现 111 个原路径不存在，经主代理恢复本地文件和重建 vendor 后再次检查，**只剩 1 个旧路径**：trainer vendor 中旧 `scripts` 空包的 `__init__.py`。该旧包随 daemon vendor 迁移退役；现入口为 `custom_nodes/comfyui-anima-trainer/_vendor/anima_daemon/__init__.py`。`nodes.py` 已使用 `._vendor.anima_daemon.client`，`sync_vendor.py` 的 `TRAINER_PACKAGE_DIRS` 也仅登记 `anima_daemon`；因此不补造不再被消费的旧 vendor 包。

## 本地保留与上游适配

| 能力 | 当前代码承接位置与审查事实 | 验收边界 |
| --- | --- | --- |
| 本地 Anima 适配器 | `networks/__init__.py`、`networks/lora_anima/` 保留 LoKr、LoHa、GLoKr、OrthoInit、Hydra、Chimera、DyLoRA、VeRA 等注册及互斥规则；没有拿 Qwen/MOSS 的绑定替换 Anima network factory | 旧权重加载、保存和混合精度由受限回归验证；此表不证明数值表现 |
| 量化 | `library/runtime/int8_linear.py`、`library/runtime/convrot/` 与 `library/training/convrot.py` 存在，`train.py` 仍调用 `maybe_apply_convrot_base` | 本次不执行量化模型训练或显存性能基准 |
| V100 | `requirements-v100.txt` 保留 Torch `2.10.0+cu129`；`networks/flash_attn_v100_compat.py`、attention dispatcher 的 provider/兼容层、FP32 adapter master 与 eager 路径仍在树中；更新器使用同一 runtime resolver 识别受保护环境并跳过普通 `uv sync` | `runtimeapi-tests.txt` 保存 32 项通过记录；浏览器目录将未登记的 V100 标为不可用。未进行 V100 实机安装、训练或指纹变化实测 |
| 普通/整理环境 | `library/runtime/profiles.py` 定义 `default`、`curation`、`v100`；登记文件为 `configs/custom/runtime.toml`；`/api/system/runtimes` 返回可用性，Vue 仅提交已登记 profile；`task_service` → daemon 持久化实际解释器和环境。整理任务固定使用 `curation`，未独立登记时继承普通环境 | 隔离环境已确认 Torch `2.14.1+cu132`、anime-tools `0.7.15` 和 11 阶段 schema，见下文安装证据；本地 `.venv` 更新曾因磁盘不足中断，固定依赖完整重装后的实际导入与 443 项通过、1 项修正后定向通过已记录 |
| 纯工具兼容 | `library/curation_compat/` 包含 MIT LICENSE、SOURCE.json 与固定来源说明，训练所需语法/桶/路径/sidecar 使用该门面；不把完整 anime_tools 安装进 V100 2.10 环境 | 43 个记录文件的 vendored SHA256 全部匹配；静态检索未发现该目录直接导入 torch 或完整 anime_tools |
| daemon 迁包 | `anima_daemon/` 为实现，`scripts/daemon/` 为兼容入口；旧 Job 字段与 `rc/returncode` 双读、WebUI sidecar、串行队列、attempt、continuation 保留 | API/状态机验证使用临时目录和停止的 worker；不启动真实训练 |
| 恢复边界 | 同 job 重启持久化 `progress_epoch/progress_start_offset`，只用当前进程起点之后的终态；daemon JSONL 追加保留先前 samples/steps。已登记 runtime 的名称、解释器、环境与登记存在性在恢复和启动前复核；未知 Job 字段平铺保留 | 新增回归仅临时 JSONL/登记文件与 mock 进程对象；不把旧 paused 事件当作新进程成功证据 |
| 三模型隔离 | Qwen 自身 request/loader/checkpoint/cache；MOSS 自身 request/model/cache；共同复用 `library.lines` 和 blockswap，不把 Anima 缓存布局或续训协议套到 Qwen/MOSS | `/continue` 显式拒绝非 Anima family；Qwen 官方目录/分片与 Comfy 单文件及量化拒绝边界仍存在 |
| MOSS WebUI API | `webui/api/moss.py` 与 `moss_service.py`、`moss_workspace.py` 提供 schema/profile/preflight/任务/结果；`webui/server.py` 注册 `/api/moss`；`task_service` 合并 prepared request 的 model/cache/output context | MOSS tune 可提交，不在本次验收运行其中训练步骤；图像和报告须绑定实际 job |
| Qwen/MOSS 权重元数据 | `library/lines/adapters.py` 在读取 tensor 前拒绝显式 `base_model` 跨模型族，写出时模型族由绑定决定；保留缺少该字段的历史格式兼容 | 只凭模型族不足以验证所有 tensor shape；完整加载仍由原有 strict load 负责 |
| Qwen/MOSS 看板事件 | 两条训练入口通过 `library.lines.progress` 向 daemon 提供的 jobdir 写标准 run_start/step/ckpt/run_end；step 使用现有看板的 `loss/average` 与 `lr` 键；独立 CLI 继续保留其 stdout/JSON report | 只增通用作业身份 env；Anima stop/resume 协议仍独立，不给两模型赋予 `/continue` 语义；真实训练未执行 |
| Turbo 新旧能力 | `setup.py` / `RunContext` / `distill.run_loop` 使用上游拆分结构；保留 soft-rank、CDM、DMAD、dual-pool、动态 sigma 与本地 OrthoInit、REPA、mean-variance | REPA 在 GAN 前即时 backward 的原有编排保留；新旧功能组合未做训练数值验收 |
| Turbo checkpoint 身份 | setup 将 preflight layout/base SHA 绑到实际 model；`TurboDMDNetwork.save_student` 在 plain/dual/expert 分派前注入 `adapter_identity_metadata` 与 `ss_num_blocks` | 不以仅设置 `model.anima_base_sha256` 代替保存 metadata；实际权重序列化数值测试未在此审计执行 |
| Anima 验证与预览 | `validation.py` 恢复 DiT→VAE→PE 三阶段及 FM 辅助；`anima/training.py` 增加分离采样/解码 helper，保留本地 sample_dir、JSONL sample、暂停/结束预览、ER-SDE | 未运行 CMMD、FM 前向或图片生成；wrapper 在解码异常时也恢复 VAE 原设备 |
| 原生整理与 Vue 模型工作区 | `DatasetBrowser` 原生挂载 `CurationWorkspace`；11 阶段、版本梯级、标注差异、标签/官方子句编辑、OCR 框、mask 叠加、分组、建议审阅、排除恢复、设置及导出报告均有 Vue 消费者；三模型由 `modelWorkspace` 与独立 Qwen/MOSS profile store 驱动配置、预处理、任务、看板和结果入口 | `ui-qa/qa-final.json`、`export-roundtrip.json` 及 19 张截图证实实际 API 的临时工作区交互和桌面/390px 布局；结果归属使用模拟任务单独验证。没有执行模型阶段或真实生成 |
| 整理任务与历史操作 | `webui/api/curation.py` → `curation_service` → `curation-stage` → `scripts/curation/runner.py` 复用官方 schema/parser/底层处理函数，不使用官方独立 JobManager；预处理/主阶段/后处理保存在一个 daemon job 的有序 steps 中 | snapshot 与报告摘要校验、已登记 roots、路径/符号链接边界及文件漂移拒绝有专门回归；历史 apply/undo 从原 job 快照恢复 roots/workspace/env，源码未发现退回当前页面 roots 的路径 |

## 本轮找到并处理的语义缺口

1. Qwen 三个冲突测试没有被文本解析闭合。已合并 shared `library.lines` / `runtime.blockswap` 边界与 WebUI torch-free 检查，保留 backward 测试的 `qwen_training` 标记；没有执行该 backward 测试。
2. daemon 的显式解释器提交会丢 `runtime_env`，续训及自动 chain 也会丢 profile 的环境。已贯通同步 client→HTTP→manager，并使续训元数据在 worker 可以取任务前发布。默认 profile 名持久化为 `default`。切换 chain profile 时重新解析，未切换时继承已保存解释器和环境。
3. history 启动清理与本地保留要求冲突。启动不调用 prune；默认 retention 为 0，仅显式 CLI 清理。`/unpause`、`/continue` 与旧 `/resume` 状态分派独立，冻结 attempt 保持占用，released attempt 释放设备槽位但仍属于活跃 lineage。
4. Turbo 冲突内容曾把旧单体 main 和新 `ctx` loop 拼接，并删掉 parser/metrics 的上游字段。已以上游结构承接本地功能，补齐配置、上下文、REPA metrics 和三种保存格式的 base identity。
5. Anima validation 曾回到旧双阶段实现，丢掉上游 `sample_image_latents` / `decode_latents_to_image` 和 FM 辅助。已恢复两文件的成套调用链；本地训练预览与停止约定保留。
6. 只读检查发现 Spectrum SEA 参数、FSG/CFG++ side-channel、inference CLI 参数和旧 `set_hydra_content` import 断链，以及 vocab-pack cache stamp 校验遗漏，主代理修复后已重新读取确认：SEA 三参数和 parser 定义存在；FSG/CFG++ 从 generation 传到 side-channel；旧内容调用改用存在的 `set_hydra_crossattn`；cache 检查恢复 `check_cache_stamp`。此处为源码复核，不代表图像数值验证。
7. Qwen/MOSS 原训练脚本只有 stdout 与结束 JSON report，且 daemon jobdir env 原本只给 Anima。已补标准 JSONL 生命周期和 common job 身份 env，使两模型看板可以读取逐步 loss/lr；没有把 Anima training command 集合扩展为两模型。初始化失败也写 0 步 error 终态，KeyboardInterrupt 由既有 `run_scope` 写 stopped；强制杀进程仍由 daemon 进程状态决定终态。
8. 最后一轮独立审查定位到旧 paused JSONL 污染重启终态、pinned interpreter 绕过登记和未知 Job 字段写回丢失，已按主代理要求修复。每次 spawn 前只记录流起点，不删除原文件；daemon sink 追加并隔开无换行尾部，普通 CLI 维持原覆盖语义。普通 continue、token continuation、released requeue 和最终 launch 都校验当前 runtime，不静默更换 path/env。unknown 字段保存在私有容器并平铺写回，public 的 continuation 仍只返回原摘要白名单。

## 缺项复核状态与剩余验收

以下按最后一次读取更新。源码断链已修复的项目与尚未执行的平台/集成验收分开记录。

| 断链/待验收项 | 证据及影响 | 责任边界 |
| --- | --- | --- |
| 本地 mod-distill 命令 | 已重新确认 `utilities.cmd_distill_mod` 与 `scripts/distill_mod/` 六个实现文件恢复；registry 原断链消除 | 源码修复已复核，未执行蒸馏 |
| LoRA fusion 工具迁移 | 已重新确认 `utilities.cmd_merge_loras` 启动 `scripts/toolkits/merge_loras.py` | CLI 消费者路径已修复，未使用真实权重融合 |
| PID 历史 bench | 两个 bench 仍使用原路径，`scripts/calibration/fit_color_calib.py` 已恢复，消费者目标重新存在 | 历史脚本断链已消除，未执行模型校准 |
| 隔离运行环境 | `environment214-smoke.txt` 记录 Torch `2.14.1+cu132`、TorchVision `0.29.1+cu132`、Transformers `5.16.1`、anime-tools `0.7.15`、FlashAttention `2.8.3+cu132torch2.14`；`cuda_visible False`；11 个整理阶段可读取。`new-environment-tests.txt` 记录该隔离环境 106 项通过 | 已证实隔离环境的安装与受限检查；不能据此推导 GPU 可用性、全部依赖声明兼容或项目 `.venv` 恢复成功 |
| 项目环境恢复与最终回归 | `project-environment-update.txt` 保存 FlashAttention 安装时 `No space left on device` 的中断；恢复过程写入 `project-environment-recovery.txt`，最终白名单和执行结果由主代理生成 | 实际导入确认 Torch 2.14.1；统一白名单 443 项通过，保留 independent-A 的预期修正后 1 项定向复验通过。详见交付记录与 final-recheck.txt |
| SAM3 依赖声明 | SAM3 声明 `numpy<2`，本项目声明并通过 uv override 使用 `numpy>=2`；该 override 继承固定上游，`uv pip check` 仍报告这 1 项冲突 | 这是保留的已知声明冲突，不是全部依赖检查通过；本次没有执行 SAM3 模型推理验证其数值兼容性 |
| 原生整理交互 | `qa-final.json` 记录真实 API 返回 synthetic OCR/mask sidecar，显示 2 个 OCR 框并过滤其中 1 个；切换外层工作区标签保留未保存 caption；导出应用在审阅确认前禁用。`export-roundtrip.json` 记录实际临时源标注写入、撤销恢复和不再显示重复撤销入口 | 此前“仅有组件/schema”的待验收项已关闭；测试素材为临时 fixture，模型驱动的打标、OCR、SAM 推理未执行 |
| 三模型 UI 与结果归属 | 实际 API 验证 MOSS profile 保留 rank 24、运行环境目录、MOSS 预处理继承所选环境、缓存状态及桌面/390px 无横向溢出；`simulated-results.json` 分别请求 `/api/moss/tasks/fixture-moss/...` 和 `/api/qwen21/tasks/fixture-qwen21/...` | 模型族和 job 归属链已验证；任务完成、manifest 与图片响应在该结果检查中是明确模拟，不能作为真实模型生成或训练结果证据 |
| 平台与模型数值 | Windows、ROCm、V100 实机、真实 Anima/Qwen/MOSS/Turbo 训练数值均未在本审计执行 | 按批准计划明确保留为未执行项；真实训练由用户负责 |

## 保存的验收证据

本节仅摘录已存在并读到终态的证据，不重新执行测试。所有路径均相对仓库外的 `/home/buxinzi/Projects/AI/MonadForge-sync-backups/20261018-continuation/`，各批测试可能重叠，不合计为唯一测试数。

| 证据文件 | 已观察结果 | 能支持的结论 |
| --- | --- | --- |
| `pure-tests.txt` / `preprocess-tests.txt` | 50 passed / 24 passed | 纯工具、已审查的预处理范围回归已执行；预处理白名单另存 `preprocess-whitelist.json` |
| `control-tests.txt` / `telemetry-tests.txt` | 33 passed / 50 passed | 控制协议、恢复边界与通用进度事件相关批次已执行 |
| `runtimeapi-tests.txt` / `compat-tests.txt` | 32 passed / 59 passed | runtime/API 和兼容性相关批次已执行 |
| `new-environment-tests.txt` | 106 passed，1 条 FastAPI/Starlette httpx 弃用警告 | 新隔离环境中的已选非训练检查通过，不替代最终统一白名单 |
| `frontend-final-tests.txt` | 53 项：52 passed，1 skipped，0 failed | Vue/表单/工作区相关已有前端检查已执行；跳过项保留为跳过 |
| `frontend-build.txt` / `ui-qa/final-build.log` | 两次生产构建均完成，最后一次 `built in 4.08s` | 前端生产产物可构建；最后日志仍有大于 500 kB chunk 的体积提示 |
| `ui-qa/qa-final.json` | `passed: true`，`errors/failed/blocked` 均为空 | 系统 Chromium 经已有 Playwright 对真实本地 API 的最终交互复核通过；Browser 插件不可用这一工具边界已记入报告 |
| `ui-qa/export-roundtrip.json` | `state: success`、`sourceWritten: true`、`sourceRestored: true`、`repeatUndoVisible: 0` | 临时素材的预览→应用→撤销实际闭环通过；原始素材未参与 |
| `ui-qa/simulated-results.json` | `simulation: true`、两模型各自 task results/thumbnail 请求、`errors: []` | Qwen/MOSS 结果不会随当前工作区模型切换串到另一模型；响应为明确模拟，不代表生成通过 |
| `ui-qa/interactive-final-console.json` | `errors/failed/blocked` 均为空 | 保存的最终交互没有捕获到控制台错误或失败请求 |

保留早期失败日志以便追踪：`curation-tests.txt` 曾有 1 项失败，`config-tests.txt` 与 `protocol-tests.txt` 各有 2 项失败，`integration-tests.txt` 曾有 1 项失败；`static-final-draft.txt` 曾报告 `cond_embed` 未定义。当前源码已调整 SAM 阶段报告断言、`train_adaln` 默认、专用显存 profile 规则、`artists_shard` CLI 与运行时注入字段的 closure allowlist；`generation.py` 的对应 Hydra 调用也已使用该作用域内的 embedding。后续 `compat-tests.txt`、`new-environment-tests.txt` 和 `static-latest.txt` 保存了通过结果，但旧日志本身不重写为通过。这些早期失败已在最终统一白名单中通过；该轮另发现的 independent-A 退役假设冲突已修正并通过定向复验，保留两份真实日志。

本轮追加审查沿实际调用链读取了三模型的 schema/profile/cache/preflight/job/result API、运行环境目录及 task service 转交、Vue model/profile store、配置切换和预处理消费者、任务结果与看板模型身份，以及整理 snapshot/report/apply/undo 和原生编辑组件。未发现需要新增源码修复的具体致命遗漏；该结论限于静态调用链与上述保存的证据，不覆盖未执行的模型前向、平台实机和数值行为。

## 本审计实际执行的检查

- 读取固定 Git 对象与两份历史同步报告，对 1,952 个上游路径做工作树覆盖，反查 774 个本地独有路径，并追踪缺失运行脚本的消费者。
- daemon/Qwen/MOSS 归属范围 79 个 Python 文件 AST 解析通过；Turbo 四个模块 AST 解析和 Ruff 未定义名/重复定义检查通过；validation/helper 两文件同类检查通过。
- 直接运行无模型配置解析：默认 `TurboConfig` 133 字段，以及 REPA、OrthoInit、soft-rank、CDM、DMAD、dual-pool、mean-variance 解析/快照；扫描 Turbo `cfg.*` 引用未发现未声明字段。
- `train.py`、`library/anima`、`library/training`、`library/inference` 的函数参数与固定上游比较，明确定位上述语义缺口；对直接内部函数调用的 keyword 做 AST 核对，未发现其余可静态解析的参数不匹配。该检查不覆盖动态 dispatch 和模型实际前向。
- 对 `library/curation_compat/SOURCE.json` 中 43 个文件逐个计算 SHA256，全部符合记录。
- 新增受限回归文件供主代理审查运行：`test_daemon_migration_contract.py`、`test_moss_interface.py`、`test_adapter_metadata_boundary.py`、`test_turbo_config_migration.py`、`test_turbo_metadata_boundary.py`、`test_lines_progress.py`、`test_daemon_recovery_boundaries.py`，以及 `test_training_continuation.py::test_continuation_submit_rechecks_runtime_before_copying_state`。进度测试仅从源 AST 抽取公开 wrapper，用 mock driver 发事件，绝不 import/执行两模型实际训练 loop；daemon env/恢复测试不启动子进程，token 测试仅用既有微型合成状态 fixture。本文件不把“新增测试”写成“测试通过”。

收尾复核只读取源码与既有证据并更新本文件，没有重新执行测试、环境安装或浏览器操作，没有执行全量 pytest、模型训练、梯度计算、生成、性能基准或真实数据集写入。本审计没有暂存、提交、push 或改动历史；最终集成与提交状态由主代理交付记录确认。
