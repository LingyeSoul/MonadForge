# PR 9: 逻辑训练任务与安全续训

PR 9 将一次用户可见的训练从“单个 daemon 进程”提升为“一个逻辑任务、多个物理 attempt”。停止或失败后的续训会创建新的 job 目录，但 WebUI 与 run analyzer 仍将整条 attempt 链展示为一个训练任务。

本文同时记录合并审查后的安全加固：恢复状态所有权、成功后的 rolling state 生命周期、并发 resume 原子预约、任务历史查询隔离，以及 Windows 托盘运行态确认。

## 数据模型

每个物理 daemon job 仍独立保存在：

```text
output/daemon/jobs/<job_id>/
  job.json
  config.snapshot.toml
  progress.jsonl
  stdout.log
  sample/
```

`job.json` 新增以下关系字段：

- `root_job_id`: 逻辑任务 id。初始 attempt 默认等于自己的 `job_id`。
- `parent_job_id`: 当前 attempt 由哪个 attempt 恢复而来。
- `attempt_index`: 从 0 开始的物理尝试序号。

每次 resume 都创建新的物理 job，旧 attempt 的日志、样本、退出原因和进程信息保持不可变。daemon 的 `/job-groups` 接口、WebUI `TaskInfo.attempts` 和 run analyzer 按 `root_job_id` 聚合这些记录。

## 不可变配置快照

WebUI 提交训练时解析当时的合并配置，并由 daemon 写入该 job 的 `config.snapshot.toml`。后续 attempt 继续使用这份快照，不重新解释可能已经改变的 preset、profile 或 GUI 配置。

快照同时提供训练目标、数据 manifest 和 config signature。训练器在 `run_start` 事件中写出 config/data signature，daemon 将它们持久化到 job 元数据，用于恢复候选筛选和诊断。

配置与数据 signature 只能证明两个状态兼容，不能证明状态属于同一次逻辑训练。因此 PR 9 审查后增加了独立的所有权协议。

## Recovery ownership schema v3

daemon 启动训练进程时注入：

```text
ANIMA_DAEMON_JOB_ID=<physical attempt id>
ANIMA_DAEMON_ROOT_JOB_ID=<logical root id>
```

`CheckpointSaver` 将两者写入 schema v3 `train_state.json`：

```json
{
  "schema_version": 3,
  "job_id": "20260812-...",
  "root_job_id": "20260811-...",
  "global_step": 1200,
  "config_signature": "...",
  "dataset_signature": "..."
}
```

daemon 自动恢复要求候选状态同时满足：

1. 状态目录完整并带有 publication marker。
2. schema 至少为 v3，且 `job_id` 非空。
3. `root_job_id` 与当前逻辑任务完全一致。
4. config/data signature 匹配。
5. 若快照给出了 `target_steps`，状态的 `global_step` 尚未达到该目标。

训练器的 load hook 在 daemon 环境中再次校验 `root_job_id`，防止候选发现与实际加载之间的路径替换或调用绕过。

`current_epoch` 表示当前已经进入的 epoch，不表示该 epoch 已完成。因此 epoch-only 任务在最后一轮中断时仍允许恢复；daemon 不会仅凭 `current_epoch == target_epochs` 误判训练已经完成。正常完成的任务会进入终态并清理 rolling state，不再提供自动恢复候选。

### Legacy 兼容边界

schema v1/v2 没有可靠的逻辑任务所有权。daemon 不再自动把这类状态关联到某个 job，即使 config/data signature 相同。

显式命令行 `--resume <state-dir>` 在非 daemon 环境中仍可读取旧状态。这保留手工迁移和历史训练恢复能力，但所有权判断由操作者承担。旧 job 仍可查看、删除历史或作为独立 logical root 展示。

## Rolling state 生命周期

`<output_name>-rolling-state` 是运行中 crash/stop 恢复用的临时 optimizer state，不是成功训练后的续训入口。

训练正常完成后，main process 始终删除 rolling state，即使未启用 `checkpointing_epochs`。这避免后续同名新训练静默拾取陈旧状态，也释放 optimizer 等大体积 artifact。

以下内容不会因此被删除：

- 最终模型文件。
- 显式 `save_state_on_train_end` 产生的 end state。
- 中断/失败后用于 resume 的有效状态。
- daemon job 历史、日志、progress 和 sample。

需要在成功训练后继续增加步数时，应显式选择保存的 end state 或创建明确的后续训练配置。

## 同一 root 的原子 resume

daemon HTTP server 是多线程的。仅检查“latest attempt”和“没有 active attempt”不足以阻止两个并发请求同时创建 `attempt_index == 1` 的分支。

`JobManager` 现在维护进程内 `_resuming_roots` reservation：

- 在 manager lock 内完成 source/latest/active 检查并预约 `root_job_id`。
- 状态发现和新 attempt 创建期间保持该 root 的 reservation。
- 新 attempt 注册完成或任一步骤失败后在 `finally` 中释放。
- 不同 root 仍可并发准备；同一 root 的第二个请求返回 HTTP 409。

## WebUI 任务历史隔离

Pinia store 的 `tasks` 是 Dashboard、训练入口和重复提交检查共享的 canonical task 集合。历史页的 filter/page 不再写入这份共享状态。

- `fetchTasks()` 固定获取 canonical 第一页，最多 500 条，并更新共享 `tasks`。
- `fetchTaskPage()` 返回 `{tasks, total}`，不修改 canonical 列表或全局查询状态。
- `TaskMonitorView` 本地维护 filter、page、page size、total、error 和历史结果。
- Task Monitor 轮询同时刷新 canonical tasks、queue status 和当前历史页。
- legacy resume 确认使用被点击的历史行，不依赖该任务是否恰好存在于 canonical 列表。

因此在 Task Monitor 选择 `failed` 或翻到后续页，不会再隐藏 Dashboard 的 active task，也不会削弱其他视图的重复提交判断。

## Windows 托盘状态

托盘不再仅凭 `/health.active_job` 显示“正在运行”。它会读取 job 详情，并确认：

- job state 仍为 `running`；
- `(pid, create_time)` 仍对应同一个活进程。

终态 job、陈旧 active id、详情读取失败或已退出/复用的 PID 都回落到 idle/error，避免 daemon 启动或进程退出窗口中误显示运行态。

## WebUI 与 run analyzer 聚合

WebUI 的 task service 使用 `/job-groups` 恢复逻辑任务，将所有 attempt 的 stdout、progress、metrics 和退出原因串联展示。当前 attempt 的状态决定任务卡片状态，attempt history 保留每次启动时间、恢复 step、exit code 和 terminal reason。

run analyzer discovery 同样按 `root_job_id` 合并 daemon jobs，并为每个事件和 sample 保留 attempt 来源。跨 attempt 的 loss/step 时间线、样本路径和详情页因此不会丢失物理边界。

## 删除边界

删除逻辑任务历史会删除该 root 下所有 terminal attempt 的 daemon-owned job 目录。它不会删除 `output/ckpt` 下的模型或训练 state，因为这些可能被其他流程引用。

活动 attempt 所属的 logical group 不允许删除。删除失败时 manager 会重新加载仍存在的 job 记录，保持内存与磁盘一致。

## 验证

审查加固包含以下回归覆盖：

- owner 匹配状态可恢复；异 root 和 v3 ownerless 状态被拒绝。
- daemon load hook 校验 root；非 daemon 显式 legacy resume 保持兼容。
- 成功训练在无 `checkpointing_epochs` 时也删除 rolling state。
- 同一 root 的两个并发 resume 最多创建一个 child attempt。
- epoch-only 任务在目标 epoch 中途停止后仍可恢复。
- 托盘只对仍存活的 running job 显示运行态。
- Pinia/Vue TypeScript 契约通过 `vue-tsc`，生产前端通过 Vite build。

最终命令与通过数量以合并前的 CI/本地验证记录为准。
