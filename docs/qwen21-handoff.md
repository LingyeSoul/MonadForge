# Qwen-Image 2.1 移植交接（WIP，禁止直接视为可合并）

交接日期：2026-09-26。交接基线：MonadForge `d0701f58`，提交交接前与 `LingyeSoul/MonadForge` 的 `main` 一致。
移植参考：`sorryhyun/anima_lora` 的 `qwen21` 分支，已核对的参考提交为 `50bf09d5ad9d47bf055fa3282f0f802362c73352`，不是对其最新分支状态的承诺。

用户要求停止当前实现，由上游接手。本 PR 保存现有实现和未完成的 UI 融入工作，不宣称功能完成、测试通过或可以直接合并。

## 必须保留的需求和安全边界

- ComfyUI 单文件是对官方 Diffusers 模型目录及权重切片的补充，不能删除后者；DiT、文本编码器、VAE 可以分别覆盖，tokenizer/processor 仍需要小文件目录。
- 保留上游自动参数语义：自动 swap/激活预留、alpha、warmup 等；空值表示自动，显式 `0` 不能变成默认值。
- 当前实现明确拒绝 INT8/FP8/W4A8 等量化权重，不自动反量化，不承诺量化兼容；单文件主要目标为 Comfy-Org BF16 组件。
- 不另做 Qwen 页面或侧栏入口；融入现有配置、预处理、任务、看板、测试和图库，沿用原有主题、壁纸、布局、帮助和多语言机制。
- Anima 的方法、硬件预设、配置合并、恢复训练路径不得被 Qwen 改写或假冒支持。
- **禁止实际训练测试。** 用户负责训练验收，不得运行 backward、optimizer step、完整训练冒烟或训练基准以完成本交接。
- 此前曾发生内存占满导致机器卡死；后续任何获准的非训练检查必须先设硬内存上限、禁用 swap，不能顺手加载完整 DiT/文本编码器、跑完整缓存或生成。不可直接运行全部 Qwen 测试，部分测试包含 backward。

## 已有实现与入口

| 范围 | 文件与当前实现 |
| --- | --- |
| 模型与训练移植 | `library/qwen21/`：上游训练/缓存/生成、LoRA、block swap、扫描、请求默认值和校验；`configs/*.json` 为组件配置，不含权重 |
| 模型来源 | `loader.py`、`checkpoint.py`：官方目录/切片路径与 Comfy 单文件路径；映射 fused gate/up、文本前缀与 VAE 形状，检查键/形状/量化标记；meta 初始化和赋值加载 |
| scheduler | 独立 DiT 且无显式官方根目录时使用内置配置；仅覆盖 TE/VAE 不改变 scheduler；保留显式覆盖 |
| CLI | `tasks.py`、`scripts/tasks/qwen21.py`、`scripts/qwen21/`：`qwen21-cache/train/generate/processor/cache-train`，支持队列 |
| 缓存后训练 | `scripts/qwen21/workflow.py`：同一 job 中先缓存子进程，成功退出后才启动训练子进程，训练输入固定为缓存输出 |
| daemon | `scripts/daemon/` 与 WebUI client：按 job 保存/传递 stall timeout，Qwen 使用 900 秒；其他 job 保持原默认语义 |
| WebUI API | `webui/api/qwen21.py`、`webui/services/qwen21_service.py`：schema、路径解析、提交、结果接口；本次增加 profile、预检查、缓存状态、任务绑定结果 |
| 依赖 | `pyproject.toml`、`uv.lock`：Diffusers 固定到 `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`；更新 Hub/safetensors 要求，打包组件 JSON |

完整权重加载、真实 GPU offload、训练数值和显存/内存占用未被验收；不能据此承诺与上游约 20 GB 内存表现一致。

## 已批准、正在实现的 UI 融入方案

1. 在现有 `ConfigEditor.vue` 工具栏选择 Anima / Qwen-Image 2.1；预处理和测试跟随配置页模型上下文，不在其他页面另加模型选择器。
2. 配置仍使用原生字段、文件选择器、帮助、搜索、保存/重载/训练/测试入口；Qwen 只展示真实支持的 LoRA 与自动硬件参数。
3. Qwen profiles 保存于 `configs/custom/qwen21/` 的 TOML，分 models/cache/train/generate；自动空值不写入 TOML，读取时从请求默认值还原。
4. 切换模型或 profile 有保存/丢弃/取消确认；浏览器旧草稿只通过显式导入迁移，不能静默覆盖服务器配置。
5. 预处理使用 Qwen 缓存字段及状态，不混用 Anima PE/条件缓存；训练前只检查路径、文件头、缓存成对关系及元数据，缺缓存时让用户选择预处理或缓存后训练。
6. 生成使用现有测试按钮；结果固定绑定 job 的输出和 manifest，而非当前表单目录；复用任务详情和图库。
7. 任务列表统一；看板依据选中任务的模型显示，而不是编辑器当前模型；不能把 Qwen 伪装成 Anima 恢复训练或训练中采样。
8. 旧 `/qwen21` 路由跳转 `/config?model=qwen21`，不恢复独立页面。

对应文件已写入但**未完成新一轮验证**：

- `stores/modelWorkspace.ts`、`stores/qwenWorkspace.ts`：模型上下文、profile、草稿和动作状态。
- `components/QwenFields.vue`、`QwenConfigFields.vue`、`QwenPreprocessContent.vue`、`QwenTaskResults.vue`：嵌入式内容，不是独立页面。
- `ConfigEditor.vue`、`PreprocessView.vue`、`TaskMonitorView.vue`、`TrainingDashboard.vue`、`App.vue`、`router.ts`：现有入口整合。
- `ConfigField.vue`、`ModelPathField.vue`、`SampleGallery.vue`：共享字段/路径/图库扩展，必须检查 Anima 回归。
- `i18n/qwen21.ts` 与四语言入口：新文案已在树中，但未完成逐项显示验收。
- `webui/services/qwen21_workspace.py`：profile 规范化/原子保存、限制大小的 safetensors 头解析、切片与缓存检查。

以上前端路径相对 `webui/frontend/src/`。旧独立 `Qwen21View.vue` 和 `QwenRequestForm.vue` 不在交接树内；不要把它们的删除当作迁移已经验收。

## 当前已知风险与接手顺序

优先检查，不要先扩功能或重做设计：

1. **静态与构建门禁尚未重跑。** Vue 大段模板条件化、TypeScript 类型收窄、新 store 与多语言引用必须先检查；此前 Python API 曾出现 E402，调整导入后未形成完整门禁结论。
2. **边界测试需要复核。** `tests/test_qwen21_boundary.py` 的 WebUI 白名单尚未包含新增 `qwen21_workspace.py`，该模块直接引用 Qwen requests；应审查边界设计后更新测试，而不是删除隔离检查。
3. **profile 行为需要实测。** 保存过程中继续编辑、创建失败后的名称/基线恢复、切换竞态、未保存确认及每模型草稿保留；检查规范化是否正确保留空值与零。
4. **文件/API 边界需要审查。** profile 路径和 symlink、缓存 metadata、切片索引一致性、结果 manifest 白名单、任务 argv 输出目录固定、缩略图；保留的旧目录结果接口也要一起审查。
5. **共享控件回归风险。** nullable float 输入和 blur、目录选择、字段稳定 ID、已有 Anima 字段、任务日志与图库布局、HelpPanel 展开时机、编辑模型路径后解析结果失效处理。
6. **只读预检查不能加载模型。** 新增 workspace 和 API 的导入链应保持 torch-free；校验失败不得吞错或降级。检查 service 对 workspace 的延迟导入关系。
7. **依赖为全项目共享。** 先确认 Diffusers/Hub/Transformers 导入兼容，再做 Anima 非训练回归；历史上发生过 `huggingface_hub.resolve_revision` 导入不兼容，不能仅凭锁文件存在宣称解决。
8. **文档和测试收尾。** 当前新增前端测试主要覆盖旧 `inputValue` 空值/零转换，不等于新共享控件和完整流程覆盖；profile、缓存状态、任务结果接口也缺少本轮验收。

接手后按照仓库规范执行 Ruff、前端 `npm test` / `npm run build` 及最小相关检查，但应先审阅测试是否会加载模型或执行训练，设置资源限制，并遵守用户授权。仓库通用数值 bench 要求并不覆盖用户的禁止训练指令；训练相关合并门禁应显式保持未完成，而非擅自运行。

## 验证证据：仅限整合前版本

2026-09-24 的本地非训练验证报告记录：Python 54 passed（排除一个训练测试，未运行 LoRA backward 测试文件）；前端 19 passed、生产构建成功；11 项真实 HTTP 检查，没有提交有效训练 job。旧独立页面进行了路径、空值/零与高级参数浏览器检查。

组件检查包括官方大模型文件头（DiT 297 / TE 750 / VAE 238 tensors）、CPU BF16 VAE 32×32 往返与单文件/目录结果比较、INT8 头检查拒绝。未物化完整 DiT/TE 权重；未训练、未执行 backward/optimizer、未使用 GPU、未下载权重。该批检查内存硬上限 4 GiB、swap 0，最大记录约 2.8 GiB。

这些结果来自本地保留报告，不是本 PR 的可复现 CI 附件；本地 output、模型、日志和截图不上传。**上述通过结果不能用于证明后续共享 UI/profile/API 改动通过。** 交接时只做文件差异与交付审查，不再继续功能实现、启动服务或运行模型。

## 交付与合并状态

这是供维护者继续工作的 Draft PR，不应直接合并。未上传本地工具配置目录、用户数据、模型、缓存、输出日志或凭据。所有实际训练验收留给用户；如需调整该边界，必须另行取得明确授权。
