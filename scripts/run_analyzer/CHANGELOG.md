# CHANGELOG — run-analyzer 训练档案分析器

本文件记录 run-analyzer 相对上游首版的完整交付清单与变更理由，
对应 PR（feat: run-analyzer training analysis dashboard）。

## 版本：v1.0.0（首版交付）

### 新增：训练档案分析器（scripts/run_analyzer/）

独立 Web 工具，读取 MonadForge 训练器实际写入的数据（progress.jsonl /
tensorboard events / snapshot.toml / stdout.log / job.json / sample 目录），
以交互图表展示每次训练的损失曲线、参数与日志。

- **三个视图**：训练档案（索引 + 详情长卷）、对比（多轮叠加）、监控（实时监控器）
- **索引页**：状态筛选 + 搜索 + 列排序、行内迷你曲线、勾选对比（8 项上限）、状态计数、来源徽标
- **详情长卷**：KPI 数字墙、圆弧进度、曲线（平滑/对数/每轮色带 E# 标注/采样点↔lightbox 闭环）、
  关键参数微标行 + 分组参数表、采样画廊（lightbox 大图）、事件流 + stdout 尾部过滤
- **对比页**：多轮 loss 叠加、每轮分段色带 + E# 标注、每轮均值点、tooltip 附轮次
- **监控器**：运行下拉（键盘可操作）、进度条（每轮刻度 + 当前轮 volt 底衬）、扫描光、
  LIVE KPI、事件时间轴（警告过滤）、每轮统计联动缩放、采样胶片带
- **交互**：实时轮询（2s/5s）、快捷键（g+r/m/c、/、?）、localStorage 状态记忆、
  视图切换 clip-path 揭示、逐字加载、坐标十字空态、prefers-reduced-motion 降级

### 新增：采样预览按图片分辨率匹配

- 服务端 `sample_list` 用 PIL 只读文件头（毫秒级）+ 进程缓存，为每条采样附加 `w/h`
- 画廊/胶片带 `aspect-ratio: var(--ar)` + `object-fit: cover`：容器比例 = 源图真实比例，
  零裁剪零变形（修复 832×1216 源图被固定 3:4/1:1 容器裁切的问题）
- img `load` 后 `naturalWidth/naturalHeight` 兜底校正（服务端缺失时）
- 缩略图与 lightbox 头部显示分辨率徽标 `832×1216`
- 监控胶片带改横向滚动：按训练时间线低→高步数完整展示全部采样，
  新采样右侧闪烁 + 贴右自动跟随（首帧不滚动）

### 新增：启动脚本集成（start-webui-linux.sh）

- `./start-webui-linux.sh` 同时拉起 WebUI（8000）与 run-analyzer（8320），
  自动打开两个标签；`Ctrl+C`/退出时 trap 清理两个服务，不留孤儿进程；
  单端口被占用不影响另一服务

### 修复：设计合规统一（对照《终末地官网设计体系拆解》）

- 删除体系外荧光色 `--green/--pink`：错误状态收敛为 volt 静态点 + 加粗
  （与运行中 volt 脉冲区分）；STOP 标线改 paper 反相标签
- 对比/lr/norm 图表调色板灰阶化（paper + g1~g7），选中序列 volt 高亮保留
- 全部色值令牌化：`--g9`/`--volt-dim`/`--volt-faint`/`--ts-nano` 新增，
  `hexA()` 统一 rgba，内联 `#999` 等类化（`.stage-sub`/`.spark-empty`/`.parse-err`）
- 字号合规：低于 `ts-micro`(11px) 的图表标注统一升至 `ts-nano`(10px)，
  杜绝 8.5/9px 损害长期使用舒适度
- 结构清理：mon-film 死代码删除、queued 状态静态化、startTyping 尊重
  reduced-motion、内联高度类化（`chart-h-*`/`chart-gap`/`masthead-tight`）

### 已知限制

- D11 扫描光 / D14 新采样闪烁仅设计实现，尚未在真实运行中任务上实测
- 对比图 tooltip 仅显示 avg（服务端暂不下发 loss/current，留待后续）
