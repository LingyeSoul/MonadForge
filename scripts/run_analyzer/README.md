# run-analyzer — 训练档案分析器

独立 Web 工具：读取 MonadForge 训练器实际写入的数据（progress.jsonl /
tensorboard events / snapshot.toml / stdout.log / job.json / sample 目录），
在浏览器中以交互图表详细展示每次训练的损失曲线、参数与日志。

UI 设计严格遵循《终末地官网设计体系拆解》（`终末地官网/analysis/design-system-report.md`）：
碳黑/纸白/电能黄角色分工、1px 细线、矩形裁切（无圆角）、左侧索引轨、
编辑式构图；黄色仅用于选中/进行中/关键动作；不复制官网任何素材。

## 启动

```bash
# 在 MonadForge 仓库根目录
.venv/bin/python -m scripts.run_analyzer.server --port 8320 --open
```

浏览器打开 `http://127.0.0.1:8320/`。依赖：fastapi / uvicorn / tensorboard
（venv 已有）；ECharts 已 vendored 于 `static/echarts.min.js`，离线可用。

## 数据源（与训练器写入代码一一对应）

| 来源 | 内容 | 写入方 |
|---|---|---|
| `output/daemon/jobs/<id>/progress.jsonl` | run_start/step/val/ckpt/sample/log/run_end 事件流 | library/training/progress.py |
| `<log_dir>/network_train/events.out.tfevents.*` | 全量标量（loss/lr/norm/vr） | accelerate tensorboard tracker |
| `<log_dir>/<run>.snapshot.toml` | 合并参数（按来源分组） | train.py --print-config |
| `output/daemon/jobs/<id>/stdout.log` | 原始输出（尾部/过滤） | daemon |
| `output/daemon/jobs/<id>/job.json` | daemon 元数据 | scripts/daemon/jobs.py |

数据源优先级：`progress.jsonl > tensorboard > stdout`；无 daemon 任务的内联
CLI 运行（仅 `output/logs/` 目录）也会被索引。

## API

- `GET /api/runs` — 运行索引（状态/步数/轮数/final loss/来源标记）
- `GET /api/runs/{id}` — 完整分析（序列 + 每轮 min/max/mean/std + 参数 + 事件 + 采样）
- `GET /api/runs/{id}/live?since=ts` — 增量事件（实时轮询）
- `GET /api/runs/{id}/stdout?tail=&q=` — stdout 尾部/过滤
- `POST /api/compare {ids}` — 多轮叠加对比
- `GET /api/runs/{id}/samples/{fn}` — 采样图

## 交互要点

- 三个视图：训练档案（索引 + 详情长卷）、对比（多轮叠加）、监控（实时监控器，左轨 03）
- 监控器：顶栏 = 运行下拉选择 + 状态/ LIVE 指示 + 进度条（step/E/时长/it·s）；左侧 LOSS 主舞台 +
  LR 图（2:1 与每轮统计表并排，统计行点击联动缩放）；右侧 LIVE KPI + 事件流；底部全宽采样胶片带；
  自动跟随运行中任务，2s 增量刷新，任务结束自动停表
- 曲线页：平滑滑杆（1–200）、对数/线性、每轮范围色带（E# min~max 标注）、原始点线开关、
  序列开关、滚轮缩放/框选、tooltip 显示所属轮次、每轮明细表点击跳转该轮
- 实时模式：运行中任务 2s 增量刷新，索引页 5s 刷新；切回前台立即刷新（visibilitychange）
- 键盘：Tab 全链路可达，`Enter` 进入；`Esc` 关闭采样遮罩
- `prefers-reduced-motion` 自动关闭动画

## 采样预览（按图片分辨率匹配）

- 服务端读取每张采样图实际尺寸（PIL 只读文件头 + 进程缓存），随 `/api/runs/{id}` 下发 `w/h`
- 画廊缩略图与监控胶片带按 `--ar`（真实宽高比）渲染：`aspect-ratio: var(--ar)` + `object-fit: cover`，
  与源图比例一致时零裁剪、零变形；加载后 `naturalWidth` 兜底校正
- 缩略图与 lightbox 头部显示分辨率徽标（如 `832×1216`）
- 监控胶片带为横向滚动条带：按训练时间线从左（早）到右（新）完整展示全部采样，
  新采样闪烁并自动跟随（仅当用户已贴右时）；132px 定宽、真实比例展示

## 设计合规

UI 严格遵循《终末地官网设计体系拆解》：颜色角色仅 碳黑/纸白/电能黄/中性灰（无体系外荧光色）；
错误状态用 volt 静态点 + 加粗表达（与运行中 volt 脉冲区分）；多序列图表（对比/lr/norm）使用
灰阶梯度调色板，选中序列以 volt 高亮；全部色值/字号收敛为设计令牌（`tokens.css` + `theme.js`），
最低字号 `--ts-nano`(10px)；无圆角/阴影/玻璃拟态；1px 细线；黄色仅用于状态/选中/动作/扫描。
