# run-analyzer 任务书（TASKBOOK）

> 目标：在遵循《终末地官网设计体系拆解》的设计与美学约束前提下，完成
> 模块化重构 + 18 项 UI 进化 + 3 项已确认修复。每完成一个批次执行一次代码审查，
> 全部完成后执行一次总审查。状态跟踪见 `STATE.md`。

## 1. 设计红线（所有批次强制）

| 红线 | 说明 |
|---|---|
| R1 无圆角 | 全局 `border-radius: 0`；ECharts tooltip/图表元素同样 0 圆角 |
| R2 无阴影 | 无 `box-shadow`；ECharts tooltip 用 `extraCssText` 压制默认阴影 |
| R3 无玻璃拟态 | 无 `backdrop-filter` / 大面积模糊 `filter` |
| R4 黄色克制 | 黄色仅用于：状态点、选中态、动作锚点、扫描光、进度填充、KPI 变更反馈 |
| R5 键盘完整 | Tab 全链路 + volt 焦点环；新组件（下拉/面板）↑↓/Enter/Esc + aria |
| R6 动效克制 | 200-300ms easeOutQuad；`prefers-reduced-motion` 全部关闭 |
| R7 模块边界 | `echarts` 只出现在 `core/charts.js`；视图间不直接操作对方 DOM（事件总线） |
| R8 零 JS 错误 | 每批 Playwright 断言 `pageerror` / console error 为 0 |

## 2. 模块化架构（批 0 落地）

```
scripts/run_analyzer/static/
├── index.html                  # 骨架：视图容器 + <script type="module" src="/js/main.js">
├── echarts.min.js              # vendored，经典脚本，暴露 window.echarts
├── css/
│   ├── tokens.css              # 设计令牌 :root + reset
│   ├── base.css                # 全局/左轨/main/报头/表格/chips/stage/meta/toast/响应式/reduced-motion
│   ├── components.css          # checkbox/seg/sc-chip/block-btn/text-btn/empty/lightbox/kpi
│   ├── view-runs.css           # 档案索引
│   ├── view-detail.css         # 详情（head/sec-nav/总览/参数/画廊/事件/每轮表）
│   ├── view-monitor.css        # 监控器（mon-bar/下拉/网格/胶片带）
│   └── view-compare.css        # 对比
└── js/
    ├── main.js                 # 启动、路由表、视图生命周期、索引轨展开
    ├── theme.js                # JS 设计令牌：C 色板 / ST 状态 / MONO/SANS 字体
    ├── core/
    │   ├── dom.js              # $/$$/esc/fmtNum/fmtLr/fmtDur/fmtTs/toast/api
    │   ├── charts.js           # ECharts 基建：stageBase/mountChart/preserveZoom/rolling/zoomTo/chartRegistry/epochAt
    │   ├── state.js            # localStorage 记忆（A3）+ cmpSel 共享选中集
    │   └── shortcuts.js        # 全局快捷键（A2）
    ├── charts/
    │   ├── loss.js             # lossChartOption(data, ui) —— 档案/监控共用（色带/标注/采样/扫描光）
    │   ├── lr.js               # lrChartOption(data, log)
    │   └── norm.js             # normChartOption(data)
    ├── components/
    │   ├── dropdown.js         # 自定义下拉（F2，批 1 落地）
    │   ├── lightbox.js         # 采样预览遮罩
    │   ├── kpi.js              # KPI 数字墙渲染 + 变更反馈（E17，批 4）+ 圆弧进度（C7，批 4）
    │   └── states.js           # 空态/加载态（E18，批 4）
    └── views/
        ├── runs.js             # 档案索引（B4/B5/B6 批 2）
        ├── detail.js           # 详情编排：章节导航/实时轮询/挂载五章节
        ├── curves.js           # 曲线视图（控件/每轮表/采样闭环 C8/跨图联动 C9，批 4）
        ├── params.js           # 参数视图（C10，批 4）
        ├── samples.js          # 采样画廊
        ├── events.js           # 事件视图（stdout 查看器）
        ├── compare.js          # 对比页
        └── monitor.js          # 监控器（D11-D15 批 3）
```

### 模块契约
- 视图统一导出 `mount(root, ctx)`（ctx = 路由上下文）；销毁由路由切换时整容器替换
- 视图状态模块私有（`detail`/`monUI`/`runsCache`/`cmpSel` 等各归其模块）
- 跨视图通信走 `CustomEvent`（命名空间 `ra:*`）：
  - `ra:open-run`（id）→ 路由跳转
  - `ra:open-sample`（sample, runId）→ lightbox 监听
  - `ra:compare-picked`（ids）→ compare/runs 同步勾选态
  - `ra:run-ended`（name）→ toast 提示
- `charts/loss.js` 是档案与监控共用的 option 构建器；ECharts 调用仅在 `core/charts.js`
- CSS 与视图成对：`view-x.js` + `view-x.css`

## 3. 批次任务表

### 批 0：模块化重构（先行，零行为变化）
| 任务 | 内容 | 验收 |
|---|---|---|
| 0.1 | 建目录骨架 + tokens.css/base.css 拆分 | CSS 令牌/全局规则迁移，无样式回归 |
| 0.2 | theme.js + core/dom.js + core/charts.js + charts/{loss,lr,norm}.js | 图表行为与现有一致 |
| 0.3 | main.js 路由 + views/{runs,detail,curves,params,samples,events,compare,monitor}.js | 全部视图行为一致 |
| 0.4 | components/{lightbox,states}.js + 各 view CSS 拆分 | 样式与现有一致 |
| 0.5 | 全量回归（R1-R8 + 既有断言） | 行为零变化 |

### 批 1：基础修复 + 导航效率
| 任务 | 内容 | 验收 |
|---|---|---|
| 1.1 F1 | KPI 溢出：网格 `minmax(0,1fr)` + `CKPT/SAMPLE` 精简 | 单元格 right ≤ stage right |
| 1.2 F2 | components/dropdown.js 自定义下拉（触发/面板/键盘/aria） | 开合/键盘/选中样式；R1-R7 |
| 1.3 F3 | 来源徽标 tooltip 语义 + 无数据曲线空态 | 悬停提示正确；空态呈现 |
| 1.4 A2 | core/shortcuts.js：g+r/m/c、/、? 键位面板 | 快捷键全生效；输入框聚焦不干扰 |
| 1.5 A3 | core/state.js：控件状态/最后运行/监控选中持久化 | 刷新后恢复；索引高亮上次运行 |

### 批 2：档案索引页
| 任务 | 内容 | 验收 |
|---|---|---|
| 2.1 | server：index mtime 缓存 | 连续两次 /api/runs 第二次不重解析（日志验证） |
| 2.2 | server：sparkline 60 点降采样入 index_payload | 每行 spark 数组长度 ≤60 |
| 2.3 B4 | 行内 sparkline SVG（56×20，hover 转 volt） | 23 行全部渲染；hover 变色 |
| 2.4 B5 | 行内勾选 + 底部浮动动作条「对比 N 项」 | 勾选≥2 出现动作条；点击带入对比页 |
| 2.5 B6 | 状态筛选 chips 计数 | 计数与列表一致 |

### 批 3：监控器实时态
| 任务 | 内容 | 验收 |
|---|---|---|
| 3.1 D11 | 最新 step 扫描光（volt 竖线，仅运行中） | 运行中显示、结束消失 |
| 3.2 D12 | 进度条每轮刻度 + 当前轮段 | 刻度位置 = epoch_spans |
| 3.3 D13 | 事件流时间轴（类型锚点 + `+Ns` 微标 + 警告过滤） | 三类锚点正确；过滤生效 |
| 3.4 D14 | 采样实时高亮（volt 闪 300ms）+ hover volt 轮廓 | 新采样闪烁；hover 生效 |
| 3.5 D15 | 队列可见性（顶栏「队列 N」+ 下拉 queued 脉冲点） | 计数正确；样式生效 |

### 批 4：详情页深度 + 动效
| 任务 | 内容 | 验收 |
|---|---|---|
| 4.1 C7 | 总览圆弧进度（黑图版 + volt 弧 + 中心 step） | 弧比例正确；无 total 显示 — |
| 4.2 C8 | 采样↔曲线闭环（scatter 点击开 lightbox） | 点击采样点打开对应图 |
| 4.3 C9 | 跨图 zoom 联动（loss↔lr/norm，防循环） | 单图缩放其余跟随 |
| 4.4 C10 | 参数关键微标行 | 白名单键渲染正确 |
| 4.5 E16 | 视图切换 clip-path 揭示 | 切换视图标题展开动画（reduced-motion 关闭） |
| 4.6 E17 | KPI 数字变更反馈（300ms volt 淡出） | 数值变化触发一次 |
| 4.7 E18 | 空态坐标十字 + 加载逐字揭示 | 空态/加载态呈现 |

### 总审查
- 全量回归（批 0-4 所有断言重跑）
- 模块边界审计（grep echarts 仅 core/charts.js；无跨视图 DOM）
- 设计合规审计（R1-R6 全绿）
- 代码质量总评（按 chinese-code-review 规范出具报告）
- STATE.md 收尾

## 4. 审查规范
- 每批结束 → 代码审查（chinese-code-review：分级标注 必须修复/建议修改/仅供参考/问题）
- 审查范围：本批新增/修改文件 + 模块边界 + 回归结果
- [必须修复] 不修完不进下一批；[建议修改] 记入 STATE.md 待办
- 总审查后出具最终报告
