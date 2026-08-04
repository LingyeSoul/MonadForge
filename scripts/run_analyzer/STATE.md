# run-analyzer 状态机记录（STATE）

## 1. 状态机定义

```
                    ┌──────────────────────────────┐
                    ▼                              │
 PLANNED → IN_PROGRESS → SELF_CHECK → REVIEW ──→ REVIEWED → DONE
    ▲                    │             │  ↑           │
    └────────────────────┘             │  └──── FIX ───┘
        (计划调整)                      │       ↑
                                       └───────┘ (总审查发现问题 → 回 FIX)
```

| 状态 | 含义 | 进入条件 |
|---|---|---|
| PLANNED | 已计划 | 任务书写入 |
| IN_PROGRESS | 执行中 | 批次开始 |
| SELF_CHECK | 自测中 | 实现完成，跑验收 |
| REVIEW | 审查中 | 自测通过 |
| FIX | 返修中 | 审查发现 [必须修复]/[建议修改] |
| REVIEWED | 审查通过 | 无 [必须修复] 遗留 |
| DONE | 完成 | 批次全项通过 + 回归绿 |

迁移规则：
- 一次只有一个批次处于 IN_PROGRESS
- FIX 修复后回 SELF_CHECK 重新验证，再 REVIEW
- 每批必须经历 REVIEW 才能 DONE；总审查在所有批次 DONE 后执行

## 2. 批次状态总表

| 批次 | 任务数 | 状态 | 审查结论 | 遗留项 |
|---|---|---|---|---|
| 批 0 模块化重构 | 5 | DONE | 通过（2×[建议修改] 未使用 import，已修） | 无 |
| 批 1 基础修复+导航 | 5 | DONE | 通过（15/15） | 无 |
| 批 2 档案索引页 | 5 | DONE | 通过（12/12）；[建议修改] 循环依赖已修 | 无 |
| 批 2 档案索引页 | 5 | PLANNED | — | — |
| 批 3 监控器实时态 | 5 | DONE | 通过（12/12）；[建议修改] 渲染顺序已修 | 扫描光待真实运行任务实测 |
| 批 4 详情+动效 | 7 | DONE | 通过（12/12） | 无 |
| 总审查 | — | DONE | 通过 | 扫描光/D14 闪烁待真实运行任务实测 |

## 3. 模块状态表

| 模块 | 职责 | 状态 | 审查 |
|---|---|---|---|
| css/tokens.css | 设计令牌 | DONE | 通过 |
| css/base.css | 全局/左轨/报头 | DONE | 通过 |
| css/components.css | 通用组件 | DONE | 通过 |
| css/view-*.css | 视图样式 | DONE | 通过 |
| js/main.js | 路由/生命周期 | DONE | 通过 |
| js/theme.js | JS 令牌 | DONE | 通过 |
| js/core/dom.js | 工具函数 | DONE | 通过 |
| js/core/charts.js | ECharts 基建 | DONE | 通过 |
| js/core/state.js | 记忆/选中集 | DONE | 通过（A3 批 1 扩展） |
| js/core/shortcuts.js | 快捷键 | PLANNED | 批 1 |
| js/charts/loss.js | loss option | DONE | 通过 |
| js/charts/lr.js | lr option | DONE | 通过 |
| js/charts/norm.js | norm option | DONE | 通过 |
| js/components/dropdown.js | 下拉 | PLANNED | 批 1 |
| js/components/lightbox.js | 采样预览 | DONE | 通过 |
| js/components/kpi.js | KPI/圆弧 | PLANNED | 批 4 |
| js/components/states.js | 空态/加载 | DONE | 通过（E18 批 4 扩展） |
| js/views/runs.js | 档案索引 | DONE | 通过（B4-B6 批 2） |
| js/views/detail.js | 详情编排 | DONE | 通过（C7 批 4） |
| js/views/curves.js | 曲线视图 | DONE | 通过（C8/C9 批 4） |
| js/views/params.js | 参数视图 | DONE | 通过（C10 批 4） |
| js/views/samples.js | 采样画廊 | DONE | 通过 |
| js/views/events.js | 事件视图 | DONE | 通过 |
| js/views/compare.js | 对比 | DONE | 通过 |
| js/views/monitor.js | 监控器 | DONE | 通过（D11-D15 批 3） |

## 4. 状态迁移日志

| 时间 | 批次/模块 | 迁移 | 说明 |
|---|---|---|---|
| 批 0 | 全部模块 | PLANNED → DONE | 模块化重构完成；回归 22/22；设计合规 radius/shadow/blur=0；零 JS 错误。审查：2×[建议修改] 未使用 import 已修复 |
| 批 1 | — | PLANNED → DONE | F1 溢出修复/F2 自定义下拉/F3 徽标语义+空态/A2 快捷键/A3 状态记忆；回归 15/15 |
| 批 2 | — | PLANNED → DONE | server mtime 缓存 + sparkline 60 点；B4 行内迷你曲线/B5 勾选+浮动条/B6 状态计数；回归 12/12；循环依赖下沉 core/dom.js |

## 5. 总审查记录（2026-08-04）

### 回归
- 批 0：22/22｜批 1：18/18｜批 2：12/12｜批 3：12/12｜批 4：12/12（合计 76/76，零 JS 错误）

### 模块边界审计
- `echarts` 仅出现在 `core/charts.js` ✓（grep 验证）
- 全部 21 个 JS 模块 ESM 解析通过（node type:module 环境）✓
- 视图间无跨 DOM 操作（detail/curves 的 `#sec-curves` 为既定编排契约）✓
- 循环依赖已消除（parseHash 下沉 core/dom.js）✓

### 设计合规审计
- `border-radius / box-shadow / backdrop-filter` 全 0（含 tooltip 弹出态、下拉面板、浮动条、圆弧图版、帮助面板）✓
- 黄色仅用于：状态点/选中态/动作锚点/扫描光/进度填充/采样点/KPI 变更反馈 ✓
- 键盘：下拉 ↑↓/Enter/Esc + aria、快捷键 g//?、焦点环 volt ✓
- prefers-reduced-motion 关闭全部动画 ✓

### 审查发现的问题（全部已修复）
1. [必须修复] samples.js 模板补丁吞掉 `}` 致 ESM 解析失败 → 已修
2. [必须修复] mountChart 跨运行导航持有陈旧 DOM 实例 → isConnected 校验重建
3. [建议修改] 未使用 import ×2、循环依赖（main↔runs）、node --check CJS 假阳性 → 已修/已建 ESM 验证流程

### 遗留（需真实运行任务实测）
- D11 扫描光、D14 新采样闪烁：无运行中任务时无法端到端验证（逻辑与空数据 markLine 安全性已验证）

| 修复对比轮次 | — | — | 对比图新增轮次分段：每运行半透明色带 + E# 标注层 + tooltip 每序列附 E#；新增「轮次分段」开关；服务端 compare 下发 epoch_spans；回归 10/10。修复 opt.series 覆盖 bands 的笔误 |
| 修复对比轮次·真因 | — | — | 根因：custom 色带 series 的 4 元数组数据 [x0,x1,y0,y1] 污染 y 轴自动取值域（取 [1] 元素=轮次结束 step，y 域变 [2,4678]），色带被裁出视野、曲线被压到底部。修复：loss/compare 图显式锁定 yAxis min/max（由 loss 数据计算），bands 关闭时色带数据置空。像素级验证：色带可见（bandHit 2457/623）、曲线分布恢复正常、开关/对数/监控/对比全回归 7/7。此修复同时解决详情图与监控图长期存在的曲线压底问题 |
| 对比 E# 标签不可见 | — | — | 根因：标签置于 yMax*1.06（=曲线最高点处），8px 小字与曲线重叠不可辨。修复：统一置于图表顶部留白区 globalMax*(1.06~1.095) 两级错开，y 轴 max 扩至 1.18×globalMax，字号 9；像素验证 2544 亮像素/38 标签，回归 5/5 |
| 采样区修复 | — | — | [必须修复] lightbox CSS 在批 0 拆分时整体丢失（grep 零命中）→ 补回 components.css（fixed/居中/遮罩/无圆角无阴影）；[必须修复] .g-cap 文件名无截断 → ellipsis+min-width:0+title 全名。验证：lightbox fixed 全屏居中/详情+监控双入口/Esc 关闭，g-cap 视觉零溢出，回归 6/6 + 设计合规全零 |
| 采样预览按分辨率匹配 | — | — | [必须修复] 源图 832×1216（比例 0.684）被固定 3:4/1:1 容器 cover 裁剪（监控裁 32%）。修复：analyze.py sample_list 附加 w/h（PIL 读头 + 进程缓存）；画廊/胶片带 aspect-ratio: var(--ar)，img 100%×100% cover；onload 兜底 naturalWidth 修正；g-cap 分辨率徽标 832×1216；lightbox lb-meta 附尺寸 + load 兜底；胶片带改 flex 横向滚动显示全部采样（去 slice(-6)）。验证：详情 20/20 比例 == natural、零裁剪、徽标、滚动 2659>1006 生效、lightbox 双入口、设计合规 87/87 |
| 胶片带步数升序 | — | — | [必须修复] 胶片带原 slice().reverse() 使高 step 在左。修复：去 reverse 升序（step 138→2760 从左到右）；flash 标记改为 i >= len-newCount（新采样在右）；自动跟随逻辑（贴右时跟随最新，但用 prevCount 保存旧值判定，避免首帧空容器误滚）。验证 6/6：升序/首帧 scrollLeft=0/滚动/徽标 |
| 设计合规统一 | — | — | 对照《终末地官网设计体系拆解》全面审查（7 CSS + 全 JS）：A. 删除体系外 green/pink 令牌，错误状态收敛为 volt 静态点+加粗（st-error/ev-row.err/l-err/tl-err/parse-err），STOP 标线改 paper 反相；B. 对比/lr/norm 调色板灰阶化（对比 8 线 paper+g1~g7、选中序列 volt 高亮保留）；C. 令牌收敛（--g9、--volt-dim/faint、--ts-nano(10px) 新增，hexA() 统一 rgba，#0c0c0c/#161616/内联 #999 全部类化）；D. 字号 <11px 全升 ts-nano/ts-micro；E. mon-film 死代码删除、queued 静态 volt、startTyping 尊重 reduced-motion、内联高度类化 chart-h-*/chart-gap。验证：全视图回归 11/11 + 错误状态 volt 静态点确认 + 颜色/字号静态审计零残留 |