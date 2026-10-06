---
name: MonadForge WebUI
description: 玄鼎工坊 —— 余烬琥珀点缀的暗色 Anima/LoRA 训练工作台
colors:
  ember: "#EBA375"
  on-ember: "#251C16"
  teal: "#88BDB5"
  on-teal: "#152522"
  gold: "#E3BC75"
  bg-deep: "#111315"
  bg-base: "#141618"
  bg-surface: "#1C1F22"
  bg-elevated: "#272B2F"
  bg-hover: "#2C3136"
  text-primary: "#EEF0F2"
  text-secondary: "#A8AFB6"
  text-muted: "#929AA3"
  text-disabled: "#69727B"
  border-subtle: "#2A2E33"
  border-default: "#383E44"
  border-strong: "#555F69"
  error: "#F18C96"
  info: "#91B9EA"
  success: "#89C5A2"
  ember-light: "#9D4C23"
  teal-light: "#34756C"
  gold-light: "#956914"
  bg-base-light: "#F6F7F8"
  bg-surface-light: "#FFFFFF"
  text-primary-light: "#242A30"
  text-secondary-light: "#626C76"
  border-subtle-light: "#E0E5E9"
typography:
  headline:
    fontFamily: "Geist, 'Microsoft YaHei', 'Noto Sans SC', sans-serif"
    fontSize: "26px"
    fontWeight: 600
    lineHeight: 1.35
  title:
    fontFamily: "Geist, 'Microsoft YaHei', 'Noto Sans SC', sans-serif"
    fontSize: "15px"
    fontWeight: 600
    lineHeight: 1.5
  body:
    fontFamily: "Geist, 'Microsoft YaHei', 'Noto Sans SC', sans-serif"
    fontSize: "14px"
    fontWeight: 400
  label:
    fontFamily: "Geist, 'Microsoft YaHei', 'Noto Sans SC', sans-serif"
    fontSize: "13px"
    fontWeight: 500
  mono-metric:
    fontFamily: "'JetBrains Mono', ui-monospace, monospace"
    fontSize: "20px"
    fontWeight: 500
    lineHeight: 1.6
rounded:
  sm: "4px"
  md: "6px"
  lg: "8px"
components:
  button-primary:
    backgroundColor: "{colors.ember}"
    textColor: "{colors.on-ember}"
    rounded: "{rounded.sm}"
    height: "34px"
  button-tonal:
    backgroundColor: "rgb(235 163 117 / 12%)"
    textColor: "{colors.ember}"
    rounded: "{rounded.sm}"
    height: "34px"
  button-outlined:
    backgroundColor: "transparent"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    height: "34px"
  card-tonal:
    backgroundColor: "{colors.bg-surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.md}"
    padding: "16px"
  input-outlined:
    backgroundColor: "{colors.bg-surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.sm}"
    typography: "{typography.label}"
  nav-item-active:
    backgroundColor: "rgb(235 163 117 / 12%)"
    textColor: "{colors.ember}"
    rounded: "5px"
    height: "38px"
  metric-value:
    typography: "{typography.mono-metric}"
    textColor: "{colors.ember}"
---

# Design System: MonadForge WebUI

## Overview

**Creative North Star: "玄鼎工坊 The Ember Forge"**

暗色基底是炉膛，余烬琥珀是炉火。MonadForge 的界面是一座温暖的工坊而非冷冰冰的控制面板：深炭色的表面由 1px 描边勾勒出层次，余烬琥珀（Ember，#EBA375）只在关键操作、活跃状态和焦点反馈处点亮，像炉膛里被拨亮的炭火——稀缺，所以醒目。工作台整体致密而安静：34px 的按钮、13px 的表单字段、紧凑的导航项，把像素留给真正重要的东西——训练读数、损失曲线和采样图。庆祝时刻（任务完成的 confetti）是这座工坊仅有的"烟火"，点到即止。

界面是"温暖工坊・余烬点缀"：密度优先、读数精确、装饰最少，但琥珀点缀、磨砂玻璃（可选）和庆贺动效保留了手艺人的温度。组件手感是**精密机床按键**——低高度、平实表面、按下下沉 1px、焦点光晕，可靠、即时、不炫耀。所有数值读数（loss、学习率、步数、GPU 温度）一律使用 JetBrains Mono 等宽字体，正文与控件永不用 mono；这是"机器读数"与"人读文字"的分工。

本文件描述主 WebUI（`webui/frontend/`，Vuetify 4 MD3 blueprint）。暗色是默认与规范主题；亮色是一套完整并行的低饱和主题（*-light tokens）。`scripts/run_analyzer/` 不在此系统内（遵循其外部设计系统报告）；Windows 托盘是原生 UI，同样不在范围。

**Key Characteristics:**
- 暗色优先的温暖工坊：炭色色阶分层 + 1px 描边，卡片零阴影
- Ember 琥珀是唯一品牌强调色，稀缺使用；Teal 只作数据第二系列与选中态
- 机器读数全部 JetBrains Mono（20px/500 起），人读文字 Geist
- 紧凑密度：34px 按钮、38px 导航项、13px 字段、60px 顶栏
- 无全大写、无字距（letter-spacing 锁 0，中英日韩四语友好）
- 动效即反馈：进场上浮 240ms、按下下沉 1px、焦点 ember 光晕；prefers-reduced-motion 全局尊重

## Colors

色板是"炉膛中性色 + 三点炉火"：近 90% 的屏幕面积由炭色色阶和两级文字色承担，强调色只在交互与语义处出现。状态色是低饱和的粉彩调（MD3 风格），与品牌强调色完全独立（见 `main.scss` 首注释）。

### Primary
- **余烬琥珀 Ember**（#EBA375，亮色 #9D4C23）：唯一品牌强调色。主按钮、激活导航项、焦点光晕、进度环、loss 曲线、关键指标值。文字压在其上用 on-ember（#251C16）。
- **余烬辉光 Ember Glow**（rgb(235 163 117 / 12%)）：ember 的半透明形态，用于激活导航底色、tonal 按钮、焦点环（3px）。它是"余烬的呼吸"，不是实色。

### Secondary
- **青瓷绿 Teal**（#88BDB5，亮色 #34756C）：数据第二系列——学习率曲线、显存/内存读数、采样卡选中态（hover 描边 + 最新样本光环）。刻意与 ember 拉开色温，两色永不同屏争抢同一角色。

### Tertiary
- **熔金 Gold**（#E3BC75，亮色 #956914）：单点警告/注意强调（如暂存分辨率视图），同时是语义 warning 色。用得比 teal 更少。

### Neutral
- **炉膛色阶**：bg-deep（#111315，导航）→ bg-base（#141618，页面/顶栏）→ bg-surface（#1C1F22，卡片与字段）→ bg-elevated（#272B2F，骨架屏）→ bg-hover（#2C3136，悬停）。
- **文字阶梯**：text-primary（#EEF0F2）/ text-secondary（#A8AFB6）/ text-muted（#929AA3）/ text-disabled（#69727B）；配套 Vuetify 语义透明度 0.92 / 0.64 / 0.32。
- **描边阶梯**：border-subtle（#2A2E33，卡片与分隔线）→ border-default（#383E44，控件描边/滚动条）→ border-strong（#555F69，强调悬停）；字段描边用 border-color @ 0.28 透明度。
- **状态色**（与品牌独立）：error 玫红 #F18C96、info 雾蓝 #91B9EA、success 苔绿 #89C5A2。GPU 温度语义阈值：≥80°C error、≥65°C warning、否则 success。

### Named Rules
**The Scarce Ember Rule（余烬稀缺）.** Ember 及其辉光占任一屏幕面积 ≤10%。主操作区至多一个实色 ember 按钮；其余动作用 tonal / outlined。稀缺即价值——满屏琥珀等于没有琥珀。
**The Cool Companion Rule（冷伴规则）.** 同一视图需要第二数据系列或"选中"语义时用 Teal，永远不用第二个暖色。Teal 不参与品牌表达，只做数据与状态。

## Typography

**Display/Body Font:** Geist（回退 Microsoft YaHei → Noto Sans SC → sans-serif）
**Label/Mono Font:** JetBrains Mono（回退 ui-monospace → monospace）

**Character:** Geist 的几何人文感配上 JetBrains Mono 的工程读数感——"工坊里的仪表"。中文环境无缝回退到系统黑体，字号阶梯为四语设计，无字距、无全大写。

### Hierarchy
- **Headline**（600，26px，1.35）：页面主标题（workspace-heading h1）；移动端降至 22px。
- **Section Title**（600，15px，1.5）：区块标题（workspace-section-title）。
- **Subtitle**（600，13–18px）：卡片标题（text-subtitle-2 13px / text-h6 18px）。
- **Body**（400，14px）：正文与应用基准字号（`.v-application { font-size: 14px }`）。
- **Label/Field**（500，13px）：表单字段、导航项、面包屑；字段内文字 13px。
- **Caption**（12px）：次要说明、表头；**Micro**（11px）：导航分组标签、品牌副标题。
- **Mono Metric**（500，20px/1.6，JetBrains Mono）：指标数值；大读数 26px（进度百分比）；引擎徽章 10px。

### Named Rules
**The Machine Readings Rule（机器读数规则）.** 一切数值——loss、LR、步数、epoch、温度、显存、路径、命令行——一律 JetBrains Mono。正文、标题、按钮永不用 mono（`.font-mono-field` 是字段的 mono 变体）。
**The No Shouting Rule（不喊叫规则）.** 无全大写按钮/标签，字距全局锁定 0（`letter-spacing: 0 !important`）。层级靠字号与字重，不靠变形。

## Layout

固定工作台外壳 + 流体内容区：

- **外壳**：左侧导航 240px（可收为 72px rail，图标 + tooltip），顶栏 60px（折叠按钮 + 面包屑"分组 › 页面" + 引擎徽章 + 指南按钮）。移动端（<600px）导航变临时抽屉。
- **内容区**：fluid container；≥960px 内边距 26px 32px，以下 16px。整页 flex 列布局、内部滚动——日志卡固定 320px 高、事件列表 200px 内滚，页面骨架不随内容塌陷。
- **节奏**：区块间距 24px；卡片内边距 16px；采样网格 gap 10px；Vuetify 4px 间距网格（pa-4 等）。
- **网格**：指标瓦片 `repeat(auto-fit, minmax(min(100%, 150px), 1fr))`（系统监控 155px）；采样图 `repeat(auto-fill, minmax(140px, 1fr))` 1:1 方卡。
- **断点**：960px（列折叠、内边距切换）与 599px（隐藏面包屑分组与引擎徽章、对话框标题降号、系统网格 2 列）。
- **页首模式**：每个视图以 workspace-heading 开场——左侧 h1 + 13px 次要说明，右侧动作区；空状态用 72px 描边方框图标 + 18px 次要标题 + 单个引导按钮。

## Elevation & Depth

**没有环境阴影。** 深度来自两条轴：炉膛色阶（deep → base → surface → elevated，每层一级灰度）与 1px 描边（subtle/default/strong）。卡片 elevation 0、展开面板阴影移除；悬浮菜单/对话框落在近不透明的 overlay token（#1C1F22）上保证可读性。光晕（glow）是**状态反馈**而非氛围：焦点环 `0 0 0 3px ember@12%`、激活导航项、最新采样样本的 teal 光环（`0 0 0 1px teal, 0 0 12px rgba(199,91,26,0.18)`）。

可选的**磨砂玻璃外观**（`html.forge-glass`）：表面 token 换为半透明值，`backdrop-filter: blur(16px) saturate(1.25)` 只施加于导航与顶栏两个单实例表面（性能与驱动兼容考虑）；浮层保持 ≥90% 不透明的可读性下限。可选**壁纸**画在 `<html>` canvas 上（浏览器最底层绘制），叠加主题化遮罩（暗色 55% / 亮色 +10% 起步）。

### Shadow Vocabulary
- **Focus Ring**（`box-shadow: 0 0 0 3px rgb(235 163 117 / 12%)`）：字段与键盘焦点（配全局 `:focus-visible` 2px ember 描边、offset 3px）。
- **Latest Sample Halo**（`box-shadow: 0 0 0 1px #88BDB5, 0 0 12px rgba(199,91,26,0.18)`）：采样网格最新一张的标记。
- **Nav Active Bar**（`box-shadow: inset 2px 0 #EBA375`）：激活导航项的左侧余烬条。

### Named Rules
**The Glow-Is-Feedback Rule（光晕即反馈）.** 光晕只回答"我在哪/什么在发生"——焦点、激活、最新。永远不用作装饰、不用作环境氛围。
**The One-Pixel Frame Rule（一像素框规则）.** 分层靠 1px 描边与色阶，不靠投影。新表面若需"浮起来"，先加一级炉膛色阶，再考虑描边，最后才允许 glow。

## Shapes

方正偏冷的几何语言，圆角克制：字段与按钮 4px（radius-sm）、卡片与面板 6px（radius-md）、采样方卡与空状态标牌 8px（radius-lg）。唯二的例外：导航项 5px（介于按钮与卡片之间，专属外壳）与徽章/滚动条 3px。全部矩形——无圆形容器（进度环是绘制的圆环，不是容器形状）。描边是这个系统的"轮廓线"：tonal 卡片 = surface 底 + 1px subtle 描边 + 隐藏 underlay，这是默认卡片形态。

## Components

组件手感：**精密机床按键**——可靠、即时、不炫耀。所有组件遵守 compact 密度与 4px 基础网格。

### Buttons
- **Shape:** 4px 圆角，min-height 34px，字重 500，无全大写。
- **Primary:** 实色 ember 底 + on-ember 文字（flat variant）。
- **Hover / Focus / Active:** 按下下沉 `translateY(1px)`；键盘焦点 2px ember 描边；无弹跳。
- **Secondary:** tonal（ember@12% 底 + ember 文字）用于次要动作；outlined 用于低频动作；text/icon 用于外壳操作。无色 elevated/flat 按钮统一走 bg-surface token。

### Chips
- **Style:** 小号 tonal 为主，字重 500；状态 chip 用语义色（error/success/warning）。
- **State:** 计数 chip 用 x-small outlined（如"128 points"）。

### Cards / Containers
- **Corner Style:** 6px；**Background:** bg-surface；**Border:** 1px border-subtle；**Shadow:** 无（见 Elevation）；**Internal Padding:** 16px。
- 标题可换行（`overflow-wrap: anywhere`）；tonal 变体是默认形态。

### Inputs / Fields
- **Style:** outlined + compact，13px 文字，4px 圆角，bg-surface 底；描边 border-color @ 0.28。
- **Focus:** `0 0 0 3px` ember 辉光（160ms ease 过渡）；标签 13px。
- **Mono variant:** 路径/数值字段用 `.font-mono-field`（JetBrains Mono）。

### Navigation
- 左侧 240px / rail 72px；分组标签 11px text-muted；导航项 38px、5px 圆角、13px/500。
- **默认** text-secondary → **hover** bg-hover + text-primary（140ms）→ **激活** ember 文字 + ember@12% 底 + 左侧 2px inset 余烬条。
- 品牌区 80px：32px logo（6px 圆角）+ 17px/650 名称 + 11px 副标题。

### Metric Tile（签名组件）
标签 12px text-secondary，值 JetBrains Mono 500 20px/1.6，可带语义色（温度阈值、显存警告）。网格 auto-fit 平铺，是"仪表读数"的基本原子。

### Sample Gallery（签名组件）
1:1 方卡（8px 圆角、1px subtle 描边），hover 上浮 2px + teal 描边（120ms），底部渐变遮罩信息层淡入；最新样本带 teal 光环；步数徽章 10px 黑底白字。

### 动效词汇
页面进场上浮 240ms（GSAP power2.out，y 6px）、fadeSlideUp 240ms ease-out、骨架 shimmer 1.5s、错误 shake 300ms、任务完成 confetti；tooltip 延迟 350ms。全部让位于 `prefers-reduced-motion: reduce`。

## Do's and Don'ts

### Do:
- **Do** 数值读数一律 JetBrains Mono；表单路径/命令字段用 `.font-mono-field`。
- **Do** 新表面先选炉膛色阶层级（deep/base/surface/elevated），再补 1px 描边。
- **Do** 焦点/激活/最新用 ember 辉光反馈，配全局 `:focus-visible` 描边。
- **Do** 每视图保持 workspace-heading 页首与 workspace-empty 空状态的既有模式。
- **Do** 整页 flex 列 + 内部滚动，让日志/事件列表内滚而非撑开页面。
- **Do** 动效短促（≤300ms）且尊重 prefers-reduced-motion。
- **Do** 亮色主题用 *-light tokens 的低饱和对应色，不做高对比反转。

### Don't:
- **Don't** 高饱和渐变滥用——不做大渐变背景、大面积高饱和色块、发光标题。
- **Don't** 装饰性动效——不弹跳、不拟物、不为动而动；动效只表达状态变化。
- **Don't** 插画风插画堆砌——空状态用 72px 描边标牌 + 一句话 + 一个动作，不上插画/吉祥物。
- **Don't** 用投影分层替代色阶与描边；glow 不做环境装饰（The Glow-Is-Feedback Rule）。
- **Don't** 同屏用两个暖色强调；第二系列用 teal（The Cool Companion Rule）。
- **Don't** 全大写或字距样式（letter-spacing 锁 0）；不用 radius 超过 8px 的大圆角容器。
