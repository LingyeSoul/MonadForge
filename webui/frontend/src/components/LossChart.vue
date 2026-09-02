<template>
  <div ref="container" class="loss-chart-container">
    <svg
      v-if="data.length > 1"
      :viewBox="`0 0 ${width} ${height}`"
      preserveAspectRatio="none"
      class="loss-chart-svg"
    >
      <!-- Grid lines -->
      <line
        v-for="(y, i) in gridYs"
        :key="'gy' + i"
        :x1="paddingLeft"
        :y1="y"
        :x2="width - paddingRight"
        :y2="y"
        stroke="var(--border-subtle)"
        stroke-width="1"
      />
      <!-- Y-axis labels -->
      <text
        v-for="(tick, i) in yTicks"
        :key="'yl' + i"
        :x="paddingLeft - 6"
        :y="tick.y + 4"
        fill="var(--text-muted)"
        font-size="10"
        text-anchor="end"
        font-family="var(--font-mono)"
      >
        {{ tick.label }}
      </text>
      <!-- Gradient fill -->
      <defs>
        <linearGradient :id="gradientId" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" :stop-color="color" stop-opacity="0.25" />
          <stop offset="100%" :stop-color="color" stop-opacity="0.02" />
        </linearGradient>
      </defs>
      <polygon :points="areaPoints" :fill="`url(#${gradientId})`" />
      <!-- Loss line -->
      <polyline
        :points="linePoints"
        fill="none"
        :stroke="color"
        stroke-width="1.5"
        stroke-linejoin="round"
        stroke-linecap="round"
      />
    </svg>
    <div v-else class="loss-chart-empty text-medium-emphasis text-caption">
      {{ emptyLabel }}
    </div>
    <div v-if="label && data.length > 1" class="loss-chart-label text-caption text-medium-emphasis">
      {{ label }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted, onUnmounted } from 'vue'

const props = withDefaults(defineProps<{
  data: { step: number; value: number }[]
  color?: string
  label?: string
  height?: number
  emptyLabel?: string
}>(), {
  color: '#C75B1A',
  label: '',
  height: 200,
  emptyLabel: 'Waiting for data...',
})

const container = ref<HTMLElement>()
const width = ref(600)
const gradientId = `lg-${Math.random().toString(36).slice(2, 8)}`

// Metrics commits re-map the whole history into `data`; plotting every
// point means rebuilding a multi-thousand-point SVG string and re-rastering
// it per commit. LTTB keeps the curve's shape (including spikes) with a
// bounded point budget.
const PLOT_POINT_BUDGET = 400
const PLOT_DOWNSAMPLE_ABOVE = 600

interface ChartPoint {
  step: number
  value: number
}

function lttb(data: ChartPoint[], threshold: number): ChartPoint[] {
  const n = data.length
  if (n <= threshold) return data
  const sampled: ChartPoint[] = [data[0]]
  const bucketSize = (n - 2) / (threshold - 2)
  let anchor = data[0]
  let anchorIndex = 0
  for (let i = 0; i < threshold - 2; i++) {
    const curStart = Math.floor(i * bucketSize) + 1
    const curEnd = Math.floor((i + 1) * bucketSize) + 1
    const nextStart = curEnd
    const nextEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, n)
    let avgIndex = 0
    let avgValue = 0
    for (let j = nextStart; j < nextEnd; j++) {
      avgIndex += j
      avgValue += data[j].value
    }
    const avgN = nextEnd - nextStart
    const avgX = avgN > 0 ? avgIndex / avgN : (nextStart + nextEnd) / 2
    const avgY = avgN > 0 ? avgValue / avgN : 0
    let bestIndex = curStart
    let bestScore = -1
    for (let j = curStart; j < curEnd; j++) {
      const dx = avgX - anchorIndex
      const dy = avgY - anchor.value
      const score = ((j - anchorIndex) * dy - (data[j].value - anchor.value) * dx) ** 2
      if (score > bestScore) {
        bestScore = score
        bestIndex = j
      }
    }
    anchor = data[bestIndex]
    anchorIndex = bestIndex
    sampled.push(anchor)
  }
  sampled.push(data[n - 1])
  return sampled
}

const paddingLeft = 50
const paddingRight = 16
const paddingTop = 12
const paddingBottom = 20

let ro: ResizeObserver | null = null
onMounted(() => {
  if (container.value) {
    ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        width.value = e.contentRect.width
      }
    })
    ro.observe(container.value)
  }
})
onUnmounted(() => ro?.disconnect())

// Domain from the FULL series so LTTB's point selection can't clip the
// true min/max off the y-axis.
const yDomain = computed(() => {
  if (props.data.length === 0) return { min: 0, max: 1 }
  let min = Infinity
  let max = -Infinity
  for (const d of props.data) {
    if (d.value < min) min = d.value
    if (d.value > max) max = d.value
  }
  if (min === max) {
    // Degenerate (constant) series: expand symmetrically around the value
    // by a magnitude-relative margin, not a hardcoded ±0.001. A hardcoded
    // absolute margin collapses a tiny value like lr=1e-5 onto the baseline.
    const mag = Math.abs(min) || 1
    min -= mag * 0.1
    max += mag * 0.1
  }
  const pad = (max - min) * 0.1
  return { min: min - pad, max: max + pad }
})

const plotData = computed<ChartPoint[]>(() => {
  if (props.data.length > PLOT_DOWNSAMPLE_ABOVE) {
    return lttb(props.data, PLOT_POINT_BUDGET)
  }
  return props.data
})

const plotWidth = computed(() => width.value - paddingLeft - paddingRight)
const plotHeight = computed(() => props.height - paddingTop - paddingBottom)

function xFor(i: number): number {
  const count = plotData.value.length
  if (count <= 1) return paddingLeft
  return paddingLeft + (i / (count - 1)) * plotWidth.value
}

function yFor(val: number): number {
  const { min, max } = yDomain.value
  const ratio = (val - min) / (max - min)
  return paddingTop + plotHeight.value * (1 - ratio)
}

const linePoints = computed(() =>
  plotData.value.map((d, i) => `${xFor(i).toFixed(1)},${yFor(d.value).toFixed(1)}`).join(' ')
)

const areaPoints = computed(() => {
  if (plotData.value.length === 0) return ''
  const bottom = paddingTop + plotHeight.value
  const first = `${xFor(0).toFixed(1)},${bottom}`
  const last = `${xFor(plotData.value.length - 1).toFixed(1)},${bottom}`
  return `${first} ${linePoints.value} ${last}`
})

const yTicks = computed(() => {
  const { min, max } = yDomain.value
  const range = max - min
  const step = niceStep(range, 4)
  const start = Math.ceil(min / step) * step
  const ticks: { y: number; label: string }[] = []
  for (let v = start; v <= max; v += step) {
    // Format by magnitude so both loss (~0.08) and lr (~1e-5) read cleanly
    // without manual per-series config.
    const av = Math.abs(v)
    let label: string
    if (av === 0) label = '0'
    else if (av >= 1) label = v.toFixed(3)
    else if (av >= 0.001) label = v.toFixed(5)
    else label = v.toExponential(1)
    ticks.push({ y: yFor(v), label })
  }
  return ticks
})

const gridYs = computed(() => yTicks.value.map((t) => t.y))

function niceStep(range: number, targetTicks: number): number {
  const rough = range / targetTicks
  const mag = Math.pow(10, Math.floor(Math.log10(rough)))
  const norm = rough / mag
  let nice: number
  if (norm <= 1) nice = 1
  else if (norm <= 2) nice = 2
  else if (norm <= 5) nice = 5
  else nice = 10
  return nice * mag
}
</script>

<style scoped>
.loss-chart-container {
  position: relative;
  width: 100%;
}
.loss-chart-svg {
  width: 100%;
  height: v-bind(height + 'px');
  display: block;
}
.loss-chart-empty {
  height: v-bind(height + 'px');
  display: flex;
  align-items: center;
  justify-content: center;
}
.loss-chart-label {
  position: absolute;
  top: 4px;
  right: 8px;
}
</style>
