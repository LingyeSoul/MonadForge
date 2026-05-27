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

const yDomain = computed(() => {
  if (props.data.length === 0) return { min: 0, max: 1 }
  const vals = props.data.map((d) => d.value)
  let min = Math.min(...vals)
  let max = Math.max(...vals)
  if (min === max) {
    min -= 0.001
    max += 0.001
  }
  const pad = (max - min) * 0.1
  return { min: min - pad, max: max + pad }
})

const plotWidth = computed(() => width.value - paddingLeft - paddingRight)
const plotHeight = computed(() => props.height - paddingTop - paddingBottom)

function xFor(i: number): number {
  if (props.data.length <= 1) return paddingLeft
  return paddingLeft + (i / (props.data.length - 1)) * plotWidth.value
}

function yFor(val: number): number {
  const { min, max } = yDomain.value
  const ratio = (val - min) / (max - min)
  return paddingTop + plotHeight.value * (1 - ratio)
}

const linePoints = computed(() =>
  props.data.map((d, i) => `${xFor(i).toFixed(1)},${yFor(d.value).toFixed(1)}`).join(' ')
)

const areaPoints = computed(() => {
  if (props.data.length === 0) return ''
  const bottom = paddingTop + plotHeight.value
  const first = `${xFor(0).toFixed(1)},${bottom}`
  const last = `${xFor(props.data.length - 1).toFixed(1)},${bottom}`
  return `${first} ${linePoints.value} ${last}`
})

const yTicks = computed(() => {
  const { min, max } = yDomain.value
  const range = max - min
  const step = niceStep(range, 4)
  const start = Math.ceil(min / step) * step
  const ticks: { y: number; label: string }[] = []
  for (let v = start; v <= max; v += step) {
    ticks.push({ y: yFor(v), label: v < 0.01 ? v.toExponential(1) : v.toFixed(4) })
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
