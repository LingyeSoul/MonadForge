<template>
  <v-container fluid class="pa-4 d-flex flex-column" style="flex: 1 1 0; min-height: 0; overflow: hidden;">
    <!-- Header -->
    <div class="workspace-heading dashboard-heading">
      <div>
        <h1>{{ t('dashTitle') }}</h1>
        <p>Anima / {{ t('dashMetrics') }}</p>
      </div>
      <v-spacer />
      <v-btn
        v-if="wandbRunUrl"
        :href="wandbRunUrl"
        target="_blank"
        variant="tonal"
        size="small"
        prepend-icon="mdi-chart-box"
        class="mr-2"
      >
        {{ t('dashWandbBoard') }}
      </v-btn>
      <v-select
        v-if="trainingTasks.length > 0"
        v-model="selectedTaskId"
        :items="trainingTaskItems"
        item-title="label"
        item-value="task_id"
        density="compact"
        variant="outlined"
        hide-details
        style="max-width: 260px;"
        :label="t('dashSelectTask')"
      />
    </div>

    <!-- No training task state -->
    <div v-if="trainingTasks.length === 0" class="dashboard-idle">
      <section v-if="sysCards.length" class="system-overview">
        <h2 class="workspace-section-title">{{ t('dashSysMon') }}</h2>
        <div class="system-grid">
          <div v-for="card in sysCards" :key="card.key" class="system-metric">
            <div class="metric-label">{{ card.label }}</div>
            <div class="metric-value" :style="{ color: card.color }">{{ card.value }}</div>
            <v-progress-linear v-if="card.percent !== undefined" :model-value="card.percent"
              :color="card.color" height="3" class="mt-3" />
          </div>
        </div>
      </section>
      <section class="workspace-empty dashboard-empty">
        <div class="workspace-empty__mark">
          <v-icon :icon="taskStore.taskListError ? 'mdi-cloud-alert-outline' : 'mdi-chart-timeline-variant'" size="32" />
        </div>
        <h2>{{ t(taskStore.taskListError ? 'taskLoadFailed' : 'dashNoTask') }}</h2>
        <v-btn v-if="taskStore.taskListError" variant="outlined" prepend-icon="mdi-refresh" @click="taskStore.fetchTasks()">
          {{ t('ppRefresh') }}
        </v-btn>
        <v-btn v-else to="/config" color="primary" prepend-icon="mdi-plus">{{ t('dashOpenConfig') }}</v-btn>
      </section>
    </div>

    <!-- Dashboard content -->
    <div v-else class="dashboard-content">
      <!-- Row 1: Progress + Metrics -->
      <v-row density="comfortable" class="training-summary" style="flex: 0 0 auto;">
        <!-- Progress Ring -->
        <v-col cols="12" md="4">
          <section class="progress-overview">
            <h2 class="workspace-section-title">{{ t('dashProgress') }}</h2>
            <div class="progress-overview__body">
              <v-progress-circular
                :model-value="progressPercent"
                :size="112"
                :width="6"
                color="primary"
              >
                <div class="text-center">
                  <div class="progress-value">{{ progressPercent.toFixed(0) }}%</div>
                  <div class="text-caption text-medium-emphasis">
                    {{ m.total_steps > 0 ? `${m.step} / ${m.total_steps}` : '— / —' }}
                  </div>
                </div>
              </v-progress-circular>

              <div class="progress-details">
                <div class="text-body-2">
                  {{ t('dashEpoch') }}: <strong>{{ m.epoch }}</strong>
                  <template v-if="m.total_epochs"> / {{ m.total_epochs }}</template>
                </div>
                <div v-if="m.elapsed" class="text-caption text-medium-emphasis mt-1">
                  {{ t('dashElapsed') }}: {{ m.elapsed }}
                  <template v-if="m.eta"> &middot; {{ t('dashEta') }}: {{ m.eta }}</template>
                </div>
                <div v-if="m.speed" class="text-caption text-medium-emphasis">
                  {{ m.speed }}
                </div>
              </div>
            </div>
          </section>
        </v-col>

        <!-- Key Metrics Grid -->
        <v-col cols="12" md="8">
          <section class="metrics-overview">
            <h2 class="workspace-section-title">{{ t('dashMetrics') }}</h2>
            <div class="metrics-grid">
                <div v-for="card in metricCards" :key="card.key" class="training-metric">
                  <div class="metric-label">{{ card.label }}</div>
                  <div class="metric-value" :style="{ color: card.color }">
                    {{ card.value }}
                  </div>
                </div>
            </div>
          </section>
        </v-col>
      </v-row>

      <!-- Row 1.5: System Monitoring -->
      <section v-if="sysCards.length" class="system-overview" style="flex: 0 0 auto;">
        <h2 class="workspace-section-title">{{ t('dashSysMon') }}</h2>
        <div class="system-grid">
            <div v-for="card in sysCards" :key="card.key" class="system-metric">
              <div class="metric-label">{{ card.label }}</div>
              <div class="metric-value" :style="{ color: card.color }">
                {{ card.value }}
              </div>
              <v-progress-linear
                v-if="card.percent !== undefined"
                :model-value="card.percent"
                :color="card.color"
                height="3"
                class="mt-3"
              />
            </div>
        </div>
      </section>

      <!-- Row 1.7: Sample previews. Renders whenever sampling is enabled so the
           user gets an explicit "waiting/disabled" state instead of an empty
           dashboard that reads as broken. -->
      <SampleGallery
        v-if="selectedTaskId && m.sampling_enabled"
        :samples="m.sample_history"
        :task-id="selectedTaskId"
      />
      <v-card
        v-else-if="selectedTaskId"
        variant="tonal"
        class="pa-4"
        style="flex: 0 0 auto;"
      >
        <div class="d-flex align-center">
          <v-icon icon="mdi-image-off-outline" size="small" class="mr-2" />
          <div class="text-body-2 text-medium-emphasis">
            {{ t('dashSamplingDisabled') }}
          </div>
        </div>
      </v-card>

      <!-- Row 2: Loss Curve + LR Curve -->
      <v-row density="comfortable" style="flex: 0 0 auto;">
        <v-col cols="12" md="6">
          <v-card variant="tonal" class="pa-4 h-100">
            <div class="d-flex align-center mb-2">
              <div class="text-subtitle-2">{{ t('dashLossCurve') }}</div>
              <v-spacer />
              <v-chip v-if="m.loss_history.length > 0" size="x-small" variant="outlined">
                {{ m.loss_history.length }} {{ t('dashPoints') }}
              </v-chip>
            </div>
            <LossChart
              :data="lossChartData"
              color="var(--forge-ember)"
              :height="220"
              :empty-label="t('dashWaitingLoss')"
            />
          </v-card>
        </v-col>
        <v-col cols="12" md="6">
          <v-card variant="tonal" class="pa-4 h-100">
            <div class="d-flex align-center mb-2">
              <div class="text-subtitle-2">{{ t('dashLrCurve') }}</div>
              <v-spacer />
              <v-chip v-if="m.lr_history.length > 0" size="x-small" variant="outlined">
                {{ m.lr_history.length }} {{ t('dashPoints') }}
              </v-chip>
            </div>
            <LossChart
              :data="lrChartData"
              color="var(--forge-amber)"
              :height="220"
              :empty-label="t('dashWaitingLr')"
            />
          </v-card>
        </v-col>
      </v-row>

      <!-- Row 3: Events Timeline -->
      <v-card v-if="m.events.length > 0" variant="tonal" class="pa-4" style="flex: 0 0 auto;">
        <div class="text-subtitle-2 mb-2">{{ t('dashEvents') }}</div>
        <div class="events-list" style="max-height: 200px; overflow-y: auto;">
          <div
            v-for="(evt, i) in reversedEvents"
            :key="i"
            class="d-flex align-center py-1"
          >
            <v-icon
              :icon="evt.type === 'epoch' ? 'mdi-arrow-right-bold-circle' : 'mdi-content-save'"
              size="small"
              :color="evt.type === 'epoch' ? 'primary' : 'success'"
              class="mr-2"
            />
            <span class="text-body-2">
              <template v-if="evt.type === 'epoch'">
                {{ t('dashEventEpoch', { epoch: evt.epoch, total: evt.total_epochs ?? '?' }) }}
              </template>
              <template v-else>
                {{ t('dashEventCheckpoint') }}
              </template>
            </span>
            <v-spacer />
            <span class="text-caption text-medium-emphasis ml-2">
              {{ evt.elapsed || '' }}
            </span>
          </div>
        </div>
      </v-card>

      <!-- Row 4: Live Log (fixed height, scrolls internally). Shares the
           dashboard's single WS via :external instead of opening its own. -->
      <v-card v-if="selectedTaskId" variant="tonal" style="flex: 0 0 320px; display: flex; flex-direction: column; overflow: hidden;">
        <v-card-title class="text-subtitle-2 d-flex align-center pa-3 pb-0">
          <v-icon icon="mdi-console-line" size="small" class="mr-2" />
          {{ t('dashLiveLog') }}
        </v-card-title>
        <v-card-text class="pa-2 d-flex flex-column" style="flex: 1 1 0; min-height: 0; overflow: hidden;">
          <LogStream :task-id="selectedTaskId" :external="streamHandle" />
        </v-card-text>
      </v-card>
      <v-card v-else variant="tonal" style="flex: 0 0 auto;">
        <div class="d-flex align-center pa-4 text-body-2 text-medium-emphasis">
          <v-icon icon="mdi-console-line" size="small" class="mr-2" />
          {{ t('dashSelectTask') }}
        </div>
      </v-card>
    </div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, watch, shallowRef, onMounted, onUnmounted } from 'vue'
import { useTrainingStore } from '../stores/training'
import { useTaskStore } from '../stores/task'
import { useTrainingStream } from '../composables/useTrainingStream'
import { useI18n } from '../composables/useI18n'
import LossChart from '../components/LossChart.vue'
import LogStream from '../components/LogStream.vue'
import SampleGallery from '../components/SampleGallery.vue'

const { t } = useI18n()
const trainingStore = useTrainingStore()
const taskStore = useTaskStore()
const m = computed(() => trainingStore.metrics)

const wandbRunUrl = computed(() => m.value.wandb_run_url || null)

// Hardware stats (polled independently)
const hw = ref<Record<string, any>>({})
async function fetchHwStats() {
  try {
    const res = await fetch('/api/system/hw-stats')
    if (res.ok) hw.value = await res.json()
  } catch { /* ignore */ }
}

const selectedTaskId = ref('')
const streamHandle = shallowRef<ReturnType<typeof useTrainingStream> | null>(null)

const trainingTasks = computed(() =>
  taskStore.tasks.filter((t) => t.category === 'training')
)

const trainingTaskItems = computed(() =>
  trainingTasks.value.map((t) => ({
    task_id: t.task_id,
    label: `${t.command} (${t.task_id.slice(0, 8)})`,
  }))
)

const progressPercent = computed(() => {
  if (m.value.total_steps <= 0) return 0
  return (m.value.step / m.value.total_steps) * 100
})

const lossChartData = computed(() =>
  m.value.step_history.map((step, i) => ({ step, value: m.value.loss_history[i] }))
)

const lrChartData = computed(() =>
  m.value.step_history.map((step, i) => ({ step, value: m.value.lr_history[i] }))
)

const reversedEvents = computed(() => [...m.value.events].reverse())

const metricCards = computed(() => {
  const cards: { key: string; label: string; value: string; color?: string }[] = [
    { key: 'loss', label: t('dashLoss'), value: m.value.avr_loss > 0 ? m.value.avr_loss.toFixed(5) : '—', color: 'var(--forge-ember)' },
    { key: 'lr', label: t('dashLearningRate'), value: m.value.lr > 0 ? m.value.lr.toExponential(2) : '—', color: 'var(--forge-amber)' },
    { key: 'speed', label: t('dashSpeed'), value: m.value.speed || '—' },
    { key: 'step', label: t('dashStep'), value: m.value.total_steps > 0 ? `${m.value.step}/${m.value.total_steps}` : '—' },
  ]
  if (m.value.router_h !== null && m.value.router_h !== undefined) {
    cards.push({ key: 'router_h', label: t('dashRouterH'), value: m.value.router_h.toFixed(3), color: 'var(--forge-amber)' })
  }
  if (m.value.avg_key_norm !== null && m.value.avg_key_norm !== undefined) {
    cards.push({ key: 'avg_key_norm', label: t('dashAvgKeyNorm'), value: m.value.avg_key_norm.toFixed(4) })
  }
  if (m.value.keys_scaled !== null && m.value.keys_scaled !== undefined) {
    cards.push({ key: 'keys_scaled', label: t('dashKeysScaled'), value: String(m.value.keys_scaled) })
  }
  return cards
})

const sysCards = computed(() => {
  const cards: { key: string; label: string; value: string; color?: string; percent?: number }[] = []
  const v = hw.value
  if (!v || Object.keys(v).length === 0) return cards

  if (v.gpu_util_percent !== undefined) {
    cards.push({ key: 'gpu_util', label: t('dashGpuUtil'), value: `${v.gpu_util_percent}%`, color: 'rgb(var(--v-theme-success))', percent: v.gpu_util_percent })
  }
  if (v.gpu_mem_total_gb !== undefined && v.gpu_mem_total_gb > 0) {
    const pct = v.gpu_mem_total_gb > 0 ? Math.round((v.gpu_mem_used_gb / v.gpu_mem_total_gb) * 100) : 0
    let memLabel = `${v.gpu_mem_used_gb}/${v.gpu_mem_total_gb} GB`
    if (v.gpu_mem_reserved_gb !== undefined && v.gpu_mem_reserved_gb !== v.gpu_mem_used_gb) {
      memLabel += ` (R: ${v.gpu_mem_reserved_gb})`
    }
    cards.push({ key: 'gpu_mem', label: t('dashGpuMem'), value: memLabel, color: 'rgb(var(--v-theme-warning))', percent: pct })
  }
  if (v.gpu_temp_c !== undefined) {
    const color = v.gpu_temp_c >= 80 ? 'rgb(var(--v-theme-error))' : v.gpu_temp_c >= 65 ? 'rgb(var(--v-theme-warning))' : 'rgb(var(--v-theme-success))'
    cards.push({ key: 'gpu_temp', label: t('dashGpuTemp'), value: `${v.gpu_temp_c}°C`, color })
  }
  if (v.cpu_percent !== undefined) {
    cards.push({ key: 'cpu', label: t('dashCpu'), value: `${v.cpu_percent}%`, color: 'rgb(var(--v-theme-info))', percent: v.cpu_percent })
  }
  if (v.mem_total_gb) {
    cards.push({ key: 'mem', label: t('dashMem'), value: `${v.mem_used_gb}/${v.mem_total_gb} GB`, color: 'var(--forge-amber)', percent: v.mem_percent })
  }
  return cards
})

// Auto-select the first running training task
onMounted(async () => {
  await taskStore.fetchTasks()
  autoSelect()
})

function autoSelect() {
  if (!trainingTasks.value.some(task => task.task_id === selectedTaskId.value)) {
    const preferred = trainingTasks.value.find(task => task.state === 'running') ?? trainingTasks.value[0]
    selectedTaskId.value = preferred?.task_id ?? ''
  }
}

// Connect/disconnect WS when selected task changes
watch(selectedTaskId, (id) => {
  streamHandle.value?.disconnect()
  trainingStore.reset()
  streamHandle.value = id ? useTrainingStream(id) : null
  if (streamHandle.value) void streamHandle.value.connect()
})

// Periodically refresh task list + hw stats
let refreshTimer = 0
let hwTimer = 0
onMounted(() => {
  fetchHwStats()
  hwTimer = window.setInterval(fetchHwStats, 3000)
  refreshTimer = window.setInterval(async () => {
    await taskStore.fetchTasks()
    autoSelect()
  }, 5000)
})
onUnmounted(() => {
  streamHandle.value?.disconnect()
  if (refreshTimer) clearInterval(refreshTimer)
  if (hwTimer) clearInterval(hwTimer)
})
</script>

<style scoped>
.dashboard-heading { flex-shrink: 0; }
.dashboard-content { display: flex; flex-direction: column; flex: 1 1 0; min-height: 0; overflow-y: auto; overflow-x: hidden; gap: 24px; }
.dashboard-idle { display: flex; flex-direction: column; flex: 1; min-height: 0; overflow-y: auto; }
.dashboard-empty { flex: 1; min-height: 280px; }
.training-summary { border-top: 1px solid var(--border-subtle); border-bottom: 1px solid var(--border-subtle); margin: 0; padding: 20px 0; }
.progress-overview, .metrics-overview { height: 100%; padding: 0 16px; }
.progress-overview { border-right: 1px solid var(--border-subtle); padding-left: 0; }
.progress-overview__body { display: flex; align-items: center; gap: 20px; margin-top: 20px; flex-wrap: wrap; }
.progress-value { font: 500 26px var(--font-mono); }
.progress-details { min-width: 0; color: var(--text-secondary); line-height: 1.7; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 150px), 1fr)); grid-auto-flow: dense; margin-top: 12px; }
.training-metric { min-width: 0; padding: 12px 12px 12px 0; }
.metric-label { color: var(--text-secondary); font-size: 12px; margin-bottom: 8px; }
.metric-value { font: 500 20px/1.6 var(--font-mono); overflow-wrap: anywhere; }
.system-overview { padding: 20px 0; border-bottom: 1px solid var(--border-subtle); }
.system-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 155px), 1fr)); grid-auto-flow: dense; gap: 24px; margin-top: 20px; }
.system-metric { min-width: 0; }
.system-metric .metric-value { font-size: 16px; }
@media (max-width: 959px) {
  .progress-overview { border-right: 0; border-bottom: 1px solid var(--border-subtle); padding-bottom: 20px; }
  .metrics-overview { padding: 20px 0 0; }
}
@media (max-width: 599px) {
  .dashboard-heading { gap: 12px; }
  .dashboard-heading :deep(.v-select) { flex-basis: 100%; max-width: none !important; }
  .system-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
