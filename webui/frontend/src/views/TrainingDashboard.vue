<template>
  <v-container fluid class="pa-4 d-flex flex-column" style="flex: 1 1 0; min-height: 0; overflow: hidden;">
    <!-- Header -->
    <div class="d-flex align-center mb-1">
      <div class="text-h5">{{ t('dashTitle') }}</div>
      <v-spacer />
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
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('dashSubtitle') }}</div>

    <!-- No training task state -->
    <div v-if="trainingTasks.length === 0" class="d-flex align-center justify-center flex-grow-1">
      <div class="text-center text-medium-emphasis">
        <v-icon icon="mdi-chart-line" size="48" class="mb-2" />
        <div>{{ t('dashNoTask') }}</div>
      </div>
    </div>

    <!-- Dashboard content -->
    <div v-else class="d-flex flex-column" style="flex: 1 1 0; min-height: 0; overflow-y: auto; gap: 20px;">
      <!-- Row 1: Progress + Metrics -->
      <v-row dense style="flex: 0 0 auto;">
        <!-- Progress Ring -->
        <v-col cols="12" md="4">
          <v-card variant="tonal" class="pa-4 h-100">
            <div class="d-flex flex-column align-center justify-center" style="gap: 12px;">
              <v-progress-circular
                :model-value="progressPercent"
                :size="140"
                :width="12"
                color="primary"
              >
                <div class="text-center">
                  <div class="text-h5 font-weight-bold">{{ progressPercent.toFixed(0) }}%</div>
                  <div class="text-caption text-medium-emphasis">
                    {{ m.total_steps > 0 ? `${m.step} / ${m.total_steps}` : '— / —' }}
                  </div>
                </div>
              </v-progress-circular>

              <div class="text-center">
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
          </v-card>
        </v-col>

        <!-- Key Metrics Grid -->
        <v-col cols="12" md="8">
          <v-card variant="tonal" class="pa-4 h-100">
            <div class="text-subtitle-2 mb-3">{{ t('dashMetrics') }}</div>
            <v-row dense>
              <v-col v-for="card in metricCards" :key="card.key" cols="6" sm="4">
                <div class="metric-card pa-3 rounded-lg" style="background: rgba(255,255,255,0.03);">
                  <div class="text-caption text-medium-emphasis mb-1">{{ card.label }}</div>
                  <div class="text-h6 font-weight-medium" :style="{ color: card.color }">
                    {{ card.value }}
                  </div>
                </div>
              </v-col>
            </v-row>
          </v-card>
        </v-col>
      </v-row>

      <!-- Row 1.5: System Monitoring -->
      <v-card variant="tonal" class="pa-4" style="flex: 0 0 auto;">
        <div class="text-subtitle-2 mb-3">{{ t('dashSysMon') }}</div>
        <v-row dense>
          <v-col v-for="card in sysCards" :key="card.key" cols="6" sm="4" md="2">
            <div class="metric-card pa-3 rounded-lg" style="background: rgba(255,255,255,0.03);">
              <div class="text-caption text-medium-emphasis mb-1">{{ card.label }}</div>
              <div class="text-h6 font-weight-medium" :style="{ color: card.color }">
                {{ card.value }}
              </div>
              <v-progress-linear
                v-if="card.percent !== undefined"
                :model-value="card.percent"
                :color="card.color"
                height="4"
                rounded
                class="mt-1"
              />
            </div>
          </v-col>
        </v-row>
      </v-card>

      <!-- Row 2: Loss Curve + LR Curve -->
      <v-row dense style="flex: 0 0 auto;">
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
              color="#BB86FC"
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
              color="#CF6679"
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

      <!-- Row 4: Live Log (fixed height, scrolls internally) -->
      <v-card variant="tonal" style="flex: 0 0 320px; display: flex; flex-direction: column; overflow: hidden;">
        <v-card-title class="text-subtitle-2 d-flex align-center pa-3 pb-0">
          <v-icon icon="mdi-console-line" size="small" class="mr-2" />
          {{ t('dashLiveLog') }}
        </v-card-title>
        <v-card-text class="pa-2 d-flex flex-column" style="flex: 1 1 0; min-height: 0; overflow: hidden;">
          <LogStream :task-id="selectedTaskId" />
        </v-card-text>
      </v-card>
    </div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useTrainingStore } from '../stores/training'
import { useTaskStore } from '../stores/task'
import { useTrainingStream } from '../composables/useTrainingStream'
import { useI18n } from '../composables/useI18n'
import LossChart from '../components/LossChart.vue'
import LogStream from '../components/LogStream.vue'

const { t } = useI18n()
const trainingStore = useTrainingStore()
const taskStore = useTaskStore()
const m = computed(() => trainingStore.metrics)

// Hardware stats (polled independently)
const hw = ref<Record<string, any>>({})
async function fetchHwStats() {
  try {
    const res = await fetch('/api/system/hw-stats')
    if (res.ok) hw.value = await res.json()
  } catch { /* ignore */ }
}

const selectedTaskId = ref('')
let stream: ReturnType<typeof useTrainingStream> | null = null

// Filter tasks that look like training commands
const _TRAINING_COMMANDS = new Set([
  'lora', 'lora-gui', 'exp-postfix', 'exp-chimera', 'exp-ip-adapter', 'exp-easycontrol',
  'exp-soft-tokens', 'distill-mod', 'dcw', 'dcw-train',
])

const trainingTasks = computed(() =>
  taskStore.tasks.filter((t) => _TRAINING_COMMANDS.has(t.command))
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
    { key: 'loss', label: t('dashLoss'), value: m.value.avr_loss > 0 ? m.value.avr_loss.toFixed(5) : '—', color: '#BB86FC' },
    { key: 'lr', label: t('dashLearningRate'), value: m.value.lr > 0 ? m.value.lr.toExponential(2) : '—', color: '#CF6679' },
    { key: 'speed', label: t('dashSpeed'), value: m.value.speed || '—' },
    { key: 'step', label: t('dashStep'), value: m.value.total_steps > 0 ? `${m.value.step}/${m.value.total_steps}` : '—' },
  ]
  if (m.value.router_h !== null && m.value.router_h !== undefined) {
    cards.push({ key: 'router_h', label: t('dashRouterH'), value: m.value.router_h.toFixed(3), color: '#03DAC6' })
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
    cards.push({ key: 'gpu_util', label: t('dashGpuUtil'), value: `${v.gpu_util_percent}%`, color: '#4CAF50', percent: v.gpu_util_percent })
  }
  if (v.gpu_mem_total_gb !== undefined && v.gpu_mem_total_gb > 0) {
    const pct = v.gpu_mem_total_gb > 0 ? Math.round((v.gpu_mem_used_gb / v.gpu_mem_total_gb) * 100) : 0
    let memLabel = `${v.gpu_mem_used_gb}/${v.gpu_mem_total_gb} GB`
    if (v.gpu_mem_reserved_gb !== undefined && v.gpu_mem_reserved_gb !== v.gpu_mem_used_gb) {
      memLabel += ` (R: ${v.gpu_mem_reserved_gb})`
    }
    cards.push({ key: 'gpu_mem', label: t('dashGpuMem'), value: memLabel, color: '#FF9800', percent: pct })
  }
  if (v.gpu_temp_c !== undefined) {
    const color = v.gpu_temp_c >= 80 ? '#F44336' : v.gpu_temp_c >= 65 ? '#FF9800' : '#4CAF50'
    cards.push({ key: 'gpu_temp', label: t('dashGpuTemp'), value: `${v.gpu_temp_c}°C`, color })
  }
  if (v.cpu_percent !== undefined) {
    cards.push({ key: 'cpu', label: t('dashCpu'), value: `${v.cpu_percent}%`, color: '#2196F3', percent: v.cpu_percent })
  }
  if (v.mem_total_gb) {
    cards.push({ key: 'mem', label: t('dashMem'), value: `${v.mem_used_gb}/${v.mem_total_gb} GB`, color: '#9C27B0', percent: v.mem_percent })
  }
  return cards
})

// Auto-select the first running training task
onMounted(async () => {
  await taskStore.fetchTasks()
  autoSelect()
})

function autoSelect() {
  const running = trainingTasks.value.find((t) => t.state === 'running')
  if (running && !selectedTaskId.value) {
    selectedTaskId.value = running.task_id
  }
}

// Connect/disconnect WS when selected task changes
watch(selectedTaskId, (id) => {
  stream?.disconnect()
  trainingStore.reset()
  if (id) {
    stream = useTrainingStream(id)
    stream.connect()
  }
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
  if (refreshTimer) clearInterval(refreshTimer)
  if (hwTimer) clearInterval(hwTimer)
})
</script>

<style scoped>
.metric-card {
  border: 1px solid rgba(255, 255, 255, 0.05);
}
</style>
