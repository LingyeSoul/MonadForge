<template>
  <v-container fluid class="pa-4 d-flex flex-column" style="flex: 1 1 auto; min-height: 0;">
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
    <div v-else class="d-flex flex-column gap-3" style="flex: 1 1 auto; min-height: 0; overflow-y: auto;">
      <!-- Row 1: Progress + Metrics -->
      <v-row dense>
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
                    {{ m.step }} / {{ m.total_steps }}
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
                  {{ m.speed }} it/s
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

      <!-- Row 2: Loss Curve -->
      <v-card variant="tonal" class="pa-4">
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

      <!-- Row 3: Events Timeline -->
      <v-card v-if="m.events.length > 0" variant="tonal" class="pa-4">
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

      <!-- Row 4: Live Log (collapsible) -->
      <v-expansion-panels variant="accordion">
        <v-expansion-panel>
          <v-expansion-panel-title>
            <v-icon icon="mdi-console-line" size="small" class="mr-2" />
            {{ t('dashLiveLog') }}
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <LogStream :task-id="selectedTaskId" />
          </v-expansion-panel-text>
        </v-expansion-panel>
      </v-expansion-panels>
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

const reversedEvents = computed(() => [...m.value.events].reverse())

const metricCards = computed(() => {
  const cards: { key: string; label: string; value: string; color?: string }[] = [
    { key: 'loss', label: t('dashLoss'), value: m.value.avr_loss > 0 ? m.value.avr_loss.toFixed(5) : '—', color: '#BB86FC' },
    { key: 'speed', label: t('dashSpeed'), value: m.value.speed ? `${m.value.speed} it/s` : '—' },
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

// Periodically refresh task list
let refreshTimer = 0
onMounted(() => {
  refreshTimer = window.setInterval(async () => {
    await taskStore.fetchTasks()
    autoSelect()
  }, 5000)
})
onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer)
})
</script>

<style scoped>
.metric-card {
  border: 1px solid rgba(255, 255, 255, 0.05);
}
</style>
