<template>
  <v-container fluid class="pa-4 distill-page">
    <div class="d-flex align-center mb-1">
      <div class="text-h5">{{ t('distTitle') }}</div>
      <v-spacer />
      <v-btn
        v-if="!guideOpen"
        variant="tonal"
        size="small"
        prepend-icon="mdi-book-open-variant"
        @click="guideOpen = true"
      >
        {{ t('distGuide') }}
      </v-btn>
    </div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('distSubtitle') }}</div>

    <!-- Method selector -->
    <v-card variant="tonal" class="mb-4">
      <v-card-title class="text-subtitle-1">
        <v-icon icon="mdi-flask" class="mr-2" />
        {{ t('distMethod') }}
        <v-chip
          v-if="selectedMethodMeta?.experimental"
          size="x-small"
          color="warning"
          variant="tonal"
          class="ml-2"
        >
          {{ t('distExperimental') }}
        </v-chip>
      </v-card-title>
      <v-card-text>
        <v-row>
          <v-col cols="12" md="4">
            <v-select
              v-model="selectedMethod"
              :items="methods"
              item-title="label"
              item-value="key"
              :label="t('distMethod')"
              variant="outlined"
              density="compact"
              hide-details
              @update:model-value="loadConfig"
            />
          </v-col>
          <v-col cols="12" md="8" class="d-flex align-center">
            <v-alert
              v-if="selectedMethodMeta?.experimental"
              type="warning"
              variant="tonal"
              density="compact"
              class="w-100"
              icon="mdi-flask-outline"
            >
              {{ t('distExperimentalWarning') }}
            </v-alert>
          </v-col>
        </v-row>
      </v-card-text>
    </v-card>

    <!-- Empty state -->
    <div
      v-if="!config.sections.length"
      class="d-flex align-center justify-center flex-grow-1 text-medium-emphasis"
      style="min-height: 240px"
    >
      <div class="text-center">
        <v-icon icon="mdi-flask-empty-outline" size="64" class="mb-4" />
        <div class="text-h6">{{ t('distSelectHint') }}</div>
      </div>
    </div>

    <!-- Two-column form + guide -->
    <v-row v-else>
      <v-col cols="12" :lg="guideOpen ? 8 : 12" class="form-column">
        <v-expansion-panels v-model="openPanels" multiple variant="accordion">
          <v-expansion-panel
            v-for="section in config.sections"
            :key="section.name"
            elevation="0"
          >
            <v-expansion-panel-title>
              <v-icon icon="mdi-cog-outline" class="mr-2" size="small" />
              {{ section.name === 'root' ? t('distGeneral') : section.name }}
              <v-chip size="x-small" class="ml-2" variant="outlined">{{ section.fields.length }}</v-chip>
            </v-expansion-panel-title>
            <v-expansion-panel-text>
              <v-row dense>
                <v-col
                  v-for="field in section.fields"
                  :key="field.key"
                  cols="12"
                  :md="isWideField(field) ? 12 : 6"
                >
                  <DistillField :field="field" />
                </v-col>
              </v-row>
            </v-expansion-panel-text>
          </v-expansion-panel>
        </v-expansion-panels>

        <!-- Action bar -->
        <v-card variant="tonal" class="mt-4 pa-3">
          <div class="d-flex flex-wrap align-center ga-3">
            <v-btn
              :color="dirty ? 'warning' : 'primary'"
              prepend-icon="mdi-content-save"
              :loading="saving"
              :disabled="!dirty"
              @click="saveConfig"
            >
              {{ t('distSave') }}{{ dirty ? ' *' : '' }}
            </v-btn>
            <v-btn
              variant="outlined"
              prepend-icon="mdi-refresh"
              :disabled="loading"
              @click="loadConfig"
            >
              {{ t('distReload') }}
            </v-btn>
            <v-divider vertical class="mx-1 hidden-sm-and-down" />
            <v-btn
              color="success"
              prepend-icon="mdi-play-circle"
              :loading="isRunning || trainingLaunching"
              :disabled="isRunning || !selectedMethod"
              @click="startTraining"
            >
              {{ t('distTrain') }}
            </v-btn>
            <v-btn
              color="error"
              prepend-icon="mdi-stop"
              variant="outlined"
              :disabled="!isRunning"
              @click="stopTraining"
            >
              {{ t('distStop') }}
            </v-btn>
            <v-spacer />
            <v-btn
              v-if="!guideOpen"
              variant="text"
              size="small"
              prepend-icon="mdi-book-open-variant"
              @click="guideOpen = true"
            >
              {{ t('distGuide') }}
            </v-btn>
          </div>
        </v-card>
      </v-col>

      <!-- Right: sticky help panel -->
      <v-col v-if="guideOpen" cols="12" lg="4">
        <div class="help-panel-sticky">
          <v-card variant="outlined" class="guide-card">
            <v-card-title class="text-subtitle-1 d-flex align-center">
              <v-icon icon="mdi-book-open-variant" class="mr-2" />
              {{ t('distGuide') }}
              <v-spacer />
              <v-btn icon="mdi-close" size="small" variant="text" @click="guideOpen = false" />
            </v-card-title>
            <v-divider />
            <v-card-text class="guide-html" v-html="guideHtml" />
          </v-card>
        </div>
      </v-col>
    </v-row>

    <!-- Active tasks -->
    <template v-if="distillTasks.length > 0 || config.sections.length > 0">
      <v-divider class="my-4" />
      <div class="d-flex align-center mb-2">
        <div class="text-subtitle-1">{{ t('distActiveTasks') }}</div>
        <v-spacer />
        <v-btn variant="text" size="small" prepend-icon="mdi-refresh" @click="taskStore.fetchTasks()">
          {{ t('distRefresh') }}
        </v-btn>
      </div>
      <v-list v-if="distillTasks.length > 0" density="compact">
        <v-list-item
          v-for="task in distillTasks"
          :key="task.task_id"
          :title="task.command"
          :subtitle="`${t('taskState')}: ${task.state} | PID: ${task.pid ?? '—'}`"
        >
          <template #append>
            <v-chip size="small" :color="stateColor(task.state)" variant="tonal">{{ task.state }}</v-chip>
            <v-btn
              v-if="task.state === 'running' || task.state === 'pending'"
              icon="mdi-stop"
              size="small"
              variant="text"
              color="error"
              @click="taskStore.cancelTask(task.task_id)"
            />
          </template>
        </v-list-item>
      </v-list>
      <div v-else class="text-medium-emphasis text-body-2">{{ t('distNoTasks') }}</div>
    </template>
  </v-container>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useAppStore } from '../stores/app'
import { useI18n } from '../composables/useI18n'
import DistillField from '../components/DistillField.vue'

const { t } = useI18n()
const taskStore = useTaskStore()
const notify = useNotifyStore()
const appStore = useAppStore()

taskStore.fetchTasks()

export interface DistillMethodSummary {
  key: string
  label: string
  config_path: string
  task_command: string
  experimental?: boolean
}

export interface DistillField {
  key: string
  value: any
  type: string
  comment: string
}

export interface DistillSection {
  name: string
  fields: DistillField[]
}

export interface DistillConfigResponse {
  method: string
  sections: DistillSection[]
}

const methods = ref<DistillMethodSummary[]>([])
const selectedMethod = ref('')
const config = reactive<DistillConfigResponse>({ method: '', sections: [] })
const originalHash = ref('')
const saving = ref(false)
const loading = ref(false)
const guideOpen = ref(false)
const guideHtml = ref('')
const trainingLaunching = ref(false)
const openPanels = ref<number[]>([])

const selectedMethodMeta = computed(() =>
  methods.value.find(m => m.key === selectedMethod.value)
)

const isRunning = computed(() => {
  const method = selectedMethodMeta.value
  if (!method) return false
  return taskStore.tasks.some(
    tp => tp.command === method.task_command && tp.state === 'running'
  )
})

const distillTasks = computed(() => {
  const cmds = new Set(methods.value.map(m => m.task_command))
  return taskStore.tasks.filter(tp => cmds.has(tp.command))
})

const currentHash = computed(() => {
  return JSON.stringify(config.sections.map(s => ({ name: s.name, fields: s.fields.map(f => ({ key: f.key, value: f.value })) })))
})

const dirty = computed(() => currentHash.value !== originalHash.value)

function isWideField(field: DistillField) {
  if (field.type === 'list') return true
  if (typeof field.value === 'string' && String(field.value).length > 40) return true
  return false
}

async function loadMethods() {
  const res = await fetch('/api/distill/methods')
  methods.value = await res.json()
  if (methods.value.length && !selectedMethod.value) {
    selectedMethod.value = methods.value[0].key
    await loadConfig()
  }
}

async function loadConfig() {
  if (!selectedMethod.value) {
    config.sections = []
    guideHtml.value = ''
    return
  }
  loading.value = true
  try {
    const res = await fetch(`/api/distill/config?method=${encodeURIComponent(selectedMethod.value)}`)
    const data = await res.json()
    Object.assign(config, data)
    originalHash.value = JSON.stringify(data.sections.map((s: DistillSection) => ({
      name: s.name,
      fields: s.fields.map(f => ({ key: f.key, value: f.value })),
    })))
    openPanels.value = data.sections.map((_: unknown, i: number) => i)
    await loadGuide()
  } finally {
    loading.value = false
  }
}

async function loadGuide() {
  if (!selectedMethod.value) {
    guideHtml.value = ''
    return
  }
  try {
    const res = await fetch(
      `/api/distill/guide?method=${encodeURIComponent(selectedMethod.value)}&lang=${encodeURIComponent(appStore.language)}`
    )
    const data = await res.json()
    guideHtml.value = data.html || ''
  } catch {
    guideHtml.value = ''
  }
}

async function saveConfig() {
  saving.value = true
  try {
    const updates: Record<string, Record<string, any>> = {}
    for (const section of config.sections) {
      updates[section.name] = {}
      for (const field of section.fields) {
        updates[section.name][field.key] = field.value
      }
    }
    const res = await fetch(`/api/distill/config?method=${encodeURIComponent(selectedMethod.value)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates }),
    })
    if (res.ok) {
      notify.show(t('distSaved'), 'success')
      await loadConfig()
    } else {
      notify.show(t('distSaveFailed'), 'error')
    }
  } finally {
    saving.value = false
  }
}

async function startTraining() {
  const method = selectedMethodMeta.value
  if (!method) return
  if (dirty.value) await saveConfig()
  trainingLaunching.value = true
  try {
    const taskId = await taskStore.startTask(method.task_command)
    if (taskId) {
      notify.show(t('notifyTaskStarted', { command: method.task_command }), 'success')
    } else {
      notify.show(t('notifyTaskStartFailed', { command: method.task_command }), 'error')
    }
  } finally {
    trainingLaunching.value = false
  }
}

function stopTraining() {
  const method = selectedMethodMeta.value
  if (!method) return
  const task = taskStore.tasks.find(
    tp => tp.command === method.task_command && tp.state === 'running'
  )
  if (task) taskStore.cancelTask(task.task_id)
}

function stateColor(state: string) {
  if (state === 'running' || state === 'stopping') return 'info'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}

onMounted(loadMethods)

watch(() => appStore.language, () => {
  if (selectedMethod.value) loadGuide()
})
</script>

<style scoped>
.distill-page {
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
}

.form-column {
  min-height: 0;
}

.help-panel-sticky {
  position: sticky;
  top: 16px;
}

.guide-card {
  max-height: calc(100vh - 120px);
  overflow-y: auto;
  border-left: 3px solid var(--forge-ember);
}

.guide-html :deep(h1),
.guide-html :deep(h2),
.guide-html :deep(h3) {
  margin-top: 0.8em;
  margin-bottom: 0.4em;
}

.guide-html :deep(p) {
  margin-bottom: 0.5em;
  line-height: 1.6;
}

.guide-html :deep(code) {
  background: var(--bg-deep);
  color: var(--forge-amber);
  padding: 1px 5px;
  border-radius: 3px;
  font-family: var(--font-mono);
  font-size: 0.85em;
}

.guide-html :deep(pre) {
  background: var(--bg-deep);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 12px;
}

/* Section panels hover */
:deep(.v-card) {
  transition: border-color 0.2s, box-shadow 0.2s;
}
:deep(.v-card:hover) {
  border-color: var(--border-default);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

:deep(.v-expansion-panel-title) {
  border-left: 3px solid transparent;
  transition: border-color 0.2s;
}
:deep(.v-expansion-panel--active .v-expansion-panel-title) {
  border-left-color: var(--forge-ember);
}
</style>
