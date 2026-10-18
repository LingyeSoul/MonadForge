<template>
  <v-container fluid class="pa-4 distill-page">
    <WorkspaceHeading :title="t('distTitle')" :subtitle="t('distSubtitle')" />
    <v-card variant="tonal" class="mb-4">
      <v-card-text>
        <v-select :model-value="selectedMethod || undefined" :items="methods" item-title="label" item-value="key"
          :label="t('distMethod')" :disabled="busy" variant="outlined" density="compact" hide-details
          data-testid="distill-method" @update:model-value="method => requestChange(() => loadConfig(method))" />
      </v-card-text>
    </v-card>
    <v-alert v-if="loadError" type="error" variant="tonal" class="mb-4">{{ loadError }}</v-alert>
    <v-progress-linear v-if="loading" indeterminate class="mb-4" />
    <v-alert v-if="selectedMethodMeta?.experimental" type="warning" variant="tonal" density="compact" class="mb-4">{{ t('distExperimentalWarning') }}</v-alert>
    <ConfigurationPicker v-if="!selectedMethod" :title="t('distMethod')" :items="methods.map(method => ({ value: method.key, label: method.label }))"
      :disabled="busy" :empty-text="t('distSelectHint')" @select="method => requestChange(() => loadConfig(method))" />
    <ConfigWorkspaceLayout v-if="config.sections.length" v-model:search="search" v-model:show-guide="guideOpen" :field-count="fieldCount" filterable>
      <v-expansion-panels v-if="filteredSections.length" :model-value="search?.trim() ? filteredSections.map(section => section.name) : openPanels"
        multiple variant="accordion" @update:model-value="value => { if (!search?.trim() && Array.isArray(value)) openPanels = value.map(String) }">
        <v-expansion-panel v-for="section in filteredSections" :key="section.name" :value="section.name" elevation="0">
          <v-expansion-panel-title>
            <v-icon icon="mdi-cog-outline" class="mr-2" size="small" />{{ section.name === 'root' ? t('distGeneral') : section.name }}
            <v-chip size="x-small" class="ml-2" variant="outlined">{{ section.fields.length }}</v-chip>
          </v-expansion-panel-title>
          <v-expansion-panel-text><v-row><v-col v-for="field in section.fields" :key="field.key" cols="12" :sm="isWideField(field) ? 12 : 6"><DistillField :field="field" :disabled="busy" /></v-col></v-row></v-expansion-panel-text>
        </v-expansion-panel>
      </v-expansion-panels>
      <div v-else class="workspace-empty"><v-icon icon="mdi-magnify" size="32" /><h2>{{ t('cfgNoMatchingFields') }}</h2><v-btn variant="outlined" @click="search = ''">{{ t('cfgClearSearch') }}</v-btn></div>
      <div class="d-flex flex-wrap align-center ga-3 mt-4">
        <v-btn :color="dirty ? 'warning' : 'primary'" prepend-icon="mdi-content-save" :loading="saving" :disabled="!dirty || busy" @click="saveConfig">{{ t('distSave') }}</v-btn>
        <v-btn variant="outlined" prepend-icon="mdi-refresh" :disabled="busy" @click="requestChange(() => loadConfig())">{{ t('distReload') }}</v-btn>
        <v-btn color="primary" prepend-icon="mdi-play" :loading="trainingLaunching" :disabled="isRunning || busy" @click="startTraining">{{ t('distTrain') }}</v-btn>
        <TaskStopButton :task-id="runningTask?.task_id" :text="t('distStop')" />
      </div>
      <template #guide><HelpPanel :guide-html="guideHtml" :field-help="{}" height="min(680px, calc(100dvh - 120px))" /></template>
    </ConfigWorkspaceLayout>
    <template v-if="distillTasks.length || config.sections.length || taskStore.taskListError">
      <v-divider class="my-4" />
      <div class="d-flex align-center mb-2"><h2 class="workspace-section-title">{{ t('distActiveTasks') }}</h2><v-spacer /><v-btn variant="text" size="small" prepend-icon="mdi-refresh" @click="taskStore.fetchTasks()">{{ t('distRefresh') }}</v-btn></div>
      <v-alert v-if="taskStore.taskListError" type="error" variant="tonal" density="compact">{{ t('taskLoadFailed') }}</v-alert>
      <v-list v-else-if="distillTasks.length" density="compact"><v-list-item v-for="task in visibleTasks" :key="task.task_id"
        :title="task.command" :subtitle="`${task.task_id} | ${t('taskState')}: ${task.state}`" :to="{ path: '/tasks', query: { task: task.task_id } }">
        <template #append><v-chip size="small" :color="stateColor(task.state)" variant="tonal">{{ task.state }}</v-chip><TaskStopButton v-if="isTaskStoppable(task.state)" :task-id="task.task_id" /></template>
      </v-list-item></v-list>
      <p v-else class="text-medium-emphasis text-body-2">{{ t('distNoTasks') }}</p>
      <v-btn v-if="hasHiddenTasks" variant="text" size="small" class="mt-1" @click="showAllTasks = !showAllTasks">{{ showAllTasks ? t('taskShowLess') : t('taskShowAll', { count: distillTasks.length }) }}</v-btn>
    </template>
    <UnsavedChangesDialog :model-value="Boolean(pendingChange)" :busy="saving || loading" :error="loadError" @cancel="cancelChange" @discard="finishChange(false)" @save="finishChange(true)" />
  </v-container>
</template>

<script setup lang="ts">
import { isTaskActive, isTaskStoppable } from '../utils/taskState.ts'
import WorkspaceHeading from '../components/WorkspaceHeading.vue'
import { ref, reactive, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import { onBeforeRouteLeave } from 'vue-router'
import ConfigWorkspaceLayout from '../components/ConfigWorkspaceLayout.vue'
import ConfigurationPicker from '../components/ConfigurationPicker.vue'
import HelpPanel from '../components/HelpPanel.vue'
import UnsavedChangesDialog from '../components/UnsavedChangesDialog.vue'
import TaskStopButton from '../components/TaskStopButton.vue'
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
const guideOpen = ref(true)
const guideHtml = ref('')
const trainingLaunching = ref(false)
const openPanels = ref<string[]>([])
const search = ref<string | null>('')
const loadError = ref('')
const busy = computed(() => loading.value || saving.value || trainingLaunching.value)
const filteredSections = computed(() => {
  const terms = (search.value ?? '').toLowerCase().split(/\s+/).filter(Boolean)
  return config.sections.map(section => ({ ...section, fields: section.fields.filter(field => terms.every(term => `${field.key} ${field.comment}`.toLowerCase().includes(term))) })).filter(section => section.fields.length)
})
const fieldCount = computed(() => filteredSections.value.reduce((sum, section) => sum + section.fields.length, 0))
const runningTask = computed(() => taskStore.tasks.find(task => task.command === selectedMethodMeta.value?.task_command && isTaskStoppable(task.state)))
let loadSequence = 0
const pendingChange = ref<(() => Promise<void>) | null>(null)
let cancelRoute: (() => void) | undefined
async function requestChange(action: () => Promise<void>) {
  if (busy.value) return
  if (dirty.value) pendingChange.value = action
  else await action()
}
function cancelChange() { pendingChange.value = null; cancelRoute?.(); cancelRoute = undefined }
async function finishChange(save: boolean) {
  if (!pendingChange.value || busy.value) return
  if (save && !await saveConfig()) return
  const action = pendingChange.value
  pendingChange.value = null
  cancelRoute = undefined
  await action()
}
onBeforeRouteLeave(() => {
  if (!dirty.value) return true
  return new Promise<boolean>(resolve => { pendingChange.value = async () => { resolve(true) }; cancelRoute = () => resolve(false) })
})
onBeforeUnmount(() => { ++loadSequence; cancelRoute?.() })

const selectedMethodMeta = computed(() =>
  methods.value.find(m => m.key === selectedMethod.value)
)

const isRunning = computed(() => {
  const method = selectedMethodMeta.value
  if (!method) return false
  return taskStore.tasks.some(
    tp => tp.command === method.task_command && isTaskActive(tp.state)
  )
})

const distillTasks = computed(() => {
  const cmds = new Set(methods.value.map(m => m.task_command))
  return taskStore.tasks.filter(tp => cmds.has(tp.command))
})

// ── Task list preview (collapse long lists) ────────────────────

const TASK_PREVIEW_LIMIT = 4
const showAllTasks = ref(false)
const visibleTasks = computed(() =>
  showAllTasks.value ? distillTasks.value : distillTasks.value.slice(0, TASK_PREVIEW_LIMIT)
)
const hasHiddenTasks = computed(() => distillTasks.value.length > TASK_PREVIEW_LIMIT)

const currentHash = computed(() => {
  return JSON.stringify(config.sections.map(s => ({ name: s.name, fields: s.fields.map(f => ({ key: f.key, value: f.value })) })))
})

const dirty = computed(() => config.sections.length > 0 && currentHash.value !== originalHash.value)

function isWideField(field: DistillField) {
  if (field.type === 'list') return true
  if (typeof field.value === 'string' && String(field.value).length > 40) return true
  return false
}

async function loadMethods() {
  loading.value = true
  try {
    const res = await fetch('/api/distill/methods')
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    methods.value = await res.json()
  } catch (cause) { loadError.value = t('cfgErrLoadMethods', { error: String(cause) }) }
  finally { loading.value = false }
}

async function loadConfig(method = selectedMethod.value) {
  if (!method) return
  const sequence = ++loadSequence
  loading.value = true
  loadError.value = ''
  try {
    const res = await fetch(`/api/distill/config?method=${encodeURIComponent(method)}`)
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
    if (sequence !== loadSequence) return
    selectedMethod.value = method
    Object.assign(config, data)
    originalHash.value = currentHash.value
    openPanels.value = []
    search.value = ''
    await loadGuide()
  } catch (cause) { if (sequence === loadSequence) loadError.value = t('cfgErrLoadConfig', { error: String(cause) }) }
  finally { if (sequence === loadSequence) loading.value = false }
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

async function saveConfig(): Promise<boolean> {
  if (saving.value || loading.value) return false
  saving.value = true
  loadError.value = ''
  try {
    const updates = Object.fromEntries(config.sections.map(section => [section.name, Object.fromEntries(section.fields.map(field => [field.key, field.value]))]))
    const res = await fetch(`/api/distill/config?method=${encodeURIComponent(selectedMethod.value)}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ updates }),
    })
    if (!res.ok) { const data = await res.json().catch(() => ({})); throw new Error(data.detail || `HTTP ${res.status}`) }
    originalHash.value = currentHash.value
    notify.show(t('distSaved'), 'success')
    return true
  } catch (cause) {
    loadError.value = t('cfgErrSave', { error: String(cause) })
    return false
  } finally { saving.value = false }
}

async function startTraining() {
  const method = selectedMethodMeta.value
  if (!method || busy.value || isRunning.value) return
  trainingLaunching.value = true
  try {
    if (dirty.value && !await saveConfig()) return
    const taskId = await taskStore.startTask(method.task_command)
    notify.show(t(taskId ? 'notifyTaskStarted' : 'notifyTaskStartFailed', { command: method.task_command }), taskId ? 'success' : 'error')
  } finally { trainingLaunching.value = false }
}

function stateColor(state: string) {
  if (state === 'paused') return 'warning'
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
.distill-page { flex: 1 1 0; min-height: 0; overflow-y: auto; }
</style>
