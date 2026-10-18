<template>
  <v-container fluid class="pa-4">
    <WorkspaceHeading :title="t('sysTitle')" :subtitle="t('sysSubtitle')" />
    <v-alert v-if="loadError" type="error" variant="tonal" class="mb-4">{{ loadError }}<v-btn variant="text" :loading="loading" @click="loadSystem">{{ t('ppRefresh') }}</v-btn></v-alert>
    <v-progress-linear v-if="loading" indeterminate class="mb-3" />

    <v-row>
      <!-- Left column: model paths, downloads, update -->
      <v-col cols="12" md="6" class="d-flex flex-column ga-4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-cube-outline" class="mr-2" />
            {{ t('sysCoreModels') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-3">{{ t('sysCoreModelsDesc') }}</div>
            <div
              v-for="mp in modelPaths"
              :key="mp.id"
              class="mb-3"
            >
              <v-text-field
                v-model="mp.path"
                :label="t(`sysModel_${mp.id}`)"
                variant="outlined"
                density="compact"
                hide-details="auto"
                :loading="mp.validating"
                :disabled="modelPathsSaving"
                :title="mp.path"
                @update:model-value="onModelPathChange(mp)"
              >
                <template #append-inner>
                  <v-icon
                    v-if="mp.exists === true"
                    icon="mdi-check-circle"
                    color="success"
                    size="small"
                    class="mr-1"
                  />
                  <v-icon
                    v-else-if="mp.exists === false"
                    icon="mdi-alert-circle"
                    color="warning"
                    size="small"
                    class="mr-1"
                  />
                  <v-btn
                    icon="mdi-folder-open"
                    size="x-small"
                    variant="text"
                    :aria-label="`${t('cfSelectModel')}: ${t(`sysModel_${mp.id}`)}`"
                    @click="openModelBrowser(mp)"
                  />
                </template>
              </v-text-field>
            </div>
            <v-btn
              color="primary"
              block
              :disabled="!modelPathsDirty || loading"
              :loading="modelPathsSaving"
              prepend-icon="mdi-content-save"
              @click="saveModelPaths"
            >
              {{ t('sysSavePaths') }}{{ modelPathsDirty ? ' *' : '' }}
            </v-btn>
          </v-card-text>
        </v-card>

        <!-- Download Status -->
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-download" class="mr-2" />
            {{ t('sysDownloads') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-3">{{ t('sysDownloadsDesc') }}</div>
            <v-list density="compact" class="mb-3">
              <v-list-item
                v-for="group in modelGroups"
                :key="group.id"
              >
                <template #prepend>
                  <v-icon :icon="group.installed ? 'mdi-check-circle' : 'mdi-alert-circle'" :color="group.installed ? 'success' : 'warning'" />
                </template>
                <v-list-item-title :title="t(`sysModel_${group.id}`)">{{ t(`sysModel_${group.id}`) }}</v-list-item-title>
                <v-list-item-subtitle>
                  <v-chip size="x-small" :color="group.installed ? 'success' : 'warning'" variant="tonal">
                    {{ group.installed ? t('sysInstalled') : t('sysMissing') }}
                  </v-chip>
                </v-list-item-subtitle>
                <template #append>
                  <v-btn
                    size="small"
                    variant="text"
                    :loading="isRunning(`download-${group.id}`)"
                    @click="runTask(`download-${group.id}`)"
                  >
                    {{ group.installed ? t('sysRedownload') : t('sysDownload') }}
                  </v-btn>
                </template>
              </v-list-item>
            </v-list>
            <v-btn color="primary" block :loading="isRunning('download-models')" @click="runTask('download-models')">
              {{ t('sysDownloadAll') }}
            </v-btn>
          </v-card-text>
        </v-card>

        <!-- Self Update -->
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-update" class="mr-2" />
            {{ t('sysUpdate') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-3">{{ t('sysUpdateDesc') }}</div>
            <v-btn color="primary" block :loading="isRunning('update')" @click="showUpdateDlg = true">
              {{ t('sysUpdateBtn') }}
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Right column: environment, quick actions, appearance -->
      <v-col cols="12" md="6" class="d-flex flex-column ga-4">
        <!-- Environment -->
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-information-outline" class="mr-2" />
            {{ t('sysEnvironment') }}
          </v-card-title>
          <v-card-text>
            <v-list density="compact">
              <v-list-item :title="t('sysBackend')" :subtitle="t('sysBackendVal')" />
              <v-list-item :title="t('sysFrontend')" :subtitle="t('sysFrontendVal')" />
              <v-list-item :title="t('sysProtocol')" :subtitle="t('sysProtocolVal')" />
            </v-list>
          </v-card-text>
        </v-card>

        <!-- Quick Actions -->
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-lightning-bolt" class="mr-2" />
            {{ t('sysQuickActions') }}
          </v-card-title>
          <v-card-text>
            <div class="d-flex flex-column ga-2">
              <v-btn variant="outlined" block prepend-icon="mdi-cog-transfer-outline" to="/config">
                {{ t('sysConfigEditor') }}
              </v-btn>
              <v-btn variant="outlined" block prepend-icon="mdi-console" to="/tasks">
                {{ t('sysTaskMonitor') }}
              </v-btn>
              <v-btn variant="outlined" block prepend-icon="mdi-test-tube" :loading="isRunning('test-unit')" @click="runTask('test-unit')">
                {{ t('sysRunTests') }}
              </v-btn>
            </div>
          </v-card-text>
        </v-card>

        <!-- Appearance -->
        <AppearanceSettings />
      </v-col>
    </v-row>

    <v-divider class="my-4" />

    <div class="d-flex align-center mb-2">
      <div class="text-subtitle-1">{{ t('sysAllTasks') }}</div>
      <v-spacer />
      <v-btn variant="text" size="small" prepend-icon="mdi-refresh" :loading="tasksLoading" @click="refreshTasks">
        {{ t('ppRefresh') }}
      </v-btn>
    </div>

    <v-list v-if="taskStore.tasks.length > 0" density="compact">
      <v-list-item
        v-for="task in visibleTasks"
        :key="task.task_id"
        :title="task.command"
        :subtitle="`${task.task_id.slice(0, 8)} | ${t('taskState')}: ${task.state} | PID: ${task.pid ?? '—'}`"
      >
        <template #append>
          <v-chip size="small" :color="stateColor(task.state)" variant="tonal">{{ task.state }}</v-chip>
          <v-btn v-if="isTaskStoppable(task.state)" icon="mdi-stop" size="small" variant="text" color="error" :aria-label="t('taskCancel')" @click="pendingCancel = task.task_id" />
        </template>
      </v-list-item>
    </v-list>
    <div v-else class="text-medium-emphasis text-body-2">{{ t('sysNoTasks') }}</div>
    <v-btn
      v-if="hasHiddenTasks"
      variant="text"
      size="small"
      class="mt-1"
      :prepend-icon="showAllTasks ? 'mdi-chevron-up' : 'mdi-chevron-down'"
      @click="showAllTasks = !showAllTasks"
    >
      {{ showAllTasks ? t('taskShowLess') : t('taskShowAll', { count: taskStore.tasks.length }) }}
    </v-btn>

    <!-- Model file browser dialog -->
    <v-dialog v-model="showBrowserDlg" max-width="700" scrollable>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-folder-open" class="mr-2" />
          {{ t('cfSelectModel') }}
        </v-card-title>
        <v-card-subtitle class="text-caption pb-2 system-browser-path">{{ browserCurrentDir }}</v-card-subtitle>
        <v-divider />
        <v-progress-linear v-if="browserLoading" indeterminate />
        <v-alert v-if="browserError" type="error" variant="tonal">{{ browserError }}</v-alert>
        <v-card-text style="max-height: 50vh" class="pa-0">
          <v-list density="compact" class="py-0">
            <v-list-item
              v-if="browserParent"
              prepend-icon="mdi-arrow-up-bold"
              :title="t('cfParentDir')"
              :disabled="browserLoading"
              @click="browseTo(browserParent)"
            />
            <v-list-item
              v-for="dir in browserSubdirs"
              :key="dir.path"
              prepend-icon="mdi-folder"
              :title="dir.name"
              :disabled="browserLoading"
              @click="browseTo(dir.path)"
            />
            <v-divider v-if="browserSubdirs.length > 0 && browserFiles.length > 0" />
            <v-list-item
              v-for="file in browserFiles"
              :key="file.path"
              :active="browserSelected === file.path"
              :disabled="browserLoading"
              @click="browserSelected = file.path"
            >
              <template #prepend>
                <v-icon icon="mdi-file-outline" class="mr-2" />
              </template>
              <v-list-item-title>{{ file.name }}</v-list-item-title>
              <v-list-item-subtitle>{{ file.size_human }}</v-list-item-subtitle>
              <template #append>
                <v-icon v-if="browserSelected === file.path" icon="mdi-check" color="primary" />
              </template>
            </v-list-item>
            <v-list-item v-if="!browserLoading && !browserError && browserSubdirs.length === 0 && browserFiles.length === 0">
              <v-list-item-title class="text-medium-emphasis">{{ t('cfNoFiles') }}</v-list-item-title>
            </v-list-item>
          </v-list>
        </v-card-text>
        <v-divider />
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showBrowserDlg = false">{{ t('cfCancel') }}</v-btn>
          <v-btn color="primary" :disabled="!browserSelected || browserLoading || Boolean(browserError)" @click="confirmBrowser">{{ t('cfConfirm') }}</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
    <v-dialog :model-value="Boolean(pendingCancel)" max-width="480" :persistent="canceling" @update:model-value="value => { if (!value) pendingCancel = null }">
      <v-card :title="t('taskCancel')" :text="t('taskCancelConfirm')"><v-card-actions><v-spacer /><v-btn :disabled="canceling" @click="pendingCancel = null">{{ t('cfCancel') }}</v-btn><v-btn color="error" variant="flat" :loading="canceling" @click="cancelTask">{{ t('taskCancel') }}</v-btn></v-card-actions></v-card>
    </v-dialog>
    <v-dialog v-model="discardOpen" max-width="480" persistent>
      <v-card :title="t('settingsUnsavedTitle')" :text="t('settingsDiscardBody')"><v-card-actions><v-spacer /><v-btn @click="resolveDiscard(false)">{{ t('cfCancel') }}</v-btn><v-btn color="warning" @click="resolveDiscard(true)">{{ t('dsDiscard') }}</v-btn></v-card-actions></v-card>
    </v-dialog>

    <!-- Update dialog -->
    <v-dialog v-model="showUpdateDlg" max-width="450">
      <v-card>
        <v-card-title>{{ t('sysUpdate') }}</v-card-title>
        <v-card-text>
          <v-checkbox v-model="updateDryRun" :label="t('sysUpdateDryRun')" density="compact" hide-details class="mb-2" />
          <div class="text-subtitle-2 mb-1">{{ t('sysUpdateConflictPolicy') }}</div>
          <v-radio-group v-model="updateConflictPolicy" density="compact" hide-details>
            <v-radio :label="t('sysUpdateKeepConflicts')" value="keep" />
            <v-radio :label="t('sysUpdateOverwrite')" value="overwrite" />
          </v-radio-group>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showUpdateDlg = false">{{ t('dsCancel') }}</v-btn>
          <v-btn color="primary" :loading="isRunning('update')" @click="runUpdate">
            {{ t('sysUpdateBtn') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { isTaskActive, isTaskStoppable } from '../utils/taskState.ts'
import { onMounted, onBeforeUnmount, ref, computed } from 'vue'
import { onBeforeRouteLeave } from 'vue-router'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useI18n } from '../composables/useI18n'
import AppearanceSettings from '../components/AppearanceSettings.vue'
import WorkspaceHeading from '../components/WorkspaceHeading.vue'

const taskStore = useTaskStore()
const notify = useNotifyStore()
const { t } = useI18n()
const loading = ref(false), loadError = ref(''), tasksLoading = ref(false)
const pendingCancel = ref<string | null>(null), canceling = ref(false)
const submitting = ref(new Set<string>())
async function refreshTasks() {
  tasksLoading.value = true
  try { await taskStore.fetchTasks() } finally { tasksLoading.value = false }
}
async function cancelTask() {
  if (!pendingCancel.value || canceling.value) return
  canceling.value = true
  try { if (await taskStore.cancelTask(pendingCancel.value)) pendingCancel.value = null } finally { canceling.value = false }
}

// ── Model groups (download status) ─────────────────────────────

interface ModelGroup {
  id: string
  installed: boolean
  files: { path: string; exists: boolean }[]
}

const modelGroups = ref<ModelGroup[]>([])

async function fetchModelGroups() {
  const res = await fetch('/api/system/models')
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  modelGroups.value = data.groups || []
}

// ── Model paths (configurable) ─────────────────────────────────

interface ModelPath {
  id: string
  toml_key: string
  path: string
  resolved: string
  exists: boolean
  validating: boolean
  _original: string
}

const modelPaths = ref<ModelPath[]>([])
const modelPathsSaving = ref(false)

const modelPathsDirty = computed(() =>
  modelPaths.value.some(mp => mp.path !== mp._original)
)
const discardOpen = ref(false)
let discardResolve: ((value: boolean) => void) | undefined
function resolveDiscard(value: boolean) { discardOpen.value = false; discardResolve?.(value); discardResolve = undefined }
async function requestLeave(): Promise<boolean> {
  if (modelPathsSaving.value) return false
  if (!modelPathsDirty.value) return true
  discardResolve?.(false)
  discardOpen.value = true
  return new Promise(resolve => { discardResolve = resolve })
}
function beforeUnload(event: BeforeUnloadEvent) { if (modelPathsDirty.value) { event.preventDefault(); event.returnValue = '' } }
onBeforeRouteLeave(requestLeave)

async function fetchModelPaths() {
  const res = await fetch('/api/system/model-paths')
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  modelPaths.value = (data.paths || []).map((p: any) => ({
    ...p,
    validating: false,
    _original: p.path,
  }))
}

async function loadSystem() {
  if (loading.value) return
  if (!await requestLeave()) return
  loading.value = true
  loadError.value = ''
  try { await Promise.all([fetchModelGroups(), fetchModelPaths()]) }
  catch (cause) { loadError.value = t('sysLoadFailed', { error: cause instanceof Error ? cause.message : String(cause) }) }
  finally { loading.value = false }
}

let validateTimers: Record<string, ReturnType<typeof setTimeout>> = {}

function onModelPathChange(mp: ModelPath) {
  if (validateTimers[mp.id]) clearTimeout(validateTimers[mp.id])
  mp.validating = true
  const path = mp.path
  validateTimers[mp.id] = setTimeout(async () => {
    try {
      const res = await fetch(`/api/files/validate?path=${encodeURIComponent(path)}`)
      const data = await res.json()
      if (mp.path === path) mp.exists = data.exists
    } catch {
      if (mp.path === path) mp.exists = false
    } finally {
      if (mp.path === path) mp.validating = false
    }
  }, 500)
}

async function saveModelPaths() {
  if (modelPathsSaving.value || !modelPathsDirty.value) return
  modelPathsSaving.value = true
  try {
    const body = modelPaths.value.map(mp => ({ key: mp.toml_key, value: mp.path }))
    const res = await fetch('/api/system/model-paths', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const data = await res.json().catch(() => null)
      throw new Error(data?.detail || `HTTP ${res.status}`)
    }
    modelPaths.value.forEach(mp => { mp._original = mp.path })
    notify.show(t('sysPathsSaved'), 'success')
    // A refresh failure after a successful write must not report a failed save.
    try { await Promise.all([fetchModelPaths(), fetchModelGroups()]) }
    catch (cause) { loadError.value = t('sysLoadFailed', { error: cause instanceof Error ? cause.message : String(cause) }) }
  } catch (cause) {
    notify.show(t('cfgErrSave', { error: cause instanceof Error ? cause.message : String(cause) }), 'error')
  } finally {
    modelPathsSaving.value = false
  }
}

// ── File browser dialog ────────────────────────────────────────

const showBrowserDlg = ref(false)
const browserTarget = ref<ModelPath | null>(null)
const browserCurrentDir = ref('')
const browserParent = ref<string | null>(null)
const browserSubdirs = ref<{ name: string; path: string }[]>([])
const browserFiles = ref<{ name: string; path: string; size_human: string }[]>([])
const browserSelected = ref<string | null>(null)
const browserLoading = ref(false), browserError = ref('')
let browserRequest = 0

const _DEFAULT_DIRS: Record<string, string> = {
  anima_dit: 'models/diffusion_models',
  anima_te: 'models/text_encoders',
  anima_vae: 'models/vae',
}

function openModelBrowser(mp: ModelPath) {
  browserTarget.value = mp
  browserSelected.value = null
  const val = mp.path
  let startDir = _DEFAULT_DIRS[mp.id] || 'models'
  if (val) {
    const lastSlash = Math.max(val.lastIndexOf('/'), val.lastIndexOf('\\'))
    if (lastSlash > 0) startDir = val.substring(0, lastSlash)
  }
  browseTo(startDir)
  showBrowserDlg.value = true
}

async function browseTo(dirPath: string) {
  const request = ++browserRequest
  browserSelected.value = null
  browserLoading.value = true
  browserError.value = ''
  browserCurrentDir.value = dirPath
  browserParent.value = null
  browserSubdirs.value = []
  browserFiles.value = []
  try {
    const res = await fetch(`/api/files/browse?dir=${encodeURIComponent(dirPath)}&ext=.safetensors`)
    const data = await res.json()
    if (!res.ok || data.error) throw new Error(data.error || data.detail || `HTTP ${res.status}`)
    if (request !== browserRequest) return
    browserCurrentDir.value = data.current_dir || data.current_dir_abs || dirPath
    browserParent.value = data.parent
    browserSubdirs.value = data.subdirs || []
    browserFiles.value = data.files || []
  } catch (cause) { if (request === browserRequest) browserError.value = cause instanceof Error ? cause.message : String(cause) }
  finally { if (request === browserRequest) browserLoading.value = false }
}

function confirmBrowser() {
  if (!browserLoading.value && !browserError.value && browserSelected.value && browserTarget.value) {
    browserTarget.value.path = browserSelected.value
    onModelPathChange(browserTarget.value)
    showBrowserDlg.value = false
  }
}

// ── Update dialog ──────────────────────────────────────────────

const showUpdateDlg = ref(false)
const updateDryRun = ref(false)
const updateConflictPolicy = ref('keep')

async function runUpdate() {
  const args: string[] = []
  if (updateDryRun.value) args.push('--dry-run')
  if (updateConflictPolicy.value === 'overwrite') args.push('--yes-overwrite')
  showUpdateDlg.value = false
  await runTask('update', args)
}

// ── Task list preview (collapse long lists) ────────────────────

const TASK_PREVIEW_LIMIT = 4
const showAllTasks = ref(false)
const visibleTasks = computed(() =>
  showAllTasks.value ? taskStore.tasks : taskStore.tasks.slice(0, TASK_PREVIEW_LIMIT)
)
const hasHiddenTasks = computed(() => taskStore.tasks.length > TASK_PREVIEW_LIMIT)

// ── Helpers ────────────────────────────────────────────────────

function isRunning(command: string) {
  return submitting.value.has(command) || taskStore.tasks.some(tp => tp.command === command && isTaskActive(tp.state))
}

function stateColor(state: string) {
  if (state === 'paused') return 'warning'
  if (state === 'running' || state === 'stopping') return 'info'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}

async function runTask(command: string, args: string[] = []) {
  if (isRunning(command)) return
  submitting.value.add(command)
  const taskId = await taskStore.startTask(command, args).finally(() => submitting.value.delete(command))
  if (taskId) {
    notify.show(t('notifyTaskStarted', { command }), 'success')
  } else {
    notify.show(t('notifyTaskStartFailed', { command }), 'error')
  }
}

// ── Init ───────────────────────────────────────────────────────

onMounted(() => {
  window.addEventListener('beforeunload', beforeUnload)
  loadSystem()
  refreshTasks()
})
onBeforeUnmount(() => { Object.values(validateTimers).forEach(clearTimeout); browserRequest++; discardResolve?.(false); window.removeEventListener('beforeunload', beforeUnload) })
</script>

<style scoped>
.system-browser-path { white-space: normal; overflow-wrap: anywhere; }
/* Environment info: mono font for paths/values */
:deep(.v-list-item-subtitle) {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-secondary);
}

/* Download progress: brand styling */
:deep(.v-progress-linear) {
  border-radius: 4px;
}

/* System cards: subtle brand border on hover */
:deep(.v-card) {
  transition: border-color 0.2s;
}

:deep(.v-card:hover) {
  border-color: var(--border-default);
}
</style>
