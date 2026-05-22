<template>
  <v-container fluid class="pa-4">
    <div class="text-h5 mb-1">{{ t('sysTitle') }}</div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('sysSubtitle') }}</div>

    <v-row>
      <!-- Core Model Paths -->
      <v-col cols="12" md="6">
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
                    @click="openModelBrowser(mp)"
                  />
                </template>
              </v-text-field>
            </div>
            <v-btn
              color="primary"
              block
              :disabled="!modelPathsDirty"
              :loading="modelPathsSaving"
              prepend-icon="mdi-content-save"
              @click="saveModelPaths"
            >
              {{ t('sysSavePaths') }}{{ modelPathsDirty ? ' *' : '' }}
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Download Status -->
      <v-col cols="12" md="6">
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
                <v-list-item-title>{{ t(`sysModel_${group.id}`) }}</v-list-item-title>
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
      </v-col>

      <!-- Self Update -->
      <v-col cols="12" md="6">
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

      <!-- Environment -->
      <v-col cols="12" md="6">
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
      </v-col>

      <!-- Quick Actions -->
      <v-col cols="12" md="6">
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
      </v-col>
    </v-row>

    <v-divider class="my-4" />

    <div class="d-flex align-center mb-2">
      <div class="text-subtitle-1">{{ t('sysAllTasks') }}</div>
      <v-spacer />
      <v-btn variant="text" size="small" prepend-icon="mdi-refresh" @click="taskStore.fetchTasks()">
        {{ t('ppRefresh') }}
      </v-btn>
    </div>

    <v-list v-if="taskStore.tasks.length > 0" density="compact">
      <v-list-item
        v-for="task in taskStore.tasks"
        :key="task.task_id"
        :title="task.command"
        :subtitle="`${task.task_id.slice(0, 8)} | ${t('taskState')}: ${task.state} | PID: ${task.pid ?? '—'}`"
      >
        <template #append>
          <v-chip size="small" :color="stateColor(task.state)" variant="tonal">{{ task.state }}</v-chip>
          <v-btn v-if="task.state === 'running'" icon="mdi-stop" size="small" variant="text" color="error" @click="taskStore.cancelTask(task.task_id)" />
        </template>
      </v-list-item>
    </v-list>
    <div v-else class="text-medium-emphasis text-body-2">{{ t('sysNoTasks') }}</div>

    <!-- Model file browser dialog -->
    <v-dialog v-model="showBrowserDlg" max-width="700" scrollable>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-folder-open" class="mr-2" />
          {{ t('cfSelectModel') }}
        </v-card-title>
        <v-card-subtitle class="text-caption pb-2">{{ browserCurrentDir }}</v-card-subtitle>
        <v-divider />
        <v-card-text style="max-height: 50vh" class="pa-0">
          <v-list density="compact" class="py-0">
            <v-list-item
              v-if="browserParent"
              prepend-icon="mdi-arrow-up-bold"
              :title="t('cfParentDir')"
              @click="browseTo(browserParent)"
            />
            <v-list-item
              v-for="dir in browserSubdirs"
              :key="dir.path"
              prepend-icon="mdi-folder"
              :title="dir.name"
              @click="browseTo(dir.path)"
            />
            <v-divider v-if="browserSubdirs.length > 0 && browserFiles.length > 0" />
            <v-list-item
              v-for="file in browserFiles"
              :key="file.path"
              :active="browserSelected === file.path"
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
            <v-list-item v-if="browserSubdirs.length === 0 && browserFiles.length === 0">
              <v-list-item-title class="text-medium-emphasis">{{ t('cfNoFiles') }}</v-list-item-title>
            </v-list-item>
          </v-list>
        </v-card-text>
        <v-divider />
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showBrowserDlg = false">{{ t('cfCancel') }}</v-btn>
          <v-btn color="primary" :disabled="!browserSelected" @click="confirmBrowser">{{ t('cfConfirm') }}</v-btn>
        </v-card-actions>
      </v-card>
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
import { onMounted, ref, computed } from 'vue'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useI18n } from '../composables/useI18n'

const taskStore = useTaskStore()
const notify = useNotifyStore()
const { t } = useI18n()
taskStore.fetchTasks()

// ── Model groups (download status) ─────────────────────────────

interface ModelGroup {
  id: string
  installed: boolean
  files: { path: string; exists: boolean }[]
}

const modelGroups = ref<ModelGroup[]>([])

async function fetchModelGroups() {
  try {
    const res = await fetch('/api/system/models')
    if (!res.ok) return
    const data = await res.json()
    modelGroups.value = data.groups || []
  } catch { /* ignore */ }
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

async function fetchModelPaths() {
  try {
    const res = await fetch('/api/system/model-paths')
    if (!res.ok) return
    const data = await res.json()
    modelPaths.value = (data.paths || []).map((p: any) => ({
      ...p,
      validating: false,
      _original: p.path,
    }))
  } catch { /* ignore */ }
}

let validateTimers: Record<string, ReturnType<typeof setTimeout>> = {}

function onModelPathChange(mp: ModelPath) {
  if (validateTimers[mp.id]) clearTimeout(validateTimers[mp.id])
  mp.validating = true
  validateTimers[mp.id] = setTimeout(async () => {
    try {
      const res = await fetch(`/api/files/validate?path=${encodeURIComponent(mp.path)}`)
      const data = await res.json()
      mp.exists = data.exists
    } catch {
      mp.exists = false
    } finally {
      mp.validating = false
    }
  }, 500)
}

async function saveModelPaths() {
  modelPathsSaving.value = true
  try {
    const body = modelPaths.value.map(mp => ({ key: mp.toml_key, value: mp.path }))
    const res = await fetch('/api/system/model-paths', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) throw new Error('Save failed')
    notify.show(t('sysPathsSaved'), 'success')
    await fetchModelPaths()
    await fetchModelGroups() // refresh download status with new paths
  } catch {
    notify.show(t('sysPathsSaveFailed'), 'error')
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
  browserSelected.value = null
  try {
    const res = await fetch(`/api/files/browse?dir=${encodeURIComponent(dirPath)}&ext=.safetensors`)
    const data = await res.json()
    if (data.error) return
    browserCurrentDir.value = data.current_dir || data.current_dir_abs || dirPath
    browserParent.value = data.parent
    browserSubdirs.value = data.subdirs || []
    browserFiles.value = data.files || []
  } catch { /* ignore */ }
}

function confirmBrowser() {
  if (browserSelected.value && browserTarget.value) {
    browserTarget.value.path = browserSelected.value
    onModelPathChange(browserTarget.value)
    showBrowserDlg.value = false
  }
}

// ── Update dialog ──────────────────────────────────────────────

const showUpdateDlg = ref(false)
const updateDryRun = ref(false)
const updateConflictPolicy = ref('keep')

function runUpdate() {
  const args: string[] = []
  if (updateDryRun.value) args.push('--dry-run')
  if (updateConflictPolicy.value === 'overwrite') args.push('--yes-overwrite')
  showUpdateDlg.value = false
  taskStore.startTask('update', args).then(taskId => {
    if (taskId) {
      notify.show(t('notifyTaskStarted', { command: t('sysUpdate') }), 'success')
    } else {
      notify.show(t('notifyTaskStartFailed', { command: t('sysUpdate') }), 'error')
    }
  })
}

// ── Helpers ────────────────────────────────────────────────────

function isRunning(command: string) {
  return taskStore.tasks.some(tp => tp.command === command && tp.state === 'running')
}

function stateColor(state: string) {
  if (state === 'running') return 'info'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}

async function runTask(command: string) {
  const taskId = await taskStore.startTask(command)
  if (taskId) {
    notify.show(t('notifyTaskStarted', { command }), 'success')
  } else {
    notify.show(t('notifyTaskStartFailed', { command }), 'error')
  }
}

// ── Init ───────────────────────────────────────────────────────

onMounted(() => {
  fetchModelGroups()
  fetchModelPaths()
})
</script>
