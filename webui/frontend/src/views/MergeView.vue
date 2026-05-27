<template>
  <v-container fluid class="pa-4">
    <div class="text-h5 mb-1">{{ t('mgTitle') }}</div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('mgSubtitle') }}</div>

    <v-row>
      <!-- Left: File Browser -->
      <v-col cols="12" md="5">
        <v-card variant="tonal" class="mb-4">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-folder-open-outline" class="mr-2" />
            {{ t('mgDirs') }}
          </v-card-title>
          <v-card-text>
            <v-select
              v-model="selectedDir"
              :items="adapterDirs"
              item-title="name"
              item-value="path"
              :label="t('mgAdapterDir')"
              variant="outlined"
              density="compact"
              hide-details
              @update:model-value="onDirChange"
            />

            <v-list v-if="adapterFiles.length > 0" density="compact" class="mt-3" max-height="400" style="overflow-y: auto">
              <v-list-item
                v-for="file in adapterFiles"
                :key="file.path"
                :active="selectedFile?.path === file.path"
                @click="selectFile(file)"
              >
                <v-list-item-title class="text-body-2">{{ file.name }}</v-list-item-title>
                <v-list-item-subtitle>{{ file.size_human }} | {{ formatDate(file.mtime) }}</v-list-item-subtitle>
              </v-list-item>
            </v-list>
            <div v-else-if="selectedDir" class="text-medium-emphasis text-body-2 mt-3">
              {{ t('mgNoFiles') }}
            </div>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Right: Scan Results + Config -->
      <v-col cols="12" md="7">
        <!-- Scan result card -->
        <v-card v-if="scanResult" variant="tonal" class="mb-4" :border="verdictBorder">
          <v-card-title class="text-subtitle-1">
            <v-icon :icon="verdictIcon" class="mr-2" :color="verdictColor" />
            {{ t('mgScanResult') }}
          </v-card-title>
          <v-card-text>
            <v-chip :color="verdictColor" variant="tonal" class="mb-2">
              {{ t(`mgVerdict_${scanResult.verdict}`) }}
            </v-chip>
            <div class="text-body-2 mb-2">{{ t('mgTotalKeys') }}: {{ scanResult.total_keys }}</div>
            <div v-if="Object.keys(scanResult.counts).length > 0">
              <div class="text-subtitle-2 mb-1">{{ t('mgKeyCounts') }}</div>
              <div class="d-flex flex-wrap ga-1">
                <v-chip v-for="(count, family) in scanResult.counts" :key="family" size="small" variant="outlined">
                  {{ family }}: {{ count }}
                </v-chip>
              </div>
            </div>
          </v-card-text>
        </v-card>

        <!-- Merge Configuration -->
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-call-merge" class="mr-2" />
            {{ t('mgConfig') }}
          </v-card-title>
          <v-card-text>
            <v-row>
              <v-col cols="12" md="6">
                <v-text-field
                  v-model="baseDit"
                  :label="t('mgBaseDit')"
                  hint="models/diffusion_models/anima-base-v1.0.safetensors"
                  persistent-hint
                  variant="outlined"
                  density="compact"
                />
              </v-col>
              <v-col cols="12" md="3">
                <v-select
                  v-model="dtype"
                  :items="['bf16', 'fp16', 'fp32']"
                  label="dtype"
                  variant="outlined"
                  density="compact"
                  hide-details
                />
              </v-col>
              <v-col cols="12" md="3">
                <v-text-field
                  v-model.number="multiplier"
                  :label="t('mgMultiplier')"
                  :hint="t('mgMultiplierHint')"
                  persistent-hint
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  variant="outlined"
                  density="compact"
                />
              </v-col>
            </v-row>
            <v-row>
              <v-col cols="12" md="6">
                <v-text-field
                  v-model="outputPath"
                  :label="t('mgOutputPath')"
                  hint="Leave empty for auto"
                  persistent-hint
                  variant="outlined"
                  density="compact"
                />
              </v-col>
              <v-col cols="12" md="6">
                <v-switch
                  v-model="allowPartial"
                  :label="t('mgAllowPartial')"
                  :hint="t('mgPartialHint')"
                  persistent-hint
                  color="primary"
                  density="compact"
                />
              </v-col>
            </v-row>

            <v-alert type="warning" variant="tonal" density="compact" class="mt-2">
              <span v-html="t('mgWarning')" />
            </v-alert>
          </v-card-text>
          <v-card-actions>
            <v-btn
              color="primary"
              prepend-icon="mdi-merge"
              :loading="isRunning('merge')"
              :disabled="!selectedFile"
              @click="runMerge"
            >
              {{ t('mgMergeBtn') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>

    <v-divider class="my-4" />

    <div class="text-subtitle-1 mb-2">{{ t('mgActiveTasks') }}</div>
    <v-list v-if="mergeTasks.length > 0" density="compact">
      <v-list-item
        v-for="task in mergeTasks"
        :key="task.task_id"
        :title="task.command"
        :subtitle="`${t('taskState')}: ${task.state} | PID: ${task.pid ?? '—'}`"
      >
        <template #append>
          <v-chip size="small" :color="stateColor(task.state)" variant="tonal">{{ task.state }}</v-chip>
          <v-btn v-if="task.state === 'running'" icon="mdi-stop" size="small" variant="text" color="error" @click="taskStore.cancelTask(task.task_id)" />
        </template>
      </v-list-item>
    </v-list>
    <div v-else class="text-medium-emphasis text-body-2">{{ t('mgNoTasks') }}</div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useI18n } from '../composables/useI18n'

const taskStore = useTaskStore()
const notify = useNotifyStore()
const { t } = useI18n()
taskStore.fetchTasks()

// ── State ─────────────────────────────────────────────────────

interface AdapterDir { name: string; path: string }
interface AdapterFile { name: string; path: string; size: number; size_human: string; mtime: string }
interface ScanResult { verdict: string; counts: Record<string, number>; total_keys: number; metadata: Record<string, string> }

const adapterDirs = ref<AdapterDir[]>([])
const selectedDir = ref('')
const adapterFiles = ref<AdapterFile[]>([])
const selectedFile = ref<AdapterFile | null>(null)
const scanResult = ref<ScanResult | null>(null)

const baseDit = ref('models/diffusion_models/anima-base-v1.0.safetensors')
const dtype = ref('bf16')
const multiplier = ref(1.0)
const allowPartial = ref(false)
const outputPath = ref('')

// ── Load directories ──────────────────────────────────────────

async function fetchDirs() {
  try {
    const res = await fetch('/api/merge/dirs')
    if (!res.ok) return
    const data = await res.json()
    adapterDirs.value = data.dirs || []
    if (adapterDirs.value.length > 0 && !selectedDir.value) {
      selectedDir.value = adapterDirs.value[0].path
      await onDirChange(selectedDir.value)
    }
  } catch { /* ignore */ }
}

onMounted(fetchDirs)

async function onDirChange(dirPath: string) {
  selectedFile.value = null
  scanResult.value = null
  if (!dirPath) { adapterFiles.value = []; return }
  try {
    const res = await fetch(`/api/merge/files?dir=${encodeURIComponent(dirPath)}`)
    if (!res.ok) return
    const data = await res.json()
    adapterFiles.value = data.files || []
  } catch { adapterFiles.value = [] }
}

// ── File selection & scan ─────────────────────────────────────

async function selectFile(file: AdapterFile) {
  selectedFile.value = file
  scanResult.value = null
  try {
    const res = await fetch('/api/merge/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_path: file.path }),
    })
    if (res.ok) scanResult.value = await res.json()
  } catch { /* ignore */ }
}

// ── Verdict display ───────────────────────────────────────────

const verdictColor = computed(() => {
  const v = scanResult.value?.verdict
  if (v === 'ok') return 'success'
  if (v === 'partial') return 'warning'
  if (v === 'block') return 'error'
  return 'grey'
})

const verdictIcon = computed(() => {
  const v = scanResult.value?.verdict
  if (v === 'ok') return 'mdi-check-circle'
  if (v === 'partial') return 'mdi-alert'
  if (v === 'block') return 'mdi-block-helper'
  return 'mdi-help-circle'
})

const verdictBorder = computed(() => {
  const v = scanResult.value?.verdict
  if (v === 'ok') return 'success thin'
  if (v === 'partial') return 'warning thin'
  if (v === 'block') return 'error thin'
  return undefined
})

// ── Merge ─────────────────────────────────────────────────────

async function runMerge() {
  if (!selectedFile.value) return
  const args = ['--adapter', selectedFile.value.path]
  if (baseDit.value) args.push('--base_dit', baseDit.value)
  if (multiplier.value !== 1.0) args.push('--multiplier', String(multiplier.value))
  if (dtype.value !== 'bf16') args.push('--dtype', dtype.value)
  if (outputPath.value) args.push('--output', outputPath.value)
  if (allowPartial.value) args.push('--allow_partial')
  const taskId = await taskStore.startTask('merge', args)
  if (taskId) {
    notify.show(t('notifyTaskStarted', { command: t('mgMergeBtn') }), 'success')
  } else {
    notify.show(t('notifyTaskStartFailed', { command: t('mgMergeBtn') }), 'error')
  }
}

// ── Helpers ───────────────────────────────────────────────────

function formatDate(iso: string) {
  try { return new Date(iso).toLocaleDateString() } catch { return iso }
}

function isRunning(command: string) {
  return taskStore.tasks.some(tp => tp.command === command && tp.state === 'running')
}

const mergeTasks = computed(() =>
  taskStore.tasks.filter(tp => tp.command === 'merge')
)

function stateColor(state: string) {
  if (state === 'running') return 'info'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}
</script>

<style scoped>
/* File tree: hover highlight */
:deep(.v-list-item:hover) {
  background: rgba(199, 91, 26, 0.06) !important;
}

/* Weight slider: ember track */
:deep(.v-slider-track__fill) {
  background: linear-gradient(90deg, var(--forge-ember), var(--forge-amber)) !important;
}
:deep(.v-slider-thumb) {
  color: var(--forge-ember) !important;
}

/* Merge strategy radio: card style */
:deep(.v-selection-control) {
  padding: 8px 12px;
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  transition: border-color 0.15s;
}
:deep(.v-selection-control--selected) {
  border-color: var(--forge-ember) !important;
  background: rgba(199, 91, 26, 0.04);
}
</style>
