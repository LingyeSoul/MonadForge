<template>
  <v-container fluid class="pa-4 d-flex flex-column merge-page">
    <div class="text-h5 mb-1">{{ t('mgTitle') }}</div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('mgSubtitle') }}</div>

    <!-- Mode toggle: DiT-bake vs LoRA⊕LoRA fusion -->
    <div class="d-flex align-center mb-4">
      <span class="text-body-2 text-medium-emphasis mr-3">{{ t('merge_mode') }}</span>
      <v-btn-toggle v-model="mode" mandatory color="primary" density="compact" variant="outlined">
        <v-btn value="dit">{{ t('merge_mode_dit') }}</v-btn>
        <v-btn value="loras">{{ t('merge_mode_loras') }}</v-btn>
      </v-btn-toggle>
    </div>

    <div class="merge-scroll flex-grow-1">
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

            <!-- Custom directory: allows users to browse adapters outside the
                 hard-coded server directories. Works for both DiT-bake and
                 LoRA-fusion modes because they share the same file browser. -->
            <div class="d-flex ga-2 align-start mt-3">
              <v-text-field
                v-model="customDirInput"
                :label="t('mgCustomDir')"
                :hint="t('mgCustomDirHint')"
                persistent-hint
                variant="outlined"
                density="compact"
                hide-details="auto"
                class="flex-grow-1"
                @keydown.enter="addCustomDir"
              />
              <v-btn
                color="primary"
                variant="tonal"
                :disabled="!customDirInput.trim()"
                @click="addCustomDir"
              >
                {{ t('mgAddCustomDir') }}
              </v-btn>
            </div>
            <div v-if="customDirs.length > 0" class="d-flex flex-wrap ga-1 mt-2">
              <v-chip
                v-for="dir in customDirs"
                :key="dir.path"
                size="small"
                closable
                variant="tonal"
                color="primary"
                :title="t('mgCustomDirRemove')"
                @click:close="removeCustomDir(dir.path)"
              >
                {{ customDirLabel(dir.path) }}
              </v-chip>
            </div>

            <!-- LoRA-fusion mode: multi-select hint + selection count -->
            <div v-if="mode === 'loras'" class="text-caption text-medium-emphasis mt-3 mb-1">
              {{ t('merge_lora_select_hint') }}
            </div>

            <v-list v-if="adapterFiles.length > 0" density="compact" class="mt-3" max-height="400" style="overflow-y: auto">
              <v-list-item
                v-for="file in adapterFiles"
                :key="file.path"
                :active="isFileActive(file)"
                @click="onFileClick(file)"
              >
                <template v-if="mode === 'loras'" #prepend>
                  <v-icon :icon="isFileSelected(file) ? 'mdi-checkbox-marked' : 'mdi-checkbox-blank-outline'" />
                </template>
                <v-list-item-title class="text-body-2">{{ file.name }}</v-list-item-title>
                <v-list-item-subtitle>{{ file.size_human }} | {{ formatDate(file.mtime) }}</v-list-item-subtitle>
              </v-list-item>
            </v-list>
            <div v-else-if="selectedDir" class="text-medium-emphasis text-body-2 mt-3">
              {{ t('mgNoFiles') }}
            </div>

            <!-- LoRA-fusion selection summary -->
            <div v-if="mode === 'loras'" class="text-body-2 mt-2">
              <v-chip size="small" variant="tonal" color="primary">
                {{ t('merge_lora_selected') }} {{ selectedFiles.length }}
              </v-chip>
              <span v-if="selectedFiles.length > 0 && selectedFiles.length < 2" class="text-warning ml-2 text-caption">
                {{ t('merge_lora_need_two') }}
              </span>
            </div>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Right: Scan Results + Config -->
      <v-col cols="12" md="7">
        <!-- Scan result card (DiT mode only) -->
        <v-card v-if="mode === 'dit' && scanResult" variant="tonal" class="mb-4" :border="verdictBorder">
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

        <!-- Analysis interference banner (LoRA-fusion mode only) -->
        <v-alert
          v-if="mode === 'loras' && analyzeBanner"
          :type="analyzeBanner.type"
          variant="tonal"
          density="comfortable"
          class="mb-4"
          closable
          @click:close="analyzeBanner = null"
        >
          {{ bannerText(analyzeBanner) }}
        </v-alert>

        <!-- ════ DiT-bake Configuration ════ -->
        <v-card v-if="mode === 'dit'" variant="tonal">
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
                  :hint="t('mgBaseDitHint')"
                  persistent-hint
                  variant="outlined"
                  density="compact"
                />
              </v-col>
              <v-col cols="12" md="3">
                <v-select
                  v-model="dtype"
                  :items="['bf16', 'fp16', 'fp32']"
                  :label="t('mgDtype')"
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
                  :hint="t('mgOutputPathHint')"
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

        <!-- ════ LoRA-fusion Configuration ════ -->
        <v-card v-else variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-merge" class="mr-2" />
            {{ t('merge_lora_options') }}
          </v-card-title>
          <v-card-text>
            <v-row>
              <v-col cols="12" md="6">
                <v-text-field
                  v-model="loraWeights"
                  :label="t('merge_weights')"
                  :hint="t('merge_weights_tip')"
                  persistent-hint
                  :placeholder="t('merge_weights_placeholder')"
                  variant="outlined"
                  density="compact"
                />
              </v-col>
              <v-col cols="12" md="3">
                <v-select
                  v-model="loraNormalize"
                  :items="['global', 'per_module', 'off']"
                  :label="t('merge_normalize')"
                  :hint="t('merge_normalize_tip')"
                  persistent-hint
                  variant="outlined"
                  density="compact"
                />
              </v-col>
              <v-col cols="12" md="3">
                <v-select
                  v-model="loraDtype"
                  :items="['bf16', 'fp16', 'fp32']"
                  :label="t('mgDtype')"
                  variant="outlined"
                  density="compact"
                  hide-details
                />
              </v-col>
            </v-row>
            <v-row>
              <v-col cols="12">
                <v-text-field
                  v-model="loraOutPath"
                  :label="t('mgOutputPath')"
                  :placeholder="t('merge_lora_out_placeholder')"
                  variant="outlined"
                  density="compact"
                  hide-details
                />
              </v-col>
            </v-row>
          </v-card-text>
          <v-card-actions>
            <v-btn
              color="primary"
              prepend-icon="mdi-merge"
              :loading="isRunning('merge-loras')"
              :disabled="selectedFiles.length < 2 || weightsError !== null"
              @click="runMergeLoras"
            >
              {{ t('merge_lora_button') }}
            </v-btn>
            <v-btn
              variant="outlined"
              prepend-icon="mdi-magnify-scan"
              :loading="analyzing"
              :disabled="selectedFiles.length < 2"
              @click="runAnalyze"
            >
              {{ t('merge_analyze_button') }}
            </v-btn>
            <span v-if="weightsError" class="text-error text-caption ml-2">{{ weightsError }}</span>
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
          <v-btn v-if="task.state === 'running' || task.state === 'pending'" icon="mdi-stop" size="small" variant="text" color="error" @click="taskStore.cancelTask(task.task_id)" />
        </template>
      </v-list-item>
    </v-list>
    <div v-else class="text-medium-emphasis text-body-2">{{ t('mgNoTasks') }}</div>
    </div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useI18n } from '../composables/useI18n'

const taskStore = useTaskStore()
const notify = useNotifyStore()
const { t } = useI18n()
taskStore.fetchTasks()

// ── State ─────────────────────────────────────────────────────

type MergeMode = 'dit' | 'loras'
type OverlapBand = 'chance' | 'reinforcing' | 'cancelling' | 'elevated' | 'colliding'

interface AdapterDir { name: string; path: string }
interface CustomDir { path: string }
interface AdapterFile { name: string; path: string; size: number; size_human: string; mtime: string }
interface ScanResult { verdict: string; counts: Record<string, number>; total_keys: number; metadata: Record<string, string> }
interface AnalyzePayload {
  verdict: 'orthogonal' | 'constructive' | 'destructive'
  ratio: number
  shared: number
  modules: number
  strongest?: [string, string, number]  // [a, b, cos]
  overlap?: [string, string, number, number, OverlapBand?, number?]  // [a, b, out, xrandom, band, cos]
}
interface AnalyzeBanner {
  type: 'success' | 'warning' | 'error'
  verdict: 'orthogonal' | 'constructive' | 'destructive' | 'failed'
  ratio: number
  shared: number
  modules: number
  cos: number
  a: string
  b: string
  strong: boolean
  overlapBand: OverlapBand
  overlap: number
  overlapXRandom: number
  overlapA: string
  overlapB: string
}

const mode = ref<MergeMode>('dit')

const serverDirs = ref<AdapterDir[]>([])
const customDirs = ref<CustomDir[]>([])
const customDirInput = ref('')
const selectedDir = ref('')
const adapterFiles = ref<AdapterFile[]>([])

const adapterDirs = computed<AdapterDir[]>(() => {
  const merged: AdapterDir[] = [...serverDirs.value]
  const seen = new Set(merged.map(d => d.path))
  for (const d of customDirs.value) {
    if (!seen.has(d.path)) {
      merged.push({ name: customDirLabel(d.path), path: d.path })
      seen.add(d.path)
    }
  }
  return merged
})

function customDirLabel(path: string) {
  return `${path} (${t('mgCustomDirSuffix')})`
}

const CUSTOM_DIRS_KEY = 'merge_custom_dirs'

function loadCustomDirs() {
  try {
    const raw = localStorage.getItem(CUSTOM_DIRS_KEY)
    if (raw) {
      const parsed = JSON.parse(raw) as unknown[]
      customDirs.value = parsed
        .filter((d): d is { path: string } =>
          typeof d === 'object' && d !== null &&
          'path' in d && typeof (d as Record<string, unknown>).path === 'string'
        )
        .map(d => ({ path: d.path }))
    }
  } catch {
    customDirs.value = []
  }
}

function saveCustomDirs() {
  localStorage.setItem(CUSTOM_DIRS_KEY, JSON.stringify(customDirs.value))
}

async function addCustomDir() {
  const raw = customDirInput.value.trim()
  if (!raw) return

  if (adapterDirs.value.some(d => d.path === raw)) {
    notify.show(t('mgCustomDirExists'), 'warning')
    return
  }

  try {
    const res = await fetch(`/api/files/browse?dir=${encodeURIComponent(raw)}&ext=${encodeURIComponent('.safetensors')}`)
    if (!res.ok) throw new Error()
    const data = await res.json()
    if (data.error) {
      notify.show(t('mgCustomDirInvalid'), 'warning')
      return
    }
    customDirs.value.push({ path: raw })
    saveCustomDirs()
    selectedDir.value = raw
    await onDirChange(raw)
    customDirInput.value = ''
    notify.show(t('mgCustomDirAdded', { path: raw }), 'success')
  } catch {
    notify.show(t('mgCustomDirInvalid'), 'warning')
  }
}

function removeCustomDir(path: string) {
  customDirs.value = customDirs.value.filter(d => d.path !== path)
  saveCustomDirs()
  if (selectedDir.value === path) {
    selectedDir.value = adapterDirs.value[0]?.path ?? ''
    onDirChange(selectedDir.value)
  }
}

// DiT mode
const selectedFile = ref<AdapterFile | null>(null)
const scanResult = ref<ScanResult | null>(null)
const baseDit = ref('models/diffusion_models/anima-base-v1.0.safetensors')
const dtype = ref('bf16')
const multiplier = ref(1.0)
const allowPartial = ref(false)
const outputPath = ref('')

// LoRA-fusion mode
const selectedFiles = ref<AdapterFile[]>([])
const loraWeights = ref('')
const loraNormalize = ref('global')
const loraDtype = ref('bf16')
const loraOutPath = ref('')
const analyzeBanner = ref<AnalyzeBanner | null>(null)
const analyzing = ref(false)
let analyzeTimer: ReturnType<typeof setInterval> | null = null

// ── Load directories ──────────────────────────────────────────

async function fetchDirs() {
  try {
    const res = await fetch('/api/merge/dirs')
    if (!res.ok) return
    const data = await res.json()
    serverDirs.value = data.dirs || []
    if (adapterDirs.value.length > 0 && !selectedDir.value) {
      selectedDir.value = adapterDirs.value[0].path
      await onDirChange(selectedDir.value)
    }
  } catch { /* ignore */ }
}

onMounted(() => {
  loadCustomDirs()
  fetchDirs()
})

onUnmounted(clearAnalyzeTimer)

// Leaving LoRA-fusion mode (or unmounting) must stop the analyze poller and
// drop its banner so no stale state carries over.
function clearAnalyzeTimer() {
  if (analyzeTimer) { clearInterval(analyzeTimer); analyzeTimer = null }
  analyzing.value = false
}

watch(mode, () => {
  clearAnalyzeTimer()
  analyzeBanner.value = null
})

async function onDirChange(dirPath: string) {
  selectedFile.value = null
  selectedFiles.value = []
  scanResult.value = null
  analyzeBanner.value = null
  clearAnalyzeTimer()
  if (!dirPath) { adapterFiles.value = []; return }
  try {
    const res = await fetch(`/api/merge/files?dir=${encodeURIComponent(dirPath)}`)
    if (!res.ok) return
    const data = await res.json()
    adapterFiles.value = data.files || []
  } catch { adapterFiles.value = [] }
}

// ── File selection ─────────────────────────────────────────────

function isFileActive(file: AdapterFile) {
  if (mode.value === 'dit') return selectedFile.value?.path === file.path
  return isFileSelected(file)
}

function isFileSelected(file: AdapterFile) {
  return selectedFiles.value.some(f => f.path === file.path)
}

async function onFileClick(file: AdapterFile) {
  if (mode.value === 'dit') {
    await selectFile(file)
  } else {
    // toggle membership
    const idx = selectedFiles.value.findIndex(f => f.path === file.path)
    if (idx >= 0) selectedFiles.value.splice(idx, 1)
    else selectedFiles.value.push(file)
    analyzeBanner.value = null  // selection changed → stale banner
  }
}

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

// ── DiT verdict display ───────────────────────────────────────

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

// ── Weights validation ────────────────────────────────────────

const parsedWeights = computed<string[] | null>(() => {
  const raw = loraWeights.value.trim()
  if (!raw) return []  // empty = all-1.0 sentinel (valid)
  const parts = raw.split(',').map(s => s.trim()).filter(s => s.length > 0)
  return parts
})

const weightsError = computed<string | null>(() => {
  if (selectedFiles.value.length < 2) return null  // button already disabled; don't double-warn
  const parts = parsedWeights.value
  if (parts === null) return null
  if (parts.length === 0) return null  // empty → all 1.0, valid
  if (parts.length !== selectedFiles.value.length) {
    return t('merge_weights_mismatch', { n: parts.length, m: selectedFiles.value.length })
  }
  if (parts.some(p => isNaN(Number(p)))) return t('merge_weights_tip')
  return null
})

function buildSharedArgs(withAnalyze: boolean): string[] | null {
  const paths = selectedFiles.value.map(f => f.path)
  const args = [...paths]
  if (withAnalyze) args.push('--analyze')
  const parts = parsedWeights.value
  if (parts && parts.length > 0) args.push('--weights', parts.join(','))
  args.push('--dtype', loraDtype.value)
  if (!withAnalyze) {
    args.push('--normalize', loraNormalize.value)
    if (loraOutPath.value.trim()) args.push('--out', loraOutPath.value.trim())
  }
  return args
}

// ── DiT merge ─────────────────────────────────────────────────

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

// ── LoRA fusion ───────────────────────────────────────────────

async function runMergeLoras() {
  if (selectedFiles.value.length < 2 || weightsError.value) return
  const args = buildSharedArgs(false)
  if (!args) return
  const taskId = await taskStore.startTask('merge-loras', args)
  if (taskId) {
    notify.show(t('notifyTaskStarted', { command: t('merge_lora_button') }), 'success')
  } else {
    notify.show(t('notifyTaskStartFailed', { command: t('merge_lora_button') }), 'error')
  }
}

async function runAnalyze() {
  if (selectedFiles.value.length < 2) return
  const args = buildSharedArgs(true)
  if (!args) return
  analyzing.value = true
  analyzeBanner.value = null
  const taskId = await taskStore.startTask('merge-loras', args)
  if (!taskId) {
    analyzing.value = false
    notify.show(t('notifyTaskStartFailed', { command: t('merge_analyze_button') }), 'error')
    return
  }
  // Poll the analyze task until terminal, then harvest the ANALYZE_RESULT trailer.
  clearAnalyzeTimer()
  analyzing.value = true  // clearAnalyzeTimer resets this; re-arm for the new poll
  analyzeTimer = setInterval(async () => {
    const done = await pollAnalyze(taskId)
    if (done) clearAnalyzeTimer()
  }, 1500)
}

async function pollAnalyze(taskId: string): Promise<boolean> {
  try {
    // Poll state via the single-task endpoint; the full (capped) output is
    // fetched exactly once, at terminal state, for the result trailer.
    const res = await fetch(`/api/tasks/${taskId}`)
    if (!res.ok) return false
    const info = await res.json()
    const state: string = info.state
    if (state === 'running' || state === 'pending') return false
    const outRes = await fetch(`/api/tasks/${taskId}/output`)
    if (!outRes.ok) {
      analyzeBanner.value = null
      return true
    }
    const data = await outRes.json()
    const lines: string[] = data.lines || []
    analyzeBanner.value = parseAnalyzeBanner(lines, state)
    return true
  } catch {
    return false
  }
}

function parseAnalyzeBanner(lines: string[], finalState: string): AnalyzeBanner | null {
  // The trailer is the LAST line starting with "ANALYZE_RESULT ".
  let payload: AnalyzePayload | null = null
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i]
    const idx = line.indexOf('ANALYZE_RESULT ')
    if (idx >= 0) {
      try {
        payload = JSON.parse(line.slice(idx + 'ANALYZE_RESULT '.length))
      } catch { /* malformed JSON — keep scanning / fall through */ }
      break
    }
  }
  if (!payload) {
    if (finalState === 'failed') {
      return {
        type: 'error', verdict: 'failed', ratio: 0, shared: 0, modules: 0,
        cos: 0, a: '', b: '', strong: false, overlapBand: 'chance',
        overlap: 0, overlapXRandom: 0, overlapA: '', overlapB: '',
      }
    }
    return null
  }
  const { verdict, ratio, shared, modules } = payload
  const cos = payload.strongest?.[2] ?? 0
  const a = payload.strongest?.[0] ?? ''
  const b = payload.strongest?.[1] ?? ''
  const overlap = payload.overlap?.[2] ?? 0
  const overlapXRandom = payload.overlap?.[3] ?? 0
  const overlapA = payload.overlap?.[0] ?? ''
  const overlapB = payload.overlap?.[1] ?? ''
  const overlapCos = payload.overlap?.[5]
    ?? ((overlapA === a && overlapB === b) || (overlapA === b && overlapB === a) ? cos : 0)
  let overlapBand = payload.overlap?.[4]
  // Backward-compatible classification for analysis trailers emitted before
  // the fork added the explicit band/cosine fields.
  if (!overlapBand) {
    if (overlapXRandom < 3) overlapBand = 'chance'
    else if (overlapCos >= 0.5) overlapBand = 'reinforcing'
    else if (overlapCos <= -0.5) overlapBand = 'cancelling'
    else overlapBand = overlapXRandom >= 8 ? 'colliding' : 'elevated'
  }
  const type = overlapBand === 'colliding' || overlapBand === 'cancelling'
    ? 'error'
    : overlapBand === 'elevated' || overlapBand === 'reinforcing'
      ? 'warning'
      : verdict === 'orthogonal' ? 'success' : (cos >= 0 ? 'warning' : 'error')
  return {
    type,
    verdict,
    ratio,
    shared,
    modules,
    cos,
    a,
    b,
    strong: Math.abs(cos) >= 0.5,
    overlapBand,
    overlap,
    overlapXRandom,
    overlapA,
    overlapB,
  }
}

// Render a structured banner as localized plain text (no v-html — filenames
// are user-controlled, so we never treat the result as HTML).
function bannerText(b: AnalyzeBanner): string {
  if (b.verdict === 'failed') {
    return t('notifyTaskStartFailed', { command: t('merge_analyze_button') })
  }
  const strength = b.strong ? t('merge_analysis_strong') : t('merge_analysis_moderate')
  const common = {
    ratio: b.ratio, shared: b.shared, modules: b.modules, a: b.a, b: b.b, strength,
    oa: b.overlapA, ob: b.overlapB, overlap: b.overlap, xrandom: b.overlapXRandom,
  }
  if (b.overlapBand !== 'chance') {
    return t(`merge_analysis_overlap_${b.overlapBand}`, common)
  }
  if (b.verdict === 'orthogonal') return t('merge_analysis_safe', common)
  if (b.cos >= 0) return t('merge_analysis_reinforce', common)
  return t('merge_analysis_cancel', common)
}

// ── Helpers ───────────────────────────────────────────────────

function formatDate(iso: string) {
  try { return new Date(iso).toLocaleDateString() } catch { return iso }
}

function isRunning(command: string) {
  return taskStore.tasks.some(tp => tp.command === command && tp.state === 'running')
}

const mergeTasks = computed(() =>
  taskStore.tasks.filter(tp => tp.command === 'merge' || tp.command === 'merge-loras')
)

function stateColor(state: string) {
  if (state === 'running' || state === 'stopping') return 'info'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}
</script>

<style scoped>
/* Fill the v-main flex container so the page owns a real height. */
.merge-page {
  flex: 1 1 0;
  min-height: 0;
}

/* The scrollable region below the page header. Inside we still want the
   header rows (mode toggle, two columns) and the active-tasks list to share
   one scroll context, so users can reach every task on short viewports. */
.merge-scroll {
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
}

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
