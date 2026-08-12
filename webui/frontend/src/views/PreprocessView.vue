<template>
  <v-container fluid class="pa-4 preprocess-page">
    <div class="text-h5 mb-1">{{ t('ppTitle') }}</div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('ppSubtitle') }}</div>

    <!-- Isolated preprocess run -->
    <v-card variant="tonal" class="mb-4">
      <v-card-title class="text-subtitle-1 d-flex align-center">
        <v-icon icon="mdi-source-branch" class="mr-2" />
        {{ t('ppRunSelection') }}
        <v-spacer />
        <v-btn
          icon="mdi-refresh"
          size="small"
          variant="text"
          :loading="runsLoading"
          :title="t('ppRefreshRuns')"
          @click="fetchRuns(false)"
        />
      </v-card-title>
      <v-card-text>
        <v-select
          v-model="selectedRun"
          :items="runItems"
          item-title="title"
          item-value="manifest"
          :label="t('ppRunManifest')"
          :loading="runsLoading"
          :no-data-text="t('ppNoRuns')"
          clearable
          density="compact"
          hide-details="auto"
        />
        <div v-if="selectedRunInfo" class="d-flex flex-wrap align-center ga-2 mt-3">
          <v-chip
            size="small"
            :color="runStatusColor(selectedRunInfo.status)"
            variant="tonal"
          >
            {{ t('ppRunStatus') }}: {{ selectedRunInfo.status }}
          </v-chip>
          <v-chip size="small" variant="tonal">
            {{ selectedRunInfo.source_group || selectedRunInfo.run_id }}
          </v-chip>
        </div>
        <div v-if="selectedRun" class="text-caption text-medium-emphasis mt-2 run-manifest-path">
          {{ selectedRun }}
        </div>
        <v-alert
          v-if="selectedRunInfo?.error"
          type="error"
          variant="tonal"
          density="compact"
          class="mt-3"
        >
          {{ selectedRunInfo.error }}
        </v-alert>
        <div v-if="!selectedRun" class="text-caption text-medium-emphasis mt-2">
          {{ t('ppRunAutomatic') }}
        </div>
      </v-card-text>
    </v-card>

    <!-- Status Dashboard -->
    <v-card variant="tonal" class="mb-4">
      <v-card-title class="text-subtitle-1">
        <v-icon icon="mdi-chart-box-outline" class="mr-2" />
        {{ t('ppStatus') }}
      </v-card-title>
      <v-card-text>
        <div class="d-flex flex-wrap ga-2">
          <v-chip prepend-icon="mdi-resize" :color="status.resized > 0 ? 'success' : 'default'" variant="tonal">
            {{ t('ppStatusResized') }}: {{ status.resized }}
          </v-chip>
          <v-chip prepend-icon="mdi-cached" :color="status.cache.latents > 0 ? 'success' : 'default'" variant="tonal">
            {{ t('ppStatusLatents') }}: {{ status.cache.latents }}
          </v-chip>
          <v-chip prepend-icon="mdi-text-box-outline" :color="status.cache.te > 0 ? 'success' : 'default'" variant="tonal">
            {{ t('ppStatusTe') }}: {{ status.cache.te }}
          </v-chip>
          <v-chip prepend-icon="mdi-eye-outline" :color="status.cache.pe > 0 ? 'success' : 'default'" variant="tonal">
            {{ t('ppStatusPe') }}: {{ status.cache.pe }}
          </v-chip>
          <v-chip prepend-icon="mdi-image-filter-center-focus" :color="status.masks > 0 ? 'success' : 'default'" variant="tonal">
            {{ t('ppStatusMasks') }}: {{ status.masks }}
          </v-chip>
          <v-chip prepend-icon="mdi-image-multiple-outline" :color="status.cond_resized > 0 ? 'success' : 'default'" variant="tonal">
            {{ t('ppStatusCondResized') }}: {{ status.cond_resized }}
          </v-chip>
        </div>
      </v-card-text>
    </v-card>

    <!-- Dataset Paths -->
    <v-card variant="tonal" class="mb-4">
      <v-card-title class="text-subtitle-1">
        <v-icon icon="mdi-folder-outline" class="mr-2" />
        {{ t('ppDatasetPaths') }}
      </v-card-title>
      <v-card-text>
        <v-text-field
          v-model="paths.source"
          :label="t('ppPathSource')"
          :loading="pathsLoading"
          :readonly="Boolean(selectedRun)"
          density="compact"
          hide-details="auto"
          class="mb-2"
        />
        <v-text-field
          v-model="paths.resized"
          :label="t('ppPathResized')"
          :readonly="Boolean(selectedRun)"
          density="compact"
          hide-details="auto"
          class="mb-2"
        />
        <v-text-field
          v-model="paths.cache"
          :label="t('ppPathCache')"
          :readonly="Boolean(selectedRun)"
          density="compact"
          hide-details="auto"
          class="mb-2"
        />
        <v-text-field
          v-model="paths.condSource"
          :label="t('ppPathCondSource')"
          :readonly="Boolean(selectedRun)"
          density="compact"
          hide-details="auto"
          class="mb-2"
        />
        <v-text-field
          v-model="paths.condResized"
          :label="t('ppPathCondResized')"
          :readonly="Boolean(selectedRun)"
          density="compact"
          hide-details="auto"
        />
      </v-card-text>
      <v-card-actions>
        <v-btn color="primary" size="small" prepend-icon="mdi-content-save" :loading="pathsSaving" :disabled="Boolean(selectedRun)" @click="savePaths">
          {{ t('ppSavePaths') }}
        </v-btn>
      </v-card-actions>
    </v-card>

    <!-- Settings Panel -->
    <v-expansion-panels class="mb-4">
      <v-expansion-panel>
        <v-expansion-panel-title>
          <v-icon icon="mdi-cog-outline" class="mr-2" />
          {{ t('ppSettings') }}
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <v-row>
            <!-- Resize Settings -->
            <v-col cols="12" md="3">
              <div class="text-subtitle-2 mb-2">
                {{ t(settings.multires_per_image ? 'ppResizeTargetResAll' : 'ppResizeTargetRes') }}
              </div>
              <div class="d-flex flex-wrap ga-1">
                <v-chip
                  v-for="edge in TARGET_RES_OPTIONS"
                  :key="edge"
                  size="small"
                  :variant="settings.target_res.includes(edge) ? 'flat' : 'tonal'"
                  :color="settings.target_res.includes(edge) ? 'primary' : 'default'"
                  @click="toggleTier(edge)"
                >
                  {{ edge }}
                </v-chip>
              </div>
              <div v-if="settings.target_res.length === 0" class="text-caption text-error mt-1">
                {{ t('ppTargetResNone') }}
              </div>
              <v-switch
                v-model="settings.multires_per_image"
                :label="t('ppMultiresPerImage')"
                :hint="t('ppMultiresPerImageHint')"
                :disabled="settings.target_res.length < 2"
                persistent-hint
                density="compact"
                class="mt-2"
              />
            </v-col>

            <!-- SAM Settings -->
            <v-col cols="12" md="3">
              <div class="text-subtitle-2 mb-2">{{ t('ppSamGroup') }}</div>
              <v-switch v-model="settings.run_sam_mask" :label="t('ppRunSamMask')" density="compact" hide-details class="mb-2" />
              <v-textarea
                v-model="samPromptsText"
                :label="t('ppSamPrompts')"
                :hint="t('ppSamPromptsHint')"
                rows="3"
                density="compact"
                hide-details="auto"
                class="mb-2"
              />
              <v-text-field
                v-model.number="settings.sam.threshold"
                :label="t('ppSamThreshold')"
                type="number"
                step="0.05"
                min="0"
                max="1"
                density="compact"
                hide-details="auto"
                class="mb-2"
              />
              <v-text-field
                v-model.number="settings.sam.dilate"
                :label="t('ppSamDilate')"
                type="number"
                step="1"
                min="0"
                max="64"
                density="compact"
                hide-details="auto"
              />
            </v-col>

            <!-- MIT Settings -->
            <v-col cols="12" md="3">
              <div class="text-subtitle-2 mb-2">{{ t('ppMitGroup') }}</div>
              <v-switch v-model="settings.run_mit_mask" :label="t('ppRunMitMask')" density="compact" hide-details class="mb-2" />
              <v-text-field
                v-model.number="settings.mit_text_threshold"
                :label="t('ppMitThreshold')"
                type="number"
                step="0.05"
                min="0"
                max="1"
                density="compact"
                hide-details="auto"
                class="mb-2"
              />
              <v-text-field
                v-model.number="settings.mit_dilate"
                :label="t('ppMitDilate')"
                type="number"
                step="1"
                min="0"
                max="64"
                density="compact"
                hide-details="auto"
              />
              <v-switch
                v-model="settings.mit_ctd_gate"
                :label="t('ppMitCtdGate')"
                :hint="t('ppMitCtdGateHint')"
                persistent-hint
                density="compact"
                hide-details="auto"
              />
            </v-col>

            <!-- Caption Settings -->
            <v-col cols="12" md="3">
              <div class="text-subtitle-2 mb-2">{{ t('ppCaptionGroup') }}</div>
              <v-text-field
                v-model.number="settings.caption_shuffle_variants"
                :label="t('ppShuffleVariants')"
                type="number"
                step="1"
                min="0"
                max="64"
                density="compact"
                hide-details="auto"
                class="mb-2"
              />
              <v-text-field
                v-model.number="settings.caption_tag_dropout_rate"
                :label="t('ppTagDropout')"
                type="number"
                step="0.05"
                min="0"
                max="1"
                density="compact"
                hide-details="auto"
              />
            </v-col>
          </v-row>

          <div class="mt-3">
            <v-btn color="primary" size="small" prepend-icon="mdi-content-save" @click="saveSettings">
              {{ t('ppSaveSettings') }}
            </v-btn>
          </div>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>

    <!-- Task Cards -->
    <v-row>
      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-resize" class="mr-2" />
            {{ t('ppResize') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppResizeDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn
              color="primary"
              :disabled="settings.target_res.length === 0"
              :loading="isRunning('preprocess-resize')"
              @click="runTask('preprocess-resize')"
            >
              {{ t('ppRunResize') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-cached" class="mr-2" />
            {{ t('ppCacheVae') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppCacheVaeDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('preprocess-vae')" @click="runTask('preprocess-vae')">
              {{ t('ppRunVae') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-text-box-outline" class="mr-2" />
            {{ t('ppCacheTe') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppCacheTeDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('preprocess-te')" @click="runTask('preprocess-te')">
              {{ t('ppRunTe') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-eye-outline" class="mr-2" />
            {{ t('ppCachePe') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppCachePeDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('preprocess-pe')" @click="runTask('preprocess-pe')">
              {{ t('ppRunPe') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-image-filter-center-focus" class="mr-2" />
            {{ t('ppMask') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppMaskDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('mask')" @click="runTask('mask')">
              {{ t('ppRunMask') }}
            </v-btn>
            <v-btn color="error" variant="text" size="small" :loading="isRunning('mask-clean')" @click="runTask('mask-clean')">
              {{ t('ppClean') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card variant="tonal" color="primary">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-pipe" class="mr-2" />
            {{ t('ppPipeline') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppPipelineDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('preprocess')" @click="runTask('preprocess')">
              {{ t('ppRunAll') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>

    <!-- Conditioning Preprocessing -->
    <v-divider class="my-4" />
    <div class="text-subtitle-1 mb-2">
      <v-icon icon="mdi-image-multiple-outline" class="mr-1" />
      {{ t('ppCondTitle') }}
    </div>
    <v-row>
      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-resize" class="mr-2" />
            {{ t('ppCondResize') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppCondResizeDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('preprocess-cond-resize')" @click="runTask('preprocess-cond-resize')">
              {{ t('ppCondRunResize') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-cached" class="mr-2" />
            {{ t('ppCondCacheVae') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppCondCacheVaeDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('preprocess-cond-vae')" @click="runTask('preprocess-cond-vae')">
              {{ t('ppCondRunCacheVae') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>

    <v-divider class="my-4" />

    <div class="d-flex align-center mb-2">
      <div class="text-subtitle-1">{{ t('ppActiveTasks') }}</div>
      <v-spacer />
      <v-btn variant="text" size="small" prepend-icon="mdi-refresh" @click="refresh">
        {{ t('ppRefresh') }}
      </v-btn>
    </div>

    <v-list v-if="preprocessTasks.length > 0" density="compact">
      <v-list-item
        v-for="task in preprocessTasks"
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
    <div v-else class="text-medium-emphasis text-body-2">{{ t('ppNoTasks') }}</div>
  </v-container>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useConfigStore } from '../stores/config'
import { useI18n } from '../composables/useI18n'
import { readPreprocessRun, writePreprocessRun } from '../composables/usePreprocessRunStorage'

const taskStore = useTaskStore()
const notify = useNotifyStore()
const configStore = useConfigStore()
const { t } = useI18n()
taskStore.fetchTasks()

const preprocessCommands = [
  'preprocess', 'preprocess-resize', 'preprocess-vae', 'preprocess-te',
  'preprocess-pe', 'preprocess-cond-resize', 'preprocess-cond-vae',
  'mask', 'mask-clean',
]

interface PreprocessRunSummary {
  manifest: string
  run_id: string
  source?: string | null
  source_group?: string | null
  config_hash?: string | null
  status: string
  complete: boolean
  artifacts?: Record<string, number>
  updated_at?: string | number | null
  error?: string | null
}

const runs = ref<PreprocessRunSummary[]>([])
function readSelectedRun(): string | null {
  return readPreprocessRun()
}

const selectedRun = ref<string | null>(readSelectedRun())
const runsLoading = ref(false)
const selectedRunInfo = computed(() =>
  runs.value.find(run => run.manifest === selectedRun.value) ?? null
)
const runItems = computed(() => runs.value.map(run => ({
  manifest: run.manifest,
  title: `${run.source_group || run.run_id} · ${run.config_hash || ''} · ${run.status}`,
})))

function runStatusColor(state: string) {
  if (state === 'ready') return 'success'
  if (state === 'running') return 'info'
  if (state === 'failed' || state === 'invalid') return 'error'
  return undefined
}

// ── Status dashboard ──────────────────────────────────────────

const status = reactive({
  resized: 0,
  masks: 0,
  cache: { latents: 0, te: 0, pe: 0 },
  cond_resized: 0,
})

async function fetchStatus() {
  try {
    const qs = new URLSearchParams()
    if (selectedRun.value) qs.set('manifest', selectedRun.value)
    else {
      if (configStore.variant) qs.set('variant', configStore.variant)
      if (configStore.preset) qs.set('preset', configStore.preset)
    }
    const url = '/api/preprocess/status' + (qs.toString() ? '?' + qs : '')
    const res = await fetch(url)
    if (!res.ok) return
    const data = await res.json()
    status.resized = data.resized ?? 0
    status.masks = data.masks ?? 0
    status.cache.latents = data.cache?.latents ?? 0
    status.cache.te = data.cache?.te ?? 0
    status.cache.pe = data.cache?.pe ?? 0
    status.cond_resized = data.cond_resized ?? 0
  } catch { /* ignore */ }
}

onMounted(fetchStatus)

// Refresh status when tasks finish
const runningCount = computed(() =>
  taskStore.tasks.filter(tp => preprocessCommands.includes(tp.command) && tp.state === 'running').length
)
watch(runningCount, (newVal, oldVal) => {
  if (oldVal > newVal) {
    fetchRuns(true)
    fetchStatus()
  }
})

// ── Settings ──────────────────────────────────────────────────

const defaultSettings = () => ({
  sam: { prompts: ['speech bubble', 'text bubble'], threshold: 0.5, dilate: 5 },
  run_sam_mask: true,
  run_mit_mask: true,
  caption_shuffle_variants: 4,
  caption_tag_dropout_rate: 0.1,
  mit_text_threshold: 0.8,
  mit_dilate: 5,
  mit_ctd_gate: true,
  // Free-fit tier edges (allowed: 512 768 896 1024 1280 1536). This is the
  // value resize actually consumes — the old resize_resolution scalar was a
  // no-op under free-fit. Persisted to configs/preprocess.toml.
  target_res: [1024] as number[],
  multires_per_image: false,
})

const settings = reactive(defaultSettings())
let settingsHydrating = false

// Allowed free-fit tier edges (mirror of ALLOWED_TARGET_RES in the backend).
const TARGET_RES_OPTIONS = [512, 768, 896, 1024, 1280, 1536]

function toggleTier(edge: number) {
  const idx = settings.target_res.indexOf(edge)
  if (idx >= 0) settings.target_res.splice(idx, 1)
  else settings.target_res.push(edge)
  if (settings.target_res.length < 2) settings.multires_per_image = false
}

const samPromptsText = computed({
  get: () => settings.sam.prompts.join('\n'),
  set: (val: string) => { settings.sam.prompts = val.split('\n').map(s => s.trim()).filter(Boolean) },
})

async function fetchSettings() {
  settingsHydrating = true
  try {
    const res = await fetch('/api/preprocess/settings')
    if (!res.ok) return
    const data = await res.json()
    Object.assign(settings.sam, data.sam ?? {})
    settings.run_sam_mask = data.run_sam_mask ?? true
    settings.run_mit_mask = data.run_mit_mask ?? true
    settings.caption_shuffle_variants = data.caption_shuffle_variants ?? 4
    settings.caption_tag_dropout_rate = data.caption_tag_dropout_rate ?? 0.1
    settings.mit_text_threshold = data.mit_text_threshold ?? 0.8
    settings.mit_dilate = data.mit_dilate ?? 5
    settings.mit_ctd_gate = data.mit_ctd_gate ?? true
    // Backend normalizes (drops invalid edges, guarantees ≥[1024]); trust it.
    settings.target_res = Array.isArray(data.target_res) && data.target_res.length
      ? data.target_res
      : [1024]
    settings.multires_per_image = data.multires_per_image ?? false
  } catch { /* ignore */ }
  finally { settingsHydrating = false }
}

onMounted(fetchSettings)

watch(settings, () => {
  if (!settingsHydrating && selectedRun.value) selectedRun.value = null
}, { deep: true, flush: 'sync' })

// ── Dataset paths ──────────────────────────────────────────────

const paths = reactive({ source: '', resized: '', cache: '', condSource: '', condResized: '' })
const pathsLoading = ref(false)
const pathsSaving = ref(false)

async function ensureVariant() {
  if (configStore.variant) return
  if (!configStore.methods.length) await configStore.fetchMethods()
  if (!configStore.methods.length) return
  const m = configStore.methods[0]
  if (!configStore.variants.length) await configStore.fetchVariants(m)
  const v = configStore.variants[0] || m
  await configStore.fetchMerged(v, configStore.preset || 'default')
}

async function fetchPaths() {
  pathsLoading.value = true
  try {
    await ensureVariant()
    const qs = new URLSearchParams()
    if (selectedRun.value) qs.set('manifest', selectedRun.value)
    else {
      if (configStore.variant) qs.set('variant', configStore.variant)
      if (configStore.preset) qs.set('preset', configStore.preset)
    }
    const url = '/api/preprocess/paths' + (qs.toString() ? '?' + qs : '')
    const res = await fetch(url)
    if (!res.ok) return
    const data = await res.json()
    paths.source = data.source_image_dir ?? ''
    paths.resized = data.resized_image_dir ?? ''
    paths.cache = data.lora_cache_dir ?? ''
    paths.condSource = data.conditioning_data_dir ?? ''
    paths.condResized = data.conditioning_resized_dir ?? ''
  } catch { /* ignore */ }
  finally { pathsLoading.value = false }
}

async function savePaths() {
  if (selectedRun.value) return
  pathsSaving.value = true
  try {
    await ensureVariant()
    if (!configStore.variant) {
      notify.show(t('notifyConfigSaveFailed'), 'error')
      return
    }
    const qs = new URLSearchParams()
    qs.set('variant', configStore.variant)
    const url = '/api/preprocess/paths?' + qs
    const res = await fetch(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source_image_dir: paths.source,
        resized_image_dir: paths.resized,
        lora_cache_dir: paths.cache,
        conditioning_data_dir: paths.condSource,
        conditioning_resized_dir: paths.condResized,
      }),
    })
    if (res.ok) {
      const data = await res.json()
      paths.source = data.source_image_dir ?? paths.source
      paths.resized = data.resized_image_dir ?? paths.resized
      paths.cache = data.lora_cache_dir ?? paths.cache
      paths.condSource = data.conditioning_data_dir ?? paths.condSource
      paths.condResized = data.conditioning_resized_dir ?? paths.condResized
      notify.show(t('ppPathsSaved'), 'success')
      fetchStatus()
    } else {
      notify.show(t('notifyConfigSaveFailed'), 'error')
    }
  } catch {
    notify.show(t('notifyConfigSaveFailed'), 'error')
  }
  finally { pathsSaving.value = false }
}

async function fetchRuns(selectNewest: boolean) {
  runsLoading.value = true
  try {
    // Runs are shared across training methods and source directories. Do not
    // filter by the currently selected run's source here: a persisted run can
    // belong to an older dataset, and filtering by it would hide a newly
    // completed run until the user manually cleared browser storage.
    const res = await fetch('/api/preprocess/runs')
    if (!res.ok) return
    const data = await res.json()
    runs.value = Array.isArray(data) ? data : []
    const currentExists = runs.value.some(run => run.manifest === selectedRun.value)
    if (!currentExists || selectNewest) {
      selectedRun.value = runs.value.find(run => run.complete && run.status === 'ready')?.manifest ?? null
    }
  } catch { /* ignore */ }
  finally { runsLoading.value = false }
}

watch(selectedRun, async () => {
  writePreprocessRun(selectedRun.value)
  await Promise.all([fetchStatus(), fetchPaths()])
})

watch([() => configStore.variant, () => configStore.preset], async () => {
  // A preprocessing manifest is shared across training methods/presets. Keep
  // the selection stable while refreshing the source-filtered run list.
  await fetchRuns(false)
})

onMounted(async () => {
  await fetchPaths()
  await fetchRuns(false)
})

async function saveSettings(): Promise<boolean> {
  try {
    const res = await fetch('/api/preprocess/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    })
    if (res.ok) {
      notify.show(t('ppSettingsSaved'), 'success')
      return true
    } else {
      notify.show(t('notifyConfigSaveFailed'), 'error')
    }
  } catch {
    notify.show(t('notifyConfigSaveFailed'), 'error')
  }
  return false
}

// ── Task management ──────────────────────────────────────────

const preprocessTasks = computed(() =>
  taskStore.tasks.filter(tp => preprocessCommands.includes(tp.command))
)

function isRunning(command: string) {
  return taskStore.tasks.some(tp => tp.command === command && tp.state === 'running')
}

function stateColor(state: string) {
  if (state === 'running' || state === 'stopping') return 'info'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}

async function runTask(command: string) {
  // Save settings before running tasks that consume them, so the on-disk
  // config / env vars are current. resize writes target_res to
  // configs/preprocess.toml; mask/te forward their knobs as env vars.
  const settingsConsumers = [
    'preprocess-resize',
    'preprocess-vae',
    'preprocess-cond-resize',
    'preprocess-cond-vae',
    'mask',
    'preprocess-te',
    'preprocess',
  ]
  if (settingsConsumers.includes(command)) {
    if (!(await saveSettings())) return
  }

  // Build CLI args for tasks that accept them
  const args: string[] = []

  // Pass relevant env vars for tasks that read them
  const env: Record<string, string> = {}
  if (configStore.variant) {
    env.METHOD = configStore.variant
    env.METHODS_SUBDIR = 'gui-methods'
  }
  if (configStore.preset) {
    env.PRESET = configStore.preset
  }
  // Forward the chosen free-fit tiers so the resize step uses exactly what
  // the UI shows (TARGET_RES env wins over the merged config in
  // scripts/tasks/preprocess.py::_target_res_args). Space-separated edges.
  const multiresConsumers = [
    'preprocess-resize',
    'preprocess-vae',
    'preprocess-cond-resize',
    'preprocess-cond-vae',
    'preprocess',
  ]
  if (multiresConsumers.includes(command)) {
    env.TARGET_RES = settings.target_res.join(' ')
    env.MULTIRES_PER_IMAGE = settings.multires_per_image ? '1' : '0'
  }
  if (['mask', 'preprocess'].includes(command)) {
    env.MIT_TEXT_THRESHOLD = String(settings.mit_text_threshold)
    env.MIT_DILATE = String(settings.mit_dilate)
    env.MIT_CTD_GATE = settings.mit_ctd_gate ? '1' : '0'
    env.RUN_SAM_MASK = settings.run_sam_mask ? '1' : '0'
    env.RUN_MIT_MASK = settings.run_mit_mask ? '1' : '0'
  }
  if (['preprocess-te', 'preprocess'].includes(command)) {
    env.CAPTION_SHUFFLE_VARIANTS = String(settings.caption_shuffle_variants)
    env.CAPTION_TAG_DROPOUT_RATE = String(settings.caption_tag_dropout_rate)
  }
  if (['preprocess-cond-resize', 'preprocess-cond-vae'].includes(command)) {
    if (paths.condSource) env.CONDITIONING_DATA_DIR = paths.condSource
    if (paths.condResized) env.CONDITIONING_RESIZED_DIR = paths.condResized
  }
  if (selectedRun.value) {
    env.PREPROCESS_RUN = selectedRun.value
  }

  const taskId = await taskStore.startTask(command, args, Object.keys(env).length > 0 ? env : undefined)
  if (taskId) {
    notify.show(t('notifyTaskStarted', { command }), 'success')
  } else {
    notify.show(t('notifyTaskStartFailed', { command }), 'error')
  }
}

function refresh() {
  taskStore.fetchTasks()
  fetchStatus()
  fetchRuns(false)
}
</script>

<style scoped>
.preprocess-page {
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
}

.run-manifest-path {
  overflow-wrap: anywhere;
}

/* Pipeline action cards: hover transition */
:deep(.v-card) {
  transition: border-color 0.2s, box-shadow 0.2s;
}

/* Status chips: ember tint for active/selected */
:deep(.v-chip--selected),
:deep(.v-chip[aria-selected="true"]) {
  background: rgba(199, 91, 26, 0.12) !important;
  border-color: var(--forge-ember) !important;
}

/* Progress bars: smooth border radius */
:deep(.v-progress-linear) {
  border-radius: 4px;
}
</style>
