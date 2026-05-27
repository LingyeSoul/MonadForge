<template>
  <v-container fluid class="pa-4 preprocess-page">
    <div class="text-h5 mb-1">{{ t('ppTitle') }}</div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('ppSubtitle') }}</div>

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
          density="compact"
          hide-details="auto"
          class="mb-2"
        />
        <v-text-field
          v-model="paths.resized"
          :label="t('ppPathResized')"
          density="compact"
          hide-details="auto"
          class="mb-2"
        />
        <v-text-field
          v-model="paths.cache"
          :label="t('ppPathCache')"
          density="compact"
          hide-details="auto"
          class="mb-2"
        />
        <v-text-field
          v-model="paths.condSource"
          :label="t('ppPathCondSource')"
          density="compact"
          hide-details="auto"
          class="mb-2"
        />
        <v-text-field
          v-model="paths.condResized"
          :label="t('ppPathCondResized')"
          density="compact"
          hide-details="auto"
        />
      </v-card-text>
      <v-card-actions>
        <v-btn color="primary" size="small" prepend-icon="mdi-content-save" :loading="pathsSaving" @click="savePaths">
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
              <div class="text-subtitle-2 mb-2">{{ t('ppResizeGroup') }}</div>
              <v-text-field
                v-model.number="settings.resize_resolution"
                :label="t('ppResizeResolution')"
                type="number"
                step="64"
                min="256"
                max="2048"
                density="compact"
                hide-details="auto"
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
            <v-btn color="primary" :loading="isRunning('preprocess-resize')" @click="runTask('preprocess-resize')">
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
          <v-btn v-if="task.state === 'running'" icon="mdi-stop" size="small" variant="text" color="error" @click="taskStore.cancelTask(task.task_id)" />
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

// ── Status dashboard ──────────────────────────────────────────

const status = reactive({
  resized: 0,
  masks: 0,
  cache: { latents: 0, te: 0, pe: 0 },
  cond_resized: 0,
})

async function fetchStatus() {
  try {
    const v = configStore.variant
    const p = configStore.preset
    const qs = new URLSearchParams()
    if (v) qs.set('variant', v)
    if (p) qs.set('preset', p)
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
  if (oldVal > newVal) fetchStatus()
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
  resize_resolution: 1024,
})

const settings = reactive(defaultSettings())

const samPromptsText = computed({
  get: () => settings.sam.prompts.join('\n'),
  set: (val: string) => { settings.sam.prompts = val.split('\n').map(s => s.trim()).filter(Boolean) },
})

async function fetchSettings() {
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
    settings.resize_resolution = data.resize_resolution ?? 1024
  } catch { /* ignore */ }
}

onMounted(fetchSettings)

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
    const v = configStore.variant
    const p = configStore.preset
    const qs = new URLSearchParams()
    if (v) qs.set('variant', v)
    if (p) qs.set('preset', p)
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

onMounted(fetchPaths)

async function saveSettings() {
  try {
    const res = await fetch('/api/preprocess/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    })
    if (res.ok) {
      notify.show(t('ppSettingsSaved'), 'success')
    } else {
      notify.show(t('notifyConfigSaveFailed'), 'error')
    }
  } catch {
    notify.show(t('notifyConfigSaveFailed'), 'error')
  }
}

// ── Task management ──────────────────────────────────────────

const preprocessTasks = computed(() =>
  taskStore.tasks.filter(tp => preprocessCommands.includes(tp.command))
)

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
  // Save settings before running mask/te tasks so env vars are current
  if (['mask', 'preprocess-te', 'preprocess'].includes(command)) {
    await saveSettings()
  }

  // Build CLI args for tasks that accept them
  const args: string[] = []
  if (command === 'preprocess-resize') {
    args.push('--resolution', String(settings.resize_resolution))
  }

  // Pass relevant env vars for tasks that read them
  const env: Record<string, string> = {}
  if (configStore.variant) {
    env.METHOD = configStore.variant
    env.METHODS_SUBDIR = 'gui-methods'
  }
  if (configStore.preset) {
    env.PRESET = configStore.preset
  }
  if (['mask', 'preprocess'].includes(command)) {
    env.MIT_TEXT_THRESHOLD = String(settings.mit_text_threshold)
    env.MIT_DILATE = String(settings.mit_dilate)
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
}
</script>

<style scoped>
.preprocess-page {
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
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
