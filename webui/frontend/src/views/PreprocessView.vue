<template>
  <v-container fluid class="pa-4 preprocess-page">
    <WorkspaceHeading :title="t('ppTitle')" :subtitle="subtitle">
      <v-btn data-testid="preprocess-guide" variant="text" size="small" prepend-icon="mdi-book-open-page-variant-outline" @click="openPreprocessHelp()">{{ t('cfgOpenGuide') }}</v-btn>
    </WorkspaceHeading>
    <div class="mb-4" style="max-width: 480px"><RuntimeSelector :disabled="taskStore.loading" /></div>

    <QwenPreprocessContent v-if="workspace.family !== 'anima'" ref="modelPreprocess" :key="workspace.family" />
    <template v-else>
    <v-alert v-if="loadError" type="error" variant="tonal" class="mb-4" data-testid="preprocess-load-error">
      {{ loadError }}
      <template #append><v-btn data-testid="preprocess-retry" variant="text" size="small" :loading="refreshing" @click="refresh">{{ t('cfgReload') }}</v-btn></template>
    </v-alert>
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
          :aria-label="t('ppRefreshRuns')"
          @click="fetchRuns(false)"
        />
      </v-card-title>
      <v-card-text>
        <v-select
          :model-value="selectedRun"
          @update:model-value="selectRun"
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
      <v-card-title class="text-subtitle-1 d-flex align-center">
        <v-icon icon="mdi-chart-box-outline" class="mr-2" />
        {{ t('ppStatus') }}<v-spacer />
        <v-btn data-testid="anima-cache-refresh" icon="mdi-refresh" size="small" variant="text" :loading="refreshing" :aria-label="t('ppRefresh')" @click="refresh" />
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
    <v-card variant="tonal" class="mb-4 preprocess-path-card">
      <v-card-title class="text-subtitle-1">
        <v-icon icon="mdi-folder-outline" class="mr-2" />
        {{ t('ppDatasetPaths') }}
      </v-card-title>
      <v-card-text>
        <template v-if="selectedRun">
          <p class="text-caption text-medium-emphasis mb-3">{{ t('ppRunSelection') }} · {{ t('cfReadOnly') }}</p>
          <dl class="preprocess-run-paths">
            <div v-for="entry in pathEntries" :key="entry.label" :class="{ 'preprocess-path-wide': entry.source }">
              <dt>{{ entry.label }}</dt>
              <dd>{{ entry.value || '—' }}</dd>
            </div>
          </dl>
        </template>
        <div v-else class="preprocess-path-fields">
          <ConfigField v-for="entry in pathEntries" :key="entry.key" :field="pathField(entry)" :model-value="paths[entry.key]"
            :class="{ 'preprocess-path-wide': entry.source }" @update="paths[entry.key] = String($event ?? '')" @help-click="openPreprocessHelp(entry.key)" />
        </div>
      </v-card-text>
      <v-card-actions v-if="!selectedRun" class="flex-wrap ga-2">
        <v-btn data-testid="anima-paths-save" color="primary" size="small" prepend-icon="mdi-content-save" :loading="pathsSaving" :disabled="!pathsDirty || pathsLoading || Boolean(submitting)" @click="savePaths">
          {{ t('ppSavePaths') }}
        </v-btn>
        <v-btn data-testid="anima-paths-cancel" size="small" :disabled="!pathsDirty || pathsSaving || Boolean(submitting)" @click="discardPaths">{{ t('cfCancel') }}</v-btn>
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
          <v-row class="preprocess-settings-grid">
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

          <div class="d-flex flex-wrap ga-2 mt-3">
            <v-btn data-testid="anima-settings-save" color="primary" size="small" prepend-icon="mdi-content-save" :loading="settingsSaving" :disabled="!settingsDirty || Boolean(submitting)" @click="saveSettings">
              {{ t('ppSaveSettings') }}
            </v-btn>
            <v-btn data-testid="anima-settings-cancel" size="small" variant="text" :disabled="!settingsDirty || settingsSaving || Boolean(submitting)" @click="discardSettings">{{ t('cfCancel') }}</v-btn>
          </div>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>

    <!-- Task Cards -->
    <v-row class="preprocess-task-grid">
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
              :disabled="settings.target_res.length === 0 || Boolean(submitting && submitting !== 'preprocess-resize')"
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
            <v-btn color="primary" :loading="isRunning('preprocess-vae')" :disabled="Boolean(submitting && submitting !== 'preprocess-vae')" @click="runTask('preprocess-vae')">
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
            <v-btn color="primary" :loading="isRunning('preprocess-te')" :disabled="Boolean(submitting && submitting !== 'preprocess-te')" @click="runTask('preprocess-te')">
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
            <v-btn color="primary" :loading="isRunning('preprocess-pe')" :disabled="Boolean(submitting && submitting !== 'preprocess-pe')" @click="runTask('preprocess-pe')">
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
            <v-btn color="primary" :loading="isRunning('mask')" :disabled="Boolean(submitting && submitting !== 'mask')" @click="runTask('mask')">
              {{ t('ppRunMask') }}
            </v-btn>
            <v-btn color="error" variant="text" size="small" :loading="isRunning('mask-clean')" :disabled="Boolean(submitting && submitting !== 'mask-clean')" @click="runTask('mask-clean')">
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
            <v-btn color="primary" :loading="isRunning('preprocess')" :disabled="Boolean(submitting && submitting !== 'preprocess')" @click="runTask('preprocess')">
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
    <v-row class="preprocess-task-grid">
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
            <v-btn color="primary" :loading="isRunning('preprocess-cond-resize')" :disabled="Boolean(submitting && submitting !== 'preprocess-cond-resize')" @click="runTask('preprocess-cond-resize')">
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
            <v-btn color="primary" :loading="isRunning('preprocess-cond-vae')" :disabled="Boolean(submitting && submitting !== 'preprocess-cond-vae')" @click="runTask('preprocess-cond-vae')">
              {{ t('ppCondRunCacheVae') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>

    <PreprocessTaskList :tasks="preprocessTasks" :refreshing="refreshing" @refresh="refresh" />
    </template>
    <UnsavedChangesDialog :model-value="Boolean(leaveResolve)" :busy="leaveBusy" :error="leaveError"
      @cancel="finishLeave(false)" @discard="discardAndLeave" @save="saveAndLeave" />
    <v-dialog v-model="showHelp" max-width="640" scrollable>
      <v-card>
        <HelpPanel :key="helpSession" ref="helpPanel" :field-help="pathHelp" :guide-html="guideHtml" max-height="70dvh" />
        <v-card-actions><v-spacer /><v-btn @click="showHelp = false">{{ t('guidebookClose') }}</v-btn></v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { isTaskActive } from '../utils/taskState.ts'
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { onBeforeRouteLeave } from 'vue-router'
import { useModelWorkspace } from '../stores/modelWorkspace'
import QwenPreprocessContent from '../components/QwenPreprocessContent.vue'
import RuntimeSelector from '../components/RuntimeSelector.vue'
import WorkspaceHeading from '../components/WorkspaceHeading.vue'
import PreprocessTaskList from '../components/PreprocessTaskList.vue'
import ConfigField from '../components/ConfigField.vue'
import HelpPanel from '../components/HelpPanel.vue'
import UnsavedChangesDialog from '../components/UnsavedChangesDialog.vue'
import { useActiveModelWorkspace } from '../stores/qwenWorkspace'
import { useAppStore } from '../stores/app'
import { mossText } from '../i18n/modelWorkspace'
const workspace = useModelWorkspace()
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useConfigStore, type FieldMeta } from '../stores/config'
import { useI18n } from '../composables/useI18n'
import { readPreprocessRun, writePreprocessRun } from '../composables/usePreprocessRunStorage'

const taskStore = useTaskStore()
const notify = useNotifyStore()
const configStore = useConfigStore()
const { t } = useI18n()
const app = useAppStore()
const modelPreprocess = ref<InstanceType<typeof QwenPreprocessContent> | null>(null)
const helpPanel = ref<InstanceType<typeof HelpPanel> | null>(null)
const showHelp = ref(false)
const helpSession = ref(0)
const subtitle = computed(() => workspace.family === 'anima' ? t('ppSubtitle') : workspace.family === 'moss' ? mossText(app.language, 'cacheHint') : t('qwenCacheHint'))
const guideHtml = computed(() => ['ppResizeDesc', 'ppCacheVaeDesc', 'ppCacheTeDesc', 'ppCachePeDesc', 'ppMaskDesc', 'ppPipelineDesc'].map(key => `<p>${t(key)}</p>`).join(''))
const loadError = ref('')
const refreshing = ref(false)
const submitting = ref('')
async function openPreprocessHelp(key?: string): Promise<void> {
  if (workspace.family !== 'anima') return modelPreprocess.value?.openHelp(key)
  helpSession.value++
  showHelp.value = true
  await nextTick()
  if (key) helpPanel.value?.showFieldHelp(key, 'method')
}
async function responseError(response: Response): Promise<Error> {
  let detail = `HTTP ${response.status}`
  try { const body = await response.json(); if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail) } catch { /* Use status when the response is not JSON. */ }
  return new Error(detail)
}
function errorMessage(error: unknown): string { return error instanceof Error ? error.message : String(error) }


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
    if (!res.ok) throw await responseError(res)
    const data = await res.json()
    status.resized = data.resized ?? 0
    status.masks = data.masks ?? 0
    status.cache.latents = data.cache?.latents ?? 0
    status.cache.te = data.cache?.te ?? 0
    status.cache.pe = data.cache?.pe ?? 0
    status.cond_resized = data.cond_resized ?? 0
  } catch (error) { loadError.value = errorMessage(error) }
}


// Refresh status when tasks finish
const runningCount = computed(() =>
  taskStore.tasks.filter(tp => preprocessCommands.includes(tp.command) && isTaskActive(tp.state)).length
)
watch(runningCount, (newVal, oldVal) => {
  if (workspace.family === 'anima' && oldVal > newVal) {
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
const savedSettings = ref('')
const settingsSaving = ref(false)
const settingsDirty = computed(() => Boolean(savedSettings.value) && JSON.stringify(settings) !== savedSettings.value)
function discardSettings(): void {
  if (savedSettings.value) Object.assign(settings, JSON.parse(savedSettings.value))
}
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
    if (!res.ok) throw await responseError(res)
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
    savedSettings.value = JSON.stringify(settings)
  } catch (error) { loadError.value = errorMessage(error) }
  finally { settingsHydrating = false }
}


watch(settings, () => {
  if (!settingsHydrating && selectedRun.value) selectedRun.value = null
}, { deep: true, flush: 'sync' })

// ── Dataset paths ──────────────────────────────────────────────

const paths = reactive({ source: '', resized: '', cache: '', condSource: '', condResized: '' })
const pathEntries = computed(() => [
  { key: 'source' as const, label: t('ppPathSource'), value: paths.source, source: true },
  { key: 'resized' as const, label: t('ppPathResized'), value: paths.resized },
  { key: 'cache' as const, label: t('ppPathCache'), value: paths.cache },
  { key: 'condSource' as const, label: t('ppPathCondSource'), value: paths.condSource },
  { key: 'condResized' as const, label: t('ppPathCondResized'), value: paths.condResized },
])
const pathsLoading = ref(false)
const pathsSaving = ref(false)
const savedPaths = ref('')
const pathsDirty = computed(() => Boolean(savedPaths.value) && JSON.stringify(paths) !== savedPaths.value)
const pathHelp = computed(() => Object.fromEntries(pathEntries.value.map(entry => [entry.key, entry.source ? t('ppResizeDesc') : entry.key.startsWith('cond') ? t('ppCondResizeDesc') : entry.key === 'cache' ? t('ppCacheVaeDesc') : t('ppResizeDesc')])))
function pathField(entry: typeof pathEntries.value[number]): FieldMeta {
  return { key: `preprocess.${entry.key}`, label: entry.label, value: entry.value, default_value: '', field_type: 'path',
    description: '', origin: 'method', group: 'paths', is_virtual: false, read_only: pathsLoading.value || pathsSaving.value || Boolean(submitting.value), path_kind: 'directory' }
}
function discardPaths(): void {
  if (savedPaths.value) Object.assign(paths, JSON.parse(savedPaths.value))
}

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
    if (!res.ok) throw await responseError(res)
    const data = await res.json()
    paths.source = data.source_image_dir ?? ''
    paths.resized = data.resized_image_dir ?? ''
    paths.cache = data.lora_cache_dir ?? ''
    paths.condSource = data.conditioning_data_dir ?? ''
    paths.condResized = data.conditioning_resized_dir ?? ''
    savedPaths.value = JSON.stringify(paths)
  } catch (error) { loadError.value = errorMessage(error) }
  finally { pathsLoading.value = false }
}

async function savePaths(): Promise<boolean> {
  if (selectedRun.value) return true
  if (pathsSaving.value) return false
  pathsSaving.value = true
  try {
    await ensureVariant()
    if (!configStore.variant) {
      notify.show(t('notifyConfigSaveFailed'), 'error')
      return false
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
      savedPaths.value = JSON.stringify(paths)
      notify.show(t('ppPathsSaved'), 'success')
      void fetchStatus()
      return true
    }
    throw await responseError(res)
  } catch (error) {
    notify.show(`${t('notifyConfigSaveFailed')}: ${errorMessage(error)}`, 'error')
    return false
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
    if (!res.ok) throw await responseError(res)
    const data = await res.json()
    runs.value = Array.isArray(data) ? data : []
    const currentExists = runs.value.some(run => run.manifest === selectedRun.value)
    if (!pathsDirty.value && !settingsDirty.value && (!currentExists || selectNewest)) {
      selectedRun.value = runs.value.find(run => run.complete && run.status === 'ready')?.manifest ?? null
    }
  } catch (error) { loadError.value = errorMessage(error) }
  finally { runsLoading.value = false }
}

watch(selectedRun, async () => {
  if (workspace.family !== 'anima') return
  writePreprocessRun(selectedRun.value)
  await Promise.all([fetchStatus(), fetchPaths()])
})

watch([() => configStore.variant, () => configStore.preset], async () => {
  if (workspace.family !== 'anima') return
  // A preprocessing manifest is shared across training methods/presets. Keep
  // the selection stable while refreshing the source-filtered run list.
  await fetchRuns(false)
})


async function saveSettings(): Promise<boolean> {
  if (settingsSaving.value) return false
  settingsSaving.value = true
  const payload = JSON.stringify(settings)
  try {
    const res = await fetch('/api/preprocess/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: payload,
    })
    if (res.ok) {
      savedSettings.value = payload
      notify.show(t('ppSettingsSaved'), 'success')
      return true
    }
    throw await responseError(res)
  } catch (error) {
    notify.show(`${t('notifyConfigSaveFailed')}: ${errorMessage(error)}`, 'error')
  } finally { settingsSaving.value = false }
  return false
}

// ── Task management ──────────────────────────────────────────

const preprocessTasks = computed(() =>
  taskStore.tasks.filter(tp => preprocessCommands.includes(tp.command))
)

function isRunning(command: string) {
  return submitting.value === command || taskStore.tasks.some(tp => tp.command === command && isTaskActive(tp.state))
}

async function runTask(command: string) {
  if (submitting.value || pathsSaving.value || settingsSaving.value || isRunning(command)) return
  submitting.value = command
  try {
    if (!selectedRun.value && pathsDirty.value && !(await savePaths())) return
    await submitTask(command)
  } finally { submitting.value = '' }
}

async function submitTask(command: string) {
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

async function refresh(): Promise<void> {
  refreshing.value = true
  loadError.value = ''
  try {
    await Promise.all([taskStore.fetchTasks(), fetchStatus(), fetchRuns(false),
      ...(!pathsDirty.value ? [fetchPaths()] : []), ...(!settingsDirty.value ? [fetchSettings()] : []),
    ])
  }
  finally { refreshing.value = false }
}

const activeModel = useActiveModelWorkspace()
const pageDirty = computed(() => workspace.family === 'anima' ? pathsDirty.value || settingsDirty.value : activeModel.value.dirty)
const leaveResolve = ref<((allowed: boolean) => void) | null>(null)
const leaveBusy = ref(false)
const leaveError = ref('')
function confirmLeave(): Promise<boolean> {
  if (!pageDirty.value) return Promise.resolve(true)
  leaveResolve.value?.(false)
  leaveError.value = ''
  return new Promise(resolve => { leaveResolve.value = resolve })
}
function finishLeave(allowed: boolean): void {
  leaveResolve.value?.(allowed)
  leaveResolve.value = null
}
function discardAndLeave(): void {
  if (workspace.family === 'anima') { discardPaths(); discardSettings() }
  else activeModel.value.discard()
  finishLeave(true)
}
async function saveAndLeave(): Promise<void> {
  leaveBusy.value = true
  leaveError.value = ''
  try {
    if (workspace.family === 'anima') {
      if ((pathsDirty.value && !(await savePaths())) || (settingsDirty.value && !(await saveSettings()))) {
        leaveError.value = t('notifyConfigSaveFailed')
        return
      }
    } else {
      const selected = activeModel.value
      if (selected.busy) return
      await selected.perform(selected.save)
      if (selected.error) { leaveError.value = selected.error; return }
    }
    finishLeave(true)
  } finally { leaveBusy.value = false }
}
async function selectRun(manifest: string | null): Promise<void> {
  if (await confirmLeave()) selectedRun.value = manifest
}
function beforeUnload(event: BeforeUnloadEvent): void {
  if (pageDirty.value) { event.preventDefault(); event.returnValue = '' }
}
onBeforeRouteLeave(confirmLeave)
onMounted(() => window.addEventListener('beforeunload', beforeUnload))
onBeforeUnmount(() => { finishLeave(false); window.removeEventListener('beforeunload', beforeUnload) })

watch(() => workspace.family, async (family) => {
  if (family !== 'anima') return
  loadError.value = ''
  await Promise.all([taskStore.fetchTasks(), fetchStatus(), fetchSettings(), fetchPaths()])
  await fetchRuns(false)
}, { immediate: true })
</script>

<style scoped>
.preprocess-page {
  flex: 1 1 0;
  min-height: 0;
  min-width: 0;
  overflow-y: auto;
  container-type: inline-size;
}

.run-manifest-path {
  overflow-wrap: anywhere;
}

.preprocess-path-fields, .preprocess-run-paths {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px 20px;
  min-width: 0;
}
.preprocess-path-wide { grid-column: 1 / -1; }
.preprocess-run-paths > div { min-width: 0; }
.preprocess-run-paths dt { font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; }
.preprocess-run-paths dd {
  margin: 0;
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.65;
  overflow-wrap: anywhere;
  user-select: text;
}
.preprocess-path-card :deep(.v-card-text) { padding-top: 12px; }
.preprocess-settings-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
  gap: 20px;
  margin: 0;
}
.preprocess-settings-grid > .v-col { width: auto; max-width: none; padding: 0; min-width: 0; }
:deep(.preprocess-task-grid) {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin: 0;
}
:deep(.preprocess-task-grid > .v-col) { width: auto; max-width: none; padding: 0; min-width: 0; }
:deep(.preprocess-task-grid .v-card) { display: flex; flex-direction: column; height: 100%; }
:deep(.preprocess-task-grid .v-card-text) { flex: 1; line-height: 1.65; }
:deep(.preprocess-task-grid .v-card-title) { white-space: normal; line-height: 1.5; }
:deep(.preprocess-task-grid .v-card-actions) { padding: 8px 12px 12px; flex-wrap: wrap; gap: 4px; }
.preprocess-page :deep(.v-card), .preprocess-page :deep(.v-expansion-panel) {
  background: var(--bg-form-surface, var(--bg-surface));
}
.preprocess-page :deep(.v-field) { background: var(--bg-form-field, var(--bg-surface)); }
@container (max-width: 960px) { :deep(.preprocess-task-grid) { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@container (max-width: 700px) { .preprocess-path-fields, .preprocess-run-paths { grid-template-columns: minmax(0, 1fr); } }
@container (max-width: 620px) { :deep(.preprocess-task-grid) { grid-template-columns: minmax(0, 1fr); } }

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
