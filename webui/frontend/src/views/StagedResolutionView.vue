<template>
  <v-container fluid class="pa-4 staged-resolution-page">
    <div class="d-flex align-start justify-space-between ga-4 mb-4 flex-wrap">
      <div>
        <h1 class="text-h5">{{ t('srTitle') }}</h1>
        <div class="text-body-2 text-medium-emphasis mt-1">{{ t('srSubtitle') }}</div>
      </div>
      <div class="profile-toolbar">
        <v-combobox
          v-model="store.selectedProfileName"
          :items="store.profiles"
          :label="t('srProfile')"
          density="compact"
          hide-details
        />
        <v-btn
          icon="mdi-folder-open-outline"
          variant="tonal"
          :title="t('srLoadProfile')"
          :aria-label="t('srLoadProfile')"
          :loading="store.loading"
          @click="loadCurrentProfile"
        />
        <v-btn
          icon="mdi-plus"
          variant="tonal"
          :title="t('srNewProfile')"
          :aria-label="t('srNewProfile')"
          @click="newProfileDialog = true"
        />
      </div>
    </div>

    <v-alert
      v-if="store.error"
      type="error"
      variant="tonal"
      density="compact"
      closable
      class="mb-4"
      @click:close="store.error = ''"
    >
      {{ store.error }}
    </v-alert>

    <section class="config-section">
      <div class="section-heading">
        <v-icon icon="mdi-tune-variant" size="20" />
        <h2>{{ t('srConfiguration') }}</h2>
        <v-chip v-if="store.dirty" color="warning" size="small">{{ t('srUnsaved') }}</v-chip>
      </div>

      <v-row dense>
        <v-col cols="12" sm="6" lg="2">
          <v-select
            v-model="store.plan.method"
            :items="configStore.methods"
            :label="t('cfgMethod')"
            hide-details="auto"
            @update:model-value="onMethodChange"
          />
        </v-col>
        <v-col cols="12" sm="6" lg="2">
          <v-select
            v-model="store.plan.variant"
            :items="variantItems"
            item-title="label"
            item-value="value"
            :label="t('cfgVariant')"
            hide-details="auto"
          />
        </v-col>
        <v-col cols="12" sm="6" lg="2">
          <v-select
            v-model="store.plan.preset"
            :items="configStore.presets"
            :label="t('cfgPreset')"
            hide-details="auto"
          />
        </v-col>
        <v-col cols="12" sm="6" lg="2">
          <v-text-field
            v-model.number="store.plan.max_train_steps"
            :label="t('srTotalSteps')"
            type="number"
            min="3"
            step="100"
            class="font-mono-field"
            hide-details="auto"
          />
        </v-col>
        <v-col cols="12" lg="4">
          <v-text-field
            v-model="store.plan.source_image_dir"
            :label="t('ppPathSource')"
            class="font-mono-field"
            hide-details="auto"
          />
        </v-col>
      </v-row>
    </section>

    <section class="config-section">
      <div class="section-heading">
        <v-icon icon="mdi-layers-triple-outline" size="20" />
        <h2>{{ t('srStages') }}</h2>
        <span class="text-caption text-medium-emphasis">{{ t('srThreePasses') }}</span>
      </div>

      <div class="schedule-bar mb-4" role="img" :aria-label="t('srSchedule')">
        <div
          v-for="(stage, index) in store.plan.stages"
          :key="index"
          class="schedule-segment"
          :class="`segment-${index}`"
          :style="{ width: `${Math.max(0, Number(stage.ratio) || 0)}%` }"
        >
          <span>{{ stage.resolution }} · {{ formatRatio(stage.ratio) }}%</span>
        </div>
      </div>

      <div class="stage-grid stage-grid-header" aria-hidden="true">
        <div>{{ t('srStage') }}</div>
        <div>{{ t('srResolution') }}</div>
        <div>{{ t('srRatio') }}</div>
        <div>{{ t('srBatchSize') }}</div>
        <div>{{ t('srRepeats') }}</div>
        <div>{{ t('srStepRange') }}</div>
      </div>
      <div
        v-for="(stage, index) in store.plan.stages"
        :key="`stage-${index}`"
        class="stage-grid stage-grid-row"
      >
        <div class="stage-number">
          <span class="stage-index">{{ index + 1 }}</span>
          <span>{{ t('srStageName', { n: index + 1 }) }}</span>
        </div>
        <div>
          <div class="mobile-label">{{ t('srResolution') }}</div>
          <v-select
            v-model.number="stage.resolution"
            :items="resolutionOptions"
            suffix="px"
            hide-details
          />
        </div>
        <div>
          <div class="mobile-label">{{ t('srRatio') }}</div>
          <v-text-field
            v-model.number="stage.ratio"
            type="number"
            min="0.1"
            max="100"
            step="1"
            suffix="%"
            class="font-mono-field"
            hide-details
          />
        </div>
        <div>
          <div class="mobile-label">{{ t('srBatchSize') }}</div>
          <v-text-field
            v-model.number="stage.batch_size"
            type="number"
            min="1"
            max="128"
            class="font-mono-field"
            hide-details
          />
        </div>
        <div>
          <div class="mobile-label">{{ t('srRepeats') }}</div>
          <v-text-field
            v-model.number="stage.num_repeats"
            type="number"
            min="1"
            max="10000"
            class="font-mono-field"
            hide-details
          />
        </div>
        <div class="step-range font-mono-field">
          <div class="mobile-label">{{ t('srStepRange') }}</div>
          {{ stepRange(index) }}
        </div>
      </div>

      <v-alert
        v-if="validationErrors.length"
        type="warning"
        variant="tonal"
        density="compact"
        class="mt-4"
      >
        {{ validationErrors[0] }}
      </v-alert>
    </section>

    <section class="config-section">
      <div class="section-heading">
        <v-icon icon="mdi-database-check-outline" size="20" />
        <h2>{{ t('srReadiness') }}</h2>
        <v-spacer />
        <v-btn
          icon="mdi-refresh"
          variant="text"
          size="small"
          :title="t('ppRefresh')"
          :aria-label="t('ppRefresh')"
          @click="store.fetchStatus"
        />
      </div>

      <div class="source-summary mb-3">
        <div>
          <span class="summary-value">{{ store.status?.source_images ?? 0 }}</span>
          <span class="summary-label">{{ t('srSourceImages') }}</span>
        </div>
        <div>
          <span class="summary-value">{{ store.status?.captions ?? 0 }}</span>
          <span class="summary-label">{{ t('srCaptions') }}</span>
        </div>
        <v-chip
          :color="store.status?.source_exists ? 'success' : 'error'"
          size="small"
          variant="tonal"
        >
          <v-icon
            :icon="store.status?.source_exists ? 'mdi-check-circle-outline' : 'mdi-alert-circle-outline'"
            start
          />
          {{ store.status?.source_exists ? t('srSourceFound') : t('srSourceMissing') }}
        </v-chip>
      </div>

      <div class="readiness-grid readiness-header" aria-hidden="true">
        <div>{{ t('srResolution') }}</div>
        <div>{{ t('ppStatusResized') }}</div>
        <div>{{ t('ppStatusLatents') }}</div>
        <div>{{ t('ppStatusTe') }}</div>
        <div>{{ t('taskState') }}</div>
        <div>{{ t('srOutputPaths') }}</div>
      </div>
      <div
        v-for="(stage, index) in readinessRows"
        :key="`ready-${stage.resolution}`"
        class="readiness-grid readiness-row"
      >
        <div class="resolution-cell">
          <span class="stage-dot" :class="`dot-${index}`" />
          <strong>{{ stage.resolution }}px</strong>
        </div>
        <div><span class="mobile-label">{{ t('ppStatusResized') }}</span>{{ stage.resized }}</div>
        <div><span class="mobile-label">{{ t('ppStatusLatents') }}</span>{{ stage.latents }}</div>
        <div><span class="mobile-label">{{ t('ppStatusTe') }}</span>{{ stage.text_embeddings }}</div>
        <div>
          <v-chip :color="stage.ready ? 'success' : 'warning'" size="small" variant="tonal">
            <v-icon :icon="stage.ready ? 'mdi-check' : 'mdi-progress-clock'" start />
            {{ stage.ready ? t('srReady') : t('srPending') }}
          </v-chip>
        </div>
        <div class="path-cell font-mono-field">
          <div>{{ stage.resized_dir }}</div>
          <div class="text-medium-emphasis">{{ stage.cache_dir }}</div>
        </div>
      </div>
    </section>

    <div class="action-bar">
      <div class="action-status">
        <v-icon
          :icon="store.status?.all_ready ? 'mdi-check-decagram-outline' : 'mdi-database-clock-outline'"
          :color="store.status?.all_ready ? 'success' : 'warning'"
        />
        <span>{{ store.status?.all_ready ? t('srAllReady') : t('srPreparationRequired') }}</span>
      </div>
      <div class="d-flex ga-2 flex-wrap justify-end">
        <v-btn
          color="primary"
          prepend-icon="mdi-content-save"
          :loading="store.saving"
          :disabled="validationErrors.length > 0 || !store.profileAligned || !store.dirty"
          @click="save"
        >
          {{ t('cfgSave') }}
        </v-btn>
        <v-btn
          color="secondary"
          prepend-icon="mdi-database-cog-outline"
          :loading="store.launching === 'preprocess'"
          :disabled="validationErrors.length > 0 || !store.profileAligned"
          @click="startPreprocess"
        >
          {{ t('srPreprocess') }}
        </v-btn>
        <v-btn
          color="success"
          prepend-icon="mdi-play-circle"
          :loading="store.launching === 'train'"
          :disabled="!canTrain"
          @click="startTraining"
        >
          {{ t('srStartTraining') }}
        </v-btn>
      </div>
    </div>

    <v-dialog v-model="newProfileDialog" max-width="420">
      <v-card>
        <v-card-title class="text-subtitle-1">{{ t('srNewProfile') }}</v-card-title>
        <v-card-text>
          <v-text-field
            v-model="newProfileName"
            :label="t('srProfile')"
            autofocus
            :rules="[profileNameRule]"
            @keyup.enter="createProfile"
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="newProfileDialog = false">{{ t('dsCancel') }}</v-btn>
          <v-btn color="primary" :disabled="profileNameRule(newProfileName) !== true" @click="createProfile">
            {{ t('srCreate') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useConfigStore } from '../stores/config'
import { useNotifyStore } from '../stores/notify'
import {
  useStagedResolutionStore,
  type StageReadiness,
} from '../stores/stagedResolution'
import { useI18n } from '../composables/useI18n'

const store = useStagedResolutionStore()
const configStore = useConfigStore()
const notify = useNotifyStore()
const { t } = useI18n()

const resolutionOptions = [512, 768, 896, 1024, 1280, 1536]
const newProfileDialog = ref(false)
const newProfileName = ref('')
let pollTimer: ReturnType<typeof setInterval> | null = null

const variantItems = computed(() =>
  configStore.variants.map(value => ({
    value,
    label: configStore.variantLabels[value] || value,
  })),
)

const readinessRows = computed<StageReadiness[]>(() => {
  if (store.status?.stages?.length === 3) return store.status.stages
  return store.plan.stages.map(stage => ({
    ...stage,
    resized_dir: `post_image_dataset/staged/${store.selectedProfileName}/${stage.resolution}/resized`,
    cache_dir: `post_image_dataset/staged/${store.selectedProfileName}/${stage.resolution}/cache`,
    resized: 0,
    latents: 0,
    text_embeddings: 0,
    ready: false,
  }))
})

const validationErrors = computed(() => {
  const errors: string[] = []
  const name = store.selectedProfileName.trim()
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name)) errors.push(t('srErrProfile'))
  if (!store.plan.source_image_dir.trim()) errors.push(t('srErrSource'))
  if (!Number.isInteger(store.plan.max_train_steps) || store.plan.max_train_steps < 3) {
    errors.push(t('srErrSteps'))
  }
  const resolutions = store.plan.stages.map(stage => Number(stage.resolution))
  if (new Set(resolutions).size !== 3 || resolutions.some((value, index) => index > 0 && value <= resolutions[index - 1])) {
    errors.push(t('srErrResolutionOrder'))
  }
  const total = store.plan.stages.reduce((sum, stage) => sum + Number(stage.ratio || 0), 0)
  if (store.plan.stages.some(stage => Number(stage.ratio) <= 0) || Math.abs(total - 100) > 0.000001) {
    errors.push(t('srErrRatio', { total: formatRatio(total) }))
  } else if (Number.isInteger(store.plan.max_train_steps)) {
    let previousBoundary = 0
    let cumulativeRatio = 0
    const hasEmptyStage = store.plan.stages.some((stage, index) => {
      cumulativeRatio += Number(stage.ratio)
      const boundary = index === store.plan.stages.length - 1
        ? store.plan.max_train_steps
        : Math.ceil(store.plan.max_train_steps * cumulativeRatio / 100 - 1e-12)
      const empty = boundary <= previousBoundary
      previousBoundary = boundary
      return empty
    })
    if (hasEmptyStage) errors.push(t('srErrStageSteps'))
  }
  if (store.plan.stages.some(stage => !Number.isInteger(stage.batch_size) || stage.batch_size < 1)) {
    errors.push(t('srErrBatch'))
  }
  if (store.plan.stages.some(stage => !Number.isInteger(stage.num_repeats) || stage.num_repeats < 1)) {
    errors.push(t('srErrRepeats'))
  }
  return errors
})

const canTrain = computed(() =>
  validationErrors.value.length === 0 &&
  store.selectedProfileName.trim() === store.loadedProfileName &&
  store.status?.profile === store.loadedProfileName &&
  !store.dirty &&
  !!store.status?.all_ready,
)

function formatRatio(value: number) {
  const number = Number(value || 0)
  return Number.isInteger(number) ? String(number) : number.toFixed(1)
}

function stepRange(index: number) {
  const total = Math.max(0, Number(store.plan.max_train_steps) || 0)
  let before = 0
  for (let i = 0; i < index; i++) before += Number(store.plan.stages[i].ratio || 0)
  const through = before + Number(store.plan.stages[index]?.ratio || 0)
  const start = Math.round(total * before / 100)
  const end = index === store.plan.stages.length - 1
    ? Math.max(start, total - 1)
    : Math.max(start, Math.round(total * through / 100) - 1)
  return `${start.toLocaleString()}–${end.toLocaleString()}`
}

function profileNameRule(value: string): true | string {
  return /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(value.trim()) || t('srErrProfile')
}

async function syncSelectors() {
  if (!configStore.methods.length) await configStore.fetchMethods()
  if (!configStore.presets.length) await configStore.fetchPresets()
  await configStore.fetchVariants(store.plan.method || 'lora')
}

async function onMethodChange(method: string) {
  await configStore.fetchVariants(method)
  if (!configStore.variants.includes(store.plan.variant)) {
    store.plan.variant = configStore.variants[0] || 'lora'
  }
}

async function loadCurrentProfile() {
  try {
    await store.loadProfile()
    await syncSelectors()
  } catch {
    notify.show(t('srLoadFailed'), 'error')
  }
}

function createProfile() {
  if (profileNameRule(newProfileName.value) !== true) return
  store.newProfile(newProfileName.value)
  newProfileDialog.value = false
  newProfileName.value = ''
  syncSelectors()
}

async function save() {
  if (validationErrors.value.length) return
  try {
    await store.saveProfile()
    notify.show(t('srSaved'), 'success')
  } catch {
    notify.show(t('notifyConfigSaveFailed'), 'error')
  }
}

async function startPreprocess() {
  if (validationErrors.value.length) return
  const taskId = await store.start('preprocess')
  if (taskId) notify.show(t('srPreprocessQueued'), 'success')
  else notify.show(t('notifyTaskStartFailed', { command: t('srPreprocess') }), 'error')
}

async function startTraining() {
  if (!canTrain.value) return
  const taskId = await store.start('train')
  if (taskId) notify.show(t('notifyTrainingLaunched'), 'success')
  else notify.show(t('notifyTaskStartFailed', { command: t('srStartTraining') }), 'error')
}

async function pollActivePreprocess() {
  await store.pollPreprocessTask()
}

onMounted(async () => {
  try {
    await Promise.all([store.fetchProfiles(), configStore.fetchMethods(), configStore.fetchPresets()])
    await store.loadProfile(store.selectedProfileName)
    await syncSelectors()
  } catch {
    await syncSelectors()
  }
  pollTimer = setInterval(pollActivePreprocess, 5000)
})

onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<style scoped>
.staged-resolution-page {
  height: 100%;
  overflow-y: auto;
  max-width: 1600px;
}

.profile-toolbar {
  display: grid;
  grid-template-columns: minmax(220px, 320px) 40px 40px;
  gap: 8px;
  align-items: center;
}

.config-section {
  padding: 18px 0 20px;
  border-top: 1px solid var(--border-default);
}

.section-heading {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 32px;
  margin-bottom: 14px;
}

.section-heading h2 {
  font-size: 15px;
  font-weight: 600;
  line-height: 1.4;
}

.schedule-bar {
  display: flex;
  height: 36px;
  overflow: hidden;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
}

.schedule-segment {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 0;
  color: #fff;
  font-family: var(--font-mono);
  font-size: 12px;
  white-space: nowrap;
  transition: width 180ms ease;
}

.schedule-segment span {
  overflow: hidden;
  text-overflow: ellipsis;
  padding: 0 8px;
}

.segment-0 { background: #9b4720; }
.segment-1 { background: #b87333; }
.segment-2 { background: #c58b2b; }

.stage-grid {
  display: grid;
  grid-template-columns: minmax(120px, 0.9fr) repeat(4, minmax(112px, 0.75fr)) minmax(150px, 0.9fr);
  gap: 12px;
  align-items: center;
}

.stage-grid-header,
.readiness-header {
  padding: 0 12px 8px;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 500;
}

.stage-grid-row,
.readiness-row {
  min-height: 64px;
  padding: 10px 12px;
  border-top: 1px solid var(--border-subtle);
}

.stage-grid-row:last-child,
.readiness-row:last-child {
  border-bottom: 1px solid var(--border-subtle);
}

.stage-number,
.resolution-cell {
  display: flex;
  align-items: center;
  gap: 10px;
}

.stage-index {
  display: inline-flex;
  width: 28px;
  height: 28px;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border-strong);
  border-radius: 50%;
  color: var(--forge-gold);
  font-family: var(--font-mono);
  font-size: 12px;
}

.step-range {
  color: var(--text-secondary);
  font-size: 13px;
}

.mobile-label { display: none; }

.source-summary {
  display: flex;
  align-items: center;
  gap: 28px;
  min-height: 40px;
}

.source-summary > div {
  display: flex;
  align-items: baseline;
  gap: 7px;
}

.summary-value {
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 18px;
  font-weight: 600;
}

.summary-label {
  color: var(--text-secondary);
  font-size: 12px;
}

.readiness-grid {
  display: grid;
  grid-template-columns: minmax(110px, 0.65fr) repeat(3, minmax(80px, 0.5fr)) minmax(110px, 0.65fr) minmax(300px, 2fr);
  gap: 12px;
  align-items: center;
}

.stage-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
}

.dot-0 { background: #9b4720; }
.dot-1 { background: #b87333; }
.dot-2 { background: #c58b2b; }

.path-cell {
  min-width: 0;
  overflow-wrap: anywhere;
  font-size: 11px;
  line-height: 1.5;
}

.action-bar {
  position: sticky;
  bottom: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 0;
  border-top: 1px solid var(--border-strong);
  background: rgba(12, 12, 16, 0.96);
  backdrop-filter: blur(8px);
}

.action-status {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-secondary);
  font-size: 13px;
}

@media (max-width: 959px) {
  .stage-grid-header,
  .readiness-header {
    display: none;
  }

  .stage-grid-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    padding: 14px 4px;
  }

  .stage-number {
    grid-column: 1 / -1;
  }

  .mobile-label {
    display: block;
    margin-bottom: 5px;
    color: var(--text-secondary);
    font-size: 11px;
  }

  .readiness-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    padding: 14px 4px;
  }

  .resolution-cell,
  .path-cell {
    grid-column: 1 / -1;
  }

  .action-bar {
    align-items: flex-start;
    flex-direction: column;
  }
}

@media (max-width: 599px) {
  .staged-resolution-page { padding: 12px !important; }
  .profile-toolbar {
    width: 100%;
    grid-template-columns: minmax(0, 1fr) 40px 40px;
  }
  .schedule-segment { font-size: 10px; }
  .source-summary { gap: 14px; flex-wrap: wrap; }
  .action-bar > div:last-child { width: 100%; }
  .action-bar .v-btn { flex: 1 1 auto; }
}
</style>
