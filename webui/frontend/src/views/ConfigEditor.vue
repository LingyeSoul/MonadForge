<template>
  <v-container fluid class="pa-4 config-editor">
    <v-row align="center" class="mb-4">
      <v-col cols="12" md="2">
        <v-select
          v-model="selectedMethod"
          :items="configStore.methods"
          :label="t('cfgMethod')"
          variant="outlined"
          density="compact"
          hide-details
          @update:model-value="onMethodChange"
        />
      </v-col>
      <v-col cols="12" md="3">
        <v-select
          v-model="selectedVariant"
          :items="variantItems"
          item-title="label"
          item-value="value"
          :label="t('cfgVariant')"
          variant="outlined"
          density="compact"
          hide-details
          :disabled="!selectedMethod"
          @update:model-value="onVariantChange"
        />
      </v-col>
      <v-col cols="12" md="1" class="d-flex align-center ga-1">
        <v-btn
          icon="mdi-plus"
          variant="text"
          density="compact"
          size="small"
          :disabled="!selectedMethod"
          :title="t('cfgNewVariant')"
          @click="openCreateVariant"
        />
      </v-col>
      <v-col cols="12" md="2">
        <div class="d-flex align-center ga-1">
          <v-select
            v-model="selectedPreset"
            :items="configStore.presets"
            :label="t('cfgPreset')"
            variant="outlined"
            density="compact"
            hide-details
            @update:model-value="onVariantChange"
          />
          <v-btn
            icon="mdi-plus"
            variant="text"
            density="compact"
            size="small"
            :title="t('cfgPresetCreate')"
            @click="openCreatePreset"
          />
          <v-btn
            v-if="isCustomPreset"
            icon="mdi-delete-outline"
            variant="text"
            density="compact"
            size="small"
            color="error"
            :title="t('cfgPresetDelete')"
            @click="deleteCurrentPreset"
          />
        </div>
      </v-col>
      <v-col cols="12" md="4" class="d-flex justify-end ga-2 flex-wrap">
        <v-btn
          :color="(configStore.dirty || extraArgs) ? 'warning' : 'primary'"
          :loading="configStore.loading"
          :disabled="!configStore.dirty && !extraArgs"
          prepend-icon="mdi-content-save"
          @click="onSave"
        >
          {{ t('cfgSave') }}{{ (configStore.dirty || extraArgs) ? ' *' : '' }}
        </v-btn>
        <v-btn
          variant="outlined"
          :disabled="!selectedVariant"
          prepend-icon="mdi-refresh"
          @click="loadConfig"
        >
          {{ t('cfgReload') }}
        </v-btn>
        <v-btn
          color="success"
          :disabled="!selectedVariant"
          :loading="trainingLaunching"
          prepend-icon="mdi-play-circle"
          @click="startTraining"
        >
          {{ t('cfgTrain') }}
        </v-btn>
        <v-btn
          color="secondary"
          :disabled="!selectedVariant"
          :loading="testLaunching"
          prepend-icon="mdi-test-tube"
          @click="startTest"
        >
          {{ t('cfgTest') }}
        </v-btn>
      </v-col>
    </v-row>

    <!-- WandB Tracking Panel -->
    <v-expansion-panels v-model="wandbPanel" class="mb-4" variant="accordion">
      <v-expansion-panel elevation="0">
        <v-expansion-panel-title class="text-subtitle-2">
          <v-icon icon="mdi-chart-box" class="mr-2" />
          {{ t('cfgWandbSection') }}
          <v-spacer />
          <v-switch
            v-model="wandb.enabled"
            :label="t('cfgWandbEnabled')"
            density="compact"
            hide-details
            color="primary"
            class="mr-2"
            @click.stop
            @update:model-value="saveWandbSettings"
          />
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <v-row dense>
            <v-col cols="12" md="4">
              <v-text-field
                v-model="wandb.project"
                :label="t('cfgWandbProject')"
                variant="outlined"
                density="compact"
                hide-details
                @change="saveWandbSettings"
              />
            </v-col>
            <v-col cols="12" md="4">
              <v-text-field
                v-model="wandb.run_name"
                :label="t('cfgWandbRunName')"
                variant="outlined"
                density="compact"
                hide-details
                @change="saveWandbSettings"
              />
            </v-col>
            <v-col cols="12" md="4">
              <v-text-field
                v-model="wandb.api_key"
                :label="t('cfgWandbApiKey')"
                type="password"
                variant="outlined"
                density="compact"
                hide-details
                @change="saveWandbSettings"
              />
            </v-col>
          </v-row>
          <v-row dense class="mt-2">
            <v-col cols="12" md="3">
              <v-text-field
                v-model.number="wandb.log_every_n_steps"
                :label="t('cfgWandbLogEvery')"
                type="number"
                variant="outlined"
                density="compact"
                hide-details
                min="1"
                @change="saveWandbSettings"
              />
            </v-col>
            <v-col cols="12" md="3" class="d-flex align-center">
              <v-checkbox
                v-model="wandb.log_gradients"
                :label="t('cfgWandbLogGradients')"
                density="compact"
                hide-details
                color="primary"
                @update:model-value="saveWandbSettings"
              />
            </v-col>
            <v-col cols="12" md="3" class="d-flex align-center">
              <v-checkbox
                v-model="wandb.log_weights"
                :label="t('cfgWandbLogWeights')"
                density="compact"
                hide-details
                color="primary"
                @update:model-value="saveWandbSettings"
              />
            </v-col>
            <v-col cols="12" md="3" class="d-flex align-center">
              <v-checkbox
                v-model="wandb.log_checkpoint_artifact"
                :label="t('cfgWandbLogArtifact')"
                density="compact"
                hide-details
                color="primary"
                @update:model-value="saveWandbSettings"
              />
            </v-col>
          </v-row>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>

    <v-alert
      v-if="configStore.error"
      type="error"
      variant="tonal"
      closable
      class="mb-4"
      @click:close="configStore.error = ''"
    >
      {{ configStore.error }}
    </v-alert>

    <v-progress-linear
      v-if="configStore.loading"
      indeterminate
      color="primary"
      class="mb-4"
    />

    <v-alert
      v-if="isExperimental"
      type="warning"
      variant="tonal"
      density="compact"
      class="mb-4"
      icon="mdi-flask-outline"
    >
      {{ t('cfgExperimentalWarning') }}
    </v-alert>

    <!-- Two-column layout: form left, help panel right -->
    <v-row>
      <v-col cols="12" :lg="selectedVariant ? 8 : 12" class="form-column">
        <v-card v-if="configStore.basicFields.length > 0" class="mb-4" variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-tune" class="mr-2" />
            {{ t('cfgBasicSettings') }}
          </v-card-title>
          <v-card-text>
            <v-row>
              <v-col
                v-for="field in configStore.basicFields"
                :key="field.key"
                cols="12"
                md="6"
              >
                <ConfigField
                  :field="field"
                  @update="(v) => configStore.setFieldValue(field.key, v)"
                  @help-click="(k) => onFieldHelp(k, field.origin)"
                />
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>

        <v-expansion-panels v-if="Object.keys(configStore.groupedAdvanced).length > 0" variant="accordion">
          <v-expansion-panel
            v-for="(groupFields, groupName) in configStore.groupedAdvanced"
            :key="groupName"
          >
            <v-expansion-panel-title>
              <v-icon icon="mdi-cog-outline" class="mr-2" size="small" />
              {{ groupName }}
              <v-chip size="x-small" class="ml-2" variant="outlined">{{ groupFields.length }}</v-chip>
            </v-expansion-panel-title>
            <v-expansion-panel-text>
              <v-row>
                <v-col
                  v-for="field in groupFields"
                  :key="field.key"
                  :cols="field.key === 'sample_prompts' ? 12 : undefined"
                  :md="field.key === 'sample_prompts' ? 12 : 6"
                >
                  <PreviewPromptEditor
                    v-if="field.key === 'sample_prompts'"
                    :prompt-path="String(field.value ?? 'sample_prompts.txt')"
                  />
                  <ConfigField
                    v-else
                    :field="field"
                    @update="(v) => configStore.setFieldValue(field.key, v)"
                    @help-click="(k) => onFieldHelp(k, field.origin)"
                  />
                </v-col>
              </v-row>
            </v-expansion-panel-text>
          </v-expansion-panel>
        </v-expansion-panels>

        <!-- Extra Args section -->
        <v-card v-if="selectedVariant" class="mt-4" variant="outlined">
          <v-card-title
            class="text-subtitle-1 d-flex align-center cursor-pointer"
            @click="showExtraArgs = !showExtraArgs"
          >
            <v-icon icon="mdi-code-braces" class="mr-2" size="small" />
            {{ t('cfgExtraArgs') }}
            <v-spacer />
            <v-icon :icon="showExtraArgs ? 'mdi-chevron-up' : 'mdi-chevron-down'" />
          </v-card-title>
          <v-expand-transition>
            <v-card-text v-show="showExtraArgs">
              <v-textarea
                v-model="extraArgs"
                :placeholder="t('cfgExtraArgsHint')"
                variant="outlined"
                density="compact"
                rows="4"
                auto-grow
                max-rows="10"
                hide-details
                style="font-family: monospace; font-size: 13px;"
              />
            </v-card-text>
          </v-expand-transition>
        </v-card>

        <div v-if="!configStore.loading && configStore.fields.length === 0" class="text-center pa-12">
          <v-icon icon="mdi-cog-transfer-outline" size="64" color="grey" class="mb-4" />
          <div class="text-h6 text-medium-emphasis">{{ t('cfgSelectHint') }}</div>
        </div>
      </v-col>

      <!-- Help panel: only shown when a variant is selected -->
      <v-col v-if="selectedVariant" cols="12" lg="4">
        <div class="help-panel-sticky">
          <HelpPanel
            ref="helpPanelRef"
            :variant="selectedVariant"
            :field-help="fieldHelpData"
            :guide-html="guideHtml"
            height="calc(100vh - 140px)"
          />
        </div>
      </v-col>
    </v-row>

    <!-- No-cache warning dialog -->
    <v-dialog v-model="showNoCacheDlg" max-width="500">
      <v-card>
        <v-card-title>{{ t('cfgNoCache') }}</v-card-title>
        <v-card-text>{{ t('cfgNoCacheBody') }}</v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showNoCacheDlg = false">{{ t('dsCancel') }}</v-btn>
          <v-btn color="primary" :loading="preprocessRunning" @click="runPreprocessThenTrain">
            {{ t('cfgRunPreprocess') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Checkpoint resume dialog -->
    <v-dialog v-model="showCheckpointDlg" max-width="500">
      <v-card>
        <v-card-title>{{ t('cfgCheckpointFound') }}</v-card-title>
        <v-card-text>
          {{ t('cfgCheckpointStep', { step: checkpointInfo?.step ?? 0 }) }}
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showCheckpointDlg = false">{{ t('dsCancel') }}</v-btn>
          <v-btn color="warning" @click="wipeAndTrain">
            {{ t('cfgWipe') }}
          </v-btn>
          <v-btn color="success" @click="resumeTrain">
            {{ t('cfgResume') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Create preset dialog -->
    <v-dialog v-model="showCreatePresetDlg" max-width="400">
      <v-card>
        <v-card-title>{{ t('cfgPresetCreate') }}</v-card-title>
        <v-card-text>
          <v-text-field
            v-model="newPresetName"
            :label="t('cfgPresetName')"
            :placeholder="t('cfgPresetNameHint')"
            variant="outlined"
            density="compact"
            autofocus
            @keyup.enter="confirmCreatePreset"
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showCreatePresetDlg = false">{{ t('cfgCancel') }}</v-btn>
          <v-btn
            color="primary"
            :disabled="!newPresetName.trim()"
            @click="confirmCreatePreset"
          >
            {{ t('cfgPresetCreate') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Create variant dialog -->
    <v-dialog v-model="showCreateVariantDlg" max-width="400">
      <v-card>
        <v-card-title>{{ t('cfgNewVariant') }}</v-card-title>
        <v-card-text>
          <v-text-field
            v-model="newVariantName"
            :label="t('cfgNewVariantName')"
            :placeholder="t('cfgNewVariantNameHint')"
            :rules="[variantNameRule]"
            variant="outlined"
            density="compact"
            autofocus
            @keyup.enter="confirmCreateVariant"
          />
          <v-checkbox
            v-model="seedFromCurrent"
            :label="t('cfgNewVariantSeed')"
            density="compact"
            hide-details
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showCreateVariantDlg = false">{{ t('cfgCancel') }}</v-btn>
          <v-btn
            color="primary"
            :disabled="!newVariantName.trim() || !!variantNameRule(newVariantName.trim())"
            @click="confirmCreateVariant"
          >
            {{ t('cfgNewVariant') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useConfigStore } from '../stores/config'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useAppStore } from '../stores/app'
import { useI18n } from '../composables/useI18n'
import ConfigField from '../components/ConfigField.vue'
import PreviewPromptEditor from '../components/PreviewPromptEditor.vue'
import HelpPanel from '../components/HelpPanel.vue'

const configStore = useConfigStore()
const taskStore = useTaskStore()
const notify = useNotifyStore()
const appStore = useAppStore()
const { t } = useI18n()

const selectedMethod = ref('')
const selectedVariant = ref('')
const selectedPreset = ref('default')

const trainingLaunching = ref(false)
const testLaunching = ref(false)
const preprocessRunning = ref(false)

// Extra args state
const extraArgs = ref('')
const showExtraArgs = ref(false)

// Experimental feature tracking
const isExperimental = ref(false)

// Help panel state
const helpPanelRef = ref<InstanceType<typeof HelpPanel> | null>(null)
const fieldHelpData = ref<Record<string, string>>({})
const guideHtml = ref('')

// Dialog state
const showNoCacheDlg = ref(false)
const showCheckpointDlg = ref(false)
const checkpointInfo = ref<{ state_dir: string; step: number } | null>(null)
const prelaunchResult = ref<any>(null)

// Preset management
const showCreatePresetDlg = ref(false)
const newPresetName = ref('')

// Variant management
const showCreateVariantDlg = ref(false)
const newVariantName = ref('')
const seedFromCurrent = ref(true)

// WandB settings
const wandb = ref({
  enabled: false,
  project: 'anima-lora',
  run_name: '',
  api_key: '',
  log_every_n_steps: 50,
  log_gradients: true,
  log_weights: true,
  log_checkpoint_artifact: true,
})
const wandbPanel = ref(false)

async function loadWandbSettings() {
  try {
    const res = await fetch('/api/config/wandb-settings')
    if (res.ok) wandb.value = await res.json()
  } catch { /* ignore */ }
}

async function saveWandbSettings() {
  try {
    await fetch('/api/config/wandb-settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(wandb.value),
    })
    notify.show(t('cfgWandbSaved'), 'success')
  } catch { /* ignore */ }
}

const variantNameRule = (v: string) => {
  if (!v) return true
  return /^[A-Za-z0-9_-]+$/.test(v) ? true : t('cfgNewVariantNameInvalid')
}

const isCustomPreset = computed(() => {
  const builtin = ['default', 'low_vram', 'graft', 'half', 'quarter', 'tenth', 'debug']
  return selectedPreset.value && !builtin.includes(selectedPreset.value)
})

const variantItems = computed(() =>
  configStore.variants.map(v => ({
    value: v,
    label: configStore.variantLabels[v] || v,
  }))
)

// ── Navigation ────────────────────────────────────────────────

async function onMethodChange(method: string) {
  selectedVariant.value = ''
  await configStore.fetchVariants(method)
}

async function onVariantChange() {
  if (selectedVariant.value) {
    await loadConfig()
  }
}

async function loadConfig() {
  if (selectedVariant.value) {
    await configStore.fetchMerged(selectedVariant.value, selectedPreset.value)
    await fetchFieldHelp()
    await checkExperimental()
  }
}

async function fetchFieldHelp() {
  if (!selectedVariant.value) return
  try {
    const lang = appStore.language
    const res = await fetch(`/api/config/field-help?variant=${encodeURIComponent(selectedVariant.value)}&lang=${encodeURIComponent(lang)}`)
    if (res.ok) {
      const data = await res.json()
      fieldHelpData.value = data.field_help || {}
      guideHtml.value = data.guide_html || ''
    }
  } catch {
    // Silently ignore — help is non-critical
  }
}

async function checkExperimental() {
  if (!selectedVariant.value) {
    isExperimental.value = false
    return
  }
  try {
    const res = await fetch(`/api/config/variant-meta?variant=${encodeURIComponent(selectedVariant.value)}`)
    if (res.ok) {
      const data = await res.json()
      isExperimental.value = !!data.experimental
    }
  } catch {
    isExperimental.value = false
  }
}

function onFieldHelp(key: string, origin: string) {
  helpPanelRef.value?.showFieldHelp(key, origin)
}

onMounted(async () => {
  await Promise.all([
    configStore.fetchMethods(),
    configStore.fetchPresets(),
    loadWandbSettings(),
  ])

  // Restore previous selections from store (survives page navigation)
  if (configStore.method) {
    selectedMethod.value = configStore.method
    await configStore.fetchVariants(configStore.method)
    if (configStore.variant) {
      selectedVariant.value = configStore.variant
      selectedPreset.value = configStore.preset
      await loadConfig()
    }
  }
})

// Refetch help data when language changes
watch(() => appStore.language, () => {
  if (selectedVariant.value) fetchFieldHelp()
})

// ── Save ───────────────────────────────────────────────────────

async function onSave() {
  try {
    const args = extraArgs.value.trim() || undefined
    await configStore.save(args)
    if (args) extraArgs.value = ''
    notify.show(t('notifyConfigSaved'), 'success')
  } catch (e: any) {
    notify.show(t('notifyConfigSaveFailed'), 'error')
  }
}

// ── Preset management ────────────────────────────────────────

function openCreatePreset() {
  newPresetName.value = ''
  showCreatePresetDlg.value = true
}

async function confirmCreatePreset() {
  const name = newPresetName.value.trim()
  if (!name) return
  const data: Record<string, unknown> = {}
  for (const f of configStore.fields) {
    if (f.is_virtual) continue
    if (f.key in configStore.editedValues) {
      data[f.key] = configStore.editedValues[f.key]
    } else if (f.origin !== 'base') {
      data[f.key] = f.value
    }
  }
  const ok = await configStore.createPreset(name, data)
  if (ok) {
    showCreatePresetDlg.value = false
    selectedPreset.value = name
    await loadConfig()
    notify.show(t('cfgPresetCreated', { name }), 'success')
  }
}

async function deleteCurrentPreset() {
  const name = selectedPreset.value
  if (!name || !isCustomPreset.value) return
  const ok = await configStore.deletePreset(name)
  if (ok) {
    selectedPreset.value = configStore.presets.includes('default') ? 'default' : (configStore.presets[0] || '')
    await loadConfig()
    notify.show(t('cfgPresetDeleted', { name }), 'success')
  }
}

// ── Variant management ──────────────────────────────────────

function openCreateVariant() {
  newVariantName.value = ''
  seedFromCurrent.value = true
  showCreateVariantDlg.value = true
}

async function confirmCreateVariant() {
  const name = newVariantName.value.trim()
  if (!name) return
  try {
    const body: Record<string, unknown> = { name }
    if (seedFromCurrent.value && selectedVariant.value) {
      body.seed_from = selectedVariant.value
    }
    const res = await fetch('/api/config/variants', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const data = await res.json()
      configStore.error = data.detail || 'Failed to create variant'
      return
    }
    const data = await res.json()
    configStore.variants = data.variants || []
    showCreateVariantDlg.value = false
    selectedVariant.value = `custom/${name}`
    await loadConfig()
    notify.show(t('cfgNewVariantCreated', { name }), 'success')
  } catch (e: any) {
    configStore.error = e.message
  }
}

// ── Prelaunch check ──────────────────────────────────────────

async function fetchPrelaunch(): Promise<any> {
  const params = new URLSearchParams({
    variant: selectedVariant.value,
    preset: selectedPreset.value,
  })
  const res = await fetch(`/api/config/prelaunch-check?${params}`)
  if (!res.ok) throw new Error(`Prelaunch check failed: ${res.status}`)
  return res.json()
}

async function autoSaveIfDirty() {
  if (configStore.dirty || extraArgs.value.trim()) {
    const args = extraArgs.value.trim() || undefined
    await configStore.save(args)
    if (args) extraArgs.value = ''
  }
}

// ── Training launch ──────────────────────────────────────────

async function startTraining() {
  if (!selectedVariant.value) return
  trainingLaunching.value = true
  try {
    await autoSaveIfDirty()

    const result = await fetchPrelaunch()
    prelaunchResult.value = result

    if (!result.has_cache) {
      showNoCacheDlg.value = true
      return
    }

    if (result.checkpoint) {
      checkpointInfo.value = result.checkpoint
      showCheckpointDlg.value = true
      return
    }

    await launchTrainingTask()
  } catch (e: any) {
    configStore.error = e.message
  } finally {
    trainingLaunching.value = false
  }
}

async function launchTrainingTask() {
  const env: Record<string, string> = { PRESET: selectedPreset.value }

  // Inject wandb env vars when enabled
  if (wandb.value.enabled) {
    env['WANDB_ENABLED'] = '1'
    if (wandb.value.project) env['WANDB_PROJECT'] = wandb.value.project
    if (wandb.value.run_name) env['WANDB_RUN_NAME'] = wandb.value.run_name
    if (wandb.value.api_key) env['WANDB_API_KEY'] = wandb.value.api_key
    env['WANDB_LOG_EVERY_N'] = String(wandb.value.log_every_n_steps || 50)
    if (wandb.value.log_gradients) env['WANDB_LOG_GRADIENTS'] = '1'
    if (wandb.value.log_weights) env['WANDB_LOG_WEIGHTS'] = '1'
    if (wandb.value.log_checkpoint_artifact) env['WANDB_LOG_ARTIFACT'] = '1'
  }

  const taskId = await taskStore.startTask('lora-gui', [selectedVariant.value], env)
  if (taskId) {
    notify.show(t('notifyTrainingLaunched'), 'success')
  } else {
    notify.show(t('notifyTaskStartFailed', { command: t('cfgTrain') }), 'error')
  }
}

async function resumeTrain() {
  showCheckpointDlg.value = false
  trainingLaunching.value = true
  try {
    await launchTrainingTask()
  } finally {
    trainingLaunching.value = false
  }
}

async function wipeAndTrain() {
  showCheckpointDlg.value = false
  if (!checkpointInfo.value) return
  trainingLaunching.value = true
  try {
    const stateDir = checkpointInfo.value.state_dir
    const stateDirName = stateDir.split(/[/\\]/).pop() || ''
    const outputName = stateDirName.replace('-checkpoint-state', '')
    const outputDir = stateDir.replace(/[/\\][^/\\]+$/, '').replace(/\\/g, '/')

    await fetch('/api/config/wipe-checkpoint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ output_dir: outputDir, output_name: outputName }),
    })
    await launchTrainingTask()
  } catch (e: any) {
    configStore.error = e.message
  } finally {
    trainingLaunching.value = false
  }
}

async function runPreprocessThenTrain() {
  showNoCacheDlg.value = false
  preprocessRunning.value = true
  try {
    const taskId = await taskStore.startTask('preprocess')
    if (!taskId) {
      configStore.error = 'Failed to start preprocessing'
      return
    }
    await waitForTask(taskId)
    const result = await fetchPrelaunch()
    if (result.has_cache) {
      if (result.checkpoint) {
        checkpointInfo.value = result.checkpoint
        showCheckpointDlg.value = true
      } else {
        await launchTrainingTask()
      }
    }
  } catch (e: any) {
    configStore.error = e.message
  } finally {
    preprocessRunning.value = false
    trainingLaunching.value = false
  }
}

// ── Test inference ────────────────────────────────────────────

async function startTest() {
  if (!selectedVariant.value) return
  testLaunching.value = true
  try {
    await autoSaveIfDirty()
    const meta = await fetch(`/api/config/variant-meta?variant=${selectedVariant.value}`)
    const metaData = await meta.json()
    let command = 'test'
    const family = metaData.family || ''
    if (family === 'hydralora' || family === 'fera') command = 'test-hydra'
    else if (family === 'postfix') command = 'exp-test-postfix'
    else if (family === 'ip_adapter') command = 'exp-test-ip'
    else if (family === 'easycontrol') command = 'exp-test-easycontrol'
    else if (family === 'chimera') command = 'test-hydra'

    // Pass test prompt/negative_prompt from config so the backend uses
    // the user's custom values instead of the hardcoded defaults.
    const args: string[] = []
    const prompt = configStore.getFieldValue('test_prompt')
    if (prompt) args.push('--prompt', String(prompt))
    const negPrompt = configStore.getFieldValue('test_negative_prompt')
    if (negPrompt) args.push('--negative_prompt', String(negPrompt))

    const taskId = await taskStore.startTask(command, args)
    if (taskId) {
      notify.show(t('notifyTestLaunched'), 'success')
    } else {
      notify.show(t('notifyTaskStartFailed', { command: t('cfgTest') }), 'error')
    }
  } catch (e: any) {
    configStore.error = e.message
  } finally {
    testLaunching.value = false
  }
}

// ── Utilities ─────────────────────────────────────────────────

function waitForTask(taskId: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      await taskStore.fetchTasks()
      const task = taskStore.tasks.find(t => t.task_id === taskId)
      if (!task) return
      if (task.state === 'success') {
        clearInterval(interval)
        resolve()
      } else if (task.state === 'failed' || task.state === 'cancelled') {
        clearInterval(interval)
        reject(new Error(`Task ${task.state}`))
      }
    }, 2000)
  })
}
</script>

<style scoped>
.config-editor {
  min-height: 100%;
  overflow-y: auto;
}

.help-panel-sticky {
  position: sticky;
  top: 16px;
}

/* Config section cards: subtle hover lift */
:deep(.v-card) {
  transition: border-color 0.2s, box-shadow 0.2s;
}
:deep(.v-card:hover) {
  border-color: var(--border-default);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}
</style>
