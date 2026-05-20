<template>
  <v-container fluid class="pa-4 config-editor">
    <v-row align="center" class="mb-4">
      <v-col cols="12" md="3">
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
      <v-col cols="12" md="2">
        <v-select
          v-model="selectedPreset"
          :items="configStore.presets"
          :label="t('cfgPreset')"
          variant="outlined"
          density="compact"
          hide-details
          @update:model-value="onVariantChange"
        />
      </v-col>
      <v-col cols="12" md="4" class="d-flex justify-end ga-2 flex-wrap">
        <v-btn
          :color="configStore.dirty ? 'warning' : 'primary'"
          :loading="configStore.loading"
          :disabled="!configStore.dirty"
          prepend-icon="mdi-content-save"
          @click="onSave"
        >
          {{ t('cfgSave') }}{{ configStore.dirty ? ' *' : '' }}
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
              cols="12"
              md="6"
            >
              <ConfigField
                :field="field"
                @update="(v) => configStore.setFieldValue(field.key, v)"
              />
            </v-col>
          </v-row>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>

    <div v-if="!configStore.loading && configStore.fields.length === 0" class="text-center pa-12">
      <v-icon icon="mdi-cog-transfer-outline" size="64" color="grey" class="mb-4" />
      <div class="text-h6 text-medium-emphasis">{{ t('cfgSelectHint') }}</div>
    </div>

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
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useConfigStore } from '../stores/config'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useI18n } from '../composables/useI18n'
import ConfigField from '../components/ConfigField.vue'

const configStore = useConfigStore()
const taskStore = useTaskStore()
const notify = useNotifyStore()
const { t } = useI18n()

const selectedMethod = ref('')
const selectedVariant = ref('')
const selectedPreset = ref('default')

const trainingLaunching = ref(false)
const testLaunching = ref(false)
const preprocessRunning = ref(false)

// Dialog state
const showNoCacheDlg = ref(false)
const showCheckpointDlg = ref(false)
const checkpointInfo = ref<{ state_dir: string; step: number } | null>(null)
const prelaunchResult = ref<any>(null)

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
  }
}

onMounted(async () => {
  await Promise.all([
    configStore.fetchMethods(),
    configStore.fetchPresets(),
  ])
})

// ── Save ───────────────────────────────────────────────────────

async function onSave() {
  try {
    await configStore.save()
    notify.show(t('notifyConfigSaved'), 'success')
  } catch (e: any) {
    notify.show(t('notifyConfigSaveFailed'), 'error')
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
  if (configStore.dirty) {
    await configStore.save()
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

    // No cache → prompt preprocess
    if (!result.has_cache) {
      showNoCacheDlg.value = true
      return
    }

    // Has checkpoint → prompt resume/wipe
    if (result.checkpoint) {
      checkpointInfo.value = result.checkpoint
      showCheckpointDlg.value = true
      return
    }

    // All clear → launch training
    await launchTrainingTask()
  } catch (e: any) {
    configStore.error = e.message
  } finally {
    trainingLaunching.value = false
  }
}

async function launchTrainingTask() {
  const taskId = await taskStore.startTask('lora-gui', [selectedVariant.value])
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
    // Extract output_dir and output_name from the state_dir path
    // state_dir is like "output/ckpt/last-checkpoint-state"
    const stateDir = checkpointInfo.value.state_dir
    const stateDirName = stateDir.split(/[/\\]/).pop() || ''
    const outputName = stateDirName.replace('-checkpoint-state', '')
    // Derive output_dir by removing the state dir name from the path
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
    // Start preprocessing, then poll for completion
    const taskId = await taskStore.startTask('preprocess')
    if (!taskId) {
      configStore.error = 'Failed to start preprocessing'
      return
    }
    // Poll until preprocess finishes
    await waitForTask(taskId)
    // Re-check and launch training
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
    // Determine test command based on variant family
    const meta = await fetch(`/api/config/variant-meta?variant=${selectedVariant.value}`)
    const metaData = await meta.json()
    let command = 'test'
    const family = metaData.family || ''
    if (family === 'hydralora' || family === 'fera') command = 'test-hydra'
    else if (family === 'postfix') command = 'exp-test-postfix'
    else if (family === 'ip_adapter') command = 'exp-test-ip'
    else if (family === 'easycontrol') command = 'exp-test-easycontrol'
    else if (family === 'chimera') command = 'test-hydra'

    const taskId = await taskStore.startTask(command)
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
</style>
