<template>
  <v-container fluid class="pa-4">
    <div class="text-h5 mb-1">{{ t('distTitle') }}</div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('distSubtitle') }}</div>

    <v-row>
      <!-- Left: Config editor -->
      <v-col cols="12" :md="guideHtml ? 8 : 12">
        <!-- Method selector -->
        <v-card variant="tonal" class="mb-4">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-flask" class="mr-2" />
            {{ t('distTitle') }}
          </v-card-title>
          <v-card-text>
            <v-select
              v-model="selectedMethod"
              :items="methods"
              item-title="label"
              item-value="key"
              :label="t('distMethod')"
              density="compact"
              hide-details
              @update:model-value="loadConfig"
            />
          </v-card-text>
        </v-card>

        <!-- Config sections -->
        <template v-if="config.sections.length">
          <v-card
            v-for="section in config.sections"
            :key="section.name"
            variant="outlined"
            class="mb-3"
          >
            <v-card-title class="text-subtitle-2">
              {{ section.name === 'root' ? t('distGeneral') : section.name }}
            </v-card-title>
            <v-card-text>
              <v-form>
                <template v-for="field in section.fields" :key="field.key">
                  <v-switch
                    v-if="field.type === 'bool'"
                    v-model="field.value"
                    :label="field.key"
                    :hint="field.comment"
                    density="compact"
                    hide-details="auto"
                    class="mb-1"
                  />
                  <v-text-field
                    v-else-if="field.type === 'int'"
                    v-model.number="field.value"
                    :label="field.key"
                    type="number"
                    :hint="field.comment"
                    density="compact"
                    hide-details="auto"
                    class="mb-1"
                  />
                  <v-text-field
                    v-else-if="field.type === 'float'"
                    v-model.number="field.value"
                    :label="field.key"
                    type="number"
                    step="any"
                    :hint="field.comment"
                    density="compact"
                    hide-details="auto"
                    class="mb-1"
                  />
                  <v-text-field
                    v-else-if="field.type === 'list'"
                    :model-value="JSON.stringify(field.value)"
                    @update:model-value="(v: string) => { try { field.value = JSON.parse(v) } catch {} }"
                    :label="field.key"
                    :hint="field.comment"
                    density="compact"
                    hide-details="auto"
                    class="mb-1"
                  />
                  <v-text-field
                    v-else
                    v-model="field.value"
                    :label="field.key"
                    :hint="field.comment"
                    density="compact"
                    hide-details="auto"
                    class="mb-1"
                  />
                </template>
              </v-form>
            </v-card-text>
          </v-card>
        </template>

        <!-- Action bar -->
        <div class="d-flex ga-2 mt-4">
          <v-btn
            color="warning"
            prepend-icon="mdi-content-save"
            :loading="saving"
            @click="saveConfig"
          >
            {{ t('distSave') }}
          </v-btn>
          <v-btn
            color="success"
            prepend-icon="mdi-play-circle"
            :loading="isRunning"
            :disabled="isRunning"
            @click="startTraining"
          >
            {{ t('distTrain') }}
          </v-btn>
          <v-btn
            color="error"
            prepend-icon="mdi-stop"
            :disabled="!isRunning"
            @click="stopTraining"
          >
            {{ t('distStop') }}
          </v-btn>
          <v-spacer />
          <v-btn
            v-if="!guideHtml"
            variant="text"
            prepend-icon="mdi-book-open-variant"
            @click="loadGuide"
          >
            {{ t('distGuide') }}
          </v-btn>
        </div>
      </v-col>

      <!-- Right: Guide panel -->
      <v-col v-if="guideHtml" cols="12" md="4">
        <v-card variant="tonal" class="guide-card">
          <v-card-title class="text-subtitle-1 d-flex align-center">
            <v-icon icon="mdi-book-open-variant" class="mr-2" />
            {{ t('distGuide') }}
            <v-spacer />
            <v-btn icon="mdi-close" size="small" variant="text" @click="guideHtml = ''" />
          </v-card-title>
          <v-card-text class="guide-html" v-html="guideHtml" />
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useI18n } from '../composables/useI18n'

const { t } = useI18n()
const taskStore = useTaskStore()
const notify = useNotifyStore()
taskStore.fetchTasks()

interface DistillMethodSummary {
  key: string
  label: string
  config_path: string
  task_command: string
}

interface DistillField {
  key: string
  value: any
  type: string
  comment: string
}

interface DistillSection {
  name: string
  fields: DistillField[]
}

interface DistillConfigResponse {
  method: string
  sections: DistillSection[]
}

const methods = ref<DistillMethodSummary[]>([])
const selectedMethod = ref('')
const config = reactive<DistillConfigResponse>({ method: '', sections: [] })
const saving = ref(false)
const guideHtml = ref('')

const isRunning = computed(() => {
  const method = methods.value.find(m => m.key === selectedMethod.value)
  if (!method) return false
  return taskStore.tasks.some(
    tp => tp.command === method.task_command && tp.state === 'running'
  )
})

async function loadMethods() {
  const res = await fetch('/api/distill/methods')
  methods.value = await res.json()
  if (methods.value.length) {
    selectedMethod.value = methods.value[0].key
    await loadConfig()
  }
}

async function loadConfig() {
  if (!selectedMethod.value) return
  const res = await fetch(`/api/distill/config?method=${selectedMethod.value}`)
  const data = await res.json()
  Object.assign(config, data)
  guideHtml.value = ''
}

async function loadGuide() {
  if (!selectedMethod.value) return
  const appStore = (await import('../stores/app')).useAppStore()
  const res = await fetch(
    `/api/distill/guide?method=${selectedMethod.value}&lang=${appStore.language}`
  )
  const data = await res.json()
  guideHtml.value = data.html || ''
}

async function saveConfig() {
  saving.value = true
  try {
    const updates: Record<string, Record<string, any>> = {}
    for (const section of config.sections) {
      updates[section.name] = {}
      for (const field of section.fields) {
        updates[section.name][field.key] = field.value
      }
    }
    const res = await fetch(`/api/distill/config?method=${selectedMethod.value}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates }),
    })
    if (res.ok) {
      notify.show(t('distSaved'), 'success')
    } else {
      notify.show(t('distSaveFailed'), 'error')
    }
  } finally {
    saving.value = false
  }
}

async function startTraining() {
  const method = methods.value.find(m => m.key === selectedMethod.value)
  if (!method) return
  await saveConfig()
  const taskId = await taskStore.startTask(method.task_command)
  if (taskId) {
    notify.show(t('notifyTaskStarted', { command: method.task_command }), 'success')
  } else {
    notify.show(t('notifyTaskStartFailed', { command: method.task_command }), 'error')
  }
}

function stopTraining() {
  const method = methods.value.find(m => m.key === selectedMethod.value)
  if (!method) return
  const task = taskStore.tasks.find(
    tp => tp.command === method.task_command && tp.state === 'running'
  )
  if (task) taskStore.cancelTask(task.task_id)
}

onMounted(loadMethods)
</script>

<style scoped>
.guide-card {
  position: sticky;
  top: 16px;
  max-height: calc(100vh - 120px);
  overflow-y: auto;
}

.guide-html :deep(h1),
.guide-html :deep(h2),
.guide-html :deep(h3) {
  margin-top: 0.8em;
  margin-bottom: 0.4em;
}
.guide-html :deep(p) {
  margin-bottom: 0.5em;
}
.guide-html :deep(code) {
  background: rgba(var(--v-border-color), 0.1);
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 0.85em;
}
.guide-html :deep(pre) {
  background: rgba(var(--v-border-color), 0.08);
  padding: 8px;
  border-radius: 4px;
  overflow-x: auto;
}
</style>
