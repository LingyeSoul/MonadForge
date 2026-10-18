<template>
  <v-alert v-if="qwen.error" type="error" variant="tonal" class="mb-4" data-testid="qwen-workspace-error">
    {{ qwen.error }}
    <template #append><v-btn data-testid="preprocess-retry" variant="text" size="small" :loading="qwen.busy" @click="loadWorkspace">{{ t('cfgReload') }}</v-btn></template>
  </v-alert>
  <v-progress-linear v-if="qwen.loading" indeterminate />
  <template v-if="qwen.draft">
    <v-card variant="tonal" class="mb-4">
      <v-card-title class="text-subtitle-1 d-flex align-center flex-wrap ga-2">
        <v-icon icon="mdi-source-branch" />{{ t('qwProfile') }}<v-spacer />
        <v-btn to="/config" size="small" variant="text">{{ t('navConfig') }}</v-btn>
      </v-card-title>
      <v-card-text>
        <v-chip size="small" variant="tonal">{{ qwen.label }} · {{ qwen.profile }}</v-chip>
        <v-chip v-if="qwen.dirty" size="small" color="warning" variant="tonal" class="ml-2">{{ t('cfgPendingChanges') }}</v-chip>
        <p class="text-caption text-medium-emphasis mt-2">{{ t('qwFollowsConfig') }}</p>
      </v-card-text>
    </v-card>

    <v-card variant="tonal" class="mb-4">
      <v-card-title class="text-subtitle-1 d-flex align-center">
        <v-icon icon="mdi-chart-box-outline" class="mr-2" />{{ t('ppStatus') }}<v-spacer />
        <v-btn data-testid="qwen-cache-refresh" icon="mdi-refresh" variant="text" size="small" :loading="refreshing" :disabled="qwen.busy" :aria-label="t('ppRefresh')" @click="refresh" />
      </v-card-title>
      <v-card-text v-if="qwen.cache">
        <div class="d-flex flex-wrap ga-2">
          <v-chip prepend-icon="mdi-check-all" :color="qwen.cache.ready ? 'success' : 'default'" variant="tonal">{{ p('pairs') }}: {{ qwen.cache.pairs }}</v-chip>
          <v-chip prepend-icon="mdi-text-box-outline" :color="qwen.cache.missing_text ? 'warning' : 'default'" variant="tonal">{{ p('missingText') }}: {{ qwen.cache.missing_text }}</v-chip>
          <v-chip prepend-icon="mdi-cached" :color="qwen.cache.missing_latents ? 'warning' : 'default'" variant="tonal">{{ p('missingLatents') }}: {{ qwen.cache.missing_latents }}</v-chip>
        </div>
        <p class="text-caption text-medium-emphasis mt-3 cache-directory">{{ qwen.cache.directory }}</p>
        <v-alert v-if="qwen.cache.errors.length" type="error" variant="tonal" class="mt-3"><div v-for="error in qwen.cache.errors" :key="error">{{ error }}</div></v-alert>
      </v-card-text>
      <v-card-text v-else class="text-body-2 text-medium-emphasis">{{ p('staleStatus') }}</v-card-text>
    </v-card>

    <v-card variant="tonal" class="mb-4 preprocess-path-card">
      <v-card-title class="text-subtitle-1"><v-icon icon="mdi-folder-outline" class="mr-2" />{{ t('ppDatasetPaths') }}</v-card-title>
      <v-card-text><QwenFields section="cache" :fields="pathFields" @help="openHelp" /></v-card-text>
      <v-card-actions class="flex-wrap ga-2">
        <v-btn data-testid="qwen-cache-save" color="primary" size="small" prepend-icon="mdi-content-save" :loading="saving" :disabled="!pathsDirty || qwen.busy" @click="save">{{ t('ppSavePaths') }}</v-btn>
        <v-btn data-testid="qwen-cache-paths-cancel" size="small" :disabled="!pathsDirty || qwen.busy" @click="discardPaths">{{ t('cfCancel') }}</v-btn>
        <v-btn data-testid="qwen-use-training-cache" size="small" :disabled="qwen.busy || qwen.draft.train.cache === qwen.draft.cache.out" @click="qwen.update('train', 'cache', qwen.draft.cache.out)">{{ t('qwUseTrainingCache') }}</v-btn>
      </v-card-actions>
    </v-card>

    <v-expansion-panels class="mb-4">
      <v-expansion-panel data-testid="qwen-cache-advanced">
        <v-expansion-panel-title><v-icon icon="mdi-cog-outline" class="mr-2" />{{ t('ppSettings') }}</v-expansion-panel-title>
        <v-expansion-panel-text>
          <QwenFields section="cache" :fields="settingsFields" @help="openHelp" />
          <div class="d-flex flex-wrap ga-2 mt-3">
            <v-btn data-testid="qwen-cache-settings-save" color="primary" size="small" prepend-icon="mdi-content-save" :loading="saving" :disabled="!settingsDirty || qwen.busy" @click="save">{{ t('ppSaveSettings') }}</v-btn>
            <v-btn data-testid="qwen-cache-settings-cancel" size="small" variant="text" :disabled="!settingsDirty || qwen.busy" @click="discardFields(settingsFields)">{{ t('cfCancel') }}</v-btn>
          </div>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>

    <v-row class="preprocess-task-grid">
      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1"><v-icon icon="mdi-cached" class="mr-2" />{{ t('ppCacheVae') }}</v-card-title>
          <v-card-text>{{ p('vaeHint') }}</v-card-text>
          <v-card-actions><v-btn data-testid="qwen-cache-vae" color="primary" :loading="submitting === 'vae'" :disabled="qwen.busy && submitting !== 'vae'" @click="submitCache('vae', { skip_text: true, skip_latents: false })">{{ t('ppRunVae') }}</v-btn></v-card-actions>
        </v-card>
      </v-col>
      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1"><v-icon icon="mdi-text-box-outline" class="mr-2" />{{ t('ppCacheTe') }}</v-card-title>
          <v-card-text>{{ p('textHint') }}</v-card-text>
          <v-card-actions><v-btn data-testid="qwen-cache-text" color="primary" :loading="submitting === 'text'" :disabled="qwen.busy && submitting !== 'text'" @click="submitCache('text', { skip_text: false, skip_latents: true })">{{ t('ppRunTe') }}</v-btn></v-card-actions>
        </v-card>
      </v-col>
      <v-col cols="12" md="4">
        <v-card variant="tonal" color="primary">
          <v-card-title class="text-subtitle-1"><v-icon icon="mdi-pipe" class="mr-2" />{{ t('ppPipeline') }}</v-card-title>
          <v-card-text>{{ cacheHint }}</v-card-text>
          <v-card-actions><v-btn data-testid="qwen-cache-submit" color="primary" :loading="submitting === 'all'" :disabled="(qwen.busy && submitting !== 'all') || Boolean(qwen.draft.cache.skip_text && qwen.draft.cache.skip_latents)" @click="submitCache('all', {})">{{ t('qwCacheSelected') }}</v-btn></v-card-actions>
        </v-card>
      </v-col>
    </v-row>

    <PreprocessTaskList :tasks="tasks" :refreshing="refreshing" @refresh="refresh" />
  </template>

  <v-dialog v-model="showHelp" max-width="640" scrollable>
    <v-card>
      <HelpPanel :key="helpSession" ref="helpPanel" :variant="`${qwen.family}/${qwen.profile}`" :field-help="help" :guide-html="guide" max-height="70dvh" />
      <v-card-actions><v-spacer /><v-btn @click="showHelp = false">{{ t('guidebookClose') }}</v-btn></v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'
import { useI18n } from '../composables/useI18n'
import { useActiveModelWorkspace } from '../stores/qwenWorkspace'
import { useAppStore } from '../stores/app'
import { useNotifyStore } from '../stores/notify'
import { modelFieldLabel, mossText } from '../i18n/modelWorkspace'
import { preprocessText } from '../i18n/preprocess'
import { useTaskStore } from '../stores/task'
import { modelFields, type QwenField, type Values } from '../utils/qwen21'
import QwenFields from './QwenFields.vue'
import HelpPanel from './HelpPanel.vue'
import PreprocessTaskList from './PreprocessTaskList.vue'

const qwen = useActiveModelWorkspace()
const app = useAppStore()
const taskStore = useTaskStore()
const notify = useNotifyStore()
const { t } = useI18n()
const p = (key: Parameters<typeof preprocessText>[1]) => preprocessText(app.language, key)
const helpPanel = ref<InstanceType<typeof HelpPanel> | null>(null)
const showHelp = ref(false)
const helpSession = ref(0)
const refreshing = ref(false)
const saving = ref(false)
const submitting = ref('')
const fields = computed(() => qwen.value.schema?.cache.filter(field => !modelFields.has(field.name)) ?? [])
const pathFields = computed(() => fields.value.filter(field => ['src', 'out'].includes(field.name)))
const settingsFields = computed(() => fields.value.filter(field => !pathFields.value.includes(field)))
const saved = computed(() => qwen.value.saved ? JSON.parse(qwen.value.saved) : null)
const fieldsDirty = (entries: QwenField[]) => entries.some(field => qwen.value.draft?.cache[field.name] !== saved.value?.cache[field.name])
const pathsDirty = computed(() => fieldsDirty(pathFields.value) || qwen.value.draft?.train.cache !== saved.value?.train.cache)
const settingsDirty = computed(() => fieldsDirty(settingsFields.value))
const help = computed(() => Object.fromEntries(fields.value.map(field => [`cache.${field.name}`, `${modelFieldLabel(qwen.value.family, app.language, field.name, t(`qwenField_${field.name}`))}\n${field.help}`])))
const cacheHint = computed(() => qwen.value.family === 'moss' ? mossText(app.language, 'cacheHint') : t('qwenCacheHint'))
const guide = computed(() => `<p>${cacheHint.value}</p><p>${qwen.value.family === 'moss' ? mossText(app.language, 'cacheIsolation') : t('qwCacheIsolation')}</p><p>${qwen.value.family === 'moss' ? mossText(app.language, 'sourcesHint') : t('qwenSourcesHint')}</p>`)
const tasks = computed(() => taskStore.tasks.filter(task => [`${qwen.value.family}-cache`, `${qwen.value.family}-cache-train`].includes(task.command)))
async function openHelp(key?: string): Promise<void> {
  helpSession.value++
  showHelp.value = true
  await nextTick()
  if (key) helpPanel.value?.showFieldHelp(key, 'method')
}
function discardFields(entries: QwenField[]): void {
  if (!saved.value) return
  for (const field of entries) qwen.value.update('cache', field.name, saved.value.cache[field.name])
}
function discardPaths(): void {
  discardFields(pathFields.value)
  if (saved.value) qwen.value.update('train', 'cache', saved.value.train.cache)
}
async function save(): Promise<void> {
  const selected = qwen.value
  if (selected.busy) return
  saving.value = true
  try { await selected.perform(async () => { await selected.save(); notify.show(t('notifyConfigSaved'), 'success') }) }
  finally { saving.value = false }
  if (selected.error) notify.show(selected.error, 'error')
}
async function refresh(): Promise<void> {
  const selected = qwen.value
  if (selected.busy) return
  refreshing.value = true
  try { await selected.perform(async () => { await selected.inspectCache(); await taskStore.fetchTasks() }) }
  finally { refreshing.value = false }
}
async function submitCache(kind: string, overrides: Values): Promise<void> {
  const selected = qwen.value
  if (selected.busy) return
  submitting.value = kind
  try {
    await selected.perform(async () => {
      if (selected.dirty) await selected.save()
      await selected.submit('cache', overrides)
      await taskStore.fetchTasks()
      notify.show(t('notifyTaskStarted', { command: `${selected.family}-cache` }), 'success')
    })
  } finally { submitting.value = '' }
  if (selected.error) notify.show(selected.error, 'error')
}
defineExpose({ openHelp })
async function loadWorkspace(): Promise<void> {
  const selected = qwen.value
  await selected.perform(async () => { await selected.ensureLoaded(); await selected.inspectCache(); await taskStore.fetchTasks() })
}
onMounted(loadWorkspace)
</script>

<style scoped>
.cache-directory { overflow-wrap: anywhere; }
.preprocess-path-card :deep(.v-card-text) { padding-top: 12px; }
.preprocess-path-card :deep(.v-card-actions) { padding: 8px 12px 12px; }
</style>
