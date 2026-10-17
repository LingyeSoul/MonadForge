<template>
  <div class="d-flex align-center flex-wrap ga-2 mb-4">
    <v-chip size="small" variant="tonal">{{ qwen.label }} · {{ qwen.profile }}</v-chip>
    <span class="text-caption text-medium-emphasis">{{ t('qwFollowsConfig') }}</span>
    <v-spacer /><v-btn to="/config" size="small" variant="outlined">{{ t('navConfig') }}</v-btn>
  </div>
  <v-alert v-if="qwen.error" type="error" variant="tonal" class="mb-4" data-testid="qwen-workspace-error">{{ qwen.error }}</v-alert>
  <v-progress-linear v-if="qwen.loading" indeterminate />
  <template v-if="qwen.draft">
    <v-card variant="tonal" class="mb-4">
      <v-card-title class="text-subtitle-1 d-flex align-center">{{ t('ppStatus') }}<v-spacer /><v-btn data-testid="qwen-cache-refresh" icon="mdi-refresh" variant="text" size="small" :aria-label="t('ppRefresh')" @click="refresh" /></v-card-title>
      <v-card-text v-if="qwen.cache">
        <v-chip size="small" :color="qwen.cache.ready ? 'success' : 'warning'" variant="tonal">{{ t('qwCacheSummary', { pairs: qwen.cache.pairs, text: qwen.cache.missing_text, latents: qwen.cache.missing_latents }) }}</v-chip>
        <p class="text-caption mt-2" style="overflow-wrap: anywhere">{{ qwen.cache.directory }}</p>
        <v-alert v-if="qwen.cache.errors.length" type="error" variant="tonal" class="mt-3"><div v-for="error in qwen.cache.errors" :key="error">{{ error }}</div></v-alert>
      </v-card-text>
    </v-card>
    <v-row>
      <v-col cols="12" lg="8">
        <v-card variant="tonal" class="mb-4">
          <v-card-title class="text-subtitle-1">{{ t('ppDatasetPaths') }}</v-card-title>
          <v-card-text><QwenFields section="cache" :fields="basicFields" @help="openHelp" /></v-card-text>
          <v-card-actions class="flex-wrap">
            <v-btn data-testid="qwen-cache-save" variant="text" :disabled="!qwen.dirty || qwen.busy" @click="qwen.perform(qwen.save)">{{ t('cfgSave') }}</v-btn>
            <v-btn data-testid="qwen-use-training-cache" variant="text" :disabled="qwen.busy" @click="qwen.update('train', 'cache', qwen.draft.cache.out)">{{ t('qwUseTrainingCache') }}</v-btn>
          </v-card-actions>
        </v-card>
        <v-expansion-panels class="mb-4" variant="accordion">
          <v-expansion-panel :title="t('ppSettings')" data-testid="qwen-cache-advanced"><v-expansion-panel-text><QwenFields section="cache" :fields="advancedFields" @help="openHelp" /></v-expansion-panel-text></v-expansion-panel>
        </v-expansion-panels>
        <p class="text-body-2 text-medium-emphasis mb-3">{{ cacheHint }}</p>
        <div class="d-flex ga-2 flex-wrap">
          <v-btn data-testid="qwen-cache-submit" color="primary" :loading="qwen.busy" @click="submitCache({})">{{ t('qwCacheSelected') }}</v-btn>
          <v-btn data-testid="qwen-cache-text" variant="outlined" :disabled="qwen.busy" @click="submitCache({ skip_text: false, skip_latents: true })">{{ t('qwCacheText') }}</v-btn>
          <v-btn data-testid="qwen-cache-vae" variant="outlined" :disabled="qwen.busy" @click="submitCache({ skip_text: true, skip_latents: false })">{{ t('qwCacheVae') }}</v-btn>
        </div>
      </v-col>
      <v-col cols="12" lg="4"><HelpPanel ref="helpPanel" :variant="`${qwen.family}/${qwen.profile}`" :field-help="help" :guide-html="guide" height="500" /></v-col>
    </v-row>
    <h2 class="text-subtitle-1 mt-5 mb-3">{{ t('ppActiveTasks') }}</h2>
    <v-list v-if="tasks.length" density="compact">
      <v-list-item v-for="task in tasks" :key="task.task_id" :title="task.command" :subtitle="task.state" :to="{ path: '/tasks', query: { task: task.task_id } }" />
    </v-list>
    <p v-else class="text-body-2 text-medium-emphasis">{{ t('ppNoTasks') }}</p>
  </template>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from '../composables/useI18n'
import { useActiveModelWorkspace } from '../stores/qwenWorkspace'
import { useAppStore } from '../stores/app'
import { modelFieldLabel, mossText } from '../i18n/modelWorkspace'
import { useTaskStore } from '../stores/task'
import { modelFields, type Values } from '../utils/qwen21'
import QwenFields from './QwenFields.vue'
import HelpPanel from './HelpPanel.vue'

const qwen = useActiveModelWorkspace()
const app = useAppStore()
const taskStore = useTaskStore()
const router = useRouter()
const { t } = useI18n()
const helpPanel = ref<InstanceType<typeof HelpPanel> | null>(null)
const fields = computed(() => qwen.value.schema?.cache.filter(field => !modelFields.has(field.name)) ?? [])
const basicFields = computed(() => fields.value.filter(field => ['src', 'out', 'resolution'].includes(field.name)))
const advancedFields = computed(() => fields.value.filter(field => !basicFields.value.includes(field)))
const help = computed(() => Object.fromEntries(fields.value.map(field => [`cache.${field.name}`, `${modelFieldLabel(qwen.value.family, app.language, field.name, t(`qwenField_${field.name}`))}\n${field.help}`])))
const cacheHint = computed(() => qwen.value.family === 'moss' ? mossText(app.language, 'cacheHint') : t('qwenCacheHint'))
const guide = computed(() => `<p>${cacheHint.value}</p><p>${qwen.value.family === 'moss' ? mossText(app.language, 'cacheIsolation') : t('qwCacheIsolation')}</p><p>${qwen.value.family === 'moss' ? mossText(app.language, 'sourcesHint') : t('qwenSourcesHint')}</p>`)
const tasks = computed(() => taskStore.tasks.filter(task => [`${qwen.value.family}-cache`, `${qwen.value.family}-cache-train`].includes(task.command)))
function openHelp(key: string): void { helpPanel.value?.showFieldHelp(key, 'method') }
async function refresh(): Promise<void> { const selected = qwen.value; await selected.perform(async () => { await selected.inspectCache(); await taskStore.fetchTasks() }) }
async function submitCache(overrides: Values): Promise<void> {
  const selected = qwen.value
  await selected.perform(async () => {
    if (selected.dirty) await selected.save()
    const taskId = await selected.submit('cache', overrides)
    await taskStore.fetchTasks()
    await router.push({ path: '/tasks', query: { task: taskId } })
  })
}
onMounted(() => { const selected = qwen.value; return selected.perform(async () => { await selected.ensureLoaded(); await selected.inspectCache(); await taskStore.fetchTasks() }) })
</script>
