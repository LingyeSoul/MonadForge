<template>
  <v-alert v-if="qwen.error" type="error" variant="tonal" class="mb-4" data-testid="qwen-workspace-error">{{ qwen.error }}</v-alert>
  <v-progress-linear v-if="qwen.loading" indeterminate />
  <ConfigWorkspaceLayout v-if="qwen.draft && qwen.schema" v-model:search="search" v-model:show-guide="showHelp"
    :field-count="fieldCount" filterable search-test-id="qwen-field-search" help-test-id="qwen-help-toggle">
    <v-card v-if="basicGroup" class="mb-4 config-section" variant="flat">
      <v-card-title class="text-subtitle-1">
        <v-icon icon="mdi-tune" class="mr-2" />
        {{ t(basicGroup.titleKey) }}
        <v-chip size="x-small" class="ml-2" variant="outlined">{{ basicFieldCount }}</v-chip>
      </v-card-title>
      <v-card-text>
        <QwenFields v-for="section in basicGroup.sections" :key="`basic-${section.section}`" :section="section.section" :fields="section.fields" @help="openHelp" />
      </v-card-text>
    </v-card>
    <v-expansion-panels v-if="panelGroups.length" :model-value="searching ? panelGroups.map(group => group.id) : expandedGroups"
      multiple variant="accordion" @update:model-value="onExpandedGroups">
      <v-expansion-panel v-for="group in panelGroups" :key="group.id" :value="group.id" :data-testid="`model-config-group-${group.id}`">
        <v-expansion-panel-title>
          <v-icon icon="mdi-cog-outline" class="mr-2" size="small" />
          {{ t(group.titleKey) }}
          <v-chip size="x-small" class="ml-2" variant="outlined">{{ group.sections.reduce((n, section) => n + section.fields.length, 0) }}</v-chip>
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <QwenFields v-for="section in group.sections" :key="section.section" :section="section.section" :fields="section.fields" @help="openHelp" />
          <template v-if="group.id === 'paths'">
            <v-btn data-testid="qwen-resolve" size="small" variant="text" class="mt-3" @click="inspectPaths">{{ t('qwenResolve') }}</v-btn>
            <dl v-if="paths" class="qwen-paths text-caption"><template v-for="(path, name) in paths" :key="name"><dt>{{ name }}</dt><dd>{{ path }}</dd></template></dl>
          </template>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>
    <div v-if="!groups.length" class="workspace-empty" role="status">
      <v-icon icon="mdi-magnify" size="32" />
      <h2>{{ t('cfgNoMatchingFields') }}</h2>
      <v-btn variant="outlined" @click="search = ''">{{ t('cfgClearSearch') }}</v-btn>
    </div>
    <div class="config-secondary-actions mt-4">
      <v-btn v-if="qwen.legacyAvailable" data-testid="qwen-import-legacy" variant="text" size="small" :disabled="qwen.busy" @click="qwen.perform(async () => qwen.importLegacy())">{{ t('qwImportLegacy') }}</v-btn>
      <v-btn data-testid="qwen-reset-defaults" variant="text" size="small" :disabled="qwen.busy" @click="qwen.resetDefaults()">{{ t('qwenReset') }}</v-btn>
      <v-btn to="/preprocess" variant="text" size="small" prepend-icon="mdi-image-sync-outline">{{ t('navPreprocess') }}</v-btn>
    </div>
    <template #guide>
      <HelpPanel ref="helpPanel" :variant="`${qwen.family}/${qwen.profile}`" :field-help="help" :guide-html="guide"
        height="min(680px, calc(100dvh - 120px))" />
    </template>
  </ConfigWorkspaceLayout>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from '../composables/useI18n'
import { useActiveModelWorkspace } from '../stores/qwenWorkspace'
import { useAppStore } from '../stores/app'
import { modelFieldLabel, mossText } from '../i18n/modelWorkspace'
import { requestJson, resolvedPaths } from '../utils/qwen21'
import { matchesConfigSearch } from '../utils/configSearch'
import { BASIC_GROUP_ID, groupModelConfigFields } from '../utils/modelConfig'
import ConfigWorkspaceLayout from './ConfigWorkspaceLayout.vue'
import HelpPanel from './HelpPanel.vue'
import QwenFields from './QwenFields.vue'

const qwen = useActiveModelWorkspace()
const app = useAppStore()
const { t } = useI18n()
const search = ref<string | null>('')
const showHelp = ref(true)
const expandedGroups = ref<string[]>([])
const searching = computed(() => Boolean(search.value?.trim()))
const helpPanel = ref<InstanceType<typeof HelpPanel> | null>(null)
const paths = ref<Record<string, string> | null>(null)
const allGroups = computed(() => groupModelConfigFields(qwen.value.schema, qwen.value.draft, qwen.value.family))
const fieldLabel = (name: string) => modelFieldLabel(qwen.value.family, app.language, name, t(`qwenField_${name}`))
const groups = computed(() => allGroups.value.map(group => ({ ...group, sections: group.sections.map(section => ({ ...section,
  fields: section.fields.filter(field => matchesConfigSearch({ key: field.name, description: field.help }, search.value, [fieldLabel(field.name)])),
})).filter(section => section.fields.length) })).filter(group => group.sections.length))
const basicGroup = computed(() => groups.value.find(group => group.id === BASIC_GROUP_ID))
const panelGroups = computed(() => groups.value.filter(group => group.id !== BASIC_GROUP_ID))
const basicFieldCount = computed(() => basicGroup.value?.sections.reduce((n, section) => n + section.fields.length, 0) ?? 0)
const fieldCount = computed(() => groups.value.reduce((total, group) => total + group.sections.reduce((n, section) => n + section.fields.length, 0), 0))
const help = computed(() => Object.fromEntries(allGroups.value.flatMap(group => group.sections.flatMap(section => section.fields.map(field => [`${section.section}.${field.name}`, `${fieldLabel(field.name)}\n${field.help}`])))))
const guide = computed(() => (qwen.value.family === 'moss' ? [mossText(app.language, 'sourcesHint'), mossText(app.language, 'autoHint'), mossText(app.language, 'generateHint'), mossText(app.language, 'unsupported'), t('qwResultIsolation')] : [t('qwenSourcesHint'), t('qwenAutoHint'), t('qwenGenerateHint'), t('qwUnsupported'), t('qwResultIsolation')]).map(text => `<p>${text}</p>`).join(''))
function onExpandedGroups(value: unknown): void {
  if (!searching.value && Array.isArray(value)) expandedGroups.value = value.map(String)
}
async function openHelp(key: string): Promise<void> {
  showHelp.value = true
  await nextTick()
  helpPanel.value?.showFieldHelp(key, 'method')
}
async function inspectPaths(): Promise<void> {
  const selected = qwen.value
  await selected.perform(async () => {
    const result = resolvedPaths(await requestJson(`${selected.apiRoot}/resolve`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values: { ...selected.draft?.train, ...selected.draft?.models } }) }), selected.family)
    if (qwen.value === selected) paths.value = result
  })
}
watch(() => [qwen.value.draft?.models, qwen.value.draft?.train.model], () => { paths.value = null }, { deep: true })
</script>

<style scoped>
.config-secondary-actions { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; padding-block: 12px; border-top: 1px solid var(--border-subtle); }
.qwen-paths { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 6px 12px; font-family: var(--font-mono); }
.qwen-paths dd { margin: 0; overflow-wrap: anywhere; }
@media (max-width: 600px) { .qwen-paths { grid-template-columns: 1fr; } }
</style>
