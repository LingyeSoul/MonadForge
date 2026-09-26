<template>
  <v-alert v-if="qwen.error" type="error" variant="tonal" class="mb-4" data-testid="qwen-workspace-error">{{ qwen.error }}</v-alert>
  <v-progress-linear v-if="qwen.loading" indeterminate />
  <template v-if="qwen.draft && qwen.schema">
    <div class="d-flex align-center flex-wrap ga-2 mb-4">
      <v-chip size="small" variant="tonal">Qwen-Image 2.1 · LoRA</v-chip>
      <span class="text-caption text-medium-emphasis">{{ t('qwProfileStored') }}</span>
      <v-spacer />
      <v-btn v-if="qwen.legacyAvailable" data-testid="qwen-import-legacy" variant="text" size="small" @click="qwen.perform(async () => qwen.importLegacy())">{{ t('qwImportLegacy') }}</v-btn>
      <v-btn data-testid="qwen-reset-defaults" variant="text" size="small" @click="qwen.resetDefaults()">{{ t('qwenReset') }}</v-btn>
      <v-btn to="/preprocess" variant="outlined" size="small">{{ t('navPreprocess') }}</v-btn>
    </div>
    <div class="d-flex align-center ga-2 mb-4">
      <v-text-field v-model="search" :label="t('cfgSearchFields')" data-testid="qwen-field-search" prepend-inner-icon="mdi-magnify" clearable hide-details density="compact" />
      <v-btn data-testid="qwen-help-toggle" icon="mdi-book-open-page-variant-outline" variant="text" :aria-label="t('cfgOpenGuide')" @click="showHelp = !showHelp" />
    </div>
    <v-row>
      <v-col cols="12" :lg="showHelp ? 8 : 12">
        <section v-for="group in groups" :key="group.section" class="config-section mb-5">
          <h2 class="text-subtitle-1 mb-3">{{ t(group.title) }}</h2>
          <QwenFields :section="group.section" :fields="group.basic" @help="openHelp" />
          <v-expansion-panels v-if="group.advanced.length" variant="accordion" class="mt-3">
            <v-expansion-panel :data-testid="`qwen-${group.section}-advanced`" :title="t('qwenAdvanced')">
              <v-expansion-panel-text><QwenFields :section="group.section" :fields="group.advanced" @help="openHelp" /></v-expansion-panel-text>
            </v-expansion-panel>
          </v-expansion-panels>
          <template v-if="group.section === 'models'">
            <v-btn data-testid="qwen-resolve" size="small" variant="text" class="mt-3" @click="inspectPaths">{{ t('qwenResolve') }}</v-btn>
            <dl v-if="paths" class="qwen-paths text-caption"><template v-for="(path, name) in paths" :key="name"><dt>{{ name }}</dt><dd>{{ path }}</dd></template></dl>
          </template>
        </section>
        <div v-if="groups.length === 0" class="workspace-empty">{{ t('cfgNoMatchingFields') }}</div>
      </v-col>
      <v-col v-if="showHelp" cols="12" lg="4">
        <HelpPanel ref="helpPanel" :variant="`qwen21/${qwen.profile}`" :field-help="help" :guide-html="guide" height="min(680px, calc(100dvh - 120px))" />
      </v-col>
    </v-row>
  </template>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from '../composables/useI18n'
import { useQwenWorkspace, type QwenSection } from '../stores/qwenWorkspace'
import { modelFields, requestJson, resolvedPaths, type QwenField } from '../utils/qwen21'
import HelpPanel from './HelpPanel.vue'
import QwenFields from './QwenFields.vue'

const qwen = useQwenWorkspace()
const { t } = useI18n()
const search = ref<string | null>('')
const showHelp = ref(true)
const helpPanel = ref<InstanceType<typeof HelpPanel> | null>(null)
const paths = ref<Record<string, string> | null>(null)
const definitions: { section: QwenSection; title: string }[] = [
  { section: 'models', title: 'qwenModels' }, { section: 'train', title: 'qwTrainingParameters' }, { section: 'generate', title: 'qwTestParameters' },
]
function sectionFields(section: QwenSection): QwenField[] {
  if (!qwen.schema) return []
  return section === 'models' ? qwen.schema.train.filter(f => modelFields.has(f.name)) : qwen.schema[section].filter(f => !modelFields.has(f.name))
}
const groups = computed(() => definitions.map(group => {
  const terms = (search.value ?? '').toLowerCase().split(/\s+/).filter(Boolean)
  const fields = sectionFields(group.section).filter(field => terms.every(term => `${field.name} ${t(`qwenField_${field.name}`)} ${field.help}`.toLowerCase().includes(term)))
  const advanced = (field: QwenField): boolean => group.section === 'models' ? field.name !== 'model_dir' : field.advanced
  return { ...group, basic: fields.filter(f => !advanced(f)), advanced: fields.filter(advanced) }
}).filter(group => group.basic.length + group.advanced.length > 0))
const help = computed(() => Object.fromEntries(definitions.flatMap(group => sectionFields(group.section).map(field => [`${group.section}.${field.name}`, `${t(`qwenField_${field.name}`)}\n${field.help}`]))))
const guide = computed(() => [t('qwenSourcesHint'), t('qwenAutoHint'), t('qwenGenerateHint'), t('qwUnsupported'), t('qwResultIsolation')].map(text => `<p>${text}</p>`).join(''))
function openHelp(key: string): void {
  showHelp.value = true
  helpPanel.value?.showFieldHelp(key, 'method')
}
async function inspectPaths(): Promise<void> {
  await qwen.perform(async () => {
    paths.value = resolvedPaths(await requestJson('/api/qwen21/resolve', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values: qwen.draft?.models }) }))
  })
}
</script>

<style scoped>
.qwen-paths { display: grid; grid-template-columns: auto 1fr; gap: 6px 12px; font-family: var(--font-mono); }
.qwen-paths dd { margin: 0; overflow-wrap: anywhere; }
@media (max-width: 600px) { .qwen-paths { grid-template-columns: 1fr; } }
</style>
