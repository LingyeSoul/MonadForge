<template>
  <v-card class="help-panel" variant="outlined" :height="height">
    <v-card-title class="d-flex align-center text-subtitle-1 pa-3 pb-1">
      <v-icon :icon="showField ? 'mdi-information-outline' : 'mdi-book-open-page-variant-outline'" class="mr-2" size="small" />
      {{ showField ? t('cfgFieldHelp') : t('cfgMethodGuide') }}
      <v-spacer />
      <v-btn
        v-if="showField"
        icon="mdi-arrow-left"
        size="x-small"
        variant="text"
        :title="t('cfgMethodGuide')"
        @click="backToGuide"
      />
    </v-card-title>

    <v-divider />

    <v-card-text class="help-content pa-3">
      <!-- Field help view -->
      <template v-if="showField">
        <div class="text-subtitle-2 mb-2" style="color: rgb(var(--v-theme-primary))">
          {{ activeFieldKey }}
        </div>

        <div v-if="fieldHelpText" class="field-help-text mb-3">
          {{ fieldHelpText }}
        </div>
        <div v-else class="text-medium-emphasis text-body-2 mb-3 font-italic">
          {{ t('cfgNoHelp') }}
        </div>

        <v-chip
          v-if="fieldOrigin"
          size="small"
          :color="originColor(fieldOrigin)"
          variant="tonal"
          class="mb-2"
        >
          <v-icon start icon="mdi-source-branch" />
          {{ t('cfgOriginFrom', { layer: fieldOrigin }) }}
        </v-chip>
      </template>

      <!-- Method guide view -->
      <template v-else>
        <div v-if="guideHtml" class="guide-html" v-html="guideHtml" />
        <div v-else class="text-medium-emphasis text-body-2 font-italic pa-4 text-center">
          {{ t('cfgSelectHint') }}
        </div>
      </template>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { useI18n } from '../composables/useI18n'

const props = defineProps<{
  variant?: string
  fieldHelp: Record<string, string>
  guideHtml: string
  height?: string | number
}>()

const { t } = useI18n()

const activeFieldKey = ref('')
const activeFieldOrigin = ref('')
const showField = ref(false)

const fieldHelpText = computed(() => {
  if (!activeFieldKey.value) return ''
  return props.fieldHelp[activeFieldKey.value] || ''
})

const fieldOrigin = computed(() => activeFieldOrigin.value)

function showFieldHelp(key: string, origin: string) {
  activeFieldKey.value = key
  activeFieldOrigin.value = origin
  showField.value = true
}

function backToGuide() {
  showField.value = false
  activeFieldKey.value = ''
  activeFieldOrigin.value = ''
}

function originColor(origin: string): string {
  if (origin === 'method') return 'success'
  if (origin === 'preset') return 'info'
  return 'grey'
}

// Reset to guide view when variant changes
watch(() => props.variant, () => {
  showField.value = false
  activeFieldKey.value = ''
  activeFieldOrigin.value = ''
})

defineExpose({ showFieldHelp })
</script>

<style scoped>
.help-panel {
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.help-content {
  overflow-y: auto;
  flex: 1;
}

.field-help-text {
  font-size: 0.875rem;
  line-height: 1.6;
  white-space: pre-wrap;
}

/* Style the method guide HTML to match the dark theme */
.guide-html :deep(h1),
.guide-html :deep(h2),
.guide-html :deep(h3) {
  color: rgb(var(--v-theme-on-surface));
  margin-top: 0.75em;
  margin-bottom: 0.5em;
}

.guide-html :deep(p) {
  margin-bottom: 0.5em;
  line-height: 1.6;
}

.guide-html :deep(code) {
  background: rgba(var(--v-theme-on-surface), 0.08);
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 0.85em;
}

.guide-html :deep(pre) {
  background: rgba(var(--v-theme-on-surface), 0.05);
  padding: 8px 12px;
  border-radius: 4px;
  overflow-x: auto;
  font-size: 0.85em;
}

.guide-html :deep(a) {
  color: rgb(var(--v-theme-primary));
  text-decoration: none;
}

.guide-html :deep(a:hover) {
  text-decoration: underline;
}

.guide-html :deep(ul),
.guide-html :deep(ol) {
  padding-left: 1.5em;
  margin-bottom: 0.5em;
}

.guide-html :deep(li) {
  margin-bottom: 0.25em;
  line-height: 1.5;
}

.guide-html :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 0.5em 0;
}

.guide-html :deep(th),
.guide-html :deep(td) {
  border: 1px solid rgba(var(--v-border-color), 0.3);
  padding: 4px 8px;
  text-align: left;
  font-size: 0.85em;
}

.guide-html :deep(th) {
  background: rgba(var(--v-theme-on-surface), 0.05);
}

.guide-html :deep(blockquote) {
  border-left: 3px solid rgb(var(--v-theme-primary));
  padding-left: 12px;
  margin: 0.5em 0;
  color: rgba(var(--v-theme-on-surface), 0.7);
}

.guide-html :deep(.callout),
.guide-html :deep(.warning) {
  background: rgba(var(--v-theme-warning), 0.1);
  border-radius: 4px;
  padding: 8px 12px;
  margin: 0.5em 0;
}

/* Brand: active topic indicator */
.help-panel {
  border-left: 3px solid var(--forge-ember);
}

/* Code blocks: bg-deep terminal style */
.guide-html :deep(pre) {
  background: var(--bg-deep);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  font-family: var(--font-mono);
  font-size: 12px;
}

.guide-html :deep(code) {
  background: var(--bg-deep);
  color: var(--forge-amber);
  padding: 1px 5px;
  border-radius: 3px;
  font-family: var(--font-mono);
  font-size: 0.85em;
}
</style>
