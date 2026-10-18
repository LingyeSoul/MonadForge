<template>
  <div class="curation-fields">
    <section v-for="[group, fields] in groups" :key="group" class="curation-field-section">
      <h3 v-if="group" class="curation-field-heading">{{ curationFieldGroup(app.language, group) }}</h3>
      <div class="curation-field-grid">
        <div v-for="field in fields" :key="field.dest" class="curation-field" :class="{ 'curation-field--wide': ['list', 'masks', 'multi'].includes(field.kind) }">
          <v-switch v-if="field.kind === 'bool'" :model-value="Boolean(value(field))" :label="copy(field).label" :aria-describedby="helpId(field)" hide-details density="comfortable" color="primary" :disabled="disabled || gated(field)" @update:model-value="set(field, $event)" />
          <v-select v-else-if="field.kind === 'enum' || field.kind === 'multi'" :model-value="value(field)" :items="curationFieldChoices(app.language, field)" :multiple="field.kind === 'multi'" :chips="field.kind === 'multi'" :label="copy(field).label" :aria-describedby="helpId(field)" hide-details density="comfortable" :disabled="disabled || gated(field)" @update:model-value="set(field, $event)" />
          <template v-else-if="field.kind === 'masks'">
            <div class="curation-field-label">{{ copy(field).label }}</div>
            <div v-for="(mask, index) in masks(field)" :key="index" class="curation-mask-row">
              <v-text-field :model-value="mask.path" :label="mask.kind === 'soft' ? tr('maskPromptFile') : tr('maskPromptText')" hide-details density="comfortable" class="curation-mask-source" :disabled="disabled || gated(field)" @update:model-value="setMask(field, index, 'path', $event)" />
              <v-select :model-value="mask.role" :items="maskRoles" :label="tr('maskRole')" hide-details density="comfortable" :disabled="disabled || gated(field)" @update:model-value="setMask(field, index, 'role', $event)" />
              <v-select :model-value="mask.kind" :items="maskKinds" :label="tr('maskKind')" hide-details density="comfortable" :disabled="disabled || gated(field)" @update:model-value="setMask(field, index, 'kind', $event)" />
              <v-btn icon="mdi-close" size="small" variant="text" :aria-label="tr('remove')" :disabled="disabled || gated(field)" @click="removeMask(field, index)" />
            </div>
            <v-btn prepend-icon="mdi-plus" size="small" variant="text" :disabled="disabled || gated(field)" @click="addMask(field)">{{ tr('add') }}</v-btn>
          </template>
          <v-textarea v-else-if="field.kind === 'list'" :model-value="stringValue(field)" :label="copy(field).label" :aria-describedby="helpId(field)" hide-details rows="3" auto-grow density="comfortable" :disabled="disabled || gated(field)" @update:model-value="set(field, $event)" />
          <v-text-field v-else :model-value="value(field)" :label="copy(field).label" :aria-describedby="helpId(field)" :type="['int', 'float'].includes(field.kind) ? 'number' : 'text'" :step="field.kind === 'int' ? 1 : 'any'" hide-details :required="field.required" density="comfortable" :disabled="disabled || gated(field)" @update:model-value="set(field, $event)" />
          <p v-if="copy(field).help" :id="helpId(field)" class="curation-field-help">{{ copy(field).help }}</p>
        </div>
      </div>
    </section>
    <v-switch v-if="hasAdvanced" v-model="advanced" :label="tr('advanced')" class="curation-advanced" density="comfortable" hide-details color="primary" />
    <details v-if="visible.length" class="curation-technical">
      <summary>{{ tr('parameterReference') }}</summary>
      <p class="curation-field-help">{{ tr('parameterReferenceHint') }}</p>
      <dl>
        <template v-for="field in visible" :key="field.dest">
          <dt>{{ copy(field).label }} <code>{{ parameterName(field) }}</code></dt>
          <dd>{{ field.help || '—' }}</dd>
        </template>
      </dl>
    </details>
  </div>
</template>
<script setup lang="ts">
import { computed, ref, useId } from 'vue'
import { useAppStore } from '../../stores/app'
import { curationText } from '../../i18n/curation'
import { curationFieldChoices, curationFieldCopy, curationFieldGroup } from '../../i18n/curationSchema'
import { curationFieldValue, visibleCurationFields, parseMaskSource, spellMaskSource } from '../../composables/useCuration'
import type { CurationField, CurationValues } from '../../types/curation'
const props = defineProps<{ fields: CurationField[]; modelValue: CurationValues; disabled?: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [value: CurationValues] }>()
const app = useAppStore()
const tr = (key: Parameters<typeof curationText>[1]) => curationText(app.language, key)
const copy = (field: CurationField) => curationFieldCopy(app.language, field)
const id = useId()
const helpId = (field: CurationField) => `${id}-${field.dest}-help`
const parameterName = (field: CurationField) => (field as CurationField & { flags?: string[] }).flags?.join(' / ') || field.label || field.dest
const maskRoles = computed(() => [{ title: tr('maskKeep'), value: 'keep' }, { title: tr('maskIgnore'), value: 'ignore' }])
const maskKinds = computed(() => [{ title: tr('maskText'), value: 'text' }, { title: tr('maskSoft'), value: 'soft' }])
const advanced = ref(false)
const visible = computed(() => visibleCurationFields(props.fields))
const hasAdvanced = computed(() => visible.value.some(field => field.advanced))
const groups = computed(() => {
  const groups = new Map<string, CurationField[]>()
  for (const field of visible.value.filter(field => advanced.value || !field.advanced)) {
    const key = field.group || ''
    groups.set(key, [...(groups.get(key) || []), field])
  }
  return Array.from(groups.entries())
})
function value(field: CurationField): unknown { return props.modelValue[field.dest] ?? field.default }
function stringValue(field: CurationField): string { const current = value(field); return Array.isArray(current) ? current.join('\n') : String(current ?? '') }
function gated(field: CurationField): boolean { return Boolean(field.gate && field.gate !== field.dest && !(props.modelValue[field.gate] ?? props.fields.find(other => other.dest === field.gate)?.default)) }
function set(field: CurationField, value: unknown): void { emit('update:modelValue', { ...props.modelValue, [field.dest]: curationFieldValue(field, value) }) }
type Mask = { path: string; role: string; kind: string }
function masks(field: CurationField): Mask[] {
  const current = value(field)
  return Array.isArray(current) ? current.map(item => parseMaskSource(String(item))) : []
}
function setMask(field: CurationField, index: number, key: keyof Mask, value: unknown): void { const rows = masks(field); rows[index] = { ...rows[index]!, [key]: String(value || '') }; set(field, rows.map(spellMaskSource)) }
function removeMask(field: CurationField, index: number): void { set(field, masks(field).filter((_, row) => row !== index).map(spellMaskSource)) }
function addMask(field: CurationField): void { set(field, [...masks(field), { path: '', role: 'ignore', kind: 'text' }].map(spellMaskSource)) }
</script>
<style scoped>
.curation-fields { min-width: 0; }
.curation-field-section + .curation-field-section { margin-top: 28px; }
.curation-field-heading { font-size: 0.94rem; font-weight: 600; margin: 0 0 20px; }
.curation-field-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; }
.curation-field { min-width: 0; }
.curation-field--wide { grid-column: 1 / -1; }
.curation-field-label { font-size: 0.875rem; margin-bottom: 14px; }
.curation-field-help { margin: 8px 0 0; font-size: 0.8125rem; line-height: 1.6; color: rgba(var(--v-theme-on-surface), 0.75); overflow-wrap: anywhere; max-width: 84ch; }
.curation-field :deep(.v-label) { opacity: 1; }
.curation-field :deep(.v-switch .v-label) { white-space: normal; }
.curation-field :deep(.v-field) { min-width: 0; }
.curation-mask-row { display: grid; grid-template-columns: minmax(180px, 1fr) 140px 140px auto; gap: 12px; align-items: center; margin-bottom: 16px; }
.curation-advanced { margin-top: 24px; }
.curation-technical { margin-top: 20px; border-top: 1px solid rgba(var(--v-theme-on-surface), 0.14); padding-top: 16px; font-size: 0.8125rem; }
.curation-technical summary { cursor: pointer; color: rgba(var(--v-theme-on-surface), 0.78); }
.curation-technical dl { margin-top: 20px; }
.curation-technical dt { font-weight: 600; margin-top: 16px; }
.curation-technical code { margin-left: 8px; font-weight: 400; overflow-wrap: anywhere; }
.curation-technical dd { margin: 6px 0 0; line-height: 1.65; color: rgba(var(--v-theme-on-surface), 0.75); overflow-wrap: anywhere; white-space: pre-wrap; }
@media (max-width: 959px) {
  .curation-field-grid { grid-template-columns: minmax(0, 1fr); gap: 22px; }
  .curation-mask-row { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto; }
  .curation-mask-source { grid-column: 1 / -1; }
}
</style>
