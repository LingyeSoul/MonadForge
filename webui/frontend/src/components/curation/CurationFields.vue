<template>
  <div>
    <div v-for="[group, fields] in groups" :key="group" class="mb-4">
      <div v-if="group" class="text-subtitle-2 mb-2">{{ group }}</div>
      <v-row dense>
        <v-col v-for="field in fields" :key="field.dest" cols="12" :md="['list', 'masks', 'multi'].includes(field.kind) ? 12 : 6">
          <v-switch v-if="field.kind === 'bool'" :model-value="Boolean(value(field))" :label="field.label || field.dest" :hint="field.help" persistent-hint density="compact" color="primary" :disabled="disabled || gated(field)" @update:model-value="set(field, $event)" />
          <v-select v-else-if="field.kind === 'enum' || field.kind === 'multi'" :model-value="value(field)" :items="field.choices || []" :multiple="field.kind === 'multi'" :chips="field.kind === 'multi'" :label="field.label || field.dest" :hint="field.help" persistent-hint density="compact" :disabled="disabled || gated(field)" @update:model-value="set(field, $event)" />
          <template v-else-if="field.kind === 'masks'">
            <div class="text-body-2 mb-2">{{ field.label || field.dest }}</div>
            <div v-for="(mask, index) in masks(field)" :key="index" class="d-flex ga-2 align-start mb-1 flex-wrap">
              <v-text-field :model-value="mask.path" label="Path" density="compact" style="min-width: 160px; flex: 1" :disabled="disabled" @update:model-value="setMask(field, index, 'path', $event)" />
              <v-select :model-value="mask.role" :items="['keep', 'ignore']" label="Role" density="compact" style="max-width: 120px" :disabled="disabled" @update:model-value="setMask(field, index, 'role', $event)" />
              <v-select :model-value="mask.kind" :items="['text', 'soft']" label="Kind" density="compact" style="max-width: 120px" :disabled="disabled" @update:model-value="setMask(field, index, 'kind', $event)" />
              <v-btn icon="mdi-close" size="small" variant="text" :aria-label="tr('remove')" :disabled="disabled" @click="removeMask(field, index)" />
            </div>
            <v-btn prepend-icon="mdi-plus" size="small" variant="text" :disabled="disabled" @click="addMask(field)">{{ tr('add') }}</v-btn>
            <div class="text-caption text-medium-emphasis">{{ field.help }}</div>
          </template>
          <v-textarea v-else-if="field.kind === 'list'" :model-value="stringValue(field)" :label="field.label || field.dest" :hint="field.help" persistent-hint rows="3" density="compact" :disabled="disabled || gated(field)" @update:model-value="set(field, $event)" />
          <v-text-field v-else :model-value="value(field)" :label="field.label || field.dest" :type="['int', 'float'].includes(field.kind) ? 'number' : 'text'" :step="field.kind === 'int' ? 1 : 'any'" :hint="field.help" persistent-hint :required="field.required" density="compact" :disabled="disabled || gated(field)" @update:model-value="set(field, $event)" />
        </v-col>
      </v-row>
    </div>
    <v-switch v-if="hasAdvanced" v-model="advanced" :label="tr('advanced')" density="compact" hide-details />
  </div>
</template>
<script setup lang="ts">
import { computed, ref } from 'vue'
import { useAppStore } from '../../stores/app'
import { curationText } from '../../i18n/curation'
import { curationFieldValue, visibleCurationFields, parseMaskSource, spellMaskSource } from '../../composables/useCuration'
import type { CurationField, CurationValues } from '../../types/curation'
const props = defineProps<{ fields: CurationField[]; modelValue: CurationValues; disabled?: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [value: CurationValues] }>()
const app = useAppStore()
const tr = (key: Parameters<typeof curationText>[1]) => curationText(app.language, key)
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
