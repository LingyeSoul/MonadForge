<template>
  <div :data-testid="`config-field-${field.key}`">
    <ModelPathField v-if="field.field_type === 'path'" :field="field" :model-value="String(currentValue ?? '')" @update="emit('update', $event)" @help-click="emit('help-click', field.key)" />
    <v-textarea v-else-if="field.multiline" :model-value="String(currentValue ?? '')" :label="field.label ?? field.key" variant="outlined" density="compact" auto-grow rows="3" hide-details="auto" @update:model-value="emit('update', field.nullable && $event === '' ? null : $event)">
      <template #append><FieldHelpButton :field-key="field.key" @help="emit('help-click', field.key)" /></template>
    </v-textarea>
    <v-switch
      v-else-if="field.field_type === 'bool'"
      :model-value="currentValue"
      :placeholder="field.nullable ? t('qwenAuto') : undefined"
      :label="field.label ?? field.key"
      :disabled="field.read_only"
      color="primary"
      density="compact"
      hide-details="auto"
      @update:model-value="emit('update', field.nullable && $event === '' ? null : $event)"
    >
      <template #append>
        <FieldHelpButton :field-key="field.key" @help="emit('help-click', field.key)" />
        <v-chip v-if="field.read_only" size="x-small" variant="outlined" color="grey" class="ml-2" prepend-icon="mdi-lock">
          {{ t('cfReadOnly') }}
        </v-chip>
        <template v-else>
          <v-chip v-if="field.origin !== 'method'" size="x-small" variant="outlined" class="ml-2">
            {{ field.origin }}
          </v-chip>
          <v-chip v-if="field.is_virtual" size="x-small" color="warning" variant="outlined" class="ml-1">
            {{ t('cfVirtual') }}
          </v-chip>
        </template>
      </template>
    </v-switch>

    <v-select
      v-else-if="field.field_type === 'select'"
      :model-value="String(currentValue ?? '')"
      :label="field.label ?? field.key"
      :items="selectItems"
      :disabled="field.read_only"
      variant="outlined"
      density="compact"
      hide-details="auto"
      @update:model-value="emit('update', field.nullable && $event === '' ? null : $event)"
    >
      <template #append>
        <FieldHelpButton :field-key="field.key" @help="emit('help-click', field.key)" />
        <v-chip v-if="field.read_only" size="x-small" variant="outlined" color="grey" class="ml-2" prepend-icon="mdi-lock">
          {{ t('cfReadOnly') }}
        </v-chip>
        <v-chip v-else-if="field.origin !== 'method'" size="x-small" variant="outlined" class="ml-2">
          {{ field.origin }}
        </v-chip>
      </template>
    </v-select>

    <v-text-field
      v-else-if="field.field_type === 'int'"
      :model-value="currentValue"
      :placeholder="field.nullable ? t('qwenAuto') : undefined"
      :label="field.label ?? field.key"
      :disabled="field.read_only"
      type="number"
      variant="outlined"
      density="compact"
      hide-details="auto"
      class="font-mono-field"
      @update:model-value="emit('update', field.nullable && ($event === '' || $event === null) ? null : Number($event))"
    >
      <template #append-inner>
        <FieldHelpButton :field-key="field.key" @help="emit('help-click', field.key)" />
        <v-chip v-if="field.read_only" size="x-small" variant="outlined" color="grey" prepend-icon="mdi-lock">
          {{ t('cfReadOnly') }}
        </v-chip>
        <v-chip v-else-if="field.origin !== 'method'" size="x-small" variant="outlined">
          {{ field.origin }}
        </v-chip>
      </template>
    </v-text-field>

    <v-text-field
      v-else-if="field.field_type === 'float'"
      :model-value="floatDisplay"
      :label="field.label ?? field.key"
      :disabled="field.read_only"
      variant="outlined"
      density="compact"
      hide-details="auto"
      class="font-mono-field"
      :rules="[validateFloat]"
      @update:model-value="onFloatInput($event)"
      @blur="onFloatBlur"
    >
      <template #append-inner>
        <FieldHelpButton :field-key="field.key" @help="emit('help-click', field.key)" />
        <v-chip v-if="field.read_only" size="x-small" variant="outlined" color="grey" prepend-icon="mdi-lock">
          {{ t('cfReadOnly') }}
        </v-chip>
        <v-chip v-else-if="field.origin !== 'method'" size="x-small" variant="outlined">
          {{ field.origin }}
        </v-chip>
      </template>
    </v-text-field>

    <v-text-field
      v-else-if="field.field_type === 'list'"
      :model-value="JSON.stringify(currentValue)"
      :label="field.label ?? field.key"
      :disabled="field.read_only"
      variant="outlined"
      density="compact"
      hide-details="auto"
      @update:model-value="tryParse($event)"
    >
      <template #append-inner>
        <FieldHelpButton :field-key="field.key" @help="emit('help-click', field.key)" />
        <v-chip v-if="field.read_only" size="x-small" variant="outlined" color="grey" prepend-icon="mdi-lock">
          {{ t('cfReadOnly') }}
        </v-chip>
        <v-chip v-else-if="field.origin !== 'method'" size="x-small" variant="outlined">
          {{ field.origin }}
        </v-chip>
      </template>
    </v-text-field>

    <RegexSetEditor
      v-else-if="field.field_type === 'regex_set'"
      :field="field"
      @update="emit('update', $event)"
      @help-click="emit('help-click', field.key)"
    />

    <v-text-field
      v-else
      :model-value="currentValue"
      :placeholder="field.nullable ? t('qwenAuto') : undefined"
      :label="field.label ?? field.key"
      :disabled="field.read_only"
      variant="outlined"
      density="compact"
      hide-details="auto"
      @update:model-value="emit('update', field.nullable && $event === '' ? null : $event)"
    >
      <template #append-inner>
        <FieldHelpButton :field-key="field.key" @help="emit('help-click', field.key)" />
        <v-chip v-if="field.read_only" size="x-small" variant="outlined" color="grey" prepend-icon="mdi-lock">
          {{ t('cfReadOnly') }}
        </v-chip>
        <v-chip v-else-if="field.origin !== 'method'" size="x-small" variant="outlined">
          {{ field.origin }}
        </v-chip>
      </template>
    </v-text-field>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { FieldMeta } from '../stores/config'
import { useConfigStore } from '../stores/config'
import { useI18n } from '../composables/useI18n'
import RegexSetEditor from './RegexSetEditor.vue'
import ModelPathField from './ModelPathField.vue'
import type { Scalar } from '../utils/qwen21'
import FieldHelpButton from './FieldHelpButton.vue'

const props = defineProps<{ field: FieldMeta; modelValue?: Scalar }>()
const emit = defineEmits<{ update: [value: unknown]; 'help-click': [key: string] }>()
const configStore = useConfigStore()
const { t } = useI18n()

const currentValue = computed(() => props.modelValue !== undefined ? props.modelValue : configStore.getFieldValue(props.field.key))

// --- Float field with scientific notation support ---

function formatFloat(v: unknown): string {
  if (v === null || v === undefined || v === '') return ''
  const n = Number(v)
  if (Number.isNaN(n)) return String(v)
  // Show scientific notation for very small/large numbers
  if ((n !== 0 && Math.abs(n) < 0.001) || Math.abs(n) >= 1e7) return n.toExponential()
  return String(n)
}

function parseFloatInput(raw: string): number | null {
  if (raw.trim() === '') return null
  const n = Number(raw)
  return Number.isNaN(n) ? null : n
}

// Local editing buffer so typing "1e-" mid-stream doesn't emit invalid values
const floatRaw = ref<string | null>(null) // null = not editing
const floatDisplay = computed(() =>
  floatRaw.value !== null ? floatRaw.value : formatFloat(currentValue.value),
)

function onFloatInput(val: string) {
  floatRaw.value = val
  const parsed = parseFloatInput(val)
  if (parsed !== null) emit('update', parsed)
}

function onFloatBlur() {
  if (floatRaw.value !== null) {
    const parsed = parseFloatInput(floatRaw.value)
    emit('update', props.field.nullable && floatRaw.value.trim() === '' ? null : parsed ?? currentValue.value)
    floatRaw.value = null // snap back to formatted display
  }
}

function validateFloat(val: string): true | string {
  if (val.trim() === '') return true
  return parseFloatInput(val) !== null ? true : t('cfInvalidNumber')
}

// Reset editing buffer when the underlying value changes externally
watch(currentValue, () => { if (floatRaw.value !== null) floatRaw.value = null })

// --- End float field ---

const selectItems = computed(() => {
  if (props.field.key === 'sample_sampler') return ['euler', 'er_sde']
  return props.field.options ?? []
})

function tryParse(val: string) {
  try {
    emit('update', JSON.parse(val))
  } catch {
    emit('update', val)
  }
}
</script>

<style scoped>
:deep(.v-switch .v-label) {
  white-space: normal;
  overflow-wrap: anywhere;
}

/* Focus glow on all fields in this component */
:deep(.v-field--focused) {
  box-shadow: 0 0 0 3px var(--glow-ember);
  border-radius: var(--radius-md);
}

:deep(.v-field--focused .v-field__outline) {
  border-color: var(--forge-ember) !important;
}

/* Switch ember gradient track */
:deep(.v-switch__track) {
  transition: background 0.2s ease;
}

/* Select dropdown: selected item amber left border */
:deep(.v-list-item--active) {
  border-left: 3px solid var(--forge-amber);
  padding-left: 13px;
}
</style>
