<template>
  <div>
    <v-switch
      v-if="field.field_type === 'bool'"
      :model-value="currentValue"
      :label="field.key"
      :disabled="field.read_only"
      color="primary"
      density="compact"
      hide-details="auto"
      @update:model-value="emit('update', $event)"
    >
      <template #append>
        <v-icon
          icon="mdi-information-outline"
          size="small"
          class="ml-1 help-icon"
          @click.stop="emit('help-click', field.key)"
        />
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
      :label="field.key"
      :items="selectItems"
      :disabled="field.read_only"
      variant="outlined"
      density="compact"
      hide-details="auto"
      @update:model-value="emit('update', $event)"
    >
      <template #append>
        <v-icon
          icon="mdi-information-outline"
          size="small"
          class="ml-1 help-icon"
          @click.stop="emit('help-click', field.key)"
        />
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
      :label="field.key"
      :disabled="field.read_only"
      type="number"
      variant="outlined"
      density="compact"
      hide-details="auto"
      @update:model-value="emit('update', Number($event))"
    >
      <template #append-inner>
        <v-icon
          icon="mdi-information-outline"
          size="small"
          class="help-icon"
          @click.stop="emit('help-click', field.key)"
        />
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
      :label="field.key"
      :disabled="field.read_only"
      variant="outlined"
      density="compact"
      hide-details="auto"
      :rules="[validateFloat]"
      @update:model-value="onFloatInput($event)"
      @blur="onFloatBlur"
    >
      <template #append-inner>
        <v-icon
          icon="mdi-information-outline"
          size="small"
          class="help-icon"
          @click.stop="emit('help-click', field.key)"
        />
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
      :label="field.key"
      :disabled="field.read_only"
      variant="outlined"
      density="compact"
      hide-details="auto"
      @update:model-value="tryParse($event)"
    >
      <template #append-inner>
        <v-icon
          icon="mdi-information-outline"
          size="small"
          class="help-icon"
          @click.stop="emit('help-click', field.key)"
        />
        <v-chip v-if="field.read_only" size="x-small" variant="outlined" color="grey" prepend-icon="mdi-lock">
          {{ t('cfReadOnly') }}
        </v-chip>
        <v-chip v-else-if="field.origin !== 'method'" size="x-small" variant="outlined">
          {{ field.origin }}
        </v-chip>
      </template>
    </v-text-field>

    <v-text-field
      v-else
      :model-value="currentValue"
      :label="field.key"
      :disabled="field.read_only"
      variant="outlined"
      density="compact"
      hide-details="auto"
      @update:model-value="emit('update', $event)"
    >
      <template #append-inner>
        <v-icon
          icon="mdi-information-outline"
          size="small"
          class="help-icon"
          @click.stop="emit('help-click', field.key)"
        />
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

const props = defineProps<{ field: FieldMeta }>()
const emit = defineEmits<{ update: [value: unknown]; 'help-click': [key: string] }>()
const configStore = useConfigStore()
const { t } = useI18n()

const currentValue = computed(() => configStore.getFieldValue(props.field.key))

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
    emit('update', parsed ?? currentValue.value)
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
  if (props.field.key === 'sample_sampler') return ['euler', 'er_sde', 'euler_a']
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
.help-icon {
  cursor: pointer;
  opacity: 0.5;
  transition: opacity 0.2s;
}
.help-icon:hover {
  opacity: 1;
  color: rgb(var(--v-theme-primary));
}
</style>
