<template>
  <div>
    <v-switch
      v-if="field.field_type === 'bool'"
      :model-value="currentValue"
      :label="field.key"
      :hint="hintText"
      persistent-hint
      color="primary"
      density="compact"
      hide-details="auto"
      @update:model-value="emit('update', $event)"
    >
      <template #append>
        <v-chip v-if="field.origin !== 'method'" size="x-small" variant="outlined" class="ml-2">
          {{ field.origin }}
        </v-chip>
        <v-chip v-if="field.is_virtual" size="x-small" color="warning" variant="outlined" class="ml-1">
          {{ t('cfVirtual') }}
        </v-chip>
      </template>
    </v-switch>

    <v-select
      v-else-if="field.field_type === 'select'"
      :model-value="String(currentValue ?? '')"
      :label="field.key"
      :items="selectItems"
      :hint="hintText"
      persistent-hint
      variant="outlined"
      density="compact"
      hide-details="auto"
      @update:model-value="emit('update', $event)"
    >
      <template #append>
        <v-chip v-if="field.origin !== 'method'" size="x-small" variant="outlined" class="ml-2">
          {{ field.origin }}
        </v-chip>
      </template>
    </v-select>

    <v-text-field
      v-else-if="field.field_type === 'int'"
      :model-value="currentValue"
      :label="field.key"
      :hint="hintText"
      persistent-hint
      type="number"
      variant="outlined"
      density="compact"
      hide-details="auto"
      @update:model-value="emit('update', Number($event))"
    >
      <template #append-inner>
        <v-chip v-if="field.origin !== 'method'" size="x-small" variant="outlined">
          {{ field.origin }}
        </v-chip>
      </template>
    </v-text-field>

    <v-text-field
      v-else-if="field.field_type === 'float'"
      :model-value="currentValue"
      :label="field.key"
      :hint="hintText"
      persistent-hint
      type="number"
      step="any"
      variant="outlined"
      density="compact"
      hide-details="auto"
      @update:model-value="emit('update', Number($event))"
    >
      <template #append-inner>
        <v-chip v-if="field.origin !== 'method'" size="x-small" variant="outlined">
          {{ field.origin }}
        </v-chip>
      </template>
    </v-text-field>

    <v-text-field
      v-else-if="field.field_type === 'list'"
      :model-value="JSON.stringify(currentValue)"
      :label="field.key"
      :hint="hintText"
      persistent-hint
      variant="outlined"
      density="compact"
      hide-details="auto"
      @update:model-value="tryParse($event)"
    >
      <template #append-inner>
        <v-chip v-if="field.origin !== 'method'" size="x-small" variant="outlined">
          {{ field.origin }}
        </v-chip>
      </template>
    </v-text-field>

    <v-text-field
      v-else
      :model-value="currentValue"
      :label="field.key"
      :hint="hintText"
      persistent-hint
      variant="outlined"
      density="compact"
      hide-details="auto"
      @update:model-value="emit('update', $event)"
    >
      <template #append-inner>
        <v-chip v-if="field.origin !== 'method'" size="x-small" variant="outlined">
          {{ field.origin }}
        </v-chip>
      </template>
    </v-text-field>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { FieldMeta } from '../stores/config'
import { useConfigStore } from '../stores/config'
import { useI18n } from '../composables/useI18n'

const props = defineProps<{ field: FieldMeta }>()
const emit = defineEmits<{ update: [value: unknown] }>()
const configStore = useConfigStore()
const { t } = useI18n()

const currentValue = computed(() => configStore.getFieldValue(props.field.key))

const hintText = computed(() => {
  const desc = props.field.description
  const descEn = props.field.description_en
  if (!desc && !descEn) return undefined
  return desc || descEn
})

const selectItems = computed(() => {
  if (props.field.key === 'attn_mode') return ['sdpa', 'xformers', 'flash_attention', 'torch']
  return []
})

function tryParse(val: string) {
  try {
    emit('update', JSON.parse(val))
  } catch {
    emit('update', val)
  }
}
</script>
