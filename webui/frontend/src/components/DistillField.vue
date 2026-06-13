<template>
  <div class="distill-field">
    <template v-if="field.type === 'bool'">
      <v-switch
        v-model="field.value"
        :label="field.key"
        :hint="field.comment"
        color="primary"
        density="compact"
        hide-details="auto"
      />
    </template>

    <template v-else-if="field.type === 'int'">
      <v-text-field
        v-model.number="field.value"
        :label="field.key"
        :hint="field.comment"
        type="number"
        step="1"
        variant="outlined"
        density="compact"
        hide-details="auto"
        class="font-mono-field"
      />
    </template>

    <template v-else-if="field.type === 'float'">
      <v-text-field
        :model-value="floatDisplay"
        :label="field.key"
        :hint="field.comment"
        variant="outlined"
        density="compact"
        hide-details="auto"
        class="font-mono-field"
        :rules="[validateFloat]"
        @update:model-value="onFloatInput"
        @blur="onFloatBlur"
      />
    </template>

    <template v-else-if="field.type === 'list'">
      <v-textarea
        :model-value="listDisplay"
        :label="field.key"
        :hint="listHint"
        variant="outlined"
        density="compact"
        rows="2"
        auto-grow
        max-rows="6"
        hide-details="auto"
        class="font-mono-field"
        @update:model-value="onListInput"
      />
    </template>

    <template v-else>
      <v-text-field
        v-model="field.value"
        :label="field.key"
        :hint="field.comment"
        variant="outlined"
        density="compact"
        hide-details="auto"
        class="font-mono-field"
      />
    </template>

    <div v-if="field.comment" class="text-caption text-medium-emphasis mt-1 ml-1">
      {{ field.comment }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useI18n } from '../composables/useI18n'
import type { DistillField } from '../views/DistillView.vue'

const props = defineProps<{ field: DistillField }>()
const { t } = useI18n()

// --- Float field with scientific notation support (mirrors ConfigField) ---

function formatFloat(v: unknown): string {
  if (v === null || v === undefined || v === '') return ''
  const n = Number(v)
  if (Number.isNaN(n)) return String(v)
  if ((n !== 0 && Math.abs(n) < 0.001) || Math.abs(n) >= 1e7) return n.toExponential()
  return String(n)
}

function parseFloatInput(raw: string): number | null {
  if (raw.trim() === '') return null
  const n = Number(raw)
  return Number.isNaN(n) ? null : n
}

const floatRaw = ref<string | null>(null)
const floatDisplay = computed(() =>
  floatRaw.value !== null ? floatRaw.value : formatFloat(props.field.value)
)

function onFloatInput(val: string) {
  floatRaw.value = val
  const parsed = parseFloatInput(val)
  if (parsed !== null) props.field.value = parsed
}

function onFloatBlur() {
  if (floatRaw.value !== null) {
    const parsed = parseFloatInput(floatRaw.value)
    props.field.value = parsed ?? props.field.value
    floatRaw.value = null
  }
}

function validateFloat(val: string): true | string {
  if (val.trim() === '') return true
  return parseFloatInput(val) !== null ? true : t('cfInvalidNumber')
}

watch(() => props.field.value, () => { if (floatRaw.value !== null) floatRaw.value = null })

// --- List field: JSON array or one-per-line strings ---

const listDisplay = computed(() => {
  const v = props.field.value
  if (Array.isArray(v)) {
    if (v.every(x => typeof x === 'number')) return JSON.stringify(v)
    return v.map(String).join('\n')
  }
  return String(v ?? '')
})

const listHint = computed(() => t('distListHint'))

function onListInput(raw: string) {
  const trimmed = raw.trim()
  if (!trimmed) {
    props.field.value = []
    return
  }
  try {
    const parsed = JSON.parse(trimmed)
    if (Array.isArray(parsed)) {
      props.field.value = parsed
      return
    }
  } catch { /* fall through to line split */ }
  props.field.value = trimmed.split('\n').map(s => s.trim()).filter(Boolean)
}
</script>

<style scoped>
.distill-field {
  width: 100%;
}

.font-mono-field :deep(input),
.font-mono-field :deep(textarea) {
  font-family: var(--font-mono), monospace;
  font-size: 0.9rem;
}

/* Focus glow */
:deep(.v-field--focused) {
  box-shadow: 0 0 0 2px rgba(199, 91, 26, 0.15);
  border-radius: var(--radius-md);
}

:deep(.v-field--focused .v-field__outline) {
  border-color: var(--forge-ember) !important;
}

/* Switch ember track */
:deep(.v-switch__track) {
  transition: background 0.2s ease;
}
</style>
