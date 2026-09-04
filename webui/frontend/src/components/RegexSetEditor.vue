<template>
  <div class="regex-set-editor">
    <div class="d-flex align-center flex-wrap ga-1 mb-1">
      <span class="text-subtitle-2">{{ field.key }}</span>
      <v-chip v-if="field.read_only" size="x-small" variant="outlined" color="grey" prepend-icon="mdi-lock">
        {{ t('cfReadOnly') }}
      </v-chip>
      <v-chip v-else-if="field.origin !== 'method'" size="x-small" variant="outlined" class="ml-1">
        {{ field.origin }}
      </v-chip>
      <v-chip v-if="field.is_virtual" size="x-small" color="warning" variant="outlined">
        {{ t('cfVirtual') }}
      </v-chip>
      <v-spacer />
      <v-icon
        icon="mdi-information-outline"
        size="small"
        class="help-icon"
        :title="field.description || field.key"
        @click.stop="emit('help-click', field.key)"
      />
    </div>

    <div
      v-for="(row, i) in rows"
      :key="row.id"
      class="d-flex align-start ga-2 mb-2"
    >
      <v-text-field
        v-model="row.pattern"
        :label="t('cfRegexPattern')"
        :disabled="field.read_only"
        variant="outlined"
        density="compact"
        hide-details="auto"
        class="font-mono-field flex-grow-1"
        :error-messages="patternErrors(row)"
        @update:model-value="emitUpdate"
      />
      <v-text-field
        v-model="row.value"
        :label="t('cfRegexValue')"
        :disabled="field.read_only"
        variant="outlined"
        density="compact"
        hide-details="auto"
        class="font-mono-field regex-value-field"
        :error-messages="valueErrors(row)"
        @update:model-value="emitUpdate"
      />
      <v-btn
        icon="mdi-close"
        variant="text"
        density="compact"
        size="small"
        color="error"
        :disabled="field.read_only"
        :aria-label="t('cfRegexDelete')"
        :title="t('cfRegexDelete')"
        class="mt-1"
        @click="removeRow(i)"
      />
    </div>

    <div class="d-flex align-center ga-2">
      <v-btn
        prepend-icon="mdi-plus"
        variant="tonal"
        size="small"
        color="primary"
        :disabled="field.read_only"
        @click="addRow"
      >
        {{ t('cfRegexAdd') }}
      </v-btn>
      <span class="text-caption text-medium-emphasis">{{ t('cfRegexHint') }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { FieldMeta } from '../stores/config'
import { useConfigStore } from '../stores/config'
import { useI18n } from '../composables/useI18n'

interface RegexRow {
  id: number
  pattern: string
  value: string
}

const props = defineProps<{ field: FieldMeta }>()
const emit = defineEmits<{ update: [value: unknown]; 'help-click': [key: string] }>()
const configStore = useConfigStore()
const { t } = useI18n()

const currentValue = computed(() => String(configStore.getFieldValue(props.field.key) ?? ''))

// network_reg_dims takes plain ints (the trainer casts via int()); lrs take floats.
const isInt = computed(() => props.field.key === 'network_reg_dims')

// The wire format mirrors the trainer's `_parse_kv_pairs`: comma-separated
// `pattern=value` segments, value split on the FIRST '='. Ranks/LRs are
// numeric; patterns may contain '=' only after the first one (kept in value
// side of the split, so re-parse stays round-trip stable).
function parseRows(text: string): RegexRow[] {
  const out: RegexRow[] = []
  for (const segment of text.split(',')) {
    const trimmed = segment.trim()
    if (!trimmed) continue
    const eq = trimmed.indexOf('=')
    if (eq < 0) {
      out.push({ id: nextId(), pattern: trimmed, value: '' })
      continue
    }
    out.push({
      id: nextId(),
      pattern: trimmed.slice(0, eq).trim(),
      value: trimmed.slice(eq + 1).trim(),
    })
  }
  return out
}

function serialize(list: RegexRow[]): string {
  return list
    .map(r => `${r.pattern.trim()}=${r.value.trim()}`)
    .filter(s => s !== '=')
    .join(', ')
}

let nextIdCounter = 0
function nextId(): number {
  return ++nextIdCounter
}

const rows = ref<RegexRow[]>([])
let selfUpdate = false

watch(
  currentValue,
  v => {
    if (selfUpdate) {
      selfUpdate = false
      return
    }
    rows.value = parseRows(v)
  },
  { immediate: true },
)

function emitUpdate() {
  selfUpdate = true
  emit('update', serialize(rows.value))
}

function addRow() {
  rows.value.push({ id: nextId(), pattern: '', value: '' })
  emitUpdate()
}

function removeRow(index: number) {
  rows.value.splice(index, 1)
  emitUpdate()
}

// --- Per-row validation (mirrors backend `_validate_regex_set`) ---

const NUMBER_RE = /^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/
const INT_RE = /^\d+$/

function patternErrors(row: RegexRow): string {
  if (!row.pattern.trim()) return t('cfRegexPatternRequired')
  if (row.pattern.includes(',')) return t('cfRegexHint')
  try {
    new RegExp(row.pattern)
    return ''
  } catch {
    return t('cfRegexInvalid')
  }
}

function valueErrors(row: RegexRow): string {
  const v = row.value.trim()
  if (!v) return t('cfRegexInvalidValue')
  if (isInt.value) return INT_RE.test(v) ? '' : t('cfRegexInvalidValue')
  if (!NUMBER_RE.test(v)) return t('cfRegexInvalidValue')
  return Number(v) < 0 ? t('cfRegexInvalidValue') : ''
}
</script>

<style scoped>
.regex-value-field {
  flex: 0 0 140px;
  max-width: 140px;
}

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
