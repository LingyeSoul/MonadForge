<template>
  <v-card variant="outlined" class="mb-2">
    <v-card-title class="text-subtitle-2 d-flex align-center">
      <v-icon icon="mdi-text-box-outline" size="small" class="mr-2" />
      {{ t('ppeTitle') }}
      <v-spacer />
      <v-btn
        size="x-small"
        variant="text"
        icon="mdi-plus"
        :title="t('ppeAdd')"
        @click="addEntry"
      />
    </v-card-title>
    <v-card-text>
      <v-progress-linear v-if="loading" indeterminate class="mb-2" />

      <div v-if="entries.length === 0 && !loading" class="text-medium-emphasis text-body-2 py-2">
        {{ t('ppeEmpty') }}
      </div>

      <v-card
        v-for="(entry, i) in entries"
        :key="i"
        variant="tonal"
        density="compact"
        class="mb-3"
      >
        <v-card-text class="pa-3">
          <div class="d-flex align-center mb-2">
            <span class="text-caption text-medium-emphasis">#{{ i + 1 }}</span>
            <v-spacer />
            <v-btn
              size="x-small"
              variant="text"
              icon="mdi-delete-outline"
              color="error"
              :title="t('ppeRemove')"
              @click="removeEntry(i)"
            />
          </div>
          <v-textarea
            :model-value="entry.prompt"
            :label="t('ppePositive')"
            rows="2"
            auto-grow
            density="compact"
            variant="outlined"
            hide-details
            class="mb-2"
            @update:model-value="updateEntry(i, 'prompt', $event)"
          />
          <v-textarea
            :model-value="entry.negative_prompt"
            :label="t('ppeNegative')"
            rows="2"
            auto-grow
            density="compact"
            variant="outlined"
            hide-details
            @update:model-value="updateEntry(i, 'negative_prompt', $event)"
          />
        </v-card-text>
      </v-card>

      <div class="d-flex ga-2 mt-2">
        <v-btn
          size="small"
          variant="tonal"
          prepend-icon="mdi-plus"
          @click="addEntry"
        >
          {{ t('ppeAdd') }}
        </v-btn>
        <v-spacer />
        <v-btn
          size="small"
          color="primary"
          :loading="saving"
          :disabled="!dirty"
          prepend-icon="mdi-content-save"
          @click="save"
        >
          {{ t('ppeSave') }}
        </v-btn>
      </div>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { useNotifyStore } from '../stores/notify'
import { useI18n } from '../composables/useI18n'

interface PromptEntry {
  prompt: string
  negative_prompt: string
}

const props = defineProps<{ promptPath: string }>()
const notify = useNotifyStore()
const { t } = useI18n()

const entries = ref<PromptEntry[]>([])
const loading = ref(false)
const saving = ref(false)
const dirty = ref(false)

async function fetchPrompts() {
  if (!props.promptPath) return
  loading.value = true
  try {
    const res = await fetch(`/api/config/sample-prompts?path=${encodeURIComponent(props.promptPath)}`)
    const data = await res.json()
    entries.value = data.entries || []
    dirty.value = false
  } catch {
    entries.value = []
  } finally {
    loading.value = false
  }
}

async function save() {
  saving.value = true
  try {
    const res = await fetch('/api/config/sample-prompts', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: props.promptPath, entries: entries.value }),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    dirty.value = false
    notify.show(t('ppeSaved'), 'success')
  } catch {
    notify.show(t('ppeSaveFailed'), 'error')
  } finally {
    saving.value = false
  }
}

function addEntry() {
  entries.value.push({ prompt: '', negative_prompt: '' })
  dirty.value = true
}

function removeEntry(index: number) {
  entries.value.splice(index, 1)
  dirty.value = true
}

function updateEntry(index: number, key: keyof PromptEntry, value: string) {
  entries.value[index][key] = value
  dirty.value = true
}

onMounted(fetchPrompts)
watch(() => props.promptPath, fetchPrompts)
</script>
