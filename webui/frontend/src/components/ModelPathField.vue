<template>
  <div>
    <v-text-field
      :model-value="currentValue"
      :label="field.key"
      :hint="hintText"
      :loading="validating"
      persistent-hint
      variant="outlined"
      density="compact"
      hide-details="auto"
      class="font-mono-field"
      @update:model-value="onInput"
    >
      <template #append-inner>
        <v-icon
          v-if="exists === true"
          icon="mdi-check-circle"
          color="success"
          size="small"
          class="mr-1"
        />
        <v-icon
          v-else-if="exists === false"
          icon="mdi-alert-circle"
          color="warning"
          size="small"
          class="mr-1"
        />
        <v-chip v-if="field.origin !== 'method'" size="x-small" variant="outlined" class="mr-1">
          {{ field.origin }}
        </v-chip>
        <v-btn
          icon="mdi-folder-open"
          size="x-small"
          variant="text"
          @click="openBrowser"
        />
      </template>
    </v-text-field>

    <v-dialog v-model="dialog" max-width="700" scrollable>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-folder-open" class="mr-2" />
          {{ t('cfSelectModel') }}
        </v-card-title>

        <v-card-subtitle class="text-caption pb-2">
          {{ currentDir }}
        </v-card-subtitle>

        <v-divider />

        <v-card-text style="max-height: 50vh" class="pa-0">
          <v-list density="compact" class="py-0">
            <!-- Parent directory -->
            <v-list-item
              v-if="parentDir"
              prepend-icon="mdi-arrow-up-bold"
              :title="t('cfParentDir')"
              @click="navigateTo(parentDir)"
            />

            <!-- Subdirectories -->
            <v-list-item
              v-for="dir in subdirs"
              :key="dir.path"
              prepend-icon="mdi-folder"
              :title="dir.name"
              @click="navigateTo(dir.path)"
            />

            <v-divider v-if="subdirs.length > 0 && files.length > 0" />

            <!-- Files -->
            <v-list-item
              v-for="file in files"
              :key="file.path"
              :active="selectedFile === file.path"
              @click="selectedFile = file.path"
            >
              <template #prepend>
                <v-icon icon="mdi-file-outline" class="mr-2" />
              </template>
              <v-list-item-title>{{ file.name }}</v-list-item-title>
              <v-list-item-subtitle>{{ file.size_human }}</v-list-item-subtitle>
              <template #append>
                <v-icon
                  v-if="selectedFile === file.path"
                  icon="mdi-check"
                  color="primary"
                />
              </template>
            </v-list-item>

            <!-- Empty state -->
            <v-list-item v-if="subdirs.length === 0 && files.length === 0">
              <v-list-item-title class="text-medium-emphasis">
                {{ t('cfNoFiles') }}
              </v-list-item-title>
            </v-list-item>
          </v-list>
        </v-card-text>

        <v-divider />

        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="dialog = false">{{ t('cfCancel') }}</v-btn>
          <v-btn
            color="primary"
            :disabled="!selectedFile"
            @click="confirmSelection"
          >
            {{ t('cfConfirm') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import type { FieldMeta } from '../stores/config'
import { useConfigStore } from '../stores/config'
import { useI18n } from '../composables/useI18n'

const props = defineProps<{ field: FieldMeta }>()
const emit = defineEmits<{ update: [value: unknown] }>()
const configStore = useConfigStore()
const { t } = useI18n()

const currentValue = computed(() => configStore.getFieldValue(props.field.key) as string)

const hintText = computed(() => {
  const desc = props.field.description
  const descEn = props.field.description_en
  if (!desc && !descEn) return undefined
  return desc || descEn
})

// ── Existence validation ──────────────────────────────────────

const exists = ref<boolean | null>(null)
const validating = ref(false)
let validateTimer: ReturnType<typeof setTimeout> | null = null

async function checkExists(path: string) {
  if (!path) {
    exists.value = null
    return
  }
  validating.value = true
  try {
    const res = await fetch(`/api/files/validate?path=${encodeURIComponent(path)}`)
    const data = await res.json()
    exists.value = data.exists
  } catch {
    exists.value = null
  } finally {
    validating.value = false
  }
}

function onInput(val: string) {
  emit('update', val)
  if (validateTimer) clearTimeout(validateTimer)
  validateTimer = setTimeout(() => checkExists(val), 500)
}

onMounted(() => checkExists(currentValue.value))
watch(currentValue, (val) => checkExists(val))

// ── File browser dialog ───────────────────────────────────────

const dialog = ref(false)
const currentDir = ref('')
const parentDir = ref<string | null>(null)
const subdirs = ref<{ name: string; path: string }[]>([])
const files = ref<{ name: string; path: string; size: number; size_human: string; mtime: string }[]>([])
const selectedFile = ref<string | null>(null)

const _DEFAULT_DIRS: Record<string, string> = {
  pretrained_model_name_or_path: 'models/diffusion_models',
  qwen3: 'models/text_encoders',
  vae: 'models/vae',
}

function getDefaultDir(): string {
  return _DEFAULT_DIRS[props.field.key] || 'models'
}

function getInitialDir(): string {
  const val = currentValue.value
  if (val) {
    // Strip filename to get directory — handles both / and \ separators
    const lastSlash = Math.max(val.lastIndexOf('/'), val.lastIndexOf('\\'))
    if (lastSlash > 0) return val.substring(0, lastSlash)
  }
  return getDefaultDir()
}

async function openBrowser() {
  selectedFile.value = null
  await navigateTo(getInitialDir())
  dialog.value = true
}

async function navigateTo(dirPath: string) {
  selectedFile.value = null
  try {
    const ext = '.safetensors'
    const res = await fetch(`/api/files/browse?dir=${encodeURIComponent(dirPath)}&ext=${encodeURIComponent(ext)}`)
    const data = await res.json()
    if (data.error) return
    currentDir.value = data.current_dir || data.current_dir_abs || dirPath
    parentDir.value = data.parent
    subdirs.value = data.subdirs || []
    files.value = data.files || []
  } catch {
    // silent
  }
}

function confirmSelection() {
  if (selectedFile.value) {
    emit('update', selectedFile.value)
    dialog.value = false
    checkExists(selectedFile.value)
  }
}
</script>

<style scoped>
:deep(.v-field--focused) {
  box-shadow: 0 0 0 2px rgba(199, 91, 26, 0.15);
  border-radius: var(--radius-md);
}
:deep(.v-field--focused .v-field__outline) {
  border-color: var(--forge-ember) !important;
}
</style>
