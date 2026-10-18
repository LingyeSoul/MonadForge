<template>
  <div>
    <v-text-field
      :model-value="currentValue"
      :label="field.label ?? field.key"
      :hint="hintText"
      :title="currentValue"
      :readonly="field.read_only"
      :loading="validating"
      variant="outlined"
      density="compact"
      hide-details="auto"
      class="font-mono-field"
      @update:model-value="onInput"
    >
      <template #append-inner>
        <FieldHelpButton :field-key="field.key" @help="emit('help-click', field.key)" />
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
          :data-testid="`browse-${field.key}`"
          :disabled="field.read_only"
          :aria-label="`${browserTitle}: ${field.label ?? field.key}`"
          @click="openBrowser"
        />
      </template>
    </v-text-field>

    <v-dialog v-model="dialog" max-width="700" scrollable>
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-folder-open" class="mr-2" />
          {{ browserTitle }}
        </v-card-title>

        <v-card-subtitle class="text-caption pb-2 model-browser-path">
          {{ currentDir }}
        </v-card-subtitle>

        <v-divider />
        <v-progress-linear v-if="browserLoading" indeterminate />
        <v-alert v-if="browserError" type="error" variant="tonal" style="flex: 0 0 auto;">
          {{ browserError }}
          <template #append><v-btn variant="text" size="small" @click="navigateTo(currentDir)">{{ t('ppRefresh') }}</v-btn></template>
        </v-alert>
        <v-card-text style="max-height: 50vh" class="pa-0">
          <v-list density="compact" class="py-0">
            <!-- Parent directory -->
            <v-list-item
              v-if="parentDir"
              prepend-icon="mdi-arrow-up-bold"
              :title="t('cfParentDir')"
              :disabled="browserLoading"
              @click="navigateTo(parentDir)"
            />

            <!-- Subdirectories -->
            <v-list-item
              v-for="dir in subdirs"
              :key="dir.path"
              prepend-icon="mdi-folder"
              :title="dir.name"
              :disabled="browserLoading"
              @click="navigateTo(dir.path)"
            />

            <v-divider v-if="subdirs.length > 0 && visibleFiles.length > 0" />

            <!-- Files -->
            <v-list-item
              v-for="file in visibleFiles"
              :key="file.path"
              :active="selectedFile === file.path"
              :disabled="browserLoading || !!browserError"
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
            <v-list-item v-if="!browserLoading && !browserError && subdirs.length === 0 && visibleFiles.length === 0">
              <v-list-item-title class="text-medium-emphasis">
                {{ t(field.path_kind === 'directory' ? 'cfNoDirectories' : 'cfNoFiles') }}
              </v-list-item-title>
            </v-list-item>
          </v-list>
        </v-card-text>

        <v-divider />

        <v-card-actions>
          <v-btn v-if="field.path_kind === 'directory' || field.path_kind === 'either'" :data-testid="`select-directory-${field.key}`" :disabled="!browserReady || browserLoading || !!browserError || field.read_only" variant="tonal" @click="selectDirectory">{{ t('qwUseDirectory') }}</v-btn>
          <v-spacer />
          <v-btn variant="text" @click="dialog = false">{{ t('cfCancel') }}</v-btn>
          <v-btn
            v-if="field.path_kind !== 'directory'"
            color="primary"
            :disabled="!selectedFile || !browserReady || browserLoading || !!browserError || field.read_only"
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
import { ref, computed, watch, onBeforeUnmount } from 'vue'
import FieldHelpButton from './FieldHelpButton.vue'
import type { FieldMeta } from '../stores/config'
import { useConfigStore } from '../stores/config'
import { useI18n } from '../composables/useI18n'

const props = defineProps<{ field: FieldMeta; modelValue?: string }>()
const emit = defineEmits<{ update: [value: string]; 'help-click': [key: string] }>()
const configStore = useConfigStore()
const { t } = useI18n()
const browserTitle = computed(() => t(props.field.path_kind === 'directory' ? 'cfSelectDirectory' : props.field.path_kind === 'either' ? 'cfSelectPath' : 'cfSelectModel'))

const currentValue = computed(() => props.modelValue !== undefined ? props.modelValue : configStore.getFieldValue(props.field.key) as string)

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
let validationRequest = 0
let validationController: AbortController | null = null
const validatedKind = ref<'file' | 'directory' | null>(null)

async function checkExists(path: string, request: number) {
  const controller = new AbortController()
  validationController = controller
  try {
    let kind: 'file' | 'directory' | null = null
    if (props.field.path_kind !== 'directory') {
      const res = await fetch(`/api/files/validate?path=${encodeURIComponent(path)}`, { signal: controller.signal })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      if (request !== validationRequest) return
      if (data.exists === true) kind = 'file'
    }
    // The validation endpoint checks files only. Directory fields and directory
    // overrides use the existing browser endpoint so a valid folder is accepted.
    if (!kind && (props.field.path_kind === 'directory' || props.field.path_kind === 'either')) {
      const res = await fetch(`/api/files/browse?dir=${encodeURIComponent(path)}&ext=${encodeURIComponent(props.field.path_extensions ?? '.safetensors')}`, { signal: controller.signal })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      if (!data.error) kind = 'directory'
    }
    if (request !== validationRequest) return
    exists.value = kind !== null
    validatedKind.value = kind
  } catch {
    if (request === validationRequest) exists.value = null
  } finally {
    if (request === validationRequest) validating.value = false
  }
}

function onInput(val: string | null) { emit('update', val ?? '') }
watch(currentValue, (val, previous) => {
  const request = ++validationRequest
  validationController?.abort()
  if (validateTimer) clearTimeout(validateTimer)
  exists.value = null
  validatedKind.value = null
  validating.value = Boolean(val)
  if (val) validateTimer = setTimeout(() => void checkExists(val, request), previous === undefined ? 0 : 500)
}, { immediate: true, flush: 'sync' })

// ── File browser dialog ───────────────────────────────────────

const dialog = ref(false)
const browserError = ref('')
const browserLoading = ref(false)
const browserReady = ref(false)
let browserRequest = 0
let browserController: AbortController | null = null
let browserValidating = false
function cancelBrowserRequest() {
  browserRequest++
  browserController?.abort()
  browserLoading.value = false
  browserReady.value = false
  selectedFile.value = null
  if (browserValidating) {
    validationRequest++
    validationController?.abort()
    validating.value = false
    browserValidating = false
  }
}
watch(dialog, open => { if (!open) cancelBrowserRequest() }, { flush: 'sync' })
onBeforeUnmount(() => {
  validationRequest++
  validationController?.abort()
  if (validateTimer) clearTimeout(validateTimer)
  cancelBrowserRequest()
})
function selectDirectory(): void {
  if (!browserReady.value || browserLoading.value || browserError.value || props.field.read_only) return
  emit('update', currentDir.value)
  dialog.value = false
}
const currentDir = ref('')
const parentDir = ref<string | null>(null)
const subdirs = ref<{ name: string; path: string }[]>([])
const files = ref<{ name: string; path: string; size: number; size_human: string; mtime: string }[]>([])
const visibleFiles = computed(() => props.field.path_kind === 'directory' ? [] : files.value)
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
    if (props.field.path_kind === 'directory' || validatedKind.value === 'directory') return val
    // Strip filename to get directory — handles both / and \ separators
    const lastSlash = Math.max(val.lastIndexOf('/'), val.lastIndexOf('\\'))
    if (lastSlash === 0) return val[0]
    if (lastSlash === 2 && /^[A-Za-z]:/.test(val)) return val.substring(0, 3)
    if (lastSlash > 0) return val.substring(0, lastSlash)
  }
  return getDefaultDir()
}

async function openBrowser() {
  if (props.field.read_only) return
  cancelBrowserRequest()
  const opening = browserRequest
  browserError.value = ''
  currentDir.value = ''
  parentDir.value = null
  subdirs.value = []
  files.value = []
  dialog.value = true
  browserLoading.value = true
  // An either-kind path may be a directory without a trailing separator.
  // Resolve that distinction before choosing its starting folder, including
  // when the user opens the picker before the debounced check has finished.
  if (props.field.path_kind === 'either' && currentValue.value && !validatedKind.value) {
    if (validateTimer) clearTimeout(validateTimer)
    validationController?.abort()
    validating.value = true
    browserValidating = true
    await checkExists(currentValue.value, ++validationRequest)
    if (opening !== browserRequest || !dialog.value) return
    browserValidating = false
  }
  await navigateTo(getInitialDir())
}

async function navigateTo(dirPath: string) {
  const request = ++browserRequest
  browserController?.abort()
  const controller = new AbortController()
  browserController = controller
  selectedFile.value = null
  browserLoading.value = true
  browserReady.value = false
  browserError.value = ''
  currentDir.value = dirPath
  parentDir.value = null
  subdirs.value = []
  files.value = []
  try {
    const ext = props.field.path_extensions ?? '.safetensors'
    const res = await fetch(`/api/files/browse?dir=${encodeURIComponent(dirPath)}&ext=${encodeURIComponent(ext)}`, { signal: controller.signal })
    const data = await res.json()
    if (!res.ok || data.error) throw new Error(data.detail ?? data.error ?? `HTTP ${res.status}`)
    if (request !== browserRequest || !dialog.value) return
    currentDir.value = data.current_dir || data.current_dir_abs || dirPath
    parentDir.value = data.parent
    subdirs.value = data.subdirs || []
    files.value = data.files || []
    browserReady.value = true
  } catch (err) {
    if (request === browserRequest) browserError.value = err instanceof Error ? err.message : String(err)
  } finally {
    if (request === browserRequest) browserLoading.value = false
  }
}

function confirmSelection() {
  if (selectedFile.value && browserReady.value && !browserLoading.value && !browserError.value && !props.field.read_only) {
    emit('update', selectedFile.value)
    dialog.value = false
  }
}
</script>

<style scoped>
.model-browser-path { white-space: normal; overflow-wrap: anywhere; }
:deep(.v-field__input) { min-width: 0; }

:deep(.v-field--focused) {
  box-shadow: 0 0 0 2px rgba(199, 91, 26, 0.15);
  border-radius: var(--radius-md);
}
:deep(.v-field--focused .v-field__outline) {
  border-color: var(--forge-ember) !important;
}
</style>
