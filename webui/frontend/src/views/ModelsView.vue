<template>
  <v-container fluid class="pa-4 d-flex flex-column models-page">
    <div class="d-flex align-center mb-1">
      <div class="text-h5">{{ t('mmTitle') }}</div>
      <v-spacer />
      <v-btn
        variant="tonal"
        color="primary"
        prepend-icon="mdi-refresh"
        :loading="loading"
        @click="fetchModels"
      >
        {{ t('mmReload') }}
      </v-btn>
    </div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('mmSubtitle') }}</div>

    <div class="d-flex align-center ga-3 mb-3 flex-wrap">
      <v-text-field
        v-model="search"
        :label="t('mmSearch')"
        prepend-inner-icon="mdi-magnify"
        variant="outlined"
        density="compact"
        hide-details
        clearable
        max-width="320"
      />
      <v-chip v-if="models.length > 0" size="small" variant="tonal" color="primary">
        {{ t('mmTotalCount', { n: String(models.length), size: totalSize }) }}
      </v-chip>
    </div>

    <v-card variant="tonal" class="flex-grow-1 models-scroll">
      <v-table v-if="filteredModels.length > 0" hover>
        <thead>
          <tr>
            <th>{{ t('mmColName') }}</th>
            <th class="text-no-wrap">{{ t('mmColType') }}</th>
            <th class="text-no-wrap">{{ t('mmColRank') }}</th>
            <th class="text-no-wrap">{{ t('mmColSize') }}</th>
            <th class="text-no-wrap">{{ t('mmColModified') }}</th>
            <th class="text-no-wrap">{{ t('mmColActions') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="m in filteredModels" :key="m.path">
            <td>
              <div class="text-body-2 font-weight-medium">{{ m.name }}</div>
              <div class="text-caption text-medium-emphasis">{{ m.directory }}</div>
            </td>
            <td>
              <v-chip size="small" variant="tonal" :color="typeColor(m.spec)">
                {{ m.type_label }}
              </v-chip>
            </td>
            <td class="text-body-2 text-no-wrap">{{ rankText(m) }}</td>
            <td class="text-body-2 text-no-wrap">{{ m.size_human }}</td>
            <td class="text-body-2 text-no-wrap">{{ formatDate(m.mtime) }}</td>
            <td class="text-no-wrap">
              <v-btn
                icon="mdi-download"
                size="small"
                variant="text"
                :title="t('mmDownload')"
                :href="downloadUrl(m)"
              />
              <v-btn
                icon="mdi-information-outline"
                size="small"
                variant="text"
                :title="t('mmDetails')"
                @click="openDetails(m)"
              />
              <v-btn
                icon="mdi-delete-outline"
                size="small"
                variant="text"
                color="error"
                :title="t('mmDelete')"
                :loading="deletingPath === m.path"
                @click="deleteModel(m)"
              />
            </td>
          </tr>
        </tbody>
      </v-table>
      <div v-else-if="!loading" class="text-center pa-10">
        <v-icon icon="mdi-cube-outline" size="48" class="mb-3 text-medium-emphasis" />
        <div class="text-body-1">{{ t('mmEmpty') }}</div>
        <div class="text-body-2 text-medium-emphasis mt-1">{{ t('mmEmptyHint') }}</div>
      </div>
    </v-card>

    <!-- Details dialog: training metadata of one adapter -->
    <v-dialog v-model="detailsOpen" max-width="720">
      <v-card v-if="details">
        <v-card-title class="d-flex align-center ga-2">
          <span class="text-subtitle-1">{{ t('mmDetailsTitle') }}</span>
          <v-chip size="small" variant="tonal" :color="typeColor(details.spec)">
            {{ details.type_label }}
          </v-chip>
          <v-spacer />
          <v-btn icon="mdi-close" size="small" variant="text" @click="detailsOpen = false" />
        </v-card-title>
        <v-card-text>
          <div class="text-body-2 font-weight-medium mb-1">{{ details.name }}</div>
          <div class="text-caption text-medium-emphasis mb-3">
            {{ details.size_human }} · {{ details.path }}
          </div>

          <template v-if="summaryRows.length > 0">
            <v-table density="compact" class="mb-3">
              <tbody>
                <tr v-for="row in summaryRows" :key="row.label">
                  <td class="text-medium-emphasis" style="width: 40%">{{ row.label }}</td>
                  <td class="text-body-2">{{ row.value }}</td>
                </tr>
              </tbody>
            </v-table>
          </template>
          <div v-else class="text-body-2 text-medium-emphasis mb-3">
            {{ t('mmNoMetadata') }}
          </div>

          <div v-if="details.network_args" class="mb-2">
            <div class="text-subtitle-2 mb-1">{{ t('mmNetworkArgs') }}</div>
            <div class="mono-block pa-3 text-caption">
              <div v-for="(value, key) in details.network_args" :key="key">
                {{ key }}: {{ JSON.stringify(value) }}
              </div>
            </div>
          </div>

          <div v-if="Object.keys(details.metadata).length > 0">
            <div class="text-subtitle-2 mb-1">{{ t('mmRawMetadata') }}</div>
            <div class="mono-block pa-3 text-caption" style="max-height: 220px; overflow-y: auto">
              <div v-for="key in rawKeys" :key="key">{{ key }}: {{ details.metadata[key] }}</div>
            </div>
          </div>
        </v-card-text>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useI18n } from '../composables/useI18n'
import { useNotifyStore } from '../stores/notify'

interface ModelEntry {
  name: string
  path: string
  directory: string
  size: number
  size_human: string
  mtime: string
  created: string
  spec: string
  type_label: string
  dim: string
  alpha: string
  base_model: string
  output_name: string
}

interface ModelMetadata {
  path: string
  name: string
  size: number
  size_human: string
  spec: string
  type_label: string
  network_args: Record<string, unknown> | null
  metadata: Record<string, string>
}

const { t } = useI18n()
const notify = useNotifyStore()

const models = ref<ModelEntry[]>([])
const loading = ref(false)
const search = ref('')
const deletingPath = ref('')
const detailsOpen = ref(false)
const details = ref<ModelMetadata | null>(null)

const TYPE_COLORS: Record<string, string> = {
  lora: 'primary',
  lokr: 'indigo',
  glokr: 'deep-purple',
  loha: 'teal',
  dylora: 'orange',
  vera: 'pink',
  ortho: 'blue-grey',
  ortho_init: 'blue-grey',
  hydra: 'cyan',
  ortho_hydra: 'cyan',
  chimera_hydra: 'cyan',
  step_expert: 'light-green',
  stacked_experts_global_fei: 'light-green',
}

const filteredModels = computed(() => {
  const q = (search.value ?? '').trim().toLowerCase()
  if (!q) return models.value
  return models.value.filter(
    (m) =>
      m.name.toLowerCase().includes(q) ||
      m.directory.toLowerCase().includes(q) ||
      m.type_label.toLowerCase().includes(q),
  )
})

const totalSize = computed(() => {
  let n = models.value.reduce((acc, m) => acc + m.size, 0)
  for (const unit of ['B', 'KB', 'MB', 'GB']) {
    if (n < 1024) return `${n.toFixed(1)} ${unit}`
    n /= 1024
  }
  return `${n.toFixed(1)} TB`
})

const rawKeys = computed(() =>
  details.value ? Object.keys(details.value.metadata).sort() : [],
)

const summaryRows = computed(() => {
  if (!details.value) return []
  const md = details.value.metadata
  const row = (label: string, value: string | undefined) => {
    if (value === undefined || value === '' || value === 'None') return null
    return { label, value }
  }
  const stamp = (key: string) => {
    const v = md[key]
    if (!v || v === 'None') return undefined
    const n = Number(v)
    return Number.isFinite(n) && n > 0 ? formatDateTs(n * 1000) : v
  }
  const rows = [
    row(t('mmModule'), md.ss_network_module),
    row(t('mmDimAlpha'), joinDimAlpha(md.ss_network_dim, md.ss_network_alpha)),
    row(t('mmBaseModel'), md.ss_base_model_version || md.ss_sd_model_name),
    row(t('mmOptimizer'), md.ss_optimizer),
    row(t('mmLearningRate'), md.ss_learning_rate),
    row(
      t('mmEpochs'),
      md.ss_epoch && md.ss_num_epochs ? `${md.ss_epoch} / ${md.ss_num_epochs}` : undefined,
    ),
    row(t('mmSteps'), md.ss_max_train_steps),
    row(t('mmImages'), md.ss_num_train_images),
    row(t('mmSeed'), md.ss_seed),
    row(t('mmStarted'), stamp('ss_training_started_at')),
    row(t('mmFinished'), stamp('ss_training_finished_at')),
    row(t('mmComment'), md.ss_training_comment),
  ].filter((r): r is { label: string; value: string } => r !== null)
  return rows
})

function typeColor(spec: string): string {
  return TYPE_COLORS[spec] ?? 'grey'
}

function rankText(m: ModelEntry): string {
  return joinDimAlpha(m.dim, m.alpha) || '—'
}

function joinDimAlpha(dim?: string, alpha?: string): string {
  if (dim && alpha) return `${dim} × ${alpha}`
  return dim || alpha || ''
}

function formatDate(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}

function formatDateTs(ms: number): string {
  return new Date(ms).toLocaleString()
}

function downloadUrl(m: ModelEntry): string {
  return `/api/models/download?path=${encodeURIComponent(m.path)}`
}

async function fetchModels() {
  loading.value = true
  try {
    const res = await fetch('/api/models')
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    models.value = data.models ?? []
  } catch (e) {
    models.value = []
    notify.show(t('mmLoadFailed', { error: String(e) }), 'error')
  } finally {
    loading.value = false
  }
}

async function openDetails(m: ModelEntry) {
  try {
    const res = await fetch(`/api/models/metadata?path=${encodeURIComponent(m.path)}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    details.value = await res.json()
    detailsOpen.value = true
  } catch (e) {
    notify.show(t('mmLoadFailed', { error: String(e) }), 'error')
  }
}

async function deleteModel(m: ModelEntry) {
  if (!window.confirm(t('mmDeleteConfirm', { name: m.name }))) return
  deletingPath.value = m.path
  try {
    const res = await fetch(`/api/models?path=${encodeURIComponent(m.path)}`, {
      method: 'DELETE',
    })
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail ?? `HTTP ${res.status}`)
    }
    models.value = models.value.filter((x) => x.path !== m.path)
    notify.show(t('mmDeleted', { name: m.name }), 'success')
  } catch (e) {
    notify.show(t('mmDeleteFailed', { error: String(e) }), 'error')
  } finally {
    deletingPath.value = ''
  }
}

onMounted(fetchModels)
</script>

<style scoped>
.models-scroll {
  overflow-y: auto;
}
.mono-block {
  font-family: monospace;
  background: rgba(128, 128, 128, 0.08);
  border-radius: 6px;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
