<template>
  <v-container fluid class="pa-4">
    <WorkspaceHeading :title="t('adTitle')" :subtitle="t('adSubtitle')" />

    <v-row>
      <v-col cols="12" md="6">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-image-frame" class="mr-2" />
            {{ t('adIpAdapter') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-3" v-html="t('adIpDesc')" />
            <v-alert type="info" variant="tonal" density="compact" class="mb-2">
              <span v-html="t('adIpHint')" />
            </v-alert>
            <v-progress-linear v-if="statsLoading['ip']" indeterminate class="mt-2" />
            <v-alert v-if="statsErrors['ip']" data-testid="adapter-ip-stats-error" type="error" variant="tonal" density="compact" class="mt-2">
              {{ t('adSourceImages') }} · {{ statsErrors['ip'] }}
              <template #append><v-btn data-testid="adapter-ip-stats-retry" variant="text" size="small" :loading="statsLoading['ip']" @click="fetchStats('image_dataset', 'ip')">{{ t('cfgReload') }}</v-btn></template>
            </v-alert>
            <!-- Dataset stats -->
            <v-row density="comfortable" class="mt-2" v-if="adapterStats['ip']">
              <v-col cols="auto">
                <v-chip prepend-icon="mdi-image" variant="tonal"
                        :color="adapterStats['ip'].source_count > 0 ? 'success' : 'default'" size="small">
                  {{ t('adSourceImages') }}: {{ adapterStats['ip'].source_count }}
                </v-chip>
              </v-col>
              <v-col cols="auto">
                <v-chip prepend-icon="mdi-text-box" variant="tonal"
                        :color="adapterStats['ip'].caption_count > 0 ? 'success' : 'default'" size="small">
                  {{ t('adCaptions') }}: {{ adapterStats['ip'].caption_count }}
                </v-chip>
              </v-col>
              <v-col cols="auto">
                <v-chip prepend-icon="mdi-database" variant="tonal"
                        :color="adapterStats['ip'].cache.latents > 0 ? 'success' : 'default'" size="small">
                  {{ t('adCacheLatents') }}: {{ adapterStats['ip'].cache.latents }}
                </v-chip>
              </v-col>
              <v-col cols="auto">
                <v-chip prepend-icon="mdi-brain" variant="tonal"
                        :color="adapterStats['ip'].cache.te > 0 ? 'success' : 'default'" size="small">
                  {{ t('adCacheTE') }}: {{ adapterStats['ip'].cache.te }}
                </v-chip>
              </v-col>
              <v-col cols="auto">
                <v-chip prepend-icon="mdi-vector-square" variant="tonal"
                        :color="adapterStats['ip'].cache.pe > 0 ? 'success' : 'default'" size="small">
                  {{ t('adCachePE') }}: {{ adapterStats['ip'].cache.pe }}
                </v-chip>
              </v-col>
            </v-row>
            <!-- Dataset preview -->
            <div v-if="ipPreviewImages.length" class="dataset-preview mt-3">
              <div class="d-flex align-center mb-1">
                <span class="text-caption text-medium-emphasis">{{ t('adDatasetPreview') }}</span>
                <v-spacer />
                <v-btn
                  size="x-small"
                  variant="text"
                  :to="{ path: '/dataset', query: { dir: 'image_dataset' } }"
                >
                  {{ t('adBrowseDataset') }}
                </v-btn>
              </div>
              <div class="preview-thumbs">
                <v-img
                  v-for="img in ipPreviewImages"
                  :key="img.path"
                  :src="`/api/images/file/${encodeURIComponent(img.path)}?directory=image_dataset`"
                  class="preview-thumb rounded"
                  width="64"
                  height="64"
                  cover
                />
              </div>
            </div>
          </v-card-text>
        </v-card>
      </v-col>

      <v-col cols="12" md="6">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-gesture-tap-button" class="mr-2" />
            {{ t('adEasyControl') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-3" v-html="t('adEasyDesc')" />
            <v-alert type="info" variant="tonal" density="compact" class="mb-2">
              <span v-html="t('adEasyHint')" />
            </v-alert>
            <v-progress-linear v-if="statsLoading['ec']" indeterminate class="mt-2" />
            <v-alert v-if="statsErrors['ec']" data-testid="adapter-ec-stats-error" type="error" variant="tonal" density="compact" class="mt-2">
              {{ t('adSourceImages') }} · {{ statsErrors['ec'] }}
              <template #append><v-btn data-testid="adapter-ec-stats-retry" variant="text" size="small" :loading="statsLoading['ec']" @click="fetchStats('easycontrol-dataset', 'ec')">{{ t('cfgReload') }}</v-btn></template>
            </v-alert>
            <!-- Dataset stats -->
            <v-row density="comfortable" class="mt-2" v-if="adapterStats['ec']">
              <v-col cols="auto">
                <v-chip prepend-icon="mdi-image" variant="tonal"
                        :color="adapterStats['ec'].source_count > 0 ? 'success' : 'default'" size="small">
                  {{ t('adSourceImages') }}: {{ adapterStats['ec'].source_count }}
                </v-chip>
              </v-col>
              <v-col cols="auto">
                <v-chip prepend-icon="mdi-text-box" variant="tonal"
                        :color="adapterStats['ec'].caption_count > 0 ? 'success' : 'default'" size="small">
                  {{ t('adCaptions') }}: {{ adapterStats['ec'].caption_count }}
                </v-chip>
              </v-col>
              <v-col cols="auto">
                <v-chip prepend-icon="mdi-database" variant="tonal"
                        :color="adapterStats['ec'].cache.latents > 0 ? 'success' : 'default'" size="small">
                  {{ t('adCacheLatents') }}: {{ adapterStats['ec'].cache.latents }}
                </v-chip>
              </v-col>
              <v-col cols="auto">
                <v-chip prepend-icon="mdi-brain" variant="tonal"
                        :color="adapterStats['ec'].cache.te > 0 ? 'success' : 'default'" size="small">
                  {{ t('adCacheTE') }}: {{ adapterStats['ec'].cache.te }}
                </v-chip>
              </v-col>
              <v-col cols="auto">
                <v-chip prepend-icon="mdi-vector-square" variant="tonal"
                        :color="adapterStats['ec'].cache.pe > 0 ? 'success' : 'default'" size="small">
                  {{ t('adCachePE') }}: {{ adapterStats['ec'].cache.pe }}
                </v-chip>
              </v-col>
            </v-row>
            <!-- Dataset preview -->
            <div v-if="ecPreviewImages.length" class="dataset-preview mt-3">
              <div class="d-flex align-center mb-1">
                <span class="text-caption text-medium-emphasis">{{ t('adDatasetPreview') }}</span>
                <v-spacer />
                <v-btn
                  size="x-small"
                  variant="text"
                  :to="{ path: '/dataset', query: { dir: 'easycontrol-dataset' } }"
                >
                  {{ t('adBrowseDataset') }}
                </v-btn>
              </div>
              <div class="preview-thumbs">
                <v-img
                  v-for="img in ecPreviewImages"
                  :key="img.path"
                  :src="`/api/images/file/${encodeURIComponent(img.path)}?directory=easycontrol-dataset`"
                  class="preview-thumb rounded"
                  width="64"
                  height="64"
                  cover
                />
              </div>
            </div>
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('easycontrol')" @click="runTask('easycontrol')">
              {{ t('adTrainEasy') }}
            </v-btn>
            <v-btn variant="text" :loading="isRunning('easycontrol-preprocess')" @click="runTask('easycontrol-preprocess')">
              {{ t('adPreprocess') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>

    <v-divider class="my-4" />

    <div class="d-flex align-center mb-2">
      <div class="text-subtitle-1">{{ t('adActiveTasks') }}</div><v-spacer />
      <v-btn variant="text" size="small" prepend-icon="mdi-refresh" :loading="refreshingTasks" @click="refreshTasks">{{ t('ppRefresh') }}</v-btn>
    </div>
    <v-alert v-if="taskStore.taskListError" data-testid="adapter-tasks-error" type="error" variant="tonal" density="compact">
      {{ t('taskLoadFailed') }}
      <template #append><v-btn data-testid="adapter-tasks-retry" variant="text" size="small" :loading="refreshingTasks" @click="refreshTasks">{{ t('cfgReload') }}</v-btn></template>
    </v-alert>
    <v-list v-else-if="adapterTasks.length > 0" density="compact">
      <v-list-item
        v-for="task in visibleTasks"
        :key="task.task_id" :to="{ path: '/tasks', query: { task: task.task_id } }"
        :title="task.command"
        :subtitle="`${task.task_id.slice(0, 8)} | ${t('taskState')}: ${task.state} | PID: ${task.pid ?? '—'}`"
      >
        <template #append>
          <v-chip size="small" :color="stateColor(task.state)" variant="tonal">{{ task.state }}</v-chip>
          <TaskStopButton v-if="isTaskStoppable(task.state)" :task-id="task.task_id" />
        </template>
      </v-list-item>
    </v-list>
    <div v-else class="text-medium-emphasis text-body-2">{{ t('adNoTasks') }}</div>
    <v-btn
      v-if="hasHiddenTasks"
      variant="text"
      size="small"
      class="mt-1"
      :prepend-icon="showAllTasks ? 'mdi-chevron-up' : 'mdi-chevron-down'"
      @click="showAllTasks = !showAllTasks"
    >
      {{ showAllTasks ? t('taskShowLess') : t('taskShowAll', { count: adapterTasks.length }) }}
    </v-btn>
  </v-container>
</template>

<script setup lang="ts">
import { isTaskActive, isTaskStoppable } from '../utils/taskState.ts'
import WorkspaceHeading from '../components/WorkspaceHeading.vue'
import TaskStopButton from '../components/TaskStopButton.vue'
import { ref, reactive, computed, onMounted } from 'vue'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useI18n } from '../composables/useI18n'

const taskStore = useTaskStore()
const notify = useNotifyStore()
const { t } = useI18n()
const refreshingTasks = ref(false)
async function refreshTasks(): Promise<void> {
  if (refreshingTasks.value) return
  refreshingTasks.value = true
  try { await taskStore.fetchTasks() }
  finally { refreshingTasks.value = false }
}
void refreshTasks()

const adapterCommands = [
  'easycontrol', 'easycontrol-preprocess',
]

const adapterTasks = computed(() =>
  taskStore.tasks.filter(tp => adapterCommands.includes(tp.command))
)

// ── Task list preview (collapse long lists) ────────────────────

const TASK_PREVIEW_LIMIT = 4
const showAllTasks = ref(false)
const visibleTasks = computed(() =>
  showAllTasks.value ? adapterTasks.value : adapterTasks.value.slice(0, TASK_PREVIEW_LIMIT)
)
const hasHiddenTasks = computed(() => adapterTasks.value.length > TASK_PREVIEW_LIMIT)

function isRunning(command: string) {
  return taskStore.tasks.some(tp => tp.command === command && isTaskActive(tp.state))
}

function stateColor(state: string) {
  if (state === 'paused') return 'warning'
  if (state === 'running' || state === 'stopping') return 'info'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}

async function runTask(command: string) {
  if (taskStore.loading || isRunning(command)) return
  const taskId = await taskStore.startTask(command)
  if (taskId) {
    notify.show(t('notifyTaskStarted', { command }), 'success')
  } else {
    notify.show(t('notifyTaskStartFailed', { command }), 'error')
  }
}

// ── Dataset preview ──────────────────────────────────────────

interface PreviewImage { path: string; filename: string }
interface AdapterCacheStats { latents: number; te: number; pe: number }
interface AdapterStatsData { source_count: number; caption_count: number; cache: AdapterCacheStats }

const ipPreviewImages = ref<PreviewImage[]>([])
const ecPreviewImages = ref<PreviewImage[]>([])
const adapterStats = reactive<Record<string, AdapterStatsData>>({})
const statsErrors = reactive<Record<string, string>>({})
const statsLoading = reactive<Record<string, boolean>>({})

async function fetchStats(dir: string, key: string) {
  if (statsLoading[key]) return
  statsLoading[key] = true
  statsErrors[key] = ''
  try {
    const res = await fetch(`/api/preprocess/adapter-stats?dir=${encodeURIComponent(dir)}`)
    if (!res.ok) {
      let message = `HTTP ${res.status}`
      try { const body = await res.json(); if (typeof body.detail === 'string') message = body.detail } catch { /* Keep HTTP status for non-JSON responses. */ }
      throw new Error(message)
    }
    adapterStats[key] = await res.json()
  } catch (cause) {
    delete adapterStats[key]
    statsErrors[key] = cause instanceof Error ? cause.message : String(cause)
  } finally { statsLoading[key] = false }
}
async function fetchPreview(dir: string): Promise<PreviewImage[]> {
  try {
    const res = await fetch(`/api/images?directory=${encodeURIComponent(dir)}&page=1&page_size=8`)
    if (!res.ok) return []
    const data = await res.json()
    return (data.items || []).map((i: any) => ({ path: i.path, filename: i.filename }))
  } catch {
    return []
  }
}

onMounted(async () => {
  const [ip, ec] = await Promise.all([
    fetchPreview('image_dataset'),
    fetchPreview('easycontrol-dataset'),
  ])
  ipPreviewImages.value = ip
  ecPreviewImages.value = ec
  fetchStats('image_dataset', 'ip')
  fetchStats('easycontrol-dataset', 'ec')
})
</script>

<style scoped>
.preview-thumbs {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.preview-thumb {
  border: 1px solid rgba(var(--v-border-color), 0.3);
}

/* Adapter cards: hover lift */
:deep(.v-card) {
  transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s;
}
:deep(.v-card:hover) {
  border-color: var(--border-default);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  transform: translateY(-1px);
}

/* Version chip: amber accent */
:deep(.v-chip[variant="outlined"]) {
  border-color: rgba(212, 145, 42, 0.4);
  color: var(--forge-amber);
}
</style>
