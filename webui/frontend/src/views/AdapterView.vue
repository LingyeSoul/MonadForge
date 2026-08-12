<template>
  <v-container fluid class="pa-4">
    <div class="text-h5 mb-1">{{ t('adTitle') }}</div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('adSubtitle') }}</div>

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
            <!-- Dataset stats -->
            <v-row dense class="mt-2" v-if="adapterStats['ip']">
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
            <!-- Dataset stats -->
            <v-row dense class="mt-2" v-if="adapterStats['ec']">
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

    <div class="text-subtitle-1 mb-2">{{ t('adActiveTasks') }}</div>
    <v-list v-if="adapterTasks.length > 0" density="compact">
      <v-list-item
        v-for="task in adapterTasks"
        :key="task.task_id"
        :title="task.command"
        :subtitle="`${t('taskState')}: ${task.state} | PID: ${task.pid ?? '—'}`"
      >
        <template #append>
          <v-chip size="small" :color="stateColor(task.state)" variant="tonal">{{ task.state }}</v-chip>
          <v-btn v-if="task.state === 'running' || task.state === 'pending'" icon="mdi-stop" size="small" variant="text" color="error" @click="taskStore.cancelTask(task.task_id)" />
        </template>
      </v-list-item>
    </v-list>
    <div v-else class="text-medium-emphasis text-body-2">{{ t('adNoTasks') }}</div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useI18n } from '../composables/useI18n'

const taskStore = useTaskStore()
const notify = useNotifyStore()
const { t } = useI18n()
taskStore.fetchTasks()

const adapterCommands = [
  'easycontrol', 'easycontrol-preprocess',
]

const adapterTasks = computed(() =>
  taskStore.tasks.filter(tp => adapterCommands.includes(tp.command))
)

function isRunning(command: string) {
  return taskStore.tasks.some(tp => tp.command === command && tp.state === 'running')
}

function stateColor(state: string) {
  if (state === 'running' || state === 'stopping') return 'info'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}

async function runTask(command: string) {
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

async function fetchStats(dir: string, key: string) {
  try {
    const res = await fetch(`/api/preprocess/adapter-stats?dir=${encodeURIComponent(dir)}`)
    if (res.ok) adapterStats[key] = await res.json()
  } catch { /* silent */ }
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
