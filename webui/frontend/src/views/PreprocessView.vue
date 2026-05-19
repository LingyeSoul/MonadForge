<template>
  <v-container fluid class="pa-4">
    <div class="text-h5 mb-1">{{ t('ppTitle') }}</div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('ppSubtitle') }}</div>

    <v-row>
      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-resize" class="mr-2" />
            {{ t('ppResize') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppResizeDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('preprocess-resize')" @click="runTask('preprocess-resize')">
              {{ t('ppRunResize') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-cached" class="mr-2" />
            {{ t('ppCacheVae') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppCacheVaeDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('preprocess-vae')" @click="runTask('preprocess-vae')">
              {{ t('ppRunVae') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-text-box-outline" class="mr-2" />
            {{ t('ppCacheTe') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppCacheTeDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('preprocess-te')" @click="runTask('preprocess-te')">
              {{ t('ppRunTe') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-eye-outline" class="mr-2" />
            {{ t('ppCachePe') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppCachePeDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('preprocess-pe')" @click="runTask('preprocess-pe')">
              {{ t('ppRunPe') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-image-filter-center-focus" class="mr-2" />
            {{ t('ppMask') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppMaskDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('mask')" @click="runTask('mask')">
              {{ t('ppRunMask') }}
            </v-btn>
            <v-btn color="error" variant="text" size="small" :loading="isRunning('mask-clean')" @click="runTask('mask-clean')">
              {{ t('ppClean') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card variant="tonal" color="primary">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-pipe" class="mr-2" />
            {{ t('ppPipeline') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-2" v-html="t('ppPipelineDesc')" />
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('preprocess')" @click="runTask('preprocess')">
              {{ t('ppRunAll') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>

    <v-divider class="my-4" />

    <div class="d-flex align-center mb-2">
      <div class="text-subtitle-1">{{ t('ppActiveTasks') }}</div>
      <v-spacer />
      <v-btn variant="text" size="small" prepend-icon="mdi-refresh" @click="taskStore.fetchTasks()">
        {{ t('ppRefresh') }}
      </v-btn>
    </div>

    <v-list v-if="preprocessTasks.length > 0" density="compact">
      <v-list-item
        v-for="task in preprocessTasks"
        :key="task.task_id"
        :title="task.command"
        :subtitle="`${t('taskState')}: ${task.state} | PID: ${task.pid ?? '—'}`"
      >
        <template #append>
          <v-chip size="small" :color="stateColor(task.state)" variant="tonal">{{ task.state }}</v-chip>
          <v-btn v-if="task.state === 'running'" icon="mdi-stop" size="small" variant="text" color="error" @click="taskStore.cancelTask(task.task_id)" />
        </template>
      </v-list-item>
    </v-list>
    <div v-else class="text-medium-emphasis text-body-2">{{ t('ppNoTasks') }}</div>
  </v-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useTaskStore } from '../stores/task'
import { useI18n } from '../composables/useI18n'

const taskStore = useTaskStore()
const { t } = useI18n()
taskStore.fetchTasks()

const preprocessCommands = [
  'preprocess', 'preprocess-resize', 'preprocess-vae', 'preprocess-te',
  'preprocess-pe', 'mask', 'mask-clean',
]

const preprocessTasks = computed(() =>
  taskStore.tasks.filter(tp => preprocessCommands.includes(tp.command))
)

function isRunning(command: string) {
  return taskStore.tasks.some(tp => tp.command === command && tp.state === 'running')
}

function stateColor(state: string) {
  if (state === 'running') return 'info'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}

async function runTask(command: string) {
  await taskStore.startTask(command)
}
</script>
