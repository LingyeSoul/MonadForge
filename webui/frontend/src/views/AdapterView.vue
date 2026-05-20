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
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('exp-ip-adapter')" @click="runTask('exp-ip-adapter')">
              {{ t('adTrainIp') }}
            </v-btn>
            <v-btn variant="text" :loading="isRunning('exp-ip-adapter-preprocess')" @click="runTask('exp-ip-adapter-preprocess')">
              {{ t('adPreprocess') }}
            </v-btn>
          </v-card-actions>
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
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" :loading="isRunning('exp-easycontrol')" @click="runTask('exp-easycontrol')">
              {{ t('adTrainEasy') }}
            </v-btn>
            <v-btn variant="text" :loading="isRunning('exp-easycontrol-preprocess')" @click="runTask('exp-easycontrol-preprocess')">
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
          <v-btn v-if="task.state === 'running'" icon="mdi-stop" size="small" variant="text" color="error" @click="taskStore.cancelTask(task.task_id)" />
        </template>
      </v-list-item>
    </v-list>
    <div v-else class="text-medium-emphasis text-body-2">{{ t('adNoTasks') }}</div>
  </v-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useTaskStore } from '../stores/task'
import { useNotifyStore } from '../stores/notify'
import { useI18n } from '../composables/useI18n'

const taskStore = useTaskStore()
const notify = useNotifyStore()
const { t } = useI18n()
taskStore.fetchTasks()

const adapterCommands = [
  'exp-ip-adapter', 'exp-ip-adapter-preprocess',
  'exp-easycontrol', 'exp-easycontrol-preprocess',
]

const adapterTasks = computed(() =>
  taskStore.tasks.filter(tp => adapterCommands.includes(tp.command))
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
  const taskId = await taskStore.startTask(command)
  if (taskId) {
    notify.show(t('notifyTaskStarted', { command }), 'success')
  } else {
    notify.show(t('notifyTaskStartFailed', { command }), 'error')
  }
}
</script>
