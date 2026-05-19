<template>
  <v-container fluid class="pa-4">
    <div class="text-h5 mb-1">{{ t('mgTitle') }}</div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('mgSubtitle') }}</div>

    <v-card variant="tonal" class="mb-4">
      <v-card-title class="text-subtitle-1">
        <v-icon icon="mdi-call-merge" class="mr-2" />
        {{ t('mgConfig') }}
      </v-card-title>
      <v-card-text>
        <v-row>
          <v-col cols="12" md="6">
            <v-text-field
              v-model="adapterDir"
              :label="t('mgAdapterDir')"
              :hint="t('mgAdapterDirHint')"
              persistent-hint
              variant="outlined"
              density="compact"
            />
          </v-col>
          <v-col cols="12" md="3">
            <v-text-field
              v-model.number="multiplier"
              :label="t('mgMultiplier')"
              :hint="t('mgMultiplierHint')"
              persistent-hint
              type="number"
              step="0.1"
              min="0"
              max="1"
              variant="outlined"
              density="compact"
            />
          </v-col>
          <v-col cols="12" md="3">
            <v-switch
              v-model="allowPartial"
              :label="t('mgAllowPartial')"
              :hint="t('mgPartialHint')"
              persistent-hint
              color="primary"
              density="compact"
            />
          </v-col>
        </v-row>

        <v-alert type="warning" variant="tonal" density="compact" class="mt-2">
          <span v-html="t('mgWarning')" />
        </v-alert>
      </v-card-text>
      <v-card-actions>
        <v-btn
          color="primary"
          prepend-icon="mdi-merge"
          :loading="isRunning('merge')"
          :disabled="!adapterDir"
          @click="runMerge"
        >
          {{ t('mgMergeBtn') }}
        </v-btn>
      </v-card-actions>
    </v-card>

    <v-divider class="my-4" />

    <div class="text-subtitle-1 mb-2">{{ t('mgActiveTasks') }}</div>
    <v-list v-if="mergeTasks.length > 0" density="compact">
      <v-list-item
        v-for="task in mergeTasks"
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
    <div v-else class="text-medium-emphasis text-body-2">{{ t('mgNoTasks') }}</div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useTaskStore } from '../stores/task'
import { useI18n } from '../composables/useI18n'

const taskStore = useTaskStore()
const { t } = useI18n()
taskStore.fetchTasks()

const adapterDir = ref('output/ckpt')
const multiplier = ref(1.0)
const allowPartial = ref(false)

const mergeTasks = computed(() =>
  taskStore.tasks.filter(tp => tp.command === 'merge')
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

async function runMerge() {
  const args = ['--adapter_dir', adapterDir.value]
  if (multiplier.value !== 1.0) {
    args.push('--multiplier', String(multiplier.value))
  }
  if (allowPartial.value) {
    args.push('--allow_partial')
  }
  await taskStore.startTask('merge', args)
}
</script>
