<template>
  <div class="d-flex flex-column" style="height: 100%">
    <v-toolbar density="compact" color="surface">
      <v-toolbar-title class="text-subtitle-1">{{ t('navTasks') }}</v-toolbar-title>
      <v-spacer />
      <v-btn
        icon="mdi-refresh"
        variant="text"
        size="small"
        @click="taskStore.fetchTasks()"
      />
    </v-toolbar>

    <div class="flex-grow-1 d-flex flex-column overflow-hidden pa-2">
      <div v-if="taskStore.tasks.length === 0" class="text-center text-medium-emphasis pa-8">
        {{ t('taskNoActive') }}
      </div>

      <template v-for="task in taskStore.tasks" :key="task.task_id">
        <v-card
          v-if="selectedTask !== task.task_id"
          class="mb-2"
          variant="tonal"
          :color="stateColor(task.state)"
          :class="{ 'anim-pulse-glow': task.state === 'running' }"
        >
          <v-card-title class="text-body-2 d-flex align-center">
            <v-icon :icon="stateIcon(task.state)" size="small" class="mr-2" />
            {{ task.command }}
            <v-spacer />
            <v-chip size="x-small" variant="outlined">{{ task.state }}</v-chip>
          </v-card-title>
          <v-card-subtitle class="text-caption">
            {{ task.task_id.slice(0, 8) }} &middot; PID {{ task.pid ?? '—' }}
          </v-card-subtitle>
          <v-card-text v-if="task.state === 'running'" class="pt-0">
            <v-progress-linear indeterminate color="primary" height="3" class="anim-progress-stripe" />
            <div class="d-flex justify-end mt-1">
              <v-btn
                size="x-small"
                color="error"
                variant="text"
                @click="taskStore.cancelTask(task.task_id)"
              >
                {{ t('taskCancel') }}
              </v-btn>
            </div>
          </v-card-text>
          <v-card-actions v-if="task.state === 'running' || task.output_lines > 0">
            <v-btn
              size="x-small"
              variant="text"
              @click="selectedTask = task.task_id"
            >
              {{ t('taskShowLogs') }}
            </v-btn>
          </v-card-actions>
        </v-card>

        <v-card
          v-else
          class="mb-2 d-flex flex-column"
          variant="tonal"
          :color="stateColor(task.state)"
          :class="{ 'anim-pulse-glow': task.state === 'running' }"
          style="flex: 1 1 0; min-height: 0;"
        >
          <v-card-title class="text-body-2 d-flex align-center flex-shrink-0">
            <v-icon :icon="stateIcon(task.state)" size="small" class="mr-2" />
            {{ task.command }}
            <v-spacer />
            <v-chip size="x-small" variant="outlined">{{ task.state }}</v-chip>
          </v-card-title>
          <v-card-subtitle class="text-caption flex-shrink-0">
            {{ task.task_id.slice(0, 8) }} &middot; PID {{ task.pid ?? '—' }}
          </v-card-subtitle>
          <v-card-text v-if="task.state === 'running'" class="pt-0 flex-shrink-0">
            <v-progress-linear indeterminate color="primary" height="3" class="anim-progress-stripe" />
            <div class="d-flex justify-end mt-1">
              <v-btn
                size="x-small"
                color="error"
                variant="text"
                @click="taskStore.cancelTask(task.task_id)"
              >
                {{ t('taskCancel') }}
              </v-btn>
            </div>
          </v-card-text>
          <v-card-text class="d-flex flex-column pt-0" style="flex: 1 1 0; min-height: 0;">
            <LogStream :task-id="task.task_id" />
          </v-card-text>
          <v-card-actions class="flex-shrink-0">
            <v-btn
              size="x-small"
              variant="text"
              @click="selectedTask = ''"
            >
              {{ t('taskHideLogs') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useTaskStore } from '../stores/task'
import { useI18n } from '../composables/useI18n'
import LogStream from './LogStream.vue'

const taskStore = useTaskStore()
const { t } = useI18n()
const selectedTask = ref('')

function stateColor(state: string) {
  if (state === 'running') return 'primary'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}

function stateIcon(state: string) {
  if (state === 'running') return 'mdi-progress-clock'
  if (state === 'success') return 'mdi-check-circle'
  if (state === 'failed') return 'mdi-alert-circle'
  if (state === 'cancelled') return 'mdi-cancel'
  return 'mdi-clock-outline'
}
</script>
