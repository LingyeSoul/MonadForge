<template>
  <v-container fluid class="pa-4 d-flex flex-column" style="flex: 1 1 0; min-height: 0; overflow: hidden;">
    <div class="d-flex align-center mb-1">
      <div class="text-h5">{{ t('taskTitle') }}</div>
      <v-spacer />
      <v-btn variant="text" size="small" prepend-icon="mdi-refresh" @click="taskStore.fetchTasks()">
        {{ t('ppRefresh') }}
      </v-btn>
    </div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('taskSubtitle') }}</div>

    <div v-if="taskStore.tasks.length === 0" class="d-flex align-center justify-center flex-grow-1">
      <div class="text-center text-medium-emphasis">
        <v-icon icon="mdi-candle" size="48" class="mb-2 ember-icon" />
        <div>{{ t('taskNoActive') }}</div>
      </div>
    </div>

    <div v-else class="d-flex flex-column flex-grow-1" style="min-height: 0;">
      <template v-for="task in taskStore.tasks" :key="task.task_id">
        <!-- Compact card for non-selected tasks -->
        <v-card
          v-if="selectedTask !== task.task_id"
          class="mb-2"
          :class="{ 'task-complete': task.state === 'success', 'task-failed': task.state === 'failed', 'task-running': task.state === 'running' }"
          variant="tonal"
          :color="stateColor(task.state)"
        >
          <v-card-title class="text-body-2 d-flex align-center">
            <v-icon :icon="stateIcon(task.state)" size="small" class="mr-2" />
            <span class="text-truncate">{{ task.command }}</span>
            <v-spacer />
            <v-chip size="x-small" variant="outlined" class="ml-2">{{ task.state }}</v-chip>
          </v-card-title>
          <v-card-subtitle class="text-caption">
            {{ task.task_id.slice(0, 8) }} &middot; PID {{ task.pid ?? '—' }}
          </v-card-subtitle>
          <v-card-text v-if="task.state === 'running'" class="pt-0">
            <v-progress-linear indeterminate color="primary" height="2" />
          </v-card-text>
          <v-card-actions>
            <v-btn
              v-if="task.state === 'running'"
              size="x-small"
              color="error"
              variant="text"
              @click="taskStore.cancelTask(task.task_id)"
            >
              {{ t('taskCancel') }}
            </v-btn>
            <v-spacer />
            <v-btn
              v-if="task.state === 'running' || task.output_lines > 0"
              size="x-small"
              variant="text"
              @click="selectedTask = task.task_id"
            >
              {{ t('taskShowLogs') }}
            </v-btn>
          </v-card-actions>
        </v-card>

        <!-- Expanded card with full-height log stream -->
        <v-card
          v-else
          class="mb-2 d-flex flex-column"
          :class="{ 'task-complete': task.state === 'success', 'task-failed': task.state === 'failed', 'task-running': task.state === 'running' }"
          variant="tonal"
          :color="stateColor(task.state)"
          style="flex: 1 1 0; min-height: 0;"
        >
          <v-card-title class="text-body-2 d-flex align-center flex-shrink-0 py-2">
            <v-icon :icon="stateIcon(task.state)" size="small" class="mr-2" />
            <span class="text-truncate">{{ task.command }}</span>
            <v-spacer />
            <v-chip size="x-small" variant="outlined" class="ml-2">{{ task.state }}</v-chip>
          </v-card-title>
          <v-card-subtitle class="text-caption flex-shrink-0">
            {{ task.task_id.slice(0, 8) }} &middot; PID {{ task.pid ?? '—' }}
          </v-card-subtitle>
          <v-progress-linear v-if="task.state === 'running'" indeterminate color="primary" height="2" class="flex-shrink-0" />
          <v-card-text class="d-flex flex-column pa-2" style="flex: 1 1 0; min-height: 0;">
            <LogStream :task-id="task.task_id" @done="onTaskDone" />
          </v-card-text>
          <v-card-actions class="flex-shrink-0 py-1">
            <v-btn
              v-if="task.state === 'running'"
              size="x-small"
              color="error"
              variant="text"
              @click="taskStore.cancelTask(task.task_id)"
            >
              {{ t('taskCancel') }}
            </v-btn>
            <v-spacer />
            <v-btn size="x-small" variant="text" @click="selectedTask = ''">
              {{ t('taskHideLogs') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </template>
    </div>
  </v-container>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useTaskStore } from '../stores/task'
import { useI18n } from '../composables/useI18n'
import LogStream from '../components/LogStream.vue'

const taskStore = useTaskStore()
const { t } = useI18n()
const selectedTask = ref('')

let pollTimer = 0

onMounted(() => {
  taskStore.fetchTasks()
  pollTimer = window.setInterval(() => taskStore.fetchTasks(), 5000)
})

onUnmounted(() => {
  clearInterval(pollTimer)
})

function onTaskDone() {
  taskStore.fetchTasks()
}

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

<style scoped>
/* Task status left-border accents */
.task-complete {
  border-left: 3px solid var(--success) !important;
}
.task-failed {
  border-left: 3px solid var(--error) !important;
}
.task-running {
  border-left: 3px solid var(--forge-ember) !important;
}

/* Card hover */
:deep(.v-card) {
  transition: border-color 0.2s, box-shadow 0.2s;
}

/* Empty state ember icon */
.ember-icon {
  color: var(--forge-ember);
  animation: ember-glow 2.5s ease-in-out infinite;
}

@keyframes ember-glow {
  0%, 100% { opacity: 0.6; }
  50%      { opacity: 1; }
}
</style>
