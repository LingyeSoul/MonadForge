<template>
  <v-container fluid class="task-monitor pa-4 d-flex flex-column" style="flex: 1 1 0; min-height: 0; overflow: hidden;">
    <div class="task-toolbar d-flex align-center mb-1">
      <div class="text-h5">{{ t('taskTitle') }}</div>
      <v-spacer />
      <div class="task-toolbar-actions d-flex align-center">
        <v-btn
          variant="text"
          size="small"
          :disabled="!taskStore.daemonUp"
          :prepend-icon="taskStore.daemonPaused ? 'mdi-play' : 'mdi-pause'"
          :color="taskStore.daemonPaused ? 'success' : undefined"
          @click="taskStore.daemonPaused ? taskStore.resumeQueue() : taskStore.pauseQueue()"
        >
          {{ taskStore.daemonPaused ? t('taskResumeQueue') : t('taskPauseQueue') }}
        </v-btn>
        <v-btn
          v-if="!isWindows"
          variant="text"
          size="small"
          color="error"
          :disabled="!taskStore.daemonUp"
          prepend-icon="mdi-power"
          @click="showShutdownDlg = true"
        >
          {{ t('taskShutdownDaemon') }}
        </v-btn>
        <v-btn variant="text" size="small" prepend-icon="mdi-refresh" @click="refreshView">
          {{ t('ppRefresh') }}
        </v-btn>
      </div>
    </div>
    <v-btn-toggle v-model="taskFilter" density="compact" mandatory divided class="task-filter mb-3 align-self-start flex-shrink-0">
      <v-btn value="all" size="small">{{ t('taskFilterAll') }}</v-btn>
      <v-btn value="active" size="small">{{ t('taskFilterActive') }}</v-btn>
      <v-btn value="success" size="small">{{ t('taskFilterSuccess') }}</v-btn>
      <v-btn value="failed" size="small">{{ t('taskFilterFailed') }}</v-btn>
      <v-btn value="cancelled" size="small">{{ t('taskFilterCancelled') }}</v-btn>
    </v-btn-toggle>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('taskSubtitle') }}</div>

    <v-alert
      v-if="historyError"
      type="error"
      variant="tonal"
      density="compact"
      class="mb-3"
    >
      {{ t('taskLoadFailed') }}
    </v-alert>

    <div v-if="visibleTasks.length === 0" class="d-flex align-center justify-center flex-grow-1">
      <div class="text-center text-medium-emphasis">
        <v-icon icon="mdi-candle" size="48" class="mb-2 ember-icon" />
        <div>{{ taskFilter === 'all' ? t('taskNoHistory') : t('taskNoMatches') }}</div>
      </div>
    </div>

    <div
      v-else
      class="d-flex flex-column flex-grow-1"
      :style="selectedTask ? 'min-height: 0; overflow: hidden;' : 'min-height: 0; overflow-y: auto;'"
    >
      <template v-for="task in displayedTasks" :key="task.task_id">
        <!-- Compact card for non-selected tasks -->
        <v-card
          v-if="selectedTask !== task.task_id"
          class="mb-2 flex-shrink-0"
          :class="{ 'task-complete': task.state === 'success', 'task-failed': task.state === 'failed', 'task-running': task.state === 'running' || task.state === 'stopping' }"
          variant="tonal"
          :color="stateColor(task.state)"
        >
          <v-card-title class="text-body-2 d-flex align-center">
            <v-icon :icon="stateIcon(task.state)" size="small" class="mr-2" />
            <span class="text-truncate">{{ task.command }}</span>
            <v-spacer />
            <v-chip
              v-if="task.state === 'pending' && task.queue_position != null"
              size="x-small"
              variant="tonal"
              color="info"
              class="ml-1"
            >
              #{{ task.queue_position }}
            </v-chip>
            <v-chip size="x-small" variant="outlined" class="ml-2">{{ task.state }}</v-chip>
            <v-chip v-if="task.legacy" size="x-small" color="warning" variant="tonal" class="ml-1">
              {{ t('taskLegacy') }}
            </v-chip>
          </v-card-title>
          <v-card-subtitle class="task-card-subtitle text-caption">
            {{ task.task_id.slice(0, 8) }} &middot; PID {{ task.pid ?? '—' }}
            <span v-if="task.recovery_step != null"> &middot; step {{ task.recovery_step }}</span>
            <span v-if="task.last_progress?.step != null"> &middot; {{ t('taskLastProgress') }} {{ task.last_progress.step }}<span v-if="task.last_progress.total_steps != null">/{{ task.last_progress.total_steps }}</span></span>
          </v-card-subtitle>
          <v-card-text v-if="task.terminal_reason" class="text-caption pt-0 text-medium-emphasis">
            {{ t('taskTerminalReason') }}: {{ task.terminal_reason }}
          </v-card-text>
          <v-card-text v-if="task.legacy" class="text-caption pt-0 text-warning">
            {{ t('taskLegacyDescription') }}
          </v-card-text>
          <TaskAttemptHistory :task="task" />
          <v-card-text v-if="task.state === 'running' || task.state === 'stopping'" class="pt-0">
            <v-progress-linear indeterminate color="primary" height="2" />
          </v-card-text>
          <v-card-actions class="task-card-actions">
            <v-btn
              v-if="task.resumable && (task.state === 'failed' || task.state === 'cancelled')"
              size="x-small"
              color="primary"
              variant="text"
              prepend-icon="mdi-play-circle"
              @click="resumeTask(task)"
            >{{ t('taskResume') }}</v-btn>
            <v-btn
              v-if="task.state === 'success' || task.state === 'failed' || task.state === 'cancelled'"
              size="x-small"
              variant="text"
              prepend-icon="mdi-delete-outline"
              @click="deleteHistory(task)"
            >{{ t('taskDeleteHistory') }}</v-btn>
            <v-btn
              v-if="task.state === 'running' || task.state === 'pending'"
              size="x-small"
              color="error"
              variant="text"
              @click="cancelTask(task)"
            >
              {{ t('taskCancel') }}
            </v-btn>
            <v-spacer class="task-action-spacer" />
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
          :class="{ 'task-complete': task.state === 'success', 'task-failed': task.state === 'failed', 'task-running': task.state === 'running' || task.state === 'stopping' }"
          variant="tonal"
          :color="stateColor(task.state)"
          style="flex: 1 1 0; min-height: 0;"
        >
          <v-card-title class="text-body-2 d-flex align-center flex-shrink-0 py-2">
            <v-icon :icon="stateIcon(task.state)" size="small" class="mr-2" />
            <span class="text-truncate">{{ task.command }}</span>
            <v-spacer />
            <v-chip
              v-if="task.state === 'pending' && task.queue_position != null"
              size="x-small"
              variant="tonal"
              color="info"
              class="ml-1"
            >
              #{{ task.queue_position }}
            </v-chip>
            <v-chip size="x-small" variant="outlined" class="ml-2">{{ task.state }}</v-chip>
            <v-chip v-if="task.legacy" size="x-small" color="warning" variant="tonal" class="ml-1">
              {{ t('taskLegacy') }}
            </v-chip>
          </v-card-title>
          <v-card-subtitle class="task-card-subtitle text-caption flex-shrink-0">
            {{ task.task_id.slice(0, 8) }} &middot; PID {{ task.pid ?? '—' }}
            <span v-if="task.recovery_step != null"> &middot; step {{ task.recovery_step }}</span>
            <span v-if="task.last_progress?.step != null"> &middot; {{ t('taskLastProgress') }} {{ task.last_progress.step }}<span v-if="task.last_progress.total_steps != null">/{{ task.last_progress.total_steps }}</span></span>
          </v-card-subtitle>
          <div v-if="task.terminal_reason" class="text-caption text-medium-emphasis px-2">
            {{ t('taskTerminalReason') }}: {{ task.terminal_reason }}
          </div>
          <div v-if="task.legacy" class="text-caption text-warning px-2">
            {{ t('taskLegacyDescription') }}
          </div>
          <TaskAttemptHistory :task="task" />
          <v-progress-linear v-if="task.state === 'running' || task.state === 'stopping'" indeterminate color="primary" height="2" class="flex-shrink-0" />
          <v-card-text class="d-flex flex-column pa-2" style="flex: 1 1 0; min-height: 0;">
            <LogStream :task-id="task.task_id" @done="onTaskDone" />
          </v-card-text>
          <v-card-actions class="task-card-actions flex-shrink-0 py-1">
            <v-btn
              v-if="task.resumable && (task.state === 'failed' || task.state === 'cancelled')"
              size="x-small"
              color="primary"
              variant="text"
              prepend-icon="mdi-play-circle"
              @click="resumeTask(task)"
            >{{ t('taskResume') }}</v-btn>
            <v-btn
              v-if="task.state === 'success' || task.state === 'failed' || task.state === 'cancelled'"
              size="x-small"
              variant="text"
              prepend-icon="mdi-delete-outline"
              @click="deleteHistory(task)"
            >{{ t('taskDeleteHistory') }}</v-btn>
            <v-btn
              v-if="task.state === 'running' || task.state === 'pending'"
              size="x-small"
              color="error"
              variant="text"
              @click="cancelTask(task)"
            >
              {{ t('taskCancel') }}
            </v-btn>
            <v-spacer class="task-action-spacer" />
            <v-btn size="x-small" variant="text" @click="selectedTask = ''">
              {{ t('taskHideLogs') }}
            </v-btn>
          </v-card-actions>
        </v-card>
      </template>
    </div>

    <v-pagination
      v-if="historyTotal > historyPageSize"
      v-model="historyPage"
      :length="Math.ceil(historyTotal / historyPageSize)"
      density="compact"
      class="flex-shrink-0 mt-2"
      @update:model-value="loadPage"
    />

    <!-- Shutdown daemon confirmation dialog -->
    <v-dialog v-model="showShutdownDlg" max-width="450">
      <v-card>
        <v-card-title class="d-flex align-center">
          <v-icon icon="mdi-power" color="error" class="mr-2" />
          {{ t('taskShutdownDaemon') }}
        </v-card-title>
        <v-card-text>
          <div class="text-body-2 mb-3">{{ t('taskShutdownDaemonDesc') }}</div>
          <v-select
            v-model="shutdownMode"
            :items="shutdownModes"
            item-title="title"
            item-value="value"
            :label="t('taskShutdownMode')"
            density="compact"
            hide-details
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" :disabled="shutdownLoading" @click="showShutdownDlg = false">
            {{ t('dsCancel') }}
          </v-btn>
          <v-btn
            color="error"
            :loading="shutdownLoading"
            @click="confirmShutdown"
          >
            {{ t('taskShutdownConfirm') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { computed, ref, watch, onMounted, onUnmounted } from 'vue'
import { useTaskStore, mergeTaskList, type TaskFilter, type TaskInfo } from '../stores/task'
import { useI18n } from '../composables/useI18n'
import LogStream from '../components/LogStream.vue'
import TaskAttemptHistory from '../components/TaskAttemptHistory.vue'

const taskStore = useTaskStore()
const { t } = useI18n()
const selectedTask = ref('')
const taskFilter = ref<TaskFilter>('all')
const historyTasks = ref<TaskInfo[]>([])
const historyTotal = ref(0)
const historyPage = ref(1)
const historyPageSize = 25
const historyError = ref(false)
let historyRequestId = 0
const visibleTasks = computed(() => historyTasks.value)
const displayedTasks = computed(() => {
  if (!selectedTask.value) return visibleTasks.value
  return visibleTasks.value.filter((task) => task.task_id === selectedTask.value)
})

async function loadPage() {
  const requestId = ++historyRequestId
  try {
    const result = await taskStore.fetchTaskPage({
      state: taskFilter.value,
      page: historyPage.value,
      pageSize: historyPageSize,
    })
    if (requestId !== historyRequestId) return
    historyTasks.value = mergeTaskList(historyTasks.value, result.tasks)
    historyTotal.value = result.total
    historyError.value = false
  } catch {
    if (requestId !== historyRequestId) return
    historyTasks.value = []
    historyTotal.value = 0
    historyError.value = true
  }
}

watch(taskFilter, () => {
  historyPage.value = 1
  void loadPage()
})

// On Windows the daemon hosts the WebUI as a sidecar and the system-tray app is
// the proper "complete exit" entry point (its Quit already shuts the daemon
// down). So the in-WebUI "Stop Daemon" button only makes sense on non-Windows
// setups where there is no tray.
const isWindows = navigator.userAgent.includes('Windows')

// ── Shutdown daemon dialog ─────────────────────────────────────
const showShutdownDlg = ref(false)
const shutdownMode = ref<'detach' | 'cooperative-stop' | 'force'>('cooperative-stop')
const shutdownModes = computed(() => [
  { title: t('taskShutdownDetach'), value: 'detach' },
  { title: t('taskShutdownCooperative'), value: 'cooperative-stop' },
  { title: t('taskShutdownForce'), value: 'force' },
])
const shutdownLoading = ref(false)

async function confirmShutdown() {
  shutdownLoading.value = true
  try {
    await taskStore.shutdownDaemon(shutdownMode.value)
    showShutdownDlg.value = false
  } finally {
    shutdownLoading.value = false
  }
}

let pollTimer = 0

async function refreshView() {
  await taskStore.poll()
  await loadPage()
}

onMounted(() => {
  void refreshView()
  pollTimer = window.setInterval(() => void refreshView(), 5000)
})

onUnmounted(() => {
  clearInterval(pollTimer)
})

async function resumeTask(task: TaskInfo) {
  if (await taskStore.resumeTask(task.task_id, task)) await loadPage()
}

async function deleteHistory(task: TaskInfo) {
  if (!await taskStore.deleteHistory(task.task_id)) return
  const maxPage = Math.max(1, Math.ceil(Math.max(0, historyTotal.value - 1) / historyPageSize))
  historyPage.value = Math.min(historyPage.value, maxPage)
  await loadPage()
}

async function cancelTask(task: TaskInfo) {
  if (await taskStore.cancelTask(task.task_id)) await loadPage()
}

function onTaskDone() {
  void refreshView()
}

function stateColor(state: string) {
  if (state === 'running' || state === 'stopping') return 'primary'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}

function stateIcon(state: string) {
  if (state === 'running' || state === 'stopping') return 'mdi-progress-clock'
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

@media (max-width: 600px) {
  .task-monitor {
    padding: 8px !important;
  }

  .task-toolbar {
    align-items: stretch !important;
    flex-direction: column;
    gap: 4px;
  }

  .task-toolbar > .v-spacer {
    display: none;
  }

  .task-toolbar-actions {
    flex-wrap: wrap;
    gap: 2px 4px;
  }

  .task-toolbar-actions :deep(.v-btn) {
    margin-inline-start: 0 !important;
  }

  .task-filter {
    align-self: stretch !important;
    max-width: 100%;
    min-height: 36px;
    overflow-x: auto;
    overflow-y: hidden;
  }

  .task-card-subtitle {
    overflow: visible;
    text-overflow: clip;
    white-space: normal;
  }

  .task-card-actions {
    flex-wrap: wrap;
    gap: 2px 4px;
    justify-content: flex-start;
  }

  .task-card-actions :deep(.v-btn) {
    margin-inline-start: 0 !important;
  }

  .task-action-spacer {
    display: none;
  }
}
</style>
