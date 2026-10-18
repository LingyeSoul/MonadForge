<template>
  <v-divider class="my-4" />
  <div class="d-flex align-center mb-2">
    <h2 class="text-subtitle-1">{{ t('ppActiveTasks') }}</h2>
    <v-spacer />
    <v-btn data-testid="preprocess-tasks-refresh" variant="text" size="small" prepend-icon="mdi-refresh" :loading="refreshing" @click="emit('refresh')">
      {{ t('ppRefresh') }}
    </v-btn>
  </div>
  <v-alert v-if="taskStore.taskListError" type="error" variant="tonal" density="compact">{{ t('taskLoadFailed') }}</v-alert>
  <v-list v-else-if="tasks.length" density="compact" class="preprocess-task-list">
    <v-list-item v-for="task in visibleTasks" :key="task.task_id" :data-task-id="task.task_id"
      :title="task.command" :subtitle="`${t('taskState')}: ${task.state} | PID: ${task.pid ?? '—'}`"
      :to="{ path: '/tasks', query: { task: task.task_id } }">
      <template #append>
        <v-chip size="small" :color="stateColor(task.state)" variant="tonal">{{ task.state }}</v-chip>
        <TaskStopButton v-if="isTaskStoppable(task.state)" :task-id="task.task_id" />
      </template>
    </v-list-item>
  </v-list>
  <div v-else class="text-medium-emphasis text-body-2">{{ t('ppNoTasks') }}</div>
  <v-btn v-if="tasks.length > 4" data-testid="preprocess-tasks-toggle" variant="text" size="small" class="mt-1"
    :prepend-icon="showAll ? 'mdi-chevron-up' : 'mdi-chevron-down'" @click="showAll = !showAll">
    {{ showAll ? t('taskShowLess') : t('taskShowAll', { count: tasks.length }) }}
  </v-btn>
</template>

<script setup lang="ts">
import { isTaskStoppable } from '../utils/taskState.ts'
import { computed, ref } from 'vue'
import TaskStopButton from './TaskStopButton.vue'
import { useI18n } from '../composables/useI18n'
import { useTaskStore, type TaskInfo } from '../stores/task'

const props = defineProps<{ tasks: TaskInfo[]; refreshing?: boolean }>()
const emit = defineEmits<{ refresh: [] }>()
const { t } = useI18n()
const taskStore = useTaskStore()
const showAll = ref(false)
const visibleTasks = computed(() => showAll.value ? props.tasks : props.tasks.slice(0, 4))
function stateColor(state: string): string | undefined {
  if (state === 'paused') return 'warning'
  if (state === 'running' || state === 'stopping') return 'info'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}
</script>

<style scoped>
.preprocess-task-list :deep(.v-list-item__append) { gap: 8px; }
@media (max-width: 599px) {
  .preprocess-task-list :deep(.v-list-item) { padding-inline: 8px; }
  .preprocess-task-list :deep(.v-list-item-title) { white-space: normal; overflow-wrap: anywhere; }
  .preprocess-task-list :deep(.v-list-item__append) { gap: 2px; margin-left: 8px; }
}
</style>
