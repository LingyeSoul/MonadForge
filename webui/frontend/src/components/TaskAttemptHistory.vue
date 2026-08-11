<template>
  <div v-if="task.attempt_count > 1" class="attempt-history px-4 pb-2">
    <div class="d-flex align-center text-caption text-medium-emphasis">
      <v-icon icon="mdi-source-branch" size="x-small" class="mr-1" />
      <span>{{ t('taskResumedCount', { n: task.attempt_count - 1 }) }}</span>
      <v-spacer />
      <v-tooltip :text="expanded ? t('taskHideAttempts') : t('taskShowAttempts')">
        <template #activator="{ props }">
          <v-btn
            v-bind="props"
            :icon="expanded ? 'mdi-chevron-up' : 'mdi-chevron-down'"
            size="x-small"
            variant="text"
            @click="expanded = !expanded"
          />
        </template>
      </v-tooltip>
    </div>

    <v-expand-transition>
      <div v-if="expanded" class="attempt-list mt-1">
        <div
          v-for="attempt in task.attempts"
          :key="attempt.job_id"
          class="attempt-row py-2"
        >
          <div class="d-flex align-center ga-2">
            <span class="text-caption font-weight-medium">
              {{ t('taskAttemptLabel', { n: attempt.attempt_index + 1 }) }}
            </span>
            <v-chip size="x-small" variant="outlined">{{ attempt.state }}</v-chip>
            <span class="text-caption text-medium-emphasis text-truncate">
              {{ attempt.job_id.slice(0, 8) }}
            </span>
          </div>
          <div class="attempt-meta text-caption text-medium-emphasis mt-1">
            <span>{{ formatTime(attempt.started_at) }}</span>
            <span v-if="attempt.recovery_step != null">
              {{ t('taskAttemptResumedFrom', { step: attempt.recovery_step }) }}
            </span>
            <span v-if="attempt.exit_code != null">
              {{ t('taskAttemptExitCode', { code: attempt.exit_code }) }}
            </span>
          </div>
          <div v-if="attempt.terminal_reason" class="text-caption text-medium-emphasis mt-1 text-break">
            {{ attempt.terminal_reason }}
          </div>
        </div>
      </div>
    </v-expand-transition>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from '../composables/useI18n'
import type { TaskInfo } from '../stores/task'

defineProps<{ task: TaskInfo }>()

const { t } = useI18n()
const expanded = ref(false)

function formatTime(value: string | null): string {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString()
}
</script>

<style scoped>
.attempt-history {
  min-width: 0;
}

.attempt-list {
  border-top: 1px solid rgba(var(--v-border-color), var(--v-border-opacity));
}

.attempt-row + .attempt-row {
  border-top: 1px solid rgba(var(--v-border-color), calc(var(--v-border-opacity) * 0.7));
}

.attempt-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 12px;
}
</style>
