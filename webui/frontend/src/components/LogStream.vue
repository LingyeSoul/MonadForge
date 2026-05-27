<template>
  <div ref="logContainer" class="log-stream pa-2" style="flex: 1 1 0; min-height: 0; overflow-y: auto; font-family: var(--font-mono); font-size: 12px; background: var(--bg-deep); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
    <div v-if="!connected && messages.length === 0" class="text-medium-emphasis">
      {{ t('taskConnecting') }}
    </div>
    <div
      v-for="(msg, i) in messages"
      :key="i"
      class="log-line"
      :class="logLevel(msg)"
    >{{ msg }}</div>
  </div>
</template>

<script setup lang="ts">
import { watch, nextTick, ref } from 'vue'
import { useTaskStream } from '../composables/useTask'
import { useI18n } from '../composables/useI18n'

const props = defineProps<{ taskId: string }>()
const emit = defineEmits<{ done: [] }>()
const logContainer = ref<HTMLElement>()
const { t } = useI18n()

const { messages, connected, done } = useTaskStream(() => props.taskId)

watch(messages, async () => {
  await nextTick()
  if (logContainer.value) {
    logContainer.value.scrollTop = logContainer.value.scrollHeight
  }
}, { deep: true })

watch(done, (val) => {
  if (val) emit('done')
})

function logLevel(msg: string): string {
  const upper = msg.toUpperCase()
  if (upper.includes('ERROR') || upper.includes('EXCEPTION') || upper.includes('TRACEBACK')) return 'log-error'
  if (upper.includes('WARN')) return 'log-warn'
  if (upper.includes('INFO')) return 'log-info'
  return ''
}
</script>

<style scoped>
.log-line {
  white-space: pre-wrap;
  word-break: break-all;
  line-height: 1.5;
  color: var(--text-secondary);
  font-size: 12px;
  padding: 1px 0;
  animation: logEntry 0.2s ease-out both;
}

@keyframes logEntry {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}

.log-info { color: var(--info); }
.log-warn { color: var(--warning); }
.log-error { color: var(--error); font-weight: 500; }
</style>
