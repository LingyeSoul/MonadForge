<template>
  <div ref="logContainer" class="log-stream pa-2" style="flex: 1 1 0; min-height: 0; overflow-y: auto; font-family: var(--font-mono); font-size: 12px; background: var(--bg-deep); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
    <div v-if="truncatedNote" class="log-truncate-note">
      {{ t('taskLogTruncated', { total: historyTotal, kept: lines.length }) }}
    </div>
    <div v-if="!connected && lines.length === 0" class="text-medium-emphasis">
      {{ t('taskConnecting') }}
    </div>
    <div
      v-for="ln in lines"
      :key="ln.seq"
      class="log-line"
      :class="ln.level"
    >{{ ln.text }}</div>
  </div>
</template>

<script setup lang="ts">
import { computed, watch, nextTick, ref, onUnmounted } from 'vue'
import { useTaskStream, type LogStreamHandle } from '../composables/useTask'
import { useI18n } from '../composables/useI18n'

const props = defineProps<{
  taskId: string
  /** Existing stream handle (e.g. the dashboard's useTrainingStream) —
   * when provided, no second WebSocket is opened for the same task. */
  external?: LogStreamHandle | null
}>()
const emit = defineEmits<{ done: [] }>()
const logContainer = ref<HTMLElement>()
const { t } = useI18n()

const own = props.external ? null : useTaskStream(() => props.taskId)

const stream = computed<LogStreamHandle | null>(() => props.external ?? own)
const lines = computed(() => stream.value?.lines.value ?? [])
const connected = computed(() => stream.value?.connected.value ?? false)
const historyTotal = computed(() => stream.value?.historyTotal.value ?? 0)
const truncatedNote = computed(() => historyTotal.value > lines.value.length && lines.value.length > 0)

// Autoscroll per coalesced flush — but only while the user is pinned to the
// bottom, so reading up through a busy log isn't a losing fight with the
// scroll position.
watch(() => stream.value?.activity.value ?? 0, async () => {
  await nextTick()
  const el = logContainer.value
  if (!el) return
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48
  if (nearBottom) el.scrollTop = el.scrollHeight
}, { immediate: true })

watch(() => stream.value?.done.value ?? false, (val) => {
  if (val) emit('done')
})

onUnmounted(() => own?.disconnect())
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

.log-truncate-note {
  padding: 4px 6px;
  margin-bottom: 4px;
  font-size: 11px;
  color: var(--text-muted);
  border-bottom: 1px dashed var(--border-subtle);
}

.log-info { color: var(--info); }
.log-warn { color: var(--warning); }
.log-error { color: var(--error); font-weight: 500; }
</style>
