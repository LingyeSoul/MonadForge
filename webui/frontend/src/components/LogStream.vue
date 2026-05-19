<template>
  <div ref="logContainer" class="log-stream pa-2" style="max-height: 300px; overflow-y: auto; font-family: monospace; font-size: 12px; background: rgba(0,0,0,0.3); border-radius: 4px;">
    <div v-if="!connected && messages.length === 0" class="text-medium-emphasis">
      {{ t('taskConnecting') }}
    </div>
    <div v-for="(msg, i) in messages" :key="i" class="log-line">{{ msg }}</div>
  </div>
</template>

<script setup lang="ts">
import { watch, nextTick, ref } from 'vue'
import { useTaskStream } from '../composables/useTask'
import { useI18n } from '../composables/useI18n'

const props = defineProps<{ taskId: string }>()
const logContainer = ref<HTMLElement>()
const { t } = useI18n()

const { messages, connected, connect } = useTaskStream(props.taskId)
connect()

watch(messages, async () => {
  await nextTick()
  if (logContainer.value) {
    logContainer.value.scrollTop = logContainer.value.scrollHeight
  }
}, { deep: true })
</script>

<style scoped>
.log-line {
  white-space: pre-wrap;
  word-break: break-all;
  line-height: 1.4;
  color: rgba(255, 255, 255, 0.87);
}
</style>
