<template>
  <section class="qwen-task-results pa-3" data-testid="qwen-task-results">
    <div class="d-flex align-center mb-2"><span class="text-subtitle-2">{{ t('qwGenerationResults') }}</span><v-spacer /><v-btn size="small" variant="text" data-testid="qwen-results-refresh" :loading="loading" @click="load">{{ t('ppRefresh') }}</v-btn></div>
    <v-alert v-if="error" type="error" variant="tonal" density="compact">{{ error }}</v-alert>
    <SampleGallery v-if="samples.length" :task-id="taskId" :samples="samples" :image-root="`/api/qwen21/tasks/${encodeURIComponent(taskId)}/samples`" />
    <p v-else-if="!error" class="text-body-2 text-medium-emphasis">{{ t('qwAwaitResults') }}</p>
  </section>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useI18n } from '../composables/useI18n'
import type { SampleInfo } from '../stores/training'
import { requestJson, resultImages } from '../utils/qwen21'
import SampleGallery from './SampleGallery.vue'

const props = defineProps<{ taskId: string; state: string }>()
const { t } = useI18n()
const samples = ref<SampleInfo[]>([])
const error = ref('')
const loading = ref(false)
let generation = 0
async function load(): Promise<void> {
  const stamp = ++generation
  const id = props.taskId
  loading.value = true
  error.value = ''
  try {
    const images = resultImages(await requestJson(`/api/qwen21/tasks/${encodeURIComponent(id)}/results`, {}))
    if (stamp !== generation) return
    samples.value = images.map(image => ({
      attempt_id: null, path: image.file, filename: image.file, step: null, epoch: null,
      prompt: image.prompt, ts: null, received_at: Date.now(), caption: `${image.file} · m${image.multiplier} · seed ${image.seed}`,
    }))
  } catch (err) { if (stamp === generation) error.value = err instanceof Error ? err.message : String(err) }
  finally { if (stamp === generation) loading.value = false }
}
watch(() => [props.taskId, props.state], () => {
  generation++
  samples.value = []
  error.value = ''
  loading.value = false
  if (props.state === 'success') void load()
}, { immediate: true })
</script>

<style scoped>
.qwen-task-results { flex: 0 1 auto; max-height: 360px; overflow-y: auto; border-top: 1px solid var(--border-subtle); }
</style>
