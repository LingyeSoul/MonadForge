<template>
  <v-card variant="tonal" class="pa-4" style="flex: 0 0 auto;">
    <div class="d-flex align-center mb-3">
      <v-icon icon="mdi-image-multiple" size="small" class="mr-2" />
      <div class="text-subtitle-2">{{ t('dashSamples') }}</div>
      <v-spacer />
      <v-chip size="x-small" variant="outlined">
        {{ t('dashSampleHistoryCount', { n: samples.length }) }}
      </v-chip>
    </div>

    <div v-if="samples.length === 0" class="text-body-2 text-medium-emphasis">
      {{ t('dashWaitingSample', 'Waiting for samples...') }}
    </div>

    <div v-else class="sample-grid">
      <div
        v-for="(s, i) in samples"
        :key="s.path"
        class="sample-card"
        :class="{ 'sample-card-latest': i === samples.length - 1 }"
        @click="open(i)"
        :title="t('dashSampleEnlarge')"
      >
        <img
          :src="fileUrl(s)"
          :alt="s.filename"
          loading="lazy"
          class="sample-thumb"
        />
        <div class="sample-overlay">
          <div class="sample-overlay-text">
            {{ labelFor(s) }}
          </div>
        </div>
        <div class="sample-badge">{{ labelFor(s) }}</div>
      </div>
    </div>

    <!-- Click-to-enlarge dialog -->
    <v-dialog v-model="dialogOpen" max-width="90vw" max-height="90vh">
      <v-card v-if="active" color="surface" class="sample-dialog">
        <v-card-title class="d-flex align-center pa-3 pb-2">
          <v-icon icon="mdi-image" size="small" class="mr-2" />
          <div class="text-subtitle-2">{{ t('dashSampleDialogTitle') }}</div>
          <v-spacer />
          <span class="text-caption text-medium-emphasis">
            {{ labelFor(active) }}
          </span>
          <v-btn
            icon="mdi-close"
            size="small"
            variant="text"
            class="ml-2"
            @click="dialogOpen = false"
          />
        </v-card-title>
        <v-card-text class="pa-3 d-flex justify-center" style="background: rgba(0,0,0,0.4);">
          <img
            :src="fileUrl(active)"
            :alt="active.filename"
            class="sample-dialog-img"
          />
        </v-card-text>
        <v-card-text v-if="active.prompt" class="pa-3">
          <div class="text-caption text-medium-emphasis mb-1">
            {{ t('dashSamplePrompt') }}
          </div>
          <div class="text-body-2 prompt-text">{{ active.prompt }}</div>
        </v-card-text>
      </v-card>
    </v-dialog>
  </v-card>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from '../composables/useI18n'
import type { SampleInfo } from '../stores/training'

const { t } = useI18n()

const props = defineProps<{ samples: SampleInfo[]; taskId: string }>()

const dialogOpen = ref(false)
const activeIndex = ref<number | null>(null)

const active = computed<SampleInfo | null>(() => {
  if (activeIndex.value === null) return null
  return props.samples[activeIndex.value] ?? null
})

function open(i: number) {
  activeIndex.value = i
  dialogOpen.value = true
}

function fileUrl(s: SampleInfo): string {
  // filename is the basename the server uses for its whitelist check.
  return `/api/preview/runs/${encodeURIComponent(props.taskId)}/samples/file?path=${encodeURIComponent(s.filename)}`
}

function labelFor(s: SampleInfo): string {
  if (s.epoch != null && s.step != null) {
    return t('dashSampleStepEpoch', { step: s.step, epoch: s.epoch })
  }
  if (s.step != null) {
    return t('dashSampleStep', { step: s.step })
  }
  if (s.epoch != null) {
    return t('dashSampleEpoch', { epoch: s.epoch })
  }
  return s.filename
}
</script>

<style scoped>
.sample-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 10px;
}

.sample-card {
  position: relative;
  aspect-ratio: 1 / 1;
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  border: 1px solid var(--border-subtle);
  background: var(--bg-surface);
  transition: transform 0.12s ease, border-color 0.12s ease;
}

.sample-card:hover {
  transform: translateY(-2px);
  border-color: var(--forge-amber);
}

.sample-card-latest {
  border-color: var(--forge-amber);
  box-shadow: 0 0 0 1px var(--forge-amber), 0 0 12px rgba(199, 91, 26, 0.18);
}

.sample-thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.sample-badge {
  position: absolute;
  bottom: 4px;
  left: 4px;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 500;
  background: rgba(0, 0, 0, 0.65);
  color: #fff;
  line-height: 1.2;
  pointer-events: none;
}

.sample-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: flex-end;
  padding: 4px;
  background: linear-gradient(
    to top,
    rgba(0, 0, 0, 0.55) 0%,
    rgba(0, 0, 0, 0) 35%
  );
  opacity: 0;
  transition: opacity 0.15s ease;
  pointer-events: none;
}

.sample-card:hover .sample-overlay {
  opacity: 1;
}

.sample-overlay-text {
  color: #fff;
  font-size: 11px;
  line-height: 1.3;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.sample-dialog-img {
  max-width: 100%;
  max-height: 75vh;
  object-fit: contain;
  display: block;
}

.prompt-text {
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.5;
  font-family: var(--font-mono);
}
</style>
