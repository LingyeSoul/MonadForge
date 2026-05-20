<template>
  <v-dialog v-model="visible" max-width="900" scrollable>
    <v-card>
      <v-card-title class="d-flex align-center">
        <v-icon icon="mdi-book-open-page-variant-outline" class="mr-2" />
        {{ t('guidebook') }}
        <v-spacer />
        <v-btn icon="mdi-close" variant="text" size="small" @click="visible = false" />
      </v-card-title>

      <v-divider />

      <v-card-text class="guidebook-content pa-4" style="max-height: 70vh">
        <div v-if="loading" class="text-center pa-8">
          <v-progress-circular indeterminate color="primary" />
        </div>
        <div v-else-if="error" class="text-center pa-8 text-error">
          {{ error }}
        </div>
        <div v-else class="markdown-body" v-html="renderedContent" />
      </v-card-text>

      <v-divider />

      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="visible = false">
          {{ t('guidebookClose') }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { marked } from 'marked'
import { useI18n } from '../composables/useI18n'
import { useAppStore } from '../stores/app'

const { t } = useI18n()
const appStore = useAppStore()

const visible = defineModel<boolean>({ default: false })
const content = ref('')
const loading = ref(false)
const error = ref('')

const renderedContent = computed(() => {
  if (!content.value) return ''
  return marked(content.value, { breaks: true }) as string
})

async function fetchGuidebook() {
  loading.value = true
  error.value = ''
  try {
    const lang = appStore.language
    const res = await fetch(`/api/docs/guidebook?lang=${encodeURIComponent(lang)}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    content.value = data.content || ''
  } catch (e: any) {
    error.value = String(e)
  } finally {
    loading.value = false
  }
}

watch(visible, (val) => {
  if (val) fetchGuidebook()
})
</script>

<style scoped>
.markdown-body {
  font-size: 0.9rem;
  line-height: 1.7;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3) {
  margin-top: 1.2em;
  margin-bottom: 0.5em;
  font-weight: 600;
}

.markdown-body :deep(h1) { font-size: 1.5em; }
.markdown-body :deep(h2) { font-size: 1.3em; }
.markdown-body :deep(h3) { font-size: 1.1em; }

.markdown-body :deep(p) {
  margin-bottom: 0.75em;
}

.markdown-body :deep(code) {
  background: rgba(var(--v-theme-on-surface), 0.08);
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 0.85em;
  font-family: monospace;
}

.markdown-body :deep(pre) {
  background: rgba(var(--v-theme-on-surface), 0.05);
  padding: 12px 16px;
  border-radius: 6px;
  overflow-x: auto;
  margin: 0.75em 0;
}

.markdown-body :deep(pre code) {
  background: none;
  padding: 0;
}

.markdown-body :deep(a) {
  color: rgb(var(--v-theme-primary));
  text-decoration: none;
}

.markdown-body :deep(a:hover) {
  text-decoration: underline;
}

.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  padding-left: 1.5em;
  margin-bottom: 0.75em;
}

.markdown-body :deep(li) {
  margin-bottom: 0.25em;
}

.markdown-body :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 0.75em 0;
}

.markdown-body :deep(th),
.markdown-body :deep(td) {
  border: 1px solid rgba(var(--v-border-color), 0.3);
  padding: 6px 12px;
  text-align: left;
}

.markdown-body :deep(th) {
  background: rgba(var(--v-theme-on-surface), 0.05);
  font-weight: 600;
}

.markdown-body :deep(blockquote) {
  border-left: 3px solid rgb(var(--v-theme-primary));
  padding-left: 16px;
  margin: 0.75em 0;
  color: rgba(var(--v-theme-on-surface), 0.7);
}

.markdown-body :deep(img) {
  max-width: 100%;
  border-radius: 4px;
}
</style>
