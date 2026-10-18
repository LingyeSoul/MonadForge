<template>
  <div v-if="filterable" class="config-filterbar">
    <span class="config-field-count">{{ t('cfgParametersCount', { n: fieldCount }) }}</span>
    <v-text-field v-model="search" :label="t('cfgSearchFields')" :data-testid="searchTestId"
      prepend-inner-icon="mdi-magnify" clearable hide-details density="compact" class="config-search" />
    <v-btn v-if="$slots.guide" :data-testid="helpTestId"
      :icon="showGuide ? 'mdi-book-open-page-variant' : 'mdi-book-open-page-variant-outline'"
      :color="showGuide ? 'primary' : undefined" variant="text" :aria-label="t('cfgOpenGuide')"
      :aria-pressed="showGuide" @click="showGuide = !showGuide">
      <v-icon :icon="showGuide ? 'mdi-book-open-page-variant' : 'mdi-book-open-page-variant-outline'" />
      <v-tooltip activator="parent">{{ t('cfgOpenGuide') }}</v-tooltip>
    </v-btn>
  </div>
  <v-row class="config-body">
    <v-col cols="12" :lg="showGuide && $slots.guide ? 8 : 12" class="form-column">
      <slot />
    </v-col>
    <v-col v-if="showGuide && $slots.guide" cols="12" lg="4" class="config-reference">
      <div class="help-panel-sticky"><slot name="guide" /></div>
    </v-col>
  </v-row>
</template>

<script setup lang="ts">
import { useI18n } from '../composables/useI18n'
defineProps<{ fieldCount: number; filterable?: boolean; searchTestId?: string; helpTestId?: string }>()
const search = defineModel<string | null>('search', { default: '' })
const showGuide = defineModel<boolean>('showGuide', { default: true })
const { t } = useI18n()
</script>

<style scoped>
.config-filterbar { display: flex; align-items: center; gap: 12px; padding-bottom: 18px; }
.config-field-count { color: var(--text-muted); font-size: 12px; white-space: nowrap; }
.config-search { max-width: 320px; margin-left: auto; }
.config-body { margin-top: 0; }
.config-reference { border-left: 1px solid var(--border-subtle); }
.help-panel-sticky { position: sticky; top: 16px; }
@media (max-width: 1279px) {
  .config-reference { border-left: 0; border-top: 1px solid var(--border-subtle); }
  .help-panel-sticky { position: static; }
}
@media (max-width: 599px) {
  .config-filterbar { flex-wrap: wrap; }
  .config-field-count { margin-right: auto; }
  .config-search { order: 3; flex-basis: 100%; max-width: none; }
}
</style>
