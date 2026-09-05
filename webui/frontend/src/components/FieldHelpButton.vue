<template>
  <v-btn icon="mdi-information-outline" variant="text" size="x-small" class="field-help-button"
    :aria-label="`${t('cfgFieldHelp')}: ${fieldKey}`" @click.stop="$emit('help')">
    <v-icon icon="mdi-information-outline" size="16" />
    <v-tooltip activator="parent" location="top">{{ t('cfgFieldHelp') }}: {{ fieldKey }}</v-tooltip>
  </v-btn>
</template>

<script setup lang="ts">
import { useI18n } from '../composables/useI18n'

defineProps<{ fieldKey: string }>()
defineEmits<{ help: [] }>()
const { t } = useI18n()
</script>

<style scoped>
.field-help-button { color: var(--text-muted); }
.field-help-button:hover, .field-help-button:focus-visible { color: var(--forge-ember); }
</style>
