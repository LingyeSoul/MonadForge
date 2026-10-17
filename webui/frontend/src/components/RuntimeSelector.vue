<template>
  <v-select :model-value="workspace.runtimeProfile" :items="items" :label="label" :disabled="disabled || workspace.runtimeLoading" :loading="workspace.runtimeLoading" :error-messages="workspace.runtimeError" density="compact" variant="outlined" hide-details="auto" data-testid="runtime-profile-select" @update:model-value="select">
    <template #item="{ props: itemProps, item }"><v-list-item v-bind="itemProps" :subtitle="item.description" :aria-disabled="item.props.disabled" /></template>
  </v-select>
</template>
<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useAppStore } from '../stores/app'
import { useModelWorkspace, type RuntimeProfileName } from '../stores/modelWorkspace'
defineProps<{ disabled?: boolean }>()
const workspace = useModelWorkspace()
const app = useAppStore()
const label = computed(() => app.language === 'cn' || app.language.startsWith('zh') ? '运行环境' : 'Runtime environment')
const items = computed(() => workspace.runtimes.map(runtime => ({ title: runtime.name, value: runtime.name, props: { disabled: !runtime.available }, description: runtime.available ? runtime.python || '' : runtime.error || 'Unavailable' })))
function select(value: RuntimeProfileName): void {
  try { workspace.selectRuntime(value) }
  catch (cause) { workspace.runtimeError = cause instanceof Error ? cause.message : String(cause) }
}
onMounted(() => { void workspace.loadRuntimes().catch(() => {}) })
</script>
