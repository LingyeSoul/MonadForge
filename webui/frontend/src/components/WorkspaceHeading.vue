<template>
  <header class="workspace-heading">
    <div class="workspace-heading__copy">
      <h1>{{ title }}</h1>
      <p v-if="subtitle">{{ subtitle }}</p>
    </div>
    <slot />
  </header>
</template>

<script setup lang="ts">
defineProps<{ title: string; subtitle?: string }>()
</script>

<style scoped>
.workspace-heading { flex-shrink: 0; }
.workspace-heading__copy { min-width: 0; }
</style>
