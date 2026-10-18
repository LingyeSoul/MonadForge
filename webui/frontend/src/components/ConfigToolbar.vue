<template>
  <div class="config-toolbar-host">
    <div class="config-toolbar">
      <div class="config-toolbar__model" data-toolbar-slot="model"><slot name="model" /></div>
      <div class="config-toolbar__runtime" data-toolbar-slot="runtime"><slot name="runtime" /></div>
      <div class="config-toolbar__method" data-toolbar-slot="method"><slot name="method" /></div>
      <div class="config-toolbar__variant" data-toolbar-slot="variant"><slot name="variant" /></div>
      <div class="config-toolbar__preset" data-toolbar-slot="preset"><slot name="preset" /></div>
      <div class="config-toolbar__task" data-toolbar-slot="task"><slot name="task" /></div>
      <div class="config-toolbar__actions" data-toolbar-slot="actions"><slot name="actions" /></div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineSlots<{
  model(): unknown
  runtime(): unknown
  method(): unknown
  variant(): unknown
  preset(): unknown
  task(): unknown
  actions(): unknown
}>()
</script>

<style scoped>
.config-toolbar-host {
  container: config-toolbar / inline-size;
  min-width: 0;
}
.config-toolbar {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 16px;
  align-items: center;
  padding: 20px 0;
  margin-bottom: 24px;
  border-top: 1px solid var(--border-subtle);
  border-bottom: 1px solid var(--border-subtle);
}
.config-toolbar > * {
  min-width: 0;
  max-width: 100%;
}
.config-toolbar :deep(.v-input) {
  min-width: 0;
  max-width: 100%;
}
.config-toolbar__variant,
.config-toolbar__preset,
.config-toolbar__actions {
  display: flex;
  align-items: center;
  gap: 6px;
}
.config-toolbar__variant :deep(.v-input),
.config-toolbar__preset :deep(.v-input) {
  flex: 1 1 0;
}
.config-toolbar__variant :deep(.v-btn),
.config-toolbar__preset :deep(.v-btn),
.config-toolbar__actions :deep(.v-btn) {
  flex-shrink: 0;
}
.config-toolbar__actions {
  flex-wrap: wrap;
  justify-content: flex-start;
}
@container config-toolbar (min-width: 560px) {
  .config-toolbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .config-toolbar__actions {
    grid-column: 1 / -1;
    justify-content: flex-end;
  }
}
@container config-toolbar (min-width: 900px) {
  .config-toolbar { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .config-toolbar__actions { grid-column: 3 / -1; }
}
</style>
