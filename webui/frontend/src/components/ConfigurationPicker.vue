<template>
  <section class="method-library" data-testid="configuration-picker">
    <div class="method-library__heading">
      <h2 class="workspace-section-title">{{ title }}</h2>
      <span class="config-field-count">{{ items.length }}</span>
    </div>
    <div class="method-grid">
      <button v-for="item in items" :key="item.value" type="button" class="method-option"
        :disabled="disabled" :data-value="item.value" @click="$emit('select', item.value)">
        <v-icon :icon="variant ? 'mdi-file-document-outline' : 'mdi-vector-combine'" size="22" />
        <span>{{ item.label }}</span>
        <v-icon icon="mdi-arrow-top-right" size="17" class="method-option__arrow" />
      </button>
      <button v-if="createLabel" type="button" class="method-option" :disabled="disabled"
        data-testid="configuration-create" @click="$emit('create')">
        <v-icon icon="mdi-plus" size="22" /><span>{{ createLabel }}</span>
        <v-icon icon="mdi-arrow-top-right" size="17" class="method-option__arrow" />
      </button>
    </div>
    <div v-if="!items.length && !createLabel" class="workspace-empty">
      <v-icon icon="mdi-cog-transfer-outline" size="36" /><h2>{{ emptyText }}</h2>
    </div>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  title: string
  items: { value: string; label: string }[]
  variant?: boolean
  disabled?: boolean
  createLabel?: string
  emptyText: string
}>()
defineEmits<{ select: [value: string]; create: [] }>()
</script>

<style scoped>
.method-library { padding-top: 4px; }
.method-library__heading { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
.config-field-count { color: var(--text-muted); font-size: 12px; white-space: nowrap; }
.method-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 240px), 1fr)); gap: 12px; }
.method-option { display: flex; align-items: center; gap: 14px; min-height: 84px; padding: 20px; border: 1px solid var(--border-subtle); border-radius: 6px; background: var(--bg-surface); color: var(--text-primary); text-align: left; transition: border-color 160ms ease, background 160ms ease; }
.method-option > .v-icon:first-child { color: var(--forge-amber); flex-shrink: 0; }
.method-option span { flex: 1; font-size: 14px; font-weight: 500; overflow-wrap: anywhere; }
.method-option__arrow { color: var(--text-muted); transition: transform 160ms ease; }
.method-option:hover:not(:disabled) { background: var(--bg-hover); border-color: var(--border-strong); }
.method-option:hover:not(:disabled) .method-option__arrow { transform: translate(2px, -2px); }
.method-option:disabled { opacity: 0.55; cursor: wait; }
@media (max-width: 599px) { .method-option { min-height: 72px; } }
</style>
