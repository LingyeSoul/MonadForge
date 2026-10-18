<template>
  <v-dialog :model-value="modelValue" persistent max-width="500">
    <v-card :title="t('qwUnsavedTitle')">
      <v-card-text>{{ t('qwUnsavedBody') }}</v-card-text>
      <v-alert v-if="error" type="error" variant="tonal" class="mx-4">{{ error }}</v-alert>
      <v-card-actions class="flex-wrap">
        <v-btn :disabled="busy" data-testid="unsaved-cancel" @click="$emit('cancel')">{{ t('cfgCancel') }}</v-btn>
        <v-spacer />
        <v-btn :disabled="busy" data-testid="unsaved-discard" @click="$emit('discard')">{{ t('qwDiscardSwitch') }}</v-btn>
        <v-btn color="primary" :loading="busy" data-testid="unsaved-save" @click="$emit('save')">{{ t('qwSaveSwitch') }}</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
import { useI18n } from '../composables/useI18n'
defineProps<{ modelValue: boolean; busy?: boolean; error?: string }>()
defineEmits<{ cancel: []; discard: []; save: [] }>()
const { t } = useI18n()
</script>
