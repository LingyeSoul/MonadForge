<template>
  <v-btn :icon="text ? undefined : 'mdi-stop'" :prepend-icon="text ? 'mdi-stop' : undefined"
    :variant="text ? 'outlined' : 'text'" size="small" color="error" :disabled="!taskId || busy"
    :aria-label="t('taskStopConfirm')" @click.prevent.stop="openConfirm"><v-icon v-if="!text" icon="mdi-stop" /><template v-else>{{ text }}</template></v-btn>
  <v-dialog v-model="open" max-width="450" :persistent="busy">
    <v-card :title="t('taskStopConfirm')">
      <v-card-text>{{ t('taskCancelConfirm') }}<div class="text-caption mt-2" style="overflow-wrap: anywhere">{{ selectedTask }}</div></v-card-text>
      <v-alert v-if="failed" type="error" variant="tonal" class="mx-4">{{ t('notifyTaskCancelFailed') }}</v-alert>
      <v-card-actions>
        <v-spacer />
        <v-btn :disabled="busy" @click="open = false">{{ t('cfgCancel') }}</v-btn>
        <v-btn color="error" :loading="busy" @click="stop">{{ t('taskStopConfirm') }}</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from '../composables/useI18n'
import { useTaskStore } from '../stores/task'
const props = defineProps<{ taskId?: string; text?: string }>()
const { t } = useI18n()
const tasks = useTaskStore()
const open = ref(false)
const busy = ref(false)
const selectedTask = ref('')
const failed = ref(false)
function openConfirm() { selectedTask.value = props.taskId || ''; failed.value = false; open.value = true }
async function stop() {
  if (!selectedTask.value || busy.value) return
  busy.value = true
  try { if (await tasks.cancelTask(selectedTask.value)) open.value = false; else failed.value = true }
  finally { busy.value = false }
}
</script>
