<template>
  <v-container fluid class="pa-4">
    <div class="text-h5 mb-1">{{ t('sysTitle') }}</div>
    <div class="text-body-2 text-medium-emphasis mb-4">{{ t('sysSubtitle') }}</div>

    <v-row>
      <!-- Model Downloads -->
      <v-col cols="12" md="6">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-download" class="mr-2" />
            {{ t('sysDownloads') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-3">{{ t('sysDownloadsDesc') }}</div>
            <v-list density="compact" class="mb-3">
              <v-list-item
                v-for="group in modelGroups"
                :key="group.id"
              >
                <template #prepend>
                  <v-icon :icon="group.installed ? 'mdi-check-circle' : 'mdi-alert-circle'" :color="group.installed ? 'success' : 'warning'" />
                </template>
                <v-list-item-title>{{ t(`sysModel_${group.id}`) }}</v-list-item-title>
                <v-list-item-subtitle>
                  <v-chip size="x-small" :color="group.installed ? 'success' : 'warning'" variant="tonal">
                    {{ group.installed ? t('sysInstalled') : t('sysMissing') }}
                  </v-chip>
                </v-list-item-subtitle>
                <template #append>
                  <v-btn
                    size="small"
                    variant="text"
                    :loading="isRunning(`download-${group.id}`)"
                    @click="runTask(`download-${group.id}`)"
                  >
                    {{ group.installed ? t('sysRedownload') : t('sysDownload') }}
                  </v-btn>
                </template>
              </v-list-item>
            </v-list>
            <v-btn color="primary" block :loading="isRunning('download-models')" @click="runTask('download-models')">
              {{ t('sysDownloadAll') }}
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Self Update -->
      <v-col cols="12" md="6">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-update" class="mr-2" />
            {{ t('sysUpdate') }}
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-3">{{ t('sysUpdateDesc') }}</div>
            <v-btn color="primary" block :loading="isRunning('update')" @click="showUpdateDlg = true">
              {{ t('sysUpdateBtn') }}
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Environment -->
      <v-col cols="12" md="6">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-information-outline" class="mr-2" />
            {{ t('sysEnvironment') }}
          </v-card-title>
          <v-card-text>
            <v-list density="compact">
              <v-list-item :title="t('sysBackend')" :subtitle="t('sysBackendVal')" />
              <v-list-item :title="t('sysFrontend')" :subtitle="t('sysFrontendVal')" />
              <v-list-item :title="t('sysProtocol')" :subtitle="t('sysProtocolVal')" />
            </v-list>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Quick Actions -->
      <v-col cols="12" md="6">
        <v-card variant="tonal">
          <v-card-title class="text-subtitle-1">
            <v-icon icon="mdi-lightning-bolt" class="mr-2" />
            {{ t('sysQuickActions') }}
          </v-card-title>
          <v-card-text>
            <div class="d-flex flex-column ga-2">
              <v-btn variant="outlined" block prepend-icon="mdi-cog-transfer-outline" to="/config">
                {{ t('sysConfigEditor') }}
              </v-btn>
              <v-btn variant="outlined" block prepend-icon="mdi-console" to="/tasks">
                {{ t('sysTaskMonitor') }}
              </v-btn>
              <v-btn variant="outlined" block prepend-icon="mdi-test-tube" :loading="isRunning('test-unit')" @click="runTask('test-unit')">
                {{ t('sysRunTests') }}
              </v-btn>
            </div>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <v-divider class="my-4" />

    <div class="d-flex align-center mb-2">
      <div class="text-subtitle-1">{{ t('sysAllTasks') }}</div>
      <v-spacer />
      <v-btn variant="text" size="small" prepend-icon="mdi-refresh" @click="taskStore.fetchTasks()">
        {{ t('ppRefresh') }}
      </v-btn>
    </div>

    <v-list v-if="taskStore.tasks.length > 0" density="compact">
      <v-list-item
        v-for="task in taskStore.tasks"
        :key="task.task_id"
        :title="task.command"
        :subtitle="`${task.task_id.slice(0, 8)} | ${t('taskState')}: ${task.state} | PID: ${task.pid ?? '—'}`"
      >
        <template #append>
          <v-chip size="small" :color="stateColor(task.state)" variant="tonal">{{ task.state }}</v-chip>
          <v-btn v-if="task.state === 'running'" icon="mdi-stop" size="small" variant="text" color="error" @click="taskStore.cancelTask(task.task_id)" />
        </template>
      </v-list-item>
    </v-list>
    <div v-else class="text-medium-emphasis text-body-2">{{ t('sysNoTasks') }}</div>

    <!-- Update dialog -->
    <v-dialog v-model="showUpdateDlg" max-width="450">
      <v-card>
        <v-card-title>{{ t('sysUpdate') }}</v-card-title>
        <v-card-text>
          <v-checkbox v-model="updateDryRun" :label="t('sysUpdateDryRun')" density="compact" hide-details class="mb-2" />
          <div class="text-subtitle-2 mb-1">{{ t('sysUpdateConflictPolicy') }}</div>
          <v-radio-group v-model="updateConflictPolicy" density="compact" hide-details>
            <v-radio :label="t('sysUpdateKeepConflicts')" value="keep" />
            <v-radio :label="t('sysUpdateOverwrite')" value="overwrite" />
          </v-radio-group>
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="showUpdateDlg = false">{{ t('dsCancel') }}</v-btn>
          <v-btn color="primary" :loading="isRunning('update')" @click="runUpdate">
            {{ t('sysUpdateBtn') }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useTaskStore } from '../stores/task'
import { useI18n } from '../composables/useI18n'

const taskStore = useTaskStore()
const { t } = useI18n()
taskStore.fetchTasks()

// ── Model groups ──────────────────────────────────────────────

interface ModelGroup {
  id: string
  installed: boolean
  files: { path: string; exists: boolean }[]
}

const modelGroups = ref<ModelGroup[]>([])

async function fetchModelGroups() {
  try {
    const res = await fetch('/api/system/models')
    if (!res.ok) return
    const data = await res.json()
    modelGroups.value = data.groups || []
  } catch { /* ignore */ }
}

onMounted(fetchModelGroups)

// ── Update dialog ─────────────────────────────────────────────

const showUpdateDlg = ref(false)
const updateDryRun = ref(false)
const updateConflictPolicy = ref('keep')

function runUpdate() {
  const args: string[] = []
  if (updateDryRun.value) args.push('--dry-run')
  if (updateConflictPolicy.value === 'overwrite') args.push('--overwrite-conflicts')
  showUpdateDlg.value = false
  taskStore.startTask('update', args)
}

// ── Helpers ───────────────────────────────────────────────────

function isRunning(command: string) {
  return taskStore.tasks.some(tp => tp.command === command && tp.state === 'running')
}

function stateColor(state: string) {
  if (state === 'running') return 'info'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}

async function runTask(command: string) {
  await taskStore.startTask(command)
}
</script>
