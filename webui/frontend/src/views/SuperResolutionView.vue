<template>
  <v-container fluid class="pa-4 sr-page">
    <div class="page-heading mb-4">
      <div>
        <h1 class="text-h5">{{ t('superResTitle') }}</h1>
        <div class="text-body-2 text-medium-emphasis mt-1">{{ t('superResSubtitle') }}</div>
      </div>
    </div>

    <v-alert type="info" variant="tonal" density="compact" class="mb-4" role="status">
      <template #prepend><v-icon icon="mdi-source-branch" /></template>
      {{ t('superResRelation') }}
    </v-alert>
    <div class="text-caption text-medium-emphasis mb-3">
      <v-icon icon="mdi-console-line" size="small" class="mr-1" />{{ t('superResSetupNote') }}
    </div>

    <v-tabs v-model="tab" color="primary" class="mb-3" show-arrows>
      <v-tab value="infer" prepend-icon="mdi-image-search-outline">{{ t('superResTabInfer') }}</v-tab>
      <v-tab value="data" prepend-icon="mdi-database-cog-outline">{{ t('superResTabData') }}</v-tab>
      <v-tab value="train" prepend-icon="mdi-school-outline">{{ t('superResTabTrain') }}</v-tab>
    </v-tabs>

    <v-window v-model="tab" class="sr-window">
      <v-window-item value="infer">
        <section class="sr-section">
          <div class="section-heading">
            <v-icon icon="mdi-image-filter-center-focus-strong" size="20" />
            <h2>{{ t('superResInference') }}</h2>
          </div>

          <v-row dense>
            <v-col cols="12" md="6">
              <v-text-field
                v-model="infer.input"
                :label="t('superResInput')"
                :placeholder="t('superResInputHint')"
                prepend-inner-icon="mdi-folder-image"
                class="font-mono-field"
                hide-details="auto"
              />
            </v-col>
            <v-col cols="12" md="6">
              <v-text-field
                v-model="infer.output"
                :label="t('superResOutput')"
                :placeholder="t('superResOutputHint')"
                prepend-inner-icon="mdi-folder-arrow-down"
                class="font-mono-field"
                hide-details="auto"
              />
            </v-col>
            <v-col cols="12" sm="6" md="3">
              <v-select v-model="infer.version" :items="srVersions" :label="t('superResVersion')" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="3">
              <v-select v-model.number="infer.chop" :items="chopOptions" :label="t('superResChop')" suffix="px" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="3">
              <v-select v-model="infer.amp_dtype" :items="ampOptions" :label="t('superResAmp')" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="3">
              <v-text-field
                v-model="infer.ckpt"
                :label="t('superResCheckpoint')"
                :placeholder="t('superResCheckpointHint')"
                class="font-mono-field"
                hide-details="auto"
              />
            </v-col>
          </v-row>

          <div class="control-row mt-3">
            <v-switch v-model="infer.musiq" :label="t('superResMusiq')" density="compact" hide-details color="primary" />
            <v-switch v-model="infer.sheet" :label="t('superResSheet')" density="compact" hide-details color="primary" />
            <v-text-field
              v-model.number="infer.sheet_max"
              :label="t('superResSheetMax')"
              type="number"
              min="1"
              max="128"
              class="numeric-field"
              :disabled="!infer.sheet"
              hide-details="auto"
            />
            <v-btn
              color="primary"
              prepend-icon="mdi-play"
              :loading="taskStore.loading"
              :disabled="isRunning('sr-test')"
              @click="runInference"
            >
              {{ t('superResRunInference') }}
            </v-btn>
          </div>
        </section>

        <section class="sr-section">
          <div class="section-heading">
            <v-icon icon="mdi-lightning-bolt-outline" size="20" />
            <h2>{{ t('superResRsdInference') }}</h2>
          </div>
          <v-row dense>
            <v-col cols="12" sm="6" md="2">
              <v-select v-model="rsdInfer.version" :items="rsdVersions" :label="t('superResRsdVersion')" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="2">
              <v-select v-model="rsdInfer.amp_dtype" :items="ampOptions" :label="t('superResAmp')" hide-details="auto" />
            </v-col>
            <v-col cols="12" md="4">
              <v-text-field v-model="rsdInfer.input" :label="t('superResInputDir')" class="font-mono-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" md="4">
              <v-text-field v-model="rsdInfer.output" :label="t('superResOutputDir')" class="font-mono-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4" md="2">
              <v-text-field v-model.number="rsdInfer.chop" :label="t('superResChop')" type="number" min="64" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4" md="2">
              <v-text-field v-model.number="rsdInfer.overlap" :label="t('superResOverlap')" type="number" min="0" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4" md="2">
              <v-text-field v-model.number="rsdInfer.tile_batch" :label="t('superResTileBatch')" type="number" min="1" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="2">
              <v-select v-model="rsdInfer.weights" :items="['ema', 'student']" :label="t('superResWeights')" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="4">
              <v-text-field v-model="rsdInfer.ckpt" :label="t('superResCheckpoint')" class="font-mono-field" hide-details="auto" />
            </v-col>
          </v-row>
          <div class="control-row mt-3">
            <v-switch v-model="rsdInfer.evaluate" :label="t('superResEvaluate')" density="compact" hide-details color="primary" />
            <v-btn
              color="secondary"
              prepend-icon="mdi-play"
              :loading="taskStore.loading"
              :disabled="isRunning('sr-rsd-infer')"
              @click="runRsdInference"
            >
              {{ t('superResRunRsd') }}
            </v-btn>
          </div>
        </section>
      </v-window-item>

      <v-window-item value="data">
        <section class="sr-section">
          <div class="section-heading">
            <v-icon icon="mdi-image-multiple-outline" size="20" />
            <h2>{{ t('superResPool') }}</h2>
          </div>
          <v-row dense>
            <v-col cols="12" md="6">
              <v-text-field v-model="pool.src" :label="t('superResSource')" :placeholder="t('superResPoolSourceHint')" class="font-mono-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" md="6">
              <v-text-field v-model="pool.out" :label="t('superResPoolOut')" class="font-mono-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4">
              <v-text-field v-model.number="pool.min_edge" :label="t('superResMinEdge')" type="number" min="256" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4">
              <v-text-field v-model.number="pool.lap_floor" :label="t('superResSharpness')" type="number" min="0" step="0.1" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4">
              <v-text-field v-model.number="pool.limit" :label="t('superResLimit')" type="number" min="0" class="numeric-field" hide-details="auto" />
            </v-col>
          </v-row>
          <div class="control-row mt-3">
            <v-switch v-model="pool.dry_run" :label="t('superResDryRun')" density="compact" hide-details color="primary" />
            <v-btn color="primary" prepend-icon="mdi-play" :loading="taskStore.loading" @click="runPool" >{{ t('superResBuildPool') }}</v-btn>
          </div>
        </section>

        <section class="sr-section">
          <div class="section-heading">
            <v-icon icon="mdi-format-letter-case" size="20" />
            <h2>{{ t('superResTextDetect') }}</h2>
          </div>
          <v-row dense>
            <v-col cols="12" md="6">
              <v-text-field v-model="textDetect.src" :label="t('superResSource')" class="font-mono-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" md="6">
              <v-text-field v-model="textDetect.out" :label="t('superResTextOut')" class="font-mono-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="3">
              <v-switch v-model="textDetect.gpu" :label="t('superResUseGpu')" density="compact" hide-details color="primary" />
            </v-col>
            <v-col cols="12" sm="6" md="3">
              <v-switch v-model="textDetect.resume" :label="t('superResResume')" density="compact" hide-details color="primary" />
            </v-col>
            <v-col cols="12" sm="6" md="3">
              <v-text-field v-model.number="textDetect.limit" :label="t('superResLimit')" type="number" min="0" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="3">
              <v-btn block color="secondary" prepend-icon="mdi-play" :loading="taskStore.loading" @click="runTextDetect">{{ t('superResRunTextDetect') }}</v-btn>
            </v-col>
          </v-row>
          <div class="text-caption text-medium-emphasis mt-3">{{ t('superResTextDetectNote') }}</div>
        </section>

        <section class="sr-section">
          <div class="section-heading">
            <v-icon icon="mdi-test-tube" size="20" />
            <h2>{{ t('superResEval') }}</h2>
          </div>
          <v-row dense>
            <v-col cols="12" md="4">
              <v-text-field v-model="evalSet.src" :label="t('superResSource')" class="font-mono-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" md="4">
              <v-text-field v-model="evalSet.out" :label="t('superResEvalOut')" class="font-mono-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4" md="1">
              <v-text-field v-model.number="evalSet.count" :label="t('superResEvalCount')" type="number" min="1" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4" md="1">
              <v-text-field v-model.number="evalSet.scale" :label="t('superResScale')" type="number" min="2" max="4" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4" md="2">
              <v-btn block color="primary" prepend-icon="mdi-play" :loading="taskStore.loading" @click="runEvalPrep">{{ t('superResRunEvalPrep') }}</v-btn>
            </v-col>
          </v-row>
          <v-row dense class="mt-2">
            <v-col cols="12" sm="4">
              <v-select v-model="phase0.version" :items="['v1', 'v2', 'v3']" :label="t('superResVersion')" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4">
              <v-select v-model="phase0.amp_dtype" :items="ampOptions" :label="t('superResAmp')" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4">
              <v-btn block color="secondary" prepend-icon="mdi-chart-line" :loading="taskStore.loading" @click="runPhase0">{{ t('superResRunPhase0') }}</v-btn>
            </v-col>
          </v-row>
        </section>
      </v-window-item>

      <v-window-item value="train">
        <section class="sr-section">
          <div class="section-heading">
            <v-icon icon="mdi-school-outline" size="20" />
            <h2>{{ t('superResTeacher') }}</h2>
          </div>
          <v-row dense>
            <v-col cols="12" sm="6" md="2">
              <v-select v-model="teacher.version" :items="['x2', 'x4', 'x4s4']" :label="t('superResTeacherVersion')" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="2">
              <v-text-field v-model.number="teacher.iters" :label="t('superResSteps')" type="number" min="1" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="2">
              <v-text-field v-model.number="teacher.bs" :label="t('superResBatch')" type="number" min="1" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="2">
              <v-select v-model="teacher.amp_dtype" :items="ampOptions" :label="t('superResAmp')" hide-details="auto" />
            </v-col>
            <v-col cols="12" md="4">
              <v-text-field v-model="teacher.src" :label="t('superResSource')" class="font-mono-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" md="6">
              <v-text-field v-model="teacher.text_boxes" :label="t('superResTextBoxes')" class="font-mono-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="3">
              <v-text-field v-model.number="teacher.text_crop_prob" :label="t('superResTextCrop')" type="number" min="0" max="1" step="0.05" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="3">
              <v-text-field v-model.number="teacher.max_steps" :label="t('superResMaxSteps')" type="number" min="0" class="numeric-field" hide-details="auto" />
            </v-col>
          </v-row>
          <div class="control-row mt-3">
            <v-switch v-model="teacher.amp" :label="t('superResAmpEnabled')" density="compact" hide-details color="primary" />
            <v-switch v-model="teacher.compile" :label="t('superResCompile')" density="compact" hide-details color="primary" />
            <v-switch v-model="teacher.grad_ckpt" :label="t('superResGradCkpt')" density="compact" hide-details color="primary" />
            <v-btn color="primary" prepend-icon="mdi-play" :loading="taskStore.loading" @click="runTeacher">{{ t('superResTrainTeacher') }}</v-btn>
          </div>
        </section>

        <section class="sr-section">
          <div class="section-heading">
            <v-icon icon="mdi-vector-combine" size="20" />
            <h2>{{ t('superResDistill') }}</h2>
          </div>
          <v-row dense>
            <v-col cols="12" sm="6" md="2">
              <v-select v-model="rsdTrain.version" :items="rsdTrainVersions" :label="t('superResRsdVersion')" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="2">
              <v-text-field v-model.number="rsdTrain.iters" :label="t('superResSteps')" type="number" min="1" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="2">
              <v-text-field v-model.number="rsdTrain.bs" :label="t('superResBatch')" type="number" min="1" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="2">
              <v-text-field v-model.number="rsdTrain.k" :label="t('superResK')" type="number" min="1" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="2">
              <v-select v-model="rsdTrain.amp_dtype" :items="ampOptions" :label="t('superResAmp')" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="6" md="2">
              <v-text-field v-model.number="rsdTrain.max_steps" :label="t('superResMaxSteps')" type="number" min="0" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" md="8">
              <v-text-field v-model="rsdTrain.src" :label="t('superResSource')" class="font-mono-field" hide-details="auto" />
            </v-col>
          </v-row>
          <div class="control-row mt-3">
            <v-switch v-model="rsdTrain.amp" :label="t('superResAmpEnabled')" density="compact" hide-details color="primary" />
            <v-switch v-model="rsdTrain.compile" :label="t('superResCompile')" density="compact" hide-details color="primary" />
            <v-switch v-model="rsdTrain.grad_ckpt" :label="t('superResGradCkpt')" density="compact" hide-details color="primary" />
            <v-btn color="secondary" prepend-icon="mdi-play" :loading="taskStore.loading" @click="runRsdTrain">{{ t('superResTrainRsd') }}</v-btn>
          </div>
        </section>

        <section class="sr-section">
          <div class="section-heading">
            <v-icon icon="mdi-memory" size="20" />
            <h2>{{ t('superResDryRunTitle') }}</h2>
          </div>
          <v-row dense>
            <v-col cols="12" sm="4">
              <v-select v-model="dryRun.version" :items="['x4', 'x2']" :label="t('superResRsdVersion')" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4">
              <v-text-field v-model.number="dryRun.bs" :label="t('superResBatch')" type="number" min="1" class="numeric-field" hide-details="auto" />
            </v-col>
            <v-col cols="12" sm="4">
              <v-select v-model="dryRun.amp_dtype" :items="ampOptions" :label="t('superResAmp')" hide-details="auto" />
            </v-col>
          </v-row>
          <div class="control-row mt-3">
            <v-switch v-model="dryRun.amp" :label="t('superResAmpEnabled')" density="compact" hide-details color="primary" />
            <v-btn color="info" prepend-icon="mdi-play" :loading="taskStore.loading" @click="runDryRun">{{ t('superResRunDryRun') }}</v-btn>
          </div>
        </section>
      </v-window-item>
    </v-window>

    <section class="sr-section active-section">
      <div class="section-heading">
        <v-icon icon="mdi-console-line" size="20" />
        <h2>{{ t('superResActiveTasks') }}</h2>
        <v-spacer />
        <v-btn icon="mdi-refresh" variant="text" size="small" :title="t('ppRefresh')" :aria-label="t('ppRefresh')" @click="taskStore.poll" />
      </div>
      <v-list v-if="activeTasks.length" density="compact" class="task-list">
        <v-list-item v-for="task in activeTasks" :key="task.task_id" :title="task.command" :subtitle="`${t('taskState')}: ${task.state} | PID: ${task.pid ?? '-'}`">
          <template #append>
            <v-chip size="small" :color="stateColor(task.state)" variant="tonal">{{ task.state }}</v-chip>
            <v-btn v-if="task.state === 'running' || task.state === 'pending'" icon="mdi-stop" size="small" variant="text" color="error" :title="t('superResStop')" :aria-label="t('superResStop')" @click="taskStore.cancelTask(task.task_id)" />
          </template>
        </v-list-item>
      </v-list>
      <div v-else class="text-medium-emphasis text-body-2">{{ t('superResNoTasks') }}</div>
    </section>
  </v-container>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useI18n } from '../composables/useI18n'
import { useNotifyStore } from '../stores/notify'
import { useTaskStore } from '../stores/task'

type AmpDtype = 'auto' | 'fp16' | 'bf16'

const { t } = useI18n()
const notify = useNotifyStore()
const taskStore = useTaskStore()
const tab = ref('infer')
const ampOptions: AmpDtype[] = ['auto', 'fp16', 'bf16']
const srVersions = ['v1', 'v2', 'v3', 'x2', 'x4ft', 'x4s4']
const rsdVersions = ['x4', 'x2', 'x4ft']
const rsdTrainVersions = ['x4', 'x2', 'x4ft']
const chopOptions = [512, 256, 64]
const srCommands = new Set([
  'sr-prep', 'sr-phase0', 'sr-test', 'sr-build-hr-pool', 'sr-detect-text',
  'sr-train', 'sr-rsd-train', 'sr-rsd-dryrun', 'sr-rsd-infer',
])

const infer = reactive({
  input: '', output: '', version: 'v3', chop: 512, amp_dtype: 'auto' as AmpDtype,
  musiq: true, sheet: true, sheet_max: 32, ckpt: '',
})
const rsdInfer = reactive({
  version: 'x4', amp_dtype: 'auto' as AmpDtype, input: 'sr/data/lr_eval',
  output: '', chop: 256, overlap: 64, tile_batch: 8, weights: 'ema', ckpt: '', evaluate: false,
})
const pool = reactive({ src: '', out: 'sr/data/hr_pool', min_edge: 1024, lap_floor: 18, limit: 0, dry_run: false })
const textDetect = reactive({ src: 'sr/data/hr_pool', out: 'sr/data/text_boxes.json', gpu: true, resume: true, limit: 0 })
const evalSet = reactive({ src: 'image_dataset', out: 'sr/data', count: 30, scale: 4 })
const phase0 = reactive({ version: 'v3', amp_dtype: 'auto' as AmpDtype })
const teacher = reactive({
  version: 'x2', iters: 45000, bs: 8, amp: true, amp_dtype: 'auto' as AmpDtype,
  compile: false, grad_ckpt: false, src: '', text_boxes: 'sr/data/text_boxes.json',
  text_crop_prob: 0.25, max_steps: 0,
})
const rsdTrain = reactive({
  version: 'x4', iters: 18000, bs: 4, k: 5, amp: true, amp_dtype: 'auto' as AmpDtype,
  compile: false, grad_ckpt: false, src: '', max_steps: 0,
})
const dryRun = reactive({ version: 'x4', bs: 1, amp: true, amp_dtype: 'auto' as AmpDtype })
let pollTimer: ReturnType<typeof setInterval> | null = null

const activeTasks = computed(() => taskStore.tasks.filter(task => srCommands.has(task.command)))

function isRunning(command: string) {
  return taskStore.tasks.some(task => task.command === command && task.state === 'running')
}

function stateColor(state: string) {
  if (state === 'running' || state === 'stopping') return 'info'
  if (state === 'success') return 'success'
  if (state === 'failed') return 'error'
  return undefined
}

function addArg(args: string[], flag: string, value: unknown) {
  if (value === undefined || value === null || value === '') return
  args.push(flag, String(value))
}

async function launch(command: string, args: string[] = [], env: Record<string, string> = {}) {
  const taskId = await taskStore.startTask(command, args, Object.keys(env).length ? env : undefined)
  notify.show(taskId ? t('notifyTaskStarted', { command }) : t('notifyTaskStartFailed', { command }), taskId ? 'success' : 'error')
}

async function runInference() {
  if (!infer.input.trim()) {
    notify.show(t('superResInputRequired'), 'warning')
    return
  }
  const args: string[] = ['--amp_dtype', infer.amp_dtype]
  if (infer.ckpt.trim()) addArg(args, '--ckpt', infer.ckpt.trim())
  if (!infer.musiq) args.push('--no_musiq')
  if (!infer.sheet) args.push('--no_sheet')
  else addArg(args, '--sheet_max', infer.sheet_max)
  await launch('sr-test', args, {
    IN: infer.input.trim(), OUT: infer.output.trim(), VERSION: infer.version, CHOP: String(infer.chop),
  })
}

async function runRsdInference() {
  const args: string[] = ['--version', rsdInfer.version, '--amp_dtype', rsdInfer.amp_dtype]
  addArg(args, '--in_dir', rsdInfer.input.trim())
  addArg(args, '--out_dir', rsdInfer.output.trim())
  addArg(args, '--chop', rsdInfer.chop)
  addArg(args, '--overlap', rsdInfer.overlap)
  addArg(args, '--tile_batch', rsdInfer.tile_batch)
  addArg(args, '--weights', rsdInfer.weights)
  if (rsdInfer.ckpt.trim()) addArg(args, '--ckpt', rsdInfer.ckpt.trim())
  if (rsdInfer.evaluate) args.push('--eval')
  await launch('sr-rsd-infer', args)
}

async function runPool() {
  const args: string[] = []
  if (pool.src.trim()) addArg(args, '--src', pool.src.trim())
  addArg(args, '--out', pool.out.trim())
  addArg(args, '--min_edge', pool.min_edge)
  addArg(args, '--lap_floor', pool.lap_floor)
  addArg(args, '--limit', pool.limit)
  if (pool.dry_run) args.push('--dry_run')
  await launch('sr-build-hr-pool', args)
}

async function runTextDetect() {
  const args: string[] = []
  addArg(args, '--src', textDetect.src.trim())
  addArg(args, '--out', textDetect.out.trim())
  addArg(args, '--limit', textDetect.limit)
  if (textDetect.gpu) args.push('--gpu')
  else args.push('--no-gpu')
  if (textDetect.resume) args.push('--resume')
  await launch('sr-detect-text', args)
}

async function runEvalPrep() {
  const args: string[] = []
  addArg(args, '--src', evalSet.src.trim())
  addArg(args, '--out', evalSet.out.trim())
  addArg(args, '--n', evalSet.count)
  addArg(args, '--scale', evalSet.scale)
  await launch('sr-prep', args)
}

async function runPhase0() {
  await launch('sr-phase0', ['--version', phase0.version, '--amp_dtype', phase0.amp_dtype])
}

async function runTeacher() {
  const args: string[] = ['--iters', String(teacher.iters), '--bs', String(teacher.bs), '--amp_dtype', teacher.amp_dtype]
  if (!teacher.amp) args.push('--no-amp')
  if (teacher.compile) args.push('--compile')
  if (teacher.grad_ckpt) args.push('--grad_ckpt')
  addArg(args, '--src', teacher.src.trim())
  addArg(args, '--text_boxes', teacher.text_boxes.trim())
  addArg(args, '--text_crop_prob', teacher.text_crop_prob)
  addArg(args, '--max_steps', teacher.max_steps)
  await launch('sr-train', args, { VERSION: teacher.version })
}

async function runRsdTrain() {
  const args: string[] = ['--iters', String(rsdTrain.iters), '--bs', String(rsdTrain.bs), '--K', String(rsdTrain.k), '--amp_dtype', rsdTrain.amp_dtype]
  if (!rsdTrain.amp) args.push('--no-amp')
  if (rsdTrain.compile) args.push('--compile')
  if (rsdTrain.grad_ckpt) args.push('--grad_ckpt')
  addArg(args, '--src', rsdTrain.src.trim())
  addArg(args, '--max_steps', rsdTrain.max_steps)
  await launch('sr-rsd-train', args, { VERSION: rsdTrain.version })
}

async function runDryRun() {
  const args: string[] = ['--bs', String(dryRun.bs), '--version', dryRun.version, '--amp_dtype', dryRun.amp_dtype]
  if (dryRun.amp) args.push('--amp')
  await launch('sr-rsd-dryrun', args)
}

onMounted(() => {
  taskStore.poll()
  pollTimer = window.setInterval(() => taskStore.poll(), 5000)
})

onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<style scoped>
.sr-page {
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
  max-width: 1600px;
}

.page-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.sr-window { min-height: 0; }

.sr-section {
  padding: 18px 0 20px;
  border-top: 1px solid var(--border-default);
}

.section-heading {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 32px;
  margin-bottom: 14px;
}

.section-heading h2 {
  font-size: 15px;
  font-weight: 600;
  line-height: 1.4;
}

.control-row {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 16px;
  flex-wrap: wrap;
}

.control-row .v-switch { min-width: 150px; }
.numeric-field { font-family: var(--font-mono); }
.task-list { border: 1px solid var(--border-subtle); background: transparent; }
.active-section { padding-bottom: 10px; }

@media (max-width: 599px) {
  .sr-page { padding: 12px !important; }
  .control-row { align-items: stretch; justify-content: flex-start; }
  .control-row .v-btn { flex: 1 1 100%; }
  .page-heading { align-items: center; }
}
</style>
