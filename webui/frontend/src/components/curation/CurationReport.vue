<template>
  <v-card variant="tonal" data-testid="curation-report">
    <v-card-title class="text-subtitle-1 d-flex flex-wrap ga-2 align-center">{{ tr('report') }} · {{ job.stage }}<v-spacer /><v-chip size="small">{{ job.state }}</v-chip><v-btn size="small" variant="text" :disabled="loading || api.busy" @click="load">{{ tr('refresh') }}</v-btn></v-card-title>
    <v-card-text>
      <v-alert v-if="job.stage === 'export'" type="warning" variant="tonal" class="mb-3">{{ tr('exportHint') }}</v-alert>
      <p class="text-caption text-medium-emphasis mb-3">{{ job.id }} · {{ job.apply ? tr('applied') : tr(api.schema.stages.find(stage => stage.id === job.stage)?.apply === false ? 'workspaceRun' : 'dryRun') }}</p>
      <v-alert v-if="failure" type="info" variant="tonal">{{ failure }}</v-alert>
      <v-progress-linear v-if="loading" indeterminate />
      <template v-if="reportData">
        <div class="d-flex flex-wrap ga-2 mb-4"><v-chip v-for="[key, value] in summaries" :key="key" variant="outlined" size="small">{{ key }}: {{ value }}</v-chip></div>
        <div v-if="rows.length" class="report-table-wrap mb-4"><v-table density="compact"><thead><tr><th v-for="key in rowKeys" :key="key">{{ key }}</th></tr></thead><tbody><tr v-for="(row, index) in displayedRows" :key="index"><td v-for="key in rowKeys" :key="key">{{ describe(row[key]) }}</td></tr></tbody></v-table></div>
        <v-pagination v-if="rows.length > 40" v-model="rowPage" :length="Math.ceil(rows.length / 40)" density="compact" :total-visible="5" />
        <v-expansion-panels variant="accordion" class="mb-4"><v-expansion-panel :title="tr('reportDetails')"><v-expansion-panel-text><pre class="report-json">{{ JSON.stringify(reportData, null, 2) }}</pre></v-expansion-panel-text></v-expansion-panel></v-expansion-panels>
      </template>
      <v-divider class="my-4" />
      <div class="text-subtitle-2 mb-2">{{ tr('proposals') }} · {{ proposalRels.length }}</div>
      <template v-if="proposalRels.length">
        <v-select :model-value="proposalRel" :items="proposalRels" :label="tr('proposalItem')" density="compact" hide-details @update:model-value="loadProposal($event)" />
        <div class="d-flex ga-2 align-center my-2"><v-btn size="small" variant="text" :disabled="proposalIndex <= 0 || proposalLoading" @click="loadProposal(proposalRels[proposalIndex - 1]!)">{{ tr('previous') }}</v-btn><span class="text-caption">{{ proposalIndex + 1 }} / {{ proposalRels.length }} · {{ tr('reviewed') }} {{ reviewed.size }}</span><v-btn size="small" variant="text" :disabled="proposalIndex >= proposalRels.length - 1 || proposalLoading" @click="loadProposal(proposalRels[proposalIndex + 1]!)">{{ tr('next') }}</v-btn></div>
        <v-progress-linear v-if="proposalLoading" indeterminate />
        <template v-if="proposal">
          <div class="text-caption mb-2 proposal-path">{{ proposal.status }} · {{ proposal.kind }} · {{ proposal.path }}</div>
          <v-row dense><v-col cols="12" md="6"><div class="text-caption mb-1">{{ tr('before') }}</div><pre class="proposal-text">{{ proposal.before }}</pre></v-col><v-col cols="12" md="6"><div class="text-caption mb-1">{{ tr('after') }}</div><pre class="proposal-text">{{ proposal.after }}</pre></v-col></v-row>
          <div class="proposal-diff mt-2"><span>{{ diff.prefix }}</span><del>{{ diff.removed }}</del><ins>{{ diff.added }}</ins><span>{{ diff.suffix }}</span></div>
        </template>
      </template>
      <p v-else class="text-body-2 text-medium-emphasis">{{ tr('noProposals') }}</p>
      <v-expansion-panels v-if="job.roots" variant="accordion" class="mt-4"><v-expansion-panel :title="tr('rootsSnapshot')"><v-expansion-panel-text><dl><template v-for="(root, name) in job.roots" :key="name"><dt class="text-caption">{{ name }}</dt><dd class="proposal-path mb-2">{{ typeof root === 'string' ? root : root.path }}</dd></template></dl></v-expansion-panel-text></v-expansion-panel></v-expansion-panels>
      <v-checkbox v-if="reportData && canReplay" v-model="acknowledged" data-testid="curation-review-acknowledge" :label="tr(job.stage === 'export' ? 'acknowledgeExport' : 'acknowledgeReport')" density="compact" hide-details class="mt-4" />
      <div class="d-flex flex-wrap ga-2 mt-4">
        <v-btn v-if="canReplay && !job.apply" data-testid="curation-apply-report" color="primary" :disabled="!acknowledged || !reportData || api.busy || loading" @click="confirm = 'apply'">{{ tr('apply') }}</v-btn>
        <v-btn v-if="canReplay && job.apply" data-testid="curation-undo-report" color="warning" variant="outlined" :disabled="!acknowledged || !reportData || api.busy || loading" @click="confirm = 'undo'">{{ tr('undo') }}</v-btn>
        <v-btn variant="text" size="small" :to="{ path: '/tasks', query: { task: job.task_id || job.id } }">{{ tr('openTasks') }}</v-btn>
      </div>
    </v-card-text>
    <v-dialog :model-value="Boolean(confirm)" max-width="580" @update:model-value="value => { if (!value) confirm = null }"><v-card :title="tr(confirm === 'undo' ? 'confirmUndo' : 'confirmApply')"><v-card-text><p>{{ tr('confirmBody') }}</p><v-alert v-if="job.stage === 'export'" type="warning" variant="tonal" class="mt-3">{{ tr('exportHint') }}</v-alert><p class="text-caption mt-3 proposal-path">{{ job.id }} · {{ reportPath }}</p></v-card-text><v-card-actions><v-spacer /><v-btn @click="confirm = null">{{ tr('cancel') }}</v-btn><v-btn color="primary" :loading="api.busy" @click="replay">{{ tr(confirm === 'undo' ? 'undo' : 'apply') }}</v-btn></v-card-actions></v-card></v-dialog>
  </v-card>
</template>
<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { UnwrapNestedRefs } from 'vue'
import { useAppStore } from '../../stores/app'
import { captionDiff, useCuration } from '../../composables/useCuration'
import { curationText } from '../../i18n/curation'
import type { CurationJob, CurationProposal, CurationValues } from '../../types/curation'
const props = defineProps<{ job: CurationJob; api: UnwrapNestedRefs<ReturnType<typeof useCuration>> }>()
const emit = defineEmits<{ queued: [id: string] }>()
const app = useAppStore()
const tr = (key: Parameters<typeof curationText>[1]) => curationText(app.language, key)
const loading = ref(false), failure = ref(''), reportData = ref<CurationValues | null>(null), reportPath = ref('')
const proposalRels = ref<string[]>([]), proposalRel = ref(''), proposal = ref<CurationProposal | null>(null), proposalLoading = ref(false)
const reviewed = ref(new Set<string>()), acknowledged = ref(false), confirm = ref<'apply' | 'undo' | null>(null), rowPage = ref(1)
let requestIndex = 0, proposalRequestIndex = 0
const canReplay = computed(() => {
  if (props.job.operation === 'undo') return false
  const success = ['completed', 'succeeded', 'success', 'done'].includes(props.job.state)
  const recoverExport = props.job.stage === 'export' && props.job.apply && ['failed', 'cancelled', 'canceled', 'error'].includes(props.job.state)
  return (success || recoverExport) && (props.job.stage === 'export' || props.api.schema.stages.find(stage => stage.id === props.job.stage)?.replay)
})
const proposalIndex = computed(() => proposalRels.value.indexOf(proposalRel.value))
const diff = computed(() => captionDiff(proposal.value?.before || '', proposal.value?.after || ''))
const summaries = computed(() => Object.entries(reportData.value || {}).filter(([, value]) => value !== null && ['string', 'number', 'boolean'].includes(typeof value) && String(value).length < 140))
const rows = computed(() => {
  const data = reportData.value || {}
  for (const key of ['rows', 'items', 'changes', 'files', 'proposals']) if (Array.isArray(data[key])) return (data[key] as unknown[]).filter(value => value && typeof value === 'object') as CurationValues[]
  return []
})
const rowKeys = computed(() => [...new Set(rows.value.slice(0, 20).flatMap(row => Object.keys(row)))].filter(key => !['before', 'after', 'before_text', 'after_text'].includes(key)).slice(0, 7))
const displayedRows = computed(() => rows.value.slice((rowPage.value - 1) * 40, rowPage.value * 40))
function describe(value: unknown): string { return value == null ? '—' : typeof value === 'object' ? JSON.stringify(value) : String(value) }
async function load(): Promise<void> {
  const index = ++requestIndex
  proposalRequestIndex++
  loading.value = true; failure.value = ''; reportData.value = null; proposal.value = null; proposalRels.value = []; reviewed.value = new Set(); acknowledged.value = false; rowPage.value = 1
  if (['pending', 'queued', 'running', 'stopping'].includes(props.job.state)) {
    failure.value = tr('noReport'); loading.value = false; return
  }
  try {
    const report = await props.api.report(props.job.id)
    if (index !== requestIndex) return
    reportData.value = report.report; reportPath.value = report.path
    if (props.job.stage === 'export' || props.api.schema.stages.find(stage => stage.id === props.job.stage)?.replay) {
      const list = await props.api.proposals(props.job.id)
      if (index !== requestIndex) return
      proposalRels.value = list.rels
      if (list.rels[0]) await loadProposal(list.rels[0])
    }
  } catch (cause) { if (index === requestIndex) failure.value = cause instanceof Error ? cause.message : tr('noReport') }
  finally { if (index === requestIndex) loading.value = false }
}
async function loadProposal(rel: string): Promise<void> {
  if (!rel) return
  const index = ++proposalRequestIndex
  proposalRel.value = rel; proposal.value = null; proposalLoading.value = true
  try { const value = await props.api.proposal(props.job.id, rel); if (index === proposalRequestIndex) { proposal.value = value; reviewed.value = new Set([...reviewed.value, rel]) } }
  catch (cause) { if (index === proposalRequestIndex) failure.value = String(cause) }
  finally { if (index === proposalRequestIndex) proposalLoading.value = false }
}
async function replay(): Promise<void> {
  if (!confirm.value || !acknowledged.value) return
  const operation = confirm.value
  await props.api.perform(async () => { const id = await props.api.replay(props.job.id, operation); confirm.value = null; acknowledged.value = false; emit('queued', id) })
}
watch(() => [props.job.id, props.job.state], () => { void load() }, { immediate: true })
</script>
<style scoped>
.report-json { max-height: 360px; overflow: auto; font-size: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
.report-table-wrap { overflow: auto; max-height: 450px; }
.report-table-wrap td { max-width: 340px; overflow-wrap: anywhere; }
.proposal-text, .proposal-diff { white-space: pre-wrap; overflow-wrap: anywhere; border: 1px solid rgba(var(--v-theme-on-surface), .15); border-radius: 6px; padding: 12px; font-size: 12px; min-height: 60px; }
.proposal-path { overflow-wrap: anywhere; }
.proposal-diff del { background: #d84c4c33; color: #ff9696; }
.proposal-diff ins { background: #40aa7033; color: #81dca5; text-decoration: none; }
</style>
