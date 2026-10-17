<template>
  <section class="curation-workspace" data-testid="curation-workspace">
    <div class="d-flex flex-wrap align-center ga-2 mb-3">
      <div><h2 class="text-h6">{{ tr('title') }}</h2><p class="text-body-2 text-medium-emphasis">{{ tr('subtitle') }}</p></div>
      <v-spacer /><v-chip size="small" variant="tonal">{{ api.items.length }} {{ tr('dataset') }}</v-chip>
      <v-chip v-if="api.activeJobs.length" size="small" color="primary">{{ api.activeJobs.length }} {{ tr('queued') }}</v-chip>
      <v-btn icon="mdi-refresh" variant="text" size="small" :aria-label="tr('refresh')" :loading="api.loading" :disabled="api.busy" @click="refresh" />
    </div>
    <v-alert v-if="api.error" type="error" variant="tonal" closable class="mb-3" data-testid="curation-error" @click:close="api.error = ''">{{ api.error }}</v-alert>
    <v-alert v-if="notice" type="success" variant="tonal" closable class="mb-3" @click:close="notice = ''">{{ notice }}</v-alert>
    <v-alert v-if="rootsDirty && tab !== 'settings'" type="warning" variant="tonal" density="compact" class="mb-3">{{ tr('changedRoots') }} <v-btn size="small" variant="text" @click="tab = 'settings'">{{ tr('settings') }}</v-btn></v-alert>
    <v-tabs v-model="tab" color="primary" class="mb-4"><v-tab value="dataset">{{ tr('dataset') }}</v-tab><v-tab value="stages">{{ tr('stages') }} · {{ api.schema.stages.length }}</v-tab><v-tab value="jobs">{{ tr('jobs') }}</v-tab><v-tab value="settings">{{ tr('settings') }}</v-tab></v-tabs>
    <v-progress-linear v-if="api.loading" indeterminate class="mb-3" />
    <div v-show="tab === 'dataset'" class="dataset-layout">
      <aside class="dataset-sidebar">
        <v-text-field v-model="search" :label="tr('search')" prepend-inner-icon="mdi-magnify" density="compact" hide-details clearable class="mb-3" />
        <v-select v-model="folder" :items="folders" :label="tr('folder')" density="compact" hide-details clearable class="mb-3" />
        <v-select v-model="groupId" :items="groupOptions" :label="tr('group')" density="compact" hide-details clearable class="mb-3" />
        <v-btn-toggle v-model="inclusion" mandatory density="compact" variant="outlined" class="mb-3"><v-btn value="all" size="small">{{ tr('all') }}</v-btn><v-btn value="active" size="small">{{ tr('active') }}</v-btn><v-btn value="excluded" size="small">{{ tr('excluded') }}</v-btn></v-btn-toggle>
        <p class="text-caption text-medium-emphasis mb-2">{{ filteredItems.length }} / {{ api.items.length }}</p>
        <div class="dataset-list" data-testid="curation-image-list">
          <button v-for="row in pagedItems" :key="row.rel" type="button" class="dataset-row" :class="{ selected: api.item?.rel === row.rel, excluded: row.excluded }" :disabled="api.busy" @click="select(row.rel)">
            <img :src="curationFileUrl(sourcePath(row.rel), true)" :alt="row.name" loading="lazy" />
            <span class="dataset-row-copy"><strong>{{ row.name }}</strong><small>{{ row.dir || '/' }}</small><span class="d-flex flex-wrap ga-1 mt-1"><span v-for="(present, kind) in row.captions" :key="kind" class="caption-dot" :class="{ present }" :title="String(kind)" /><v-icon v-if="row.mask" icon="mdi-image-filter-center-focus" size="14" /><v-icon v-if="row.excluded" icon="mdi-cancel" size="14" /></span></span>
          </button>
          <p v-if="!filteredItems.length" class="pa-4 text-body-2 text-medium-emphasis">{{ tr('empty') }}</p>
        </div>
        <v-pagination v-if="filteredItems.length > pageSize" v-model="page" :length="Math.ceil(filteredItems.length / pageSize)" density="compact" :total-visible="3" />
        <v-alert v-if="api.groups.missing" type="info" variant="tonal" density="compact" class="mt-3">{{ tr('groupsMissing') }}</v-alert>
        <v-alert v-if="api.groups.stale" type="warning" variant="tonal" density="compact" class="mt-3">{{ tr('groupsStale') }}</v-alert>
        <v-card v-if="selectedGroup" variant="outlined" class="mt-3 pa-3"><div class="text-subtitle-2">{{ selectedGroup.artist || tr('group') }} #{{ selectedGroup.id }}</div><p class="text-body-2">{{ selectedGroup.members.length }} {{ tr('dataset') }} <span v-if="selectedGroup.mean_cosine != null">· cosine {{ selectedGroup.mean_cosine.toFixed(3) }}</span></p></v-card>
        <v-btn size="small" variant="text" prepend-icon="mdi-image-multiple-outline" class="mt-2" @click="stageId = 'groups'; tab = 'stages'">{{ tr('groups') }}</v-btn>
      </aside>
      <div class="dataset-detail">
        <CurationInspector v-if="api.item" ref="inspector" :item="api.item" :busy="api.busy" :parse="api.parse" :compose="api.compose" @dirty="captionDirty = $event" @save="saveCaption" @exclude="exclude" />
        <v-card v-else variant="tonal" class="pa-10 text-center"><v-icon icon="mdi-image-search-outline" size="48" class="mb-3" /><p>{{ tr('selectImage') }}</p><v-btn v-if="!api.items.length" variant="outlined" class="mt-4" @click="tab = 'settings'">{{ tr('settings') }}</v-btn></v-card>
      </div>
    </div>
    <div v-show="tab === 'stages'">
      <p class="text-body-2 text-medium-emphasis mb-4">{{ tr('queueHint') }}</p>
      <div class="stage-grid mb-4"><v-btn v-for="(entry, index) in api.schema.stages" :key="entry.id" :data-testid="`curation-stage-${entry.id}`" :variant="stageId === entry.id ? 'flat' : 'outlined'" :color="stageId === entry.id ? 'primary' : undefined" :prepend-icon="entry.available ? undefined : 'mdi-alert-circle-outline'" @click="stageId = entry.id">{{ String(index + 1).padStart(2, '0') }} · {{ stageLabel(entry) }}</v-btn></div>
      <p v-if="!api.schema.stages.length" class="text-medium-emphasis">{{ tr('noStages') }}</p>
      <v-card v-if="stage" variant="tonal">
        <v-card-title class="text-subtitle-1">{{ stageLabel(stage) }}</v-card-title>
        <v-card-text>
          <v-alert v-if="!stage.available" type="error" variant="tonal" class="mb-4">{{ tr('unavailable') }}: {{ stage.error }}</v-alert>
          <p v-if="stage.doc" class="text-body-2 mb-3 stage-document">{{ stage.doc }}</p>
          <v-alert v-if="stage.notes" type="info" variant="tonal" density="compact" class="mb-3">{{ stage.notes }}</v-alert>
          <v-alert v-if="stage.id === 'export'" type="warning" variant="tonal" class="mb-4">{{ tr('exportHint') }}</v-alert>
          <CurationFields :fields="stage.fields" :model-value="stageValues[stage.id] || {}" :disabled="api.busy || !stage.available" @update:model-value="stageValues[stage!.id] = $event" />
          <div class="d-flex flex-wrap align-center ga-3 mt-3">
            <v-checkbox v-if="stage.scoped" v-model="scoped" :label="tr('scope')" :disabled="!api.item || Boolean(api.item.excluded)" density="compact" hide-details />
            <span v-if="scoped && api.item" class="text-caption path-label">{{ api.item.rel }}</span>
            <v-switch v-if="stage.apply && stage.id !== 'export'" v-model="runApply" :label="tr('runMode')" density="compact" hide-details color="warning" />
          </div>
          <p v-if="!stage.apply" class="text-caption text-medium-emphasis mt-3">{{ tr('stageWrites') }}</p>
        </v-card-text>
        <v-card-actions class="flex-wrap"><v-btn data-testid="curation-run-stage" color="primary" variant="flat" :loading="api.busy" :disabled="!stage.available || rootsDirty || captionDirty || (scoped && (!api.item || Boolean(api.item.excluded)))" @click="queueStage">{{ stage.id === 'export' || (stage.apply && !runApply) ? tr('preview') : tr('run') }}</v-btn><v-btn variant="text" :disabled="api.busy" @click="stageValues[stage.id] = {}">{{ tr('reset') }}</v-btn></v-card-actions>
      </v-card>
    </div>
    <div v-show="tab === 'jobs'" class="jobs-layout">
      <v-list class="job-list" density="compact" bg-color="transparent"><v-list-item v-for="job in api.jobs" :key="job.id" :active="jobId === job.id" :title="`${job.stage} · ${job.state}`" :subtitle="`${job.apply ? tr('applied') : tr(api.schema.stages.find(stage => stage.id === job.stage)?.apply === false ? 'workspaceRun' : 'dryRun')} · ${job.created_at || job.id}`" @click="jobId = job.id"><template #prepend><v-icon :icon="['running', 'queued', 'pending'].includes(job.state) ? 'mdi-progress-clock' : job.error ? 'mdi-alert-circle-outline' : 'mdi-file-document-outline'" /></template></v-list-item><v-list-item v-if="!api.jobs.length" :title="tr('noJobs')" /></v-list>
      <div class="job-detail"><CurationReport v-if="selectedJob" :job="selectedJob" :api="api" @queued="onQueued" /><v-card v-else variant="tonal" class="pa-8 text-center">{{ tr('selectJob') }}</v-card></div>
    </div>
    <div v-show="tab === 'settings'">
      <v-alert type="info" variant="tonal" class="mb-4">{{ tr('sourceHint') }} {{ tr('settingsHint') }}</v-alert>
      <v-card variant="tonal" class="mb-4"><v-card-title class="text-subtitle-1">{{ tr('settings') }}</v-card-title><v-card-text><v-row dense><v-col v-for="name in rootNames" :key="name" cols="12" md="6"><v-text-field v-model="rootDraft[name]" :disabled="api.busy" :data-testid="`curation-root-${name}`" :label="`${name} · ${tr(name === 'src' ? 'source' : name)}`" :hint="api.roots[name].exists === false ? tr('missing') : ''" persistent-hint density="compact" /></v-col></v-row><v-btn v-if="inputDir" size="small" variant="text" @click="rootDraft.src = inputDir">{{ tr('chooseSource') }} · {{ inputDir }}</v-btn></v-card-text><v-card-actions><v-btn data-testid="curation-save-roots" variant="flat" color="primary" :disabled="!rootsDirty || api.busy" @click="saveRoots">{{ tr('saveRoots') }}</v-btn></v-card-actions></v-card>
      <v-card variant="tonal"><v-card-title class="text-subtitle-1">{{ tr('defaults') }}</v-card-title><v-card-text><v-row dense><v-col v-for="name in defaultKeys" :key="name" cols="12" md="6"><v-text-field v-model="defaultDraft[name]" :disabled="api.busy" :label="name === 'report_root' ? tr('reportRoot') : name === 'mask_root' ? tr('maskRoot') : name" density="compact" hide-details /></v-col></v-row><div v-if="resizeStage" class="mt-5"><h3 class="text-subtitle-2 mb-3">{{ tr('resizeDefaults') }}</h3><CurationFields v-model="preprocessDraft" :fields="resizeStage.fields" :disabled="api.busy" /></div></v-card-text><v-card-actions><v-btn color="primary" :disabled="api.busy" @click="saveSettings">{{ tr('saveSettings') }}</v-btn></v-card-actions></v-card>
    </div>
    <v-expansion-panels v-if="api.result" variant="accordion" class="mt-4"><v-expansion-panel :title="tr('reportResult')"><v-expansion-panel-text><pre class="operation-result">{{ JSON.stringify(api.result, null, 2) }}</pre></v-expansion-panel-text></v-expansion-panel></v-expansion-panels>
  </section>
</template>
<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { onBeforeRouteLeave } from 'vue-router'
import { useAppStore } from '../../stores/app'
import { curationFileUrl, rootNames, useCuration } from '../../composables/useCuration'
import { curationEn, curationText } from '../../i18n/curation'
import type { CurationStage, CurationValues, RootName } from '../../types/curation'
import CurationFields from './CurationFields.vue'
import CurationInspector from './CurationInspector.vue'
import CurationReport from './CurationReport.vue'
const props = defineProps<{ inputDir?: string }>()
const inputDir = computed(() => props.inputDir || '')
const app = useAppStore()
const tr = (key: Parameters<typeof curationText>[1]) => curationText(app.language, key)
const api = reactive(useCuration())
const tab = ref('dataset'), notice = ref(''), captionDirty = ref(false)
const inspector = ref<InstanceType<typeof CurationInspector> | null>(null)
const search = ref(''), folder = ref<string | null>(null), groupId = ref<number | null>(null), inclusion = ref('all'), page = ref(1)
const pageSize = 30
const folders = computed(() => [...new Set(api.items.map(item => item.dir).filter(Boolean))].sort())
const groupOptions = computed(() => api.groups.groups.map(group => ({ title: `#${group.id} · ${group.artist || '—'} · ${group.members.length}`, value: group.id })))
const selectedGroup = computed(() => api.groups.groups.find(group => group.id === groupId.value))
const filteredItems = computed(() => {
  const needle = (search.value || '').toLocaleLowerCase()
  const members = selectedGroup.value ? new Set(selectedGroup.value.members) : null
  return api.items.filter(row => (!needle || row.rel.toLocaleLowerCase().includes(needle)) && (!folder.value || row.dir === folder.value || row.dir.startsWith(`${folder.value}/`)) && (!members || members.has(row.rel)) && (inclusion.value === 'all' || row.excluded === (inclusion.value === 'excluded')))
})
const pagedItems = computed(() => filteredItems.value.slice((page.value - 1) * pageSize, page.value * pageSize))
watch([search, folder, groupId, inclusion], () => { page.value = 1 })
function sourcePath(rel: string): string { return `${api.roots.src.path.replace(/[\\/]$/, '')}/${rel}` }
const rootDraft = ref<Record<RootName, string>>({ src: '', master: '', dst: '', masks: '', out: '' })
const rootsDirty = computed(() => rootNames.some(name => rootDraft.value[name] !== api.roots[name].path))
watch(() => api.roots, () => { rootDraft.value = Object.fromEntries(rootNames.map(name => [name, api.roots[name].path])) as Record<RootName, string> }, { deep: true, immediate: true })
const defaultKeys = ['path_pattern', 'tagger_dir', 'checkpoint', 'prompt_embed', 'report_root', 'mask_root']
const defaultDraft = ref<Record<string, string>>({}), preprocessDraft = ref<CurationValues>({}), stageValues = ref<Record<string, CurationValues>>({})
watch(() => api.settings, settings => {
  defaultDraft.value = { ...settings.stage_defaults as Record<string, string> || {} }
  preprocessDraft.value = { ...settings.preprocess as CurationValues || {} }
  stageValues.value = { ...settings.values as Record<string, CurationValues> || {} }
}, { immediate: true })
const stageId = ref('autotag'), runApply = ref(false), scoped = ref(false), jobId = ref('')
const stage = computed(() => api.schema.stages.find(entry => entry.id === stageId.value) || api.schema.stages[0])
const resizeStage = computed(() => api.schema.stages.find(entry => entry.id === 'resize'))
const selectedJob = computed(() => api.jobs.find(job => job.id === jobId.value))
watch(stageId, () => { runApply.value = false; scoped.value = false })
function stageLabel(entry: CurationStage): string { return entry.id in curationEn ? tr(entry.id as keyof typeof curationEn) : entry.title }
async function leaveCaption(): Promise<boolean> { return await inspector.value?.requestLeave() ?? true }
async function select(rel: string): Promise<void> { if (api.item?.rel === rel || !await leaveCaption()) return; await api.perform(() => api.selectItem(rel)) }
async function refresh(): Promise<void> {
  if (!await leaveCaption()) return
  const rel = api.item?.rel
  await api.initialize()
  if (rel && api.items.some(row => row.rel === rel)) await api.perform(() => api.selectItem(rel))
}
async function saveCaption(kind: string, text: string): Promise<void> {
  if (!api.item) return
  await api.perform(async () => { api.result = await api.saveCaption(api.item!.rel, kind, text); notice.value = `${tr('saved')} · ${tr('stale')}`; await api.loadDataset() })
}
async function exclude(excluded: boolean): Promise<void> { if (!api.item || !await leaveCaption()) return; await api.perform(async () => { await api.exclude(api.item!.rel, excluded); notice.value = tr('restored') }) }
async function saveRoots(): Promise<void> { if (!await leaveCaption()) return; await api.perform(async () => { await api.saveRoots(rootDraft.value); captionDirty.value = false; groupId.value = null; folder.value = null; page.value = 1; notice.value = tr('saved') }) }
async function saveSettings(): Promise<void> { await api.perform(async () => { await api.saveSettings({ stage_defaults: defaultDraft.value, preprocess: preprocessDraft.value }); notice.value = tr('saved') }) }
async function queueStage(): Promise<void> {
  if (!stage.value || rootsDirty.value || captionDirty.value) return
  const selected = stage.value
  await api.perform(async () => {
    const id = await api.submit(selected.id, stageValues.value[selected.id] || {}, selected.id === 'export' ? false : Boolean(selected.apply && runApply.value), scoped.value ? api.item?.rel : undefined)
    onQueued(id)
  })
}
function onQueued(id: string): void { jobId.value = id; tab.value = 'jobs'; notice.value = tr('queued') }
let timer: ReturnType<typeof setTimeout> | undefined, disposed = false
async function poll(): Promise<void> {
  if (disposed) return
  if (api.activeJobs.length && !api.busy && !document.hidden) {
    const active = api.activeJobs.map(job => job.id)
    try {
      await api.loadJobs()
      if (active.some(id => !api.activeJobs.some(job => job.id === id))) {
        await Promise.all([api.loadDataset(), api.loadGroups()])
        if (api.item && !captionDirty.value) await api.selectItem(api.item.rel)
      }
    } catch (cause) { api.error = cause instanceof Error ? cause.message : String(cause) }
  }
  if (!disposed) timer = setTimeout(poll, 5000)
}
function beforeUnload(event: BeforeUnloadEvent): void { if (captionDirty.value) { event.preventDefault(); event.returnValue = '' } }
onBeforeRouteLeave(() => leaveCaption())
onMounted(async () => { window.addEventListener('beforeunload', beforeUnload); await api.initialize(); if (!disposed) timer = setTimeout(poll, 5000) })
onBeforeUnmount(() => { disposed = true; clearTimeout(timer); window.removeEventListener('beforeunload', beforeUnload) })
</script>
<style scoped>
.curation-workspace { min-width: 0; }
.dataset-layout, .jobs-layout { display: grid; grid-template-columns: minmax(260px, 320px) minmax(0, 1fr); gap: 20px; align-items: start; }
.dataset-sidebar, .dataset-detail, .job-list, .job-detail { min-width: 0; }
.dataset-list { max-height: 680px; overflow: auto; }
.dataset-row { display: flex; width: 100%; gap: 10px; align-items: center; padding: 9px; border: 1px solid transparent; border-radius: 6px; text-align: left; color: inherit; background: transparent; }
.dataset-row:hover { background: rgba(var(--v-theme-primary), .08); }
.dataset-row.selected { border-color: rgb(var(--v-theme-primary)); background: rgba(var(--v-theme-primary), .12); }
.dataset-row.excluded { opacity: .6; }
.dataset-row img { width: 54px; height: 62px; object-fit: contain; border-radius: 3px; background: rgba(0,0,0,.15); }
.dataset-row-copy { min-width: 0; flex: 1; }
.dataset-row-copy strong, .dataset-row-copy small { display: block; overflow-wrap: anywhere; }
.dataset-row-copy strong { font-size: 12px; font-weight: 500; }
.dataset-row-copy small { font-size: 10px; opacity: .65; }
.caption-dot { width: 6px; height: 6px; border: 1px solid rgba(var(--v-theme-on-surface), .35); border-radius: 50%; margin-top: 4px; }
.caption-dot.present { background: rgb(var(--v-theme-primary)); border-color: transparent; }
.stage-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 8px; }
.stage-document { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 220px; overflow: auto; }
.path-label { overflow-wrap: anywhere; }
.operation-result { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 350px; overflow: auto; }
@media (max-width: 900px) { .dataset-layout, .jobs-layout { grid-template-columns: minmax(0, 1fr); } .dataset-list { max-height: 290px; } .job-list { max-height: 260px; overflow: auto; } }
</style>
