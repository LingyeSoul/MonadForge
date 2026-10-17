import { computed, onScopeDispose, ref } from 'vue'
import type {
  CurationGroups, CurationItem, CurationJob, CurationProposal, CurationRoots,
  CurationRow, CurationSchema, CurationValues, ParsedCaption, RootName, CaptionClause, CurationField,
} from '../types/curation'

export const rootNames: RootName[] = ['src', 'master', 'dst', 'masks', 'out']
export interface CurationMaskSource { path: string; role: string; kind: string }
export function parseMaskSource(spec: string): CurationMaskSource {
  const [role = 'keep', kind = 'text', ...rest] = spec.split(':')
  return { role, kind, path: rest.join(':') }
}
export function spellMaskSource(mask: CurationMaskSource): string { return `${mask.role}:${mask.kind}:${mask.path}` }
export function visibleCurationFields(fields: CurationField[]) {
  return fields.filter(field => !['apply', 'from_report'].includes(field.dest)
    && (!(field.root || field.setting || field.report || field.mask || field.auto) || field.overridable))
}
export function curationFieldValue(field: CurationField, value: unknown): unknown {
  if (field.kind === 'bool') return Boolean(value)
  if (field.kind === 'int' || field.kind === 'float') {
    if (value === '' || value == null) return null
    const number = Number(value)
    return Number.isFinite(number) ? (field.kind === 'int' ? Math.trunc(number) : number) : null
  }
  if (field.kind === 'list') return String(value || '').split('\n').map(value => value.trim()).filter(Boolean)
  return value
}
export function normalizeCurationRoots(raw: unknown): CurationRoots {
  const source = (raw || {}) as Record<string, unknown>
  return Object.fromEntries(rootNames.map(key => {
    const value = source[key]
    return [key, typeof value === 'string' ? { path: value } : { path: '', ...(value as object || {}) }]
  })) as CurationRoots
}

/** Keep the caption grammar on the server; this diff only compares literal text. */
export function captionDiff(before: string, after: string) {
  let start = 0
  while (start < before.length && start < after.length && before[start] === after[start]) start++
  let end = 0
  while (end < before.length - start && end < after.length - start && before[before.length - end - 1] === after[after.length - end - 1]) end++
  return { prefix: before.slice(0, start), removed: before.slice(start, before.length - end),
    added: after.slice(start, after.length - end), suffix: end ? before.slice(-end) : '' }
}

export function curationFileUrl(path?: string | null, thumbnail = false): string {
  return path ? `/api/curation/${thumbnail ? 'thumb' : 'files'}?${new URLSearchParams({ path, ...(thumbnail ? { size: '256' } : {}) })}` : ''
}

function errorText(value: unknown): string {
  if (typeof value === 'string') return value
  if (value && typeof value === 'object' && 'detail' in value) return errorText((value as { detail: unknown }).detail)
  if (Array.isArray(value)) return value.map(errorText).join('; ')
  if (value && typeof value === 'object' && 'msg' in value) return String((value as { msg: unknown }).msg)
  return JSON.stringify(value) || 'Request failed'
}

export function useCuration() {
  const schema = ref<CurationSchema>({ stages: [] })
  const roots = ref<CurationRoots>(normalizeCurationRoots(null))
  const settings = ref<CurationValues>({})
  const items = ref<CurationRow[]>([])
  const groups = ref<CurationGroups>({ groups: [] })
  const jobs = ref<CurationJob[]>([])
  const item = ref<CurationItem | null>(null)
  const error = ref('')
  const busy = ref(false)
  const loading = ref(false)
  const result = ref<unknown>(null)
  const listInfo = ref<CurationValues>({})
  let itemRequest = 0
  let datasetRequest = 0
  let disposed = false
  const controllers = new Set<AbortController>()
  onScopeDispose(() => { disposed = true; for (const controller of controllers) controller.abort() })

  async function request<T>(path: string, body?: unknown, method = 'GET'): Promise<T> {
    const controller = new AbortController()
    controllers.add(controller)
    try {
      const response = await fetch(`/api/curation${path}`, {
        method, signal: controller.signal,
        headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
      })
      const data = await response.json().catch(() => null)
      if (!response.ok) throw new Error(data ? errorText(data) : `${response.status} ${response.statusText}`)
      return data as T
    } finally { controllers.delete(controller) }
  }
  async function perform<T>(action: () => Promise<T>): Promise<T | undefined> {
    if (busy.value) return undefined
    busy.value = true
    error.value = ''
    try { return await action() }
    catch (cause) { if (!disposed) error.value = cause instanceof Error ? cause.message : String(cause) }
    finally { busy.value = false }
  }
  async function loadSchema() {
    const data = await request<CurationSchema>('/schema')
    schema.value = data
    if (data.roots) roots.value = normalizeCurationRoots(data.roots)
  }
  async function loadRoots() {
    const data = await request<{ roots: unknown }>('/roots')
    roots.value = normalizeCurationRoots(data.roots)
  }
  async function loadDataset(query = '') {
    const index = ++datasetRequest
    const data = await request<{ items: CurationRow[]; [key: string]: unknown }>(`/dataset?${new URLSearchParams({ search: query })}`)
    if (index !== datasetRequest || disposed) return
    items.value = data.items || []
    listInfo.value = data
  }
  async function loadGroups() { groups.value = await request<CurationGroups>('/groups') }
  async function loadJobs() {
    const data = await request<{ jobs: CurationJob[] } | CurationJob[]>('/jobs')
    jobs.value = (Array.isArray(data) ? data : data.jobs || []).map(job => ({ ...job, id: job.id || job.task_id || '' }))
  }
  async function initialize() {
    loading.value = true
    await perform(async () => {
      await loadSchema()
      await loadRoots()
      const response = await request<CurationValues>('/settings')
      settings.value = response.settings as CurationValues || response
      await Promise.all([loadDataset(), loadGroups(), loadJobs()])
    })
    loading.value = false
  }
  async function selectItem(rel: string) {
    const index = ++itemRequest
    const detail = await request<CurationItem>(`/items?${new URLSearchParams({ rel })}`)
    if (index === itemRequest && !disposed) item.value = detail
    return detail
  }
  async function saveCaption(rel: string, kind: string, text: string) {
    const saved = await request<CurationValues>('/caption', { rel, kind, text }, 'PUT')
    await selectItem(rel)
    return saved
  }
  async function parse(text: string) { return request<ParsedCaption>('/parse', { text }, 'POST') }
  async function compose(flat_tags: string[], clauses: CaptionClause[]) {
    return request<{ text: string; parsed: ParsedCaption }>('/compose', { flat_tags, clauses }, 'POST')
  }
  async function saveRoots(values: Record<RootName, string>) {
    const response = await request<{ roots: unknown }>('/roots', values, 'PUT')
    roots.value = normalizeCurationRoots(response.roots)
    itemRequest++
    item.value = null
    await loadSchema()
    await Promise.all([loadDataset(), loadGroups()])
  }
  async function saveSettings(values: CurationValues) {
    const response = await request<CurationValues>('/settings', values, 'PUT')
    settings.value = response.settings as CurationValues || response
    await loadSchema()
  }
  async function exclude(rel: string, excluded: boolean) {
    result.value = await request('/exclude', { rel, excluded }, 'POST')
    await selectItem(rel)
    await loadDataset()
  }
  async function submit(stage: string, values: CurationValues, apply: boolean, rel?: string) {
    const job = await request<CurationJob>('/jobs', { stage, values, apply, ...(rel ? { rel } : {}) }, 'POST')
    await loadJobs()
    return job.id || job.task_id || ''
  }
  async function report(id: string) { return request<{ report: CurationValues; path: string }>(`/jobs/${encodeURIComponent(id)}/report`) }
  async function proposals(id: string) {
    return request<{ rels: string[]; total: number; kind: string; apply: boolean }>(`/jobs/${encodeURIComponent(id)}/proposals`)
  }
  async function proposal(id: string, rel: string) {
    return request<CurationProposal>(`/jobs/${encodeURIComponent(id)}/proposal?${new URLSearchParams({ rel })}`)
  }
  async function replay(id: string, operation: 'apply' | 'undo') {
    const job = await request<CurationJob>(`/jobs/${encodeURIComponent(id)}/${operation}`, {}, 'POST')
    await loadJobs()
    return job.id || job.task_id || ''
  }
  const activeJobs = computed(() => jobs.value.filter(job => ['queued', 'pending', 'running'].includes(job.state)))
  return { schema, roots, settings, items, groups, jobs, item, error, busy, loading, result, listInfo, activeJobs,
    perform, initialize, loadSchema, loadRoots, loadDataset, loadGroups, loadJobs, selectItem,
    saveCaption, parse, compose, saveRoots, saveSettings, exclude, submit, report, proposals, proposal, replay }
}
